"""Add LoRA outputs to the base layer's output before rounding to bf16 (vLLM sampler and PEFT trainer).

In bf16, both vLLM's LoRA layers and PEFT compute the base layer's output, round it to bf16, and only then add the LoRA
output. A rounded base output sits exactly on the bf16 grid, so a LoRA contribution below half a grid step rounds back to
the same value: small adapters (early in training, B starts at zero) are partly dropped, deterministically, while
autodiff sees the full effect. With this patch the base matmul keeps its fp32 accumulator output, the LoRA output (still
computed in the model dtype, as before) is added in fp32, and the sum is rounded once.

Enabled with `--lora_fp32_add`, which sets ENV so that vLLM's engine processes (which load
`register_vllm_plugin` through the `vllm.general_plugins` entry point) patch themselves, and calls `patch_peft()` in
the trainer. Only unquantized layers with tensor parallel size 1 are supported.
"""

import os

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

LORA_FP32_ADD_ENV = "OPEN_INSTRUCT_LORA_FP32_ADD"
# vLLM reuses compiled graphs from its cache, keyed without this monkeypatch: a graph compiled unpatched would run instead.
ENV = {LORA_FP32_ADD_ENV: "1", "VLLM_DISABLE_COMPILE_CACHE": "1"}


def enabled() -> bool:
    return os.environ.get(LORA_FP32_ADD_ENV) == "1"


class _MatmulFp32Out(torch.autograd.Function):
    """x @ w.T with an fp32 output from bf16 operands; the gradient w.r.t. x is computed as in a bf16 linear layer.

    The base weight is frozen (LoRA), so no gradient flows to it."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(w)
        return torch.mm(x, w.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (w,) = ctx.saved_tensors
        return grad.to(w.dtype) @ w, None


def base_linear_fp32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Base linear layer output in fp32 (bf16 inputs, fp32 accumulator, no rounding of the result)."""
    if weight.requires_grad:
        raise ValueError("lora_fp32_add expects a frozen base weight")
    x2 = x.reshape(-1, x.shape[-1])
    y = (
        _MatmulFp32Out.apply(x2, weight)
        if torch.is_grad_enabled()
        else torch.mm(x2, weight.t(), out_dtype=torch.float32)
    )
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*x.shape[:-1], weight.shape[0])


def patch_peft() -> None:
    """Patch peft's LoRA Linear forward: result = round_bf16(base_fp32(x) + lora(x)), lora(x) as PEFT computes it."""
    from peft.tuners.lora.layer import Linear  # noqa: PLC0415

    if getattr(Linear, "_lora_fp32_add", False):
        return
    original = Linear.forward

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if (
            self.disable_adapters
            or self.merged
            or args
            or kwargs
            or not isinstance(self.base_layer, torch.nn.Linear)
            or self.base_layer.weight.dtype != torch.bfloat16
            or any(n in self.lora_variant for n in self.active_adapters)
        ):
            return original(self, x, *args, **kwargs)
        out_dtype = x.dtype
        result = base_linear_fp32(x, self.base_layer.weight, self.base_layer.bias)
        for name in self.active_adapters:
            if name not in self.lora_A:
                continue
            lora_a, lora_b = self.lora_A[name], self.lora_B[name]
            h = self._cast_input_dtype(x, lora_a.weight.dtype)
            result = result + (lora_b(lora_a(self.lora_dropout[name](h))) * self.scaling[name]).float()
        return result.to(out_dtype)

    Linear.forward = forward
    Linear._lora_fp32_add = True
    logger.info("lora_fp32_add: patched peft LoRA Linear.forward")


def patch_vllm() -> None:
    """Patch vLLM's BaseLinearLayerWithLoRA.apply in this process (unquantized, TP = 1)."""
    from vllm.distributed import get_tensor_model_parallel_world_size  # noqa: PLC0415
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA  # noqa: PLC0415
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: PLC0415
    from vllm.platforms import current_platform  # noqa: PLC0415

    if getattr(BaseLinearLayerWithLoRA, "_lora_fp32_add", False):
        return
    if os.environ.get("VLLM_DISABLE_COMPILE_CACHE") != "1":
        raise RuntimeError("lora_fp32_add needs VLLM_DISABLE_COMPILE_CACHE=1 (a cached graph would skip the patch)")
    original = BaseLinearLayerWithLoRA.apply

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not isinstance(self.base_layer.quant_method, UnquantizedLinearMethod) or x.ndim != 2:
            return original(self, x, bias)
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("lora_fp32_add supports tensor parallel size 1 only")
        base = base_linear_fp32(x, self.base_layer.weight, bias)
        delta = torch.zeros(base.shape, dtype=x.dtype, device=x.device)
        lora_output = self.punica_wrapper.add_lora_linear(
            delta, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
        )
        if not current_platform.can_update_inplace():
            delta = lora_output
        return (base + delta.float()).to(x.dtype)

    BaseLinearLayerWithLoRA.apply = apply
    BaseLinearLayerWithLoRA._lora_fp32_add = True
    logger.info("lora_fp32_add: patched vLLM BaseLinearLayerWithLoRA.apply")


def register_vllm_plugin() -> None:
    """vLLM general plugin (loaded in every vLLM process); patches only when the environment variable is set."""
    if enabled():
        patch_vllm()
