"""Add LoRA outputs to the base layer's output before rounding to bf16 (vLLM sampler and PEFT trainer).

In bf16, both vLLM's LoRA layers and PEFT compute the base layer's output, round it to bf16, and only then add the LoRA
output. A rounded base output sits exactly on the bf16 grid, so a LoRA contribution below half a grid step rounds back to
the same value: small adapters (early in training, B starts at zero) are partly dropped, deterministically, while
autodiff sees the full effect. With this patch the base matmul keeps its fp32 accumulator output, the LoRA output (still
computed in the model dtype, as before) is added in fp32, and the sum is rounded once.

Enabled with `--lora_fp32_add`, which sets ENV so that vLLM's engine processes (which load
`register_vllm_plugin` through the `vllm.general_plugins` entry point) patch themselves, and calls `patch_peft()` in
the trainer. Only unquantized layers with tensor parallel size 1 are supported. Unsupported paths raise instead of
silently running the stock numerics. Every patching process writes a JSON report (versions, environment, which LoRA
layer classes received adapter weights, including vLLM's warm-up dummy adapters) to $REPORT_DIR_ENV, so a run records that the patch was active.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
import socket
import time

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

LORA_FP32_ADD_ENV = "OPEN_INSTRUCT_LORA_FP32_ADD"
# vLLM reuses compiled graphs from its cache, keyed without this monkeypatch: a graph compiled unpatched would run instead.
ENV = {LORA_FP32_ADD_ENV: "1", "VLLM_DISABLE_COMPILE_CACHE": "1"}
REPORT_DIR_ENV = "OPEN_INSTRUCT_LORA_NUMERICS_DIR"
# Allowlist: reports land in the run directory, which may be uploaded (no credentials such as VLLM_API_KEY).
REPORT_ENV_VARS = (
    LORA_FP32_ADD_ENV,
    REPORT_DIR_ENV,
    "VLLM_DISABLE_COMPILE_CACHE",
    "VLLM_BATCH_INVARIANT",
    "VLLM_PLUGINS",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_ATTENTION_BACKEND",
    "TORCH_COMPILE_DISABLE",
)


def enabled() -> bool:
    return os.environ.get(LORA_FP32_ADD_ENV) == "1"


def write_report(kind: str, **info) -> None:
    """Write (or overwrite) this process's report <REPORT_DIR>/lora_numerics_<kind>_<host>_<pid>.json."""
    report_dir = os.environ.get(REPORT_DIR_ENV)
    if not report_dir:
        return
    versions = {}
    for package in ("torch", "vllm", "peft", "transformers", "deepspeed"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    with open(__file__, "rb") as f:
        module_sha256 = hashlib.sha256(f.read()).hexdigest()
    report = {
        "kind": kind,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "module": __file__,
        "module_sha256": module_sha256,
        "versions": versions,
        "env": {k: os.environ[k] for k in REPORT_ENV_VARS if k in os.environ},
        **info,
    }
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"lora_numerics_{kind}_{report['host']}_{report['pid']}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
    os.replace(path + ".tmp", path)


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
        # Paths without the bf16 dead zone keep the stock forward: no adapter added, or an fp32 model (diagnostics).
        if self.disable_adapters or self.base_layer.weight.dtype == torch.float32:
            return original(self, x, *args, **kwargs)
        if (
            self.merged
            or args
            or kwargs
            or not isinstance(self.base_layer, torch.nn.Linear)
            or self.base_layer.weight.dtype != torch.bfloat16
            or any(n in self.lora_variant for n in self.active_adapters)
        ):
            raise NotImplementedError(
                "lora_fp32_add covers plain LoRA on bf16 nn.Linear layers only (no merged adapters, variants such as "
                f"DoRA, or extra forward arguments); got {type(self.base_layer).__name__} "
                f"{self.base_layer.weight.dtype}, merged={self.merged}, variants={list(self.lora_variant)}"
            )
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
    write_report("peft", patched="peft.tuners.lora.layer.Linear.forward")
    logger.info("lora_fp32_add: patched peft LoRA Linear.forward")


def _subclasses(cls) -> list[type]:
    out = []
    for sub in cls.__subclasses__():
        out += [sub, *_subclasses(sub)]
    return out


def patch_vllm() -> None:
    """Patch vLLM's BaseLinearLayerWithLoRA.apply in this process (unquantized, TP = 1).

    LoRA layer classes that do not go through this apply (fully sharded LoRA, which overrides apply; logits,
    embedding and MoE LoRA layers) raise when an adapter with weights for them is loaded (set_lora), so they cannot
    silently run with the stock numerics. Unsupported inputs to the patched apply (quantized base layers, 3-D inputs,
    tensor parallelism) raise as well."""
    import vllm.lora.layers  # noqa: PLC0415, F401  (registers every LoRA layer class)
    from vllm.distributed import get_tensor_model_parallel_world_size  # noqa: PLC0415
    from vllm.lora.layers.base import BaseLayerWithLoRA  # noqa: PLC0415
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA  # noqa: PLC0415
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: PLC0415
    from vllm.platforms import current_platform  # noqa: PLC0415

    if getattr(BaseLinearLayerWithLoRA, "_lora_fp32_add", False):
        return
    if os.environ.get("VLLM_DISABLE_COMPILE_CACHE") != "1":
        raise RuntimeError("lora_fp32_add needs VLLM_DISABLE_COMPILE_CACHE=1 (a cached graph would skip the patch)")
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"):
        raise NotImplementedError("lora_fp32_add bypasses the batch-invariant matmul of VLLM_BATCH_INVARIANT")

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not isinstance(self.base_layer.quant_method, UnquantizedLinearMethod) or x.ndim != 2:
            raise NotImplementedError(
                f"lora_fp32_add: unsupported {type(self.base_layer.quant_method).__name__} or {x.ndim}-D input"
            )
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

    covered, uncovered = [], []
    for cls in _subclasses(BaseLayerWithLoRA):
        if issubclass(cls, BaseLinearLayerWithLoRA) and all(
            "apply" not in c.__dict__ for c in cls.__mro__[: cls.__mro__.index(BaseLinearLayerWithLoRA)]
        ):
            covered.append(cls)
        else:
            uncovered.append(cls)
    loaded = set()
    covered_set = set(covered)

    def make_set_lora(original_set_lora):
        def set_lora(self, *args, **kwargs):
            # coverage of the instance's own class: subclasses (e.g. fully sharded) inherit set_lora from covered parents
            name = type(self).__name__
            if type(self) not in covered_set:
                raise NotImplementedError(f"lora_fp32_add does not cover vLLM LoRA layer {name}")
            if name not in loaded:  # eager (adapter loading), outside compiled graphs
                loaded.add(name)
                write_report("vllm", patched="BaseLinearLayerWithLoRA.apply", adapter_layer_classes=sorted(loaded))
            return original_set_lora(self, *args, **kwargs)

        return set_lora

    for cls in [BaseLayerWithLoRA, *covered, *uncovered]:
        if "set_lora" in cls.__dict__:
            cls.set_lora = make_set_lora(cls.__dict__["set_lora"])
    BaseLinearLayerWithLoRA.apply = apply
    BaseLinearLayerWithLoRA._lora_fp32_add = True
    write_report("vllm", patched="BaseLinearLayerWithLoRA.apply", adapter_layer_classes=[])
    logger.info(
        f"lora_fp32_add: patched vLLM BaseLinearLayerWithLoRA.apply ({len(covered)} covered, "
        f"{len(uncovered)} guarded LoRA layer classes)"
    )


def register_vllm_plugin() -> None:
    """vLLM general plugin (loaded in every vLLM process); patches only when the environment variable is set."""
    if enabled():
        patch_vllm()
