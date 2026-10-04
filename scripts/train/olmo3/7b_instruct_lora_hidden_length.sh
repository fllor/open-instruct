#!/bin/bash
# Experiment 1 (hidden length target): LoRA GRPO of Olmo 3 7B Instruct on a single GPU, fully synchronous.
# Reward: -|log(words / (c * L_x))| with a per-prompt target the prompt never states (HiddenLengthVerifier).
# Data comes from olmo/tools/make_hidden_length_data.py.
#   C=0.6 STEPS=10 EXP=hl_short_pilot bash scripts/train/olmo3/7b_instruct_lora_hidden_length.sh
# 8 prompts x 8 rollouts per step. Adapters land in /workspace/runs/<run_name>/lora_adapters/step_N,
# training rollouts (--save_traces) in /workspace/runs/<EXP>/rollouts.
set -euo pipefail

C=${C:-0.6}
STEPS=${STEPS:-10}
LR=${LR:-1e-5}
SEED=${SEED:-1}
EXP=${EXP:-hl_c${C}_${STEPS}steps}
DATA=${DATA:-/workspace/data/hidden_length}

export TORCH_COMPILE_DISABLE=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.0.0+debug
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

uv run --active open_instruct/grpo_fast.py \
    --exp_name "$EXP" \
    --output_dir /workspace/runs \
    --rollouts_save_path "/workspace/runs/$EXP/rollouts" \
    --model_name_or_path allenai/Olmo-3-7B-Instruct-DPO \
    --chat_template_name tokenizer_default \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 64 \
    --dataset_mixer_list "$DATA/train_c${C}.jsonl" 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list "$DATA/readout.jsonl" 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 1536 \
    --pack_length 2048 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --temperature 1.0 \
    --learning_rate "$LR" \
    --lr_scheduler_type constant \
    --total_episodes $((STEPS * 64)) \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_enforce_eager \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --beta 0.0 \
    --load_ref_policy false \
    --async_steps 0 \
    --inflight_updates false \
    --gradient_checkpointing \
    --seed "$SEED" \
    --local_eval_every 100000 \
    --save_freq 100000 \
    --save_traces \
    --push_to_hub false "$@"
