#!/bin/bash
# Downsampled LoRA GRPO run of Olmo 3 7B Instruct on a single GPU (vLLM and learner share it).
# 8 prompts x 8 rollouts per step; total_episodes=128 -> 2 training steps, so one adapter swap
# in vLLM is exercised after the first update. Extra args are appended (e.g. --total_episodes 64).
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.0.0+debug
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

uv run --active open_instruct/grpo_fast.py \
    --exp_name olmo3_7b_lora_debug \
    --output_dir /workspace/runs \
    --model_name_or_path allenai/Olmo-3-7B-Instruct-DPO \
    --chat_template_name tokenizer_default \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 64 \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --total_episodes 128 \
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
    --seed 1 \
    --local_eval_every 1000 \
    --save_freq 1000 \
    --save_traces \
    --push_to_hub false "$@"
