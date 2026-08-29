#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/sanitize_cuda_env.sh"
sanitize_cuda_env

DATA_ROOT=${DATA_ROOT:-data/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot}
DATASET_PATH=${DATASET_PATH:-$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | paste -sd:)}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-helm_robocasa}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
PRETRAIN_PATH=${PRETRAIN_PATH:-data/helm_pretrain}
WAM_MODEL_PATH="$PRETRAIN_PATH/wam"
TEXT_EMBEDDING_CACHE="$PRETRAIN_PATH/text_embeddings.pt"

python scripts/validate_pretrain_bundle.py "$PRETRAIN_PATH"

NUM_GPUS=${NUM_GPUS:-${MLP_WORKER_GPU:-1}}
NNODES=${NNODES:-${MLP_WORKER_NUM:-1}}
NODE_RANK=${NODE_RANK:-${MLP_ROLE_INDEX:-0}}
MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST:-"127.0.0.1"}}
MASTER_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}
MAX_STEPS=${MAX_STEPS:-60000}
SAVE_STEPS=${SAVE_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((BATCH_SIZE * NUM_GPUS))}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
SHARD_SIZE=${SHARD_SIZE:-1024}
NUM_SHARDS_PER_EPOCH=${NUM_SHARDS_PER_EPOCH:-100000}
EPISODE_SAMPLING_RATE=${EPISODE_SAMPLING_RATE:-0.1}
TASK_EPISODE_RATIO=${TASK_EPISODE_RATIO:-1.0}
USE_WANDB=${USE_WANDB:-0}
WAM_NUM_CONTEXT_TOKENS=${WAM_NUM_CONTEXT_TOKENS:-8}
WAM_LORA_RANK=${WAM_LORA_RANK:-4}
WAM_LORA_ALPHA=${WAM_LORA_ALPHA:-4}
WAM_LORA_DROPOUT=${WAM_LORA_DROPOUT:-0.0}
AUTO_RESUME=${AUTO_RESUME:-0}

WANDB_FLAG=()
if [ "$USE_WANDB" = "1" ]; then
  WANDB_FLAG+=(--use_wandb)
fi

RESUME_FLAG=(--no-auto-resume-from-output-dir)
if [ "$AUTO_RESUME" = "1" ]; then
  RESUME_FLAG=(--auto-resume-from-output-dir)
fi

if [ -z "$DATASET_PATH" ]; then
  echo "No dataset directories found under $DATA_ROOT" >&2
  exit 1
fi

python -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --tee=3 \
  --module utils.experiment.launch_finetune \
  --pretrain_path "$PRETRAIN_PATH" \
  --dataset_path "$DATASET_PATH" \
  --embodiment_tag ROBOCASA_GR1_TABLETOP \
  --num_gpus "$NUM_GPUS" \
  --output_dir "$OUTPUT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 1 \
  "${RESUME_FLAG[@]}" \
  --max_steps "$MAX_STEPS" \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --learning_rate 1e-4 \
  "${WANDB_FLAG[@]}" \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --shard_size "$SHARD_SIZE" \
  --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH" \
  --episode_sampling_rate "$EPISODE_SAMPLING_RATE" \
  --task_episode_ratio "$TASK_EPISODE_RATIO" \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --load_bf16 \
  --wam_num_context_tokens "$WAM_NUM_CONTEXT_TOKENS" \
  --wam_lora_rank "$WAM_LORA_RANK" \
  --wam_lora_alpha "$WAM_LORA_ALPHA" \
  --wam_lora_dropout "$WAM_LORA_DROPOUT" \
  "$@"
