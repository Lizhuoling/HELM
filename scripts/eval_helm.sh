#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}

EXP_ID=${EXP_ID:-helm_robocasa}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-60000}
MODEL_PATH=${MODEL_PATH:-outputs/${EXP_ID}/checkpoint-${CHECKPOINT_STEP}}
PRETRAIN_PATH=${PRETRAIN_PATH:-data/helm_pretrain}
python scripts/validate_pretrain_bundle.py "$PRETRAIN_PATH"
ENV_NAME=${ENV_NAME:-gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env}
N_EPISODES=${N_EPISODES:-20}
N_ENVS=${N_ENVS:-5}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
TASK_NAME=${ENV_NAME#gr1_unified/}
TASK_NAME=${TASK_NAME%_GR1ArmsAndWaistFourierHands_Env}
VIDEO_DIR=${VIDEO_DIR:-outputs/${EXP_ID}/eval_videos/${TASK_NAME}}

python -m utils.eval.rollout_policy \
  --model-path "$MODEL_PATH" \
  --pretrain-path "$PRETRAIN_PATH" \
  --env-name "$ENV_NAME" \
  --n-episodes "$N_EPISODES" \
  --n-action-steps "$N_ACTION_STEPS" \
  --n-envs "$N_ENVS" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --video-dir "$VIDEO_DIR" \
  "$@"
