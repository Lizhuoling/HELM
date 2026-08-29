#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

EXP_ID=${EXP_ID:-helm_robocasa}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-60000}
MODEL_PATH=${MODEL_PATH:-outputs/${EXP_ID}/checkpoint-${CHECKPOINT_STEP}}
PRETRAIN_PATH=${PRETRAIN_PATH:-data/helm_pretrain}
python scripts/validate_pretrain_bundle.py "$PRETRAIN_PATH"
N_EPISODES=${N_EPISODES:-20}
N_ENVS=${N_ENVS:-5}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
EVAL_DIR=${EVAL_DIR:-outputs/${EXP_ID}/eval_gr1_tabletop_all}
SUMMARY_CSV=${SUMMARY_CSV:-"${EVAL_DIR}/summary.csv"}
METRICS_CSV=${METRICS_CSV:-"${EVAL_DIR}/metrics.csv"}

SEEN_ARTICULATED_TASKS=(
  "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
)

SEEN_PICK_PLACE_TASKS=(
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
)

UNSEEN_APPEARANCE_TASKS=()
for env_name in "${SEEN_PICK_PLACE_TASKS[@]}"; do
  unseen_env_name=${env_name/PosttrainPnPNovel/EvalPnPNovel}
  unseen_env_name=${unseen_env_name/SplitA_/SplitB_}
  UNSEEN_APPEARANCE_TASKS+=("$unseen_env_name")
done

EXTRA_ROLLOUT_ARGS=("$@")

mkdir -p "$EVAL_DIR"
printf "suite,category,env_name,success_rate,n_episodes,log_path,video_dir\n" > "$SUMMARY_CSV"

evaluate_task() {
  local env_name=$1
  local suite=$2
  local category=$3
  local task_name task_dir log_path video_dir success_rate

  task_name=${env_name#gr1_unified/}
  task_name=${task_name%_GR1ArmsAndWaistFourierHands_Env}
  task_dir="${EVAL_DIR}/${task_name}"
  log_path="${task_dir}/rollout.log"
  video_dir="${task_dir}/videos"
  mkdir -p "$task_dir"

  echo "Evaluating ${env_name} [suite=${suite}, category=${category}]"
  if python -m utils.eval.rollout_policy \
    --model-path "$MODEL_PATH" \
    --pretrain-path "$PRETRAIN_PATH" \
    --env-name "$env_name" \
    --n-episodes "$N_EPISODES" \
    --n-action-steps "$N_ACTION_STEPS" \
    --n-envs "$N_ENVS" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --video-dir "$video_dir" \
    "${EXTRA_ROLLOUT_ARGS[@]}" > "$log_path" 2>&1; then
    cat "$log_path"
    success_rate=$(awk '/success rate:/ {print $NF}' "$log_path" | tail -1)
    success_rate=${success_rate:-nan}
  else
    cat "$log_path"
    success_rate="nan"
  fi

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$suite" "$category" "$env_name" "$success_rate" "$N_EPISODES" \
    "$log_path" "$video_dir" >> "$SUMMARY_CSV"
}

for env_name in "${SEEN_ARTICULATED_TASKS[@]}"; do
  evaluate_task "$env_name" "seen" "articulated"
done

for env_name in "${SEEN_PICK_PLACE_TASKS[@]}"; do
  evaluate_task "$env_name" "seen" "pick_place"
done

for env_name in "${UNSEEN_APPEARANCE_TASKS[@]}"; do
  evaluate_task "$env_name" "unseen_appearance" "pick_place"
done

python - "$SUMMARY_CSV" "$METRICS_CSV" <<'PY'
import csv
import math
import sys

summary_path = sys.argv[1]
metrics_path = sys.argv[2]
rows = []
with open(summary_path, newline="") as f:
    for row in csv.DictReader(f):
        try:
            rate = float(row["success_rate"])
        except ValueError:
            rate = math.nan
        row["success_rate"] = rate
        rows.append(row)

if not any(math.isfinite(row["success_rate"]) for row in rows):
    raise SystemExit(f"No finite success rates found in {summary_path}")

metric_filters = (
    ("pick_place", lambda row: row["suite"] == "seen" and row["category"] == "pick_place"),
    ("articulated", lambda row: row["suite"] == "seen" and row["category"] == "articulated"),
    ("seen", lambda row: row["suite"] == "seen"),
    ("unseen_appearance", lambda row: row["suite"] == "unseen_appearance"),
)

metric_rows = []
for metric_name, predicate in metric_filters:
    selected = [row for row in rows if predicate(row)]
    valid_rates = [row["success_rate"] for row in selected if math.isfinite(row["success_rate"])]
    mean_rate = sum(valid_rates) / len(valid_rates) if valid_rates else math.nan
    metric_rows.append(
        {
            "metric": metric_name,
            "success_rate": mean_rate,
            "n_valid_tasks": len(valid_rates),
            "n_expected_tasks": len(selected),
        }
    )

with open(metrics_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=("metric", "success_rate", "n_valid_tasks", "n_expected_tasks"),
    )
    writer.writeheader()
    writer.writerows(metric_rows)

print(f"Summary CSV: {summary_path}")
print(f"Metrics CSV: {metrics_path}")
for row in metric_rows:
    print(
        f"{row['metric']} success rate: {row['success_rate']:.6f} "
        f"({row['n_valid_tasks']}/{row['n_expected_tasks']} valid tasks)"
    )
PY
