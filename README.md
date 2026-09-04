# HELM

HELM is a humanoid vision-language-action policy with a world-action context backbone, a vision-language
feature extractor, and a diffusion action head. This
repository contains HELM training, control-signal inference, and RoboCasa GR1
tabletop simulation evaluation.

Supported embodiments:

- `UNITREE_G1_SONIC` for G1 demonstrations and control-signal inference.
- `ROBOCASA_GR1_TABLETOP` for RoboCasa GR1 training and simulation evaluation.

The inference API predicts action chunks.

## Getting started

- Follow [`INSTALL.md`](INSTALL.md) to create the Conda environment and install
  HELM and the optional RoboCasa simulation dependencies.
- Download the pretrained HELM checkpoint from
  [`Zhuoling98/helm_pretrain`](https://huggingface.co/Zhuoling98/helm_pretrain)
  into `data/helm_pretrain` as described in `INSTALL.md`.
- See [`DATA.md`](DATA.md) for dataset sources, access requirements, and the
  expected directory layout.

## Repository layout

```text
utils/
  configs/       HELM model, data, and training configuration
  data/          LeRobot episode loading, sharding, and normalization
  eval/          RoboCasa GR1 rollout evaluation
  experiment/    distributed training and checkpoint management
  model/         HELM backbone, action head, and context module
  policy/        HelmPolicy and the control-signal server
  vla/           Internal WAM implementation modules
scripts/         supported training, inference, and evaluation commands
```

The public policy class is `utils.policy.HelmPolicy`. Internal package paths are
implementation details and do not require additional pretrained repositories.

## Data

Dataset download instructions are maintained in [`DATA.md`](DATA.md). Each
downstream dataset must use LeRobot v2 format and contain its metadata, Parquet
trajectory data, and video files. HELM reads modality keys from the selected
embodiment config.

`UNITREE_G1_SONIC` expects:

- video: `ego_view`
- state: legs, waist, arms, hands, and projected gravity
- action: `motion_token`, `left_hand_joints`, and `right_hand_joints`
- language: `annotation.human.task_description`

`ROBOCASA_GR1_TABLETOP` expects:

- video: `ego_view_bg_crop_pad_res256_freq20`
- state/action: left arm, right arm, left hand, right hand, and waist
- language: `remarks`

## Cache text embeddings

HELM uses frozen WAM text embeddings. Create the cache before training:

```bash
python scripts/cache_text_embeddings.py \
  --wam-checkpoint data/helm_pretrain/wam \
  --dataset-path /path/to/lerobot/data \
  --output-path data/helm_pretrain/text_embeddings.pt
```

The cache is stored at `data/helm_pretrain/text_embeddings.pt`. The G1 Sonic
launcher creates it automatically when absent; the RoboCasa launcher expects a
cache covering all selected task datasets.

The pretrained bundle from
[`Zhuoling98/helm_pretrain`](https://huggingface.co/Zhuoling98/helm_pretrain)
must have this shape after downloading:

```text
data/helm_pretrain/
  config.json                 # HELM model configuration
  model-*.safetensors         # HELM checkpoint (including the action head)
  vlm/                        # VLM files
  wam/                        # WAM files
  action_head/                # action-head files
  text_embeddings.pt          # cached WAM text features
```

Validate it with `python scripts/validate_pretrain_bundle.py`. Training and
evaluation accept only `PRETRAIN_PATH`; do not provide separate pretrained
component paths.

## Train

Train G1 Sonic:

```bash
MAX_STEPS=5000 \
NUM_GPUS=8 \
BATCH_SIZE=64 \
GRADIENT_ACCUMULATION_STEPS=1 \
EXPERIMENT_NAME=helm_g1_sonic \
DATA_ROOT=/path/to/g1_sonic_data \
PRETRAIN_PATH=data/helm_pretrain \
bash scripts/train_helm_g1_sonic.sh
```

Train RoboCasa GR1 tabletop on every dataset directory under `DATA_ROOT`:

```bash
MAX_STEPS=60000 \
NUM_GPUS=8 \
BATCH_SIZE=64 \
GRADIENT_ACCUMULATION_STEPS=1 \
EXPERIMENT_NAME=helm_robocasa_unified \
DATA_ROOT=/path/to/lerobot_data \
PRETRAIN_PATH=data/helm_pretrain \
bash scripts/train_helm_robocasa.sh
```

Both launchers accept environment-variable overrides for GPU count, distributed
launch settings, batch size, sampling, LoRA rank, learning rate, checkpointing,
and output paths. Extra CLI arguments are forwarded to the training command.

## Control-signal inference

Load the policy directly:

```python
from utils.policy import HelmPolicy

policy = HelmPolicy(
    embodiment_tag="UNITREE_G1_SONIC",
    model_path="outputs/helm_g1_sonic/checkpoint-20000",
    device="cuda:0",
    pretrain_path="data/helm_pretrain",
)

action_chunk, info = policy.get_action(observation)
```

`observation` is a batched dictionary with `video`, `state`, and `language`
sub-dictionaries. Video arrays have shape `(B, T, H, W, C)`, state arrays have
shape `(B, T, D)`, and language values have shape `(B, T)`. The returned action
dictionary contains float32 control-signal chunks with shape `(B, horizon, D)`.

To keep model dependencies on a GPU server:

```bash
python scripts/serve_helm.py \
  --model-path outputs/helm_g1_sonic/checkpoint-20000 \
  --embodiment-tag UNITREE_G1_SONIC \
  --pretrain-path data/helm_pretrain
```

Clients use `utils.policy.server_client.PolicyClient`. The server only accepts
observations and returns predicted control signals; it never commands hardware.

## RoboCasa evaluation

Evaluate one task:

```bash
MODEL_PATH=outputs/helm_robocasa/checkpoint-60000 \
PRETRAIN_PATH=data/helm_pretrain \
bash scripts/eval_helm.sh
```

Evaluate the configured seen and unseen-appearance suites:

```bash
MODEL_PATH=outputs/helm_robocasa/checkpoint-60000 \
PRETRAIN_PATH=data/helm_pretrain \
bash scripts/eval_helm_suite.sh
```

The suite command writes per-task results to `summary.csv` and aggregated
success rates to `metrics.csv`.

## Validation

Run formatting and static checks:

```bash
ruff format --check utils scripts
ruff check utils scripts
python -m compileall -q utils scripts
```

The GPU benchmark consumes one recorded demonstration step and measures model
latency without data-loading time:

```bash
python scripts/benchmark_helm.py --help
```

## License

Apache License 2.0. Retained source files preserve their original NVIDIA
copyright and license headers.
