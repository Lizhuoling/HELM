# HELM

HELM is a humanoid vision-language-action policy with a vision-language
backbone, a world-action context module, and a diffusion action head. This
repository contains HELM training, control-signal inference, and RoboCasa GR1
tabletop simulation evaluation.

Supported embodiments:

- `UNITREE_G1_SONIC` for G1 demonstrations and control-signal inference.
- `ROBOCASA_GR1_TABLETOP` for RoboCasa GR1 training and simulation evaluation.

The inference API predicts action chunks. Robot drivers, safety checks, timing,
and closed-loop control are intentionally outside this repository.

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

## Requirements

- Python 3.10
- CUDA 12.8 and a CUDA GPU
- A HELM pretraining bundle at `data/helm_pretrain`
- Demonstrations in LeRobot v2 format

Install the Python environment:

```bash
uv sync --all-extras
```

For the complete Conda, CUDA, PyTorch, FlashAttention, model/data, and
RoboCasa setup sequence, see [`INSTALL.md`](INSTALL.md).

HELM requires FFmpeg 4-7 for `torchcodec==0.4.0`. If video loading reports
`Could not load libtorchcodec`, activate the `helm` environment and run
`conda install -c conda-forge "ffmpeg=7.*" -y`, then retry.

RoboCasa evaluation additionally requires the GR1 tabletop environment and its
assets. Keep it outside the HELM source tree or clone it into the ignored
`external_dependencies` directory:

```bash
mkdir -p external_dependencies
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks \
  external_dependencies/robocasa-gr1-tabletop-tasks
uv pip install "git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1"
uv pip install -e external_dependencies/robocasa-gr1-tabletop-tasks
python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/\
download_tabletop_assets.py -y
```

## Data contract

Each dataset root must contain LeRobot v2 metadata, parquet trajectory data,
and video files. HELM reads modality keys from the selected embodiment config.

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

The bundle must have this shape:

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
NUM_GPUS=1 \
DATASET_PATH=/path/to/g1_sonic_data \
PRETRAIN_PATH=data/helm_pretrain \
bash scripts/train_helm_g1_sonic.sh
```

Train RoboCasa GR1 tabletop on every dataset directory under `DATA_ROOT`:

```bash
NUM_GPUS=1 \
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
uv run python scripts/serve_helm.py \
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
uv run ruff format --check utils scripts
uv run ruff check utils scripts
python -m compileall -q utils scripts
```

The GPU benchmark consumes one recorded demonstration step and measures model
latency without data-loading time:

```bash
uv run python scripts/benchmark_helm.py --help
```

## License

Apache License 2.0. Retained source files preserve their original NVIDIA
copyright and license headers.
