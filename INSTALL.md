# HELM Environment Installation

This guide installs the environment used by the HELM policy for training,
control-signal inference, and RoboCasa GR1 tabletop simulation evaluation.
The commands below install the environment used by HELM training, inference,
and simulation evaluation.

## Prerequisites

- Linux x86_64 with an NVIDIA GPU and a working driver
- Conda (or Miniconda/Miniforge)
- Git and Git LFS (`sudo apt-get install -y git git-lfs && git lfs install`)
- `sudo` access for the RoboCasa EGL/GL packages

HELM targets Python 3.10 and CUDA 12.8. The pinned FlashAttention wheel is
for CPython 3.10 on Linux x86_64. Use a clean Conda environment so that the
CUDA and PyTorch versions remain consistent.

## Create the Conda environment

```bash
conda create -n helm python=3.10 -y
conda activate helm

pip install --upgrade pip setuptools wheel
# torchcodec 0.4.0 supports FFmpeg 4-7; do not install the FFmpeg 8 default.
conda install -c conda-forge "ffmpeg=7.*" -y

conda install -c nvidia/label/cuda-12.8.0 cuda-nvcc=12.8 cuda-toolkit=12.8 cuda-cudart-dev=12.8
conda install -c conda-forge gcc_linux-64=11 cudnn=8 libstdcxx-ng=12
```

Install the CUDA 12.8 PyTorch build:

```bash
pip install torch==2.7.1 torchvision==0.22.1 triton==3.3.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the Python dependencies used by training, inference, and evaluation:

```bash
pip install \
  albumentations==1.4.18 "huggingface-hub[cli]" opencv-python-headless \
  av==16.1.0 diffusers==0.35.1 accelerate hydra-core dm-tree lmdb==1.7.5 \
  msgpack==1.1.0 msgpack-numpy==0.4.8 pandas==2.2.3 peft==0.17.1 \
  termcolor==3.2.0 transformers==4.57.3 tyro==0.9.17 click==8.1.8 \
  datasets==3.6.0 "cryptography>=44.0.0" einops==0.8.1 gitpython==3.1.46 \
  jsonlines==4.0.0 gymnasium==1.2.2 matplotlib==3.10.1 numpy==1.26.4 \
  omegaconf==2.3.0 scipy==1.15.3 torchcodec==0.4.0 wandb==0.23.0 \
  pyzmq==27.0.1 safetensors tqdm pillow sentencepiece tiktoken tensorboard
```

If FFmpeg was already installed without a version pin, repair it with:

```bash
conda activate helm
conda install -c conda-forge "ffmpeg=7.*" -y
```

Install the pinned FlashAttention wheel before installing HELM itself:

```bash
pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

From the HELM checkout, install the package without resolving dependencies a
second time:

```bash
pip install -e . --no-deps
```

The repository also contains a `uv` configuration. After the Conda setup,
`uv sync --all-extras` is an alternative for reproducing the Python package
environment, but it may resolve/download the CUDA and FlashAttention wheels
again.

## Check the installation

```bash
python - <<'PY'
import torch
import utils

print("HELM import: OK")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
PY
```

## Models and demonstration data

Place the HELM pretrained artifact under `data/`:

```text
data/
  helm_pretrain/
    config.json
    model-*.safetensors
    vlm/
    wam/
    action_head/
    text_embeddings.pt
```

The demonstration root must contain LeRobot v2 metadata, parquet trajectory
files, and videos. See the repository README for the supported modality keys
for `UNITREE_G1_SONIC` and `ROBOCASA_GR1_TABLETOP`.

If a provided bundle does not already cover your dataset language strings,
regenerate its frozen WAM text features:

```bash
python scripts/cache_text_embeddings.py \
  --wam-checkpoint data/helm_pretrain/wam \
  --dataset-path /path/to/lerobot_dataset \
  --output-path data/helm_pretrain/text_embeddings.pt
```

The script also accepts a parent directory containing multiple LeRobot
dataset folders. Use `--device cuda` (the default) on a GPU, or select another
PyTorch device explicitly.

## RoboCasa GR1 tabletop (simulation only)

The RoboCasa dependency is needed only for the
`ROBOCASA_GR1_TABLETOP` simulation evaluation. Clone the pinned task
repository and install its dependencies inside the `helm` environment:

```bash
mkdir -p external_dependencies
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks \
  external_dependencies/robocasa-gr1-tabletop-tasks
git -C external_dependencies/robocasa-gr1-tabletop-tasks \
  checkout 4840e671596f93ca03651524b9f72ffb1aadfeff

sudo apt-get install -y libegl1-mesa-dev libglu1-mesa
CFLAGS="-DKEY_LINK_PHONE=0x1bf" pip install --no-cache-dir evdev
pip install "git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1"
pip install -e external_dependencies/robocasa-gr1-tabletop-tasks
pip install gymnasium==0.29.1 pydantic av==15.0.0 pyzmq transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 tyro
```

Download the tabletop and kitchen assets into the RoboCasa checkout:

```bash
python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y
python external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_kitchen_assets.py -y
```

The evaluation scripts use this checkout in place; no robot driver or control
loop is included. HELM inference only returns predicted control-signal chunks
to the caller.

## Next steps

Return to [`README.md`](README.md) for the training, inference, and RoboCasa
evaluation commands.

Download or copy the complete `helm_pretrain/` artifact into `data/`. It is a
self-contained bundle of real files; no additional pretrained model folders are
needed. Validate it with:

```bash
python scripts/validate_pretrain_bundle.py data/helm_pretrain
```

When launching training, set `NUM_GPUS` to the number of GPUs available to the
job. The launcher defaults to one process and checks the visible GPU count
before starting distributed training.
