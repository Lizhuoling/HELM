# Data

HELM uses large-scale human video datasets for world-action model pretraining
and robot demonstrations for downstream policy training. Dataset licenses and
access requirements are controlled by the respective dataset owners. Review
and accept their terms before downloading or redistributing any data.

## World-action model pretraining

The pretrained world-action model uses the following public datasets. Download
each dataset from its official project page:

- [Ego4D](https://ego4d-data.org/)
- [Ego-Exo4D](https://docs.ego-exo4d-data.org/getting-started/)
- [Nymeria](https://www.projectaria.com/datasets/nymeria/)
- [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
- [EPIC-KITCHENS](https://epic-kitchens.github.io/2024)

These datasets are needed only to reproduce world-action model pretraining.
They are not inputs to the supported G1 Sonic or RoboCasa fine-tuning
launchers when using the provided `data/helm_pretrain` bundle.

## RoboCasa GR1 tabletop demonstrations

Download the RoboCasa GR1 tabletop demonstrations from the official
[PhysicalAI-Robotics-GR00T-Teleop-Sim dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim).
The HELM launcher consumes the task datasets under the release's `LeRobot/`
directory. For example, place or link the downloaded data as:

```text
data/PhysicalAI-Robotics-GR00T-Teleop-Sim/
  LeRobot/
    <task-dataset-1>/
    <task-dataset-2>/
    ...
```

Then train with:

```bash
NUM_GPUS=1 \
DATA_ROOT=data/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot \
PRETRAIN_PATH=data/helm_pretrain \
bash scripts/train_helm_robocasa.sh
```

`DATA_ROOT` must point to `LeRobot/`, whose immediate child directories are
the individual task datasets. Do not point it at the parent release directory.

## G1 demonstrations

Download the collected G1 demonstrations from the
[HELM G1 dataset on Hugging Face](https://huggingface.co/datasets/Zhuoling98/HELM_g1_data).

After downloading, point `DATASET_PATH` to the LeRobot dataset root:

```bash
NUM_GPUS=1 \
DATASET_PATH=/path/to/g1_demonstrations \
PRETRAIN_PATH=data/helm_pretrain \
bash scripts/train_helm_g1_sonic.sh
```

## Robot demonstration format

Both downstream datasets must use LeRobot v2 format. Each dataset root must
contain its metadata, Parquet trajectory data, and video files. See
[`README.md`](README.md) for the modality keys expected by each embodiment and
for text-embedding cache preparation.
