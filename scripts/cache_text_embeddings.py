#!/usr/bin/env python
"""Cache frozen WAM text encoder outputs for one or more LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data.language_utils import (  # noqa: E402
    normalize_language_text,
    resolve_dataset_language_key,
)
from utils.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from utils.data.types import ModalityConfig  # noqa: E402
from utils.model.wam_vla_lora.wam_context_module import instantiate  # noqa: E402


LOGGER = logging.getLogger("helm_text_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wam-checkpoint",
        "--cleaned-checkpoint",
        default="data/helm_pretrain/wam",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/g1_data_20260706",
    )
    parser.add_argument(
        "--output-path",
        default="data/helm_pretrain/text_embeddings.pt",
    )
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument(
        "--include-empty-text",
        action="store_true",
        help="Also cache empty remarks strings from episodes.jsonl.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_wam_config(wam_checkpoint: Path) -> dict[str, Any]:
    with (wam_checkpoint / "cleaned_config.json").open("r") as f:
        return json.load(f)


def is_lerobot_dataset_dir(path: Path) -> bool:
    meta_dir = path / "meta"
    return (
        path.is_dir()
        and (meta_dir / "info.json").is_file()
        and (meta_dir / "modality.json").is_file()
        and (meta_dir / "episodes.jsonl").is_file()
        and (meta_dir / "tasks.jsonl").is_file()
    )


def resolve_dataset_dirs(dataset_path: Path) -> list[Path]:
    if is_lerobot_dataset_dir(dataset_path):
        return [dataset_path]

    dataset_dirs = sorted(
        path.parent.parent
        for path in dataset_path.glob("*/meta/info.json")
        if is_lerobot_dataset_dir(path.parent.parent)
    )
    dataset_dirs = sorted(set(dataset_dirs))
    if dataset_dirs:
        return dataset_dirs

    raise FileNotFoundError(
        f"No LeRobot dataset found at {dataset_path}. Expected either a dataset root containing "
        "meta/info.json, or a parent directory containing subdataset folders."
    )


def collect_task_texts_from_episode_meta(
    dataset_dir: Path,
    *,
    include_empty_text: bool,
) -> set[str]:
    texts: set[str] = set()
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    with episodes_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for task in record.get("tasks", []):
                normalized = str(task).strip()
                if normalized or include_empty_text:
                    texts.add(normalized)
    return texts


def collect_episode_meta_texts(
    dataset_dir: Path,
    *,
    field: str,
    include_empty_text: bool,
) -> set[str]:
    texts: set[str] = set()
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    with episodes_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            normalized = normalize_language_text(record.get(field, ""))
            if normalized or include_empty_text:
                texts.add(normalized)
    return texts


def collect_unique_texts(dataset_path: Path, *, include_empty_text: bool) -> list[str]:
    texts: set[str] = set()
    dataset_dirs = resolve_dataset_dirs(dataset_path)
    LOGGER.info("Scanning %d dataset(s) under %s", len(dataset_dirs), dataset_path)

    for dataset_dir in dataset_dirs:
        language_key = resolve_dataset_language_key(dataset_dir)
        if language_key in {"remarks", "description"}:
            texts.update(
                collect_episode_meta_texts(
                    dataset_dir,
                    field=language_key,
                    include_empty_text=include_empty_text,
                )
            )
            continue
        if language_key == "task":
            texts.update(
                collect_task_texts_from_episode_meta(
                    dataset_dir,
                    include_empty_text=include_empty_text,
                )
            )
            continue

        loader = LeRobotEpisodeLoader(
            dataset_path=dataset_dir,
            modality_configs={
                "language": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=[language_key],
                )
            },
            video_backend="torchcodec",
        )

        col = f"language.{language_key}"
        for episode_index in range(len(loader)):
            df = loader[episode_index]
            for text in df[col]:
                normalized = normalize_language_text(text)
                if normalized or include_empty_text:
                    texts.add(normalized)
    return sorted(texts)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    checkpoint = Path(args.wam_checkpoint)
    config = load_wam_config(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", trust_remote_code=True)

    texts = collect_unique_texts(Path(args.dataset_path), include_empty_text=args.include_empty_text)
    if not texts:
        raise RuntimeError("No language strings found in dataset.")
    LOGGER.info("Found %d unique language strings: %s", len(texts), texts)

    text_encoder = instantiate(config["text_encoder_config"])
    text_encoder.load_state_dict(
        load_file(checkpoint / config["kept_components"]["text_encoder"]),
        strict=True,
    )
    text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()

    feature_list = []
    mask_list = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Encoding text"):
            tokenized = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=args.text_len,
                return_tensors="pt",
            )
            token_ids = tokenized["input_ids"].to(device=device)
            attention_mask = tokenized["attention_mask"].to(device=device)
            features = text_encoder(token_ids, attention_mask).to(dtype=dtype)
            feature_list.append(features[0].detach().cpu())
            mask_list.append(attention_mask[0].detach().cpu())

    output_path = Path(args.output_path)
    if output_path.exists() and output_path.is_dir():
        output_path = output_path / "text_embeddings.pt"
    elif str(args.output_path).endswith("/"):
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "text_embeddings.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_to_index": {text: index for index, text in enumerate(texts)},
            "texts": texts,
            "text_features": torch.stack(feature_list, dim=0),
            "attention_mask": torch.stack(mask_list, dim=0),
            "text_len": args.text_len,
            "dtype": args.dtype,
            "source_checkpoint": str(checkpoint),
            "source_dataset": str(args.dataset_path),
        },
        output_path,
    )
    LOGGER.info("Saved text embedding cache to %s", output_path)


if __name__ == "__main__":
    main()
