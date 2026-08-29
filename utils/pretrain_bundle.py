"""Resolve the single HELM pretraining bundle layout."""

from pathlib import Path
import json


def resolve_pretrain_bundle(path: str | Path) -> dict[str, Path]:
    root = Path(path).expanduser()
    required = {
        "root": root,
        "vlm": root / "vlm",
        "wam": root / "wam",
        "action_head": root / "action_head",
        "text_embeddings": root / "text_embeddings.pt",
    }
    missing = [name for name, item in required.items() if not item.exists()]
    if missing:
        raise FileNotFoundError(
            f"Invalid HELM pretrain bundle {root}: missing {', '.join(missing)}. "
            "Expected root checkpoint files plus vlm/, wam/, action_head/, and text_embeddings.pt."
        )
    required_files = {
        "model config": root / "config.json",
        "VLM config": required["vlm"] / "config.json",
        "WAM config": required["wam"] / "cleaned_config.json",
        "WAM weights": required["wam"] / "dit.safetensors",
        "action-head config": required["action_head"] / "config.json",
    }
    missing_files = [name for name, item in required_files.items() if not item.is_file()]
    root_weights = list(root.glob("model*.safetensors"))
    vlm_weights = list(required["vlm"].glob("model*.safetensors"))
    if not root_weights:
        missing_files.append("model weights")
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        keys = index.get("weight_map", {})
        for prefix in ("backbone.", "action_head."):
            if not any(key.startswith(prefix) for key in keys):
                missing_files.append(f"root {prefix[:-1]} tensors")
    if not vlm_weights:
        missing_files.append("VLM weights")
    if not list(required["action_head"].glob("model*.safetensors")):
        missing_files.append("action-head weights")
    if missing_files:
        raise FileNotFoundError(
            f"Invalid HELM pretrain bundle {root}: missing {', '.join(missing_files)}."
        )
    return required
