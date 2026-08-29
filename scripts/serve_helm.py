#!/usr/bin/env python
"""Serve HELM control-signal inference over ZeroMQ."""

from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.policy.helm_policy import HelmPolicy
from utils.policy.server_client import PolicyServer
import tyro


@dataclass
class ServerConfig:
    model_path: str
    embodiment_tag: str
    pretrain_path: str = "data/helm_pretrain"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 5555
    strict: bool = True
    merge_lora: bool = True


def main(config: ServerConfig) -> None:
    policy = HelmPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        vlm_model_path=f"{config.pretrain_path}/vlm",
        wam_model_path=f"{config.pretrain_path}/wam",
        wam_text_embedding_cache=f"{config.pretrain_path}/text_embeddings.pt",
        merge_lora=config.merge_lora,
    )
    PolicyServer(policy=policy, host=config.host, port=config.port).run()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
