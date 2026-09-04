#!/usr/bin/env python
"""Validate the single-directory HELM pretraining bundle."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils.pretrain_bundle import resolve_pretrain_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/helm_pretrain")
    args = parser.parse_args()
    bundle = resolve_pretrain_bundle(args.path)
    print(f"Valid HELM pretrain bundle: {bundle['root']}")
    for name in ("vlm", "wam", "action_head", "text_embeddings"):
        print(f"  {name}: {bundle[name]}")


if __name__ == "__main__":
    main()
