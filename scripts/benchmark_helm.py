"""
python scripts/benchmark_helm.py \
  --model-path outputs/helm_robocasa/ \
  --compile-action-head
Results
  mean inference latency: 136.37 ms
  max GPU memory allocated: 18.09 GiB
  max GPU memory reserved:  18.45 GiB
  module latency mean:
    backbone:     43.11 ms
    WAM context:  74.63 ms
    action head:  63.19 ms
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from utils.data.dataset.sharded_single_step_dataset import extract_step_data
from utils.data.embodiment_tags import EmbodimentTag
from utils.data.types import MessageType, VLAStepData
from utils.policy.policy_core import HelmPolicyBase
from utils.policy.helm_policy import HelmPolicy


def _rec_to_device_dtype(x: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.to(device=device)
        if torch.is_floating_point(x):
            x = x.to(dtype=dtype)
        return x
    if isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_device_dtype(v, device, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_rec_to_device_dtype(v, device, dtype) for v in x]
    return x


def _repeat_tensor_batch(x: Any, batch_size: int) -> Any:
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x
        repeats = [batch_size] + [1] * (x.ndim - 1)
        return x.repeat(*repeats)
    if isinstance(x, dict) or hasattr(x, "items"):
        return {k: _repeat_tensor_batch(v, batch_size) for k, v in x.items()}
    if isinstance(x, list):
        return x * batch_size
    if isinstance(x, np.ndarray):
        return np.repeat(x, batch_size, axis=0)
    return x


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def _make_observation(
    policy: HelmPolicyBase,
    dataset_path: str,
    embodiment_tag: EmbodimentTag,
    episode_index: int,
    step_index: int,
    video_backend: str,
    allow_padding: bool,
) -> dict[str, Any]:
    modality_config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend=video_backend,
    )
    episode_data = dataset[episode_index]
    step_data = extract_step_data(
        episode_data,
        step_index=step_index,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        allow_padding=allow_padding,
    )

    language_key = modality_config["language"].modality_keys[0]
    return {
        "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},
        "state": {k: step_data.states[k][None] for k in step_data.states},
        "language": {language_key: [[step_data.text]]},
    }


def _prepare_model_inputs(
    policy: HelmPolicyBase,
    observation: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[str, Any]:
    unbatched_observations = policy._unbatch_observation(observation)
    processed_inputs = []

    for obs in unbatched_observations:
        vla_step_data = VLAStepData(
            images=obs["video"],
            states=obs["state"],
            actions={},
            text=obs["language"][policy.language_key][0],
            embodiment=policy.embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    model_inputs = dict(policy.collate_fn(processed_inputs))
    model_inputs = _rec_to_device_dtype(model_inputs, device=device, dtype=dtype)
    if batch_size > 1:
        model_inputs = _repeat_tensor_batch(model_inputs, batch_size)
    return model_inputs


def _run_model(policy: HelmPolicyBase, model_inputs: dict[str, Any]) -> Any:
    try:
        return policy.model.get_action(**model_inputs)
    except TypeError as exc:
        if "missing 1 required positional argument: 'inputs'" not in str(exc):
            raise
        inputs = model_inputs["inputs"] if "inputs" in model_inputs else model_inputs
        return policy.model.get_action(inputs)


def _unwrap_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    return model_inputs["inputs"] if "inputs" in model_inputs else model_inputs


def _measure_full_latency(
    policy: HelmPolicyBase,
    model_inputs: dict[str, Any],
    iterations: int,
    device: torch.device,
) -> list[float]:
    latencies_ms = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _run_model(policy, model_inputs)
        torch.cuda.synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms


def _measure_module_breakdown(
    policy: HelmPolicyBase,
    model_inputs: dict[str, Any],
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    inputs = _unwrap_inputs(model_inputs)
    backbone_ms = []
    wam_ms = []
    action_head_ms = []

    for _ in range(iterations):
        with torch.inference_mode():
            backbone_inputs, action_inputs = policy.model.prepare_input(inputs)

        start = torch.cuda.Event(enable_timing=True)
        backbone_done = torch.cuda.Event(enable_timing=True)
        wam_done = torch.cuda.Event(enable_timing=True)
        action_done = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            start.record()
            backbone_outputs = policy.model._forward_backbone(backbone_inputs)
            backbone_done.record()
            backbone_outputs = policy.model._augment_backbone_with_wam_context(
                backbone_outputs,
                action_inputs,
                inputs,
            )
            wam_done.record()
            policy.model.action_head.get_action(backbone_outputs, action_inputs)
            action_done.record()

        action_done.synchronize()
        backbone_ms.append(start.elapsed_time(backbone_done))
        wam_ms.append(backbone_done.elapsed_time(wam_done))
        action_head_ms.append(wam_done.elapsed_time(action_done))

    return {
        "backbone": sum(backbone_ms) / len(backbone_ms),
        "wam_context": sum(wam_ms) / len(wam_ms),
        "action_head": sum(action_head_ms) / len(action_head_ms),
    }


def _compile_action_head_if_requested(policy: HelmPolicyBase, args: argparse.Namespace) -> bool:
    compile_mode = None if args.compile_mode == "default" else args.compile_mode
    if not args.compile_action_head:
        return False

    print(f"Compiling DiT action head get_action with torch.compile(mode={args.compile_mode!r})...")
    policy.model.action_head.get_action = torch.compile(
        policy.model.action_head.get_action,
        mode=compile_mode,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="outputs/helm_robocasa/checkpoint-20000",
        help="Path to a trained HELM checkpoint.",
    )
    parser.add_argument(
        "--dataset-path",
        default="data/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/gr1_unified.PnPCanToDrawerClose",
        help="LeRobot dataset path used only to prepare one model input sample.",
    )
    parser.add_argument("--pretrain-path", default="data/helm_pretrain")
    parser.add_argument(
        "--embodiment-tag",
        default="ROBOCASA_GR1_TABLETOP",
        help="Embodiment tag for the checkpoint and dataset.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--breakdown-iterations",
        type=int,
        default=10,
        help="Iterations for per-module timing. Set to 0 to disable breakdown.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--compile-action-head",
        action="store_true",
        help="Compile policy.model.action_head.get_action with torch.compile before benchmarking.",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode used when --compile-action-head is set.",
    )
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--allow-padding", action="store_true")
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help="Keep PEFT LoRA layers separate instead of merging them after checkpoint loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA so GPU latency and memory can be measured.")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print("Loading policy...")
    load_start = time.perf_counter()
    policy = HelmPolicy(
        model_path=args.model_path,
        embodiment_tag=embodiment_tag,
        device=device,
        strict=True,
        vlm_model_path=f"{args.pretrain_path}/vlm",
        wam_model_path=f"{args.pretrain_path}/wam",
        wam_text_embedding_cache=f"{args.pretrain_path}/text_embeddings.pt",
        merge_lora=not args.no_merge_lora,
    )
    load_s = time.perf_counter() - load_start
    print(f"Loaded policy in {load_s:.2f}s")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Model path: {args.model_path}")
    print(f"DreamZero LoRA merged: {policy.lora_merged}")

    compiled_action_head = _compile_action_head_if_requested(policy, args)

    print("Preparing one sample outside timed region...")
    observation = _make_observation(
        policy=policy,
        dataset_path=args.dataset_path,
        embodiment_tag=embodiment_tag,
        episode_index=args.episode_index,
        step_index=args.step_index,
        video_backend=args.video_backend,
        allow_padding=args.allow_padding,
    )
    model_inputs = _prepare_model_inputs(
        policy=policy,
        observation=observation,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
    )

    print("\nBenchmark configuration")
    print("  Timed full call: policy.model.get_action only")
    print("  Data loading/preprocessing included in latency: no")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Full iterations: {args.iterations}")
    print(f"  Breakdown iterations: {args.breakdown_iterations}")
    print(f"  torch.compile action head: {compiled_action_head}")
    if compiled_action_head:
        print(f"  torch.compile mode: {args.compile_mode}")

    for _ in range(args.warmup):
        with torch.inference_mode():
            _run_model(policy, model_inputs)
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = _measure_full_latency(policy, model_inputs, args.iterations, device)
    module_breakdown = None
    if args.breakdown_iterations > 0:
        module_breakdown = _measure_module_breakdown(
            policy,
            model_inputs,
            args.breakdown_iterations,
            device,
        )

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    mean_ms = sum(latencies_ms) / len(latencies_ms)

    print("\nResults")
    print(f"  mean inference latency: {mean_ms:.2f} ms")
    print(f"  max GPU memory allocated: {_format_gib(peak_allocated)}")
    print(f"  max GPU memory reserved:  {_format_gib(peak_reserved)}")
    if module_breakdown is not None:
        print("  module latency mean:")
        print(f"    backbone:     {module_breakdown['backbone']:.2f} ms")
        print(f"    WAM context:  {module_breakdown['wam_context']:.2f} ms")
        print(f"    action head:  {module_breakdown['action_head']:.2f} ms")


if __name__ == "__main__":
    main()
