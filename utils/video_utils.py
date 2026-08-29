import math
from typing import List, Optional, Tuple

import numpy as np


def _video_decoder(video_path: str):
    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "HELM requires a working torchcodec installation. "
            "Install torchcodec==0.4.0 with FFmpeg 4-7 (for example: "
            'conda install -c conda-forge "ffmpeg=7.*" -y). '
            f"Original error: {exc}"
        ) from exc
    return VideoDecoder(
        video_path,
        device="cpu",
        dimension_order="NHWC",
        num_ffmpeg_threads=0,
    )


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "torchcodec",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    """Decode selected frames with the only HELM-supported video backend."""
    del video_backend_kwargs
    if video_backend != "torchcodec":
        raise ValueError(f"HELM supports only video_backend='torchcodec', got {video_backend!r}")
    return _video_decoder(video_path).get_frames_at(indices=indices).data.numpy()


def get_accumulate_timestamp_idxs(
    timestamps: List[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative: bool = False,
) -> Tuple[List[int], List[int], int]:
    """Map irregular render timestamps onto fixed-rate output-frame indices."""
    local_idxs: list[int] = []
    global_idxs: list[int] = []
    for local_idx, timestamp in enumerate(timestamps):
        global_idx = math.floor((timestamp - start_time) / dt + eps)
        if not allow_negative and global_idx < 0:
            continue
        if next_global_idx is None:
            next_global_idx = global_idx
        repeats = max(0, global_idx - next_global_idx + 1)
        local_idxs.extend([local_idx] * repeats)
        global_idxs.extend(range(next_global_idx, next_global_idx + repeats))
        next_global_idx += repeats
    return local_idxs, global_idxs, next_global_idx
