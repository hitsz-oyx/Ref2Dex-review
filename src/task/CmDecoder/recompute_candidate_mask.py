"""Recompute 5cm object-surface candidate masks for an existing geometry cache.

The current surface-only cache intentionally stores an all-true compatibility
bitmap.  This utility restores the historical frame-level 5cm criterion as a
sidecar without touching the cache's active ``obj_candidate_mask_5cm.npy``.
It is safe to run while a loader is using the cache; the sidecar is committed
atomically per episode after all frames have been computed.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def _mask_cpu(hand: np.ndarray, obj: np.ndarray, threshold_m: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    result = np.empty((hand.shape[0], obj.shape[1]), dtype=np.bool_)
    for frame in range(hand.shape[0]):
        result[frame] = cKDTree(np.asarray(hand[frame])).query_ball_point(
            np.asarray(obj[frame]), r=float(threshold_m), return_length=True
        ) > 0
    return result


def _mask_gpu(hand: np.ndarray, obj: np.ndarray, threshold_m: float, *, device: str, frame_batch: int) -> np.ndarray:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    result = np.empty((hand.shape[0], obj.shape[1]), dtype=np.bool_)
    torch_device = torch.device(device)
    with torch.inference_mode():
        for start in range(0, hand.shape[0], int(frame_batch)):
            stop = min(hand.shape[0], start + int(frame_batch))
            # Copies are intentional: mmap arrays are read-only and torch must
            # not retain a tensor with an unsafe writable flag.
            hand_t = torch.from_numpy(np.asarray(hand[start:stop], dtype=np.float32).copy()).to(torch_device)
            obj_t = torch.from_numpy(np.asarray(obj[start:stop], dtype=np.float32).copy()).to(torch_device)
            nearest = torch.cdist(obj_t, hand_t).amin(dim=-1)
            result[start:stop] = (nearest <= float(threshold_m)).cpu().numpy()
    return result


def recompute_episode(geometry: Path, *, threshold_m: float, device: str, frame_batch: int, output_name: str) -> tuple[int, int, float]:
    hand = np.load(geometry / "hand_points_world.npy", mmap_mode="r")
    obj = np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r")
    if hand.ndim != 3 or obj.ndim != 3 or hand.shape[0] != obj.shape[0] or hand.shape[-1] != 3 or obj.shape[-1] != 3:
        raise ValueError(f"Invalid hand/object shapes at {geometry}: hand={hand.shape}, obj={obj.shape}")
    if device.startswith("cuda"):
        mask = _mask_gpu(hand, obj, threshold_m, device=device, frame_batch=frame_batch)
    else:
        mask = _mask_cpu(hand, obj, threshold_m)
    destination = geometry / output_name
    temporary = geometry / f".{output_name}.tmp.npy"
    np.save(temporary, mask, allow_pickle=False)
    os.replace(temporary, destination)
    return int(mask.shape[0]), int(mask.sum()), float(mask.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Geometry cache root containing episodes/*/geometry")
    parser.add_argument("--threshold-m", type=float, default=0.05)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--frame-batch", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional limit for a smoke run")
    parser.add_argument("--output-name", default="obj_candidate_mask_5cm_recomputed.npy")
    args = parser.parse_args()
    if args.threshold_m <= 0.0 or args.frame_batch <= 0:
        raise ValueError("threshold and frame batch must be positive")
    geometries = sorted(path for path in args.root.glob("episodes/*/geometry") if path.is_dir())
    if not geometries:
        raise FileNotFoundError(f"No episode geometry directories under {args.root}")
    if args.max_episodes is not None:
        geometries = geometries[: max(0, int(args.max_episodes))]
        if not geometries:
            raise ValueError("--max-episodes selected no episodes")
    started = time.time()
    total_frames = total_active = 0
    for index, geometry in enumerate(geometries, start=1):
        frames, active, ratio = recompute_episode(
            geometry,
            threshold_m=float(args.threshold_m),
            device=str(args.device),
            frame_batch=int(args.frame_batch),
            output_name=str(args.output_name),
        )
        total_frames += frames
        total_active += active
        print(
            f"[{index}/{len(geometries)}] {geometry.parent.name}: "
            f"frames={frames} active={active} ratio={ratio:.4f} "
            f"elapsed_min={(time.time() - started) / 60.0:.1f}",
            flush=True,
        )
    print(
        f"completed episodes={len(geometries)} frames={total_frames} "
        f"active_points={total_active} mean_point_ratio={total_active / max(1, total_frames * 4096):.6f} "
        f"elapsed_min={(time.time() - started) / 60.0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
