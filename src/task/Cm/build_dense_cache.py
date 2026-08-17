"""Precompute the frozen DenseToken bank (Cm Scene Cache V1, layer 3).

依照 ``src/task/Cm/docs/指导/V1.md`` §14-17：对每个 (sequence, side, current,
bank) 提前执行一次 Frozen DenseToken 前向，缓存::

    dense_bank/<side>/z_scene.npy      [T,B,512,D]    float16
    dense_bank/<side>/z_hand.npy       [T,B,1538,D]   float16
    dense_bank/<side>/hand_contact.npy [T,B,1538]     float16

cache key 是 ``sequence+side+current+bank``（§15：z_hand 也必须按 bank 分别
缓存，因为 PTv3 中 hand token 受 scene 点选择影响）。DenseToken 不知道
stride/future/delta_time（§16），所以 dense bank 不含 stride 维。

输入构造复用 ``dataset_scene.gather_frame_scene_inputs``，与在线路径逐位一致
（parity Test D）。fingerprint 绑定 DenseToken checkpoint + sampling bank +
点数配置，训练启动时不匹配即 raise，绝不静默复用旧特征（§22）。

用法::

    python -m src.task.Cm.build_dense_cache \
        --root data/processed_data/cm_scene_v1 \
        --dense-checkpoint src/task/Cm/densetoken_ckpt/best.pt \
        --bank-size 4 --batch-size 64 --dtype float16 --device cuda:0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.task.Cm.cache_schema import (
    SCHEMA_NAME,
    SceneSequenceCache,
    dense_cache_fingerprint,
    iter_sequence_dirs,
    read_meta,
    scene_cache_fingerprint,
    sha256_file,
    update_meta,
    validate_scene_root,
)
from src.task.Cm.dataset_scene import (
    _normal_world_to_hand,
    _world_to_hand,
    gather_frame_scene_inputs,
)


DTYPE_MAP = {"float16": np.float16, "float32": np.float32}


def dense_fingerprint_for_root(
    root: str | Path,
    *,
    checkpoint_sha: str,
    bank_size: int,
    num_scene_points: int = 512,
) -> str:
    meta = read_meta(root)
    scene_pool = meta["scene_pool"]
    return dense_cache_fingerprint(
        checkpoint_sha=checkpoint_sha,
        schema=SCHEMA_NAME,
        sampling_fingerprint=scene_cache_fingerprint(
            schema=SCHEMA_NAME,
            num_obj_pool=int(scene_pool["object_points"]),
            num_env_pool=int(scene_pool["environment_points"]),
            num_hand_points=int(meta["num_hand_points"]),
            candidate_threshold_m=float(meta["candidate_threshold_m"]),
            sampling_seed=int(meta.get("sampling_seed", -1)),
        ),
        num_scene_points=int(num_scene_points),
        num_hand_points=int(meta["num_hand_points"]),
        bank_size=int(bank_size),
    )


def _batch_inputs(
    cache: SceneSequenceCache,
    side_arrays: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    *,
    bank: int,
    frames: range,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack scene/hand inputs for a frame batch, exactly as the dataset does."""
    points_list, normals_list, valid_list = [], [], []
    hand_list, hand_normal_list = [], []
    for frame in frames:
        points, normals, valid = gather_frame_scene_inputs(
            cache, side_arrays, bank_indices, frame=frame, bank=bank
        )
        points_list.append(points)
        normals_list.append(normals)
        valid_list.append(valid)
        pose = np.asarray(side_arrays["hand_root_pose_world"][frame], dtype=np.float32)
        hand_list.append(
            _world_to_hand(np.asarray(side_arrays["hand_points_world"][frame], dtype=np.float32), pose)
        )
        hand_normal_list.append(
            _normal_world_to_hand(
                np.asarray(side_arrays["hand_normals_world"][frame], dtype=np.float32), pose
            )
        )
    return {
        "obj_points": torch.from_numpy(np.stack(points_list)).to(device),
        "obj_normals": torch.from_numpy(np.stack(normals_list)).to(device),
        "hand_points": torch.from_numpy(np.stack(hand_list)).to(device),
        "hand_normals": torch.from_numpy(np.stack(hand_normal_list)).to(device),
        "obj_valid_mask": torch.from_numpy(np.stack(valid_list)).to(device),
    }


def build_side_dense(
    cache: SceneSequenceCache,
    side: str,
    *,
    encoder: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    bank_size: int,
    num_scene_points: int,
    batch_size: int,
    device: torch.device,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Run the frozen encoder over every (frame, bank) of one side."""
    side_arrays = cache.load_side(side)
    bank_indices = np.load(str(cache.dir / "sampling_bank" / f"{side}_indices.npy"), mmap_mode="r")
    frames = cache.frame_count
    num_hand = int(side_arrays["hand_points_world"].shape[1])
    probe = _batch_inputs(
        cache, side_arrays, bank_indices, bank=0, frames=range(0, min(1, frames)), device=device
    )
    with torch.no_grad():
        z_obj_probe, z_hand_probe, _ = encoder(
            obj_points=probe["obj_points"],
            obj_normals=probe["obj_normals"],
            hand_points=probe["hand_points"],
            hand_normals=probe["hand_normals"],
            obj_valid_mask=probe["obj_valid_mask"],
        )
    token_dim = int(z_obj_probe.shape[-1])
    if int(z_hand_probe.shape[-1]) != token_dim:
        raise ValueError("z_obj/z_hand token dims disagree")
    if int(z_obj_probe.shape[1]) != num_scene_points or int(z_hand_probe.shape[1]) != num_hand:
        raise ValueError(
            "DenseToken output point counts do not match the cache contract: "
            f"{int(z_obj_probe.shape[1])} / {int(z_hand_probe.shape[1])}"
        )
    out = {
        "z_scene": np.empty((frames, bank_size, num_scene_points, token_dim), dtype=dtype),
        "z_hand": np.empty((frames, bank_size, num_hand, token_dim), dtype=dtype),
        "hand_contact": np.empty((frames, bank_size, num_hand), dtype=dtype),
    }
    for bank in range(bank_size):
        for start in range(0, frames, batch_size):
            frame_range = range(start, min(start + batch_size, frames))
            batch = _batch_inputs(
                cache, side_arrays, bank_indices, bank=bank, frames=frame_range, device=device
            )
            with torch.no_grad():
                z_obj, z_hand, contact = encoder(
                    obj_points=batch["obj_points"],
                    obj_normals=batch["obj_normals"],
                    hand_points=batch["hand_points"],
                    hand_normals=batch["hand_normals"],
                    obj_valid_mask=batch["obj_valid_mask"],
                )
            index = slice(frame_range.start, frame_range.stop)
            out["z_scene"][index, bank] = z_obj.to(device="cpu", dtype=torch.float32).numpy()
            out["z_hand"][index, bank] = z_hand.to(device="cpu", dtype=torch.float32).numpy()
            out["hand_contact"][index, bank] = (
                contact.to(device="cpu", dtype=torch.float32).numpy()
            )
    return out


def build_dense_cache(
    root: str | Path,
    *,
    encoder: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    checkpoint_sha: str,
    bank_size: int,
    num_scene_points: int = 512,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    dtype: str = "float16",
    token_dim: int,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write the dense bank for every sequence/side and bind its fingerprint."""
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dense dtype {dtype!r}; expected one of {sorted(DTYPE_MAP)}")
    root = Path(root)
    meta = read_meta(root)
    if int(meta.get("sampling_bank_size", -1)) != int(bank_size):
        raise ValueError(
            f"{root}: sampling_bank_size in meta is {meta.get('sampling_bank_size')}, "
            f"not {bank_size}. Rebuild the sampling bank first."
        )
    fingerprint = dense_fingerprint_for_root(
        root, checkpoint_sha=checkpoint_sha, bank_size=bank_size, num_scene_points=num_scene_points
    )
    stored = meta.get("dense_cache", {})
    if stored.get("enabled") and not overwrite:
        if str(stored.get("fingerprint")) != fingerprint:
            raise ValueError(
                f"{root}: dense cache exists with a different fingerprint; pass --overwrite "
                "to rebuild. Never silently reuse stale features."
            )
    validate_scene_root(root, num_scene_points=num_scene_points, sampling_bank_size=bank_size)
    resolved_device = torch.device(device)
    stats: dict[str, int] = {"sequences": 0, "sides": 0, "skipped": 0, "frames": 0}
    for sequence_dir in iter_sequence_dirs(root):
        cache = SceneSequenceCache(sequence_dir)
        for side in ("left", "right"):
            if not (sequence_dir / side).is_dir():
                continue
            dense_dir = sequence_dir / "dense_bank" / side
            marker = dense_dir / "z_scene.npy"
            if marker.is_file() and not overwrite:
                stats["skipped"] += 1
                continue
            arrays = build_side_dense(
                cache,
                side,
                encoder=encoder,
                bank_size=int(bank_size),
                num_scene_points=int(num_scene_points),
                batch_size=int(batch_size),
                device=resolved_device,
                dtype=DTYPE_MAP[dtype],
            )
            dense_dir.mkdir(parents=True, exist_ok=True)
            for name, array in arrays.items():
                np.save(dense_dir / f"{name}.npy", array)
            stats["sides"] += 1
            stats["frames"] += cache.frame_count
        stats["sequences"] += 1
        print(f"[dense-cache] {sequence_dir.relative_to(root)} done")
    update_meta(
        root,
        dense_cache={
            "enabled": True,
            "dtype": dtype,
            "fingerprint": fingerprint,
            "checkpoint_sha256": checkpoint_sha,
            "token_dim": int(token_dim),
            "bank_size": int(bank_size),
            "num_scene_points": int(num_scene_points),
            "num_hand_points": int(meta["num_hand_points"]),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the frozen DenseToken bank")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--bank-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from src.task.Cm.dense_token import FrozenDenseTokenEncoder

    checkpoint = Path(args.dense_checkpoint).resolve()
    encoder = FrozenDenseTokenEncoder(checkpoint).to(torch.device(args.device))
    encoder.eval()
    stats = build_dense_cache(
        args.root,
        encoder=encoder,
        checkpoint_sha=sha256_file(checkpoint),
        bank_size=args.bank_size,
        num_scene_points=args.num_points,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        token_dim=encoder.token_dim,
        overwrite=args.overwrite,
    )
    print(f"[dense-cache] done: {stats}")


if __name__ == "__main__":
    main()
