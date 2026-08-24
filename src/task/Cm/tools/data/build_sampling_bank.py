"""Build the deterministic per-frame sampling banks (Cm Scene Cache V1, layer 2).

依照 ``src/task/Cm/docs/指导/V1.md`` §11-12：为每个 (sequence, side, frame)
提前生成 B=4 套 512-point 采样索引，训练时 bank 随 epoch 哈希变化，验证/测试
固定 bank=0 完全 deterministic。

Bank 内容与旧 ``Stage4CmDataset`` 的在线采样逐位一致（parity Test A/C）：
bank ``b`` 的种子取 ``stable_frame_seed(..., epoch=b, namespace="cm-object-sampling")``，
因此旧数据集在 ``base_seed=sampling_seed, epoch=b`` 时抽到的 512 点与
``bank=b`` 完全相同。索引类型 uint32，未填满的槽位用
``INVALID_INDEX=0xFFFFFFFF`` 哨兵标记。

用法::

    python -m src.task.Cm.tools.data.build_sampling_bank \
        --root data/processed_data/cm_scene_v1 \
        --bank-size 4 --num-points 512 --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.task.Cm.cache_schema import (
    INVALID_INDEX,
    SCHEMA_NAME,
    SceneSequenceCache,
    iter_sequence_dirs,
    read_meta,
    scene_cache_fingerprint,
    update_meta,
    validate_scene_root,
)
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed


def build_side_bank(
    cache: SceneSequenceCache,
    side: str,
    *,
    bank_size: int,
    num_points: int,
    sampling_seed: int,
) -> np.ndarray:
    """Generate one ``[T, bank_size, num_points]`` uint32 index bank."""
    pool = cache.num_scene_pool
    if pool >= INVALID_INDEX:
        raise ValueError(f"Scene pool size {pool} exceeds the uint32 sampling-bank limit.")
    side_arrays = cache.load_side(side)
    raw_frame_id = np.asarray(cache.raw_frame_id)
    seq_id = str(cache.shared_meta.get("seq_id", cache.dir.name))
    frames = cache.frame_count
    bank = np.full((frames, int(bank_size), int(num_points)), INVALID_INDEX, dtype=np.uint32)
    for frame in range(frames):
        candidate = cache.candidate_indices_at(side_arrays, frame)
        if candidate.size == 0:
            continue
        # Rebuild the bool mask only to reuse the exact legacy sampler; the
        # resulting selection is identical to sampling from ``candidate``
        # because np.flatnonzero yields the same ascending index order.
        mask = np.zeros(pool, dtype=bool)
        mask[candidate] = True
        for b in range(int(bank_size)):
            seed = stable_frame_seed(
                base_seed=int(sampling_seed),
                seq_id=seq_id,
                side=side,
                raw_frame_id=int(raw_frame_id[frame]),
                epoch=b,
                namespace="cm-object-sampling",
            )
            selected, valid = sample_object_indices(mask, num_samples=int(num_points), seed=seed)
            bank[frame, b] = np.where(valid, selected, INVALID_INDEX).astype(np.uint32)
    return bank


def build_all_banks(
    root: str | Path,
    *,
    bank_size: int,
    num_points: int,
    sampling_seed: int,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write ``sampling_bank/<side>_indices.npy`` for every sequence."""
    root = Path(root)
    meta = read_meta(root)
    fingerprint = scene_cache_fingerprint(
        schema=SCHEMA_NAME,
        points_per_asset=int(meta["points_per_asset"]),
        candidate_threshold_m=float(meta["candidate_threshold_m"]),
        sampling_seed=int(sampling_seed),
    )
    existing = meta.get("sampling", {})
    if existing and not overwrite:
        if int(existing.get("bank_size", -1)) != int(bank_size):
            raise ValueError(
                f"{root}: existing sampling bank has bank_size={existing.get('bank_size')}; "
                "pass --overwrite to regenerate."
            )
        if int(existing.get("num_points", -1)) != int(num_points):
            raise ValueError(
                f"{root}: existing sampling bank has num_points={existing.get('num_points')}; "
                "pass --overwrite to regenerate."
            )
        if str(existing.get("fingerprint")) != fingerprint:
            raise ValueError(
                f"{root}: existing sampling bank fingerprint does not match the current "
                "geometry cache contract; pass --overwrite to regenerate."
            )
    stats = {"sequences": 0, "sides": 0, "skipped": 0, "frames": 0}
    for sequence_dir in iter_sequence_dirs(root):
        cache = SceneSequenceCache(sequence_dir)
        # Cross-check the builder meta against the arrays actually on disk.
        if cache.num_scene_pool <= 0:
            raise ValueError(f"{sequence_dir}: scene pool must not be empty")
        bank_dir = sequence_dir / "sampling_bank"
        for side in ("left", "right"):
            if not (sequence_dir / side).is_dir():
                continue
            bank_path = bank_dir / f"{side}_indices.npy"
            if bank_path.is_file() and not overwrite:
                stats["skipped"] += 1
                continue
            bank = build_side_bank(
                cache, side, bank_size=bank_size, num_points=num_points, sampling_seed=sampling_seed
            )
            bank_dir.mkdir(parents=True, exist_ok=True)
            np.save(bank_path, bank)
            stats["sides"] += 1
            stats["frames"] += cache.frame_count
        stats["sequences"] += 1
    update_meta(
        root,
        sampling_seed=int(sampling_seed),
        sampling_bank_size=int(bank_size),
        sampling_num_points=int(num_points),
        model_scene_points=int(num_points),
        sampling={
            "seed": int(sampling_seed),
            "bank_size": int(bank_size),
            "num_points": int(num_points),
            "seed_rule": "stable_frame_seed(base_seed=sampling_seed, epoch=bank, namespace=cm-object-sampling)",
            "fingerprint": fingerprint,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    validate_scene_root(root, num_scene_points=int(num_points), sampling_bank_size=int(bank_size))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Cm scene sampling banks")
    parser.add_argument("--root", required=True, help="Cm scene cache root (stage4_cm_scene output)")
    parser.add_argument("--bank-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bank_size <= 0 or args.num_points <= 0:
        raise SystemExit("--bank-size and --num-points must be positive")
    stats = build_all_banks(
        args.root,
        bank_size=args.bank_size,
        num_points=args.num_points,
        sampling_seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"[sampling-bank] done: {stats}")


if __name__ == "__main__":
    main()
