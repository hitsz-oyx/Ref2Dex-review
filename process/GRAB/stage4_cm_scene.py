"""Build the Cm Scene Cache V1.1 (mmap scene geometry + ragged candidates).

依照 ``src/task/Cm/docs/指导/V1.1.md`` 实现 Scene Cache 的第一级。语义上
``P_t`` 变为当前手附近的 local scene points；每个 manipulated/environment
asset 各采 4096 点，sequence 间允许不同数量的 asset，object 与 environment
完全一视同仁。candidate 仍是「当前手 5cm」，但改存 ragged index 而不是 bool
mask。环境资产通过 ``GRABSeqData.get_environment_assets`` 通用接口获得，Stage4
不感知 environment == table。

用法::

    python -m process.GRAB.stage4_cm_scene \
        --grab-root dataset/GRAB/data \
        --output-root data/processed_data/cm_scene_v1 \
        --num-obj-points 4096 --num-env-points 4096 \
        --candidate-threshold 0.05 --ds-rate 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.task.Cm.cache_schema import (
    ENV_SOURCE_ENVIRONMENT,
    ENV_SOURCE_OBJECT,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    update_meta,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAB_ROOT = str(ROOT / "dataset" / "GRAB")
DEFAULT_MANO_MODEL_DIR = str(ROOT / "dataset" / "arctic" / "data" / "body_models" / "mano")
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed_data" / "cm_scene_v1"
SOURCE_FPS = 120.0
# Default tolerance for treating a per-frame-jittering environment pose as a
# rigid static asset (rotation in radians, translation in metres).
DEFAULT_STATIC_ROT_EPS = 1.0e-3
DEFAULT_STATIC_TRANS_EPS = 1.0e-3
DEFAULT_STATIC_INLIER_FRACTION = 0.9


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _resolve_sequences(args: argparse.Namespace) -> list[str]:
    """Resolve raw paths lazily so cache utilities do not require ``smplx``."""
    from process.GRAB.raw import resolve_grab_sequence_root

    if args.raw_file:
        return [str(Path(args.raw_file).resolve())]
    sequence_root = resolve_grab_sequence_root(args.grab_root)
    sequences = sorted(sequence_root.glob("*/*.npz"))
    if not args.seq:
        return [str(path) for path in sequences]
    target = args.seq[:-4] if args.seq.endswith(".npz") else args.seq
    return [
        str(path) for path in sequences
        if path.relative_to(sequence_root).with_suffix("").as_posix() == target
        or target in path.relative_to(sequence_root).with_suffix("").as_posix()
    ]


def _robust_static_pose(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a static pose without letting sparse tracker outliers dominate."""
    translation = np.median(poses[:, :3, 3], axis=0)
    median_rotation = np.median(poses[:, :3, :3], axis=0)
    u, _, vh = np.linalg.svd(median_rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation, translation


def _rotation_angle_deviation(rotations: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Angle from every frame rotation to one reference rotation, in radians."""
    relative = np.einsum("ji,tjk->tik", reference, rotations)
    trace = np.trace(relative, axis1=1, axis2=2)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def build_static_environment(
    assets: list[dict],
    *,
    static_rot_eps: float,
    static_trans_eps: float,
    static_inlier_fraction: float = DEFAULT_STATIC_INLIER_FRACTION,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Collapse per-frame env poses into one static world placement (V1.md §8).

    GRAB 的 table pose 通常只有数值抖动，但部分序列含稀疏 tracker outlier。
    使用 robust median pose，并要求足够比例的帧落在物理容差内；持续移动的
    environment 仍然 fail-fast。Dataset 对结果自动 broadcast。
    """
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    names: list[str] = []
    counts: list[int] = []
    if not 0.0 < static_inlier_fraction <= 1.0:
        raise ValueError("static_inlier_fraction must be in (0, 1]")
    for asset in assets:
        poses = np.asarray(asset["poses_world"], dtype=np.float64)
        rotation, translation = _robust_static_pose(poses)
        trans_dev = np.linalg.norm(poses[:, :3, 3] - translation, axis=1)
        rot_dev = _rotation_angle_deviation(poses[:, :3, :3], rotation)
        inlier = (trans_dev <= static_trans_eps) & (rot_dev <= static_rot_eps)
        inlier_fraction = float(inlier.mean())
        if inlier_fraction < static_inlier_fraction:
            raise NotImplementedError(
                f"Environment asset {asset['name']!r} moves within the sequence "
                f"(inliers={inlier_fraction:.1%}, required={static_inlier_fraction:.1%}, "
                f"rot_p90={np.quantile(rot_dev, 0.9):.3g} rad, "
                f"trans_p90={np.quantile(trans_dev, 0.9):.3g} m); only "
                f"'static_world' storage is supported in scene cache V1."
            )
        rotation = rotation.astype(np.float32)
        translation = translation.astype(np.float32)
        world_points = (np.asarray(asset["canonical_points"], dtype=np.float32) @ rotation.T + translation).astype(np.float32)
        world_normals = (np.asarray(asset["canonical_normals"], dtype=np.float32) @ rotation.T).astype(np.float32)
        norm = np.linalg.norm(world_normals, axis=-1, keepdims=True)
        world_normals = (world_normals / np.clip(norm, 1e-10, None)).astype(np.float32)
        points.append(world_points)
        normals.append(world_normals)
        names.append(str(asset["name"]))
        counts.append(int(world_points.shape[0]))
    if not points:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), [], []
    return np.concatenate(points, axis=0), np.concatenate(normals, axis=0), names, counts


def compute_scene_candidate_ragged(
    scene_points_world: np.ndarray,
    hand_points_world: np.ndarray,
    hand_root_pose_world: np.ndarray,
    *,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """5cm candidates over the full scene pool, stored as ragged indices.

    距离计算沿用旧 Stage4 的流程（先统一到当前手根系再 ``cdist``），保证
    object 半边的判定与旧 ``obj_candidate_mask_5cm`` 逐元素一致（Test B）。
    """
    from process.common.stage4_cm import compute_current_candidate_mask

    mask = compute_current_candidate_mask(
        scene_points_world,
        hand_points_world,
        hand_root_pose_world,
        candidate_threshold=candidate_threshold,
        frame_batch_size=frame_batch_size,
        device=device,
    )
    frames = mask.shape[0]
    counts = mask.sum(axis=1).astype(np.int64)
    offsets = np.zeros(frames + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    # np.flatnonzero on the raveled mask yields absolute indices; convert to
    # per-frame pool indices by taking the remainder modulo the pool size.
    pool = mask.shape[1]
    flat = np.flatnonzero(mask.ravel()).astype(np.int64)
    indices = (flat % pool).astype(np.int32)
    return offsets, indices


def summarize_scene_root(output_root: str | Path, *, source_sequences: int) -> dict[str, Any]:
    """Summarize all materialized sequences, including incremental retries."""
    root = Path(output_root)
    sequence_files = sorted(root.glob("*/*/shared/raw_frame_id.npy"))
    side_frames = 0
    active_side_frames = 0
    sides_present = 0
    candidate_min: int | None = None
    candidate_max: int | None = None
    for raw_frame_path in sequence_files:
        sequence_dir = raw_frame_path.parent.parent
        for side in ("left", "right"):
            offsets_path = sequence_dir / side / "candidate_offsets.npy"
            if not offsets_path.is_file():
                continue
            offsets = np.load(offsets_path, mmap_mode="r")
            counts = np.diff(offsets)
            sides_present += 1
            side_frames += int(counts.size)
            active_side_frames += int(np.count_nonzero(counts))
            if counts.size:
                current_min = int(counts.min())
                current_max = int(counts.max())
                candidate_min = current_min if candidate_min is None else min(candidate_min, current_min)
                candidate_max = current_max if candidate_max is None else max(candidate_max, current_max)
    sequences_present = len(sequence_files)
    return {
        "source_sequences": int(source_sequences),
        "sequences_present": sequences_present,
        "sequences_missing": max(int(source_sequences) - sequences_present, 0),
        "sides_present": sides_present,
        "side_frames": side_frames,
        "active_side_frames": active_side_frames,
        "candidate_min": candidate_min,
        "candidate_max": candidate_max,
    }


def build_sequence_scene(
    source: dict[str, Any],
    *,
    env_assets: list[dict],
    static_rot_eps: float,
    static_trans_eps: float,
    static_inlier_fraction: float = DEFAULT_STATIC_INLIER_FRACTION,
) -> dict[str, Any]:
    """Assemble shared scene-pool fields for one sequence."""
    env_points, env_normals, env_names, env_counts = build_static_environment(
        env_assets,
        static_rot_eps=static_rot_eps,
        static_trans_eps=static_trans_eps,
        static_inlier_fraction=static_inlier_fraction,
    )
    obj_points = np.asarray(source["obj_points_world"], dtype=np.float32)
    obj_normals = np.asarray(source["obj_normals_world"], dtype=np.float32)
    num_obj = obj_points.shape[1]
    scene_source_id = np.concatenate([
        np.full(num_obj, ENV_SOURCE_OBJECT, dtype=np.uint8),
        np.full(env_points.shape[0], ENV_SOURCE_ENVIRONMENT, dtype=np.uint8),
    ])
    scene_asset_id = np.concatenate([
        np.zeros(num_obj, dtype=np.uint16),
        np.concatenate([
            np.full(count, asset_id + 1, dtype=np.uint16)
            for asset_id, count in enumerate(env_counts)
        ]) if env_counts else np.zeros(0, dtype=np.uint16),
    ])
    asset_offsets = np.asarray(
        [0, num_obj] + [num_obj + sum(env_counts[:i + 1]) for i in range(len(env_counts))],
        dtype=np.int64,
    )
    return {
        "raw_frame_id": np.asarray(source["raw_frame_id"], dtype=np.int32),
        "obj_points_world": obj_points,
        "obj_normals_world": obj_normals,
        "env_points_world": env_points,
        "env_normals_world": env_normals,
        "scene_source_id": scene_source_id,
        "scene_asset_id": scene_asset_id,
        "asset_offsets": asset_offsets,
        "environment_names": env_names,
        "environment_counts": env_counts,
    }


def write_sequence(
    sequence_dir: Path,
    shared: dict[str, Any],
    *,
    source: dict[str, Any],
    source_path: Path,
    grab_root: Path,
    ds_rate: int,
) -> None:
    shared_dir = sequence_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw_frame_id", "obj_points_world", "obj_normals_world",
                 "env_points_world", "env_normals_world", "scene_source_id",
                 "scene_asset_id", "asset_offsets"):
        np.save(shared_dir / f"{name}.npy", shared[name])
    source_rel = source_path.resolve().relative_to(grab_root.resolve()).as_posix()
    (shared_dir / "meta.json").write_text(
        json.dumps({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "seq_id": str(source["seq_id"]),
            "dataset_name": str(source["dataset_name"]),
            "subject_id": str(source["subject_id"]),
            "seq_name": str(source["seq_name"]),
            "object_name": str(source["object_name"]),
            "source_raw_file": source_rel,
            "ds_rate": int(ds_rate),
            "source_fps": SOURCE_FPS,
            "coordinate_frame": "world",
            "environment_storage": "static_world",
            "environment_assets": shared["environment_names"],
            "assets": [
                {"name": str(source["object_name"]), "role": "manipulated", "num_points": int(source["obj_points_world"].shape[1])},
                *[
                    {"name": name, "role": "environment", "num_points": count}
                    for name, count in zip(shared["environment_names"], shared["environment_counts"])
                ],
            ],
            "scene_pool_size": int(shared["obj_points_world"].shape[1] + shared["env_points_world"].shape[0]),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_side(
    sequence_dir: Path,
    side: str,
    source: dict[str, Any],
    *,
    env_points_world: np.ndarray,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> dict[str, int]:
    side_dir = sequence_dir / side
    side_dir.mkdir(parents=True, exist_ok=True)
    hand_world = np.asarray(source[f"{side}_hand_points_world"], dtype=np.float32)
    hand_normals = np.asarray(source[f"{side}_hand_normals_world"], dtype=np.float32)
    hand_root = np.asarray(source[f"{side}_hand_root_pose"], dtype=np.float32)
    obj_world = np.asarray(source["obj_points_world"], dtype=np.float32)
    frames = obj_world.shape[0]
    scene_world = np.concatenate([obj_world, np.broadcast_to(env_points_world, (frames,) + env_points_world.shape)], axis=1)
    offsets, indices = compute_scene_candidate_ragged(
        scene_world, hand_world, hand_root,
        candidate_threshold=candidate_threshold,
        frame_batch_size=frame_batch_size,
        device=device,
    )
    np.save(side_dir / "hand_points_world.npy", hand_world)
    np.save(side_dir / "hand_normals_world.npy", hand_normals)
    np.save(side_dir / "hand_root_pose_world.npy", hand_root)
    np.save(side_dir / "candidate_offsets.npy", offsets)
    np.save(side_dir / "candidate_indices.npy", indices)
    counts = (offsets[1:] - offsets[:-1])
    return {
        "frames": int(frames),
        "candidate_min": int(counts.min()),
        "candidate_median": float(np.median(counts)),
        "candidate_max": int(counts.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw GRAB -> Cm Scene Cache V1 (mmap)")
    parser.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--mano-path", default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--seq", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument("--num-env-points", "--points-per-asset", dest="num_env_points", type=int, default=4096)
    parser.add_argument("--ds-rate", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="m")
    parser.add_argument("--static-rot-eps", type=float, default=DEFAULT_STATIC_ROT_EPS)
    parser.add_argument("--static-trans-eps", type=float, default=DEFAULT_STATIC_TRANS_EPS)
    parser.add_argument("--static-inlier-fraction", type=float, default=DEFAULT_STATIC_INLIER_FRACTION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from process.GRAB.raw import GRABRawAdapter, GRABSeqData, load_manifest_seq_paths

    args = parse_args()

    if args.manifest and (args.raw_file or args.seq):
        raise SystemExit("--manifest cannot be combined with --raw-file/--seq")
    if args.num_obj_points != 4096:
        raise SystemExit("Scene cache V1.1 requires --num-obj-points=4096")
    if args.num_env_points != args.num_obj_points:
        raise SystemExit("V1.1 requires --points-per-asset=4096 for every asset")
    if args.num_env_points <= 0:
        raise SystemExit("--points-per-asset must be positive")
    if args.ds_rate <= 0:
        raise SystemExit("--ds-rate must be positive")

    if args.manifest:
        sequences = load_manifest_seq_paths(args.manifest, args.grab_root)
    else:
        sequences = _resolve_sequences(args)
    if not sequences:
        raise FileNotFoundError("No GRAB sequences matched the requested input")

    device = _resolve_device(args.device)
    output_root = Path(args.output_root).resolve()
    grab_root = Path(args.grab_root).resolve()
    adapter = GRABRawAdapter(
        num_obj_points=args.num_obj_points,
        device=str(device),
        max_frames=args.max_frames if args.max_frames > 0 else None,
        frame_start=args.frame_start,
        grab_root=str(grab_root),
        mano_path=args.mano_path,
        ds_rate=args.ds_rate,
        obj_unit=args.obj_unit,
        nn_batch_size=args.nn_batch_size,
    )
    sides = ("left", "right") if args.side == "both" else (args.side,)
    stats: dict[str, Any] = {
        "source_sequences": len(sequences), "sequences_written": 0, "sides_written": 0,
        "skipped": 0, "empty_side": 0, "failed": 0, "frames": 0,
        "candidate_min": None, "candidate_median": 0.0, "candidate_max": None,
    }

    for raw_path_text in sequences:
        raw_path = Path(raw_path_text).resolve()
        try:
            source = adapter.process_sequence(str(raw_path))
            frame_ids = np.asarray(source["raw_frame_id"], dtype=np.int64)
            seq_data = GRABSeqData(str(raw_path))
            env_assets = adapter.get_environment_geometry(seq_data, frame_ids, num_env_points=args.num_env_points)
            shared = build_sequence_scene(
                source,
                env_assets=env_assets,
                static_rot_eps=args.static_rot_eps,
                static_trans_eps=args.static_trans_eps,
                static_inlier_fraction=args.static_inlier_fraction,
            )
            sequence_dir = output_root / str(source["subject_id"]) / str(source["seq_name"])
            pending_sides = [side for side in sides if args.overwrite or not (sequence_dir / side).is_dir()]
            if not pending_sides:
                stats["skipped"] += len(sides)
                continue
            if args.overwrite or not (sequence_dir / "shared" / "raw_frame_id.npy").is_file():
                write_sequence(
                    sequence_dir, shared, source=source, source_path=raw_path,
                    grab_root=grab_root, ds_rate=args.ds_rate,
                )
                stats["sequences_written"] += 1
            for side in sides:
                if side not in pending_sides:
                    continue
                root_key = f"{side}_hand_root_pose"
                if root_key not in source:
                    stats["empty_side"] += 1
                    continue
                side_stats = write_side(
                    sequence_dir, side, source,
                    env_points_world=shared["env_points_world"],
                    candidate_threshold=args.candidate_threshold,
                    frame_batch_size=args.frame_batch_size,
                    device=device,
                )
                stats["sides_written"] += 1
                stats["frames"] += side_stats["frames"]
                stats["candidate_min"] = (
                    side_stats["candidate_min"] if stats["candidate_min"] is None
                    else min(stats["candidate_min"], side_stats["candidate_min"])
                )
                stats["candidate_max"] = (
                    side_stats["candidate_max"] if stats["candidate_max"] is None
                    else max(stats["candidate_max"], side_stats["candidate_max"])
                )
                print(
                    f"[stage4-scene] wrote {sequence_dir / side} frames={side_stats['frames']} "
                    f"candidate[min/median/max]={side_stats['candidate_min']}/"
                    f"{side_stats['candidate_median']:.0f}/{side_stats['candidate_max']}"
                )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage4-scene] failed {raw_path}: {exc}")
            traceback.print_exc()

    root_stats = summarize_scene_root(
        output_root,
        source_sequences=max(len(sequences), stats["failed"] + len(list(output_root.glob("*/*/shared/raw_frame_id.npy")))),
    )
    update_meta(
        output_root,
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        source="raw GRAB via generic environment assets; no Stage 2/3 dependency",
        source_grab_root=str(grab_root),
        output_root=str(output_root),
        points_per_asset=int(args.num_env_points),
        model_scene_points=512,
        hand_points=int(adapter.num_hand_points),
        candidate_threshold_m=float(args.candidate_threshold),
        num_hand_points=int(adapter.num_hand_points),
        ds_rate=int(args.ds_rate),
        source_fps=SOURCE_FPS,
        effective_fps=SOURCE_FPS / int(args.ds_rate),
        environment_storage="static_world",
        static_inlier_fraction=float(args.static_inlier_fraction),
        coordinate_frame="world; Dataset maps endpoints to hand_root_t",
        stats=root_stats,
        last_run_stats=stats,
    )
    print(f"[stage4-scene] summary: {json.dumps(root_stats)}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
