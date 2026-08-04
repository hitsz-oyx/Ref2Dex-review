"""Build temporal Stage 4 Cm samples directly from raw GRAB sequences.

Stage 4 deliberately does not consume Stage 2 or Stage 3 files.  It reuses
the raw GRAB adapter only for MANO reconstruction and stable object-surface
sampling, then writes consecutive temporal pairs in the *current* hand-root
frame.  This makes the future object geometry unavailable to a Cm encoder
while retaining it as an object-flow target.
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

from process.GRAB.raw import (
    DEFAULT_GRAB_ROOT,
    DEFAULT_MANO_MODEL_DIR,
    GRABRawAdapter,
    load_manifest_seq_paths,
    resolve_grab_sequence_root,
)
from process.common.stage4_cm import write_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed_data" / "stage4" / "data" / "grab"
SHARED_SCHEMA_NAME = "ref2dex_cm_sequence_shared"
HAND_SCHEMA_NAME = "ref2dex_cm_sequence_hand"
SCHEMA_VERSION = "3.0.0"
SOURCE_FPS = 120.0


def _resolve_sequences(args: argparse.Namespace) -> list[str]:
    """Resolve raw GRAB sequences without depending on the Stage 2 entrypoint."""
    if args.raw_file:
        return [str(Path(args.raw_file).resolve())]
    sequence_root = resolve_grab_sequence_root(args.grab_root)
    sequences = sorted(sequence_root.glob("*/*.npz"))
    if not args.seq:
        return [str(path) for path in sequences]
    target = args.seq[:-4] if args.seq.endswith(".npz") else args.seq
    selected: list[str] = []
    for path in sequences:
        relative = path.relative_to(sequence_root).with_suffix("").as_posix()
        if relative == target or target in relative:
            selected.append(str(path))
    return selected


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _points_world_to_current_hand(points_world: np.ndarray, hand_root_pose_t: np.ndarray) -> np.ndarray:
    """Express points at either time in each pair's current hand-root frame."""
    rotation = hand_root_pose_t[:, :3, :3]
    translation = hand_root_pose_t[:, :3, 3]
    return np.einsum("tji,tpj->tpi", rotation, points_world - translation[:, None]).astype(np.float32)


def _normals_world_to_current_hand(normals_world: np.ndarray, hand_root_pose_t: np.ndarray) -> np.ndarray:
    rotation = hand_root_pose_t[:, :3, :3]
    normals = np.einsum("tji,tpj->tpi", rotation, normals_world)
    return (normals / np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def _compute_current_candidate_mask(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    *,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Compute only the current-frame 5cm candidate mask in bounded CUDA batches."""
    num_pairs, num_obj, _ = obj_points.shape
    candidate_mask = np.empty((num_pairs, num_obj), dtype=bool)
    batch_size = max(1, int(frame_batch_size))
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        obj = torch.from_numpy(obj_points[start:end]).to(device=device, dtype=torch.float32)
        hand = torch.from_numpy(hand_points[start:end]).to(device=device, dtype=torch.float32)
        distance = torch.cdist(obj, hand)
        candidate_mask[start:end] = distance.amin(dim=-1).cpu().numpy() <= float(candidate_threshold)
    return candidate_mask


def _scalar_from_raw(path: Path, key: str, default: float) -> float:
    with np.load(path, allow_pickle=True) as raw:
        if key not in raw.files:
            return float(default)
        value = np.asarray(raw[key])
        return float(value.item()) if value.size == 1 else float(default)


def build_shared_sequence(
    source: dict[str, Any],
    *,
    source_path: Path,
    grab_root: Path,
    ds_rate: int,
) -> dict[str, np.ndarray]:
    """Build fields shared by both hands for one cached GRAB sequence."""
    source_rel = source_path.resolve().relative_to(grab_root.resolve()).as_posix()
    return {
        "schema_name": np.asarray(SHARED_SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION),
        "source_raw_file": np.asarray(source_rel),
        "dataset_name": np.asarray(str(source["dataset_name"])),
        "seq_id": np.asarray(str(source["seq_id"])),
        "subject_id": np.asarray(str(source["subject_id"])),
        "seq_name": np.asarray(str(source["seq_name"])),
        "object_name": np.asarray(str(source["object_name"])),
        "raw_frame_id": np.asarray(source["raw_frame_id"], dtype=np.int32),
        "ds_rate": np.asarray(int(ds_rate), dtype=np.int32),
        "source_fps": np.asarray(SOURCE_FPS, dtype=np.float32),
        "coordinate_frame": np.asarray("world"),
        "obj_points_world": np.asarray(source["obj_points_world"], dtype=np.float32),
        "obj_normals_world": np.asarray(source["obj_normals_world"], dtype=np.float32),
        "obj_point_id": np.asarray(source["obj_point_id"], dtype=np.int32),
    }


def build_hand_sequence(
    source: dict[str, Any],
    *,
    side: str,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Build hand-specific fields; object geometry remains in ``shared.npz``."""
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported side {side!r}")
    root_key = f"{side}_hand_root_pose"
    if root_key not in source:
        raise ValueError(f"{source['seq_id']} has no reconstructed {side}-hand trajectory")

    obj_world = np.asarray(source["obj_points_world"], dtype=np.float32)
    hand_world = np.asarray(source[f"{side}_hand_points_world"], dtype=np.float32)
    hand_normals_world = np.asarray(source[f"{side}_hand_normals_world"], dtype=np.float32)
    hand_root_pose_world = np.asarray(source[root_key], dtype=np.float32)
    hand_cano_points = np.asarray(source[f"{side}_hand_cano_points"], dtype=np.float32)
    # Candidate masks are current-state geometry only; no future frame enters.
    hand_local = _points_world_to_current_hand(hand_world, hand_root_pose_world)
    obj_local = _points_world_to_current_hand(obj_world, hand_root_pose_world)
    candidate_mask = _compute_current_candidate_mask(
        obj_local,
        hand_local,
        candidate_threshold=candidate_threshold,
        frame_batch_size=frame_batch_size,
        device=device,
    )
    return {
        "schema_name": np.asarray(HAND_SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION),
        "side": np.asarray(side),
        "hand_root_pose_world": hand_root_pose_world,
        "hand_points_world": hand_world,
        "hand_normals_world": hand_normals_world,
        "hand_point_id": np.asarray(source[f"{side}_hand_point_id"], dtype=np.int32),
        "hand_cano_points": hand_cano_points,
        "hand_finger_id": np.asarray(source[f"{side}_hand_finger_id"], dtype=np.int32),
        "hand_region_id": np.asarray(source[f"{side}_hand_region_id"], dtype=np.int32),
        "obj_candidate_mask_5cm": candidate_mask,
    }


def _write_meta(output_root: Path, args: argparse.Namespace, stats: dict[str, int]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "shared_schema_name": SHARED_SCHEMA_NAME,
        "hand_schema_name": HAND_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "raw GRAB; no Stage 2 or Stage 3 dependency",
        "source_grab_root": str(Path(args.grab_root).resolve()),
        "output_root": str(output_root.resolve()),
        "sample_unit": "one shared sequence file plus one hand-side file",
        "coordinate_frame": "world; Dataset maps endpoints to hand_root_t",
        "future_object_policy": "not stored as a pair; Dataset creates endpoint flow at runtime",
        "num_obj_pool": int(args.num_obj_points),
        "num_hand_points": 1538,
        "ds_rate": int(args.ds_rate),
        "source_fps": SOURCE_FPS,
        "effective_fps": SOURCE_FPS / int(args.ds_rate),
        "candidate_threshold": float(args.candidate_threshold),
        "stats": stats,
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw GRAB -> complete Cm sequence cache")
    parser.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--mano-path", default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--seq", default=None, help="Exact or substring filter, for example s1/bowl_pass_1")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument(
        "--ds-rate",
        type=int,
        default=4,
        help="Raw-frame subsampling before sequence caching; default is ds_rate=4 (30 Hz).",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="First original GRAB frame to cache; raw_frame_id remains absolute.",
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="m")
    parser.add_argument("--save-compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest and (args.raw_file or args.seq):
        raise SystemExit("--manifest cannot be combined with --raw-file/--seq")
    if args.num_obj_points != 4096:
        raise SystemExit("Current Cm Stage 4 contract requires --num-obj-points=4096")
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
    stats = {"source_sequences": len(sequences), "shared_written": 0, "hand_written": 0,
             "skipped": 0, "empty_side": 0, "failed": 0, "frames": 0}

    for raw_path_text in sequences:
        raw_path = Path(raw_path_text).resolve()
        try:
            source = adapter.process_sequence(str(raw_path))
            sequence_dir = output_root / str(source["subject_id"]) / str(source["seq_name"])
            shared_path = sequence_dir / "shared.npz"
            pending_sides = [side for side in sides if args.overwrite or not (sequence_dir / f"{side}.npz").exists()]
            if not pending_sides:
                stats["skipped"] += len(sides)
                continue
            if args.overwrite or not shared_path.exists():
                shared = build_shared_sequence(
                    source, source_path=raw_path, grab_root=grab_root, ds_rate=args.ds_rate,
                )
                sequence_dir.mkdir(parents=True, exist_ok=True)
                if args.save_compressed:
                    np.savez_compressed(shared_path, **shared)
                else:
                    np.savez(shared_path, **shared)
                stats["shared_written"] += 1
            for side in sides:
                output_path = sequence_dir / f"{side}.npz"
                if side not in pending_sides:
                    stats["skipped"] += 1
                    continue
                try:
                    hand = build_hand_sequence(
                        source,
                        side=side,
                        candidate_threshold=args.candidate_threshold,
                        frame_batch_size=args.frame_batch_size,
                        device=device,
                    )
                except ValueError as exc:
                    if "no reconstructed" in str(exc):
                        stats["empty_side"] += 1
                        continue
                    raise
                if args.save_compressed:
                    np.savez_compressed(output_path, **hand)
                else:
                    np.savez(output_path, **hand)
                stats["hand_written"] += 1
                stats["frames"] += int(hand["obj_candidate_mask_5cm"].shape[0])
                candidate_counts = hand["obj_candidate_mask_5cm"].sum(axis=1)
                print(
                    f"[stage4] wrote {output_path} frames={len(hand['obj_candidate_mask_5cm'])} "
                    f"candidate[min/median/max]={int(candidate_counts.min())}/"
                    f"{int(np.median(candidate_counts))}/{int(candidate_counts.max())}"
                )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage4] failed {raw_path}: {exc}")
            traceback.print_exc()

    _write_meta(output_root, args, stats)
    write_manifest(output_root)
    print(f"[stage4] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
