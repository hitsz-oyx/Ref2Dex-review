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
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "processed_data" / "generated" / "stage4" / "grab_cm_raw_stride3"
SCHEMA_NAME = "ref2dex_cm_stage4"
SCHEMA_VERSION = "1.1.0"


def _resolve_sequences(args: argparse.Namespace) -> list[str]:
    """Resolve raw GRAB sequences without depending on the Stage 2 entrypoint."""
    if args.raw_file:
        return [str(Path(args.raw_file).resolve())]
    sequences = sorted((Path(args.grab_root) / "grab").glob("*/*.npz"))
    if not args.seq:
        return [str(path) for path in sequences]
    target = args.seq[:-4] if args.seq.endswith(".npz") else args.seq
    selected: list[str] = []
    for path in sequences:
        relative = path.relative_to(Path(args.grab_root) / "grab").with_suffix("").as_posix()
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


def _relative_wrist_pose(hand_root_pose_t: np.ndarray, hand_root_pose_t1: np.ndarray) -> np.ndarray:
    """Return T_{hand_t <- hand_t1}; both poses are T_{world <- hand}."""
    r_t = hand_root_pose_t[:, :3, :3]
    r_t1 = hand_root_pose_t1[:, :3, :3]
    t_t = hand_root_pose_t[:, :3, 3]
    t_t1 = hand_root_pose_t1[:, :3, 3]
    relative = np.tile(np.eye(4, dtype=np.float32), (len(hand_root_pose_t), 1, 1))
    relative[:, :3, :3] = np.einsum("tji,tjk->tik", r_t, r_t1).astype(np.float32)
    relative[:, :3, 3] = np.einsum("tji,tj->ti", r_t, t_t1 - t_t).astype(np.float32)
    return relative


def _compute_current_distance_statistics(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    *,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute current-frame object/hand distances in bounded CUDA batches."""
    num_pairs, num_obj, _ = obj_points.shape
    num_hand = hand_points.shape[1]
    obj_to_hand = np.empty((num_pairs, num_obj), dtype=np.float32)
    hand_to_obj = np.empty((num_pairs, num_hand), dtype=np.float32)
    batch_size = max(1, int(frame_batch_size))
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        obj = torch.from_numpy(obj_points[start:end]).to(device=device, dtype=torch.float32)
        hand = torch.from_numpy(hand_points[start:end]).to(device=device, dtype=torch.float32)
        distance = torch.cdist(obj, hand)
        obj_to_hand[start:end] = distance.amin(dim=-1).cpu().numpy().astype(np.float32)
        hand_to_obj[start:end] = distance.amin(dim=1).cpu().numpy().astype(np.float32)
    candidate_mask = obj_to_hand <= float(candidate_threshold)
    return obj_to_hand, hand_to_obj, candidate_mask


def _mirror_x(
    obj_points: np.ndarray,
    obj_normals: np.ndarray,
    obj_flow: np.ndarray,
    hand_points: np.ndarray,
    hand_normals: np.ndarray,
    hand_flow: np.ndarray,
    wrist_delta: np.ndarray,
    hand_cano_points: np.ndarray,
) -> None:
    """Reflect a complete temporal pair into the right-hand convention in place."""
    for value in (obj_points, obj_normals, obj_flow, hand_points, hand_normals, hand_flow):
        value[..., 0] *= -1.0
    hand_cano_points[..., 0] *= -1.0
    reflection = np.diag(np.asarray([-1.0, 1.0, 1.0], dtype=np.float32))
    rotation = wrist_delta[:, :3, :3]
    translation = wrist_delta[:, :3, 3]
    wrist_delta[:, :3, :3] = np.einsum("ij,tjk,kl->til", reflection, rotation, reflection)
    wrist_delta[:, :3, 3] = np.einsum("ij,tj->ti", reflection, translation)


def _mirror_hand_root_poses_world(hand_root_pose_world: np.ndarray) -> None:
    """Mirror ``T_world<-hand`` consistently with the local-frame reflection."""
    reflection = np.eye(4, dtype=np.float32)
    reflection[0, 0] = -1.0
    hand_root_pose_world[...] = np.einsum(
        "ij,tjk,kl->til", reflection, hand_root_pose_world, reflection
    )


def _scalar_from_raw(path: Path, key: str, default: float) -> float:
    with np.load(path, allow_pickle=True) as raw:
        if key not in raw.files:
            return float(default)
        value = np.asarray(raw[key])
        return float(value.item()) if value.size == 1 else float(default)


def build_stage4_sequence(
    source: dict[str, Any],
    *,
    source_path: Path,
    grab_root: Path,
    side: str,
    pair_stride: int,
    pair_hop: int,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
    ds_rate: int,
    mirror_left_to_right: bool,
) -> dict[str, np.ndarray]:
    """Convert one reconstructed raw sequence into one side-specific Stage 4 NPZ."""
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported side {side!r}")
    root_key = f"{side}_hand_root_pose"
    if root_key not in source:
        raise ValueError(f"{source['seq_id']} has no reconstructed {side}-hand trajectory")

    total_frames = int(np.asarray(source["raw_frame_id"]).shape[0])
    if pair_stride <= 0 or pair_stride >= total_frames:
        raise ValueError(
            f"pair_stride must be in [1, {total_frames - 1}] for {source['seq_id']}, got {pair_stride}"
        )
    if pair_hop <= 0:
        raise ValueError(f"pair_hop must be positive, got {pair_hop}")
    # The default raw-frame policy is an overlapping temporal window:
    # raw 0->3, 1->4, 2->5, ... (ds_rate=1, pair_stride=3, pair_hop=1).
    current_idx = np.arange(0, total_frames - pair_stride, pair_hop, dtype=np.int64)
    next_idx = current_idx + int(pair_stride)
    # These are T_world<-hand.  Keep them in Stage 4 so visualizers can
    # switch between the training hand-root frame and the original world frame
    # without changing any model input or target.
    hand_root_t = np.asarray(source[root_key], dtype=np.float32)[current_idx]
    hand_root_t1 = np.asarray(source[root_key], dtype=np.float32)[next_idx]
    hand_root_pose_world = hand_root_t.copy()
    next_hand_root_pose_world = hand_root_t1.copy()

    obj_world = np.asarray(source["obj_points_world"], dtype=np.float32)
    obj_normals_world = np.asarray(source["obj_normals_world"], dtype=np.float32)
    hand_world = np.asarray(source[f"{side}_hand_points_world"], dtype=np.float32)
    hand_normals_world = np.asarray(source[f"{side}_hand_normals_world"], dtype=np.float32)

    obj_points = _points_world_to_current_hand(obj_world[current_idx], hand_root_t)
    obj_normals = _normals_world_to_current_hand(obj_normals_world[current_idx], hand_root_t)
    obj_next_in_current_frame = _points_world_to_current_hand(obj_world[next_idx], hand_root_t)
    obj_flow = (obj_next_in_current_frame - obj_points).astype(np.float32)
    hand_points = _points_world_to_current_hand(hand_world[current_idx], hand_root_t)
    hand_normals = _normals_world_to_current_hand(hand_normals_world[current_idx], hand_root_t)
    hand_next_in_current_frame = _points_world_to_current_hand(hand_world[next_idx], hand_root_t)
    hand_flow = (hand_next_in_current_frame - hand_points).astype(np.float32)
    wrist_delta = _relative_wrist_pose(hand_root_t, hand_root_t1)
    hand_cano_points = np.asarray(source[f"{side}_hand_cano_points"], dtype=np.float32).copy()

    if mirror_left_to_right and side == "left":
        _mirror_x(
            obj_points,
            obj_normals,
            obj_flow,
            hand_points,
            hand_normals,
            hand_flow,
            wrist_delta,
            hand_cano_points,
        )
        _mirror_hand_root_poses_world(hand_root_pose_world)
        _mirror_hand_root_poses_world(next_hand_root_pose_world)

    obj_to_hand_min_dist, hand_to_obj_min_dist, candidate_mask = _compute_current_distance_statistics(
        obj_points,
        hand_points,
        candidate_threshold=candidate_threshold,
        frame_batch_size=frame_batch_size,
        device=device,
    )
    source_rel = source_path.resolve().relative_to(grab_root.resolve()).as_posix()
    framerate = _scalar_from_raw(source_path, "framerate", default=30.0)
    time_delta = float(pair_stride * ds_rate) / max(framerate, 1e-8)

    return {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION),
        "source_raw_file": np.asarray(source_rel),
        "dataset_name": np.asarray(str(source["dataset_name"])),
        "seq_id": np.asarray(str(source["seq_id"])),
        "subject_id": np.asarray(str(source["subject_id"])),
        "seq_name": np.asarray(str(source["seq_name"])),
        "object_name": np.asarray(str(source["object_name"])),
        "side": np.asarray(side),
        "raw_frame_id": np.asarray(source["raw_frame_id"], dtype=np.int32)[current_idx],
        "next_raw_frame_id": np.asarray(source["raw_frame_id"], dtype=np.int32)[next_idx],
        "pair_index": np.arange(len(current_idx), dtype=np.int32),
        "time_delta_sec": np.full((len(current_idx),), time_delta, dtype=np.float32),
        "coordinate_frame": np.asarray("hand_root_t"),
        "hand_root_pose_world": hand_root_pose_world,
        "next_hand_root_pose_world": next_hand_root_pose_world,
        "obj_points": obj_points,
        "obj_normals": obj_normals,
        "obj_point_id": np.asarray(source["obj_point_id"], dtype=np.int32),
        "obj_flow_gt": obj_flow,
        "hand_points": hand_points,
        "hand_normals": hand_normals,
        "hand_point_id": np.asarray(source[f"{side}_hand_point_id"], dtype=np.int32),
        "hand_cano_points": hand_cano_points,
        "hand_finger_id": np.asarray(source[f"{side}_hand_finger_id"], dtype=np.int32),
        "hand_region_id": np.asarray(source[f"{side}_hand_region_id"], dtype=np.int32),
        "hand_flow": hand_flow,
        "wrist_delta": wrist_delta,
        "obj_to_hand_min_dist": obj_to_hand_min_dist,
        "hand_to_obj_min_dist": hand_to_obj_min_dist,
        "obj_candidate_mask_5cm": candidate_mask,
    }


def _write_meta(output_root: Path, args: argparse.Namespace, stats: dict[str, int]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "raw GRAB; no Stage 2 or Stage 3 dependency",
        "source_grab_root": str(Path(args.grab_root).resolve()),
        "output_root": str(output_root.resolve()),
        "sample_unit": "single_sequence_single_hand_temporal_pair",
        "coordinate_frame": "hand_root_t",
        "world_pose_fields": "hand_root_pose_world, next_hand_root_pose_world (T_world<-hand)",
        "future_object_policy": "not stored; obj_flow_gt is supervision only",
        "num_obj_pool": int(args.num_obj_points),
        "num_hand_points": 1538,
        "ds_rate": int(args.ds_rate),
        "pair_stride": int(args.pair_stride),
        "pair_hop": int(args.pair_hop),
        "candidate_threshold": float(args.candidate_threshold),
        "mirror_left_to_right": bool(args.mirror_left_to_right),
        "stats": stats,
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw GRAB -> temporal Cm Stage 4 pairs")
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
        default=1,
        help="Raw-frame subsampling before temporal pairing; default keeps every GRAB frame.",
    )
    parser.add_argument(
        "--pair-stride",
        type=int,
        default=3,
        help="Future-index offset in the ds-rate sampled sequence; default is raw 1->4.",
    )
    parser.add_argument(
        "--pair-hop",
        type=int,
        default=1,
        help="Current-frame hop in the sampled sequence; default creates overlapping windows 1->4, 2->5, ...",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="m")
    parser.add_argument("--mirror-left-to-right", action="store_true")
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
        grab_root=str(grab_root),
        mano_path=args.mano_path,
        ds_rate=args.ds_rate,
        obj_unit=args.obj_unit,
        nn_batch_size=args.nn_batch_size,
    )
    sides = ("left", "right") if args.side == "both" else (args.side,)
    stats = {"source_sequences": len(sequences), "written": 0, "skipped": 0, "empty_side": 0, "failed": 0, "pairs": 0}

    for raw_path_text in sequences:
        raw_path = Path(raw_path_text).resolve()
        try:
            source = adapter.process_sequence(str(raw_path))
            for side in sides:
                output_path = output_root / str(source["subject_id"]) / f"{source['seq_name']}_{side}.npz"
                if output_path.exists() and not args.overwrite:
                    stats["skipped"] += 1
                    continue
                try:
                    stage4 = build_stage4_sequence(
                        source,
                        source_path=raw_path,
                        grab_root=grab_root,
                        side=side,
                        pair_stride=args.pair_stride,
                        pair_hop=args.pair_hop,
                        candidate_threshold=args.candidate_threshold,
                        frame_batch_size=args.frame_batch_size,
                        device=device,
                        ds_rate=args.ds_rate,
                        mirror_left_to_right=args.mirror_left_to_right,
                    )
                except ValueError as exc:
                    if "no reconstructed" in str(exc):
                        stats["empty_side"] += 1
                        continue
                    raise
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if args.save_compressed:
                    np.savez_compressed(output_path, **stage4)
                else:
                    np.savez(output_path, **stage4)
                stats["written"] += 1
                stats["pairs"] += int(stage4["pair_index"].shape[0])
                candidate_counts = stage4["obj_candidate_mask_5cm"].sum(axis=1)
                print(
                    f"[stage4] wrote {output_path} pairs={len(stage4['pair_index'])} "
                    f"candidate[min/median/max]={int(candidate_counts.min())}/"
                    f"{int(np.median(candidate_counts))}/{int(candidate_counts.max())}"
                )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage4] failed {raw_path}: {exc}")
            traceback.print_exc()

    _write_meta(output_root, args, stats)
    print(f"[stage4] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
