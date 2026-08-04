from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from process.ARCTIC.raw import assign_hand_semantics, compute_canonical_hand_surface
from process.ContactPose.stage3_export import (
    DEFAULT_CONTACTPOSE_ROOT,
    DEFAULT_MANO_PATH,
    HAND_SIDES,
    ManoLayerCache,
    SequenceRef,
    _build_hand_sequence,
    _enumerate_sequences,
)
from process.common.stage4_cm import (
    ROOT,
    build_hand_sequence,
    build_shared_sequence,
    resolve_device,
    write_manifest,
    write_meta,
)


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed_data" / "stage4" / "data" / "contactpose"


@dataclass(frozen=True)
class ContactPoseSource:
    source: dict[str, np.ndarray]
    source_path: Path


def _mano_semantics(mano_cache: ManoLayerCache, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer = mano_cache.get(side, 15)
    finger_id, region_id = assign_hand_semantics(layer, is_right=(side == "right"))
    cano_points, _cano_normals = compute_canonical_hand_surface(layer)
    return (
        np.asarray(cano_points, dtype=np.float32),
        np.asarray(finger_id, dtype=np.int32),
        np.asarray(region_id, dtype=np.int32),
    )


def _to_source(
    sequence,
    *,
    side: str,
    cano_points: np.ndarray,
    finger_id: np.ndarray,
    region_id: np.ndarray,
) -> ContactPoseSource:
    num_frames = int(sequence.num_frames)
    identity_pose = np.broadcast_to(np.eye(4, dtype=np.float32)[None, :, :], (num_frames, 4, 4)).copy()
    seq_dir = sequence.ref.path.resolve()
    source = {
        "seq_id": sequence.ref.seq_id,
        "dataset_name": "contactpose",
        "subject_id": sequence.ref.subject_id,
        "seq_name": sequence.ref.seq_name,
        "object_name": sequence.ref.object_name,
        "raw_frame_id": np.arange(num_frames, dtype=np.int32),
        "obj_points_world": np.asarray(sequence.obj_points, dtype=np.float32),
        "obj_normals_world": np.asarray(sequence.obj_normals, dtype=np.float32),
        "obj_point_id": np.arange(sequence.obj_points.shape[1], dtype=np.int32),
        f"{side}_hand_points_world": np.asarray(sequence.hand_points, dtype=np.float32),
        f"{side}_hand_normals_world": np.asarray(sequence.hand_normals, dtype=np.float32),
        f"{side}_hand_point_id": np.arange(sequence.hand_points.shape[1], dtype=np.int32),
        f"{side}_hand_cano_points": np.asarray(cano_points, dtype=np.float32),
        f"{side}_hand_finger_id": np.asarray(finger_id, dtype=np.int32),
        f"{side}_hand_region_id": np.asarray(region_id, dtype=np.int32),
        f"{side}_hand_root_pose": identity_pose,
    }
    return ContactPoseSource(source=source, source_path=seq_dir / "annotations.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ContactPose -> Cm sequence cache")
    parser.add_argument("--contactpose-root", default=str(DEFAULT_CONTACTPOSE_ROOT))
    parser.add_argument("--mano-path", default=str(DEFAULT_MANO_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--intent", choices=("use", "handoff"), default="use")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seq", default=None, help="Exact substring filter on subject_intent_object")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_fps <= 0.0:
        raise SystemExit("--source-fps must be positive")

    contactpose_root = Path(args.contactpose_root).resolve()
    mano_path = Path(args.mano_path).resolve()
    output_root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    refs = _enumerate_sequences(contactpose_root, intent=args.intent)
    if args.seq:
        refs = [ref for ref in refs if args.seq in ref.seq_name or args.seq in ref.seq_id]
    if args.limit is not None:
        refs = refs[: int(args.limit)]
    if not refs:
        raise FileNotFoundError("No ContactPose sequences matched the requested input")

    mano_cache = ManoLayerCache(mano_path)
    semantics = {
        side: _mano_semantics(mano_cache, side) for side in HAND_SIDES
    }
    side_filter = set(HAND_SIDES if args.side == "both" else (args.side,))
    stats = {
        "source_sequences": len(refs),
        "shared_written": 0,
        "hand_written": 0,
        "skipped": 0,
        "empty_side": 0,
        "failed": 0,
        "frames": 0,
    }

    for ref in refs:
        seq_dir = output_root / ref.subject_id / ref.seq_name
        shared_path = seq_dir / "shared.npz"
        wrote_shared = False
        try:
            valid_hands: list[tuple[int, str]] = []
            # Match the current ContactPose exporter convention.
            hands_meta = json.loads((ref.path / "annotations.json").read_text(encoding="utf-8"))["hands"]
            for hand_idx, hand_meta in enumerate(hands_meta[: len(HAND_SIDES)]):
                side = HAND_SIDES[hand_idx]
                if side not in side_filter:
                    continue
                if bool(hand_meta.get("valid")):
                    valid_hands.append((hand_idx, side))
            if not valid_hands:
                stats["empty_side"] += len(side_filter)
                continue
            for hand_idx, side in valid_hands:
                output_path = seq_dir / f"{side}.npz"
                if output_path.exists() and not args.overwrite:
                    stats["skipped"] += 1
                    continue
                sequence = _build_hand_sequence(
                    ref,
                    hand_idx=hand_idx,
                    side=side,
                    mano_cache=mano_cache,
                    num_obj_pool=4096,
                )
                cano_points, finger_id, region_id = semantics[side]
                prepared = _to_source(
                    sequence,
                    side=side,
                    cano_points=cano_points,
                    finger_id=finger_id,
                    region_id=region_id,
                )
                if not wrote_shared and (args.overwrite or not shared_path.exists()):
                    shared = build_shared_sequence(
                        prepared.source,
                        source_path=prepared.source_path,
                        source_root=contactpose_root,
                        ds_rate=1,
                        source_fps=args.source_fps,
                    )
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    if args.save_compressed:
                        np.savez_compressed(shared_path, **shared)
                    else:
                        np.savez(shared_path, **shared)
                    wrote_shared = True
                    stats["shared_written"] += 1
                hand = build_hand_sequence(
                    prepared.source,
                    side=side,
                    candidate_threshold=args.candidate_threshold,
                    frame_batch_size=args.frame_batch_size,
                    device=device,
                )
                if args.save_compressed:
                    np.savez_compressed(output_path, **hand)
                else:
                    np.savez(output_path, **hand)
                stats["hand_written"] += 1
                stats["frames"] += int(hand["obj_candidate_mask_5cm"].shape[0])
                candidate_counts = hand["obj_candidate_mask_5cm"].sum(axis=1)
                print(
                    f"[stage4-contactpose] wrote {output_path} frames={len(hand['obj_candidate_mask_5cm'])} "
                    f"candidate[min/median/max]={int(candidate_counts.min())}/"
                    f"{int(np.median(candidate_counts))}/{int(candidate_counts.max())}"
                )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage4-contactpose] failed {ref.seq_id}: {exc}")
            traceback.print_exc()

    write_meta(
        output_root,
        source_description="ContactPose annotations exported to Cm sequence cache",
        source_root=contactpose_root,
        ds_rate=1,
        source_fps=args.source_fps,
        candidate_threshold=args.candidate_threshold,
        stats=stats,
        extra={
            "dataset_name": "contactpose",
            "intent": args.intent,
            "mano_path": str(mano_path.resolve()),
            "note": "hand_root_pose_world is identity because ContactPose is exported in hand-root coordinates",
        },
    )
    write_manifest(output_root)
    print(f"[stage4-contactpose] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
