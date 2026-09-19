"""Run the OakInk2 30 Hz temporal-segmentation pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np

from src.task.ObjectInteractionCm.research.oakink2_segment_visualizer.run import (
    _category_groups,
    _frame_ids_for_mode,
    _load_catalog,
)
from src.task.ObjectInteractionCm.tools.data.oakink2_temporal_segments import (
    TemporalSegmentationConfig,
    segment_motion_then_contact,
)
from src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool import (
    _frame_distances,
    _load_stage3_map,
)


REPO = Path(__file__).resolve().parents[5]
DEFAULT_INDEX = Path("data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/index.json")
DEFAULT_PREVIOUS_INDEX = Path("data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/index.json")
DEFAULT_SELECTED_IDS = ("selected:1344", "selected:1772", "selected:1726")
MODIFICATION_VERSION = "V1.4.22"
SOURCE_MOCAP_FPS = 120
TARGET_FPS = 30


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _snapshot(path: Path) -> dict:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
    return commit, dirty


def _official_30hz_timeline(
    primitive_mocap_frame_ids: Sequence[int],
    video_aligned_frame_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Select the official video-aligned 30 Hz samples from a mocap primitive.

    OakInk2 mocap is 120 Hz, while ``frame_id_list`` stores the mocap frame
    associated with each 30 Hz RGB frame.  Its raw-frame increments can vary
    (observed 3/4/5), so fixed modulo/fixed-step sampling would accumulate
    phase error.
    The returned timeline positions are global RGB-frame ordinals and are
    used for temporal connectivity while raw mocap IDs remain the output IDs.
    """

    primitive = {int(value) for value in primitive_mocap_frame_ids}
    aligned = [(index, int(frame)) for index, frame in enumerate(video_aligned_frame_ids)
               if int(frame) in primitive]
    if not aligned:
        raise ValueError("primitive has no official 30 Hz video-aligned frames")
    positions = np.asarray([value[0] for value in aligned], dtype=np.int64)
    frame_ids = np.asarray([value[1] for value in aligned], dtype=np.int64)
    if len(set(frame_ids.tolist())) != len(frame_ids) or np.any(np.diff(frame_ids) <= 0):
        raise ValueError("official 30 Hz frame IDs must be strictly increasing and unique")
    if np.any(np.diff(positions) != 1):
        raise ValueError("primitive official 30 Hz timeline must be contiguous")
    return frame_ids, positions


def run(args: argparse.Namespace) -> dict:
    index, records = _load_catalog(args.index, args.previous_index)
    by_id = {record.id: record for record in records}
    static_records = _category_groups(records)["static"]
    requested_before_exclusion = list(args.segment or DEFAULT_SELECTED_IDS)
    requested_before_exclusion.extend(record.id for record in static_records)
    requested_before_exclusion = list(dict.fromkeys(requested_before_exclusion))
    excluded = list(dict.fromkeys(args.exclude_segment or []))
    missing = [
        segment_id for segment_id in [*requested_before_exclusion, *excluded]
        if segment_id not in by_id
    ]
    if missing:
        raise ValueError(f"unknown segment ids: {missing}")
    requested = [segment_id for segment_id in requested_before_exclusion if segment_id not in excluded]
    if not requested:
        raise ValueError("all requested segments were excluded")

    config = TemporalSegmentationConfig(
        window_radius=args.window_radius,
        window_translation_m=args.window_translation_mm / 1000.0,
        window_rotation_deg=args.window_rotation_deg,
        part_consensus_ratio=args.part_consensus_ratio,
        motion_max_gap=args.motion_max_gap,
        motion_min_evidence_frames=args.motion_min_evidence_frames,
        contact_threshold_m=args.contact_threshold_cm / 100.0,
        contact_max_gap=args.contact_max_gap,
    )
    config.validate()
    stage3 = _load_stage3_map(Path(index["stage3_root"]))
    distance_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for segment_id in requested:
        record = by_id[segment_id]
        sequence = str(record.row["sequence"])
        annotation_path = Path(index["annotation_root"]) / f"{sequence}.pkl"
        with annotation_path.open("rb") as stream:
            annotation = pickle.load(stream)
        mocap_frame_ids = _frame_ids_for_mode(record, annotation["raw_mano"].keys(), "primitive")
        frame_ids, timeline_positions = _official_30hz_timeline(
            mocap_frame_ids,
            annotation["frame_id_list"],
        )
        object_ids = [str(value) for value in record.row["selected_object_ids"]]
        pose_by_part = {
            object_id: {
                int(frame): np.asarray(annotation["obj_transf"][object_id][int(frame)], dtype=np.float32)
                for frame in frame_ids.tolist()
                if object_id in annotation.get("obj_transf", {})
                and int(frame) in annotation["obj_transf"][object_id]
            }
            for object_id in object_ids
        }
        distance_calls = []

        def provide_distances(wanted: Sequence[int]) -> Mapping[int, float]:
            distance_calls.append(len(wanted))
            return _frame_distances(
                stage3,
                sequence,
                object_ids,
                [int(value) for value in wanted],
                ("left", "right"),
                cache=distance_cache,
            )

        result = segment_motion_then_contact(
            pose_by_part,
            frame_ids.tolist(),
            provide_distances,
            config,
            timeline_positions=timeline_positions.tolist(),
        )
        rows.append({
            "segment_id": record.id,
            "segment_label": record.label,
            "source_status": record.status,
            "sequence": sequence,
            "primitive": record.row.get("primitive"),
            "selected_object_name": record.row.get("selected_object_name"),
            "selected_object_ids": object_ids,
            "primitive_mocap_frame_range": [int(mocap_frame_ids[0]), int(mocap_frame_ids[-1])],
            "primitive_mocap_frame_count": int(len(mocap_frame_ids)),
            "primitive_frame_range": [int(frame_ids[0]), int(frame_ids[-1])],
            "primitive_frame_count": int(len(frame_ids)),
            "timeline_position_range": [int(timeline_positions[0]), int(timeline_positions[-1])],
            "old_motion_frame_range": record.row.get("motion_frame_range"),
            "old_selected_frame_ids": record.row.get("selected_frame_ids", []),
            "old_selected_frame_count": int(record.row.get("selected_frame_count", 0)),
            "distance_source": "Stage3 cached hand_to_obj_min_dist" if distance_calls else "not_read",
            "distance_provider_call_count": len(distance_calls),
            "temporal": result,
        })

    counts = Counter(row["temporal"]["status"] for row in rows)
    report = {
        "schema_name": "ref2dex_oakink2_temporal_segmentation_pilot_v1",
        "schema_version": "1.1.0",
        "created_at": _now(),
        "work_version": MODIFICATION_VERSION,
        "source_index": _snapshot(args.index),
        "previous_index": _snapshot(args.previous_index),
        "parameters": rows[0]["temporal"]["config"] if rows else {},
        "sampling": {
            "source_mocap_fps": SOURCE_MOCAP_FPS,
            "target_fps": TARGET_FPS,
            "policy": "official annotation frame_id_list intersection with primitive",
            "output_frame_ids": "original mocap frame IDs on the official 30 Hz RGB-aligned timeline",
        },
        "requested_segment_ids": requested_before_exclusion,
        "excluded_segment_ids": excluded,
        "segment_ids": requested,
        "counts": dict(counts),
        "segments": rows,
        "conclusion": "INCONCLUSIVE",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output / "temporal_segmentation_report.json", report)
    commit, dirty = _git_state()
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "oakink2_temporal_segmentation_pilot",
        "run_id": args.output.name,
        "run_status": "COMPLETED",
        "started_at": report["created_at"],
        "completed_at": _now(),
        "work_version": MODIFICATION_VERSION,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "index": report["source_index"],
            "previous_index": report["previous_index"],
            "annotation_root": str(Path(index["annotation_root"]).resolve()),
            "stage3_root": str(Path(index["stage3_root"]).resolve()),
        },
        "parameters": report["parameters"],
        "sampling": report["sampling"],
        "excluded_segment_ids": excluded,
        "outputs": {"report": "temporal_segmentation_report.json"},
        "counts": {"segments": len(rows), **dict(counts)},
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(args.output / "run_manifest.json", manifest)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "run_status": manifest["run_status"],
        "counts": manifest["counts"],
    }, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--previous-index", type=Path, default=DEFAULT_PREVIOUS_INDEX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment", action="append")
    parser.add_argument(
        "--exclude-segment",
        action="append",
        help="Exact segment ID excluded before motion/contact processing.",
    )
    parser.add_argument("--window-radius", type=int, default=3)
    parser.add_argument("--window-translation-mm", type=float, default=2.0)
    parser.add_argument("--window-rotation-deg", type=float, default=2.0)
    parser.add_argument("--part-consensus-ratio", type=float, default=0.60)
    parser.add_argument("--motion-max-gap", type=int, default=1)
    parser.add_argument("--motion-min-evidence-frames", type=int, default=3)
    parser.add_argument("--contact-threshold-cm", type=float, default=2.0)
    parser.add_argument("--contact-max-gap", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
