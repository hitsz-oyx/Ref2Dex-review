"""Build a 30 Hz KNN/distance viewer adapter from a temporal pilot report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from src.task.ObjectInteractionCm.research.oakink2_segment_visualizer.run import (
    ManoReconstructor,
    Stage3ObjectCatalog,
    _frame_ids_for_mode,
    _load_annotation,
    _load_catalog,
    _load_trajectory,
    _sample_rows,
)
from src.task.ObjectInteractionCm.research.oakink2_temporal_segmentation.run import (
    WORK_VERSION,
    SOURCE_MOCAP_FPS,
    TARGET_FPS,
    _git_state,
    _official_30hz_timeline,
    _snapshot,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _candidate_mask(object_points: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    result = np.empty(object_points.shape[:2], dtype=bool)
    for frame in range(len(object_points)):
        result[frame] = (
            cKDTree(hand_points[frame]).query(object_points[frame], k=1, workers=-1)[0]
            <= 0.05
        )
    return result


def _save_sequence(
    output: Path,
    *,
    seq_id: str,
    object_name: str,
    source_report: Path,
    frame_ids: np.ndarray,
    object_points: np.ndarray,
    context_points: np.ndarray,
    hands: Mapping[str, np.ndarray],
    hand_faces: Mapping[str, np.ndarray],
    candidates: Mapping[str, np.ndarray],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "shared.npz",
        schema_name=np.asarray("ref2dex_cm_sequence_shared"),
        schema_version=np.asarray("3.0.0"),
        source_raw_file=np.asarray(str(source_report)),
        dataset_name=np.asarray("oakink2"),
        seq_id=np.asarray(seq_id),
        subject_id=np.asarray("official_30hz_review"),
        seq_name=np.asarray(seq_id),
        object_name=np.asarray(object_name),
        raw_frame_id=frame_ids.astype(np.int32),
        ds_rate=np.asarray(SOURCE_MOCAP_FPS // TARGET_FPS, dtype=np.int32),
        source_fps=np.asarray(float(SOURCE_MOCAP_FPS), dtype=np.float32),
        coordinate_frame=np.asarray("world"),
        obj_points_world=object_points.astype(np.float32),
        obj_normals_world=np.zeros_like(object_points, dtype=np.float32),
        context_points_world=context_points.astype(np.float32),
        obj_point_id=np.arange(object_points.shape[1], dtype=np.int32),
    )
    for side in ("left", "right"):
        hand = hands[side].astype(np.float32)
        np.savez_compressed(
            output / f"{side}.npz",
            schema_name=np.asarray("ref2dex_cm_sequence_hand"),
            schema_version=np.asarray("3.0.0"),
            side=np.asarray(side),
            hand_points_world=hand,
            hand_normals_world=np.zeros_like(hand),
            hand_point_id=np.arange(hand.shape[1], dtype=np.int32),
            hand_mesh_vertices_world=hand,
            hand_mesh_faces=hand_faces[side].astype(np.int32),
            obj_candidate_mask_5cm=candidates[side],
        )


def run(args: argparse.Namespace) -> dict:
    started_at = _now()
    start_time = time.time()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    sampling = report.get("sampling", {})
    if sampling.get("source_mocap_fps") != SOURCE_MOCAP_FPS:
        raise ValueError("report is not sourced from 120 Hz OakInk2 mocap")
    if sampling.get("target_fps") != TARGET_FPS:
        raise ValueError("report is not sampled on the official 30 Hz timeline")

    index, records = _load_catalog(args.index, args.previous_index)
    records_by_id = {record.id: record for record in records}
    selected_rows = [row for row in report["segments"] if row["temporal"]["status"] == "selected"]
    if not selected_rows:
        raise ValueError("report has no selected trajectories")

    args.output.mkdir(parents=True, exist_ok=False)
    sequences_root = args.output / "sequences"
    sequences_root.mkdir()
    object_catalog = Stage3ObjectCatalog(Path(index["stage3_root"]))
    mano = ManoReconstructor(args.mano_root, args.threads)
    entries = []
    details = []

    for ordinal, row in enumerate(selected_rows, 1):
        segment_id = str(row["segment_id"])
        record = records_by_id[segment_id]
        annotation = _load_annotation(Path(index["annotation_root"]), str(record.row["sequence"]))
        mocap_ids = _frame_ids_for_mode(record, annotation["raw_mano"].keys(), "primitive")
        full_ids, _ = _official_30hz_timeline(mocap_ids, annotation["frame_id_list"])
        trajectory = _load_trajectory(
            record,
            "primitive",
            annotation_root=Path(index["annotation_root"]),
            object_catalog=object_catalog,
            mano=mano,
            frame_ids_override=full_ids,
        )
        selected_points = np.stack([
            _sample_rows(trajectory.object_world(frame, selected=True, points_per_part=4096), 4096)
            for frame in range(trajectory.frame_count)
        ]).astype(np.float32)
        context_rows = [
            trajectory.object_world(frame, selected=False, points_per_part=args.context_points_per_part)
            for frame in range(trajectory.frame_count)
        ]
        context_counts = sorted({len(value) for value in context_rows})
        if len(context_counts) != 1:
            raise ValueError(f"{segment_id} context point count varies: {context_counts}")
        context_points = (
            np.stack(context_rows).astype(np.float32)
            if context_counts[0]
            else np.empty((trajectory.frame_count, 0, 3), dtype=np.float32)
        )
        candidates = {
            side: _candidate_mask(selected_points, trajectory.hands[side])
            for side in ("left", "right")
        }
        full_index = {int(frame): index for index, frame in enumerate(full_ids.tolist())}
        cut_ids = np.asarray(row["temporal"]["selected_frame_ids"], dtype=np.int64)
        missing = [int(frame) for frame in cut_ids if int(frame) not in full_index]
        if missing:
            raise ValueError(f"{segment_id} selected frames outside official 30 Hz primitive: {missing[:5]}")
        cut_indices = np.asarray([full_index[int(frame)] for frame in cut_ids], dtype=np.int64)

        safe_segment = segment_id.replace(":", "_")
        primitive = str(row.get("primitive") or "unknown")
        object_name = str(row.get("selected_object_name") or "object")
        for mode, indices in (
            ("cut", cut_indices),
            ("full", np.arange(len(full_ids), dtype=np.int64)),
        ):
            seq_id = f"official30hz_{safe_segment}_{mode}"
            seq_dir = sequences_root / seq_id
            _save_sequence(
                seq_dir,
                seq_id=seq_id,
                object_name=object_name,
                source_report=args.report,
                frame_ids=full_ids[indices],
                object_points=selected_points[indices],
                context_points=context_points[indices],
                hands={side: trajectory.hands[side][indices] for side in ("left", "right")},
                hand_faces=trajectory.hand_faces,
                candidates={side: candidates[side][indices] for side in ("left", "right")},
            )
            entries.append({
                "id": f"oakink2/{seq_id}",
                "source": "oakink2_mano",
                "dataset": "oakink2",
                "variant": "mano_bilateral_raw",
                "object_name": f"{object_name} | {primitive} | {mode} | official 30 Hz",
                "split": "train",
                "path": str(seq_dir.resolve()),
                "segment_id": segment_id,
                "primitive": primitive,
                "review_mode": mode,
                "frame_count": int(len(indices)),
                "source_frame_start": int(full_ids[indices[0]]),
                "source_frame_end": int(full_ids[indices[-1]]),
            })
        details.append({
            "segment_id": segment_id,
            "object": object_name,
            "primitive": primitive,
            "full_30hz_frames": int(len(full_ids)),
            "cut_30hz_frames": int(len(cut_ids)),
            "context_points_per_frame": int(context_points.shape[1]),
            "missing_context_object_ids": list(trajectory.missing_object_points),
        })
        print(
            f"[{ordinal}/{len(selected_rows)}] {segment_id} {object_name} "
            f"full={len(full_ids)} cut={len(cut_ids)}",
            flush=True,
        )

    index_payload = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
        "schema_version": "1.2.0",
        "created_at": _now(),
        "purpose": "OakInk2 official 30 Hz temporal cut/full task-semantic review",
        "source_report": str(args.report.resolve()),
        "sequences": {"train": entries, "val": [], "test": []},
    }
    _write_json(args.output / "index.json", index_payload)
    commit, dirty = _git_state()
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "oakink2_official_30hz_knn_viewer_adapter",
        "run_id": args.output.name,
        "run_status": "COMPLETED",
        "started_at": started_at,
        "completed_at": _now(),
        "work_version": WORK_VERSION,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "report": _snapshot(args.report),
            "index": _snapshot(args.index),
            "previous_index": _snapshot(args.previous_index),
            "mano_root": str(args.mano_root.resolve()),
        },
        "sampling": report["sampling"],
        "excluded_segment_ids": report.get("excluded_segment_ids", []),
        "counts": {
            "selected_segments": len(selected_rows),
            "viewer_sequences": len(entries),
            "full_30hz_frames": sum(value["full_30hz_frames"] for value in details),
            "cut_30hz_frames": sum(value["cut_30hz_frames"] for value in details),
        },
        "details": details,
        "outputs": {"index": "index.json", "sequences": "sequences"},
        "elapsed_s": time.time() - start_time,
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(args.output / "run_manifest.json", manifest)
    print(json.dumps({"output": str(args.output.resolve()), **manifest["counts"]}, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/index.json"),
    )
    parser.add_argument(
        "--previous-index",
        type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/index.json"),
    )
    parser.add_argument("--mano-root", type=Path, default=Path("dataset/arctic/data/body_models/mano"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-points-per-part", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.context_points_per_part < 1 or args.threads < 1:
        raise ValueError("context-points-per-part and threads must be positive")
    for path in (args.report, args.index, args.previous_index, args.mano_root):
        if not path.exists():
            raise FileNotFoundError(path)
    run(args)


if __name__ == "__main__":
    main()
