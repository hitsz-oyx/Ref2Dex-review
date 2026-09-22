#!/usr/bin/env python3
"""Materialize metadata inputs for the bounded V1.14b real-data six-group smoke."""
from __future__ import annotations

import argparse
import ast
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.task.ObjectInteractionCm.tools.data.build_oakink2_single_object_index import _root_map


REPO_ROOT = Path(__file__).resolve().parents[5]
SELECTION_SCHEMA = "ref2dex_oakink2_inspire_selection_v1_4"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _program_row(program: dict[str, Any], roots: dict[str, str], sequence: str,
                 selected_frames: list[int], ordinal: int) -> dict[str, Any]:
    matches = []
    for frame_range_def, item in program.items():
        ranges = ast.literal_eval(frame_range_def)
        usable = [value for value in ranges if value is not None]
        start, end = min(value[0] for value in usable), max(value[1] for value in usable)
        if min(selected_frames) < start or max(selected_frames) > end:
            continue
        object_ids = [str(value) for value in item.get("obj_list", [])]
        instances = sorted({roots.get(value, value) for value in object_ids})
        if len(instances) == 1:
            matches.append((frame_range_def, item, object_ids, instances[0], start, end, ranges))
    if len(matches) != 1:
        raise ValueError(f"{sequence}: smoke window must match exactly one single-root primitive")
    frame_range_def, item, object_ids, instance, start, end, ranges = matches[0]
    sides = [side for side, value in zip(("left", "right"), ranges) if value is not None]
    return {
        "id": f"oakink2/{sequence}/{ordinal:04d}", "sequence": sequence,
        "frame_range_def": frame_range_def, "primitive": item.get("primitive"),
        "interaction_mode": item.get("interaction_mode"), "instance_id": instance,
        "object_ids": object_ids, "selected_object_ids": object_ids,
        "selected_root_id": instance, "selected_hand_sides": sides,
        "frame_range": [int(start), int(end)], "primitive_mocap_frame_count": len(selected_frames),
        "primitive_frame_range": [selected_frames[0], selected_frames[-1]],
        "primitive_frame_count": len(selected_frames),
        "official_timeline_positions": [], "selected_frame_ids": selected_frames,
        "selected_timeline_positions": [], "selected_frame_count": len(selected_frames),
        "motion_frame_ranges": [[selected_frames[0], selected_frames[-1]]],
        "temporal": {"status": "bounded_interface_smoke_only"},
    }


def build(spec_path: Path, annotation_root: Path, object_root: Path, stage3_root: Path,
          output_root: Path, run_id: str) -> None:
    targets = (
        output_root / "source_split.json",
        output_root / "oakink_selection/index.json",
        output_root / "oakink_selection/manifest.json",
        output_root / "metadata_run_manifest.json",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite smoke metadata: {existing}")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_name") != "object_interaction_cmv2_six_source_smoke_inputs_v1_14b":
        raise ValueError("unexpected six-source smoke input spec")
    source_split = {"schema_name": "ref2dex_cmv2_v114b_smoke_source_assignment",
                    "work_version": "V1.14", "sequences": {"train": [], "val": [], "test": []}}
    for split in ("train", "val"):
        source_split["sequences"][split] = [dict(value, split=split) for value in spec["stage4"][split]]

    tree = json.loads((object_root / "object_affordance/object_part_tree.json").read_text(encoding="utf-8"))
    roots = _root_map(tree)
    selected_rows = []
    for ordinal, value in enumerate(spec["oakink2"]):
        sequence = str(value["sequence"])
        with (annotation_root / f"{sequence}.pkl").open("rb") as stream:
            annotation = pickle.load(stream)
        official = [int(frame) for frame in annotation["frame_id_list"]]
        start = int(value["raw_frame_start"]); stop = start + int(value["raw_frame_count"])
        frames = [frame for frame in official if start <= frame < stop]
        if len(frames) < 4:
            raise ValueError(f"{sequence}: smoke window has fewer than four official 30 Hz frames")
        program = json.loads((object_root / "program/program_info" / f"{sequence}.json").read_text())
        row = _program_row(program, roots, sequence, frames, ordinal)
        positions = {int(frame): index for index, frame in enumerate(official)}
        row["official_timeline_positions"] = [positions[frame] for frame in frames]
        row["selected_timeline_positions"] = list(row["official_timeline_positions"])
        for object_id in row["selected_object_ids"]:
            if not any(stage3_root.glob(f"*_{object_id.replace('@', '_')}_*.npz")):
                raise FileNotFoundError(f"{sequence}/{object_id}: Stage3 smoke geometry missing")
        selected_rows.append(row)

    selection = {
        "schema_name": SELECTION_SCHEMA, "schema_version": "1.0.0", "work_version": "V1.14",
        "source": "bounded real-data interface smoke; not a scientific selection",
        "created_at": _now(), "selection_policy": {"smoke_only": True, "official_timeline": True},
        "input_segment_count": len(selected_rows), "segment_count": len(selected_rows),
        "selected_frame_count": sum(row["selected_frame_count"] for row in selected_rows),
        "rejected_counts": {}, "failures": [], "segments": selected_rows, "conclusion": "N/A",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "source_split.json", source_split)
    _write_json(output_root / "oakink_selection/index.json", selection)
    _write_json(output_root / "oakink_selection/manifest.json", {
        "schema_name": SELECTION_SCHEMA, "work_version": "V1.14", "smoke_only": True,
        "segment_count": len(selected_rows), "conclusion": "N/A"})
    _write_json(output_root / "metadata_run_manifest.json", {
        "schema_name": "ref2dex_data_run_manifest_v1", "task": "ObjectInteractionCmv2",
        "work_version": "V1.14", "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "run_id": run_id, "run_status": "COMPLETED", "operation": "prepare_v114b_six_source_smoke_metadata",
        "created_at": _now(), "inputs": {"spec": str(spec_path.resolve()),
        "annotation_root": str(annotation_root.resolve()), "object_root": str(object_root.resolve()),
        "stage3_root": str(stage3_root.resolve())}, "outputs": {"root": str(output_root.resolve())},
        "conclusion": "N/A"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    build(args.spec.resolve(), args.annotation_root.resolve(), args.object_root.resolve(),
          args.stage3_root.resolve(), args.output_root.resolve(), args.run_id)


if __name__ == "__main__":
    main()
