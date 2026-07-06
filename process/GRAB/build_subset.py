#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np


REF2DEX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAB_ROOT = REF2DEX_ROOT / "dataset" / "GRAB" / "data"
DEFAULT_OUT_CSV = REF2DEX_ROOT / "tmp" / "manifests" / "grab_subset_100.csv"
DEFAULT_OUT_JSON = REF2DEX_ROOT / "tmp" / "manifests" / "grab_subset_100_summary.json"


def _frame_count(raw_path: Path) -> int:
    data = np.load(str(raw_path), allow_pickle=True)
    if "n_frames" in data.files:
        return int(data["n_frames"])
    for key in ("lhand", "rhand", "object", "body"):
        if key not in data.files:
            continue
        try:
            item = data[key].item()
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        params = item.get("params", {})
        transl = params.get("transl")
        if transl is not None:
            return int(len(transl))
    raise RuntimeError(f"Could not infer frame count from {raw_path}")


def _candidate_rows(grab_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path_text in sorted(glob(str(grab_root / "grab" / "*" / "*.npz"))):
        raw_path = Path(raw_path_text).resolve()
        subject_id = raw_path.parent.name
        seq_name = raw_path.stem
        object_name = seq_name.split("_")[0]
        action_name = seq_name[len(object_name) + 1 :] if seq_name.startswith(object_name + "_") else seq_name
        rows.append(
            {
                "seq_id": f"{subject_id}/{seq_name}",
                "subject_id": subject_id,
                "seq_name": seq_name,
                "object_name": object_name,
                "action_name": action_name,
                "frame_count": _frame_count(raw_path),
                "raw_path": str(raw_path.relative_to(grab_root)),
                "raw_path_abs": str(raw_path),
            }
        )
    return rows


def _select_subset(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if sample_size > len(rows):
        raise ValueError(f"sample_size={sample_size} exceeds dataset size={len(rows)}")

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["object_name"])].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    subject_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    actions_by_object: dict[str, set[str]] = defaultdict(set)

    # Phase 1: guarantee at least one sequence per object, while balancing subjects.
    for object_name in sorted(by_object):
        best = min(
            by_object[object_name],
            key=lambda row: (
                subject_counts[str(row["subject_id"])],
                int(row["frame_count"]),
                str(row["seq_id"]),
            ),
        )
        selected.append(best)
        selected_ids.add(str(best["seq_id"]))
        subject_counts[str(best["subject_id"])] += 1
        object_counts[str(best["object_name"])] += 1
        actions_by_object[str(best["object_name"])].add(str(best["action_name"]))

    if len(selected) > sample_size:
        raise ValueError(
            f"sample_size={sample_size} is smaller than number of GRAB objects={len(selected)}; "
            "cannot keep one sequence per object."
        )

    remaining = [row for row in rows if str(row["seq_id"]) not in selected_ids]

    # Phase 2: fill the rest, preferring new actions, then objects/subjects with fewer samples, then shorter seqs.
    while len(selected) < sample_size:
        best = min(
            remaining,
            key=lambda row: (
                0 if str(row["action_name"]) not in actions_by_object[str(row["object_name"])] else 1,
                object_counts[str(row["object_name"])],
                subject_counts[str(row["subject_id"])],
                int(row["frame_count"]),
                str(row["seq_id"]),
            ),
        )
        remaining.remove(best)
        selected.append(best)
        selected_ids.add(str(best["seq_id"]))
        subject_counts[str(best["subject_id"])] += 1
        object_counts[str(best["object_name"])] += 1
        actions_by_object[str(best["object_name"])].add(str(best["action_name"]))

    selected = sorted(selected, key=lambda row: str(row["seq_id"]))
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "seq_id",
        "subject_id",
        "seq_name",
        "object_name",
        "action_name",
        "frame_count",
        "raw_path",
        "raw_path_abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "sample_index": idx,
                    "seq_id": row["seq_id"],
                    "subject_id": row["subject_id"],
                    "seq_name": row["seq_name"],
                    "object_name": row["object_name"],
                    "action_name": row["action_name"],
                    "frame_count": int(row["frame_count"]),
                    "raw_path": row["raw_path"],
                    "raw_path_abs": row["raw_path_abs"],
                }
            )


def _build_summary(
    rows: list[dict[str, Any]],
    sample_size: int,
    grab_root: Path,
    out_csv: Path,
) -> dict[str, Any]:
    subject_counts = Counter(str(row["subject_id"]) for row in rows)
    object_counts = Counter(str(row["object_name"]) for row in rows)
    action_counts = Counter((str(row["object_name"]), str(row["action_name"])) for row in rows)
    frame_counts = np.asarray([int(row["frame_count"]) for row in rows], dtype=np.float64)
    return {
        "dataset_name": "grab",
        "sample_size": int(sample_size),
        "grab_root": str(grab_root.resolve()),
        "manifest_csv": str(out_csv.resolve()),
        "num_subjects": int(len(subject_counts)),
        "num_objects": int(len(object_counts)),
        "num_object_actions": int(len(action_counts)),
        "subject_counts": dict(sorted(subject_counts.items())),
        "object_count_min": int(min(object_counts.values())) if object_counts else 0,
        "object_count_max": int(max(object_counts.values())) if object_counts else 0,
        "frame_count_mean": float(frame_counts.mean()) if frame_counts.size else 0.0,
        "frame_count_median": float(np.median(frame_counts)) if frame_counts.size else 0.0,
        "frame_count_p90": float(np.percentile(frame_counts, 90)) if frame_counts.size else 0.0,
        "frame_count_max": int(frame_counts.max()) if frame_counts.size else 0,
        "frame_count_min": int(frame_counts.min()) if frame_counts.size else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced exact-sequence manifest for a GRAB subset.")
    parser.add_argument("--grab-path", type=str, default=str(DEFAULT_GRAB_ROOT))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    args = parser.parse_args()

    grab_root = Path(args.grab_path).resolve()
    rows = _candidate_rows(grab_root)
    selected = _select_subset(rows, int(args.sample_size))
    out_csv = Path(args.out_csv).resolve()
    out_json = Path(args.out_json).resolve()
    _write_csv(out_csv, selected)
    summary = _build_summary(selected, int(args.sample_size), grab_root, out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_csv={out_csv}")
    print(f"wrote_json={out_json}")
    print(f"sample_size={summary['sample_size']}")
    print(f"num_subjects={summary['num_subjects']}")
    print(f"num_objects={summary['num_objects']}")
    print(f"num_object_actions={summary['num_object_actions']}")
    print(f"frame_count_mean={summary['frame_count_mean']:.3f}")
    print(f"frame_count_p90={summary['frame_count_p90']:.3f}")


if __name__ == "__main__":
    main()
