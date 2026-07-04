#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REF2DEX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_CSV = REF2DEX_ROOT / "tmp" / "manifests" / "grab_subset100_hand_jobs.csv"
DEFAULT_OUT_JSON = REF2DEX_ROOT / "tmp" / "manifests" / "grab_subset100_hand_jobs_summary.json"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_index",
        "dataset_name",
        "seq_id",
        "subject_id",
        "seq_name",
        "object_name",
        "side",
        "num_total_frames",
        "num_valid_frames",
        "valid_ratio",
        "npz_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(
                {
                    "job_index": idx,
                    "dataset_name": row["dataset_name"],
                    "seq_id": row["seq_id"],
                    "subject_id": row["subject_id"],
                    "seq_name": row["seq_name"],
                    "object_name": row["object_name"],
                    "side": row["side"],
                    "num_total_frames": row["num_total_frames"],
                    "num_valid_frames": row["num_valid_frames"],
                    "valid_ratio": f"{float(row['valid_ratio']):.8f}",
                    "npz_path": row["npz_path"],
                }
            )


def _build_rows(processed_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for npz_path in sorted(processed_root.glob("*/*.npz")):
        with np.load(str(npz_path), allow_pickle=True) as data:
            dataset_name = str(data["dataset_name"].item() if "dataset_name" in data.files else "unknown")
            seq_id = str(data["seq_id"].item() if "seq_id" in data.files else f"{npz_path.parent.name}/{npz_path.stem}")
            subject_id, seq_name = seq_id.split("/", 1)
            object_name = str(data["object_name"].item() if "object_name" in data.files else seq_name.split("_")[0])
            for side in ("right", "left"):
                key = f"{side}_hand_valid"
                if key not in data.files:
                    continue
                valid = np.asarray(data[key], dtype=bool)
                num_total = int(valid.shape[0])
                num_valid = int(valid.sum())
                if num_valid <= 0:
                    continue
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "seq_id": seq_id,
                        "subject_id": subject_id,
                        "seq_name": seq_name,
                        "object_name": object_name,
                        "side": side,
                        "num_total_frames": num_total,
                        "num_valid_frames": num_valid,
                        "valid_ratio": float(num_valid / max(num_total, 1)),
                        "npz_path": str(npz_path.resolve()),
                    }
                )
    rows.sort(key=lambda row: (row["subject_id"], row["seq_name"], row["side"]))
    return rows


def _build_summary(processed_root: Path, rows: list[dict[str, Any]], out_csv: Path) -> dict[str, Any]:
    subject_counts = Counter(str(row["subject_id"]) for row in rows)
    side_counts = Counter(str(row["side"]) for row in rows)
    object_counts = Counter(str(row["object_name"]) for row in rows)
    valid_frames = np.asarray([int(row["num_valid_frames"]) for row in rows], dtype=np.float64)
    return {
        "processed_root": str(processed_root.resolve()),
        "manifest_csv": str(out_csv.resolve()),
        "num_jobs": int(len(rows)),
        "num_subjects": int(len(subject_counts)),
        "num_objects": int(len(object_counts)),
        "side_counts": dict(sorted(side_counts.items())),
        "subject_counts": dict(sorted(subject_counts.items())),
        "valid_frames_mean": float(valid_frames.mean()) if valid_frames.size else 0.0,
        "valid_frames_median": float(np.median(valid_frames)) if valid_frames.size else 0.0,
        "valid_frames_p90": float(np.percentile(valid_frames, 90)) if valid_frames.size else 0.0,
        "valid_frames_max": int(valid_frames.max()) if valid_frames.size else 0,
        "valid_frames_min": int(valid_frames.min()) if valid_frames.size else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-hand Stage 2 job manifest from processed Ref2Dex GRAB data.")
    parser.add_argument("--processed-root", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    args = parser.parse_args()

    processed_root = Path(args.processed_root).resolve()
    rows = _build_rows(processed_root)
    out_csv = Path(args.out_csv).resolve()
    out_json = Path(args.out_json).resolve()
    _write_csv(out_csv, rows)
    summary = _build_summary(processed_root, rows, out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_csv={out_csv}")
    print(f"wrote_json={out_json}")
    print(f"num_jobs={summary['num_jobs']}")
    print(f"side_counts={json.dumps(summary['side_counts'], sort_keys=True)}")
    print(f"valid_frames_mean={summary['valid_frames_mean']:.3f}")


if __name__ == "__main__":
    main()
