#!/usr/bin/env python3
"""Export two repaired Inspire sequences and build a viewer-ready pilot index."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[5]
EXPORT_MODULE = "src.task.ObjectInteractionCm.tools.data.retarget_stage4_bilateral_inspire"
POINTS_PER_SIDE = 1538


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_info() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    return commit, dirty


def _components(points: np.ndarray, radius_m: float = 0.02) -> tuple[int, list[int]]:
    tree = cKDTree(points)
    pairs = np.asarray(list(tree.query_pairs(float(radius_m))), dtype=np.int64)
    if not len(pairs):
        return len(points), [1] * min(6, len(points))
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(points), len(points)),
    ).tocsr()
    count, labels = connected_components(graph, directed=False)
    sizes = np.sort(np.bincount(labels))[::-1]
    return int(count), [int(value) for value in sizes[:6]]


def _active_frame(geometry: Path, preferred: int) -> int:
    candidate = np.load(geometry / "obj_candidate_mask_5cm.npy", mmap_mode="r")
    active = np.asarray(candidate, dtype=bool)
    if active.ndim == 2:
        active = active.any(axis=1)
    preferred = min(max(int(preferred), 0), len(active) - 1)
    if active[preferred]:
        return preferred
    indices = np.flatnonzero(active)
    return int(indices[len(indices) // 2]) if len(indices) else preferred


def _sequence_report(
    dataset: str,
    sequence: str,
    cache_root: Path,
    parent_root: Path,
    preferred_frame: int,
) -> dict[str, Any]:
    geometry = cache_root / sequence / "geometry"
    hand = np.load(geometry / "hand_points_world.npy", mmap_mode="r")
    normals = np.load(geometry / "hand_normals_world.npy", mmap_mode="r")
    source_frames = np.load(geometry / "source_frame_id.npy", mmap_mode="r")
    frame = _active_frame(geometry, preferred_frame)
    future = min(frame + 10, len(hand) - 1)
    sides: dict[str, Any] = {}
    max_extent_m = 0.0
    max_normal_error = 0.0
    for side_index, side in enumerate(("left", "right")):
        start = side_index * POINTS_PER_SIDE
        stop = start + POINTS_PER_SIDE
        points = np.asarray(hand[frame, start:stop], dtype=np.float64)
        side_normals = np.asarray(normals[frame, start:stop], dtype=np.float64)
        parent = np.load(parent_root / sequence / f"{side}.npz", mmap_mode="r")
        parent_points = np.asarray(parent["hand_points_world"][frame], dtype=np.float64)
        bbox_m = points.max(axis=0) - points.min(axis=0)
        max_extent_m = max(max_extent_m, float(bbox_m.max()))
        normal_error = float(np.max(np.abs(np.linalg.norm(side_normals, axis=1) - 1.0)))
        max_normal_error = max(max_normal_error, normal_error)
        self_nn_m = cKDTree(points).query(points, k=2)[0][:, 1]
        to_parent_m = cKDTree(parent_points).query(points, k=1)[0]
        components, largest = _components(points)
        sides[side] = {
            "bbox_mm": np.round(bbox_m * 1000.0, 3).tolist(),
            "centroid_mm": np.round(points.mean(axis=0) * 1000.0, 3).tolist(),
            "self_nearest_neighbor_mm_p50_p95_max": np.round(
                np.percentile(self_nn_m * 1000.0, [50, 95, 100]), 3
            ).tolist(),
            "components_at_20mm": components,
            "largest_components": largest,
            "nearest_parent_mano_mm_p50_p95_max": np.round(
                np.percentile(to_parent_m * 1000.0, [50, 95, 100]), 3
            ).tolist(),
            "normal_unit_max_abs_error": normal_error,
        }
    flow_mm = np.linalg.norm(
        np.asarray(hand[future], dtype=np.float64) - np.asarray(hand[frame], dtype=np.float64),
        axis=1,
    ) * 1000.0
    passed = bool(
        hand.shape[1:] == (2 * POINTS_PER_SIDE, 3)
        and normals.shape == hand.shape
        and max_extent_m < 0.30
        and max_normal_error < 1e-5
        and np.isfinite(hand[frame]).all()
        and np.isfinite(hand[future]).all()
    )
    return {
        "dataset": dataset,
        "sequence": sequence,
        "frame_count": int(len(hand)),
        "visualization_frame": frame,
        "future_frame": future,
        "source_frame": int(source_frames[frame]),
        "future_source_frame": int(source_frames[future]),
        "hand_flow_mm_p50_p95_max": np.round(
            np.percentile(flow_mm, [50, 95, 100]), 3
        ).tolist(),
        "max_side_extent_mm": max_extent_m * 1000.0,
        "max_normal_unit_error": max_normal_error,
        "engineering_gate": "PASS" if passed else "FAIL",
        "sides": sides,
    }


def _export(
    *,
    python: Path,
    source: str,
    sequence: str,
    input_root: Path,
    output_root: Path,
    work_version: str,
) -> tuple[list[str], str]:
    command = [
        str(python),
        "-u",
        "-m",
        EXPORT_MODULE,
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--source",
        source,
        "--sequence",
        sequence,
        "--work-version",
        work_version,
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"{source} export failed ({result.returncode}):\n{combined}")
    return command, combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grab-input-root",
        type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/grab_mano_30hz"),
    )
    parser.add_argument(
        "--arctic-input-root",
        type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/arctic_mano_30hz"),
    )
    parser.add_argument("--grab-sequence", default="s1/airplane_fly_1")
    parser.add_argument("--arctic-sequence", default="s01/box_use_01")
    parser.add_argument("--work-version", default="V1.4.6")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_at = _now()
    commit, dirty = _git_info()
    config = {
        "work_version": args.work_version,
        "grab_input_root": str(args.grab_input_root.resolve()),
        "arctic_input_root": str(args.arctic_input_root.resolve()),
        "grab_sequence": args.grab_sequence,
        "arctic_sequence": args.arctic_sequence,
        "surface_seed": 2024,
        "points_per_side": POINTS_PER_SIDE,
        "full_export": False,
    }
    _write_json(output / "config.json", config)
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "inspire_double_fk_repair_pilot",
        "run_id": output.name,
        "run_status": "RUNNING",
        "started_at": started_at,
        "work_version": args.work_version,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "config": str((output / "config.json").resolve()),
        "output": str(output),
    }
    _write_json(output / "run_manifest.json", manifest)
    logs: list[str] = []
    commands: list[list[str]] = []
    try:
        for dataset, sequence, input_root in (
            ("grab", args.grab_sequence, args.grab_input_root),
            ("arctic", args.arctic_sequence, args.arctic_input_root),
        ):
            command, log = _export(
                python=args.python,
                source=dataset,
                sequence=sequence,
                input_root=input_root.resolve(),
                output_root=output / "cache" / dataset,
                work_version=args.work_version,
            )
            commands.append(command)
            logs.append(f"$ {shlex.join(command)}\n{log}")
        (output / "export.log").write_text("\n".join(logs), encoding="utf-8")

        reports = [
            _sequence_report(
                "grab",
                args.grab_sequence,
                output / "cache" / "grab",
                args.grab_input_root.resolve(),
                103,
            ),
            _sequence_report(
                "arctic",
                args.arctic_sequence,
                output / "cache" / "arctic",
                args.arctic_input_root.resolve(),
                0,
            ),
        ]
        passed = all(item["engineering_gate"] == "PASS" for item in reports)
        report = {
            "run_id": output.name,
            "status": "ENGINEERING_PASS" if passed else "ENGINEERING_FAIL",
            "double_fk_fix": "canonical_zero_q_to_visual_local_before_per_frame_fk",
            "sequences": reports,
        }
        _write_json(output / "pilot_report.json", report)
        index = {
            "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
            "schema_version": "1.2.0",
            "created_at": _now(),
            "knn_k": 0,
            "object_pool_points": 4096,
            "model_object_points": 1024,
            "hand_points_per_stream": 3076,
            "max_union_hand_points": 3076,
            "source_probability": {"inspire_f1": 1.0},
            "pilot_only": True,
            "sequences": {
                "train": [
                    {
                        "source": "inspire_f1",
                        "id": f"grab/{args.grab_sequence}",
                        "path": str((output / "cache/grab" / args.grab_sequence).resolve()),
                        "dataset": "grab",
                    },
                    {
                        "source": "inspire_f1",
                        "id": f"arctic/{args.arctic_sequence}",
                        "path": str((output / "cache/arctic" / args.arctic_sequence).resolve()),
                        "dataset": "arctic",
                    },
                ],
                "val": [],
                "test": [],
            },
            "counts": {"train": 2, "val": 0, "test": 0},
        }
        _write_json(output / "index.json", index)
        if not passed:
            raise RuntimeError("pilot geometry engineering gate failed; see pilot_report.json")
        manifest.update(
            {
                "run_status": "COMPLETED",
                "completed_at": _now(),
                "commands": [shlex.join(command) for command in commands],
                "metadata_snapshot": [
                    str((output / "cache/grab" / args.grab_sequence / "geometry/manifest.json").resolve()),
                    str((output / "cache/arctic" / args.arctic_sequence / "geometry/manifest.json").resolve()),
                ],
                "report": str((output / "pilot_report.json").resolve()),
                "index": str((output / "index.json").resolve()),
                "conclusion": "SUPPORTED_ENGINEERING_PILOT_PENDING_USER_VISUAL_REVIEW",
            }
        )
        _write_json(output / "run_manifest.json", manifest)
        print(json.dumps(report, ensure_ascii=False))
    except Exception as exc:
        if logs and not (output / "export.log").is_file():
            (output / "export.log").write_text("\n".join(logs), encoding="utf-8")
        manifest.update(
            {
                "run_status": "FAILED",
                "completed_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
