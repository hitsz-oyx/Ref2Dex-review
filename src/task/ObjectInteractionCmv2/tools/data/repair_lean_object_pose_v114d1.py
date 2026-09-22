#!/usr/bin/env python3
"""Recover rigid object poses in local lean caches with fixed point correspondence."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.task.ObjectInteractionCmv2.mixed_training import sha256_file
from src.task.ObjectInteractionCmv2.multi_domain import LEAN_CACHE_MANIFEST_SCHEMA, LEAN_GEOMETRY_SCHEMA


WORK_VERSION = "V1.14"
MAX_REPLAY_M = 2e-4
REPO_ROOT = Path(__file__).resolve().parents[5]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def recover_rigid_poses(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return poses mapping frame-0 points to each frame and max point replay error."""
    stream = np.asarray(points)
    if stream.ndim != 3 or stream.shape[1:] != (4096, 3) or len(stream) < 2:
        raise ValueError("object stream must be [T,4096,3] with T>=2")
    reference = np.asarray(stream[0], dtype=np.float64)
    reference_center = reference.mean(axis=0)
    centered_reference = reference - reference_center
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (len(stream), 4, 4)).copy()
    maximum = 0.0
    for frame in range(len(stream)):
        target = np.asarray(stream[frame], dtype=np.float64)
        target_center = target.mean(axis=0)
        covariance = centered_reference.T @ (target - target_center)
        left, _, right_t = np.linalg.svd(covariance)
        row_rotation = left @ right_t
        if np.linalg.det(row_rotation) < 0:
            left[:, -1] *= -1
            row_rotation = left @ right_t
        translation = target_center - reference_center @ row_rotation
        replay = reference @ row_rotation + translation
        error = float(np.linalg.norm(replay - target, axis=1).max())
        maximum = max(maximum, error)
        poses[frame, :3, :3] = row_rotation.T.astype(np.float32)
        poses[frame, :3, 3] = translation.astype(np.float32)
    return poses, maximum


def _validate_pose(points: np.ndarray, poses: np.ndarray) -> float:
    reference = np.asarray(points[0], dtype=np.float64)
    maximum = 0.0
    for frame, pose in enumerate(np.asarray(poses, dtype=np.float64)):
        rotation = pose[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
                or abs(float(np.linalg.det(rotation)) - 1.0) > 2e-5
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7)):
            raise ValueError(f"frame {frame}: recovered pose is not SE(3)")
        replay = reference @ rotation.T + pose[:3, 3]
        maximum = max(maximum, float(np.linalg.norm(
            replay - np.asarray(points[frame], dtype=np.float64), axis=1).max()))
    if maximum > MAX_REPLAY_M:
        raise ValueError(f"object pose replay {maximum} exceeds {MAX_REPLAY_M} m")
    return maximum


def _repair_sequence(path: Path) -> dict[str, Any]:
    geometry = path / "geometry"
    manifest_path = geometry / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != LEAN_GEOMETRY_SCHEMA:
        raise ValueError(f"{path}: not a V1.14c lean sequence")
    points = np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r")
    pose_path = geometry / "obj_pose_world.npy"
    backup_path = geometry / "obj_pose_world.v114d1_backup.npy"
    if manifest.get("object_pose_source") == "fixed_correspondence_kabsch_frame0_v1":
        maximum = _validate_pose(points, np.load(pose_path, mmap_mode="r"))
        return {"id": manifest["sequence_id"], "frames": len(points), "max_replay_m": maximum,
                "status": "ALREADY_REPAIRED"}
    poses, fit_error = recover_rigid_poses(points)
    maximum = _validate_pose(points, poses)
    if not backup_path.exists():
        shutil.copy2(pose_path, backup_path)
    temporary = geometry / "obj_pose_world.v114d1.tmp.npy"
    np.save(temporary, poses)
    os.replace(temporary, pose_path)
    manifest.update({
        "object_pose_source": "fixed_correspondence_kabsch_frame0_v1",
        "object_pose_reference_frame": 0,
        "object_pose_fit_max_replay_m": fit_error,
        "object_pose_validated_max_replay_m": maximum,
        "object_pose_repair_work_version": WORK_VERSION,
    })
    _write_json(manifest_path, manifest)
    return {"id": manifest["sequence_id"], "frames": len(points), "max_replay_m": maximum,
            "status": "REPAIRED"}


def run(args: argparse.Namespace) -> int:
    root = args.cache_root.resolve()
    index_path = root / "index.json"
    cache_manifest_path = root / "cache_manifest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if (cache_manifest.get("schema_name") != LEAN_CACHE_MANIFEST_SCHEMA
            or cache_manifest.get("work_version") != WORK_VERSION):
        raise ValueError("pose repair requires a completed V1.14 lean cache")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    if dirty:
        raise ValueError("formal pose repair requires a clean committed worktree")
    entries = [row for split in ("train", "val", "test")
               for row in index["sequences"][split]]
    report_path = root / f"run_manifest_{args.run_id}.json"
    report = {"schema_name": "ref2dex_data_run_manifest_v1", "task": "ObjectInteractionCmv2",
              "work_version": WORK_VERSION, "run_id": args.run_id, "run_status": "RUNNING",
              "operation": "v114d1_lean_object_pose_repair", "started_at": _now(),
              "git_commit": commit, "input_index_sha256": sha256_file(index_path),
              "expected_sequences": len(entries), "completed_sequences": 0,
              "completed_frames": 0, "failures": [], "max_replay_m": 0.0,
              "outputs": {"cache_root": str(root)}, "conclusion": "INCONCLUSIVE"}
    _write_json(report_path, report)
    for ordinal, row in enumerate(entries, 1):
        try:
            value = _repair_sequence(Path(row["path"]))
            report["completed_sequences"] += 1
            report["completed_frames"] += int(value["frames"])
            report["max_replay_m"] = max(float(report["max_replay_m"]),
                                         float(value["max_replay_m"]))
            _write_json(report_path, report)
            if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(entries):
                print(json.dumps({"index": ordinal, "total": len(entries), **value}), flush=True)
        except Exception as error:
            report["failures"].append({"id": row.get("id"),
                                       "error": f"{type(error).__name__}: {error}"})
            _write_json(report_path, report)
            traceback.print_exc()
    if report["failures"]:
        report.update(run_status="FAILED", finished_at=_now(), conclusion="INVALID_IMPLEMENTATION")
        _write_json(report_path, report)
        return 1
    cache_manifest["object_pose_repair"] = {
        "schema": "fixed_correspondence_kabsch_frame0_v1", "run_id": args.run_id,
        "git_commit": commit, "sequence_count": len(entries),
        "frame_count": int(report["completed_frames"]),
        "max_replay_m": float(report["max_replay_m"]), "threshold_m": MAX_REPLAY_M,
    }
    cache_manifest["validation"] = {"bad_count": 0, "bad_examples": []}
    _write_json(cache_manifest_path, cache_manifest)
    report.update(run_status="COMPLETED", finished_at=_now(), conclusion="N/A",
                  output_cache_manifest_sha256=sha256_file(cache_manifest_path))
    _write_json(report_path, report)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
