"""Read-only full-frame label-to-configured-FK gate before coupled training."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from src.base import load_config
from ...kinematics import InspireKinematics, NATIVE_TO_URDF
from ...pointflow import DifferentiableInspireSurface


def reference(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tolerance-mm", type=float, default=0.002)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.tolerance_mm <= 0:
        parser.error("batch-size and tolerance-mm must be positive")
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a directory basename")
    cfg = load_config(args.config)
    index_path = Path(cfg.data.view_root) / "index.json"
    index = json.loads(index_path.read_text())
    if index["work_version"] != cfg.work_version:
        raise ValueError("Runner requires matching training and view work_version")
    output = Path(__file__).parent / "output" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sampler = str(getattr(cfg.model, "surface_sampling", "legacy_urdf"))
    surface = DifferentiableInspireSurface(
        cfg.model.surface_urdf, sample_count=cfg.meta.num_hand_points,
        surface_sampling=sampler, surface_seed=2024,
    ).to(args.device).eval()
    kin = InspireKinematics(cfg.model.surface_urdf)
    lower = np.array([j.lower for j in kin.joints if j.q_index >= 0])[NATIVE_TO_URDF][6:]
    upper = np.array([j.upper for j in kin.joints if j.q_index >= 0])[NATIVE_TO_URDF][6:]
    write_json(output / "config.json", {"training": cfg.to_dict(), "verification": vars(args)})
    metadata = {
        "view_index": reference(index_path), "splits": ["train", "val"],
        "coordinate_frame": "world", "units": "m", "surface_sampling": sampler,
        "point_count": int(cfg.meta.num_hand_points), "surface_seed": 2024,
        "state": "native18", "control": "finger6_and_wrist", "checkpoint_loaded": False,
    }
    write_json(output / "metadata.json", metadata)
    manifest = {
        "task": "CmDecoderv2", "run_id": args.run_id,
        "work_version": cfg.work_version, "operation_category": ["diagnostic"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "command": [sys.executable, "-m", "src.task.CmDecoderv2.research.dexplore_contract_audit.verify_coupled_view", *sys.argv[1:]],
        "config_snapshot": "config.json", "metadata_snapshot": "metadata.json",
        "seed": 2024, "initial_checkpoint": None, "output_root": str(output.resolve()),
        "inputs": [reference(index_path), reference(args.config), reference(cfg.model.surface_urdf)],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "run_manifest.json", manifest)
    rows, all_inputs = [], []
    with torch.inference_mode(), (output / "metrics.jsonl").open("w") as metrics:
        for split in ("train", "val"):
            for entry in index["sequences"][split]:
                paths = [Path(entry["q_native"]), Path(entry["wrist_pose_world"]),
                         Path(entry["geometry_root"]) / "knn_hand_points_world.npy"]
                before = [reference(path) for path in paths]
                q, wrist, expected = [np.load(path, mmap_mode="r") for path in paths]
                if q.shape != (len(q), 18) or wrist.shape != (len(q), 4, 4) or expected.shape != (len(q), cfg.meta.num_hand_points, 3):
                    raise ValueError(f"Invalid shapes: {entry['id']}")
                if not np.isfinite(q).all() or not np.isfinite(wrist).all():
                    raise ValueError(f"Nonfinite state: {entry['id']}")
                # Explicit native mimic equation is independent of expand_finger_q.
                mimic = q[:, [7, 9, 11, 13, 16, 17]] - q[:, [6, 8, 10, 12, 15, 15]] * [1.05, 1.05, 1.05, 1.05, .6, .8]
                residual = float(np.abs(mimic).max())
                violation = float(max(0., np.maximum(lower - q[:, 6:], q[:, 6:] - upper).max()))
                epe_sum, max_error, point_count = 0., 0., 0
                for offset in range(0, len(q), args.batch_size):
                    block = slice(offset, offset + args.batch_size)
                    fingers = torch.tensor(q[block][:, [6, 8, 10, 12, 14, 15]], device=args.device)
                    poses = torch.tensor(np.asarray(wrist[block]), device=args.device)
                    actual = surface(fingers, poses).cpu().numpy()
                    errors = np.linalg.norm(actual - expected[block], axis=-1) * 1000.
                    if not np.isfinite(errors).all():
                        raise ValueError(f"Nonfinite geometry: {entry['id']}")
                    epe_sum += float(errors.sum(dtype=np.float64))
                    max_error = max(max_error, float(errors.max()))
                    point_count += errors.size
                if before != [reference(path) for path in paths]:
                    raise RuntimeError(f"Inputs changed during verification: {entry['id']}")
                all_inputs.extend(before)
                row = {"id": entry["id"], "split": split, "frames": len(q),
                       "point_count": point_count, "mean_epe_mm": epe_sum / point_count,
                       "max_epe_mm": max_error, "max_mimic_residual_rad": residual,
                       "max_limit_violation_rad": violation,
                       "passed": max_error <= args.tolerance_mm and residual <= 1e-6 and violation <= 1e-6}
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                metrics.flush()
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f"verified {len(rows)} sequences", flush=True)
    total_points = sum(row["point_count"] for row in rows)
    summary = {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "sequences": len(rows), "frames": sum(row["frames"] for row in rows),
        "point_count": total_points,
        "mean_epe_mm": sum(row["mean_epe_mm"] * row["point_count"] for row in rows) / total_points,
        "max_epe_mm": max(row["max_epe_mm"] for row in rows),
        "max_mimic_residual_rad": max(row["max_mimic_residual_rad"] for row in rows),
        "max_limit_violation_rad": max(row["max_limit_violation_rad"] for row in rows),
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(output / "input_files.json", all_inputs)
    write_json(output / "verification.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit("Configured label-to-FK correspondence gate failed")


if __name__ == "__main__":
    main()
