from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an exact paired GRAB viewer index")
    parser.add_argument("--sequence", default="s1/airplane_fly_1")
    parser.add_argument(
        "--mano-root", type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/grab_mano_30hz"),
    )
    parser.add_argument(
        "--inspire-root", type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mano = (REPO_ROOT / args.mano_root).resolve() if not args.mano_root.is_absolute() else args.mano_root.resolve()
    inspire = (
        (REPO_ROOT / args.inspire_root).resolve()
        if not args.inspire_root.is_absolute() else args.inspire_root.resolve()
    )
    mano_sequence = mano / args.sequence
    inspire_sequence = inspire / args.sequence
    output = (REPO_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    with np.load(mano_sequence / "shared.npz", allow_pickle=False) as shared:
        mano_frames = np.asarray(shared["raw_frame_id"], dtype=np.int32)
        mano_object = np.asarray(shared["obj_points_world"], dtype=np.float32)
        mano_pose = np.asarray(shared["obj_pose_world"], dtype=np.float32)
    geometry = inspire_sequence / "geometry"
    inspire_frames = np.load(geometry / "source_frame_id.npy", mmap_mode="r")
    inspire_object = np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r")
    inspire_pose = np.load(geometry / "obj_pose_world.npy", mmap_mode="r")

    checks = {
        "frame_count_equal": len(mano_frames) == len(inspire_frames),
        "source_frame_id_exact": bool(np.array_equal(mano_frames, inspire_frames)),
        "object_points_exact": bool(np.array_equal(mano_object, inspire_object)),
        "object_pose_exact": bool(np.array_equal(mano_pose, inspire_pose)),
    }
    if not all(checks.values()):
        _write_json(output / "pair_validation.json", {"sequence": args.sequence, "checks": checks})
        raise ValueError(f"MANO/Inspire pair validation failed: {checks}")

    index = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
        "schema_version": "viewer-only-paired-v1",
        "model_object_points": 1024,
        "viewer_only": True,
        "sequences": {
            "train": [
                {
                    "id": f"grab/{args.sequence}",
                    "path": str(mano_sequence),
                    "source": "mano",
                    "variant": "mano_bilateral_raw",
                    "dataset": "grab",
                },
                {
                    "id": f"grab/{args.sequence}",
                    "path": str(inspire_sequence),
                    "source": "inspire_f1",
                    "variant": "inspire_geometric",
                    "dataset": "grab",
                },
            ],
            "val": [],
            "test": [],
        },
    }
    report = {
        "sequence": args.sequence,
        "frames": int(len(mano_frames)),
        "first_source_frame": int(mano_frames[0]),
        "last_source_frame": int(mano_frames[-1]),
        "mano_path": str(mano_sequence),
        "inspire_path": str(inspire_sequence),
        "checks": checks,
        "conclusion": "SUPPORTED",
    }
    _write_json(output / "index.json", index)
    _write_json(output / "pair_validation.json", report)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    _write_json(
        output / "run_manifest.json",
        {
            "schema_name": "ref2dex_run_manifest_v1",
            "task": "ObjectInteractionCm",
            "operation": "paired_grab_viewer_index_prepare",
            "run_id": output.name,
            "run_status": "COMPLETED",
            "started_at": _now(),
            "completed_at": _now(),
            "modification_version": "V1.4.7",
            "base_commit": commit,
            "worktree_dirty": dirty,
            "command": shlex.join([sys.executable, *sys.argv]),
            "inputs": {"mano": str(mano_sequence), "inspire": str(inspire_sequence)},
            "outputs": {"index": "index.json", "validation": "pair_validation.json"},
            "conclusion": "SUPPORTED",
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
