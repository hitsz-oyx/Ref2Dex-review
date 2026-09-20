"""Prepare an ignored source index for the coupled geometric cache builder.

The existing V1.3 cache builder owns object/KNN schema validation.  This
adapter only supplies parent object geometry and regenerates the 1538-point
Inspire surface from the fitted six-drive reference.  It never modifies the
parent cache or fitted tensors.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.task.ObjectInteractionCm.research.hand_region_sampling.build_trajectory_preview import _sample_correspondence
from src.task.ObjectInteractionCm.research.hand_region_sampling.run import _build_inspire_pool
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel, _pose_from_native
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_v1_3_cache import _sample_inspire_surface

ROOT = Path(__file__).resolve().parents[5]
SOURCE_SCHEMA = "ref2dex_object_interaction_cm_index_v1_1"


def _safe_id(value: str) -> str:
    return value.replace("/", "_")


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    os.symlink(source.resolve(), target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, default=ROOT / "data/processed_data/cm_object_v2_surface512_object_pose_20260830")
    parser.add_argument("--source-index", type=Path, default=ROOT / "data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")
    parser.add_argument("--surface-seed", type=int, default=2024)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    source_payload = json.loads(args.source_index.read_text(encoding="utf-8"))
    if source_payload.get("schema_name") != SOURCE_SCHEMA:
        raise ValueError(f"Expected {SOURCE_SCHEMA}, got {source_payload.get('schema_name')!r}")
    entries = []
    for split in ("train", "val"):
        entries.extend(dict(item) for item in source_payload["sequences"][split] if item.get("variant") == "inspire_rl")
    if args.smoke:
        entries = entries[:1]
    helper = InspireUrdfModel(args.urdf)
    pool, _ = _build_inspire_pool(args.urdf, 0.20)
    correspondence = _sample_correspondence(pool, 1538, args.surface_seed)
    args.output_root.mkdir(parents=True)
    rewritten = {"train": [], "val": [], "test": []}
    for item in entries:
        name = str(item["id"]).replace("/", "_")
        fit_path = args.fit_root / name / "interaction_hand_inspire.pt"
        if not fit_path.is_file():
            raise FileNotFoundError(fit_path)
        fitted = torch.load(fit_path, map_location="cpu", weights_only=True).numpy().astype(np.float32, copy=False)
        parent = args.parent_root / str(item["subject_id"]) / f"{item['object_name']}_{item['action_name']}"
        shared = parent / "shared"
        parent_paths = {
            "obj_points": shared / "obj_points_world.npy",
            "obj_normals": shared / "obj_normals_world.npy",
            "obj_pose": shared / "obj_pose_world.npy",
            "source_frame": shared / "raw_frame_id.npy",
        }
        if not all(path.is_file() for path in parent_paths.values()):
            raise FileNotFoundError(f"Missing parent geometry for {item['id']}: {parent}")
        frame_count = len(fitted)
        for path in parent_paths.values():
            if len(np.load(path, mmap_mode="r")) != frame_count:
                raise ValueError(f"Frame mismatch for {item['id']}: {path}")
        relative = Path("sequences") / str(item["split"]) / "inspire_rl" / name
        geometry = args.output_root / relative / "geometry"
        geometry.mkdir(parents=True, exist_ok=False)
        old_pose = np.asarray(np.load(parent_paths["obj_pose"], mmap_mode="r"), dtype=np.float32)
        old_points = np.asarray(np.load(parent_paths["obj_points"], mmap_mode="r"), dtype=np.float32)
        old_normals = np.asarray(np.load(parent_paths["obj_normals"], mmap_mode="r"), dtype=np.float32)
        target_pose = np.stack([_pose_from_native(row[198:201], row[201:205]) for row in fitted]).astype(np.float32)
        local_points = (old_points - old_pose[:, None, :3, 3]) @ old_pose[:, :3, :3]
        local_normals = old_normals @ old_pose[:, :3, :3]
        np.save(geometry / "obj_points_pool_world.npy", local_points @ np.transpose(target_pose[:, :3, :3], (0, 2, 1)) + target_pose[:, None, :3, 3])
        np.save(geometry / "obj_normals_pool_world.npy", local_normals @ np.transpose(target_pose[:, :3, :3], (0, 2, 1)))
        np.save(geometry / "obj_pose_world.npy", target_pose)
        raw = np.asarray(np.load(parent_paths["source_frame"], mmap_mode="r"), dtype=np.int64)
        np.save(geometry / "source_frame_id.npy", raw.astype(np.int32))
        np.save(geometry / "frame_time.npy", (raw / 120.0).astype(np.float32))
        hand_points = np.empty((frame_count, 1538, 3), dtype=np.float32)
        hand_normals = np.empty_like(hand_points)
        _sample_inspire_surface(helper, fitted[:, 373:391], correspondence, hand_points, hand_normals, batch_size=32)
        np.save(geometry / "hand_points_world.npy", hand_points)
        np.save(geometry / "hand_normals_world.npy", hand_normals)
        manifest = {
            "schema_name": "ref2dex_coupled_geometric_source_v1", "sequence_id": item["id"],
            "split": item["split"], "variant": "inspire_rl", "frame_count": frame_count,
            "coordinate_frame": "dexplore_native_object_pose_world", "object_pose_source": str(parent.resolve()),
            "source_type": "dexplore_rl_native_q_coupled_geometric_reference",
            "source_fps": 120.0, "effective_fps": 30.0, "ds_rate": 4,
            "rl_q": {"tensor": str(fit_path.resolve()), "native_slice": [373, 391], "parameterization": "six_drive_coupled_reference"},
            "surface_sampling": {"point_count": 1538, "seed": args.surface_seed, "asset": "inspire_hand_right_urdf_visuals"},
            "training_eligible": not args.smoke,
        }
        (geometry / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rewritten[item["split"]].append({**item, "path": str(relative), "frame_count": frame_count, "variant": "inspire_rl"})
    index = {
        "schema_name": SOURCE_SCHEMA, "schema_version": "1.1.0-coupled-geometric-v1", "experiment_schema": "ref2dex_coupled_geometric_source_v1",
        "source_index": str(args.source_index.resolve()), "fit_root": str(args.fit_root.resolve()), "parent_root": str(args.parent_root.resolve()),
        "coordinate_frame": "object_pose_t", "decoder_hand_points_per_stream": 1538,
        "split_policy": "existing_train_val_inspire_rl_only", "sequences": rewritten,
        "counts": {split: len(values) for split, values in rewritten.items()},
        "training_eligible": not args.smoke,
    }
    (args.output_root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1", "task": "CmDecoderv2",
        "operation": "prepare_coupled_geometric_source", "run_id": args.output_root.name,
        "run_status": "COMPLETED", "work_version": "V1.1.13",
        "operation_category": ["data", "operation"], "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_commit": commit,
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "command": [sys.executable, *sys.argv], "fit_root": str(args.fit_root.resolve()),
        "parent_root": str(args.parent_root.resolve()), "source_index": str(args.source_index.resolve()),
        "output_root": str(args.output_root.resolve()), "counts": index["counts"],
        "training_eligible": not args.smoke, "conclusion": "SUPPORTED",
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "counts": index["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
