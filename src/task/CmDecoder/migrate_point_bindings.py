"""Add static point-to-link bindings to an existing layered CmDecoder cache.

This migration deliberately reuses the existing world geometry and task points;
it does not rerun mesh sampling or frame FK for every sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmDecoder.q_optimizer import DifferentiableInspireHand


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("dataset.py"), Path(__file__).with_name("q_optimizer.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _binding_for_episode(model: DifferentiableInspireHand, geometry: Path) -> tuple[np.ndarray, np.ndarray]:
    q_full = np.load(geometry / "q_full.npy", mmap_mode="r", allow_pickle=False)
    hand_world = np.load(geometry / "hand_points_world.npy", mmap_mode="r", allow_pickle=False)
    wrist = np.load(geometry / "wrist_pose_world.npy", mmap_mode="r", allow_pickle=False)
    current = (np.asarray(hand_world[0]) - np.asarray(wrist[0, :3, 3])) @ np.asarray(wrist[0, :3, :3])
    q = torch.from_numpy(np.asarray(q_full[0, 6:], dtype=np.float32)[None].copy())
    points = torch.from_numpy(np.asarray(current, dtype=np.float32)[None].copy())
    groups = model.bind_points(q, points)
    link_to_id = {name: index for index, name in enumerate(model.link_order)}
    link_index = np.empty(model.num_hand_points, dtype=np.int16)
    local_points = np.empty((model.num_hand_points, 3), dtype=np.float32)
    for link, indices, local in groups:
        index = link_to_id[link]
        idx = indices.detach().cpu().numpy()
        link_index[idx] = index
        local_points[idx] = local[0].detach().cpu().numpy()
    reconstructed = model.points(
        q,
        cached_link_index=torch.from_numpy(link_index),
        cached_local_points=torch.from_numpy(local_points),
    )
    error_m = torch.linalg.vector_norm(reconstructed - points, dim=-1).max().item()
    if error_m > 1e-5:
        raise RuntimeError(f"Binding validation failed for {geometry}: max error={error_m} m")
    return link_index, local_points


def migrate(args: argparse.Namespace) -> None:
    root = args.cache_root.resolve()
    manifest_path = root / "v4" / args.v4_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = DifferentiableInspireHand(args.robot_urdf, num_hand_points=args.num_hand_points, sample_seed=args.seed)
    implementation_hash = _implementation_hash()
    for ordinal, (episode, relative) in enumerate(manifest["cache_dirs"].items(), start=1):
        geometry = root / relative / "geometry"
        link_file = geometry / "hand_point_link_index.npy"
        local_file = geometry / "hand_points_local.npy"
        if not link_file.is_file() or not local_file.is_file():
            link_index, local_points = _binding_for_episode(model, geometry)
            np.save(link_file, link_index, allow_pickle=False)
            np.save(local_file, local_points, allow_pickle=False)
        geometry_manifest_path = geometry / "manifest.json"
        geometry_manifest = json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
        geometry_manifest["implementation_sha256"] = implementation_hash
        geometry_manifest["hand_binding_semantics"] = "link_index_and_link_local_point_v1"
        geometry_manifest["hand_binding_link_order"] = model.link_order
        geometry_manifest["fields"] = sorted(set(geometry_manifest.get("fields", [])) | {"hand_point_link_index", "hand_points_local"})
        geometry_manifest_path.write_text(json.dumps(geometry_manifest, indent=2), encoding="utf-8")
        if ordinal % 20 == 0 or ordinal == 1:
            print(f"bindings {ordinal}/{len(manifest['cache_dirs'])}: {episode}", flush=True)

    horizon_root = args.horizon_root.resolve() if args.horizon_root is not None else root.parent / "hrdexdb_inspire_f1_3hz"
    horizon_manifest_path = horizon_root / "v2" / args.v2_manifest
    if horizon_manifest_path.is_file():
        horizon_manifest = json.loads(horizon_manifest_path.read_text(encoding="utf-8"))
        for episode, relative in horizon_manifest["cache_dirs"].items():
            source_relative = manifest["cache_dirs"][episode]
            source_geometry = root / source_relative / "geometry"
            task = horizon_root / relative / "task"
            for name in ("hand_point_link_index", "hand_points_local"):
                source = source_geometry / f"{name}.npy"
                target = task / f"{name}.npy"
                if not target.is_file():
                    target.write_bytes(source.read_bytes())
            task_manifest_path = task / "manifest.json"
            task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
            task_manifest["hand_binding_semantics"] = "link_index_and_link_local_point_v1"
            task_manifest["hand_binding_link_order"] = model.link_order
            task_manifest["fields"] = sorted(set(task_manifest.get("fields", [])) | {"hand_point_link_index", "hand_points_local"})
            task_manifest_path.write_text(json.dumps(task_manifest, indent=2), encoding="utf-8")
        horizon_manifest["hand_binding_semantics"] = "link_index_and_link_local_point_v1"
        horizon_manifest["hand_binding_link_order"] = model.link_order
        horizon_manifest_path.write_text(json.dumps(horizon_manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/processed_data/cm_decoder/hrdexdb_inspire_f1"))
    parser.add_argument("--v4-manifest", default="selection_576_seed42.json")
    parser.add_argument("--v2-manifest", default="selection_576_object_disjoint_seed42.json")
    parser.add_argument("--horizon-root", type=Path, default=None)
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "dataset"
        / "HRDexDB"
        / "assets"
        / "robots"
        / "xarm_inspire_f1_right.urdf",
    )
    parser.add_argument("--num-hand-points", type=int, default=1538)
    parser.add_argument("--seed", type=int, default=42)
    migrate(parser.parse_args())


if __name__ == "__main__":
    main()
