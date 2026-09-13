"""Build a paired MANO-source / actual-Inspire decoder view.

The target stream is the original Dexplore actual Inspire trajectory.  The Cm
source stream is a MANO parent surface sampled at the same raw frame ids and
placed in the actual object pose.  Source and target geometry are kept as
separate fields so this view cannot silently train on the geometric reference
as its target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points

from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_v1_3_cache import (
    _build_mano_pool,
    _sample_correspondence,
    _sample_mano_surface,
)

ROOT = Path(__file__).resolve().parents[5]
KNN_K = 32
OBJECT_POOL_POINTS = 4096
HAND_POINTS = 10135
RADIUS_M = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_batch(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points - pose[:, None, :3, 3]) @ pose[:, :3, :3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-index", type=Path, default=ROOT / "data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/index.json")
    parser.add_argument("--object-index", type=Path, default=ROOT / "data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json")
    parser.add_argument("--parent-root", type=Path, default=ROOT / "data/processed_data/cm_object_v2_surface512_object_pose_20260830")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mano-model-root", type=Path, default=ROOT / "dataset/arctic/data/body_models/mano")
    parser.add_argument("--surface-seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--geometry-batch-size", type=int, default=8)
    parser.add_argument("--knn-batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-missing-mano", action="store_true", help="Explicitly exclude sequences without complete MANO provenance")
    return parser.parse_args()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), target)


def _build_source_geometry(actual_geometry: Path, parent: Path, output: Path, correspondence: dict[str, np.ndarray], device: torch.device, args: argparse.Namespace) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    object_points = np.load(actual_geometry / "obj_points_pool_world.npy", mmap_mode="r")
    object_normals = np.load(actual_geometry / "obj_normals_pool_world.npy", mmap_mode="r")
    object_pose = np.load(actual_geometry / "obj_pose_world.npy", mmap_mode="r")
    raw = np.load(actual_geometry / "source_frame_id.npy", mmap_mode="r")
    if object_points.shape[1:] != (OBJECT_POOL_POINTS, 3) or object_normals.shape != object_points.shape:
        raise ValueError(f"Unexpected actual object geometry: {actual_geometry}")
    parent_raw = np.asarray(np.load(parent / "shared/raw_frame_id.npy"), dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(parent_raw)}
    if any(int(value) not in lookup for value in raw):
        raise KeyError(f"MANO parent lacks raw frame ids for {actual_geometry}")

    frame_count = len(raw)
    points_path = output / "knn_hand_points_world.npy"
    normals_path = output / "knn_hand_normals_world.npy"
    indices_path = output / "obj_knn_indices.npy"
    mask_path = output / "obj_candidate_mask_2cm.npy"
    points = np.lib.format.open_memmap(points_path, mode="w+", dtype=np.float32, shape=(frame_count, HAND_POINTS, 3))
    normals = np.lib.format.open_memmap(normals_path, mode="w+", dtype=np.float32, shape=points.shape)
    _sample_mano_surface(parent, np.asarray(raw, dtype=np.int64), np.asarray(object_pose, dtype=np.float32), correspondence, points, normals, batch_size=args.geometry_batch_size)
    indices = np.lib.format.open_memmap(indices_path, mode="w+", dtype=np.uint16, shape=(frame_count, OBJECT_POOL_POINTS, KNN_K))
    active = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=(frame_count, OBJECT_POOL_POINTS))
    for start in range(0, frame_count, max(1, args.knn_batch_size)):
        stop = min(frame_count, start + max(1, args.knn_batch_size))
        pose = np.asarray(object_pose[start:stop], dtype=np.float32)
        obj_local = _frame_batch(np.asarray(object_points[start:stop]), pose)
        hand_local = _frame_batch(np.asarray(points[start:stop]), pose)
        with torch.inference_mode():
            result = knn_points(torch.as_tensor(obj_local, device=device), torch.as_tensor(hand_local, device=device), K=KNN_K, return_sorted=True)
            indices[start:stop] = result.idx.cpu().numpy().astype(np.uint16)
            active[start:stop] = (result.dists.clamp_min(0.0).sqrt() <= RADIUS_M).any(dim=-1).cpu().numpy()
    for array in (points, normals, indices, active):
        array.flush()
    del points, normals, indices, active
    manifest = {
        "schema_name": "ref2dex_cm_decoder_v2_paired_cm_source_v1",
        "sequence_id": actual_geometry.parent.parent.name,
        "frame_count": frame_count,
        "source_type": "mano_parent_surface_at_actual_object_pose",
        "coordinate_frame": "object_pose_t",
        "raw_frame_id_source": str((actual_geometry / "source_frame_id.npy").resolve()),
        "parent_root": str(parent.resolve()),
        "hand_points": HAND_POINTS,
        "knn_k": KNN_K,
        "interaction_radius_m": RADIUS_M,
        "surface_seed": int(args.surface_seed),
        "surface_correspondence_barycentric_sha256": hashlib.sha256(np.asarray(correspondence["barycentric"], dtype=np.float32).tobytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = _parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    actual_index = json.loads(args.actual_index.read_text(encoding="utf-8"))
    object_index = json.loads(args.object_index.read_text(encoding="utf-8"))
    object_entries = {item["id"]: item for split in ("train", "val") for item in object_index["sequences"][split] if item.get("variant") == "inspire_rl"}
    entries = [item for split in ("train", "val") for item in actual_index["sequences"][split]]
    if args.smoke:
        entries = [actual_index["sequences"]["train"][0], actual_index["sequences"]["val"][0]]
    pool, _ = _build_mano_pool(args.mano_model_root, 0.20)
    correspondence = _sample_correspondence(pool, HAND_POINTS, args.surface_seed)
    device = torch.device(args.device)
    output_sequences = {"train": [], "val": [], "test": []}
    excluded = []
    args.output_root.mkdir(parents=True)
    for item in entries:
        sequence_id = str(item["id"])
        if sequence_id not in object_entries:
            raise KeyError(f"Actual sequence is absent from object index: {sequence_id}")
        actual_geometry = Path(item["geometry_root"])
        parent = args.parent_root / str(item["subject_id"]) / f"{item['object_name']}_{item['action_name']}"
        required_parent = [
            parent / "shared/raw_frame_id.npy",
            parent / "shared/obj_pose_world.npy",
            parent / "right/hand_mesh_vertices_world.npy",
            parent / "right/hand_mesh_faces.npy",
        ]
        if not all(path.is_file() for path in required_parent):
            if not args.skip_missing_mano:
                raise FileNotFoundError(f"Missing MANO provenance for {sequence_id}: {[str(path) for path in required_parent if not path.is_file()]}")
            excluded.append({"id": sequence_id, "split": item["split"], "reason": "missing_mano_provenance", "missing": [str(path) for path in required_parent if not path.is_file()]})
            print(f"[skip] {item['split']} {sequence_id}: missing MANO provenance", flush=True)
            continue
        source_rel = Path("sequences") / item["split"] / item["id"].replace("/", "_") / "cm_source"
        source_output = args.output_root / source_rel
        manifest = _build_source_geometry(actual_geometry, parent, source_output, correspondence, device, args)
        value = dict(item)
        value["variant"] = "inspire_rl"
        value["paired_source"] = "mano"
        value["geometry_root"] = str(actual_geometry.resolve())
        value["cm_geometry_root"] = str(source_output.resolve())
        value["cm_hand_points"] = str((source_output / "knn_hand_points_world.npy").resolve())
        value["cm_hand_normals"] = str((source_output / "knn_hand_normals_world.npy").resolve())
        value["cm_active_mask"] = str((source_output / "obj_candidate_mask_2cm.npy").resolve())
        value["q_native"] = str(Path(item["q_native"]).resolve())
        value["wrist_pose_world"] = str(Path(item["wrist_pose_world"]).resolve())
        value["paired_source_manifest"] = str((source_output / "manifest.json").resolve())
        output_sequences[item["split"]].append(value)
        print(f"[done] {item['split']} {sequence_id} frames={manifest['frame_count']}", flush=True)
    index = {
        "schema_name": "ref2dex_cm_decoder_v2_dexplore_paired_view_v1",
        "schema_version": "1.0.0",
        "modification_version": "V1.1.14",
        "source_actual_index": str(args.actual_index.resolve()),
        "source_object_index": str(args.object_index.resolve()),
        "split_contract": {"train": "mano_source_actual_inspire", "val": "mano_source_actual_inspire", "test": "not_built"},
        "cm_source_contract": {"source": "MANO parent surface", "target": "Dexplore actual Inspire", "coordinate_frame": "object_pose_t", "hand_points": HAND_POINTS, "knn_k": KNN_K, "interaction_radius_m": RADIUS_M},
        "sequences": output_sequences,
        "counts": {key: len(value) for key, value in output_sequences.items()},
        "excluded_sequences": excluded,
        "training_eligible": not args.smoke,
    }
    (args.output_root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_name": "ref2dex.run_manifest_v1", "task": "CmDecoderv2", "operation": "prepare_mano_actual_finetune_view",
        "run_id": args.output_root.name, "run_status": "COMPLETED", "modification_version": "V1.1.14",
        "operation_category": ["data", "operation"], "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "command": [sys.executable, *sys.argv], "output_root": str(args.output_root.resolve()),
        "counts": index["counts"], "excluded_sequences": excluded, "training_eligible": not args.smoke, "conclusion": "SUPPORTED" if not args.smoke else "INCONCLUSIVE",
        "source_actual_index": str(args.actual_index.resolve()), "source_object_index": str(args.object_index.resolve()),
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "counts": index["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
