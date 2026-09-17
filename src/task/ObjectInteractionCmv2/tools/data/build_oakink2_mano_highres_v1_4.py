#!/usr/bin/env python3
"""Produce a Cmv2 4096-point MANO/KNN32 OakInk2 cache from official annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from src.task.ObjectInteractionCm.tools.data import build_bilateral_mano_v1_4_cache as mano_cache
from src.task.ObjectInteractionCm.tools.data import export_oakink2_inspire_v1_4 as legacy
from src.task.ObjectInteractionCmv2.tools.data.backfill_oakink2_inspire_v1_4 import RawAnnotationGeometryStore


MODIFICATION_VERSION = "V1.4.5"
OBJECT_POINTS = 4096
MANO_PER_SIDE = 2048
MANO_POINTS = 4096
DECODER_PER_SIDE = 1538
DECODER_POINTS = 3076
KNN_K = 32
REPO_ROOT = Path(__file__).resolve().parents[5]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_id(value: str) -> str:
    suffix = value[len("oakink2/"):] if value.startswith("oakink2/") else value
    return legacy._safe_id(suffix)


def _target(root: Path, row: Mapping[str, Any]) -> Path:
    return root / "sequences" / "train" / "mano" / "oakink2" / _safe_id(str(row["id"]))


def _validate(path: Path) -> dict[str, Any]:
    geometry = path / "geometry"; manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8")); frames = int(manifest["frame_count"])
    expected = {"obj_points_pool_world.npy": (frames, OBJECT_POINTS, 3), "obj_normals_pool_world.npy": (frames, OBJECT_POINTS, 3),
                "hand_points_world.npy": (frames, DECODER_POINTS, 3), "hand_normals_world.npy": (frames, DECODER_POINTS, 3),
                "knn_hand_points_world.npy": (frames, MANO_POINTS, 3), "knn_hand_normals_world.npy": (frames, MANO_POINTS, 3),
                "obj_knn_indices.npy": (frames, OBJECT_POINTS, KNN_K), "obj_candidate_mask_2cm.npy": (frames, OBJECT_POINTS),
                "obj_pose_world.npy": (frames, 4, 4), "source_frame_id.npy": (frames,), "frame_time.npy": (frames,)}
    for name, shape in expected.items():
        value = np.load(geometry / name, mmap_mode="r")
        if value.shape != shape: raise ValueError(f"{path}/{name}: expected {shape}, got {value.shape}")
        if value.dtype.kind == "f" and not np.isfinite(np.asarray(value[: min(2, frames)])).all(): raise ValueError(f"{path}/{name}: non-finite")
    index_max = int(np.asarray(np.load(geometry / "obj_knn_indices.npy", mmap_mode="r")).max())
    if index_max >= MANO_POINTS: raise ValueError(f"{path}: KNN index outside MANO stream")
    ids = np.asarray(np.load(geometry / "source_frame_id.npy", mmap_mode="r"))
    if np.any(np.diff(ids) <= 0) or abs(float(manifest["effective_fps"]) - 30.0) > 1e-4: raise ValueError(f"{path}: not increasing 30 Hz stream")
    poses = np.asarray(np.load(geometry / "obj_pose_world.npy", mmap_mode="r"))
    if not all(legacy._is_se3(value) for value in poses[:: max(1, len(poses) // 16)]): raise ValueError(f"{path}: invalid object SE(3)")
    return {"id": manifest["sequence_id"], "path": str(path.resolve()), "frames": frames, "knn_index_max": index_max}


def _decoder(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.linspace(0, MANO_PER_SIDE - 1, DECODER_PER_SIDE).round().astype(np.int64)
    ids = np.concatenate((ids, MANO_PER_SIDE + ids))
    return np.ascontiguousarray(points[:, ids]), np.ascontiguousarray(normals[:, ids])


def _process(row: Mapping[str, Any], annotation: Mapping[str, Any], stage3: RawAnnotationGeometryStore,
             reconstructor: Any, correspondences: dict[str, dict[str, np.ndarray]], output: Path,
             device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    destination = _target(output, row)
    if (destination / "geometry" / "manifest.json").is_file():
        if not args.resume: raise FileExistsError(destination)
        return _validate(destination)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or destination.exists(): raise FileExistsError(f"incomplete/existing target: {destination}")
    geometry = partial / "geometry"; geometry.mkdir(parents=True)
    try:
        frame_ids = np.asarray(row["selected_frame_ids"], dtype=np.int64); timeline = np.asarray(row["selected_timeline_positions"], dtype=np.int64); object_ids = [str(value) for value in row["selected_object_ids"]]
        local_points, local_normals, poses, _ = stage3.parts(object_ids, frame_ids)
        seed = int.from_bytes(hashlib.blake2b(str(row["id"]).encode(), digest_size=8).digest(), "little") & 0xFFFFFFFF
        objects, normals, pose, point_ids = legacy._sample_object_pool(local_points, local_normals, poses, seed=seed)
        high_points: list[np.ndarray] = []; high_normals: list[np.ndarray] = []
        for side in ("left", "right"):
            vertices, _ = reconstructor.reconstruct(annotation["raw_mano"], frame_ids.tolist(), side)
            faces = np.asarray(reconstructor.faces[side], dtype=np.int64); reference = np.asarray(correspondences[side]["reference_faces"], dtype=np.int64)
            if faces.shape != reference.shape or not np.array_equal(faces, reference): raise ValueError(f"{row['id']}/{side}: official MANO topology differs from fixed correspondence")
            points, side_normals = mano_cache._sample_surface_chunk(vertices, faces, correspondences[side])
            high_points.append(points); high_normals.append(side_normals)
        high = np.ascontiguousarray(np.concatenate(high_points, axis=1)); high_normals_value = np.ascontiguousarray(np.concatenate(high_normals, axis=1)); decoder, decoder_normals = _decoder(high, high_normals_value)
        for name, value in (("obj_points_pool_world", objects), ("obj_normals_pool_world", normals), ("obj_pose_world", pose), ("obj_point_id", point_ids),
                            ("source_frame_id", frame_ids.astype(np.int32)), ("frame_time", timeline.astype(np.float32) / legacy.TARGET_FPS),
                            ("knn_hand_points_world", high), ("knn_hand_normals_world", high_normals_value), ("hand_points_world", decoder), ("hand_normals_world", decoder_normals),
                            ("obj_candidate_mask_5cm", np.ones((len(frame_ids), OBJECT_POINTS), dtype=bool))): np.save(geometry / f"{name}.npy", value)
        mano_cache._build_knn(geometry, device=device, frame_batch_size=args.knn_frame_batch, object_chunk=args.knn_object_chunk)
        _write_json(geometry / "manifest.json", {"schema_name": "ref2dex_object_interaction_cmv2_oakink2_mano_v1_4", "schema_version": "1.0.0", "modification_version": MODIFICATION_VERSION,
            "sequence_id": str(row["id"]), "dataset": "oakink2", "source_dataset": "oakink2", "source": "mano", "hand_variant": "mano", "split": "train", "coordinate_frame": "object_pose_t",
            "object_representation": "single_root_articulated_world_points_with_reference_pose", "hand_side": "bilateral_merged_left_then_right", "frame_count": len(frame_ids), "object_pool_points": OBJECT_POINTS,
            "decoder_hand_points": DECODER_POINTS, "decoder_points_per_side": DECODER_PER_SIDE, "knn_hand_points": MANO_POINTS, "knn_points_per_side": MANO_PER_SIDE, "knn_k": KNN_K, "knn_index_dtype": "uint16",
            "interaction_radius_m": 0.02, "hand_supervision_radius_m": 0.02, "effective_fps": 30.0, "source_fps": 120.0, "source_type": "oakink2_official_quaternion_mano_fixed_surface_correspondence",
            "selected_root_id": row["selected_root_id"], "selected_object_ids": object_ids, "selection_index_id": str(row["id"]), "surface_sampling": {"method": "fixed MANO reference triangle barycentric", "seed": 2024, "points_per_side": MANO_PER_SIDE, "cross_frame_fixed": True}})
        destination.parent.mkdir(parents=True, exist_ok=True); os.replace(partial, destination)
        return _validate(destination)
    except Exception:
        if partial.exists(): shutil.rmtree(partial)
        raise


def run(args: argparse.Namespace) -> int:
    selection_path = args.selection_index.resolve(); selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_name") != legacy.SELECTION_SCHEMA: raise ValueError(f"unexpected selection schema: {selection.get('schema_name')}")
    rows = list(selection.get("segments", [])); rows = rows[args.offset: args.offset + args.limit] if args.limit else rows[args.offset:]
    if not rows: raise ValueError("no selected OakInk2 segments")
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError(f"CUDA is required, got {device}")
    torch.cuda.set_device(device); commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    manifest_path = output / f"run_manifest_{args.run_id}.json"; manifest = {"schema_name": "ref2dex_data_run_manifest_v1", "task": "ObjectInteractionCmv2", "operation": "oakink2_mano_highres_export", "run_id": args.run_id, "run_status": "STARTED", "started_at": _now(), "modification_version": MODIFICATION_VERSION, "base_commit": commit, "device": str(device), "expected_segments": len(rows), "completed_segments": 0, "completed_frames": 0, "failures": [], "inputs": {"selection_index": str(selection_path), "annotation_root": str(args.annotation_root.resolve()), "stage3_root": str(args.stage3_root.resolve())}, "outputs": {"root": str(output)}, "conclusion": "INCONCLUSIVE"}; _write_json(manifest_path, manifest)
    stage3_map = legacy._load_stage3_map(args.stage3_root.resolve()); reconstructor = legacy._ManoReconstructor(args.mano_root.resolve(), device, args.mano_batch_size); correspondences = mano_cache._correspondences(args.mano_root.resolve()); records = []
    for ordinal, row in enumerate(rows, 1):
        try:
            with (args.annotation_root.resolve() / f"{row['sequence']}.pkl").open("rb") as stream: annotation = pickle.load(stream)
            stage3 = RawAnnotationGeometryStore(stage3_map, str(row["sequence"]), annotation, args.canonical_root.resolve() if args.canonical_root else None)
            value = _process(row, annotation, stage3, reconstructor, correspondences, output, device, args)
            records.append({"source": "mano", "hand_variant": "mano", "id": str(row["id"]), "path": value["path"], "dataset": "oakink2", "split": "train", "frame_count": value["frames"]})
            manifest["completed_segments"] = len(records); manifest["completed_frames"] += int(value["frames"]); _write_json(manifest_path, manifest); print(json.dumps({"status": "COMPLETED", "index": ordinal, "total": len(rows), **value}), flush=True)
        except Exception as exc:
            manifest["failures"].append({"selection_id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"}); _write_json(manifest_path, manifest); traceback.print_exc()
    if manifest["failures"]:
        manifest.update({"run_status": "FAILED", "finished_at": _now(), "conclusion": "INVALID_IMPLEMENTATION"}); _write_json(manifest_path, manifest); return 1
    index = {"schema_name": "ref2dex_object_interaction_cm_oakink2_index_v1_4", "schema_version": "1.1.0", "created_at": _now(), "modification_version": MODIFICATION_VERSION, "object_pool_points": OBJECT_POINTS, "model_object_points": 1024, "decoder_hand_points_per_stream": DECODER_POINTS, "knn_hand_points_per_stream": {"mano": MANO_POINTS}, "max_knn_hand_points": MANO_POINTS, "knn_k": KNN_K, "split_policy": {"oakink2": "existing selected train segments"}, "sequences": {"train": records, "val": [], "test": []}, "counts": {"train": len(records), "val": 0, "test": 0, "frames": sum(int(row["frame_count"]) for row in records)}, "selection_index": str(selection_path)}
    _write_json(output / "index.json", index); _write_json(output / "cache_manifest.json", {"schema_name": "ref2dex_cmv2_highres_cache_manifest_v1", "modification_version": MODIFICATION_VERSION, "dataset": "oakink2", "variant": "mano", "total_sequences": len(records), "total_frames": manifest["completed_frames"], "hand_contract": "decoder bilateral 3076 compatibility points; MANO KNN bilateral 4096 points", "effective_fps": 30.0, "knn_k": KNN_K, "validation": {"bad_count": 0, "bad_examples": []}})
    manifest.update({"run_status": "COMPLETED", "finished_at": _now(), "outputs": {"root": str(output), "index": str((output / "index.json").resolve()), "cache_manifest": str((output / "cache_manifest.json").resolve())}, "conclusion": "SUPPORTED"}); _write_json(manifest_path, manifest); print(json.dumps({"status": "COMPLETED", "segments": len(records), "frames": manifest["completed_frames"]}), flush=True); return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-index", type=Path, required=True); parser.add_argument("--annotation-root", type=Path, required=True); parser.add_argument("--stage3-root", type=Path, required=True); parser.add_argument("--canonical-root", type=Path); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--mano-root", type=Path, required=True); parser.add_argument("--device", default="cuda:2"); parser.add_argument("--mano-batch-size", type=int, default=128); parser.add_argument("--knn-frame-batch", type=int, default=2); parser.add_argument("--knn-object-chunk", type=int, default=512); parser.add_argument("--run-id", required=True); parser.add_argument("--limit", type=int, default=0); parser.add_argument("--offset", type=int, default=0); parser.add_argument("--resume", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__": main()
