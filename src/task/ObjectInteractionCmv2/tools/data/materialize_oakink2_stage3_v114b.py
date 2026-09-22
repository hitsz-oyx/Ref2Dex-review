#!/usr/bin/env python3
"""Materialize the existing OakInk2 Stage3 contract from a portable raw bundle.

The output is intentionally consumable by the existing OakInk2 selection,
MANO/Inspire cache, and part-adapter producers.  It does not introduce a new
selection policy or coordinate convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from src.task.ObjectInteractionCm.tools.data.export_oakink2_inspire_v1_4 import (
    _ManoReconstructor,
    _is_se3,
)


WORK_VERSION = "V1.14"
SCHEMA_NAME = "train_corr_static_v2"
OBJECT_POINTS = 4096
CONTACT_KEEP_M = 0.05
REPO_ROOT = Path(__file__).resolve().parents[5]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


def sample_object_surface(mesh_path: Path, count: int = OBJECT_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample a mesh in its canonical, metre-valued frame."""
    import trimesh

    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(faces):
        raise ValueError(f"invalid OakInk2 mesh: {mesh_path}")
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if not np.isfinite(area2).all() or float(area2.sum()) <= 0:
        raise ValueError(f"degenerate OakInk2 mesh: {mesh_path}")
    rng = np.random.default_rng(_seed(mesh_path.parent.name))
    chosen = rng.choice(len(faces), size=int(count), replace=True, p=area2 / area2.sum())
    uv = rng.random((int(count), 2))
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1.0 - uv[flip]
    selected = triangles[chosen]
    points = selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    normals = cross[chosen] / area2[chosen, None]
    return np.ascontiguousarray(points, dtype=np.float32), np.ascontiguousarray(normals, dtype=np.float32)


def hand_face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float32)[:, np.asarray(faces, dtype=np.int64)]
    centres = triangles.mean(axis=2)
    normals = np.cross(triangles[:, :, 1] - triangles[:, :, 0],
                       triangles[:, :, 2] - triangles[:, :, 0])
    normals /= np.clip(np.linalg.norm(normals, axis=2, keepdims=True), 1e-8, None)
    return (np.ascontiguousarray(centres, dtype=np.float32),
            np.ascontiguousarray(normals, dtype=np.float32))


def object_frame_points(world_points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    rotations = np.asarray(poses[:, :3, :3], dtype=np.float32)
    translations = np.asarray(poses[:, :3, 3], dtype=np.float32)
    return np.einsum("tpk,tkv->tpv", world_points - translations[:, None], rotations)


def hand_to_object_distances(object_points: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    tree = cKDTree(np.asarray(object_points, dtype=np.float32))
    distances = tree.query(np.asarray(hand_points).reshape(-1, 3), workers=-1)[0]
    return np.ascontiguousarray(distances.reshape(hand_points.shape[:2]), dtype=np.float32)


def _mesh_path(object_root: Path, object_id: str) -> Path:
    for family in ("object_repair", "object_raw"):
        candidate = object_root / family / "align_ds" / object_id / "model.obj"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"OakInk2 object mesh unavailable: {object_id}")


def build_sequence(
    annotation_path: Path,
    object_root: Path,
    output_root: Path,
    reconstructor: _ManoReconstructor,
    *,
    sequence_ordinal: int,
    frame_offset: int = 0,
    max_frames: int = 0,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    with annotation_path.open("rb") as stream:
        annotation: Mapping[str, Any] = pickle.load(stream)
    raw_mano = annotation.get("raw_mano", {})
    official = set(int(value) for value in annotation.get("mocap_frame_id_list", raw_mano))
    frame_ids = sorted(official & {int(value) for value in raw_mano})
    frame_ids = frame_ids[int(frame_offset):]
    if max_frames:
        frame_ids = frame_ids[: int(max_frames)]
    if not frame_ids:
        raise ValueError(f"{annotation_path.stem}: no common raw MANO frames")
    hand_geometry = {}
    for side in ("left", "right"):
        vertices, _ = reconstructor.reconstruct(raw_mano, frame_ids, side)
        hand_geometry[side] = hand_face_geometry(vertices, reconstructor.faces[side])
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for object_id in (str(value) for value in annotation.get("obj_list", [])):
        frame_map = annotation.get("obj_transf", {}).get(object_id)
        if not isinstance(frame_map, Mapping):
            skipped.append({"sequence": annotation_path.stem, "object": object_id, "reason": "missing_obj_transf"})
            continue
        available = np.asarray([value for value in frame_ids if int(value) in frame_map], dtype=np.int64)
        if len(available) != len(frame_ids):
            skipped.append({"sequence": annotation_path.stem, "object": object_id, "reason": "incomplete_obj_transf"})
            continue
        poses = np.stack([np.asarray(frame_map[int(value)], dtype=np.float32) for value in available])
        if not all(_is_se3(value) for value in poses):
            skipped.append({"sequence": annotation_path.stem, "object": object_id, "reason": "invalid_obj_transf"})
            continue
        object_points, object_normals = sample_object_surface(_mesh_path(object_root, object_id))
        for side in ("left", "right"):
            hand_centres, hand_normals_world = hand_geometry[side]
            local_hand = object_frame_points(hand_centres, poses)
            local_hand_normals = np.einsum(
                "tpk,tkv->tpv", hand_normals_world, np.asarray(poses[:, :3, :3], dtype=np.float32))
            distances = hand_to_object_distances(object_points, local_hand)
            keep = np.flatnonzero(np.min(distances, axis=1) <= CONTACT_KEEP_M)
            if not len(keep):
                skipped.append({"sequence": annotation_path.stem, "object": object_id, "side": side,
                                "reason": "no_interacting_frame"})
                continue
            stem = f"{sequence_ordinal:05d}_{object_id.replace('@', '_')}_{side}"
            destination = output_root / f"{stem}.npz"
            if destination.exists() and not overwrite:
                raise FileExistsError(destination)
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    schema_name=np.asarray(SCHEMA_NAME), schema_version=np.asarray("2.0.0"),
                    dataset_id=np.asarray("oakink2"), dataset_name=np.asarray("OakInk2"),
                    seq_id=np.asarray(f"{annotation_path.stem}/{object_id}/{side}"), side=np.asarray(side),
                    raw_frame_id=available[keep].astype(np.int32),
                    obj_points=object_points, obj_normals=object_normals,
                    hand_points=local_hand[keep], hand_normals=local_hand_normals[keep],
                    hand_to_obj_min_dist=distances[keep],
                    coordinate_frame=np.asarray("object"), obj_root_pose_world=poses[keep],
                    work_version=np.asarray(WORK_VERSION),
                )
            os.replace(temporary, destination)
            rows.append({"file": str(destination.resolve()),
                         "seq_id": f"{annotation_path.stem}/{object_id}/{side}",
                         "object_id": object_id, "side": side, "frames": int(len(keep)),
                         "min_dist_m": float(distances[keep].min())})
    return rows, skipped


def run(args: argparse.Namespace) -> int:
    if args.offset < 0 or args.frame_offset < 0 or args.limit < 0 or args.max_frames < 0:
        raise ValueError("offset, frame-offset, limit, and max-frames must be non-negative")
    annotation_root = args.annotation_root.resolve()
    object_root = args.object_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    annotation_paths = sorted(annotation_root.glob("*.pkl"))[int(args.offset):]
    if args.limit:
        annotation_paths = annotation_paths[: int(args.limit)]
    if not annotation_paths:
        raise ValueError("no OakInk2 annotations selected")
    import torch
    device = torch.device(args.device)
    reconstructor = _ManoReconstructor(args.mano_root.resolve(), device, args.mano_batch_size)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    counts = Counter()
    for relative_ordinal, path in enumerate(annotation_paths):
        ordinal = int(args.offset) + relative_ordinal
        try:
            produced, omitted = build_sequence(
                path, object_root, output_root, reconstructor,
                sequence_ordinal=ordinal, frame_offset=args.frame_offset,
                max_frames=args.max_frames, overwrite=args.overwrite)
            rows.extend(produced); skipped.extend(omitted)
            counts.update(row["side"] for row in produced)
        except Exception as error:
            failures.append({"sequence": path.stem, "error": f"{type(error).__name__}: {error}",
                             "traceback": traceback.format_exc()})
    stats = {"schema_name": "ref2dex_oakink2_stage3_materialization_v114b", "work_version": WORK_VERSION,
             "source": str(annotation_root), "output_root": str(output_root), "num_sequences": len(annotation_paths),
             "num_outputs": len(rows), "num_frames": sum(int(row["frames"]) for row in rows),
             "side_counts": dict(counts), "outputs": rows, "skipped": skipped, "failures": failures}
    stats_path = output_root / f"oakink2_stage3_stats_{int(args.offset):04d}.json"
    _write_json(stats_path, stats)
    manifest = {"schema_name": "ref2dex_data_run_manifest_v1", "task": "ObjectInteractionCmv2",
                "work_version": WORK_VERSION, "run_id": args.run_id,
                "run_status": "FAILED" if failures else "COMPLETED", "created_at": _now(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                "operation": "oakink2_portable_raw_to_existing_stage3_contract",
                "inputs": {"annotation_root": str(annotation_root), "object_root": str(object_root),
                           "mano_root": str(args.mano_root.resolve())},
                "parameters": {"object_points": OBJECT_POINTS, "contact_keep_m": CONTACT_KEEP_M,
                               "frame_offset": int(args.frame_offset), "max_frames": int(args.max_frames)},
                "outputs": {"stage3_root": str(output_root), "stats": str(stats_path)},
                "conclusion": "INVALID_IMPLEMENTATION" if failures else "N/A"}
    _write_json(output_root / f"run_manifest_{args.run_id}.json", manifest)
    return int(bool(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mano-batch-size", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
