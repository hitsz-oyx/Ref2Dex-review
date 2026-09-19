#!/usr/bin/env python3
"""Build the V1.3 ObjectInteractionCm cache with offline batched KNN indices.

The source is the completed V1.2.5 cache.  Its object geometry and 1538-point
decoder stream are copied byte-for-byte.  A second, source-specific hand
surface stream is generated for offline object-to-hand KNN:

* MANO: 2048 points
* Inspire: 10135 points

The cache stores only ``[T,4096,32]`` uint16 hand indices.  Training gathers
those indices after its normal random 1024-point object sampling and computes
the 32 distances again.  This keeps the radius threshold configurable without
repeating the full object-to-hand search.

``worker`` mode is safe to run concurrently: each worker owns a deterministic
sequence shard and writes only sequence directories plus its own report.
``finalize`` verifies that all source entries were completed exactly once and
writes the shared index and run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from pytorch3d.ops import knn_points

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task.ObjectInteractionCm.research.hand_region_sampling.run import (  # noqa: E402
    INSPIRE_RATIO_POINTS,
    NUM_POINTS,
    SURFACE_SEED,
    _build_inspire_pool,
    _build_mano_pool,
)
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (  # noqa: E402
    InspireUrdfModel,
    NATIVE_Q_START,
    NUM_DOFS,
    _load_tensor,
    _to_object_frame,
)
from src.task.ObjectInteractionCm.research.hand_region_sampling.build_trajectory_preview import (  # noqa: E402
    _sample_correspondence,
)


SOURCE_INDEX_SCHEMA = "ref2dex_object_interaction_cm_index_v1_1"
INDEX_SCHEMA = "ref2dex_object_interaction_cm_index_v1_2"
CACHE_SCHEMA = "ref2dex_object_interaction_cm_dexplore_rl_v1_3"
OBJECT_POOL_POINTS = 4096
DECODER_HAND_POINTS = 1538
KNN_K = 32
INTERACTION_RADIUS_M = 0.02
HAND_SUPERVISION_RADIUS_M = 0.02
MANO_KNN_POINTS = NUM_POINTS
INSPIRE_KNN_POINTS = INSPIRE_RATIO_POINTS
DEFAULT_GEOMETRY_BATCH = 64
DEFAULT_KNN_BATCH = 4


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    return value.replace("/", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _source_entries(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != SOURCE_INDEX_SCHEMA:
        raise ValueError(f"Expected {SOURCE_INDEX_SCHEMA}, got {payload.get('schema_name')!r}")
    entries: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        values = payload.get("sequences", {}).get(split, [])
        if not isinstance(values, list):
            raise ValueError(f"Malformed source split {split}: {type(values)}")
        for item in values:
            if not isinstance(item, dict):
                raise ValueError(f"Malformed source entry: {item!r}")
            if item.get("split", split) != split:
                raise ValueError(f"Source entry split mismatch: {item!r}")
            if item.get("variant") not in {"mano", "inspire_rl"}:
                raise ValueError(f"Unsupported source variant: {item.get('variant')!r}")
            required = ("id", "path", "source", "variant", "split")
            missing = [key for key in required if key not in item]
            if missing:
                raise ValueError(f"Source entry missing {missing}: {item!r}")
            entries.append(dict(item))
    if not entries:
        raise ValueError(f"No source entries in {index_path}")
    return entries


def _entry_key(entry: dict[str, Any]) -> str:
    return f"{entry['split']}::{entry['variant']}::{entry['id']}"


def _resolve_source_path(source_root: Path, entry: dict[str, Any]) -> Path:
    raw = Path(str(entry["path"]))
    return raw if raw.is_absolute() else (source_root / raw).resolve()


def _copy_source_geometry(source_geometry: Path, output_geometry: Path) -> dict[str, np.ndarray]:
    required = (
        "obj_points_pool_world.npy",
        "obj_normals_pool_world.npy",
        "obj_pose_world.npy",
        "source_frame_id.npy",
        "frame_time.npy",
        "hand_points_world.npy",
        "hand_normals_world.npy",
    )
    for name in required:
        path = source_geometry / name
        if not path.is_file():
            raise FileNotFoundError(f"V1.2.5 geometry is missing {path}")
        shutil.copyfile(path, output_geometry / name)
    arrays = {
        "obj": np.load(output_geometry / "obj_points_pool_world.npy", mmap_mode="r"),
        "obj_normals": np.load(output_geometry / "obj_normals_pool_world.npy", mmap_mode="r"),
        "pose": np.load(output_geometry / "obj_pose_world.npy", mmap_mode="r"),
        "raw": np.load(output_geometry / "source_frame_id.npy", mmap_mode="r"),
        "time": np.load(output_geometry / "frame_time.npy", mmap_mode="r"),
        "decoder_hand": np.load(output_geometry / "hand_points_world.npy", mmap_mode="r"),
        "decoder_normals": np.load(output_geometry / "hand_normals_world.npy", mmap_mode="r"),
    }
    obj = arrays["obj"]
    if obj.ndim != 3 or obj.shape[1:] != (OBJECT_POOL_POINTS, 3):
        raise ValueError(f"Unexpected object shape in {source_geometry}: {obj.shape}")
    if arrays["obj_normals"].shape != obj.shape:
        raise ValueError("Object normal shape mismatch")
    if arrays["pose"].shape != (len(obj), 4, 4):
        raise ValueError("Object pose shape mismatch")
    if arrays["raw"].shape != (len(obj),) or arrays["time"].shape != (len(obj),):
        raise ValueError("Frame metadata shape mismatch")
    if arrays["decoder_hand"].shape != (len(obj), DECODER_HAND_POINTS, 3):
        raise ValueError(f"Expected decoder hand [T,{DECODER_HAND_POINTS},3], got {arrays['decoder_hand'].shape}")
    if arrays["decoder_normals"].shape != arrays["decoder_hand"].shape:
        raise ValueError("Decoder normal shape mismatch")
    return arrays


def _sample_mano_surface(
    parent_root: Path,
    source_frame_ids: np.ndarray,
    target_pose: np.ndarray,
    correspondence: dict[str, np.ndarray],
    output_points: np.ndarray,
    output_normals: np.ndarray,
    *,
    batch_size: int,
) -> None:
    vertices_path = parent_root / "right" / "hand_mesh_vertices_world.npy"
    faces_path = parent_root / "right" / "hand_mesh_faces.npy"
    old_pose_path = parent_root / "shared" / "obj_pose_world.npy"
    old_raw_path = parent_root / "shared" / "raw_frame_id.npy"
    for path in (vertices_path, faces_path, old_pose_path, old_raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"MANO provenance file is unavailable: {path}")
    vertices_world = np.load(vertices_path, mmap_mode="r")
    faces = np.asarray(np.load(faces_path, mmap_mode="r"), dtype=np.int64)
    old_pose = np.load(old_pose_path, mmap_mode="r")
    old_raw = np.asarray(np.load(old_raw_path, mmap_mode="r"), dtype=np.int64)
    if vertices_world.ndim != 3 or vertices_world.shape[1:] != (778, 3):
        raise ValueError(f"Invalid MANO mesh vertices {vertices_path}: {vertices_world.shape}")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError(f"Invalid MANO mesh faces {faces_path}: {faces.shape}")
    if old_pose.shape != (len(old_raw), 4, 4):
        raise ValueError(f"Invalid MANO object poses {old_pose_path}: {old_pose.shape}")
    raw_to_index = {int(value): index for index, value in enumerate(old_raw)}
    parent_indices = np.asarray([raw_to_index.get(int(value), -1) for value in source_frame_ids], dtype=np.int64)
    if np.any(parent_indices < 0):
        missing = source_frame_ids[parent_indices < 0][:5].tolist()
        raise KeyError(f"MANO source frame ids are absent from {old_raw_path}: {missing}")
    face_ids = np.asarray(correspondence["face_ids"], dtype=np.int64)
    barycentric = np.asarray(correspondence["barycentric"], dtype=np.float32)
    if face_ids.shape != (output_points.shape[1],) or barycentric.shape != (output_points.shape[1], 3):
        raise ValueError("MANO correspondence shape mismatch")
    if np.any(face_ids < 0) or np.any(face_ids >= len(faces)):
        raise ValueError("MANO correspondence has an out-of-range face")
    for start in range(0, len(source_frame_ids), max(1, int(batch_size))):
        stop = min(len(source_frame_ids), start + max(1, int(batch_size)))
        indices = parent_indices[start:stop]
        old_vertices = np.asarray(vertices_world[indices], dtype=np.float32)
        old_object_pose = np.asarray(old_pose[indices], dtype=np.float32)
        local_vertices = (
            old_vertices - old_object_pose[:, None, :3, 3]
        ) @ old_object_pose[:, :3, :3]
        triangles = local_vertices[:, faces[face_ids]]
        points_local = np.einsum("pj,bpjd->bpd", barycentric, triangles, optimize=True)
        cross = np.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0])
        normals_local = cross / np.clip(np.linalg.norm(cross, axis=-1, keepdims=True), 1e-12, None)
        rotation = np.asarray(target_pose[start:stop, :3, :3], dtype=np.float32)
        translation = np.asarray(target_pose[start:stop, :3, 3], dtype=np.float32)
        output_points[start:stop] = points_local @ np.transpose(rotation, (0, 2, 1)) + translation[:, None, :]
        output_normals[start:stop] = normals_local @ np.transpose(rotation, (0, 2, 1))
        output_normals[start:stop] /= np.clip(
            np.linalg.norm(output_normals[start:stop], axis=-1, keepdims=True), 1e-8, None
        )


def _inspire_sample_groups(
    model: InspireUrdfModel,
    correspondence: dict[str, np.ndarray],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    face_ids = np.asarray(correspondence["face_ids"], dtype=np.int64)
    visual_ids = np.asarray(correspondence["visual_ids"], dtype=np.int64)
    barycentric = np.asarray(correspondence["barycentric"], dtype=np.float32)
    groups: list[tuple[int, np.ndarray, np.ndarray]] = []
    for visual_id in sorted(set(int(value) for value in visual_ids)):
        mask = visual_ids == visual_id
        visual = model.visuals[visual_id]
        triangles = visual.vertices[visual.faces[face_ids[mask]]]
        points = np.einsum("pj,pjd->pd", barycentric[mask], triangles, optimize=True)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals = cross / np.clip(np.linalg.norm(cross, axis=-1, keepdims=True), 1e-12, None)
        groups.append((visual_id, points.astype(np.float32), normals.astype(np.float32)))
    return groups


def _sample_inspire_surface(
    model: InspireUrdfModel,
    q_native: np.ndarray,
    correspondence: dict[str, np.ndarray],
    output_points: np.ndarray,
    output_normals: np.ndarray,
    *,
    batch_size: int,
) -> None:
    if q_native.ndim != 2 or q_native.shape[1:] != (NUM_DOFS,):
        raise ValueError(f"Unexpected Inspire q shape: {q_native.shape}")
    if len(q_native) != len(output_points):
        raise ValueError("Inspire q/geometry frame mismatch")
    groups = _inspire_sample_groups(model, correspondence)
    if not groups:
        raise ValueError("Inspire correspondence has no samples")
    for start in range(0, len(q_native), max(1, int(batch_size))):
        stop = min(len(q_native), start + max(1, int(batch_size)))
        count = stop - start
        frame_transforms: dict[int, np.ndarray] = {
            visual_id: np.empty((count, 4, 4), dtype=np.float32)
            for visual_id, _, _ in groups
        }
        for local_frame, native_q in enumerate(np.asarray(q_native[start:stop], dtype=np.float64)):
            links = model.link_transforms(model.qpos_to_urdf_order(native_q))
            for visual_id, _, _ in groups:
                visual = model.visuals[visual_id]
                frame_transforms[visual_id][local_frame] = (
                    links[visual.link] @ visual.local_transform
                ).astype(np.float32)
        for visual_id, local_points, local_normals in groups:
            transform = frame_transforms[visual_id]
            points = np.einsum(
                "bij,pj->bpi", transform[:, :3, :3], local_points, optimize=True
            ) + transform[:, None, :3, 3]
            normals = np.einsum(
                "bij,pj->bpi", transform[:, :3, :3], local_normals, optimize=True
            )
            normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
            visual_mask = np.asarray(correspondence["visual_ids"], dtype=np.int64) == visual_id
            output_points[start:stop, visual_mask] = points
            output_normals[start:stop, visual_mask] = normals


def _to_frame_batch(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (
        np.asarray(points_world, dtype=np.float32) - np.asarray(pose[:, None, :3, 3], dtype=np.float32)
    ) @ np.asarray(pose[:, :3, :3], dtype=np.float32)


def _knn_batch(
    object_local: np.ndarray,
    hand_local: np.ndarray,
    decoder_local: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_tensor = torch.as_tensor(np.asarray(object_local, dtype=np.float32), device=device)
    hand_tensor = torch.as_tensor(np.asarray(hand_local, dtype=np.float32), device=device)
    decoder_tensor = torch.as_tensor(np.asarray(decoder_local, dtype=np.float32), device=device)
    with torch.inference_mode():
        result = knn_points(
            object_tensor,
            hand_tensor,
            K=KNN_K,
            return_nn=False,
            return_sorted=True,
        )
        distances = result.dists.clamp_min(0.0).sqrt()
        decoder_result = knn_points(
            decoder_tensor,
            object_tensor,
            K=1,
            return_nn=False,
            return_sorted=True,
        )
        decoder_distances = decoder_result.dists.clamp_min(0.0).sqrt()[..., 0]
    return (
        result.idx.detach().cpu().numpy().astype(np.uint16, copy=False),
        (distances <= INTERACTION_RADIUS_M).any(dim=-1).detach().cpu().numpy().astype(bool),
        (decoder_distances <= HAND_SUPERVISION_RADIUS_M).detach().cpu().numpy().astype(bool),
        decoder_distances.min(dim=-1).values.detach().cpu().numpy().astype(np.float32),
    )


def _build_knn_arrays(
    geometry: Path,
    arrays: dict[str, np.ndarray],
    *,
    knn_hand_count: int,
    device: torch.device,
    batch_size: int,
) -> None:
    frame_count = len(arrays["obj"])
    knn_hand = np.load(geometry / "knn_hand_points_world.npy", mmap_mode="r")
    if knn_hand.shape != (frame_count, knn_hand_count, 3):
        raise ValueError(f"KNN hand shape mismatch: {knn_hand.shape}")
    decoder_hand = arrays["decoder_hand"]
    indices_path = geometry / "obj_knn_indices.npy"
    interaction_path = geometry / "obj_candidate_mask_2cm.npy"
    supervision_path = geometry / "hand_supervision_mask_2cm.npy"
    min_distance_path = geometry / "hand_min_object_distance_m.npy"
    indices = open_memmap(str(indices_path), mode="w+", dtype=np.uint16, shape=(frame_count, OBJECT_POOL_POINTS, KNN_K))
    interaction = open_memmap(
        str(interaction_path), mode="w+", dtype=np.bool_, shape=(frame_count, OBJECT_POOL_POINTS)
    )
    supervision = open_memmap(
        str(supervision_path), mode="w+", dtype=np.bool_, shape=(frame_count, DECODER_HAND_POINTS)
    )
    min_distance = open_memmap(str(min_distance_path), mode="w+", dtype=np.float32, shape=(frame_count,))
    try:
        for start in range(0, frame_count, max(1, int(batch_size))):
            stop = min(frame_count, start + max(1, int(batch_size)))
            pose = np.asarray(arrays["pose"][start:stop], dtype=np.float32)
            object_local = _to_frame_batch(arrays["obj"][start:stop], pose)
            hand_local = _to_frame_batch(knn_hand[start:stop], pose)
            decoder_local = _to_frame_batch(decoder_hand[start:stop], pose)
            batch_indices, batch_interaction, batch_supervision, batch_min_distance = _knn_batch(
                object_local, hand_local, decoder_local, device=device
            )
            indices[start:stop] = batch_indices
            interaction[start:stop] = batch_interaction
            supervision[start:stop] = batch_supervision
            min_distance[start:stop] = batch_min_distance
    finally:
        indices.flush()
        interaction.flush()
        supervision.flush()
        min_distance.flush()
        del indices, interaction, supervision, min_distance


def _finite_array(path: Path, *, chunk_size: int = 64) -> bool:
    array = np.load(path, mmap_mode="r")
    if array.ndim == 0:
        return bool(np.isfinite(array))
    for start in range(0, len(array), chunk_size):
        if not np.isfinite(np.asarray(array[start:start + chunk_size])).all():
            return False
    return True


def _validate_sequence(path: Path, variant: str, *, expected_knn_points: int) -> dict[str, Any]:
    geometry = path / "geometry"
    required = {
        "obj": geometry / "obj_points_pool_world.npy",
        "obj_normals": geometry / "obj_normals_pool_world.npy",
        "pose": geometry / "obj_pose_world.npy",
        "raw": geometry / "source_frame_id.npy",
        "time": geometry / "frame_time.npy",
        "decoder_hand": geometry / "hand_points_world.npy",
        "decoder_normals": geometry / "hand_normals_world.npy",
        "knn_hand": geometry / "knn_hand_points_world.npy",
        "knn_normals": geometry / "knn_hand_normals_world.npy",
        "indices": geometry / "obj_knn_indices.npy",
        "interaction": geometry / "obj_candidate_mask_2cm.npy",
        "supervision": geometry / "hand_supervision_mask_2cm.npy",
        "min_distance": geometry / "hand_min_object_distance_m.npy",
        "manifest": geometry / "manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Sequence cache is incomplete: {missing}")
    arrays = {key: np.load(value, mmap_mode="r") for key, value in required.items() if key != "manifest"}
    frame_count = int(arrays["obj"].shape[0])
    expected = {
        "obj": (frame_count, OBJECT_POOL_POINTS, 3),
        "obj_normals": (frame_count, OBJECT_POOL_POINTS, 3),
        "pose": (frame_count, 4, 4),
        "raw": (frame_count,),
        "time": (frame_count,),
        "decoder_hand": (frame_count, DECODER_HAND_POINTS, 3),
        "decoder_normals": (frame_count, DECODER_HAND_POINTS, 3),
        "knn_hand": (frame_count, expected_knn_points, 3),
        "knn_normals": (frame_count, expected_knn_points, 3),
        "indices": (frame_count, OBJECT_POOL_POINTS, KNN_K),
        "interaction": (frame_count, OBJECT_POOL_POINTS),
        "supervision": (frame_count, DECODER_HAND_POINTS),
        "min_distance": (frame_count,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} shape {arrays[key].shape} != {shape}")
    if arrays["indices"].dtype != np.uint16:
        raise TypeError(f"{path}: KNN index dtype must be uint16, got {arrays['indices'].dtype}")
    if int(np.asarray(arrays["indices"]).max()) >= expected_knn_points:
        raise ValueError(f"{path}: KNN index exceeds hand point count {expected_knn_points}")
    if any(not _finite_array(required[key]) for key in ("obj", "obj_normals", "pose", "decoder_hand", "decoder_normals", "knn_hand", "knn_normals", "min_distance")):
        raise ValueError(f"{path}: non-finite geometry")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_name") != CACHE_SCHEMA:
        raise ValueError(f"{path}: unexpected manifest schema {manifest.get('schema_name')!r}")
    if manifest.get("knn_k") != KNN_K or manifest.get("knn_hand_points") != expected_knn_points:
        raise ValueError(f"{path}: manifest KNN contract mismatch")
    return {
        "path": str(path),
        "variant": variant,
        "frames": frame_count,
        "knn_hand_points": expected_knn_points,
        "active_frames": int(np.asarray(arrays["interaction"], dtype=bool).any(axis=1).sum()),
        "active_object_points": int(np.asarray(arrays["interaction"], dtype=bool).sum()),
        "supervised_hand_points": int(np.asarray(arrays["supervision"], dtype=bool).sum()),
    }


def _sequence_knn_count(variant: str) -> int:
    if variant == "mano":
        return MANO_KNN_POINTS
    if variant == "inspire_rl":
        return INSPIRE_KNN_POINTS
    raise ValueError(f"Unsupported variant {variant!r}")


def _make_correspondence(
    variant: str,
    *,
    mano_pool: Any,
    inspire_pool: Any,
    surface_seed: int,
) -> dict[str, np.ndarray]:
    pool = mano_pool if variant == "mano" else inspire_pool
    return _sample_correspondence(pool, _sequence_knn_count(variant), surface_seed)


def _process_sequence(
    entry: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    mano_pool: Any,
    inspire_pool: Any,
    inspire_model: InspireUrdfModel,
    surface_seed: int,
    device: torch.device,
    geometry_batch_size: int,
    knn_batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    source_path = _resolve_source_path(source_root, entry)
    source_geometry = source_path / "geometry"
    source_manifest_path = source_geometry / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source geometry manifest is unavailable: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    variant = str(entry["variant"])
    knn_count = _sequence_knn_count(variant)
    relative_path = Path(str(entry["path"]))
    sequence_output = output_root / relative_path
    final_geometry = sequence_output / "geometry"
    if (final_geometry / "manifest.json").is_file():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing sequence: {sequence_output}")
        validation = _validate_sequence(sequence_output, variant, expected_knn_points=knn_count)
        print(f"[resume] reusing {entry['id']} ({variant})", flush=True)
        return {
            **entry,
            "frame_count": validation["frames"],
            "knn_hand_points": knn_count,
            "cache_path": str(relative_path),
            "validation": validation,
        }
    if sequence_output.exists():
        raise FileExistsError(
            f"Partial V1.3 sequence exists at {sequence_output}; remove it explicitly before retrying"
        )
    partial_output = sequence_output.with_name(sequence_output.name + ".partial")
    if partial_output.exists():
        raise FileExistsError(f"Partial V1.3 sequence exists at {partial_output}")
    partial_geometry = partial_output / "geometry"
    partial_geometry.mkdir(parents=True, exist_ok=False)
    try:
        arrays = _copy_source_geometry(source_geometry, partial_geometry)
        frame_count = len(arrays["obj"])
        correspondence = _make_correspondence(
            variant,
            mano_pool=mano_pool,
            inspire_pool=inspire_pool,
            surface_seed=surface_seed,
        )
        knn_points_path = partial_geometry / "knn_hand_points_world.npy"
        knn_normals_path = partial_geometry / "knn_hand_normals_world.npy"
        knn_points = open_memmap(
            str(knn_points_path), mode="w+", dtype=np.float32, shape=(frame_count, knn_count, 3)
        )
        knn_normals = open_memmap(
            str(knn_normals_path), mode="w+", dtype=np.float32, shape=(frame_count, knn_count, 3)
        )
        try:
            if variant == "mano":
                parent_value = source_manifest.get("input_parent_cache")
                if not parent_value:
                    raise ValueError(f"MANO source has no input_parent_cache: {source_manifest_path}")
                parent_root = Path(str(parent_value)).expanduser()
                if not parent_root.is_absolute():
                    parent_root = (source_root / parent_root).resolve()
                _sample_mano_surface(
                    parent_root,
                    np.asarray(arrays["raw"], dtype=np.int64),
                    np.asarray(arrays["pose"], dtype=np.float32),
                    correspondence,
                    knn_points,
                    knn_normals,
                    batch_size=geometry_batch_size,
                )
            else:
                rl_q = source_manifest.get("rl_q")
                if not isinstance(rl_q, dict) or not rl_q.get("tensor"):
                    raise ValueError(f"Inspire source has no rl_q.tensor: {source_manifest_path}")
                q_path = Path(str(rl_q["tensor"])).expanduser()
                if not q_path.is_absolute():
                    q_path = (source_root / q_path).resolve()
                dex_data = _load_tensor(q_path)
                q_native = dex_data[:, NATIVE_Q_START:NATIVE_Q_START + NUM_DOFS]
                if len(q_native) != frame_count:
                    raise ValueError(f"Inspire q/geometry frame mismatch: {len(q_native)} != {frame_count}")
                _sample_inspire_surface(
                    inspire_model,
                    q_native,
                    correspondence,
                    knn_points,
                    knn_normals,
                    batch_size=geometry_batch_size,
                )
        finally:
            knn_points.flush()
            knn_normals.flush()
            del knn_points, knn_normals
        _build_knn_arrays(
            partial_geometry,
            arrays,
            knn_hand_count=knn_count,
            device=device,
            batch_size=knn_batch_size,
        )
        manifest = {
            "schema_name": CACHE_SCHEMA,
            "schema_version": "1.0.0",
            "sequence_id": str(entry["id"]),
            "parent_seq_id": str(entry.get("parent_seq_id", entry["id"])),
            "split": str(entry["split"]),
            "variant": variant,
            "source": str(entry["source"]),
            "source_type": (
                "mano_parent_mesh_topology_area_uniform_knn"
                if variant == "mano"
                else "dexplore_rl_native_q_urdf_area_uniform_knn"
            ),
            "coordinate_frame": "object_pose_t",
            "world_frame": source_manifest.get("world_frame", "dexplore_native_object_pose_world"),
            "hand_side": "right",
            "object_pool_points": OBJECT_POOL_POINTS,
            "decoder_hand_points": DECODER_HAND_POINTS,
            "knn_hand_points": knn_count,
            "knn_k": KNN_K,
            "knn_index_dtype": "uint16",
            "knn_index_shape": [frame_count, OBJECT_POOL_POINTS, KNN_K],
            "knn_index_semantics": "object_pool_point_to_current_frame_knn_hand_point",
            "effective_fps": source_manifest.get("effective_fps", 30.0),
            "source_fps": source_manifest.get("source_fps", 120.0),
            "ds_rate": source_manifest.get("ds_rate", 4),
            "interaction_radius_m": INTERACTION_RADIUS_M,
            "hand_supervision_radius_m": HAND_SUPERVISION_RADIUS_M,
            "candidate_mask_file": "obj_candidate_mask_2cm.npy",
            "candidate_mask_shape": [frame_count, OBJECT_POOL_POINTS],
            "candidate_semantics": "full_object_pool_has_knn_hand_neighbor_within_2cm",
            "hand_supervision_mask_file": "hand_supervision_mask_2cm.npy",
            "hand_supervision_mask_shape": [frame_count, DECODER_HAND_POINTS],
            "distance_storage": "runtime_recompute_from_cached_32_indices",
            "surface_sampling": {
                "method": "global_surface_area_uniform_triangle_barycentric",
                "seed": int(surface_seed),
                "point_count": knn_count,
                "correspondence": {
                    "face_ids": np.asarray(correspondence["face_ids"], dtype=np.int64).tolist(),
                    "visual_ids": np.asarray(correspondence["visual_ids"], dtype=np.int64).tolist(),
                    "barycentric_sha256": hashlib.sha256(
                        np.asarray(correspondence["barycentric"], dtype=np.float32).tobytes()
                    ).hexdigest(),
                },
                "asset": (
                    "mano_parent_mesh_topology" if variant == "mano" else "inspire_hand_right_urdf_visuals"
                ),
            },
            "input_parent_cache": str(source_path.resolve()),
            "source_manifest": str(source_manifest_path.resolve()),
            "rl_q": source_manifest.get("rl_q") if variant == "inspire_rl" else None,
        }
        _write_json(partial_geometry / "manifest.json", manifest)
        sequence_output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_output, sequence_output)
        validation = _validate_sequence(sequence_output, variant, expected_knn_points=knn_count)
        print(
            f"[done] {entry['split']}/{variant}/{entry['id']} "
            f"frames={validation['frames']} active={validation['active_frames']}",
            flush=True,
        )
        return {
            **entry,
            "frame_count": frame_count,
            "knn_hand_points": knn_count,
            "cache_path": str(relative_path),
            "validation": validation,
        }
    except Exception:
        if partial_output.exists():
            shutil.rmtree(partial_output)
        raise


def _worker(
    *,
    index_path: Path,
    output_root: Path,
    shard_index: int,
    num_shards: int,
    device_name: str,
    surface_seed: int,
    geometry_batch_size: int,
    knn_batch_size: int,
    resume: bool,
    work_version: str,
) -> int:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}")
    entries = _source_entries(index_path)
    selected = [entry for ordinal, entry in enumerate(entries) if ordinal % num_shards == shard_index]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "workers").mkdir(parents=True, exist_ok=True)
    mano_model_dir = REPO_ROOT / "dataset/arctic/data/body_models/mano"
    urdf_path = Path("/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf")
    if not (mano_model_dir / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(f"MANO model is unavailable: {mano_model_dir}")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Inspire URDF is unavailable: {urdf_path}")
    mano_pool, _ = _build_mano_pool(mano_model_dir, 0.20)
    inspire_pool, _ = _build_inspire_pool(urdf_path, 0.20)
    inspire_model = InspireUrdfModel(urdf_path)
    source_root = index_path.parent.resolve()
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for V1.3 GPU-batched worker")
        torch.cuda.set_device(device)
    records = []
    for entry in selected:
        records.append(
            _process_sequence(
                entry,
                source_root=source_root,
                output_root=output_root,
                mano_pool=mano_pool,
                inspire_pool=inspire_pool,
                inspire_model=inspire_model,
                surface_seed=surface_seed,
                device=device,
                geometry_batch_size=geometry_batch_size,
                knn_batch_size=knn_batch_size,
                resume=resume,
            )
        )
    commit, dirty = _git_info()
    report = {
        "schema_name": "ref2dex_object_interaction_cm_v1_3_worker_report",
        "run_id": f"oicm-v1.3-worker-{shard_index}",
        "run_status": "COMPLETED",
        "created_at": _now(),
        "work_version": work_version,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "source_index": str(index_path),
        "source_index_sha256": _sha256(index_path),
        "output_root": str(output_root.resolve()),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "device": device_name,
        "surface_seed": surface_seed,
        "geometry_batch_size": geometry_batch_size,
        "knn_batch_size": knn_batch_size,
        "records": records,
        "sequence_count": len(records),
    }
    report_path = output_root / "workers" / f"shard_{shard_index:02d}.json"
    _write_json(report_path, report)
    print(json.dumps({"worker_report": str(report_path), "sequence_count": len(records)}, ensure_ascii=False))
    return 0


def _finalize(
    *,
    index_path: Path,
    output_root: Path,
    num_shards: int,
    work_version: str,
) -> int:
    source_payload = json.loads(index_path.read_text(encoding="utf-8"))
    source_entries = _source_entries(index_path)
    expected = {_entry_key(entry): entry for entry in source_entries}
    reports = []
    for shard_index in range(num_shards):
        report_path = output_root / "workers" / f"shard_{shard_index:02d}.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing worker report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_status") != "COMPLETED":
            raise RuntimeError(f"Worker is not completed: {report_path}")
        if report.get("source_index_sha256") != _sha256(index_path):
            raise ValueError(f"Worker source index hash mismatch: {report_path}")
        reports.append(report)
    records: dict[str, dict[str, Any]] = {}
    for report in reports:
        for record in report.get("records", []):
            key = _entry_key(record)
            if key in records:
                raise ValueError(f"Duplicate worker record: {key}")
            records[key] = record
    if set(records) != set(expected):
        missing = sorted(set(expected) - set(records))
        extra = sorted(set(records) - set(expected))
        raise ValueError(f"Worker coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    validation = []
    sequences: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for entry in source_entries:
        record = records[_entry_key(entry)]
        sequence_path = output_root / str(entry["path"])
        check = _validate_sequence(
            sequence_path,
            str(entry["variant"]),
            expected_knn_points=_sequence_knn_count(str(entry["variant"])),
        )
        validation.append(check)
        merged = {
            **entry,
            "path": str(Path(str(entry["path"]))),
            "frame_count": int(record["frame_count"]),
            "knn_hand_points": int(record["knn_hand_points"]),
        }
        sequences[str(entry["split"])].append(merged)
    for split in sequences:
        sequences[split].sort(key=lambda item: str(item["id"]))
    assignment = {
        "schema_name": "ref2dex_object_interaction_cm_v1_3_assignment",
        "work_version": work_version,
        "source_index": str(index_path.resolve()),
        "source_index_sha256": _sha256(index_path),
        "split_preserved": True,
        "sequence_count": len(source_entries),
        "sequences": sequences,
    }
    _write_json(output_root / "assignment.json", assignment)
    counts = {
        split: {
            "grab": sum(item["source"] == "grab" for item in values),
            "inspire_f1": sum(item["source"] == "inspire_f1" for item in values),
            "frames": sum(int(item["frame_count"]) for item in values),
        }
        for split, values in sequences.items()
    }
    index = {
        "schema_name": INDEX_SCHEMA,
        "schema_version": "1.2.0",
        "created_at": _now(),
        "experiment_schema": CACHE_SCHEMA,
        "source_index": str(index_path.resolve()),
        "source_index_sha256": _sha256(index_path),
        "source_probability": source_payload.get("source_probability", {"grab": 0.5, "inspire_f1": 0.5}),
        "stride_policy": source_payload.get(
            "stride_policy",
            {"grab": list(range(1, 11)), "inspire_f1": list(range(1, 11))},
        ),
        "object_pool_points": OBJECT_POOL_POINTS,
        "model_object_points": int(source_payload.get("model_object_points", 1024)),
        "decoder_hand_points_per_stream": DECODER_HAND_POINTS,
        "knn_hand_points_per_stream": {"mano": MANO_KNN_POINTS, "inspire_f1": INSPIRE_KNN_POINTS},
        "knn_k": KNN_K,
        "knn_index_dtype": "uint16",
        "interaction_radius_m": INTERACTION_RADIUS_M,
        "hand_supervision_radius_m": HAND_SUPERVISION_RADIUS_M,
        "distance_storage": "runtime_recompute_from_cached_32_indices",
        "coordinate_frame": "object_pose_t",
        "hand_side": "right",
        "split_policy": "exactly_preserved_from_v1_2_5",
        "assignment_manifest": "assignment.json",
        "source_roots": source_payload.get("source_roots", {}),
        "sequences": sequences,
        "counts": counts,
    }
    _write_json(output_root / "index.json", index)
    _write_json(output_root / "validation_summary.json", validation)
    commit, dirty = _git_info()
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "object_interaction_cm_v1_3_offline_knn_cache",
        "run_id": f"oicm-dexplore-rl-v1-3-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "run_status": "COMPLETED",
        "created_at": _now(),
        "work_version": work_version,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "source_index": str(index_path.resolve()),
        "source_index_sha256": _sha256(index_path),
        "parameters": {
            "object_pool_points": OBJECT_POOL_POINTS,
            "decoder_hand_points": DECODER_HAND_POINTS,
            "knn_hand_points": {"mano": MANO_KNN_POINTS, "inspire_f1": INSPIRE_KNN_POINTS},
            "knn_k": KNN_K,
            "interaction_radius_m": INTERACTION_RADIUS_M,
            "hand_supervision_radius_m": HAND_SUPERVISION_RADIUS_M,
            "surface_seed": SURFACE_SEED,
            "num_shards": num_shards,
        },
        "worker_reports": [str(output_root / "workers" / f"shard_{i:02d}.json") for i in range(num_shards)],
        "counts": {
            "sequences": len(source_entries),
            "frames": sum(value["frames"] for value in counts.values()),
            "by_split": counts,
        },
        "outputs": {
            "index": str(output_root / "index.json"),
            "assignment": str(output_root / "assignment.json"),
            "validation_summary": str(output_root / "validation_summary.json"),
        },
        "conclusion": "SUPPORTED",
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest["counts"], ensure_ascii=False, indent=2))
    return 0


def _pilot(
    *,
    index_path: Path,
    output_root: Path,
    sequence_id: str,
    variant: str | None,
    device_name: str,
    surface_seed: int,
    geometry_batch_size: int,
    knn_batch_size: int,
    work_version: str,
) -> int:
    entries = _source_entries(index_path)
    matches = [entry for entry in entries if str(entry["id"]) == sequence_id]
    if variant is not None:
        matches = [entry for entry in matches if str(entry["variant"]) == variant]
    if len(matches) != 1:
        raise ValueError(f"Expected one pilot entry for {sequence_id!r}/{variant!r}, got {len(matches)}")
    mano_model_dir = REPO_ROOT / "dataset/arctic/data/body_models/mano"
    urdf_path = Path("/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf")
    mano_pool, _ = _build_mano_pool(mano_model_dir, 0.20)
    inspire_pool, _ = _build_inspire_pool(urdf_path, 0.20)
    inspire_model = InspireUrdfModel(urdf_path)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output_root.mkdir(parents=True, exist_ok=True)
    record = _process_sequence(
        matches[0],
        source_root=index_path.parent.resolve(),
        output_root=output_root,
        mano_pool=mano_pool,
        inspire_pool=inspire_pool,
        inspire_model=inspire_model,
        surface_seed=surface_seed,
        device=device,
        geometry_batch_size=geometry_batch_size,
        knn_batch_size=knn_batch_size,
        resume=False,
    )
    validation = record["validation"]
    commit, dirty = _git_info()
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "object_interaction_cm_v1_3_offline_knn_pilot",
        "run_id": output_root.name,
        "run_status": "COMPLETED",
        "created_at": _now(),
        "work_version": work_version,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "source_index": str(index_path.resolve()),
        "sequence": record,
        "parameters": {
            "knn_k": KNN_K,
            "interaction_radius_m": INTERACTION_RADIUS_M,
            "hand_supervision_radius_m": HAND_SUPERVISION_RADIUS_M,
            "surface_seed": surface_seed,
        },
        "validation": validation,
        "conclusion": "SUPPORTED",
    }
    _write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps({"output": str(output_root), "validation": validation}, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "worker", "finalize"), required=True)
    parser.add_argument(
        "--source-index",
        default="data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence", default=None, help="Pilot sequence id")
    parser.add_argument("--variant", choices=("mano", "inspire_rl"), default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--surface-seed", type=int, default=SURFACE_SEED)
    parser.add_argument("--geometry-batch-size", type=int, default=DEFAULT_GEOMETRY_BATCH)
    parser.add_argument("--knn-batch-size", type=int, default=DEFAULT_KNN_BATCH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--work-version", default="V1.3")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.surface_seed != SURFACE_SEED:
        raise ValueError(f"V1.3 sampling is pinned to surface_seed={SURFACE_SEED}")
    if args.geometry_batch_size <= 0 or args.knn_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    index_path = Path(args.source_index).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if args.mode == "pilot":
        if not args.sequence:
            raise ValueError("--sequence is required in pilot mode")
        return _pilot(
            index_path=index_path,
            output_root=output_root,
            sequence_id=args.sequence,
            variant=args.variant,
            device_name=args.device,
            surface_seed=args.surface_seed,
            geometry_batch_size=args.geometry_batch_size,
            knn_batch_size=args.knn_batch_size,
            work_version=args.work_version,
        )
    if args.mode == "worker":
        return _worker(
            index_path=index_path,
            output_root=output_root,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            device_name=args.device,
            surface_seed=args.surface_seed,
            geometry_batch_size=args.geometry_batch_size,
            knn_batch_size=args.knn_batch_size,
            resume=args.resume,
            work_version=args.work_version,
        )
    return _finalize(
        index_path=index_path,
        output_root=output_root,
        num_shards=args.num_shards,
        work_version=args.work_version,
    )


if __name__ == "__main__":
    raise SystemExit(main())
