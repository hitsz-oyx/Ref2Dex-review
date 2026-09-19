#!/usr/bin/env python3
"""Build the bilateral V1.4 MANO cache from the Stage-4 NPZ streams.

The decoder-compatible stream keeps 1538 face centres per side (3076 total).
The independent offline-KNN stream samples 2048 points per MANO mesh side
(4096 total) with fixed triangle/barycentric correspondence and seed 2024.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

for _name, _value in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CACHE_SCHEMA = "ref2dex_object_interaction_cm_bilateral_mano_v1_4"
INDEX_SCHEMA = "ref2dex_object_interaction_cm_index_v1_2"
OBJECT_POINTS = 4096
DECODER_POINTS_PER_SIDE = 1538
KNN_POINTS_PER_SIDE = 2048
MERGED_DECODER_POINTS = DECODER_POINTS_PER_SIDE * 2
MERGED_KNN_POINTS = KNN_POINTS_PER_SIDE * 2
KNN_K = 32
RADIUS_M = 0.02
SURFACE_SEED = 2024
MODIFICATION_VERSION = "V1.4.21"


@dataclass(frozen=True)
class Entry:
    dataset: str
    sequence_id: str
    relative_id: str
    split: str
    source_path: Path
    output_path: Path


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _scalar(array: np.ndarray) -> Any:
    return np.asarray(array).item()


def _entries(
    split_index: Path,
    grab_root: Path,
    arctic_root: Path,
    output_root: Path,
) -> list[Entry]:
    payload = json.loads(split_index.read_text(encoding="utf-8"))
    result: list[Entry] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        for item in payload.get("sequences", {}).get(split, []):
            dataset = str(item.get("dataset", ""))
            if dataset not in {"grab", "arctic"}:
                continue
            sequence_id = str(item["id"])
            prefix = dataset + "/"
            if not sequence_id.startswith(prefix):
                raise ValueError(f"Malformed {dataset} sequence id: {sequence_id}")
            if sequence_id in seen:
                raise ValueError(f"Duplicate source sequence: {sequence_id}")
            seen.add(sequence_id)
            relative_id = sequence_id[len(prefix):]
            source_root = grab_root if dataset == "grab" else arctic_root
            source_path = source_root / relative_id
            if not all((source_path / name).is_file() for name in ("shared.npz", "left.npz", "right.npz")):
                raise FileNotFoundError(f"Incomplete bilateral source: {source_path}")
            result.append(Entry(
                dataset=dataset,
                sequence_id=sequence_id,
                relative_id=relative_id,
                split=split,
                source_path=source_path,
                output_path=output_root / "sequences" / split / "mano" / dataset / relative_id,
            ))
    expected_grab = sum(entry.dataset == "grab" for entry in result)
    expected_arctic = sum(entry.dataset == "arctic" for entry in result)
    if expected_grab != 1335 or expected_arctic != 301:
        raise ValueError(
            f"Expected complete GRAB/ARCTIC assignment 1335/301, got {expected_grab}/{expected_arctic}"
        )
    return result


def _load_mano_reference(model_dir: Path, side: str) -> tuple[np.ndarray, np.ndarray]:
    from smplx import MANO

    model = MANO(
        str(model_dir), is_rhand=side == "right", use_pca=True,
        num_pca_comps=24, flat_hand_mean=True,
    ).to("cpu")
    with torch.inference_mode():
        vertices = model().vertices[0].detach().cpu().numpy().astype(np.float64)
    return vertices, np.asarray(model.faces, dtype=np.int64)


def _surface_correspondence(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    count: int = KNN_POINTS_PER_SIDE,
    seed: int = SURFACE_SEED,
) -> dict[str, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = np.flatnonzero(areas > 1e-12)
    if not len(valid):
        raise ValueError("MANO reference mesh has no positive-area triangle")
    probabilities = areas[valid] / areas[valid].sum()
    rng = np.random.default_rng(int(seed))
    face_ids = valid[rng.choice(len(valid), size=int(count), replace=True, p=probabilities)]
    u = np.sqrt(rng.random(int(count)))
    v = rng.random(int(count))
    barycentric = np.stack((1.0 - u, u * (1.0 - v), u * v), axis=1).astype(np.float32)
    return {"face_ids": face_ids.astype(np.int64), "barycentric": barycentric}


def _correspondences(model_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        vertices, faces = _load_mano_reference(model_dir, side)
        corr = _surface_correspondence(vertices, faces)
        corr["reference_faces"] = faces
        result[side] = corr
    return result


def _correspondence_hash(corr: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(corr["face_ids"], dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(corr["barycentric"], dtype=np.float32).tobytes())
    return digest.hexdigest()


def _sample_surface_chunk(
    vertices: np.ndarray,
    faces: np.ndarray,
    correspondence: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    face_ids = np.asarray(correspondence["face_ids"], dtype=np.int64)
    barycentric = np.asarray(correspondence["barycentric"], dtype=np.float32)
    triangles = np.asarray(vertices, dtype=np.float32)[:, np.asarray(faces, dtype=np.int64)[face_ids]]
    points = np.einsum("pj,bpjd->bpd", barycentric, triangles, optimize=True).astype(np.float32)
    cross = np.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0])
    normals = cross / np.clip(np.linalg.norm(cross, axis=-1, keepdims=True), 1e-12, None)
    return points, normals.astype(np.float32)


class ArcticManoProvider:
    def __init__(self, model_dir: Path, device: torch.device, batch_size: int) -> None:
        from smplx import MANO

        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.models = {
            "left": MANO(str(model_dir), is_rhand=False, use_pca=False, flat_hand_mean=False).to(device),
            "right": MANO(str(model_dir), is_rhand=True, use_pca=False, flat_hand_mean=False).to(device),
        }

    def vertices(
        self,
        raw_path: Path,
        raw_frame_ids: np.ndarray,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        mano_data = np.load(raw_path, allow_pickle=True).item()[side]
        frames = np.asarray(raw_frame_ids, dtype=np.int64)
        rot = np.asarray(mano_data["rot"], dtype=np.float32)[frames]
        pose = np.asarray(mano_data["pose"], dtype=np.float32)[frames]
        trans = np.asarray(mano_data["trans"], dtype=np.float32)[frames]
        shape = np.asarray(mano_data["shape"], dtype=np.float32)
        result: list[np.ndarray] = []
        model = self.models[side]
        with torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                stop = min(len(frames), start + self.batch_size)
                count = stop - start
                betas = shape
                if betas.ndim == 1:
                    betas = np.broadcast_to(betas[None], (count, len(betas)))
                else:
                    betas = betas[frames[start:stop]] if len(betas) > 1 else np.broadcast_to(betas, (count, betas.shape[-1]))
                output = model(
                    global_orient=torch.as_tensor(rot[start:stop], device=self.device),
                    hand_pose=torch.as_tensor(pose[start:stop], device=self.device),
                    betas=torch.as_tensor(np.asarray(betas, dtype=np.float32), device=self.device),
                    transl=torch.as_tensor(trans[start:stop], device=self.device),
                )
                result.append(output.vertices.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(result, axis=0), np.asarray(model.faces, dtype=np.int64)


def _resolve_arctic_raw(shared: Any, arctic_meta: dict[str, Any]) -> Path:
    source_root = Path(str(arctic_meta["source_root"]))
    relative = Path(str(_scalar(shared["source_raw_file"])))
    path = relative if relative.is_absolute() else source_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"ARCTIC MANO parameters unavailable: {path}")
    return path


def _write_arctic_object_fields(
    geometry: Path,
    shared: Any,
    arctic_meta: dict[str, Any],
    raw_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree
    from process.ARCTIC.raw import (
        axis_angle_to_rotmat,
        build_SE3,
        load_object_mesh,
        sample_mesh_surface,
        transform_object_points_batch,
    )

    raw_frame_ids = np.asarray(shared["raw_frame_id"], dtype=np.int64)
    object_params = np.asarray(np.load(str(raw_path).replace(".mano.npy", ".object.npy")), dtype=np.float32)
    selected = object_params[raw_frame_ids]
    object_name = str(_scalar(shared["object_name"]))
    mesh, parts = load_object_mesh(object_name, unit=str(arctic_meta.get("obj_unit", "mm")))
    canonical, _, _ = sample_mesh_surface(mesh, OBJECT_POINTS, seed=42)
    sample_parts: np.ndarray | None
    if parts is None:
        sample_parts = None
        part_ids = np.zeros((OBJECT_POINTS,), dtype=np.int32)
    else:
        sample_parts = np.asarray(parts[cKDTree(mesh.vertices).query(canonical, k=1)[1]], dtype=bool)
        part_ids = sample_parts.astype(np.int32)
    first_points, _ = transform_object_points_batch(
        np.asarray(canonical, dtype=np.float32),
        np.zeros_like(canonical, dtype=np.float32),
        sample_parts,
        selected[:1, 0],
        selected[:1, 1:4],
        selected[:1, 4:7] / 1000.0,
        device,
    )
    cached_first = np.asarray(shared["obj_points_world"][0], dtype=np.float32)
    max_error = float(np.max(np.abs(first_points[0] - cached_first)))
    if max_error > 2e-5:
        raise ValueError(f"ARCTIC object correspondence mismatch: {max_error:.8f} m")
    rotations = axis_angle_to_rotmat(torch.as_tensor(selected[:, 1:4], dtype=torch.float32))
    root_pose = build_SE3(rotations, torch.as_tensor(selected[:, 4:7] / 1000.0)).numpy().astype(np.float32)
    np.save(geometry / "obj_point_id.npy", np.asarray(shared["obj_point_id"], dtype=np.int32))
    np.save(geometry / "obj_part_id.npy", part_ids)
    np.save(geometry / "obj_articulation.npy", selected[:, 0:1].astype(np.float32))
    np.save(geometry / "obj_root_pose_world.npy", root_pose)
    return {
        "object_representation": "articulated_world_points_identity_reference_pose",
        "object_part_count": int(len(np.unique(part_ids))),
        "object_correspondence_max_error_m": max_error,
        "object_raw_parameter_file": str(Path(str(raw_path).replace(".mano.npy", ".object.npy")).resolve()),
    }


def _build_knn(
    geometry: Path,
    *,
    device: torch.device,
    frame_batch_size: int,
    object_chunk: int,
) -> None:
    obj = np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r")
    hand = np.load(geometry / "knn_hand_points_world.npy", mmap_mode="r")
    decoder = np.load(geometry / "hand_points_world.npy", mmap_mode="r")
    frame_count = len(obj)
    indices = np.lib.format.open_memmap(
        geometry / "obj_knn_indices.npy", mode="w+", dtype=np.uint16,
        shape=(frame_count, OBJECT_POINTS, KNN_K),
    )
    candidate = np.lib.format.open_memmap(
        geometry / "obj_candidate_mask_2cm.npy", mode="w+", dtype=np.bool_,
        shape=(frame_count, OBJECT_POINTS),
    )
    supervision = np.lib.format.open_memmap(
        geometry / "hand_supervision_mask_2cm.npy", mode="w+", dtype=np.bool_,
        shape=(frame_count, MERGED_DECODER_POINTS),
    )
    minimum = np.lib.format.open_memmap(
        geometry / "hand_min_object_distance_m.npy", mode="w+", dtype=np.float32,
        shape=(frame_count,),
    )
    batch_size = max(1, int(frame_batch_size))
    chunk_size = max(KNN_K, int(object_chunk))
    try:
        with torch.inference_mode():
            for frame_start in range(0, frame_count, batch_size):
                frame_stop = min(frame_count, frame_start + batch_size)
                hand_tensor = torch.as_tensor(
                    np.array(hand[frame_start:frame_stop], dtype=np.float32, copy=True), device=device
                )
                object_tensor = torch.as_tensor(
                    np.array(obj[frame_start:frame_stop], dtype=np.float32, copy=True), device=device
                )
                for start in range(0, OBJECT_POINTS, chunk_size):
                    stop = min(OBJECT_POINTS, start + chunk_size)
                    distance = torch.cdist(object_tensor[:, start:stop], hand_tensor)
                    values, ids = torch.topk(distance, k=KNN_K, dim=-1, largest=False, sorted=True)
                    indices[frame_start:frame_stop, start:stop] = ids.cpu().numpy().astype(np.uint16)
                    candidate[frame_start:frame_stop, start:stop] = (values[..., 0] <= RADIUS_M).cpu().numpy()
                decoder_tensor = torch.as_tensor(
                    np.array(decoder[frame_start:frame_stop], dtype=np.float32, copy=True), device=device
                )
                frame_min = torch.full((frame_stop - frame_start,), float("inf"), device=device)
                for start in range(0, MERGED_DECODER_POINTS, chunk_size):
                    stop = min(MERGED_DECODER_POINTS, start + chunk_size)
                    distance = torch.cdist(decoder_tensor[:, start:stop], object_tensor).amin(dim=-1)
                    supervision[frame_start:frame_stop, start:stop] = (distance <= RADIUS_M).cpu().numpy()
                    frame_min = torch.minimum(frame_min, distance.amin(dim=-1))
                minimum[frame_start:frame_stop] = frame_min.cpu().numpy().astype(np.float32)
    finally:
        for value in (indices, candidate, supervision, minimum):
            value.flush()
        del indices, candidate, supervision, minimum


def _write_hand_streams(
    geometry: Path,
    entry: Entry,
    shared: Any,
    correspondences: dict[str, dict[str, np.ndarray]],
    *,
    arctic_provider: ArcticManoProvider | None,
    arctic_meta: dict[str, Any],
    device: torch.device,
    geometry_batch_size: int,
) -> tuple[int, dict[str, Any]]:
    frame_count = int(np.asarray(shared["obj_points_world"]).shape[0])
    decoder_points = np.lib.format.open_memmap(
        geometry / "hand_points_world.npy", mode="w+", dtype=np.float32,
        shape=(frame_count, MERGED_DECODER_POINTS, 3),
    )
    decoder_normals = np.lib.format.open_memmap(
        geometry / "hand_normals_world.npy", mode="w+", dtype=np.float32,
        shape=(frame_count, MERGED_DECODER_POINTS, 3),
    )
    knn_points = np.lib.format.open_memmap(
        geometry / "knn_hand_points_world.npy", mode="w+", dtype=np.float32,
        shape=(frame_count, MERGED_KNN_POINTS, 3),
    )
    knn_normals = np.lib.format.open_memmap(
        geometry / "knn_hand_normals_world.npy", mode="w+", dtype=np.float32,
        shape=(frame_count, MERGED_KNN_POINTS, 3),
    )
    candidate_5cm = np.zeros((frame_count, OBJECT_POINTS), dtype=bool)
    raw_path: Path | None = None
    if entry.dataset == "arctic":
        raw_path = _resolve_arctic_raw(shared, arctic_meta)
        if arctic_provider is None:
            raise RuntimeError("ARCTIC MANO provider was not initialized")
    try:
        for side_index, side in enumerate(("left", "right")):
            with np.load(entry.source_path / f"{side}.npz", allow_pickle=False) as side_data:
                points = np.asarray(side_data["hand_points_world"], dtype=np.float32)
                normals = np.asarray(side_data["hand_normals_world"], dtype=np.float32)
                if points.shape != (frame_count, DECODER_POINTS_PER_SIDE, 3) or normals.shape != points.shape:
                    raise ValueError(f"{entry.source_path}/{side}.npz has invalid decoder geometry")
                d0 = side_index * DECODER_POINTS_PER_SIDE
                d1 = d0 + DECODER_POINTS_PER_SIDE
                decoder_points[:, d0:d1] = points
                decoder_normals[:, d0:d1] = normals
                candidate_5cm |= np.asarray(side_data["obj_candidate_mask_5cm"], dtype=bool)
                if entry.dataset == "grab":
                    vertices = np.asarray(side_data["hand_mesh_vertices_world"], dtype=np.float32)
                    faces = np.asarray(side_data["hand_mesh_faces"], dtype=np.int64)
                else:
                    assert raw_path is not None and arctic_provider is not None
                    vertices, faces = arctic_provider.vertices(
                        raw_path, np.asarray(shared["raw_frame_id"], dtype=np.int64), side
                    )
            reference_faces = np.asarray(correspondences[side]["reference_faces"], dtype=np.int64)
            if faces.shape != reference_faces.shape or not np.array_equal(faces, reference_faces):
                raise ValueError(f"{entry.sequence_id}/{side}: MANO topology differs from reference asset")
            k0 = side_index * KNN_POINTS_PER_SIDE
            k1 = k0 + KNN_POINTS_PER_SIDE
            for start in range(0, frame_count, max(1, int(geometry_batch_size))):
                stop = min(frame_count, start + max(1, int(geometry_batch_size)))
                sampled_points, sampled_normals = _sample_surface_chunk(
                    vertices[start:stop], faces, correspondences[side]
                )
                knn_points[start:stop, k0:k1] = sampled_points
                knn_normals[start:stop, k0:k1] = sampled_normals
        np.save(geometry / "obj_candidate_mask_5cm.npy", candidate_5cm)
    finally:
        for value in (decoder_points, decoder_normals, knn_points, knn_normals):
            value.flush()
        del decoder_points, decoder_normals, knn_points, knn_normals
    return frame_count, {"arctic_raw_mano": str(raw_path.resolve()) if raw_path else None}


def _process_entry(
    entry: Entry,
    *,
    correspondences: dict[str, dict[str, np.ndarray]],
    arctic_provider: ArcticManoProvider | None,
    arctic_meta: dict[str, Any],
    device: torch.device,
    geometry_batch_size: int,
    knn_batch_size: int,
    object_chunk: int,
    resume: bool,
) -> dict[str, Any]:
    final_geometry = entry.output_path / "geometry"
    if (final_geometry / "manifest.json").is_file():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {entry.output_path}")
        return _validate_sequence(entry.output_path)
    partial = entry.output_path.with_name(entry.output_path.name + ".partial")
    if entry.output_path.exists() or partial.exists():
        raise FileExistsError(f"Existing incomplete output requires explicit cleanup: {entry.output_path}")
    geometry = partial / "geometry"
    geometry.mkdir(parents=True)
    started = time.time()
    try:
        with np.load(entry.source_path / "shared.npz", allow_pickle=False) as shared:
            obj = np.asarray(shared["obj_points_world"], dtype=np.float32)
            obj_normals = np.asarray(shared["obj_normals_world"], dtype=np.float32)
            raw_frames = np.asarray(shared["raw_frame_id"], dtype=np.int32)
            if obj.ndim != 3 or obj.shape[1:] != (OBJECT_POINTS, 3) or obj_normals.shape != obj.shape:
                raise ValueError(f"{entry.source_path}: invalid object arrays")
            np.save(geometry / "obj_points_pool_world.npy", obj)
            np.save(geometry / "obj_normals_pool_world.npy", obj_normals)
            np.save(geometry / "source_frame_id.npy", raw_frames)
            source_fps = float(_scalar(shared["source_fps"]))
            np.save(geometry / "frame_time.npy", raw_frames.astype(np.float32) / source_fps)
            if "obj_pose_world" in shared.files:
                pose = np.asarray(shared["obj_pose_world"], dtype=np.float32)
            else:
                pose = np.broadcast_to(np.eye(4, dtype=np.float32), (len(obj), 4, 4)).copy()
            np.save(geometry / "obj_pose_world.npy", pose)
            frame_count, provenance = _write_hand_streams(
                geometry, entry, shared, correspondences,
                arctic_provider=arctic_provider, arctic_meta=arctic_meta,
                device=device, geometry_batch_size=geometry_batch_size,
            )
            if entry.dataset == "arctic":
                assert provenance["arctic_raw_mano"] is not None
                object_meta = _write_arctic_object_fields(
                    geometry, shared, arctic_meta, Path(provenance["arctic_raw_mano"]), device
                )
            else:
                np.save(geometry / "obj_point_id.npy", np.asarray(shared["obj_point_id"], dtype=np.int32))
                object_meta = {"object_representation": "rigid_se3"}
        _build_knn(
            geometry, device=device, frame_batch_size=knn_batch_size, object_chunk=object_chunk
        )
        manifest = {
            "schema_name": CACHE_SCHEMA,
            "schema_version": "1.0.0",
            "work_version": MODIFICATION_VERSION,
            "sequence_id": entry.sequence_id,
            "dataset": entry.dataset,
            "source": "mano",
            "split": entry.split,
            "source_path": str(entry.source_path.resolve()),
            "coordinate_frame": "world" if entry.dataset == "arctic" else "object_pose_t",
            "hand_side": "bilateral_merged_left_then_right",
            "frame_count": frame_count,
            "object_pool_points": OBJECT_POINTS,
            "decoder_hand_points": MERGED_DECODER_POINTS,
            "decoder_points_per_side": DECODER_POINTS_PER_SIDE,
            "knn_hand_points": MERGED_KNN_POINTS,
            "knn_points_per_side": KNN_POINTS_PER_SIDE,
            "knn_k": KNN_K,
            "knn_index_dtype": "uint16",
            "interaction_radius_m": RADIUS_M,
            "hand_supervision_radius_m": RADIUS_M,
            "effective_fps": 30.0,
            "surface_sampling": {
                "method": "global_surface_area_uniform_triangle_barycentric",
                "seed": SURFACE_SEED,
                "cross_frame_fixed": True,
                "left_sha256": _correspondence_hash(correspondences["left"]),
                "right_sha256": _correspondence_hash(correspondences["right"]),
            },
            **object_meta,
            **provenance,
        }
        _write_json(geometry / "manifest.json", manifest)
        entry.output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, entry.output_path)
        result = _validate_sequence(entry.output_path)
        result["elapsed_s"] = time.time() - started
        return result
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def _finite(path: Path, chunk: int = 32) -> bool:
    value = np.load(path, mmap_mode="r")
    for start in range(0, len(value), chunk):
        if not np.isfinite(np.asarray(value[start:start + chunk])).all():
            return False
    return True


def _array_max(path: Path, chunk: int = 32) -> int:
    value = np.load(path, mmap_mode="r")
    maximum = 0
    for start in range(0, len(value), chunk):
        maximum = max(maximum, int(np.asarray(value[start:start + chunk]).max()))
    return maximum


def _active_frame_count(path: Path, chunk: int = 256) -> int:
    value = np.load(path, mmap_mode="r")
    total = 0
    for start in range(0, len(value), chunk):
        total += int(np.asarray(value[start:start + chunk], dtype=bool).any(axis=1).sum())
    return total


def _validate_sequence(sequence: Path) -> dict[str, Any]:
    geometry = sequence / "geometry"
    manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8"))
    frame_count = int(manifest["frame_count"])
    expected = {
        "obj_points_pool_world.npy": ((frame_count, OBJECT_POINTS, 3), np.dtype(np.float32)),
        "obj_normals_pool_world.npy": ((frame_count, OBJECT_POINTS, 3), np.dtype(np.float32)),
        "obj_pose_world.npy": ((frame_count, 4, 4), np.dtype(np.float32)),
        "source_frame_id.npy": ((frame_count,), np.dtype(np.int32)),
        "frame_time.npy": ((frame_count,), np.dtype(np.float32)),
        "hand_points_world.npy": ((frame_count, MERGED_DECODER_POINTS, 3), np.dtype(np.float32)),
        "hand_normals_world.npy": ((frame_count, MERGED_DECODER_POINTS, 3), np.dtype(np.float32)),
        "knn_hand_points_world.npy": ((frame_count, MERGED_KNN_POINTS, 3), np.dtype(np.float32)),
        "knn_hand_normals_world.npy": ((frame_count, MERGED_KNN_POINTS, 3), np.dtype(np.float32)),
        "obj_knn_indices.npy": ((frame_count, OBJECT_POINTS, KNN_K), np.dtype(np.uint16)),
        "obj_candidate_mask_2cm.npy": ((frame_count, OBJECT_POINTS), np.dtype(np.bool_)),
        "hand_supervision_mask_2cm.npy": ((frame_count, MERGED_DECODER_POINTS), np.dtype(np.bool_)),
        "hand_min_object_distance_m.npy": ((frame_count,), np.dtype(np.float32)),
        "obj_candidate_mask_5cm.npy": ((frame_count, OBJECT_POINTS), np.dtype(np.bool_)),
        "obj_point_id.npy": ((OBJECT_POINTS,), np.dtype(np.int32)),
    }
    for name, (shape, dtype) in expected.items():
        path = geometry / name
        if not path.is_file():
            raise FileNotFoundError(path)
        value = np.load(path, mmap_mode="r")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{path}: expected {shape}/{dtype}, got {value.shape}/{value.dtype}")
    index_max = _array_max(geometry / "obj_knn_indices.npy")
    if index_max >= MERGED_KNN_POINTS:
        raise ValueError(f"{geometry}: out-of-range KNN index")
    finite_names = (
        "obj_points_pool_world.npy", "obj_normals_pool_world.npy", "obj_pose_world.npy",
        "frame_time.npy", "hand_points_world.npy", "hand_normals_world.npy",
        "knn_hand_points_world.npy", "knn_hand_normals_world.npy", "hand_min_object_distance_m.npy",
    )
    if not all(_finite(geometry / name) for name in finite_names):
        raise ValueError(f"{geometry}: non-finite array")
    if manifest["dataset"] == "arctic":
        for name, shape in {
            "obj_part_id.npy": (OBJECT_POINTS,),
            "obj_articulation.npy": (frame_count, 1),
            "obj_root_pose_world.npy": (frame_count, 4, 4),
        }.items():
            value = np.load(geometry / name, mmap_mode="r")
            if value.shape != shape:
                raise ValueError(f"{geometry / name}: expected {shape}, got {value.shape}")
    return {
        "sequence_id": manifest["sequence_id"],
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "path": str(sequence.resolve()),
        "frames": frame_count,
        "knn_index_max": index_max,
        "active_frames": _active_frame_count(geometry / "obj_candidate_mask_2cm.npy"),
    }


def _worker(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    entries = _entries(
        args.split_index.resolve(), args.grab_root.resolve(), args.arctic_root.resolve(), output_root
    )
    if args.sequence:
        selected = set(args.sequence)
        entries = [entry for entry in entries if entry.sequence_id in selected]
        missing = selected - {entry.sequence_id for entry in entries}
        if missing:
            raise ValueError(f"Unknown requested sequences: {sorted(missing)}")
    entries = [entry for index, entry in enumerate(entries) if index % args.num_shards == args.shard_index]
    if args.limit:
        entries = entries[:args.limit]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"A CUDA device is required, got {args.device}")
    arctic_meta = json.loads((args.arctic_root / "meta.json").read_text(encoding="utf-8"))
    model_dir = Path(str(arctic_meta["mano_path"])).resolve()
    correspondences = _correspondences(model_dir)
    needs_arctic = any(entry.dataset == "arctic" for entry in entries)
    arctic_provider = ArcticManoProvider(model_dir, device, args.mano_batch_size) if needs_arctic else None
    commit, dirty = _git_info()
    started_at = _now()
    initial_report = {
        "schema_name": "ref2dex_cache_worker_report_v1",
        "run_id": args.run_id,
        "run_status": "STARTED",
        "work_version": MODIFICATION_VERSION,
        "started_at": started_at,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "device": str(device),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_sequences": len(entries),
        "completed_sequences": 0,
        "completed_frames": 0,
        "failures": [],
        "results": [],
    }
    _write_json(output_root / "workers" / f"shard_{args.shard_index:02d}.json", initial_report)
    if args.shard_index == 0 and not (output_root / "run_manifest.json").exists():
        _write_json(output_root / "run_manifest.json", {
            "schema_name": "ref2dex_data_run_manifest_v1",
            "run_id": args.run_id,
            "run_status": "STARTED",
            "work_version": MODIFICATION_VERSION,
            "operation_category": ["code", "data", "operation"],
            "started_at": started_at,
            "base_commit": commit,
            "worktree_dirty": dirty,
            "command": " ".join(sys.argv),
            "inputs": {
                "split_index": str(args.split_index.resolve()),
                "grab_root": str(args.grab_root.resolve()),
                "arctic_root": str(args.arctic_root.resolve()),
            },
            "outputs": {"root": str(output_root)},
            "expected": {
                "sequences": len(_entries(
                    args.split_index.resolve(), args.grab_root.resolve(),
                    args.arctic_root.resolve(), output_root,
                )),
                "num_shards": args.num_shards,
            },
        })
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, entry in enumerate(entries, 1):
        try:
            result = _process_entry(
                entry, correspondences=correspondences, arctic_provider=arctic_provider,
                arctic_meta=arctic_meta, device=device,
                geometry_batch_size=args.geometry_batch_size,
                knn_batch_size=args.knn_batch_size, object_chunk=args.object_chunk,
                resume=args.resume,
            )
            results.append(result)
            print(json.dumps({"status": "COMPLETED", "index": index, "total": len(entries), **result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failures.append({"sequence_id": entry.sequence_id, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps({"status": "FAILED", "sequence_id": entry.sequence_id, "error": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
            traceback.print_exc()
    report = {
        "schema_name": "ref2dex_cache_worker_report_v1",
        "run_id": args.run_id,
        "run_status": "COMPLETED" if not failures else "FAILED",
        "work_version": MODIFICATION_VERSION,
        "started_at": started_at,
        "finished_at": _now(),
        "base_commit": commit,
        "worktree_dirty": dirty,
        "device": str(device),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_sequences": len(entries),
        "completed_sequences": len(results),
        "completed_frames": sum(int(value["frames"]) for value in results),
        "failures": failures,
        "results": results,
    }
    _write_json(output_root / "workers" / f"shard_{args.shard_index:02d}.json", report)
    return 1 if failures else 0


def _write_correspondence_asset(output_root: Path, correspondences: dict[str, dict[str, np.ndarray]]) -> Path:
    path = output_root / "mano_surface_correspondence.npz"
    np.savez_compressed(
        path,
        left_face_ids=correspondences["left"]["face_ids"],
        left_barycentric=correspondences["left"]["barycentric"],
        right_face_ids=correspondences["right"]["face_ids"],
        right_barycentric=correspondences["right"]["barycentric"],
    )
    return path


def _finalize(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    entries = _entries(
        args.split_index.resolve(), args.grab_root.resolve(), args.arctic_root.resolve(), output_root
    )
    if args.sequence:
        selected = set(args.sequence)
        entries = [entry for entry in entries if entry.sequence_id in selected]
    arctic_meta = json.loads((args.arctic_root / "meta.json").read_text(encoding="utf-8"))
    correspondences = _correspondences(Path(str(arctic_meta["mano_path"])).resolve())
    validations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for entry in entries:
        try:
            validations.append(_validate_sequence(entry.output_path))
        except Exception as exc:
            failures.append({"sequence_id": entry.sequence_id, "error": f"{type(exc).__name__}: {exc}"})
    if failures:
        _write_json(output_root / "validation_summary.json", {
            "run_status": "FAILED", "validated": len(validations), "failures": failures,
        })
        raise RuntimeError(f"Cannot finalize: {len(failures)} sequences failed validation")
    correspondence_path = _write_correspondence_asset(output_root, correspondences)
    split_entries: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    by_id = {entry.sequence_id: entry for entry in entries}
    for value in validations:
        entry = by_id[value["sequence_id"]]
        split_entries[entry.split].append({
            "source": "mano",
            "id": entry.sequence_id,
            "path": str(entry.output_path.resolve()),
            "dataset": entry.dataset,
            "split": entry.split,
            "frame_count": int(value["frames"]),
            "variant": "mano_bilateral",
        })
    counts = {split: len(values) for split, values in split_entries.items()}
    frame_counts = {
        split: sum(int(value["frame_count"]) for value in values)
        for split, values in split_entries.items()
    }
    index = {
        "schema_name": INDEX_SCHEMA,
        "schema_version": "1.4.0",
        "work_version": MODIFICATION_VERSION,
        "created_at": _now(),
        "object_pool_points": OBJECT_POINTS,
        "model_object_points": 1024,
        "decoder_hand_points_per_stream": MERGED_DECODER_POINTS,
        "knn_hand_points_per_stream": {"mano": MERGED_KNN_POINTS},
        "max_knn_hand_points": MERGED_KNN_POINTS,
        "knn_k": KNN_K,
        "interaction_radius_m": RADIUS_M,
        "hand_supervision_radius_m": RADIUS_M,
        "source_probability": {"mano": 1.0},
        "split_policy": {
            "grab": "preserved exactly from V1.4 Inspire index",
            "arctic": "all train",
        },
        "counts": counts,
        "frame_counts": frame_counts,
        "sequences": split_entries,
    }
    _write_json(output_root / "index.json", index)
    _write_json(output_root / "validation_summary.json", {
        "run_status": "COMPLETED",
        "validated_sequences": len(validations),
        "validated_frames": sum(int(value["frames"]) for value in validations),
        "counts": counts,
        "frame_counts": frame_counts,
        "failures": [],
        "results": validations,
    })
    commit, dirty = _git_info()
    _write_json(output_root / "run_manifest.json", {
        "schema_name": "ref2dex_data_run_manifest_v1",
        "run_id": args.run_id,
        "run_status": "COMPLETED",
        "work_version": MODIFICATION_VERSION,
        "operation_category": ["code", "data", "operation"],
        "created_at": _now(),
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": " ".join(sys.argv),
        "inputs": {
            "split_index": str(args.split_index.resolve()),
            "grab_root": str(args.grab_root.resolve()),
            "arctic_root": str(args.arctic_root.resolve()),
        },
        "outputs": {
            "root": str(output_root),
            "index": str((output_root / "index.json").resolve()),
            "validation": str((output_root / "validation_summary.json").resolve()),
            "mano_surface_correspondence": str(correspondence_path.resolve()),
        },
        "counts": counts,
        "frame_counts": frame_counts,
        "result": "SUPPORTED",
    })
    print(json.dumps({"status": "COMPLETED", "counts": counts, "frame_counts": frame_counts}, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "finalize"))
    parser.add_argument("--grab-root", type=Path, required=True)
    parser.add_argument("--arctic-root", type=Path, required=True)
    parser.add_argument("--split-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--geometry-batch-size", type=int, default=64)
    parser.add_argument("--mano-batch-size", type=int, default=256)
    parser.add_argument("--knn-batch-size", type=int, default=1)
    parser.add_argument("--object-chunk", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"Invalid shard {args.shard_index}/{args.num_shards}")
    raise SystemExit(_worker(args) if args.mode == "worker" else _finalize(args))


if __name__ == "__main__":
    main()
