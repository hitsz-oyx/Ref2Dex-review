#!/usr/bin/env python3
"""Build the right-hand MANO/RL-Inspire ObjectInteractionCm cache.

The converter intentionally creates a new geometry layout instead of mutating
the existing GRAB or HRDexDB caches.  Dexplore's native object pose and
Inspire qpos are used as the target world frame; existing GRAB geometry is
first expressed in its object frame and then re-projected with that pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


SCHEMA_NAME = "ref2dex_object_interaction_cm_dexplore_rl_v1"
INDEX_SCHEMA_NAME = "ref2dex_object_interaction_cm_index_v1_1"
NUM_OBJECT_POINTS = 4096
NUM_HAND_POINTS = 1538
NATIVE_Q_START = 245 + 32 * 4
NUM_DOFS = 18
NATIVE_TO_URDF = np.asarray(
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9],
    dtype=np.int64,
)


def _vec(text: Optional[str], size: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(size, default, dtype=np.float64)
    values = [float(value) for value in text.split()]
    if len(values) != size:
        raise ValueError(f"Expected {size} values, got {len(values)} in {text!r}")
    return np.asarray(values, dtype=np.float64)


def _origin(element: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform
    transform[:3, 3] = _vec(element.get("xyz"), 3)
    transform[:3, :3] = Rotation.from_euler("xyz", _vec(element.get("rpy"), 3)).as_matrix()
    return transform


def _motion_transform(joint_type: str, axis: np.ndarray, q: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if joint_type == "prismatic":
        transform[:3, 3] = axis * q
    elif joint_type in ("revolute", "continuous"):
        transform[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
    elif joint_type != "fixed":
        raise ValueError(f"Unsupported URDF joint type: {joint_type}")
    return transform


def _resolve_mesh(filename: str, urdf_path: Path) -> Path:
    filename = filename.replace("package://", "").replace("file://", "")
    if filename.startswith("$(find ") and ")/" in filename:
        filename = filename.split(")/", 1)[1]
    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = urdf_path.parent / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"URDF mesh does not exist: {candidate}")
    return candidate


@dataclass
class Joint:
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    q_index: int


@dataclass
class Visual:
    link: str
    vertices: np.ndarray
    faces: np.ndarray
    local_transform: np.ndarray


class InspireUrdfModel:
    """Small deterministic FK/surface loader for the Dexplore Inspire URDF."""

    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = urdf_path.resolve()
        root = ET.parse(self.urdf_path).getroot()
        links = [str(element.get("name")) for element in root.findall("link")]
        if any(name == "None" for name in links):
            raise ValueError(f"Unnamed link in {self.urdf_path}")
        self.link_names = links
        self.joints: list[Joint] = []
        child_links: set[str] = set()
        q_index = 0
        for element in root.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None or parent.get("link") is None or child.get("link") is None:
                raise ValueError(f"Malformed joint in {self.urdf_path}")
            joint_type = element.get("type", "fixed")
            axis_element = element.find("axis")
            axis = _vec(axis_element.get("xyz") if axis_element is not None else None, 3)
            norm = float(np.linalg.norm(axis))
            if norm > 0:
                axis = axis / norm
            current_index = q_index if joint_type != "fixed" else -1
            if joint_type != "fixed":
                q_index += 1
            parent_name = str(parent.get("link"))
            child_name = str(child.get("link"))
            self.joints.append(Joint(
                parent=parent_name,
                child=child_name,
                joint_type=joint_type,
                origin=_origin(element.find("origin")),
                axis=axis,
                q_index=current_index,
            ))
            child_links.add(child_name)
        if q_index != NUM_DOFS:
            raise ValueError(f"Expected {NUM_DOFS} actuated joints, got {q_index}")
        self.root_link = next(name for name in self.link_names if name not in child_links)
        self.children: dict[str, list[Joint]] = {}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)
        self.visuals: list[Visual] = []
        for link in root.findall("link"):
            link_name = str(link.get("name"))
            for visual in link.findall("visual"):
                mesh = visual.find("./geometry/mesh")
                if mesh is None or mesh.get("filename") is None:
                    continue
                mesh_path = _resolve_mesh(str(mesh.get("filename")), self.urdf_path)
                loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
                if not isinstance(loaded, trimesh.Trimesh):
                    raise TypeError(f"Expected a mesh at {mesh_path}, got {type(loaded)}")
                vertices = np.asarray(loaded.vertices, dtype=np.float64)
                faces = np.asarray(loaded.faces, dtype=np.int32)
                scale = _vec(mesh.get("scale"), 3, default=1.0)
                vertices *= scale[None, :]
                self.visuals.append(Visual(
                    link=link_name,
                    vertices=vertices,
                    faces=faces,
                    local_transform=_origin(visual.find("origin")),
                ))
        if not self.visuals:
            raise ValueError(f"No visual meshes found in {self.urdf_path}")

    def qpos_to_urdf_order(self, native_qpos: np.ndarray) -> np.ndarray:
        native_qpos = np.asarray(native_qpos, dtype=np.float64)
        if native_qpos.shape != (NUM_DOFS,):
            raise ValueError(f"Expected native qpos ({NUM_DOFS},), got {native_qpos.shape}")
        urdf_qpos = np.empty_like(native_qpos)
        urdf_qpos[NATIVE_TO_URDF] = native_qpos
        return urdf_qpos

    def link_transforms(self, urdf_qpos: np.ndarray) -> dict[str, np.ndarray]:
        urdf_qpos = np.asarray(urdf_qpos, dtype=np.float64)
        if urdf_qpos.shape != (NUM_DOFS,):
            raise ValueError(f"Expected URDF qpos ({NUM_DOFS},), got {urdf_qpos.shape}")
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}

        def visit(parent: str) -> None:
            for joint in self.children.get(parent, []):
                transforms[joint.child] = (
                    transforms[parent]
                    @ joint.origin
                    @ _motion_transform(
                        joint.joint_type,
                        joint.axis,
                        urdf_qpos[joint.q_index] if joint.q_index >= 0 else 0.0,
                    )
                )
                visit(joint.child)

        visit(self.root_link)
        if len(transforms) != len(self.link_names):
            raise ValueError(f"Disconnected URDF links: {sorted(set(self.link_names) - set(transforms))}")
        return transforms

    def surface_samples(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample fixed local surface points and normals, preserving visual-link IDs."""
        triangles: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        link_ids: list[int] = []
        areas: list[np.ndarray] = []
        for visual_id, visual in enumerate(self.visuals):
            # Keep samples in the mesh-local frame.  The visual origin is
            # applied exactly once together with the link FK transform below.
            tri = visual.vertices[visual.faces]
            cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > 1e-12
            triangles.append(tri[valid])
            normal = cross[valid] / np.clip(np.linalg.norm(cross[valid], axis=1, keepdims=True), 1e-12, None)
            normals.append(normal)
            areas.append(area[valid])
            link_ids.extend([visual_id] * int(valid.sum()))
        all_triangles = np.concatenate(triangles, axis=0)
        all_normals = np.concatenate(normals, axis=0)
        all_areas = np.concatenate(areas, axis=0)
        all_links = np.asarray(link_ids, dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        probabilities = all_areas / np.clip(all_areas.sum(), 1e-12, None)
        indices = rng.choice(len(all_triangles), size=int(count), replace=True, p=probabilities)
        tri = all_triangles[indices]
        r1 = np.sqrt(rng.random(int(count)))
        r2 = rng.random(int(count))
        points = tri[:, 0] + r1[:, None] * (tri[:, 1] - tri[:, 0]) + r2[:, None] * (tri[:, 2] - tri[:, 0])
        return points.astype(np.float32), all_normals[indices].astype(np.float32), all_links[indices]


def _stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git_info(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(seq_id: str) -> str:
    return seq_id.replace("/", "_")


def _dex_name(seq_id: str) -> str:
    return _safe_name(seq_id)


def _pose_from_native(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64)).as_matrix().T
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = np.asarray(position, dtype=np.float32)
    return pose


def _to_object_frame(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float32) - pose[:3, 3]) @ pose[:3, :3]


def _from_object_frame(points_local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.asarray(points_local, dtype=np.float32) @ pose[:3, :3].T + pose[:3, 3]


def _normals_to_frame(normals_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    values = np.asarray(normals_world, dtype=np.float32) @ pose[:3, :3]
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8, None)


def _normals_from_frame(normals_local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    values = np.asarray(normals_local, dtype=np.float32) @ pose[:3, :3].T
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8, None)


def _load_tensor(path: Path) -> np.ndarray:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected tensor at {path}, got {type(value)}")
    array = value.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 2 or array.shape[1] < NATIVE_Q_START + NUM_DOFS:
        raise ValueError(f"Invalid Inspire tensor shape at {path}: {array.shape}")
    q = array[:, NATIVE_Q_START:NATIVE_Q_START + NUM_DOFS]
    if not np.isfinite(q).all():
        raise ValueError(f"Non-finite Inspire qpos at {path}")
    return array


def _parent_entries(index_path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != INDEX_SCHEMA_NAME:
        raise ValueError(f"Unsupported parent index schema: {payload.get('schema_name')!r}")
    result: dict[str, list[dict[str, Any]]] = {}
    for split, entries in payload.get("sequences", {}).items():
        selected = []
        for entry in entries:
            if str(entry.get("source")) != "grab":
                continue
            seq_id = str(entry["id"])
            selected.append({"id": seq_id, "split": split})
        result[str(split)] = selected
    return result


def _sequence_metadata(grab_root: Path, seq_id: str) -> dict[str, str]:
    path = grab_root / seq_id / "shared" / "meta.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    seq_name = str(payload.get("seq_name") or seq_id.split("/", 1)[-1])
    return {
        "subject_id": str(payload.get("subject_id") or seq_id.split("/", 1)[0]),
        "seq_name": seq_name,
        "object_name": str(payload.get("object_name") or seq_name.split("_", 1)[0]),
        "action_name": seq_name.split("_", 1)[1] if "_" in seq_name else seq_name,
    }


def _right_mano_available(grab_root: Path, seq_id: str) -> bool:
    """Check the right-hand files without reading their full payload."""
    root = grab_root / seq_id / "right"
    try:
        hand = np.load(root / "hand_points_world.npy", mmap_mode="r")
        normals = np.load(root / "hand_normals_world.npy", mmap_mode="r")
        active = np.load(root / "candidate_active_5cm.npy", mmap_mode="r")
        return hand.shape[1:] == (NUM_HAND_POINTS, 3) and normals.shape == hand.shape and active.ndim == 1
    except (OSError, ValueError, FileNotFoundError):
        return False


def _assign_variants(entries: dict[str, list[dict[str, Any]]], grab_root: Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    assigned: dict[str, list[dict[str, Any]]] = {}
    for split, values in entries.items():
        enriched = []
        for item in values:
            meta = _sequence_metadata(grab_root, item["id"])
            enriched.append({**item, **meta, "mano_available": _right_mano_available(grab_root, item["id"])})
        if split == "test":
            for item in enriched:
                if not item["mano_available"]:
                    raise ValueError(f"Test sequence lacks valid right MANO geometry: {item['id']}")
                item["variant"] = "mano"
            assigned[split] = sorted(enriched, key=lambda x: x["id"])
            continue
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in enriched:
            groups.setdefault((item["object_name"], item["action_name"]), []).append(item)
        counts = {"mano": 0, "inspire_rl": 0}
        for group_key in sorted(groups):
            group = sorted(groups[group_key], key=lambda x: (x["subject_id"], x["id"]))
            start = _stable_int(seed, split, *group_key) % 2
            for index, item in enumerate(group):
                if not item["mano_available"]:
                    item["variant"] = "inspire_rl"
                    counts["inspire_rl"] += 1
                    continue
                preferred = "inspire_rl" if (index + start) % 2 else "mano"
                other = "mano" if preferred == "inspire_rl" else "inspire_rl"
                if counts[preferred] > len(enriched) // 2:
                    preferred = other
                item["variant"] = preferred
                counts[preferred] += 1
        assigned[split] = sorted(enriched, key=lambda x: x["id"])
    return assigned


def _intersection(
    entries: dict[str, list[dict[str, Any]]],
    grab_root: Path,
    dexplore_grab_root: Path,
    rl_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    missing: list[str] = []
    result: dict[str, list[dict[str, Any]]] = {}
    for split, values in entries.items():
        kept = []
        for item in values:
            seq_id = item["id"]
            old_path = grab_root / seq_id
            dex_name = _dex_name(seq_id)
            dexplore_path = dexplore_grab_root / dex_name
            dex_path = rl_root / dex_name / "interaction_hand_inspire.pt"
            if not old_path.is_dir() or not dexplore_path.is_dir() or not dex_path.is_file():
                missing.append(seq_id)
                continue
            kept.append(item)
        result[split] = kept
    return result, missing


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))


def _fk_surface(
    model: InspireUrdfModel,
    q_native: np.ndarray,
    sampled_points: np.ndarray,
    sampled_normals: np.ndarray,
    sampled_visual_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.empty((len(q_native), len(sampled_points), 3), dtype=np.float32)
    normals = np.empty_like(points)
    for frame, native in enumerate(q_native):
        links = model.link_transforms(model.qpos_to_urdf_order(native))
        for visual_id in np.unique(sampled_visual_ids):
            mask = sampled_visual_ids == visual_id
            transform = links[model.visuals[int(visual_id)].link] @ model.visuals[int(visual_id)].local_transform
            points[frame, mask] = sampled_points[mask] @ transform[:3, :3].T + transform[:3, 3]
            normals[frame, mask] = sampled_normals[mask] @ transform[:3, :3].T
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return points, normals


def _convert_sequence(
    item: dict[str, Any],
    *,
    output_root: Path,
    grab_root: Path,
    rl_root: Path,
    model: InspireUrdfModel,
    sampled_points: np.ndarray,
    sampled_normals: np.ndarray,
    sampled_visual_ids: np.ndarray,
    surface_seed: int,
) -> dict[str, Any]:
    seq_id = item["id"]
    old_root = grab_root / seq_id
    old_shared = old_root / "shared"
    old_right = old_root / "right"
    old_obj = np.load(old_shared / "obj_points_world.npy", mmap_mode="r")
    old_obj_normals = np.load(old_shared / "obj_normals_world.npy", mmap_mode="r")
    old_pose = np.load(old_shared / "obj_pose_world.npy", mmap_mode="r")
    old_raw = np.load(old_shared / "raw_frame_id.npy", mmap_mode="r")
    dex_tensor_path = rl_root / _dex_name(seq_id) / "interaction_hand_inspire.pt"
    dex_data = _load_tensor(dex_tensor_path)
    q_native = dex_data[:, NATIVE_Q_START:NATIVE_Q_START + NUM_DOFS]
    if len(old_obj) != len(dex_data) or old_obj.shape[1:] != (NUM_OBJECT_POINTS, 3):
        raise ValueError(f"Frame/object mismatch for {seq_id}: old={old_obj.shape}, dex={dex_data.shape}")
    if len(old_raw) != len(dex_data):
        raise ValueError(f"Frame metadata mismatch for {seq_id}")

    T = len(dex_data)
    pose = np.empty((T, 4, 4), dtype=np.float32)
    object_world = np.empty((T, NUM_OBJECT_POINTS, 3), dtype=np.float32)
    object_normals_world = np.empty_like(object_world)
    hand_world = np.empty((T, NUM_HAND_POINTS, 3), dtype=np.float32)
    hand_normals_world = np.empty_like(hand_world)
    if item["variant"] == "mano":
        if not (old_right / "hand_points_world.npy").is_file() or not (old_right / "hand_normals_world.npy").is_file():
            raise ValueError(f"Missing right MANO hand geometry for {seq_id}")
        old_hand = np.load(old_right / "hand_points_world.npy", mmap_mode="r")
        old_hand_normals = np.load(old_right / "hand_normals_world.npy", mmap_mode="r")
        old_active = np.load(old_right / "candidate_active_5cm.npy", mmap_mode="r").astype(bool)
        if old_hand.shape[1:] != (NUM_HAND_POINTS, 3) or old_hand_normals.shape != old_hand.shape:
            raise ValueError(f"Expected right MANO hand [T,{NUM_HAND_POINTS},3] for {seq_id}")
        if len(old_active) != len(dex_data):
            raise ValueError(f"Candidate activity frame mismatch for {seq_id}")
    for frame in range(T):
        pose[frame] = _pose_from_native(dex_data[frame, 198:201], dex_data[frame, 201:205])
        object_local = _to_object_frame(old_obj[frame], old_pose[frame])
        object_normals_local = _normals_to_frame(old_obj_normals[frame], old_pose[frame])
        object_world[frame] = _from_object_frame(object_local, pose[frame])
        object_normals_world[frame] = _normals_from_frame(object_normals_local, pose[frame])
        if item["variant"] == "mano":
            hand_local = _to_object_frame(old_hand[frame], old_pose[frame])
            hand_normals_local = _normals_to_frame(old_hand_normals[frame], old_pose[frame])
            hand_world[frame] = _from_object_frame(hand_local, pose[frame])
            hand_normals_world[frame] = _normals_from_frame(hand_normals_local, pose[frame])
    if item["variant"] == "inspire_rl":
        hand_world[:], hand_normals_world[:] = _fk_surface(
            model, q_native, sampled_points, sampled_normals, sampled_visual_ids
        )
        active = dex_data[:, 205] > 0.5
        candidate_semantics = "dexplore_rl_native_object_contact_flag"
    else:
        active = np.asarray(old_active, dtype=bool)
        candidate_semantics = "parent_grab_right_candidate_active_5cm"

    seq_out = output_root / "sequences" / item["split"] / item["variant"] / _safe_name(seq_id)
    geometry = seq_out / "geometry"
    geometry.mkdir(parents=True, exist_ok=False)
    _save_array(geometry / "obj_points_pool_world.npy", object_world)
    _save_array(geometry / "obj_normals_pool_world.npy", object_normals_world)
    _save_array(geometry / "obj_pose_world.npy", pose)
    _save_array(geometry / "source_frame_id.npy", np.asarray(old_raw, dtype=np.int32))
    _save_array(geometry / "frame_time.npy", np.asarray(old_raw, dtype=np.float32) / 120.0)
    _save_array(geometry / "hand_points_world.npy", hand_world)
    _save_array(geometry / "hand_normals_world.npy", hand_normals_world)
    _save_array(geometry / "obj_candidate_mask_5cm.npy", active.astype(bool))
    geometry_manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": "1.0.0",
        "sequence_id": seq_id,
        "parent_seq_id": seq_id,
        "split": item["split"],
        "variant": item["variant"],
        "source": "grab" if item["variant"] == "mano" else "inspire_f1",
        "source_type": "mano_parent_geometry" if item["variant"] == "mano" else "dexplore_rl_native_q",
        "coordinate_frame": "object_pose_t",
        "world_frame": "dexplore_native_object_pose_world",
        "hand_side": "right",
        "object_pool_points": NUM_OBJECT_POINTS,
        "hand_points": NUM_HAND_POINTS,
        "effective_fps": 30.0,
        "source_fps": 120.0,
        "ds_rate": 4,
        "candidate_threshold_m": 0.05,
        "candidate_semantics": candidate_semantics,
        "candidate_mask_shape": [int(T)],
        "surface_sampling": {
            "method": "area_weighted_triangle_barycentric",
            "seed": int(surface_seed),
            "asset": "inspire_hand_right_urdf_visuals",
        },
        "rl_q": {
            "tensor": str(dex_tensor_path),
            "native_slice": [NATIVE_Q_START, NATIVE_Q_START + NUM_DOFS],
            "native_to_urdf": NATIVE_TO_URDF.tolist(),
        } if item["variant"] == "inspire_rl" else None,
        "input_parent_cache": str(old_root),
    }
    (geometry / "manifest.json").write_text(json.dumps(geometry_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "source": "grab" if item["variant"] == "mano" else "inspire_f1",
        "id": seq_id,
        "parent_seq_id": seq_id,
        "path": str(seq_out.relative_to(output_root)),
        "variant": item["variant"],
        "split": item["split"],
        "subject_id": item["subject_id"],
        "object_name": item["object_name"],
        "action_name": item["action_name"],
        "frame_count": int(T),
        "source_frame_first": int(old_raw[0]),
        "source_frame_last": int(old_raw[-1]),
    }


def _write_assignment(path: Path, assignments: dict[str, list[dict[str, Any]]], missing: list[str], seed: int) -> None:
    counts: dict[str, dict[str, int]] = {}
    for split, values in assignments.items():
        counts[split] = {"mano": sum(x.get("variant") == "mano" for x in values),
                         "inspire_rl": sum(x.get("variant") == "inspire_rl" for x in values)}
    payload = {
        "schema_name": "ref2dex_object_interaction_cm_dexplore_rl_assignment_v1",
        "seed": int(seed),
        "missing_parent_sequences": sorted(missing),
        "counts": counts,
        "sequences": {split: values for split, values in assignments.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_index(
    output_root: Path,
    entries: dict[str, list[dict[str, Any]]],
    *,
    missing: list[str],
    assignment_path: Path,
    rl_root: Path,
) -> None:
    counts = {}
    for split, values in entries.items():
        counts[split] = {
            "grab": sum(item["source"] == "grab" for item in values),
            "inspire_f1": sum(item["source"] == "inspire_f1" for item in values),
        }
    index = {
        "schema_name": INDEX_SCHEMA_NAME,
        "schema_version": "1.1.0",
        "created_at": _now(),
        "experiment_schema": SCHEMA_NAME,
        "source_probability": {"grab": 0.5, "inspire_f1": 0.5},
        # Both converted sources are sampled at 30 Hz.  Keep the default
        # action horizons identical; older OI caches used even Inspire
        # strides because their source frame rates differed.
        "stride_policy": {"grab": list(range(1, 11)), "inspire_f1": list(range(1, 11))},
        "object_pool_points": NUM_OBJECT_POINTS,
        "model_object_points": 1024,
        "hand_points_per_stream": NUM_HAND_POINTS,
        "max_union_hand_points": NUM_HAND_POINTS,
        "split_policy": {
            "parent": "existing ObjectInteractionCm GRAB split seed42",
            "train_val_variant": "stratified subject/object/action deterministic assignment",
            "test_variant": "MANO only",
            "same_parent_seq_across_variants": False,
        },
        "source_roots": {
            "grab_parent_cache": str((output_root.parent / "cm_object_v2_surface512_object_pose_20260830").resolve()),
            "dexplore_grab": str((output_root.parent / "dexplore_grab").resolve()),
            "inspire_rl": str(rl_root.resolve()),
        },
        "assignment_manifest": str(assignment_path.relative_to(output_root)),
        "missing_parent_sequences": sorted(missing),
        "sequences": entries,
        "counts": counts,
    }
    (output_root / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate_sequence(path: Path, expected_variant: str) -> dict[str, Any]:
    geometry = path / "geometry"
    arrays = {
        "obj": np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r"),
        "obj_normals": np.load(geometry / "obj_normals_pool_world.npy", mmap_mode="r"),
        "pose": np.load(geometry / "obj_pose_world.npy", mmap_mode="r"),
        "hand": np.load(geometry / "hand_points_world.npy", mmap_mode="r"),
        "hand_normals": np.load(geometry / "hand_normals_world.npy", mmap_mode="r"),
        "active": np.load(geometry / "obj_candidate_mask_5cm.npy", mmap_mode="r"),
    }
    T = arrays["obj"].shape[0]
    if arrays["obj"].shape != (T, NUM_OBJECT_POINTS, 3) or arrays["obj_normals"].shape != arrays["obj"].shape:
        raise ValueError(f"Object shape validation failed for {path}")
    if arrays["hand"].shape != (T, NUM_HAND_POINTS, 3) or arrays["hand_normals"].shape != arrays["hand"].shape:
        raise ValueError(f"Hand shape validation failed for {path}")
    if arrays["pose"].shape != (T, 4, 4) or arrays["active"].shape != (T,):
        raise ValueError(f"Metadata shape validation failed for {path}")
    for name, value in arrays.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Non-finite {name} array for {path}")
    pose = np.asarray(arrays["pose"][0])
    rotation_error = float(np.max(np.abs(pose[3] - np.asarray([0, 0, 0, 1], dtype=np.float32))))
    orth_error = float(np.max(np.abs(pose[:3, :3] @ pose[:3, :3].T - np.eye(3))))
    if rotation_error > 1e-5 or orth_error > 1e-4:
        raise ValueError(f"Invalid object pose for {path}: bottom={rotation_error}, orth={orth_error}")
    return {
        "path": str(path),
        "variant": expected_variant,
        "frames": int(T),
        "active_frames": int(np.asarray(arrays["active"], dtype=bool).sum()),
        "min_object_norm": float(np.min(np.linalg.norm(arrays["obj_normals"], axis=-1))),
        "min_hand_norm": float(np.min(np.linalg.norm(arrays["hand_normals"], axis=-1))),
    }


def _existing_entry(item: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Recover an entry for a sequence completed before a resumable run."""
    seq_out = output_root / "sequences" / item["split"] / item["variant"] / _safe_name(item["id"])
    frame_ids = np.load(seq_out / "geometry" / "source_frame_id.npy", mmap_mode="r")
    return {
        "source": "grab" if item["variant"] == "mano" else "inspire_f1",
        "id": item["id"],
        "parent_seq_id": item["id"],
        "path": str(seq_out.relative_to(output_root)),
        "variant": item["variant"],
        "split": item["split"],
        "subject_id": item["subject_id"],
        "object_name": item["object_name"],
        "action_name": item["action_name"],
        "frame_count": int(len(frame_ids)),
        "source_frame_first": int(frame_ids[0]),
        "source_frame_last": int(frame_ids[-1]),
    }


def _exact_contact_check(seq_id: str, geometry_path: Path, grab_root: Path, rl_root: Path) -> dict[str, Any]:
    """Pilot-only check of RL native contact against actual 5 cm surface distance."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {"available": False}
    object_points = np.load(geometry_path / "obj_points_pool_world.npy", mmap_mode="r")
    hand_points = np.load(geometry_path / "hand_points_world.npy", mmap_mode="r")
    active = np.load(geometry_path / "obj_candidate_mask_5cm.npy", mmap_mode="r").astype(bool)
    actual = np.zeros(len(active), dtype=bool)
    for frame in range(len(active)):
        actual[frame] = bool(np.min(cKDTree(object_points[frame]).query(hand_points[frame], k=1, workers=1)[0]) < 0.05)
    return {
        "available": True,
        "export_active_frames": int(active.sum()),
        "geometry_active_frames": int(actual.sum()),
        "disagreement_frames": int(np.count_nonzero(active != actual)),
        "disagreement_fraction": float(np.mean(active != actual)),
    }


def _refresh_rl_candidates(output_root: Path, entries: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Use the same 5 cm geometry rule for every generated RL hand stream."""
    from scipy.spatial import cKDTree

    reports: list[dict[str, Any]] = []
    for split, values in entries.items():
        for item in values:
            if item["variant"] != "inspire_rl":
                continue
            geometry = output_root / item["path"] / "geometry"
            object_world = np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r")
            pose = np.load(geometry / "obj_pose_world.npy", mmap_mode="r")
            hand_world = np.load(geometry / "hand_points_world.npy", mmap_mode="r")
            object_local = _to_object_frame(object_world[0], pose[0])
            tree = cKDTree(object_local)
            active = np.zeros((len(object_world),), dtype=bool)
            for frame in range(len(object_world)):
                hand_local = _to_object_frame(hand_world[frame], pose[frame])
                active[frame] = bool(np.min(tree.query(hand_local, k=1, workers=1)[0]) < 0.05)
            target = geometry / "obj_candidate_mask_5cm.npy"
            temporary = geometry / "obj_candidate_mask_5cm.refresh.npy"
            np.save(temporary, active)
            temporary.replace(target)
            manifest_path = geometry / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["candidate_semantics"] = "generated_rl_hand_to_object_surface_5cm"
            manifest["candidate_mask_source"] = "obj_candidate_mask_5cm.npy refreshed from generated hand geometry"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            reports.append({
                "id": item["id"],
                "split": split,
                "frames": int(len(active)),
                "active_frames": int(active.sum()),
            })
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--sequence", default=None, help="Parent id such as s1/airplane_fly_1 (pilot only)")
    parser.add_argument("--variant", choices=("mano", "inspire_rl"), default="inspire_rl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-index", default="data/processed_data/object_interaction_cm_v1_1/index.json")
    parser.add_argument("--grab-root", default="data/processed_data/cm_object_v2_surface512_object_pose_20260830")
    parser.add_argument("--dexplore-grab-root", default="data/processed_data/dexplore_grab/sequences")
    parser.add_argument("--rl-root", default="data/processed_data/inspire_rl")
    parser.add_argument("--urdf", default="/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surface-seed", type=int, default=2024)
    parser.add_argument("--resume", action="store_true", help="Resume a partially generated full output directory")
    parser.add_argument("--refresh-rl-candidates", action="store_true", help="Refresh RL masks using generated geometry and the 5 cm rule")
    parser.add_argument("--work-version", default="V1.2.5")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[5]
    output_root = Path(args.output).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    parent_index = Path(args.parent_index).resolve()
    grab_root = Path(args.grab_root).resolve()
    dexplore_grab_root = Path(args.dexplore_grab_root).resolve()
    rl_root = Path(args.rl_root).resolve()
    urdf_path = Path(args.urdf).resolve()
    if not parent_index.is_file() or not grab_root.is_dir() or not dexplore_grab_root.is_dir() or not rl_root.is_dir() or not urdf_path.is_file():
        raise FileNotFoundError("parent index, GRAB cache, dexplore_grab sequences, inspire_rl root and URDF must exist")

    parent = _parent_entries(parent_index)
    parent, missing = _intersection(parent, grab_root, dexplore_grab_root, rl_root)
    if args.mode == "pilot":
        if not args.sequence:
            raise ValueError("--sequence is required in pilot mode")
        chosen = None
        for split, values in parent.items():
            for item in values:
                if item["id"] == args.sequence:
                    chosen = {**item, **_sequence_metadata(grab_root, item["id"]), "variant": args.variant}
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(f"Pilot sequence is not in the parent/Dexplore intersection: {args.sequence}")
        assignments = {chosen["split"]: [chosen]}
    else:
        assignments = _assign_variants(parent, grab_root, args.seed)

    assignment_path = output_root / "assignment.json"
    _write_assignment(assignment_path, assignments, missing, args.seed)
    model = InspireUrdfModel(urdf_path)
    sampled_points, sampled_normals, sampled_visual_ids = model.surface_samples(NUM_HAND_POINTS, args.surface_seed)
    converted: dict[str, list[dict[str, Any]]] = {split: [] for split in assignments}
    validation: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for item in assignments.get(split, []):
            existing = output_root / "sequences" / item["split"] / item["variant"] / _safe_name(item["id"])
            if args.resume and (existing / "geometry" / "manifest.json").is_file():
                print(f"[{args.mode}] reusing {item['id']} ({item['variant']})", flush=True)
                entry = _existing_entry(item, output_root)
                converted.setdefault(split, []).append(entry)
                validation.append(_validate_sequence(existing, item["variant"]))
                continue
            print(f"[{args.mode}] converting {item['id']} ({item['variant']})", flush=True)
            entry = _convert_sequence(
                item,
                output_root=output_root,
                grab_root=grab_root,
                rl_root=rl_root,
                model=model,
                sampled_points=sampled_points,
                sampled_normals=sampled_normals,
                sampled_visual_ids=sampled_visual_ids,
                surface_seed=args.surface_seed,
            )
            converted.setdefault(split, []).append(entry)
            validation.append(_validate_sequence(output_root / entry["path"], item["variant"]))

    _build_index(output_root, converted, missing=missing, assignment_path=assignment_path, rl_root=rl_root)
    candidate_refresh = None
    if args.refresh_rl_candidates:
        candidate_refresh = _refresh_rl_candidates(output_root, converted)
    commit, dirty = _git_info(repo_root)
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "work_version": str(args.work_version),
        "operation": "dexplore_rl_right_hand_cache_conversion",
        "run_id": f"oicm-dexplore-rl-{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "run_status": "COMPLETED",
        "created_at": _now(),
        "git_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "parent_index": str(parent_index),
            "parent_index_sha256": _sha256(parent_index),
            "grab_root": str(grab_root),
            "dexplore_grab_root": str(dexplore_grab_root),
            "dexplore_rl_root": str(rl_root),
            "urdf": str(urdf_path),
            "urdf_sha256": _sha256(urdf_path),
        },
        "parameters": {
            "mode": args.mode,
            "resume": bool(args.resume),
            "refresh_rl_candidates": bool(args.refresh_rl_candidates),
            "seed": args.seed,
            "surface_seed": args.surface_seed,
            "surface_points": NUM_HAND_POINTS,
            "native_q_slice": [NATIVE_Q_START, NATIVE_Q_START + NUM_DOFS],
            "native_to_urdf": NATIVE_TO_URDF.tolist(),
            "right_hand_only": True,
            "effective_fps": 30.0,
        },
        "counts": {
            "parent_intersection": sum(len(values) for values in parent.values()),
            "missing_parent_sequences": len(missing),
            "converted_sequences": sum(len(values) for values in converted.values()),
            "by_split": {split: len(values) for split, values in converted.items()},
            "by_variant": {
                "mano": sum(item["variant"] == "mano" for values in converted.values() for item in values),
                "inspire_rl": sum(item["variant"] == "inspire_rl" for values in converted.values() for item in values),
            },
        },
        "outputs": {
            "assignment": str(assignment_path),
            "index": str(output_root / "index.json"),
            "validation_summary": str(output_root / "validation_summary.json"),
        },
        "validation": validation,
        "candidate_refresh": {
            "source": "generated hand/object geometry in object frame",
            "threshold_m": 0.05,
            "rl_sequences": len(candidate_refresh or []),
            "reports": candidate_refresh or [],
        },
        "conclusion": "SUPPORTED" if validation else "INCONCLUSIVE",
    }
    (output_root / "validation_summary.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.mode == "pilot" and args.variant == "inspire_rl":
        contact_check = _exact_contact_check(
            args.sequence,
            output_root / converted[next(iter(converted))][0]["path"] / "geometry",
            grab_root,
            rl_root,
        )
        (output_root / "pilot_contact_check.json").write_text(json.dumps(contact_check, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"pilot_contact_check": contact_check}, indent=2), flush=True)
    print(json.dumps(run_manifest["counts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
