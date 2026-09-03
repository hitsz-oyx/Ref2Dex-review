"""URDF/FK hand reconstruction for robot Stage 3 samples.

Robot hands deliberately use a separate path from MANO.  Stage 3 stores a
fixed set of link-local surface samples and their link indices.  At sample
time we perturb only the hand qpos, run URDF FK, and transform the resulting
points/normals into the clean hand-root frame.  The arm/base qpos is kept
fixed so the hand-root pose remains the same nuisance-free coordinate frame.
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np


_IO_CACHE: dict[Path, tuple[Any, Any]] = {}


def _load_robot_model(urdf_path: str | Path) -> tuple[Any, Any]:
    path = Path(urdf_path).expanduser().resolve()
    cached = _IO_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        from dataset.HRDexDB.hrdexdb_contact_heatmaps import hrdexdb_io as io
    except ModuleNotFoundError as exc:
        # DataLoader workers can import the task's ``dataset.py`` as the
        # top-level module named ``dataset``, shadowing the repository's
        # namespace package.  Load the self-contained HRDexDB IO module by
        # file path in that case, avoiding a fragile PYTHONPATH ordering.
        if "dataset.HRDexDB" not in str(exc):
            raise
        module_path = (
            Path(__file__).resolve().parents[3]
            / "dataset/HRDexDB/hrdexdb_contact_heatmaps/hrdexdb_io.py"
        )
        spec = importlib.util.spec_from_file_location("ref2dex_hrdexdb_io", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load HRDexDB IO module from {module_path}") from exc
        io = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = io
        spec.loader.exec_module(io)

    value = (io, io.parse_urdf(path))
    _IO_CACHE[path] = value
    return value


def _scalar_string(data: dict[str, np.ndarray], key: str) -> str:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar string, got {value.shape}")
    return str(value.item())


def perturb_robot_hand_from_frame(
    data: dict[str, np.ndarray],
    *,
    frame_idx: int,
    seed: int,
    apply_perturb: bool,
    perturb_prob: float = 1.0,
    noise_scale: float = 1.0,
    coordinate_frame: str = "hand_root",
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return one robot hand frame in the requested coordinate frame.

    The returned boolean reports whether q-space noise was actually applied.
    ``data`` is the complete NPZ payload loaded by ``CorrStaticDatasetV2``.
    """

    required = (
        "robot_qpos",
        "robot_hand_qpos_indices",
        "robot_qpos_lower",
        "robot_qpos_upper",
        "robot_hand_qpos_noise_std",
        "robot_hand_points_local",
        "robot_hand_normals_local",
        "robot_hand_point_link_index",
        "robot_link_names",
        "robot_c2r",
        "hand_root_pose",
        "robot_urdf_path",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Robot Stage 3 payload is missing {missing}")

    q = np.asarray(data["robot_qpos"][frame_idx], dtype=np.float64).copy()
    q_indices = np.asarray(data["robot_hand_qpos_indices"], dtype=np.int64).reshape(-1)
    lower = np.asarray(data["robot_qpos_lower"], dtype=np.float64).reshape(-1)
    upper = np.asarray(data["robot_qpos_upper"], dtype=np.float64).reshape(-1)
    std = np.asarray(data["robot_hand_qpos_noise_std"], dtype=np.float64).reshape(-1)
    if not np.isfinite(noise_scale) or float(noise_scale) <= 0.0:
        raise ValueError(f"noise_scale must be positive and finite, got {noise_scale!r}")
    std = std * float(noise_scale)
    if q.ndim != 1 or lower.shape != q.shape or upper.shape != q.shape:
        raise ValueError(f"Robot qpos/limit shapes disagree: q={q.shape}, lower={lower.shape}, upper={upper.shape}")
    if std.shape != q_indices.shape or np.any(q_indices < 0) or np.any(q_indices >= len(q)):
        raise ValueError(f"Invalid robot hand qpos indices/std: {q_indices.shape}, {std.shape}, q={q.shape}")

    rng = np.random.default_rng(int(seed))
    applied = bool(apply_perturb and float(perturb_prob) > 0.0 and rng.random() <= float(perturb_prob))
    if applied:
        q[q_indices] += rng.normal(0.0, std, size=len(q_indices))
        q = np.clip(q, lower, upper)

    urdf_path = _scalar_string(data, "robot_urdf_path")
    io, urdf = _load_robot_model(urdf_path)
    link_names = [str(value) for value in np.asarray(data["robot_link_names"]).reshape(-1).tolist()]
    link_index = np.asarray(data["robot_hand_point_link_index"], dtype=np.int64).reshape(-1)
    local_points = np.asarray(data["robot_hand_points_local"], dtype=np.float64)
    local_normals = np.asarray(data["robot_hand_normals_local"], dtype=np.float64)
    if local_points.shape != local_normals.shape or local_points.shape[0] != len(link_index):
        raise ValueError(f"Robot local binding shapes disagree: points={local_points.shape}, normals={local_normals.shape}, links={link_index.shape}")
    c2r = np.asarray(data["robot_c2r"], dtype=np.float64).reshape(4, 4)
    root_pose = np.asarray(data["hand_root_pose"][frame_idx], dtype=np.float64).reshape(4, 4)
    link_tfs = io.compute_link_transforms(urdf, q)

    points_robot = np.empty_like(local_points)
    normals_robot = np.empty_like(local_normals)
    for index, name in enumerate(link_names):
        mask = link_index == index
        if not np.any(mask):
            continue
        transform = np.asarray(link_tfs.get(name, np.eye(4)), dtype=np.float64)
        points_robot[mask] = local_points[mask] @ transform[:3, :3].T + transform[:3, 3]
        normals_robot[mask] = local_normals[mask] @ transform[:3, :3].T

    points_world = points_robot @ c2r[:3, :3].T + c2r[:3, 3]
    normals_world = normals_robot @ c2r[:3, :3].T
    if str(coordinate_frame) == "object":
        if "obj_root_pose_world" not in data:
            raise KeyError("Object-frame robot reconstruction requires obj_root_pose_world")
        target_pose = np.asarray(data["obj_root_pose_world"][frame_idx], dtype=np.float64).reshape(4, 4)
    elif str(coordinate_frame) == "hand_root":
        target_pose = root_pose
    else:
        raise ValueError(f"Unsupported coordinate_frame={coordinate_frame!r}")
    target_r = target_pose[:3, :3]
    target_t = target_pose[:3, 3]
    points_target = (points_world - target_t) @ target_r
    normals_target = normals_world @ target_r
    normals_target /= np.clip(np.linalg.norm(normals_target, axis=-1, keepdims=True), 1e-8, None)
    return points_target.astype(np.float32), normals_target.astype(np.float32), applied


def decode_robot_link_names(value: np.ndarray) -> list[str]:
    """Decode a scalar JSON/list field for diagnostics and schema checks."""
    array = np.asarray(value)
    if array.size == 1:
        item = array.item()
        if isinstance(item, str):
            try:
                decoded = json.loads(item)
                if isinstance(decoded, list):
                    return [str(x) for x in decoded]
            except json.JSONDecodeError:
                return [item]
    return [str(x) for x in array.reshape(-1).tolist()]
