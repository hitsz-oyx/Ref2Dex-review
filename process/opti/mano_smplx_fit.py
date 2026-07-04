#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified MANO fitting via smplx.MANO.

目的：
    1. 用 processed NPZ + meta.json 中记录的 MANO 配置复现 GT。
    2. 提供一个纯几何的逐帧优化器（jt_loss + 可选 vert_loss）。
    3. 产出与现有 viewer / pkl 消费者兼容的 payload，便于和 CPF 结果对比。

和 `arctic_mano_cpf_fit.py` 的区别：
    - 不依赖 manotorch / CPF / anchor loss。
    - 优化变量直接是 smplx.MANO 的
      `global_orient + hand_pose + transl (+ betas)`。
    - ARCTIC / GRAB 共用一条前向链，仅由 meta.json 决定
      `flat_hand_mean` / `v_template` 等配置。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# numpy legacy shim (chumpy / old smplx expect np.bool etc.)
# ---------------------------------------------------------------------------
for _legacy_name, _legacy_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)


def _stub_libmesh() -> None:
    """Stub uncompiled thirdparty.libmesh dependency used by CPF collision utils."""
    import types

    if "thirdparty.libmesh.inside_mesh" in sys.modules:
        return

    def _safe_check_mesh_contains(mesh, points, hash_resolution=512):
        return np.zeros(len(points), dtype=bool)

    mod = types.ModuleType("thirdparty.libmesh.inside_mesh")
    mod.check_mesh_contains = _safe_check_mesh_contains
    mod.MeshIntersector = lambda *a, **kw: None
    mod.TriangleIntersector2d = lambda *a, **kw: None
    sys.modules["thirdparty.libmesh.inside_mesh"] = mod
    if "thirdparty.libmesh" not in sys.modules:
        pkg = types.ModuleType("thirdparty.libmesh")
        pkg.__path__ = []
        sys.modules["thirdparty.libmesh"] = pkg
    sys.modules["thirdparty.libmesh"].inside_mesh = mod


_stub_libmesh()


import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from manotorch.utils.anchorutils import anchor_load, recover_anchor_batch
from smplx import MANO
import trimesh


def _first_existing_path(*candidates):
    for c in candidates:
        if c is None:
            continue
        p = Path(c) if not isinstance(c, Path) else c
        if p.exists():
            return p
    return None


REF2DEX_ROOT = Path(__file__).resolve().parents[2]
ARCTIC_DATA_ROOT = (
    REF2DEX_ROOT / "dataset" / "arctic" / "data"
)
CPF_ROOT = _first_existing_path(
    os.environ.get("CPF_ROOT"),
    REF2DEX_ROOT.parent / "CPF",
    Path.home() / "CPF",
)
if CPF_ROOT is None:
    raise RuntimeError("CPF repo not found; set CPF_ROOT or place CPF next to Ref2Dex.")
CPF_ASSETS_DIR = CPF_ROOT / "assets"
ANCHOR_ROOT = CPF_ASSETS_DIR / "anchor"
for _p in (str(REF2DEX_ROOT), str(CPF_ROOT), str(CPF_ROOT / "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from lib.postprocess.geo_loss import FieldLoss
from process.opti.object_sdf_grid import (
    ObjectSDFGrid,
    default_object_sdf_cache_root,
    load_or_build_object_sdf_grid,
)

DEFAULT_PROCESSED_ROOT = REF2DEX_ROOT / "processed_data" / "arctic"
DEFAULT_MANO_DIR = ARCTIC_DATA_ROOT / "body_models" / "mano"
DEFAULT_OUTPUT_ROOT = REF2DEX_ROOT / "processed_data" / "generated" / "mano_fit"
DEFAULT_SDF_REF_ROOT = _first_existing_path(
    os.environ.get("SDF_REF_ROOT"),
    REF2DEX_ROOT.parent / "SDF_ref",
)
CPF_CONTACT_CACHE_VERSION = 1
CPF_CONTACT_RANGE_THRESHOLD_MM = 10.0
CPF_CONTACT_TOPK = 32
CPF_CONTACT_ELASTI_THRESHOLD_MM = 45.0
CPF_CONTACT_ELASTI_CUTOFF = 0.1

DEFAULT_CONTACT_THRESHOLD = 0.030

MANO_JOINT_REORDER = [
    0, 13, 14, 15, 16,
    1, 2, 3, 17,
    4, 5, 6, 18,
    10, 11, 12, 19,
    7, 8, 9, 20,
]
MANO_TIP_VERTEX_INDICES = {
    "right": [745, 317, 444, 556, 673],
    "left": [745, 317, 445, 556, 673],
}
MANO_PARENTS_STD_16 = np.asarray(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14],
    dtype=np.int64,
)


@dataclass
class ProcessedSequence:
    dataset_name: str
    seq_id: str
    subject_id: str
    seq_name: str
    object_name: str
    object_mesh_path: Optional[Path]
    side: str
    npz_path: Path
    flat_hand_mean: bool
    mano_vtemplate_path: Optional[Path]
    frame_ids: np.ndarray
    preprocess_frame_idx: np.ndarray
    selected_indices: np.ndarray
    global_orient_aa: torch.Tensor
    hand_pose_aa: torch.Tensor
    mano_betas: torch.Tensor
    mano_translation: torch.Tensor
    mano_vertices_world: torch.Tensor
    mano_joints_world_user: torch.Tensor
    mano_root_pose: torch.Tensor
    obj_trajectory: torch.Tensor
    obj_verts_world: torch.Tensor
    obj_normals_world: torch.Tensor
    obj_point_id: np.ndarray
    obj_keep_8cm_preopt: np.ndarray
    hand_valid: np.ndarray
    hand_point_id: np.ndarray
    hand_cano_points: np.ndarray
    hand_cano_normals: np.ndarray
    hand_finger_id: np.ndarray
    hand_region_id: np.ndarray


def _slice_frame_ids(num_frames: int, start_frame: int, end_frame: Optional[int],
                     frame_step: int, max_frames: int) -> np.ndarray:
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    stop = num_frames if end_frame is None or end_frame < 0 else min(end_frame, num_frames)
    if stop <= start_frame:
        raise ValueError("Requested frame range is empty")
    frame_ids = np.arange(start_frame, stop, frame_step, dtype=np.int64)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    if frame_ids.size == 0:
        raise ValueError("No frames selected after applying frame filters")
    return frame_ids


def _load_mano_v1_assets(mano_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    import scipy.sparse as sp

    pkl_path = str(mano_root / "MANO_RIGHT.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    J = data["J_regressor"]
    if sp.issparse(J):
        J = J.toarray()
    faces = data["f"]
    if sp.issparse(faces):
        faces = faces.toarray()
    faces = np.asarray(faces, dtype=np.int64)
    return np.asarray(J, dtype=np.float32), faces


def _verts_to_21_joints_np(verts: np.ndarray, J_regressor: np.ndarray,
                           tip_indices, joint_reorder) -> np.ndarray:
    joints_16 = np.einsum("...vj,rv->...rj", verts, J_regressor)
    tips = verts[..., tip_indices, :]
    joints_21 = np.concatenate([joints_16, tips], axis=-2)
    return joints_21[..., joint_reorder, :]


def _resolve_hand_valid(data: np.lib.npyio.NpzFile, side_key: str,
                        selected: np.ndarray, contact_thresh: float) -> np.ndarray:
    for key in (f"{side_key}_frame_keep_3cm", f"{side_key}_hand_valid"):
        if key in data.files:
            return np.asarray(data[key][selected], dtype=bool)
    min_key = f"{side_key}_hand_min_dist_to_obj"
    if min_key in data.files:
        return np.asarray(data[min_key][selected] <= contact_thresh, dtype=bool)
    dist_key = f"{side_key}_hand_to_obj_dist"
    if dist_key in data.files:
        dist = data[dist_key][selected]
        return dist.min(axis=1) <= contact_thresh
    raise KeyError(
        f"NPZ missing frame_keep_3cm / hand_valid / '{min_key}' / '{dist_key}', "
        "cannot derive contact mask."
    )


def _rotmat_to_axis_angle(rot_mats: np.ndarray) -> np.ndarray:
    rot = Rotation.from_matrix(rot_mats.reshape(-1, 3, 3))
    return rot.as_rotvec().reshape(*rot_mats.shape[:-2], 3).astype(np.float32)


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)
    quat_wxyz = quat_wxyz.reshape(*rotvec.shape[:-1], 4)
    quat_wxyz = np.where(quat_wxyz[..., :1] < 0.0, -quat_wxyz, quat_wxyz)
    return quat_wxyz.astype(np.float32)


def _compose_world_quats(global_orient_aa: np.ndarray,
                         hand_pose_aa: np.ndarray,
                         parents: np.ndarray) -> np.ndarray:
    T = global_orient_aa.shape[0]
    world_rot = np.zeros((T, 16, 3, 3), dtype=np.float64)
    world_rot[:, 0] = Rotation.from_rotvec(global_orient_aa).as_matrix()
    local_rot = Rotation.from_rotvec(hand_pose_aa.reshape(-1, 3)).as_matrix()
    local_rot = local_rot.reshape(T, 15, 3, 3)
    for j in range(1, 16):
        parent = int(parents[j])
        world_rot[:, j] = np.matmul(world_rot[:, parent], local_rot[:, j - 1])
    quat_xyzw = Rotation.from_matrix(world_rot.reshape(-1, 3, 3)).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)
    quat_wxyz = quat_wxyz.reshape(T, 16, 4)
    quat_wxyz = np.where(quat_wxyz[..., :1] < 0.0, -quat_wxyz, quat_wxyz)
    return quat_wxyz.astype(np.float32)


def _scalarize_loss_tensor(x: torch.Tensor) -> torch.Tensor:
    """Normalize possibly-[1]-shaped loss tensors to 0-d scalars."""
    if x.ndim == 0:
        return x
    return x.reshape(())


def _penetration_barrier_from_sdf(
    sdf: torch.Tensor,
    penetration_tol_m: float,
    penetration_scale_m: float,
) -> torch.Tensor:
    penetration = torch.relu(-(sdf + float(penetration_tol_m)))
    active = penetration > 0.0
    if not bool(active.any()):
        return torch.zeros((), dtype=sdf.dtype, device=sdf.device)
    scale = max(float(penetration_scale_m), 1e-6)
    return torch.mean(torch.square(penetration[active] / scale))


def _load_mesh_sdf_from_asset_class(sdf_ref_root: Path):
    sdf_ref_root = Path(sdf_ref_root).resolve()
    if str(sdf_ref_root) not in sys.path:
        sys.path.insert(0, str(sdf_ref_root))
    try:
        from sdf.get_sdf_from_asset import MeshSDFfromAsset
    except Exception as e:
        raise RuntimeError(
            f"Failed to import MeshSDFfromAsset from {sdf_ref_root}; "
            f"check pytorch3d/open3d in the active environment."
        ) from e
    return MeshSDFfromAsset


def _read_dataset_meta(processed_root: Path) -> Dict:
    meta_path = processed_root / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_npz_key(data: np.lib.npyio.NpzFile, *keys: str) -> str:
    for key in keys:
        if key in data.files:
            return key
    raise KeyError(f"Missing keys {keys}; available keys={data.files}")


def _load_npz_array(data: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray:
    return np.asarray(data[_first_npz_key(data, *keys)])


def _select_betas_array(betas: np.ndarray, selected: np.ndarray) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        return np.repeat(betas[None, :], len(selected), axis=0).astype(np.float32)
    if betas.ndim == 2 and betas.shape[0] == 1:
        return np.repeat(betas, len(selected), axis=0).astype(np.float32)
    return np.asarray(betas[selected], dtype=np.float32)


def _ensure_preprocess_frame_idx(num_frames: int, preprocess_frame_idx: Optional[np.ndarray]) -> np.ndarray:
    if preprocess_frame_idx is None:
        return np.arange(num_frames, dtype=np.int64)
    return np.asarray(preprocess_frame_idx, dtype=np.int64)


def _face_centers_from_verts(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(verts[:, faces].mean(axis=2), dtype=np.float32)


def _face_normals_from_verts(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[:, faces[:, 0], :]
    v1 = verts[:, faces[:, 1], :]
    v2 = verts[:, faces[:, 2], :]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / np.clip(norms, 1e-10, None)).astype(np.float32)


def _points_world_to_obj(points_world: np.ndarray, obj_root_pose: np.ndarray) -> np.ndarray:
    R = obj_root_pose[:, :3, :3]
    t = obj_root_pose[:, :3, 3]
    return np.einsum("tji,tpj->tpi", R, points_world - t[:, None, :]).astype(np.float32)


def _points_world_to_obj_torch(points_world: torch.Tensor, obj_root_pose: torch.Tensor) -> torch.Tensor:
    if points_world.ndim == 2:
        R = obj_root_pose[:3, :3]
        t = obj_root_pose[:3, 3]
        return torch.einsum("ji,pj->pi", R, points_world - t[None, :])
    if points_world.ndim == 3:
        R = obj_root_pose[:, :3, :3]
        t = obj_root_pose[:, :3, 3]
        return torch.einsum("bji,bpj->bpi", R, points_world - t[:, None, :])
    raise ValueError(f"Unsupported points_world shape: {tuple(points_world.shape)}")


def _normals_world_to_obj(normals_world: np.ndarray, obj_root_pose: np.ndarray) -> np.ndarray:
    R = obj_root_pose[:, :3, :3]
    return np.einsum("tji,tpj->tpi", R, normals_world).astype(np.float32)


def _static_points_obj_to_world(points_obj: np.ndarray, obj_root_pose: np.ndarray) -> np.ndarray:
    R = obj_root_pose[:, :3, :3]
    t = obj_root_pose[:, :3, 3]
    return (np.einsum("tij,pj->tpi", R, points_obj) + t[:, None, :]).astype(np.float32)


def _static_normals_obj_to_world(normals_obj: np.ndarray, obj_root_pose: np.ndarray) -> np.ndarray:
    R = obj_root_pose[:, :3, :3]
    return np.einsum("tij,pj->tpi", R, normals_obj).astype(np.float32)


def _sample_object_surface_points_normals(
    object_mesh_path: Path,
    num_samples: int,
    seed: int,
) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(object_mesh_path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh from {object_mesh_path}, got {type(mesh)!r}")

    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        surface_points, face_idx = trimesh.sample.sample_surface(mesh, int(num_samples))
    finally:
        np.random.set_state(rng_state)
    surface_normals = mesh.face_normals[np.asarray(face_idx, dtype=np.int64)]
    return (
        mesh,
        np.asarray(surface_points, dtype=np.float32),
        np.asarray(surface_normals, dtype=np.float32),
    )


def _approx_vertex_signed_distance_to_surface(
    hand_verts_world: torch.Tensor,
    surface_points_world: torch.Tensor,
    surface_normals_world: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if surface_points_world.numel() == 0:
        return torch.empty(
            (int(hand_verts_world.shape[0]),),
            dtype=hand_verts_world.dtype,
            device=hand_verts_world.device,
        )

    topk = max(1, min(int(topk), int(surface_points_world.shape[0])))
    pair_dist = torch.cdist(hand_verts_world.unsqueeze(0), surface_points_world.unsqueeze(0)).squeeze(0)
    nn_dist, nn_idx = torch.topk(pair_dist, k=topk, dim=1, largest=False, sorted=False)
    nn_points = surface_points_world[nn_idx]
    nn_normals = F.normalize(surface_normals_world[nn_idx], dim=-1, eps=1e-8)
    offset = hand_verts_world[:, None, :] - nn_points
    signed = torch.sum(offset * nn_normals, dim=-1)
    if topk == 1:
        return signed[:, 0]
    weights = 1.0 / nn_dist.clamp_min(1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.sum(weights * signed, dim=1)


def _surface_signed_penetration_loss(
    hand_verts_world: torch.Tensor,
    surface_points_world: torch.Tensor,
    surface_normals_world: torch.Tensor,
    penetration_tol_m: float,
    penetration_scale_m: float,
    topk: int,
) -> torch.Tensor:
    signed = _approx_vertex_signed_distance_to_surface(
        hand_verts_world=hand_verts_world,
        surface_points_world=surface_points_world,
        surface_normals_world=surface_normals_world,
        topk=topk,
    )
    if signed.numel() == 0:
        return torch.zeros((), dtype=hand_verts_world.dtype, device=hand_verts_world.device)
    return _penetration_barrier_from_sdf(
        signed,
        penetration_tol_m=penetration_tol_m,
        penetration_scale_m=penetration_scale_m,
    )


def _compute_penetration_depth_mm(
    hand_verts_world: np.ndarray,
    obj_root_pose: np.ndarray,
    object_mesh: trimesh.Trimesh,
) -> np.ndarray:
    hand_verts_obj = _points_world_to_obj(hand_verts_world.astype(np.float32), obj_root_pose.astype(np.float32))
    num_frames = int(hand_verts_obj.shape[0])
    penetration_depth_mm = np.zeros((num_frames,), dtype=np.float32)
    try:
        import open3d as o3d

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(object_mesh.vertices, dtype=np.float64)),
                o3d.utility.Vector3iVector(np.asarray(object_mesh.faces, dtype=np.int32)),
            )
        )
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)
        for frame_idx in range(num_frames):
            signed = scene.compute_signed_distance(
                o3d.core.Tensor(hand_verts_obj[frame_idx], dtype=o3d.core.Dtype.Float32)
            ).numpy()
            signed = np.asarray(signed, dtype=np.float32)
            penetration_depth_mm[frame_idx] = max(0.0, float(np.max(-signed))) * 1000.0
        return penetration_depth_mm
    except Exception:
        pass

    for frame_idx in range(num_frames):
        signed = trimesh.proximity.signed_distance(object_mesh, hand_verts_obj[frame_idx])
        signed = np.asarray(signed, dtype=np.float32)
        penetration_depth_mm[frame_idx] = max(0.0, float(np.max(signed))) * 1000.0
    return penetration_depth_mm


def _compute_bidirectional_nn_per_frame(
    hand_points: np.ndarray,
    obj_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_frames = int(hand_points.shape[0])
    num_hand = int(hand_points.shape[1])
    num_obj = int(obj_points.shape[1])
    hand_to_obj_idx = np.zeros((num_frames, num_hand), dtype=np.int32)
    hand_to_obj_dist = np.zeros((num_frames, num_hand), dtype=np.float32)
    obj_to_hand_idx = np.zeros((num_frames, num_obj), dtype=np.int32)
    obj_to_hand_dist = np.zeros((num_frames, num_obj), dtype=np.float32)
    for t in range(num_frames):
        obj_tree = cKDTree(obj_points[t])
        hand_tree = cKDTree(hand_points[t])
        hand_to_obj_dist[t], hand_to_obj_idx[t] = obj_tree.query(hand_points[t], k=1)
        obj_to_hand_dist[t], obj_to_hand_idx[t] = hand_tree.query(obj_points[t], k=1)
    return (
        hand_to_obj_idx.astype(np.int32),
        hand_to_obj_dist.astype(np.float32),
        obj_to_hand_idx.astype(np.int32),
        obj_to_hand_dist.astype(np.float32),
    )


def _collapse_constant_rows(x: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim >= 2 and x.shape[0] > 0 and np.allclose(x, x[:1], atol=atol, rtol=0.0):
        return np.asarray(x[0], dtype=x.dtype)
    return x


def _pad_object_crops(
    obj_points_world_full: np.ndarray,
    obj_normals_world_full: np.ndarray,
    obj_point_id_full: np.ndarray,
    obj_root_pose: np.ndarray,
    obj_keep_8cm_preopt_full: np.ndarray,
    obj_keep_8cm_postopt_full: np.ndarray,
    obj_to_hand_idx_full: np.ndarray,
    obj_to_hand_dist_full: np.ndarray,
    opt_hand_face_centers_world: np.ndarray,
    opt_hand_normals_world: np.ndarray,
) -> Dict[str, np.ndarray]:
    num_frames, _, _ = obj_points_world_full.shape
    final_keep_full = np.asarray(obj_keep_8cm_preopt_full | obj_keep_8cm_postopt_full, dtype=bool)
    crop_counts = final_keep_full.sum(axis=1).astype(np.int32)
    max_crop = int(crop_counts.max()) if crop_counts.size > 0 else 0

    obj_points_world_crop = np.zeros((num_frames, max_crop, 3), dtype=np.float32)
    obj_normals_world_crop = np.zeros((num_frames, max_crop, 3), dtype=np.float32)
    obj_points_obj_crop = np.zeros((num_frames, max_crop, 3), dtype=np.float32)
    obj_normals_obj_crop = np.zeros((num_frames, max_crop, 3), dtype=np.float32)
    obj_crop_valid_mask = np.zeros((num_frames, max_crop), dtype=bool)
    obj_crop_raw_idx = np.full((num_frames, max_crop), -1, dtype=np.int32)
    obj_crop_point_id = np.full((num_frames, max_crop), -1, dtype=np.int32)
    obj_keep_8cm_preopt = np.zeros((num_frames, max_crop), dtype=bool)
    obj_keep_8cm_postopt = np.zeros((num_frames, max_crop), dtype=bool)
    obj_keep_8cm_final = np.zeros((num_frames, max_crop), dtype=bool)
    obj_to_hand_nn_id = np.full((num_frames, max_crop), -1, dtype=np.int32)
    obj_to_hand_dist = np.full((num_frames, max_crop), np.inf, dtype=np.float32)
    obj_to_hand_signed_dist = np.zeros((num_frames, max_crop), dtype=np.float32)
    hand_to_obj_nn_local_id = np.full((num_frames, opt_hand_face_centers_world.shape[1]), -1, dtype=np.int32)
    hand_to_obj_dist = np.full((num_frames, opt_hand_face_centers_world.shape[1]), np.inf, dtype=np.float32)
    obj_spring_1cm = np.zeros((num_frames, max_crop), dtype=bool)
    hand_spring_1cm = np.zeros((num_frames, opt_hand_face_centers_world.shape[1]), dtype=bool)
    obj_contact_region_3cm = np.zeros((num_frames, max_crop), dtype=bool)
    obj_context_region_8cm = np.zeros((num_frames, max_crop), dtype=bool)

    for t in range(num_frames):
        keep_idx = np.flatnonzero(final_keep_full[t]).astype(np.int32)
        n_keep = int(keep_idx.shape[0])
        if n_keep == 0:
            continue

        obj_points_world_t = obj_points_world_full[t, keep_idx]
        obj_normals_world_t = obj_normals_world_full[t, keep_idx]
        obj_points_world_crop[t, :n_keep] = obj_points_world_t
        obj_normals_world_crop[t, :n_keep] = obj_normals_world_t
        obj_crop_valid_mask[t, :n_keep] = True
        obj_crop_raw_idx[t, :n_keep] = keep_idx
        obj_crop_point_id[t, :n_keep] = obj_point_id_full[keep_idx]
        obj_keep_8cm_preopt[t, :n_keep] = obj_keep_8cm_preopt_full[t, keep_idx]
        obj_keep_8cm_postopt[t, :n_keep] = obj_keep_8cm_postopt_full[t, keep_idx]
        obj_keep_8cm_final[t, :n_keep] = True

        root_pose_t = obj_root_pose[t:t + 1]
        obj_points_obj_crop[t, :n_keep] = _points_world_to_obj(obj_points_world_t[None], root_pose_t)[0]
        obj_normals_obj_crop[t, :n_keep] = _normals_world_to_obj(obj_normals_world_t[None], root_pose_t)[0]

        hand_nn_id_t = obj_to_hand_idx_full[t, keep_idx].astype(np.int32)
        hand_nn_dist_t = obj_to_hand_dist_full[t, keep_idx].astype(np.float32)
        obj_to_hand_nn_id[t, :n_keep] = hand_nn_id_t
        obj_to_hand_dist[t, :n_keep] = hand_nn_dist_t

        hand_points_t = opt_hand_face_centers_world[t, hand_nn_id_t]
        hand_normals_t = opt_hand_normals_world[t, hand_nn_id_t]
        delta_t = obj_points_world_t - hand_points_t
        obj_to_hand_signed_dist[t, :n_keep] = np.einsum("ij,ij->i", delta_t, hand_normals_t).astype(np.float32)

        obj_spring_1cm[t, :n_keep] = hand_nn_dist_t <= 0.01
        obj_contact_region_3cm[t, :n_keep] = hand_nn_dist_t <= 0.03
        obj_context_region_8cm[t, :n_keep] = hand_nn_dist_t <= 0.08

        crop_tree = cKDTree(obj_points_world_t)
        hand_to_obj_dist_t, hand_to_obj_idx_t = crop_tree.query(opt_hand_face_centers_world[t], k=1)
        hand_to_obj_nn_local_id[t] = hand_to_obj_idx_t.astype(np.int32)
        hand_to_obj_dist[t] = hand_to_obj_dist_t.astype(np.float32)
        hand_spring_1cm[t] = hand_to_obj_dist_t <= 0.01

    return {
        "obj_points_world_crop": obj_points_world_crop,
        "obj_normals_world_crop": obj_normals_world_crop,
        "obj_points_obj_crop": obj_points_obj_crop,
        "obj_normals_obj_crop": obj_normals_obj_crop,
        "obj_crop_valid_mask": obj_crop_valid_mask,
        "obj_crop_raw_idx": obj_crop_raw_idx,
        "obj_crop_point_id": obj_crop_point_id,
        "obj_keep_8cm_preopt": obj_keep_8cm_preopt,
        "obj_keep_8cm_postopt": obj_keep_8cm_postopt,
        "obj_keep_8cm_final": obj_keep_8cm_final,
        "obj_to_hand_nn_id": obj_to_hand_nn_id,
        "obj_to_hand_dist": obj_to_hand_dist,
        "obj_to_hand_signed_dist": obj_to_hand_signed_dist,
        "hand_to_obj_nn_local_id": hand_to_obj_nn_local_id,
        "hand_to_obj_dist": hand_to_obj_dist,
        "obj_spring_1cm": obj_spring_1cm,
        "hand_spring_1cm": hand_spring_1cm,
        "obj_contact_region_3cm": obj_contact_region_3cm,
        "obj_context_region_8cm": obj_context_region_8cm,
    }


def _resolve_processed_npz(processed_root: Path, subject_id: str, seq_name: str) -> Path:
    candidates = [
        processed_root / subject_id / f"{seq_name}.npz",
        processed_root / "train" / f"{subject_id}_{seq_name}.npz",
        processed_root / f"{subject_id}_{seq_name}.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing processed NPZ for seq_id={subject_id}/{seq_name}; "
        f"checked: {[str(p) for p in candidates]}"
    )


def _default_cpf_cache_root(processed_root: Path) -> Path:
    return processed_root / f"_cpf_contact_cache_v{CPF_CONTACT_CACHE_VERSION}"


def _resolve_cpf_cache_path(cache_root: Path, seq_id: str, side: str) -> Path:
    subject_id, seq_name = seq_id.split("/", 1)
    return cache_root / subject_id / f"{seq_name}_{side}.npz"


def _is_cpf_cache_valid(
    cache_data: np.lib.npyio.NpzFile,
    npz_path: Path,
    side: str,
    anchor_root: Path,
) -> bool:
    try:
        cache_version = int(np.asarray(cache_data["cache_version"]).item())
        cached_side = str(np.asarray(cache_data["side"]).item())
        cached_source = str(np.asarray(cache_data["source_npz"]).item())
        cached_mtime_ns = int(np.asarray(cache_data["source_mtime_ns"]).item())
        cached_num_frames = int(np.asarray(cache_data["num_frames"]).item())
        cached_anchor_root = str(np.asarray(cache_data["anchor_root"]).item())
        cached_range_threshold = float(np.asarray(cache_data["range_threshold_mm"]).item())
        cached_topk = int(np.asarray(cache_data["topk"]).item())
        cached_elasti_threshold = float(np.asarray(cache_data["elasti_threshold_mm"]).item())
        cached_elasti_cutoff = float(np.asarray(cache_data["elasti_cutoff"]).item())
    except Exception:
        return False

    if cache_version != CPF_CONTACT_CACHE_VERSION:
        return False
    if cached_side != str(side):
        return False
    if cached_source != str(npz_path):
        return False
    if cached_anchor_root != str(anchor_root.resolve()):
        return False
    if abs(cached_range_threshold - float(CPF_CONTACT_RANGE_THRESHOLD_MM)) > 1e-6:
        return False
    if cached_topk != int(CPF_CONTACT_TOPK):
        return False
    if abs(cached_elasti_threshold - float(CPF_CONTACT_ELASTI_THRESHOLD_MM)) > 1e-6:
        return False
    if abs(cached_elasti_cutoff - float(CPF_CONTACT_ELASTI_CUTOFF)) > 1e-6:
        return False
    try:
        source_stat = npz_path.stat()
    except FileNotFoundError:
        return False
    if cached_mtime_ns != int(source_stat.st_mtime_ns):
        return False

    try:
        with np.load(str(npz_path), mmap_mode="r") as src_data:
            verts_key = _first_npz_key(
                src_data,
                f"{side.lower()}_hand_verts_world",
                f"{side.lower()}_hand_verts",
            )
            actual_num_frames = int(src_data[verts_key].shape[0])
    except Exception:
        return False
    return cached_num_frames == actual_num_frames


def _resolve_vtemplate_path(meta: Dict, seq_id: str, side_key: str) -> Optional[Path]:
    mano_meta = meta.get("mano_meta", {})
    seq_meta = mano_meta.get(seq_id, {})
    vtemplate_rel = seq_meta.get(f"{side_key}_hand_vtemplate_path")
    if not vtemplate_rel:
        return None
    vtemplate_path = Path(vtemplate_rel)
    if vtemplate_path.is_absolute():
        return vtemplate_path
    data_root = meta.get("data_root")
    if data_root:
        return Path(data_root) / vtemplate_path
    return vtemplate_path


def _resolve_object_mesh_path(meta: Dict, dataset_name: str, object_name: str) -> Optional[Path]:
    candidates = []
    object_asset_root = meta.get("object_asset_root")
    if object_asset_root:
        candidates.append(Path(object_asset_root) / object_name / "mesh.obj")
    candidates.extend([
        REF2DEX_ROOT / "assets" / dataset_name / "objects" / object_name / "mesh.obj",
        REF2DEX_ROOT / "render" / "assets" / "objects" / object_name / "mesh.obj",
        REF2DEX_ROOT / "preprocess" / "assets" / "objects" / object_name / "mesh.obj",
    ])
    return _first_existing_path(*candidates)


def _build_anchor_tables(anchor_mapping: dict, n_regions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    rev_anchor_mapping = {
        region_id: [] for region_id in range(n_regions)
    }
    for anchor_id, region_id in anchor_mapping.items():
        rev_anchor_mapping[int(region_id)].append(int(anchor_id))

    anchor_id_background = len(anchor_mapping.keys())
    region_id_background = len(rev_anchor_mapping.keys())
    anchor_padding_len = max(len(v) for v in rev_anchor_mapping.values())

    anchor_lookup_table = np.zeros((n_regions, anchor_padding_len), dtype=np.int64)
    anchor_id_table = np.full((n_regions, anchor_padding_len), anchor_id_background, dtype=np.int64)
    anchor_mask_table = np.zeros((n_regions, anchor_padding_len), dtype=bool)
    for region_id, anchor_ids in rev_anchor_mapping.items():
        if not anchor_ids:
            continue
        n_anchor = len(anchor_ids)
        anchor_lookup_table[region_id, :n_anchor] = np.asarray(anchor_ids, dtype=np.int64)
        anchor_id_table[region_id, :n_anchor] = np.asarray(anchor_ids, dtype=np.int64)
        anchor_mask_table[region_id, :n_anchor] = True
    return (
        anchor_lookup_table,
        anchor_id_table,
        anchor_mask_table,
        anchor_id_background,
        region_id_background,
    )


def _elasti_fn_torch(x: torch.Tensor, range_th: float = CPF_CONTACT_ELASTI_THRESHOLD_MM) -> torch.Tensor:
    x = torch.clamp(x, max=float(range_th))
    res = 0.5 * torch.cos((np.pi / float(range_th)) * x) + 0.5
    return torch.where(res < 1e-8, torch.zeros_like(res), res)


def compute_contact_info_torch(
    hand_verts_world: torch.Tensor,
    obj_verts_world: torch.Tensor,
    hand_palm_vid: torch.Tensor,
    merged_vertex_assignment: torch.Tensor,
    anchor_pos_world: torch.Tensor,
    anchor_lookup_table: torch.Tensor,
    anchor_id_table: torch.Tensor,
    anchor_mask_table: torch.Tensor,
    anchor_id_background: int,
    region_id_background: int,
    range_threshold: float = CPF_CONTACT_RANGE_THRESHOLD_MM,
    n_samples: int = CPF_CONTACT_TOPK,
    elasti_threshold: float = CPF_CONTACT_ELASTI_THRESHOLD_MM,
    elasti_cutoff: float = CPF_CONTACT_ELASTI_CUTOFF,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = hand_verts_world.device
    hand_palm_vid = hand_palm_vid.to(device=device, dtype=torch.long)
    merged_vertex_assignment = merged_vertex_assignment.to(device=device, dtype=torch.long)
    anchor_lookup_table = anchor_lookup_table.to(device=device, dtype=torch.long)
    anchor_id_table = anchor_id_table.to(device=device, dtype=torch.long)
    anchor_mask_table = anchor_mask_table.to(device=device, dtype=torch.bool)

    palm_verts_world = hand_verts_world[hand_palm_vid]
    selected_assignment = merged_vertex_assignment[hand_palm_vid]
    num_obj = int(obj_verts_world.shape[0])
    anchor_padding_len = int(anchor_id_table.shape[1])
    if num_obj == 0:
        return (
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int16),
            np.zeros((0, anchor_padding_len), dtype=np.int16),
            np.zeros((0, anchor_padding_len), dtype=np.float32),
            np.zeros((0, anchor_padding_len), dtype=np.uint8),
        )

    topk = min(int(n_samples), int(palm_verts_world.shape[0]))
    dist_mat = torch.cdist(obj_verts_world.unsqueeze(0), palm_verts_world.unsqueeze(0)).squeeze(0) * 1000.0
    dist_topk, order_idx = torch.topk(dist_mat, k=topk, dim=1, largest=False, sorted=True)
    valid_mask = dist_topk < float(range_threshold)
    contact_mask = torch.any(valid_mask, dim=1)

    origin_regions = selected_assignment[order_idx]
    mode_weight = torch.where(valid_mask, float(range_threshold) - dist_topk, torch.zeros_like(dist_topk))
    n_regions = int(anchor_lookup_table.shape[0])
    region_scores = torch.zeros((num_obj, n_regions), dtype=torch.float32, device=device)
    region_scores.scatter_add_(1, origin_regions, mode_weight)
    target_region = torch.argmax(region_scores, dim=1)

    vertex_contact = contact_mask.to(torch.uint8)
    contact_region_id = torch.full((num_obj,), int(region_id_background), dtype=torch.long, device=device)
    anchor_id = torch.full((num_obj, anchor_padding_len), int(anchor_id_background), dtype=torch.long, device=device)
    anchor_elasti = torch.zeros((num_obj, anchor_padding_len), dtype=torch.float32, device=device)
    anchor_padding_mask = torch.zeros((num_obj, anchor_padding_len), dtype=torch.uint8, device=device)

    if bool(contact_mask.any()):
        contact_idx = torch.nonzero(contact_mask, as_tuple=False).squeeze(1)
        region_sel = target_region[contact_idx]
        anchor_lookup_sel = anchor_lookup_table[region_sel]
        anchor_id_sel = anchor_id_table[region_sel]
        anchor_mask_sel = anchor_mask_table[region_sel]
        obj_contact = obj_verts_world[contact_idx]
        anchor_pos_sel = anchor_pos_world[anchor_lookup_sel]
        dist_anchor = torch.linalg.norm(obj_contact[:, None, :] - anchor_pos_sel, dim=-1) * 1000.0
        elasti = _elasti_fn_torch(dist_anchor, range_th=elasti_threshold)
        elasti = torch.where(
            anchor_mask_sel,
            torch.clamp_min(elasti, float(elasti_cutoff)),
            torch.zeros_like(elasti),
        )

        contact_region_id[contact_idx] = region_sel
        anchor_id[contact_idx] = anchor_id_sel
        anchor_elasti[contact_idx] = elasti
        anchor_padding_mask[contact_idx] = anchor_mask_sel.to(torch.uint8)

    return (
        vertex_contact.detach().cpu().numpy().astype(np.uint8),
        contact_region_id.detach().cpu().numpy().astype(np.int16),
        anchor_id.detach().cpu().numpy().astype(np.int16),
        anchor_elasti.detach().cpu().numpy().astype(np.float32),
        anchor_padding_mask.detach().cpu().numpy().astype(np.uint8),
    )


def compute_contact_info(
    hand_verts_world: np.ndarray,
    obj_verts_world: np.ndarray,
    hand_palm_vid: np.ndarray,
    merged_vertex_assignment: np.ndarray,
    n_regions: int,
    anchor_pos_world: np.ndarray,
    anchor_mapping: dict,
    range_threshold: float = CPF_CONTACT_RANGE_THRESHOLD_MM,
    n_samples: int = CPF_CONTACT_TOPK,
    elasti_threshold: float = CPF_CONTACT_ELASTI_THRESHOLD_MM,
    elasti_cutoff: float = CPF_CONTACT_ELASTI_CUTOFF,
    pad_vertex: bool = True,
    pad_anchor: bool = True,
):
    if not pad_vertex or not pad_anchor:
        raise NotImplementedError("Torch contact path currently supports pad_vertex=True and pad_anchor=True only.")
    (
        anchor_lookup_table,
        anchor_id_table,
        anchor_mask_table,
        anchor_id_background,
        region_id_background,
    ) = _build_anchor_tables(anchor_mapping, n_regions)
    device = torch.device("cpu")
    return compute_contact_info_torch(
        hand_verts_world=torch.as_tensor(hand_verts_world, dtype=torch.float32, device=device),
        obj_verts_world=torch.as_tensor(obj_verts_world, dtype=torch.float32, device=device),
        hand_palm_vid=torch.as_tensor(hand_palm_vid, dtype=torch.long, device=device),
        merged_vertex_assignment=torch.as_tensor(merged_vertex_assignment, dtype=torch.long, device=device),
        anchor_pos_world=torch.as_tensor(anchor_pos_world, dtype=torch.float32, device=device),
        anchor_lookup_table=torch.as_tensor(anchor_lookup_table, dtype=torch.long, device=device),
        anchor_id_table=torch.as_tensor(anchor_id_table, dtype=torch.long, device=device),
        anchor_mask_table=torch.as_tensor(anchor_mask_table, dtype=torch.bool, device=device),
        anchor_id_background=anchor_id_background,
        region_id_background=region_id_background,
        range_threshold=range_threshold,
        n_samples=n_samples,
        elasti_threshold=elasti_threshold,
        elasti_cutoff=elasti_cutoff,
    )


def load_sequence(
    processed_root: Path,
    meta: Dict,
    mano_dir: Path,
    seq_id: str,
    side: str,
    frame_step: int,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: int,
    contact_thresh: float,
) -> ProcessedSequence:
    side_key = side.lower()
    if side_key not in {"left", "right"}:
        raise ValueError(f"Unsupported side: {side}")

    subject_id, seq_name = seq_id.split("/", 1)
    npz_path = _resolve_processed_npz(processed_root, subject_id, seq_name)
    flat_hand_mean = bool(meta.get("mano_config", {}).get("flat_hand_mean", False))
    mano_vtemplate_path = _resolve_vtemplate_path(meta, seq_id, side_key)
    with np.load(str(npz_path), allow_pickle=True) as data:
        verts_key = _first_npz_key(
            data,
            f"{side_key}_hand_verts_world",
            f"{side_key}_hand_verts",
        )
        pose_key = _first_npz_key(data, f"{side_key}_hand_joint_axis_angle")
        betas_key = _first_npz_key(data, f"{side_key}_hand_betas")
        root_pose_key = _first_npz_key(data, f"{side_key}_hand_root_pose")
        obj_root_pose_key = _first_npz_key(data, "obj_root_pose")
        raw_frame_key = _first_npz_key(data, "raw_frame_id", "frame_id")

        num_pre_frames = int(data[verts_key].shape[0])
        raw_frame_ids_all = np.asarray(data[raw_frame_key], dtype=np.int64)
        preprocess_frame_idx_all = _ensure_preprocess_frame_idx(
            num_pre_frames,
            data["preprocess_frame_idx"] if "preprocess_frame_idx" in data.files else None,
        )
        selected = _slice_frame_ids(
            num_frames=num_pre_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            max_frames=max_frames,
        )

        dataset_name = str(
            data["dataset_name"].item() if "dataset_name" in data.files else meta.get("dataset_name", "arctic")
        ).strip().lower() or "arctic"
        object_name = str(
            data["object_name"].item() if "object_name" in data.files else seq_name.split("_")[0]
        )
        object_mesh_path = _resolve_object_mesh_path(meta, dataset_name, object_name)

        frame_ids = raw_frame_ids_all[selected]
        preprocess_frame_idx = preprocess_frame_idx_all[selected]

        J_regressor, mano_faces = _load_mano_v1_assets(mano_dir)
        verts_np = np.asarray(data[verts_key][selected], dtype=np.float32)
        hand_pose_aa_np = np.asarray(data[pose_key][selected], dtype=np.float32)
        betas_np = _select_betas_array(np.asarray(data[betas_key]), selected)
        root_pose_np = np.asarray(data[root_pose_key][selected], dtype=np.float32)
        obj_root_pose_np = np.asarray(data[obj_root_pose_key][selected], dtype=np.float32)
        obj_verts_world_np = _load_npz_array(data, "obj_points_world", "obj_points")[selected].astype(np.float32)
        obj_normals_world_np = _load_npz_array(data, "obj_normals_world", "obj_normals")[selected].astype(np.float32)
        obj_point_id_np = _load_npz_array(data, "obj_point_id").astype(np.int32)
        obj_keep_8cm_preopt_np = (
            np.asarray(data[f"{side_key}_obj_keep_8cm"][selected], dtype=bool)
            if f"{side_key}_obj_keep_8cm" in data.files
            else (
                np.asarray(data[f"obj_to_{side_key}_hand_dist"][selected] <= 0.08, dtype=bool)
                if f"obj_to_{side_key}_hand_dist" in data.files
                else None
            )
        )
        if obj_keep_8cm_preopt_np is None:
            raise KeyError(
                f"NPZ missing '{side_key}_obj_keep_8cm' and 'obj_to_{side_key}_hand_dist'; "
                "cannot derive pre-opt object crop mask."
            )

        transl_np = (
            np.asarray(data[f"{side_key}_hand_translation"][selected], dtype=np.float32)
            if f"{side_key}_hand_translation" in data.files
            else root_pose_np[:, :3, 3].astype(np.float32)
        )
        global_orient_aa_np = (
            np.asarray(data[f"{side_key}_hand_global_orient_aa"][selected], dtype=np.float32)
            if f"{side_key}_hand_global_orient_aa" in data.files
            else _rotmat_to_axis_angle(root_pose_np[:, :3, :3].astype(np.float64))
        )

        tip_ids = MANO_TIP_VERTEX_INDICES[side_key]
        joints_user_np = _verts_to_21_joints_np(verts_np, J_regressor, tip_ids, MANO_JOINT_REORDER)
        hand_valid_np = _resolve_hand_valid(data, side_key, selected, contact_thresh)

        if f"{side_key}_hand_point_id" in data.files:
            hand_point_id_np = _load_npz_array(data, f"{side_key}_hand_point_id").astype(np.int32)
        else:
            hand_point_id_np = np.arange(int(mano_faces.shape[0]), dtype=np.int32)
        hand_cano_points_np = (
            _load_npz_array(data, f"{side_key}_hand_cano_points").astype(np.float32)
            if f"{side_key}_hand_cano_points" in data.files
            else np.zeros((hand_point_id_np.shape[0], 3), dtype=np.float32)
        )
        hand_cano_normals_np = (
            _load_npz_array(data, f"{side_key}_hand_cano_normals").astype(np.float32)
            if f"{side_key}_hand_cano_normals" in data.files
            else np.zeros((hand_point_id_np.shape[0], 3), dtype=np.float32)
        )
        hand_finger_id_np = (
            _load_npz_array(data, f"{side_key}_hand_finger_id").astype(np.int32)
            if f"{side_key}_hand_finger_id" in data.files
            else np.zeros((hand_point_id_np.shape[0],), dtype=np.int32)
        )
        hand_region_id_np = (
            _load_npz_array(data, f"{side_key}_hand_region_id").astype(np.int32)
            if f"{side_key}_hand_region_id" in data.files
            else np.zeros((hand_point_id_np.shape[0],), dtype=np.int32)
        )

    return ProcessedSequence(
        dataset_name=dataset_name,
        seq_id=seq_id,
        subject_id=subject_id,
        seq_name=seq_name,
        object_name=object_name,
        object_mesh_path=object_mesh_path,
        side=side_key,
        npz_path=npz_path,
        flat_hand_mean=flat_hand_mean,
        mano_vtemplate_path=mano_vtemplate_path,
        frame_ids=frame_ids,
        preprocess_frame_idx=preprocess_frame_idx,
        selected_indices=selected,
        global_orient_aa=torch.from_numpy(global_orient_aa_np),
        hand_pose_aa=torch.from_numpy(hand_pose_aa_np),
        mano_betas=torch.from_numpy(betas_np),
        mano_translation=torch.from_numpy(transl_np),
        mano_vertices_world=torch.from_numpy(verts_np),
        mano_joints_world_user=torch.from_numpy(joints_user_np),
        mano_root_pose=torch.from_numpy(root_pose_np),
        obj_trajectory=torch.from_numpy(obj_root_pose_np),
        obj_verts_world=torch.from_numpy(obj_verts_world_np),
        obj_normals_world=torch.from_numpy(obj_normals_world_np),
        obj_point_id=obj_point_id_np,
        obj_keep_8cm_preopt=obj_keep_8cm_preopt_np,
        hand_valid=hand_valid_np,
        hand_point_id=hand_point_id_np,
        hand_cano_points=hand_cano_points_np,
        hand_cano_normals=hand_cano_normals_np,
        hand_finger_id=hand_finger_id_np,
        hand_region_id=hand_region_id_np,
    )


class SmplxManoFitter:
    def __init__(self, device: torch.device, mano_dir: Path, side: str,
                 flat_hand_mean: bool,
                 mano_vtemplate_path: Optional[Path],
                 optimize_betas: bool, lr: float, n_iter: int,
                 joint_target_weight: float, vertex_target_weight: float,
                 pose_prior_weight: float, transl_prior_weight: float,
                 shape_prior_weight: float,
                 lambda_contact_loss: float,
                 lambda_repulsion_loss: float,
                 repulsion_mode: str,
                 repulsion_query: float,
                 repulsion_threshold: float,
                 repulsion_surface_topk: int,
                 penetration_tol_m: float,
                 penetration_scale_m: float,
                 anchor_root: Path) -> None:
        self.device = device
        self.side = side
        self.optimize_betas = optimize_betas
        self.lr = lr
        self.n_iter = n_iter
        self.joint_target_weight = joint_target_weight
        self.vertex_target_weight = vertex_target_weight
        self.pose_prior_weight = pose_prior_weight
        self.transl_prior_weight = transl_prior_weight
        self.shape_prior_weight = shape_prior_weight
        self.lambda_contact_loss = lambda_contact_loss
        self.lambda_repulsion_loss = lambda_repulsion_loss
        self.repulsion_mode = str(repulsion_mode)
        self.repulsion_query = repulsion_query
        self.repulsion_threshold = repulsion_threshold
        self.repulsion_surface_topk = int(repulsion_surface_topk)
        self.penetration_tol_m = float(penetration_tol_m)
        self.penetration_scale_m = float(penetration_scale_m)
        self.flat_hand_mean = flat_hand_mean
        self.mano_vtemplate_path = mano_vtemplate_path
        self.anchor_root = Path(anchor_root).resolve()
        self.object_sdf: Optional[ObjectSDFGrid] = None
        self.object_mesh_sdf_asset = None

        mano_kwargs = {}
        if mano_vtemplate_path is not None:
            v_template = trimesh.load(str(mano_vtemplate_path), process=False).vertices.astype(np.float32)
            mano_kwargs["v_template"] = v_template

        self.mano = MANO(
            model_path=str(mano_dir),
            is_rhand=(side == "right"),
            use_pca=False,
            flat_hand_mean=flat_hand_mean,
            **mano_kwargs,
        ).to(device)
        J_regressor_np, faces_np = _load_mano_v1_assets(mano_dir)
        self.J_regressor = torch.from_numpy(J_regressor_np).float().to(device)
        self.faces = faces_np.astype(np.int64)
        self.tip_ids = torch.tensor(
            MANO_TIP_VERTEX_INDICES[side], device=device, dtype=torch.long
        )
        anchor_face_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(str(self.anchor_root))
        self.anchor_face_idx = torch.from_numpy(anchor_face_idx).long().unsqueeze(0).to(device)
        self.anchor_weight = torch.from_numpy(anchor_weight).float().unsqueeze(0).to(device)
        self.merged_vertex_assignment = np.asarray(merged_vertex_assignment, dtype=np.int64)
        self.merged_vertex_assignment_t = torch.from_numpy(self.merged_vertex_assignment).long().to(device)
        self.anchor_mapping = anchor_mapping
        self.n_anchor_regions = int(self.merged_vertex_assignment.max()) + 1
        self.palm_vid = np.loadtxt(str(CPF_ASSETS_DIR / "hand_palm_full.txt"), dtype=int)
        self.palm_vid_t = torch.from_numpy(np.asarray(self.palm_vid, dtype=np.int64)).long().to(device)
        (
            anchor_lookup_table,
            anchor_id_table,
            anchor_mask_table,
            self.anchor_id_background,
            self.region_id_background,
        ) = _build_anchor_tables(self.anchor_mapping, self.n_anchor_regions)
        self.anchor_lookup_table_t = torch.from_numpy(anchor_lookup_table).long().to(device)
        self.anchor_id_table_t = torch.from_numpy(anchor_id_table).long().to(device)
        self.anchor_mask_table_t = torch.from_numpy(anchor_mask_table).bool().to(device)

    def _align_betas_to_model(self, betas: torch.Tensor) -> torch.Tensor:
        """Match input betas dim to the loaded MANO shapedirs dim.

        Current Ref2Dex MANO assets use 10 shape coefficients, so this is a no-op
        in the normal ARCTIC/GRAB pipeline. We still keep the shim to stay robust
        to alternative MANO assets whose shape space dimension differs.
        """
        actual_num_betas = int(self.mano.shapedirs.shape[-1])
        if betas.shape[-1] == actual_num_betas:
            return betas
        betas = betas[..., :min(betas.shape[-1], actual_num_betas)]
        if actual_num_betas > betas.shape[-1]:
            pad = torch.zeros(
                *betas.shape[:-1],
                actual_num_betas - betas.shape[-1],
                device=betas.device,
                dtype=betas.dtype,
            )
            betas = torch.cat([betas, pad], dim=-1)
        return betas

    def _forward_batch(
        self,
        global_orient_aa: torch.Tensor,
        hand_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = self._align_betas_to_model(betas)
        if hand_pose_aa.ndim == 2:
            hand_pose_aa = hand_pose_aa.view(hand_pose_aa.shape[0], 15, 3)
        out = self.mano(
            global_orient=global_orient_aa,
            hand_pose=hand_pose_aa.view(hand_pose_aa.shape[0], 45),
            betas=betas,
            transl=transl,
        )
        verts = out.vertices
        joints_std = out.joints
        joints_16 = torch.einsum("bvj,rv->brj", verts, self.J_regressor)
        tips = verts[:, self.tip_ids]
        joints_21 = torch.cat([joints_16, tips], dim=1)[:, MANO_JOINT_REORDER]
        return verts, joints_std, joints_21

    def _forward(self,
                 global_orient_aa: torch.Tensor,
                 hand_pose_aa: torch.Tensor,
                 betas: torch.Tensor,
                 transl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        verts, joints_std, joints_21 = self._forward_batch(
            global_orient_aa.unsqueeze(0),
            hand_pose_aa.view(1, 15, 3),
            betas.unsqueeze(0),
            transl.unsqueeze(0),
        )
        verts = verts.squeeze(0)
        joints_std = joints_std.squeeze(0)
        joints_21 = joints_21.squeeze(0)
        return verts, joints_std, joints_21

    def prepare_cpf_contact_data(
        self,
        gt_verts_world: np.ndarray,
        obj_verts_world: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        gt_verts_world_t = torch.as_tensor(gt_verts_world, dtype=torch.float32, device=self.device)
        obj_verts_world_t = torch.as_tensor(obj_verts_world, dtype=torch.float32, device=self.device)
        anchor_pos_world_t = recover_anchor_batch(
            gt_verts_world_t.unsqueeze(0),
            self.anchor_face_idx,
            self.anchor_weight,
        ).squeeze(0)
        vertex_contact, contact_region_id, anchor_id, anchor_elasti, anchor_padding_mask = compute_contact_info_torch(
            hand_verts_world=gt_verts_world_t,
            obj_verts_world=obj_verts_world_t,
            hand_palm_vid=self.palm_vid_t,
            merged_vertex_assignment=self.merged_vertex_assignment_t,
            anchor_pos_world=anchor_pos_world_t,
            anchor_lookup_table=self.anchor_lookup_table_t,
            anchor_id_table=self.anchor_id_table_t,
            anchor_mask_table=self.anchor_mask_table_t,
            anchor_id_background=self.anchor_id_background,
            region_id_background=self.region_id_background,
            range_threshold=CPF_CONTACT_RANGE_THRESHOLD_MM,
            n_samples=CPF_CONTACT_TOPK,
            elasti_threshold=CPF_CONTACT_ELASTI_THRESHOLD_MM,
            elasti_cutoff=CPF_CONTACT_ELASTI_CUTOFF,
        )
        return {
            "vertex_contact": vertex_contact.astype(np.uint8),
            "contact_region_id": contact_region_id.astype(np.int16),
            "anchor_id": anchor_id.astype(np.int16),
            "anchor_elasti": anchor_elasti.astype(np.float32),
            "anchor_padding_mask": anchor_padding_mask.astype(np.uint8),
            "contact_ratio": float(vertex_contact.mean()),
        }

    @staticmethod
    def _slice_cpf_contact_cache_arrays(cache_arrays: Dict[str, np.ndarray], selected_indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "vertex_contact": cache_arrays["vertex_contact"][selected_indices],
            "contact_region_id": cache_arrays["contact_region_id"][selected_indices],
            "anchor_id": cache_arrays["anchor_id"][selected_indices],
            "anchor_elasti": cache_arrays["anchor_elasti"][selected_indices],
            "anchor_padding_mask": cache_arrays["anchor_padding_mask"][selected_indices],
            "contact_ratio": cache_arrays["contact_ratio"][selected_indices],
        }

    def build_cpf_contact_cache(
        self,
        npz_path: Path,
        seq_id: str,
        side: str,
        cache_path: Path,
        progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        side_key = side.lower()
        with np.load(str(npz_path), allow_pickle=False) as data:
            verts_key = _first_npz_key(
                data,
                f"{side_key}_hand_verts_world",
                f"{side_key}_hand_verts",
            )
            hand_verts_all = np.asarray(data[verts_key], dtype=np.float32)
            obj_points_all = _load_npz_array(data, "obj_points_world", "obj_points").astype(np.float32)

        num_frames = int(hand_verts_all.shape[0])
        num_obj_points = int(obj_points_all.shape[1])
        anchor_padding_len = int(self.anchor_id_table_t.shape[1])
        cache_arrays = {
            "vertex_contact": np.zeros((num_frames, num_obj_points), dtype=np.uint8),
            "contact_region_id": np.full((num_frames, num_obj_points), self.region_id_background, dtype=np.int16),
            "anchor_id": np.full((num_frames, num_obj_points, anchor_padding_len), self.anchor_id_background, dtype=np.int16),
            "anchor_elasti": np.zeros((num_frames, num_obj_points, anchor_padding_len), dtype=np.float32),
            "anchor_padding_mask": np.zeros((num_frames, num_obj_points, anchor_padding_len), dtype=np.uint8),
            "contact_ratio": np.zeros((num_frames,), dtype=np.float32),
        }

        report_every = max(1, num_frames // 10)
        for frame_idx in range(num_frames):
            if progress and (frame_idx == 0 or frame_idx == num_frames - 1 or frame_idx % report_every == 0):
                print(
                    f"[cpf-cache] build seq={seq_id} side={side_key} "
                    f"frame {frame_idx + 1}/{num_frames}"
                )
            frame_cache = self.prepare_cpf_contact_data(
                hand_verts_all[frame_idx],
                obj_points_all[frame_idx],
            )
            cache_arrays["vertex_contact"][frame_idx] = frame_cache["vertex_contact"]
            cache_arrays["contact_region_id"][frame_idx] = frame_cache["contact_region_id"]
            cache_arrays["anchor_id"][frame_idx] = frame_cache["anchor_id"]
            cache_arrays["anchor_elasti"][frame_idx] = frame_cache["anchor_elasti"]
            cache_arrays["anchor_padding_mask"][frame_idx] = frame_cache["anchor_padding_mask"]
            cache_arrays["contact_ratio"][frame_idx] = frame_cache["contact_ratio"]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            cache_version=np.asarray(CPF_CONTACT_CACHE_VERSION, dtype=np.int32),
            side=np.asarray(side_key),
            source_npz=np.asarray(str(npz_path)),
            source_mtime_ns=np.asarray(npz_path.stat().st_mtime_ns, dtype=np.int64),
            num_frames=np.asarray(num_frames, dtype=np.int32),
            range_threshold_mm=np.asarray(CPF_CONTACT_RANGE_THRESHOLD_MM, dtype=np.float32),
            topk=np.asarray(CPF_CONTACT_TOPK, dtype=np.int32),
            elasti_threshold_mm=np.asarray(CPF_CONTACT_ELASTI_THRESHOLD_MM, dtype=np.float32),
            elasti_cutoff=np.asarray(CPF_CONTACT_ELASTI_CUTOFF, dtype=np.float32),
            anchor_root=np.asarray(str(self.anchor_root)),
            **cache_arrays,
        )
        return cache_arrays

    def load_or_build_cpf_contact_cache(
        self,
        sequence: ProcessedSequence,
        cache_root: Path,
        rebuild: bool = False,
        progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        cache_path = _resolve_cpf_cache_path(cache_root, sequence.seq_id, sequence.side)
        if (not rebuild) and cache_path.exists():
            with np.load(str(cache_path), allow_pickle=False) as cache_data:
                if _is_cpf_cache_valid(cache_data, sequence.npz_path, sequence.side, self.anchor_root):
                    print(f"[cpf-cache] hit {cache_path}")
                    cache_arrays = {
                        "vertex_contact": np.asarray(cache_data["vertex_contact"]),
                        "contact_region_id": np.asarray(cache_data["contact_region_id"]),
                        "anchor_id": np.asarray(cache_data["anchor_id"]),
                        "anchor_elasti": np.asarray(cache_data["anchor_elasti"]),
                        "anchor_padding_mask": np.asarray(cache_data["anchor_padding_mask"]),
                        "contact_ratio": np.asarray(cache_data["contact_ratio"]),
                    }
                    return self._slice_cpf_contact_cache_arrays(cache_arrays, sequence.selected_indices)
            print(f"[cpf-cache] stale {cache_path}, rebuilding")

        print(f"[cpf-cache] build {cache_path}")
        cache_arrays = self.build_cpf_contact_cache(
            npz_path=sequence.npz_path,
            seq_id=sequence.seq_id,
            side=sequence.side,
            cache_path=cache_path,
            progress=progress,
        )
        return self._slice_cpf_contact_cache_arrays(cache_arrays, sequence.selected_indices)

    def _prepare_cpf_contact_tensors(
        self,
        cpf_contact_data: Optional[Dict[str, np.ndarray]],
        obj_verts_world_t: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        device = self.device
        tensors = {
            "indexed_anchor_id": None,
            "indexed_anchor_elasti": None,
            "indexed_vertex_id": None,
            "indexed_elasti_k": None,
            "obj_contact_verts_t": None,
            "contact_ratio": 0.0,
        }
        if cpf_contact_data is None:
            return tensors

        vertex_contact_t = torch.from_numpy(cpf_contact_data["vertex_contact"]).long().to(device)
        tensors["contact_ratio"] = float(cpf_contact_data["contact_ratio"])
        anchor_id_t = torch.from_numpy(cpf_contact_data["anchor_id"]).long().to(device)
        anchor_elasti_t = torch.from_numpy(cpf_contact_data["anchor_elasti"]).float().to(device)
        anchor_padding_mask_t = torch.from_numpy(cpf_contact_data["anchor_padding_mask"]).long().to(device)
        if int(vertex_contact_t.sum().item()) == 0 or obj_verts_world_t is None:
            return tensors

        anchor_id_pos = anchor_id_t[vertex_contact_t == 1]
        anchor_elasti_pos = anchor_elasti_t[vertex_contact_t == 1]
        anchor_padding_mask_pos = anchor_padding_mask_t[vertex_contact_t == 1]
        indexed_anchor_id = anchor_id_pos[anchor_padding_mask_pos == 1]
        indexed_anchor_elasti = anchor_elasti_pos[anchor_padding_mask_pos == 1]
        vertex_id = torch.arange(anchor_id_pos.shape[0], device=device)[:, None].repeat_interleave(
            anchor_padding_mask_pos.shape[1], dim=1
        )
        indexed_vertex_id = vertex_id[anchor_padding_mask_pos == 1]
        tip_anchor_mask = torch.zeros(indexed_anchor_id.shape[0], dtype=torch.bool, device=device)
        for tip_anchor_id in [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]:
            tip_anchor_mask = tip_anchor_mask | (indexed_anchor_id == tip_anchor_id)
        indexed_elasti_k = torch.where(
            tip_anchor_mask,
            torch.tensor(1.0, device=device),
            torch.tensor(0.1, device=device),
        )
        obj_contact_verts_t = obj_verts_world_t[vertex_contact_t == 1]
        tensors.update(
            indexed_anchor_id=indexed_anchor_id,
            indexed_anchor_elasti=indexed_anchor_elasti,
            indexed_vertex_id=indexed_vertex_id,
            indexed_elasti_k=indexed_elasti_k,
            obj_contact_verts_t=obj_contact_verts_t,
        )
        return tensors

    def reconstruct_batch(
        self,
        global_orient_aa: torch.Tensor,
        hand_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            verts, joints_std, joints_user = self._forward_batch(
                global_orient_aa.to(self.device).float(),
                hand_pose_aa.to(self.device).float(),
                betas.to(self.device).float(),
                transl.to(self.device).float(),
            )
        return {
            "verts": verts.cpu().numpy(),
            "joints_std": joints_std.cpu().numpy(),
            "joints_user": joints_user.cpu().numpy(),
        }

    def reconstruct(self,
                    global_orient_aa: torch.Tensor,
                    hand_pose_aa: torch.Tensor,
                    betas: torch.Tensor,
                    transl: torch.Tensor) -> Dict[str, np.ndarray]:
        batch = self.reconstruct_batch(
            global_orient_aa.unsqueeze(0),
            hand_pose_aa.unsqueeze(0),
            betas.unsqueeze(0),
            transl.unsqueeze(0),
        )
        return {
            "verts": batch["verts"][0],
            "joints_std": batch["joints_std"][0],
            "joints_user": batch["joints_user"][0],
        }

    def fit_batch(
        self,
        global_orient_aa_init: torch.Tensor,
        hand_pose_aa_init: torch.Tensor,
        betas_init: torch.Tensor,
        transl_init: torch.Tensor,
        gt_joints_user: torch.Tensor,
        gt_verts: torch.Tensor,
        obj_root_pose: Optional[torch.Tensor] = None,
        obj_verts_world: Optional[torch.Tensor] = None,
        obj_normals_world: Optional[torch.Tensor] = None,
        repulsion_obj_verts_world: Optional[torch.Tensor] = None,
        repulsion_obj_normals_world: Optional[torch.Tensor] = None,
        cpf_contact_data_list: Optional[list[Optional[Dict[str, np.ndarray]]]] = None,
        perturb_init_scale: float = 0.0,
        progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        device = self.device
        batch_size = int(global_orient_aa_init.shape[0])

        global_orient = global_orient_aa_init.detach().to(device).float().clone()
        hand_pose = hand_pose_aa_init.detach().to(device).float().clone()
        if hand_pose.ndim == 2:
            hand_pose = hand_pose.view(batch_size, 15, 3)
        betas = betas_init.detach().to(device).float().clone()
        transl = transl_init.detach().to(device).float().clone()

        if perturb_init_scale > 0.0:
            global_orient = global_orient + perturb_init_scale * torch.randn_like(global_orient)
            hand_pose = hand_pose + perturb_init_scale * torch.randn_like(hand_pose)
            transl = transl + (0.5 * perturb_init_scale) * torch.randn_like(transl)

        global_orient.requires_grad_(True)
        hand_pose.requires_grad_(True)
        transl.requires_grad_(True)
        betas.requires_grad_(self.optimize_betas)

        params = [global_orient, hand_pose, transl]
        if self.optimize_betas:
            params.append(betas)
        optimizer = torch.optim.Adam(params, lr=self.lr)

        gt_joints_user = gt_joints_user.detach().to(device).float()
        gt_verts = gt_verts.detach().to(device).float()
        obj_root_pose_t = (
            obj_root_pose.detach().to(device).float()
            if obj_root_pose is not None else None
        )
        obj_verts_world_t = (
            obj_verts_world.detach().to(device).float()
            if obj_verts_world is not None else None
        )
        obj_normals_world_t = (
            obj_normals_world.detach().to(device).float()
            if obj_normals_world is not None else None
        )
        repulsion_obj_verts_world_t = (
            repulsion_obj_verts_world.detach().to(device).float()
            if repulsion_obj_verts_world is not None else None
        )
        repulsion_obj_normals_world_t = (
            repulsion_obj_normals_world.detach().to(device).float()
            if repulsion_obj_normals_world is not None else None
        )
        init_hand_pose = hand_pose.detach().clone()
        init_transl = transl.detach().clone()
        init_betas = betas.detach().clone()

        cpf_tensors = []
        contact_ratio_np = np.zeros((batch_size,), dtype=np.float32)
        if cpf_contact_data_list is None:
            cpf_contact_data_list = [None] * batch_size
        for frame_idx, cpf_contact_data in enumerate(cpf_contact_data_list):
            obj_frame = obj_verts_world_t[frame_idx] if obj_verts_world_t is not None else None
            tensors = self._prepare_cpf_contact_tensors(cpf_contact_data, obj_frame)
            cpf_tensors.append(tensors)
            contact_ratio_np[frame_idx] = float(tensors["contact_ratio"])

        with torch.no_grad():
            init_verts, init_joints_std, init_joints_user = self._forward_batch(
                global_orient, hand_pose, betas, transl
            )
            init_jt_rms = torch.sqrt(((init_joints_user - gt_joints_user) ** 2).sum(dim=-1).mean(dim=-1))
            init_vert_rms = torch.sqrt(((init_verts - gt_verts) ** 2).sum(dim=-1).mean(dim=-1))

        history = []
        for step in range(self.n_iter):
            optimizer.zero_grad()
            pred_verts, pred_joints_std, pred_joints_user = self._forward_batch(
                global_orient, hand_pose, betas, transl
            )
            jt_loss_pf = ((pred_joints_user - gt_joints_user) ** 2).mean(dim=(1, 2))
            if self.vertex_target_weight > 0.0:
                vert_loss_pf = ((pred_verts - gt_verts) ** 2).mean(dim=(1, 2))
            else:
                vert_loss_pf = torch.zeros((batch_size,), device=device)

            contact_loss_pf = []
            for frame_idx in range(batch_size):
                cpf_frame = cpf_tensors[frame_idx]
                if (
                    self.lambda_contact_loss > 0.0
                    and cpf_frame["indexed_anchor_id"] is not None
                    and cpf_frame["indexed_anchor_id"].numel() > 0
                    and cpf_frame["obj_contact_verts_t"] is not None
                ):
                    pred_anchor = recover_anchor_batch(
                        pred_verts[frame_idx:frame_idx + 1],
                        self.anchor_face_idx,
                        self.anchor_weight,
                    ).squeeze(0)
                    anchor_pos = pred_anchor[cpf_frame["indexed_anchor_id"]]
                    loss_frame = FieldLoss.contact_loss(
                        anchor_pos,
                        cpf_frame["obj_contact_verts_t"][cpf_frame["indexed_vertex_id"]],
                        cpf_frame["indexed_anchor_elasti"],
                        cpf_frame["indexed_elasti_k"],
                    )
                else:
                    loss_frame = torch.zeros((), device=device)
                contact_loss_pf.append(loss_frame)
            contact_loss_pf = torch.stack(contact_loss_pf)

            repulsion_loss_pf = []
            for frame_idx in range(batch_size):
                if (
                    self.lambda_repulsion_loss > 0.0
                ):
                    if self.repulsion_mode == "sdf_grid" and self.object_sdf is not None and obj_root_pose_t is not None:
                        pred_verts_obj = _points_world_to_obj_torch(pred_verts[frame_idx], obj_root_pose_t[frame_idx])
                        loss_frame = self.object_sdf.penetration_loss(
                            pred_verts_obj,
                            penetration_tol_m=self.penetration_tol_m,
                            penetration_scale_m=self.penetration_scale_m,
                        )
                    elif (
                        self.repulsion_mode == "mesh_sdf_asset"
                        and self.object_mesh_sdf_asset is not None
                        and obj_root_pose_t is not None
                    ):
                        pred_verts_obj = _points_world_to_obj_torch(pred_verts[frame_idx], obj_root_pose_t[frame_idx])
                        sdf_frame = self.object_mesh_sdf_asset.query_sdf(
                            pred_verts_obj,
                            reduce_parts=True,
                            return_details=False,
                        )
                        sdf_frame = sdf_frame[0] if sdf_frame.ndim == 2 else sdf_frame.reshape(-1)
                        loss_frame = _penetration_barrier_from_sdf(
                            sdf_frame,
                            penetration_tol_m=self.penetration_tol_m,
                            penetration_scale_m=self.penetration_scale_m,
                        )
                    elif (
                        self.repulsion_mode == "signed_surface"
                        and repulsion_obj_verts_world_t is not None
                        and repulsion_obj_normals_world_t is not None
                    ):
                        loss_frame = _surface_signed_penetration_loss(
                            hand_verts_world=pred_verts[frame_idx],
                            surface_points_world=repulsion_obj_verts_world_t[frame_idx],
                            surface_normals_world=repulsion_obj_normals_world_t[frame_idx],
                            penetration_tol_m=self.penetration_tol_m,
                            penetration_scale_m=self.penetration_scale_m,
                            topk=self.repulsion_surface_topk,
                        )
                    elif (
                        repulsion_obj_verts_world_t is not None
                        and repulsion_obj_normals_world_t is not None
                    ):
                        loss_frame = FieldLoss.full_repulsion_loss(
                            pred_verts[frame_idx],
                            repulsion_obj_verts_world_t[frame_idx],
                            repulsion_obj_normals_world_t[frame_idx],
                            query=self.repulsion_query,
                            threshold=self.repulsion_threshold,
                        )
                        loss_frame = _scalarize_loss_tensor(loss_frame)
                    else:
                        loss_frame = torch.zeros((), device=device)
                else:
                    loss_frame = torch.zeros((), device=device)
                repulsion_loss_pf.append(loss_frame)
            repulsion_loss_pf = torch.stack(repulsion_loss_pf)

            pose_prior_pf = ((hand_pose - init_hand_pose) ** 2).mean(dim=(1, 2)) if self.pose_prior_weight > 0.0 else torch.zeros((batch_size,), device=device)
            transl_prior_pf = ((transl - init_transl) ** 2).mean(dim=1) if self.transl_prior_weight > 0.0 else torch.zeros((batch_size,), device=device)
            shape_prior_pf = ((betas - init_betas) ** 2).mean(dim=1) if self.shape_prior_weight > 0.0 else torch.zeros((batch_size,), device=device)

            loss_pf = (
                self.joint_target_weight * jt_loss_pf
                + self.vertex_target_weight * vert_loss_pf
                + self.lambda_contact_loss * contact_loss_pf
                + self.lambda_repulsion_loss * repulsion_loss_pf
                + self.pose_prior_weight * pose_prior_pf
                + self.transl_prior_weight * transl_prior_pf
                + self.shape_prior_weight * shape_prior_pf
            )
            loss = loss_pf.mean()
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))
            if progress and (step % max(1, self.n_iter // 10) == 0 or step == self.n_iter - 1):
                print(
                    f"    [smplx-batch] step {step:04d} jt={jt_loss_pf.mean().item():.4e} "
                    f"vert={vert_loss_pf.mean().item():.4e} "
                    f"contact={contact_loss_pf.mean().item():.4e} "
                    f"repul={repulsion_loss_pf.mean().item():.4e}"
                )

        with torch.no_grad():
            opt_verts, opt_joints_std, opt_joints_user = self._forward_batch(
                global_orient, hand_pose, betas, transl
            )
            final_jt_rms = torch.sqrt(((opt_joints_user - gt_joints_user) ** 2).sum(dim=-1).mean(dim=-1))
            final_vert_rms = torch.sqrt(((opt_verts - gt_verts) ** 2).sum(dim=-1).mean(dim=-1))
            final_jt_loss_pf = ((opt_joints_user - gt_joints_user) ** 2).mean(dim=(1, 2))
            final_vert_loss_pf = ((opt_verts - gt_verts) ** 2).mean(dim=(1, 2)) if self.vertex_target_weight > 0.0 else torch.zeros((batch_size,), device=device)

            final_contact_loss_pf = []
            for frame_idx in range(batch_size):
                cpf_frame = cpf_tensors[frame_idx]
                if (
                    self.lambda_contact_loss > 0.0
                    and cpf_frame["indexed_anchor_id"] is not None
                    and cpf_frame["indexed_anchor_id"].numel() > 0
                    and cpf_frame["obj_contact_verts_t"] is not None
                ):
                    pred_anchor = recover_anchor_batch(
                        opt_verts[frame_idx:frame_idx + 1],
                        self.anchor_face_idx,
                        self.anchor_weight,
                    ).squeeze(0)
                    anchor_pos = pred_anchor[cpf_frame["indexed_anchor_id"]]
                    loss_frame = FieldLoss.contact_loss(
                        anchor_pos,
                        cpf_frame["obj_contact_verts_t"][cpf_frame["indexed_vertex_id"]],
                        cpf_frame["indexed_anchor_elasti"],
                        cpf_frame["indexed_elasti_k"],
                    )
                else:
                    loss_frame = torch.zeros((), device=device)
                final_contact_loss_pf.append(loss_frame)
            final_contact_loss_pf = torch.stack(final_contact_loss_pf)

            final_repulsion_loss_pf = []
            for frame_idx in range(batch_size):
                if (
                    self.lambda_repulsion_loss > 0.0
                ):
                    if self.repulsion_mode == "sdf_grid" and self.object_sdf is not None and obj_root_pose_t is not None:
                        opt_verts_obj = _points_world_to_obj_torch(opt_verts[frame_idx], obj_root_pose_t[frame_idx])
                        loss_frame = self.object_sdf.penetration_loss(
                            opt_verts_obj,
                            penetration_tol_m=self.penetration_tol_m,
                            penetration_scale_m=self.penetration_scale_m,
                        )
                    elif (
                        self.repulsion_mode == "mesh_sdf_asset"
                        and self.object_mesh_sdf_asset is not None
                        and obj_root_pose_t is not None
                    ):
                        opt_verts_obj = _points_world_to_obj_torch(opt_verts[frame_idx], obj_root_pose_t[frame_idx])
                        sdf_frame = self.object_mesh_sdf_asset.query_sdf(
                            opt_verts_obj,
                            reduce_parts=True,
                            return_details=False,
                        )
                        sdf_frame = sdf_frame[0] if sdf_frame.ndim == 2 else sdf_frame.reshape(-1)
                        loss_frame = _penetration_barrier_from_sdf(
                            sdf_frame,
                            penetration_tol_m=self.penetration_tol_m,
                            penetration_scale_m=self.penetration_scale_m,
                        )
                    elif (
                        self.repulsion_mode == "signed_surface"
                        and repulsion_obj_verts_world_t is not None
                        and repulsion_obj_normals_world_t is not None
                    ):
                        loss_frame = _surface_signed_penetration_loss(
                            hand_verts_world=opt_verts[frame_idx],
                            surface_points_world=repulsion_obj_verts_world_t[frame_idx],
                            surface_normals_world=repulsion_obj_normals_world_t[frame_idx],
                            penetration_tol_m=self.penetration_tol_m,
                            penetration_scale_m=self.penetration_scale_m,
                            topk=self.repulsion_surface_topk,
                        )
                    elif (
                        repulsion_obj_verts_world_t is not None
                        and repulsion_obj_normals_world_t is not None
                    ):
                        loss_frame = FieldLoss.full_repulsion_loss(
                            opt_verts[frame_idx],
                            repulsion_obj_verts_world_t[frame_idx],
                            repulsion_obj_normals_world_t[frame_idx],
                            query=self.repulsion_query,
                            threshold=self.repulsion_threshold,
                        )
                        loss_frame = _scalarize_loss_tensor(loss_frame)
                    else:
                        loss_frame = torch.zeros((), device=device)
                else:
                    loss_frame = torch.zeros((), device=device)
                final_repulsion_loss_pf.append(loss_frame)
            final_repulsion_loss_pf = torch.stack(final_repulsion_loss_pf)

            pose_prior_pf = ((hand_pose - init_hand_pose) ** 2).mean(dim=(1, 2)) if self.pose_prior_weight > 0.0 else torch.zeros((batch_size,), device=device)
            transl_prior_pf = ((transl - init_transl) ** 2).mean(dim=1) if self.transl_prior_weight > 0.0 else torch.zeros((batch_size,), device=device)
            shape_prior_pf = ((betas - init_betas) ** 2).mean(dim=1) if self.shape_prior_weight > 0.0 else torch.zeros((batch_size,), device=device)
            final_loss_pf = (
                self.joint_target_weight * final_jt_loss_pf
                + self.vertex_target_weight * final_vert_loss_pf
                + self.lambda_contact_loss * final_contact_loss_pf
                + self.lambda_repulsion_loss * final_repulsion_loss_pf
                + self.pose_prior_weight * pose_prior_pf
                + self.transl_prior_weight * transl_prior_pf
                + self.shape_prior_weight * shape_prior_pf
            )

        opt_global_orient_aa = global_orient.detach().cpu().numpy().astype(np.float32)
        opt_hand_pose_aa = hand_pose.detach().cpu().numpy().astype(np.float32).reshape(batch_size, 15, 3)
        opt_betas = betas.detach().cpu().numpy().astype(np.float32)
        opt_transl = transl.detach().cpu().numpy().astype(np.float32)

        local_quat = np.concatenate(
            [
                _rotvec_to_quat_wxyz(opt_global_orient_aa)[:, None, :],
                _rotvec_to_quat_wxyz(opt_hand_pose_aa),
            ],
            axis=1,
        )
        world_quat = _compose_world_quats(
            opt_global_orient_aa, opt_hand_pose_aa, MANO_PARENTS_STD_16
        )

        return {
            "opt_hand_global_orient_aa": opt_global_orient_aa,
            "opt_hand_pose_axis_angle": opt_hand_pose_aa,
            "opt_hand_pose_quat": local_quat,
            "opt_hand_pose_quat_world": world_quat,
            "opt_hand_tsl": opt_transl,
            "opt_hand_betas": opt_betas,
            "opt_hand_verts": opt_verts.detach().cpu().numpy().astype(np.float32),
            "opt_hand_joints_std": opt_joints_std.detach().cpu().numpy().astype(np.float32),
            "opt_hand_joints_user": opt_joints_user.detach().cpu().numpy().astype(np.float32),
            "init_jt_rms": init_jt_rms.detach().cpu().numpy().astype(np.float32),
            "final_jt_rms": final_jt_rms.detach().cpu().numpy().astype(np.float32),
            "init_vert_rms": init_vert_rms.detach().cpu().numpy().astype(np.float32),
            "final_vert_rms": final_vert_rms.detach().cpu().numpy().astype(np.float32),
            "contact_ratio": contact_ratio_np,
            "final_loss": final_loss_pf.detach().cpu().numpy().astype(np.float32),
            "loss_history": np.asarray(history, dtype=np.float32),
        }

    def fit_frame(
        self,
        global_orient_aa_init: torch.Tensor,
        hand_pose_aa_init: torch.Tensor,
        betas_init: torch.Tensor,
        transl_init: torch.Tensor,
        gt_joints_user: torch.Tensor,
        gt_verts: torch.Tensor,
        obj_root_pose: Optional[torch.Tensor] = None,
        obj_verts_world: Optional[torch.Tensor] = None,
        obj_normals_world: Optional[torch.Tensor] = None,
        repulsion_obj_verts_world: Optional[torch.Tensor] = None,
        repulsion_obj_normals_world: Optional[torch.Tensor] = None,
        cpf_contact_data: Optional[Dict[str, np.ndarray]] = None,
        perturb_init_scale: float = 0.0,
        progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        batch_result = self.fit_batch(
            global_orient_aa_init=global_orient_aa_init.unsqueeze(0),
            hand_pose_aa_init=hand_pose_aa_init.unsqueeze(0),
            betas_init=betas_init.unsqueeze(0),
            transl_init=transl_init.unsqueeze(0),
            gt_joints_user=gt_joints_user.unsqueeze(0),
            gt_verts=gt_verts.unsqueeze(0),
            obj_root_pose=(obj_root_pose.unsqueeze(0) if obj_root_pose is not None else None),
            obj_verts_world=(obj_verts_world.unsqueeze(0) if obj_verts_world is not None else None),
            obj_normals_world=(obj_normals_world.unsqueeze(0) if obj_normals_world is not None else None),
            repulsion_obj_verts_world=(
                repulsion_obj_verts_world.unsqueeze(0) if repulsion_obj_verts_world is not None else None
            ),
            repulsion_obj_normals_world=(
                repulsion_obj_normals_world.unsqueeze(0) if repulsion_obj_normals_world is not None else None
            ),
            cpf_contact_data_list=[cpf_contact_data],
            perturb_init_scale=perturb_init_scale,
            progress=progress,
        )
        return {
            "opt_hand_global_orient_aa": batch_result["opt_hand_global_orient_aa"][0],
            "opt_hand_pose_axis_angle": batch_result["opt_hand_pose_axis_angle"][0],
            "opt_hand_pose_quat": batch_result["opt_hand_pose_quat"][0],
            "opt_hand_pose_quat_world": batch_result["opt_hand_pose_quat_world"][0],
            "opt_hand_tsl": batch_result["opt_hand_tsl"][0],
            "opt_hand_betas": batch_result["opt_hand_betas"][0],
            "opt_hand_verts": batch_result["opt_hand_verts"][0],
            "opt_hand_joints_std": batch_result["opt_hand_joints_std"][0],
            "opt_hand_joints_user": batch_result["opt_hand_joints_user"][0],
            "init_jt_rms": float(batch_result["init_jt_rms"][0]),
            "final_jt_rms": float(batch_result["final_jt_rms"][0]),
            "init_vert_rms": float(batch_result["init_vert_rms"][0]),
            "final_vert_rms": float(batch_result["final_vert_rms"][0]),
            "contact_ratio": float(batch_result["contact_ratio"][0]),
            "final_loss": float(batch_result["final_loss"][0]),
            "loss_history": batch_result["loss_history"],
        }


def _write_mano_opt_meta(
    base_out_dir: Path,
    source_meta: Dict,
    sequence: ProcessedSequence,
    mano_dir: Path,
    args: argparse.Namespace,
) -> None:
    raw_fps = float(source_meta.get("raw_fps", 0.0) or 0.0)
    raw_dt = float(source_meta.get("raw_dt", (1.0 / raw_fps) if raw_fps > 0 else 0.0) or 0.0)
    preprocess_stride = int(source_meta.get("preprocess_stride", 1) or 1)
    dt_effective = float(source_meta.get("dt_effective", raw_dt * preprocess_stride) or 0.0)
    meta_out = {
        "schema_name": "mano_opt_single_hand",
        "schema_version": "0.2.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_preprocess_root": str(Path(args.processed_root).resolve()),
        "output_root": str(base_out_dir.resolve()),
        "dataset_name": sequence.dataset_name,
        "raw_fps": raw_fps,
        "raw_dt": raw_dt,
        "preprocess_stride": preprocess_stride,
        "dt_effective": dt_effective,
        "mano_opt_frame_policy": {
            "use_frame_keep_3cm": True,
            "frame_keep_thresh": 0.03,
            "drop_non_keep_frames": bool(args.respect_hand_valid),
            "keep_raw_frame_id": True,
            "keep_preprocess_frame_idx": True,
        },
        "object_crop_policy": {
            "obj_crop_thresh": 0.08,
            "obj_crop_source": "preopt_or_postopt_union",
            "spring_contact_thresh": 0.01,
        },
        "mano_config": {
            "mano_dir": str(mano_dir),
            "flat_hand_mean": bool(sequence.flat_hand_mean),
            "use_pca": False,
            "mano_vtemplate_path": str(sequence.mano_vtemplate_path) if sequence.mano_vtemplate_path else None,
        },
        "optimizer_config": {
            "lr": float(args.lr),
            "n_iter": int(args.n_iter),
            "frame_batch_size": int(args.frame_batch_size),
            "joint_target_weight": float(args.joint_target_weight),
            "vertex_target_weight": float(args.vertex_target_weight),
            "lambda_contact_loss": float(args.lambda_contact_loss),
            "lambda_repulsion_loss": float(args.lambda_repulsion_loss),
            "repulsion_mode": str(args.repulsion_mode),
            "pose_prior_weight": float(args.pose_prior_weight),
            "transl_prior_weight": float(args.transl_prior_weight),
            "shape_prior_weight": float(args.shape_prior_weight),
            "optimize_betas": bool(args.optimize_betas),
            "repulsion_surface_samples": int(args.repulsion_surface_samples),
            "repulsion_surface_topk": int(args.repulsion_surface_topk),
            "penetration_tol_mm": float(args.penetration_tol_mm),
            "penetration_scale_mm": float(args.penetration_scale_mm),
            "sdf_grid_resolution": int(args.sdf_grid_resolution),
            "sdf_grid_padding_m": float(args.sdf_grid_padding_m),
            "sdf_ref_root": str(Path(args.sdf_ref_root).resolve()) if args.sdf_ref_root else None,
        },
    }
    with (base_out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Unified MANO fitting via smplx.MANO"
    )
    parser.add_argument("--processed-root", type=str, default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--seq-id", type=str, default="s01/box_grab_01")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument(
        "--contact-thresh",
        type=float,
        default=DEFAULT_CONTACT_THRESHOLD,
        help="Fallback frame-keep threshold in meters when Stage 1 NPZ lacks explicit 3cm mask.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mano-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--joint-target-weight", type=float, default=1.0)
    parser.add_argument("--vertex-target-weight", type=float, default=0.0)
    parser.add_argument("--lambda-contact-loss", type=float, default=10.0)
    parser.add_argument("--lambda-repulsion-loss", type=float, default=0.5)
    parser.add_argument(
        "--repulsion-mode",
        type=str,
        default="sdf_grid",
        choices=["sdf_grid", "mesh_sdf_asset", "signed_surface", "cpf_proxy"],
    )
    parser.add_argument("--repulsion-query", type=float, default=0.030)
    parser.add_argument("--repulsion-threshold", type=float, default=0.080)
    parser.add_argument("--repulsion-surface-samples", type=int, default=4096)
    parser.add_argument("--repulsion-surface-topk", type=int, default=4)
    parser.add_argument("--penetration-tol-mm", type=float, default=0.5)
    parser.add_argument("--penetration-scale-mm", type=float, default=5.0)
    parser.add_argument("--repulsion-surface-seed", type=int, default=0)
    parser.add_argument("--sdf-grid-resolution", type=int, default=96)
    parser.add_argument("--sdf-grid-padding-m", type=float, default=0.05)
    parser.add_argument(
        "--sdf-ref-root",
        type=str,
        default=(str(DEFAULT_SDF_REF_ROOT) if DEFAULT_SDF_REF_ROOT is not None else None),
    )
    parser.add_argument("--pose-prior-weight", type=float, default=0.0)
    parser.add_argument("--transl-prior-weight", type=float, default=0.0)
    parser.add_argument("--shape-prior-weight", type=float, default=0.0)
    parser.add_argument("--optimize-betas", action="store_true", default=False)
    parser.add_argument("--respect-hand-valid", dest="respect_hand_valid", action="store_true")
    parser.add_argument(
        "--keep-all-frames",
        dest="respect_hand_valid",
        action="store_false",
        help="Do not drop frames outside Stage 1 frame_keep_3cm; keep them in output for debug.",
    )
    parser.set_defaults(respect_hand_valid=True)
    parser.add_argument("--perturb-init-scale", type=float, default=0.0)
    parser.add_argument("--anchor-root", type=str, default=str(ANCHOR_ROOT))
    parser.add_argument("--cpf-cache-root", type=str, default=None)
    parser.add_argument("--rebuild-cpf-cache", action="store_true", default=False)
    parser.add_argument("--sdf-cache-root", type=str, default=None)
    parser.add_argument("--rebuild-sdf-cache", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--progress", action="store_true", default=False)
    args = parser.parse_args()

    processed_root = Path(args.processed_root).resolve()
    meta = _read_dataset_meta(processed_root)
    mano_dir = Path(
        args.mano_dir
        or meta.get("mano_model_dir")
        or DEFAULT_MANO_DIR
    ).resolve()
    if not (mano_dir / "MANO_RIGHT.pkl").exists():
        raise FileNotFoundError(f"MANO assets not found under {mano_dir}")
    if not Path(args.anchor_root).exists():
        raise FileNotFoundError(f"CPF anchor assets not found under {args.anchor_root}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        print("[warn] CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    sequence = load_sequence(
        processed_root=processed_root,
        meta=meta,
        mano_dir=mano_dir,
        seq_id=args.seq_id,
        side=args.side,
        frame_step=args.frame_step,
        start_frame=args.start_frame,
        end_frame=(None if args.end_frame < 0 else args.end_frame),
        max_frames=args.max_frames,
        contact_thresh=args.contact_thresh,
    )
    if meta.get("mano_config", {}).get("requires_vtemplate") and sequence.mano_vtemplate_path is None:
        raise FileNotFoundError(
            f"meta.json marks dataset as requiring per-sequence vtemplate, but none was found for "
            f"{sequence.seq_id} side={sequence.side}"
        )
    J_regressor, mano_faces = _load_mano_v1_assets(mano_dir)
    mano_faces = np.asarray(mano_faces, dtype=np.int64)

    fitter = SmplxManoFitter(
        device=device,
        mano_dir=mano_dir,
        side=sequence.side,
        flat_hand_mean=sequence.flat_hand_mean,
        mano_vtemplate_path=sequence.mano_vtemplate_path,
        optimize_betas=args.optimize_betas,
        lr=args.lr,
        n_iter=args.n_iter,
        joint_target_weight=args.joint_target_weight,
        vertex_target_weight=args.vertex_target_weight,
        pose_prior_weight=args.pose_prior_weight,
        transl_prior_weight=args.transl_prior_weight,
        shape_prior_weight=args.shape_prior_weight,
        lambda_contact_loss=args.lambda_contact_loss,
        lambda_repulsion_loss=args.lambda_repulsion_loss,
        repulsion_mode=args.repulsion_mode,
        repulsion_query=args.repulsion_query,
        repulsion_threshold=args.repulsion_threshold,
        repulsion_surface_topk=args.repulsion_surface_topk,
        penetration_tol_m=args.penetration_tol_mm / 1000.0,
        penetration_scale_m=args.penetration_scale_mm / 1000.0,
        anchor_root=Path(args.anchor_root),
    )
    cpf_cache_root = Path(args.cpf_cache_root).resolve() if args.cpf_cache_root else _default_cpf_cache_root(processed_root)
    sdf_cache_root = (
        Path(args.sdf_cache_root).resolve()
        if args.sdf_cache_root
        else default_object_sdf_cache_root(REF2DEX_ROOT)
    )

    n_frames = sequence.frame_ids.size
    frame_batch_size = max(1, int(args.frame_batch_size))
    num_betas = int(sequence.mano_betas.shape[1])
    num_hand_points = int(mano_faces.shape[0])
    opt_pose_quat = np.zeros((n_frames, 16, 4), dtype=np.float32)
    opt_pose_quat_world = np.zeros((n_frames, 16, 4), dtype=np.float32)
    opt_global_orient_aa = np.zeros((n_frames, 3), dtype=np.float32)
    opt_hand_pose_aa = np.zeros((n_frames, 15, 3), dtype=np.float32)
    opt_tsl = np.zeros((n_frames, 3), dtype=np.float32)
    opt_betas = np.zeros((n_frames, num_betas), dtype=np.float32)
    opt_verts = np.zeros((n_frames, 778, 3), dtype=np.float32)
    opt_face_centers = np.zeros((n_frames, num_hand_points, 3), dtype=np.float32)
    opt_joints_user = np.zeros((n_frames, 21, 3), dtype=np.float32)
    opt_joints_std = np.zeros((n_frames, 16, 3), dtype=np.float32)
    gt_verts_consistent = np.zeros((n_frames, 778, 3), dtype=np.float32)
    gt_face_centers_consistent = np.zeros((n_frames, num_hand_points, 3), dtype=np.float32)
    gt_joints_std_consistent = np.zeros((n_frames, 16, 3), dtype=np.float32)
    gt_joints_user_consistent = np.zeros((n_frames, 21, 3), dtype=np.float32)
    init_jt_rms = np.zeros((n_frames,), dtype=np.float32)
    final_jt_rms = np.zeros((n_frames,), dtype=np.float32)
    init_vert_rms = np.zeros((n_frames,), dtype=np.float32)
    final_vert_rms = np.zeros((n_frames,), dtype=np.float32)
    contact_ratio = np.zeros((n_frames,), dtype=np.float32)
    final_loss = np.zeros((n_frames,), dtype=np.float32)
    penetration_depth = np.zeros((n_frames,), dtype=np.float32)
    cpf_contact_cache_selected = None
    repulsion_surface_points_world = None
    repulsion_surface_normals_world = None
    object_mesh_for_eval = None

    for start in range(0, n_frames, frame_batch_size):
        end = min(start + frame_batch_size, n_frames)
        gt_recon = fitter.reconstruct_batch(
            sequence.global_orient_aa[start:end],
            sequence.hand_pose_aa[start:end],
            sequence.mano_betas[start:end],
            sequence.mano_translation[start:end],
        )
        gt_verts_consistent[start:end] = gt_recon["verts"]
        gt_joints_std_consistent[start:end] = gt_recon["joints_std"]
        gt_joints_user_consistent[start:end] = gt_recon["joints_user"]
        gt_face_centers_consistent[start:end] = gt_recon["verts"][:, mano_faces].mean(axis=2)

    process_mask = (
        np.asarray(sequence.hand_valid, dtype=bool)
        if args.respect_hand_valid
        else np.ones((n_frames,), dtype=bool)
    )
    skipped_idx = np.where(~process_mask)[0]
    n_skipped = int(skipped_idx.size)
    for t in skipped_idx:
        opt_global_orient_aa[t] = sequence.global_orient_aa[t].cpu().numpy()
        opt_hand_pose_aa[t] = sequence.hand_pose_aa[t].cpu().numpy()
        opt_tsl[t] = sequence.mano_translation[t].cpu().numpy()
        opt_betas[t] = sequence.mano_betas[t].cpu().numpy()
        opt_verts[t] = sequence.mano_vertices_world[t].cpu().numpy()
        opt_face_centers[t] = opt_verts[t][mano_faces].mean(axis=1)
        opt_joints_user[t] = sequence.mano_joints_world_user[t].cpu().numpy()
        opt_joints_std[t] = gt_joints_std_consistent[t]
        opt_pose_quat[t] = np.concatenate(
            [
                _rotvec_to_quat_wxyz(opt_global_orient_aa[t:t + 1]),
                _rotvec_to_quat_wxyz(opt_hand_pose_aa[t]),
            ],
            axis=0,
        )
        opt_pose_quat_world[t] = _compose_world_quats(
            opt_global_orient_aa[t:t + 1], opt_hand_pose_aa[t:t + 1], MANO_PARENTS_STD_16
        )[0]
        init_jt_rms[t] = np.sqrt(
            ((gt_joints_user_consistent[t] - opt_joints_user[t]) ** 2).sum(axis=-1).mean()
        )
        final_jt_rms[t] = init_jt_rms[t]
        init_vert_rms[t] = np.sqrt(
            ((gt_verts_consistent[t] - opt_verts[t]) ** 2).sum(axis=-1).mean()
        )
        final_vert_rms[t] = init_vert_rms[t]
        final_loss[t] = 0.0

    need_obj_verts = args.lambda_contact_loss > 0.0
    need_obj_normals = False
    need_cpf_contact = args.lambda_contact_loss > 0.0
    if args.lambda_repulsion_loss > 0.0:
        if args.repulsion_mode == "sdf_grid":
            if sequence.object_mesh_path is not None and sequence.object_mesh_path.exists():
                fitter.object_sdf = load_or_build_object_sdf_grid(
                    mesh_path=sequence.object_mesh_path,
                    dataset_name=sequence.dataset_name,
                    object_name=sequence.object_name,
                    device=device,
                    cache_root=sdf_cache_root,
                    resolution=args.sdf_grid_resolution,
                    padding_m=args.sdf_grid_padding_m,
                    rebuild=args.rebuild_sdf_cache,
                    progress=args.progress,
                )
                object_mesh_for_eval = trimesh.load(str(sequence.object_mesh_path), process=False, force="mesh")
                print(
                    f"[smplx-fit] sdf_grid repulsion using res={args.sdf_grid_resolution} "
                    f"pad={args.sdf_grid_padding_m:.3f}m from {sequence.object_mesh_path}"
                )
            else:
                print("[warn] object mesh missing; sdf_grid repulsion disabled")
        elif args.repulsion_mode == "mesh_sdf_asset":
            if not args.sdf_ref_root:
                raise RuntimeError("--repulsion-mode mesh_sdf_asset requires --sdf-ref-root")
            if sequence.object_mesh_path is not None and sequence.object_mesh_path.exists():
                MeshSDFfromAsset = _load_mesh_sdf_from_asset_class(Path(args.sdf_ref_root))
                fitter.object_mesh_sdf_asset = MeshSDFfromAsset(
                    str(sequence.object_mesh_path),
                    device=str(device),
                    check_manifold=False,
                )
                object_mesh_for_eval = trimesh.load(str(sequence.object_mesh_path), process=False, force="mesh")
                print(
                    f"[smplx-fit] mesh_sdf_asset repulsion from {sequence.object_mesh_path} "
                    f"(sdf_ref_root={Path(args.sdf_ref_root).resolve()})"
                )
            else:
                print("[warn] object mesh missing; mesh_sdf_asset repulsion disabled")
        elif args.repulsion_mode == "signed_surface":
            if sequence.object_mesh_path is not None and sequence.object_mesh_path.exists():
                object_mesh_for_eval, surface_points_obj, surface_normals_obj = _sample_object_surface_points_normals(
                    sequence.object_mesh_path,
                    num_samples=args.repulsion_surface_samples,
                    seed=args.repulsion_surface_seed,
                )
                obj_root_pose_np_full = sequence.obj_trajectory.cpu().numpy()
                repulsion_surface_points_world = torch.from_numpy(
                    _static_points_obj_to_world(surface_points_obj, obj_root_pose_np_full)
                )
                repulsion_surface_normals_world = torch.from_numpy(
                    _static_normals_obj_to_world(surface_normals_obj, obj_root_pose_np_full)
                )
                print(
                    f"[smplx-fit] signed_surface repulsion using mesh samples "
                    f"{args.repulsion_surface_samples} from {sequence.object_mesh_path}"
                )
            else:
                print(
                    "[warn] object mesh missing; signed_surface repulsion falls back to "
                    "processed object points/normals"
                )
                repulsion_surface_points_world = sequence.obj_verts_world.clone()
                repulsion_surface_normals_world = sequence.obj_normals_world.clone()
        else:
            repulsion_surface_points_world = sequence.obj_verts_world.clone()
            repulsion_surface_normals_world = sequence.obj_normals_world.clone()
    if need_cpf_contact:
        cpf_contact_cache_selected = fitter.load_or_build_cpf_contact_cache(
            sequence=sequence,
            cache_root=cpf_cache_root,
            rebuild=args.rebuild_cpf_cache,
            progress=args.progress,
        )
        contact_ratio[:] = cpf_contact_cache_selected["contact_ratio"].astype(np.float32)
    fit_idx = np.where(process_mask)[0]
    num_fit_batches = (fit_idx.size + frame_batch_size - 1) // frame_batch_size if fit_idx.size > 0 else 0

    for batch_id, start in enumerate(range(0, fit_idx.size, frame_batch_size), start=1):
        batch_idx = fit_idx[start:start + frame_batch_size]
        if args.progress or batch_id == 1 or batch_id == num_fit_batches or batch_id % max(1, num_fit_batches // 10) == 0:
            print(
                f"[smplx-fit] batch {batch_id}/{num_fit_batches} seq={sequence.seq_id} "
                f"side={sequence.side} frames={int(batch_idx[0])}-{int(batch_idx[-1])}"
            )

        batch_idx_t = torch.as_tensor(batch_idx, dtype=torch.long)
        cpf_contact_data_list = [None] * len(batch_idx)
        if need_cpf_contact:
            for local_i, frame_idx in enumerate(batch_idx):
                cpf_contact_data_list[local_i] = {
                    "vertex_contact": cpf_contact_cache_selected["vertex_contact"][frame_idx],
                    "contact_region_id": cpf_contact_cache_selected["contact_region_id"][frame_idx],
                    "anchor_id": cpf_contact_cache_selected["anchor_id"][frame_idx],
                    "anchor_elasti": cpf_contact_cache_selected["anchor_elasti"][frame_idx],
                    "anchor_padding_mask": cpf_contact_cache_selected["anchor_padding_mask"][frame_idx],
                    "contact_ratio": float(cpf_contact_cache_selected["contact_ratio"][frame_idx]),
                }

        result = fitter.fit_batch(
            global_orient_aa_init=sequence.global_orient_aa[batch_idx_t],
            hand_pose_aa_init=sequence.hand_pose_aa[batch_idx_t],
            betas_init=sequence.mano_betas[batch_idx_t],
            transl_init=sequence.mano_translation[batch_idx_t],
            gt_joints_user=sequence.mano_joints_world_user[batch_idx_t],
            gt_verts=sequence.mano_vertices_world[batch_idx_t],
            obj_root_pose=sequence.obj_trajectory[batch_idx_t],
            obj_verts_world=(sequence.obj_verts_world[batch_idx_t] if need_obj_verts else None),
            obj_normals_world=(sequence.obj_normals_world[batch_idx_t] if need_obj_normals else None),
            repulsion_obj_verts_world=(
                repulsion_surface_points_world[batch_idx_t] if repulsion_surface_points_world is not None else None
            ),
            repulsion_obj_normals_world=(
                repulsion_surface_normals_world[batch_idx_t] if repulsion_surface_normals_world is not None else None
            ),
            cpf_contact_data_list=cpf_contact_data_list,
            perturb_init_scale=args.perturb_init_scale,
            progress=args.progress,
        )

        opt_global_orient_aa[batch_idx] = result["opt_hand_global_orient_aa"]
        opt_hand_pose_aa[batch_idx] = result["opt_hand_pose_axis_angle"]
        opt_pose_quat[batch_idx] = result["opt_hand_pose_quat"]
        opt_pose_quat_world[batch_idx] = result["opt_hand_pose_quat_world"]
        opt_tsl[batch_idx] = result["opt_hand_tsl"]
        opt_betas[batch_idx] = result["opt_hand_betas"]
        opt_verts[batch_idx] = result["opt_hand_verts"]
        opt_face_centers[batch_idx] = result["opt_hand_verts"][:, mano_faces].mean(axis=2)
        opt_joints_user[batch_idx] = result["opt_hand_joints_user"]
        opt_joints_std[batch_idx] = result["opt_hand_joints_std"]
        init_jt_rms[batch_idx] = result["init_jt_rms"]
        final_jt_rms[batch_idx] = result["final_jt_rms"]
        init_vert_rms[batch_idx] = result["init_vert_rms"]
        final_vert_rms[batch_idx] = result["final_vert_rms"]
        contact_ratio[batch_idx] = result["contact_ratio"]
        final_loss[batch_idx] = result["final_loss"]

    obj_points_world_np = sequence.obj_verts_world.cpu().numpy()
    obj_normals_world_np = sequence.obj_normals_world.cpu().numpy()
    obj_root_pose_np = sequence.obj_trajectory.cpu().numpy()
    if object_mesh_for_eval is None and sequence.object_mesh_path is not None and sequence.object_mesh_path.exists():
        object_mesh_for_eval = trimesh.load(str(sequence.object_mesh_path), process=False, force="mesh")
    if isinstance(object_mesh_for_eval, trimesh.Trimesh):
        penetration_depth = _compute_penetration_depth_mm(
            hand_verts_world=opt_verts,
            obj_root_pose=obj_root_pose_np,
            object_mesh=object_mesh_for_eval,
        )
    init_normals_world = _face_normals_from_verts(gt_verts_consistent, mano_faces)
    opt_normals_world = _face_normals_from_verts(opt_verts, mano_faces)
    init_face_centers_obj = _points_world_to_obj(gt_face_centers_consistent, obj_root_pose_np)
    init_normals_obj = _normals_world_to_obj(init_normals_world, obj_root_pose_np)
    opt_face_centers_obj = _points_world_to_obj(opt_face_centers, obj_root_pose_np)
    opt_normals_obj = _normals_world_to_obj(opt_normals_world, obj_root_pose_np)
    _, _, obj_to_hand_idx_full_post, obj_to_hand_dist_full_post = _compute_bidirectional_nn_per_frame(
        opt_face_centers,
        obj_points_world_np,
    )
    obj_keep_8cm_postopt_full = obj_to_hand_dist_full_post <= 0.08
    crop_all = _pad_object_crops(
        obj_points_world_full=obj_points_world_np,
        obj_normals_world_full=obj_normals_world_np,
        obj_point_id_full=sequence.obj_point_id,
        obj_root_pose=obj_root_pose_np,
        obj_keep_8cm_preopt_full=sequence.obj_keep_8cm_preopt,
        obj_keep_8cm_postopt_full=obj_keep_8cm_postopt_full,
        obj_to_hand_idx_full=obj_to_hand_idx_full_post,
        obj_to_hand_dist_full=obj_to_hand_dist_full_post,
        opt_hand_face_centers_world=opt_face_centers,
        opt_hand_normals_world=opt_normals_world,
    )

    export_idx = fit_idx if args.respect_hand_valid else np.arange(n_frames, dtype=np.int64)
    export_idx = np.asarray(export_idx, dtype=np.int64)
    n_export = int(export_idx.size)
    opt_messages = [
        ("optimized" if process_mask[int(i)] else "copied_preprocess_frame_keep_false")
        for i in export_idx
    ]
    optimization_success = np.asarray([process_mask[int(i)] for i in export_idx], dtype=bool)

    default_out_root = DEFAULT_OUTPUT_ROOT / f"{sequence.dataset_name}_smplx_cpf"
    base_out_dir = Path(args.output_dir) if args.output_dir else default_out_root
    if base_out_dir.name == sequence.subject_id:
        out_dir = base_out_dir
    else:
        out_dir = base_out_dir / sequence.subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sequence.seq_name}_{sequence.side}.pkl"
    _write_mano_opt_meta(base_out_dir, meta, sequence, mano_dir, args)

    init_pose_quat_world = _compose_world_quats(
        sequence.global_orient_aa.cpu().numpy(),
        sequence.hand_pose_aa.cpu().numpy(),
        MANO_PARENTS_STD_16,
    )

    payload = {
        "seq_id": sequence.seq_id,
        "dataset_name": sequence.dataset_name,
        "subject_id": sequence.subject_id,
        "seq_name": sequence.seq_name,
        "object_name": sequence.object_name,
        "side": sequence.side,
        "raw_frame_id": sequence.frame_ids[export_idx],
        "preprocess_frame_idx": sequence.preprocess_frame_idx[export_idx],
        "frame_ids": sequence.frame_ids[export_idx],
        "source_preprocess_file": str(sequence.npz_path),
        "frame_keep_thresh": 0.03,
        "mano_file": str(sequence.npz_path),
        "object_file": str(sequence.npz_path),
        "object_mesh_path": (
            str(sequence.object_mesh_path)
            if sequence.object_mesh_path is not None else str(sequence.npz_path)
        ),
        "obj_root_pose": obj_root_pose_np[export_idx],
        "obj_points_world_crop": crop_all["obj_points_world_crop"][export_idx],
        "obj_normals_world_crop": crop_all["obj_normals_world_crop"][export_idx],
        "obj_points_obj_crop": crop_all["obj_points_obj_crop"][export_idx],
        "obj_normals_obj_crop": crop_all["obj_normals_obj_crop"][export_idx],
        "obj_crop_valid_mask": crop_all["obj_crop_valid_mask"][export_idx],
        "obj_crop_raw_idx": crop_all["obj_crop_raw_idx"][export_idx],
        "obj_crop_point_id": crop_all["obj_crop_point_id"][export_idx],
        "obj_keep_8cm_preopt": crop_all["obj_keep_8cm_preopt"][export_idx],
        "obj_keep_8cm_postopt": crop_all["obj_keep_8cm_postopt"][export_idx],
        "obj_keep_8cm_final": crop_all["obj_keep_8cm_final"][export_idx],
        "init_hand_face_centers_world": gt_face_centers_consistent[export_idx],
        "init_hand_normals_world": init_normals_world[export_idx],
        "init_hand_verts_world": gt_verts_consistent[export_idx],
        "init_hand_face_centers_obj": init_face_centers_obj[export_idx],
        "init_hand_normals_obj": init_normals_obj[export_idx],
        "opt_hand_face_centers_world": opt_face_centers[export_idx],
        "opt_hand_normals_world": opt_normals_world[export_idx],
        "opt_hand_verts_world": opt_verts[export_idx],
        "opt_hand_joints_world": opt_joints_std[export_idx],
        "opt_hand_face_centers_obj": opt_face_centers_obj[export_idx],
        "opt_hand_normals_obj": opt_normals_obj[export_idx],
        "opt_hand_pose_quat": opt_pose_quat[export_idx],
        "opt_hand_pose_quat_world": opt_pose_quat_world[export_idx],
        "opt_hand_global_orient_aa": opt_global_orient_aa[export_idx],
        "opt_hand_pose_axis_angle": opt_hand_pose_aa[export_idx],
        "opt_hand_tsl": opt_tsl[export_idx],
        "opt_hand_betas": opt_betas[export_idx],
        "hand_point_id": sequence.hand_point_id,
        "hand_cano_points": sequence.hand_cano_points,
        "hand_cano_normals": sequence.hand_cano_normals,
        "hand_finger_id": sequence.hand_finger_id,
        "hand_region_id": sequence.hand_region_id,
        "obj_to_hand_nn_id": crop_all["obj_to_hand_nn_id"][export_idx],
        "obj_to_hand_dist": crop_all["obj_to_hand_dist"][export_idx],
        "obj_to_hand_signed_dist": crop_all["obj_to_hand_signed_dist"][export_idx],
        "hand_to_obj_nn_local_id": crop_all["hand_to_obj_nn_local_id"][export_idx],
        "hand_to_obj_dist": crop_all["hand_to_obj_dist"][export_idx],
        "obj_spring_1cm": crop_all["obj_spring_1cm"][export_idx],
        "hand_spring_1cm": crop_all["hand_spring_1cm"][export_idx],
        "obj_contact_region_3cm": crop_all["obj_contact_region_3cm"][export_idx],
        "obj_context_region_8cm": crop_all["obj_context_region_8cm"][export_idx],
        "contact_ratio": contact_ratio[export_idx],
        "final_loss": final_loss[export_idx],
        "init_jt_rms": init_jt_rms[export_idx],
        "final_jt_rms": final_jt_rms[export_idx],
        "init_vert_rms": init_vert_rms[export_idx],
        "final_vert_rms": final_vert_rms[export_idx],
        "penetration_depth": penetration_depth[export_idx],
        "optimization_success": optimization_success,
        "optimization_message": opt_messages,
        "mano_wrist_pos": sequence.mano_joints_world_user[:, 0].cpu().numpy()[export_idx],
        "mano_wrist_rot_aa": sequence.global_orient_aa.cpu().numpy()[export_idx],
        "mano_pose_quat": init_pose_quat_world[export_idx],
        "mano_betas": sequence.mano_betas.cpu().numpy()[export_idx],
        "mano_translation": sequence.mano_translation.cpu().numpy()[export_idx],
        "mano_joints_world_user": sequence.mano_joints_world_user.cpu().numpy()[export_idx],
        "mano_vertices_world": sequence.mano_vertices_world.cpu().numpy()[export_idx],
        "object_trajectory": sequence.obj_trajectory.cpu().numpy()[export_idx],
        "hand_valid": sequence.hand_valid[export_idx],
        "opt_hand_verts": opt_verts[export_idx],
        "opt_hand_face_centers": opt_face_centers[export_idx],
        "opt_hand_joints_user": opt_joints_user[export_idx],
        "opt_hand_joints_std": opt_joints_std[export_idx],
        "gt_verts_consistent": gt_verts_consistent[export_idx],
        "gt_face_centers_consistent": gt_face_centers_consistent[export_idx],
        "config": {
            "method": "smplx",
            "dataset_name": sequence.dataset_name,
            "flat_hand_mean": sequence.flat_hand_mean,
            "use_pca": False,
            "mano_dir": str(mano_dir),
            "mano_vtemplate_path": str(sequence.mano_vtemplate_path) if sequence.mano_vtemplate_path else None,
            "lr": args.lr,
            "n_iter": args.n_iter,
            "joint_target_weight": args.joint_target_weight,
            "vertex_target_weight": args.vertex_target_weight,
            "lambda_contact_loss": args.lambda_contact_loss,
            "lambda_repulsion_loss": args.lambda_repulsion_loss,
            "repulsion_mode": args.repulsion_mode,
            "repulsion_query": args.repulsion_query,
            "repulsion_threshold": args.repulsion_threshold,
            "repulsion_surface_samples": args.repulsion_surface_samples,
            "repulsion_surface_topk": args.repulsion_surface_topk,
            "penetration_tol_mm": args.penetration_tol_mm,
            "penetration_scale_mm": args.penetration_scale_mm,
            "repulsion_surface_seed": args.repulsion_surface_seed,
            "sdf_grid_resolution": args.sdf_grid_resolution,
            "sdf_grid_padding_m": args.sdf_grid_padding_m,
            "sdf_ref_root": args.sdf_ref_root,
            "pose_prior_weight": args.pose_prior_weight,
            "transl_prior_weight": args.transl_prior_weight,
            "shape_prior_weight": args.shape_prior_weight,
            "optimize_betas": args.optimize_betas,
            "frame_step": args.frame_step,
            "frame_batch_size": args.frame_batch_size,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "frame_keep_thresh": 0.03,
            "contact_thresh": args.contact_thresh,
            "obj_crop_thresh": 0.08,
            "obj_crop_source": "preopt_or_postopt_union",
            "spring_contact_thresh": 0.01,
            "respect_hand_valid": args.respect_hand_valid,
            "perturb_init_scale": args.perturb_init_scale,
            "anchor_root": str(args.anchor_root),
            "cpf_cache_root": str(cpf_cache_root),
            "sdf_cache_root": str(sdf_cache_root),
            "rebuild_cpf_cache": args.rebuild_cpf_cache,
            "rebuild_sdf_cache": args.rebuild_sdf_cache,
            "cpf_contact_cache_version": CPF_CONTACT_CACHE_VERSION,
            "cpf_contact_range_threshold_mm": CPF_CONTACT_RANGE_THRESHOLD_MM,
            "cpf_contact_topk": CPF_CONTACT_TOPK,
            "cpf_contact_elasti_threshold_mm": CPF_CONTACT_ELASTI_THRESHOLD_MM,
            "cpf_contact_elasti_cutoff": CPF_CONTACT_ELASTI_CUTOFF,
            "penetration_depth_unit": "mm",
        },
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"[smplx-fit] saved to {out_path}")
    if n_export > 0:
        print(
            f"[smplx-fit] export_frames={n_export}/{n_frames} "
            f"init_jt_rms mean={init_jt_rms[export_idx].mean() * 1000:.3f} mm, "
            f"final_jt_rms mean={final_jt_rms[export_idx].mean() * 1000:.3f} mm, "
            f"init_vert_rms mean={init_vert_rms[export_idx].mean() * 1000:.3f} mm, "
            f"final_vert_rms mean={final_vert_rms[export_idx].mean() * 1000:.3f} mm, "
            f"skipped={n_skipped}/{n_frames}"
        )
    else:
        print(f"[smplx-fit] no frames exported for {sequence.seq_id} side={sequence.side}")


if __name__ == "__main__":
    main()
