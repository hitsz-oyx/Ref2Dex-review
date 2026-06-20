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
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from manotorch.utils.anchorutils import anchor_load, recover_anchor, recover_anchor_batch
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
from lib.utils.contact import dumped_process_contact_info

DEFAULT_PROCESSED_ROOT = REF2DEX_ROOT / "processed_data" / "arctic"
DEFAULT_MANO_DIR = ARCTIC_DATA_ROOT / "body_models" / "mano"
DEFAULT_OUTPUT_ROOT = REF2DEX_ROOT / "outputs" / "mano_fit"

DEFAULT_CONTACT_THRESHOLD = 0.010

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
    hand_valid: np.ndarray


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
    for suffix in ("_1cm", "_2cm", ""):
        key = f"{side_key}_hand_valid{suffix}"
        if key in data.files:
            return np.asarray(data[key][selected], dtype=bool)
    dist_key = f"{side_key}_hand_to_obj_dist"
    if dist_key in data.files:
        dist = data[dist_key][selected]
        return dist.min(axis=1) <= contact_thresh
    raise KeyError(
        f"NPZ missing valid mask and '{dist_key}', cannot derive contact mask."
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


def _read_dataset_meta(processed_root: Path) -> Dict:
    meta_path = processed_root / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _elasti_fn(x: np.ndarray, range_th: float = 45.0) -> np.ndarray:
    x = x.copy()
    np.putmask(x, x > range_th, range_th)
    res = 0.5 * np.cos((np.pi / range_th) * x) + 0.5
    np.putmask(res, res < 1e-8, 0)
    return res


def _mode_of_regions(assignment: np.ndarray, n_regions: int,
                     weights: Optional[np.ndarray] = None) -> int:
    if weights is None:
        weights = np.ones_like(assignment, dtype=np.float32)
    res = np.zeros((n_regions,), dtype=np.float32)
    for region_id in range(n_regions):
        res[region_id] = np.sum((assignment == region_id).astype(np.float32) * weights)
    return int(np.argmax(res))


def compute_contact_info(
    hand_verts_world: np.ndarray,
    obj_verts_world: np.ndarray,
    hand_palm_vid: np.ndarray,
    merged_vertex_assignment: np.ndarray,
    n_regions: int,
    anchor_pos_world: np.ndarray,
    anchor_mapping: dict,
    range_threshold: float = 10.0,
    n_samples: int = 32,
    elasti_threshold: float = 45.0,
    elasti_cutoff: float = 0.1,
    pad_vertex: bool = True,
    pad_anchor: bool = True,
):
    palm_verts_world = hand_verts_world[hand_palm_vid]
    selected_assignment = merged_vertex_assignment[hand_palm_vid]
    rev_anchor_mapping = {
        region_id: [] for region_id in range(n_regions)
    }
    for anchor_id, region_id in anchor_mapping.items():
        rev_anchor_mapping[region_id].append(anchor_id)

    dist_mat = cdist(obj_verts_world, palm_verts_world) * 1000.0
    order_idx = np.argsort(dist_mat, axis=1)[:, :n_samples]
    vertex_blob = []
    for point_id in range(obj_verts_world.shape[0]):
        dist_vec = dist_mat[point_id, order_idx[point_id]]
        valid_mask = dist_vec < range_threshold
        if not valid_mask.any():
            vertex_blob.append({"contact": 0})
            continue

        masked_dist = dist_vec[valid_mask]
        valid_local_idx = np.where(valid_mask)[0]
        origin_valid_idx = order_idx[point_id, valid_local_idx]
        origin_regions = selected_assignment[origin_valid_idx]
        mode_weight = (range_threshold - masked_dist).astype(np.float32)
        target_region = _mode_of_regions(origin_regions, n_regions, weights=mode_weight)
        anchor_list = rev_anchor_mapping[target_region]
        obj_pt = obj_verts_world[point_id:point_id + 1]
        anchor_pts = anchor_pos_world[anchor_list]
        dist_anchor = cdist(obj_pt, anchor_pts).squeeze(0) * 1000.0
        elasti = _elasti_fn(dist_anchor, range_th=elasti_threshold)
        np.putmask(elasti, elasti < elasti_cutoff, elasti_cutoff)
        vertex_blob.append({
            "contact": 1,
            "region": int(target_region),
            "anchor_id": list(anchor_list),
            "anchor_elasti": elasti.tolist(),
        })

    return dumped_process_contact_info(
        vertex_blob,
        anchor_mapping=anchor_mapping,
        pad_vertex=pad_vertex,
        pad_anchor=pad_anchor,
        elasti_th=0.0,
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
    object_name = seq_name.split("_")[0]
    npz_path = _resolve_processed_npz(processed_root, subject_id, seq_name)
    flat_hand_mean = bool(meta.get("mano_config", {}).get("flat_hand_mean", False))
    dataset_name = str(meta.get("dataset_name", "arctic")).strip().lower() or "arctic"
    mano_vtemplate_path = _resolve_vtemplate_path(meta, seq_id, side_key)
    object_mesh_path = _resolve_object_mesh_path(meta, dataset_name, object_name)

    data = np.load(str(npz_path), allow_pickle=True)
    verts_key = f"{side_key}_hand_verts"
    pose_key = f"{side_key}_hand_joint_axis_angle"
    betas_key = f"{side_key}_hand_betas"
    root_pose_key = f"{side_key}_hand_root_pose"
    obj_root_pose_key = "obj_root_pose"

    all_frame_ids = data.get("frame_id", np.arange(data[verts_key].shape[0]))
    selected = _slice_frame_ids(
        num_frames=len(all_frame_ids),
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        max_frames=max_frames,
    )
    frame_ids = all_frame_ids[selected]

    J_regressor, _ = _load_mano_v1_assets(mano_dir)
    verts_np = data[verts_key][selected].astype(np.float32)
    hand_pose_aa_np = data[pose_key][selected].astype(np.float32)
    betas_np = data[betas_key][selected].astype(np.float32)
    root_pose_np = data[root_pose_key][selected].astype(np.float32)
    obj_root_pose_np = data[obj_root_pose_key][selected].astype(np.float32)
    obj_verts_world_np = data["obj_points"][selected].astype(np.float32)
    obj_normals_world_np = data["obj_normals"][selected].astype(np.float32)
    transl_np = root_pose_np[:, :3, 3].astype(np.float32)
    global_orient_aa_np = _rotmat_to_axis_angle(root_pose_np[:, :3, :3].astype(np.float64))

    tip_ids = MANO_TIP_VERTEX_INDICES[side_key]
    joints_user_np = _verts_to_21_joints_np(verts_np, J_regressor, tip_ids, MANO_JOINT_REORDER)
    hand_valid_np = _resolve_hand_valid(data, side_key, selected, contact_thresh)

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
        hand_valid=hand_valid_np,
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
                 repulsion_query: float,
                 repulsion_threshold: float,
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
        self.repulsion_query = repulsion_query
        self.repulsion_threshold = repulsion_threshold
        self.flat_hand_mean = flat_hand_mean
        self.mano_vtemplate_path = mano_vtemplate_path

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
        anchor_face_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(str(anchor_root))
        self.anchor_face_idx = torch.from_numpy(anchor_face_idx).long().unsqueeze(0).to(device)
        self.anchor_weight = torch.from_numpy(anchor_weight).float().unsqueeze(0).to(device)
        self.merged_vertex_assignment = np.asarray(merged_vertex_assignment, dtype=np.int64)
        self.anchor_mapping = anchor_mapping
        self.n_anchor_regions = int(self.merged_vertex_assignment.max()) + 1
        self.palm_vid = np.loadtxt(str(CPF_ASSETS_DIR / "hand_palm_full.txt"), dtype=int)

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

    def _forward(self,
                 global_orient_aa: torch.Tensor,
                 hand_pose_aa: torch.Tensor,
                 betas: torch.Tensor,
                 transl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = self._align_betas_to_model(betas)
        out = self.mano(
            global_orient=global_orient_aa.unsqueeze(0),
            hand_pose=hand_pose_aa.view(1, 45),
            betas=betas.unsqueeze(0),
            transl=transl.unsqueeze(0),
        )
        verts = out.vertices.squeeze(0)
        joints_std = out.joints.squeeze(0)
        joints_16 = torch.einsum("vj,rv->rj", verts, self.J_regressor)
        tips = verts[self.tip_ids]
        joints_21 = torch.cat([joints_16, tips], dim=0)[MANO_JOINT_REORDER]
        return verts, joints_std, joints_21

    def prepare_cpf_contact_data(
        self,
        gt_verts_world: np.ndarray,
        obj_verts_world: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        anchor_pos_world = recover_anchor(
            gt_verts_world,
            self.anchor_face_idx.squeeze(0).detach().cpu().numpy(),
            self.anchor_weight.squeeze(0).detach().cpu().numpy(),
        )
        vertex_contact, contact_region_id, anchor_id, anchor_elasti, anchor_padding_mask = compute_contact_info(
            hand_verts_world=np.asarray(gt_verts_world, dtype=np.float32),
            obj_verts_world=np.asarray(obj_verts_world, dtype=np.float32),
            hand_palm_vid=self.palm_vid,
            merged_vertex_assignment=self.merged_vertex_assignment,
            n_regions=self.n_anchor_regions,
            anchor_pos_world=np.asarray(anchor_pos_world, dtype=np.float32),
            anchor_mapping=self.anchor_mapping,
        )
        return {
            "vertex_contact": vertex_contact.astype(np.int64),
            "contact_region_id": contact_region_id.astype(np.int64),
            "anchor_id": anchor_id.astype(np.int64),
            "anchor_elasti": anchor_elasti.astype(np.float32),
            "anchor_padding_mask": anchor_padding_mask.astype(np.int64),
            "contact_ratio": float(vertex_contact.mean()),
        }

    def reconstruct(self,
                    global_orient_aa: torch.Tensor,
                    hand_pose_aa: torch.Tensor,
                    betas: torch.Tensor,
                    transl: torch.Tensor) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            verts, joints_std, joints_user = self._forward(
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

    def fit_frame(
        self,
        global_orient_aa_init: torch.Tensor,
        hand_pose_aa_init: torch.Tensor,
        betas_init: torch.Tensor,
        transl_init: torch.Tensor,
        gt_joints_user: torch.Tensor,
        gt_verts: torch.Tensor,
        obj_verts_world: Optional[torch.Tensor] = None,
        obj_normals_world: Optional[torch.Tensor] = None,
        cpf_contact_data: Optional[Dict[str, np.ndarray]] = None,
        perturb_init_scale: float = 0.0,
        progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        device = self.device

        global_orient = global_orient_aa_init.detach().to(device).float().clone()
        hand_pose = hand_pose_aa_init.detach().to(device).float().clone()
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
        obj_verts_world_t = (
            obj_verts_world.detach().to(device).float()
            if obj_verts_world is not None else None
        )
        obj_normals_world_t = (
            obj_normals_world.detach().to(device).float()
            if obj_normals_world is not None else None
        )
        init_global_orient = global_orient.detach().clone()
        init_hand_pose = hand_pose.detach().clone()
        init_transl = transl.detach().clone()
        init_betas = betas.detach().clone()

        indexed_anchor_id = None
        indexed_anchor_elasti = None
        indexed_vertex_id = None
        indexed_elasti_k = None
        obj_contact_verts_t = None
        contact_ratio = 0.0
        if cpf_contact_data is not None:
            vertex_contact_t = torch.from_numpy(cpf_contact_data["vertex_contact"]).long().to(device)
            contact_ratio = float(cpf_contact_data["contact_ratio"])
            anchor_id_t = torch.from_numpy(cpf_contact_data["anchor_id"]).long().to(device)
            anchor_elasti_t = torch.from_numpy(cpf_contact_data["anchor_elasti"]).float().to(device)
            anchor_padding_mask_t = torch.from_numpy(cpf_contact_data["anchor_padding_mask"]).long().to(device)
            if int(vertex_contact_t.sum().item()) > 0 and obj_verts_world_t is not None:
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

        with torch.no_grad():
            init_verts, init_joints_std, init_joints_user = self._forward(
                global_orient, hand_pose, betas, transl
            )
            init_jt_rms = torch.sqrt(((init_joints_user - gt_joints_user) ** 2).sum(dim=-1).mean())
            init_vert_rms = torch.sqrt(((init_verts - gt_verts) ** 2).sum(dim=-1).mean())

        history = []
        for step in range(self.n_iter):
            optimizer.zero_grad()
            pred_verts, pred_joints_std, pred_joints_user = self._forward(
                global_orient, hand_pose, betas, transl
            )
            jt_loss = F.mse_loss(pred_joints_user, gt_joints_user)
            if self.vertex_target_weight > 0.0:
                vert_loss = F.mse_loss(pred_verts, gt_verts)
            else:
                vert_loss = torch.zeros((), device=device)
            if (
                self.lambda_contact_loss > 0.0
                and indexed_anchor_id is not None
                and indexed_anchor_id.numel() > 0
                and obj_contact_verts_t is not None
            ):
                pred_anchor = recover_anchor_batch(pred_verts.unsqueeze(0), self.anchor_face_idx, self.anchor_weight)
                pred_anchor = pred_anchor.squeeze(0)
                anchor_pos = pred_anchor[indexed_anchor_id]
                contact_loss = FieldLoss.contact_loss(
                    anchor_pos,
                    obj_contact_verts_t[indexed_vertex_id],
                    indexed_anchor_elasti,
                    indexed_elasti_k,
                )
            else:
                contact_loss = torch.zeros((), device=device)
            if (
                self.lambda_repulsion_loss > 0.0
                and obj_verts_world_t is not None
                and obj_normals_world_t is not None
            ):
                repulsion_loss = FieldLoss.full_repulsion_loss(
                    pred_verts,
                    obj_verts_world_t,
                    obj_normals_world_t,
                    query=self.repulsion_query,
                    threshold=self.repulsion_threshold,
                )
            else:
                repulsion_loss = torch.zeros((), device=device)
            pose_prior = F.mse_loss(hand_pose, init_hand_pose) if self.pose_prior_weight > 0.0 else torch.zeros((), device=device)
            transl_prior = F.mse_loss(transl, init_transl) if self.transl_prior_weight > 0.0 else torch.zeros((), device=device)
            shape_prior = F.mse_loss(betas, init_betas) if self.shape_prior_weight > 0.0 else torch.zeros((), device=device)
            loss = (
                self.joint_target_weight * jt_loss
                + self.vertex_target_weight * vert_loss
                + self.lambda_contact_loss * contact_loss
                + self.lambda_repulsion_loss * repulsion_loss
                + self.pose_prior_weight * pose_prior
                + self.transl_prior_weight * transl_prior
                + self.shape_prior_weight * shape_prior
            )
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))
            if progress and (step % max(1, self.n_iter // 10) == 0 or step == self.n_iter - 1):
                print(
                    f"    [smplx] step {step:04d} jt={jt_loss.item():.4e} "
                    f"vert={vert_loss.item():.4e} contact={contact_loss.item():.4e} "
                    f"repul={repulsion_loss.item():.4e}"
                )

        with torch.no_grad():
            opt_verts, opt_joints_std, opt_joints_user = self._forward(
                global_orient, hand_pose, betas, transl
            )
            final_jt_rms = torch.sqrt(((opt_joints_user - gt_joints_user) ** 2).sum(dim=-1).mean())
            final_vert_rms = torch.sqrt(((opt_verts - gt_verts) ** 2).sum(dim=-1).mean())

        opt_global_orient_aa = global_orient.detach().cpu().numpy().astype(np.float32)
        opt_hand_pose_aa = hand_pose.detach().cpu().numpy().reshape(15, 3).astype(np.float32)
        opt_betas = betas.detach().cpu().numpy().astype(np.float32)
        opt_transl = transl.detach().cpu().numpy().astype(np.float32)

        local_quat = np.concatenate(
            [
                _rotvec_to_quat_wxyz(opt_global_orient_aa[None]),
                _rotvec_to_quat_wxyz(opt_hand_pose_aa),
            ],
            axis=0,
        )
        world_quat = _compose_world_quats(
            opt_global_orient_aa[None], opt_hand_pose_aa[None], MANO_PARENTS_STD_16
        )[0]

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
            "init_jt_rms": float(init_jt_rms.item()),
            "final_jt_rms": float(final_jt_rms.item()),
            "init_vert_rms": float(init_vert_rms.item()),
            "final_vert_rms": float(final_vert_rms.item()),
            "contact_ratio": contact_ratio,
            "loss_history": np.asarray(history, dtype=np.float32),
        }


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
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--contact-thresh", type=float, default=DEFAULT_CONTACT_THRESHOLD)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mano-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--joint-target-weight", type=float, default=1.0)
    parser.add_argument("--vertex-target-weight", type=float, default=0.0)
    parser.add_argument("--lambda-contact-loss", type=float, default=10.0)
    parser.add_argument("--lambda-repulsion-loss", type=float, default=0.5)
    parser.add_argument("--repulsion-query", type=float, default=0.030)
    parser.add_argument("--repulsion-threshold", type=float, default=0.080)
    parser.add_argument("--pose-prior-weight", type=float, default=0.0)
    parser.add_argument("--transl-prior-weight", type=float, default=0.0)
    parser.add_argument("--shape-prior-weight", type=float, default=0.0)
    parser.add_argument("--optimize-betas", action="store_true", default=False)
    parser.add_argument("--respect-hand-valid", action="store_true", default=False)
    parser.add_argument("--perturb-init-scale", type=float, default=0.0)
    parser.add_argument("--anchor-root", type=str, default=str(ANCHOR_ROOT))
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
        repulsion_query=args.repulsion_query,
        repulsion_threshold=args.repulsion_threshold,
        anchor_root=Path(args.anchor_root),
    )

    n_frames = sequence.frame_ids.size
    opt_pose_quat = np.zeros((n_frames, 16, 4), dtype=np.float32)
    opt_pose_quat_world = np.zeros((n_frames, 16, 4), dtype=np.float32)
    opt_global_orient_aa = np.zeros((n_frames, 3), dtype=np.float32)
    opt_hand_pose_aa = np.zeros((n_frames, 15, 3), dtype=np.float32)
    opt_tsl = np.zeros((n_frames, 3), dtype=np.float32)
    opt_betas = np.zeros((n_frames, 10), dtype=np.float32)
    opt_verts = np.zeros((n_frames, 778, 3), dtype=np.float32)
    opt_face_centers = np.zeros((n_frames, 1538, 3), dtype=np.float32)
    opt_joints_user = np.zeros((n_frames, 21, 3), dtype=np.float32)
    opt_joints_std = np.zeros((n_frames, 16, 3), dtype=np.float32)
    gt_verts_consistent = np.zeros((n_frames, 778, 3), dtype=np.float32)
    gt_face_centers_consistent = np.zeros((n_frames, 1538, 3), dtype=np.float32)
    init_jt_rms = np.zeros((n_frames,), dtype=np.float32)
    final_jt_rms = np.zeros((n_frames,), dtype=np.float32)
    init_vert_rms = np.zeros((n_frames,), dtype=np.float32)
    final_vert_rms = np.zeros((n_frames,), dtype=np.float32)
    contact_ratio = np.zeros((n_frames,), dtype=np.float32)
    final_loss = np.zeros((n_frames,), dtype=np.float32)

    n_skipped = 0
    for t in range(n_frames):
        if args.progress or t % max(1, n_frames // 10) == 0:
            print(
                f"[smplx-fit] frame {t + 1}/{n_frames} seq={sequence.seq_id} "
                f"side={sequence.side} valid={bool(sequence.hand_valid[t])}"
            )

        gt_recon = fitter.reconstruct(
            sequence.global_orient_aa[t],
            sequence.hand_pose_aa[t],
            sequence.mano_betas[t],
            sequence.mano_translation[t],
        )
        gt_verts_consistent[t] = gt_recon["verts"]
        gt_face_centers_consistent[t] = gt_recon["verts"][mano_faces].mean(axis=1)
        cpf_contact_data = fitter.prepare_cpf_contact_data(
            sequence.mano_vertices_world[t].cpu().numpy(),
            sequence.obj_verts_world[t].cpu().numpy(),
        )
        contact_ratio[t] = cpf_contact_data["contact_ratio"]

        if args.respect_hand_valid and not sequence.hand_valid[t]:
            n_skipped += 1
            opt_global_orient_aa[t] = sequence.global_orient_aa[t].cpu().numpy()
            opt_hand_pose_aa[t] = sequence.hand_pose_aa[t].cpu().numpy()
            opt_tsl[t] = sequence.mano_translation[t].cpu().numpy()
            opt_betas[t] = sequence.mano_betas[t].cpu().numpy()
            opt_verts[t] = sequence.mano_vertices_world[t].cpu().numpy()
            opt_face_centers[t] = opt_verts[t][mano_faces].mean(axis=1)
            opt_joints_user[t] = sequence.mano_joints_world_user[t].cpu().numpy()
            opt_joints_std[t] = gt_recon["joints_std"]
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
                ((gt_recon["joints_user"] - opt_joints_user[t]) ** 2).sum(axis=-1).mean()
            )
            final_jt_rms[t] = init_jt_rms[t]
            init_vert_rms[t] = np.sqrt(
                ((gt_recon["verts"] - opt_verts[t]) ** 2).sum(axis=-1).mean()
            )
            final_vert_rms[t] = init_vert_rms[t]
            continue

        result = fitter.fit_frame(
            global_orient_aa_init=sequence.global_orient_aa[t],
            hand_pose_aa_init=sequence.hand_pose_aa[t],
            betas_init=sequence.mano_betas[t],
            transl_init=sequence.mano_translation[t],
            gt_joints_user=sequence.mano_joints_world_user[t],
            gt_verts=sequence.mano_vertices_world[t],
            obj_verts_world=sequence.obj_verts_world[t],
            obj_normals_world=sequence.obj_normals_world[t],
            cpf_contact_data=cpf_contact_data,
            perturb_init_scale=args.perturb_init_scale,
            progress=args.progress,
        )

        opt_global_orient_aa[t] = result["opt_hand_global_orient_aa"]
        opt_hand_pose_aa[t] = result["opt_hand_pose_axis_angle"]
        opt_pose_quat[t] = result["opt_hand_pose_quat"]
        opt_pose_quat_world[t] = result["opt_hand_pose_quat_world"]
        opt_tsl[t] = result["opt_hand_tsl"]
        opt_betas[t] = result["opt_hand_betas"]
        opt_verts[t] = result["opt_hand_verts"]
        opt_face_centers[t] = result["opt_hand_verts"][mano_faces].mean(axis=1)
        opt_joints_user[t] = result["opt_hand_joints_user"]
        opt_joints_std[t] = result["opt_hand_joints_std"]
        init_jt_rms[t] = result["init_jt_rms"]
        final_jt_rms[t] = result["final_jt_rms"]
        init_vert_rms[t] = result["init_vert_rms"]
        final_vert_rms[t] = result["final_vert_rms"]
        contact_ratio[t] = result["contact_ratio"]
        final_loss[t] = result["loss_history"][-1] if result["loss_history"].size > 0 else 0.0

    default_out_root = DEFAULT_OUTPUT_ROOT / f"{sequence.dataset_name}_smplx_cpf"
    base_out_dir = Path(args.output_dir) if args.output_dir else default_out_root
    if base_out_dir.name == sequence.subject_id:
        out_dir = base_out_dir
    else:
        out_dir = base_out_dir / sequence.subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sequence.seq_name}_{sequence.side}.pkl"

    payload = {
        "seq_id": sequence.seq_id,
        "subject_id": sequence.subject_id,
        "seq_name": sequence.seq_name,
        "object_name": sequence.object_name,
        "side": sequence.side,
        "frame_ids": sequence.frame_ids,
        "mano_file": str(sequence.npz_path),
        "object_file": str(sequence.npz_path),
        "object_mesh_path": (
            str(sequence.object_mesh_path)
            if sequence.object_mesh_path is not None else str(sequence.npz_path)
        ),
        "mano_wrist_pos": sequence.mano_joints_world_user[:, 0].cpu().numpy(),
        "mano_wrist_rot_aa": sequence.global_orient_aa.cpu().numpy(),
        "mano_pose_quat": _compose_world_quats(
            sequence.global_orient_aa.cpu().numpy(),
            sequence.hand_pose_aa.cpu().numpy(),
            MANO_PARENTS_STD_16,
        ),
        "mano_betas": sequence.mano_betas.cpu().numpy(),
        "mano_translation": sequence.mano_translation.cpu().numpy(),
        "mano_joints_world_user": sequence.mano_joints_world_user.cpu().numpy(),
        "mano_vertices_world": sequence.mano_vertices_world.cpu().numpy(),
        "object_trajectory": sequence.obj_trajectory.cpu().numpy(),
        "hand_valid": sequence.hand_valid,
        "opt_hand_pose_quat": opt_pose_quat,
        "opt_hand_pose_quat_world": opt_pose_quat_world,
        "opt_hand_global_orient_aa": opt_global_orient_aa,
        "opt_hand_pose_axis_angle": opt_hand_pose_aa,
        "opt_hand_tsl": opt_tsl,
        "opt_hand_betas": opt_betas,
        "opt_hand_verts": opt_verts,
        "opt_hand_face_centers": opt_face_centers,
        "opt_hand_joints_user": opt_joints_user,
        "opt_hand_joints_std": opt_joints_std,
        "penetration_depth": np.zeros((n_frames,), dtype=np.float32),
        "contact_ratio": contact_ratio,
        "final_loss": final_loss,
        "gt_verts_consistent": gt_verts_consistent,
        "gt_face_centers_consistent": gt_face_centers_consistent,
        "init_jt_rms": init_jt_rms,
        "final_jt_rms": final_jt_rms,
        "init_vert_rms": init_vert_rms,
        "final_vert_rms": final_vert_rms,
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
            "repulsion_query": args.repulsion_query,
            "repulsion_threshold": args.repulsion_threshold,
            "pose_prior_weight": args.pose_prior_weight,
            "transl_prior_weight": args.transl_prior_weight,
            "shape_prior_weight": args.shape_prior_weight,
            "optimize_betas": args.optimize_betas,
            "frame_step": args.frame_step,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "contact_thresh": args.contact_thresh,
            "respect_hand_valid": args.respect_hand_valid,
            "perturb_init_scale": args.perturb_init_scale,
            "anchor_root": str(args.anchor_root),
        },
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"[smplx-fit] saved to {out_path}")
    print(
        f"[smplx-fit] init_jt_rms mean={init_jt_rms.mean() * 1000:.3f} mm, "
        f"final_jt_rms mean={final_jt_rms.mean() * 1000:.3f} mm, "
        f"init_vert_rms mean={init_vert_rms.mean() * 1000:.3f} mm, "
        f"final_vert_rms mean={final_vert_rms.mean() * 1000:.3f} mm, "
        f"skipped={n_skipped}/{n_frames}"
    )


if __name__ == "__main__":
    main()
