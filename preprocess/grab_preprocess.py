#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAB 数据集预处理脚本（仿造 arctic_preprocess.py 字段规范）
==============================================================

本脚本将 GRAB 原始数据 (`.npz` 序列) 处理成 Ref2Dex 训练所需的 `.npz` 格式，
**字段命名与 `arctic_preprocess.py` 完全一致**——这样下游消费者
（`arctic_mano_cpf_fit.py` / `arctic_mano_toch_fit.py` / 可视化 / 学习模型）
对两个数据集可以共用同一套读取逻辑，只需把数据源切换即可。

================================================================================
与 ARCTIC 字段规范的兼容策略
================================================================================
    - 物体采样点数固定为 2048（与 arctic_preprocess 对齐）。
    - 手部采用 **face-center 采样 1538 点**（与 arctic_preprocess 对齐）。
    - 关节存 16 关节世界四元数 [w, x, y, z]（与 arctic_preprocess 对齐）。
    - 存 1538 维手点 + 法向 + 双向 NN + flow。
    - 接触 valid 掩码采用 1/2/3cm 三阈值（与 arctic_preprocess 对齐）。

================================================================================
GRAB 与 ARCTIC 的关键差异
================================================================================
    1. **无铰接物体**：GRAB 中所有物体都是刚体（box、mug、apple 等），
       不需要 R_z(-angle) 顶部件旋转，直接做全局刚体变换即可。
    2. **MANO 配置不同**：GRAB 需要 `flat_hand_mean=True`，且每条序列 /
       每只手都要注入自己的 `v_template` 才能精确复现 GT。
    3. **时间轴保留完整**：本脚本不再按接触过滤帧；只做可选均匀降采样，
       并通过 `frame_id` 保留原始帧索引。
    4. **直接使用 fullpose**：GRAB 原始数据已提供 45D `fullpose`，
       因此 `*_hand_joint_axis_angle` 直接来自原始 fullpose，而不是再从 PCA 反解。
    5. **无 fitting_err 字段**：GRAB 不提供 ARCTIC 那类逐帧拟合误差，
       因此 flow_valid 仅在最后一帧为 False（没有下一帧）。

================================================================================
输出 .npz 字段规范（与 arctic_preprocess 完全一致）
================================================================================
    {
        "seq_id": str,                            # 序列标识, e.g. "s1/airplane_pass_1"
        "frame_id": (T,) int,

        # ----- 物体（每一帧）-----
        "obj_points": (T, 2048, 3) float,
        "obj_normals": (T, 2048, 3) float,
        "obj_point_id": (2048,) int,
        "obj_root_pose": (T, 4, 4) float,
        "obj_flow": (T, 2048, 3) float,
        "obj_flow_valid": (T, 2048) bool,

        # ----- 右手（每一帧，face-center 采样 1538 点）-----
        "right_hand_points": (T, 1538, 3) float,
        "right_hand_normals": (T, 1538, 3) float,
        "right_hand_point_id": (1538,) int,
        "right_hand_root_pose": (T, 4, 4) float,
        "right_hand_joint_axis_angle": (T, 15, 3) float,
        "right_hand_joint_axis_angle_delta": (T, 15, 3) float,
        "right_hand_joint_quat": (T, 16, 4) float,
        "right_hand_betas": (T, 10) float,
        "right_hand_verts": (T, 778, 3) float,
        "right_hand_finger_id": (1538,) int,
        "right_hand_region_id": (1538,) int,
        "right_hand_flow": (T, 1538, 3) float,
        "right_hand_flow_valid": (T, 1538) bool,
        "right_hand_to_obj_nn_id": (T, 1538) int,
        "right_hand_to_obj_dist": (T, 1538) float,
        "right_hand_min_dist_to_obj": (T,) float,
        "right_hand_valid": (T,) bool,           # 3cm 接触掩码
        "right_hand_valid_2cm": (T,) bool,      # 2cm 接触掩码（CPF 对齐）
        "right_hand_valid_1cm": (T,) bool,      # 1cm 接触掩码（CPF 硬截断对齐）

        # ----- 左手（同上；若原始序列缺失该手，则 hand_* 全部填 0）-----
        "left_hand_*": ...,

        # ----- 共享 -----
        "obj_to_right_hand_nn_id": (T, 2048) int,
        "obj_to_right_hand_dist": (T, 2048) float,
        "obj_to_left_hand_nn_id": (T, 2048) int,
        "obj_to_left_hand_dist": (T, 2048) float,
    }

================================================================================
输出文件策略（与 ARCTIC 对齐：单文件含双手所有字段）
================================================================================
    每条 GRAB 序列输出 1 个 .npz（含 left_hand_* + right_hand_* + obj_*），
    与 arctic_preprocess 输出格式完全一致：
        <output_root>/<subject>/<action>.npz

    - 双手都按真实 `right_*` / `left_*` 槽位写入
    - `--hand` 仅保留命令兼容，不再影响输出语义
    - *_hand_is_mano / *_hand_has_mano_params 不写入 .npz，
      改为写入 meta.json 的 mano_meta 字段

================================================================================
split 划分
================================================================================
    用户要求：所有序列都归到 train 集（与 arctic_preprocess 对齐）。
    不再按物体名划分 test/val。

================================================================================
使用方法
================================================================================
    # 处理右手（默认）
    python grab_preprocess.py --grab-path /path/to/GRAB --mano-path /path/to/mano \\
        --output-root ./processed_data/grab --hand right

    # 处理左手
    python grab_preprocess.py --grab-path /path/to/GRAB --mano-path /path/to/mano \\
        --output-root ./processed_data/grab --hand left

    # 自定义采样数 / 接触阈值
    python grab_preprocess.py --num_obj_points 4096 --contact_distance_thresh 0.025

================================================================================
依赖
================================================================================
    - numpy, scipy, tqdm
    - trimesh（mesh 处理 + 法向）
    - smplx（MANO forward）
    - pytorch

**无需 GRAB 原仓库**：原 repo 的 tools/ObjectModel/parse_npz 只是 numpy
字典访问的便利封装，本脚本直接用 np.load 解析 .npz 即可。
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as op
import pickle
import re
import sys
import traceback
from glob import glob
from typing import Optional

import numpy as np
from tqdm import tqdm

# ============================================================
# numpy legacy shim（smplx / chumpy 需要 np.bool / np.int 等）
# ============================================================
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

import torch
import trimesh
from scipy.spatial import cKDTree
from smplx import MANO


# ============================================================
# 路径配置（可通过 CLI 覆盖）
# ============================================================
REF2DEX_ROOT = op.dirname(op.dirname(op.abspath(__file__)))
# MANO 模型目录（含 MANO_LEFT.pkl / MANO_RIGHT.pkl）
DEFAULT_MANO_MODEL_DIR = op.join(REF2DEX_ROOT, "dataset", "arctic", "data", "body_models", "mano")
# GRAB 原始数据根目录（含 grab/、tools/、tools/object_meshes/contact_meshes/）
DEFAULT_GRAB_ROOT = op.join(REF2DEX_ROOT, "dataset", "GRAB", "data")
# 默认输出根目录
DEFAULT_OUTPUT_ROOT = op.join(REF2DEX_ROOT, "processed_data", "grab")
SHARED_ASSET_ROOT = op.join(REF2DEX_ROOT, "assets", "shared")
SHARED_MANO_ASSET_ROOT = op.join(SHARED_ASSET_ROOT, "mano")
OBJECT_ASSET_ROOT = op.join(REF2DEX_ROOT, "assets", "grab", "objects")

# GRAB 标准 split 划分（与 TOCH 原 repo 一致）
GRAB_TEST_OBJECTS = ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste']
GRAB_VAL_OBJECTS = ['apple', 'toothbrush', 'elephant', 'hand']

# ARCTIC-preprocess 兼容常量
MANO_NUM_JOINTS = 16
MANO_NUM_FACES = 1538
MANO_NUM_HAND_POINTS = MANO_NUM_FACES
# 关节 → finger_id 映射（与 arctic_preprocess 完全一致）
JOINT_TO_FINGER = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
FITTING_ERR_THRESH = 10.0   # 没用上（GRAB 不提供 fitting_err），但保留对齐
CONTACT_DISTANCE_THRESH = 0.03  # 3cm 默认接触阈值
CONTACT_THRESH_2CM = 0.02
CONTACT_THRESH_1CM = 0.01

REGION_PALM = 0
REGION_FINGERTIP = 1
REGION_FINGER_PAD = 2


# ============================================================
# 通用工具（仿造 arctic_preprocess）
# ============================================================
def axis_angle_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues 公式：(..., 3) -> (..., 3, 3)。"""
    angle = torch.norm(axis_angle + 1e-8, p=2, dim=-1, keepdim=True)
    axis = axis_angle / angle
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    row1 = torch.cat([torch.zeros_like(x), -z, y], dim=-1)
    row2 = torch.cat([z, torch.zeros_like(y), -x], dim=-1)
    row3 = torch.cat([-y, x, torch.zeros_like(z)], dim=-1)
    K = torch.stack([row1, row2, row3], dim=-2)
    I = torch.eye(3, device=axis_angle.device).expand(*K.shape[:-2], -1, -1)
    return I + sin.unsqueeze(-1) * K + (1 - cos.unsqueeze(-1)) * (K @ K)


def rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3)。"""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    near_zero = theta < 1e-6
    theta_safe = theta.clone()
    theta_safe[near_zero] = 1.0
    x = (R[..., 2, 1] - R[..., 1, 2]) / (2 * theta_safe)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (2 * theta_safe)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (2 * theta_safe)
    aa = torch.stack([x, y, z], dim=-1) * theta.unsqueeze(-1)
    aa[near_zero] = 0.0
    return aa


def build_SE3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """R: (..., 3, 3)  t: (..., 3) -> (..., 4, 4)。"""
    batch_shape = R.shape[:-2]
    device = R.device
    pose = torch.eye(4, device=device).expand(*batch_shape, 4, 4).contiguous()
    pose[..., :3, :3] = R
    pose[..., :3, 3] = t
    return pose


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Shepperd 方法：(..., 3, 3) -> (..., 4) [w, x, y, z]。"""
    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]

    s0 = torch.sqrt(torch.clamp(m00 + m11 + m22 + 1.0, min=1e-10)) * 2
    q0_w = 0.25 * s0
    q0_x = (m21 - m12) / s0
    q0_y = (m02 - m20) / s0
    q0_z = (m10 - m01) / s0
    q0 = torch.stack([q0_w, q0_x, q0_y, q0_z], dim=-1)

    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-10)) * 2
    q1_w = (m21 - m12) / s1
    q1_x = 0.25 * s1
    q1_y = (m01 + m10) / s1
    q1_z = (m02 + m20) / s1
    q1 = torch.stack([q1_w, q1_x, q1_y, q1_z], dim=-1)

    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-10)) * 2
    q2_w = (m02 - m20) / s2
    q2_x = (m01 + m10) / s2
    q2_y = 0.25 * s2
    q2_z = (m12 + m21) / s2
    q2 = torch.stack([q2_w, q2_x, q2_y, q2_z], dim=-1)

    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-10)) * 2
    q3_w = (m10 - m01) / s3
    q3_x = (m02 + m20) / s3
    q3_y = (m12 + m21) / s3
    q3_z = 0.25 * s3
    q3 = torch.stack([q3_w, q3_x, q3_y, q3_z], dim=-1)

    trace = m00 + m11 + m22
    use_branch0 = trace > 0
    m00_is_max = (m00 >= m11) & (m00 >= m22)
    m11_is_max = (~m00_is_max) & (m11 >= m22)
    q = q0
    sel1 = (~use_branch0) & m00_is_max
    q = torch.where(sel1.unsqueeze(-1), q1, q)
    sel2 = (~use_branch0) & (~m00_is_max) & m11_is_max
    q = torch.where(sel2.unsqueeze(-1), q2, q)
    sel3 = (~use_branch0) & (~m00_is_max) & (~m11_is_max)
    q = torch.where(sel3.unsqueeze(-1), q3, q)
    q = q / torch.norm(q, dim=-1, keepdim=True)
    return q


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """面法向 (F, 3)。"""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    normals = mesh.face_normals.copy()
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(norms, 1e-10, None)
    return normals


def sample_mesh_surface(mesh: trimesh.Trimesh, n_points: int, seed: int = 42):
    """均匀随机表面采样，返回 points / face_idx / barycentric。"""
    np.random.seed(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    vertices = mesh.vertices
    faces = mesh.faces[face_idx]
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    triangles = np.stack([v0, v1, v2], axis=1)
    barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
    barycentric = np.nan_to_num(barycentric, nan=1.0 / 3.0)
    return points, face_idx, barycentric


def interpolate_vertex_attributes(face_idx, barycentric, vert_attr, faces):
    """重心坐标插值。"""
    fv = faces[face_idx]
    attr0 = vert_attr[fv[:, 0]]
    attr1 = vert_attr[fv[:, 1]]
    attr2 = vert_attr[fv[:, 2]]
    bary = barycentric
    return bary[:, 0:1] * attr0 + bary[:, 1:2] * attr1 + bary[:, 2:3] * attr2


def assign_hand_semantics(mano_layer: MANO, is_right: bool) -> tuple:
    """为每个 face 分配 finger_id / region_id（与 arctic_preprocess 一致）。"""
    device = next(mano_layer.parameters()).device
    with torch.no_grad():
        output = mano_layer()
        verts = output.vertices.squeeze(0).cpu().numpy()
        joints = output.joints.squeeze(0).cpu().numpy()
    faces = mano_layer.faces
    face_centers = verts[faces].mean(axis=1)
    num_faces = face_centers.shape[0]
    tree = cKDTree(joints)
    _, idxs = tree.query(face_centers, k=1)

    def joint_to_finger(j_idx):
        return JOINT_TO_FINGER[j_idx] if j_idx < len(JOINT_TO_FINGER) else 0

    finger_id = np.array([joint_to_finger(j) for j in idxs], dtype=np.int32)

    tip_joints = {3, 6, 9, 12, 15}
    palm_joints = {0}
    region_id = np.zeros(num_faces, dtype=np.int32)
    for i in range(num_faces):
        if idxs[i] in tip_joints:
            region_id[i] = REGION_FINGERTIP
        elif idxs[i] in palm_joints:
            region_id[i] = REGION_PALM
        else:
            region_id[i] = REGION_FINGER_PAD
    return finger_id, region_id


def load_object_canonical_mesh(obj_name: str, grab_root: str, unit: str = "m") -> trimesh.Trimesh:
    """加载 GRAB 物体 canonical mesh。

    路径约定：{grab_root}/tools/object_meshes/contact_meshes/{obj_name}.ply

    Args:
        obj_name: 物体名
        grab_root: GRAB 根目录
        unit: mesh 顶点单位
            - 'm'   : 已在米（GRAB 默认就是这个单位，实测物体 mesh abs_max ∈ [0.02, 1.35]）
            - 'mm'  : 毫米，内部 /1000 转米
            - 'auto': 启发式（max>10 视为 mm），不推荐

    经验数据（GRAB 真实物体 .ply）：
        airplane=0.089, apple=0.049, body=1.345, coffeemug=0.058,
        cubelarge=0.060, fryingpan=0.118, ... 全部在米单位。
    """
    p = op.join(grab_root, "tools", "object_meshes", "contact_meshes", f"{obj_name}.ply")
    if not op.exists(p):
        p = op.join(grab_root, "tools", "object_meshes", "contact_meshes", f"{obj_name}.obj")
    if not op.exists(p):
        raise FileNotFoundError(f"GRAB object mesh not found: {obj_name} (tried {p})")
    mesh = trimesh.load(p, process=False)
    # ============================================================
    # 单位处理（实测 GRAB = 米，仿 arctic_preprocess.load_object_mesh）
    # ============================================================
    if unit == "m":
        pass
    elif unit == "mm":
        mesh.vertices = mesh.vertices / 1000.0
    elif unit == "auto":
        if np.abs(mesh.vertices).max() > 10.0:
            mesh.vertices = mesh.vertices / 1000.0
    else:
        raise ValueError(f"Unknown unit {unit!r}; expected 'm', 'mm', or 'auto'")
    return mesh


def compute_hand_joint_quaternions(global_orient, hand_pose, parents) -> torch.Tensor:
    """沿运动学树组合 16 关节世界旋转 -> quat [w, x, y, z]。"""
    T = global_orient.shape[0]
    device = global_orient.device
    R_world = torch.zeros((T, 16, 3, 3), device=device, dtype=global_orient.dtype)
    R_world[:, 0] = axis_angle_to_rotmat(global_orient)
    hand_pose_aa = hand_pose.reshape(T, 15, 3)
    R_local = axis_angle_to_rotmat(hand_pose_aa)
    for j in range(1, 16):
        p = int(parents[j])
        R_world[:, j] = R_world[:, p] @ R_local[:, j - 1]
    return rotmat_to_quat(R_world)


# ============================================================
# GRAB 序列解析（兼容两种路径：有 GRAB tools / 无 GRAB tools）
# ============================================================
class GRABSeqData:
    """GRAB 单条序列的数据封装（直接用 np.load，无外部依赖）。"""

    def __init__(self, npz_path: str):
        self.path = npz_path
        # GRAB .npz 是 pickle dict，用 allow_pickle=True 才能读子对象
        raw = np.load(npz_path, allow_pickle=True)
        self._raw = raw
        # obj_name = 文件名中第一个 "_" 前的部分（GRAB 约定：{obj}_{action}.npz）
        base = op.basename(npz_path).split(".")[0]
        self.obj_name = base.split("_")[0]
        self.n_comps = int(raw.get("n_comps", 24)) if "n_comps" in raw.files else 24
        # n_frames = rhand.params.transl 的第一维
        rhand = raw["rhand"].item()
        self.n_frames = int(rhand["params"]["transl"].shape[0])

    @staticmethod
    def _hand_key(side: str) -> str:
        """GRAB .npz 中手部 key 简称：'right'/'left' → 'rhand'/'lhand'。"""
        return f"{side[0]}hand"

    def get_hand_params(self, side: str) -> dict:
        """返回 {global_orient, hand_pose, transl, fullpose, betas}。"""
        hand = self._raw[self._hand_key(side)].item()
        params = hand["params"]
        betas = hand.get("betas", None)
        if not isinstance(betas, np.ndarray):
            betas = np.zeros((10,), dtype=np.float32)
        else:
            betas = np.asarray(betas, dtype=np.float32)
        return {
            "global_orient": params["global_orient"],   # (T, 3)
            "hand_pose": params["hand_pose"],            # (T, 24) PCA 压缩
            "transl": params["transl"],                  # (T, 3)
            "fullpose": params.get("fullpose", None),
            "betas": betas,
        }

    def get_hand_vtemp_relpath(self, side: str) -> str:
        """返回手部 canonical 顶点相对 `grab_root` 的路径。"""
        hand = self._raw[self._hand_key(side)].item()
        return str(hand["vtemp"])

    def get_object_params(self) -> dict:
        """返回 {global_orient, transl, ...}。"""
        obj = self._raw["object"].item()
        params = obj["params"]
        return {
            "global_orient": params["global_orient"],
            "transl": params["transl"],
        }

    def get_contact_object(self) -> np.ndarray:
        """返回 (T, V_obj) int8 矩阵：每帧每个物点接触的 SMPL-X 身体 part id。

        编码：
            0     = 无接触
            其他  = SMPL-X 身体 part id（如 22=右手腕, 41-55=右手 15 关节, 46=右前臂…）
        """
        if "contact" in self._raw.files:
            c = self._raw["contact"].item()
            if "object" in c:
                return np.asarray(c["object"], dtype=np.int8)
        return None


# ============================================================
# 物体 forward（仿造 arctic_preprocess，无铰接）
# ============================================================
def transform_object_points(
    obj_points_canonical: np.ndarray,
    obj_normals_canonical: np.ndarray,
    face_idx: np.ndarray,
    barycentric: np.ndarray,
    obj_mesh: trimesh.Trimesh,
    global_rot: np.ndarray,
    global_trans: np.ndarray,
    canonical_faces: np.ndarray,
) -> tuple:
    """把物体点从 canonical 变到 world（GRAB 无铰接，无 R_arti）。"""
    R_global = axis_angle_to_rotmat(
        torch.from_numpy(global_rot[None, :])
    ).squeeze(0).numpy()
    verts_posed = (R_global @ obj_mesh.vertices.T).T + global_trans
    posed_points = interpolate_vertex_attributes(
        face_idx, barycentric, verts_posed, canonical_faces
    )
    posed_mesh = trimesh.Trimesh(vertices=verts_posed, faces=canonical_faces, process=False)
    posed_mesh.fix_normals()
    posed_vn = posed_mesh.vertex_normals
    posed_normals = interpolate_vertex_attributes(
        face_idx, barycentric, posed_vn, canonical_faces
    )
    norms = np.linalg.norm(posed_normals, axis=-1, keepdims=True)
    posed_normals = posed_normals / np.clip(norms, 1e-10, None)
    return posed_points, posed_normals


# ============================================================
# GRAB 帧级过滤（仿造 TOCH filter_contact_frames）
# ============================================================
def filter_contact_frames(seq_data: GRABSeqData, side: str) -> np.ndarray:
    """返回 (n_frames,) bool 掩码：True = 该帧手-物接触的物点占比 > 阈值。

    GRAB contact 字段真实编码（实测）：
        contact['object']: (T, V_obj) int8
            0 = 无接触
            其他值 = 接触的 SMPL-X 身体 part id（如 43=右前臂, 55=右中指, ...）

    side='right' → 取 part id ∈ {22, 41..55}（右手腕 + 右手 15 关节）
    side='left'  → 取 part id ∈ {21, 26..40}（左手腕 + 左手 15 关节）

    判定：单帧有接触的物点数 / V_obj > 0.5% → 算接触帧。
    """
    contact_obj = seq_data.get_contact_object()
    n_frames = seq_data.n_frames
    if contact_obj is None:
        return np.ones(n_frames, dtype=bool)
    # SMPL-X 手腕 + 手指关节的 part id 范围
    if side == "right":
        hand_part_ids = set([22] + list(range(41, 56)))
    else:
        hand_part_ids = set([21] + list(range(26, 41)))
    # 物点级 part id 矩阵
    V_obj = contact_obj.shape[1] if contact_obj.ndim == 2 else 1
    is_hand_part = np.isin(contact_obj, list(hand_part_ids))   # (T, V_obj)
    hand_contact_ratio = is_hand_part.sum(axis=1) / float(V_obj)  # (T,)
    return hand_contact_ratio > 5e-3   # >0.5% 物点算接触帧


# ============================================================
# 单条序列处理（核心）
# ============================================================
class GRABPreprocessor:
    """GRAB 预处理器（仿造 ArcticPreprocessor 接口）。"""

    def __init__(
        self,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        num_obj_points: int = 2048,
        device: str = "cpu",
        dt: float = 1.0 / 30.0,
        contact_distance_thresh: float = CONTACT_DISTANCE_THRESH,
        max_frames: Optional[int] = None,
        grab_root: str = DEFAULT_GRAB_ROOT,
        mano_path: str = DEFAULT_MANO_MODEL_DIR,
        hand: str = "right",
        ds_rate: int = 1,
        obj_unit: str = "m",
    ):
        self.output_root = output_root
        self.num_obj_points = num_obj_points
        self.device = "cpu"   # 强制 CPU，与 arctic_preprocess 一致
        self.dt = dt
        self.contact_distance_thresh = float(contact_distance_thresh)
        self.max_frames = max_frames
        self.grab_root = grab_root
        self.hand = hand
        self.ds_rate = max(1, int(ds_rate))
        self.obj_unit = obj_unit

        # 默认 MANO（用于手部语义标签等静态分析）
        print(f"[Preprocessor] Loading default MANO for {hand} hand (flat_hand_mean=True)...")
        self.default_mano = MANO(
            mano_path,
            is_rhand=(hand == "right"),
            use_pca=True,         # GRAB 用 PCA24
            num_pca_comps=24,
            flat_hand_mean=True,
        ).to(self.device)
        # 实际 forward 用的 MANO（按 subject vtemp 注入）—— 运行时构造并 cache
        self._mano_cache = {}    # key: (vtemp_path, is_rhand) -> ManoLayer
        self.mano_path = mano_path
        # 兼容性：保持 self.mano 指向 default
        self.mano = self.default_mano

        # 手部语义标签（cache）
        print(f"[Preprocessor] Computing hand semantics...")
        self.finger_id, self.region_id = assign_hand_semantics(self.default_mano, is_right=(hand == "right"))
        self.faces = self.default_mano.faces.astype(np.int64)
        self.num_hand_points = self.faces.shape[0]
        self.hand_point_id = np.arange(self.num_hand_points, dtype=np.int32)
        self.mano_parents = self.default_mano.parents.detach().cpu().numpy().astype(np.int64)

        # 物体采样缓存
        self._obj_cache = {}
        # GRAB split 划分
        self.splits = {
            "test": GRAB_TEST_OBJECTS,
            "val": GRAB_VAL_OBJECTS,
            "train": [],   # 运行时填
        }
        self._split_of_seq = {}   # path -> "train"/"val"/"test"

    def _get_mano_for_vtemp(self, vtemp_path: str, is_rhand: bool) -> MANO:
        """根据受试者 vtemp 构造（或取 cache）MANO layer。

        GRAB 关键：每受试者手型不同，必须用 v_template 注入，
        否则 forward 出来的顶点是 MANO 平均手型，与受试者 GT 不一致。
        """
        key = (vtemp_path, is_rhand)
        if key not in self._mano_cache:
            v_template = trimesh.load(vtemp_path, process=False).vertices.astype(np.float32)
            m = MANO(
                self.mano_path,
                is_rhand=is_rhand,
                use_pca=True,
                num_pca_comps=24,
                flat_hand_mean=True,
                v_template=v_template,        # ★ 注入受试者手型
            ).to(self.device)
            self._mano_cache[key] = m
        return self._mano_cache[key]

    def _split_for_obj(self, obj_name: str) -> str:
        # 用户要求：全部当 train（与 arctic_preprocess 对齐，不分 test/val）
        return "train"

    def _get_obj_sampling(self, obj_name: str):
        if obj_name not in self._obj_cache:
            mesh = load_object_canonical_mesh(obj_name, self.grab_root, unit=self.obj_unit)
            pts, face_idx, bary = sample_mesh_surface(mesh, self.num_obj_points, seed=42)
            mesh.fix_normals()
            vn = mesh.vertex_normals
            normals = interpolate_vertex_attributes(face_idx, bary, vn, mesh.faces)
            norms = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = normals / np.clip(norms, 1e-10, None)
            self._obj_cache[obj_name] = {
                "mesh": mesh,
                "faces": mesh.faces.copy(),
                "points": pts,
                "normals": normals,
                "point_id": np.arange(self.num_obj_points, dtype=np.int32),
                "face_idx": face_idx,
                "barycentric": bary,
            }
            print(f"[Preprocessor] Cached {obj_name}: {len(mesh.vertices)} verts, "
                  f"{self.num_obj_points} samples")
        return self._obj_cache[obj_name]

    # ---- MANO forward (PCA + flat_hand_mean=True) ----
    def _mano_forward(self, T: int, hand_params: dict) -> tuple:
        """对 T 帧做 MANO forward，返回 (verts (T, 778, 3), joints (T, 16, 3))。"""
        rot = torch.from_numpy(hand_params["global_orient"]).float().to(self.device)
        pose = torch.from_numpy(hand_params["hand_pose"]).float().to(self.device)
        tsl = torch.from_numpy(hand_params["transl"]).float().to(self.device)
        betas_np = hand_params["betas"]
        if betas_np.ndim == 1:
            betas = torch.from_numpy(betas_np).float().unsqueeze(0).expand(T, -1).to(self.device)
        else:
            betas = torch.from_numpy(betas_np).float().to(self.device)
        # 截断 betas 到模型实际 num_betas（GRAB 存 (10,)，MANO 模型可能只有 9）
        actual_num_betas = self.mano.shapedirs.shape[-1]
        if betas.shape[-1] != actual_num_betas:
            betas = betas[..., :min(betas.shape[-1], actual_num_betas)]
            if actual_num_betas > betas.shape[-1]:
                # 模型维度 > betas 维度：补零
                pad = torch.zeros(*betas.shape[:-1], actual_num_betas - betas.shape[-1],
                                 device=betas.device, dtype=betas.dtype)
                betas = torch.cat([betas, pad], dim=-1)
        out = self.mano(global_orient=rot, hand_pose=pose, betas=betas, transl=tsl)
        return out.vertices, out.joints  # (T, 778, 3), (T, 16, 3)

    def _select_frame_ids(self, n_frames: int) -> np.ndarray:
        frame_ids = np.arange(0, n_frames, self.ds_rate, dtype=np.int32)
        if self.max_frames is not None:
            frame_ids = frame_ids[:self.max_frames]
        if frame_ids.size == 0:
            raise ValueError("No frames selected after applying ds_rate / max_frames")
        return frame_ids

    def _decode_hand_pose_axis_angle(self, mano_layer: MANO,
                                     hand_params: dict,
                                     frame_ids: np.ndarray) -> np.ndarray:
        fullpose = hand_params.get("fullpose")
        if isinstance(fullpose, np.ndarray) and fullpose.ndim == 2 and fullpose.shape[1] == 45:
            return np.asarray(fullpose[frame_ids], dtype=np.float32).reshape(-1, 15, 3)

        pose_pca = np.asarray(hand_params["hand_pose"][frame_ids], dtype=np.float32)
        try:
            th_comps = mano_layer.th_comps.detach().cpu().numpy()
            hand_pose_aa = pose_pca @ th_comps.T
        except Exception:
            hand_pose_aa = np.zeros((len(frame_ids), 45), dtype=np.float32)
        return hand_pose_aa.reshape(-1, 15, 3).astype(np.float32)

    def _empty_hand_payload(self, T: int) -> dict:
        return {
            "points": np.zeros((T, self.num_hand_points, 3), dtype=np.float32),
            "normals": np.zeros((T, self.num_hand_points, 3), dtype=np.float32),
            "root_pose": np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
            "joint_axis_angle": np.zeros((T, 15, 3), dtype=np.float32),
            "joint_axis_angle_delta": np.zeros((T, 15, 3), dtype=np.float32),
            "joint_quat": np.zeros((T, 16, 4), dtype=np.float32),
            "betas": np.zeros((T, 10), dtype=np.float32),
            "verts": np.zeros((T, 778, 3), dtype=np.float32),
            "flow": np.zeros((T, self.num_hand_points, 3), dtype=np.float32),
            "flow_valid": np.zeros((T, self.num_hand_points), dtype=bool),
            "to_obj_nn_id": np.zeros((T, self.num_hand_points), dtype=np.int32),
            "to_obj_dist": np.zeros((T, self.num_hand_points), dtype=np.float32),
            "obj_to_hand_nn_id": np.zeros((T, self.num_obj_points), dtype=np.int32),
            "obj_to_hand_dist": np.zeros((T, self.num_obj_points), dtype=np.float32),
            "min_dist_to_obj": np.zeros((T,), dtype=np.float32),
            "valid": np.zeros((T,), dtype=bool),
            "valid_2cm": np.zeros((T,), dtype=bool),
            "valid_1cm": np.zeros((T,), dtype=bool),
            "is_mano": False,
            "has_mano_params": False,
            "vtemplate_path": None,
        }

    def _process_one_hand(self, side: str, seq_data: "GRABSeqData",
                          frame_ids: np.ndarray, obj_points: np.ndarray) -> dict:
        T = len(frame_ids)
        hand_key = GRABSeqData._hand_key(side)
        if hand_key not in seq_data._raw.files:
            return self._empty_hand_payload(T)

        hand_params = seq_data.get_hand_params(side)
        hand_params_sel = {}
        for k, v in hand_params.items():
            if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == seq_data.n_frames:
                hand_params_sel[k] = np.asarray(v[frame_ids], dtype=np.float32)
            else:
                hand_params_sel[k] = v

        vtemp_relpath = seq_data.get_hand_vtemp_relpath(side)
        vtemp_path = op.join(self.grab_root, vtemp_relpath)
        mano_for_seq = self._get_mano_for_vtemp(vtemp_path, is_rhand=(side == "right"))

        old_mano = self.mano
        self.mano = mano_for_seq
        try:
            verts_t, _ = self._mano_forward(T, hand_params_sel)
        finally:
            self.mano = old_mano

        verts = verts_t.detach().cpu().numpy().astype(np.float32)
        face_pts = verts[:, self.faces].mean(axis=2).astype(np.float32)
        normals = np.zeros_like(face_pts)
        for t in range(T):
            normals[t] = compute_face_normals(verts[t], self.faces).astype(np.float32)

        global_orient = np.asarray(hand_params_sel["global_orient"], dtype=np.float32)
        transl = np.asarray(hand_params_sel["transl"], dtype=np.float32)
        R_root = axis_angle_to_rotmat(torch.from_numpy(global_orient)).cpu().numpy()
        root_pose = build_SE3(
            torch.from_numpy(R_root),
            torch.from_numpy(transl).float(),
        ).cpu().numpy().astype(np.float32)

        joint_aa = self._decode_hand_pose_axis_angle(mano_for_seq, hand_params, frame_ids)
        joint_aa_delta = np.zeros_like(joint_aa)
        if T > 1:
            joint_aa_delta[:-1] = joint_aa[1:] - joint_aa[:-1]

        joint_quat = compute_hand_joint_quaternions(
            torch.from_numpy(global_orient).float(),
            torch.from_numpy(joint_aa.reshape(T, 45)).float(),
            self.mano_parents,
        ).cpu().numpy().astype(np.float32)

        betas_raw = hand_params["betas"]
        if betas_raw.ndim == 1:
            betas = np.repeat(betas_raw[None, :], T, axis=0).astype(np.float32)
        else:
            betas = np.asarray(betas_raw[frame_ids], dtype=np.float32)

        hand_flow = np.zeros((T, self.num_hand_points, 3), dtype=np.float32)
        if T > 1:
            hand_flow[:-1] = face_pts[1:] - face_pts[:-1]
        hand_flow_valid = np.ones((T, self.num_hand_points), dtype=bool)
        hand_flow_valid[-1] = False

        hand_to_obj_nn_id = np.zeros((T, self.num_hand_points), dtype=np.int32)
        hand_to_obj_dist = np.zeros((T, self.num_hand_points), dtype=np.float32)
        obj_to_hand_nn_id = np.zeros((T, self.num_obj_points), dtype=np.int32)
        obj_to_hand_dist = np.zeros((T, self.num_obj_points), dtype=np.float32)
        for t in range(T):
            tree_obj = cKDTree(obj_points[t])
            dists_h2o, idxs_h2o = tree_obj.query(face_pts[t], k=1)
            hand_to_obj_nn_id[t] = idxs_h2o
            hand_to_obj_dist[t] = dists_h2o.astype(np.float32)
            tree_hand = cKDTree(face_pts[t])
            dists_o2h, idxs_o2h = tree_hand.query(obj_points[t], k=1)
            obj_to_hand_nn_id[t] = idxs_o2h
            obj_to_hand_dist[t] = dists_o2h.astype(np.float32)

        min_per_frame = hand_to_obj_dist.min(axis=1)
        return {
            "points": face_pts,
            "normals": normals,
            "root_pose": root_pose,
            "joint_axis_angle": joint_aa.astype(np.float32),
            "joint_axis_angle_delta": joint_aa_delta.astype(np.float32),
            "joint_quat": joint_quat,
            "betas": betas,
            "verts": verts,
            "flow": hand_flow,
            "flow_valid": hand_flow_valid,
            "to_obj_nn_id": hand_to_obj_nn_id,
            "to_obj_dist": hand_to_obj_dist,
            "obj_to_hand_nn_id": obj_to_hand_nn_id,
            "obj_to_hand_dist": obj_to_hand_dist,
            "min_dist_to_obj": min_per_frame.astype(np.float32),
            "valid": min_per_frame <= self.contact_distance_thresh,
            "valid_2cm": min_per_frame <= CONTACT_THRESH_2CM,
            "valid_1cm": min_per_frame <= CONTACT_THRESH_1CM,
            "is_mano": True,
            "has_mano_params": True,
            "vtemplate_path": vtemp_relpath,
        }

    def process_sequence(self, seq_path: str) -> dict:
        """处理单条 GRAB 序列并保留完整时间轴。"""
        seq_data = GRABSeqData(seq_path)
        obj_params = seq_data.get_object_params()
        n_frames = seq_data.n_frames
        frame_ids = self._select_frame_ids(n_frames)
        T = len(frame_ids)
        obj_name = seq_data.obj_name

        obj_rot_aa = np.asarray(obj_params["global_orient"][frame_ids], dtype=np.float32)
        obj_trans = np.asarray(obj_params["transl"][frame_ids], dtype=np.float32)
        if np.abs(obj_trans).max() > 5.0:
            obj_trans = obj_trans / 1000.0

        obj_cache = self._get_obj_sampling(obj_name)
        obj_points = np.zeros((T, self.num_obj_points, 3), dtype=np.float32)
        obj_normals = np.zeros((T, self.num_obj_points, 3), dtype=np.float32)
        for t in range(T):
            pts, nmls = transform_object_points(
                obj_cache["points"], obj_cache["normals"],
                obj_cache["face_idx"], obj_cache["barycentric"],
                obj_cache["mesh"],
                obj_rot_aa[t], obj_trans[t],
                obj_cache["faces"],
            )
            obj_points[t] = pts.astype(np.float32)
            obj_normals[t] = nmls.astype(np.float32)

        R_obj = axis_angle_to_rotmat(torch.from_numpy(obj_rot_aa)).cpu().numpy()
        obj_root_pose = build_SE3(
            torch.from_numpy(R_obj),
            torch.from_numpy(obj_trans).float(),
        ).cpu().numpy().astype(np.float32)

        obj_flow = np.zeros((T, self.num_obj_points, 3), dtype=np.float32)
        if T > 1:
            obj_flow[:-1] = obj_points[1:] - obj_points[:-1]
        obj_flow_valid = np.ones((T, self.num_obj_points), dtype=bool)
        obj_flow_valid[-1] = False

        right_data = self._process_one_hand("right", seq_data, frame_ids, obj_points)
        left_data = self._process_one_hand("left", seq_data, frame_ids, obj_points)

        parent_dir = op.basename(op.dirname(seq_path))
        m = re.match(r"^s(\d+)$", parent_dir)
        subj_id = f"s{parent_dir[1:]}" if m else parent_dir
        action_name = op.basename(seq_path).replace(".npz", "")
        seq_id = f"{subj_id}/{action_name}"

        output = {
            "seq_id": seq_id,
            "frame_id": frame_ids.astype(np.int32),
            "obj_points": obj_points,
            "obj_normals": obj_normals,
            "obj_point_id": obj_cache["point_id"],
            "obj_root_pose": obj_root_pose,
            "obj_flow": obj_flow,
            "obj_flow_valid": obj_flow_valid,
            "obj_to_right_hand_nn_id": right_data["obj_to_hand_nn_id"],
            "obj_to_right_hand_dist": right_data["obj_to_hand_dist"],
            "obj_to_left_hand_nn_id": left_data["obj_to_hand_nn_id"],
            "obj_to_left_hand_dist": left_data["obj_to_hand_dist"],

            "right_hand_points": right_data["points"],
            "right_hand_normals": right_data["normals"],
            "right_hand_point_id": self.hand_point_id,
            "right_hand_root_pose": right_data["root_pose"],
            "right_hand_joint_axis_angle": right_data["joint_axis_angle"],
            "right_hand_joint_axis_angle_delta": right_data["joint_axis_angle_delta"],
            "right_hand_joint_quat": right_data["joint_quat"],
            "right_hand_betas": right_data["betas"],
            "right_hand_verts": right_data["verts"],
            "right_hand_finger_id": self.finger_id,
            "right_hand_region_id": self.region_id,
            "right_hand_flow": right_data["flow"],
            "right_hand_flow_valid": right_data["flow_valid"],
            "right_hand_to_obj_nn_id": right_data["to_obj_nn_id"],
            "right_hand_to_obj_dist": right_data["to_obj_dist"],
            "right_hand_min_dist_to_obj": right_data["min_dist_to_obj"],
            "right_hand_valid": right_data["valid"],
            "right_hand_valid_2cm": right_data["valid_2cm"],
            "right_hand_valid_1cm": right_data["valid_1cm"],

            "left_hand_points": left_data["points"],
            "left_hand_normals": left_data["normals"],
            "left_hand_point_id": self.hand_point_id,
            "left_hand_root_pose": left_data["root_pose"],
            "left_hand_joint_axis_angle": left_data["joint_axis_angle"],
            "left_hand_joint_axis_angle_delta": left_data["joint_axis_angle_delta"],
            "left_hand_joint_quat": left_data["joint_quat"],
            "left_hand_betas": left_data["betas"],
            "left_hand_verts": left_data["verts"],
            "left_hand_finger_id": self.finger_id,
            "left_hand_region_id": self.region_id,
            "left_hand_flow": left_data["flow"],
            "left_hand_flow_valid": left_data["flow_valid"],
            "left_hand_to_obj_nn_id": left_data["to_obj_nn_id"],
            "left_hand_to_obj_dist": left_data["to_obj_dist"],
            "left_hand_min_dist_to_obj": left_data["min_dist_to_obj"],
            "left_hand_valid": left_data["valid"],
            "left_hand_valid_2cm": left_data["valid_2cm"],
            "left_hand_valid_1cm": left_data["valid_1cm"],

            "right_hand_is_mano": bool(right_data["is_mano"]),
            "right_hand_has_mano_params": bool(right_data["has_mano_params"]),
            "right_hand_vtemplate_path": right_data["vtemplate_path"],
            "left_hand_is_mano": bool(left_data["is_mano"]),
            "left_hand_has_mano_params": bool(left_data["has_mano_params"]),
            "left_hand_vtemplate_path": left_data["vtemplate_path"],
        }
        return output

    def save_sequence(self, output: dict, seq_path: str) -> str:
        """保存为单个 .npz（含双手所有字段，与 ARCTIC 路径约定一致）。"""
        parent_dir = op.basename(op.dirname(seq_path))
        seq_name = op.basename(seq_path).replace(".npz", "")
        out_dir = op.join(self.output_root, parent_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = op.join(out_dir, f"{seq_name}.npz")

        save_dict = {
            k: v for k, v in output.items()
            if "is_mano" not in k
            and "has_mano_params" not in k
            and not k.endswith("_vtemplate_path")
        }
        np.savez_compressed(out_path, **save_dict)
        return out_path

    def run(self, seq_filter: Optional[str] = None):
        """处理一批 GRAB 序列。

        seq_filter 形如：
            - 'airplane'             → 所有 airplane 开头的序列
            - 'airplane_fly_1'       → 文件名包含 airplane_fly_1 的序列
            - 's1'                   → s1 受试者的所有序列
        """
        all_seqs = sorted(glob(op.join(self.grab_root, "grab", "*", "*.npz")))
        if seq_filter:
            if "/" in seq_filter:
                # 保留兼容：'obj/action'
                obj, action = seq_filter.split("/", 1)
                all_seqs = [s for s in all_seqs
                            if obj in op.basename(s) and action in op.basename(s)]
            else:
                # 单字符串：匹配文件名或父目录
                all_seqs = [s for s in all_seqs
                            if seq_filter in op.basename(s)
                            or seq_filter in op.basename(op.dirname(s))]
        print(f"[Preprocessor] Found {len(all_seqs)} GRAB sequences")

        stats = {"success": 0, "fail": 0, "skipped": 0, "sequences": []}
        mano_meta = {}
        contact_stats = {}
        for seq in tqdm(all_seqs, desc="GRAB preprocessing"):
            try:
                seq_data = GRABSeqData(seq)
                parent_dir = op.basename(op.dirname(seq))
                name = op.basename(seq).replace(".npz", "")
                out_path = op.join(self.output_root, parent_dir, f"{name}.npz")
                if op.exists(out_path):
                    stats["skipped"] += 1
                    continue
                output = self.process_sequence(seq)
                self.save_sequence(output, seq)
                stats["success"] += 1
                stats["sequences"].append(output["seq_id"])
                mano_meta[output["seq_id"]] = {
                    "right_hand_is_mano": bool(output["right_hand_is_mano"]),
                    "right_hand_has_mano_params": bool(output["right_hand_has_mano_params"]),
                    "right_hand_vtemplate_path": output["right_hand_vtemplate_path"],
                    "left_hand_is_mano": bool(output["left_hand_is_mano"]),
                    "left_hand_has_mano_params": bool(output["left_hand_has_mano_params"]),
                    "left_hand_vtemplate_path": output["left_hand_vtemplate_path"],
                }
                contact_stats[output["seq_id"]] = {
                    "num_object_frames": int(len(output["frame_id"])),
                    "num_right_valid_frames": int(output["right_hand_valid"].sum()),
                    "num_left_valid_frames": int(output["left_hand_valid"].sum()),
                    "right_valid_ratio": float(output["right_hand_valid"].mean()),
                    "left_valid_ratio": float(output["left_hand_valid"].mean()),
                    "num_right_valid_frames_2cm": int(output["right_hand_valid_2cm"].sum()),
                    "num_left_valid_frames_2cm": int(output["left_hand_valid_2cm"].sum()),
                    "right_valid_ratio_2cm": float(output["right_hand_valid_2cm"].mean()),
                    "left_valid_ratio_2cm": float(output["left_hand_valid_2cm"].mean()),
                    "num_right_valid_frames_1cm": int(output["right_hand_valid_1cm"].sum()),
                    "num_left_valid_frames_1cm": int(output["left_hand_valid_1cm"].sum()),
                    "right_valid_ratio_1cm": float(output["right_hand_valid_1cm"].mean()),
                    "left_valid_ratio_1cm": float(output["left_hand_valid_1cm"].mean()),
                }
                tqdm.write(f"  OK: {output['seq_id']} (T={len(output['frame_id'])})")
            except Exception as e:
                stats["fail"] += 1
                tqdm.write(f"  FAIL: {seq} - {e}")
                traceback.print_exc()

        # meta.json
        # GRAB dt 用序列真实的 framerate 算（顶层字段，1335 条全 120Hz → dt=1/120）
        fps_set = set()
        for seq in all_seqs[:5]:   # 探 5 条足够（实测全 120Hz）
            raw = np.load(seq, allow_pickle=True)
            if "framerate" in raw.files:
                fps_set.add(int(raw["framerate"]))
        primary_fps = max(fps_set) if fps_set else 120
        meta_dt = (self.ds_rate / primary_fps) if primary_fps > 0 else self.dt
        # 统计实际出现的物体（用户要求：所有数据都当 train）
        all_objects = sorted(set(GRABSeqData(s).obj_name for s in all_seqs))
        all_subjects = sorted(set(op.basename(op.dirname(s)) for s in all_seqs))
        meta = {
            "dataset_name": "grab",
            "data_root": self.grab_root,
            "mano_model_dir": self.mano_path,
            "shared_asset_root": SHARED_ASSET_ROOT,
            "shared_mano_asset_root": SHARED_MANO_ASSET_ROOT,
            "object_asset_root": OBJECT_ASSET_ROOT,
            "output_root": self.output_root,
            "subjects": all_subjects,
            "rigid_objects": all_objects,
            "num_obj_points": self.num_obj_points,
            "num_hand_points": self.num_hand_points,
            "dt": meta_dt,
            "contact_distance_thresh": self.contact_distance_thresh,
            "mano_config": {
                "is_rhand": None,
                "use_pca": False,
                "flat_hand_mean": True,
                "num_shape_coeffs": 10,
                "num_faces": MANO_NUM_FACES,
                "num_joints": MANO_NUM_JOINTS,
                "requires_vtemplate": True,
            },
            "splits": {
                "train": all_objects,
                "test": [],
                "val": [],
                "_note": "用户要求：所有序列都当 train（与 arctic_preprocess 对齐）",
            },
            "num_sequences_processed": stats["success"],
            "num_sequences_failed": stats["fail"],
            "num_sequences_skipped": stats["skipped"],
            "processed_sequences": stats["sequences"],
            "grab_dataset_meta": {
                "framerate_hz": primary_fps,
                "ds_rate": self.ds_rate,
                "n_comps": 24,
                "has_body": True,
                "has_table": True,
                "all_objects": all_objects,
                "all_subjects": all_subjects,
                "motion_intents": sorted({
                    op.basename(s).split("_")[1]
                    for s in all_seqs
                    if len(op.basename(s).split("_")) >= 2
                }),
                "subjects": all_subjects,
            },
            "no_grab_repo_needed": True,
            "mano_meta": mano_meta,
            "extras": {
                "contact_stats": contact_stats,
            },
        }
        meta_path = op.join(self.output_root, "meta.json")
        os.makedirs(self.output_root, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n[Preprocessor] Done. Success: {stats['success']}, "
              f"Fail: {stats['fail']}, Skipped: {stats['skipped']}")
        print(f"[Preprocessor] Meta written to: {meta_path}")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GRAB -> Ref2Dex Preprocessor (compatible with arctic_preprocess fields)")
    parser.add_argument("--grab-path", type=str, default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--mano-path", type=str, default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num_obj_points", type=int, default=2048)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--contact_distance_thresh", type=float, default=CONTACT_DISTANCE_THRESH)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="Deprecated compatibility arg. 双手都会被处理，值仅保留兼容旧命令。")
    parser.add_argument("--ds-rate", type=int, default=1,
                        help="Down-sample rate (every N-th frame kept). 1=keep all.")
    parser.add_argument("--obj-unit", type=str, default="m",
                        choices=["m", "mm", "auto"],
                        help="物体 mesh.obj 顶点单位。GRAB 实测 = 'm'（abs_max 0.02~1.35m）。"
                             "内部如需会 /1000 转米，pipeline 统一 m。")
    parser.add_argument("--seq", type=str, default=None,
                        help="Sequence filter, e.g. 'mug' (object) or 'mug/pass_1' (object/action).")
    args = parser.parse_args()

    preprocessor = GRABPreprocessor(
        output_root=args.output_root,
        num_obj_points=args.num_obj_points,
        dt=args.dt,
        contact_distance_thresh=args.contact_distance_thresh,
        max_frames=(args.max_frames if args.max_frames > 0 else None),
        grab_root=args.grab_path,
        mano_path=args.mano_path,
        hand=args.hand,
        ds_rate=args.ds_rate,
        obj_unit=args.obj_unit,
    )
    preprocessor.run(seq_filter=args.seq)


if __name__ == "__main__":
    main()
