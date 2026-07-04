#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 Ref2Dex 预处理数据上运行 ContactOpt 风格的手部姿态优化。

功能概述：
    本脚本实现了 ContactOpt 方法的复现，用于优化手部姿态以改善与物体的接触效果。

核心思想：
    1. 接收预处理后的手-物数据（MANO 参数 + 物体点云）
    2. 使用 DeepContact 网络或几何方法预测"理想接触图"
    3. 通过梯度下降优化 MANO 姿态，最小化当前接触与目标接触的差异

两种 MANO 参数模式：
    - contactopt_pca: 原始 ContactOpt 方式，使用 manopth 的 15 维 PCA MANO
      （flat_hand_mean=False，与 ContactOpt 论文一致）
    - native: 使用 smplx.MANO，保持 GRAB 数据原有的 flat_hand_mean=True
      和 per-sequence v_template，避免投影到不同 MANO 空间引入的误差
    - auto: 自动选择，对于 GRAB/有 vtemplate/flat_hand 的数据用 native 模式

与 Ref2Dex 现有 stage2 的区别：
    - 这是一个独立的 baseline，不是对现有 stage2 的替代
    - 核心差异在于使用 ContactOpt 的接触损失函数，而非关节损失

典型用法：
    python contactopt_fit.py --seq-id s1/bowl_pass_1 --contact-source deepcontact
    python contactopt_fit.py --seq-id s1/bowl_pass_1 --contact-source geom --mano-param-mode native
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
# numpy 遗留类型兼容
# ---------------------------------------------------------------------------
# 为旧版 numpy 添加被移除的类型别名，避免调用旧代码时报错
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
import torch.nn.functional as F
from manopth.manolayer import ManoLayer
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle
from smplx import MANO
import trimesh


# ===========================================================================
# 全局路径常量
# ===========================================================================
# Ref2Dex 根目录（脚本向上两级）
REF2DEX_ROOT = Path(__file__).resolve().parents[2]
# ContactOpt 仓库路径（Ref2Dex 同级目录）
DEFAULT_CONTACTOPT_ROOT = REF2DEX_ROOT.parent / "ContactOpt"
# 预处理数据根目录（GRAB 数据）
DEFAULT_PROCESSED_ROOT = REF2DEX_ROOT / "processed_data" / "grab"
# 输出目录
DEFAULT_OUTPUT_ROOT = REF2DEX_ROOT / "processed_data" / "generated" / "mano_fit_contactopt"

# 将 ContactOpt 添加到 Python 路径，以便导入其模块
if str(DEFAULT_CONTACTOPT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_CONTACTOPT_ROOT))

# 导入 ContactOpt 核心组件
import contactopt.diffcontact as diffcontact  # noqa: E402
import contactopt.util as cutil  # noqa: E402
from contactopt.deepcontact_net import DeepContactNet  # noqa: E402


# ===========================================================================
# MANO 模型常量
# ===========================================================================

# MANO 手指尖顶点在 MANO 模型中的索引
# right: 右手索引（拇指、食指、中指、无名指、小指）
# left: 左手索引
MANO_TIP_VERTEX_INDICES = {
    "right": [745, 317, 444, 556, 673],  # 5 个指尖顶点
    "left": [745, 317, 445, 556, 673],
}

# MANO 关节重排索引
# MANO 模型输出的 21 个关节顺序与标准顺序不同，需要重排
# 标准顺序：root, index(3), middle(3), ring(3), pinky(3), thumb(4)
MANO_JOINT_REORDER = [
    0, 13, 14, 15, 16,   # root + pinky base
    1, 2, 3, 17,         # index finger
    4, 5, 6, 18,         # middle finger
    10, 11, 12, 19,      # ring finger
    7, 8, 9, 20,         # thumb
]

# MANO 手掌封闭面片索引
# 用于计算完整的手部法向量（包含手掌区域）
MANO_CLOSE_FACES = np.asarray(
    [
        [92, 38, 122],
        [234, 92, 122],
        [239, 234, 122],
        [279, 239, 122],
        [215, 279, 122],
        [215, 122, 118],
        [215, 118, 117],
        [215, 117, 119],
        [215, 119, 120],
        [215, 120, 108],
        [215, 108, 79],
        [215, 79, 78],
        [215, 78, 121],
        [214, 215, 121],
    ],
    dtype=np.int64,
)


# ===========================================================================
# 数据类
# ===========================================================================

@dataclass
class ProcessedSequence:
    """
    预处理序列数据的数据类。

    存储从 Ref2Dex 预处理 NPZ 文件加载的完整序列信息，包括：
    - 序列元信息（数据集名、ID、物体名等）
    - MANO 模型参数（姿态、形状、位移）
    - 3D 数据（手部顶点、关节、物体点云）
    """
    dataset_name: str           # 数据集名称，如 "grab"
    seq_id: str                 # 序列 ID，格式 "subject_id/seq_name"
    subject_id: str             # 主体 ID
    seq_name: str               # 序列名称
    object_name: str            # 物体名称
    object_mesh_path: Optional[Path]  # 物体网格文件路径
    side: str                   # "right" 或 "left"
    npz_path: Path               # 预处理 NPZ 文件路径
    flat_hand_mean: bool        # MANO 的 flat_hand_mean 参数
    mano_vtemplate_path: Optional[Path]  # 自定义顶点模板路径（GRAB 数据特有）
    frame_ids: np.ndarray       # 帧 ID 数组
    preprocess_frame_idx: np.ndarray  # 预处理时的帧索引映射
    selected_indices: np.ndarray  # 选中的帧索引
    hand_valid: np.ndarray      # 手部有效性标志
    global_orient_aa: np.ndarray  # 全局旋转（轴角表示）
    hand_pose_aa: np.ndarray    # 手部姿态（轴角表示，45 维）
    mano_betas: np.ndarray      # MANO 形状参数（10 维）
    mano_translation: np.ndarray  # MANO 平移向量
    mano_vertices_world: np.ndarray  # MANO 手部顶点（世界坐标系）
    mano_joints_world_user: np.ndarray  # MANO 关节（世界坐标系，用户格式）
    obj_root_pose: np.ndarray   # 物体根节点姿态
    obj_points_world: np.ndarray  # 物体点云（世界坐标系）
    obj_normals_world: np.ndarray  # 物体点云法向量


# ===========================================================================
# 辅助函数
# ===========================================================================

def _first_existing_path(*candidates) -> Optional[Path]:
    """
    返回第一个存在的路径。

    Args:
        *candidates: 可变数量的路径候选

    Returns:
        第一个存在的 Path 对象，或 None（如果都不存在）
    """
    for c in candidates:
        if c is None:
            continue
        p = Path(c) if not isinstance(c, Path) else c
        if p.exists():
            return p
    return None


def _first_npz_key(data: np.lib.npyio.NpzFile, *names: str) -> str:
    """
    返回 NPZ 文件中第一个存在的键。

    用于兼容不同版本的 NPZ 文件键名差异。

    Args:
        data: 已加载的 NPZ 文件对象
        *names: 可能的键名列表

    Returns:
        第一个存在的键名

    Raises:
        KeyError: 如果所有键都不存在
    """
    for name in names:
        if name in data.files:
            return name
    raise KeyError(f"Missing required npz key among: {names}")


def _resolve_processed_npz(processed_root: Path, subject_id: str, seq_name: str) -> Path:
    """
    解析预处理 NPZ 文件的完整路径。

    Args:
        processed_root: 预处理数据根目录
        subject_id: 主体 ID
        seq_name: 序列名称

    Returns:
        NPZ 文件的完整路径

    Raises:
        FileNotFoundError: 如果文件不存在
    """
    npz_path = processed_root / subject_id / f"{seq_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed sequence not found: {npz_path}")
    return npz_path


def _slice_frame_ids(
    num_frames: int,
    start_frame: int,
    end_frame: Optional[int],
    frame_step: int,
    max_frames: int,
) -> np.ndarray:
    """
    根据帧范围和步长选择帧索引。

    Args:
        num_frames: 序列总帧数
        start_frame: 起始帧索引（含）
        end_frame: 结束帧索引（不含），None 表示到末尾
        frame_step: 帧采样步长
        max_frames: 最大帧数，-1 表示不限制

    Returns:
        选中的帧索引数组

    Raises:
        ValueError: 如果参数无效
    """
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


def _ensure_preprocess_frame_idx(num_frames: int, arr: Optional[np.ndarray]) -> np.ndarray:
    """
    确保 preprocess_frame_idx 数组存在且长度正确。

    如果数组为 None，则生成默认的 [0, 1, 2, ..., num_frames-1]。

    Args:
        num_frames: 帧数
        arr: 预处理帧索引数组或 None

    Returns:
        有效的帧索引数组

    Raises:
        ValueError: 如果数组长度不匹配
    """
    if arr is None:
        return np.arange(num_frames, dtype=np.int64)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    if arr.shape[0] != num_frames:
        raise ValueError(
            f"preprocess_frame_idx length mismatch: got {arr.shape[0]}, expected {num_frames}"
        )
    return arr


def _rotmat_to_axis_angle_np(rot_mats: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为轴角表示（NumPy 版本）。

    Args:
        rot_mats: 旋转矩阵数组，形状 (N, 3, 3)

    Returns:
        轴角表示数组，形状 (N, 3)
    """
    rot = torch.from_numpy(np.asarray(rot_mats, dtype=np.float32))
    return matrix_to_axis_angle(rot).detach().cpu().numpy().astype(np.float32)


def _select_betas_array(betas: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """
    根据选中的帧索引选择对应的 betas 参数。

    处理 betas 可能是单帧（1D）或序列（2D）的情况。

    Args:
        betas: MANO 形状参数
        selected: 选中的帧索引

    Returns:
        选中帧对应的 betas 数组
    """
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        # 单帧 betas，复制到所有选中帧
        return np.repeat(betas[None], selected.shape[0], axis=0)
    if betas.ndim == 2:
        if betas.shape[0] == 1:
            # 序列中只有一个 betas，复制
            return np.repeat(betas, selected.shape[0], axis=0)
        # 正常情况：每个帧有独立的 betas
        return betas[selected]
    raise ValueError(f"Unsupported betas shape: {betas.shape}")


def _load_mano_v1_assets(mano_root: Path, side: str = "right") -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 MANO v1 模型的资产文件。

    包括关节回归矩阵和面片数据。

    Args:
        mano_root: MANO 模型根目录
        side: "right" 或 "left"

    Returns:
        (关节回归矩阵, 面片数据) 的元组
    """
    import scipy.sparse as sp

    side_key = str(side).lower()
    pkl_name = "MANO_LEFT.pkl" if side_key == "left" else "MANO_RIGHT.pkl"
    pkl_path = str(mano_root / pkl_name)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    J = data["J_regressor"]
    # 稀疏矩阵转换为密集矩阵
    if sp.issparse(J):
        J = J.toarray()
    faces = data["f"]
    if sp.issparse(faces):
        faces = faces.toarray()
    faces = np.asarray(faces, dtype=np.int64)
    return np.asarray(J, dtype=np.float32), faces


def _verts_to_21_joints_np(
    verts: np.ndarray,
    J_regressor: np.ndarray,
    tip_indices: list[int],
    joint_reorder: list[int],
) -> np.ndarray:
    """
    从顶点计算 21 个 MANO 关节（NumPy 版本）。

    包含 16 个 MANO 关节 + 5 个指尖顶点。

    Args:
        verts: 手部顶点 (..., 778, 3)
        J_regressor: 关节回归矩阵 (16, 778)
        tip_indices: 5 个指尖顶点索引
        joint_reorder: 关节重排索引

    Returns:
        21 个关节坐标 (..., 21, 3)
    """
    # 用回归矩阵计算 16 个关节
    joints_16 = np.einsum("...vj,rv->...rj", verts, J_regressor)
    # 提取 5 个指尖顶点
    tips = verts[..., tip_indices, :]
    # 拼接并重排
    joints_21 = np.concatenate([joints_16, tips], axis=-2)
    return joints_21[..., joint_reorder, :]


def _resolve_object_mesh_path(meta: Dict, dataset_name: str, object_name: str) -> Optional[Path]:
    """
    解析物体网格文件的路径。

    尝试多个可能的路径位置。

    Args:
        meta: 元数据字典
        dataset_name: 数据集名称
        object_name: 物体名称

    Returns:
        物体网格文件路径或 None
    """
    candidates = []
    object_asset_root = meta.get("object_asset_root")
    if object_asset_root:
        candidates.append(Path(object_asset_root) / object_name / "mesh.obj")
    # 尝试 Ref2Dex 仓库中的多个可能位置
    candidates.extend(
        [
            REF2DEX_ROOT / "assets" / dataset_name / "objects" / object_name / "mesh.obj",
            REF2DEX_ROOT / "render" / "assets" / "objects" / object_name / "mesh.obj",
            REF2DEX_ROOT / "preprocess" / "assets" / "objects" / object_name / "mesh.obj",
        ]
    )
    return _first_existing_path(*candidates)


def _resolve_vtemplate_path(meta: Dict, seq_id: str, side_key: str) -> Optional[Path]:
    """
    解析 GRAB 数据自定义顶点模板的路径。

    GRAB 数据集为每个序列提供了个性化的 MANO 顶点模板，
    用于更准确地表示特定人的手部形状。

    Args:
        meta: 元数据字典
        seq_id: 序列 ID
        side_key: "right" 或 "left"

    Returns:
        vtemplate 文件路径或 None
    """
    mano_meta = meta.get("mano_meta", {})
    seq_meta = mano_meta.get(seq_id, {})
    vtemplate_rel = seq_meta.get(f"{side_key}_hand_vtemplate_path")
    if not vtemplate_rel:
        return _resolve_grab_vtemplate_from_raw(meta, seq_id, side_key)
    vtemplate_path = Path(vtemplate_rel)
    if vtemplate_path.is_absolute():
        return vtemplate_path
    data_root = meta.get("data_root")
    if data_root:
        return Path(data_root) / vtemplate_path
    return vtemplate_path


def _candidate_grab_roots(meta: Dict) -> list[Path]:
    """
    获取可能的 GRAB 数据根目录列表。

    从 meta.json 的多个字段中收集可能的路径。

    Args:
        meta: 元数据字典

    Returns:
        可能的 GRAB 根目录列表
    """
    roots = []
    for key in ("source_data_root", "data_root", "grab_root"):
        value = meta.get(key)
        if value:
            roots.append(Path(value))
    # 添加默认路径
    roots.append(REF2DEX_ROOT / "dataset" / "GRAB" / "data")
    # 去重
    deduped = []
    seen = set()
    for root in roots:
        root = root.expanduser()
        marker = str(root)
        if marker not in seen:
            seen.add(marker)
            deduped.append(root)
    return deduped


def _resolve_grab_vtemplate_from_raw(meta: Dict, seq_id: str, side_key: str) -> Optional[Path]:
    """
    从原始 GRAB NPZ 文件中解析 vtemplate 路径。

    Args:
        meta: 元数据字典
        seq_id: 序列 ID
        side_key: "right" 或 "left"

    Returns:
        vtemplate 路径或 None
    """
    if "/" not in seq_id:
        return None
    subject_id, seq_name = seq_id.split("/", 1)
    hand_key = f"{side_key[0]}hand"  # "rhand" 或 "lhand"
    for grab_root in _candidate_grab_roots(meta):
        raw_candidates = [
            grab_root / "grab" / subject_id / f"{seq_name}.npz",
            grab_root / subject_id / f"{seq_name}.npz",
        ]
        for raw_path in raw_candidates:
            if not raw_path.exists():
                continue
            try:
                with np.load(str(raw_path), allow_pickle=True) as raw:
                    if hand_key not in raw.files:
                        continue
                    hand = raw[hand_key].item()
                    vtemplate_rel = hand.get("vtemp")
            except Exception:
                continue
            if not vtemplate_rel:
                continue
            vtemplate_path = Path(str(vtemplate_rel))
            if vtemplate_path.is_absolute():
                return vtemplate_path
            candidate = grab_root / vtemplate_path
            if candidate.exists():
                return candidate
            return candidate
    return None


def _get_sequence_mano_config(meta: Dict, seq_id: str, side_key: str) -> Tuple[bool, Optional[Path]]:
    """
    获取序列的 MANO 配置。

    Args:
        meta: 元数据字典
        seq_id: 序列 ID
        side_key: "right" 或 "left"

    Returns:
        (flat_hand_mean, mano_vtemplate_path) 元组
    """
    mano_config = meta.get("mano_config", {})
    flat_hand_mean = bool(mano_config.get("flat_hand_mean", False))
    mano_vtemplate_path = _resolve_vtemplate_path(meta, seq_id, side_key)
    return flat_hand_mean, mano_vtemplate_path


def _align_betas_to_model(model, betas: torch.Tensor) -> torch.Tensor:
    """
    调整 betas 维度以匹配 MANO 模型要求。

    处理 betas 维度与模型期望不匹配的情况（截断或填充）。

    Args:
        model: MANO 模型
        betas: 输入的 betas 张量

    Returns:
        调整后的 betas 张量
    """
    actual_num_betas = int(model.shapedirs.shape[-1])
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


def _make_smplx_mano(
    mano_root: Path,
    side: str,
    flat_hand_mean: bool,
    mano_vtemplate_path: Optional[Path],
    device: torch.device,
) -> MANO:
    """
    创建 smplx.MANO 模型实例（用于 native 模式）。

    Args:
        mano_root: MANO 模型目录
        side: "right" 或 "left"
        flat_hand_mean: 是否使用 flat hand mean
        mano_vtemplate_path: 自定义顶点模板路径（可选）
        device: 计算设备

    Returns:
        MANO 模型实例
    """
    mano_kwargs = {}
    if mano_vtemplate_path is not None:
        # 加载自定义顶点模板
        v_template = trimesh.load(str(mano_vtemplate_path), process=False).vertices.astype(np.float32)
        mano_kwargs["v_template"] = v_template
    return MANO(
        model_path=str(mano_root),
        is_rhand=(side == "right"),
        use_pca=False,  # native 模式不使用 PCA
        flat_hand_mean=flat_hand_mean,
        **mano_kwargs,
    ).to(device)


def _verts_to_21_joints_torch(
    verts: torch.Tensor,
    J_regressor: torch.Tensor,
    tip_ids: torch.Tensor,
) -> torch.Tensor:
    """
    从顶点计算 21 个 MANO 关节（PyTorch 版本）。

    Args:
        verts: 手部顶点 (B, 778, 3)
        J_regressor: 关节回归矩阵 (16, 778)
        tip_ids: 5 个指尖顶点索引

    Returns:
        21 个关节坐标 (B, 21, 3)
    """
    joints_16 = torch.einsum("bvj,rv->brj", verts, J_regressor)
    tips = verts[:, tip_ids]
    joints_21 = torch.cat([joints_16, tips], dim=1)[:, MANO_JOINT_REORDER]
    return joints_21


# ===========================================================================
# 数据加载
# ===========================================================================

def load_sequence(
    processed_root: Path,
    meta: Dict,
    mano_root: Path,
    seq_id: str,
    side: str,
    frame_step: int,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: int,
    respect_hand_valid: bool,
) -> Tuple[ProcessedSequence, np.ndarray]:
    """
    加载预处理序列数据。

    从 NPZ 文件中读取所有需要的字段，并构建 ProcessedSequence 对象。

    Args:
        processed_root: 预处理数据根目录
        meta: 元数据字典
        mano_root: MANO 模型目录
        seq_id: 序列 ID
        side: "right" 或 "left"
        frame_step: 帧采样步长
        start_frame: 起始帧
        end_frame: 结束帧，None 表示到末尾
        max_frames: 最大帧数
        respect_hand_valid: 是否根据 hand_valid 过滤帧

    Returns:
        (ProcessedSequence 对象, 关节回归矩阵) 元组
    """
    side_key = side.lower()

    # 解析 subject_id 和 seq_name
    subject_id, seq_name = seq_id.split("/", 1)
    npz_path = _resolve_processed_npz(processed_root, subject_id, seq_name)

    # 获取 MANO 配置
    flat_hand_mean, mano_vtemplate_path = _get_sequence_mano_config(meta, seq_id, side_key)
    J_regressor, _ = _load_mano_v1_assets(mano_root, side=side_key)
    tip_ids = MANO_TIP_VERTEX_INDICES[side_key]

    with np.load(str(npz_path), allow_pickle=True) as data:
        # 确定键名（兼容不同版本）
        verts_key = _first_npz_key(data, f"{side_key}_hand_verts_world", f"{side_key}_hand_verts")
        pose_key = _first_npz_key(data, f"{side_key}_hand_joint_axis_angle")
        raw_frame_key = _first_npz_key(data, "raw_frame_id", "frame_id")

        num_pre_frames = int(data[verts_key].shape[0])

        # 计算选中的帧索引
        selected = _slice_frame_ids(
            num_frames=num_pre_frames,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            max_frames=-1,
        )

        # 获取手部有效性标志
        hand_valid_all = np.asarray(data[f"{side_key}_hand_valid"], dtype=bool)
        if hand_valid_all.shape[0] != num_pre_frames:
            raise ValueError(
                f"{side_key}_hand_valid length mismatch: got {hand_valid_all.shape[0]}, expected {num_pre_frames}"
            )

        # 根据 hand_valid 过滤帧
        if respect_hand_valid:
            selected = selected[hand_valid_all[selected]]
        if max_frames > 0:
            selected = selected[:max_frames]
        if selected.size == 0:
            raise ValueError("No frames selected after applying frame filters and hand_valid mask")

        # 获取帧 ID
        frame_ids = np.asarray(data[raw_frame_key], dtype=np.int64)[selected]
        preprocess_frame_idx = _ensure_preprocess_frame_idx(
            num_pre_frames,
            data["preprocess_frame_idx"] if "preprocess_frame_idx" in data.files else None,
        )[selected]

        # 获取数据集和物体名称
        dataset_name = str(
            data["dataset_name"].item()
            if "dataset_name" in data.files
            else meta.get("dataset_name", "grab")
        ).strip().lower()
        object_name = str(
            data["object_name"].item()
            if "object_name" in data.files
            else seq_name.split("_")[0]
        )
        object_mesh_path = _resolve_object_mesh_path(meta, dataset_name, object_name)

        # 获取手部数据
        mano_vertices_world = np.asarray(data[verts_key][selected], dtype=np.float32)

        # 获取根节点姿态
        root_pose = (
            np.asarray(data[f"{side_key}_hand_root_pose"][selected], dtype=np.float32)
            if f"{side_key}_hand_root_pose" in data.files
            else None
        )

        # 获取全局旋转
        if f"{side_key}_hand_global_orient_aa" in data.files:
            global_orient_aa = np.asarray(data[f"{side_key}_hand_global_orient_aa"][selected], dtype=np.float32)
        elif root_pose is not None:
            global_orient_aa = _rotmat_to_axis_angle_np(root_pose[:, :3, :3])
        else:
            raise KeyError(
                f"Missing {side_key}_hand_global_orient_aa and {side_key}_hand_root_pose in {npz_path}"
            )

        # 获取手部姿态
        hand_pose_aa = np.asarray(data[pose_key][selected], dtype=np.float32)

        # 获取平移
        if f"{side_key}_hand_translation" in data.files:
            mano_translation = np.asarray(data[f"{side_key}_hand_translation"][selected], dtype=np.float32)
        elif root_pose is not None:
            mano_translation = root_pose[:, :3, 3].astype(np.float32)
        else:
            raise KeyError(
                f"Missing {side_key}_hand_translation and {side_key}_hand_root_pose in {npz_path}"
            )

        # 获取形状参数
        mano_betas = _select_betas_array(np.asarray(data[f"{side_key}_hand_betas"]), selected)

        # 获取物体数据
        obj_root_pose = np.asarray(data["obj_root_pose"][selected], dtype=np.float32)
        obj_points_key = _first_npz_key(data, "obj_points_world", "obj_points")
        obj_normals_key = _first_npz_key(data, "obj_normals_world", "obj_normals")
        obj_points_world = np.asarray(data[obj_points_key][selected], dtype=np.float32)
        obj_normals_world = np.asarray(data[obj_normals_key][selected], dtype=np.float32)
        hand_valid = hand_valid_all[selected]

        # 计算关节
        mano_joints_world_user = _verts_to_21_joints_np(
            mano_vertices_world,
            J_regressor,
            tip_ids,
            MANO_JOINT_REORDER,
        ).astype(np.float32)

    # 构建序列对象
    sequence = ProcessedSequence(
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
        hand_valid=hand_valid,
        global_orient_aa=global_orient_aa,
        hand_pose_aa=hand_pose_aa,
        mano_betas=mano_betas,
        mano_translation=mano_translation,
        mano_vertices_world=mano_vertices_world,
        mano_joints_world_user=mano_joints_world_user,
        obj_root_pose=obj_root_pose,
        obj_points_world=obj_points_world,
        obj_normals_world=obj_normals_world,
    )
    return sequence, J_regressor


# ===========================================================================
# 设备相关
# ===========================================================================

def _device_from_arg(device_arg: str) -> torch.device:
    """
    根据参数解析计算设备。

    Args:
        device_arg: 设备字符串 ("cuda", "cpu", 等)

    Returns:
        torch.device 对象
    """
    device = torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        print("[warn] CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")
    return device


# ===========================================================================
# 几何计算
# ===========================================================================

def _make_hand_tform(transl: torch.Tensor) -> torch.Tensor:
    """
    创建手部变换矩阵（仅平移）。

    Args:
        transl: 平移向量 (B, 3)

    Returns:
        4x4 变换矩阵 (B, 4, 4)
    """
    tform = torch.eye(4, device=transl.device, dtype=torch.float32).unsqueeze(0).repeat(transl.shape[0], 1, 1)
    tform[:, :3, 3] = transl
    return tform


def _make_hand_faces(mano_faces: np.ndarray, closed: bool, device: torch.device) -> torch.Tensor:
    """
    创建手部面片数组。

    Args:
        mano_faces: MANO 模型基础面片
        closed: 是否添加手掌封闭面片
        device: 计算设备

    Returns:
        面片数组
    """
    faces = mano_faces
    if closed:
        faces = np.concatenate([faces, MANO_CLOSE_FACES], axis=0)
    return torch.as_tensor(faces, dtype=torch.long, device=device)


def _compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    计算顶点法向量。

    使用 PyTorch3D 的 Meshes 结构计算面法向量并加权平均到顶点。

    Args:
        verts: 顶点坐标 (B, V, 3)
        faces: 面片索引 (F, 3)

    Returns:
        顶点法向量 (B, V, 3)，已归一化
    """
    batch_size = verts.shape[0]
    mesh = Meshes(verts=verts, faces=faces.unsqueeze(0).repeat(batch_size, 1, 1))
    normals = mesh.verts_normals_padded()
    return F.normalize(normals, dim=2, eps=1e-12)


def _compute_hand_object_features(
    hand_verts: torch.Tensor,
    hand_joints: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_normals: torch.Tensor,
    hand_faces_closed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算手-物几何特征。

    这些特征用于 DeepContact 网络的输入。

    每个顶点的特征包括：
    - one_hot: 全 1 向量（占位）
    - 到质心的法向投影
    - 到最近点的距离
    - 沿法向的距离
    - 到所有关节的距离

    Args:
        hand_verts: 手部顶点 (B, 778, 3)
        hand_joints: 手部关节 (B, 21, 3)
        obj_verts: 物体顶点 (B, N, 3)
        obj_normals: 物体法向量 (B, N, 3)
        hand_faces_closed: 手部封闭面片

    Returns:
        (hand_feats, obj_feats) 元组，每个特征维度是 21
    """
    # 计算手部法向量
    hand_normals = _compute_vertex_normals(hand_verts, hand_faces_closed)

    # 找到最近邻点
    _, _, obj_nearest = knn_points(obj_verts, hand_verts, K=1, return_nn=True)
    _, _, hand_nearest = knn_points(hand_verts, obj_verts, K=1, return_nn=True)
    obj_nearest = obj_nearest[:, :, 0, :]
    hand_nearest = hand_nearest[:, :, 0, :]

    # 计算质心
    hand_centroid = hand_verts.mean(dim=1, keepdim=True)
    obj_centroid = obj_verts.mean(dim=1, keepdim=True)

    # 手部特征
    hand_one_hot = torch.ones((hand_verts.shape[0], hand_verts.shape[1], 1), device=hand_verts.device)
    hand_vec_to_closest = hand_nearest - hand_verts
    hand_dist_to_closest = torch.linalg.norm(hand_vec_to_closest, dim=2, keepdim=True)
    hand_dist_along_normal = torch.sum(hand_vec_to_closest * hand_normals, dim=2, keepdim=True)
    hand_dist_to_joint = torch.linalg.norm(hand_verts[:, :, None, :] - hand_joints[:, None, :, :], dim=3)
    hand_dot_to_centroid = torch.sum((hand_verts - obj_centroid) * hand_normals, dim=2, keepdim=True)

    # 物体特征
    obj_one_hot = torch.zeros((obj_verts.shape[0], obj_verts.shape[1], 1), device=obj_verts.device)
    obj_vec_to_closest = obj_nearest - obj_verts
    obj_dist_to_closest = torch.linalg.norm(obj_vec_to_closest, dim=2, keepdim=True)
    obj_dist_along_normal = torch.sum(obj_vec_to_closest * obj_normals, dim=2, keepdim=True)
    obj_dist_to_joint = torch.linalg.norm(obj_verts[:, :, None, :] - hand_joints[:, None, :, :], dim=3)
    obj_dot_to_centroid = torch.sum((obj_verts - hand_centroid) * obj_normals, dim=2, keepdim=True)

    # 拼接特征
    hand_feats = torch.cat(
        [hand_one_hot, hand_dot_to_centroid, hand_dist_to_closest, hand_dist_along_normal, hand_dist_to_joint],
        dim=2,
    )
    obj_feats = torch.cat(
        [obj_one_hot, obj_dot_to_centroid, obj_dist_to_closest, obj_dist_along_normal, obj_dist_to_joint],
        dim=2,
    )
    return hand_feats, obj_feats


# ===========================================================================
# PCA MANO 初始化（ContactOpt 原始方式）
# ===========================================================================

def _fit_contactopt_pca_pose(
    sequence: ProcessedSequence,
    mano_root: Path,
    device: torch.device,
    batch_size: int,
    n_iter: int,
    lr: float,
) -> Dict[str, np.ndarray]:
    """
    将 GRAB 的 full axis-angle MANO 参数转换为 ContactOpt 的 PCA 格式。

    由于 ContactOpt 使用 15 维 PCA MANO，而 GRAB 使用 45 维 axis-angle，
    需要进行参数转换和优化拟合。

    流程：
    1. 用 GRAB 参数（前向传播）获取目标关节位置
    2. 用 PCA MANO 优化，使输出关节逼近目标

    Args:
        sequence: 序列数据
        mano_root: MANO 模型目录
        device: 计算设备
        batch_size: 批大小
        n_iter: 优化迭代次数
        lr: 学习率

    Returns:
        包含优化结果的字典
    """
    # 创建三个 MANO 模型用于转换
    # 1. src_model: 原始 45 维 axis-angle（flat_hand_mean=True，与 GRAB 一致）
    src_model = ManoLayer(
        mano_root=str(mano_root),
        joint_rot_mode="axisang",
        use_pca=False,
        center_idx=None,
        flat_hand_mean=True,
        side="right",
    ).to(device)

    # 2. pca45_model: 45 维 PCA（用于投影）
    pca45_model = ManoLayer(
        mano_root=str(mano_root),
        use_pca=True,
        ncomps=45,
        flat_hand_mean=False,
        side="right",
    ).to(device)

    # 3. pca15_model: 15 维 PCA（最终使用的模型）
    pca15_model = ManoLayer(
        mano_root=str(mano_root),
        use_pca=True,
        ncomps=15,
        flat_hand_mean=False,
        side="right",
    ).to(device)

    T = sequence.frame_ids.shape[0]
    out_pose18 = np.zeros((T, 18), dtype=np.float32)  # 输出 PCA 姿态
    src_vert_err_mm = np.zeros((T,), dtype=np.float32)  # 原始模型的顶点误差
    pca_vert_err_mm = np.zeros((T,), dtype=np.float32)  # PCA 模型的顶点误差
    init_verts_world = np.zeros_like(sequence.mano_vertices_world)  # 初始化顶点
    init_joints_model = np.zeros_like(sequence.mano_joints_world_user)  # 初始化关节

    # 转换为张量
    global_orient = torch.from_numpy(sequence.global_orient_aa).to(device=device, dtype=torch.float32)
    hand_pose = torch.from_numpy(sequence.hand_pose_aa.reshape(T, -1)).to(device=device, dtype=torch.float32)
    betas = torch.from_numpy(sequence.mano_betas).to(device=device, dtype=torch.float32)
    transl = torch.from_numpy(sequence.mano_translation).to(device=device, dtype=torch.float32)
    gt_verts = torch.from_numpy(sequence.mano_vertices_world).to(device=device, dtype=torch.float32)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        pose48 = torch.cat([global_orient[start:end], hand_pose[start:end]], dim=1)
        beta_batch = betas[start:end]
        tform = _make_hand_tform(transl[start:end])

        # 计算原始模型的顶点误差
        with torch.no_grad():
            src_verts_world, _ = cutil.forward_mano(src_model, pose48, beta_batch, [tform])
            src_vert_err_mm[start:end] = (
                torch.linalg.norm(src_verts_world - gt_verts[start:end], dim=2).mean(dim=1) * 1000.0
            ).detach().cpu().numpy()

        # 转换为 PCA 参数
        full_axang = pose48.detach().clone()
        full_axang[:, 3:] -= pca45_model.th_hands_mean
        pca_mat = pca45_model.th_selected_comps.T
        pca_shape = full_axang[:, 3:].mm(pca_mat)

        # 创建优化变量
        pca_in = torch.zeros((end - start, 18), dtype=torch.float32, device=device)
        pca_in[:, :3] = pose48[:, :3]  # 全局旋转保持不变
        pca_in[:, 3:] = pca_shape[:, :15]  # PCA 姿态取前 15 维
        pca_in.requires_grad_(True)

        # 优化
        optimizer = torch.optim.Adam([pca_in], lr=lr, amsgrad=True)
        loss_fn = torch.nn.L1Loss()
        with torch.no_grad():
            target_joints_local = cutil.forward_mano(src_model, pose48, beta_batch, [])[1]

        for _ in range(n_iter):
            optimizer.zero_grad()
            _, hand_joints = cutil.forward_mano(pca15_model, pca_in, beta_batch, [])
            loss = loss_fn(hand_joints, target_joints_local)
            loss.backward()
            optimizer.step()

        # 保存结果
        with torch.no_grad():
            verts_world, joints_world = cutil.forward_mano(pca15_model, pca_in, beta_batch, [tform])
            pca_vert_err_mm[start:end] = (
                torch.linalg.norm(verts_world - gt_verts[start:end], dim=2).mean(dim=1) * 1000.0
            ).detach().cpu().numpy()
            out_pose18[start:end] = pca_in.detach().cpu().numpy()
            init_verts_world[start:end] = verts_world.detach().cpu().numpy()
            init_joints_model[start:end] = joints_world.detach().cpu().numpy()

    return {
        "pose18": out_pose18,
        "src_vert_err_mm": src_vert_err_mm,
        "pca_vert_err_mm": pca_vert_err_mm,
        "init_verts_world": init_verts_world,
        "init_joints_model": init_joints_model,
    }


# ===========================================================================
# Native MANO 前向传播和初始化
# ===========================================================================

def _forward_native_mano(
    mano_model: MANO,
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    J_regressor: torch.Tensor,
    tip_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Native MANO 模型的前向传播。

    Args:
        mano_model: smplx.MANO 模型
        global_orient: 全局旋转 (B, 3)
        hand_pose: 手部姿态 (B, 45)
        betas: 形状参数 (B, 10)
        transl: 平移 (B, 3)
        J_regressor: 关节回归矩阵
        tip_ids: 指尖索引

    Returns:
        (vertices, joints_21) 元组
    """
    if hand_pose.ndim == 3:
        hand_pose = hand_pose.reshape(hand_pose.shape[0], 45)
    betas = _align_betas_to_model(mano_model, betas)
    out = mano_model(
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        transl=transl,
    )
    verts = out.vertices
    joints_21 = _verts_to_21_joints_torch(verts, J_regressor, tip_ids)
    return verts, joints_21


def _init_native_pose(
    sequence: ProcessedSequence,
    mano_model: MANO,
    J_regressor: torch.Tensor,
    tip_ids: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    初始化 Native MANO 参数。

    由于 native 模式直接使用原始参数，不需要转换，
    只需要做一次前向传播获取顶点。

    Args:
        sequence: 序列数据
        mano_model: MANO 模型
        J_regressor: 关节回归矩阵
        tip_ids: 指尖索引
        device: 计算设备
        batch_size: 批大小

    Returns:
        包含初始化结果的字典
    """
    T = sequence.frame_ids.shape[0]
    pose48 = np.zeros((T, 48), dtype=np.float32)
    src_vert_err_mm = np.zeros((T,), dtype=np.float32)
    init_verts_world = np.zeros_like(sequence.mano_vertices_world)
    init_joints_model = np.zeros_like(sequence.mano_joints_world_user)

    # 转换为张量
    global_orient = torch.from_numpy(sequence.global_orient_aa).to(device=device, dtype=torch.float32)
    hand_pose = torch.from_numpy(sequence.hand_pose_aa.reshape(T, -1)).to(device=device, dtype=torch.float32)
    betas = torch.from_numpy(sequence.mano_betas).to(device=device, dtype=torch.float32)
    transl = torch.from_numpy(sequence.mano_translation).to(device=device, dtype=torch.float32)
    gt_verts = torch.from_numpy(sequence.mano_vertices_world).to(device=device, dtype=torch.float32)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        with torch.no_grad():
            verts_world, joints_world = _forward_native_mano(
                mano_model=mano_model,
                global_orient=global_orient[start:end],
                hand_pose=hand_pose[start:end],
                betas=betas[start:end],
                transl=transl[start:end],
                J_regressor=J_regressor,
                tip_ids=tip_ids,
            )
            src_vert_err_mm[start:end] = (
                torch.linalg.norm(verts_world - gt_verts[start:end], dim=2).mean(dim=1) * 1000.0
            ).detach().cpu().numpy()
            pose48[start:end, :3] = global_orient[start:end].detach().cpu().numpy()
            pose48[start:end, 3:] = hand_pose[start:end].detach().cpu().numpy()
            init_verts_world[start:end] = verts_world.detach().cpu().numpy()
            init_joints_model[start:end] = joints_world.detach().cpu().numpy()

    return {
        "pose48": pose48,
        "src_vert_err_mm": src_vert_err_mm,
        "pca_vert_err_mm": np.full((T,), np.nan, dtype=np.float32),
        "init_verts_world": init_verts_world,
        "init_joints_model": init_joints_model,
    }


# ===========================================================================
# DeepContact 模型加载和接触推断
# ===========================================================================

def _load_deepcontact_model(checkpoint_path: Path, device: torch.device) -> DeepContactNet:
    """
    加载 DeepContact 预训练模型。

    Args:
        checkpoint_path: 检查点文件路径
        device: 计算设备

    Returns:
        加载好的模型（eval 模式）
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DeepContact checkpoint not found: {checkpoint_path}")
    model = DeepContactNet()
    state = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _infer_contact_targets(
    contact_source: str,
    model: Optional[DeepContactNet],
    hand_verts: torch.Tensor,
    hand_joints: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_normals: torch.Tensor,
    hand_faces_closed: torch.Tensor,
    caps_top: float,
    caps_bot: float,
    caps_rad: float,
    cont_method: int,
) -> Dict[str, torch.Tensor]:
    """
    推断目标接触图。

    两种来源：
    1. geom: 纯几何计算，基于胶囊 SDF
    2. deepcontact: 使用 DeepContact 深度网络预测

    Args:
        contact_source: "geom" 或 "deepcontact"
        model: DeepContact 模型（deepcontact 模式需要）
        其他参数: 几何和接触计算参数

    Returns:
        包含接触特征的字典
    """
    hand_normals = _compute_vertex_normals(hand_verts, hand_faces_closed)

    if contact_source == "geom":
        # 几何方法：使用胶囊 SDF 计算接触
        obj_contact_target, hand_contact_target = diffcontact.calculate_contact_capsule(
            hand_verts,
            hand_normals,
            obj_verts,
            obj_normals,
            caps_top=caps_top,
            caps_bot=caps_bot,
            caps_rad=caps_rad,
            caps_on_hand=False,
            contact_norm_method=cont_method,
        )
        hand_feats, obj_feats = _compute_hand_object_features(
            hand_verts, hand_joints, obj_verts, obj_normals, hand_faces_closed
        )
    elif contact_source == "deepcontact":
        # 深度学习方法：使用 DeepContact 网络
        if model is None:
            raise RuntimeError("DeepContact model is required when --contact-source=deepcontact")
        hand_feats, obj_feats = _compute_hand_object_features(
            hand_verts, hand_joints, obj_verts, obj_normals, hand_faces_closed
        )
        with torch.no_grad():
            network_out = model(hand_verts, hand_feats, obj_verts, obj_feats)
            hand_contact_target = cutil.class_to_val(network_out["contact_hand"]).unsqueeze(2)
            obj_contact_target = cutil.class_to_val(network_out["contact_obj"]).unsqueeze(2)
    else:
        raise ValueError(f"Unsupported contact source: {contact_source}")

    return {
        "hand_feats": hand_feats,
        "obj_feats": obj_feats,
        "hand_normals": hand_normals,
        "hand_contact_target": hand_contact_target,
        "obj_contact_target": obj_contact_target,
    }


# ===========================================================================
# 接触优化
# ===========================================================================

def _is_thin_object(object_name: str) -> bool:
    """
    判断物体是否为薄物体。

    薄物体（如牙刷、刀、眼镜）需要特殊处理穿透代价。

    Args:
        object_name: 物体名称

    Returns:
        是否为薄物体
    """
    return object_name.lower() in {"toothbrush", "knife", "eyeglasses", "scissors"}


def _optimize_pose_contactopt(
    data: Dict[str, torch.Tensor],
    hand_contact_target: torch.Tensor,
    obj_contact_target: torch.Tensor,
    mano_root: Path,
    object_name: str,
    n_iter: int,
    lr: float,
    w_cont_hand: float,
    w_cont_obj: float,
    ncomps: int,
    w_cont_asym: float,
    w_opt_trans: float,
    w_opt_pose: float,
    w_opt_rot: float,
    caps_top: float,
    caps_bot: float,
    caps_rad: float,
    caps_on_hand: bool,
    contact_norm_method: int,
    w_pen_cost: float,
    w_obj_rot: float,
    pen_it: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Dict[str, torch.Tensor]]]:
    """
    ContactOpt PCA 模式的姿态优化。

    优化变量：
    - opt_vector[:, 0:3]: 全局旋转增量
    - opt_vector[:, 3:ncomps+3]: PCA 姿态增量
    - opt_vector[:, ncomps+3:ncomps+6]: 平移增量
    - opt_vector[:, ncomps+6:]: 物体旋转增量

    损失函数：
    - 物体接触损失
    - 手部接触损失
    - 穿透代价（可选）

    Args:
        data: 输入数据字典
        hand_contact_target: 目标手部接触图
        obj_contact_target: 目标物体接触图
        其他参数: 优化超参数

    Returns:
        (优化姿态, 变换矩阵, 物体旋转, 优化状态历史) 元组
    """
    batch_size = data["hand_pose_aug"].shape[0]
    device = data["hand_pose_aug"].device

    # 优化向量初始化
    opt_vector = torch.zeros((batch_size, ncomps + 6 + 3), device=device)
    opt_vector.requires_grad_(True)

    # 创建 MANO 模型
    mano_model = ManoLayer(
        mano_root=str(mano_root),
        use_pca=True,
        ncomps=ncomps,
        side="right",
        flat_hand_mean=False,
    ).to(device)

    obj_normals_sampled = data["obj_normals_aug"]
    optimizer = torch.optim.Adam([opt_vector], lr=lr, amsgrad=True)
    loss_criterion = torch.nn.L1Loss(reduction="none")
    opt_state = []

    # 薄物体标志
    is_thin = torch.full(
        (batch_size,),
        1 if _is_thin_object(object_name) else 0,
        dtype=torch.float32,
        device=device,
    )

    for _ in range(n_iter):
        optimizer.zero_grad()

        # 构建优化姿态
        mano_pose_out = torch.cat(
            [opt_vector[:, 0:3] * w_opt_rot, opt_vector[:, 3:ncomps + 3] * w_opt_pose],
            dim=1,
        )
        mano_pose_out[:, :18] += data["hand_pose_aug"]  # 加上初始姿态
        tform_out = cutil.translation_to_tform(opt_vector[:, ncomps + 3:ncomps + 6] * w_opt_trans)

        # 前向传播
        hand_verts, hand_joints = cutil.forward_mano(
            mano_model,
            mano_pose_out,
            data["hand_beta_aug"],
            [data["hand_mTc_aug"], tform_out],
        )

        # 计算手部法向量
        if contact_norm_method != 0 and not caps_on_hand:
            hand_faces = data["hand_faces_closed"].unsqueeze(0).repeat(batch_size, 1, 1)
            hand_normals = Meshes(verts=hand_verts, faces=hand_faces).verts_normals_padded()
        else:
            hand_normals = torch.zeros_like(hand_verts)

        obj_verts = data["obj_sampled_verts_aug"]
        obj_normals = obj_normals_sampled

        # 物体旋转
        obj_rot_mat = axis_angle_to_matrix(opt_vector[:, ncomps + 6:])
        if w_obj_rot > 0:
            obj_verts = cutil.apply_rot(obj_rot_mat, obj_verts, around_centroid=True)
            obj_normals = cutil.apply_rot(obj_rot_mat, obj_normals)

        # 计算接触
        contact_obj, contact_hand = diffcontact.calculate_contact_capsule(
            hand_verts,
            hand_normals,
            obj_verts,
            obj_normals,
            caps_top=caps_top,
            caps_bot=caps_bot,
            caps_rad=caps_rad,
            caps_on_hand=caps_on_hand,
            contact_norm_method=contact_norm_method,
        )

        # 非对称接触损失：缺失接触的惩罚更大
        contact_obj_sub = obj_contact_target - contact_obj
        contact_obj_weighted = contact_obj_sub + F.relu(contact_obj_sub) * w_cont_asym
        loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(1, 2))

        contact_hand_sub = hand_contact_target - contact_hand
        contact_hand_weighted = contact_hand_sub + F.relu(contact_hand_sub) * w_cont_asym
        loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        loss = loss_contact_obj * w_cont_obj + loss_contact_hand * w_cont_hand

        # 穿透代价
        if w_pen_cost > 0 and _ >= pen_it:
            pen_cost = diffcontact.calculate_penetration_cost(
                hand_verts,
                hand_normals,
                obj_verts,
                obj_normals,
                is_thin,
                contact_norm_method,
            )
            loss = loss + pen_cost.mean(dim=1) * w_pen_cost

        # 记录状态
        opt_state.append({"loss": loss.detach().cpu()})
        loss.mean().backward()
        optimizer.step()

    # 聚合变换
    tform_full_out = cutil.aggregate_tforms([data["hand_mTc_aug"], tform_out])
    return mano_pose_out, tform_full_out, obj_rot_mat, opt_state


def _optimize_pose_native(
    data: Dict[str, torch.Tensor],
    hand_contact_target: torch.Tensor,
    obj_contact_target: torch.Tensor,
    mano_model: MANO,
    J_regressor: torch.Tensor,
    tip_ids: torch.Tensor,
    object_name: str,
    n_iter: int,
    lr: float,
    w_cont_hand: float,
    w_cont_obj: float,
    w_cont_asym: float,
    w_opt_trans: float,
    w_opt_pose: float,
    w_opt_rot: float,
    caps_top: float,
    caps_bot: float,
    caps_rad: float,
    caps_on_hand: bool,
    contact_norm_method: int,
    w_pen_cost: float,
    w_obj_rot: float,
    w_pose_prior: float,
    w_rot_prior: float,
    w_trans_prior: float,
    w_vert_prior: float,
    pen_it: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Dict[str, torch.Tensor]]]:
    """
    Native 模式的姿态优化。

    与 ContactOpt 模式的区别：
    1. 优化变量是 full axis-angle（45 维）而非 PCA 参数
    2. 添加了先验损失（pose/rot/trans/vert prior）
    3. 使用 smplx.MANO 而非 manopth

    Args:
        data: 输入数据字典
        其他参数: 优化超参数（包含先验权重）

    Returns:
        (全局旋转, 手部姿态, 平移, 物体旋转, 优化状态) 元组
    """
    batch_size = data["hand_global_orient_aug"].shape[0]
    device = data["hand_global_orient_aug"].device

    # 优化向量：(3 + 45 + 3 + 3) = 54 维
    opt_vector = torch.zeros((batch_size, 3 + 45 + 3 + 3), device=device)
    opt_vector.requires_grad_(True)

    obj_normals_sampled = data["obj_normals_aug"]
    optimizer = torch.optim.Adam([opt_vector], lr=lr, amsgrad=True)
    loss_criterion = torch.nn.L1Loss(reduction="none")
    opt_state = []

    is_thin = torch.full(
        (batch_size,),
        1 if _is_thin_object(object_name) else 0,
        dtype=torch.float32,
        device=device,
    )

    for it in range(n_iter):
        optimizer.zero_grad()

        # 构建优化姿态（加上增量）
        global_orient_out = data["hand_global_orient_aug"] + opt_vector[:, 0:3] * w_opt_rot
        hand_pose_out = data["hand_pose_aa_aug"] + opt_vector[:, 3:48] * w_opt_pose
        transl_out = data["hand_transl_aug"] + opt_vector[:, 48:51] * w_opt_trans

        # 前向传播
        hand_verts, hand_joints = _forward_native_mano(
            mano_model=mano_model,
            global_orient=global_orient_out,
            hand_pose=hand_pose_out,
            betas=data["hand_beta_aug"],
            transl=transl_out,
            J_regressor=J_regressor,
            tip_ids=tip_ids,
        )

        # 计算手部法向量
        if contact_norm_method != 0 and not caps_on_hand:
            hand_faces = data["hand_faces_closed"].unsqueeze(0).repeat(batch_size, 1, 1)
            hand_normals = Meshes(verts=hand_verts, faces=hand_faces).verts_normals_padded()
        else:
            hand_normals = torch.zeros_like(hand_verts)

        obj_verts = data["obj_sampled_verts_aug"]
        obj_normals = obj_normals_sampled

        # 物体旋转
        obj_rot_mat = axis_angle_to_matrix(opt_vector[:, 51:54])
        if w_obj_rot > 0:
            obj_verts = cutil.apply_rot(obj_rot_mat, obj_verts, around_centroid=True)
            obj_normals = cutil.apply_rot(obj_rot_mat, obj_normals)

        # 计算接触
        contact_obj, contact_hand = diffcontact.calculate_contact_capsule(
            hand_verts,
            hand_normals,
            obj_verts,
            obj_normals,
            caps_top=caps_top,
            caps_bot=caps_bot,
            caps_rad=caps_rad,
            caps_on_hand=caps_on_hand,
            contact_norm_method=contact_norm_method,
        )

        # 接触损失
        contact_obj_sub = obj_contact_target - contact_obj
        contact_obj_weighted = contact_obj_sub + F.relu(contact_obj_sub) * w_cont_asym
        loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(1, 2))

        contact_hand_sub = hand_contact_target - contact_hand
        contact_hand_weighted = contact_hand_sub + F.relu(contact_hand_sub) * w_cont_asym
        loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        loss = loss_contact_obj * w_cont_obj + loss_contact_hand * w_cont_hand

        # 先验损失：防止姿态变化过大
        pose_prior_target = data.get("hand_pose_aa_prior", data["hand_pose_aa_aug"])
        orient_prior_target = data.get("hand_global_orient_prior", data["hand_global_orient_aug"])
        transl_prior_target = data.get("hand_transl_prior", data["hand_transl_aug"])
        verts_prior_target = data.get("hand_verts_prior", data["hand_verts_aug"])

        loss_pose_prior = torch.square(hand_pose_out - pose_prior_target).mean(dim=1)
        loss_rot_prior = torch.square(global_orient_out - orient_prior_target).mean(dim=1)
        loss_trans_prior = torch.square(transl_out - transl_prior_target).mean(dim=1)
        loss_vert_prior = torch.square(hand_verts - verts_prior_target).mean(dim=(1, 2))

        loss = (
            loss
            + loss_pose_prior * w_pose_prior
            + loss_rot_prior * w_rot_prior
            + loss_trans_prior * w_trans_prior
            + loss_vert_prior * w_vert_prior
        )

        # 穿透代价
        if w_pen_cost > 0 and it >= pen_it:
            pen_cost = diffcontact.calculate_penetration_cost(
                hand_verts,
                hand_normals,
                obj_verts,
                obj_normals,
                is_thin,
                contact_norm_method,
            )
            loss = loss + pen_cost.mean(dim=1) * w_pen_cost

        # 记录状态
        opt_state.append(
            {
                "loss": loss.detach().cpu(),
                "loss_contact_obj": loss_contact_obj.detach().cpu(),
                "loss_contact_hand": loss_contact_hand.detach().cpu(),
                "loss_pose_prior": loss_pose_prior.detach().cpu(),
                "loss_rot_prior": loss_rot_prior.detach().cpu(),
                "loss_trans_prior": loss_trans_prior.detach().cpu(),
                "loss_vert_prior": loss_vert_prior.detach().cpu(),
            }
        )
        loss.mean().backward()
        optimizer.step()

    return global_orient_out, hand_pose_out, transl_out, obj_rot_mat, opt_state


# ===========================================================================
# 随机重启
# ===========================================================================

def _run_random_restarts(
    data: Dict[str, torch.Tensor],
    hand_contact_target: torch.Tensor,
    obj_contact_target: torch.Tensor,
    mano_root: Path,
    object_name: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    执行随机重启优化（ContactOpt PCA 模式）。

    每次重启在初始姿态上添加随机扰动，保留损失最小的结果。

    Args:
        data: 输入数据
        hand_contact_target: 目标手部接触
        obj_contact_target: 目标物体接触
        mano_root: MANO 根目录
        object_name: 物体名称
        args: 命令行参数

    Returns:
        (最佳姿态, 最佳变换, 最佳物体旋转, 最佳损失) 元组
    """
    batch_size = data["hand_pose_aug"].shape[0]
    device = data["hand_pose_aug"].device
    mtc_orig = data["hand_mTc_aug"].detach().clone()
    best_loss = torch.full((batch_size,), float("inf"), device=device)
    best_pose = None
    best_mTc = None
    best_obj_rot = None

    for re_it in range(args.rand_re):
        # 重置到原始姿态
        data["hand_mTc_aug"] = mtc_orig.detach().clone()

        # 添加随机扰动（除了第一次）
        if args.rand_re > 1 and re_it > 0:
            random_rot_mat = euler_angles_to_matrix(
                torch.randn((batch_size, 3), device=device) * args.rand_re_rot / 180.0 * np.pi,
                "ZYX",
            )
            data["hand_mTc_aug"][:, :3, :3] = torch.bmm(random_rot_mat, data["hand_mTc_aug"][:, :3, :3])
            data["hand_mTc_aug"][:, :3, 3] += torch.randn((batch_size, 3), device=device) * args.rand_re_trans

        # 运行优化
        out_pose, out_mTc, obj_rot, opt_state = _optimize_pose_contactopt(
            data=data,
            hand_contact_target=hand_contact_target,
            obj_contact_target=obj_contact_target,
            mano_root=mano_root,
            object_name=object_name,
            n_iter=args.n_iter,
            lr=args.lr,
            w_cont_hand=args.w_cont_hand,
            w_cont_obj=1.0,
            ncomps=args.ncomps,
            w_cont_asym=args.w_cont_asym,
            w_opt_trans=args.w_opt_trans,
            w_opt_pose=args.w_opt_pose,
            w_opt_rot=args.w_opt_rot,
            caps_top=args.caps_top,
            caps_bot=args.caps_bot,
            caps_rad=args.caps_rad,
            caps_on_hand=args.caps_hand,
            contact_norm_method=args.cont_method,
            w_pen_cost=args.w_pen_cost,
            w_obj_rot=args.w_obj_rot,
            pen_it=args.pen_it,
        )

        # 更新最佳结果
        loss_val = opt_state[-1]["loss"].to(device)
        if best_pose is None:
            best_pose = out_pose.detach().clone()
            best_mTc = out_mTc.detach().clone()
            best_obj_rot = obj_rot.detach().clone()

        update_mask = loss_val < best_loss
        best_loss = torch.where(update_mask, loss_val, best_loss)
        best_pose[update_mask] = out_pose.detach()[update_mask]
        best_mTc[update_mask] = out_mTc.detach()[update_mask]
        best_obj_rot[update_mask] = obj_rot.detach()[update_mask]

    data["hand_mTc_aug"] = mtc_orig
    return best_pose, best_mTc, best_obj_rot, best_loss.detach().cpu()


def _run_random_restarts_native(
    data: Dict[str, torch.Tensor],
    hand_contact_target: torch.Tensor,
    obj_contact_target: torch.Tensor,
    mano_model: MANO,
    J_regressor: torch.Tensor,
    tip_ids: torch.Tensor,
    object_name: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    执行随机重启优化（Native 模式）。

    与 _run_random_restarts 类似，但使用 Native MANO 模型。

    Returns:
        (最佳全局旋转, 最佳手部姿态, 最佳平移, 最佳物体旋转, 最佳损失) 元组
    """
    batch_size = data["hand_global_orient_aug"].shape[0]
    device = data["hand_global_orient_aug"].device
    global_orig = data["hand_global_orient_aug"].detach().clone()
    transl_orig = data["hand_transl_aug"].detach().clone()
    best_loss = torch.full((batch_size,), float("inf"), device=device)
    best_global = None
    best_pose = None
    best_transl = None
    best_obj_rot = None

    for re_it in range(args.rand_re):
        # 创建副本以避免修改原始数据
        data_run = dict(data)
        data_run["hand_global_orient_aug"] = global_orig.detach().clone()
        data_run["hand_transl_aug"] = transl_orig.detach().clone()
        data_run["hand_global_orient_prior"] = global_orig
        data_run["hand_transl_prior"] = transl_orig
        data_run["hand_pose_aa_prior"] = data["hand_pose_aa_aug"]
        data_run["hand_verts_prior"] = data["hand_verts_aug"]

        # 添加随机扰动
        if args.rand_re > 1 and re_it > 0:
            random_rot_mat = euler_angles_to_matrix(
                torch.randn((batch_size, 3), device=device) * args.rand_re_rot / 180.0 * np.pi,
                "ZYX",
            )
            base_rot = axis_angle_to_matrix(data_run["hand_global_orient_aug"])
            data_run["hand_global_orient_aug"] = matrix_to_axis_angle(torch.bmm(random_rot_mat, base_rot))
            data_run["hand_transl_aug"] = (
                data_run["hand_transl_aug"]
                + torch.randn((batch_size, 3), device=device) * args.rand_re_trans
            )

        # 运行优化
        out_global, out_pose, out_transl, obj_rot, opt_state = _optimize_pose_native(
            data=data_run,
            hand_contact_target=hand_contact_target,
            obj_contact_target=obj_contact_target,
            mano_model=mano_model,
            J_regressor=J_regressor,
            tip_ids=tip_ids,
            object_name=object_name,
            n_iter=args.n_iter,
            lr=args.lr,
            w_cont_hand=args.w_cont_hand,
            w_cont_obj=1.0,
            w_cont_asym=args.w_cont_asym,
            w_opt_trans=args.w_opt_trans,
            w_opt_pose=args.w_opt_pose,
            w_opt_rot=args.w_opt_rot,
            caps_top=args.caps_top,
            caps_bot=args.caps_bot,
            caps_rad=args.caps_rad,
            caps_on_hand=args.caps_hand,
            contact_norm_method=args.cont_method,
            w_pen_cost=args.w_pen_cost,
            w_obj_rot=args.w_obj_rot,
            w_pose_prior=args.w_pose_prior,
            w_rot_prior=args.w_rot_prior,
            w_trans_prior=args.w_trans_prior,
            w_vert_prior=args.w_vert_prior,
            pen_it=args.pen_it,
        )

        # 更新最佳结果
        loss_val = opt_state[-1]["loss"].to(device)
        if best_global is None:
            best_global = out_global.detach().clone()
            best_pose = out_pose.detach().clone()
            best_transl = out_transl.detach().clone()
            best_obj_rot = obj_rot.detach().clone()

        update_mask = loss_val < best_loss
        best_loss = torch.where(update_mask, loss_val, best_loss)
        best_global[update_mask] = out_global.detach()[update_mask]
        best_pose[update_mask] = out_pose.detach()[update_mask]
        best_transl[update_mask] = out_transl.detach()[update_mask]
        best_obj_rot[update_mask] = obj_rot.detach()[update_mask]

    return best_global, best_pose, best_transl, best_obj_rot, best_loss.detach().cpu()


# ===========================================================================
# 评估指标计算
# ===========================================================================

def _compute_contact_ratio(contact_obj: torch.Tensor) -> np.ndarray:
    """
    计算接触比例。

    Args:
        contact_obj: 物体接触值 (B, N, 1)

    Returns:
        每帧的接触比例 (B,)
    """
    return contact_obj.squeeze(-1).mean(dim=1).detach().cpu().numpy().astype(np.float32)


def _compute_penetration_depth_mm(
    hand_verts: torch.Tensor,
    obj_verts: torch.Tensor,
    obj_normals: torch.Tensor,
) -> np.ndarray:
    """
    计算穿透深度（毫米）。

    对于每个手部顶点，找到最近的物体顶点，检查是否穿透。

    Args:
        hand_verts: 手部顶点 (B, 778, 3)
        obj_verts: 物体顶点 (B, N, 3)
        obj_normals: 物体法向量 (B, N, 3)

    Returns:
        每帧的最大穿透深度 (B,)
    """
    _, nearest_idx, _ = knn_points(hand_verts, obj_verts, K=1, return_nn=True)
    closest_obj_verts = cutil.batched_index_select(obj_verts, 1, nearest_idx.squeeze(2))
    closest_obj_normals = cutil.batched_index_select(obj_normals, 1, nearest_idx.squeeze(2))
    delta = hand_verts - closest_obj_verts
    signed = torch.sum(delta * closest_obj_normals, dim=2)  # 沿法向的投影
    penetration = torch.relu(-signed).max(dim=1).values * 1000.0  # 负值表示穿透
    return penetration.detach().cpu().numpy().astype(np.float32)


def _compute_face_centers(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    计算面片中心点。

    Args:
        verts: 顶点 (B, V, 3)
        faces: 面片索引 (F, 3)

    Returns:
        面片中心 (B, F, 3)
    """
    return verts[:, faces].mean(axis=2).astype(np.float32)


def _maybe_rotate_object(
    obj_points: torch.Tensor,
    obj_normals: torch.Tensor,
    obj_rot: torch.Tensor,
    enabled: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可选地旋转物体。

    Args:
        obj_points: 物体点
        obj_normals: 物体法向量
        obj_rot: 旋转矩阵
        enabled: 是否执行旋转

    Returns:
        (旋转后的点, 旋转后的法向量) 元组
    """
    if not enabled:
        return obj_points, obj_normals
    return (
        cutil.apply_rot(obj_rot, obj_points, around_centroid=True),
        cutil.apply_rot(obj_rot, obj_normals),
    )


# ===========================================================================
# 命令行参数解析
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回:
        包含所有参数的 Namespace 对象
    """
    parser = argparse.ArgumentParser(description="Run ContactOpt baseline on Ref2Dex processed data")

    # ----- 路径参数 -----
    parser.add_argument("--processed-root", type=str, default=str(DEFAULT_PROCESSED_ROOT),
                        help="预处理数据根目录")
    parser.add_argument("--seq-id", type=str, required=True,
                        help="序列 ID，格式 'subject_id/seq_name'，例如 's1/bowl_pass_1'")
    parser.add_argument("--side", type=str, default="right",
                        help="'right' 或 'left'")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，默认使用 processed_data/generated/mano_fit_contactopt")
    parser.add_argument("--contactopt-root", type=str, default=str(DEFAULT_CONTACTOPT_ROOT),
                        help="ContactOpt 仓库路径")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="DeepContact 检查点路径")

    # ----- 设备参数 -----
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")

    # ----- 帧选择参数 -----
    parser.add_argument("--frame-step", type=int, default=1,
                        help="帧采样步长，每隔 N 帧取一帧")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="起始帧索引")
    parser.add_argument("--end-frame", type=int, default=-1,
                        help="结束帧索引，-1 表示到末尾")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="最大帧数，-1 表示不限制")
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument("--respect-hand-valid", dest="respect_hand_valid", action="store_true")
    valid_group.add_argument("--no-respect-hand-valid", dest="respect_hand_valid", action="store_false")
    parser.set_defaults(respect_hand_valid=True)

    # ----- 批处理参数 -----
    parser.add_argument("--frame-batch-size", type=int, default=16,
                        help="优化时的批大小")
    parser.add_argument("--contact-source", type=str, default="deepcontact",
                        choices=["deepcontact", "geom"],
                        help="接触图来源：deepcontact 使用网络预测，geom 使用几何计算")

    # ----- PCA 初始化参数（仅 ContactOpt 模式） -----
    parser.add_argument("--pca-fit-batch-size", type=int, default=32,
                        help="PCA 拟合批大小")
    parser.add_argument("--pca-fit-iter", type=int, default=200,
                        help="PCA 拟合迭代次数")
    parser.add_argument("--pca-fit-lr", type=float, default=0.03,
                        help="PCA 拟合学习率")

    # ----- MANO 参数模式 -----
    parser.add_argument(
        "--mano-param-mode",
        type=str,
        default="auto",
        choices=["auto", "native", "contactopt_pca"],
        help="auto: 自动选择（GRAB 数据用 native）；native: 使用 smplx.MANO；contactopt_pca: 使用 manopth PCA MANO"
    )

    # ----- 优化参数 -----
    parser.add_argument("--lr", type=float, default=0.01,
                        help="优化学习率")
    parser.add_argument("--n-iter", type=int, default=250,
                        help="优化迭代次数")
    parser.add_argument("--export-init-only", action="store_true", default=False,
                        help="只导出初始化结果，跳过优化")

    # ----- 接触损失权重 -----
    parser.add_argument("--w-cont-hand", type=float, default=2.5,
                        help="手部接触损失权重")
    parser.add_argument("--ncomps", type=int, default=15,
                        help="PCA 组件数（ContactOpt 模式）")
    parser.add_argument("--w-cont-asym", type=float, default=2.0,
                        help="接触非对称损失权重")
    parser.add_argument("--w-opt-trans", type=float, default=0.3,
                        help="平移优化权重")
    parser.add_argument("--w-opt-rot", type=float, default=1.0,
                        help="旋转优化权重")
    parser.add_argument("--w-opt-pose", type=float, default=1.0,
                        help="姿态优化权重")

    # ----- 胶囊接触参数 -----
    parser.add_argument("--caps-rad", type=float, default=0.001,
                        help="胶囊半径（米）")
    parser.add_argument("--caps-hand", action="store_true",
                        help="胶囊放在手部而非物体上")
    parser.add_argument("--cont-method", type=int, default=0,
                        help="接触归一化方法")
    parser.add_argument("--caps-top", type=float, default=0.0005,
                        help="胶囊顶部距离（米）")
    parser.add_argument("--caps-bot", type=float, default=-0.001,
                        help="胶囊底部距离（米）")

    # ----- 穿透代价参数 -----
    parser.add_argument("--w-pen-cost", type=float, default=320.0,
                        help="穿透代价权重")
    parser.add_argument("--pen-it", type=int, default=0,
                        help="穿透代价开始迭代")
    parser.add_argument("--w-obj-rot", type=float, default=0.0,
                        help="物体旋转权重（实验性）")

    # ----- 先验权重（仅 Native 模式） -----
    parser.add_argument("--w-pose-prior", type=float, default=1.0,
                        help="姿态先验权重")
    parser.add_argument("--w-rot-prior", type=float, default=1.0,
                        help="旋转先验权重")
    parser.add_argument("--w-trans-prior", type=float, default=10.0,
                        help="平移先验权重")
    parser.add_argument("--w-vert-prior", type=float, default=10.0,
                        help="顶点先验权重")

    # ----- 随机重启参数 -----
    parser.add_argument("--rand-re", type=int, default=1,
                        help="随机重启次数")
    parser.add_argument("--rand-re-trans", type=float, default=0.02,
                        help="随机重启平移扰动（米）")
    parser.add_argument("--rand-re-rot", type=float, default=5.0,
                        help="随机重启旋转扰动（度）")

    return parser.parse_args()


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    """
    主函数：加载数据、运行优化、保存结果。

    流程：
        1. 解析参数和加载元数据
        2. 加载序列数据
        3. 初始化 MANO 参数（PCA 或 Native）
        4. 逐批运行接触优化
        5. 计算评估指标
        6. 保存结果到 PKL 文件
    """
    args = parse_args()

    # 警告：物体旋转优化会改变坐标系
    if abs(float(args.w_obj_rot)) > 0:
        print(
            "[warn] --w-obj-rot != 0 rotates the object during optimization, but obj_root_pose/object points "
            "are saved in the original frame. Keep it at 0 for dataset generation unless object pose output is updated."
        )

    # 加载元数据
    processed_root = Path(args.processed_root).resolve()
    meta_path = processed_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under {processed_root}")
    meta = json.loads(meta_path.read_text())

    # 设置路径
    contactopt_root = Path(args.contactopt_root).resolve()
    if str(contactopt_root) not in sys.path:
        sys.path.insert(0, str(contactopt_root))

    # DeepContact 检查点
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (contactopt_root / "checkpoints" / "deepcontact_checkpoint.pt").resolve()
    )

    # MANO 模型目录
    mano_root = Path(meta.get("mano_model_dir")).resolve()
    if not (mano_root / "MANO_RIGHT.pkl").exists():
        raise FileNotFoundError(f"MANO assets not found under {mano_root}")

    # 设置设备
    device = _device_from_arg(args.device)

    # 加载序列数据
    sequence, J_regressor = load_sequence(
        processed_root=processed_root,
        meta=meta,
        mano_root=mano_root,
        seq_id=args.seq_id,
        side=args.side,
        frame_step=args.frame_step,
        start_frame=args.start_frame,
        end_frame=(None if args.end_frame < 0 else args.end_frame),
        max_frames=args.max_frames,
        respect_hand_valid=args.respect_hand_valid,
    )
    print(
        f"[contactopt-fit] seq={sequence.seq_id} side={sequence.side} "
        f"frames={sequence.frame_ids.size} object={sequence.object_name}"
    )

    # 加载 MANO 面片
    _, mano_faces = _load_mano_v1_assets(mano_root, side=sequence.side)
    hand_faces_closed = _make_hand_faces(mano_faces, closed=True, device=device)

    # 检查 vtemplate 要求
    requires_vtemplate = bool(meta.get("mano_config", {}).get("requires_vtemplate", False))
    if requires_vtemplate and sequence.mano_vtemplate_path is None:
        raise RuntimeError(
            f"meta.json marks {sequence.dataset_name} as requiring v_template, "
            f"but no {sequence.side}_hand_vtemplate_path was found for {sequence.seq_id}"
        )

    # 确定 MANO 参数模式
    effective_mano_mode = args.mano_param_mode
    if effective_mano_mode == "auto":
        if requires_vtemplate or sequence.mano_vtemplate_path is not None or sequence.flat_hand_mean:
            effective_mano_mode = "native"
        else:
            effective_mano_mode = "contactopt_pca"

    print(
        f"[contactopt-fit] mano_param_mode={effective_mano_mode} "
        f"flat_hand_mean={sequence.flat_hand_mean} "
        f"vtemplate={sequence.mano_vtemplate_path}"
    )

    # 检查左手支持
    if sequence.side != "right" and effective_mano_mode != "native":
        raise NotImplementedError("Left hand is currently supported only in native MANO mode.")

    # 导出模式提示
    if args.export_init_only:
        print("[contactopt-fit] export_init_only=True, optimization is skipped and init hand is written as opt_*")

    # 准备辅助数据
    J_regressor_t = torch.from_numpy(J_regressor).to(device=device, dtype=torch.float32)
    tip_ids_t = torch.as_tensor(MANO_TIP_VERTEX_INDICES[sequence.side], device=device, dtype=torch.long)

    # 模型初始化
    native_model = None
    pca_model = None
    pca_fit = None
    native_fit = None

    if effective_mano_mode == "native":
        # Native 模式：使用 smplx.MANO
        native_model = _make_smplx_mano(
            mano_root=mano_root,
            side=sequence.side,
            flat_hand_mean=sequence.flat_hand_mean,
            mano_vtemplate_path=sequence.mano_vtemplate_path,
            device=device,
        )
        native_fit = _init_native_pose(
            sequence=sequence,
            mano_model=native_model,
            J_regressor=J_regressor_t,
            tip_ids=tip_ids_t,
            device=device,
            batch_size=max(1, args.pca_fit_batch_size),
        )
        print(
            f"[contactopt-fit] native init vert err mm: "
            f"src={native_fit['src_vert_err_mm'].mean():.3f}"
        )
    else:
        # ContactOpt PCA 模式：转换到 15 维 PCA 空间
        pca_fit = _fit_contactopt_pca_pose(
            sequence=sequence,
            mano_root=mano_root,
            device=device,
            batch_size=max(1, args.pca_fit_batch_size),
            n_iter=max(1, args.pca_fit_iter),
            lr=float(args.pca_fit_lr),
        )
        print(
            f"[contactopt-fit] pca init vert err mm: "
            f"src={pca_fit['src_vert_err_mm'].mean():.3f}, pca={pca_fit['pca_vert_err_mm'].mean():.3f}"
        )
        pca_model = ManoLayer(
            mano_root=str(mano_root),
            use_pca=True,
            ncomps=15,
            flat_hand_mean=False,
            side="right",
        ).to(device)

    # 确定要处理的帧索引
    process_mask = (
        np.asarray(sequence.hand_valid, dtype=bool)
        if args.respect_hand_valid
        else np.ones((sequence.frame_ids.shape[0],), dtype=bool)
    )
    fit_idx = np.where(process_mask)[0]
    if fit_idx.size == 0:
        raise RuntimeError("No frames selected for optimization after applying hand_valid mask")

    # 加载 DeepContact 模型
    deepcontact_model = None
    if args.contact_source == "deepcontact" and (not args.export_init_only):
        deepcontact_model = _load_deepcontact_model(checkpoint_path, device)

    # 分配输出数组
    T = int(fit_idx.size)
    opt_verts = np.zeros((T, 778, 3), dtype=np.float32)
    opt_joints_model = np.zeros((T, 21, 3), dtype=np.float32)
    opt_pose18 = np.zeros((T, 18), dtype=np.float32)
    opt_global_orient_aa = np.zeros((T, 3), dtype=np.float32)
    opt_hand_pose_aa = np.zeros((T, 45), dtype=np.float32)
    opt_hand_translation = np.zeros((T, 3), dtype=np.float32)
    opt_mTc = np.zeros((T, 4, 4), dtype=np.float32)
    opt_obj_rot = np.zeros((T, 3, 3), dtype=np.float32)
    final_loss = np.zeros((T,), dtype=np.float32)
    contact_ratio = np.zeros((T,), dtype=np.float32)
    penetration_depth = np.zeros((T,), dtype=np.float32)
    target_obj_contact_all = np.zeros((T, sequence.obj_points_world.shape[1]), dtype=np.float32)
    final_obj_contact_all = np.zeros((T, sequence.obj_points_world.shape[1]), dtype=np.float32)

    # 初始化数据
    if effective_mano_mode == "native":
        init_pack = native_fit
        pca_pose18_all = np.full((T, 18), np.nan, dtype=np.float32)
    else:
        init_pack = pca_fit
        pca_pose18_all = pca_fit["pose18"][fit_idx]

    init_verts_world_all = init_pack["init_verts_world"][fit_idx]
    init_joints_model_all = init_pack["init_joints_model"][fit_idx]
    init_global_orient_aa_all = sequence.global_orient_aa[fit_idx].reshape(T, 3).astype(np.float32)
    init_hand_pose_aa_all = sequence.hand_pose_aa[fit_idx].reshape(T, 45).astype(np.float32)
    init_hand_translation_all = sequence.mano_translation[fit_idx].reshape(T, 3).astype(np.float32)

    # 逐批处理
    total_batches = (T + max(1, args.frame_batch_size) - 1) // max(1, args.frame_batch_size)
    for batch_id, start in enumerate(range(0, T, max(1, args.frame_batch_size)), start=1):
        end = min(start + args.frame_batch_size, T)
        batch_idx = fit_idx[start:end]
        B = end - start

        print(
            f"[contactopt-fit] batch {batch_id}/{total_batches} "
            f"frames={int(sequence.frame_ids[batch_idx[0]])}-{int(sequence.frame_ids[batch_idx[-1]])}"
        )

        # 准备输入数据
        hand_beta_aug = torch.from_numpy(sequence.mano_betas[batch_idx]).to(device=device, dtype=torch.float32)
        hand_verts_aug = torch.from_numpy(init_verts_world_all[start:end]).to(device=device, dtype=torch.float32)
        hand_joints_aug = torch.from_numpy(init_joints_model_all[start:end]).to(device=device, dtype=torch.float32)
        obj_sampled_verts_aug = torch.from_numpy(sequence.obj_points_world[batch_idx]).to(device=device, dtype=torch.float32)
        obj_normals_aug = torch.from_numpy(sequence.obj_normals_world[batch_idx]).to(device=device, dtype=torch.float32)
        obj_sampled_idx = torch.arange(obj_sampled_verts_aug.shape[1], device=device).long().unsqueeze(0).repeat(B, 1)

        if args.export_init_only:
            # 导出模式：跳过优化
            out_global = torch.from_numpy(init_global_orient_aa_all[start:end]).to(device=device, dtype=torch.float32)
            out_hand_pose = torch.from_numpy(init_hand_pose_aa_all[start:end]).to(device=device, dtype=torch.float32)
            out_transl = torch.from_numpy(init_hand_translation_all[start:end]).to(device=device, dtype=torch.float32)
            out_mTc = _make_hand_tform(out_transl)
            out_pose = (
                torch.from_numpy(pca_pose18_all[start:end]).to(device=device, dtype=torch.float32)
                if effective_mano_mode != "native"
                else torch.full((B, 18), float("nan"), device=device, dtype=torch.float32)
            )
            obj_rot_mat = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
            final_verts_t = hand_verts_aug
            final_joints_t = hand_joints_aug
            final_loss[start:end] = 0.0

            with torch.no_grad():
                final_hand_normals_t = _compute_vertex_normals(final_verts_t, hand_faces_closed)
                final_obj_contact_t, _ = diffcontact.calculate_contact_capsule(
                    final_verts_t,
                    final_hand_normals_t,
                    obj_sampled_verts_aug,
                    obj_normals_aug,
                    caps_top=args.caps_top,
                    caps_bot=args.caps_bot,
                    caps_rad=args.caps_rad,
                    caps_on_hand=args.caps_hand,
                    contact_norm_method=args.cont_method,
                )
            target_obj_contact_t = final_obj_contact_t
        else:
            # 推断目标接触
            target_pack = _infer_contact_targets(
                contact_source=args.contact_source,
                model=deepcontact_model,
                hand_verts=hand_verts_aug,
                hand_joints=hand_joints_aug,
                obj_verts=obj_sampled_verts_aug,
                obj_normals=obj_normals_aug,
                hand_faces_closed=hand_faces_closed,
                caps_top=args.caps_top,
                caps_bot=args.caps_bot,
                caps_rad=args.caps_rad,
                cont_method=args.cont_method,
            )

            # 构建数据字典
            data = {
                "hand_beta_aug": hand_beta_aug,
                "hand_verts_aug": hand_verts_aug,
                "hand_feats_aug": target_pack["hand_feats"],
                "obj_sampled_verts_aug": obj_sampled_verts_aug,
                "obj_feats_aug": target_pack["obj_feats"],
                "obj_normals_aug": obj_normals_aug,
                "obj_sampled_idx": obj_sampled_idx,
                "hand_faces_closed": hand_faces_closed,
            }

            if effective_mano_mode == "native":
                data["hand_global_orient_aug"] = torch.from_numpy(
                    init_global_orient_aa_all[start:end]
                ).to(device=device, dtype=torch.float32)
                data["hand_pose_aa_aug"] = torch.from_numpy(
                    init_hand_pose_aa_all[start:end]
                ).to(device=device, dtype=torch.float32)
                data["hand_transl_aug"] = torch.from_numpy(
                    init_hand_translation_all[start:end]
                ).to(device=device, dtype=torch.float32)
            else:
                hand_pose_aug = torch.from_numpy(pca_pose18_all[start:end]).to(device=device, dtype=torch.float32)
                hand_mTc_aug = _make_hand_tform(
                    torch.from_numpy(sequence.mano_translation[batch_idx]).to(device=device, dtype=torch.float32)
                )
                data["hand_pose_aug"] = hand_pose_aug
                data["hand_mTc_aug"] = hand_mTc_aug

            # 运行优化
            if effective_mano_mode == "native":
                if args.rand_re > 1:
                    out_global, out_hand_pose, out_transl, obj_rot_mat, best_loss = _run_random_restarts_native(
                        data=data,
                        hand_contact_target=target_pack["hand_contact_target"],
                        obj_contact_target=target_pack["obj_contact_target"],
                        mano_model=native_model,
                        J_regressor=J_regressor_t,
                        tip_ids=tip_ids_t,
                        object_name=sequence.object_name,
                        args=args,
                    )
                    final_loss[start:end] = best_loss.numpy().astype(np.float32)
                else:
                    out_global, out_hand_pose, out_transl, obj_rot_mat, opt_state = _optimize_pose_native(
                        data=data,
                        hand_contact_target=target_pack["hand_contact_target"],
                        obj_contact_target=target_pack["obj_contact_target"],
                        mano_model=native_model,
                        J_regressor=J_regressor_t,
                        tip_ids=tip_ids_t,
                        object_name=sequence.object_name,
                        n_iter=args.n_iter,
                        lr=args.lr,
                        w_cont_hand=args.w_cont_hand,
                        w_cont_obj=1.0,
                        w_cont_asym=args.w_cont_asym,
                        w_opt_trans=args.w_opt_trans,
                        w_opt_pose=args.w_opt_pose,
                        w_opt_rot=args.w_opt_rot,
                        caps_top=args.caps_top,
                        caps_bot=args.caps_bot,
                        caps_rad=args.caps_rad,
                        caps_on_hand=args.caps_hand,
                        contact_norm_method=args.cont_method,
                        w_pen_cost=args.w_pen_cost,
                        w_obj_rot=args.w_obj_rot,
                        w_pose_prior=args.w_pose_prior,
                        w_rot_prior=args.w_rot_prior,
                        w_trans_prior=args.w_trans_prior,
                        w_vert_prior=args.w_vert_prior,
                        pen_it=args.pen_it,
                    )
                    final_loss[start:end] = opt_state[-1]["loss"].numpy().astype(np.float32)

                # 前向传播获取最终顶点
                with torch.no_grad():
                    final_verts_t, final_joints_t = _forward_native_mano(
                        mano_model=native_model,
                        global_orient=out_global,
                        hand_pose=out_hand_pose,
                        betas=hand_beta_aug,
                        transl=out_transl,
                        J_regressor=J_regressor_t,
                        tip_ids=tip_ids_t,
                    )
                    out_mTc = _make_hand_tform(out_transl)
                    out_pose = torch.full((B, 18), float("nan"), device=device, dtype=torch.float32)
            else:
                if args.rand_re > 1:
                    out_pose, out_mTc, obj_rot_mat, best_loss = _run_random_restarts(
                        data=data,
                        hand_contact_target=target_pack["hand_contact_target"],
                        obj_contact_target=target_pack["obj_contact_target"],
                        mano_root=mano_root,
                        object_name=sequence.object_name,
                        args=args,
                    )
                    final_loss[start:end] = best_loss.numpy().astype(np.float32)
                else:
                    out_pose, out_mTc, obj_rot_mat, opt_state = _optimize_pose_contactopt(
                        data=data,
                        hand_contact_target=target_pack["hand_contact_target"],
                        obj_contact_target=target_pack["obj_contact_target"],
                        mano_root=mano_root,
                        object_name=sequence.object_name,
                        n_iter=args.n_iter,
                        lr=args.lr,
                        w_cont_hand=args.w_cont_hand,
                        w_cont_obj=1.0,
                        ncomps=args.ncomps,
                        w_cont_asym=args.w_cont_asym,
                        w_opt_trans=args.w_opt_trans,
                        w_opt_pose=args.w_opt_pose,
                        w_opt_rot=args.w_opt_rot,
                        caps_top=args.caps_top,
                        caps_bot=args.caps_bot,
                        caps_rad=args.caps_rad,
                        caps_on_hand=args.caps_hand,
                        contact_norm_method=args.cont_method,
                        w_pen_cost=args.w_pen_cost,
                        w_obj_rot=args.w_obj_rot,
                        pen_it=args.pen_it,
                    )
                    final_loss[start:end] = opt_state[-1]["loss"].numpy().astype(np.float32)

                # 前向传播获取最终顶点
                with torch.no_grad():
                    final_verts_t, final_joints_t = cutil.forward_mano(pca_model, out_pose, hand_beta_aug, [out_mTc])
                    out_global = torch.full((B, 3), float("nan"), device=device, dtype=torch.float32)
                    out_hand_pose = torch.full((B, 45), float("nan"), device=device, dtype=torch.float32)
                    out_transl = out_mTc[:, :3, 3]

            # 计算最终接触
            with torch.no_grad():
                final_obj_points_t, final_obj_normals_t = _maybe_rotate_object(
                    obj_sampled_verts_aug,
                    obj_normals_aug,
                    obj_rot_mat,
                    enabled=(args.w_obj_rot > 0),
                )
                final_hand_normals_t = _compute_vertex_normals(final_verts_t, hand_faces_closed)
                final_obj_contact_t, _ = diffcontact.calculate_contact_capsule(
                    final_verts_t,
                    final_hand_normals_t,
                    final_obj_points_t,
                    final_obj_normals_t,
                    caps_top=args.caps_top,
                    caps_bot=args.caps_bot,
                    caps_rad=args.caps_rad,
                    caps_on_hand=args.caps_hand,
                    contact_norm_method=args.cont_method,
                )
            target_obj_contact_t = target_pack["obj_contact_target"]

        # 更新物体（用于评估）
        with torch.no_grad():
            final_obj_points_t, final_obj_normals_t = _maybe_rotate_object(
                obj_sampled_verts_aug,
                obj_normals_aug,
                obj_rot_mat,
                enabled=(args.w_obj_rot > 0),
            )

        # 保存结果
        opt_verts[start:end] = final_verts_t.detach().cpu().numpy()
        opt_joints_model[start:end] = final_joints_t.detach().cpu().numpy()
        opt_pose18[start:end] = out_pose.detach().cpu().numpy()
        opt_global_orient_aa[start:end] = out_global.detach().cpu().numpy()
        opt_hand_pose_aa[start:end] = out_hand_pose.detach().cpu().numpy()
        opt_hand_translation[start:end] = out_transl.detach().cpu().numpy()
        opt_mTc[start:end] = out_mTc.detach().cpu().numpy()
        opt_obj_rot[start:end] = obj_rot_mat.detach().cpu().numpy()
        contact_ratio[start:end] = _compute_contact_ratio(final_obj_contact_t)
        penetration_depth[start:end] = _compute_penetration_depth_mm(
            final_verts_t,
            final_obj_points_t,
            final_obj_normals_t,
        )
        target_obj_contact_all[start:end] = target_obj_contact_t.squeeze(-1).detach().cpu().numpy()
        final_obj_contact_all[start:end] = final_obj_contact_t.squeeze(-1).detach().cpu().numpy()

    # 计算评估指标
    tip_ids = MANO_TIP_VERTEX_INDICES[sequence.side]
    gt_verts = sequence.mano_vertices_world[fit_idx]
    gt_joints = sequence.mano_joints_world_user[fit_idx]
    init_joints_user_all = _verts_to_21_joints_np(
        init_verts_world_all,
        J_regressor,
        tip_ids,
        MANO_JOINT_REORDER,
    ).astype(np.float32)
    opt_joints_user = _verts_to_21_joints_np(
        opt_verts,
        J_regressor,
        tip_ids,
        MANO_JOINT_REORDER,
    ).astype(np.float32)

    # 计算 RMS 误差
    init_vert_rms = np.sqrt(((init_verts_world_all - gt_verts) ** 2).sum(axis=-1).mean(axis=1)).astype(np.float32)
    final_vert_rms = np.sqrt(((opt_verts - gt_verts) ** 2).sum(axis=-1).mean(axis=1)).astype(np.float32)
    init_jt_rms = np.sqrt(((init_joints_user_all - gt_joints) ** 2).sum(axis=-1).mean(axis=1)).astype(np.float32)
    final_jt_rms = np.sqrt(((opt_joints_user - gt_joints) ** 2).sum(axis=-1).mean(axis=1)).astype(np.float32)

    # 构建输出路径
    default_out_root = DEFAULT_OUTPUT_ROOT / f"{sequence.dataset_name}_{args.contact_source}"
    base_out_dir = Path(args.output_dir).resolve() if args.output_dir else default_out_root
    out_dir = base_out_dir / sequence.subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sequence.seq_name}_{sequence.side}.pkl"

    # 构建输出 payload
    payload = {
        # 元信息
        "seq_id": sequence.seq_id,
        "dataset_name": sequence.dataset_name,
        "subject_id": sequence.subject_id,
        "seq_name": sequence.seq_name,
        "object_name": sequence.object_name,
        "side": sequence.side,
        "frame_ids": sequence.frame_ids[fit_idx],
        "raw_frame_id": sequence.frame_ids[fit_idx],
        "preprocess_frame_idx": sequence.preprocess_frame_idx[fit_idx],
        "source_preprocess_file": str(sequence.npz_path),
        "object_mesh_path": str(sequence.object_mesh_path) if sequence.object_mesh_path else None,
        "obj_root_pose": sequence.obj_root_pose[fit_idx],

        # 接触信息
        "contact_source": args.contact_source,
        "target_obj_contact": target_obj_contact_all,
        "final_obj_contact": final_obj_contact_all,
        "contact_ratio": contact_ratio,
        "penetration_depth": penetration_depth,
        "penetration_metric": "nearest_object_point_normal_approx",
        "final_loss": final_loss,

        # 优化结果
        "opt_hand_pose_pca": opt_pose18,
        "opt_hand_global_orient_aa": opt_global_orient_aa,
        "opt_hand_pose_aa": opt_hand_pose_aa.reshape(T, 15, 3),
        "opt_hand_pose_axis_angle": opt_hand_pose_aa.reshape(T, 15, 3),
        "opt_hand_translation": opt_hand_translation,
        "opt_hand_mTc": opt_mTc,
        "opt_obj_rot": opt_obj_rot,
        "opt_hand_verts_world": opt_verts,
        "opt_hand_verts": opt_verts,
        "opt_hand_joints_user": opt_joints_user,
        "opt_hand_joints_model": opt_joints_model,
        "opt_hand_face_centers_world": _compute_face_centers(opt_verts, mano_faces),
        "opt_hand_face_centers": _compute_face_centers(opt_verts, mano_faces),

        # 初始化结果
        "init_hand_pose_pca": pca_pose18_all,
        "init_hand_global_orient_aa": init_global_orient_aa_all,
        "init_hand_pose_aa": init_hand_pose_aa_all.reshape(T, 15, 3),
        "init_hand_pose_axis_angle": init_hand_pose_aa_all.reshape(T, 15, 3),
        "init_hand_translation": init_hand_translation_all,
        "init_hand_verts_world": init_verts_world_all,
        "init_joints_user": init_joints_user_all,
        "init_joints_model": init_joints_model_all,
        "init_hand_face_centers_world": _compute_face_centers(init_verts_world_all, mano_faces),
        "init_hand_face_centers": _compute_face_centers(init_verts_world_all, mano_faces),

        # 评估指标
        "init_vert_rms": init_vert_rms,
        "final_vert_rms": final_vert_rms,
        "init_jt_rms": init_jt_rms,
        "final_jt_rms": final_jt_rms,
        "contactopt_src_vert_err_mm": init_pack["src_vert_err_mm"][fit_idx],
        "contactopt_pca_vert_err_mm": init_pack["pca_vert_err_mm"][fit_idx],
        "hand_valid": sequence.hand_valid[fit_idx],
        "optimization_success": np.ones((T,), dtype=bool),
        "optimization_message": np.asarray(
            ["init_only" if args.export_init_only else "optimized"] * T,
            dtype=object,
        ),

        # 配置信息
        "config": {
            "method": "contactopt",
            "contactopt_root": str(contactopt_root),
            "checkpoint": str(checkpoint_path) if args.contact_source == "deepcontact" else None,
            "mano_root": str(mano_root),
            "mano_dir": str(mano_root),
            "processed_root": str(processed_root),
            "mano_param_mode": str(effective_mano_mode),
            "flat_hand_mean": bool(sequence.flat_hand_mean),
            "mano_vtemplate_path": str(sequence.mano_vtemplate_path) if sequence.mano_vtemplate_path else None,
            "contact_source": args.contact_source,
            "respect_hand_valid": bool(args.respect_hand_valid),
            "frame_step": int(args.frame_step),
            "start_frame": int(args.start_frame),
            "end_frame": int(args.end_frame),
            "max_frames": int(args.max_frames),
            "frame_batch_size": int(args.frame_batch_size),
            "pca_fit_batch_size": int(args.pca_fit_batch_size),
            "pca_fit_iter": int(args.pca_fit_iter),
            "pca_fit_lr": float(args.pca_fit_lr),
            "lr": float(args.lr),
            "n_iter": int(args.n_iter),
            "export_init_only": bool(args.export_init_only),
            "w_cont_hand": float(args.w_cont_hand),
            "ncomps": int(args.ncomps),
            "w_cont_asym": float(args.w_cont_asym),
            "w_opt_trans": float(args.w_opt_trans),
            "w_opt_rot": float(args.w_opt_rot),
            "w_opt_pose": float(args.w_opt_pose),
            "caps_rad": float(args.caps_rad),
            "caps_hand": bool(args.caps_hand),
            "cont_method": int(args.cont_method),
            "caps_top": float(args.caps_top),
            "caps_bot": float(args.caps_bot),
            "w_pen_cost": float(args.w_pen_cost),
            "pen_it": int(args.pen_it),
            "w_obj_rot": float(args.w_obj_rot),
            "w_pose_prior": float(args.w_pose_prior),
            "w_rot_prior": float(args.w_rot_prior),
            "w_trans_prior": float(args.w_trans_prior),
            "w_vert_prior": float(args.w_vert_prior),
            "penetration_metric": "nearest_object_point_normal_approx",
            "rand_re": int(args.rand_re),
            "rand_re_trans": float(args.rand_re_trans),
            "rand_re_rot": float(args.rand_re_rot),
        },
    }

    # 保存结果
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[contactopt-fit] wrote {out_path}")
    print(
        f"[contactopt-fit] final vert rms mean={final_vert_rms.mean() * 1000.0:.3f} mm "
        f"(init={init_vert_rms.mean() * 1000.0:.3f} mm)"
    )


if __name__ == "__main__":
    main()
