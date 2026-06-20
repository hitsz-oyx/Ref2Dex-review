#!/usr/bin/env python3
"""
ARCTIC 数据集预处理脚本
=======================

本脚本将 ARCTIC 原始数据 (`.mano.npy` + `.object.npy`) 处理成 Ref2Dex 训练所需的
`.npz` 格式，每条序列对应一个 `.npz` 文件，里面包含：

    - 物体表面采样点（canonical 拓扑稳定）
    - 双手的 face center 采样点（1538 点 = MANO face 数量）
    - 各自的法向、流场 (flow)
    - 手与物体之间的最近邻 (NN) 索引和距离
    - 手的 finger_id / region_id 语义标签

使用方法:
    python arctic_preprocess.py                              # 处理所有序列
    python arctic_preprocess.py --seq s05/box_grab_01        # 处理单条序列
    python arctic_preprocess.py --subject s01                # 处理某个 subject 的全部序列
    python arctic_preprocess.py --num_obj_points 4096        # 自定义物体采样数
    python arctic_preprocess.py --output_root ./processed    # 自定义输出目录

输出 .npz 字段规范（每个序列）:
    {
        # ----- 元信息 -----
        "seq_id": str,                            # 序列标识, e.g. "s01/box_grab_01"
        "frame_id": (T,) int,                     # 帧号 0..T-1
        "dt": float,                              # 帧间隔秒数, 默认 1/30

        # ----- 物体（每一帧）-----
        "obj_points": (T, No, 3) float,           # 物体表面点世界坐标
        "obj_normals": (T, No, 3) float,          # 对应点法向
        "obj_point_id": (No,) int,                # canonical 索引 (跨帧一致)
        "obj_root_pose": (T, 4, 4) float,         # 物体根节点 SE(3)

        # ----- 右手（每一帧，face-center 采样 1538 点）-----
        # **形状: (T, ...)**，所有帧都保留，不会被删除
        "right_hand_points": (T, 1538, 3) float,
        "right_hand_normals": (T, 1538, 3) float,
        "right_hand_point_id": (1538,) int,
        "right_hand_root_pose": (T, 4, 4) float,
        "right_hand_joint_axis_angle": (T, 15, 3) float,        # 15 个手指关节 axis-angle
        "right_hand_joint_axis_angle_delta": (T, 15, 3) float,  # 帧间差分
        "right_hand_joint_quat": (T, 16, 4) float,              # 16 关节世界旋转四元数 [w,x,y,z]
        "right_hand_betas": (T, 10) float,                     # MANO 形状参数 β（每帧相同）
        "right_hand_verts": (T, 778, 3) float,                 # MANO 完整 778 顶点（世界坐标）
        "right_hand_finger_id": (1538,) int,      # 0=palm, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
        "right_hand_region_id": (1538,) int,      # 0=palm, 1=fingertip, 2=finger_pad
        "right_hand_flow": (T, 1538, 3) float,            # 帧间位移
        "right_hand_flow_valid": (T, 1538) bool,          # fitting_err 阈值判定
        "right_hand_to_obj_nn_id": (T, 1538) int,
        "right_hand_to_obj_dist": (T, 1538) float,
        "right_hand_min_dist_to_obj": (T,) float,         # 该手每帧到物体最近距离（米）
        "right_hand_valid": (T,) bool,                    # 3cm 接触掩码: True=该手 face-center ≤ 3cm
        "right_hand_valid_2cm": (T,) bool,               # 2cm 接触掩码: True=该手 face-center ≤ 2cm
        "right_hand_valid_1cm": (T,) bool,               # 1cm 接触掩码: True=该手 face-center ≤ 1cm
                                                                                # （与 CPF 内部 range_threshold=10mm 严格对齐）

        # ----- 左手（同上）-----
        "left_hand_points": (T, 1538, 3) float,
        "left_hand_normals": (T, 1538, 3) float,
        "left_hand_point_id": (1538,) int,
        "left_hand_root_pose": (T, 4, 4) float,
        "left_hand_joint_axis_angle": (T, 15, 3) float,
        "left_hand_joint_axis_angle_delta": (T, 15, 3) float,
        "left_hand_joint_quat": (T, 16, 4) float,
        "left_hand_betas": (T, 10) float,
        "left_hand_verts": (T, 778, 3) float,
        "left_hand_finger_id": (1538,) int,
        "left_hand_region_id": (1538,) int,
        "left_hand_flow": (T, 1538, 3) float,
        "left_hand_flow_valid": (T, 1538) bool,
        "left_hand_to_obj_nn_id": (T, 1538) int,
        "left_hand_to_obj_dist": (T, 1538) float,
        "left_hand_min_dist_to_obj": (T,) float,
        "left_hand_valid": (T,) bool,

        # ----- 物体（保持 T 帧）-----
        "obj_points": (T, No, 3) float,
        "obj_normals": (T, No, 3) float,
        "obj_point_id": (No,) int,
        "obj_root_pose": (T, 4, 4) float,
        "obj_flow": (T, No, 3) float,
        "obj_flow_valid": (T, No) bool,
        "frame_id": (T,) int,                    # 物体帧编号 0..T-1

        # ----- obj→hand 字段（保持 T 帧）-----
        "obj_to_right_hand_nn_id": (T, No) int,
        "obj_to_right_hand_dist": (T, No) float,
        "obj_to_left_hand_nn_id": (T, No) int,
        "obj_to_left_hand_dist": (T, No) float,

        # ----- 手-物接触掩码规则（双阈值）-----
        # 3cm 掩码（*_hand_valid, 向后兼容）:
        #     该手 1538 face-center 点 min dist ≤ 0.03m → True
        #     用途: "手在物体附近" → 用于训练时 soft contact 概率化 / 评估
        # 2cm 掩码（*_hand_valid_2cm, 新增）:
        #     同样的 1538 face-center 点 min dist ≤ 0.02m → True
        #     用途: "手真的在接触" → 用于 CPF 等强接触优化的过滤
        #     原因: CPF 内部用 407 palm 顶点做 range_threshold=20mm 的接触判定，
        #           2cm face-center 比 3cm 更接近 CPF 内部口径，命中率更高
        # 双向独立判断: 右手无接触只影响 right_hand_valid[*]=False（2cm 同理）
        # **不删除任何帧、不清零任何数据**，所有数组保持 (T, ...) 形状
        # 下游用 *_hand_valid 掩码自行决定如何处理（过滤 / 加权 / 丢弃）

        # ----- 符号 -----
        # T = 物体总帧数 = frame_id.shape[0] = 所有 hand_* 数组的时间维度

MANO 采样说明:
    不同于直接采样顶点，本脚本使用**面中心 (face center)**：每个三角面取一个点，
    共 1538 个点（MANO 拓扑固定 1538 个 face）。这种"拓扑稳定"采样保证：
        - face i 在不同手型下都对应同一个三角形
        - 跨帧的点对应关系天然保持（不需要 NN 配准）
        - 法向取 face normal，比顶点法向更自然
"""

import argparse          # 命令行参数解析
import json               # 读取 parts.json（铰接物体顶/底部件标签）
import os                 # 文件路径、目录创建
import os.path as op      # 路径拼接
import sys
import time
import traceback          # 异常追踪打印
from glob import glob     # 通配符查找 .mano.npy

import numpy as np        # 数值计算核心库
from typing import Optional  # 用于 max_frames: Optional[int]

# ============================================================
# 修复 numpy 1.24+ 兼容性
# ============================================================
# smplx / MANO 库内部仍使用已废弃的 np.bool / np.int / np.float 等属性。
# numpy 1.24 起移除这些，需要在 import smplx 之前手动补回来。
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float_
if not hasattr(np, 'complex'):
    np.complex = np.complex_
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'unicode'):
    np.unicode = np.str_
if not hasattr(np, 'str'):
    np.str = np.str_

import torch              # 深度学习框架（MANO forward 走 PyTorch）
import trimesh            # 网格处理库（法向、face normal、barycentric 等）
from scipy.spatial import cKDTree   # 高效最近邻查询（用于手-物对应）
from smplx import MANO    # MANO 手部参数化人体模型
from tqdm import tqdm     # 进度条

# ============================================================
# 路径配置（可通过命令行参数覆盖）
# ============================================================
REF2DEX_ROOT = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
# MANO 模型文件目录（包含 MANO_LEFT.pkl / MANO_RIGHT.pkl）
MANO_MODEL_DIR = op.join(REF2DEX_ROOT, "dataset", "arctic", "data", "body_models", "mano")
# ARCTIC 原始数据根目录
DATA_ROOT = op.join(REF2DEX_ROOT, "dataset", "arctic", "data", "arctic_data", "data")
RAW_SEQS_DIR = op.join(DATA_ROOT, "raw_seqs")        # 输入: 存 .mano.npy / .object.npy
META_DIR = op.join(DATA_ROOT, "meta")                 # 物体模板、拆分标签等元数据
OBJECT_VTEMPLATE_DIR = op.join(META_DIR, "object_vtemplates")
SHARED_ASSET_ROOT = op.join(REF2DEX_ROOT, "assets", "shared")
SHARED_MANO_ASSET_ROOT = op.join(SHARED_ASSET_ROOT, "mano")
OBJECT_ASSET_ROOT = op.join(REF2DEX_ROOT, "assets", "arctic", "objects")
# 默认输出根目录
DEFAULT_OUTPUT_ROOT = op.join(REF2DEX_ROOT, "processed_data", "arctic")

# ARCTIC 包含 10 个被试 (s01..s10)
SUBJECTS = [f"s{i:02d}" for i in range(1, 11)]

# ARCTIC 中的铰接物体集合（具有顶/底部件 + 铰接角参数）
# 这些物体的 .object.npy 第 0 列是铰接角（弧度），其余是刚体位姿
ARTICULATED_OBJECTS = {
    "box", "capsulemachine", "espressomachine", "ketchup",
    "laptop", "microwave", "mixer", "notebook", "phone",
    "scissors", "waffleiron",
}

# MANO 关节层级（smplx MANO 输出一共 16 个关节）:
#   0    = wrist（腕）
#   1-3  = thumb（拇指）  → finger_id=1
#   4-6  = index（食指）  → finger_id=2
#   7-9  = middle（中指） → finger_id=3
#   10-12 = ring（无名指）→ finger_id=4
#   13-15 = pinky（小指） → finger_id=5
# 关节 axis-angle 的形状为 [15, 3]，因为根节点（腕）不算在 hand_pose 里
MANO_NUM_JOINTS = 16  # smplx MANO 输出的关节总数
MANO_NUM_FACES = 1538  # 标准 MANO 拓扑：1538 个三角面
MANO_NUM_HAND_POINTS = MANO_NUM_FACES  # 每个 face 取一个中心点，共 1538 个采样点

# MANO 关节索引 → finger_id 的映射
# 0=palm, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
# 注意顺序是按"父关节 -> 子关节 -> 末梢"排列
JOINT_TO_FINGER = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

# MANO fitting 误差阈值（单位：毫米，ARCTIC 原始数据单位）
# 当某帧的拟合误差 > 10mm 时，认为该帧的 hand 数据不可靠
# 该帧对应的 hand_flow_valid 会被置为 False
FITTING_ERR_THRESH = 10.0

# 手-物接触距离阈值（单位：米）
# 当手部表面到物体的最近距离 > 该阈值时，认为该帧手与物体"无接触"
# 此时该手的表面数据/流场/接触对应会被屏蔽（置 0），并通过
# `*_hand_in_contact` 掩码告知下游消费者跳过该帧该手
CONTACT_DISTANCE_THRESH = 0.03   # 3 cm 默认值，可通过 CLI 覆盖

# MANO 顶点的 region 语义 ID（基于"最近关节"启发式划分）
REGION_PALM = 0         # 手掌/腕部
REGION_FINGERTIP = 1    # 指尖（接触物体的主要区域）
REGION_FINGER_PAD = 2   # 手指内侧（指肚，接触物体的次要区域）
REGION_FINGER_SIDE = 3  # 手指两侧
REGION_FINGER_BACK = 4  # 手指背面
REGION_HAND_BACK = 5    # 手背


# ============================================================
# 辅助函数
# ============================================================


def axis_angle_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    将 (..., 3) 的 axis-angle 旋转向量转换为 (..., 3, 3) 旋转矩阵（Rodrigues 公式）。

    数学原理:
        给定旋转轴 n（单位向量）和角度 θ，旋转矩阵为:
            R = I + sin(θ) * K + (1 - cos(θ)) * K^2
        其中 K 是 n 的反对称矩阵（叉积矩阵）。

    Args:
        axis_angle: (..., 3) 旋转向量。模长 = 角度，方向 = 旋转轴。

    Returns:
        R: (..., 3, 3) 旋转矩阵（正交阵，det=+1）
    """
    # 取模长 = 角度（+1e-8 防止 0 角度时除零）
    angle = torch.norm(axis_angle + 1e-8, p=2, dim=-1, keepdim=True)  # (..., 1)
    axis = axis_angle / angle                                          # (..., 3)
    cos = torch.cos(angle)                                             # (..., 1)
    sin = torch.sin(angle)                                             # (..., 1)

    # 拆分 axis 三个分量用于构建反对称矩阵 K
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]           # 均为 (..., 1)

    # 构造反对称矩阵 K（叉积矩阵 K，使得 K @ v = axis × v）
    # K = [[ 0, -z,  y],
    #      [ z,  0, -x],
    #      [-y,  x,  0]]
    row1 = torch.cat([torch.zeros_like(x), -z, y], dim=-1)              # (..., 3)
    row2 = torch.cat([z, torch.zeros_like(y), -x], dim=-1)
    row3 = torch.cat([-y, x, torch.zeros_like(z)], dim=-1)
    K = torch.stack([row1, row2, row3], dim=-2)                         # (..., 3, 3)

    # 单位阵（与 K 同形状）
    I = torch.eye(3, device=axis_angle.device).reshape(1, 3, 3).expand(*K.shape[:-2], -1, -1)
    # Rodrigues 公式
    R = I + sin.unsqueeze(-1) * K + (1 - cos.unsqueeze(-1)) * (K @ K)
    return R


def rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    将 (..., 3, 3) 旋转矩阵转换为 (..., 3) axis-angle 向量。
    是 axis_angle_to_rotmat 的逆操作。

    数值细节:
        接近 0 角度旋转时直接返回 0 向量（避免除零）。
    """
    # trace(R) = 1 + 2*cos(theta)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1) / 2
    # 截断到 [-1, 1] 防止 acos 数值越界
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)                                       # (...,)

    # 标记近零旋转（直接置 0）
    near_zero = theta < 1e-6
    theta_safe = theta.clone()
    theta_safe[near_zero] = 1.0  # 占位避免除零

    # 旋转轴 = (R - R^T) / (2*sin(theta))  的反对称部分
    x = (R[..., 2, 1] - R[..., 1, 2]) / (2 * theta_safe)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (2 * theta_safe)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (2 * theta_safe)

    # axis-angle = axis * theta
    aa = torch.stack([x, y, z], dim=-1) * theta.unsqueeze(-1)
    aa[near_zero] = 0.0
    return aa


def build_SE3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    从旋转矩阵 R 和平移向量 t 构造 4x4 SE(3) 齐次变换矩阵。

        T = [[R, t],
             [0, 1]]

    Args:
        R: (..., 3, 3) 旋转矩阵
        t: (..., 3)    平移向量

    Returns:
        pose: (..., 4, 4) SE(3) 变换矩阵
    """
    batch_shape = R.shape[:-2]
    device = R.device
    # 初始化为单位 SE(3) 矩阵（齐次坐标）
    pose = torch.eye(4, device=device).expand(*batch_shape, 4, 4).contiguous()
    pose[..., :3, :3] = R   # 左上 3x3 = 旋转
    pose[..., :3, 3] = t    # 右上 3x1 = 平移
    return pose


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    将 (..., 3, 3) 旋转矩阵转换为 (..., 4) 单位四元数 [w, x, y, z]。

    使用 **Shepperd 方法**（分支选择）保证数值稳定性：
        - 优先选择 w 分支（trace > 0）
        - 否则选最大对角元对应的分支（x/y/z）
    避免在旋转角接近 π 时除零。

    数学原理（四分支对应四元数的四种"最大分量"）:
        w-branch: w = 0.5 * sqrt(1 + trace)
                  x = (R21 - R12) / (4w)
                  y = (R02 - R20) / (4w)
                  z = (R10 - R01) / (4w)
        x-branch: x = 0.5 * sqrt(1 + R00 - R11 - R22)
                  ...
        y/z 分支同理，对应置换下标。

    Args:
        R: (..., 3, 3) 旋转矩阵（正交阵，det=+1）

    Returns:
        q: (..., 4) 单位四元数 [w, x, y, z]
    """
    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]

    # ---- 分支 0：w 最大（最常见，trace > 0 的旋转角度 < π）----
    s0 = torch.sqrt(torch.clamp(m00 + m11 + m22 + 1.0, min=1e-10)) * 2  # = 4w
    q0_w = 0.25 * s0
    q0_x = (m21 - m12) / s0
    q0_y = (m02 - m20) / s0
    q0_z = (m10 - m01) / s0
    q0 = torch.stack([q0_w, q0_x, q0_y, q0_z], dim=-1)

    # ---- 分支 1：x 最大（R00 是最大对角元）----
    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-10)) * 2  # = 4x
    q1_w = (m21 - m12) / s1
    q1_x = 0.25 * s1
    q1_y = (m01 + m10) / s1
    q1_z = (m02 + m20) / s1
    q1 = torch.stack([q1_w, q1_x, q1_y, q1_z], dim=-1)

    # ---- 分支 2：y 最大（R11 最大）----
    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-10)) * 2  # = 4y
    q2_w = (m02 - m20) / s2
    q2_x = (m01 + m10) / s2
    q2_y = 0.25 * s2
    q2_z = (m12 + m21) / s2
    q2 = torch.stack([q2_w, q2_x, q2_y, q2_z], dim=-1)

    # ---- 分支 3：z 最大（R22 最大）----
    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-10)) * 2  # = 4z
    q3_w = (m10 - m01) / s3
    q3_x = (m02 + m20) / s3
    q3_y = (m12 + m21) / s3
    q3_z = 0.25 * s3
    q3 = torch.stack([q3_w, q3_x, q3_y, q3_z], dim=-1)

    # ---- 选分支：trace > 0 用分支 0；否则挑最大对角元对应的分支 ----
    trace = m00 + m11 + m22
    use_branch0 = trace > 0

    m00_is_max = (m00 >= m11) & (m00 >= m22)
    m11_is_max = (~m00_is_max) & (m11 >= m22)

    # 默认 q0
    q = q0
    # 替换为分支 1
    sel1 = (~use_branch0) & m00_is_max
    q = torch.where(sel1.unsqueeze(-1), q1, q)
    # 替换为分支 2
    sel2 = (~use_branch0) & (~m00_is_max) & m11_is_max
    q = torch.where(sel2.unsqueeze(-1), q2, q)
    # 替换为分支 3（剩下就是 R22 最大）
    sel3 = (~use_branch0) & (~m00_is_max) & (~m11_is_max)
    q = torch.where(sel3.unsqueeze(-1), q3, q)

    # 单位化（防止极小数值漂移）
    q = q / torch.norm(q, dim=-1, keepdim=True)
    return q


def compute_mesh_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    计算 mesh 的**顶点法向**（带方向修正：确保朝外）。

    Args:
        vertices: (V, 3) 顶点坐标
        faces:    (F, 3) 三角形顶点索引

    Returns:
        normals:  (V, 3) 单位长度的顶点法向（朝外）
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()  # 关键：修正面 winding 和法向方向，确保朝外
    normals = mesh.vertex_normals.copy()
    return normals


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    计算 mesh 的**面法向**（per-face normal），用于 face-center 采样。

    为什么用 face normal 而不是 vertex normal?
        - 我们对手部采用 face center 采样（每面一个点）
        - face normal 跟采样点天然对应，比 vertex normal 更合适
        - vertex normal 需要从相邻 face 法向加权平均，会扩散到不属于该 face 的方向

    Args:
        vertices: (V, 3) 顶点坐标（已经是"posed"姿态）
        faces:    (F, 3) 三角形顶点索引

    Returns:
        face_normals: (F, 3) 单位长度的面法向（朝外）
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()  # 确保方向一致
    normals = mesh.face_normals.copy()
    # 单位化（数值上其实已归一化，这里是双保险）
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(norms, 1e-10, None)
    return normals


def sample_mesh_surface(mesh: trimesh.Trimesh, n_points: int, seed: int = 42):
    """
    在 mesh 表面**均匀随机采样** n_points 个点。

    Returns:
        points:      (N, 3)  采样点坐标（canonical 空间）
        face_idx:    (N,)    每个点所属的 face 索引
        barycentric: (N, 3)  每个点在所属 face 内的重心坐标（用于插值顶点属性）
    """
    np.random.seed(seed)  # 保证跨进程一致
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    vertices = mesh.vertices
    faces = mesh.faces[face_idx]
    v0 = vertices[faces[:, 0]]    # (N, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    triangles = np.stack([v0, v1, v2], axis=1)  # (N, 3, 3) 三个顶点
    # 把采样点转成 face 内的重心坐标
    barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
    # trimesh 在数值奇异时会产生 NaN，退化为均匀重心
    barycentric = np.nan_to_num(barycentric, nan=1.0 / 3.0)
    return points, face_idx, barycentric


def interpolate_vertex_attributes(
    face_idx: np.ndarray,
    barycentric: np.ndarray,
    vert_attr: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    在采样点位置**插值**每顶点属性（如法向、位置等）。

    利用重心坐标，在每个 face 上做仿射组合:
        attr(p) = u * attr(v0) + v * attr(v1) + w * attr(v2)
    其中 (u, v, w) 是点 p 在该 face 中的重心坐标。

    Args:
        face_idx:    (N,)   每个采样点的 face 索引
        barycentric: (N, 3) 每个采样点的重心坐标
        vert_attr:   (V, D) 顶点属性（如位置、法向）
        faces:       (F, 3) face 顶点索引表

    Returns:
        sampled: (N, D) 采样点上的属性
    """
    fv = faces[face_idx]      # (N, 3) 每个采样点所在 face 的三个顶点索引
    attr0 = vert_attr[fv[:, 0]]   # (N, D)
    attr1 = vert_attr[fv[:, 1]]
    attr2 = vert_attr[fv[:, 2]]
    bary = barycentric            # (N, 3)
    return (
        bary[:, 0:1] * attr0
        + bary[:, 1:2] * attr1
        + bary[:, 2:3] * attr2
    )


def assign_hand_semantics(
    mano_layer: MANO,
    is_right: bool,
) -> tuple:
    """
    给 MANO 的每个 face（中心点）分配**语义标签**:
        - finger_id: 0=palm, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
        - region_id: 0=palm, 1=fingertip, 2=finger_pad

    算法思路（启发式）:
        1. 在 canonical（零姿态）下手部 forward 一次，得到 778 顶点和 16 关节位置
        2. 计算每个 face 的中心点（1538 个）
        3. 对每个 face center，找最近的关节 → 得到 finger_id
        4. 如果最近关节是某个手指的 tip（DIP），标记为 fingertip (region=1)
           如果最近关节是 wrist（joint 0），标记为 palm (region=0)
           否则标记为 finger_pad (region=2)

    注意:
        - 这里在 canonical 空间算一次即可，因为 face 拓扑固定、关节相对位置大致固定
        - 整个数据集的 finger_id / region_id 都是一样的（不随姿态变化）

    Returns:
        finger_id: (1538,) int32  每个 face 属于哪根手指
        region_id: (1538,) int32  每个 face 属于哪个区域
    """
    device = next(mano_layer.parameters()).device

    # 1) Canonical forward
    with torch.no_grad():
        output = mano_layer()  # 默认零姿态
        verts = output.vertices.squeeze(0)    # (778, 3)
        joints = output.joints.squeeze(0)     # (16, 3)

    verts_np = verts.cpu().numpy()
    joints_np = joints.cpu().numpy()
    faces = mano_layer.faces                 # (1538, 3) int

    # 2) Face center = 三个顶点坐标的均值
    face_centers = verts_np[faces].mean(axis=1)   # (1538, 3)
    num_faces = face_centers.shape[0]              # 1538

    # 3) 为每个 face center 找最近的关节 → 映射到 finger_id
    from scipy.spatial import cKDTree
    tree = cKDTree(joints_np)
    _, idxs = tree.query(face_centers, k=1)        # (1538,)

    # 用 JOINT_TO_FINGER 表把关节索引转成 finger_id（防御性：超出范围填 0=palm）
    def joint_to_finger(j_idx):
        return JOINT_TO_FINGER[j_idx] if j_idx < len(JOINT_TO_FINGER) else 0

    finger_id = np.array([joint_to_finger(j) for j in idxs], dtype=np.int32)

    # 4) 划分 region:
    #    - 关节 0 是 wrist → palm
    #    - 关节 3, 6, 9, 12, 15 是 thumb/index/middle/ring/pinky 的 DIP（指尖）→ fingertip
    #    - 其余 MCP / PIP → finger_pad
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


def load_object_mesh(obj_name: str, unit: str = "m") -> tuple:
    """
    加载物体的**canonical mesh**（规范姿态，原始坐标系）和 parts 标签。

    Args:
        obj_name: 物体名称, e.g. "box", "phone"
        unit:     mesh.obj 顶点坐标单位，'m' 或 'mm'。
                  统一用 'm' 即可（推荐）；如物体原始是 mm，本函数会除 1000 转成 m。
                  旧启发式 'auto' 保留兼容（max>10→mm，否则 m），但不推荐。

    Returns:
        mesh:  trimesh.Trimesh  canonical 姿态的 mesh（**统一返回 m**）
        parts: (V,) bool  True=顶部件, False=底部件（铰接物体才有；刚体物体返回 None）
    """
    obj_dir = op.join(OBJECT_VTEMPLATE_DIR, obj_name)
    mesh_p = op.join(obj_dir, "mesh.obj")
    parts_p = op.join(obj_dir, "parts.json")

    mesh = trimesh.load(mesh_p, process=False)
    if unit == "mm":
        mesh.vertices = mesh.vertices / 1000.0
    elif unit == "m":
        pass  # 已经在 m 单位
    elif unit == "auto":
        # ⚠️ 不推荐的旧启发式：max>10 视为 mm；可能漏判小物体
        if mesh.vertices.max() > 10:
            mesh.vertices = mesh.vertices / 1000.0
    else:
        raise ValueError(f"Unknown unit {unit!r}; expected 'm', 'mm', or 'auto'")

    parts = None
    if op.exists(parts_p):
        with open(parts_p, "r") as f:
            parts_raw = json.load(f)
        parts = np.array(parts_raw, dtype=bool)   # (V,)

    return mesh, parts


def compute_hand_joint_quaternions(
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    parents: np.ndarray,
) -> torch.Tensor:
    """
    沿 MANO 运动学树组合全局旋转，得到 16 个关节的世界空间四元数。

    关节顺序（与 smplx MANO 一致）:
        0        = wrist（腕，根）
        1, 2, 3  = thumb（拇指 CMC/MCP/IP）
        4, 5, 6  = index（食指 MCP/PIP/DIP）
        7, 8, 9  = middle（中指 MCP/PIP/DIP）
        10,11,12 = ring（无名指 MCP/PIP/DIP）
        13,14,15 = pinky（小指 MCP/PIP/DIP）

    hand_pose 形状 (T, 45) 顺序: [thumb(0-2), index(3-5), middle(6-8), ring(9-11), pinky(12-14)]
    即 hand_pose[:, 3*k:3*(k+1)] 对应关节 3*k+1（thumb→index→middle→ring→pinky）。

    Args:
        global_orient: (T, 3)  全局根节点 axis-angle
        hand_pose:     (T, 45) 15 个手指关节 axis-angle（展平）
        parents:       (16,)   MANO 运动学树 parents
                       parents[0] = -1（根）, parents[1]=0, parents[2]=1, ...

    Returns:
        quat_world: (T, 16, 4) 16 关节的世界旋转四元数 [w, x, y, z]
    """
    T = global_orient.shape[0]
    device = global_orient.device

    # 1) 根节点 R_world[0] = R(global_orient)
    R_world = torch.zeros((T, 16, 3, 3), device=device, dtype=global_orient.dtype)
    R_world[:, 0] = axis_angle_to_rotmat(global_orient)        # (T, 3, 3)

    # 2) hand_pose 重塑为 (T, 15, 3)，再转 (T, 15, 3, 3) 局部旋转
    hand_pose_aa = hand_pose.reshape(T, 15, 3)
    R_local = axis_angle_to_rotmat(hand_pose_aa)               # (T, 15, 3, 3)

    # 3) 沿运动学树组合：R_world[j] = R_world[parent[j]] @ R_local[j-1]
    for j in range(1, 16):
        p = int(parents[j])
        R_world[:, j] = R_world[:, p] @ R_local[:, j - 1]

    # 4) 转四元数
    quat_world = rotmat_to_quat(R_world)                      # (T, 16, 4)
    return quat_world


def transform_object_points(
    obj_points_canonical: np.ndarray,
    obj_normals_canonical: np.ndarray,
    face_idx: np.ndarray,
    barycentric: np.ndarray,
    obj_mesh: trimesh.Trimesh,
    parts: np.ndarray,
    arti_angle: float,
    global_rot: np.ndarray,
    global_trans: np.ndarray,
    canonical_faces: np.ndarray,
) -> tuple:
    """
    把物体表面点从 **canonical 空间** 变换到 **世界（posed）空间**。

    变换流程（与 ARCTIC 官方实现一致）:
        Step 1: 铰接变换（仅铰接物体）—— 顶部件绕 z 轴旋转 arti_angle
        Step 2: 全局刚体变换 —— 旋转 R_global + 平移 t_global
        Step 3: 用 barycentric 插值得到采样点位置
        Step 4: 重新计算 posed mesh 的法向并插值得到采样点法向

    注意 (与参考实现对齐):
        参考脚本 (`visualize_arctic_dataset.py`) 用的是 R_z(-angle) 复合 .T 操作，
        等价于在 column 向量约定下应用 R_z(-angle)。本函数保持一致。

    Args:
        obj_points_canonical: (No, 3) canonical 空间下的采样点
        obj_normals_canonical: (No, 3) canonical 空间下的法向
        face_idx:              (No,)   每个采样点所属 face
        barycentric:           (No, 3) 重心坐标
        obj_mesh:              canonical 网格（用于取 vertices）
        parts:                 (V,) bool  顶/底部件 mask（刚体物体传 None）
        arti_angle:            float   铰接角（弧度）
        global_rot:            (3,)    全局旋转的 axis-angle
        global_trans:          (3,)    全局平移（米）
        canonical_faces:       (F, 3)  face 顶点索引

    Returns:
        posed_points:  (No, 3) 世界空间下的采样点
        posed_normals: (No, 3) 世界空间下的法向
    """
    vertices_canon = obj_mesh.vertices.copy()    # (V, 3) canonical 顶点
    faces = canonical_faces

    # Step 1: 铰接变换（仅对铰接物体）
    # 参考实现约定: 用 R_z(-angle) 旋转顶部件
    if parts is not None and np.any(parts):
        c = np.cos(arti_angle)
        s = np.sin(arti_angle)
        # R_z(-angle) = [[cos,  sin, 0],
        #                [-sin, cos, 0],
        #                [0,    0,   1]]
        R_arti = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)

        # 只旋转顶部件的顶点，底部件保持不变
        verts_articulated = vertices_canon.copy()
        top_mask = parts
        verts_articulated[top_mask] = (R_arti @ vertices_canon[top_mask].T).T
    else:
        # 刚体物体：跳过铰接
        verts_articulated = vertices_canon

    # Step 2: 全局刚体变换
    R_global = axis_angle_to_rotmat(
        torch.from_numpy(global_rot[None, :])
    ).squeeze(0).numpy()

    # 应用旋转 + 平移，得到 posed 顶点
    verts_posed = (R_global @ verts_articulated.T).T + global_trans

    # Step 3: 通过 barycentric 插值，把 canonical 空间采样点映射到 posed 空间
    posed_points = interpolate_vertex_attributes(
        face_idx, barycentric, verts_posed, faces
    )

    # Step 4: 法向要重新计算（用 posed 后的 mesh）
    # 因为旋转后 face 方向变了，不能直接 rotate 原来的法向（铰接物体底/顶分别旋转时尤其重要）
    posed_mesh = trimesh.Trimesh(vertices=verts_posed, faces=faces, process=False)
    posed_mesh.fix_normals()  # 修正方向
    posed_vertex_normals = posed_mesh.vertex_normals

    posed_normals = interpolate_vertex_attributes(
        face_idx, barycentric, posed_vertex_normals, faces
    )

    # 重新归一化
    norms = np.linalg.norm(posed_normals, axis=-1, keepdims=True)
    posed_normals = posed_normals / np.clip(norms, 1e-10, None)

    return posed_points, posed_normals


# ============================================================
# 核心处理逻辑
# ============================================================

class ArcticPreprocessor:
    """
    ARCTIC 数据集预处理器。

    职责:
        1. 初始化 MANO 左右手模型（canonical）
        2. 计算手部语义标签（finger_id / region_id）
        3. 对每条 ARCTIC 序列:
            - 读 .mano.npy + .object.npy
            - forward MANO 得到 778 顶点
            - 转换成 face-center 采样（1538 点）
            - 对物体表面采样（2048 点）
            - 计算 flow / NN / 根节点 pose
            - 序列化为 .npz 文件

    实例变量说明:
        - mano_r / mano_l: 左右手 MANO 模型
        - right_finger_id / right_region_id: 右手的语义标签
        - left_finger_id / left_region_id: 左手的语义标签
        - right_faces / left_faces: MANO 的 face 索引表（固定 1538 个）
        - num_hand_points: 1538
        - right_hand_point_id: 右手 face 中心点编号 [0..1537]
        - _obj_cache: 物体名 → canonical 采样结果（避免重复采样）
    """

    def __init__(
        self,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        num_obj_points: int = 2048,
        device: str = "cuda:0",
        dt: float = 1.0 / 30.0,
        contact_distance_thresh: float = CONTACT_DISTANCE_THRESH,
        max_frames: Optional[int] = None,
        obj_unit: str = "mm",      # mesh.obj 单位：'m' / 'mm' / 'auto'
                                    # 默认 'mm'：ARCTIC 原始 mesh.obj 是 mm（box max=244.91mm），
                                    # 函数内部会 /1000 转成 m，pipeline 统一 m。
    ):
        """
        Args:
            output_root:           输出根目录
            num_obj_points:        每个物体表面采样点数（默认 2048）
            device:                PyTorch 设备（参数保留，目前强制用 CPU）
            dt:                    帧间隔秒数（默认 1/30，对应 30 FPS）
            contact_distance_thresh: 手-物接触距离阈值（米），默认 0.03m (3cm)
                                    超过该距离的帧会被判定为"无接触"并屏蔽该手的数据
            max_frames:            单条序列最多处理前 N 帧（None=不限制）。
                                    主要用于 quick smoke test；与 cpf_fit 的
                                    --max-frames 不冲突（cpf_fit 在此基础上再步长采样）。
        """
        self.output_root = output_root
        self.num_obj_points = num_obj_points
        # 强制使用 CPU（该环境无 CUDA / 不想强依赖）
        self.device = "cpu"
        self.dt = dt
        self.contact_distance_thresh = float(contact_distance_thresh)
        self.max_frames = max_frames  # None=不限；>0=只处理前 N 帧（quick test 用）
        self.obj_unit = obj_unit      # mesh.obj 单位：'m' / 'mm' / 'auto'

        # ----- 加载 MANO 左右手模型 -----
        print(f"[Preprocessor] Loading MANO models...")
        # 重要：ARCTIC 提供的 MANO 参数是用 `flat_hand_mean=False` 拟合的
        # （与 arctic_mano_collision_opt._build_arctic_mano_layer 一致）。
        # 若误用 `flat_hand_mean=True`，smplx 内部会把手部 pose 增量解释为
        # 相对"平直"姿态，导致指尖位置偏移最多约 6.5cm。
        self.mano_r = MANO(MANO_MODEL_DIR, is_rhand=True, use_pca=False,
                            flat_hand_mean=False).to(self.device)
        self.mano_l = MANO(MANO_MODEL_DIR, is_rhand=False, use_pca=False,
                            flat_hand_mean=False).to(self.device)

        # ----- 计算手部语义标签（finger_id / region_id） -----
        # 整个数据集共用同一组标签（拓扑稳定），只算一次缓存即可
        print(f"[Preprocessor] Computing hand semantics...")
        (
            self.right_finger_id,
            self.right_region_id,
        ) = assign_hand_semantics(self.mano_r, is_right=True)
        (
            self.left_finger_id,
            self.left_region_id,
        ) = assign_hand_semantics(self.mano_l, is_right=False)

        # ----- 缓存 MANO face 索引 -----
        # MANO 拓扑固定，face 索引在 forward 之后可以直接从 layer.faces 取
        self.right_faces = self.mano_r.faces.astype(np.int64)   # (1538, 3)
        self.left_faces = self.mano_l.faces.astype(np.int64)
        self.num_hand_points = self.right_faces.shape[0]        # 1538

        # ----- 缓存 MANO 运动学树 parents -----
        # 用于关节链组合，得到 16 关节世界旋转以计算四元数
        # parents[0] = -1（根），parents[1..15] 描述 15 个手指关节的父子关系
        # smplx 把它存成 torch.Tensor，统一转到 numpy 便于后续处理
        self.mano_parents = self.mano_r.parents.detach().cpu().numpy().astype(np.int64)   # (16,)

        # 持久化的点 ID（左右手各自 0..1537，方便按 face 索引取同一拓扑点）
        self.right_hand_point_id = np.arange(self.num_hand_points, dtype=np.int32)
        self.left_hand_point_id = np.arange(self.num_hand_points, dtype=np.int32)

        # 物体采样缓存（同一物体只需采样一次）
        self._obj_cache = {}

        print(f"[Preprocessor] Output dir: {output_root}")
        print(f"[Preprocessor] Device: {self.device}")
        print(f"[Preprocessor] Num obj points: {num_obj_points}")
        print(f"[Preprocessor] Contact distance thresh: "
              f"{self.contact_distance_thresh*100:.2f} cm")

    def _get_obj_sampling(self, obj_name: str):
        """
        加载或计算指定物体的**canonical 表面采样**。

        一次采样后会缓存到 self._obj_cache，同一物体名的所有序列共享同一组采样点。
        这样不同序列/帧的物体点云天然具备**跨帧一致的 point_id**。

        缓存内容:
            - mesh:   canonical 网格
            - parts:  顶/底部件 mask（None 表示刚体）
            - faces:  (F, 3) face 顶点索引
            - points: (No, 3) 采样点坐标
            - normals:(No, 3) 采样点法向
            - point_id: (No,) 持久化点编号
            - face_idx: (No,) 每个点所属 face
            - barycentric: (No, 3) 重心坐标
            - sample_parts: (No,) bool  采样点属于顶/底部件

        Returns:
            cache 字典
        """
        if obj_name not in self._obj_cache:
            mesh, parts = load_object_mesh(obj_name, unit=self.obj_unit)

            # 1) 在 canonical mesh 表面均匀采样
            pts, face_idx, bary = sample_mesh_surface(
                mesh, self.num_obj_points, seed=42
            )

            # 2) 计算 canonical 法向
            mesh.fix_normals()
            vn = mesh.vertex_normals
            normals = interpolate_vertex_attributes(
                face_idx, bary, vn, mesh.faces
            )
            norms = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = normals / np.clip(norms, 1e-10, None)

            # 3) 标记每个采样点属于顶/底部件（用最近顶点的 part）
            if parts is not None:
                tree = cKDTree(mesh.vertices)
                _, nearest_v = tree.query(pts, k=1)
                sample_parts = parts[nearest_v]
            else:
                sample_parts = None

            cache = {
                "mesh": mesh,
                "parts": parts,
                "faces": mesh.faces.copy(),
                "points": pts,           # (No, 3)
                "normals": normals,      # (No, 3)
                "point_id": np.arange(self.num_obj_points, dtype=np.int32),
                "face_idx": face_idx,
                "barycentric": bary,
                "sample_parts": sample_parts,
            }
            self._obj_cache[obj_name] = cache
            print(f"[Preprocessor] Cached {obj_name}: {len(mesh.vertices)} verts, "
                  f"{self.num_obj_points} samples")

        return self._obj_cache[obj_name]

    def process_sequence(
        self, mano_p: str
    ) -> dict:
        """
        处理单条 ARCTIC 序列，返回输出字典。

        主要步骤:
            1. 解析路径 → 拿到 subject / seq_name / obj_name
            2. 读 .mano.npy + .object.npy
            3. 分批 forward MANO 左右手（避免一次性吃光内存）
            4. 把 MANO 顶点转成 face-center 采样（1538 点）
            5. 计算每帧的 posed 物体采样点（铰接 + 全局变换）
            6. 构造 root_pose（SE(3)）
            7. 计算 flow（t -> t+1）
            8. 计算双向 NN（手↔物）
            9. 组装输出 dict

        Args:
            mano_p:  .mano.npy 文件的绝对路径

        Returns:
            output:  包含所有 .npz 字段的 dict（见文件顶部 docstring）
        """
        # 路径解析: RAW_SEQS_DIR/s01/box_grab_01.mano.npy
        seq_rel = mano_p.replace(RAW_SEQS_DIR + "/", "")
        seq_rel = seq_rel.replace(".mano.npy", "")
        subject, seq_name = seq_rel.split("/")
        obj_name = seq_name.split("_")[0]   # "box_grab_01" → "box"

        # 同名 .object.npy 存了物体参数
        obj_p = mano_p.replace(".mano.npy", ".object.npy")

        # ----- 读原始数据 -----
        mano_data = np.load(mano_p, allow_pickle=True).item()    # dict
        obj_params = np.load(obj_p, allow_pickle=True)            # (T, 7)

        T = obj_params.shape[0]   # 帧数

        # ----- 裁帧 (max_frames) -----
        # 目的：quick smoke test 时不需要跑整条 700+ 帧的序列。
        # 在所有按 T 切片的逻辑之前裁一次，下游 NN / flow / valid 等会自动
        # 只算被保留的 0..T' 帧。
        if self.max_frames is not None and T > self.max_frames:
            T = int(self.max_frames)
            obj_params = obj_params[:T]
            for side in ("right", "left"):
                for k in ("rot", "pose", "trans"):
                    if k in mano_data[side]:
                        mano_data[side][k] = mano_data[side][k][:T]
            # shape 不裁（一条序列一个 subject 一套手型）

        # ----- 提取并转 torch 的 MANO 参数 -----
        # rot: 全局根节点旋转（axis-angle）
        # pose: 15 个手指关节 axis-angle（展平 45 = 15 * 3）
        # trans: 根节点平移
        # shape: 形状参数 betas
        rot_r = torch.FloatTensor(mano_data["right"]["rot"]).to(self.device)      # (T, 3)
        pose_r = torch.FloatTensor(mano_data["right"]["pose"]).to(self.device)    # (T, 45)
        trans_r = torch.FloatTensor(mano_data["right"]["trans"]).to(self.device)  # (T, 3)
        shape_r = torch.FloatTensor(mano_data["right"]["shape"]).to(self.device)  # (10,)

        rot_l = torch.FloatTensor(mano_data["left"]["rot"]).to(self.device)
        pose_l = torch.FloatTensor(mano_data["left"]["pose"]).to(self.device)
        trans_l = torch.FloatTensor(mano_data["left"]["trans"]).to(self.device)
        shape_l = torch.FloatTensor(mano_data["left"]["shape"]).to(self.device)

        # ----- 提取物体参数（7 列含义）-----
        #   col 0    = 铰接角（弧度）
        #   col 1:4  = 物体全局旋转 axis-angle
        #   col 4:7  = 物体全局平移（毫米）
        arti = obj_params[:, 0:1]           # (T, 1) 铰接角
        obj_rot_aa = obj_params[:, 1:4]     # (T, 3) axis-angle
        obj_trans = obj_params[:, 4:7]      # (T, 3) 毫米
        obj_trans_m = obj_trans / 1000.0    # 转米（与 MANO 单位一致）

        # ----- MANO Forward（分批避免 OOM）-----
        batch_size = min(T, 320)
        all_right_verts = []
        all_right_joints = []
        all_left_verts = []
        all_left_joints = []

        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            b = end - i

            # 右手: shape 兼容 (10,) 和 (T, 10) 两种存储形式
            shape_r_batch = shape_r.expand(b, -1) if shape_r.ndim == 1 else shape_r[:b]
            out_r = self.mano_r(
                global_orient=rot_r[i:end],
                hand_pose=pose_r[i:end],
                betas=shape_r_batch,
                transl=trans_r[i:end],
            )
            all_right_verts.append(out_r.vertices.detach().cpu().numpy())
            all_right_joints.append(out_r.joints.detach().cpu().numpy())

            # 左手同理
            shape_l_batch = shape_l.expand(b, -1) if shape_l.ndim == 1 else shape_l[:b]
            out_l = self.mano_l(
                global_orient=rot_l[i:end],
                hand_pose=pose_l[i:end],
                betas=shape_l_batch,
                transl=trans_l[i:end],
            )
            all_left_verts.append(out_l.vertices.detach().cpu().numpy())
            all_left_joints.append(out_l.joints.detach().cpu().numpy())

        # 拼成 (T, 778, 3) / (T, 16, 3)
        right_verts = np.concatenate(all_right_verts, axis=0)     # (T, 778, 3)
        right_joints = np.concatenate(all_right_joints, axis=0)   # (T, 16, 3)
        left_verts = np.concatenate(all_left_verts, axis=0)       # (T, 778, 3)
        left_joints = np.concatenate(all_left_joints, axis=0)     # (T, 16, 3)

        # ----- MANO 顶点 → face-center 采样（1538 点）-----
        # 每个 face 三个顶点取均值，得到一个"中心点"
        # 由于 MANO 拓扑固定，face i 在所有手型下都是同一个三角形 → 拓扑稳定
        right_face_pts = right_verts[:, self.right_faces].mean(axis=2)  # (T, 1538, 3)
        left_face_pts = left_verts[:, self.left_faces].mean(axis=2)     # (T, 1538, 3)

        # ----- 计算每帧的面法向（per-face normal）-----
        # 用 posed mesh 的 face normal（不是 vertex normal），与 face center 天然对应
        right_normals = np.zeros_like(right_face_pts)
        left_normals = np.zeros_like(left_face_pts)

        for t in range(T):
            right_normals[t] = compute_face_normals(
                right_verts[t], self.right_faces
            )
            left_normals[t] = compute_face_normals(
                left_verts[t], self.left_faces
            )

        # ----- 物体：读 canonical 采样，逐帧变换 -----
        obj_cache = self._get_obj_sampling(obj_name)
        obj_canon_points = obj_cache["points"]       # (No, 3)
        obj_canon_normals = obj_cache["normals"]      # (No, 3)
        obj_point_id = obj_cache["point_id"]          # (No,)
        obj_face_idx = obj_cache["face_idx"]          # (No,)
        obj_barycentric = obj_cache["barycentric"]    # (No, 3)
        obj_mesh = obj_cache["mesh"]
        obj_parts = obj_cache["parts"]
        obj_faces = obj_cache["faces"]

        # 逐帧变换到世界坐标
        obj_points = np.zeros((T, self.num_obj_points, 3), dtype=np.float32)
        obj_normals = np.zeros((T, self.num_obj_points, 3), dtype=np.float32)

        for t in range(T):
            pts, nmls = transform_object_points(
                obj_canon_points, obj_canon_normals,
                obj_face_idx, obj_barycentric,
                obj_mesh, obj_parts,
                arti[t, 0], obj_rot_aa[t], obj_trans_m[t],
                obj_faces,
            )
            obj_points[t] = pts
            obj_normals[t] = nmls

        # ----- 计算根节点 SE(3) pose -----
        # 右手根节点: rot_r (axis-angle) + trans_r
        R_r = axis_angle_to_rotmat(rot_r.cpu()).numpy()
        right_root_pose = build_SE3(
            torch.from_numpy(R_r),
            trans_r.cpu(),
        ).numpy()

        # 左手根节点
        R_l = axis_angle_to_rotmat(rot_l.cpu()).numpy()
        left_root_pose = build_SE3(
            torch.from_numpy(R_l),
            trans_l.cpu(),
        ).numpy()

        # 物体根节点
        R_o = axis_angle_to_rotmat(torch.from_numpy(obj_rot_aa)).numpy()
        obj_root_pose = build_SE3(
            torch.from_numpy(R_o),
            torch.from_numpy(obj_trans_m),
        ).numpy()

        # ----- 手部 15 个手指关节的 axis-angle -----
        # pose_r 形状是 (T, 45) = 15 * 3，重塑为 (T, 15, 3)
        # 顺序：thumb(0-2), index(3-5), middle(6-8), ring(9-11), pinky(12-14)
        right_joint_aa = pose_r.cpu().numpy().reshape(T, 15, 3)   # (T, 15, 3)
        left_joint_aa = pose_l.cpu().numpy().reshape(T, 15, 3)

        # ----- 帧间差分（用于序列建模，捕捉"动作增量"）-----
        right_joint_aa_delta = np.zeros_like(right_joint_aa)
        left_joint_aa_delta = np.zeros_like(left_joint_aa)
        if T > 1:
            right_joint_aa_delta[:-1] = right_joint_aa[1:] - right_joint_aa[:-1]
            left_joint_aa_delta[:-1] = left_joint_aa[1:] - left_joint_aa[:-1]
        # 最后一帧没有下一帧，差分置 0

        # ----- 16 关节世界旋转四元数 [w, x, y, z] -----
        # 沿 MANO 运动学树组合 global_orient + hand_pose，再转四元数
        right_joint_quat = compute_hand_joint_quaternions(
            rot_r, pose_r, self.mano_parents
        ).cpu().numpy().astype(np.float32)                         # (T, 16, 4)
        left_joint_quat = compute_hand_joint_quaternions(
            rot_l, pose_l, self.mano_parents
        ).cpu().numpy().astype(np.float32)                          # (T, 16, 4)

        # ----- MANO 形状参数 β -----
        # ARCTIC 里 shape 是 (10,) 或 (T, 10)，统一广播为 (T, 10)
        if shape_r.ndim == 1:
            right_betas = shape_r.cpu().numpy()[None, :].repeat(T, axis=0).astype(np.float32)
            left_betas = shape_l.cpu().numpy()[None, :].repeat(T, axis=0).astype(np.float32)
        else:
            right_betas = shape_r.cpu().numpy().astype(np.float32)
            left_betas = shape_l.cpu().numpy().astype(np.float32)

        # ----- 流场（t -> t+1 的位移）-----
        right_hand_flow = np.zeros((T, self.num_hand_points, 3), dtype=np.float32)
        left_hand_flow = np.zeros((T, self.num_hand_points, 3), dtype=np.float32)
        obj_flow = np.zeros((T, self.num_obj_points, 3), dtype=np.float32)

        if T > 1:
            right_hand_flow[:-1] = right_face_pts[1:] - right_face_pts[:-1]
            left_hand_flow[:-1] = left_face_pts[1:] - left_face_pts[:-1]
            obj_flow[:-1] = obj_points[1:] - obj_points[:-1]
        # 最后一帧 flow 置 0

        # ----- flow 有效性掩码 -----
        # ARCTIC 提供了每帧的 MANO fitting 误差（毫米）
        # 误差过大的帧（>10mm）认为手部不可靠，flow_valid 设为 False
        fitting_err_r = np.array(mano_data["right"]["fitting_err"], dtype=np.float32)
        fitting_err_l = np.array(mano_data["left"]["fitting_err"], dtype=np.float32)

        hand_flow_valid_r = np.ones((T, self.num_hand_points), dtype=bool)
        hand_flow_valid_l = np.ones((T, self.num_hand_points), dtype=bool)

        if fitting_err_r.ndim == 1 and len(fitting_err_r) == T:
            bad_frames_r = fitting_err_r > FITTING_ERR_THRESH
            hand_flow_valid_r[bad_frames_r] = False
        if fitting_err_l.ndim == 1 and len(fitting_err_l) == T:
            bad_frames_l = fitting_err_l > FITTING_ERR_THRESH
            hand_flow_valid_l[bad_frames_l] = False

        # 物体 flow 默认为全部有效（ARCTIC 中物体位姿很可靠）
        obj_flow_valid = np.ones((T, self.num_obj_points), dtype=bool)

        # ----- 手-物双向最近邻（cKDTree）-----
        # 对每个手点找最近的物点；反之亦然
        # 用于训练接触预测、接触图等
        right_hand_to_obj_nn_id = np.zeros((T, self.num_hand_points), dtype=np.int32)
        right_hand_to_obj_dist = np.zeros((T, self.num_hand_points), dtype=np.float32)
        left_hand_to_obj_nn_id = np.zeros((T, self.num_hand_points), dtype=np.int32)
        left_hand_to_obj_dist = np.zeros((T, self.num_hand_points), dtype=np.float32)
        obj_to_right_hand_nn_id = np.zeros((T, self.num_obj_points), dtype=np.int32)
        obj_to_right_hand_dist = np.zeros((T, self.num_obj_points), dtype=np.float32)
        obj_to_left_hand_nn_id = np.zeros((T, self.num_obj_points), dtype=np.int32)
        obj_to_left_hand_dist = np.zeros((T, self.num_obj_points), dtype=np.float32)

        for t in range(T):
            # 右手 → 物体（在手点位置查物体 kd-tree）
            tree_obj = cKDTree(obj_points[t])
            dists, idxs = tree_obj.query(right_face_pts[t], k=1)
            right_hand_to_obj_nn_id[t] = idxs
            right_hand_to_obj_dist[t] = dists

            # 左手 → 物体
            dists, idxs = tree_obj.query(left_face_pts[t], k=1)
            left_hand_to_obj_nn_id[t] = idxs
            left_hand_to_obj_dist[t] = dists

            # 物体 → 右手
            tree_r = cKDTree(right_face_pts[t])
            dists, idxs = tree_r.query(obj_points[t], k=1)
            obj_to_right_hand_nn_id[t] = idxs
            obj_to_right_hand_dist[t] = dists

            # 物体 → 左手
            tree_l = cKDTree(left_face_pts[t])
            dists, idxs = tree_l.query(obj_points[t], k=1)
            obj_to_left_hand_nn_id[t] = idxs
            obj_to_left_hand_dist[t] = dists

        # ============================================================
        # 手-物接触距离判定（per-frame, per-hand）：只算掩码，不删/不清
        # ============================================================
        # 规则:
        #   - 每帧计算该手所有采样点到物体的最近距离 min_dist
        #   - 若 min_dist ≤ thresh  → 该帧该手 valid=True
        #   - 若 min_dist > thresh  → 该帧该手 valid=False
        #   - **不删除任何帧、不清零任何数据**：hand 数组始终保持 (T, ...) 形状
        #   - 双手独立判断，右手 valid 只描述右手，左手 valid 只描述左手
        #   - 下游消费者按需使用 valid 掩码（过滤/加权/丢弃）
        #   - 物体数据 (obj_*) 始终保留 T 帧，与 valid 无关
        #   - hand_flow / hand_flow_valid 仍按 fitting_err 阈值判断（与 contact 无关）
        thresh = self.contact_distance_thresh

        # 每帧最近距离 + 接触掩码
        right_min_per_frame = right_hand_to_obj_dist.min(axis=1)        # (T,)
        left_min_per_frame = left_hand_to_obj_dist.min(axis=1)          # (T,)
        # 3cm 掩码：默认接触阈值（CLI: --contact_distance_thresh 控制，向后兼容）
        right_valid = right_min_per_frame <= thresh                     # (T,) bool
        left_valid = left_min_per_frame <= thresh                       # (T,) bool
        # 2cm 掩码：硬编码 0.02m，与 CPF 内部 palm 顶点 range_threshold 一致
        # 用途：CPF / 强接触优化前先过滤掉 3cm~2cm 之间的"擦边"帧
        CONTACT_THRESH_2CM = 0.02  # 与 CPF range_threshold=20mm 对齐
        right_valid_2cm = right_min_per_frame <= CONTACT_THRESH_2CM     # (T,) bool
        left_valid_2cm = left_min_per_frame <= CONTACT_THRESH_2CM       # (T,) bool
        # 1cm 掩码：与 CPF range_threshold=10mm 严格对齐
        # 用途：避免 ARCTIC "face-center 离物 2cm / palm 离物 2.2cm" 边缘帧被
        #      contact_loss 误激活（contact 硬截断在 10mm 内）
        CONTACT_THRESH_1CM = 0.01
        right_valid_1cm = right_min_per_frame <= CONTACT_THRESH_1CM     # (T,) bool
        left_valid_1cm = left_min_per_frame <= CONTACT_THRESH_1CM       # (T,) bool

        # 将 min 距离也保存下来，下游可做 soft contact 软接触 label
        right_min_dist = right_min_per_frame.astype(np.float32)         # (T,)
        left_min_dist = left_min_per_frame.astype(np.float32)           # (T,)

        # ============================================================
        # 组装输出 dict（字段顺序与 .npz 写入顺序无关）
        # ============================================================
        seq_id = f"{subject}/{seq_name}"

        output = {
            # ----- 元信息 -----
            "seq_id": seq_id,
            "frame_id": np.arange(T, dtype=np.int32),         # 物体帧编号 (T,)

            # ----- 物体（保留全部 T 帧）-----
            "obj_points": obj_points.astype(np.float32),
            "obj_normals": obj_normals.astype(np.float32),
            "obj_point_id": obj_point_id,
            "obj_root_pose": obj_root_pose.astype(np.float32),
            "obj_flow": obj_flow.astype(np.float32),
            "obj_flow_valid": obj_flow_valid,

            # ----- 右手（始终 (T, ...) 形状，不删除任何帧）-----
            "right_hand_points": right_face_pts.astype(np.float32),
            "right_hand_normals": right_normals.astype(np.float32),
            "right_hand_point_id": self.right_hand_point_id,
            "right_hand_root_pose": right_root_pose.astype(np.float32),
            "right_hand_joint_axis_angle": right_joint_aa.astype(np.float32),
            "right_hand_joint_axis_angle_delta": right_joint_aa_delta.astype(np.float32),
            "right_hand_joint_quat": right_joint_quat,
            "right_hand_betas": right_betas,
            "right_hand_verts": right_verts.astype(np.float32),
            "right_hand_finger_id": self.right_finger_id,
            "right_hand_region_id": self.right_region_id,
            "right_hand_flow": right_hand_flow.astype(np.float32),
            "right_hand_flow_valid": hand_flow_valid_r,
            "right_hand_to_obj_nn_id": right_hand_to_obj_nn_id,
            "right_hand_to_obj_dist": right_hand_to_obj_dist.astype(np.float32),
            "right_hand_min_dist_to_obj": right_min_dist,    # (T,) 该手每帧到物最近距离
            "right_hand_valid": right_valid,                  # (T,) bool 3cm 接触掩码
            "right_hand_valid_2cm": right_valid_2cm,         # (T,) bool 2cm 接触掩码（CPF 对齐）
            "right_hand_valid_1cm": right_valid_1cm,         # (T,) bool 1cm 接触掩码（CPF 硬截断对齐）
            "obj_to_right_hand_nn_id": obj_to_right_hand_nn_id,
            "obj_to_right_hand_dist": obj_to_right_hand_dist.astype(np.float32),

            # ----- 左手（始终 (T, ...) 形状）-----
            "left_hand_points": left_face_pts.astype(np.float32),
            "left_hand_normals": left_normals.astype(np.float32),
            "left_hand_point_id": self.left_hand_point_id,
            "left_hand_root_pose": left_root_pose.astype(np.float32),
            "left_hand_joint_axis_angle": left_joint_aa.astype(np.float32),
            "left_hand_joint_axis_angle_delta": left_joint_aa_delta.astype(np.float32),
            "left_hand_joint_quat": left_joint_quat,
            "left_hand_betas": left_betas,
            "left_hand_verts": left_verts.astype(np.float32),
            "left_hand_finger_id": self.left_finger_id,
            "left_hand_region_id": self.left_region_id,
            "left_hand_flow": left_hand_flow.astype(np.float32),
            "left_hand_flow_valid": hand_flow_valid_l,
            "left_hand_to_obj_nn_id": left_hand_to_obj_nn_id,
            "left_hand_to_obj_dist": left_hand_to_obj_dist.astype(np.float32),
            "left_hand_min_dist_to_obj": left_min_dist,
            "left_hand_valid": left_valid,                    # (T,) bool 3cm 接触掩码
            "left_hand_valid_2cm": left_valid_2cm,           # (T,) bool 2cm 接触掩码（CPF 对齐）
            "left_hand_valid_1cm": left_valid_1cm,           # (T,) bool 1cm 接触掩码（CPF 硬截断对齐）
            "obj_to_left_hand_nn_id": obj_to_left_hand_nn_id,
            "obj_to_left_hand_dist": obj_to_left_hand_dist.astype(np.float32),
        }

        return output

    def save_sequence(self, output: dict, mano_p: str):
        """
        把处理好的 dict 存为 .npz 压缩文件。

        文件路径: output_root / s01 / box_grab_01.npz
        """
        seq_rel = mano_p.replace(RAW_SEQS_DIR + "/", "")
        seq_rel = seq_rel.replace(".mano.npy", "")
        out_path = op.join(self.output_root, seq_rel + ".npz")
        os.makedirs(op.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **output)
        return out_path

    def save_meta(self, stats: dict, extra_info: dict = None):
        """
        把本次数据集生成的**超参配置 + 统计信息**写到 output_root/meta.json。

        写入内容:
            - 数据集生成的所有超参（contact_distance_thresh, num_obj_points, dt ...）
            - MANO 模型的初始化选项
            - 输入/输出路径
            - 处理时间戳
            - 处理结果统计（成功 / 失败 / 跳过 / 处理的序列数）

        Args:
            stats:      run() 里累计的成功/失败/跳过计数及序列列表
            extra_info: 可选补充信息（如每个序列的接触统计）
        """
        meta = {
            # ---- 路径 ----
            "dataset_name": "arctic",
            "data_root": DATA_ROOT,
            "raw_seqs_dir": RAW_SEQS_DIR,
            "object_vtemplate_dir": OBJECT_VTEMPLATE_DIR,
            "object_asset_root": OBJECT_ASSET_ROOT,
            "mano_model_dir": MANO_MODEL_DIR,
            "shared_asset_root": SHARED_ASSET_ROOT,
            "shared_mano_asset_root": SHARED_MANO_ASSET_ROOT,
            "output_root": self.output_root,
            "subjects": SUBJECTS,
            "articulated_objects": sorted(list(ARTICULATED_OBJECTS)),

            # ---- 采样与时间超参 ----
            "num_obj_points": self.num_obj_points,
            "num_hand_points": self.num_hand_points,    # 1538
            "dt": self.dt,

            # ---- 手-物接触过滤超参 ----
            "contact_distance_thresh": self.contact_distance_thresh,
            "contact_distance_thresh_cm": self.contact_distance_thresh * 100.0,
            "fitting_err_thresh_mm": FITTING_ERR_THRESH,

            # ---- MANO 初始化超参 ----
            "mano_config": {
                "is_rhand": True,
                "use_pca": False,
                "flat_hand_mean": False,
                "num_shape_coeffs": 10,
                "num_faces": MANO_NUM_FACES,
                "num_joints": MANO_NUM_JOINTS,
            },

            # ---- 设备 ----
            "device": self.device,

            # ---- 处理统计 ----
            "num_sequences_processed": stats.get("success", 0),
            "num_sequences_failed": stats.get("fail", 0),
            "num_sequences_skipped": stats.get("skipped", 0),
            "processed_sequences": stats.get("sequences", []),
        }
        if extra_info:
            meta["extras"] = extra_info

        meta_path = op.join(self.output_root, "meta.json")
        os.makedirs(self.output_root, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta_path

    def run(self, seq_filter: str = None):
        """
        处理一批序列。

        seq_filter 支持的形式:
            - None:           处理所有序列
            - "s01/box_grab_01": 精确匹配单条序列
            - "s01" 或 "s01/": 处理某个 subject 的全部序列

        已处理过的序列会自动跳过（看 .npz 是否存在）。
        处理完成后会在 output_root/meta.json 写入超参与统计。
        """
        # 找到所有 .mano.npy
        all_mano_ps = sorted(glob(op.join(RAW_SEQS_DIR, "*/*.mano.npy")))

        if seq_filter is None:
            mano_ps = all_mano_ps
        elif "/" in seq_filter:
            # 精确单序列
            target = seq_filter.rstrip("/")
            mano_ps = [p for p in all_mano_ps
                       if p.replace(RAW_SEQS_DIR + "/", "").replace(".mano.npy", "") == target]
        else:
            # subject 前缀
            subject = seq_filter.rstrip("/")
            mano_ps = [p for p in all_mano_ps
                       if p.replace(RAW_SEQS_DIR + "/", "").startswith(subject + "/")]

        print(f"[Preprocessor] Found {len(mano_ps)} sequences to process")

        stats = {
            "success": 0,
            "fail": 0,
            "skipped": 0,
            "sequences": [],       # 处理成功的 seq_id 列表
        }
        # 收集每序列的接触统计，便于写到 meta.json
        contact_stats = {}

        for mano_p in tqdm(mano_ps, desc="Processing sequences"):
            seq_rel = mano_p.replace(RAW_SEQS_DIR + "/", "")
            seq_rel = seq_rel.replace(".mano.npy", "")

            # 增量处理：跳过已存在的
            out_path = op.join(self.output_root, seq_rel + ".npz")
            if op.exists(out_path):
                stats["skipped"] += 1
                continue

            try:
                output = self.process_sequence(mano_p)
                self.save_sequence(output, mano_p)
                stats["success"] += 1
                stats["sequences"].append(seq_rel)

                # 记录每序列的接触统计（valid 比例：3cm / 2cm / 1cm 三阈值）
                T_obj = int(output["frame_id"].shape[0])
                n_r_valid = int(output["right_hand_valid"].sum())
                n_l_valid = int(output["left_hand_valid"].sum())
                n_r_valid_2cm = int(output["right_hand_valid_2cm"].sum())
                n_l_valid_2cm = int(output["left_hand_valid_2cm"].sum())
                n_r_valid_1cm = int(output["right_hand_valid_1cm"].sum())
                n_l_valid_1cm = int(output["left_hand_valid_1cm"].sum())
                contact_stats[seq_rel] = {
                    "num_object_frames": T_obj,
                    # 3cm 阈值
                    "num_right_valid_frames": n_r_valid,
                    "num_left_valid_frames": n_l_valid,
                    "right_valid_ratio": n_r_valid / T_obj if T_obj else 0.0,
                    "left_valid_ratio": n_l_valid / T_obj if T_obj else 0.0,
                    # 2cm 阈值（CPF 对齐口径）
                    "num_right_valid_frames_2cm": n_r_valid_2cm,
                    "num_left_valid_frames_2cm": n_l_valid_2cm,
                    "right_valid_ratio_2cm": n_r_valid_2cm / T_obj if T_obj else 0.0,
                    "left_valid_ratio_2cm": n_l_valid_2cm / T_obj if T_obj else 0.0,
                    # 1cm 阈值（CPF 硬截断对齐口径）
                    "num_right_valid_frames_1cm": n_r_valid_1cm,
                    "num_left_valid_frames_1cm": n_l_valid_1cm,
                    "right_valid_ratio_1cm": n_r_valid_1cm / T_obj if T_obj else 0.0,
                    "left_valid_ratio_1cm": n_l_valid_1cm / T_obj if T_obj else 0.0,
                }
                tqdm.write(
                    f"  OK: {seq_rel} (obj={T_obj} frames, "
                    f"R_valid={n_r_valid} ({contact_stats[seq_rel]['right_valid_ratio']:.1%}), "
                    f"L_valid={n_l_valid} ({contact_stats[seq_rel]['left_valid_ratio']:.1%}), "
                    f"R_valid_2cm={n_r_valid_2cm} ({contact_stats[seq_rel]['right_valid_ratio_2cm']:.1%}), "
                    f"L_valid_2cm={n_l_valid_2cm} ({contact_stats[seq_rel]['left_valid_ratio_2cm']:.1%}), "
                    f"R_valid_1cm={n_r_valid_1cm} ({contact_stats[seq_rel]['right_valid_ratio_1cm']:.1%}), "
                    f"L_valid_1cm={n_l_valid_1cm} ({contact_stats[seq_rel]['left_valid_ratio_1cm']:.1%}))"
                )
            except Exception as e:
                # 单条失败不影响其他序列
                stats["fail"] += 1
                tqdm.write(f"  FAIL: {seq_rel} - {e}")
                traceback.print_exc()

        print(f"\n[Preprocessor] Done. "
              f"Success: {stats['success']}, "
              f"Fail: {stats['fail']}, "
              f"Skipped: {stats['skipped']}")

        # 写入 meta.json
        meta_path = self.save_meta(
            stats, extra_info={"contact_stats": contact_stats}
        )
        print(f"[Preprocessor] Meta written to: {meta_path}")


# ============================================================
# 入口
# ============================================================

def main():
    """
    CLI 入口。

    用法:
        python arctic_preprocess.py                              # 处理所有
        python arctic_preprocess.py --seq s05/box_grab_01        # 单序列
        python arctic_preprocess.py --subject s01                # 单 subject 全部
        python arctic_preprocess.py --num_obj_points 4096        # 改采样数
        python arctic_preprocess.py --output_root /tmp/x         # 改输出目录
    """
    parser = argparse.ArgumentParser(description="ARCTIC -> Ref2Dex Preprocessor")
    parser.add_argument("--seq", type=str, default=None,
                        help="处理单条序列, e.g. s05/box_grab_01")
    parser.add_argument("--subject", type=str, default=None,
                        help="处理某个 subject 的全部序列, e.g. s01")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help=f"输出目录 (默认: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--num_obj_points", type=int, default=2048,
                        help="每个物体的表面采样点数 (默认: 2048)")
    parser.add_argument("--dt", type=float, default=1.0/30.0,
                        help="帧间隔秒数 (默认: 1/30 = 30 FPS)")
    parser.add_argument("--contact_distance_thresh", type=float, default=CONTACT_DISTANCE_THRESH,
                        help=f"手-物接触距离阈值 (米, 默认: {CONTACT_DISTANCE_THRESH} = 3cm). "
                             "超过该距离的帧会屏蔽该手的所有数据。")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="单条序列最多处理前 N 帧 (0=不限)。用于 quick smoke test。")
    parser.add_argument("--obj-unit", type=str, default="mm",
                        choices=["m", "mm", "auto"],
                        help="物体 mesh.obj 顶点坐标单位。"
                             "默认 'mm'（ARCTIC 原始 mesh.obj 数值是 mm，如 box max=244.91mm），"
                             "内部会 /1000 转成 m，pipeline 统一输出 m。"
                             "如物体原始就是 m，传 'm'。"
                             "'auto' = 旧启发式（不推荐）。")
    args = parser.parse_args()

    # 构造预处理器
    preprocessor = ArcticPreprocessor(
        output_root=args.output_root,
        num_obj_points=args.num_obj_points,
        dt=args.dt,
        contact_distance_thresh=args.contact_distance_thresh,
        max_frames=(args.max_frames if args.max_frames > 0 else None),
        obj_unit=args.obj_unit,
    )
    # 选择过滤模式
    if args.seq is not None:
        seq_filter = args.seq
    elif args.subject is not None:
        seq_filter = args.subject
    else:
        seq_filter = None
    preprocessor.run(seq_filter=seq_filter)


if __name__ == "__main__":
    main()
