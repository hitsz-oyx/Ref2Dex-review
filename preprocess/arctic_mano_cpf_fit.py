#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 ARCTIC 数据集中的 MANO 手部轨迹"retarget"到自身的 CPF 优化脚本（processed-only）。

================================================================================
脚本作用
================================================================================
本脚本替代原始的 ``arctic_mano_collision_opt.py``：把里面基于 Isaac Gym 的
"FK 链 + 碰撞 SDF" 优化流程，换成 CPF 仓库 (https://github.com/lixiny/
CPF) 自带的 ``GeOptimizer`` (Grasp Energy Optimizer)。

为什么要换：
    原脚本依赖 Isaac Gym 的仿真 + 碰撞 SDF 来对每帧 MANO pose 做碰撞修正。
    Isaac Gym 要求每个 β (MANO 形状参数) 必须有一个对应的 URDF 才能仿真——
    而 β 是 10 维连续参数，遍历代价极高。
    CPF 的方案则直接用可微的 ``manotorch.ManoLayer``，把 β 当作 forward 输入
    一次性前向 → 出 778 顶点 + 21 关节，完全不需要 URDF。

================================================================================
数据源：只吃 processed NPZ
================================================================================
输入：
    - preprocess/arctic_preprocess.py 生成的 NPZ 文件
      （含 hand_verts / hand_joint_quat / hand_betas / obj_points 等字段，
       以及架构文档 §8 约定的 {side}_hand_valid 接触掩码）
    - 每帧从 NPZ 直接读 world 坐标，无需 mm→m、obj_root_pose 变换
    - contact 距离阈值由 NPZ 里的 hand_valid 提供；缺字段时 fallback
      到 hand_to_obj_dist.min(axis=1) <= DEFAULT_CONTACT_THRESHOLD 现算

为什么不再支持 raw 模式（.mano.npy + .object.npy + .obj 网格）：
    - raw 路径要再跑一遍 smplx.MANO forward + mm→m 变换 + 物体 canonical→world
      矩阵乘法，逻辑几乎完全复刻 arctic_preprocess.py，重复且易错
    - 数据集里 processed NPZ 是唯一权威的"消费数据"形态；raw 只服务于
      preprocess 一道工序，retarget 没必要再绕回去

================================================================================
输入输出
================================================================================
输入：
    - processed NPZ（``s01/box_grab_01.npz`` 等），由 ``load_sequence`` 加载。
    - 同一 NPZ 里已经包含 hand / obj / NN / flow / valid 全部字段。

输出：
    - pickle 文件，落盘到 ``<Ref2Dex>/outputs/mano_fit/arctic_cpf/<subject>/<seq>_<side>.pkl``
    - 内部字段 (opt_hand_pose_quat / opt_hand_tsl / opt_hand_verts /
      opt_hand_joints_user / opt_hand_joints_std / penetration_depth /
      contact_ratio / final_loss) 与原 ``arctic_mano_collision_opt.py`` 输出
      完全兼容，下游消费者不用改。
    - 同时把架构文档 §8 的 hand_valid 掩码原样写入 payload。

================================================================================
每帧流程
================================================================================
    1. 从 NPZ 读手部参数 → ManoLayer forward（如果需要做 anchor 接触分析），
       否则直接用 NPZ 里的 778 顶点 + 21 关节。
    2. 读 {side}_hand_valid[t]：
         - True  → 走 CPF 优化
         - False → opt_* 直接 = ARCTIC GT，跳过整个优化循环
    3. 走 CPF 时：现算 per-vertex contact 信息
         - 每个 obj 顶点，找最近 32 个 palm 顶点
         - 距离 < 20mm 才算接触候选
         - 候选 palm 顶点的 region 按距离倒数加权做 majority voting
         - 取该 region 的 32 个 anchor，对每个算 distance → elasti
       输出的 5 个张量喂给 GeOptimizer
    4. GeOptimizer 跑 n_iter 步 (默认 100) Adam 优化：
         - 接触损失 (anchor elasti 拉拽)
         - 排斥损失 (手-物 < 5mm 推开)
         - 解剖学正则
         - 关节目标损失 (我们加的：把 21 关节 L2 拉向 GT)
       β 固定 (optimize_hand_shape=False)，只优化 tsl + 16 关节 quat
    5. recover_hand() 拿到最终的 verts + joints；落盘

================================================================================
关键依赖
================================================================================
    - PyTorch + manotorch (可微 MANO 前向)
    - CPF/lib/postprocess/geo_optim.py 的 GeOptimizer
    - numpy / scipy (距离矩阵、投票)
    - arctic_preprocess.py 生成的 NPZ

================================================================================
使用
================================================================================
    conda activate graspenv
    python preprocess/arctic_mano_cpf_fit.py --seq-id s01/box_grab_01 \\
        --frame-step 4 --n-iter 100 --device cuda --progress
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# numpy legacy shim (chumpy / older smplx still expect np.bool etc.)
# ---------------------------------------------------------------------------
# ===========================================================================
# numpy 旧版类型兼容垫片
# ===========================================================================
# 背景：chumpy / 老版 smplx 等会 import numpy.bool / numpy.int 等已被
# numpy 1.20+ 移除的别名。一旦用 NumPy ≥ 1.20 而这些老库还没更新，import 时
# 会报 "module 'numpy' has no attribute 'bool'"。
# 解决办法：在脚本最早期把这些别名以 builtins 类型 (Python 的 bool/int/float)
# 重新注入到 np.__dict__，让老库拿到"看起来对、行为也对"的对象。
#
# 注意：
#   - 仅当属性不存在时才注入，避免覆盖 numpy 1.20+ 的新增同名属性。
#   - 用 Python 原生类型 (bool/int) 而非 np.bool_，因为 chumpy 实际上做的是
#     Python 层面的 isinstance 检查，原生类型兼容性最好。
for _legacy_name, _legacy_value in {
    "bool": bool,        # numpy.bool → Python bool
    "int": int,          # numpy.int → Python int
    "float": float,      # numpy.float → Python float
    "complex": complex,  # numpy.complex → Python complex
    "object": object,    # numpy.object → Python object
    "unicode": str,      # numpy.unicode → Python str (Python 2 残留)
    "str": str,          # numpy.str → Python str
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)


# ===========================================================================
# 唯一仍需 stub 的 CPF 子模块：thirdparty.libmesh.inside_mesh
# ===========================================================================
# 背景：CPF 核心入口 ``lib.postprocess.geo_optim.GeOptimizer`` 的 import 链会
# 间接拉入 ``lib.utils.collision``，后者第 7 行写死
#   from thirdparty.libmesh.inside_mesh import check_mesh_contains
# 该模块依赖一个未编译的 Cython 扩展 (triangle_hash.pyx)，在没有编译过
# libmesh 的环境里 import 会直接报 ModuleNotFoundError。
#
# 我们的 retarget 流程用的是 anchor elasti 距离场，**不会**调用
# solid_intersection_volume()（libmesh 唯一会触发的函数），所以 stub 一个
# 返回 "no points inside" 的安全占位即可：万一被调到也只是返回 0 穿深，
# 不会污染优化结果。
# ---------------------------------------------------------------------------
def _stub_libmesh() -> None:
    """Stub the uncompiled Cython extension: thirdparty.libmesh.inside_mesh."""
    import types
    if "thirdparty.libmesh.inside_mesh" in sys.modules:
        return  # 已被 stub 过或被真模块 import 过，不要覆盖

    def _safe_check_mesh_contains(mesh, points, hash_resolution=512):
        # Retarget 路径不会触达这里；返回 "无点在物体内" 作为最安全的占位。
        return np.zeros(len(points), dtype=bool)

    mod = types.ModuleType("thirdparty.libmesh.inside_mesh")
    mod.check_mesh_contains = _safe_check_mesh_contains
    mod.MeshIntersector = lambda *a, **kw: None
    mod.TriangleIntersector2d = lambda *a, **kw: None
    sys.modules["thirdparty.libmesh.inside_mesh"] = mod

    # 父包占位：'from thirdparty.libmesh.inside_mesh import X' 必须先解析
    # 'thirdparty.libmesh' 这个 package 对象。
    if "thirdparty.libmesh" not in sys.modules:
        pkg = types.ModuleType("thirdparty.libmesh")
        pkg.__path__ = []  # 标记为 package
        sys.modules["thirdparty.libmesh"] = pkg
    sys.modules["thirdparty.libmesh"].inside_mesh = mod


_stub_libmesh()


# ===========================================================================
# 路径与 sys.path 设置：定位 CPF 仓库 / ARCTIC 数据集 / Ref2Dex 工具
# ===========================================================================
# 整体策略：脚本启动时不预设任何硬编码路径，而是按以下优先级查找：
#   1. 环境变量 (CPF_ROOT / ARCTIC_DATA_ROOT)
#   2. 与 Ref2Dex 同级的兄弟目录 CPF/
#   3. 用户的 $HOME/CPF
#   4. ARCTIC 默认路径 Ref2Dex/dataset/arctic/data
# 这种设计让脚本在容器、ssh 远程、本地等环境下都能直接运行。
def _first_existing_path(*candidates):
    """从若干候选路径中返回第一个实际存在的。都不存在则返回 None。"""
    for c in candidates:
        if c is None:
            continue
        p = Path(c) if not isinstance(c, Path) else c
        if p.exists():
            return p
    return None


# Ref2Dex 仓库根目录：脚本在 preprocess/ 下，所以 parents[1] 是 Ref2Dex/
REF2DEX_ROOT = Path(__file__).resolve().parents[1]

# 定位 CPF 仓库根
CPF_ROOT = _first_existing_path(
    os.environ.get("CPF_ROOT"),
    REF2DEX_ROOT.parent / "CPF",  # 兄弟目录：~/test_ws/CPF
    Path.home() / "CPF",
)
if CPF_ROOT is None:
    raise RuntimeError("CPF repo not found; set CPF_ROOT or place CPF next to Ref2Dex.")

# CPF 资源目录：MANO pkl 与 anchor 权重都从这取
# 注意：CPF/assets/mano_v1_2 是符号链接，指向 Ref2Dex 的 MANO 资源（见 README）
CPF_ASSETS_DIR = CPF_ROOT / "assets"
MANO_ASSETS_ROOT = CPF_ASSETS_DIR / "mano"   # MANO pkl（被 ManoLayer 加载）
ANCHOR_ROOT = CPF_ASSETS_DIR / "anchor"      # anchor / region / merged_vertex_assignment

# ARCTIC 原始数据集根目录
ARCTIC_DATA_ROOT = _first_existing_path(
    os.environ.get("ARCTIC_DATA_ROOT"),
    REF2DEX_ROOT / "dataset" / "arctic" / "data",
    REF2DEX_ROOT / "data" / "ARCTIC",
) or (REF2DEX_ROOT / "dataset" / "arctic" / "data")

# 第三方 / 内部工具
ACO_COMPAT_ROOT = REF2DEX_ROOT / "third_party" / "aco_compat"  # 旋转转换 / MANO 增强
PREPROCESS_ROOT = REF2DEX_ROOT / "preprocess"                   # 旧脚本工具

# 把上述路径都注入到 sys.path，方便后面 ``import lib.xxx`` / ``from dataset.xxx`` 解析
for _p in (str(REF2DEX_ROOT), str(CPF_ROOT), str(CPF_ROOT / "lib"),
          str(ACO_COMPAT_ROOT), str(PREPROCESS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # 插到 path[0]，优先级最高


# ===========================================================================
# 真正 import——顺序敏感：先做 stub，再 import CPF 库
# ===========================================================================
import torch

from scipy.spatial.distance import cdist  # 距离矩阵
from scipy.spatial.transform import Rotation

# manotorch：可微 MANO 前向 + 8-region 锚点
from manotorch.anchorlayer import AnchorLayer
from manotorch.manolayer import ManoLayer
from manotorch.utils.anchorutils import (
    anchor_load,             # 加载 anchor / palm 顶点 / region 划分
    get_rev_anchor_mapping,  # region → anchor_id 列表
    recover_anchor,          # 给定手顶点，恢复 32 个 anchor 的世界坐标
    region_select_and_mask,  # 给定 region id 和 0~31 的 anchor id，生成 mask
)

# CPF 库：核心优化器和辅助
from lib.postprocess.geo_optim import GeOptimizer, _normalize_quaternion   # 主优化器
from lib.utils.contact import dumped_process_contact_info  # 接触信息 dict → tensor 的转换
from lib.utils.collision import penetration_loss_hand_in_obj  # 评估用：手-物穿透深度


# ===========================================================================
# 关节名 / 顶点索引常量：与 arctic_mano_collision_opt.py 完全一致
# ===========================================================================
# MANO 21 关节的语义名称：wrist + 5 手指 × 4 关节 = 21
MANO_TARGET_NAMES = [
    "wrist",
    "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
]

# MANO 关节重排表：SNAP order (manotorch 用的) → user order (ARCTIC 用的)
# - SNAP 顺序：[wrist, index, middle, ring, pinky, thumb] 按 SNAP 拓扑排序
# - User 顺序：[wrist, thumb(4), index(4), middle(4), ring(4), pinky(4)]
# 该表把 SNAP-index 21 关节变成 user-index 21 关节
MANO_JOINT_REORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
# 注意：数组里 ≥ 16 的下标指的是"在 SNAP 21 关节中的 5 指尖顶点序号"（虚拟指端）
# manotorch 的 21 关节里 index 16~20 其实是 5 个"指尖顶点"，不是骨骼关节

# 5 指尖对应的 MANO 顶点 id（右手 778 顶点中的索引）
# - thumb_tip  = 745
# - index_tip  = 317
# - middle_tip = 444
# - ring_tip   = 556
# - pinky_tip  = 673
# 左手小指稍有不同（445 vs 444），其他共用
MANO_TIP_VERTEX_INDICES = {
    "right": [745, 317, 444, 556, 673],
    "left":  [745, 317, 445, 556, 673],
}

# MANO_JOINT_REORDER 的逆（限制在前 16 个真正的骨骼关节）
# 输入：SNAP 顺序的 16 关节（manotorch ManoLayer.joints 的前 16 行）
# 输出：标准 MANO 16 关节顺序（用于与原 arctic_mano_collision_opt.py 的输出兼容）
# 详细推算：MANO 标准 16 关节顺序是 [wrist, thumb(3), index(3), middle(3), ring(3), pinky(3)]
# 在 SNAP 里 [wrist, thumb(3), index(3), middle(3), ring(3), pinky(3)] 的下标是
# [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
SNAP_TO_STD_16 = [0, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 1, 2, 3]

# MANO 标准 16 关节 kinematic tree（与 smplx / ARCTIC preprocess 一致）：
# [wrist,
#  thumb1, thumb2, thumb3,
#  index1, index2, index3,
#  middle1, middle2, middle3,
#  ring1, ring2, ring3,
#  pinky1, pinky2, pinky3]
MANO_PARENTS_STD_16 = np.asarray(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14],
    dtype=np.int64,
)


# ===========================================================================
# ARCTIC 序列数据结构 + 帧选择工具
# ===========================================================================
@dataclass
class ArcticSequence:
    """单个 ARCTIC 序列的所有数据。

    字段含义：
        seq_id:           完整 ID，如 "s01/box_grab_01"
        subject_id:       受试者 ID，如 "s01"
        seq_name:         动作名，如 "box_grab_01"
        object_name:      物体名（用于查找物体网格）
        side:             "right" 或 "left"
        frame_ids:        选中的帧 id 数组 (numpy int64)
        wrist_pos:        wrist 在 world 坐标 (T, 3)
        wrist_rot_aa:     wrist 旋转（axis-angle）(T, 3)
        mano_joints_world: 21 关节 (T, 21, 3)，user order
        mano_vertices_world: 778 顶点 (T, 778, 3)
        obj_trajectory:   物体 4x4 变换 (T, 4, 4)
        mano_betas:       形状参数 (T, 10)
        mano_pose_quat_world: 16 关节世界旋转 quat (T, 16, 4) wxyz
        mano_pose_quat_local: 16 关节局部旋转 quat (T, 16, 4) wxyz，manotorch 输入
        mano_translation: wrist 平移 (T, 3)
        mano_path:        MANO 数据的 .pkl 路径
        object_path:      ARCTIC 物体轨迹的 .npy 路径
        object_mesh_path: 物体 canonical 网格路径（用于碰撞）
    """
    seq_id: str
    subject_id: str
    seq_name: str
    object_name: str
    side: str
    frame_ids: np.ndarray
    wrist_pos: torch.Tensor
    wrist_rot_aa: torch.Tensor
    mano_joints_world: torch.Tensor
    mano_vertices_world: torch.Tensor
    obj_trajectory: torch.Tensor
    mano_betas: torch.Tensor
    mano_pose_quat_world: torch.Tensor
    mano_pose_quat_local: torch.Tensor
    mano_translation: torch.Tensor
    mano_path: Path
    object_path: Path
    object_mesh_path: Path


def _slice_frame_ids(num_frames: int, start_frame: int, end_frame: Optional[int],
                     frame_step: int, max_frames: int) -> np.ndarray:
    """根据 CLI 参数 (--start-frame / --end-frame / --frame-step / --max-frames)
    从原始 [0, num_frames) 中选出一组帧 id。

    规则：
        - start_frame 必须 >= 0
        - end_frame 为 None 或 -1 视为 num_frames
        - frame_step: 步长
        - max_frames: 上限帧数 (0 表示无限制)
    """
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    stop = num_frames if end_frame is None or end_frame < 0 else min(end_frame, num_frames)
    if stop <= start_frame:
        raise ValueError("Requested frame range is empty")
    frame_ids = np.arange(start_frame, stop, frame_step, dtype=np.int64)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]  # 截断到 max_frames
    if frame_ids.size == 0:
        raise ValueError("No frames selected after applying frame filters")
    return frame_ids


def _load_mano_v1_assets(mano_root: Path):
    """从 MANO v1.0 pkl（1538 face）加载 J_regressor 和 faces。

    返回：
        J_regressor: (16, 778) numpy float32 — 16 关节的线性回归器
        faces:       (1538, 3) numpy int64   — 三角面索引
    """
    import pickle
    import scipy.sparse as sp
    pkl_path = str(mano_root / "MANO_RIGHT.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    J = data["J_regressor"]
    if sp.issparse(J):
        J = J.toarray()
    f = data["f"]
    if sp.issparse(f):
        f = f.toarray().astype(np.int64)
    else:
        f = np.asarray(f, dtype=np.int64)
    return np.asarray(J, dtype=np.float32), f


def _verts_to_21_joints(verts: np.ndarray, J_regressor: np.ndarray,
                         tip_indices, joint_reorder) -> np.ndarray:
    """从 778 顶点直接算 21 关节（无需 MANO forward）。

    原理：MANO 的 16 标准关节 = J_regressor (16, 778) @ verts (778, 3)，
          5 指尖 = verts[tip_idx]，再拼起来重排到 user order。

    Args:
        verts:        (..., 778, 3) numpy 顶点
        J_regressor:  (16, 778) numpy
        tip_indices:  list of 5 int, 指尖 vertex id
        joint_reorder: list of 21 int, 重排映射
    Returns:
        (..., 21, 3) numpy
    """
    joints_16 = np.einsum("...vj,rv->...rj", verts, J_regressor)  # (..., 16, 3)
    tips = verts[..., tip_indices, :]                              # (..., 5, 3)
    joints_21 = np.concatenate([joints_16, tips], axis=-2)         # (..., 21, 3)
    return joints_21[..., joint_reorder, :]                       # (..., 21, 3)


def _quat_world_to_local(quat_world_wxyz: np.ndarray,
                         parents: np.ndarray) -> np.ndarray:
    """把 16 关节 world quat 转成 MANO local quat。

    NPZ 的 `*_hand_joint_quat` 来自 `arctic_preprocess.py` 的
    `compute_hand_joint_quaternions()`，语义是沿运动学树累乘后的世界旋转；
    `manotorch.ManoLayer(rot_mode="quat")` 则要求输入根节点全局旋转 + 15 个局部旋转。
    """
    quat_world_wxyz = np.asarray(quat_world_wxyz, dtype=np.float64)
    if quat_world_wxyz.ndim != 3 or quat_world_wxyz.shape[1:] != (16, 4):
        raise ValueError(
            f"Expected world quats with shape (T, 16, 4), got {quat_world_wxyz.shape}"
        )

    quat_world_xyzw = np.concatenate(
        [quat_world_wxyz[..., 1:], quat_world_wxyz[..., :1]], axis=-1
    )
    rot_world = Rotation.from_quat(quat_world_xyzw.reshape(-1, 4)).as_matrix()
    rot_world = rot_world.reshape(-1, 16, 3, 3)

    rot_local = np.zeros_like(rot_world)
    rot_local[:, 0] = rot_world[:, 0]
    for joint_idx in range(1, 16):
        parent_idx = int(parents[joint_idx])
        rot_local[:, joint_idx] = np.matmul(
            np.swapaxes(rot_world[:, parent_idx], -1, -2),
            rot_world[:, joint_idx],
        )

    quat_local_xyzw = Rotation.from_matrix(rot_local.reshape(-1, 3, 3)).as_quat()
    quat_local_xyzw = quat_local_xyzw.reshape(-1, 16, 4)
    quat_local_wxyz = np.concatenate(
        [quat_local_xyzw[..., 3:4], quat_local_xyzw[..., :3]], axis=-1
    )
    quat_local_wxyz = np.where(
        quat_local_wxyz[..., :1] < 0.0, -quat_local_wxyz, quat_local_wxyz
    )
    return quat_local_wxyz.astype(np.float32)


# ===========================================================================
# processed 模式默认接触阈值（米）
# ===========================================================================
# 用途：NPZ 里没有 {side}_hand_valid_2cm 字段（旧版 preprocess 输出）时，用
# right_hand_to_obj_dist.min(axis=1) <= DEFAULT_CONTACT_THRESHOLD 现算一个
# bool 掩码，避免空接触帧让 CPF 内部 loss 出 NaN。
# 改 0.02 的原因：与 CPF 内部 palm 顶点 range_threshold=20mm 对齐，
# 比默认 0.03 严，可消除 "palm 离物 2.2cm" 之类的 NaN 来源（见架构文档 §8 双阈值设计）。
DEFAULT_CONTACT_THRESHOLD = 0.010  # 10 mm，与 CPF range_threshold=1cm 对齐
# （CPF compute_contact_info 的硬截断 range_threshold 默认 20mm 在本文件中
# 已改为 10mm，见 compute_contact_info(...) 调用处。）


def _resolve_hand_valid(data: np.lib.npyio.NpzFile, side_key: str,
                         selected: np.ndarray, contact_thresh: float) -> np.ndarray:
    """从 NPZ 里读出 (T,) bool 接触掩码；找不到字段则从 hand_to_obj_dist 现算。

    优先读 ``{side}_hand_valid_1cm``（新 preprocess 产物，1cm 阈值，与 CPF
    内部 palm 顶点 range_threshold=10mm 严格对齐）；
    找不到再读 ``{side}_hand_valid_2cm``（2cm 阈值，过渡兼容）；
    再没有读 ``{side}_hand_valid``（3cm 阈值，向后兼容）。
    最后 fallback 到 ``{side}_hand_to_obj_dist[selected].min(axis=1) <= contact_thresh``。
    """
    valid_key_1cm = f"{side_key}_hand_valid_1cm"
    if valid_key_1cm in data.files:
        return np.asarray(data[valid_key_1cm][selected], dtype=bool)
    valid_key_2cm = f"{side_key}_hand_valid_2cm"
    if valid_key_2cm in data.files:
        return np.asarray(data[valid_key_2cm][selected], dtype=bool)
    valid_key_3cm = f"{side_key}_hand_valid"
    if valid_key_3cm in data.files:
        return np.asarray(data[valid_key_3cm][selected], dtype=bool)
    dist_key = f"{side_key}_hand_to_obj_dist"
    if dist_key in data.files:
        dist = data[dist_key][selected]                # (T, 1538) 米
        return dist.min(axis=1) <= contact_thresh      # (T,) bool
    raise KeyError(
        f"NPZ missing both '{valid_key_1cm}' / '{valid_key_2cm}' / '{valid_key_3cm}' and '{dist_key}'; "
        f"cannot derive contact mask."
    )


def load_sequence(
    processed_root: Path,
    seq_id: str,
    side: str,
    frame_step: int,
    start_frame: int,
    end_frame: Optional[int],
    max_frames: int,
    contact_thresh: float = DEFAULT_CONTACT_THRESHOLD,
) -> ArcticSequence:
    """从 processed_data 的 .npz 文件加载序列（无 .obj mesh，无需 MANO forward）。

    流程：
        1. 解析 seq_id → subject_id + seq_name
        2. 加载 .npz（含 hand / obj / NN / flow / valid 全部字段）
        3. 选帧：按 frame_step / start_frame / end_frame / max_frames 选子集
        4. 从 778 顶点 + J_regressor 算 21 关节（user order）
        5. 读 right/left_hand_valid 掩码（fallback 到 hand_to_obj_dist 现算）

    返回的 ArcticSequence 字段：
        - 所有 *_world 数据都是 NPZ 里的 world 坐标（物体已是 world）
        - 新增 ``hand_valid`` 字段（不是 dataclass 字段，单独挂在 attribute 上）
    """
    side_key = side.lower()
    if side_key not in {"left", "right"}:
        raise ValueError(f"Unsupported side: {side}")

    # 解析 seq_id → subject_id, seq_name（object_name 只在传入时用，不用于路径）
    subject_id, seq_name = seq_id.split("/", 1)
    object_name = seq_name.split("_")[0]

    # .npz 路径
    npz_path = processed_root / subject_id / f"{seq_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing processed NPZ: {npz_path}")

    # 加载 .npz
    data = np.load(str(npz_path), allow_pickle=True)

    verts_key = f"{side_key}_hand_verts"
    pose_quat_key = f"{side_key}_hand_joint_quat"
    betas_key = f"{side_key}_hand_betas"
    root_pose_key = f"{side_key}_hand_root_pose"

    # 帧选择：用 frame_id 数组来选
    all_frame_ids = data.get("frame_id", np.arange(data[verts_key].shape[0]))
    selected = _slice_frame_ids(
        num_frames=len(all_frame_ids),
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        max_frames=max_frames,
    )
    frame_ids = all_frame_ids[selected]

    # ---------- 加载 MANO v1.0 辅助资产 ----------
    # 从 dataset/arctic 下拿 MANO v1.0 pkl 的 J_regressor 算 21 关节
    mano_model_root = ARCTIC_DATA_ROOT / "body_models" / "mano"
    J_regressor, mano_faces = _load_mano_v1_assets(mano_model_root)  # (16, 778), (1538, 3)

    # ---------- 手部数据 ----------
    verts_np = data[verts_key][selected].astype(np.float32)        # (T, 778, 3)
    pose_quat_world_np = data[pose_quat_key][selected].astype(np.float32)  # (T, 16, 4)
    pose_quat_local_np = _quat_world_to_local(pose_quat_world_np, MANO_PARENTS_STD_16)
    betas_np = data[betas_key][selected].astype(np.float32)             # (T, 10)
    root_pose = data[root_pose_key][selected]                          # (T, 4, 4)
    tsl_np = np.asarray(root_pose[:, :3, 3], dtype=np.float32)                  # (T, 3)

    # 21 关节（world 坐标）：
    tip_ids = MANO_TIP_VERTEX_INDICES[side_key]
    joints_user_np = _verts_to_21_joints(verts_np, J_regressor, tip_ids, MANO_JOINT_REORDER)

    # 转 torch
    verts_t = torch.from_numpy(verts_np)
    joints_t = torch.from_numpy(joints_user_np)
    betas_t = torch.from_numpy(betas_np)
    pose_quat_world_t = torch.from_numpy(pose_quat_world_np)
    pose_quat_local_t = torch.from_numpy(pose_quat_local_np)
    tsl_t = torch.from_numpy(tsl_np)
    wrist_pos_t = joints_t[:, 0]  # (T, 3) wrist 位置

    # ---------- 物体数据 ----------
    # obj_points/obj_normals 已在 world 坐标，无需额外变换
    # obj_root_pose (T, 4, 4) 只留作 payload 记录
    obj_root_pose = data["obj_root_pose"][selected].astype(np.float32)       # (T, 4, 4)
    obj_trajectory = torch.from_numpy(obj_root_pose)                         # (T, 4, 4)
    # obj_points[t] / obj_normals[t] 也在 NPZ 里是 world 坐标
    obj_verts_world = data["obj_points"][selected].astype(np.float32)         # (T, 2048, 3)
    obj_normals_world = data["obj_normals"][selected].astype(np.float32)      # (T, 2048, 3)

    # ---------- 接触掩码 ----------
    # 优先读 NPZ 自带 hand_valid（架构文档 §8 约定）；没有就 fallback。
    hand_valid_np = _resolve_hand_valid(data, side_key, selected, contact_thresh)

    seq = ArcticSequence(
        seq_id=seq_id,
        subject_id=subject_id,
        seq_name=seq_name,
        object_name=object_name,
        side=side_key,
        frame_ids=frame_ids,
        wrist_pos=wrist_pos_t,
        wrist_rot_aa=torch.zeros(len(selected), 3, dtype=torch.float32),  # 不用 AA（payload 里参考用）
        mano_joints_world=joints_t,
        mano_vertices_world=verts_t,
        obj_trajectory=obj_trajectory,
        mano_betas=betas_t,
        mano_pose_quat_world=pose_quat_world_t,
        mano_pose_quat_local=pose_quat_local_t,
        mano_translation=tsl_t,
        mano_path=npz_path,
        object_path=npz_path,
        object_mesh_path=npz_path,  # processed 模式没有独立 mesh 路径
    )
    # extra: valid 掩码（架构文档 §8 的 *_hand_valid）+ 物体 world 坐标
    seq.hand_valid = hand_valid_np                  # (T,) bool
    seq.obj_verts_world = obj_verts_world            # (T, 2048, 3)
    seq.obj_normals_world = obj_normals_world        # (T, 2048, 3)
    return seq


# ===========================================================================
# 单帧接触信息计算：内联 CPF 的 prepare_contact_info.py worker
# ===========================================================================
# 背景：CPF 原始的 ``prepare_contact_info.py`` 是个多进程脚本，对 GT 数据
#       （DEX-YCB 等）做同样的事。这里我们只对当前这一帧的 obj 顶点算接触，
#       没必要起多进程——所以内联过来，单线程直接跑。
def _elasti_fn(x: np.ndarray, range_th: float = 45.0) -> np.ndarray:
    """把距离 (mm) 转成 elasti 权重 (0~1)。公式：0.5 * cos(π/range_th * d) + 0.5。

    性质：
        - d=0     → 1.0（最贴合 = 权重最大）
        - d=range → 0.0（超出范围 = 权重为 0）
        - 中间值   → 余弦衰减
    在 anchor 距离场中，elasti 直接作为接触损失的权重。
    """
    x = x.copy()
    np.putmask(x, x > range_th, range_th)        # 超过 range 的截到 range
    res = 0.5 * np.cos((np.pi / range_th) * x) + 0.5
    np.putmask(res, res < 1e-8, 0)               # 极小数清零（数值稳定）
    return res


def _mode_of_regions(assignment: np.ndarray, n_regions: int,
                     weights: Optional[np.ndarray] = None) -> int:
    """对一组 region id 投票，返回加权众数。

    实现：累加每个 region 的权重和（不是严格的 mode，而是 weighted mode），
    返回得分最高的 region。
    """
    if weights is None:
        weights = np.ones_like(assignment, dtype=np.float32)
    res = np.zeros((n_regions,), dtype=np.float32)
    for bin_id in range(n_regions):
        res[bin_id] = np.sum((assignment == bin_id).astype(np.float32) * weights)
    return int(np.argmax(res))


def compute_contact_info(
    hand_verts_world: np.ndarray,  # (778, 3)
    obj_verts_world: np.ndarray,  # (N_OBJ, 3)
    hand_palm_vid: np.ndarray,  # vertex indices eligible for contact
    merged_vertex_assignment: np.ndarray,  # (778,) int — region per vert
    n_regions: int,
    anchor_pos_world: np.ndarray,  # (32, 3) — recovered anchor positions
    anchor_mapping: dict,  # region_id -> list[anchor_id]
    range_threshold: float = 20.0,  # mm
    n_samples: int = 32,
    elasti_threshold: float = 45.0,
    elasti_cutoff: float = 0.1,
    pad_vertex: bool = True,
    pad_anchor: bool = True,
):
    """对单帧计算 per-vertex 接触信息（CPF 喂给 GeOptimizer 的输入格式）。

    输入：
        hand_verts_world:        (778, 3) 当前帧手部所有顶点
        obj_verts_world:         (N_OBJ, 3) 物体当前帧顶点
        hand_palm_vid:           (M,) palm 区域（接触候选）的 vertex id
        merged_vertex_assignment: (778,)  每个手顶点属于哪个 region (0~7)
        n_regions:               region 总数（CPF 用 8）
        anchor_pos_world:        (32, 3) 32 个 anchor 的世界坐标
        anchor_mapping:          {region_id: [anchor_id, ...]} region → anchor 列表
        range_threshold:         接触距离阈值（mm），默认 10mm（与 valid_1cm 对齐）
        n_samples:               每个 obj 顶点最多采样的近邻手顶点数
        elasti_threshold:        elasti 余弦衰减的范围（mm），默认 45mm
        elasti_cutoff:           elasti 下限（避免梯度消失）
        pad_vertex:              是否对 obj 顶点 pad
        pad_anchor:              是否对 anchor pad

    算法（对应 CPF 原始的 prepare_contact_info.py worker）：
        对每个 obj 顶点 i：
            1. 找手 vertex 中最近的 n_samples 个
            2. 距离 < range_threshold 的算接触候选
            3. 候选手 vertex 的 region 按 1/d 加权投票 → target_region
            4. 取 target_region 的所有 anchor（共 4 个）
            5. 对每个 anchor 算 distance + elasti

    返回（5 个张量，直接喂给 GeOptimizer.set_opt_val）：
        vertex_contact:     (N_OBJ,)            0/1 是否接触
        contact_region_id:  (N_OBJ,)            region id (>=0) 或 -1（未接触）
        anchor_id:          (N_OBJ, max_anchor) anchor id (>=0) 或 0（padding）
        anchor_elasti:      (N_OBJ, max_anchor) elasti 权重
        anchor_padding_mask:(N_OBJ, max_anchor) True=有效，False=padding
    """
    # region_id → 该 region 下所有 anchor id 的列表
    rev_anchor_mapping = get_rev_anchor_mapping(anchor_mapping, n_region=n_regions)
    # 只考虑 palm 顶点作为接触候选
    selected_hand_verts = hand_verts_world[hand_palm_vid]    # (M, 3)
    selected_assignment = merged_vertex_assignment[hand_palm_vid]  # (M,) region

    # 计算每个 obj 顶点到每个 palm 顶点的距离（m → mm）
    all_dist = cdist(obj_verts_world, selected_hand_verts) * 1000.0  # (N_OBJ, M) in mm
    # 按距离排序，取前 n_samples 个
    order_idx = np.argsort(all_dist, axis=1)                          # (N_OBJ, M)
    sorted_dist_sampled = np.take_along_axis(all_dist, order_idx[:, :n_samples], axis=1)  # (N_OBJ, K)

    # 逐个 obj 顶点构建 contact info dict
    vertex_blob = []
    for point_id in range(obj_verts_world.shape[0]):
        dist_vec = sorted_dist_sampled[point_id]            # (K,) 前 K 个近邻的距离
        valid_mask = dist_vec < range_threshold             # (K,) bool：是否在接触范围内

        if not valid_mask.any():
            # 没有接触候选：这个 obj 顶点不接触
            vertex_blob.append({"contact": 0})
            continue

        masked_dist = dist_vec[valid_mask]                  # 距离
        valid_local_idx = np.where(valid_mask)[0]            # 在 K 个近邻里的位置
        origin_valid_idx = order_idx[point_id, valid_local_idx]  # 映射回 M 维 palm 中的 id
        origin_regions = selected_assignment[origin_valid_idx]    # 这些 palm 顶点的 region

        # 加权众数投票：权重 = (range_threshold - 距离)——越近权重越大
        mode_weight = (range_threshold - masked_dist).astype(np.float32)
        target_region = _mode_of_regions(origin_regions, n_regions, weights=mode_weight)

        # 取该 region 的 anchor 列表
        anchor_list = rev_anchor_mapping[target_region]
        obj_pt = obj_verts_world[point_id:point_id + 1]    # (1, 3)
        anchor_pts = anchor_pos_world[anchor_list]          # (n_anchor, 3)
        dist_mat = cdist(obj_pt, anchor_pts).squeeze(0) * 1000.0  # (n_anchor,) mm
        elasti_mat = _elasti_fn(dist_mat, range_th=elasti_threshold)  # (n_anchor,) 0~1
        # 截断极小 elasti，避免反向传播时梯度消失
        np.putmask(elasti_mat, elasti_mat < elasti_cutoff, elasti_cutoff)

        vertex_blob.append({
            "contact": 1,
            "region": int(target_region),
            "anchor_id": list(anchor_list),
            "anchor_dist": dist_mat.tolist(),
            "anchor_elasti": elasti_mat.tolist(),
        })

    # 把 list[dict] 转成 GeOptimizer 想要的 5 个张量
    return dumped_process_contact_info(
        vertex_blob,
        anchor_mapping=anchor_mapping,
        pad_vertex=pad_vertex,
        pad_anchor=pad_anchor,
        elasti_th=0.0,
    )


# ===========================================================================
# ArcticCpfFitter：GeOptimizer 的薄包装，加上"关节目标 L2"额外项
# ===========================================================================
# 设计动机：
#   原始 GeOptimizer 内部只有：
#     - 接触损失（anchor elasti 拉拽）
#     - 排斥损失（手-物 < 5mm 推开）
#     - 解剖学正则（关节限位）
#   但这些都不"显式"把 21 关节位置拉回 ARCTIC GT，所以优化可能偏离 GT 太远。
#   我们额外加一个 L2 joint target loss 起到"软" retarget 的作用——
#   既保持 ARCTIC 动作的语义，又让接触/碰撞符合物理。
class ArcticCpfFitter:
    """Fit a single hand frame with CPF + an extra joint-target regulariser.

    流程：
        1. __init__ 时构造 manotorch.ManoLayer + AnchorLayer + GeOptimizer
        2. fit() 接收一帧的所有数据，调用 GeOptimizer 优化 tsl + 16 quat
        3. 训练循环里每步都额外算 jt_loss，注入到 backward
        4. 训练完调用 recover_hand() 拿最终 verts + joints
    """

    def __init__(
        self,
        device: torch.device,
        side: str,
        mano_assets_root: str = "assets/mano",
        anchor_root: str = "assets/anchor",
        lr: float = 1e-2,
        n_iter: int = 400,
        lambda_contact_loss: float = 10.0,
        lambda_repulsion_loss: float = 0.5,
        repulsion_query: float = 0.030,
        repulsion_threshold: float = 0.080,
        joint_target_weight: float = 5.0,
        tip_weight_thumb: float = 25.0,
        tip_weight_index: float = 20.0,
        tip_weight_middle: float = 10.0,
        tip_weight_ring: float = 7.0,
        tip_weight_pinky: float = 5.0,
    ) -> None:
        """构造 fitter：manotorch.ManoLayer + AnchorLayer + CPF GeOptimizer。

        关键参数：
            device:                GPU / CPU 设备
            mano_assets_root:      manotorch 用的 MANO 资源路径（指向 MANO_LEFT/RIGHT.pkl）
            anchor_root:           32 个 anchor 权重的路径
            lr / n_iter:           Adam 学习率 / 优化迭代次数
            lambda_contact_loss:   接触损失权重（CPF 内部）
            lambda_repulsion_loss: 排斥损失权重
            repulsion_query:       排斥查询半径（m），默认 3cm
            repulsion_threshold:   排斥最大半径（m），默认 8cm
            joint_target_weight:   我加的 jt_loss 权重
            tip_weight_*:          jt_loss 里对每根手指的权重（指尖更重要）
        """
        self.device = device
        self.side = side
        self.lr = lr
        self.n_iter = n_iter
        self.joint_target_weight = joint_target_weight
        # jt_loss 用的指尖权重：拇指 25（最重要）、小指 5（最不重要）
        # 总共 21 个权重：1 wrist + 5×4 关节 = 21
        self._tip_weights = torch.tensor(
            [1.0,  # wrist
             tip_weight_thumb, tip_weight_thumb, tip_weight_thumb, tip_weight_thumb,  # thumb
             tip_weight_index, tip_weight_index, tip_weight_index, tip_weight_index,  # index
             tip_weight_middle, tip_weight_middle, tip_weight_middle, tip_weight_middle,  # middle
             tip_weight_ring, tip_weight_ring, tip_weight_ring, tip_weight_ring,  # ring
             tip_weight_pinky, tip_weight_pinky, tip_weight_pinky, tip_weight_pinky],  # pinky
            device=device, dtype=torch.float32,
        )

        # manotorch.ManoLayer：可微 MANO 前向，用 quat 旋转，16+5 关节
        # 注意：manotorch 的 quat mode 会强制走 flat_hand_mean 语义
        # （库内直接 warning: "Quat mode doesn't support ... non flat_hand_mean"）。
        # ARCTIC 的原始 MANO 参数来自 flat_hand_mean=False 的 smplx 拟合，因此
        # 即便修正了左右手 / world→local quat，init 也不会与 NPZ GT 完全重合。
        self.mano_layer = ManoLayer(
            rot_mode="quat",          # 我们用 quat 而非 axis-angle
            side=side,                # 不传则 manotorch 默认 right hand
            use_pca=False,            # 不压缩 pose
            mano_assets_root=mano_assets_root,
            center_idx=9,             # 中心关节（middle MCP）的索引
            flat_hand_mean=True,      # 用 flat_hand_mean（manotorch 推荐）
        ).to(device)

        # CPF 的主优化器：内含 anchor layer / mano layer / 优化器
        self.geoptim = GeOptimizer(
            device=device,
            lr=lr,
            n_iter=n_iter,
            verbose=False,            # 不打 CPF 自己的日志
            lambda_contact_loss=lambda_contact_loss,
            lambda_repulsion_loss=lambda_repulsion_loss,
            repulsion_query=repulsion_query,
            repulsion_threshold=repulsion_threshold,
        )
        # 重要：GeOptimizer 内部默认用 "assets/anchor" 这种相对路径
        # 我们要覆盖成我们找到的绝对路径（CPF_ROOT/assets/anchor）
        self.geoptim.anchor_layer = AnchorLayer(anchor_root).to(device)
        # 共享 mano_layer 的权重（确保两边的 forward 一致）
        self.geoptim.mano_layer = self.mano_layer

    # ------------------------------------------------------------------ public
    def fit(
        self,
        hand_shape: torch.Tensor,  # (10,)
        hand_tsl_init: torch.Tensor,  # (3,)
        hand_pose_quat_init: torch.Tensor,  # (16, 4)  wxyz
        hand_joints_gt_world: torch.Tensor,  # (21, 3) user order, **wrist-centered world**
        hand_verts_gt_world: torch.Tensor,  # (778, 3) **wrist-centered world** (NPZ direct)
        obj_verts_world: torch.Tensor,  # (N_OBJ, 3)
        obj_normals_world: torch.Tensor,  # (N_OBJ, 3)
        obj_faces: torch.Tensor,  # (NF, 3) long
        vertex_contact: np.ndarray,  # (N_OBJ,) {0,1}
        contact_region_id: np.ndarray,  # (N_OBJ,) padded
        anchor_id: np.ndarray,  # (NCONT, anchor_padding_len) padded
        anchor_elasti: np.ndarray,  # (NCONT, anchor_padding_len)
        anchor_padding_mask: np.ndarray,  # (NCONT, anchor_padding_len)
        hand_tip_vid: torch.Tensor,  # (5,) tip vertex ids in user's order
        progress: bool = False,
        only_jt_loss: bool = False,  # True=只优化 jt_loss，禁用 CPF 原生损失（ablation 用）
    ) -> Dict[str, np.ndarray]:
        """对单帧跑 CPF + 关节目标 L2 优化，返回 dict 形式的 opt_* 数组。

        参数说明：
            hand_shape:            (10,) ARCTIC 形状参数 β
            hand_tsl_init:         (3,) ARCTIC wrist 平移（m）
            hand_pose_quat_init:   (16, 4) ARCTIC 16 关节 quat（wxyz）
            hand_joints_gt_world:  (21, 3) ARCTIC 21 关节（user order, wrist-centered world）
            hand_verts_gt_world:   (778, 3) ARCTIC 778 顶点（wrist-centered world）
            obj_verts_world:       (N_OBJ, 3) 物体当前帧世界坐标顶点
            obj_normals_world:     (N_OBJ, 3) 物体法线
            obj_faces:             (NF, 3) 物体三角面（long）
            vertex_contact:        (N_OBJ,) 0/1
            contact_region_id:     (N_OBJ,) padded region id
            anchor_id:             (N_OBJ, K) padded anchor id
            anchor_elasti:         (N_OBJ, K) padded elasti 权重
            anchor_padding_mask:   (N_OBJ, K) 1=有效，0=padding
            hand_tip_vid:          (5,) 5 指尖在 user order 里的 vertex id
            progress:              是否每 10% step 打一次损失

        返回 dict：
            opt_hand_pose_quat:    (16, 4) wxyz
            opt_hand_tsl:          (3,)
            opt_hand_verts:        (778, 3)  wrist-centered world
            opt_hand_joints_user:  (21, 3) user order, wrist-centered world
            opt_hand_joints_std:   (16, 3) 标准 MANO order
            penetration_depth:     标量
            contact_ratio:         标量，0~1
            loss_history:          dict[list]，每 step 的 loss
            final_loss:            标量

        ⚠️ 坐标系约定（重要！）：
            manotorch.ManoLayer 的 ``verts/joints`` 在内部减去了 ``center_joint``（即
            center_idx=9 处关节，middle MCP）的位置——这是 manotorch 的"middle-MCP-centered"
            坐标系。
            但 ARCTIC NPZ 的 ``right_hand_verts / right_hand_joints_world`` 来自 SMPL-X
            MANO forward，是"wrist-centered"坐标系（manotorch vs smplx 差 90+mm 系统差）。
            本函数:
              - jt_loss target 用 ``hand_joints_gt_world``（NPZ 直接给，wrist-centered world）
              - opt_joints = ``mano_out.joints + center_joint + vec_tsl``（手动 uncenter）
              - opt_verts  = ``recover_hand() + center_joint``（手动 uncenter）
            这样才能让 opt 跟 NPZ GT 在同一坐标系下比较。
        """
        device = self.device
        n_total_obj = vertex_contact.shape[0]
        n_regions = int(contact_region_id.max()) + 1  # 8 (CPF 默认)

        # ===========================================================================
        # 设备对齐：load_sequence() 把数据放在 CPU（NPZ 读入的固有产物），
        # 而 self.mano_layer / self.anchor_layer 都在 self.device（GPU）。
        # CPF 的 loss_fn 里会做 th_shapedirs @ betas 等 matmul，若 betas 在 CPU
        # 会直接 RuntimeError。在 fit() 入口统一搬到 self.device，从根源消除
        # 设备不一致问题，避免每个调用点单独 .to() 的特判蔓延。
        # ===========================================================================
        hand_shape = hand_shape.detach().to(device).float()
        hand_tsl_init = hand_tsl_init.detach().to(device).float()
        hand_pose_quat_init = hand_pose_quat_init.detach().to(device).float()
        # hand_joints_gt_world 已在下面的 "jt_loss target" 段落统一搬到 device

        # 把 numpy contact info 转成 torch tensor 上 GPU
        vertex_contact_t = torch.from_numpy(vertex_contact).long().to(device)
        contact_region_id_t = torch.from_numpy(contact_region_id).long().to(device)
        anchor_id_t = torch.from_numpy(anchor_id).long().to(device)
        anchor_elasti_t = torch.from_numpy(anchor_elasti).float().to(device)
        anchor_padding_mask_t = torch.from_numpy(anchor_padding_mask).long().to(device)

        # 关键：CPF 的 set_opt_val 在 mode="hand" 下，需要 obj 在 *canonical* 空间
        # 然后由 obj_tsl/obj_rot 转换。我们这里"反向"用：物体固定在 ARCTIC 给出
        # 的 world 位置，所以把 world 顶点当作 canonical，obj_tsl/obj_rot 给 0
        # ——GeOptimizer 内部 obj_tsl_var=0 加上 identity 旋转，等价于"不动物体"。
        obj_tsl_init = torch.zeros(3, device=device, dtype=torch.float32)
        obj_rot_init = torch.zeros(3, device=device, dtype=torch.float32)

        # 注入所有变量到 GeOptimizer
        # - mode="hand": 优化手，物体固定
        # - hand_shape_init: 固定 β（不优化）
        # - hand_pose_gt[0] = wrist quat（不优化，固定）
        # - hand_pose_init[1..15] = 手指 quat（可优化）
        # - obj_verts_gt / obj_normals_gt: 物体的 GT（固定）世界坐标
        self.geoptim.set_opt_val(
            mode="hand",
            vertex_contact=vertex_contact_t,
            contact_region_id=contact_region_id_t,
            anchor_id=anchor_id_t,
            anchor_elasti=anchor_elasti_t,
            anchor_padding_mask=anchor_padding_mask_t,
            obj_faces=obj_faces.long().to(device),
            # hand
            hand_shape_init=hand_shape.detach().clone(),  # 固定 β
            hand_tsl_init=hand_tsl_init.detach().clone(),  # wrist 平移（可优化）
            hand_pose_gt=([0], hand_pose_quat_init[0:1].detach().clone()),  # wrist quat 固定
            hand_pose_init=(list(range(1, 16)),
                            hand_pose_quat_init[1:].detach().clone()),  # 手指 quat 可优化
            # obj (固定)
            obj_verts_gt=obj_verts_world.to(device).float(),
            obj_normals_gt=obj_normals_world.to(device).float(),
            obj_tsl_init=obj_tsl_init,  # 0（不变）
            obj_rot_init=obj_rot_init,  # 0（不变）
        )

        # 强制 β 固定：CPF set_opt_val 默认 optimize_hand_shape=True
        # 我们通过 ctrl_val 关掉它，并把 β 当作 GT 注入 const_val
        # 这样 loss_fn 会走"非优化 hand shape"分支，不会对 β 求梯度
        self.geoptim.const_val["hand_shape_gt"] = hand_shape.detach().clone()
        self.geoptim.ctrl_val["optimize_hand_shape"] = False
        # 重建 Adam optimizer：把 β 排除在参数组外
        self._build_param_optimizer()

        # ===========================================================================
        # jt_loss target：直接用 NPZ 提供的 21 joints (wrist-centered world)
        # ===========================================================================
        # ⚠️ BUG FIX（之前注释错了）:
        # 原代码用 ``mano_out.joints + hand_tsl_init`` 作为 jt_loss target，理由是
        # "manotorch 跟 NPZ 差 34mm，用 manotorch 自洽避免初始 loss 不为零"。
        # 但实测 manotorch 跟 ARCTIC NPZ 不是差 34mm，而是差 90+mm —— 因为：
        #   - manotorch 输出 verts/joints 是 middle-MCP-centered（center_idx=9）
        #   - ARCTIC NPZ 的 verts/joints 是 wrist-centered（用 smplx MANO 生成）
        # 两者在不同坐标系下！manotorch 减完 center_joint 后 + vec_tsl 仍差 90+mm。
        # 修复：jt_loss target 直接用 NPZ 的 hand_joints_gt_world（wrist-centered world，
        # 跟 opt_joints 用同一坐标系）。
        # 副作用：jt_loss 初始值不为 0（~90mm），但 jt_loss 跟 contact_loss 协同优化后会收敛。
        hand_joints_gt_world = hand_joints_gt_world.detach().to(device).float()
        gt_joints_consistent = hand_joints_gt_world  # (21, 3) wrist-centered world
        # 顶点 GT 也直接用 NPZ 提供的（wrist-centered world）
        gt_verts_consistent = hand_verts_gt_world.detach().to(device).float()  # (778, 3)

        # ===========================================================================
        # 训练循环：每步算 CPF 原生 loss + 我们加的 jt_loss
        # ===========================================================================
        history = {"loss": [], "contact": [], "repulsion": [], "jt": []}
        last_loss = None
        # 记录 init tsl 用于打印 drift
        self._init_tsl = self.geoptim.opt_val["hand_tsl_var"].detach().clone()
        for step in range(self.n_iter):
            self.geoptim.optimizer.zero_grad()

            # CPF 原生损失（contact + repulsion + anatomical priors）
            # only_jt_loss=True 时只跑 jt_loss，不用 CPF（ablation 用）
            if only_jt_loss:
                loss_base = torch.zeros((), device=device)
                info = {}
            else:
                loss_base, info = self.geoptim.loss_fn(
                    self.geoptim.opt_val,
                    self.geoptim.const_val,
                    self.geoptim.ctrl_val,
                    self.geoptim.coef_val,
                )

            # --- jt_loss：把 21 关节位置 L2 拉向 NPZ GT (wrist-centered world) ---
            # target 是 NPZ 提供的 hand_joints_gt_world（跟 opt_joints 同坐标系）。
            # opt_joints 必须 uncenter：``mano_out.joints + center_joint + vec_tsl``
            # 因为 manotorch 输出是 middle-MCP-centered（减了 center_joint），而 NPZ
            # GT 是 wrist-centered。
            # ⚠️ bug fix: 不能用 self.geoptim.recover_hand()！
            # recover_hand() 内部对 hand_pose_var_val / hand_shape_var / hand_tsl_var
            # 全部 .detach()（geo_optim.py:665, 669, 673），返回的 joints 完全脱离
            # 计算图 → jt_loss 是常数，对 total.backward() 无贡献 → 手被强行拉到
            # 与 GT 差异 4-5cm 的"锚点接触"位置。
            # 修复：自己 forward 整条 opt_val → mano forward → joints 链。
            vec_pose = self.geoptim.assemble_pose_vec(
                self.geoptim.const_val["hand_pose_gt_idx"],
                self.geoptim.const_val["hand_pose_gt_val"],
                self.geoptim.const_val["hand_pose_var_idx"],
                self.geoptim.opt_val["hand_pose_var_val"],   # ⚠️ 不 detach
            )
            vec_pose = _normalize_quaternion(vec_pose)
            vec_tsl = self.geoptim.opt_val["hand_tsl_var"]   # ⚠️ 不 detach
            vec_shape = self.geoptim.const_val["hand_shape_gt"]
            mano_out = self.geoptim.mano_layer(
                vec_pose.unsqueeze(0), vec_shape.unsqueeze(0)
            )
            # 关键 uncenter: + center_joint（middle MCP world）把 middle-MCP-centered
            # 坐标系转回 wrist-centered world，再加 vec_tsl
            opt_joints_user = (mano_out.joints.squeeze(0)
                               + mano_out.center_joint.squeeze(0)
                               + vec_tsl)  # (21, 3) wrist-centered world
            # 按关节维度算 L2
            diff = (opt_joints_user - gt_joints_consistent).pow(2).sum(dim=-1)  # (21,)
            jt_loss = (diff * self._tip_weights).mean()  # 加权平均

            # 总损失 = CPF 原生 + 我们的关节目标
            total = loss_base + self.joint_target_weight * jt_loss
            total.backward()
            self.geoptim.optimizer.step()
            # ReduceLROnPlateau：loss 长期不降时降 lr
            self.geoptim.scheduler.step(total)

            # 记录历史
            history["loss"].append(loss_base.item())
            history["contact"].append(info.get("contact_loss", float("nan")))
            history["repulsion"].append(info.get("repulsion_loss", float("nan")))
            history["jt"].append(jt_loss.item())
            last_loss = total.item()

            # 进度打印：每 10% step 打一行
            if progress and (step % max(1, self.n_iter // 10) == 0 or step == self.n_iter - 1):
                print(f"    [cpf] step {step:04d} base={loss_base.item():.4e} "
                      f"contact={info.get('contact_loss', 0):.4e} "
                      f"repul={info.get('repulsion_loss', 0):.4e} "
                      f"jt={jt_loss.item():.4e}")

        # ===========================================================================
        # 最终恢复：把优化后的变量前向一次，得到最终 verts + joints
        # ===========================================================================
        # recover_hand() 内部输出是 middle-MCP-centered world（manotorch 默认行为），
        # 我们要的是 wrist-centered world（跟 NPZ GT 同坐标系），所以要 + center_joint
        # 做 uncenter。
        opt_verts, opt_joints_user, _ = self.geoptim.recover_hand(squeeze_out=True)
        opt_pose_quat = self.geoptim.recover_hand_pose()  # (16, 4) wxyz
        opt_tsl = self.geoptim.opt_val["hand_tsl_var"].detach().clone()
        # 用优化后的 pose 重新算一次 center_joint（uncenter 需要）
        with torch.no_grad():
            final_mano_out = self.geoptim.mano_layer(
                opt_pose_quat.unsqueeze(0), hand_shape.unsqueeze(0)
            )
            final_center = final_mano_out.center_joint.squeeze(0)
        # uncenter: middle-MCP-centered → wrist-centered
        opt_verts = opt_verts + final_center
        opt_joints_user = opt_joints_user + final_center
        # 抽 16 个标准 MANO 关节（按 SNAP_TO_STD_16 表做逆排列）
        opt_joints_std = opt_joints_user[SNAP_TO_STD_16]

        # 评估：手-物穿透深度（用 CPF 自己的 penetration_loss）
        # 用 uncenter 后的 opt_verts（wrist-centered world）来评估
        pen_depth = float(torch.sqrt(
            penetration_loss_hand_in_obj(opt_verts.detach(),
                                          obj_verts_world.to(device).float(),
                                          obj_faces.to(device).long())
        ).item())

        return {
            "opt_hand_pose_quat": opt_pose_quat.cpu().numpy(),
            "opt_hand_tsl": opt_tsl.cpu().numpy(),
            "opt_hand_verts": opt_verts.detach().cpu().numpy(),
            "opt_hand_joints_user": opt_joints_user.detach().cpu().numpy(),
            "opt_hand_joints_std": opt_joints_std.detach().cpu().numpy(),
            "penetration_depth": pen_depth,
            "contact_ratio": float(vertex_contact.mean()),
            "gt_verts_consistent": gt_verts_consistent.cpu().numpy(),   # NPZ GT
            "gt_joints_consistent": gt_joints_consistent.cpu().numpy(), # NPZ GT
            "loss_history": {k: np.asarray(v) for k, v in history.items()},
            "final_loss": float(last_loss) if last_loss is not None else float("nan"),
        }

    # ----------------------------------------------------------------- helpers
    def _build_param_optimizer(self) -> None:
        """构造 Adam 优化器：自动根据 ctrl_val 决定哪些变量被优化。

        关键：
            - hand_tsl_var 总是被优化（除非 ctrl 关掉）
            - hand_pose_var_val 总是被优化
            - β（hand_shape）已经被我们手动排除
            - obj_rot_var / obj_tsl_var 在 mode="hand" 下不会被优化
        """
        param = []
        if self.geoptim.ctrl_val["optimize_hand_tsl"]:
            param.append({"params": [self.geoptim.opt_val["hand_tsl_var"]],
                          "lr": 0.1 * self.lr})  # 平移用更小的 lr（数值大）
        if self.geoptim.ctrl_val["optimize_hand_pose"]:
            param.append({"params": [self.geoptim.opt_val["hand_pose_var_val"]]})
        if self.geoptim.ctrl_val["optimize_obj"]:  # mode="hand" 下通常 False
            param.append({"params": [self.geoptim.opt_val["obj_rot_var"]]})
            param.append({"params": [self.geoptim.opt_val["obj_tsl_var"]],
                          "lr": 0.1 * self.lr})
        if not param:
            raise RuntimeError("ArcticCpfFitter: no optimisable variables — nothing to do.")
        self.geoptim.optimizer = torch.optim.Adam(param, lr=self.lr)
        # ReduceLROnPlateau：loss 不降时降 lr
        self.geoptim.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.geoptim.optimizer, min_lr=1e-5, mode="min", factor=0.5, patience=20)
        self.geoptim.optimizing = True


# ===========================================================================
# 顶层入口 main()
# ===========================================================================
# 整体流程（5 步）：
#   1. 解析 CLI → 构造 device / 路径
#   2. load_sequence() 加载 processed 序列（NPZ 全部字段一次读入）
#   3. 加载 CPF 资源（anchor / palm / region）+ 物体 world 坐标
#   4. 构造 ArcticCpfFitter（CPF 优化器 + 我们加的 jt_loss）
#   5. 逐帧：hand_valid=True 跑 CPF 优化 → False 直接保留 GT → 落盘
#
# "hand_valid=False 不参与 loss" 的语义（架构文档 §8）：
#     接触距离内（≤ thresh）→ hand_valid[t]=True → 跑 CPF 优化
#     接触距离外（>  thresh）→ hand_valid[t]=False → opt_* = ARCTIC GT
#     来源：优先 NPZ 里的 {side}_hand_valid；找不到 fallback 到
#     {side}_hand_to_obj_dist.min(axis=1) <= contact_thresh
def main() -> None:
    # Force line-buffered stdout/stderr so progress is visible in real time
    # when output is redirected to a file (e.g. `python script.py > log &`).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass  # Python < 3.7 or already closed

    # -------------------- 1. CLI 解析 --------------------
    parser = argparse.ArgumentParser(
        description="ARCTIC MANO retarget via CPF (processed NPZ only, no Isaac Gym)")
    # processed 数据源
    parser.add_argument("--processed-root", type=str,
                        default=str(REF2DEX_ROOT / "processed_data" / "arctic"))
    parser.add_argument("--seq-id", type=str, default="s01/box_grab_01")
    parser.add_argument("--side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--frame-step", type=int, default=4)  # 每 4 帧采一帧
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)  # -1 = 到结尾
    parser.add_argument("--max-frames", type=int, default=0)  # 0 = 不限
    # fallback contact 距离阈值（NPZ 缺 hand_valid 时使用）
    parser.add_argument("--contact-thresh", type=float, default=DEFAULT_CONTACT_THRESHOLD,
                        help="Fallback contact distance threshold (m) when NPZ "
                             "lacks {side}_hand_valid; same convention as arctic_preprocess.")
    # 设备/优化超参
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-iter", type=int, default=400)        # 每帧 Adam 步数
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-contact-loss", type=float, default=10.0)
    parser.add_argument("--only-jt-loss", action="store_true",
                        help="Ablation: 只用 jt_loss 优化，禁用 CPF 原生损失 "
                             "(contact + repulsion + anatomical priors)。用于验证 jt_loss 单兵能力。")
    parser.add_argument("--lambda-repulsion-loss", type=float, default=0.5)
    parser.add_argument("--repulsion-query", type=float, default=0.030)
    parser.add_argument("--repulsion-threshold", type=float, default=0.080)
    parser.add_argument("--joint-target-weight", type=float, default=30.0)
    # 输出/资源
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mano-assets-root", type=str, default=str(MANO_ASSETS_ROOT))
    parser.add_argument("--anchor-root", type=str, default=str(ANCHOR_ROOT))
    parser.add_argument("--progress", action="store_true", default=False)
    args = parser.parse_args()

    # 资源存在性检查
    if not Path(args.mano_assets_root).exists():
        raise FileNotFoundError(f"MANO assets not found at {args.mano_assets_root}")
    if not Path(args.anchor_root).exists():
        raise FileNotFoundError(f"Anchor assets not found at {args.anchor_root}")

    # 设备选择
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if not torch.cuda.is_available() and device.type == "cuda":
        print("[warn] CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    # -------------------- 2. 加载 processed 序列 --------------------
    processed_root = Path(args.processed_root).resolve()
    sequence = load_sequence(
        processed_root=processed_root,
        seq_id=args.seq_id,
        side=args.side,
        frame_step=args.frame_step,
        start_frame=args.start_frame,
        end_frame=(None if args.end_frame < 0 else args.end_frame),
        max_frames=args.max_frames,
        contact_thresh=args.contact_thresh,
    )
    # 加载 v1.0 MANO faces（1538, 3）用于 face center 输出
    mano_model_root = ARCTIC_DATA_ROOT / "body_models" / "mano"
    _, _mano_faces = _load_mano_v1_assets(mano_model_root)
    _mano_faces = np.asarray(_mano_faces, dtype=np.int64)
    n_valid = int(sequence.hand_valid.sum())
    n_total = len(sequence.hand_valid)
    print(f"[cpf-fit] loaded seq={sequence.seq_id} side={sequence.side} "
          f"T={n_total}  valid={n_valid} ({n_valid / max(1, n_total):.1%})")

    # -------------------- 3. 加载 CPF 资源 --------------------
    # CPF 内部用相对路径读 anchor / MANO pkl，所以 chdir 到 CPF 根
    os.chdir(CPF_ROOT)

    # anchor_load 返回：
    #   - face_vert_idx:           (32, 3) 32 个 anchor 各自"代表"的面
    #   - anchor_weight:           (32, ?) anchor 在面内的 barycentric 权重
    #   - merged_vertex_assignment:(778,) 每个手顶点属于哪个 region (0~7)
    #   - anchor_mapping:          {region_id: [anchor_id, ...]} region → anchor 列表
    face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(
        str(ANCHOR_ROOT))
    # palm 区域（接触候选顶点）的 778 顶点 id
    palm_vid = np.loadtxt(str(CPF_ASSETS_DIR / "hand_palm_full.txt"), dtype=int)
    n_regions = int(merged_vertex_assignment.max()) + 1  # 8 (CPF 默认)

    # processed 模式：物体表面点 + 法线已在 world 坐标（NPZ 直接给出）
    obj_verts_world_arr = sequence.obj_verts_world        # (T, 2048, 3)
    obj_normals_world_arr = sequence.obj_normals_world    # (T, 2048, 3)
    # 无 obj_faces（NPZ 里没存三角面）→ 传空；penetration_depth 评估会被跳过
    obj_faces = np.zeros((0, 3), dtype=np.int64)

    # -------------------- 4. 构造 fitter --------------------
    fitter = ArcticCpfFitter(
        device=device,
        side=sequence.side,
        mano_assets_root=args.mano_assets_root,
        anchor_root=args.anchor_root,
        lr=args.lr,
        n_iter=args.n_iter,
        lambda_contact_loss=args.lambda_contact_loss,
        lambda_repulsion_loss=args.lambda_repulsion_loss,
        repulsion_query=args.repulsion_query,
        repulsion_threshold=args.repulsion_threshold,
        joint_target_weight=args.joint_target_weight,
    )

    # 5 指尖在 user order 里的 vertex id（MANO_TIP_VERTEX_INDICES 给出）
    hand_tip_vid = torch.tensor(MANO_TIP_VERTEX_INDICES[sequence.side],
                                device=device, dtype=torch.long)
    n_frames = sequence.frame_ids.size
    hand_valid = sequence.hand_valid  # (T,) bool，架构文档 §8

    # 输出缓冲区：每个数组都按帧顺序写入
    opt_pose = np.zeros((n_frames, 16, 4), dtype=np.float32)         # (T, 16, 4) wxyz
    opt_tsl = np.zeros((n_frames, 3), dtype=np.float32)             # (T, 3)
    opt_verts = np.zeros((n_frames, 778, 3), dtype=np.float32)      # (T, 778, 3)
    opt_face_centers = np.zeros((n_frames, 1538, 3), dtype=np.float32)  # (T, 1538, 3)
    opt_joints_user = np.zeros((n_frames, 21, 3), dtype=np.float32) # (T, 21, 3) user order
    opt_joints_std = np.zeros((n_frames, 16, 3), dtype=np.float32)  # (T, 16, 3) std order
    pen_depth_arr = np.zeros((n_frames,), dtype=np.float32)         # (T,) 穿透深度
    contact_ratio_arr = np.zeros((n_frames,), dtype=np.float32)     # (T,) 接触比例
    final_loss_arr = np.zeros((n_frames,), dtype=np.float32)        # (T,) 最终 loss
    gt_verts_consistent_arr = np.zeros((n_frames, 778, 3), dtype=np.float32)       # manotorch 自洽 GT
    gt_face_centers_consistent_arr = np.zeros((n_frames, 1538, 3), dtype=np.float32)

    # -------------------- 5. 逐帧处理 --------------------
    for t in range(n_frames):
        # 进度打印
        if args.progress or (n_frames >= 1 and (t % max(1, n_frames // 20) == 0)):
            print(f"[cpf-fit] frame {t + 1}/{n_frames}  seq={sequence.seq_id}  "
                  f"side={sequence.side}  valid={bool(hand_valid[t])}  "
                  f"beta_norm={float(sequence.mano_betas[t].norm()):.3f}")

        # 取这一帧的 GT 手部数据
        gt_verts = sequence.mano_vertices_world[t].detach().cpu().numpy()  # (778, 3)
        gt_joints = sequence.mano_joints_world[t].detach()                # (21, 3) device
        obj_verts_world = obj_verts_world_arr[t]      # (2048, 3) world
        obj_normals_world = obj_normals_world_arr[t]  # (2048, 3) world

        # ---- Pre-filter by hand_valid（架构文档 §8）----
        # 接触距离外的帧（手在空中 / 太远）→ 不跑 CPF，直接保留 ARCTIC GT
        # 行为：opt_* = ARCTIC GT；penetration / contact_ratio / final_loss = 0
        if not hand_valid[t]:
            opt_pose[t] = sequence.mano_pose_quat_local[t].detach().cpu().numpy()
            opt_tsl[t] = sequence.mano_translation[t].detach().cpu().numpy()
            opt_verts[t] = gt_verts
            opt_face_centers[t] = gt_verts[_mano_faces].mean(axis=1)
            opt_joints_user[t] = gt_joints.detach().cpu().numpy()
            opt_joints_std[t] = gt_joints.detach().cpu().numpy()[SNAP_TO_STD_16]
            # ⚠️ BUG FIX: gt_verts_consistent / gt_face_centers_consistent 也要填！
            # 否则 invalid 帧在可视化 show_mode=2 时 green 手是 (0,0,0) 全零 → 不可见，
            # 而 opt 有数据 → red 手可见 → 造成"绿色被红色带偏"的错觉。
            gt_verts_consistent_arr[t] = np.asarray(gt_verts, dtype=np.float32)
            gt_face_centers_consistent_arr[t] = np.asarray(
                gt_verts[_mano_faces].mean(axis=1), dtype=np.float32)
            # pen_depth_arr / contact_ratio_arr / final_loss_arr 保持 0.0
            continue

        # ---- 正常路径：现算 per-vertex contact info → fitter.fit() ----
        # 用 GT 手顶点（"完美手"）恢复 32 个 anchor 的世界坐标
        anchor_pos_world = recover_anchor(gt_verts, face_vert_idx, anchor_weight)
        anchor_pos_world = np.asarray(anchor_pos_world)  # torch → numpy

        # 计算 5 个张量：vertex_contact / contact_region_id / anchor_id / anchor_elasti / anchor_padding_mask
        (vertex_contact, contact_region_id,
         anchor_id, anchor_elasti, anchor_padding_mask) = compute_contact_info(
            hand_verts_world=gt_verts,
            obj_verts_world=obj_verts_world.astype(np.float32),
            hand_palm_vid=palm_vid,
            merged_vertex_assignment=merged_vertex_assignment,
            n_regions=n_regions,
            anchor_pos_world=anchor_pos_world,
            anchor_mapping=anchor_mapping,
        )

        # 跑 CPF 优化
        result = fitter.fit(
            hand_shape=sequence.mano_betas[t].detach().float(),
            hand_tsl_init=sequence.mano_translation[t].detach().float(),
            hand_pose_quat_init=sequence.mano_pose_quat_local[t].detach().float(),
            hand_joints_gt_world=gt_joints.float(),
            hand_verts_gt_world=torch.from_numpy(gt_verts).float().to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ),
            obj_verts_world=torch.from_numpy(obj_verts_world).float(),
            obj_normals_world=torch.from_numpy(obj_normals_world).float(),
            obj_faces=torch.from_numpy(obj_faces).long(),
            vertex_contact=vertex_contact,
            contact_region_id=contact_region_id,
            anchor_id=anchor_id,
            anchor_elasti=anchor_elasti,
            anchor_padding_mask=anchor_padding_mask,
            hand_tip_vid=hand_tip_vid,
            progress=args.progress,
            only_jt_loss=args.only_jt_loss,
        )

        # 把优化结果写回缓冲区
        opt_verts_t = result["opt_hand_verts"]    # (778, 3)
        opt_pose[t] = result["opt_hand_pose_quat"]
        opt_tsl[t] = result["opt_hand_tsl"]
        opt_verts[t] = opt_verts_t
        opt_face_centers[t] = opt_verts_t[_mano_faces].mean(axis=1)  # (1538, 3)
        opt_joints_user[t] = result["opt_hand_joints_user"]
        opt_joints_std[t] = result["opt_hand_joints_std"]
        pen_depth_arr[t] = result["penetration_depth"]
        contact_ratio_arr[t] = result["contact_ratio"]
        final_loss_arr[t] = result["final_loss"]
        gt_verts_consistent_arr[t] = result["gt_verts_consistent"]
        gt_face_centers_consistent_arr[t] = result["gt_verts_consistent"][_mano_faces].mean(axis=1)

    # -------------------- 6. 落盘 --------------------
    # 输出路径：统一落到 <Ref2Dex>/outputs/mano_fit/arctic_cpf/
    default_out = REF2DEX_ROOT / "outputs" / "mano_fit" / "arctic_cpf" / sequence.subject_id
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sequence.seq_name}_{sequence.side}.pkl"
    # 构造 payload：与原 arctic_mano_collision_opt.py 兼容
    payload = {
        # 元信息
        "seq_id": sequence.seq_id,
        "subject_id": sequence.subject_id,
        "seq_name": sequence.seq_name,
        "object_name": sequence.object_name,
        "side": sequence.side,
        "frame_ids": sequence.frame_ids,
        # 数据源路径
        "mano_file": str(sequence.mano_path),
        "object_file": str(sequence.object_path),
        "object_mesh_path": str(sequence.object_mesh_path),
        # 原始 ARCTIC 数据（下游消费者经常用）
        "mano_wrist_pos": sequence.wrist_pos.detach().cpu().numpy(),
        "mano_wrist_rot_aa": sequence.wrist_rot_aa.detach().cpu().numpy(),
        "mano_pose_quat": sequence.mano_pose_quat_world.detach().cpu().numpy(),
        "mano_pose_quat_local": sequence.mano_pose_quat_local.detach().cpu().numpy(),
        "mano_betas": sequence.mano_betas.detach().cpu().numpy(),
        "mano_translation": sequence.mano_translation.detach().cpu().numpy(),
        "mano_joints_world_user": sequence.mano_joints_world.detach().cpu().numpy(),
        "mano_vertices_world": sequence.mano_vertices_world.detach().cpu().numpy(),
        "object_trajectory": sequence.obj_trajectory.detach().cpu().numpy(),
        # valid 掩码（架构文档 §8）
        "hand_valid": sequence.hand_valid,
        # 优化后的手部数据
        "opt_hand_pose_quat": opt_pose,           # (T, 16, 4) wxyz
        "opt_hand_tsl": opt_tsl,                 # (T, 3)
        "opt_hand_verts": opt_verts,             # (T, 778, 3)
        "opt_hand_face_centers": opt_face_centers, # (T, 1538, 3)
        "opt_hand_joints_user": opt_joints_user, # (T, 21, 3) user order
        "opt_hand_joints_std": opt_joints_std,   # (T, 16, 3) standard MANO order
        # 评估指标
        "penetration_depth": pen_depth_arr,      # (T,)
        "contact_ratio": contact_ratio_arr,      # (T,)
        "final_loss": final_loss_arr,            # (T,)
        # manotorch 自洽 GT（与优化链路同源，用于可视化公平对比）
        "gt_verts_consistent": gt_verts_consistent_arr,             # (T, 778, 3)
        "gt_face_centers_consistent": gt_face_centers_consistent_arr, # (T, 1538, 3)
        # 配置（用于回溯）
        "config": {
            "n_iter": args.n_iter,
            "lr": args.lr,
            "lambda_contact_loss": args.lambda_contact_loss,
            "lambda_repulsion_loss": args.lambda_repulsion_loss,
            "repulsion_query": args.repulsion_query,
            "repulsion_threshold": args.repulsion_threshold,
            "joint_target_weight": args.joint_target_weight,
            "frame_step": args.frame_step,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "contact_thresh": args.contact_thresh,
        },
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)  # 写 pkl，下游消费者（artimano2dex / 可视化等）直接读
    print(f"[cpf-fit] Saved retarget result to {out_path}")
    # 打印整条序列的平均指标：渗透深度 / 接触比例 / 最终 loss
    print(f"[cpf-fit] mean penetration_depth = {pen_depth_arr.mean():.6f}, "
          f"mean contact_ratio = {contact_ratio_arr.mean():.4f}, "
          f"mean final_loss = {final_loss_arr.mean():.4f}")


# ===========================================================================
# 脚本入口
# ===========================================================================
# 当文件被直接运行（不是被 import）时，调用 main()
# 通常用法：
#   python preprocess/arctic_mano_cpf_fit.py --seq-id s01/box_grab_01 ...
if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
