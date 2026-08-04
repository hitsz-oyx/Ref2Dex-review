#!/usr/bin/env python3
"""ARCTIC raw-data adapter used by :mod:`process.ARCTIC.stage2_optimize`.

It parses raw ARCTIC MANO/object trajectories, reconstructs hand geometry,
samples articulated object surfaces, and returns in-memory fields for the
common Stage 2 writer. It does not write a Stage 1 dataset.
"""
import json               # 读取 parts.json（铰接物体顶/底部件标签）
import os.path as op      # 路径拼接
from pathlib import Path

import numpy as np        # 数值计算核心库
from typing import Optional, Union  # 用于 max_frames: Optional[int]

# ============================================================
# 修复 numpy 1.24+ 兼容性
# ============================================================
# smplx / MANO 库内部仍使用已废弃的 numpy 类型别名。
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

import torch              # 深度学习框架（MANO forward 走 PyTorch）
import trimesh            # 网格处理库（法向、face normal、barycentric 等）
from scipy.spatial import cKDTree   # 高效最近邻查询（用于手-物对应）
from smplx import MANO    # MANO 手部参数化人体模型

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
DEFAULT_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_NN_BATCH_SIZE = 16
DEFAULT_MANO_BATCH_SIZE = 1024 if torch.cuda.is_available() else 320

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




def build_SE3(R: torch.Tensor, t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
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
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=R.dtype, device=R.device)
    else:
        t = t.to(device=R.device, dtype=R.dtype)
    batch_shape = R.shape[:-2]
    device = R.device
    # 初始化为单位 SE(3) 矩阵（齐次坐标）
    pose = torch.eye(4, device=device).expand(*batch_shape, 4, 4).contiguous()
    pose[..., :3, :3] = R   # 左上 3x3 = 旋转
    pose[..., :3, 3] = t    # 右上 3x1 = 平移
    return pose






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


def compute_face_normals_batched(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    批量计算 posed MANO mesh 的面法向。

    Args:
        vertices: (T, V, 3)
        faces:    (F, 3)

    Returns:
        normals:  (T, F, 3)
    """
    triangles = vertices[:, faces]  # (T, F, 3, 3)
    edge_01 = triangles[:, :, 1] - triangles[:, :, 0]
    edge_02 = triangles[:, :, 2] - triangles[:, :, 0]
    normals = np.cross(edge_01, edge_02)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.clip(norms, 1e-10, None)


def resolve_torch_device(device: str) -> str:
    """Resolve user-facing device string to an actually available torch device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(torch.device(device))


def nearest_neighbor_batch(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    device: str,
    frame_batch_size: int,
) -> tuple:
    """
    分帧批量计算 src -> dst 最近邻。

    Args:
        src_points: (T, Ns, 3)
        dst_points: (T, Nd, 3)
        device:     torch 设备；CUDA 时走 torch.cdist
        frame_batch_size: 每次并行处理多少帧

    Returns:
        src_to_dst_idx/dist
    """
    T, num_src, _ = src_points.shape
    src_to_dst_idx = np.zeros((T, num_src), dtype=np.int32)
    src_to_dst_dist = np.zeros((T, num_src), dtype=np.float32)

    if device.startswith("cuda"):
        torch_device = torch.device(device)
        for start in range(0, T, frame_batch_size):
            end = min(start + frame_batch_size, T)
            src_t = torch.from_numpy(src_points[start:end]).float().to(torch_device)
            dst_t = torch.from_numpy(dst_points[start:end]).float().to(torch_device)
            dist = torch.cdist(src_t, dst_t)
            src_dist, src_idx = dist.min(dim=2)
            src_to_dst_idx[start:end] = src_idx.detach().cpu().numpy().astype(np.int32)
            src_to_dst_dist[start:end] = src_dist.detach().cpu().numpy().astype(np.float32)
        return src_to_dst_idx, src_to_dst_dist

    for t in range(T):
        tree_dst = cKDTree(dst_points[t])
        dists, idxs = tree_dst.query(src_points[t], k=1)
        src_to_dst_idx[t] = idxs.astype(np.int32)
        src_to_dst_dist[t] = dists.astype(np.float32)
    return src_to_dst_idx, src_to_dst_dist


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


def compute_canonical_hand_surface(mano_layer: MANO) -> tuple:
    """返回 canonical MANO face-center 点和 face normal。"""
    with torch.no_grad():
        output = mano_layer()
        verts = output.vertices.squeeze(0).detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(mano_layer.faces, dtype=np.int64)
    face_centers = verts[faces].mean(axis=1).astype(np.float32)
    face_normals = compute_face_normals(verts, faces).astype(np.float32)
    return face_centers, face_normals




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






def transform_object_points_batch(
    obj_points_canonical: np.ndarray,
    obj_normals_canonical: np.ndarray,
    sample_parts: Optional[np.ndarray],
    arti_angle: np.ndarray,
    global_rot: np.ndarray,
    global_trans: np.ndarray,
    device: str,
) -> tuple:
    """
    批量把 canonical 采样点/法向变换到 posed 世界空间。

    这里不再逐帧重建 posed mesh 再插值，而是直接对采样点执行：
        1. 顶部件铰接旋转（若有）
        2. 物体整体刚体变换

    因为这两步都属于刚体变换，所以对采样点和法向直接施加同样旋转即可。
    """
    torch_device = torch.device(device)
    arti_angle = np.asarray(arti_angle, dtype=np.float32).reshape(-1)
    global_rot = np.asarray(global_rot, dtype=np.float32)
    global_trans = np.asarray(global_trans, dtype=np.float32)
    T = global_rot.shape[0]

    points = torch.from_numpy(obj_points_canonical).float().to(torch_device)
    normals = torch.from_numpy(obj_normals_canonical).float().to(torch_device)
    points = points.unsqueeze(0).expand(T, -1, -1).clone()
    normals = normals.unsqueeze(0).expand(T, -1, -1).clone()

    if sample_parts is not None and np.any(sample_parts):
        top_mask = torch.from_numpy(sample_parts.astype(np.bool_)).to(torch_device)
        c = torch.cos(torch.from_numpy(arti_angle).float().to(torch_device))
        s = torch.sin(torch.from_numpy(arti_angle).float().to(torch_device))
        R_arti = torch.zeros((T, 3, 3), device=torch_device, dtype=torch.float32)
        R_arti[:, 0, 0] = c
        R_arti[:, 0, 1] = s
        R_arti[:, 1, 0] = -s
        R_arti[:, 1, 1] = c
        R_arti[:, 2, 2] = 1.0
        points[:, top_mask] = torch.matmul(points[:, top_mask], R_arti.transpose(1, 2))
        normals[:, top_mask] = torch.matmul(normals[:, top_mask], R_arti.transpose(1, 2))

    R_global = axis_angle_to_rotmat(
        torch.from_numpy(global_rot).float().to(torch_device)
    )
    trans = torch.from_numpy(global_trans).float().to(torch_device)
    posed_points = torch.matmul(points, R_global.transpose(1, 2)) + trans[:, None, :]
    posed_normals = torch.matmul(normals, R_global.transpose(1, 2))
    posed_normals = posed_normals / torch.clamp(
        torch.linalg.norm(posed_normals, dim=-1, keepdim=True),
        min=1e-10,
    )

    return (
        posed_points.detach().cpu().numpy().astype(np.float32),
        posed_normals.detach().cpu().numpy().astype(np.float32),
    )


# ============================================================
# 核心处理逻辑
# ============================================================

class ArcticRawAdapter:
    """Parse ARCTIC sequences into the minimal common Stage 2 source fields."""

    def __init__(
        self,
        num_obj_points: int = 4096,
        device: str = DEFAULT_TORCH_DEVICE,
        preprocess_stride: int = 1,
        max_frames: Optional[int] = None,
        obj_unit: str = "mm",      # mesh.obj 单位：'m' / 'mm' / 'auto'
                                    # 默认 'mm'：ARCTIC 原始 mesh.obj 是 mm（box max=244.91mm），
                                    # 函数内部会 /1000 转成 m，pipeline 统一 m。
        nn_batch_size: int = DEFAULT_NN_BATCH_SIZE,
        mano_batch_size: int = DEFAULT_MANO_BATCH_SIZE,
    ):
        """
        Args:
            num_obj_points:        每个物体表面采样点数（固定 4096）
            device:                PyTorch 设备（如 cuda / cuda:3 / cpu）
            preprocess_stride:     预处理阶段物理降采样步长。1=保留全部帧。
            max_frames:            单条序列最多处理前 N 帧（None=不限制）。
                                    主要用于 quick smoke test。
        """
        self.num_obj_points = num_obj_points
        self.device = resolve_torch_device(device)
        self.preprocess_stride = max(1, int(preprocess_stride))
        self.max_frames = max_frames  # None=不限；>0=只处理前 N 帧（quick test 用）
        self.obj_unit = obj_unit      # mesh.obj 单位：'m' / 'mm' / 'auto'
        self.nn_batch_size = max(1, int(nn_batch_size))
        self.mano_batch_size = max(1, int(mano_batch_size))

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
            self.right_hand_cano_points,
            self.right_hand_cano_normals,
        ) = compute_canonical_hand_surface(self.mano_r)
        (
            self.left_finger_id,
            self.left_region_id,
        ) = assign_hand_semantics(self.mano_l, is_right=False)
        (
            self.left_hand_cano_points,
            self.left_hand_cano_normals,
        ) = compute_canonical_hand_surface(self.mano_l)

        # ----- 缓存 MANO face 索引 -----
        # MANO 拓扑固定，face 索引在 forward 之后可以直接从 layer.faces 取
        self.right_faces = self.mano_r.faces.astype(np.int64)   # (1538, 3)
        self.left_faces = self.mano_l.faces.astype(np.int64)
        self.num_hand_points = self.right_faces.shape[0]        # 1538

        # 持久化的点 ID（左右手各自 0..1537，方便按 face 索引取同一拓扑点）
        self.right_hand_point_id = np.arange(self.num_hand_points, dtype=np.int32)
        self.left_hand_point_id = np.arange(self.num_hand_points, dtype=np.int32)

        # 物体采样缓存（同一物体只需采样一次）
        self._obj_cache = {}

        print(f"[Preprocessor] Device: {self.device}")
        print(f"[Preprocessor] Preprocess stride: {self.preprocess_stride}")
        print(f"[Preprocessor] MANO batch size: {self.mano_batch_size}")
        print(f"[Preprocessor] NN frame batch size: {self.nn_batch_size}")
        print(f"[Preprocessor] Num obj points: {num_obj_points}")

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
        """Parse one sequence and return fields consumed by the Stage 2 writer."""
        # 路径解析: RAW_SEQS_DIR/s01/box_grab_01.mano.npy
        seq_rel = Path(mano_p).resolve().relative_to(Path(RAW_SEQS_DIR).resolve()).as_posix()
        seq_rel = seq_rel.replace(".mano.npy", "")
        subject, seq_name = seq_rel.split("/")
        obj_name = seq_name.split("_")[0]   # "box_grab_01" → "box"

        # 同名 .object.npy 存了物体参数
        obj_p = mano_p.replace(".mano.npy", ".object.npy")

        # ----- 读原始数据 -----
        mano_data = np.load(mano_p, allow_pickle=True).item()    # dict
        obj_params = np.load(obj_p, allow_pickle=True)            # (T, 7)

        T_raw = int(obj_params.shape[0])
        raw_frame_id = np.arange(0, T_raw, self.preprocess_stride, dtype=np.int32)
        if self.max_frames is not None:
            raw_frame_id = raw_frame_id[: int(self.max_frames)]
        if raw_frame_id.size == 0:
            raise ValueError("No frames selected after applying preprocess_stride / max_frames")

        obj_params = obj_params[raw_frame_id]
        for side in ("right", "left"):
            for k in ("rot", "pose", "trans"):
                if k in mano_data[side]:
                    mano_data[side][k] = mano_data[side][k][raw_frame_id]
        T = int(raw_frame_id.shape[0])
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
        batch_size = min(T, self.mano_batch_size)
        all_right_verts = []
        all_right_joints = []
        all_left_verts = []
        all_left_joints = []

        for i in range(0, T, batch_size):
            end = min(i + batch_size, T)
            b = end - i

            # 右手: shape 兼容 (10,) 和 (T, 10) 两种存储形式
            shape_r_batch = shape_r.expand(b, -1) if shape_r.ndim == 1 else shape_r[i:end]
            out_r = self.mano_r(
                global_orient=rot_r[i:end],
                hand_pose=pose_r[i:end],
                betas=shape_r_batch,
                transl=trans_r[i:end],
            )
            all_right_verts.append(out_r.vertices.detach().cpu().numpy())
            # joints: (b, 16, 3) — joint 0 = wrist (world space, post forward)
            all_right_joints.append(out_r.joints.detach().cpu().numpy())

            # 左手同理
            shape_l_batch = shape_l.expand(b, -1) if shape_l.ndim == 1 else shape_l[i:end]
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
        right_joints = np.concatenate(all_right_joints, axis=0)  # (T, 16, 3)
        left_verts = np.concatenate(all_left_verts, axis=0)       # (T, 778, 3)
        left_joints = np.concatenate(all_left_joints, axis=0)     # (T, 16, 3)

        # ----- MANO 顶点 → face-center 采样（1538 点）-----
        # 每个 face 三个顶点取均值，得到一个"中心点"
        # 由于 MANO 拓扑固定，face i 在所有手型下都是同一个三角形 → 拓扑稳定
        right_face_pts = right_verts[:, self.right_faces].mean(axis=2)  # (T, 1538, 3)
        left_face_pts = left_verts[:, self.left_faces].mean(axis=2)     # (T, 1538, 3)

        # ----- 计算每帧的面法向（per-face normal）-----
        # 用 posed mesh 的 face normal（不是 vertex normal），与 face center 天然对应
        right_normals = compute_face_normals_batched(
            right_verts, self.right_faces
        ).astype(np.float32)
        left_normals = compute_face_normals_batched(
            left_verts, self.left_faces
        ).astype(np.float32)

        # ----- 物体：读 canonical 采样，逐帧变换 -----
        obj_cache = self._get_obj_sampling(obj_name)
        obj_canon_points = obj_cache["points"]       # (No, 3)
        obj_canon_normals = obj_cache["normals"]      # (No, 3)
        obj_point_id = obj_cache["point_id"]          # (No,)
        obj_sample_parts = obj_cache["sample_parts"]
        obj_points, obj_normals = transform_object_points_batch(
            obj_canon_points,
            obj_canon_normals,
            obj_sample_parts,
            arti[:, 0],
            obj_rot_aa,
            obj_trans_m,
            self.device,
        )

        # ----- 物体根节点 SE(3) pose -----
        R_o = axis_angle_to_rotmat(
            torch.from_numpy(obj_rot_aa).float().to(self.device)
        ).detach().cpu()
        obj_root_pose = build_SE3(
            R_o,
            torch.from_numpy(obj_trans_m).float().to(R_o.device),
        ).detach().cpu().numpy()

        # ----- 手部根节点 SE(3) pose -----
        #   origin  = MANO joint 0 (wrist) 的 world 坐标
        #   rotation = MANO global_orient 的 axis-angle → 旋转矩阵
        #   T_world_from_hand_root = [R_root, wrist; 0 1]
        R_hand_r = axis_angle_to_rotmat(rot_r).detach().cpu()
        right_hand_root_pose = build_SE3(
            R_hand_r,
            right_joints[:, 0, :],  # (T, 3) wrist position in world
        ).numpy().astype(np.float32)
        R_hand_l = axis_angle_to_rotmat(rot_l).detach().cpu()
        left_hand_root_pose = build_SE3(
            R_hand_l,
            left_joints[:, 0, :],
        ).numpy().astype(np.float32)

        # Stage 2 only needs hand -> object NN for frame filtering and
        # the sampled-normal penetration diagnostic.
        right_hand_to_obj_nn_id, right_hand_to_obj_dist = nearest_neighbor_batch(
            right_face_pts.astype(np.float32),
            obj_points.astype(np.float32),
            self.device,
            self.nn_batch_size,
        )
        left_hand_to_obj_nn_id, left_hand_to_obj_dist = nearest_neighbor_batch(
            left_face_pts.astype(np.float32),
            obj_points.astype(np.float32),
            self.device,
            self.nn_batch_size,
        )

        right_min_dist = right_hand_to_obj_dist.min(axis=1).astype(np.float32)
        left_min_dist = left_hand_to_obj_dist.min(axis=1).astype(np.float32)

        # Minimal in-memory source schema consumed by process.common.stage2.
        seq_id = f"{subject}/{seq_name}"

        output = {
            "seq_id": seq_id,
            "dataset_name": "arctic",
            "subject_id": subject,
            "seq_name": seq_name,
            "object_name": obj_name,
            "raw_frame_id": raw_frame_id.astype(np.int32),
            "obj_points_world": obj_points.astype(np.float32),
            "obj_normals_world": obj_normals.astype(np.float32),
            "obj_point_id": obj_point_id,
            "obj_root_pose": obj_root_pose.astype(np.float32),
            "right_hand_points_world": right_face_pts.astype(np.float32),
            "right_hand_normals_world": right_normals.astype(np.float32),
            "right_hand_point_id": self.right_hand_point_id,
            "right_hand_cano_points": self.right_hand_cano_points.astype(np.float32),
            "right_hand_finger_id": self.right_finger_id,
            "right_hand_region_id": self.right_region_id,
            "right_hand_to_obj_nn_id": right_hand_to_obj_nn_id,
            "right_hand_min_dist_to_obj": right_min_dist,
            "right_hand_root_pose": right_hand_root_pose,
            "left_hand_points_world": left_face_pts.astype(np.float32),
            "left_hand_normals_world": left_normals.astype(np.float32),
            "left_hand_point_id": self.left_hand_point_id,
            "left_hand_cano_points": self.left_hand_cano_points.astype(np.float32),
            "left_hand_finger_id": self.left_finger_id,
            "left_hand_region_id": self.left_region_id,
            "left_hand_to_obj_nn_id": left_hand_to_obj_nn_id,
            "left_hand_min_dist_to_obj": left_min_dist,
            "left_hand_root_pose": left_hand_root_pose,
        }
        return output
