#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRAB raw-data adapter used by :mod:`process.GRAB.stage2_optimize`.

It parses raw GRAB sequences, reconstructs MANO geometry, samples the object
surface, and returns in-memory fields for the common Stage 2 writer. It does
not write a Stage 1 dataset. GRAB uses ``flat_hand_mean=True`` and injects the
per-sequence hand ``v_template`` supplied by the dataset.
"""
from __future__ import annotations

import csv
import json
import os.path as op
import re
from pathlib import Path
from typing import Optional

import numpy as np

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
REF2DEX_ROOT = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
# MANO 模型目录（含 MANO_LEFT.pkl / MANO_RIGHT.pkl）
DEFAULT_MANO_MODEL_DIR = op.join(REF2DEX_ROOT, "dataset", "arctic", "data", "body_models", "mano")
# GRAB 数据集根目录；兼容以下布局：
#   1) {root}/grab/{subject}/*.npz + {root}/tools/...
#   2) {root}/data/{subject}/*.npz + {root}/tools/...
#   3) {root}/{subject}/*.npz + {root}/tools/...
DEFAULT_GRAB_ROOT = op.join(REF2DEX_ROOT, "dataset", "GRAB")
DEFAULT_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_NN_BATCH_SIZE = 16
SHARED_ASSET_ROOT = op.join(REF2DEX_ROOT, "assets", "shared")
SHARED_MANO_ASSET_ROOT = op.join(SHARED_ASSET_ROOT, "mano")
OBJECT_ASSET_ROOT = op.join(REF2DEX_ROOT, "assets", "grab", "objects")

# ARCTIC-preprocess 兼容常量
MANO_NUM_JOINTS = 16
MANO_NUM_FACES = 1538
MANO_NUM_HAND_POINTS = MANO_NUM_FACES
# 关节 → finger_id 映射（与 arctic_preprocess 完全一致）
JOINT_TO_FINGER = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
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




def build_SE3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """R: (..., 3, 3)  t: (..., 3) -> (..., 4, 4)。"""
    batch_shape = R.shape[:-2]
    device = R.device
    pose = torch.eye(4, device=device).expand(*batch_shape, 4, 4).contiguous()
    pose[..., :3, :3] = R
    pose[..., :3, 3] = t
    return pose




def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """面法向 (F, 3)。"""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    normals = mesh.face_normals.copy()
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(norms, 1e-10, None)
    return normals


def compute_face_normals_batched(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """批量计算面法向，避免逐帧构造 trimesh。"""
    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.clip(norms, 1e-10, None)


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


def compute_canonical_hand_surface(mano_layer: MANO) -> tuple:
    """返回 canonical MANO face-center 点和 face normal。"""
    with torch.no_grad():
        output = mano_layer()
        verts = output.vertices.squeeze(0).detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(mano_layer.faces, dtype=np.int64)
    face_centers = verts[faces].mean(axis=1).astype(np.float32)
    face_normals = compute_face_normals(verts, faces).astype(np.float32)
    return face_centers, face_normals


def resolve_grab_sequence_root(grab_root: str | Path) -> Path:
    """Return the directory that directly contains subject subdirectories.

    GRAB is stored either as ``{root}/grab/{subject}/*.npz``,
    ``{root}/data/{subject}/*.npz``, or ``{root}/data/grab/{subject}/*.npz``.
    """
    root = Path(grab_root).resolve()
    candidates = (
        root / "grab",
        root / "data" / "grab",
        root / "data",
        root,
    )
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("*/*.npz"), None) is not None:
            return candidate
    raise FileNotFoundError(
        f"No GRAB sequences found under {root}. Expected one of "
        f"{root / 'grab'}, {root / 'data' / 'grab'}, {root / 'data'}, "
        f"or {root} to contain */*.npz"
    )




def _resolve_manifest_seq_path(manifest_path: str, grab_root: str, row: dict, line_num: int) -> str:
    sequence_root = resolve_grab_sequence_root(grab_root)
    raw_path_text = str(
        row.get("raw_path")
        or row.get("raw_path_abs")
        or row.get("source_path")
        or row.get("path")
        or ""
    ).strip()
    seq_id_text = str(row.get("seq_id") or row.get("sequence") or "").strip()
    if raw_path_text:
        candidate = Path(raw_path_text)
        if not candidate.is_absolute():
            candidate_grab = Path(grab_root) / raw_path_text
            candidate_repo = Path(REF2DEX_ROOT) / raw_path_text
            if candidate_grab.exists():
                candidate = candidate_grab
            elif candidate_repo.exists():
                candidate = candidate_repo
            else:
                candidate = candidate_grab
    elif seq_id_text:
        if "/" not in seq_id_text:
            raise ValueError(
                f"Manifest {manifest_path} line {line_num}: seq_id must be '<subject>/<sequence>', got {seq_id_text!r}"
            )
        subject_id, seq_name = seq_id_text.split("/", 1)
        candidate = sequence_root / subject_id / f"{seq_name}.npz"
    else:
        raise ValueError(
            f"Manifest {manifest_path} line {line_num}: expected one of raw_path/raw_path_abs/source_path/path/seq_id"
        )
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} line {line_num}: missing sequence file {candidate}")
    return str(candidate)


def load_manifest_seq_paths(manifest_path: str, grab_root: str) -> list[str]:
    manifest = Path(manifest_path).resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    seq_paths: list[str] = []
    seen = set()
    suffix = manifest.suffix.lower()
    if suffix == ".csv":
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for line_num, row in enumerate(reader, start=2):
                seq_path = _resolve_manifest_seq_path(str(manifest), grab_root, row, line_num)
                if seq_path not in seen:
                    seen.add(seq_path)
                    seq_paths.append(seq_path)
    elif suffix == ".json":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Manifest {manifest} must be a JSON list, got {type(payload)!r}")
        for idx, item in enumerate(payload, start=1):
            if isinstance(item, str):
                row = {"seq_id": item}
            elif isinstance(item, dict):
                row = item
            else:
                raise ValueError(f"Manifest {manifest} item {idx}: unsupported type {type(item)!r}")
            seq_path = _resolve_manifest_seq_path(str(manifest), grab_root, row, idx)
            if seq_path not in seen:
                seen.add(seq_path)
                seq_paths.append(seq_path)
    else:
        line_num = 0
        for raw_line in manifest.read_text(encoding="utf-8").splitlines():
            line_num += 1
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue
            row = {"seq_id": line}
            seq_path = _resolve_manifest_seq_path(str(manifest), grab_root, row, line_num)
            if seq_path not in seen:
                seen.add(seq_path)
                seq_paths.append(seq_path)
    return seq_paths


def resolve_torch_device(device: str) -> str:
    device = str(device).strip()
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Preprocessor] CUDA unavailable, falling back to CPU.")
        return "cpu"
    return device or "cpu"


def nearest_neighbor_batch(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    device: str,
    frame_batch_size: int,
):
    """Compute exact src -> dst nearest neighbors for batched frames.

    Args:
        src_points: (T, Ns, 3)
        dst_points: (T, Nd, 3)
        device: torch device string
        frame_batch_size: chunk size along T for GPU cdist
    """
    T, Ns = src_points.shape[:2]
    src_to_dst_idx = np.zeros((T, Ns), dtype=np.int32)
    src_to_dst_dist = np.zeros((T, Ns), dtype=np.float32)

    if str(device).startswith("cuda"):
        torch_device = torch.device(device)
        chunk = max(1, int(frame_batch_size))
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            src_t = torch.from_numpy(src_points[start:end]).to(torch_device, dtype=torch.float32)
            dst_t = torch.from_numpy(dst_points[start:end]).to(torch_device, dtype=torch.float32)
            dist = torch.cdist(src_t, dst_t, p=2)
            src_dist, src_idx = dist.min(dim=2)
            src_to_dst_idx[start:end] = src_idx.detach().cpu().numpy().astype(np.int32)
            src_to_dst_dist[start:end] = src_dist.detach().cpu().numpy().astype(np.float32)
        return src_to_dst_idx, src_to_dst_dist

    for t in range(T):
        tree_dst = cKDTree(dst_points[t])
        d_src, i_src = tree_dst.query(src_points[t], k=1)
        src_to_dst_idx[t] = i_src
        src_to_dst_dist[t] = d_src.astype(np.float32)
    return src_to_dst_idx, src_to_dst_dist


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
    base = Path(grab_root).resolve()
    search_roots = [base]
    for extra in (base / "data", base / "grab", base.parent):
        if extra not in search_roots:
            search_roots.append(extra)
    for root in search_roots:
        for ext in (".ply", ".obj"):
            candidate = root / "tools" / "object_meshes" / "contact_meshes" / f"{obj_name}{ext}"
            if candidate.exists():
                p = candidate
                break
        else:
            continue
        break
    else:
        p = op.join(OBJECT_ASSET_ROOT, obj_name, "mesh.obj")
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




def load_environment_mesh(mesh_relpath: str, grab_root: str, seq_root: Optional[str] = None, unit: str = "m") -> trimesh.Trimesh:
    """Load a canonical environment mesh by its dataset-relative path.

    兼容两种布局：路径相对 ``grab_root``（如 ``tools/...``），或相对
    ``seq_root``（``resolve_grab_sequence_root`` 返回的目录）。
    """
    candidates = [op.join(grab_root, mesh_relpath)]
    if seq_root is not None:
        candidates.append(op.join(seq_root, mesh_relpath))
        candidates.append(op.join(op.dirname(seq_root.rstrip("/")), mesh_relpath))
    path = next((p for p in candidates if op.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Environment mesh not found: {mesh_relpath} (tried {candidates})")
    mesh = trimesh.load(path, process=False)
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


def sample_environment_surface(mesh: trimesh.Trimesh, num_points: int, seed: int = 42) -> tuple:
    """Stable canonical surface sampling for one environment asset.

    与物体采样一致：固定 seed 的均匀表面采样 + 重心坐标插值 vertex normals，
    保证同一 canonical point index 在所有序列/所有帧中对应同一表面点。
    """
    points, face_idx, bary = sample_mesh_surface(mesh, num_points, seed=seed)
    mesh.fix_normals()
    vertex_normals = mesh.vertex_normals
    normals = interpolate_vertex_attributes(face_idx, bary, vertex_normals, mesh.faces)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(norms, 1e-10, None)
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
    )




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

    def get_environment_assets(self) -> list[dict]:
        """Return generic environment assets stored in this raw sequence.

        返回的每个 asset 是一个通用 dict：
            name            资产名（例如 "table"），不含任何数据集特定语义
            mesh_relpath    canonical mesh 相对 grab_root 的路径
            global_orient   (T_raw, 3) axis-angle
            transl          (T_raw, 3)

        上层（Stage4 scene builder）只看到 "environment assets"，
        不知道也不需要知道 environment == table。
        """
        assets: list[dict] = []
        if "table" in self._raw.files:
            table = self._raw["table"].item()
            params = table.get("params", {})
            mesh_relpath = table.get("table_mesh", None)
            if mesh_relpath is None or "global_orient" not in params or "transl" not in params:
                raise ValueError(f"{self.path}: malformed table environment entry")
            assets.append({
                "name": "table",
                "mesh_relpath": str(mesh_relpath),
                "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
                "transl": np.asarray(params["transl"], dtype=np.float32),
            })
        return assets

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


def transform_object_points_batch(
    obj_points_canonical: np.ndarray,
    obj_normals_canonical: np.ndarray,
    global_rot: np.ndarray,
    global_trans: np.ndarray,
    device: str,
) -> tuple:
    """批量 rigid transform GRAB object points/normals to world space."""
    rot_t = torch.from_numpy(global_rot).float().to(device)
    R_global = axis_angle_to_rotmat(rot_t).detach().cpu().numpy()
    posed_points = np.einsum("tij,pj->tpi", R_global, obj_points_canonical) + global_trans[:, None, :]
    posed_normals = np.einsum("tij,pj->tpi", R_global, obj_normals_canonical)
    norms = np.linalg.norm(posed_normals, axis=-1, keepdims=True)
    posed_normals = posed_normals / np.clip(norms, 1e-10, None)
    return posed_points.astype(np.float32), posed_normals.astype(np.float32)

# ============================================================
# 单条序列处理（核心）
# ============================================================
class GRABRawAdapter:
    """Parse one raw GRAB sequence into the in-memory Stage 2 source fields."""

    def __init__(
        self,
        num_obj_points: int = 4096,
        device: str = DEFAULT_TORCH_DEVICE,
        max_frames: Optional[int] = None,
        frame_start: int = 0,
        grab_root: str = DEFAULT_GRAB_ROOT,
        mano_path: str = DEFAULT_MANO_MODEL_DIR,
        ds_rate: int = 1,
        obj_unit: str = "m",
        nn_batch_size: int = DEFAULT_NN_BATCH_SIZE,
        require_subject_vtemplate: bool = False,
    ):
        self.num_obj_points = num_obj_points
        self.device = resolve_torch_device(device)
        self.max_frames = max_frames
        self.frame_start = max(0, int(frame_start))
        self.grab_root = grab_root
        self.ds_rate = max(1, int(ds_rate))
        self.obj_unit = obj_unit
        self.nn_batch_size = max(1, int(nn_batch_size))
        self.require_subject_vtemplate = bool(require_subject_vtemplate)
        self._sequence_root = resolve_grab_sequence_root(self.grab_root)

        # 默认 MANO（用于手部语义标签等静态分析）
        print("[Preprocessor] Loading default MANO for both hands (flat_hand_mean=True)...")
        self.default_mano_r = MANO(
            mano_path,
            is_rhand=True,
            use_pca=True,
            num_pca_comps=24,
            flat_hand_mean=True,
        ).to(self.device)
        self.default_mano_l = MANO(
            mano_path,
            is_rhand=False,
            use_pca=True,
            num_pca_comps=24,
            flat_hand_mean=True,
        ).to(self.device)
        # 实际 forward 用的 MANO（按 subject vtemp 注入）—— 运行时构造并 cache
        self._mano_cache = {}    # key: (vtemp_path, is_rhand) -> ManoLayer
        self._missing_vtemp_warned = set()
        self.mano_path = mano_path

        # 手部语义标签（cache）
        print(f"[Preprocessor] Computing hand semantics...")
        self.right_finger_id, self.right_region_id = assign_hand_semantics(self.default_mano_r, is_right=True)
        self.left_finger_id, self.left_region_id = assign_hand_semantics(self.default_mano_l, is_right=False)
        self.right_hand_cano_points, self.right_hand_cano_normals = compute_canonical_hand_surface(self.default_mano_r)
        self.left_hand_cano_points, self.left_hand_cano_normals = compute_canonical_hand_surface(self.default_mano_l)
        self.right_faces = self.default_mano_r.faces.astype(np.int64)
        self.left_faces = self.default_mano_l.faces.astype(np.int64)
        self.num_hand_points = self.right_faces.shape[0]
        self.hand_point_id = np.arange(self.num_hand_points, dtype=np.int32)
        # 物体采样缓存
        self._obj_cache = {}
        # 环境资产 canonical surface 采样缓存：key -> (points, normals)
        self._env_cache: dict = {}
        # 解析后的 sequence 根目录，用于环境 mesh 与 subject asset 相对路径
        print(
            f"[Preprocessor] Device: {self.device} | "
            f"NN batch size: {self.nn_batch_size}"
        )

    def _get_mano_for_vtemp(self, vtemp_path: str, is_rhand: bool) -> MANO:
        """根据受试者 vtemp 构造（或取 cache）MANO layer。

        GRAB 关键：每受试者手型不同，必须用 v_template 注入，
        否则 forward 出来的顶点是 MANO 平均手型，与受试者 GT 不一致。
        """
        key = (vtemp_path, is_rhand)
        if key not in self._mano_cache:
            mano_kwargs = {
                "is_rhand": is_rhand,
                "use_pca": True,
                "num_pca_comps": 24,
                "flat_hand_mean": True,
            }
            if op.exists(vtemp_path):
                v_template = trimesh.load(vtemp_path, process=False).vertices.astype(np.float32)
                mano_kwargs["v_template"] = v_template  # type: ignore[assignment]
            elif self.require_subject_vtemplate:
                raise FileNotFoundError(
                    f"Missing required GRAB subject v_template: {vtemp_path}. "
                    "Check --grab-root (expected dataset/GRAB or dataset/GRAB/data) "
                    "and the raw GRAB tools/subject_meshes assets."
                )
            elif vtemp_path not in self._missing_vtemp_warned:
                print(
                    f"[Preprocessor] WARNING: missing subject v_template {vtemp_path}. "
                    "Falling back to default MANO mean shape."
                )
                self._missing_vtemp_warned.add(vtemp_path)
            m = MANO(self.mano_path, **mano_kwargs).to(self.device)
            self._mano_cache[key] = m
        return self._mano_cache[key]

    def _resolve_grab_asset_path(self, relative_path: str) -> str:
        """Resolve GRAB assets for both dataset/GRAB and dataset/GRAB/data roots."""
        candidate = Path(str(relative_path))
        if candidate.is_absolute():
            return str(candidate)
        roots = (
            Path(self.grab_root).resolve(),
            Path(self.grab_root).resolve() / "data",
            self._sequence_root,
            self._sequence_root.parent,
        )
        for root in roots:
            resolved = root / candidate
            if resolved.exists():
                return str(resolved)
        return str(Path(self.grab_root).resolve() / candidate)

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

    def _get_env_sampling(self, mesh_relpath: str, asset_name: str, num_points: int, seq_root: str) -> tuple:
        key = (str(mesh_relpath), asset_name, int(num_points))
        if key not in self._env_cache:
            mesh = load_environment_mesh(mesh_relpath, self.grab_root, seq_root=seq_root, unit=self.obj_unit)
            self._env_cache[key] = sample_environment_surface(mesh, int(num_points), seed=42)
            print(f"[Preprocessor] Cached environment {asset_name}: {len(mesh.vertices)} verts, {num_points} samples")
        return self._env_cache[key]

    def get_environment_geometry(
        self,
        seq_data: "GRABSeqData",
        frame_ids: np.ndarray,
        *,
        num_env_points: int,
    ) -> list[dict]:
        """Return posed environment assets for the selected frames.

        每个返回的 asset：
            name                资产名（如 "table"）
            canonical_points    (N_E, 3) canonical 表面采样点
            canonical_normals   (N_E, 3)
            poses_world         (T, 4, 4) 每帧世界位姿（T == len(frame_ids)）

        Stage4 scene builder 负责判断 pose 是否序列内恒定并决定
        static_world / dynamic_world 存储方式；本函数保持通用。
        """
        assets = seq_data.get_environment_assets()
        seq_root = str(self._sequence_root)
        result: list[dict] = []
        for asset in assets:
            cano_points, cano_normals = self._get_env_sampling(
                asset["mesh_relpath"], asset["name"], num_env_points, seq_root,
            )
            rot_aa = np.asarray(asset["global_orient"][frame_ids], dtype=np.float32)
            transl = np.asarray(asset["transl"][frame_ids], dtype=np.float32)
            if np.abs(transl).max() > 5.0:
                transl = transl / 1000.0
            rot_t = torch.from_numpy(rot_aa).float().to(self.device)
            R_env = axis_angle_to_rotmat(rot_t).detach().cpu().numpy()
            poses_world = build_SE3(
                torch.from_numpy(R_env).float(),
                torch.from_numpy(transl).float(),
            ).cpu().numpy().astype(np.float32)
            result.append({
                "name": asset["name"],
                "canonical_points": cano_points,
                "canonical_normals": cano_normals,
                "poses_world": poses_world,
            })
        return result

    # ---- MANO forward (PCA + flat_hand_mean=True) ----
    def _mano_forward(self, mano: MANO, T: int, hand_params: dict) -> tuple:
        """对 T 帧做 MANO forward，返回 (verts (T, 778, 3), joints (T, 16, 3))。"""
        rot = torch.from_numpy(hand_params["global_orient"]).float().to(self.device)
        pose = torch.from_numpy(hand_params["hand_pose"]).float().to(self.device)
        tsl = torch.from_numpy(hand_params["transl"]).float().to(self.device)
        betas_np = hand_params["betas"]
        if betas_np.ndim == 1:
            betas = torch.from_numpy(betas_np).float().unsqueeze(0).expand(T, -1).to(self.device)
        else:
            betas = torch.from_numpy(betas_np).float().to(self.device)
        # 对齐 betas 到当前加载的 MANO shapedirs 维度。
        # 在当前 Ref2Dex 环境里是 10 维；这里保留兼容层，避免替换 MANO 资产后因维度不同报错。
        actual_num_betas = mano.shapedirs.shape[-1]
        if betas.shape[-1] != actual_num_betas:
            betas = betas[..., :min(betas.shape[-1], actual_num_betas)]
            if actual_num_betas > betas.shape[-1]:
                # 模型维度 > betas 维度：补零
                pad = torch.zeros(*betas.shape[:-1], actual_num_betas - betas.shape[-1],
                                 device=betas.device, dtype=betas.dtype)
                betas = torch.cat([betas, pad], dim=-1)
        out = mano(global_orient=rot, hand_pose=pose, betas=betas, transl=tsl)
        return out.vertices, out.joints  # (T, 778, 3), (T, 16, 3)

    def _select_frame_ids(self, n_frames: int) -> np.ndarray:
        frame_ids = np.arange(self.frame_start, n_frames, self.ds_rate, dtype=np.int32)
        if self.max_frames is not None:
            frame_ids = frame_ids[:self.max_frames]
        if frame_ids.size == 0:
            raise ValueError("No frames selected after applying ds_rate / max_frames")
        return frame_ids

    def _empty_hand_payload(self, T: int) -> dict:
        return {
            "points": np.zeros((T, self.num_hand_points, 3), dtype=np.float32),
            "normals": np.zeros((T, self.num_hand_points, 3), dtype=np.float32),
            "to_obj_nn_id": np.full((T, self.num_hand_points), -1, dtype=np.int32),
            "min_dist_to_obj": np.full((T,), np.inf, dtype=np.float32),
            # Intentionally NO "root_pose": absent-hand path must not emit one,
            # and stage2 pack_stage2_hand treats hand_root_pose as optional.
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
        vtemp_path = self._resolve_grab_asset_path(vtemp_relpath)
        mano_for_seq = self._get_mano_for_vtemp(vtemp_path, is_rhand=(side == "right"))

        verts_t, joints_t = self._mano_forward(mano_for_seq, T, hand_params_sel)

        verts = verts_t.detach().cpu().numpy().astype(np.float32)
        joints = joints_t.detach().cpu().numpy().astype(np.float32)  # (T, 16, 3) — joint 0 = wrist
        faces = self.right_faces if side == "right" else self.left_faces
        face_pts = verts[:, faces].mean(axis=2).astype(np.float32)
        normals = compute_face_normals_batched(verts, faces).astype(np.float32)

        # 手部根节点 SE(3) pose：origin = wrist (joint 0), rotation = global_orient
        rot_aa = torch.from_numpy(hand_params_sel["global_orient"]).float().to(self.device)
        R_hand = axis_angle_to_rotmat(rot_aa).detach().cpu().numpy()
        hand_root_pose = build_SE3(
            torch.from_numpy(R_hand).float(),
            torch.from_numpy(joints[:, 0, :]).float(),
        ).cpu().numpy().astype(np.float32)

        hand_to_obj_nn_id, hand_to_obj_dist = nearest_neighbor_batch(
            face_pts,
            obj_points,
            device=self.device,
            frame_batch_size=self.nn_batch_size,
        )

        min_per_frame = hand_to_obj_dist.min(axis=1)

        # ---- MANO parameters (for stage 2/3 cross-dataset storage) ----
        # Persist raw pose / orient / transl and the per-subject v_template so
        # the train side can re-run MANO forward deterministically without
        # depending on which hand_params were re-fitted downstream.
        v_template = None
        if op.exists(vtemp_path):
            v_template = trimesh.load(vtemp_path, process=False).vertices.astype(np.float32)
        betas_np = hand_params.get("betas", None)
        if betas_np is None:
            betas_out = np.zeros((T, 10), dtype=np.float32)
        else:
            betas_out = np.asarray(betas_np, dtype=np.float32)
            if betas_out.ndim == 1:
                betas_out = np.broadcast_to(betas_out, (T, betas_out.shape[0])).astype(np.float32).copy()
        mano_payload = {
            "mano_global_orient": np.asarray(hand_params_sel["global_orient"], dtype=np.float32).copy(),
            "mano_transl": np.asarray(hand_params_sel["transl"], dtype=np.float32).copy(),
            "mano_pose": np.asarray(hand_params_sel["hand_pose"], dtype=np.float32).copy(),
            "mano_betas": betas_out,
            "mano_v_template": v_template,
        }

        return {
            "points": face_pts,
            "normals": normals,
            "to_obj_nn_id": hand_to_obj_nn_id,
            "min_dist_to_obj": min_per_frame.astype(np.float32),
            "root_pose": hand_root_pose,
            "mano": mano_payload,
        }

    def process_sequence(self, seq_path: str) -> dict:
        """处理单条 GRAB 序列并保留完整时间轴。"""
        seq_data = GRABSeqData(seq_path)
        obj_params = seq_data.get_object_params()
        n_frames = seq_data.n_frames
        frame_ids = self._select_frame_ids(n_frames)
        obj_name = seq_data.obj_name

        obj_rot_aa = np.asarray(obj_params["global_orient"][frame_ids], dtype=np.float32)
        obj_trans = np.asarray(obj_params["transl"][frame_ids], dtype=np.float32)
        if np.abs(obj_trans).max() > 5.0:
            obj_trans = obj_trans / 1000.0

        obj_cache = self._get_obj_sampling(obj_name)
        obj_points, obj_normals = transform_object_points_batch(
            obj_cache["points"],
            obj_cache["normals"],
            obj_rot_aa,
            obj_trans,
            device=self.device,
        )

        R_obj = axis_angle_to_rotmat(
            torch.from_numpy(obj_rot_aa).float().to(self.device)
        ).detach().cpu().numpy()
        obj_root_pose = build_SE3(
            torch.from_numpy(R_obj).float(),
            torch.from_numpy(obj_trans).float(),
        ).cpu().numpy().astype(np.float32)

        right_data = self._process_one_hand("right", seq_data, frame_ids, obj_points)
        left_data = self._process_one_hand("left", seq_data, frame_ids, obj_points)

        parent_dir = op.basename(op.dirname(seq_path))
        m = re.match(r"^s(\d+)$", parent_dir)
        subj_id = f"s{parent_dir[1:]}" if m else parent_dir
        action_name = op.basename(seq_path).replace(".npz", "")
        seq_id = f"{subj_id}/{action_name}"

        # ---- MANO config (per docs/指导.md cross-dataset compatibility) ----
        # GRAB is fixed: PCA24 + flat_hand_mean=True + per-subject v_template.
        # We still emit the descriptive fields so the stage 3 schema is
        # dataset-agnostic and the train-time MANO loader can dispatch on
        # them.
        mano_config = {
            "mano_use_pca": True,
            "mano_num_pca_comps": 24,
            "mano_flat_hand_mean": True,
            "mano_pose_repr": "pca",
        }

        output = {
            "seq_id": seq_id,
            "dataset_name": "grab",
            "subject_id": subj_id,
            "seq_name": action_name,
            "object_name": obj_name,
            "raw_frame_id": frame_ids.astype(np.int32),
            "obj_points_world": obj_points,
            "obj_normals_world": obj_normals,
            "obj_repr": "rigid_canonical",
            "obj_points_canonical": obj_cache["points"].astype(np.float32),
            "obj_normals_canonical": obj_cache["normals"].astype(np.float32),
            "obj_point_id": obj_cache["point_id"],
            "obj_root_pose": obj_root_pose,
            "right_hand_points_world": right_data["points"],
            "right_hand_normals_world": right_data["normals"],
            "right_hand_point_id": self.hand_point_id,
            "right_hand_cano_points": self.right_hand_cano_points.astype(np.float32),
            "right_hand_finger_id": self.right_finger_id,
            "right_hand_region_id": self.right_region_id,
            "right_hand_to_obj_nn_id": right_data["to_obj_nn_id"],
            "right_hand_min_dist_to_obj": right_data["min_dist_to_obj"],
            "left_hand_points_world": left_data["points"],
            "left_hand_normals_world": left_data["normals"],
            "left_hand_point_id": self.hand_point_id,
            "left_hand_cano_points": self.left_hand_cano_points.astype(np.float32),
            "left_hand_finger_id": self.left_finger_id,
            "left_hand_region_id": self.left_region_id,
            "left_hand_to_obj_nn_id": left_data["to_obj_nn_id"],
            "left_hand_min_dist_to_obj": left_data["min_dist_to_obj"],
        }
        if "root_pose" in right_data:
            output["right_hand_root_pose"] = right_data["root_pose"]
        if "root_pose" in left_data:
            output["left_hand_root_pose"] = left_data["root_pose"]
        # Per-hand MANO parameters (raw pose/orient/transl + subject v_template
        # + configuration). Stage 2 will forward these to the Stage 3 npz so
        # we can re-run MANO forward on the train side for cross-dataset
        # reconstruction and hand PCA perturbation.
        for hand_side, hand_data in (("right", right_data), ("left", left_data)):
            if "mano" in hand_data:
                for key, value in hand_data["mano"].items():
                    output[f"{hand_side}_{key}"] = value
        for key, value in mano_config.items():
            output[f"right_{key}"] = value
            output[f"left_{key}"] = value
        return output
