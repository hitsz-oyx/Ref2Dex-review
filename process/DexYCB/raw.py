#!/usr/bin/env python3
"""DexYCB raw-data adapter for :mod:`process.DexYCB.stage4_cm`.

Parses raw DexYCB capture sessions (subject/capture/{meta.yml,pose.npz,labels_*.npz}),
reconstructs MANO hand geometry (right-hand only, per-sequence single side) and
samples YCB object surfaces, then returns in-memory fields for the common
Stage 4 writer.

Coordinate frames
-----------------
DexYCB sequence-level ``pose.npz`` stores both ``pose_y`` and ``pose_m`` in
the reference/master-camera frame.  That camera is the serial whose entry in
``extrinsics_<date>.yml`` is identity.  The frame is fixed for the whole
sequence, so it serves as the Stage 4 ``world`` frame without another
extrinsic transform.  Object quaternions in this file use ``(x, y, z, w)``.
"""
from __future__ import annotations

import os
import os.path as op
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial import cKDTree
from smplx import MANO

from process.ARCTIC.raw import (
    REGION_FINGER_PAD,
    REGION_FINGERTIP,
    REGION_PALM,
    assign_hand_semantics,
    compute_canonical_hand_surface,
)

# ============================================================
# 路径配置（可通过命令行参数 / 环境变量覆盖）
# ============================================================
REF2DEX_ROOT = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
_DEXYCB_ROOT_CANDIDATES = [
    os.environ.get("REF2DEX_DEXYCB_ROOT", ""),
    op.join(REF2DEX_ROOT, "data", "raw_data", "DexYCB"),
]
DEXYCB_ROOT = next(
    (path for path in _DEXYCB_ROOT_CANDIDATES if path and op.isdir(op.join(path, "raw", "calibration"))),
    _DEXYCB_ROOT_CANDIDATES[1],
)
# 解析 raw/ 目录（包含 calibration/、models/、<subject>/ 子目录）
DEXYCB_RAW_ROOT = op.join(DEXYCB_ROOT, "raw")
# MANO 模型目录（复用全仓库共享的 MANO 资源，与 GRAB/ARCTIC 同源）
MANO_MODEL_DIR_CANDIDATES = [
    os.environ.get("REF2DEX_MANO_ROOT", ""),
    op.join(REF2DEX_ROOT, "data", "raw_data", "ARCTIC", "arctic", "data", "body_models", "mano"),
    op.join(REF2DEX_ROOT, "data", "raw_data", "shared", "mano"),
    op.join(REF2DEX_ROOT, "assets", "shared", "mano"),
]
MANO_MODEL_DIR = next(
    (path for path in MANO_MODEL_DIR_CANDIDATES if path and op.isfile(op.join(path, "MANO_LEFT.pkl"))),
    MANO_MODEL_DIR_CANDIDATES[1],
)

DEFAULT_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MANO_BATCH_SIZE = 256 if torch.cuda.is_available() else 64
DEFAULT_NN_BATCH_SIZE = 8

# YCB object id (1-21) → 物体目录名（与 models/<name>/ 一一对应）
YCB_ID_TO_NAME = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}
YCB_ID_TO_NAME[0] = "background"  # placeholder, never used in grasp


# ============================================================
# 辅助函数
# ============================================================


def parse_extrinsics_yml(path: str) -> dict[str, np.ndarray]:
    """Parse ``extrinsics_<date>/extrinsics.yml`` to a {serial: 3x4 np.ndarray} dict.

    The yaml stores each camera's 3x4 SE(3) as a 12-tuple under the
    ``extrinsics`` key.  The matrix maps camera frame → reference-camera frame.
    """
    loader = yaml.SafeLoader
    loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )
    with open(path, "r") as handle:
        data = yaml.load(handle, Loader=loader)
    raw = data["extrinsics"]
    out: dict[str, np.ndarray] = {}
    for serial, value in raw.items():
        # 防止 yaml 把 tuple 解析成 list
        arr = np.asarray(value, dtype=np.float64).reshape(3, 4)
        out[str(serial)] = arr.astype(np.float32)
    return out


def parse_intrinsics_yml(path: str) -> dict[str, dict[str, np.ndarray]]:
    """Parse ``calibration/intrinsics/<serial>_<W>x<H>.yml``.

    Each file has:
        color: {fx, fy, ppx, ppy}
        depth: {fx, fy, ppx, ppy}
        extrinsics: 3x4 SE(3)  (depth → color 配准)
    Returns:  {color_K(3,3), color_dist=None, depth_K(3,3), depth_to_color_SE3(3,4)}
    """
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    color = data["color"]
    depth = data["depth"]
    color_K = np.array(
        [
            [color["fx"], 0.0, color["ppx"]],
            [0.0, color["fy"], color["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    depth_K = np.array(
        [
            [depth["fx"], 0.0, depth["ppx"]],
            [0.0, depth["fy"], depth["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    extr = np.asarray(data["extrinsics"], dtype=np.float64).reshape(3, 4)
    return {
        "color_K": color_K.astype(np.float32),
        "depth_K": depth_K.astype(np.float32),
        "depth_to_color_SE3": extr.astype(np.float32),
    }


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert DexYCB ``(..., 4)`` quaternions in ``(x, y, z, w)`` order."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q.shape}")
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # 归一化（防御性）
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if np.any(n < 1e-8) or not np.all(np.isfinite(n)):
        raise ValueError("DexYCB object pose contains a zero or non-finite quaternion")
    w = w / n
    x = x / n
    y = y / n
    z = z / n
    R = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    return R.astype(np.float32)


def build_SE3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(..., 3, 3) + (..., 3) → (..., 4, 4)."""
    pose = np.zeros((*R.shape[:-2], 4, 4), dtype=np.float32)
    pose[..., :3, :3] = R
    pose[..., :3, 3] = t
    pose[..., 3, 3] = 1.0
    return pose


def transform_points(points: np.ndarray, SE3: np.ndarray) -> np.ndarray:
    """Apply batched SE(3) to a point cloud.

    Args:
        points: (T, N, 3) in source frame
        SE3:    (T, 4, 4) source → target
    Returns:
        (T, N, 3) in target frame
    """
    R = SE3[..., :3, :3]
    t = SE3[..., :3, 3]
    return np.einsum("tij,tnj->tni", R, points) + t[:, None, :]


def transform_normals(normals: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply batched rotation (no translation) to normals. (T, N, 3) × (T, 3, 3)."""
    return np.einsum("tij,tnj->tni", R, normals)


def find_reference_camera(
    extrinsics: dict[str, np.ndarray], serials: list[str], *, atol: float = 1e-5
) -> str:
    """Return the capture serial defining the sequence-level pose frame.

    DexYCB calibration represents all cameras relative to one reference
    camera.  Its 3x4 extrinsic is identity.  Requiring exactly one such serial
    prevents silently mixing a camera-frame MANO trajectory with object poses
    expressed in a different frame.
    """
    identity = np.concatenate(
        [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1
    )
    matches = [
        str(serial)
        for serial in serials
        if str(serial) in extrinsics
        and np.allclose(np.asarray(extrinsics[str(serial)]), identity, atol=atol, rtol=0.0)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one identity-extrinsic DexYCB reference camera, "
            f"got {matches} among serials={serials}"
        )
    return matches[0]


def trim_invalid_mano_prefix(
    raw_frame_id: np.ndarray, pose_y: np.ndarray, pose_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim leading all-zero MANO annotations without changing time spacing."""
    valid_mano = np.any(np.abs(pose_m) > 0.0, axis=(1, 2))
    valid_idx = np.flatnonzero(valid_mano)
    if valid_idx.size == 0:
        raise ValueError("DexYCB sequence has no valid MANO annotations")
    first_valid, last_valid = int(valid_idx[0]), int(valid_idx[-1])
    if not np.all(valid_mano[first_valid : last_valid + 1]):
        raise ValueError(
            "DexYCB MANO annotations contain internal gaps; trimming them would "
            "break runtime-stride time semantics"
        )
    keep = slice(first_valid, last_valid + 1)
    return raw_frame_id[keep], pose_y[keep], pose_m[keep]


def expand_mano_pca_pose(coefficients: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Expand DexYCB's 45 MANO PCA coefficients to joint axis-angles."""
    if coefficients.ndim != 2 or coefficients.shape[-1] != 45:
        raise ValueError(f"Expected MANO PCA coefficients shaped (T,45), got {coefficients.shape}")
    if components.shape != (45, 45):
        raise ValueError(f"Expected MANO PCA components shaped (45,45), got {components.shape}")
    return coefficients @ components


def sample_mesh_surface(
    mesh: trimesh.Trimesh, n_points: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """均匀随机采样 n_points 个表面点，返回 (points, face_idx, barycentric)。"""
    np.random.seed(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    faces = mesh.faces[face_idx]
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]
    triangles = np.stack([v0, v1, v2], axis=1)
    bary = trimesh.triangles.points_to_barycentric(triangles, points)
    bary = np.nan_to_num(bary, nan=1.0 / 3.0)
    return points.astype(np.float32), face_idx, bary.astype(np.float32)


def load_ycb_object_mesh(ycb_id: int, models_root: str) -> trimesh.Trimesh:
    """加载 YCB 物体 mesh。YCB 模型是 m 为单位（master_chef_can 约 0.1m 直径）。

    Args:
        ycb_id: 1-21，对应 YCB_ID_TO_NAME
        models_root: calibration 同级的 models/ 目录
    """
    name = YCB_ID_TO_NAME[int(ycb_id)]
    mesh_path = op.join(models_root, name, "textured_simple.obj")
    if not op.isfile(mesh_path):
        raise FileNotFoundError(f"YCB mesh not found: {mesh_path}")
    # YCB 顶点 v 行为 6 列 (x y z r g b) ，trimesh 会自动处理
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    # 兜底单位：若 |max|>10，视为 mm，需要 /1000
    if np.abs(np.asarray(mesh.vertices)).max() > 10.0:
        mesh.vertices = np.asarray(mesh.vertices) / 1000.0
    return mesh


# ============================================================
# 核心处理
# ============================================================


class DexYCBRawAdapter:
    """Parse DexYCB capture sessions into the common Stage 4 source schema.

    Single-hand per sequence: ``mano_sides`` from meta.yml gives exactly one
    side string.  The current adapter deliberately supports right-hand
    captures only; the Stage 4 entry point filters out left-hand captures.
    """

    def __init__(
        self,
        num_obj_points: int = 4096,
        device: str = DEFAULT_TORCH_DEVICE,
        preprocess_stride: int = 1,
        max_frames: Optional[int] = None,
        nn_batch_size: int = DEFAULT_NN_BATCH_SIZE,
        mano_batch_size: int = DEFAULT_MANO_BATCH_SIZE,
        dexycb_root: Optional[str] = None,
    ):
        self.num_obj_points = int(num_obj_points)
        self.device = str(torch.device(device) if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
        self.preprocess_stride = max(1, int(preprocess_stride))
        self.max_frames = max_frames
        self.nn_batch_size = max(1, int(nn_batch_size))
        self.mano_batch_size = max(1, int(mano_batch_size))
        self.dexycb_root = Path(dexycb_root).resolve() if dexycb_root else Path(DEXYCB_ROOT).resolve()
        # calibration / models 都位于 <root>/raw/ 之下，兼容两种 layout：
        #   1) 传入的 dexycb_root 已经是 raw/（含 calibration/）
        #   2) 传入的 dexycb_root 是外层（<...>/DexYCB），需要再下钻到 raw/
        if (self.dexycb_root / "calibration").is_dir():
            self.models_root = str(self.dexycb_root / "models")
            self.calibration_root = str(self.dexycb_root / "calibration")
        else:
            self.models_root = str(self.dexycb_root / "raw" / "models")
            self.calibration_root = str(self.dexycb_root / "raw" / "calibration")

        # ----- 加载 MANO 右手模型（所有 DexYCB 被试都是右手序列）-----
        print(f"[Preprocessor] Loading MANO right hand (DexYCB PCA, flat_hand_mean=False)...")
        self.mano_r = MANO(
            MANO_MODEL_DIR,
            is_rhand=True,
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=False,
        ).to(self.device)
        # smplx 在请求全部 45 个分量时会禁用内部 PCA 展开；
        # DexYCB pose_m 仍存储 PCA 系数，因此在 MANO forward 前显式展开。
        self.right_hand_components = torch.as_tensor(
            np.asarray(self.mano_r.np_hand_components, dtype=np.float32),
            device=self.device,
        )

        # ----- 计算手部语义 + canonical face center + 法向 -----
        self.right_finger_id, self.right_region_id = assign_hand_semantics(
            self.mano_r, is_right=True
        )
        (
            self.right_hand_cano_points,
            self.right_hand_cano_normals,
        ) = compute_canonical_hand_surface(self.mano_r)
        self.right_faces = np.asarray(self.mano_r.faces, dtype=np.int64)
        self.num_hand_points = self.right_faces.shape[0]
        self.right_hand_point_id = np.arange(self.num_hand_points, dtype=np.int32)

        # 物体采样缓存：同一 YCB id 跨 capture 共享 surface sampling
        self._obj_cache: dict[int, dict] = {}
        # extrinsics 缓存：每 (subject, extrinsics_date) 共享
        self._extr_cache: dict[str, dict[str, np.ndarray]] = {}

        print(f"[Preprocessor] Device: {self.device}")
        print(f"[Preprocessor] Preprocess stride: {self.preprocess_stride}")
        print(f"[Preprocessor] MANO batch size: {self.mano_batch_size}")
        print(f"[Preprocessor] NN frame batch size: {self.nn_batch_size}")
        print(f"[Preprocessor] Num obj points: {self.num_obj_points}")
        print(f"[Preprocessor] DexYCB root: {self.dexycb_root}")

    # ------------------------------------------------------------
    # 内部：缓存读取
    # ------------------------------------------------------------

    def _get_extrinsics(self, subject_dir: str, extrinsics_date: str) -> dict[str, np.ndarray]:
        """Reference camera ← camera SE(3) for each camera serial."""
        key = f"{subject_dir}|{extrinsics_date}"
        if key not in self._extr_cache:
            path = op.join(
                self.calibration_root,
                f"extrinsics_{extrinsics_date}",
                "extrinsics.yml",
            )
            if not op.isfile(path):
                raise FileNotFoundError(
                    f"extrinsics yml not found: {path} "
                    f"(subject={subject_dir} date={extrinsics_date})"
                )
            self._extr_cache[key] = parse_extrinsics_yml(path)
        return self._extr_cache[key]

    def _get_obj_sampling(self, ycb_id: int) -> dict:
        """YCB 物体规范空间采样缓存（跨 capture 共享）。"""
        if ycb_id not in self._obj_cache:
            mesh = load_ycb_object_mesh(ycb_id, self.models_root)
            pts, face_idx, bary = sample_mesh_surface(mesh, self.num_obj_points, seed=42)
            mesh.fix_normals()
            vn = mesh.vertex_normals
            # 重心插值求法向
            fv = mesh.faces[face_idx]
            normals = (
                bary[:, 0:1] * vn[fv[:, 0]]
                + bary[:, 1:2] * vn[fv[:, 1]]
                + bary[:, 2:3] * vn[fv[:, 2]]
            )
            norms = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = normals / np.clip(norms, 1e-10, None)
            self._obj_cache[int(ycb_id)] = {
                "mesh": mesh,
                "faces": np.asarray(mesh.faces, dtype=np.int64),
                "points": pts.astype(np.float32),
                "normals": normals.astype(np.float32),
                "point_id": np.arange(self.num_obj_points, dtype=np.int32),
                "face_idx": face_idx,
                "barycentric": bary.astype(np.float32),
            }
            print(
                f"[Preprocessor] Cached YCB id={ycb_id} ({YCB_ID_TO_NAME[int(ycb_id)]}): "
                f"{len(mesh.vertices)} verts, {self.num_obj_points} samples"
            )
        return self._obj_cache[int(ycb_id)]

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------

    def process_sequence(
        self,
        capture_dir: str,
        view_serial: Optional[str] = None,
    ) -> dict:
        """Parse a single DexYCB capture and return the Stage 4 source fields.

        Args:
            capture_dir: 形如 ``<root>/20201022-subject-10/20201022_112447``
                         即 subject_dir/capture_name 的路径
            view_serial: 可选的 reference-camera serial 校验值。
                         ``pose.npz`` 只提供 reference-camera 坐标的聚合姿态，
                         因此指定非 reference serial 会直接报错。
        """
        capture_path = Path(capture_dir).resolve()
        subject_dir = capture_path.parent
        capture_name = capture_path.name
        subject_dir_name = subject_dir.name  # e.g. "20201022-subject-10"

        # 从 subject_dir 名称解析 subject_id（"20201022-subject-10" → "subject-10"）
        if "-" in subject_dir_name and "subject-" in subject_dir_name:
            subject_id = subject_dir_name.split("subject-", 1)[1]
            subject_id = f"subject-{subject_id}"
        else:
            subject_id = subject_dir_name

        # ----- meta.yml -----
        meta = yaml.safe_load((capture_path / "meta.yml").read_text(encoding="utf-8"))
        num_frames = int(meta["num_frames"])
        ycb_ids = [int(x) for x in meta["ycb_ids"]]
        grasp_idx = int(meta["ycb_grasp_ind"])
        grasp_obj_id = ycb_ids[grasp_idx]
        grasp_obj_name = YCB_ID_TO_NAME[grasp_obj_id]
        mano_side = str(meta["mano_sides"][0])  # "right"/"left" depends on subject
        if mano_side != "right":
            # 当前只支持 right-hand MANO。Stage4 入口会预先过滤左手；
            # 直接调用 adapter 时仍 fail-fast。
            raise NotImplementedError(
                f"DexYCB subject has mano_side={mano_side!r}; only 'right' is wired up in this adapter"
            )
        mano_calib_name = str(meta["mano_calib"][0])  # e.g. "20201022_105224_subject-10_right"
        serials = [str(s) for s in meta["serials"]]
        extrinsics_date = str(meta["extrinsics"])

        # sequence-level pose.npz 使用 extrinsic=单位阵的 reference camera 坐标。
        extr_dict = self._get_extrinsics(str(subject_dir), extrinsics_date)
        reference_serial = find_reference_camera(extr_dict, serials)
        if view_serial is not None and str(view_serial) != reference_serial:
            raise ValueError(
                f"pose.npz is expressed in reference camera {reference_serial!r}; "
                f"requested non-reference view_serial={view_serial!r}"
            )
        view_serial = reference_serial

        # ----- pose.npz (per-frame, world frame) -----
        pose = np.load(capture_path / "pose.npz")
        pose_y = np.asarray(pose["pose_y"], dtype=np.float32)  # (T, N_obj, 7) world
        pose_m = np.asarray(pose["pose_m"], dtype=np.float32)  # (T, 1, 51)
        if pose_y.shape[0] != num_frames or pose_m.shape[0] != num_frames:
            raise ValueError(
                f"pose.npz frames mismatch: meta={num_frames} "
                f"pose_y={pose_y.shape[0]} pose_m={pose_m.shape[0]}"
            )

        # ----- 帧降采样 -----
        raw_frame_id = np.arange(0, num_frames, self.preprocess_stride, dtype=np.int32)
        if self.max_frames is not None:
            raw_frame_id = raw_frame_id[: int(self.max_frames)]
        if raw_frame_id.size == 0:
            raise ValueError("No frames selected after applying preprocess_stride / max_frames")
        pose_y = pose_y[raw_frame_id]            # (T, N_obj, 7)
        pose_m = pose_m[raw_frame_id]            # (T, 1, 51)
        # DexYCB 用全 0 pose_m 表示当前帧没有有效手部标注。当前
        # subject-10 右手序列的无效帧只出现在序列开头；裁掉它们不会
        # 改变保留帧的时间间隔。如果以后数据出现内部断裂则 fail-fast。
        raw_frame_id, pose_y, pose_m = trim_invalid_mano_prefix(
            raw_frame_id, pose_y, pose_m
        )
        T = int(raw_frame_id.shape[0])

        # ----- MANO betas（每被试一个，在 calibration 目录） -----
        mano_yaml = op.join(
            self.calibration_root, f"mano_{mano_calib_name}", "mano.yml"
        )
        if not op.isfile(mano_yaml):
            raise FileNotFoundError(f"MANO betas not found: {mano_yaml}")
        betas = np.asarray(
            yaml.safe_load(open(mano_yaml, "r"))["betas"], dtype=np.float32
        )  # (10,)
        if betas.shape[0] != 10:
            raise ValueError(
                f"Expected 10 MANO betas, got {betas.shape[0]} in {mano_yaml}"
            )
        betas_t = torch.from_numpy(betas).to(self.device).expand(T, 10)  # (T, 10)

        # ----- MANO 参数拆解（已在 reference/master-camera frame）-----
        # pose_m[:, 0, 0:3]   = wrist rotation (axis-angle)
        # pose_m[:, 0, 3:48]  = 45 MANO PCA coefficients
        # pose_m[:, 0, 48:51] = wrist translation
        wrist_rot_aa = torch.from_numpy(pose_m[:, 0, 0:3]).to(self.device)             # (T, 3)
        finger_pose_coeff = torch.from_numpy(pose_m[:, 0, 3:48]).to(self.device)        # (T, 45)
        finger_pose_aa = expand_mano_pca_pose(
            finger_pose_coeff, self.right_hand_components
        )                                                                                # (T, 45)
        wrist_trans_world = torch.from_numpy(pose_m[:, 0, 48:51]).to(self.device)       # (T, 3)

        # ----- MANO Forward（分批）-----
        all_verts, all_joints = [], []
        for i in range(0, T, self.mano_batch_size):
            end = min(i + self.mano_batch_size, T)
            out = self.mano_r(
                global_orient=wrist_rot_aa[i:end],
                hand_pose=finger_pose_aa[i:end],
                betas=betas_t[i:end],
                transl=wrist_trans_world[i:end],
            )
            all_verts.append(out.vertices.detach().cpu().numpy())
            all_joints.append(out.joints.detach().cpu().numpy())
        right_verts = np.concatenate(all_verts, axis=0).astype(np.float32)    # (T, 778, 3)
        right_joints = np.concatenate(all_joints, axis=0).astype(np.float32)  # (T, 16, 3)

        # ----- face center 采样 1538 点 -----
        right_face_pts = right_verts[:, self.right_faces].mean(axis=2)  # (T, 1538, 3)

        # ----- face normals（用 posed mesh 的 per-face normal）-----
        triangles = right_verts[:, self.right_faces]    # (T, F, 3, 3)
        e01 = triangles[:, :, 1] - triangles[:, :, 0]
        e02 = triangles[:, :, 2] - triangles[:, :, 0]
        normals = np.cross(e01, e02)
        n_norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        right_normals = (normals / np.clip(n_norms, 1e-10, None)).astype(np.float32)  # (T, 1538, 3)

        # ----- 物体：只取 grasped object 的 6D pose -----
        grasp_pose_y = pose_y[:, grasp_idx, :]  # (T, 7)  quat(4) + trans(3)  world frame
        obj_quat = grasp_pose_y[:, 0:4]         # (T, 4)  x y z w
        obj_trans = grasp_pose_y[:, 4:7]        # (T, 3)  meters (reference frame)
        R_obj = quat_to_rotmat(obj_quat)        # (T, 3, 3)
        obj_root_pose = build_SE3(R_obj, obj_trans)  # (T, 4, 4)

        obj_cache = self._get_obj_sampling(grasp_obj_id)
        obj_canon_points = obj_cache["points"]    # (No, 3)
        obj_canon_normals = obj_cache["normals"]  # (No, 3)
        obj_points_world = transform_points(
            np.broadcast_to(obj_canon_points[None], (T, self.num_obj_points, 3)).copy(),
            obj_root_pose,
        ).astype(np.float32)  # (T, No, 3)
        obj_normals_world = transform_normals(
            np.broadcast_to(obj_canon_normals[None], (T, self.num_obj_points, 3)).copy(),
            R_obj,
        ).astype(np.float32)  # (T, No, 3)

        # ----- 手部根节点 SE(3) pose -----
        # wrist 位置 = MANO forward 输出的 joint 0 (world)
        # wrist 朝向 = global_orient (axis-angle) → 旋转矩阵
        from process.ARCTIC.raw import axis_angle_to_rotmat
        R_hand_r = axis_angle_to_rotmat(wrist_rot_aa).detach().cpu().numpy()
        right_hand_root_pose = build_SE3(
            R_hand_r, right_joints[:, 0, :].astype(np.float32)
        )  # (T, 4, 4)

        seq_id = f"{subject_id}/{capture_name}"
        return {
            "seq_id": seq_id,
            "dataset_name": "dexycb",
            "subject_id": subject_id,
            "seq_name": capture_name,
            "object_name": grasp_obj_name,
            "reference_camera_serial": reference_serial,
            "raw_frame_id": raw_frame_id.astype(np.int32),
            # ----- object -----
            "obj_points_world": obj_points_world,
            "obj_normals_world": obj_normals_world,
            "obj_point_id": obj_cache["point_id"],
            "obj_root_pose": obj_root_pose.astype(np.float32),
            # ----- right hand -----
            "right_hand_points_world": right_face_pts.astype(np.float32),
            "right_hand_normals_world": right_normals,
            "right_hand_point_id": self.right_hand_point_id,
            "right_hand_cano_points": self.right_hand_cano_points.astype(np.float32),
            "right_hand_finger_id": self.right_finger_id,
            "right_hand_region_id": self.right_region_id,
            "right_hand_root_pose": right_hand_root_pose,
            # ----- left hand 不存在（单手 sequence） -----
            # Stage 4 要求 'right_hand_root_pose' 等键存在；left 缺失时
            # stage4_cm.py 不调用 build_hand_sequence(side="left")，所以无需
            # 填 left_* 键。如果以后想跑双被试（没有），再补齐。
        }
