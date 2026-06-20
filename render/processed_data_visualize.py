#!/usr/bin/env python3
"""
Ref2Dex Processed Dataset Visualizer (Open3D)

Usage:
    python visualize.py [--seq s01/box_grab_01]
                        [--data_root ../processed_data/arctic]
                        [--arctic_root ../dataset/arctic/data/arctic_data/data]

Keyboard controls:
    Left / H        : Previous frame
    Right / L       : Next frame
    Up / K          : Jump 10 frames forward
    Home/End        : First/Last frame
    Space           : Play/Pause
    N               : Toggle normals
    F               : Toggle flow arrows
    J               : Toggle joint markers
    B               : Toggle bones (MANO skeleton)
    W               : Toggle object wireframe
    O               : Toggle object point cloud
    1               : Toggle right hand
    2               : Toggle left hand
    3 / M           : Cycle hand display: GT only → opt only → both
    T               : Toggle trajectory lines
    S               : Next sequence
    R               : Reset camera
    H               : Toggle help overlay
    Escape/Q        : Quit
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import os.path as op
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

# ============================================================
# Paths
# ============================================================
THIS_DIR = op.dirname(op.abspath(__file__))
REF2DEX_ROOT = op.abspath(op.join(THIS_DIR, ".."))
ASSETS_ROOT = op.join(REF2DEX_ROOT, "assets")
DATA_ROOT_DEFAULT = op.join(REF2DEX_ROOT, "processed_data", "arctic")
ARCTIC_ROOT_DEFAULT = op.join(THIS_DIR, "..", "dataset", "arctic", "data", "arctic_data", "data")
OUTPUTS_ROOT = op.join(REF2DEX_ROOT, "outputs", "mano_fit")
LEGACY_OPTI_ROOT = op.join(REF2DEX_ROOT, "mano_opti_data")

# ============================================================
# Colors
# ============================================================
# 旧的 per-finger 着色（palm/thumb/index/middle/ring/pinky 6 色）已删除。
# 现在原始手和优化手都用统一颜色（COLOR_ORIG_HAND / COLOR_OPTI_HAND），
# 方便直接对比两个手整体的几何差异。

COLOR_OBJ_POINTS = (0.40, 0.80, 1.00)  # light blue
COLOR_OBJ_WIREFRAME = (0.70, 0.90, 1.00)  # lighter blue
COLOR_NORMALS = (1.00, 0.40, 0.40)  # red
COLOR_FLOW = (0.20, 1.00, 0.20)  # green
COLOR_JOINTS = (1.00, 0.20, 0.20)  # red
COLOR_TRAJECTORY_WRIST = (1.00, 0.60, 0.20)  # orange
COLOR_TRAJECTORY_OBJ = (0.20, 0.60, 1.00)  # blue
COLOR_BACKGROUND = (0.15, 0.15, 0.15)  # dark gray
# Hand colors (统一色，不再按手指分色)
# - 原始手（preprocess 输出 / ARCTIC GT）：纯绿色
# - 优化后手（CPF 优化输出）：纯红色
# 二者叠加时容易看出 CPF 调整了几何
COLOR_ORIG_HAND = (0.30, 1.00, 0.30)  # 绿色：原始手
COLOR_OPTI_HAND = (1.00, 0.25, 0.25)  # 红色：CPF 优化手

# MANO bone chains (joint indices)
MANO_BONE_CHAINS = [
    [0, 1, 2, 3, 4],      # thumb
    [0, 5, 6, 7, 8],      # index
    [0, 9, 10, 11, 12],   # middle
    [0, 13, 14, 15, 16],  # ring
    [0, 17, 18, 19, 20],  # pinky
]

# smplx MANO joint order: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
# But preprocess uses 16 joints from MANO without PCA
# Actually, the processed data has 21 joints (standard MANO output)
MANO_JOINT_FINGER = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]


def _load_dataset_meta(data_root: str) -> dict:
    meta_path = op.join(data_root, "meta.json")
    if not op.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Viewer] Failed to read meta.json from {meta_path}: {e}")
        return {}


def _resolve_dataset_name(meta: dict, data_root: str) -> str:
    dataset_name = str(meta.get("dataset_name", "")).strip().lower()
    if dataset_name:
        return dataset_name
    root_name = op.basename(op.abspath(data_root)).lower()
    for known_name in ("arctic", "grab"):
        if known_name in root_name:
            return known_name
    return "arctic"


def _resolve_default_opti_root(dataset_name: str) -> str:
    candidates = [
        op.join(OUTPUTS_ROOT, f"{dataset_name}_smplx_cpf"),
        op.join(OUTPUTS_ROOT, f"{dataset_name}_cpf"),
        op.join(LEGACY_OPTI_ROOT, f"{dataset_name}_smplx_cpf"),
        op.join(LEGACY_OPTI_ROOT, dataset_name),
    ]
    for candidate in candidates:
        if op.exists(candidate):
            return candidate
    return candidates[0]


def _iter_object_asset_roots(meta: dict, dataset_name: str, assets_root: str):
    roots = []
    object_asset_root = meta.get("object_asset_root")
    if object_asset_root:
        roots.append(object_asset_root)
    roots.extend([
        op.join(assets_root, dataset_name, "objects"),
        op.join(assets_root, "objects"),
    ])
    seen = set()
    for root in roots:
        abs_root = op.abspath(root)
        if abs_root in seen:
            continue
        seen.add(abs_root)
        yield abs_root


def _resolve_object_asset_dir(meta: dict, dataset_name: str, assets_root: str, obj_name: str):
    for root in _iter_object_asset_roots(meta, dataset_name, assets_root):
        candidate = op.join(root, obj_name)
        if op.exists(op.join(candidate, "mesh.obj")):
            return candidate
    return None


def make_lineset(points, edges, color, width=1.0):
    """Create an open3d LineSet from points and edges."""
    if len(edges) == 0:
        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        return lines
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    colors = np.tile(np.asarray(color, dtype=np.float64), (len(edges), 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_pointcloud(points, colors=None):
    """Create an open3d PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def make_sphere(center, radius=0.008, color=(1, 0, 0)):
    """Create a small sphere mesh at a position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere


def make_arrow_batch(origins, vectors, color=(0, 1, 0), scale=1.0, density=4):
    """
    Create line segments representing arrows from origins in vector directions.
    Returns a LineSet with one segment per arrow.
    """
    if len(origins) == 0:
        return make_lineset(np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), color)
    
    origins = np.asarray(origins, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    tips = origins + vectors * scale
    
    N = len(origins)
    all_pts = np.zeros((N * 2, 3), dtype=np.float64)
    all_pts[0::2] = origins
    all_pts[1::2] = tips
    
    edges = np.arange(N * 2, dtype=np.int32).reshape(-1, 2)
    
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    colors_arr = np.tile(np.asarray(color, dtype=np.float64), (N, 1))
    ls.colors = o3d.utility.Vector3dVector(colors_arr)
    return ls


# ============================================================
# Helpers: rotation
# ============================================================

def axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle to (..., 3, 3) rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(axis_angle + 1e-8, axis=-1, keepdims=True)
    axis = axis_angle / angle
    cos = np.cos(angle)
    sin = np.sin(angle)
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    row1 = np.concatenate([np.zeros_like(x), -z, y], axis=-1)
    row2 = np.concatenate([z, np.zeros_like(y), -x], axis=-1)
    row3 = np.concatenate([-y, x, np.zeros_like(z)], axis=-1)
    K = np.stack([row1, row2, row3], axis=-2)
    I = np.eye(3).reshape(1, 3, 3)
    R = I + sin[..., None] * K + (1 - cos[..., None]) * (K @ K)
    return R


# ============================================================
# Main Viewer
# ============================================================

class Ref2DexViewer:
    """Open3D-based visualizer for Ref2Dex processed data."""

    def __init__(self, args):
        self.args = args
        self.data_root = op.abspath(args.data_root)
        self.meta = _load_dataset_meta(self.data_root)
        self.dataset_name = _resolve_dataset_name(self.meta, self.data_root)
        self.arctic_root = op.abspath(args.arctic_root) if args.arctic_root else None
        self.assets_root = op.abspath(args.assets_root) if args.assets_root else ASSETS_ROOT
        self.opti_root = (
            op.abspath(args.opti_root)
            if args.opti_root else _resolve_default_opti_root(self.dataset_name)
        )
        print(
            f"[Viewer] dataset={self.dataset_name} "
            f"assets_root={self.assets_root} opti_root={self.opti_root}"
        )

        # State
        self.seq_idx = 0
        self.frame_idx = 0
        self.is_playing = False
        self.last_step_time = 0.0
        self.autoplay_hz = args.fps

        # Toggle flags
        self.show_normals = args.normals
        self.show_flow = args.flow
        self.show_joints = args.joints
        self.show_bones = args.bones
        self.show_wireframe = args.wireframe
        self.show_object_pc = True
        self.show_right_hand = True
        self.show_left_hand = True
        self.show_trajectory = args.trajectory
        self.show_help = False
        # MANO CPF optimized hand overlay
        # 显示模式: 0 = GT only, 1 = opt only, 2 = both (GT=green, opt=red)
        # 启动模式: --show-opti-hand → 2, 默认 0
        self.show_mode = 2 if args.show_opti_hand else 0
        self.show_opti_hand = bool(self.show_mode)  # 兼容旧逻辑
        self.opti_right = None   # dict-style payload for right-hand pkl (or None)
        self.opti_left = None    # ditto for left-hand pkl
        # open3d 0.19 没有 set_visible → 用空点云 hide
        self._empty_pc = np.zeros((0, 3), dtype=np.float64)
        self._opti_was_missing = False  # for de-duped "current frame not in pkl" warnings

        # Geometry stride (subsample for performance)
        self.normal_stride = args.normal_stride
        self.flow_stride = args.flow_stride
        self.flow_scale = args.flow_scale

        # Get sorted sequence list
        all_npz = sorted(glob.glob(op.join(self.data_root, "*/*.npz")))
        if len(all_npz) == 0:
            print(f"[ERROR] No .npz files found in {self.data_root}")
            sys.exit(1)
        self.seq_paths = all_npz
        print(f"[Viewer] Found {len(self.seq_paths)} sequences in {self.data_root}")

        # Override sequence if specified
        if args.seq is not None:
            for i, p in enumerate(self.seq_paths):
                if args.seq in p:
                    self.seq_idx = i
                    break
            else:
                print(f"[Warning] Sequence '{args.seq}' not found, using first.")

        # Data containers (filled by load_sequence)
        self.data = None          # dict with all npz fields
        self.T = 0                # number of frames
        self.obj_name = None
        self.canonical_obj_verts = None
        self.canonical_obj_faces = None
        self.obj_parts = None     # parts.json bool array
        self.arti_angles = None   # (T,) articulation angles (if arctic_root available)

        # Open3D geometries (created once, updated per frame)
        self.geom = {}            # name -> (geometry, visible_by_default) 
        self.added_names = set()  # names of geometries currently added to vis
        self.vis = None
        self.help_panel = None    # text overlay

        # Track if trajectory was created (one-time)
        self._trajectory_created = False

        # Load first sequence & create window
        self._load_sequence(self.seq_idx)
        self._create_window()
        self._create_geometries()

        # If showing optimized hand, jump to the first frame that has opti data
        if self.show_opti_hand:
            self._goto_first_opti_frame()

        self._update_frame()

        # Force-reset view so that the camera frames the actual data
        # (otherwise it looks at the empty (0,0,0) bounding box of uninitialized
        # point clouds, making the scene appear blank)
        try:
            self.vis.reset_view_point(True)
        except Exception:
            pass
        # Also nudge the camera a bit so it isn't orthogonal
        view_ctl = self.vis.get_view_control()
        try:
            view_ctl.set_zoom(0.7)
        except Exception:
            pass

        # Set a sensible default camera based on the data bbox
        try:
            self._set_initial_camera()
        except Exception as e:
            print(f"[Viewer] Initial camera set failed: {e}")

        print(self._help_text())
        print(f"[Viewer] Ready. Sequence [{self.seq_idx+1}/{len(self.seq_paths)}] "
              f"{self.data['seq_id']} | Frames: {self.T}")

    # ----------------------------------------------------------
    # Sequence loading
    # ----------------------------------------------------------

    def _load_sequence(self, seq_idx: int):
        """Load a processed sequence and its corresponding assets."""
        seq_path = self.seq_paths[seq_idx]
        self.data = dict(np.load(seq_path, allow_pickle=True))
        self.T = self.data['frame_id'].shape[0]
        self.frame_idx = 0
        self.is_playing = False

        # Determine object name from seq_id
        seq_id = str(self.data['seq_id'])
        # e.g. "s01/box_grab_01" -> "box"
        obj_name = seq_id.split("/")[1].split("_")[0]
        self.obj_name = obj_name

        # Load canonical object mesh from dataset-specific asset root.
        obj_asset_dir = _resolve_object_asset_dir(
            self.meta, self.dataset_name, self.assets_root, obj_name
        )
        mesh_obj_path = op.join(obj_asset_dir, "mesh.obj") if obj_asset_dir else None
        if mesh_obj_path and op.exists(mesh_obj_path):
            mesh = trimesh.load(mesh_obj_path, process=False)
            self.canonical_obj_verts = np.asarray(mesh.vertices, dtype=np.float64)
            self.canonical_obj_faces = np.asarray(mesh.faces, dtype=np.int32)
        else:
            print(f"[Warning] Object mesh not found for {obj_name} under dataset={self.dataset_name}")
            self.canonical_obj_verts = np.zeros((0, 3), dtype=np.float64)
            self.canonical_obj_faces = np.zeros((0, 3), dtype=np.int32)

        # Load parts.json for articulated objects
        parts_path = op.join(obj_asset_dir, "parts.json") if obj_asset_dir else ""
        if obj_asset_dir and op.exists(parts_path):
            with open(parts_path) as f:
                parts_raw = json.load(f)
            self.obj_parts = np.array(parts_raw, dtype=bool)
        else:
            self.obj_parts = None

        # Load articulation angles from ARCTIC raw data (if available)
        self.arti_angles = None
        if self.dataset_name == "arctic" and self.arctic_root is not None:
            raw_seqs_dir = op.join(self.arctic_root, "raw_seqs")
            subject = seq_id.split("/")[0]
            seq_name = seq_id.split("/")[1]
            obj_npy_path = op.join(raw_seqs_dir, subject, f"{seq_name}.object.npy")
            if op.exists(obj_npy_path):
                obj_params = np.load(obj_npy_path)  # (T, 7)
                self.arti_angles = obj_params[:, 0].astype(np.float64)  # (T,)
                print(f"[Viewer] Loaded articulation angles for {seq_id}")

        # Compute obj mesh edges from faces (for wireframe)
        if len(self.canonical_obj_faces) > 0:
            edges = np.concatenate([
                self.canonical_obj_faces[:, [0, 1]],
                self.canonical_obj_faces[:, [1, 2]],
                self.canonical_obj_faces[:, [2, 0]],
            ], axis=0)
            edges = np.sort(edges, axis=1)
            self.obj_edges = np.unique(edges, axis=0)
        else:
            self.obj_edges = np.zeros((0, 2), dtype=np.int32)

        # Face data for the wireframe
        self.obj_faces = self.canonical_obj_faces.copy()

        # Reset trajectory flag
        self._trajectory_created = False

        # Load MANO CPF optimization result for this sequence (if available)
        self.opti_right = self._load_opti_data("right")
        self.opti_left = self._load_opti_data("left")
        self._opti_was_missing = False

        print(f"[Viewer] Loaded: {seq_id} | Object: {obj_name} | Frames: {self.T}")

    def _get_posed_obj_verts(self, frame_idx: int):
        """
        Get posed object mesh vertices for a given frame.
        Applies articulation (if available) + global transform.
        """
        verts = self.canonical_obj_verts.copy()
        
        # Apply articulation (rotate top part around z-axis)
        if self.arti_angles is not None and self.obj_parts is not None:
            angle = self.arti_angles[frame_idx]
            c = np.cos(angle)
            s = np.sin(angle)
            R_arti = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
            top_mask = self.obj_parts
            if np.any(top_mask):
                verts[top_mask] = (R_arti @ verts[top_mask].T).T
        
        # Apply global transform
        root_pose = self.data['obj_root_pose'][frame_idx]  # (4, 4)
        R = root_pose[:3, :3].astype(np.float64)
        t = root_pose[:3, 3].astype(np.float64)
        verts = (R @ verts.T).T + t
        return verts

    def _get_posed_obj_edges(self, frame_idx: int):
        """Get posed wireframe edges for the object at a given frame."""
        verts = self._get_posed_obj_verts(frame_idx)
        return verts, self.obj_edges

    def _load_opti_data(self, side: str):
        """Load MANO CPF optimization result for current sequence + side.

        Looks for `<opti_root>/<subject>/<seq_name>_<side>.pkl` (the layout used
        by the current `outputs/mano_fit/*` tree and older optimization outputs).
        Returns the unpickled payload
        dict, or ``None`` if the file is missing / fails to load.

        The payload is expected to contain at least:
            - ``frame_ids`` (Tpkl,) int
            - ``opt_hand_face_centers`` (Tpkl, 1538, 3) float
            - ``penetration_depth`` (Tpkl,) float
            - ``contact_ratio`` (Tpkl,) float
        """
        seq_id = str(self.data['seq_id'])
        subject_id = seq_id.split("/")[0]
        seq_name = seq_id.split("/")[1]
        pkl_path = op.join(self.opti_root, subject_id, f"{seq_name}_{side}.pkl")
        if not op.exists(pkl_path):
            print(f"[Viewer] No MANO opti pkl for {side} hand: {pkl_path}")
            return None
        try:
            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            print(f"[Viewer] Failed to load {pkl_path}: {e}")
            return None
        n = int(payload['opt_hand_face_centers'].shape[0])
        fids = payload['frame_ids']
        print(f"[Viewer] Loaded MANO opti pkl: {pkl_path} "
              f"({n} frames, frame_ids={fids[:3].tolist()}..{fids[-1].tolist() if n else []})")
        return payload

    # ----------------------------------------------------------
    # Open3D window & geometries
    # ----------------------------------------------------------

    def _create_window(self):
        """Create the Open3D visualization window and register callbacks."""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name=f"Ref2Dex Visualizer - {self.data['seq_id']}",
            width=self.args.width,
            height=self.args.height,
            left=50,
            top=50,
            visible=True,
        )

        # Set rendering options
        opt = self.vis.get_render_option()
        opt.background_color = np.array(COLOR_BACKGROUND)
        opt.point_size = 3.0
        opt.line_width = 1.5
        opt.show_coordinate_frame = True

        print(f"[Viewer] Window created on DISPLAY={os.environ.get('DISPLAY', '?')}")
        print(f"[Viewer] Window size: {self.args.width}x{self.args.height}")

        # Register key callbacks
        # Arrow keys (RIGHT/LEFT) are unreliable across Open3D versions; use ASCII alternatives
        # 0/9 = prev/next seq, 7/8 = prev/next frame (vim-style for safety)
        # We try multiple key codes for arrows as fallback
        try:
            self.vis.register_key_callback(ord('L'), lambda v: self._next_frame(1))   # L = next
            self.vis.register_key_callback(ord('H'), lambda v: self._prev_frame(1))   # H = prev
            self.vis.register_key_callback(ord('K'), lambda v: self._next_frame(10))  # K = +10
            self.vis.register_key_callback(ord('J'), lambda v: self._prev_frame(10))  # J = -10
        except Exception as e:
            print(f"[Viewer] ASCII key register failed: {e}")
        # Arrow key fallbacks (multiple code sets)
        for kc in (262, 65363, 0xff53):
            try: self.vis.register_key_callback(kc, lambda v: self._next_frame(1))
            except: pass
        for kc in (263, 65361, 0xff51):
            try: self.vis.register_key_callback(kc, lambda v: self._prev_frame(1))
            except: pass
        self.vis.register_key_callback(268, lambda v: self._goto_frame(0))     # HOME
        self.vis.register_key_callback(269, lambda v: self._goto_frame(-1))    # END
        self.vis.register_key_callback(32, lambda v: self._toggle_play())      # SPACE
        self.vis.register_key_callback(ord('N'), lambda v: self._toggle_flag('normals'))
        self.vis.register_key_callback(ord('F'), lambda v: self._toggle_flag('flow'))
        self.vis.register_key_callback(ord('J'), lambda v: self._toggle_flag('joints'))
        self.vis.register_key_callback(ord('B'), lambda v: self._toggle_flag('bones'))
        self.vis.register_key_callback(ord('W'), lambda v: self._toggle_flag('wireframe'))
        self.vis.register_key_callback(ord('O'), lambda v: self._toggle_flag('object_pc'))
        self.vis.register_key_callback(ord('1'), lambda v: self._toggle_flag('right_hand'))
        self.vis.register_key_callback(ord('2'), lambda v: self._toggle_flag('left_hand'))
        self.vis.register_key_callback(ord('3'), lambda v: self._toggle_flag('opti_hand'))
        self.vis.register_key_callback(ord('M'), lambda v: self._toggle_flag('opti_hand'))  # alias
        self.vis.register_key_callback(ord('T'), lambda v: self._toggle_flag('trajectory'))
        self.vis.register_key_callback(ord('S'), lambda v: self._next_sequence())
        self.vis.register_key_callback(ord('R'), lambda v: self._reset_camera())
        self.vis.register_key_callback(ord('H'), lambda v: self._toggle_flag('help'))
        self.vis.register_key_callback(256, lambda v: self._quit())            # ESCAPE
        self.vis.register_key_callback(ord('Q'), lambda v: self._quit())       # Q

    def _create_geometries(self):
        """Create all geometric objects. Added to vis on first use."""
        T = self.T
        data = self.data

        # Number of hand points = 1538 (face centers)
        num_hand_pts = data['right_hand_points'].shape[1]
        num_obj_pts = data['obj_points'].shape[1]

        # ---- Right Hand PointCloud ----
        # 原始手：统一绿色（COLOR_ORIG_HAND），不再按 finger_id 着色
        colors_r = np.tile(np.array(COLOR_ORIG_HAND, dtype=np.float64), (num_hand_pts, 1))
        self.geom['right_hand'] = (
            make_pointcloud(np.zeros((num_hand_pts, 3)), colors_r),
            True,
        )

        # ---- Left Hand PointCloud ----
        # 原始手：统一绿色（COLOR_ORIG_HAND），不再按 finger_id 着色
        colors_l = np.tile(np.array(COLOR_ORIG_HAND, dtype=np.float64), (num_hand_pts, 1))
        self.geom['left_hand'] = (
            make_pointcloud(np.zeros((num_hand_pts, 3)), colors_l),
            True,
        )

        # ---- Object PointCloud ----
        self.geom['object_pc'] = (
            make_pointcloud(np.zeros((num_obj_pts, 3)),
                          np.tile(COLOR_OBJ_POINTS, (num_obj_pts, 1))),
            True,
        )

        # ---- Object Wireframe ----
        self.geom['wireframe'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_OBJ_WIREFRAME),
            self.show_wireframe,
        )

        # ---- Normals (Right Hand) ----
        self.geom['normals_rh'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_NORMALS),
            self.show_normals,
        )
        # ---- Normals (Left Hand) ----
        self.geom['normals_lh'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_NORMALS),
            self.show_normals,
        )
        # ---- Normals (Object) ----
        self.geom['normals_obj'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_NORMALS),
            self.show_normals,
        )

        # ---- Flow Arrows ----
        self.geom['flow_rh'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_FLOW),
            self.show_flow,
        )
        self.geom['flow_lh'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_FLOW),
            self.show_flow,
        )
        self.geom['flow_obj'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_FLOW),
            self.show_flow,
        )

        # ---- Joint Markers (as PointCloud for efficient per-frame update) ----
        self.geom['joints'] = (
            make_pointcloud(np.zeros((12, 3))),
            self.show_joints,
        )

        # ---- Bones (LineSet) ----
        self.geom['bones'] = (
            make_lineset(np.zeros((21, 3)), np.zeros((0, 2), dtype=np.int32), (1, 1, 1)),
            self.show_bones,
        )

        # ---- Trajectory Lines ----
        self.geom['trajectory'] = (
            make_lineset(np.zeros((1, 3)), np.zeros((0, 2), dtype=np.int32), COLOR_TRAJECTORY_WRIST),
            self.show_trajectory,
        )

        # ---- Optimized (MANO CPF) Hand PointCloud (per side) ----
        # Default visible only when --show-opti-hand is set AND the corresponding
        # pkl was successfully loaded for the current sequence.
        opti_color_arr = np.tile(np.array(COLOR_OPTI_HAND, dtype=np.float64), (num_hand_pts, 1))
        self.geom['opti_hand_r'] = (
            make_pointcloud(np.zeros((num_hand_pts, 3)), opti_color_arr.copy()),
            bool(self.show_opti_hand and self.opti_right is not None),
        )
        self.geom['opti_hand_l'] = (
            make_pointcloud(np.zeros((num_hand_pts, 3)), opti_color_arr.copy()),
            bool(self.show_opti_hand and self.opti_left is not None),
        )

        # Add all default-visible geometries
        for name in ['right_hand', 'left_hand', 'object_pc']:
            geom, _ = self.geom[name]
            self.vis.add_geometry(geom)
            self.added_names.add(name)

        # Add toggleable ones based on initial flags
        for name, (geom, default_vis) in self.geom.items():
            if default_vis and name not in self.added_names:
                self.vis.add_geometry(geom)
                self.added_names.add(name)

    def _add_geom(self, name):
        """Add a geometry to the visualizer by name."""
        if name in self.added_names:
            return
        geom, _ = self.geom[name]
        self.vis.add_geometry(geom)
        self.added_names.add(name)

    def _remove_geom(self, name):
        """Remove a geometry from the visualizer by name."""
        if name not in self.added_names:
            return
        geom, _ = self.geom[name]
        self.vis.remove_geometry(geom)
        self.added_names.discard(name)

    def _update_gt_visibility(self):
        """GT 手显隐 = show_side_attr AND show_mode != 1
        show_mode=0 → GT on
        show_mode=1 → GT off（让 opt 单独显示）
        show_mode=2 → GT on
        """
        for side, attr in [('right', 'show_right_hand'), ('left', 'show_left_hand')]:
            show = getattr(self, attr) and self.show_mode != 1
            if show:
                self._add_geom(f'{side}_hand')
            else:
                self._remove_geom(f'{side}_hand')

    # ----------------------------------------------------------
    # Frame update
    # ----------------------------------------------------------

    def _update_frame(self):
        """Update all geometry positions/colors for the current frame."""
        f = self.frame_idx
        T = self.T
        data = self.data
        vis = self.vis

        # ---- Right Hand ----
        # show_mode=1 → 用空点云 hide；show_mode=2 → 用 manotorch 自洽 GT 做公平对比
        if self.show_mode == 1:
            rh_pts = self._empty_pc
        elif self.show_mode == 2:
            consistent = self._get_consistent_gt_pts('r', f)
            rh_pts = consistent if consistent is not None else data['right_hand_points'][f]
        else:
            rh_pts = data['right_hand_points'][f]
        geom_rh, _ = self.geom['right_hand']
        geom_rh.points = o3d.utility.Vector3dVector(rh_pts.astype(np.float64))
        # Colors are static (finger_id based), no need to update
        if 'right_hand' in self.added_names:
            vis.update_geometry(geom_rh)

        # ---- Left Hand ----
        if self.show_mode == 1:
            lh_pts = self._empty_pc
        elif self.show_mode == 2:
            consistent = self._get_consistent_gt_pts('l', f)
            lh_pts = consistent if consistent is not None else data['left_hand_points'][f]
        else:
            lh_pts = data['left_hand_points'][f]
        geom_lh, _ = self.geom['left_hand']
        geom_lh.points = o3d.utility.Vector3dVector(lh_pts.astype(np.float64))
        if 'left_hand' in self.added_names:
            vis.update_geometry(geom_lh)

        # ---- Object PointCloud ----
        obj_pts = data['obj_points'][f]
        geom_obj, _ = self.geom['object_pc']
        geom_obj.points = o3d.utility.Vector3dVector(obj_pts.astype(np.float64))
        if 'object_pc' in self.added_names:
            vis.update_geometry(geom_obj)

        # ---- Object Wireframe ----
        if self.show_wireframe:
            posed_verts, edges = self._get_posed_obj_edges(f)
            geom_wf, _ = self.geom['wireframe']
            geom_wf.points = o3d.utility.Vector3dVector(posed_verts.astype(np.float64))
            geom_wf.lines = o3d.utility.Vector2iVector(edges.astype(np.int32))
            wf_colors = np.tile(np.array(COLOR_OBJ_WIREFRAME, dtype=np.float64), (len(edges), 1))
            geom_wf.colors = o3d.utility.Vector3dVector(wf_colors)
            if 'wireframe' in self.added_names:
                vis.update_geometry(geom_wf)

        # ---- Normals ----
        self._update_normals('rh', data['right_hand_points'][f], data['right_hand_normals'][f])
        self._update_normals('lh', data['left_hand_points'][f], data['left_hand_normals'][f])
        self._update_normals('obj', data['obj_points'][f], data['obj_normals'][f])

        # ---- Flow Arrows ----
        num_hand_pts = data['right_hand_points'].shape[1]
        num_obj_pts = data['obj_points'].shape[1]
        self._update_flow('rh', data['right_hand_points'][f], data.get('right_hand_flow', np.zeros((T, num_hand_pts, 3)))[f])
        self._update_flow('lh', data['left_hand_points'][f], data.get('left_hand_flow', np.zeros((T, num_hand_pts, 3)))[f])
        self._update_flow('obj', data['obj_points'][f], data.get('obj_flow', np.zeros((T, num_obj_pts, 3)))[f])

        # ---- Joints (21 sphere meshes merged) ----
        self._update_joints()

        # ---- Bones ----
        self._update_bones()

        # ---- Trajectory (one-time creation) ----
        if self.show_trajectory and not self._trajectory_created:
            self._update_trajectory()
            self._trajectory_created = True

        # ---- Optimized (MANO CPF) Hand Overlay ----
        # show_mode=0 → opt hide（空点云）；1/2 → opt 显示当前帧
        for side in ['r', 'l']:
            opt_geom, _ = self.geom[f'opti_hand_{side}']
            if f'opti_hand_{side}' in self.added_names and self.show_mode == 0:
                opt_geom.points = o3d.utility.Vector3dVector(self._empty_pc)
                vis.update_geometry(opt_geom)
        # mode 1/2 走原路径（_update_opti_hand 内部用 show_opti_hand 决定更新内容）
        self._update_opti_hand('r')
        self._update_opti_hand('l')

    def _update_normals(self, prefix: str, points, normals):
        """Update normal line segments for a given geometry prefix."""
        geom_name = f'normals_{prefix}'
        if not self.show_normals or geom_name not in self.added_names:
            return

        stride = self.normal_stride
        pts = np.asarray(points[::stride], dtype=np.float64)
        nmls = np.asarray(normals[::stride], dtype=np.float64)
        normal_len = 0.01  # 1cm normal length

        N = len(pts)
        if N == 0:
            return

        all_pts = np.zeros((N * 2, 3), dtype=np.float64)
        all_pts[0::2] = pts
        all_pts[1::2] = pts + nmls * normal_len
        edges = np.arange(N * 2, dtype=np.int32).reshape(-1, 2)

        geom, _ = self.geom[geom_name]
        geom.points = o3d.utility.Vector3dVector(all_pts)
        geom.lines = o3d.utility.Vector2iVector(edges)
        colors = np.tile(np.array(COLOR_NORMALS, dtype=np.float64), (N, 1))
        geom.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(geom)

    def _update_flow(self, prefix: str, points, flow):
        """Update flow arrow line segments."""
        geom_name = f'flow_{prefix}'
        if not self.show_flow or geom_name not in self.added_names:
            return

        stride = self.flow_stride
        pts = np.asarray(points[::stride], dtype=np.float64)
        vecs = np.asarray(flow[::stride], dtype=np.float64)

        # Only show non-zero flows
        norms = np.linalg.norm(vecs, axis=-1)
        valid = norms > 1e-6
        pts = pts[valid]
        vecs = vecs[valid]

        if len(pts) == 0:
            return

        scale = self.flow_scale
        tips = pts + vecs * scale
        N = len(pts)
        all_pts = np.zeros((N * 2, 3), dtype=np.float64)
        all_pts[0::2] = pts
        all_pts[1::2] = tips
        edges = np.arange(N * 2, dtype=np.int32).reshape(-1, 2)

        geom, _ = self.geom[geom_name]
        geom.points = o3d.utility.Vector3dVector(all_pts)
        geom.lines = o3d.utility.Vector2iVector(edges)
        colors = np.tile(np.array(COLOR_FLOW, dtype=np.float64), (N, 1))
        geom.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(geom)

    def _update_joints(self):
        """Update joint marker positions using a PointCloud.

        Uses wrist positions from root_pose + fingertip positions
        (dynamically determined by finding face-center points farthest
        from the wrist) for approximate joint visualization.
        """
        if not self.show_joints:
            return

        f = self.frame_idx
        data = self.data

        rh_pose = data['right_hand_root_pose'][f]
        lh_pose = data['left_hand_root_pose'][f]
        wrist_r = rh_pose[:3, 3].astype(np.float64)
        wrist_l = lh_pose[:3, 3].astype(np.float64)

        rh_pts = data['right_hand_points'][f]
        lh_pts = data['left_hand_points'][f]

        # Find 5 fingertips dynamically as the 5 face-centers farthest from
        # the wrist along each finger direction (we use the region_id field
        # to identify the 5 fingers and pick the farthest point per finger).
        tip_indices = self._find_fingertip_indices(data['right_hand_finger_id'],
                                                    rh_pts, wrist_r)
        if not tip_indices:
            # Fallback: use 5 points farthest from wrist
            dists = np.linalg.norm(rh_pts - wrist_r, axis=1)
            tip_indices = [int(i) for i in np.argsort(dists)[-5:][::-1]]
        tip_indices_l = self._find_fingertip_indices(data['left_hand_finger_id'],
                                                     lh_pts, wrist_l)
        if not tip_indices_l:
            dists = np.linalg.norm(lh_pts - wrist_l, axis=1)
            tip_indices_l = [int(i) for i in np.argsort(dists)[-5:][::-1]]

        joint_positions = []
        joint_colors = []

        # Right wrist
        joint_positions.append(wrist_r)
        joint_colors.append((1.0, 0.2, 0.2))
        # Right fingertips
        for idx in tip_indices:
            joint_positions.append(rh_pts[idx].astype(np.float64))
            joint_colors.append((1.0, 0.6, 0.2))

        # Left wrist
        joint_positions.append(wrist_l)
        joint_colors.append((0.2, 0.2, 1.0))
        # Left fingertips
        for idx in tip_indices_l:
            joint_positions.append(lh_pts[idx].astype(np.float64))
            joint_colors.append((0.2, 0.6, 1.0))

        geom, _ = self.geom['joints']
        geom.points = o3d.utility.Vector3dVector(np.array(joint_positions))
        geom.colors = o3d.utility.Vector3dVector(np.array(joint_colors))
        if 'joints' in self.added_names:
            self.vis.update_geometry(geom)

    def _find_fingertip_indices(self, finger_id, points, wrist):
        """For each finger (1-5), find the face-center point farthest from the wrist.

        Returns a list of 5 indices (one per finger, ordered thumb/pinky).
        Returns empty list if no finger_id > 0 is found.
        """
        indices = []
        for f_id in range(1, 6):  # 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
            mask = finger_id == f_id
            if not np.any(mask):
                continue
            finger_pts = points[mask]
            finger_idx = np.where(mask)[0]
            dists = np.linalg.norm(finger_pts - wrist, axis=1)
            local_tip = int(np.argmax(dists))
            indices.append(int(finger_idx[local_tip]))
        return indices

    def _update_bones(self):
        """Update MANO skeleton bone lines (wrist-to-fingertip skeleton)."""
        if not self.show_bones:
            return

        f = self.frame_idx
        data = self.data

        rh_pts = data['right_hand_points'][f]
        lh_pts = data['left_hand_points'][f]
        wrist_r = data['right_hand_root_pose'][f][:3, 3]
        wrist_l = data['left_hand_root_pose'][f][:3, 3]

        # Find fingertip face-center indices dynamically for each finger
        # by using region_id (1 = fingertip) or fall back to farthest from wrist
        tip_indices_r = self._find_fingertip_indices(
            data['right_hand_finger_id'], rh_pts, wrist_r
        )
        tip_indices_l = self._find_fingertip_indices(
            data['left_hand_finger_id'], lh_pts, wrist_l
        )

        bone_pts = []
        bone_edges = []
        bone_colors_list = []
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # Right hand: wrist -> each fingertip
        bone_pts.append(wrist_r.astype(np.float64))
        # 骨头统一绿色（COLOR_ORIG_HAND），不再按手指分色
        for i, idx in enumerate(tip_indices_r):
            bone_pts.append(rh_pts[idx].astype(np.float64))
            bone_edges.append([0, i + 1])
            bone_colors_list.append(COLOR_ORIG_HAND)  # 统一绿色

        offset = len(bone_pts)

        # Left hand: wrist -> each fingertip
        bone_pts.append(wrist_l.astype(np.float64))
        for i, idx in enumerate(tip_indices_l):
            bone_pts.append(lh_pts[idx].astype(np.float64))
            bone_edges.append([offset, offset + i + 1])
            bone_colors_list.append(COLOR_ORIG_HAND)  # 统一绿色

        all_pts = np.array(bone_pts, dtype=np.float64)
        all_edges = np.array(bone_edges, dtype=np.int32)

        geom, _ = self.geom['bones']
        geom.points = o3d.utility.Vector3dVector(all_pts)
        geom.lines = o3d.utility.Vector2iVector(all_edges)
        geom.colors = o3d.utility.Vector3dVector(np.array(bone_colors_list, dtype=np.float64))
        if 'bones' in self.added_names:
            self.vis.update_geometry(geom)

    def _update_trajectory(self):
        """Create trajectory lines (wrist + object center across all frames)."""
        if not self.show_trajectory:
            return
        
        T = self.T
        data = self.data
        
        # Compute object center trajectory
        obj_centers = np.mean(data['obj_points'][:T], axis=1).astype(np.float64)
        
        # Right wrist trajectory (from root pose)
        wrist_r_traj = data['right_hand_root_pose'][:T, :3, 3].astype(np.float64)
        wrist_l_traj = data['left_hand_root_pose'][:T, :3, 3].astype(np.float64)
        
        # Build line set: wrist_R + wrist_L + obj_center
        # Sub-sample every Nth frame for cleaner look
        step = max(1, T // 100)  # ~100 points max
        
        def make_traj_edges(pts):
            N = len(pts)
            pts_flat = pts.reshape(-1, 3)
            edges = np.stack([np.arange(N-1), np.arange(1, N)], axis=1)
            return pts_flat, edges
        
        all_pts_list = []
        all_edges_list = []
        all_colors_list = []
        
        for traj, color in [(wrist_r_traj[::step], COLOR_TRAJECTORY_WRIST),
                             (wrist_l_traj[::step], (0.7, 0.5, 1.0)),
                             (obj_centers[::step], COLOR_TRAJECTORY_OBJ)]:
            pts, edges = make_traj_edges(traj)
            offset = len(all_pts_list)
            all_pts_list.extend(pts)
            all_edges_list.extend(edges + offset)
            all_colors_list.extend([color] * len(edges))
        
        if len(all_pts_list) == 0:
            return
        
        all_pts = np.array(all_pts_list, dtype=np.float64)
        all_edges = np.array(all_edges_list, dtype=np.int32)
        all_colors = np.array(all_colors_list, dtype=np.float64)
        
        geom, _ = self.geom['trajectory']
        geom.points = o3d.utility.Vector3dVector(all_pts)
        geom.lines = o3d.utility.Vector2iVector(all_edges)
        geom.colors = o3d.utility.Vector3dVector(all_colors)
        if 'trajectory' in self.added_names:
            self.vis.update_geometry(geom)

    def _get_consistent_gt_pts(self, side: str, frame_idx: int):
        """获取 manotorch 自洽 GT 的 face centers，用于与优化手公平对比。

        返回 (1538, 3) numpy array，或 None（如果 pkl 不包含该字段 / 帧不在 pkl 中）。
        """
        payload = self.opti_right if side == 'r' else self.opti_left
        if payload is None or 'gt_face_centers_consistent' not in payload:
            return None
        current_frame_id = int(self.data['frame_id'][frame_idx])
        pkl_frame_ids = np.asarray(payload['frame_ids'])
        matches = np.where(pkl_frame_ids == current_frame_id)[0]
        if matches.size == 0:
            return None
        pkl_idx = int(matches[0])
        return np.asarray(payload['gt_face_centers_consistent'][pkl_idx], dtype=np.float64)

    def _update_opti_hand(self, side: str):
        """Update optimized (MANO CPF) hand point cloud for ``side`` ('r' or 'l').

        Uses strict frame_id matching: if the current npz frame is not present
        in the pkl's ``frame_ids`` array, the geometry is replaced with an
        empty point cloud (invisible) and a single console warning is printed
        per consecutive miss streak (suppressed on subsequent misses, re-armed
        on the next match).
        """
        geom_name = f'opti_hand_{side}'
        if not self.show_opti_hand or geom_name not in self.added_names:
            return

        payload = self.opti_right if side == 'r' else self.opti_left
        if payload is None:
            return

        f = self.frame_idx
        current_frame_id = int(self.data['frame_id'][f])
        pkl_frame_ids = np.asarray(payload['frame_ids'])
        matches = np.where(pkl_frame_ids == current_frame_id)[0]

        geom, _ = self.geom[geom_name]
        if matches.size == 0:
            # Strict miss: empty the point cloud so nothing is drawn, warn once.
            geom.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self.vis.update_geometry(geom)
            if not self._opti_was_missing:
                print(f"[Viewer] Frame id={current_frame_id} not in MANO opti pkl "
                      f"({side}); optimized overlay hidden. "
                      f"pkl has {pkl_frame_ids.size} frame(s).")
                self._opti_was_missing = True
            return

        # Match found: write points
        pkl_idx = int(matches[0])
        opt_pts = np.asarray(payload['opt_hand_face_centers'][pkl_idx], dtype=np.float64)
        geom.points = o3d.utility.Vector3dVector(opt_pts)
        self.vis.update_geometry(geom)
        if self._opti_was_missing:
            print(f"[Viewer] Frame id={current_frame_id} found in MANO opti pkl "
                  f"({side}) at pkl idx {pkl_idx}; overlay restored.")
            self._opti_was_missing = False
        # Stash latest metrics for the title bar (used by _on_frame_change)
        pen = float(payload.get('penetration_depth', np.nan)[pkl_idx]) \
            if 'penetration_depth' in payload else float('nan')
        cr = float(payload.get('contact_ratio', np.nan)[pkl_idx]) \
            if 'contact_ratio' in payload else float('nan')
        if side == 'r':
            self._last_opti_metrics_r = (pen, cr)
        else:
            self._last_opti_metrics_l = (pen, cr)

    # ----------------------------------------------------------
    # Toggle helpers
    # ----------------------------------------------------------

    def _toggle_flag(self, flag_name: str):
        """Toggle a boolean flag and update visualization accordingly."""
        if flag_name == 'normals':
            self.show_normals = not self.show_normals
            for suffix in ['rh', 'lh', 'obj']:
                name = f'normals_{suffix}'
                if self.show_normals:
                    self._add_geom(name)
                else:
                    self._remove_geom(name)
            if self.show_normals:
                self._update_normals('rh', self.data['right_hand_points'][self.frame_idx],
                                     self.data['right_hand_normals'][self.frame_idx])
                self._update_normals('lh', self.data['left_hand_points'][self.frame_idx],
                                     self.data['left_hand_normals'][self.frame_idx])
                self._update_normals('obj', self.data['obj_points'][self.frame_idx],
                                     self.data['obj_normals'][self.frame_idx])
            print(f"[Viewer] Normals: {'ON' if self.show_normals else 'OFF'}")

        elif flag_name == 'flow':
            self.show_flow = not self.show_flow
            for suffix in ['rh', 'lh', 'obj']:
                name = f'flow_{suffix}'
                if self.show_flow:
                    self._add_geom(name)
                else:
                    self._remove_geom(name)
            if self.show_flow:
                T = self.T
                f = self.frame_idx
                num_hand_pts = self.data['right_hand_points'].shape[1]
                num_obj_pts = self.data['obj_points'].shape[1]
                self._update_flow('rh', self.data['right_hand_points'][f],
                                  self.data.get('right_hand_flow', np.zeros((T, num_hand_pts, 3)))[f])
                self._update_flow('lh', self.data['left_hand_points'][f],
                                  self.data.get('left_hand_flow', np.zeros((T, num_hand_pts, 3)))[f])
                self._update_flow('obj', self.data['obj_points'][f],
                                  self.data.get('obj_flow', np.zeros((T, num_obj_pts, 3)))[f])
            print(f"[Viewer] Flow: {'ON' if self.show_flow else 'OFF'}")

        elif flag_name == 'joints':
            self.show_joints = not self.show_joints
            if self.show_joints:
                self._update_joints()
            else:
                geom, _ = self.geom['joints']
                if 'joints' in self.added_names:
                    self.vis.remove_geometry(geom)
                    self.added_names.discard('joints')
            print(f"[Viewer] Joints: {'ON' if self.show_joints else 'OFF'}")

        elif flag_name == 'bones':
            self.show_bones = not self.show_bones
            name = 'bones'
            if self.show_bones:
                self._add_geom(name)
                self._update_bones()
            else:
                self._remove_geom(name)
            print(f"[Viewer] Bones: {'ON' if self.show_bones else 'OFF'}")

        elif flag_name == 'wireframe':
            self.show_wireframe = not self.show_wireframe
            name = 'wireframe'
            if self.show_wireframe:
                self._add_geom(name)
                posed_verts, edges = self._get_posed_obj_edges(self.frame_idx)
                geom, _ = self.geom[name]
                geom.points = o3d.utility.Vector3dVector(posed_verts.astype(np.float64))
                geom.lines = o3d.utility.Vector2iVector(edges.astype(np.int32))
                wf_colors = np.tile(np.array(COLOR_OBJ_WIREFRAME, dtype=np.float64), (len(edges), 1))
                geom.colors = o3d.utility.Vector3dVector(wf_colors)
                self.vis.update_geometry(geom)
            else:
                self._remove_geom(name)
            print(f"[Viewer] Wireframe: {'ON' if self.show_wireframe else 'OFF'}")

        elif flag_name == 'object_pc':
            self.show_object_pc = not self.show_object_pc
            name = 'object_pc'
            if self.show_object_pc:
                self._add_geom(name)
            else:
                self._remove_geom(name)
            print(f"[Viewer] Object point cloud: {'ON' if self.show_object_pc else 'OFF'}")

        elif flag_name == 'right_hand':
            self.show_right_hand = not self.show_right_hand
            name = 'right_hand'
            if self.show_right_hand:
                self._add_geom(name)
            else:
                self._remove_geom(name)
            print(f"[Viewer] Right hand: {'ON' if self.show_right_hand else 'OFF'}")

        elif flag_name == 'left_hand':
            self.show_left_hand = not self.show_left_hand
            name = 'left_hand'
            if self.show_left_hand:
                self._add_geom(name)
            else:
                self._remove_geom(name)
            print(f"[Viewer] Left hand: {'ON' if self.show_left_hand else 'OFF'}")

        elif flag_name == 'opti_hand':
            # 三态循环: 0 (GT only) → 1 (opt only) → 2 (both) → 0
            # 显隐完全由 _update_frame() 按 show_mode 决定（hide by empty point cloud）
            self.show_mode = (self.show_mode + 1) % 3
            self.show_opti_hand = bool(self.show_mode)
            mode_str = ["GT only", "OPT only", "GT + OPT"][self.show_mode]
            # 立即 force 一次 frame update
            self._update_frame()
            print(f"[Viewer] Hand display mode: {mode_str} "
                  f"(0=GT only, 1=OPT only, 2=both)")

        elif flag_name == 'trajectory':
            self.show_trajectory = not self.show_trajectory
            name = 'trajectory'
            if self.show_trajectory:
                self._update_trajectory()
                self._add_geom(name)
            else:
                self._remove_geom(name)
            print(f"[Viewer] Trajectory: {'ON' if self.show_trajectory else 'OFF'}")

        elif flag_name == 'help':
            self.show_help = not self.show_help
            print(f"[Viewer] Help: {'ON' if self.show_help else 'OFF'}")

    # ----------------------------------------------------------
    # Navigation
    # ----------------------------------------------------------

    def _goto_first_opti_frame(self):
        """Jump to the first frame that has optimized data in the pkl."""
        for side_short, attr in [('r', 'opti_right'), ('l', 'opti_left')]:
            payload = getattr(self, attr)
            if payload is None:
                continue
            pkl_fids = np.asarray(payload['frame_ids'])
            if len(pkl_fids) == 0:
                continue
            first_fid = int(pkl_fids[0])
            # Find the NPZ index matching this frame_id
            npz_fids = self.data['frame_id']
            matches = np.where(npz_fids == first_fid)[0]
            if matches.size > 0:
                self.frame_idx = int(matches[0])
                print(f"[Viewer] Jumped to first opti frame: NPZ idx={self.frame_idx}, "
                      f"frame_id={first_fid}")
                return
        print("[Viewer] No opti frame found, staying at frame 0")

    def _next_frame(self, step=1):
        new_idx = min(self.T - 1, self.frame_idx + step)
        if new_idx == self.frame_idx:
            return True
        print(f"[Viewer] _next_frame({step}): {self.frame_idx} -> {new_idx}", flush=True)
        self.frame_idx = new_idx
        self.is_playing = False
        self._on_frame_change()
        self.vis.update_renderer()  # force redraw
        return True

    def _prev_frame(self, step=1):
        new_idx = max(0, self.frame_idx - step)
        if new_idx == self.frame_idx:
            return True
        print(f"[Viewer] _prev_frame({step}): {self.frame_idx} -> {new_idx}", flush=True)
        self.frame_idx = new_idx
        self.is_playing = False
        self._on_frame_change()
        self.vis.update_renderer()  # force redraw
        return True

    def _goto_frame(self, frame_idx: int):
        if frame_idx < 0:
            frame_idx = self.T - 1
        self.frame_idx = max(0, min(self.T - 1, frame_idx))
        self.is_playing = False
        self._on_frame_change()
        return True

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.last_step_time = 0.0
        print(f"[Viewer] {'Playing' if self.is_playing else 'Paused'}")
        return True

    def _next_sequence(self):
        """Switch to the next sequence in the list."""
        self.seq_idx = (self.seq_idx + 1) % len(self.seq_paths)
        self._load_sequence(self.seq_idx)
        self._reload_scene()
        print(f"[Viewer] Sequence [{self.seq_idx+1}/{len(self.seq_paths)}] "
              f"{self.data['seq_id']} | Frames: {self.T}")
        return True

    def _on_frame_change(self):
        """Called when frame changes. Updates geometries and window title."""
        self._update_frame()
        title = f"Ref2Dex Visualizer - {self.data['seq_id']} | Frame {self.frame_idx+1}/{self.T} ({int(self.data['frame_id'][self.frame_idx])})"
        # Append opti metrics if the overlay is on and we have a match.
        if self.show_opti_hand:
            pen_r = getattr(self, '_last_opti_metrics_r', None)
            pen_l = getattr(self, '_last_opti_metrics_l', None)
            metrics = []
            if self.opti_right is not None and pen_r is not None:
                metrics.append(f"R pen={pen_r[0]:.4f} cr={pen_r[1]:.3f}")
            if self.opti_left is not None and pen_l is not None:
                metrics.append(f"L pen={pen_l[0]:.4f} cr={pen_l[1]:.3f}")
            if metrics:
                title += " | opti: " + ", ".join(metrics)
        try:
            self.vis.set_window_name(title)
        except AttributeError:
            pass  # not supported in some Open3D versions

    def _reload_scene(self):
        """Reload the entire scene after a sequence change."""
        # Remove all geometries
        for name in list(self.added_names):
            self._remove_geom(name)

        # Recreate base geometries
        self.geom.clear()
        self.added_names.clear()
        self._create_geometries()
        self._update_frame()

        title = f"Ref2Dex Visualizer - {self.data['seq_id']} | Frame 1/{self.T} ({int(self.data['frame_id'][0])})"
        try:
            self.vis.set_window_name(title)
        except AttributeError:
            pass

    def _reset_camera(self):
        """Reset camera to view the entire scene."""
        self.vis.reset_view_point(True)
        # Also nudge the camera back a bit
        view_ctl = self.vis.get_view_control()
        try:
            view_ctl.set_zoom(0.7)
        except Exception:
            pass
        return True

    def _set_initial_camera(self):
        """Compute scene bbox and place camera to frame the whole scene.

        This works around Open3D's tendency to put the camera inside the
        bounding box when geometries start at (0,0,0).
        """
        f = self.frame_idx
        data = self.data

        # Collect all current points
        all_pts = []
        for key in ['right_hand_points', 'left_hand_points', 'obj_points']:
            pts = data[key][f]
            all_pts.append(pts)
        all_pts = np.concatenate(all_pts, axis=0)  # (N, 3)

        bbox_min = all_pts.min(axis=0)
        bbox_max = all_pts.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        extent = (bbox_max - bbox_min).max()

        if extent < 1e-6:
            extent = 1.0

        # Place camera looking from front-right-above
        cam_pos = center + np.array([1.0, -1.0, 0.6]) * extent * 0.9

        view_ctl = self.vis.get_view_control()
        # Set lookat target
        view_ctl.set_lookat(center)
        # Set front direction (pointing back at target)
        view_ctl.set_front((center - cam_pos) / np.linalg.norm(center - cam_pos))
        # Up direction
        view_ctl.set_up([0, 0, 1])
        # Zoom
        view_ctl.set_zoom(0.55)
        return True

    def _quit(self):
        self.is_playing = False
        self.vis.close()
        return True

    # ----------------------------------------------------------
    # Info text
    # ----------------------------------------------------------

    def _help_text(self):
        return (
            "\n=== Ref2Dex Visualizer Controls ===\n"
            "  Left / H      : Previous frame\n"
            "  Right / L     : Next frame\n"
            "  Up / K        : Jump 10 frames forward\n"
            "  Home/End      : First/Last frame\n"
            "  Space         : Play/Pause\n"
            "  N             : Toggle normals\n"
            "  F             : Toggle flow arrows\n"
            "  J             : Toggle joint markers\n"
            "  B             : Toggle bones\n"
            "  W             : Toggle object wireframe\n"
            "  O             : Toggle object point cloud\n"
            "  1             : Toggle right hand\n"
            "  2             : Toggle left hand\n"
            "  3 / M         : Cycle hand display: GT only → opt only → both\n"
            "  T             : Toggle trajectory paths\n"
            "  S             : Next sequence\n"
            "  R             : Reset camera\n"
            "  H             : Show/hide this help\n"
            "  Escape/Q      : Quit\n"
            "==================================\n"
        )

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------

    def run(self):
        """Main rendering loop."""
        try:
            self._on_frame_change()

            while self.vis.poll_events():
                # Handle autoplay
                now = time.time()
                if self.is_playing and self.autoplay_hz > 0:
                    if self.last_step_time == 0.0:
                        self.last_step_time = now
                    elif (now - self.last_step_time) >= (1.0 / self.autoplay_hz):
                        if self.frame_idx >= self.T - 1:
                            self.frame_idx = 0
                        else:
                            self.frame_idx += 1
                        self._on_frame_change()
                        self.last_step_time = now

                self.vis.update_renderer()
                time.sleep(0.005)  # small sleep to prevent busy-waiting

        except KeyboardInterrupt:
            pass
        finally:
            self.vis.destroy_window()


# ============================================================
# Main entry point
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Ref2Dex processed data visualizer (Open3D)"
    )
    parser.add_argument("--seq", type=str, default=None,
                        help="Sequence identifier, e.g. s01/box_grab_01")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT_DEFAULT,
                        help=f"Processed data root (default: {DATA_ROOT_DEFAULT})")
    parser.add_argument("--arctic_root", type=str, default=ARCTIC_ROOT_DEFAULT,
                        help=f"ARCTIC raw data root for articulation (default: {ARCTIC_ROOT_DEFAULT})")
    parser.add_argument("--assets_root", type=str, default=None,
                        help="Asset root directory (default: <Ref2Dex>/assets)")
    parser.add_argument("--opti-root", type=str, default=None,
                        help="Optimization result root. Default: auto-probe outputs/mano_fit/* then legacy mano_opti_data/*")
    parser.add_argument("--show-opti-hand", action="store_true", default=False,
                        help="Overlay the MANO CPF optimized hand (light green) "
                             "for visual comparison with the original hand.")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    parser.add_argument("--normals", action="store_true", default=False,
                        help="Show normals by default")
    parser.add_argument("--flow", action="store_true", default=False,
                        help="Show flow arrows by default")
    parser.add_argument("--joints", action="store_true", default=True,
                        help="Show joint markers by default")
    parser.add_argument("--bones", action="store_true", default=True,
                        help="Show bones by default")
    parser.add_argument("--wireframe", action="store_true", default=True,
                        help="Show object wireframe by default")
    parser.add_argument("--trajectory", action="store_true", default=True,
                        help="Show trajectory lines by default")
    parser.add_argument("--normal_stride", type=int, default=16,
                        help="Subsample stride for normals display (default: 16)")
    parser.add_argument("--flow_stride", type=int, default=32,
                        help="Subsample stride for flow arrows (default: 32)")
    parser.add_argument("--flow_scale", type=float, default=1.0,
                        help="Flow arrow scale multiplier (default: 1.0)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    viewer = Ref2DexViewer(args)
    viewer.run()


if __name__ == "__main__":
    main()
