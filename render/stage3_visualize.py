from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.task.correspondence_ptv3.data import (
    augment_geometry,
    compute_input_knn,
    sample_object_indices,
    stable_frame_seed,
)


os.environ.setdefault("DISPLAY", "localhost:10.0")


# ============================================================
# 交互键（与 _register_callbacks 注册顺序一致）
# ============================================================
# ←/A (263, ord("A"))   : 上一帧
# →/D (262, ord("D"))   : 下一帧
# [ / ]                 : epoch -1 / +1（重新触发 augment / 采样种子）
# ----------------------------------------------------------- 几何层切换
# O                     : toggle show_pool        （4096 候选池，灰）
# C                     : toggle show_candidates  （≤5cm 内的 obj，蓝）
# S                     : toggle show_selected    （512 真训练 token，contact 着色）
# H                     : toggle show_clean_hand  （GT 优化后手，橙）
# N                     : toggle show_noisy_hand  （扰动后手，洋红）
# ----------------------------------------------------------- KNN 连线
# K                     : toggle show_clean_knn   （GT KNN，黄线）
# I                     : toggle show_input_knn   （扰动后 KNN，青线）
# ----------------------------------------------------------- 增强
# G                     : toggle augment          （整体旋转/平移/缩放）
# P                     : toggle hand_perturb     （手部位姿/平移扰动）
# ============================================================

# ============================================================
# 颜色对照表（与 refresh() 中各 layer 的颜色常量一致）
# ============================================================
# 几何层（点云，refresh() 内 hardcode）
# -----------------------------------------------------------
# pool         : 灰       (0.32, 0.35, 0.39)   4096 候选池
# candidates   : 蓝       (0.20, 0.55, 0.96)   ≤5cm 内的 obj
# selected     : 接触色带（_contact_colors）    512 真训练 token，per-point 上色
#                  0cm → 红 (1.00, 0.12, 0.10)
#                  2.5cm → 紫 (0.73, 0.47, 0.54)
#                  5cm+ → 蓝紫 (0.45, 0.82, 0.98)
# clean_hand   : 橙       (0.95, 0.58, 0.12)   GT 优化后手
# noisy_hand   : 洋红     (0.85, 0.16, 0.85)   扰动后手
# -----------------------------------------------------------
# 连线层（_set_lines，refresh() 内 hardcode）
# -----------------------------------------------------------
# clean_knn    : 黄       (1.00, 0.88, 0.10)   GT KNN 连线
# input_knn    : 青       (0.08, 0.95, 0.92)   扰动后 KNN 连线
# ============================================================


REQUIRED_FIELDS = {
    "schema_name",
    "raw_frame_id",
    "obj_points",
    "obj_normals",
    "obj_point_id",
    "hand_points",
    "hand_normals",
    "hand_cano_points",
    "obj_to_hand_min_dist",
    "obj_candidate_mask_5cm",
    "gt_obj_to_hand_knn_idx",
}


def _scalar(data: Any, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return str(value.item()) if value.size == 1 else default


def _validate(data: Any, path: Path) -> dict[str, int]:
    missing = REQUIRED_FIELDS.difference(data.keys())
    if missing:
        raise KeyError(f"{path}: missing Stage 3 fields {sorted(missing)}")
    if _scalar(data, "schema_name") != "train_corr_static":
        raise ValueError(
            f"{path}: expected schema_name='train_corr_static', "
            f"got {_scalar(data, 'schema_name')!r}"
        )
    obj = np.asarray(data["obj_points"])
    hand = np.asarray(data["hand_points"])
    obj_normals = np.asarray(data["obj_normals"])
    hand_normals = np.asarray(data["hand_normals"])
    min_dist = np.asarray(data["obj_to_hand_min_dist"])
    candidate = np.asarray(data["obj_candidate_mask_5cm"])
    knn = np.asarray(data["gt_obj_to_hand_knn_idx"])
    if obj.ndim != 3 or obj.shape[-1] != 3:
        raise ValueError(f"obj_points must be [T,4096,3], got {obj.shape}")
    frames, pool, _ = obj.shape
    expected = {
        "obj_normals": (frames, pool, 3),
        "obj_point_id": (pool,),
        "hand_points": (frames, hand.shape[1], 3),
        "hand_normals": (frames, hand.shape[1], 3),
        "hand_cano_points": (hand.shape[1], 3),
        "obj_to_hand_min_dist": (frames, pool),
        "obj_candidate_mask_5cm": (frames, pool),
    }
    for key, shape in expected.items():
        actual = np.asarray(data[key]).shape
        if actual != shape:
            raise ValueError(f"{key}: expected {shape}, got {actual}")
    if knn.shape[:2] != (frames, pool) or knn.ndim != 3:
        raise ValueError(f"gt_obj_to_hand_knn_idx has invalid shape {knn.shape}")
    if pool != 4096 or hand.shape[1] != 1538:
        raise ValueError(f"Expected object pool 4096 and hand 1538, got {pool}, {hand.shape[1]}")
    if not np.array_equal(candidate, min_dist <= 0.05):
        mismatch = int(np.count_nonzero(candidate != (min_dist <= 0.05)))
        raise ValueError(f"5cm candidate mask disagrees with distances at {mismatch} points")
    if np.any(knn[~candidate] != -1):
        raise ValueError("Non-candidate KNN rows must be padded with -1")
    if np.any(knn[candidate] < 0) or np.any(knn[candidate] >= hand.shape[1]):
        raise ValueError("Candidate KNN rows contain an invalid hand index")
    if not all(np.isfinite(value).all() for value in (obj, obj_normals, hand, hand_normals, min_dist)):
        raise ValueError("Stage 3 geometry contains non-finite values")
    return {
        "frames": frames,
        "pool": pool,
        "hand": hand.shape[1],
        "k": knn.shape[2],
        "candidate_min": int(candidate.sum(axis=1).min()),
        "candidate_median": int(np.median(candidate.sum(axis=1))),
        "candidate_max": int(candidate.sum(axis=1).max()),
    }


@dataclass
class RuntimeFrame:
    pool_points: np.ndarray
    candidate_points: np.ndarray
    selected_points: np.ndarray
    selected_colors: np.ndarray
    selected_valid: np.ndarray
    clean_hand_points: np.ndarray
    noisy_hand_points: np.ndarray
    clean_knn_idx: np.ndarray
    input_knn_idx: np.ndarray
    input_knn_valid: np.ndarray
    selected_idx: np.ndarray
    sample_seed: int
    augmentation_seed: int
    hand_perturbed: bool
    distance_scale: float


def _contact_colors(distance: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(distance) / 0.05, 0.0, 1.0)
    return np.stack(
        [1.0 - 0.55 * value, 0.12 + 0.70 * value, 0.10 + 0.88 * value],
        axis=-1,
    )


def _build_runtime_frame(
    data: Any,
    frame: int,
    epoch: int,
    args: argparse.Namespace,
) -> RuntimeFrame:
    seq_id = _scalar(data, "seq_id", "unknown")
    side = _scalar(data, "side", "")
    raw_frame_id = int(np.asarray(data["raw_frame_id"])[frame])
    sample_seed = stable_frame_seed(
        base_seed=args.base_seed,
        seq_id=seq_id,
        side=side,
        raw_frame_id=raw_frame_id,
        epoch=epoch,
    )
    selected_idx, selected_valid = sample_object_indices(
        np.asarray(data["obj_candidate_mask_5cm"][frame]),
        num_samples=args.num_obj_points,
        seed=sample_seed,
    )
    safe_idx = np.maximum(selected_idx, 0)
    selected_obj = np.asarray(data["obj_points"][frame, safe_idx], dtype=np.float32).copy()
    selected_normal = np.asarray(data["obj_normals"][frame, safe_idx], dtype=np.float32).copy()
    selected_distance = np.asarray(
        data["obj_to_hand_min_dist"][frame, safe_idx],
        dtype=np.float32,
    ).copy()
    clean_knn = np.asarray(
        data["gt_obj_to_hand_knn_idx"][frame, safe_idx],
        dtype=np.int64,
    ).copy()
    selected_obj[~selected_valid] = 0
    selected_normal[~selected_valid] = 0
    selected_distance[~selected_valid] = 0
    clean_knn[~selected_valid] = -1

    hand = np.asarray(data["hand_points"][frame], dtype=np.float32)
    hand_normal = np.asarray(data["hand_normals"][frame], dtype=np.float32)
    augmentation_seed = stable_frame_seed(
        base_seed=args.base_seed,
        seq_id=seq_id,
        side=side,
        raw_frame_id=raw_frame_id,
        epoch=epoch,
        namespace="augmentation",
    )
    common_kwargs = {
        "hand_points": hand,
        "hand_normals": hand_normal,
        "seed": augmentation_seed,
        "apply_hand_perturb": bool(args.hand_perturb),
        "hand_rot_std_deg": float(args.hand_rot_std_deg),
        "hand_trans_std": float(args.hand_trans_std),
        "hand_perturb_prob": float(args.hand_perturb_prob),
        "apply_global_aug": bool(args.augment),
        "augment_rotation": bool(args.augment_rotation),
        "rotation_range_deg": float(args.rotation_range),
        "augment_translation": bool(args.augment_translation),
        "translation_range": float(args.translation_range),
        "augment_scale": bool(args.augment_scale),
        "scale_range": tuple(args.scale_range),
    }
    selected_geometry = augment_geometry(
        obj_points=selected_obj,
        obj_normals=selected_normal,
        **common_kwargs,
    )
    pool_geometry = augment_geometry(
        obj_points=np.asarray(data["obj_points"][frame], dtype=np.float32),
        obj_normals=np.asarray(data["obj_normals"][frame], dtype=np.float32),
        **common_kwargs,
    )
    selected_distance *= float(selected_geometry.distance_scale)
    input_knn_idx, input_knn_valid = compute_input_knn(
        selected_geometry.input_obj_points,
        selected_geometry.input_hand_points,
        selected_valid,
        k_cross=int(data["gt_obj_to_hand_knn_idx"].shape[2]),
    )
    candidate_mask = np.asarray(data["obj_candidate_mask_5cm"][frame], dtype=bool)
    return RuntimeFrame(
        pool_points=pool_geometry.gt_obj_points,
        candidate_points=pool_geometry.gt_obj_points[candidate_mask],
        selected_points=selected_geometry.gt_obj_points,
        selected_colors=_contact_colors(selected_distance),
        selected_valid=selected_valid,
        clean_hand_points=selected_geometry.gt_hand_points,
        noisy_hand_points=selected_geometry.input_hand_points,
        clean_knn_idx=clean_knn,
        input_knn_idx=input_knn_idx,
        input_knn_valid=input_knn_valid,
        selected_idx=selected_idx,
        sample_seed=sample_seed,
        augmentation_seed=augmentation_seed,
        hand_perturbed=selected_geometry.hand_perturbed,
        distance_scale=selected_geometry.distance_scale,
    )


def _knn_lines(
    obj: np.ndarray,
    hand: np.ndarray,
    knn_idx: np.ndarray,
    valid_obj: np.ndarray,
    ranks: int,
    edge_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    max_rank = min(max(0, ranks), knn_idx.shape[1])
    for obj_idx in np.flatnonzero(valid_obj):
        for rank in range(max_rank):
            if edge_valid is not None and not bool(edge_valid[obj_idx, rank]):
                continue
            hand_idx = int(knn_idx[obj_idx, rank])
            if hand_idx < 0 or hand_idx >= len(hand):
                continue
            start = len(points)
            points.extend((obj[obj_idx], hand[hand_idx]))
            lines.append([start, start + 1])
    if not lines:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    return np.asarray(points, dtype=np.float64), np.asarray(lines, dtype=np.int32)


class Stage3Viewer:
    def __init__(self, data: Any, args: argparse.Namespace) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.data = data
        self.args = args
        self.frame = int(np.clip(args.frame, 0, len(data["raw_frame_id"]) - 1))
        self.epoch = max(0, int(args.epoch))
        self.step = max(1, int(args.step))
        self.show_pool = bool(args.show_pool)
        self.show_candidates = bool(args.show_candidates)
        self.show_selected = bool(args.show_selected)
        self.show_clean_hand = bool(args.show_clean_hand)
        self.show_noisy_hand = bool(args.show_noisy_hand)
        self.show_clean_knn = bool(args.show_clean_knn)
        self.show_input_knn = bool(args.show_input_knn)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window(
            window_name=f"Stage 3 audit: {_scalar(data, 'seq_id')} {_scalar(data, 'side')}",
            width=args.width,
            height=args.height,
        ):
            raise RuntimeError(
                "Open3D window creation failed. Check DISPLAY; expected "
                f"{os.environ.get('DISPLAY')!r}."
            )
        options = self.vis.get_render_option()
        options.point_size = float(args.point_size)
        options.line_width = float(args.line_width)
        options.background_color = np.asarray([0.035, 0.035, 0.045])
        self.geometries: dict[str, Any] = {}
        self._register_callbacks()
        self.refresh(reset_view=True)

    def _register_callbacks(self) -> None:
        for key in (262, ord("D")):
            self.vis.register_key_callback(key, lambda vis: self._move(+self.step))
        for key in (263, ord("A")):
            self.vis.register_key_callback(key, lambda vis: self._move(-self.step))
        self.vis.register_key_callback(ord("["), lambda vis: self._change_epoch(-1))
        self.vis.register_key_callback(ord("]"), lambda vis: self._change_epoch(+1))
        for key, field in (
            ("O", "show_pool"),
            ("C", "show_candidates"),
            ("S", "show_selected"),
            ("H", "show_clean_hand"),
            ("N", "show_noisy_hand"),
            ("K", "show_clean_knn"),
            ("I", "show_input_knn"),
        ):
            self.vis.register_key_callback(ord(key), lambda vis, name=field: self._toggle(name))
        self.vis.register_key_callback(ord("G"), lambda vis: self._toggle_arg("augment"))
        self.vis.register_key_callback(ord("P"), lambda vis: self._toggle_arg("hand_perturb"))

    def _move(self, delta: int) -> bool:
        self.frame = int(np.clip(self.frame + delta, 0, len(self.data["raw_frame_id"]) - 1))
        self.refresh()
        return False

    def _change_epoch(self, delta: int) -> bool:
        self.epoch = max(0, self.epoch + delta)
        self.refresh()
        return False

    def _toggle(self, field: str) -> bool:
        setattr(self, field, not bool(getattr(self, field)))
        self.refresh()
        return False

    def _toggle_arg(self, field: str) -> bool:
        setattr(self.args, field, not bool(getattr(self.args, field)))
        self.refresh()
        return False

    def _set_point_cloud(
        self,
        name: str,
        points: np.ndarray | None,
        color: np.ndarray,
        per_point_color: np.ndarray | None = None,
    ) -> None:
        if points is None or len(points) == 0:
            self._remove(name)
            return
        geometry = self.geometries.get(name)
        is_new = geometry is None
        if is_new:
            geometry = self.o3d.geometry.PointCloud()
            self.geometries[name] = geometry
        geometry.points = self.o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        colors = (
            np.asarray(per_point_color, dtype=np.float64)
            if per_point_color is not None
            else np.tile(np.asarray(color, dtype=np.float64)[None], (len(points), 1))
        )
        geometry.colors = self.o3d.utility.Vector3dVector(colors)
        if is_new:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            self.vis.update_geometry(geometry)

    def _set_lines(
        self,
        name: str,
        points: np.ndarray,
        lines: np.ndarray,
        color: np.ndarray,
    ) -> None:
        # Never submit an empty LineSet to Open3D. This is the source of the
        # SimpleShaderForLineSet "Binding failed with empty geometry" warning.
        if len(lines) == 0:
            self._remove(name)
            return
        geometry = self.geometries.get(name)
        is_new = geometry is None
        if is_new:
            geometry = self.o3d.geometry.LineSet()
            self.geometries[name] = geometry
        geometry.points = self.o3d.utility.Vector3dVector(points)
        geometry.lines = self.o3d.utility.Vector2iVector(lines)
        geometry.colors = self.o3d.utility.Vector3dVector(
            np.tile(np.asarray(color, dtype=np.float64)[None], (len(lines), 1))
        )
        if is_new:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            self.vis.update_geometry(geometry)

    def _remove(self, name: str) -> None:
        geometry = self.geometries.pop(name, None)
        if geometry is not None:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)

    def refresh(self, reset_view: bool = False) -> None:
        runtime = _build_runtime_frame(self.data, self.frame, self.epoch, self.args)
        valid = runtime.selected_valid
        self._set_point_cloud(
            "pool",
            runtime.pool_points if self.show_pool else None,
            np.asarray([0.32, 0.35, 0.39]),
        )
        self._set_point_cloud(
            "candidates",
            runtime.candidate_points if self.show_candidates else None,
            np.asarray([0.20, 0.55, 0.96]),
        )
        self._set_point_cloud(
            "selected",
            runtime.selected_points[valid] if self.show_selected else None,
            np.asarray([1.0, 0.2, 0.1]),
            runtime.selected_colors[valid] if self.show_selected else None,
        )
        self._set_point_cloud(
            "clean_hand",
            runtime.clean_hand_points if self.show_clean_hand else None,
            np.asarray([0.95, 0.58, 0.12]),
        )
        self._set_point_cloud(
            "noisy_hand",
            runtime.noisy_hand_points if self.show_noisy_hand else None,
            np.asarray([0.85, 0.16, 0.85]),
        )

        if self.show_clean_knn:
            points, lines = _knn_lines(
                runtime.selected_points,
                runtime.clean_hand_points,
                runtime.clean_knn_idx,
                valid,
                self.args.knn_ranks,
            )
        else:
            points, lines = np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)
        self._set_lines("clean_knn", points, lines, np.asarray([1.0, 0.88, 0.10]))

        if self.show_input_knn:
            points, lines = _knn_lines(
                runtime.selected_points,
                runtime.noisy_hand_points,
                runtime.input_knn_idx,
                valid,
                self.args.knn_ranks,
                runtime.input_knn_valid,
            )
        else:
            points, lines = np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)
        self._set_lines("input_knn", points, lines, np.asarray([0.08, 0.95, 0.92]))

        candidate_count = int(np.asarray(self.data["obj_candidate_mask_5cm"][self.frame]).sum())
        raw_frame = int(np.asarray(self.data["raw_frame_id"])[self.frame])
        print(
            f"\r[stage3] frame={self.frame}/{len(self.data['raw_frame_id']) - 1} "
            f"raw={raw_frame} epoch={self.epoch} candidates={candidate_count} "
            f"selected={int(valid.sum())} pad={int((~valid).sum())} "
            f"global_aug={self.args.augment} hand_noise={runtime.hand_perturbed} "
            f"seed={runtime.sample_seed}   ",
            end="",
            flush=True,
        )
        if reset_view:
            self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self) -> None:
        try:
            self.vis.run()
        finally:
            print()
            self.vis.destroy_window()

    def smoke(self, frame_count: int) -> None:
        try:
            for _ in range(max(0, int(frame_count) - 1)):
                self.frame = min(self.frame + self.step, len(self.data["raw_frame_id"]) - 1)
                self.refresh()
        finally:
            print()
            self.vis.destroy_window()


def _add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true")
    group.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stage 3 full pools, epoch samples, augmentations, and KNN"
    )
    parser.add_argument("--input", required=True, help="Stage 3 .npz file")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-obj-points", type=int, default=512)
    parser.add_argument("--knn-ranks", type=int, default=1)

    _add_bool_flag(parser, "augment", default=False)
    _add_bool_flag(parser, "hand-perturb", default=False)
    _add_bool_flag(parser, "augment-rotation", default=True)
    _add_bool_flag(parser, "augment-translation", default=False)
    _add_bool_flag(parser, "augment-scale", default=False)
    parser.add_argument("--rotation-range", type=float, default=180.0)
    parser.add_argument("--translation-range", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, nargs=2, default=(0.9, 1.1))
    parser.add_argument("--hand-rot-std-deg", type=float, default=10.0)
    parser.add_argument("--hand-trans-std", type=float, default=0.01)
    parser.add_argument("--hand-perturb-prob", type=float, default=1.0)

    _add_bool_flag(parser, "show-pool", default=False)
    _add_bool_flag(parser, "show-candidates", default=False)
    _add_bool_flag(parser, "show-selected", default=True)
    _add_bool_flag(parser, "show-clean-hand", default=True)
    _add_bool_flag(parser, "show-noisy-hand", default=False)
    _add_bool_flag(parser, "show-clean-knn", default=True)
    _add_bool_flag(parser, "show-input-knn", default=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=0,
        help="Open a window, update N frames, then close (renderer regression check)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--line-width", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input).expanduser().resolve()
    # NpzFile extracts an entire member on every ``data[key]`` access. Keeping
    # it lazy makes each arrow-key refresh repeatedly reread the large KNN
    # tensor and looks like a frozen viewer. Load every member once instead.
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    stats = _validate(data, path)
    frame = int(np.clip(args.frame, 0, stats["frames"] - 1))
    runtime = _build_runtime_frame(data, frame, max(0, args.epoch), args)
    print(
        f"[stage3] {path}\n"
        f"  seq={_scalar(data, 'seq_id')} side={_scalar(data, 'side')} "
        f"frames={stats['frames']}\n"
        f"  pool={stats['pool']} hand={stats['hand']} K={stats['k']} "
        f"candidate[min/median/max]="
        f"{stats['candidate_min']}/{stats['candidate_median']}/{stats['candidate_max']}\n"
        f"  frame={frame} epoch={max(0, args.epoch)} "
        f"selected={int(runtime.selected_valid.sum())} "
        f"padding={int((~runtime.selected_valid).sum())}"
    )
    if args.check_only:
        return
    print(
        "Keys: Left/A, Right/D frames; [/] epoch; O pool; C candidates; "
        "S selected512; H clean hand; N noisy hand; K clean KNN; "
        "I input KNN; G global augment; P hand perturb"
    )
    viewer = Stage3Viewer(data, args)
    if args.smoke_frames > 0:
        viewer.smoke(args.smoke_frames)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
