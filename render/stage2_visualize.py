from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("DISPLAY", "localhost:10.0")


# ============================================================
# 交互键（与 _register_callbacks 注册顺序一致；底部 print 行也会再打印一次）
# ============================================================
# ←/A (263, ord("A"))   : 上一帧
# →/D (262, ord("D"))   : 下一帧
# O                     : 切换物体显示（show_object）
# H                     : 切换手部显示（show_hand）
# C                     : 切换热力图上色（heatmap，距离 ≤5cm 蓝→红渐变）
# F                     : 切换坐标系（object_frame：world ↔ obj）
# ============================================================

# ============================================================
# 颜色对照表（与 refresh() 中各 layer 的颜色常量一致）
# ============================================================
# object (默认)   : 蓝灰     (0.55, 0.62, 0.70)   无热力图时的物体色
# object (热力图)  : 距离色带（_distance_colors）  每个 obj 点的最近手距映射
#                  0cm → 红 (1.00, 0.15, 0.12)
#                  2.5cm → 紫红 (0.725, 0.475, 0.545)
#                  5cm+ → 暗紫蓝 (0.45, 0.80, 0.97)
# hand            : 橙       (0.96, 0.52, 0.12)   手部固定橙色
# ============================================================


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {path}")
    if data.get("schema_name") != "ref2dex_opti":
        raise ValueError(
            f"{path} is not a common Stage 2 file "
            f"(schema_name={data.get('schema_name')!r})"
        )
    required = (
        "raw_frame_id",
        "obj_points_world",
        "obj_normals_world",
        "hand_points_world",
        "hand_normals_world",
        "obj_root_pose",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing Stage 2 fields: {missing}")
    return data


def _frame_min_distance(obj: np.ndarray, hand: np.ndarray) -> float:
    # Chunking avoids a large temporary [4096, 1538, 3] array.
    import torch

    with torch.no_grad():
        distance = torch.cdist(
            torch.from_numpy(np.asarray(obj, dtype=np.float32)),
            torch.from_numpy(np.asarray(hand, dtype=np.float32)),
        )
    return float(distance.min())


def _world_to_object(
    points: np.ndarray,
    normals: np.ndarray,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    points_obj = (points - translation) @ rotation
    normals_obj = normals @ rotation
    return points_obj.astype(np.float64), normals_obj.astype(np.float64)


def _distance_colors(distance: np.ndarray, limit: float = 0.05) -> np.ndarray:
    value = np.clip(distance / limit, 0.0, 1.0)
    return np.stack(
        [1.0 - 0.55 * value, 0.15 + 0.65 * value, 0.12 + 0.85 * value],
        axis=-1,
    )


def _nearest_distance(obj: np.ndarray, hand: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        result = torch.cdist(
            torch.from_numpy(np.asarray(obj, dtype=np.float32)),
            torch.from_numpy(np.asarray(hand, dtype=np.float32)),
        ).amin(dim=1)
    return result.numpy()


class Stage2Viewer:
    def __init__(self, data: dict[str, Any], args: argparse.Namespace) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.data = data
        self.frame = int(np.clip(args.frame, 0, len(data["raw_frame_id"]) - 1))
        self.step = max(1, int(args.step))
        self.object_frame = bool(args.object_frame)
        self.show_object = bool(args.show_object)
        self.show_hand = bool(args.show_hand)
        self.heatmap = bool(args.heatmap)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window(
            window_name=f"Stage 2: {data.get('seq_id', '')} {data.get('side', '')}",
            width=args.width,
            height=args.height,
        ):
            raise RuntimeError(
                "Open3D window creation failed. Check DISPLAY; expected "
                f"{os.environ.get('DISPLAY')!r}."
            )
        self.geometries: dict[str, Any] = {}
        self._register_callbacks()
        self.refresh(reset_view=True)

    def _register_callbacks(self) -> None:
        for key in (262, ord("D")):  # GLFW right
            self.vis.register_key_callback(key, lambda vis: self._move(+self.step))
        for key in (263, ord("A")):  # GLFW left
            self.vis.register_key_callback(key, lambda vis: self._move(-self.step))
        self.vis.register_key_callback(ord("O"), lambda vis: self._toggle("show_object"))
        self.vis.register_key_callback(ord("H"), lambda vis: self._toggle("show_hand"))
        self.vis.register_key_callback(ord("C"), lambda vis: self._toggle("heatmap"))
        self.vis.register_key_callback(ord("F"), lambda vis: self._toggle("object_frame"))

    def _move(self, delta: int) -> bool:
        self.frame = int(np.clip(self.frame + delta, 0, len(self.data["raw_frame_id"]) - 1))
        self.refresh()
        return False

    def _toggle(self, field: str) -> bool:
        setattr(self, field, not bool(getattr(self, field)))
        self.refresh()
        return False

    def _set_point_cloud(
        self,
        name: str,
        points: np.ndarray | None,
        colors: np.ndarray | None,
    ) -> None:
        if points is None or len(points) == 0:
            geometry = self.geometries.pop(name, None)
            if geometry is not None:
                self.vis.remove_geometry(geometry, reset_bounding_box=False)
            return
        geometry = self.geometries.get(name)
        is_new = geometry is None
        if is_new:
            geometry = self.o3d.geometry.PointCloud()
            self.geometries[name] = geometry
        geometry.points = self.o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        geometry.colors = self.o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        if is_new:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            self.vis.update_geometry(geometry)

    def refresh(self, reset_view: bool = False) -> None:
        index = self.frame
        obj = np.asarray(self.data["obj_points_world"][index], dtype=np.float64)
        obj_normal = np.asarray(self.data["obj_normals_world"][index], dtype=np.float64)
        hand = np.asarray(self.data["hand_points_world"][index], dtype=np.float64)
        hand_normal = np.asarray(self.data["hand_normals_world"][index], dtype=np.float64)
        if self.object_frame:
            pose = np.asarray(self.data["obj_root_pose"][index], dtype=np.float64)
            obj, obj_normal = _world_to_object(obj, obj_normal, pose)
            hand, hand_normal = _world_to_object(hand, hand_normal, pose)

        if self.heatmap:
            colors = _distance_colors(_nearest_distance(obj, hand))
        else:
            colors = np.tile(np.asarray([[0.55, 0.62, 0.70]]), (len(obj), 1))
        self._set_point_cloud("object", obj if self.show_object else None, colors)
        hand_colors = np.tile(np.asarray([[0.96, 0.52, 0.12]]), (len(hand), 1))
        self._set_point_cloud("hand", hand if self.show_hand else None, hand_colors)

        raw_frame = int(self.data["raw_frame_id"][index])
        saved_min = float(np.asarray(self.data.get("frame_min_hand_obj_dist", []))[index])
        print(
            f"\r[stage2] frame={index}/{len(self.data['raw_frame_id']) - 1} "
            f"raw={raw_frame} min={saved_min * 100:.2f}cm "
            f"coord={'object' if self.object_frame else 'world'} "
            f"object={self.show_object} hand={self.show_hand} heatmap={self.heatmap}   ",
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
    parser = argparse.ArgumentParser(description="Visualize common Ref2Dex Stage 2 data")
    parser.add_argument("--input", required=True, help="Stage 2 .pkl file")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    _add_bool_flag(parser, "object-frame", default=False)
    _add_bool_flag(parser, "show-object", default=True)
    _add_bool_flag(parser, "show-hand", default=True)
    _add_bool_flag(parser, "heatmap", default=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=0,
        help="Open a window, update N frames, then close (renderer regression check)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input).expanduser().resolve()
    data = _load(path)
    frame_count = int(np.asarray(data["raw_frame_id"]).shape[0])
    min_dist = np.asarray(data.get("frame_min_hand_obj_dist", []), dtype=np.float32)
    if min_dist.shape != (frame_count,):
        raise ValueError(
            f"frame_min_hand_obj_dist must have shape ({frame_count},), got {min_dist.shape}"
        )
    print(
        f"[stage2] {path}\n"
        f"  seq={data.get('seq_id')} side={data.get('side')} frames={frame_count}\n"
        f"  object={np.asarray(data['obj_points_world']).shape} "
        f"hand={np.asarray(data['hand_points_world']).shape}\n"
        f"  min-distance={min_dist.min() * 100:.2f}..{min_dist.max() * 100:.2f}cm "
        f"(all frames must be <=5cm)"
    )
    if float(min_dist.max(initial=0.0)) > 0.050001:
        raise ValueError("Stage 2 contains a frame farther than the 5cm keep threshold")
    if args.check_only:
        return
    print("Keys: Left/A previous, Right/D next, O object, H hand, C heatmap, F coordinate frame")
    viewer = Stage2Viewer(data, args)
    if args.smoke_frames > 0:
        viewer.smoke(args.smoke_frames)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
