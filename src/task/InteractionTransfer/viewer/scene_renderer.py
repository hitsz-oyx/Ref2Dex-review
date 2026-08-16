"""scene 渲染工具：atomic 批量重建 + 自动 panel spacing + 通用 adder。

viser 1.0.30 的 ``MeshHandle`` 没有 vertices setter，无法原地更新 mesh，
因此统一采用 ``server.atomic()`` 内整体 remove/re-add（viewer_gty 已验证
该方式播放不闪烁）。
"""
from __future__ import annotations

import numpy as np


def object_diameter(vertices: np.ndarray) -> float:
    return float(np.linalg.norm(np.ptp(vertices, axis=0)))


def auto_spacing(object_vertices: np.ndarray) -> float:
    """panel 间距：max(0.15, 1.5 * object diameter)，与 InteractionDynamics 一致。"""
    return max(0.15, 1.5 * object_diameter(object_vertices))


def panel_offsets(count: int, spacing: float) -> np.ndarray:
    """count 个 panel 的 x 偏移；奇数时中间 panel 恒在原点。"""
    return (np.arange(count) - (count - 1) / 2.0) * spacing


class SceneBatch:
    """收集本帧全部 handles，atomic 重建，避免客户端看到空帧闪烁。"""

    def __init__(self, server):
        self.server = server
        self.handles = []

    def clear(self) -> None:
        with self.server.atomic():
            for handle in self.handles:
                handle.remove()
            self.handles = []

    def _keep(self, handle):
        self.handles.append(handle)
        return handle

    def mesh(self, name: str, vertices: np.ndarray, faces: np.ndarray,
             color, opacity: float = 1.0, visible: bool = True,
             wireframe: bool = False):
        return self._keep(self.server.scene.add_mesh_simple(
            name, vertices, faces.astype(np.int32), color=color,
            opacity=opacity, visible=visible, wireframe=wireframe))

    def cloud(self, name: str, points: np.ndarray, colors, point_size: float,
              visible: bool = True):
        return self._keep(self.server.scene.add_point_cloud(
            name, points=points, colors=colors, point_size=point_size, visible=visible))

    def arrows(self, name: str, points: np.ndarray, flow: np.ndarray, color,
               stride: int = 1, visible: bool = True, line_width: float = 2.0):
        idx = np.arange(0, len(points), max(1, int(stride)))
        segs = np.stack([points[idx], points[idx] + flow[idx]], axis=1).astype(np.float32)
        colors = np.tile(np.asarray(color, np.uint8), (len(idx), 2, 1))
        return self._keep(self.server.scene.add_line_segments(
            name, points=segs, colors=colors, line_width=line_width, visible=visible))

    def label(self, name: str, position, text: str):
        return self._keep(self.server.scene.add_label(name, text, position=position))


def viridis(values: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    return (cm.viridis(np.clip(values, 0.0, 1.0))[:, :3] * 255).astype(np.uint8)


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-12)
