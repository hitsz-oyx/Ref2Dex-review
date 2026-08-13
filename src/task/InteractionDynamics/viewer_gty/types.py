"""与模型、MANO 和 URDF 解耦的三路对比数据类型。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComparisonTrajectory:
    name: str
    source_raw_file: str
    object_name: str
    subject: str
    raw_frame_ids: np.ndarray
    object_vertices: np.ndarray
    object_faces: np.ndarray
    gt_vertices: np.ndarray
    pred_vertices: np.ndarray
    mano_faces: np.ndarray
    anchors: np.ndarray
    gt_y: np.ndarray
    pred_y: np.ndarray
    pred_metrics: dict
    robot_name: str
    robot_vertices: np.ndarray
    robot_faces: np.ndarray
    robot_available: np.ndarray
    robot_metrics: dict

    @property
    def frames(self) -> int:
        return len(self.raw_frame_ids)
