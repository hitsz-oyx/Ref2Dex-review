"""与具体手参数化解耦的 viewer 数据接口。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class HandFrame:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class ObjectTrajectory:
    vertices: np.ndarray
    faces: np.ndarray
    poses: np.ndarray

    def world_vertices(self, frame: int) -> np.ndarray:
        pose = self.poses[frame]
        return self.vertices @ pose[:3, :3].T + pose[:3, 3]


@dataclass(frozen=True)
class VisualizationSample:
    object_trajectory: ObjectTrajectory
    pred_hand: list[HandFrame]
    gt_hand: Optional[list[HandFrame]]
    grasp_proxy: Optional[bool]
    collision: Optional[bool] = None
    max_penetration_mm: Optional[float] = None
    label: str = ""


class HandBackend(Protocol):
    name: str

    def decode_prediction(self, sample: dict, prediction: object) -> list[HandFrame]: ...
    def decode_gt(self, sample: dict) -> Optional[list[HandFrame]]: ...


class ManoBackend:
    """第一版 MANO backend；接收 provider 已恢复的 mesh tensor。"""
    name = "MANO"

    @staticmethod
    def _frames(vertices: np.ndarray, faces: np.ndarray) -> list[HandFrame]:
        return [HandFrame(np.asarray(frame, np.float32), np.asarray(faces, np.int32))
                for frame in vertices]

    def decode_prediction(self, sample: dict, prediction: object) -> list[HandFrame]:
        vertices, faces = prediction
        return self._frames(vertices, faces)

    def decode_gt(self, sample: dict) -> Optional[list[HandFrame]]:
        vertices, faces = sample["gt_mesh"]
        return self._frames(vertices, faces)
