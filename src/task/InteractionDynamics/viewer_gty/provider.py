"""读取离线三路 comparison cache，不执行模型或优化。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.task.InteractionDynamics.viewer_gty.types import ComparisonTrajectory


class GTYComparisonProvider:
    def __init__(self, cache: str | Path) -> None:
        self.paths = sorted(Path(cache).glob("*.pt"))
        if not self.paths:
            raise ValueError(f"没有 V20.5 comparison cache: {cache}")
        self._headers = [torch.load(path, map_location="cpu") for path in self.paths]
        self.names = tuple(f"{row['object_name']} | {row['subject']} | {row['source_raw_file']}"
                           for row in self._headers)
        robots = set.intersection(*(set(row["robot_results"]) for row in self._headers))
        self.robots = tuple(sorted(robots))

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int, robot: str) -> ComparisonTrajectory:
        row = self._headers[index]; result = row["robot_results"][robot]
        array = lambda value: value.numpy() if torch.is_tensor(value) else np.asarray(value)
        return ComparisonTrajectory(self.names[index], row["source_raw_file"], row["object_name"],
            row["subject"], array(row["raw_frame_ids"]), array(row["object_vertices"]),
            array(row["object_faces"]), array(row["gt_vertices_object"]),
            array(row["pred_vertices_object"]), array(row["mano_faces"]), array(row["anchors"]),
            array(row["gt_y"]), array(row["pred_y"]), row["pred_metrics"], robot,
            array(result["vertices_object"]),
            array(result["faces"]), array(result["available"]).astype(bool), result["metrics"])
