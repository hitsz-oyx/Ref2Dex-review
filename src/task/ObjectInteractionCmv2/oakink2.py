"""Small read-only OakInk2 Stage3 adapter for the V1.0 rigid pilot."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class OakInk2RigidDataset(Dataset):
    """Loads adjacent frames from existing single-object object-centered NPZ files."""

    def __init__(self, root: str | Path, *, max_files: int = 4, max_frames: int = 32,
                 object_points: int = 1024, hand_points: int = 1538) -> None:
        self.root = Path(root).expanduser().resolve()
        paths = sorted(self.root.glob("*.npz"))[: int(max_files)]
        if not paths:
            raise FileNotFoundError(f"No OakInk2 NPZ files under {self.root}")
        self.records: List[tuple[Path, int]] = []
        self.object_points, self.hand_points = int(object_points), int(hand_points)
        self._cache = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                frames = min(len(z["raw_frame_id"]) - 1, int(max_frames))
            self.records.extend((path, i) for i in range(max(0, frames)))
        if not self.records:
            raise ValueError("OakInk2 files contain no adjacent frames")

    def __len__(self):
        return len(self.records)

    def _load(self, path):
        if path not in self._cache:
            self._cache[path] = np.load(path, allow_pickle=False)
        return self._cache[path]

    def __getitem__(self, index):
        path, frame = self.records[int(index)]
        z = self._load(path)
        points = np.asarray(z["obj_points"][: self.object_points], dtype=np.float32)
        normals = np.asarray(z["obj_normals"][: self.object_points], dtype=np.float32)
        hand = np.asarray(z["hand_points"][frame, : self.hand_points], dtype=np.float32)
        hand_next = np.asarray(z["hand_points"][frame + 1, : self.hand_points], dtype=np.float32)
        hand_normals = np.asarray(z["hand_normals"][frame, : self.hand_points], dtype=np.float32)
        pose_t = np.asarray(z["obj_root_pose_world"][frame], dtype=np.float32)
        pose_next = np.asarray(z["obj_root_pose_world"][frame + 1], dtype=np.float32)
        relative = np.linalg.inv(pose_t) @ pose_next
        homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
        future = (homogeneous @ relative.T)[:, :3]
        return {
            "obj_points": torch.from_numpy(points), "obj_normals": torch.from_numpy(normals),
            "hand_points": torch.from_numpy(hand), "hand_normals": torch.from_numpy(hand_normals),
            "hand_flow": torch.from_numpy(hand_next - hand), "obj_flow_gt": torch.from_numpy(future - points),
            "hand_valid_mask": torch.ones(len(hand), dtype=torch.bool),
            "obj_link_id": torch.zeros(len(points), dtype=torch.long), "num_links": torch.tensor(1),
            "raw_frame_id": torch.tensor(int(z["raw_frame_id"][frame])), "source_file": str(path),
        }
