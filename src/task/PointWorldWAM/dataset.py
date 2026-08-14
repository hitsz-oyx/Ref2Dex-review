from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def fixed_point_indices(pool_size: int, count: int) -> np.ndarray:
    if count > pool_size:
        raise ValueError(f"采样数 {count} 大于点池 {pool_size}")
    return np.linspace(0, pool_size - 1, count, dtype=np.int64)


@dataclass(frozen=True)
class WindowRef:
    shared_path: Path
    hand_path: Path
    start: int
    motion: float


class GRABWindowDataset(Dataset):
    """读取现有 GRAB sequence cache，保持跨时间固定 point ID。"""

    def __init__(
        self,
        root: str,
        horizon: int = 11,
        object_points: int = 1024,
        hand_points: int = 256,
        max_sequences: int = 2,
        sequence_offset: int = 0,
        max_windows: int = 64,
        window_stride: int = 2,
        side: str = "right",
    ) -> None:
        if horizon != 11:
            raise ValueError("PointWorld V0 固定 horizon=11")
        self.root = Path(root)
        self.horizon = horizon
        self.object_points = object_points
        self.hand_points = hand_points
        self.side = side
        shared_paths = sorted(self.root.glob("*/*/shared.npz"))
        candidates: List[Tuple[float, Path, Path, int]] = []
        sequence_scores: List[Tuple[float, Path, Path, List[Tuple[float, int]]]] = []
        for shared_path in shared_paths:
            hand_path = shared_path.with_name(f"{side}.npz")
            if not hand_path.is_file():
                continue
            with np.load(shared_path, allow_pickle=False) as shared:
                obj = np.asarray(shared["obj_points_world"], dtype=np.float32)
            if len(obj) < horizon:
                continue
            per_sequence: List[Tuple[float, int]] = []
            for start in range(0, len(obj) - horizon + 1, window_stride):
                endpoint = np.linalg.norm(obj[start + horizon - 1] - obj[start], axis=-1)
                per_sequence.append((float(endpoint.mean()), start))
            sequence_scores.append((max(score for score, _ in per_sequence), shared_path, hand_path, per_sequence))

        # Overfit 阶段优先选择实际发生刚体运动的 sequence/window，避免静态解占优。
        sequence_scores.sort(key=lambda item: (-item[0], str(item[1])))
        sequence_stop = sequence_offset + max_sequences if max_sequences else None
        for _, shared_path, hand_path, per_sequence in sequence_scores[sequence_offset:sequence_stop]:
            for motion, start in per_sequence:
                candidates.append((motion, shared_path, hand_path, start))
        candidates.sort(key=lambda item: (-item[0], str(item[1]), item[3]))
        if max_windows:
            candidates = candidates[:max_windows]
        self.windows = [WindowRef(shared, hand, start, motion) for motion, shared, hand, start in candidates]
        if not self.windows:
            raise FileNotFoundError(f"{self.root} 下没有可用的 {side} hand 11 帧窗口")

        with np.load(self.windows[0].shared_path, allow_pickle=False) as shared, np.load(
            self.windows[0].hand_path, allow_pickle=False
        ) as hand:
            self.obj_indices = fixed_point_indices(shared["obj_points_world"].shape[1], object_points)
            self.hand_indices = fixed_point_indices(hand["hand_points_world"].shape[1], hand_points)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ref = self.windows[index]
        sl = slice(ref.start, ref.start + self.horizon)
        with np.load(ref.shared_path, allow_pickle=False) as shared, np.load(
            ref.hand_path, allow_pickle=False
        ) as hand:
            obj = np.asarray(shared["obj_points_world"][sl][:, self.obj_indices], np.float32)
            obj_normals = np.asarray(shared["obj_normals_world"][sl][:, self.obj_indices], np.float32)
            hand_points = np.asarray(hand["hand_points_world"][sl][:, self.hand_indices], np.float32)
            hand_normals = np.asarray(hand["hand_normals_world"][sl][:, self.hand_indices], np.float32)
            obj_ids = np.asarray(shared["obj_point_id"][self.obj_indices], np.int64)
            hand_ids = np.asarray(hand["hand_point_id"][self.hand_indices], np.int64)
            raw_frame_id = np.asarray(shared["raw_frame_id"][sl], np.int64)
            seq_id = str(shared["seq_id"])

        c0 = obj[0].mean(axis=0, keepdims=True)
        obj = obj - c0
        hand_points = hand_points - c0
        return {
            "object_points": torch.from_numpy(obj),
            "object_normals": torch.from_numpy(obj_normals),
            "hand_points": torch.from_numpy(hand_points),
            "hand_normals": torch.from_numpy(hand_normals),
            "obj_point_id": torch.from_numpy(obj_ids),
            "hand_point_id": torch.from_numpy(hand_ids),
            "raw_frame_id": torch.from_numpy(raw_frame_id),
            "window_index": torch.tensor(index, dtype=torch.long),
            "motion": torch.tensor(ref.motion, dtype=torch.float32),
            "sequence": seq_id,
        }


def build_dataset(data_cfg) -> GRABWindowDataset:
    return GRABWindowDataset(**vars(data_cfg))
