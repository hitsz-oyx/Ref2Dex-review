from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import fixed_point_indices


@dataclass(frozen=True)
class TransitionRef:
    shared_path: Path
    right_path: Path
    current: int
    gap: int
    motion: float


class GRABOneStepDataset(Dataset):
    """只保留右手可解释、左手不参与的 GRAB transition。"""

    def __init__(
        self,
        root: str,
        sequences: List[str],
        object_points: int = 1024,
        hand_points: int = 256,
        gaps: List[int] = (1,),
        max_transitions: int = 64,
    ) -> None:
        self.root = Path(root)
        self.object_points = object_points
        self.hand_points = hand_points
        self.gaps = tuple(int(gap) for gap in gaps)
        if not self.gaps or min(self.gaps) < 1:
            raise ValueError("gaps 必须为正整数")
        candidates: List[TransitionRef] = []
        for sequence in sequences:
            directory = self.root / sequence
            shared_path = directory / "shared.npz"
            right_path = directory / "right.npz"
            left_path = directory / "left.npz"
            if not shared_path.is_file() or not right_path.is_file() or not left_path.is_file():
                raise FileNotFoundError(f"transition cache 不完整: {directory}")
            with np.load(shared_path, allow_pickle=False) as shared, np.load(
                right_path, allow_pickle=False
            ) as right, np.load(left_path, allow_pickle=False) as left:
                obj = np.asarray(shared["obj_points_world"], dtype=np.float32)
                raw_frame_id = np.asarray(shared["raw_frame_id"], dtype=np.int64)
                right_active = np.asarray(right["obj_candidate_mask_5cm"], dtype=bool).any(1)
                left_active = np.asarray(left["obj_candidate_mask_5cm"], dtype=bool).any(1)
                source_fps = float(np.asarray(shared["source_fps"]).item())
                if source_fps != 120.0:
                    raise ValueError(f"{shared_path}: source_fps={source_fps}，预期 120")
                for gap in self.gaps:
                    for current in range(len(obj) - gap):
                        future = current + gap
                        if not (
                            right_active[current]
                            and right_active[future]
                            and not left_active[current]
                            and not left_active[future]
                        ):
                            continue
                        raw_gap = int(raw_frame_id[future] - raw_frame_id[current])
                        if raw_gap != 4 * gap:
                            raise ValueError(
                                f"{shared_path}: cache gap={gap} 对应 raw gap={raw_gap}，预期 {4 * gap}"
                            )
                        motion = float(
                            np.linalg.norm(obj[future] - obj[current], axis=-1).mean()
                        )
                        candidates.append(
                            TransitionRef(shared_path, right_path, current, gap, motion)
                        )
        candidates.sort(
            key=lambda ref: (-ref.motion, str(ref.shared_path), ref.current, ref.gap)
        )
        self.transitions = candidates[:max_transitions] if max_transitions else candidates
        if not self.transitions:
            raise FileNotFoundError("没有满足 right-active/left-inactive 的 transition")
        with np.load(self.transitions[0].shared_path, allow_pickle=False) as shared, np.load(
            self.transitions[0].right_path, allow_pickle=False
        ) as right:
            self.obj_indices = fixed_point_indices(
                shared["obj_points_world"].shape[1], object_points
            )
            self.hand_indices = fixed_point_indices(
                right["hand_points_world"].shape[1], hand_points
            )

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ref = self.transitions[index]
        future = ref.current + ref.gap
        with np.load(ref.shared_path, allow_pickle=False) as shared, np.load(
            ref.right_path, allow_pickle=False
        ) as right:
            obj_current = np.asarray(
                shared["obj_points_world"][ref.current, self.obj_indices], np.float32
            )
            obj_next = np.asarray(
                shared["obj_points_world"][future, self.obj_indices], np.float32
            )
            obj_normals = np.asarray(
                shared["obj_normals_world"][ref.current, self.obj_indices], np.float32
            )
            hand_current = np.asarray(
                right["hand_points_world"][ref.current, self.hand_indices], np.float32
            )
            hand_next = np.asarray(
                right["hand_points_world"][future, self.hand_indices], np.float32
            )
            hand_normals = np.asarray(
                right["hand_normals_world"][ref.current, self.hand_indices], np.float32
            )
            raw_current = int(shared["raw_frame_id"][ref.current])
            raw_future = int(shared["raw_frame_id"][future])
            source_fps = float(np.asarray(shared["source_fps"]).item())
            sequence = str(np.asarray(shared["seq_id"]).item())
        center = obj_current.mean(axis=0, keepdims=True)
        obj_current = obj_current - center
        obj_next = obj_next - center
        hand_current = hand_current - center
        hand_next = hand_next - center
        return {
            "object_points": torch.from_numpy(obj_current),
            "object_normals": torch.from_numpy(obj_normals),
            "hand_points": torch.from_numpy(hand_current),
            "hand_normals": torch.from_numpy(hand_normals),
            "hand_flow": torch.from_numpy(hand_next - hand_current),
            "object_flow": torch.from_numpy(obj_next - obj_current),
            "dt": torch.tensor((raw_future - raw_current) / source_fps, dtype=torch.float32),
            "gap": torch.tensor(ref.gap, dtype=torch.int64),
            "raw_frame_id": torch.tensor(raw_current, dtype=torch.int64),
            "next_raw_frame_id": torch.tensor(raw_future, dtype=torch.int64),
            "motion": torch.tensor(ref.motion, dtype=torch.float32),
            "sequence": sequence,
        }


def build_one_step_dataset(data_cfg) -> GRABOneStepDataset:
    return GRABOneStepDataset(**vars(data_cfg))
