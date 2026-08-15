from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .dataset import fixed_point_indices


@dataclass(frozen=True)
class WAMTransitionRef:
    shared_path: Path
    left_path: Path
    right_path: Path
    current: int


def relative_action(
    current_orient: np.ndarray,
    next_orient: np.ndarray,
    current_pose: np.ndarray,
    next_pose: np.ndarray,
    current_transl: np.ndarray,
    next_transl: np.ndarray,
) -> np.ndarray:
    relative_rotation = Rotation.from_rotvec(current_orient).inv() * Rotation.from_rotvec(
        next_orient
    )
    return np.concatenate(
        [
            next_transl - current_transl,
            relative_rotation.as_rotvec().astype(np.float32),
            next_pose - current_pose,
        ]
    ).astype(np.float32)


class GRABWAMDataset(Dataset):
    """完整双手相邻 transition；至少一只手在当前帧接近物体。"""

    def __init__(
        self,
        root: str,
        grab_raw_root: str,
        object_points: int = 1024,
        debug_max_transitions: Optional[int] = 128,
    ) -> None:
        self.root = Path(root)
        self.grab_raw_root = Path(grab_raw_root)
        self.object_points = object_points
        candidates = []
        self.raw_params = {}
        for shared_path in sorted(self.root.glob("*/*/shared.npz")):
            left_path = shared_path.with_name("left.npz")
            right_path = shared_path.with_name("right.npz")
            if not left_path.is_file() or not right_path.is_file():
                continue
            with np.load(shared_path, allow_pickle=False) as shared, np.load(
                left_path, allow_pickle=False
            ) as left, np.load(right_path, allow_pickle=False) as right:
                raw_frame_ids = np.asarray(shared["raw_frame_id"], dtype=np.int64)
                source_raw_file = str(np.asarray(shared["source_raw_file"]).item())
                raw_path = self.grab_raw_root / source_raw_file
                if not raw_path.is_file():
                    raise FileNotFoundError(f"cache 对应的 GRAB raw 不存在: {raw_path}")
                left_active = np.asarray(left["obj_candidate_mask_5cm"], dtype=bool).any(1)
                right_active = np.asarray(right["obj_candidate_mask_5cm"], dtype=bool).any(1)
                if not np.all(np.diff(raw_frame_ids) == 4):
                    raise ValueError(f"{shared_path}: V1 要求连续 cache 帧 raw gap=4")
                for current in range(1, len(raw_frame_ids) - 1):
                    if left_active[current] or right_active[current]:
                        candidates.append(
                            WAMTransitionRef(shared_path, left_path, right_path, current)
                        )
                raw = np.load(raw_path, allow_pickle=True)
                per_side = {}
                for side, raw_key in (("left", "lhand"), ("right", "rhand")):
                    hand = raw[raw_key].item()
                    params = hand["params"]
                    betas = hand.get("betas")
                    if not isinstance(betas, np.ndarray):
                        betas = np.zeros(10, dtype=np.float32)
                    per_side[side] = {
                        "global_orient": np.asarray(params["global_orient"], np.float32),
                        "hand_pose": np.asarray(params["hand_pose"], np.float32),
                        "transl": np.asarray(params["transl"], np.float32),
                        "betas": np.asarray(betas, np.float32),
                    }
                self.raw_params[str(shared_path)] = per_side
                raw.close()
        if not candidates:
            raise FileNotFoundError("没有合法的双手 interaction transition")
        if debug_max_transitions and len(candidates) > debug_max_transitions:
            selected = np.linspace(
                0, len(candidates) - 1, debug_max_transitions, dtype=np.int64
            )
            candidates = [candidates[int(index)] for index in selected]
        self.transitions = candidates
        with np.load(self.transitions[0].shared_path, allow_pickle=False) as shared:
            self.obj_indices = fixed_point_indices(
                shared["obj_points_world"].shape[1], object_points
            )

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ref = self.transitions[index]
        previous, current, future = ref.current - 1, ref.current, ref.current + 1
        with np.load(ref.shared_path, allow_pickle=False) as shared, np.load(
            ref.left_path, allow_pickle=False
        ) as left, np.load(ref.right_path, allow_pickle=False) as right:
            obj = np.asarray(
                shared["obj_points_world"][[previous, current, future]][
                    :, self.obj_indices
                ],
                np.float32,
            )
            obj_normals = np.asarray(
                shared["obj_normals_world"][current, self.obj_indices], np.float32
            )
            raw_ids = np.asarray(
                shared["raw_frame_id"][[current, future]], dtype=np.int64
            )
            sequence = str(np.asarray(shared["seq_id"]).item())
            hands = {}
            for side, cache in (("left", left), ("right", right)):
                hands[side] = {
                    "points": np.asarray(
                        cache["hand_points_world"][[current, future]], np.float32
                    ),
                    "normals": np.asarray(cache["hand_normals_world"][current], np.float32),
                }
        center = obj[1].mean(axis=0, keepdims=True)
        obj = obj - center
        result = {
            "object_points": torch.from_numpy(obj[1]),
            "object_normals": torch.from_numpy(obj_normals),
            "prev_object_flow": torch.from_numpy(obj[1] - obj[0]),
            "target_object_flow": torch.from_numpy(obj[2] - obj[1]),
            "raw_frame_id": torch.tensor(int(raw_ids[0]), dtype=torch.int64),
            "next_raw_frame_id": torch.tensor(int(raw_ids[1]), dtype=torch.int64),
            "sequence": sequence,
        }
        params = self.raw_params[str(ref.shared_path)]
        for side in ("left", "right"):
            current_raw, future_raw = int(raw_ids[0]), int(raw_ids[1])
            hand_params = params[side]
            current_orient = hand_params["global_orient"][current_raw]
            next_orient = hand_params["global_orient"][future_raw]
            current_pose = hand_params["hand_pose"][current_raw]
            next_pose = hand_params["hand_pose"][future_raw]
            current_transl = hand_params["transl"][current_raw]
            next_transl = hand_params["transl"][future_raw]
            points = hands[side]["points"] - center
            betas = hand_params["betas"]
            if betas.ndim > 1:
                betas = betas[current_raw]
            result.update(
                {
                    f"{side}_hand_points": torch.from_numpy(points[0]),
                    f"{side}_next_hand_points": torch.from_numpy(points[1]),
                    f"{side}_hand_normals": torch.from_numpy(hands[side]["normals"]),
                    f"{side}_global_orient": torch.from_numpy(current_orient.copy()),
                    f"{side}_hand_pose": torch.from_numpy(current_pose.copy()),
                    f"{side}_transl": torch.from_numpy(
                        (current_transl - center.squeeze(0)).copy()
                    ),
                    f"{side}_betas": torch.from_numpy(np.asarray(betas, np.float32).copy()),
                    f"{side}_action": torch.from_numpy(
                        relative_action(
                            current_orient,
                            next_orient,
                            current_pose,
                            next_pose,
                            current_transl,
                            next_transl,
                        )
                    ),
                }
            )
        return result

    def action_statistics(self) -> Dict[str, torch.Tensor]:
        actions = {side: [] for side in ("left", "right")}
        for index in range(len(self)):
            item = self[index]
            for side in actions:
                actions[side].append(item[f"{side}_action"])
        result = {}
        for side, values in actions.items():
            stacked = torch.stack(values)
            result[f"{side}_mean"] = stacked.mean(0)
            result[f"{side}_std"] = stacked.std(0).clamp_min(1e-4)
        return result


def build_wam_dataset(data_cfg) -> GRABWAMDataset:
    return GRABWAMDataset(**vars(data_cfg))
