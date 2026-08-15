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
class WAMChunkRef:
    shared_path: Path
    left_path: Path
    right_path: Path
    current: int


def relative_action_chunk(
    orient: np.ndarray,
    pose: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    """Return every future MANO state relative to chunk frame 0."""
    current_rotation = Rotation.from_rotvec(orient[0])
    relative_rotation = current_rotation.inv() * Rotation.from_rotvec(orient[1:])
    return np.concatenate(
        [
            transl[1:] - transl[0],
            relative_rotation.as_rotvec().astype(np.float32),
            pose[1:] - pose[0],
        ],
        axis=-1,
    ).astype(np.float32)


class GRABWAMChunkDataset(Dataset):
    """GRAB 10-frame joint world/action windows with broad interaction filtering."""

    def __init__(
        self,
        root: str,
        grab_raw_root: str,
        object_points: int = 1024,
        chunk_size: int = 10,
        debug_max_windows: Optional[int] = 128,
        require_interaction_in_window: bool = True,
    ) -> None:
        self.root = Path(root)
        self.grab_raw_root = Path(grab_raw_root)
        self.object_points = int(object_points)
        self.chunk_size = int(chunk_size)
        candidates = []
        self.raw_params = {}
        self.subjects = set()
        for shared_path in sorted(self.root.glob("*/*/shared.npz")):
            left_path = shared_path.with_name("left.npz")
            right_path = shared_path.with_name("right.npz")
            if not left_path.is_file() or not right_path.is_file():
                continue
            with np.load(shared_path, allow_pickle=False) as shared, np.load(
                left_path, allow_pickle=False
            ) as left, np.load(right_path, allow_pickle=False) as right:
                raw_frame_ids = np.asarray(shared["raw_frame_id"], dtype=np.int64)
                if not np.all(np.diff(raw_frame_ids) == 4):
                    raise ValueError(f"{shared_path}: chunk 要求连续 cache 帧 raw gap=4")
                subject = str(np.asarray(shared["subject_id"]).item())
                self.subjects.add(subject)
                left_active = np.asarray(left["obj_candidate_mask_5cm"], dtype=bool).any(1)
                right_active = np.asarray(right["obj_candidate_mask_5cm"], dtype=bool).any(1)
                active = left_active | right_active
                for current in range(1, len(raw_frame_ids) - self.chunk_size):
                    if require_interaction_in_window and not active[
                        current : current + self.chunk_size + 1
                    ].any():
                        continue
                    candidates.append(
                        WAMChunkRef(shared_path, left_path, right_path, current)
                    )
                source_raw_file = str(np.asarray(shared["source_raw_file"]).item())
                raw_path = self.grab_raw_root / source_raw_file
                if not raw_path.is_file():
                    raise FileNotFoundError(f"cache 对应的 GRAB raw 不存在: {raw_path}")
                with np.load(raw_path, allow_pickle=True) as raw:
                    per_side = {}
                    for side, raw_key in (("left", "lhand"), ("right", "rhand")):
                        hand = raw[raw_key].item()
                        params = hand["params"]
                        betas = hand.get("betas")
                        if not isinstance(betas, np.ndarray):
                            betas = np.zeros(10, dtype=np.float32)
                        per_side[side] = {
                            "global_orient": np.asarray(
                                params["global_orient"], np.float32
                            ),
                            "hand_pose": np.asarray(params["hand_pose"], np.float32),
                            "transl": np.asarray(params["transl"], np.float32),
                            "betas": np.asarray(betas, np.float32),
                        }
                self.raw_params[str(shared_path)] = per_side
        if not candidates:
            raise FileNotFoundError("没有合法的双手 chunk window")
        self.full_window_count = len(candidates)
        if debug_max_windows and len(candidates) > debug_max_windows:
            selected = np.linspace(
                0, len(candidates) - 1, debug_max_windows, dtype=np.int64
            )
            candidates = [candidates[int(index)] for index in selected]
        self.windows = candidates
        with np.load(self.windows[0].shared_path, allow_pickle=False) as shared:
            self.obj_indices = fixed_point_indices(
                shared["obj_points_world"].shape[1], self.object_points
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ref = self.windows[index]
        current = ref.current
        frames = np.arange(current, current + self.chunk_size + 1)
        with np.load(ref.shared_path, allow_pickle=False) as shared, np.load(
            ref.left_path, allow_pickle=False
        ) as left, np.load(ref.right_path, allow_pickle=False) as right:
            previous_object = np.asarray(
                shared["obj_points_world"][current - 1, self.obj_indices], np.float32
            )
            objects = np.asarray(
                shared["obj_points_world"][frames][:, self.obj_indices], np.float32
            )
            object_normals = np.asarray(
                shared["obj_normals_world"][current, self.obj_indices], np.float32
            )
            raw_ids = np.asarray(shared["raw_frame_id"][frames], dtype=np.int64)
            subject = str(np.asarray(shared["subject_id"]).item())
            sequence = str(np.asarray(shared["seq_id"]).item())
            hand_data = {}
            for side, cache in (("left", left), ("right", right)):
                hand_data[side] = {
                    "points": np.asarray(
                        cache["hand_points_world"][frames], np.float32
                    ),
                    "normals": np.asarray(
                        cache["hand_normals_world"][current], np.float32
                    ),
                    "canonical": np.asarray(cache["hand_cano_points"], np.float32),
                    "finger_id": np.asarray(cache["hand_finger_id"], np.int64),
                    "region_id": np.asarray(cache["hand_region_id"], np.int64),
                }
        center = objects[0].mean(axis=0, keepdims=True)
        objects_centered = objects - center
        result = {
            "object_points": torch.from_numpy(objects_centered[0]),
            "object_normals": torch.from_numpy(object_normals),
            "prev_object_flow": torch.from_numpy(objects[0] - previous_object),
            "world_chunk": torch.from_numpy(objects_centered[1:] - objects_centered[0]),
            "raw_frame_id": torch.tensor(int(raw_ids[0]), dtype=torch.int64),
            "sequence": sequence,
            "subject_id": subject,
        }
        params = self.raw_params[str(ref.shared_path)]
        for side in ("left", "right"):
            hand = hand_data[side]
            hand_points = hand["points"] - center
            hand_params = params[side]
            orientations = hand_params["global_orient"][raw_ids]
            poses = hand_params["hand_pose"][raw_ids]
            translations = hand_params["transl"][raw_ids]
            betas = hand_params["betas"]
            if betas.ndim > 1:
                betas = betas[raw_ids[0]]
            result.update(
                {
                    f"{side}_hand_points": torch.from_numpy(hand_points[0]),
                    f"{side}_hand_chunk": torch.from_numpy(hand_points[1:]),
                    f"{side}_hand_normals": torch.from_numpy(hand["normals"]),
                    f"{side}_hand_cano_points": torch.from_numpy(hand["canonical"]),
                    f"{side}_finger_id": torch.from_numpy(hand["finger_id"]),
                    f"{side}_region_id": torch.from_numpy(hand["region_id"]),
                    f"{side}_global_orient": torch.from_numpy(
                        orientations[0].copy()
                    ),
                    f"{side}_hand_pose": torch.from_numpy(poses[0].copy()),
                    f"{side}_transl": torch.from_numpy(
                        (translations[0] - center.squeeze(0)).copy()
                    ),
                    f"{side}_betas": torch.from_numpy(
                        np.asarray(betas, np.float32).copy()
                    ),
                    f"{side}_action_chunk": torch.from_numpy(
                        relative_action_chunk(orientations, poses, translations)
                    ),
                }
            )
        return result

    def statistics(self) -> Dict[str, torch.Tensor]:
        actions = {side: [] for side in ("left", "right")}
        world_sum = torch.zeros(self.chunk_size, 3, dtype=torch.float64)
        world_square_sum = torch.zeros(self.chunk_size, 3, dtype=torch.float64)
        world_count = torch.zeros(self.chunk_size, dtype=torch.float64)
        for index in range(len(self)):
            item = self[index]
            world = item["world_chunk"].double()
            world_sum += world.sum(1)
            world_square_sum += world.square().sum(1)
            world_count += world.shape[1]
            for side in actions:
                actions[side].append(item[f"{side}_action_chunk"])
        result = {}
        world_mean = world_sum / world_count[:, None]
        world_var = world_square_sum / world_count[:, None] - world_mean.square()
        result["world_mean"] = world_mean[:, None].float()
        result["world_std"] = (
            world_var.clamp_min(0).sqrt()[:, None].float().clamp_min(1e-5)
        )
        for side, values in actions.items():
            stacked = torch.stack(values, dim=0)
            result[f"{side}_mean"] = stacked.mean(0)
            result[f"{side}_std"] = stacked.std(0).clamp_min(1e-4)
        return result


def build_wam_chunk_dataset(data_cfg) -> GRABWAMChunkDataset:
    return GRABWAMChunkDataset(**vars(data_cfg))
