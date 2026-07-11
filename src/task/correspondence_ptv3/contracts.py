from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Stage3FrameRef:
    path: str
    frame_idx: int


@dataclass(frozen=True)
class Stage3Frame:
    seq_id: str
    side: str
    raw_frame_id: int
    obj_points: np.ndarray
    obj_normals: np.ndarray
    obj_point_id: np.ndarray
    hand_points: np.ndarray
    hand_normals: np.ndarray
    hand_point_id: np.ndarray
    obj_to_hand_min_dist: np.ndarray
    obj_candidate_mask_5cm: np.ndarray
    gt_obj_to_hand_knn_idx: np.ndarray
    hand_cano_points: np.ndarray
    hand_finger_id: np.ndarray
    hand_region_id: np.ndarray


@dataclass(frozen=True)
class EdgeSample:
    idx: np.ndarray
    valid_mask: np.ndarray
    loss_weight: np.ndarray
    stats: dict[str, np.ndarray]


@dataclass(frozen=True)
class ContactLossResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
