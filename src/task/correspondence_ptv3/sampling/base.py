from __future__ import annotations

from typing import Protocol

import numpy as np

from src.task.correspondence_ptv3.contracts import EdgeSample


class EdgeSampler(Protocol):
    def sample(
        self,
        *,
        gt_obj_points: np.ndarray,
        gt_hand_points: np.ndarray,
        obj_valid: np.ndarray,
        seed: int,
    ) -> EdgeSample:
        ...
