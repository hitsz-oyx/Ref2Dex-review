from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.task.correspondence_ptv3.contracts import EdgeSample


@dataclass(frozen=True)
class DenseEdgeSampler:

    def sample(
        self,
        *,
        gt_obj_points: np.ndarray,
        gt_hand_points: np.ndarray,
        obj_valid: np.ndarray,
        seed: int,
    ) -> EdgeSample:
        del seed
        num_obj = gt_obj_points.shape[0]
        num_hand = gt_hand_points.shape[0]
        all_hand_idx = np.arange(num_hand, dtype=np.int64)
        logit_idx = np.broadcast_to(all_hand_idx[None, :], (num_obj, num_hand)).copy()
        logit_valid = np.broadcast_to(
            np.asarray(obj_valid, dtype=bool)[:, None],
            (num_obj, num_hand),
        ).copy()
        logit_weight = logit_valid.astype(np.float32)
        logit_idx[~logit_valid] = -1
        return EdgeSample(
            idx=logit_idx,
            valid_mask=logit_valid,
            loss_weight=logit_weight,
        )
