from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.task.correspondence_ptv3.contracts import EdgeSample


@dataclass(frozen=True)
class DenseEdgeSampler:
    logit_near_radius: float
    logit_far_min_radius: float

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

        near_count = np.zeros((num_obj,), dtype=np.int64)
        far_count = np.zeros((num_obj,), dtype=np.int64)
        valid_obj_idx = np.flatnonzero(obj_valid)
        if valid_obj_idx.size == 0:
            return EdgeSample(
                idx=logit_idx,
                valid_mask=logit_valid,
                loss_weight=logit_weight,
                stats={"near_count": near_count, "far_count": far_count},
            )

        obj = torch.from_numpy(np.asarray(gt_obj_points[valid_obj_idx], dtype=np.float32))
        hand = torch.from_numpy(np.asarray(gt_hand_points, dtype=np.float32))
        distance = torch.cdist(obj, hand).numpy()
        near_count[valid_obj_idx] = np.count_nonzero(
            distance <= float(self.logit_near_radius),
            axis=-1,
        ).astype(np.int64)
        far_count[valid_obj_idx] = np.count_nonzero(
            distance > float(self.logit_far_min_radius),
            axis=-1,
        ).astype(np.int64)
        return EdgeSample(
            idx=logit_idx,
            valid_mask=logit_valid,
            loss_weight=logit_weight,
            stats={"near_count": near_count, "far_count": far_count},
        )
