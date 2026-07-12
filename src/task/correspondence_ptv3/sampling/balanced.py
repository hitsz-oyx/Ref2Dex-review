from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.task.correspondence_ptv3.contracts import EdgeSample


@dataclass(frozen=True)
class BalancedEdgeSampler:
    k_near_logit: int
    k_far_logit: int
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
        k_logit = int(self.k_near_logit) + int(self.k_far_logit)
        num_obj = gt_obj_points.shape[0]
        logit_idx = np.full((num_obj, k_logit), -1, dtype=np.int64)
        logit_valid = np.zeros((num_obj, k_logit), dtype=bool)
        logit_weight = np.zeros((num_obj, k_logit), dtype=np.float32)
        valid_obj_idx = np.flatnonzero(obj_valid)
        if valid_obj_idx.size == 0:
            return EdgeSample(
                idx=logit_idx,
                valid_mask=logit_valid,
                loss_weight=logit_weight,
            )

        obj = torch.from_numpy(np.asarray(gt_obj_points[valid_obj_idx], dtype=np.float32))
        hand = torch.from_numpy(np.asarray(gt_hand_points, dtype=np.float32))
        distance = torch.cdist(obj, hand).numpy()
        rng = np.random.default_rng(seed)
        for row, obj_idx in enumerate(valid_obj_idx.tolist()):
            dist_row = distance[row]
            near_candidates = np.flatnonzero(dist_row <= float(self.logit_near_radius))
            if near_candidates.size > 0:
                n_near = min(int(self.k_near_logit), int(near_candidates.size))
                chosen = rng.choice(near_candidates, size=int(n_near), replace=False)
                logit_idx[obj_idx, :n_near] = chosen
                logit_valid[obj_idx, :n_near] = True
                logit_weight[obj_idx, :n_near] = 1.0

            far_candidates = np.flatnonzero(dist_row > float(self.logit_far_min_radius))
            if far_candidates.size > 0:
                n_far = min(int(self.k_far_logit), int(far_candidates.size))
                chosen = rng.choice(far_candidates, size=int(n_far), replace=False)
                start = int(self.k_near_logit)
                logit_idx[obj_idx, start:start + n_far] = chosen
                logit_valid[obj_idx, start:start + n_far] = True
                logit_weight[obj_idx, start:start + n_far] = 1.0
        return EdgeSample(
            idx=logit_idx,
            valid_mask=logit_valid,
            loss_weight=logit_weight,
        )
