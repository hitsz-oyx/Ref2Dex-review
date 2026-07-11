from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.task.correspondence_ptv3.contracts import EdgeSample


def validate_stratified_edge_sampler_config(
    distance_edges: tuple[float, ...] | list[float],
    quotas: tuple[int, ...] | list[int],
) -> tuple[np.ndarray, np.ndarray]:
    edges = tuple(float(value) for value in distance_edges)
    quota_values = tuple(int(value) for value in quotas)
    if len(edges) != 4:
        raise ValueError("logit_stratified_distance_edges must contain exactly 4 values.")
    if len(quota_values) != len(edges) + 1:
        raise ValueError("logit_stratified_quotas must have len(distance_edges) + 1 entries.")
    if sum(quota_values) != 128:
        raise ValueError("logit_stratified_quotas must sum to 128.")
    if any(quota < 0 for quota in quota_values):
        raise ValueError("logit_stratified_quotas must be non-negative.")
    if any(edge <= 0.0 for edge in edges) or any(
        curr <= prev for prev, curr in zip(edges[:-1], edges[1:])
    ):
        raise ValueError("logit_stratified_distance_edges must be positive and strictly increasing.")
    return np.asarray(edges, dtype=np.float32), np.asarray(quota_values, dtype=np.int64)


def stratified_distance_bucket_candidates(
    dist_row: np.ndarray,
    distance_edges: np.ndarray,
) -> list[np.ndarray]:
    edge0, edge1, edge2, edge3 = (float(value) for value in distance_edges)
    return [
        np.flatnonzero(dist_row <= edge0),
        np.flatnonzero((dist_row > edge0) & (dist_row <= edge1)),
        np.flatnonzero((dist_row > edge1) & (dist_row < edge2)),
        np.flatnonzero((dist_row >= edge2) & (dist_row <= edge3)),
        np.flatnonzero(dist_row > edge3),
    ]


@dataclass(frozen=True)
class StratifiedEdgeSampler:
    distance_edges: tuple[float, ...] | list[float]
    quotas: tuple[int, ...] | list[int]
    logit_near_radius: float
    logit_far_min_radius: float

    def __post_init__(self) -> None:
        edges, quotas = validate_stratified_edge_sampler_config(
            self.distance_edges,
            self.quotas,
        )
        object.__setattr__(self, "distance_edges", tuple(float(value) for value in edges.tolist()))
        object.__setattr__(self, "quotas", tuple(int(value) for value in quotas.tolist()))

    def sample(
        self,
        *,
        gt_obj_points: np.ndarray,
        gt_hand_points: np.ndarray,
        obj_valid: np.ndarray,
        seed: int,
    ) -> EdgeSample:
        edges, quota_values = validate_stratified_edge_sampler_config(
            self.distance_edges,
            self.quotas,
        )
        k_logit = int(quota_values.sum())
        num_obj = int(gt_obj_points.shape[0])
        logit_idx = np.full((num_obj, k_logit), -1, dtype=np.int64)
        logit_valid = np.zeros((num_obj, k_logit), dtype=bool)
        logit_weight = np.zeros((num_obj, k_logit), dtype=np.float32)
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
        rng = np.random.default_rng(seed)
        for row, obj_idx in enumerate(valid_obj_idx.tolist()):
            selected: list[int] = []
            remaining_parts: list[np.ndarray] = []
            dist_row = distance[row]
            for layer_idx, candidates in enumerate(
                stratified_distance_bucket_candidates(dist_row, edges)
            ):
                shuffled = rng.permutation(candidates)
                take_count = min(int(quota_values[layer_idx]), int(shuffled.size))
                selected.extend(int(value) for value in shuffled[:take_count])
                remaining_parts.append(np.asarray(shuffled[take_count:], dtype=np.int64))

            offsets = [0] * len(remaining_parts)

            def refill_round_robin(layer_indices: tuple[int, ...]) -> None:
                while len(selected) < k_logit:
                    progress = False
                    for layer_idx in layer_indices:
                        offset = offsets[layer_idx]
                        remaining = remaining_parts[layer_idx]
                        if offset >= int(remaining.size):
                            continue
                        selected.append(int(remaining[offset]))
                        offsets[layer_idx] += 1
                        progress = True
                        if len(selected) >= k_logit:
                            break
                    if not progress:
                        break

            refill_round_robin((1, 2, 3))
            if len(selected) < k_logit:
                refill_round_robin((0,))
            if len(selected) < k_logit:
                refill_round_robin((4,))

            count = min(k_logit, len(selected))
            if count <= 0:
                continue
            chosen = np.asarray(selected[:count], dtype=np.int64)
            logit_idx[obj_idx, :count] = chosen
            logit_valid[obj_idx, :count] = True
            logit_weight[obj_idx, :count] = 1.0
            selected_dist = dist_row[chosen]
            near_count[obj_idx] = int(np.count_nonzero(selected_dist <= self.logit_near_radius))
            far_count[obj_idx] = int(np.count_nonzero(selected_dist > self.logit_far_min_radius))
        return EdgeSample(
            idx=logit_idx,
            valid_mask=logit_valid,
            loss_weight=logit_weight,
            stats={"near_count": near_count, "far_count": far_count},
        )
