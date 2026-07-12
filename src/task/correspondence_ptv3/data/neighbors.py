from __future__ import annotations

import numpy as np
import torch


def _compute_runtime_context_neighbors(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_ctx: int,
    ctx_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_obj = obj_points.shape[0]
    ctx_idx = np.full((num_obj, k_ctx), -1, dtype=np.int64)
    ctx_valid = np.zeros((num_obj, k_ctx), dtype=bool)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return ctx_idx, ctx_valid
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    distance = torch.cdist(obj, hand).numpy()
    for row, obj_idx in enumerate(valid_obj_idx.tolist()):
        dist_row = distance[row]
        ctx_candidates = np.flatnonzero(dist_row <= float(ctx_radius))
        if ctx_candidates.size <= 0:
            continue
        order = np.argsort(dist_row[ctx_candidates], kind="stable")
        chosen_ctx = ctx_candidates[order[:k_ctx]]
        ctx_count = int(chosen_ctx.size)
        ctx_idx[obj_idx, :ctx_count] = chosen_ctx
        ctx_valid[obj_idx, :ctx_count] = True
    return ctx_idx, ctx_valid


def _compute_input_knn(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_cross: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_obj = obj_points.shape[0]
    result = np.full((num_obj, k_cross), -1, dtype=np.int64)
    valid = np.zeros((num_obj, k_cross), dtype=bool)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return result, valid
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    topk = min(int(k_cross), int(hand.shape[0]))
    idx = torch.topk(torch.cdist(obj, hand), k=topk, dim=-1, largest=False).indices.numpy()
    result[valid_obj_idx, :topk] = idx
    valid[valid_obj_idx, :topk] = True
    return result, valid


__all__ = [
    "_compute_input_knn",
    "_compute_runtime_context_neighbors",
]
