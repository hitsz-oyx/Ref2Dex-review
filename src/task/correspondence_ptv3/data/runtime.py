from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


def stable_frame_seed(
    *,
    base_seed: int,
    seq_id: str,
    side: str,
    raw_frame_id: int,
    epoch: int,
    namespace: str = "object-sampling",
) -> int:
    payload = (
        f"{int(base_seed)}\0{seq_id}\0{side}\0{int(raw_frame_id)}\0"
        f"{int(epoch)}\0{namespace}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def sample_object_indices(
    candidate_mask: np.ndarray,
    *,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_idx = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    rng = np.random.default_rng(seed)
    selected = np.full((num_samples,), -1, dtype=np.int64)
    valid = np.zeros((num_samples,), dtype=bool)
    if candidate_idx.size == 0:
        return selected, valid
    if candidate_idx.size >= num_samples:
        selected[:] = rng.choice(candidate_idx, size=num_samples, replace=False)
        valid[:] = True
        return selected, valid
    shuffled = rng.permutation(candidate_idx)
    selected[: shuffled.size] = shuffled
    valid[: shuffled.size] = True
    return selected, valid


def compute_runtime_context_neighbors(
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


def compute_input_knn(
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


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _random_axis(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return axis / norm


@dataclass(frozen=True)
class AugmentedGeometry:
    input_obj_points: np.ndarray
    input_obj_normals: np.ndarray
    input_hand_points: np.ndarray
    input_hand_normals: np.ndarray
    gt_obj_points: np.ndarray
    gt_obj_normals: np.ndarray
    gt_hand_points: np.ndarray
    gt_hand_normals: np.ndarray
    distance_scale: float
    hand_perturbed: bool


def augment_geometry(
    *,
    obj_points: np.ndarray,
    obj_normals: np.ndarray,
    hand_points: np.ndarray,
    hand_normals: np.ndarray,
    seed: int,
    apply_hand_perturb: bool,
    hand_rot_std_deg: float,
    hand_trans_std: float,
    hand_perturb_prob: float,
    apply_global_aug: bool,
    augment_rotation: bool,
    rotation_range_deg: float,
    augment_translation: bool,
    translation_range: float,
    augment_scale: bool,
    scale_range: tuple[float, float],
) -> AugmentedGeometry:
    rng = np.random.default_rng(seed)
    gt_obj_points = np.asarray(obj_points, dtype=np.float32).copy()
    gt_obj_normals = np.asarray(obj_normals, dtype=np.float32).copy()
    gt_hand_points = np.asarray(hand_points, dtype=np.float32).copy()
    gt_hand_normals = np.asarray(hand_normals, dtype=np.float32).copy()
    input_obj_points = gt_obj_points.copy()
    input_obj_normals = gt_obj_normals.copy()
    input_hand_points = gt_hand_points.copy()
    input_hand_normals = gt_hand_normals.copy()

    hand_perturbed = False
    if (
        apply_hand_perturb
        and hand_perturb_prob > 0
        and (hand_rot_std_deg > 0 or hand_trans_std > 0)
        and rng.random() <= hand_perturb_prob
    ):
        angle = np.deg2rad(rng.normal(0.0, hand_rot_std_deg))
        rotation = _rotation_matrix(_random_axis(rng), float(angle))
        translation = rng.normal(0.0, hand_trans_std, size=3)
        center = input_hand_points.mean(axis=0)
        input_hand_points = (
            (input_hand_points - center) @ rotation.T + center + translation
        ).astype(np.float32)
        input_hand_normals = (input_hand_normals @ rotation.T).astype(np.float32)
        hand_perturbed = True

    distance_scale = 1.0
    if apply_global_aug:
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        if augment_rotation and rotation_range_deg > 0:
            angle = np.deg2rad(rng.uniform(-rotation_range_deg, rotation_range_deg))
            rotation = _rotation_matrix(_random_axis(rng), float(angle))
        if augment_translation and translation_range > 0:
            translation = rng.uniform(-translation_range, translation_range, size=3)
        if augment_scale:
            distance_scale = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))

        def transform_points(points: np.ndarray) -> np.ndarray:
            return ((points @ rotation.T) * distance_scale + translation).astype(np.float32)

        def transform_normals(normals: np.ndarray) -> np.ndarray:
            transformed = normals @ rotation.T
            norm = np.linalg.norm(transformed, axis=-1, keepdims=True)
            return (transformed / np.clip(norm, 1e-8, None)).astype(np.float32)

        input_obj_points = transform_points(input_obj_points)
        input_hand_points = transform_points(input_hand_points)
        gt_obj_points = transform_points(gt_obj_points)
        gt_hand_points = transform_points(gt_hand_points)
        input_obj_normals = transform_normals(input_obj_normals)
        input_hand_normals = transform_normals(input_hand_normals)
        gt_obj_normals = transform_normals(gt_obj_normals)
        gt_hand_normals = transform_normals(gt_hand_normals)

    return AugmentedGeometry(
        input_obj_points=input_obj_points,
        input_obj_normals=input_obj_normals,
        input_hand_points=input_hand_points,
        input_hand_normals=input_hand_normals,
        gt_obj_points=gt_obj_points,
        gt_obj_normals=gt_obj_normals,
        gt_hand_points=gt_hand_points,
        gt_hand_normals=gt_hand_normals,
        distance_scale=distance_scale,
        hand_perturbed=hand_perturbed,
    )


__all__ = [
    "AugmentedGeometry",
    "augment_geometry",
    "compute_input_knn",
    "compute_runtime_context_neighbors",
    "sample_object_indices",
    "stable_frame_seed",
]
