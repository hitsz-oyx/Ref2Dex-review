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


def sample_random_supervision_edges(
    *,
    num_obj_points: int,
    num_hand_points: int,
    obj_valid_mask: np.ndarray,
    num_supervision_edges: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_supervision_edges <= 0:
        raise ValueError("num_supervision_edges must be positive.")
    if num_hand_points <= 0:
        raise ValueError("num_hand_points must be positive.")
    if num_supervision_edges > num_hand_points:
        raise ValueError(
            "num_supervision_edges must be <= num_hand_points for no-replacement sampling."
        )
    obj_valid_mask = np.asarray(obj_valid_mask, dtype=bool)
    edge_idx = np.full((num_obj_points, num_supervision_edges), -1, dtype=np.int64)
    edge_valid = np.zeros((num_obj_points, num_supervision_edges), dtype=bool)
    rng = np.random.default_rng(seed)
    all_hand_idx = np.arange(num_hand_points, dtype=np.int64)
    for obj_idx in np.flatnonzero(obj_valid_mask).tolist():
        chosen = rng.choice(all_hand_idx, size=num_supervision_edges, replace=False)
        edge_idx[obj_idx] = chosen
        edge_valid[obj_idx] = True
    return edge_idx, edge_valid


def sample_contact_supervision_edges(
    *,
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    contact_radius: float,
    quotas: tuple[int, int, int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample contact-positive edges for the auxiliary supervision stream.

    For every valid object point, distance ``d`` to each hand point defines
    a soft target ``y = max(1 - d/r, 0)``. Hand candidates are bucketed by
    target strength into four bins (weak / medium / strong / very strong,
    in that order so the bin layout matches the diagnostic bins) and each
    bin is filled up to its quota *without* refilling from other bins.

    Args:
        gt_obj_points: ``[N_obj, 3]`` clean GT object points (the *same*
            geometry used for the edge target so the auxiliary supervision
            distribution matches the loss).
        gt_hand_points: ``[N_hand, 3]`` clean GT hand points.
        obj_valid_mask: ``[N_obj]`` boolean mask for valid object points.
        contact_radius: radius ``r`` used by the contact target definition.
        quotas: per-bin maximum sample count, ordered
            ``(weak, medium, strong, very_strong)``. The total maximum
            is therefore ``sum(quotas)`` edges per object.
        seed: PRNG seed; together with the surrounding frame signature it
            makes the auxiliary supervision stream deterministic and
            independent of the random128 baseline stream.

    Returns:
        edge_idx: ``[N_obj, sum(quotas)]`` int64 hand point indices.
            Unfilled slots are -1.
        edge_valid: ``[N_obj, sum(quotas)]`` bool mask. Only valid edges
            have ``y > 0`` by construction.
    """
    if contact_radius <= 0.0:
        raise ValueError("contact_radius must be positive.")
    if len(quotas) != 4:
        raise ValueError(
            f"quotas must have 4 entries (weak/medium/strong/very_strong), got {len(quotas)}."
        )
    quotas_t = tuple(int(q) for q in quotas)
    if any(q < 0 for q in quotas_t):
        raise ValueError(f"quotas must be non-negative, got {quotas_t}.")
    total_quota = sum(quotas_t)
    if total_quota <= 0:
        raise ValueError("Sum of quotas must be positive.")

    obj_pts = np.asarray(gt_obj_points, dtype=np.float32)
    hand_pts = np.asarray(gt_hand_points, dtype=np.float32)
    obj_valid_mask = np.asarray(obj_valid_mask, dtype=bool)
    num_obj = int(obj_pts.shape[0])
    num_hand = int(hand_pts.shape[0])
    if num_obj <= 0 or num_hand <= 0:
        return (
            np.full((num_obj, total_quota), -1, dtype=np.int64),
            np.zeros((num_obj, total_quota), dtype=bool),
        )

    obj_t = torch.from_numpy(obj_pts)
    hand_t = torch.from_numpy(hand_pts)
    edge_idx = np.full((num_obj, total_quota), -1, dtype=np.int64)
    edge_valid = np.zeros((num_obj, total_quota), dtype=bool)
    rng = np.random.default_rng(seed)

    # Compute the full clean GT distance matrix once; with N_obj=512 and
    # N_hand=1538 this is ~7.87e5 pairs/frame, which is cheap.
    distance = torch.cdist(obj_t, hand_t).numpy()
    r = float(contact_radius)

    for obj_idx in np.flatnonzero(obj_valid_mask).tolist():
        dist_row = distance[obj_idx]
        very_strong = np.flatnonzero(dist_row <= 0.25 * r)
        strong = np.flatnonzero((dist_row > 0.25 * r) & (dist_row <= 0.50 * r))
        medium = np.flatnonzero((dist_row > 0.50 * r) & (dist_row <= 0.75 * r))
        weak = np.flatnonzero((dist_row > 0.75 * r) & (dist_row < r))
        candidate_bins = (weak, medium, strong, very_strong)

        write_col = 0
        for candidates, quota in zip(candidate_bins, quotas_t):
            if quota <= 0 or candidates.size == 0:
                continue
            take_count = min(quota, int(candidates.size))
            chosen = rng.choice(candidates, size=take_count, replace=False)
            slot = write_col + take_count
            edge_idx[obj_idx, write_col:slot] = chosen
            edge_valid[obj_idx, write_col:slot] = True
            write_col = slot
    return edge_idx, edge_valid


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
class PerturbedGeometry:
    input_obj_points: np.ndarray
    input_obj_normals: np.ndarray
    input_hand_points: np.ndarray
    input_hand_normals: np.ndarray
    gt_obj_points: np.ndarray
    gt_obj_normals: np.ndarray
    gt_hand_points: np.ndarray
    gt_hand_normals: np.ndarray
    obj_perturbed: bool


def perturb_object_geometry(
    *,
    obj_points: np.ndarray,
    obj_normals: np.ndarray,
    hand_points: np.ndarray,
    hand_normals: np.ndarray,
    seed: int,
    apply_obj_perturb: bool,
    obj_rot_std_deg: float,
    obj_trans_std: float,
    obj_perturb_prob: float,
) -> PerturbedGeometry:
    """Simulate a noisy *object* pose in hand-root frame: one shared SE(3)
    on the 512 sampled object points, clean hand geometry, clean GT for both.

    No global scene-level augmentation is applied — in hand-root frame the
    ``T_hand^{-1}`` canonicalization already eliminates scene-level SE(3)
    degrees of freedom, so global augmentation would only re-introduce a
    nuisance the data preprocessing has already removed.
    """
    rng = np.random.default_rng(seed)
    gt_obj_points = np.asarray(obj_points, dtype=np.float32).copy()
    gt_obj_normals = np.asarray(obj_normals, dtype=np.float32).copy()
    gt_hand_points = np.asarray(hand_points, dtype=np.float32).copy()
    gt_hand_normals = np.asarray(hand_normals, dtype=np.float32).copy()
    input_obj_points = gt_obj_points.copy()
    input_obj_normals = gt_obj_normals.copy()
    input_hand_points = gt_hand_points.copy()
    input_hand_normals = gt_hand_normals.copy()

    obj_perturbed = False
    if (
        apply_obj_perturb
        and obj_perturb_prob > 0
        and (obj_rot_std_deg > 0 or obj_trans_std > 0)
        and rng.random() <= obj_perturb_prob
    ):
        angle = np.deg2rad(rng.normal(0.0, obj_rot_std_deg))
        rotation = _rotation_matrix(_random_axis(rng), float(angle))
        translation = rng.normal(0.0, obj_trans_std, size=3)
        input_obj_points = (input_obj_points @ rotation.T + translation).astype(np.float32)
        input_obj_normals = (input_obj_normals @ rotation.T).astype(np.float32)
        norms = np.linalg.norm(input_obj_normals, axis=-1, keepdims=True)
        input_obj_normals = (input_obj_normals / np.clip(norms, 1e-8, None)).astype(np.float32)
        obj_perturbed = True

    return PerturbedGeometry(
        input_obj_points=input_obj_points,
        input_obj_normals=input_obj_normals,
        input_hand_points=input_hand_points,
        input_hand_normals=input_hand_normals,
        gt_obj_points=gt_obj_points,
        gt_obj_normals=gt_obj_normals,
        gt_hand_points=gt_hand_points,
        gt_hand_normals=gt_hand_normals,
        obj_perturbed=obj_perturbed,
    )
