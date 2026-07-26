from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

_CONTACT_POSITIVE_EPS = 1e-4


def _sample_without_replacement_torch(
    candidates: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample up to ``count`` entries from a 1D candidate tensor."""
    if count <= 0 or candidates.numel() == 0:
        return candidates[:0]
    take_count = min(int(count), int(candidates.numel()))
    order = torch.randperm(int(candidates.numel()), generator=generator, device=candidates.device)
    return candidates[order[:take_count]]


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
    obj_valid_mask_t = torch.as_tensor(obj_valid_mask, dtype=torch.bool)
    edge_idx = torch.full((num_obj_points, num_supervision_edges), -1, dtype=torch.long)
    edge_valid = torch.zeros((num_obj_points, num_supervision_edges), dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    all_hand_idx = torch.arange(num_hand_points, dtype=torch.long)
    for obj_idx in torch.nonzero(obj_valid_mask_t, as_tuple=False).squeeze(-1).tolist():
        chosen = _sample_without_replacement_torch(
            all_hand_idx,
            count=num_supervision_edges,
            generator=generator,
        )
        edge_idx[obj_idx, : chosen.numel()] = chosen
        edge_valid[obj_idx, : chosen.numel()] = True
    return edge_idx.numpy(), edge_valid.numpy()


def sample_contact_supervision_edges(
    *,
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    contact_radius: float,
    quotas: tuple[int, int, int, int],
    hard_negative_quota: int,
    hard_negative_distance_range: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample auxiliary edges with stratified positives plus hard negatives.

    For every valid object point, distance ``d`` to each hand point defines
    a soft target ``y = max(1 - d/r, 0)``. Hand candidates are bucketed by
    target strength into four bins (weak / medium / strong / very strong,
    in that order so the slot layout matches the diagnostic bins) and each
    bin is filled up to its quota *without* refilling from other bins. A
    final block of hard negatives samples ``y=0`` pairs from a narrow
    distance band just outside ``contact_radius``.

    Args:
        gt_obj_points: ``[N_obj, 3]`` clean GT object points (the *same*
            geometry used for the edge target so the auxiliary supervision
            distribution matches the loss).
        gt_hand_points: ``[N_hand, 3]`` clean GT hand points.
        obj_valid_mask: ``[N_obj]`` boolean mask for valid object points.
        contact_radius: radius ``r`` used by the contact target definition.
        quotas: per-bin maximum sample count, ordered
            ``(weak, medium, strong, very_strong)``. The total maximum
            positive count is ``sum(quotas)`` edges per object.
        hard_negative_quota: maximum number of near-contact negatives
            sampled per object.
        hard_negative_distance_range: closed-open interval
            ``[d_min, d_max)`` used to define hard negatives.
        seed: PRNG seed; together with the surrounding frame signature it
            makes the auxiliary supervision stream deterministic and
            independent of the random128 baseline stream.

    Returns:
        edge_idx: ``[N_obj, sum(quotas) + hard_negative_quota]`` int64 hand
            point indices.
            Unfilled slots are -1.
        edge_valid: ``[N_obj, sum(quotas) + hard_negative_quota]`` bool
            mask. Valid edges can be either ``y > 0`` positives or sampled
            hard negatives with ``y = 0``.
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
    hard_negative_quota = int(hard_negative_quota)
    if hard_negative_quota < 0:
        raise ValueError(f"hard_negative_quota must be non-negative, got {hard_negative_quota}.")
    neg_min, neg_max = (float(hard_negative_distance_range[0]), float(hard_negative_distance_range[1]))
    if not (contact_radius <= neg_min < neg_max):
        raise ValueError(
            "hard_negative_distance_range must satisfy "
            "contact_radius <= min < max."
        )
    total_quota = sum(quotas_t) + hard_negative_quota
    if total_quota <= 0:
        raise ValueError("Sum of quotas must be positive.")

    obj_pts = torch.as_tensor(gt_obj_points, dtype=torch.float32)
    hand_pts = torch.as_tensor(gt_hand_points, dtype=torch.float32)
    obj_valid_mask_t = torch.as_tensor(obj_valid_mask, dtype=torch.bool)
    num_obj = int(obj_pts.shape[0])
    num_hand = int(hand_pts.shape[0])
    if num_obj <= 0 or num_hand <= 0:
        return (
            np.full((num_obj, total_quota), -1, dtype=np.int64),
            np.zeros((num_obj, total_quota), dtype=bool),
        )

    edge_idx = torch.full((num_obj, total_quota), -1, dtype=torch.long)
    edge_valid = torch.zeros((num_obj, total_quota), dtype=torch.bool)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    # Compute the full clean GT distance matrix once; with N_obj=512 and
    # N_hand=1538 this is ~7.87e5 pairs/frame, which is cheap. Use float64
    # here so sampler binning matches the downstream target recomputation as
    # closely as possible near the ``d ~= r`` boundary.
    distance = torch.cdist(obj_pts.double(), hand_pts.double())
    r = float(contact_radius)

    for obj_idx in torch.nonzero(obj_valid_mask_t, as_tuple=False).squeeze(-1).tolist():
        dist_row = distance[obj_idx]
        target_row = (1.0 - dist_row / r).clamp(0.0, 1.0)
        # Keep a tiny positive margin so an edge that is numerically
        # borderline at sampling time does not later collapse to y=0 when the
        # dataset recomputes the target in float32 from gathered coordinates.
        weak = torch.nonzero(
            (target_row > _CONTACT_POSITIVE_EPS) & (target_row <= 0.25),
            as_tuple=False,
        ).squeeze(-1)
        medium = torch.nonzero(
            (target_row > 0.25) & (target_row <= 0.50),
            as_tuple=False,
        ).squeeze(-1)
        strong = torch.nonzero(
            (target_row > 0.50) & (target_row <= 0.75),
            as_tuple=False,
        ).squeeze(-1)
        very_strong = torch.nonzero(
            (target_row > 0.75) & (target_row <= 1.0),
            as_tuple=False,
        ).squeeze(-1)
        candidate_bins = (weak, medium, strong, very_strong)
        hard_negative = torch.nonzero(
            (dist_row >= neg_min) & (dist_row < neg_max) & (target_row <= 0.0),
            as_tuple=False,
        ).squeeze(-1)

        write_col = 0
        for candidates, quota in zip(candidate_bins, quotas_t):
            slot_end = write_col + quota
            if quota <= 0 or candidates.numel() == 0:
                write_col = slot_end
                continue
            chosen = _sample_without_replacement_torch(
                candidates,
                count=quota,
                generator=generator,
            )
            edge_idx[obj_idx, write_col : write_col + chosen.numel()] = chosen
            edge_valid[obj_idx, write_col : write_col + chosen.numel()] = True
            write_col = slot_end
        neg_slot_end = write_col + hard_negative_quota
        if hard_negative_quota > 0 and hard_negative.numel() > 0:
            chosen = _sample_without_replacement_torch(
                hard_negative,
                count=hard_negative_quota,
                generator=generator,
            )
            edge_idx[obj_idx, write_col : write_col + chosen.numel()] = chosen
            edge_valid[obj_idx, write_col : write_col + chosen.numel()] = True
        write_col = neg_slot_end
    return edge_idx.numpy(), edge_valid.numpy()


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
    noisy_hand_points: np.ndarray | None = None,
    noisy_hand_normals: np.ndarray | None = None,
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
    input_hand_points = (
        np.asarray(noisy_hand_points, dtype=np.float32).copy()
        if noisy_hand_points is not None
        else gt_hand_points.copy()
    )
    input_hand_normals = (
        np.asarray(noisy_hand_normals, dtype=np.float32).copy()
        if noisy_hand_normals is not None
        else gt_hand_normals.copy()
    )

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
