from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


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
        raise ValueError("num_supervision_edges must be <= num_hand_points for no-replacement sampling.")
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
    obj_perturbed: bool


def augment_geometry(
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
    apply_global_aug: bool,
    augment_rotation: bool,
    rotation_range_deg: float,
    augment_translation: bool,
    translation_range: float,
) -> AugmentedGeometry:
    """In hand-root frame we simulate a noisy *object* pose: one shared SE(3)
    on the 512 sampled object points, clean hand geometry, clean GT for both.
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
        # All 512 sampled obj points share the same SE(3): simulating an
        # object pose estimation error after clean candidate selection.
        input_obj_points = (input_obj_points @ rotation.T + translation).astype(np.float32)
        input_obj_normals = (input_obj_normals @ rotation.T).astype(np.float32)
        norms = np.linalg.norm(input_obj_normals, axis=-1, keepdims=True)
        input_obj_normals = (input_obj_normals / np.clip(norms, 1e-8, None)).astype(np.float32)
        obj_perturbed = True

    if apply_global_aug:
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        if augment_rotation and rotation_range_deg > 0:
            angle = np.deg2rad(rng.uniform(-rotation_range_deg, rotation_range_deg))
            rotation = _rotation_matrix(_random_axis(rng), float(angle))
        if augment_translation and translation_range > 0:
            translation = rng.uniform(-translation_range, translation_range, size=3)

        def transform_points(points: np.ndarray) -> np.ndarray:
            return (points @ rotation.T + translation).astype(np.float32)

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
        obj_perturbed=obj_perturbed,
    )
