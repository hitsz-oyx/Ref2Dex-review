from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.task.correspondence_ptv3_v2.losses import contact_target_from_distance, quality_focal_loss_map
from src.task.correspondence_ptv3_v2.sampling import sample_random_supervision_edges, stable_frame_seed


def test_contact_target_from_distance_values() -> None:
    distance = torch.tensor([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02], dtype=torch.float32)
    target = contact_target_from_distance(distance, contact_radius=0.01)
    expected = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0, 0.0], dtype=torch.float32)
    assert torch.allclose(target, expected)
    assert torch.all(target >= 0.0)
    assert torch.all(target <= 1.0)


def test_contact_target_from_distance_rejects_non_positive_radius() -> None:
    distance = torch.tensor([0.0], dtype=torch.float32)
    for radius in (0.0, -0.01):
        try:
            contact_target_from_distance(distance, contact_radius=radius)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for non-positive contact_radius.")


def test_random_edge_sampling_shape_no_duplicates_and_distance_independent() -> None:
    obj_valid_mask = np.asarray([True, True, False, True], dtype=bool)
    sample_a = sample_random_supervision_edges(
        num_obj_points=4,
        num_hand_points=256,
        obj_valid_mask=obj_valid_mask,
        num_supervision_edges=128,
        seed=123,
    )
    sample_b = sample_random_supervision_edges(
        num_obj_points=4,
        num_hand_points=256,
        obj_valid_mask=obj_valid_mask,
        num_supervision_edges=128,
        seed=123,
    )
    edge_idx, edge_valid = sample_a
    assert edge_idx.shape == (4, 128)
    assert edge_valid.shape == (4, 128)
    assert np.array_equal(sample_a[0], sample_b[0])
    assert np.array_equal(sample_a[1], sample_b[1])
    for obj_idx in np.flatnonzero(obj_valid_mask):
        chosen = edge_idx[obj_idx]
        assert len(np.unique(chosen)) == 128


def test_random_edge_sampling_same_epoch_same_indices_different_epoch_resamples() -> None:
    seed_epoch0 = stable_frame_seed(base_seed=42, seq_id="seq", side="right", raw_frame_id=7, epoch=0, namespace="supervision-edges")
    seed_epoch0_b = stable_frame_seed(base_seed=42, seq_id="seq", side="right", raw_frame_id=7, epoch=0, namespace="supervision-edges")
    seed_epoch1 = stable_frame_seed(base_seed=42, seq_id="seq", side="right", raw_frame_id=7, epoch=1, namespace="supervision-edges")
    obj_valid_mask = np.asarray([True, True], dtype=bool)
    sample0 = sample_random_supervision_edges(
        num_obj_points=2,
        num_hand_points=256,
        obj_valid_mask=obj_valid_mask,
        num_supervision_edges=128,
        seed=seed_epoch0,
    )[0]
    sample0_b = sample_random_supervision_edges(
        num_obj_points=2,
        num_hand_points=256,
        obj_valid_mask=obj_valid_mask,
        num_supervision_edges=128,
        seed=seed_epoch0_b,
    )[0]
    sample1 = sample_random_supervision_edges(
        num_obj_points=2,
        num_hand_points=256,
        obj_valid_mask=obj_valid_mask,
        num_supervision_edges=128,
        seed=seed_epoch1,
    )[0]
    assert np.array_equal(sample0, sample0_b)
    assert not np.array_equal(sample0, sample1)


def test_qfl_beta_zero_matches_bce_with_logits() -> None:
    logits = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    target = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    qfl = quality_focal_loss_map(logits, target, beta=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    assert torch.allclose(qfl, bce)


def test_qfl_soft_targets_backward_and_zero_near_match() -> None:
    logits = torch.tensor([-1.0986123, 0.0, 1.0986123], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
    loss = quality_focal_loss_map(logits, target, beta=2.0).sum()
    loss.backward()
    assert logits.grad is not None
    near_match_logits = torch.logit(target.clamp(1e-4, 1.0 - 1e-4)).detach()
    near_match_loss = quality_focal_loss_map(near_match_logits, target, beta=2.0)
    assert torch.all(near_match_loss < 1e-6)
