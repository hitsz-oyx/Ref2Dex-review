from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.base import MetricStat
from src.task.correspondence_ptv3_v2 import model as model_module
from src.task.correspondence_ptv3_v2.losses import (
    contact_target_from_distance,
    quality_focal_loss_map,
    reduce_loss_map_per_object,
)
from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner
from src.task.correspondence_ptv3_v2.sampling import (
    sample_contact_supervision_edges,
    sample_random_supervision_edges,
    stable_frame_seed,
)


def test_contact_target_from_distance_values() -> None:
    distance = torch.tensor([0.0, 0.005, 0.01, 0.015, 0.02, 0.04], dtype=torch.float32)
    target = contact_target_from_distance(distance, contact_radius=0.02)
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


def test_reduce_loss_map_per_object_all_invalid_returns_zero_and_backward() -> None:
    loss_map = torch.randn(2, 3, requires_grad=True)
    edge_mask = torch.zeros(2, 3, dtype=torch.bool)
    obj_valid_mask = torch.zeros(2, dtype=torch.bool)
    loss = reduce_loss_map_per_object(loss_map, edge_mask, obj_valid_mask)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert loss_map.grad is not None


def test_qfl_modulation_gradient_not_detached() -> None:
    logits = torch.tensor([0.2], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([0.75], dtype=torch.float32)
    loss = quality_focal_loss_map(logits, target, beta=2.0).sum()
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_runner_coverage_metrics_use_metric_stat() -> None:
    runner = CorrespondencePTV3V2Runner.__new__(CorrespondencePTV3V2Runner)
    class Meta:
        quality_focal_beta = 2.0
        loss_cross_edge_weight = 1.0
        loss_contact_aux_weight = 1.0
    class Train:
        diagnostic_every_steps = 20
    class Cfg:
        meta = Meta()
        train = Train()
    runner.cfg = Cfg()
    runner.global_step = 19
    preds = {
        "pred_cross_random_logits": torch.zeros(1, 2, 3),
        "pred_cross_random_prob": torch.full((1, 2, 3), 0.5),
        "pred_cross_contact_aux_logits": torch.zeros(1, 2, 2),
        "pred_cross_contact_aux_prob": torch.full((1, 2, 2), 0.5),
    }
    batch = {
        "runtime_obj_valid_mask": torch.tensor([[True, True]]),
        "random_edge_valid_mask": torch.tensor([[[True, True, False], [True, False, False]]]),
        "random_edge_contact_target": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "contact_edge_valid_mask": torch.tensor([[[True, True], [True, True]]]),
        "contact_edge_contact_target": torch.tensor([[[0.5, 0.0], [0.7, 0.9]]]),
        "hand_contact_target": torch.tensor([[0.0, 0.5, 1.0, 0.0]]),
    }
    losses, aux = runner._compute_losses(preds, batch, compute_diagnostics=True)
    # Random stream metric keys are preserved as ``cross_edge_random_*``.
    assert isinstance(aux["random_sampled_nonzero_edge_fraction"], MetricStat)
    assert isinstance(aux["random_object_nonzero_edge_coverage"], MetricStat)
    assert aux["random_sampled_nonzero_edge_fraction"].total == 1.0
    assert aux["random_sampled_nonzero_edge_fraction"].count == 3.0
    assert aux["random_object_nonzero_edge_coverage"].total == 1.0
    assert aux["random_object_nonzero_edge_coverage"].count == 2.0
    # Loss shape: random QFL + contact aux QFL (hand is removed in v2.1).
    assert set(losses) == {"cross_edge_random", "cross_edge_contact_aux"}
    expected_contact_qfl = quality_focal_loss_map(
        torch.zeros(4), torch.tensor([0.5, 0.0, 0.7, 0.9]), beta=2.0
    )
    assert torch.allclose(losses["cross_edge_contact_aux"], expected_contact_qfl.mean())
    assert "hand_contact" not in losses
    # Hand contact is observed as a GT distribution only.
    assert "hand_contact_qfl" not in aux
    assert "hand_contact_nonzero_mae" not in aux
    assert isinstance(aux["hand_contact_target_count"], MetricStat)
    # Required random-stream diagnostic keys.
    expected_random_keys = {
        "cross_edge_random_qfl",
        "cross_edge_random_bce",
        "cross_edge_random_mae",
        "cross_edge_random_oracle_bce",
        "cross_edge_random_oracle_bce_global",
        "cross_edge_random_excess_bce",
        "cross_edge_random_excess_bce_global",
        "cross_edge_random_nonzero_mae",
        "cross_edge_random_nonzero_bce",
        "cross_edge_random_nonzero_pred_mean",
        "cross_edge_random_nonzero_target_mean",
        "cross_edge_random_zero_pred_mean",
        "cross_edge_random_zero_pred_p95",
        "cross_edge_random_zero_pred_p99",
        "zero_baseline_qfl",
        "zero_baseline_bce",
        "zero_baseline_mae",
    }
    for name in (
        "edge_y_0_025",
        "edge_y_025_050",
        "edge_y_050_075",
        "edge_y_075_100",
    ):
        expected_random_keys.add(f"random_{name}_count")
        expected_random_keys.add(f"random_{name}_mae")
    missing = expected_random_keys - aux.keys()
    assert not missing, f"Missing random-stream metrics: {sorted(missing)}"
    # Required contact-auxiliary diagnostic keys.
    expected_contact_aux_keys = {
        "contact_aux_qfl",
        "contact_aux_bce",
        "contact_aux_soft_bce",
        "contact_aux_mae",
        "contact_aux_hard_neg_count",
        "contact_aux_hard_neg_pred_mean",
        "contact_aux_hard_neg_pred_p95",
        "contact_aux_pred_mean",
        "contact_aux_target_mean",
        "contact_aux_nonzero_mae",
        "contact_aux_object_coverage",
        "contact_aux_valid_edge_count",
    }
    for name in (
        "edge_y_0_025",
        "edge_y_025_050",
        "edge_y_050_075",
        "edge_y_075_100",
    ):
        expected_contact_aux_keys.add(f"contact_aux_{name}_count")
        expected_contact_aux_keys.add(f"contact_aux_{name}_mae")
    missing = expected_contact_aux_keys - aux.keys()
    assert not missing, f"Missing contact-aux metrics: {sorted(missing)}"
    # Required hand-contact GT observation keys.
    for key in (
        "hand_contact_target_nonzero_fraction",
        "hand_contact_target_nonzero_mean",
        "hand_contact_target_mean",
        "hand_target_edge_y_0_025_fraction",
        "hand_target_edge_y_025_050_fraction",
        "hand_target_edge_y_050_075_fraction",
        "hand_target_edge_y_075_100_fraction",
    ):
        assert key in aux, f"Missing hand-contact GT metric: {key}"
    # Losses must only contain cross-edge + hand contact (obj_contact removed).
    assert "obj_contact" not in aux
    assert "obj_contact_qfl" not in aux
    assert "obj_contact_bce" not in aux
    assert "obj_contact_mae" not in aux


def test_model_output_contract_and_dense_cross_api() -> None:
    original_backbone = model_module.PTv3DenseBackbone

    class FakeBackbone(torch.nn.Module):
        def __init__(self, meta, in_channels):
            super().__init__()
            self.output_dim = 8
        def forward(self, feat, coord, valid_mask):
            del coord, valid_mask
            return feat[..., :8]

    class Meta:
        num_obj_points = 2
        num_hand_points = 3
        point_feat_dim = 11
        ptv3_repo_path = ""
        ptv3_grid_size = 0.003
        ptv3_order = ("z",)
        ptv3_stride = (2, 2, 2, 2)
        ptv3_enc_depths = (1, 1, 1, 1, 1)
        ptv3_enc_channels = (8, 8, 8, 8, 8)
        ptv3_enc_num_head = (1, 1, 1, 1, 1)
        ptv3_enc_patch_size = (8, 8, 8, 8, 8)
        ptv3_dec_depths = (1, 1, 1, 1)
        ptv3_dec_channels = (8, 8, 8, 8)
        ptv3_dec_num_head = (1, 1, 1, 1)
        ptv3_dec_patch_size = (8, 8, 8, 8)
        ptv3_mlp_ratio = 4.0
        ptv3_qkv_bias = True
        ptv3_attn_drop = 0.0
        ptv3_proj_drop = 0.0
        ptv3_drop_path = 0.0
        ptv3_pre_norm = True
        ptv3_shuffle_orders = False
        ptv3_enable_rpe = False
        ptv3_enable_flash = False
        ptv3_upcast_attention = False
        ptv3_upcast_softmax = False

    class Cfg:
        meta = Meta()

    model_module.PTv3DenseBackbone = FakeBackbone
    try:
        model = model_module.StaticHOCPTv3V2(Cfg())
        batch = {
            "points": torch.randn(1, 5, 3),
            "normals": torch.randn(1, 5, 3),
            "point_valid_mask": torch.tensor([[True, True, True, True, True]]),
            "runtime_obj_valid_mask": torch.tensor([[True, True]]),
            "random_edge_idx": torch.tensor([[[0, 1], [1, 2]]]),
            "random_edge_valid_mask": torch.tensor([[[True, True], [True, True]]]),
            "contact_edge_idx": torch.tensor([[[2, 0], [0, 1]]]),
            "contact_edge_valid_mask": torch.tensor([[[True, True], [True, True]]]),
        }
        out = model(batch)
        # v2.1 forward contract: only the two cross-edge streams are predicted.
        assert set(out) == {
            "pred_cross_random_logits",
            "pred_cross_random_prob",
            "pred_cross_contact_aux_logits",
            "pred_cross_contact_aux_prob",
        }
        assert hasattr(model, "_predict_cross_edges")
        dense = model.predict_dense_cross_for_object(batch, obj_idx=0)
        assert tuple(dense.shape) == (1, 3)
    finally:
        model_module.PTv3DenseBackbone = original_backbone


def test_v2_has_no_old_task_dependency_strings() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "task" / "correspondence_ptv3_v2"
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        # Skip architecture / design docs: those record the v1 design for
        # historical context and are not live code. Only enforce on .py.
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "src.task.correspondence_ptv3." not in text
        assert "_base_: src.task.correspondence_ptv3." not in text
        # obj_contact branches are no longer needed in v2.
        assert "pred_obj_contact_logits" not in text
        assert "pred_obj_contact_prob" not in text
        assert "obj_contact_bce" not in text
        assert "obj_contact_qfl" not in text
        assert "obj_contact_mae" not in text
        # v2.1 ablation renamed the supervision edge set to ``random_*``
        # and added a separate ``contact_*`` set; the old keys must be
        # gone.
        assert "supervision_edge_idx" not in text
        assert "supervision_edge_valid_mask" not in text
        # Forbid the bare obj_contact field (the loss function name
        # `contact_target_from_distance` is allowed and is excluded by the
        # function-name check below).
        import re

        bare_contact_target = re.search(r"(?<![A-Za-z_])contact_target(?![A-Za-z_])", text)
        assert bare_contact_target is None, (
            f"{path}: stray contact_target field reference (must use edge_contact_target)"
        )


# ---------------------------------------------------------------------------
# sample_contact_supervision_edges tests (Step 9 in docs/指导.md).
# The 6 mandatory tests cover the full contract: sparse/no-contact behavior,
# 4-bin positive layout, hard-negative tail slots, seed reproducibility, and
# random128 stream regression.
# ---------------------------------------------------------------------------

_QUOTAS = (4, 4, 4, 4)
_HARD_NEGATIVE_QUOTA = 2
_HARD_NEGATIVE_RANGE = (0.02, 0.03)


def _build_synthetic_geometry(num_obj: int, num_hand: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic obj/hand point cloud where some hand points are
    within ``contact_radius`` of each obj point so the sampler can find
    contact-aware candidates.
    """
    rng = np.random.default_rng(seed)
    obj = rng.normal(size=(num_obj, 3)).astype(np.float32) * 0.01
    hand = rng.normal(size=(num_hand, 3)).astype(np.float32) * 0.01
    return obj, hand


def test_sample_contact_sampler_outside_auxiliary_range_yields_zero_valid_edges() -> None:
    """Test 1: when there are neither positives nor near hard negatives, nothing is sampled."""
    obj = np.zeros((2, 3), dtype=np.float32)
    hand = np.full((16, 3), 5.0, dtype=np.float32)  # 5m away, far beyond r=0.02
    edge_idx, edge_valid = sample_contact_supervision_edges(
        gt_obj_points=obj,
        gt_hand_points=hand,
        obj_valid_mask=np.array([True, True]),
        contact_radius=0.02,
        quotas=_QUOTAS,
        hard_negative_quota=_HARD_NEGATIVE_QUOTA,
        hard_negative_distance_range=_HARD_NEGATIVE_RANGE,
        seed=42,
    )
    assert edge_idx.shape == (2, sum(_QUOTAS) + _HARD_NEGATIVE_QUOTA)
    assert edge_valid.shape == (2, sum(_QUOTAS) + _HARD_NEGATIVE_QUOTA)
    assert int(edge_valid.sum()) == 0


def test_sample_contact_sampler_only_weak_fills_only_weak() -> None:
    """Test 2: with only weak candidates, weak fills up to its quota and
    the other bins stay empty (no refill)."""
    obj = np.zeros((1, 3), dtype=np.float32)
    # 16 hand points at distance 0.018m (weak bin: 0.75r < d < r with r=0.02).
    hand = np.zeros((16, 3), dtype=np.float32)
    hand[:, 0] = 0.018
    edge_idx, edge_valid = sample_contact_supervision_edges(
        gt_obj_points=obj,
        gt_hand_points=hand,
        obj_valid_mask=np.array([True]),
        contact_radius=0.02,
        quotas=_QUOTAS,
        hard_negative_quota=_HARD_NEGATIVE_QUOTA,
        hard_negative_distance_range=_HARD_NEGATIVE_RANGE,
        seed=42,
    )
    valid_count = int(edge_valid[0].sum())
    # weak quota is 4; medium/strong/very_strong are all 0.
    assert valid_count == _QUOTAS[0]
    # All sampled slots fall in the weak bin (the first 4 columns).
    for col in range(_QUOTAS[0]):
        assert bool(edge_valid[0, col])
    for col in range(_QUOTAS[0], sum(_QUOTAS) + _HARD_NEGATIVE_QUOTA):
        assert not bool(edge_valid[0, col])


def test_sample_contact_sampler_each_bin_fully_filled() -> None:
    """Test 3: when every bin has >= quota candidates, total is sum(quotas)."""
    obj = np.zeros((1, 3), dtype=np.float32)
    hand = np.zeros((sum(_QUOTAS) * 2, 3), dtype=np.float32)
    hand[sum(_QUOTAS) :, 0] = 5.0
    # 4 hand points in each bin (weak / medium / strong / very strong).
    offsets = [0.018, 0.012, 0.006, 0.002]
    for i, off in enumerate(offsets):
        hand[i * 4 : (i + 1) * 4, 0] = off
    edge_idx, edge_valid = sample_contact_supervision_edges(
        gt_obj_points=obj,
        gt_hand_points=hand,
        obj_valid_mask=np.array([True]),
        contact_radius=0.02,
        quotas=_QUOTAS,
        hard_negative_quota=_HARD_NEGATIVE_QUOTA,
        hard_negative_distance_range=_HARD_NEGATIVE_RANGE,
        seed=42,
    )
    assert int(edge_valid.sum()) == sum(_QUOTAS)
    # The first 4 slots are weak, the next 4 medium, etc. (no refill
    # means the bin layout is preserved end-to-end).
    for i, off in enumerate(offsets):
        for col in range(_QUOTAS[i]):
            assert bool(edge_valid[0, i * _QUOTAS[i] + col])
    # Sanity-check: each sampled hand point lies in the expected bin.
    for i, off in enumerate(offsets):
        for col in range(_QUOTAS[i]):
            sampled_off = hand[edge_idx[0, i * _QUOTAS[i] + col], 0]
            assert abs(float(sampled_off) - off) < 1e-5


def test_sample_contact_sampler_adds_hard_negatives_in_tail_slots() -> None:
    """Test 4: hard negatives occupy the tail slots and keep y == 0."""
    obj = np.zeros((1, 3), dtype=np.float32)
    hand = np.zeros((24, 3), dtype=np.float32)
    hand[18:, :] = 5.0
    hand[0:4, 0] = 0.018
    hand[4:8, 0] = 0.012
    hand[8:12, 0] = 0.006
    hand[12:16, 0] = 0.002
    hand[16:18, 0] = 0.024
    edge_idx, edge_valid = sample_contact_supervision_edges(
        gt_obj_points=obj,
        gt_hand_points=hand,
        obj_valid_mask=np.array([True]),
        contact_radius=0.02,
        quotas=_QUOTAS,
        hard_negative_quota=_HARD_NEGATIVE_QUOTA,
        hard_negative_distance_range=_HARD_NEGATIVE_RANGE,
        seed=42,
    )
    obj_t = torch.from_numpy(obj)
    hand_t = torch.from_numpy(hand)
    safe_idx = torch.from_numpy(edge_idx).clamp(min=0)
    distance = torch.norm(hand_t[safe_idx] - obj_t.unsqueeze(1), dim=-1)
    target = contact_target_from_distance(distance, contact_radius=0.02)
    neg_slice = slice(sum(_QUOTAS), sum(_QUOTAS) + _HARD_NEGATIVE_QUOTA)
    assert torch.all(torch.from_numpy(edge_valid)[0, neg_slice])
    assert torch.all(target[0, neg_slice] == 0)


def test_sample_contact_sampler_excludes_near_zero_boundary_edges() -> None:
    """Borderline ``d ~= r`` edges must not be marked valid."""
    obj = np.zeros((1, 3), dtype=np.float32)
    hand = np.full((8, 3), 5.0, dtype=np.float32)
    hand[0, 0] = 0.02 - 5e-7  # positive but too close to the y=0 boundary
    edge_idx, edge_valid = sample_contact_supervision_edges(
        gt_obj_points=obj,
        gt_hand_points=hand,
        obj_valid_mask=np.array([True]),
        contact_radius=0.02,
        quotas=_QUOTAS,
        hard_negative_quota=_HARD_NEGATIVE_QUOTA,
        hard_negative_distance_range=_HARD_NEGATIVE_RANGE,
        seed=42,
    )
    assert int(edge_valid.sum()) == 0


def test_sample_contact_sampler_is_deterministic_under_same_seed() -> None:
    """Test 5: same seed reproduces the same edge_idx."""
    obj, hand = _build_synthetic_geometry(num_obj=4, num_hand=128, seed=0)
    kwargs = dict(
        gt_obj_points=obj,
        gt_hand_points=hand,
        obj_valid_mask=np.array([True, True, True, True]),
        contact_radius=0.02,
        quotas=_QUOTAS,
        hard_negative_quota=_HARD_NEGATIVE_QUOTA,
        hard_negative_distance_range=_HARD_NEGATIVE_RANGE,
    )
    idx1, valid1 = sample_contact_supervision_edges(seed=42, **kwargs)
    idx2, valid2 = sample_contact_supervision_edges(seed=42, **kwargs)
    assert np.array_equal(idx1, idx2)
    assert np.array_equal(valid1, valid2)


def test_sample_random_supervision_edges_unchanged_by_contact_sampler() -> None:
    """Test 6 (regression): the random128 baseline stream remains byte-identical.

    The digest was recorded from the post-v2.1 implementation while the
    random stream still used the original ``namespace=supervision-edges``.
    Any future change to the random128 baseline sampling must update this
    test intentionally.
    """
    seed = stable_frame_seed(
        base_seed=42,
        seq_id="seq",
        side="right",
        raw_frame_id=7,
        epoch=0,
        namespace="supervision-edges",
    )
    obj_valid_mask = np.array([True, True, False, True])
    edge_idx, edge_valid = sample_random_supervision_edges(
        num_obj_points=4,
        num_hand_points=256,
        obj_valid_mask=obj_valid_mask,
        num_supervision_edges=128,
        seed=seed,
    )
    digest = hashlib.sha256(edge_idx.tobytes() + edge_valid.tobytes()).hexdigest()
    assert digest == "634970ffc88ba02c1a2d9f4d4825941dde7204c3a49397d38e8babf48a1fa861"
