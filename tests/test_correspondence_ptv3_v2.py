from __future__ import annotations

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
        loss_hand_contact_weight = 1.0
    class Cfg:
        meta = Meta()
    runner.cfg = Cfg()
    preds = {
        "pred_cross_contact_logits": torch.zeros(1, 2, 3),
        "pred_cross_contact_prob": torch.full((1, 2, 3), 0.5),
        "pred_hand_contact_logits": torch.zeros(1, 4),
        "pred_hand_contact_prob": torch.full((1, 4), 0.5),
    }
    batch = {
        "runtime_obj_valid_mask": torch.tensor([[True, True]]),
        "supervision_edge_valid_mask": torch.tensor([[[True, True, False], [True, False, False]]]),
        "edge_contact_target": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        "hand_contact_target": torch.tensor([[0.0, 0.5, 1.0, 0.0]]),
    }
    losses, aux = runner._compute_losses(preds, batch)
    assert isinstance(aux["sampled_nonzero_edge_fraction"], MetricStat)
    assert isinstance(aux["object_nonzero_edge_coverage"], MetricStat)
    assert aux["sampled_nonzero_edge_fraction"].total == 1.0
    assert aux["sampled_nonzero_edge_fraction"].count == 3.0
    assert aux["object_nonzero_edge_coverage"].total == 1.0
    assert aux["object_nonzero_edge_coverage"].count == 2.0
    # Loss shape: cross-edge QFL + hand contact QFL.
    assert set(losses) == {"cross_edge_contact", "hand_contact"}
    assert isinstance(aux["hand_contact_nonzero_count"], MetricStat)
    # Diagnostic metrics must all be present after the v2 metric audit.
    expected_keys = {
        "cross_edge_qfl",
        "cross_edge_bce",
        "cross_edge_mae",
        "cross_edge_oracle_bce",
        "cross_edge_oracle_bce_global",
        "cross_edge_excess_bce",
        "cross_edge_excess_bce_global",
        "cross_edge_nonzero_mae",
        "cross_edge_nonzero_bce",
        "cross_edge_nonzero_pred_mean",
        "cross_edge_nonzero_target_mean",
        "cross_edge_zero_pred_mean",
        "cross_edge_zero_pred_p95",
        "cross_edge_zero_pred_p99",
        "zero_baseline_qfl",
        "zero_baseline_bce",
        "zero_baseline_mae",
        "hand_contact_qfl",
        "hand_contact_bce",
        "hand_contact_mae",
        "hand_contact_nonzero_mae",
        "hand_contact_nonzero_pred_mean",
        "hand_contact_nonzero_target_mean",
    }
    for name in (
        "edge_y_0_025",
        "edge_y_025_050",
        "edge_y_050_075",
        "edge_y_075_100",
    ):
        expected_keys.add(f"{name}_count")
        expected_keys.add(f"{name}_mae")
    missing = expected_keys - aux.keys()
    assert not missing, f"Missing diagnostic metrics: {sorted(missing)}"
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
            "supervision_edge_idx": torch.tensor([[[0, 1], [1, 2]]]),
            "supervision_edge_valid_mask": torch.tensor([[[True, True], [True, True]]]),
        }
        out = model(batch)
        # v2 forward contract: cross-edge outputs + hand contact outputs.
        assert set(out) == {
            "pred_cross_contact_logits",
            "pred_cross_contact_prob",
            "pred_hand_contact_logits",
            "pred_hand_contact_prob",
        }
        assert hasattr(model, "hand_contact_head")
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
        # Forbid the bare obj_contact field (the loss function name
        # `contact_target_from_distance` is allowed and is excluded by the
        # function-name check below).
        import re

        bare_contact_target = re.search(r"(?<![A-Za-z_])contact_target(?![A-Za-z_])", text)
        assert bare_contact_target is None, (
            f"{path}: stray contact_target field reference (must use edge_contact_target)"
        )
