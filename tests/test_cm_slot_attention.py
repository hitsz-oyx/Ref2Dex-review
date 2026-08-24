from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from src.task.Cm.model import CmFlowHead, CmFlowModel
from src.task.Cm.runner import CmActionRunner, internal_flow_smooth_l1, scaled_flow_smooth_l1


def test_cm_flow_head_uses_full_hand_motion_inputs_and_slot_bottleneck() -> None:
    torch.manual_seed(7)
    head = CmFlowHead(
        dense_token_dim=8,
        cm_dim=16,
        num_cm_tokens=4,
        num_slot_iters=2,
    )
    batch_size, num_obj, num_hand = 2, 5, 7
    output = head(
        z_obj=torch.randn(batch_size, num_obj, 8),
        z_hand=torch.randn(batch_size, num_hand, 8),
        dense_hand_contact=torch.rand(batch_size, num_hand),
        obj_points=torch.randn(batch_size, num_obj, 3),
        obj_normals=torch.randn(batch_size, num_obj, 3),
        hand_points=torch.randn(batch_size, num_hand, 3),
        hand_normals=torch.randn(batch_size, num_hand, 3),
        hand_flow=torch.randn(batch_size, num_hand, 3),
        obj_valid_mask=torch.tensor([[True, True, False, True, False], [True] * num_obj]),
    )

    assert head.hand_motion_encoder[0].in_features == 8 + 10
    assert output["cm_tokens"].shape == (batch_size, 4, 16)
    assert output["cm_assignment"].shape == (batch_size, 4, num_hand)
    assert output["cm_slot_weights"].shape == (batch_size, 4, num_hand)
    assert output["slot_gate"].shape == (batch_size, 4)
    assert output["slot_nonzero_prob"].shape == (batch_size, 4)
    assert output["decoder_slot_usage"].shape == (batch_size, 4)
    assert output["cm_anchor_pos"].shape == (batch_size, 4, 3)
    assert output["cm_anchor_normal"].shape == (batch_size, 4, 3)
    assert output["pred_obj_flow"].shape == (batch_size, num_obj, 3)
    torch.testing.assert_close(output["cm_assignment"].sum(dim=1), torch.ones(batch_size, num_hand))
    torch.testing.assert_close(output["cm_slot_weights"].sum(dim=-1), torch.ones(batch_size, 4))
    torch.testing.assert_close(
        output["decoder_slot_usage"].sum(dim=-1),
        torch.ones(batch_size),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(output["cm_anchor_normal"], dim=-1),
        torch.ones(batch_size, 4),
    )
    torch.testing.assert_close(output["pred_obj_flow"][0, ~torch.tensor([True, True, False, True, False])], torch.zeros(2, 3))
    assert torch.isfinite(output["cm_tokens"]).all()

    repeated_inputs = {
        "z_obj": torch.randn(batch_size, num_obj, 8),
        "z_hand": torch.randn(batch_size, num_hand, 8),
        "dense_hand_contact": torch.rand(batch_size, num_hand),
        "obj_points": torch.randn(batch_size, num_obj, 3),
        "obj_normals": torch.randn(batch_size, num_obj, 3),
        "hand_points": torch.randn(batch_size, num_hand, 3),
        "hand_normals": torch.randn(batch_size, num_hand, 3),
        "hand_flow": torch.randn(batch_size, num_hand, 3),
        "obj_valid_mask": torch.ones(batch_size, num_obj, dtype=torch.bool),
    }
    # Hard-Concrete samples gates while training; extraction/evaluation is
    # deterministic by construction.
    head.eval()
    repeated_output = head(**repeated_inputs)
    repeated_output_again = head(**repeated_inputs)
    for key in (
        "cm_tokens", "cm_assignment", "cm_slot_weights", "slot_gate", "slot_hard_mask",
        "slot_nonzero_prob", "decoder_slot_usage", "pred_obj_flow",
    ):
        torch.testing.assert_close(repeated_output_again[key], repeated_output[key])


def test_cm_flow_head_restores_metric_anchor_coordinates_after_internal_scaling() -> None:
    torch.manual_seed(11)
    head = CmFlowHead(
        dense_token_dim=4,
        cm_dim=8,
        num_cm_tokens=2,
        num_slot_iters=1,
        internal_point_flow_scale=100.0,
    )
    hand_points = torch.randn(1, 3, 3)
    output = head(
        z_obj=torch.randn(1, 2, 4),
        z_hand=torch.randn(1, 3, 4),
        dense_hand_contact=torch.rand(1, 3),
        obj_points=torch.randn(1, 2, 3),
        obj_normals=torch.randn(1, 2, 3),
        hand_points=hand_points,
        hand_normals=torch.randn(1, 3, 3),
        hand_flow=torch.randn(1, 3, 3),
        obj_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    expected_anchor_m = torch.einsum("bkh,bhd->bkd", output["cm_slot_weights"], hand_points)
    torch.testing.assert_close(output["cm_anchor_pos"], expected_anchor_m)
    # A nonzero internal-centimetre decoder output must be restored to metres.
    head.eval()
    with torch.no_grad():
        head.edge_flow_head.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        head.slot_gate_head[-1].weight.zero_()
        head.slot_gate_head[-1].bias.fill_(100.0)
    output = head(
        z_obj=torch.randn(1, 2, 4), z_hand=torch.randn(1, 3, 4),
        dense_hand_contact=torch.rand(1, 3), obj_points=torch.randn(1, 2, 3),
        obj_normals=torch.randn(1, 2, 3), hand_points=hand_points,
        hand_normals=torch.randn(1, 3, 3), hand_flow=torch.randn(1, 3, 3),
        obj_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    expected_m = torch.tensor([0.01, 0.02, 0.03]).expand(1, 2, 3)
    torch.testing.assert_close(output["pred_obj_flow"], expected_m)


def test_slot_gate_fallback_keeps_one_dynamic_slot_active() -> None:
    torch.manual_seed(13)
    head = CmFlowHead(dense_token_dim=4, cm_dim=8, num_cm_tokens=2, num_slot_iters=1)
    head.eval()
    with torch.no_grad():
        head.slot_gate_head[-1].weight.zero_()
        head.slot_gate_head[-1].bias.fill_(-100.0)
    output = head(
        z_obj=torch.randn(1, 3, 4), z_hand=torch.randn(1, 4, 4),
        dense_hand_contact=torch.rand(1, 4), obj_points=torch.randn(1, 3, 3),
        obj_normals=torch.randn(1, 3, 3), hand_points=torch.randn(1, 4, 3),
        hand_normals=torch.randn(1, 4, 3), hand_flow=torch.randn(1, 4, 3),
        obj_valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(output["slot_hard_mask"].sum(dim=-1), torch.ones(1, dtype=torch.long))
    torch.testing.assert_close(output["decoder_slot_usage"].sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(output["pred_obj_flow"], torch.zeros_like(output["pred_obj_flow"]))


def test_no_gate_routes_through_every_configured_slot() -> None:
    torch.manual_seed(17)
    head = CmFlowHead(
        dense_token_dim=4, cm_dim=8, num_cm_tokens=4, num_slot_iters=1,
        use_slot_gate=False,
    )
    head.eval()
    output = head(
        z_obj=torch.randn(2, 3, 4), z_hand=torch.randn(2, 5, 4),
        dense_hand_contact=torch.rand(2, 5), obj_points=torch.randn(2, 3, 3),
        obj_normals=torch.randn(2, 3, 3), hand_points=torch.randn(2, 5, 3),
        hand_normals=torch.randn(2, 5, 3), hand_flow=torch.randn(2, 5, 3),
        obj_valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(output["slot_nonzero_prob"], torch.ones(2, 4))
    torch.testing.assert_close(output["slot_hard_mask"], torch.ones(2, 4, dtype=torch.bool))
    torch.testing.assert_close(output["slot_fallback_used"], torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(output["decoder_slot_usage"].sum(dim=-1), torch.ones(2))


def test_gate_warmup_forces_all_slots_without_changing_gate_probabilities() -> None:
    torch.manual_seed(18)
    head = CmFlowHead(
        dense_token_dim=4, cm_dim=8, num_cm_tokens=4, num_slot_iters=1,
        use_slot_gate=True, slot_threshold=0.85,
    )
    head.eval()
    with torch.no_grad():
        head.slot_gate_head[-1].weight.zero_()
        head.slot_gate_head[-1].bias.fill_(-100.0)
    head.set_slot_gate_runtime(force_all_slots=True, threshold=0.0)
    output = head(
        z_obj=torch.randn(2, 3, 4), z_hand=torch.randn(2, 5, 4),
        dense_hand_contact=torch.rand(2, 5), obj_points=torch.randn(2, 3, 3),
        obj_normals=torch.randn(2, 3, 3), hand_points=torch.randn(2, 5, 3),
        hand_normals=torch.randn(2, 5, 3), hand_flow=torch.randn(2, 5, 3),
        obj_valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(output["slot_hard_mask"], torch.ones(2, 4, dtype=torch.bool))
    assert torch.all(output["slot_nonzero_prob"] < 1e-4)
    assert float(output["slot_gate_threshold"]) == 0.0
    assert float(output["slot_gate_force_all"]) == 1.0


def test_gate_warmup_schedule_has_full_open_and_linear_ramp() -> None:
    runner = object.__new__(CmActionRunner)
    runner.cfg = SimpleNamespace(meta=SimpleNamespace(
        use_slot_gate=True,
        gate_warmup_enabled=True,
        gate_warmup_full_epochs=5,
        gate_warmup_ramp_epochs=5,
        slot_threshold=0.85,
        loss_slot_count_weight=1e-3,
    ))
    assert runner._gate_warmup_state(4) == (True, 0.0, 0.0, 0.0)
    assert runner._gate_warmup_state(5) == (False, 0.0, 0.0, 0.0)
    force_all, threshold, count_weight, progress = runner._gate_warmup_state(7)
    assert not force_all
    assert threshold == 0.425
    assert count_weight == 5e-4
    assert progress == 0.5
    assert runner._gate_warmup_state(9) == (False, 0.85, 1e-3, 1.0)


def test_time_condition_requires_and_uses_physical_seconds() -> None:
    torch.manual_seed(19)
    head = CmFlowHead(
        dense_token_dim=4, cm_dim=8, num_cm_tokens=1, num_slot_iters=1,
        use_slot_gate=False, use_time_condition=True,
    )
    inputs = dict(
        z_obj=torch.randn(1, 3, 4), z_hand=torch.randn(1, 5, 4),
        dense_hand_contact=torch.rand(1, 5), obj_points=torch.randn(1, 3, 3),
        obj_normals=torch.randn(1, 3, 3), hand_points=torch.randn(1, 5, 3),
        hand_normals=torch.randn(1, 5, 3), hand_flow=torch.randn(1, 5, 3),
        obj_valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    try:
        head(**inputs)
    except ValueError as error:
        assert "delta_time_s is required" in str(error)
    else:
        raise AssertionError("Time-conditioned head accepted a missing time interval.")
    short = head(**inputs, delta_time_s=torch.tensor([1.0 / 30.0]))
    long = head(**inputs, delta_time_s=torch.tensor([10.0 / 30.0]))
    assert not torch.allclose(short["cm_tokens"], long["cm_tokens"])


def test_geometry_only_object_decoder_ignores_dense_object_context() -> None:
    torch.manual_seed(23)
    head = CmFlowHead(
        dense_token_dim=4,
        cm_dim=8,
        num_cm_tokens=2,
        num_slot_iters=1,
        use_slot_gate=False,
        use_object_context=False,
        use_time_condition=False,
    )
    head.eval()
    with torch.no_grad():
        nn.init.normal_(head.edge_flow_head.weight, std=0.1)
    inputs = dict(
        z_hand=torch.randn(1, 5, 4),
        dense_hand_contact=torch.rand(1, 5),
        obj_points=torch.randn(1, 3, 3),
        obj_normals=F.normalize(torch.randn(1, 3, 3), dim=-1),
        hand_points=torch.randn(1, 5, 3),
        hand_normals=F.normalize(torch.randn(1, 5, 3), dim=-1),
        hand_flow=torch.randn(1, 5, 3),
        obj_valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    first = head(z_obj=torch.randn(1, 3, 4), **inputs)
    second = head(z_obj=torch.randn(1, 3, 4) * 1000.0, **inputs)
    assert head.object_context_encoder is None
    assert head.edge_backbone[0].in_features == head.cm_dim + 12
    torch.testing.assert_close(first["pred_obj_flow"], second["pred_obj_flow"])
    torch.testing.assert_close(first["decoder_slot_usage"], second["decoder_slot_usage"])


def test_dense_token_input_stays_in_metres_and_internal_loss_scales_gradient() -> None:
    class CapturingDense(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_dim = 4
            self.received: dict[str, torch.Tensor] = {}

        def forward(self, **kwargs):
            self.received = kwargs
            batch_size = kwargs["obj_points"].shape[0]
            return (
                torch.zeros(batch_size, 2, 4),
                torch.zeros(batch_size, 3, 4),
                torch.zeros(batch_size, 3),
            )

    class PassthroughHead(nn.Module):
        def forward(self, **kwargs):
            return {"pred_obj_flow": kwargs["obj_points"]}

    model = object.__new__(CmFlowModel)
    nn.Module.__init__(model)
    dense = CapturingDense()
    model.dense_encoder = dense
    model.head = PassthroughHead()
    obj_points_m = torch.tensor([[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]])
    batch = {
        "obj_points": obj_points_m,
        "obj_normals": torch.zeros_like(obj_points_m),
        "hand_points": torch.zeros(1, 3, 3),
        "hand_normals": torch.zeros(1, 3, 3),
        "hand_flow": torch.zeros(1, 3, 3),
        "obj_valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    model(batch)
    torch.testing.assert_close(dense.received["obj_points"], obj_points_m)

    pred_m = torch.tensor([[[0.01, 0.0, 0.0]]], requires_grad=True)
    gt_m = torch.zeros_like(pred_m)
    valid = torch.ones(1, 1, dtype=torch.bool)
    loss_m = internal_flow_smooth_l1(pred_m, gt_m, valid, beta_m=0.01, internal_scale=1.0)
    loss_cm = internal_flow_smooth_l1(pred_m, gt_m, valid, beta_m=0.01, internal_scale=100.0)
    grad_m = torch.autograd.grad(loss_m, pred_m, retain_graph=True)[0]
    grad_cm = torch.autograd.grad(loss_cm, pred_m)[0]
    torch.testing.assert_close(grad_cm, grad_m * 100.0)


def test_public_meter_output_and_scaled_loss_cancel_target_scale() -> None:
    scale = 20.0
    pred_scaled = torch.tensor([[[1.2, 0.0, 0.0]]], requires_grad=True)
    gt_m = torch.tensor([[[0.05, 0.0, 0.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    pred_m = pred_scaled / scale
    indirect = scaled_flow_smooth_l1(
        pred_m, gt_m, valid, beta_m=0.005, target_scale=scale,
    )
    direct = F.smooth_l1_loss(
        torch.linalg.vector_norm(pred_scaled - gt_m * scale, dim=-1),
        torch.zeros(1, 1),
        beta=0.005 * scale,
    )
    indirect_grad = torch.autograd.grad(indirect, pred_scaled, retain_graph=True)[0]
    direct_grad = torch.autograd.grad(direct, pred_scaled)[0]
    torch.testing.assert_close(indirect, direct)
    torch.testing.assert_close(indirect_grad, direct_grad)


def test_epoch_ratios_and_test_stride_wandb_selection() -> None:
    runner = object.__new__(CmActionRunner)
    runner.cfg = SimpleNamespace(meta=SimpleNamespace(loss_active_overlap_weight=0.0))
    epoch = runner.select_epoch_metrics({
        "train_epoch/flow/epe_mm": 4.0,
        "train_epoch/flow/gt_norm_mm": 8.0,
        "train_epoch/flow/pred_norm_mm": 6.0,
    })
    assert epoch["epoch/flow/relative_epe"] == 0.5
    assert epoch["epoch/flow/norm_ratio"] == 0.75
    assert epoch["epoch/flow/zero_flow_improvement"] == 0.5
    selected = runner.select_eval_metrics({
        "val/stride_1/flow/epe_mm": 1.0,
        "test/stride_10/flow/epe_mm": 10.0,
        "test/stride_2/flow/epe_mm": 2.0,
        "test/mean_stride_epe_mm": 4.0,
    })
    assert "val/stride_1/flow/epe_mm" in selected
    assert "test/stride_10/flow/epe_mm" in selected
    assert "test/stride_2/flow/epe_mm" not in selected


def test_required_flow_calibration_metadata_is_strict() -> None:
    runner = object.__new__(CmActionRunner)
    runner.cfg = SimpleNamespace(
        meta=SimpleNamespace(
            coordinate_frame="hand_root_t", num_obj_points=512, num_hand_points=1538,
            flow_target_rms_m=0.1, object_flow_target_scale=10.0,
            require_flow_calibration=True,
        ),
        data=SimpleNamespace(active_only=True, min_stride=1, max_stride=10),
    )
    metadata = {
        "coordinate_frame": "hand_root_t", "num_obj_points": 512, "num_hand_points": 1538,
    }
    try:
        runner.configure_data(metadata)
    except ValueError as exc:
        assert "Missing Cm flow calibration metadata" in str(exc)
    else:
        raise AssertionError("Expected missing calibration metadata to fail.")
    metadata.update({
        "flow_target_rms_m": 0.1,
        "flow_target_scale": 10.0,
        "statistics_split": "train",
        "statistics_active_only": True,
        "statistics_num_obj_points": 512,
        "statistics_stride_distribution": "uniform_1_to_10",
        "statistics_stride_weighting": "equal_per_stride",
        "statistics_point_weighting": "per_pair_capped_at_num_obj_points",
    })
    runner.configure_data(metadata)
