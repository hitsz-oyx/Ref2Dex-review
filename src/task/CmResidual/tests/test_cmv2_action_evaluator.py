"""Action-conditioned Cmv2 contract tests without a real checkpoint or IsaacGym."""
from pathlib import Path

import pytest
import torch

from src.task.CmResidual.cm_v2_action_evaluator import (
    Cmv2ActionEvaluator,
    NominalHandSweep,
    build_nominal_link_poses,
    compose_nominal_targets,
    effect_metrics,
    object_pose_delta_to_xi,
    pairwise_rank_agreement,
)
from src.task.CmDecoderv2.kinematics import InspireKinematics


class _StubAdapter:
    """Deterministic stand-in for the frozen checkpoint used only by contract tests."""

    def predict(self, object_points, object_normals, hand_points, hand_normals,
                hand_flow, delta_time_s, hand_valid_mask):
        batch = object_points.shape[0]
        delta_xi = torch.zeros(batch, 6, device=object_points.device)
        delta_xi[:, 0] = hand_flow[..., 0].mean(-1)
        return {
            "delta_xi_root": delta_xi,
            "obj_flow_pred": torch.zeros(batch, 1024, 3, device=object_points.device),
            "cm_tokens": torch.zeros(batch, 16, 32, device=object_points.device),
            "token_anchors": torch.zeros(batch, 16, 3, device=object_points.device),
            "token_normals": torch.zeros(batch, 16, 3, device=object_points.device),
            "token_mass": torch.zeros(batch, 16, device=object_points.device),
            "token_mask": torch.zeros(batch, 16, dtype=torch.bool, device=object_points.device),
            "p_effect": torch.zeros(batch, device=object_points.device),
            "cm_context": torch.zeros(batch, 640, device=object_points.device),
        }


def _sweep(candidate_offsets=(0.0, 0.1, 0.2), source="controller_fk"):
    current = torch.zeros(1, 1538, 3)
    next_points = current[:, None].expand(1, len(candidate_offsets), 1538, 3).clone()
    for index, offset in enumerate(candidate_offsets):
        next_points[:, index, :, 0] += offset
    normals = torch.zeros_like(current)
    normals[..., 2] = 1.0
    return NominalHandSweep(current, normals, next_points, normals[:, None].expand_as(next_points), source)


def test_action_evaluator_uses_nominal_sweep_and_selects_effect():
    x = torch.linspace(-0.05, 0.05, 1024)
    object_points = torch.stack((x, x.square(), torch.zeros_like(x)), -1)[None]
    object_normals = torch.zeros_like(object_points)
    object_normals[..., 2] = 1.0
    sweep = _sweep()
    evaluator = Cmv2ActionEvaluator(_StubAdapter())
    desired = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    result = evaluator.evaluate(
        object_points, object_normals, sweep,
        candidate_actions=torch.zeros(1, 3, 18),
        candidate_valid_mask=torch.tensor([[True, True, False]]),
        desired_delta_xi=desired)
    assert result["schema"] == "cmv2_action_effect_v1"
    assert result["predicted_delta_xi"].shape == (1, 3, 6)
    assert result["encoded_tokens"].shape == (1, 3, 16, 40)
    assert torch.equal(result["encoded_tokens"], torch.zeros_like(result["encoded_tokens"]))
    assert result["effect_score"].shape == (1, 3)
    assert evaluator.select_best(result).tolist() == [1]
    assert torch.isinf(result["effect_score"][0, 2]).item() is True
    agreement = pairwise_rank_agreement(
        result["effect_score"].masked_fill(~result["candidate_valid_mask"], 0),
        torch.tensor([[0.1, 0.0, 0.5]]), result["candidate_valid_mask"])
    assert agreement.item() == 1.0


def test_no_future_reference_surface_can_enter_evaluator():
    with pytest.raises(ValueError, match="controller_fk"):
        _sweep(source="reference_future").validate()


def test_effect_metric_identity_and_rotation():
    predicted = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target = predicted.clone()
    metrics = effect_metrics(predicted, target)
    torch.testing.assert_close(metrics["translation_error_m"], torch.zeros(1))
    torch.testing.assert_close(metrics["rotation_error_rad"], torch.zeros(1))
    rotated = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966]])
    assert effect_metrics(rotated, target)["rotation_error_rad"].item() > 1.5


def test_reference_object_pose_delta_is_local_one_step_effect():
    current = torch.eye(4)[None]
    following = current.clone()
    following[0, 0, 3] = 0.02
    following[0, 0, 0] = 0.0
    following[0, 1, 1] = 0.0
    following[0, 0, 1] = -1.0
    following[0, 1, 0] = 1.0
    effect = object_pose_delta_to_xi(current, following)
    torch.testing.assert_close(effect[0, :3], torch.tensor([0.02, 0.0, 0.0]), atol=1e-6, rtol=0.0)
    assert abs(effect[0, 5].item() - 1.5707963) < 1e-5


def test_existing_residual_semantics_and_pinned_fk():
    base = torch.zeros(1, 18)
    residual = torch.zeros(1, 2, 18)
    residual[0, 1, 6] = 1.0
    current = torch.zeros(1, 18)
    lower = torch.tensor([-2.0] * 6 + [0.0] * 12)
    upper = torch.tensor([2.0] * 6 + [2.0] * 12)
    targets, details = compose_nominal_targets(
        base, residual, current, lower, upper,
        translation_scale_m=0.015, rotation_scale_rad=0.20, finger_scale_rad=0.08)
    torch.testing.assert_close(targets[0, 0], details["base_targets"][0, 0])
    assert targets[0, 1, 6] > targets[0, 0, 6]

    urdf = Path("src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")
    if not urdf.is_file():
        pytest.skip("machine-local Inspire URDF is not present in the isolated worktree")
    poses = build_nominal_link_poses(InspireKinematics(urdf), targets)
    assert poses.shape == (1, 2, 18, 4, 4)
    assert torch.isfinite(poses).all()
