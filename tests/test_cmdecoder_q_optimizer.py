import numpy as np
import torch

from src.task.CmDecoder.q_optimizer import (
    _axis_motion,
    optimize_q_from_hand_points,
    reconstruct_hand_points,
    rotvec_to_matrix,
)


def test_axis_motion_rotates_about_z():
    transform = _axis_motion(np.array([0.0, 0.0, 1.0]), torch.tensor([torch.pi / 2]), "revolute")
    point = torch.tensor([1.0, 0.0, 0.0, 1.0])

    result = transform[0] @ point

    torch.testing.assert_close(result[:3], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0.0)


def test_rotvec_identity_has_finite_zero_gradient():
    rotvec = torch.zeros((1, 3), requires_grad=True)
    matrix = rotvec_to_matrix(rotvec)

    matrix.sum().backward()

    torch.testing.assert_close(matrix, torch.eye(3).unsqueeze(0))
    assert torch.isfinite(rotvec.grad).all()


class _FakeHand:
    device = torch.device("cpu")
    q_lower = torch.zeros(6)
    q_upper = torch.ones(6)

    def bind_points(self, hand_q, current_points):
        return None

    def points(self, hand_q, *, bindings=None, cached_link_index=None, cached_local_points=None):
        del bindings, cached_link_index, cached_local_points
        zeros = torch.zeros_like(hand_q)
        return torch.stack([hand_q, zeros, zeros], dim=-1)


def test_q_optimization_starts_at_qt_and_reduces_point_error():
    hand = _FakeHand()
    q_t = torch.full((1, 6), 0.1)
    q_target = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    current = hand.points(q_t)
    target = hand.points(q_target)

    result = optimize_q_from_hand_points(
        hand,
        q_t=q_t,
        current_hand_points=current,
        target_hand_points=target,
        steps=100,
        lr=0.05,
        prior_weight=0.0,
    )

    initial_error = torch.linalg.vector_norm(current - target, dim=-1).mean()
    fitted_error = torch.linalg.vector_norm(result["points"] - target, dim=-1).mean()
    assert fitted_error < initial_error * 0.02
    assert torch.all(result["q"] >= hand.q_lower)
    assert torch.all(result["q"] <= hand.q_upper)
    assert result["wrist_delta_translation"].shape == (1, 3)
    assert result["wrist_delta_rotvec"].shape == (1, 3)


def test_reconstruct_hand_points_propagates_q_and_wrist_gradients():
    hand = _FakeHand()
    q_t = torch.full((1, 6), 0.1)
    current = hand.points(q_t)
    predicted_q = torch.full((1, 6), 0.2, requires_grad=True)
    translation = torch.tensor([[0.01, -0.02, 0.03]], requires_grad=True)
    rotvec = torch.tensor([[0.0, 0.0, 0.1]], requires_grad=True)

    points = reconstruct_hand_points(
        hand,
        q_t=q_t,
        current_hand_points=current,
        predicted_q=predicted_q,
        wrist_translation=translation,
        wrist_rotvec=rotvec,
    )
    points.square().mean().backward()

    assert points.shape == (1, 6, 3)
    assert torch.isfinite(predicted_q.grad).all() and predicted_q.grad.abs().sum() > 0
    assert torch.isfinite(translation.grad).all() and translation.grad.abs().sum() > 0
    assert torch.isfinite(rotvec.grad).all() and rotvec.grad.abs().sum() > 0


def test_reconstruct_hand_points_accepts_cached_binding():
    hand = _FakeHand()
    q_t = torch.full((1, 6), 0.1)
    current = hand.points(q_t)
    predicted_q = torch.full((1, 6), 0.2)
    translation = torch.zeros((1, 3))
    rotvec = torch.zeros((1, 3))
    cached_link = torch.zeros((6,), dtype=torch.int16)
    cached_local = torch.zeros((6, 3))

    result = reconstruct_hand_points(
        hand,
        q_t=q_t,
        current_hand_points=current,
        predicted_q=predicted_q,
        wrist_translation=translation,
        wrist_rotvec=rotvec,
        cached_link_index=cached_link,
        cached_local_points=cached_local,
    )

    assert result.shape == (1, 6, 3)
