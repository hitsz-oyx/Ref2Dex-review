from __future__ import annotations

import torch

from src.task.CmDecoderv2.model import TemporalD2Core
from src.task.CmDecoderv2.runner import axis_angle_to_matrix, rotation_geodesic


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(batch, 4, 16, 32, generator=generator),
        torch.randn(batch, 4, 16, 3, generator=generator),
        torch.randn(batch, 4, 16, 3, generator=generator),
        torch.randn(batch, 15, generator=generator),
        torch.randn(batch, 18, 10, generator=generator),
    )


def test_temporal_d2_shapes_identity_initialization_and_gradients() -> None:
    model = TemporalD2Core(dropout=0.0)
    values = _inputs()
    result = model(*values)
    assert result["pred_q_delta"].shape == (2, 4, 6)
    assert result["pred_wrist_translation"].shape == (2, 4, 3)
    assert result["pred_wrist_rotvec"].shape == (2, 4, 3)
    assert torch.count_nonzero(result["pred_q_delta"]) == 0
    loss = sum(value.square().mean() for value in result.values())
    loss.backward()
    assert model.q_head.weight.grad is not None


def test_slot_permutation_invariance() -> None:
    model = TemporalD2Core(dropout=0.0).eval()
    with torch.no_grad():
        model.q_head.weight.normal_(generator=torch.Generator().manual_seed(3))
    cm, pos, normal, state, links = _inputs(batch=1)
    permutations = torch.stack([torch.randperm(16, generator=torch.Generator().manual_seed(10 + frame)) for frame in range(4)])
    gather_cm = permutations[None, :, :, None].expand(1, 4, 16, 32)
    gather_anchor = permutations[None, :, :, None].expand(1, 4, 16, 3)
    reference = model(cm, pos, normal, state, links)["pred_q_delta"]
    permuted = model(torch.gather(cm, 2, gather_cm), torch.gather(pos, 2, gather_anchor), torch.gather(normal, 2, gather_anchor), state, links)["pred_q_delta"]
    torch.testing.assert_close(permuted, reference, atol=2e-6, rtol=2e-6)


def test_rotation_geodesic_is_finite_at_identity() -> None:
    rotvec = torch.zeros(2, 4, 3, requires_grad=True)
    matrix = axis_angle_to_matrix(rotvec)
    target = torch.eye(3).expand_as(matrix)
    loss = rotation_geodesic(matrix, target).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(rotvec.grad).all()
