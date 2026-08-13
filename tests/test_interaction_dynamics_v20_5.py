from pathlib import Path

import torch

from src.task.InteractionDynamics.research.gty_generalization.optimize_robot_from_gty import representative_frames
from src.task.InteractionDynamics.viewer_gty.robot_backend import UrdfHandBackend, scan_robot_assets


ASSETS = Path("data/raw_data/robot_hands")


def test_robot_asset_scan_finds_four_requested_hands() -> None:
    assert set(scan_robot_assets(ASSETS)) == {"Allegro", "LEAP", "Shadow", "Barrett"}


def test_allegro_fk_shape_limits_and_gradient() -> None:
    backend = UrdfHandBackend(scan_robot_assets(ASSETS)["Allegro"])
    q = backend.limits.mean(1).repeat(2, 1).requires_grad_()
    vertices = backend.vertices(q, torch.zeros(2, 3), torch.zeros(2, 3))
    assert backend.dof == 16 and vertices.shape[0] == 2 and vertices.shape[-1] == 3
    assert backend.faces.max() < vertices.shape[1]
    vertices.square().mean().backward(); assert q.grad is not None and torch.isfinite(q.grad).all()


def test_representative_frames_are_valid_and_unique() -> None:
    y = torch.zeros(8, 128, 7).transpose(0, 1)
    y[:, 2:, 3] = 3; y[:, :2, 3] = 1
    frames = representative_frames(y)
    assert frames == sorted(set(frames)) and len(frames) >= 4
    assert all(0 <= value <= 8 for value in frames)
