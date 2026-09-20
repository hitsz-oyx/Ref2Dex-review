from pathlib import Path

import torch

from src.task.CmResidual.dexplore_cm_geometry import (
    ACTION_DIM, DExploreCmv2GeometryBridge, HAND_POINTS, OBJECT_POINTS,
    dexplore_action_to_native_targets, dexplore_root_pose,
)


ROOT = Path(__file__).resolve().parents[4]


def test_dexplore_pd_mapping_keeps_wrist_delta_finger_range_and_coupling():
    current = torch.zeros(1, ACTION_DIM)
    current[:, :6] = torch.tensor([1., 2., 3., .1, .2, .3])
    action = torch.zeros_like(current)
    action[:, :6] = torch.tensor([.2, -.4, .5, .1, -.2, .3])
    lower = torch.zeros(ACTION_DIM)
    upper = torch.ones(ACTION_DIM)
    target = dexplore_action_to_native_targets(action, current, lower, upper)
    torch.testing.assert_close(target[:, :3], current[:, :3] + action[:, :3])
    torch.testing.assert_close(target[:, 3:6], current[:, 3:6] + torch.pi * action[:, 3:6])
    torch.testing.assert_close(target[:, 6], torch.full((1,), .5))
    torch.testing.assert_close(target[:, 7], torch.full((1,), .525))
    torch.testing.assert_close(target[:, 16], torch.full((1,), .3))
    torch.testing.assert_close(target[:, 17], torch.full((1,), .4))


def test_dexplore_root_xyzw_pose_is_identity_and_translation_preserving():
    root = torch.zeros(2, 13)
    root[:, :3] = torch.tensor([[1., 2., 3.], [-1., 0., .5]])
    root[:, 6] = 1.0
    pose = dexplore_root_pose(root)
    torch.testing.assert_close(pose[:, :3, :3], torch.eye(3).expand(2, 3, 3))
    torch.testing.assert_close(pose[:, :3, 3], root[:, :3])


def test_bridge_builds_exact_cmv2_geometry_and_nominal_flow_from_pinned_assets():
    hand = ROOT / "third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
    obj = ROOT / "third_party/DExplore/dexplore/data/assets/mjcf/airplane.urdf"
    bridge = DExploreCmv2GeometryBridge(hand_urdf=hand, object_urdf=obj, device="cpu", seed=42)
    current = torch.zeros(2, ACTION_DIM)
    root = torch.zeros(2, 13)
    root[:, 6] = 1.0
    geometry = bridge.current(current, root)
    assert geometry.object_points.shape == (2, OBJECT_POINTS, 3)
    assert geometry.hand_points.shape == (2, HAND_POINTS, 3)
    torch.testing.assert_close(geometry.object_normals.norm(dim=-1), torch.ones(2, OBJECT_POINTS), atol=1e-5, rtol=0)
    torch.testing.assert_close(geometry.hand_normals.norm(dim=-1), torch.ones(2, HAND_POINTS), atol=1e-5, rtol=0)
    actions = torch.zeros(2, 3, ACTION_DIM)
    points, normals, flow = bridge.nominal_hand_sweep(current, actions, torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM))
    assert points.shape == normals.shape == (2, HAND_POINTS, 3)
    assert flow.shape == (2, 3, HAND_POINTS, 3)
    assert torch.isfinite(flow).all()
    # DExplore's zero action targets finger-range midpoints, not the current
    # joint state; identical candidates must nevertheless yield identical flow.
    torch.testing.assert_close(flow[:, 0], flow[:, 1], atol=1e-6, rtol=0)
    torch.testing.assert_close(flow[:, 1], flow[:, 2], atol=1e-6, rtol=0)
