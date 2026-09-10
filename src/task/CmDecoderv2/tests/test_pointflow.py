from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.task.CmDecoderv2.kinematics import extract_finger_q
from src.task.CmDecoderv2.pointflow import DifferentiableInspireSurface, world_to_object


URDF = Path("src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")
VIEW_ROOT = Path("data/processed_data/cm_decoder_v2/dexplore_rl_v1_1")
GEOMETRY_ROOT = Path("data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/train/inspire_rl/s1_airplane_lift/geometry")
V13_VIEW_ROOT = Path("data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135_pilot")
V13_GEOMETRY_ROOT = Path(
    "data/processed_data/object_interaction_cm_dexplore_rl_v1_3/"
    "sequences/train/inspire_rl/s1_airplane_lift/geometry"
)


def test_differentiable_surface_matches_cache_and_has_gradients() -> None:
    surface = DifferentiableInspireSurface(URDF, sample_count=1538, surface_seed=2024)
    q_native = np.load(VIEW_ROOT / "sequences/train/s1_airplane_lift/q_native.npy", mmap_mode="r")[0]
    wrist = np.load(VIEW_ROOT / "sequences/train/s1_airplane_lift/wrist_pose_world.npy", mmap_mode="r")[0]
    expected = np.load(GEOMETRY_ROOT / "hand_points_world.npy", mmap_mode="r")[0]
    finger_q = torch.tensor(extract_finger_q(q_native)[None], dtype=torch.float32, requires_grad=True)
    wrist_pose = torch.from_numpy(np.asarray(wrist[None], dtype=np.float32).copy())
    actual = surface(finger_q, wrist_pose)
    np.testing.assert_allclose(actual.detach().numpy()[0], expected, atol=2e-5, rtol=0.0)
    actual.square().mean().backward()
    assert finger_q.grad is not None
    assert torch.isfinite(finger_q.grad).all()


def test_world_to_object_supports_horizon_points() -> None:
    pose = torch.eye(4).expand(2, 4, 4).clone()
    pose[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    points = torch.zeros(2, 4, 1538, 3)
    result = world_to_object(points, pose)
    assert result.shape == points.shape
    assert torch.isfinite(result).all()


def test_v13_surface_uses_10135_cache_correspondence_and_has_gradients() -> None:
    surface = DifferentiableInspireSurface(
        URDF,
        sample_count=10135,
        surface_seed=2024,
        surface_sampling="v1_3_cache",
    )
    q_native = np.load(
        V13_VIEW_ROOT / "sequences/train/s1_airplane_lift/q_native.npy",
        mmap_mode="r",
    )[0]
    wrist = np.load(
        V13_VIEW_ROOT / "sequences/train/s1_airplane_lift/wrist_pose_world.npy",
        mmap_mode="r",
    )[0]
    expected = np.load(V13_GEOMETRY_ROOT / "knn_hand_points_world.npy", mmap_mode="r")[0]
    finger_q = torch.tensor(
        extract_finger_q(q_native)[None],
        dtype=torch.float32,
        requires_grad=True,
    )
    wrist_pose = torch.from_numpy(np.asarray(wrist[None], dtype=np.float32).copy())
    actual = surface(finger_q, wrist_pose)
    assert actual.shape == (1, 10135, 3)
    np.testing.assert_allclose(actual.detach().numpy()[0], expected, atol=2e-5, rtol=0.0)
    actual.square().mean().backward()
    assert finger_q.grad is not None
    assert torch.isfinite(finger_q.grad).all()
