from types import SimpleNamespace

import numpy as np
import torch

from src.task.InteractionDynamics.dataset import inverse_transform_points, transform_points
from src.task.Posetoken.dataset import build_canonical_patch_map
from src.task.Posetoken.model import StaticPoseEncoder


def _pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def test_global_se3_does_not_change_root_points_or_pose_tokens():
    rng = np.random.default_rng(4)
    points_root = rng.normal(size=(96, 3)).astype(np.float32) * .05
    canonical = rng.normal(size=(96, 3)).astype(np.float32) * .05
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
    root_pose = _pose(np.eye(3, dtype=np.float32), np.array([.3, -.2, .5], np.float32))
    global_pose = _pose(rotation, np.array([2., -1., .4], np.float32))
    world_a = inverse_transform_points(points_root, root_pose)
    world_b = inverse_transform_points(world_a, global_pose)
    root_pose_b = global_pose @ root_pose
    recovered_a = transform_points(world_a, root_pose)
    recovered_b = transform_points(world_b, root_pose_b)
    np.testing.assert_allclose(recovered_a, recovered_b, atol=1e-5, rtol=1e-5)

    meta = SimpleNamespace(model_dim=24, global_dim=16, attention_heads=6,
                           patch_size=8, num_patches=4, num_hand_points=96)
    model = StaticPoseEncoder(SimpleNamespace(meta=meta)).eval()
    patch = build_canonical_patch_map(torch.from_numpy(canonical), 4, 8)
    def encode(points: np.ndarray):
        return model({
            "hand_points_root": torch.from_numpy(points)[None],
            "hand_cano_points": torch.from_numpy(canonical)[None],
            "patch_knn_idx": patch[None],
        })
    with torch.no_grad():
        output_a, output_b = encode(recovered_a), encode(recovered_b)
    torch.testing.assert_close(output_a["pose_tokens"], output_b["pose_tokens"],
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output_a["global_pose_token"], output_b["global_pose_token"],
                               atol=1e-5, rtol=1e-5)
