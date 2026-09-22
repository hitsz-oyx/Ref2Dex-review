from __future__ import annotations

import numpy as np

from src.task.ObjectInteractionCmv2.tools.data.repair_lean_object_pose_v114d1 import (
    _validate_pose,
    recover_rigid_poses,
)


def test_recover_rigid_pose_replays_fixed_correspondence() -> None:
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(4096, 3)).astype(np.float32) * 0.1
    angles = (0.0, 0.2, -0.4)
    frames = []
    for frame, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        translation = np.asarray([frame * 0.01, -frame * 0.02, 0.03], dtype=np.float32)
        frames.append(reference @ rotation.T + translation)
    points = np.stack(frames)
    poses, fit = recover_rigid_poses(points)
    assert fit < 1e-6
    assert _validate_pose(points, poses) < 1e-6
    assert np.allclose(poses[1, :3, 3], [0.01, -0.02, 0.0], atol=1e-6)
