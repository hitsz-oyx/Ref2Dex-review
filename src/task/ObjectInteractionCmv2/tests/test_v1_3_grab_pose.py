import numpy as np
import pytest

from src.task.ObjectInteractionCmv2.grab import GrabManoTransitions
from src.task.ObjectInteractionCmv2.tests.test_v1_2_grab_contract import make_cache


def test_direct_pose_gt_matches_current_object_frame_and_rejects_mismatch(tmp_path):
    geometry = make_cache(tmp_path)
    pose = np.load(geometry / "obj_pose_world.npy")
    points = np.load(geometry / "obj_points_pool_world.npy")
    local = np.linspace(-0.01, 0.01, 4096, dtype=np.float32)
    for t in range(4):
        points[t, :, 0] = pose[t, 0, 3] + local
    np.save(geometry / "obj_points_pool_world.npy", points)
    dataset = GrabManoTransitions(tmp_path / "index.json", tmp_path / "run_manifest.json",
                                  "train", direct_pose_gt=True)
    sample = dataset[0]
    assert np.allclose(sample["delta_translation_gt"].numpy(), [0.1, 0, 0], atol=1e-6)
    assert np.allclose(sample["delta_rotation_gt"].numpy(), np.eye(3), atol=1e-6)
    assert np.allclose(sample["obj_flow_gt"].numpy(), [0.1, 0, 0], atol=1e-6)
    points[1, 0, 0] += 0.001
    np.save(geometry / "obj_points_pool_world.npy", points)
    # mmap arrays are process cached; clear them after updating the fixture file.
    from src.task.ObjectInteractionCmv2.grab import _arrays
    _arrays.cache_clear()
    with pytest.raises(ValueError, match="Pose/point mismatch"):
        dataset[0]
