from pathlib import Path

import numpy as np
import pytest
import torch

from src.task.CmDecoderv2.kinematics import InspireKinematics, QUERY_LINKS
from src.task.CmDecoderv2.rl.residual_contract import (
    NATIVE_DOF_NAMES, actual_queries, coupled_finger_bounds, expand_native_targets,
    inverse_pose, matrix_pose, native_sim_indices, native_to_sim, pose_matrix,
    sim_to_native, wrist_native_target,
)

URDF = Path(__file__).resolve().parents[1] / "assets/inspire_hand_new/inspire_hand_right.urdf"


@pytest.mark.parametrize("names", [list(NATIVE_DOF_NAMES), list(reversed(NATIVE_DOF_NAMES))])
def test_sim_mapping_is_name_based_and_invertible(names):
    index = torch.tensor(native_sim_indices(names))
    q = torch.arange(54).reshape(3, 18)
    torch.testing.assert_close(sim_to_native(native_to_sim(q, index), index), q)
    for j, name in enumerate(names):
        torch.testing.assert_close(native_to_sim(q, index)[:, j], q[:, NATIVE_DOF_NAMES.index(name)])


def test_unknown_dof_names_fail_closed():
    with pytest.raises(ValueError):
        native_sim_indices(["joint"] * 18)


@pytest.mark.skipif(not URDF.exists(), reason="Inspire asset unavailable")
def test_actual_query_parity_preserves_uncoupled_state():
    fk = InspireKinematics(URDF)
    q = np.array([-.2, .1, 1.2, .2, -1.48, -2.1, .4, .9, .2, .7, .3, .6, .5, .1, .5, .1, .4, .2])
    links = fk.link_transforms_native(q)
    obj = np.eye(4)
    obj[:3, 3] = [-.1, -.2, .9]
    state, features = actual_queries(torch.tensor(q[None], dtype=torch.float32),
                                    torch.tensor(np.stack([links[k] for k in QUERY_LINKS])[None], dtype=torch.float32),
                                    torch.tensor(obj[None], dtype=torch.float32))
    torch.testing.assert_close(features[0], torch.from_numpy(fk.query_features_native(q, obj)), atol=1e-6, rtol=1e-5)
    assert state.shape == (1, 15)
    assert features.shape == (1, 18, 10)
    coupled = fk.query_features(q[[6, 8, 10, 12, 14, 15]], links["hand_base_link"], obj)
    assert np.max(np.abs(coupled - features[0].numpy())) > .01


@pytest.mark.skipif(not URDF.exists(), reason="Inspire asset unavailable")
def test_wrist_inverse_matches_urdf_fk_and_nearby_angle_branch():
    fk = InspireKinematics(URDF)
    rng = np.random.default_rng(42)
    q = rng.uniform(-3, 3, (48, 18))
    q[:, 2] = 1.1
    q[0, 3:6] = [.2, -1.486, -2.096]
    pose = np.stack([fk.wrist_pose_from_native(row) for row in q])
    target = wrist_native_target(torch.tensor(pose, dtype=torch.float64),
                                 torch.tensor(fk._zero_hand_base_inverse), torch.tensor(q[:, :6]))
    np.testing.assert_allclose(target.numpy(), q[:, :6], atol=1e-9)


def test_pose_inverse_xyzw_and_quaternion_sign():
    pose = torch.tensor([[1., 2., 3., .1, .2, .3, .9]])
    matrix = pose_matrix(pose)
    torch.testing.assert_close(matrix @ inverse_pose(matrix), torch.eye(4)[None], atol=1e-6, rtol=1e-6)
    pose[:, 3:] *= -1
    torch.testing.assert_close(pose_matrix(pose), matrix)
    torch.testing.assert_close(pose_matrix(matrix_pose(matrix)), matrix)


def test_coupled_limits_are_applied_before_expansion():
    lower, upper = torch.zeros(18), torch.ones(18)
    upper[16] = .12
    lo, hi = coupled_finger_bounds(lower, upper)
    targets = expand_native_targets(torch.full((2, 6), 99.).clamp(lo, hi))
    assert (targets >= lower).all()
    assert (targets <= upper + 1e-6).all()
    assert targets[0, 15] == pytest.approx(.2)
    assert targets[0, 16] == pytest.approx(.12)
