from pathlib import Path
import numpy as np
import pytest

from src.task.CmDecoderv2.kinematics import InspireKinematics, expand_finger_q, extract_finger_q
from src.task.CmDecoderv2.research.dexplore_contract_audit.contracts import (
    existing_splits, joint_least_squares, mimic_residual, reconstruct_native,
    sampled_frames, validate_pair,
)

URDF = Path("src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")


@pytest.fixture
def kin():
    if not URDF.is_file():
        pytest.skip("Local licensed Inspire asset unavailable")
    return InspireKinematics(URDF)


def test_exact_coupled_state_and_clamp(kin):
    q = expand_finger_q(np.array([.3, .4, .5, .6, .2, .3]))[None]
    np.testing.assert_allclose(mimic_residual(q), 0, atol=1e-12)
    np.testing.assert_allclose(reconstruct_native(q, kin), q)
    q[:, 6] = 4
    assert reconstruct_native(q, kin)[0, 6] == kin.finger_upper[0]


def test_independent_distal_state_not_represented(kin):
    q = expand_finger_q(np.array([.3, .4, .5, .6, .2, .3]))[None]
    q[0, 7] += .7
    assert mimic_residual(q)[0, 0] == pytest.approx(.7)
    assert reconstruct_native(q, kin)[0, 7] == pytest.approx(.315)
    ls = joint_least_squares(q, kin)
    assert np.square(ls[:, 6:] - q[:, 6:]).sum() < np.square(reconstruct_native(q, kin)[:, 6:] - q[:, 6:]).sum()
    # Closed-form normal equation for this single uncapped finger pair.
    assert ls[0, 6] == pytest.approx((q[0, 6] + 1.05 * q[0, 7]) / (1 + 1.05 ** 2))


def test_native_observation_fk_does_not_reexpand_mimic_joint(kin):
    native = expand_finger_q(np.array([.3, .4, .5, .6, .2, .3]))
    native[7] += .4
    wrist = kin.wrist_pose_from_native(native)
    actual = kin.query_features_native(native, np.eye(4))
    coupled = kin.query_features(extract_finger_q(native), wrist, np.eye(4))
    assert not np.allclose(actual[:, 1:4], coupled[:, 1:4])


def test_pair_fields_and_sign_bits():
    geo = np.zeros((5, 598), dtype=np.float32)
    geo[:, 204] = 1
    actual = geo.copy()
    actual[:, 373:391] = .3
    validate_pair(geo, actual)
    actual[0, 51] = -0.0
    with pytest.raises(ValueError, match="Columns"):
        validate_pair(geo, actual)
    with pytest.raises(ValueError, match="frame counts"):
        validate_pair(geo, np.repeat(geo[:1], 6, axis=0))
    actual = geo.copy()
    actual[0, 375] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_pair(geo, actual)


def test_sampling_and_identity():
    np.testing.assert_array_equal(sampled_frames(3, 8), [0, 1, 2])
    np.testing.assert_array_equal(sampled_frames(9, 3), [0, 4, 8])
    index = {"sequences": {"train": [{"parent_seq_id": "s1/mouse_lift"}], "val": [], "test": []}}
    assert existing_splits(index) == {"s1_mouse_lift": "train"}
    index["sequences"]["test"] = index["sequences"]["train"]
    with pytest.raises(ValueError, match="Duplicate"):
        existing_splits(index)
    with pytest.raises(ValueError):
        sampled_frames(0, 3)


def test_invalid_native_shape_and_nan():
    with pytest.raises(ValueError):
        mimic_residual(np.zeros((3, 12)))
    with pytest.raises(ValueError):
        mimic_residual(np.full((3, 18), np.nan))
