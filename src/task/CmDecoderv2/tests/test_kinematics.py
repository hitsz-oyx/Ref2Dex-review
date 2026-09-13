from __future__ import annotations

from pathlib import Path

import numpy as np

from src.task.CmDecoderv2.kinematics import (
    INDEPENDENT_FINGER_NATIVE_INDICES,
    QUERY_LINKS,
    InspireKinematics,
    expand_finger_q,
    extract_finger_q,
)


URDF = Path("src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")


def test_mimic_mapping_and_independent_roundtrip() -> None:
    finger = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    native = expand_finger_q(finger)
    np.testing.assert_allclose(native[INDEPENDENT_FINGER_NATIVE_INDICES], finger)
    np.testing.assert_allclose(native[[7, 9, 11, 13, 16, 17]], [0.21, 0.315, 0.42, 0.525, 0.42, 0.56])
    np.testing.assert_allclose(extract_finger_q(native), finger)


def test_explicit_wrist_state_reproduces_native_fk() -> None:
    model = InspireKinematics(URDF)
    finger = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.3])
    native = expand_finger_q(finger, wrist_native=np.asarray([0.1, -0.2, 0.3, 0.2, -0.1, 0.4]))
    native_links = model.link_transforms_native(native)
    wrist = native_links["hand_base_link"]
    state_links = model.link_transforms_from_state(finger, wrist)
    for link in QUERY_LINKS:
        np.testing.assert_allclose(state_links[link], native_links[link], atol=1e-7)
    features = model.query_features(finger, wrist, np.eye(4))
    assert features.shape == (18, 10)
    assert np.isfinite(features).all()
