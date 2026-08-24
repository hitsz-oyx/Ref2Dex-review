import numpy as np
import pytest

from src.task.CmDecoder.viewer import _compose_hand_q, _pair_indices


def test_pair_indices_match_contiguous_stride_selection():
    source_ids = np.array([1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15])
    current, target = _pair_indices(source_ids, stride=3)

    assert current.tolist() == [0, 4, 5, 6, 7, 8]
    assert target.tolist() == [3, 7, 8, 9, 10, 11]


def test_compose_hand_q_uses_current_arm_as_frame_and_replaces_fingers():
    q_full = np.arange(12, dtype=np.float32)
    hand_q = np.linspace(0.1, 0.6, 6, dtype=np.float32)

    result = _compose_hand_q(q_full, hand_q)

    np.testing.assert_array_equal(result[:6], q_full[:6])
    np.testing.assert_allclose(result[6:], hand_q)
    np.testing.assert_array_equal(q_full, np.arange(12, dtype=np.float32))


def test_compose_hand_q_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="q_full"):
        _compose_hand_q(np.zeros(6), np.zeros(6))
