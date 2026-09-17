import numpy as np
import pytest

from src.task.ObjectInteractionCm.tools.data.merge_dual_hand_stream import (
    merge_bilateral_hand_arrays,
    merge_bilateral_hand_stream,
)


def test_bilateral_stream_is_left_then_right_and_flow_is_stable():
    left = np.zeros((2, 3, 3), dtype=np.float64)
    right = np.ones((2, 2, 3), dtype=np.float64)
    out = merge_bilateral_hand_stream(left, right, left, right, left + 2.0, right + 3.0)
    assert out["hand_points"].shape == (2, 5, 3)
    np.testing.assert_allclose(out["hand_points"][:, :3], 0.0)
    np.testing.assert_allclose(out["hand_points"][:, 3:], 1.0)
    np.testing.assert_allclose(out["hand_flow"][:, :3], 2.0)
    np.testing.assert_allclose(out["hand_flow"][:, 3:], 3.0)


def test_bilateral_stream_rejects_mismatched_frames_and_nonfinite_values():
    with pytest.raises(ValueError, match="frame counts"):
        merge_bilateral_hand_arrays(np.zeros((2, 1, 3)), np.zeros((3, 1, 3)))
    bad = np.zeros((1, 1, 3), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        merge_bilateral_hand_arrays(bad, np.zeros_like(bad))
