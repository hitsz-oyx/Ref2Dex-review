from __future__ import annotations

import numpy as np

from src.task.CmDecoderv2.research.inspire_rollout_effect.run import (
    _effect_colors,
    _effective_object_flow,
    _summary,
)


def test_invalid_oicm_effect_is_semantically_zero() -> None:
    raw = np.asarray([[0.01, -0.02, 0.03]], dtype=np.float32)
    np.testing.assert_array_equal(_effective_object_flow(raw, False), np.zeros_like(raw))
    np.testing.assert_array_equal(_effective_object_flow(raw, True), raw)


def test_summary_separates_raw_and_effective_far_effect() -> None:
    arrays = {
        "min_hand_object_distance_mm": np.asarray([100.0, 40.0], dtype=np.float32),
        "effect_rms_mm": np.asarray([12.0, 5.0], dtype=np.float32),
        "effective_effect_rms_mm": np.asarray([0.0, 5.0], dtype=np.float32),
        "oicm_sample_valid": np.asarray([False, True]),
    }
    summary = _summary(arrays)
    assert summary["far_gt_50mm"]["frames"] == 1
    assert summary["far_gt_50mm"]["effect_rms_mean_mm"] == 12.0
    assert summary["far_gt_50mm"]["effective_effect_rms_mean_mm"] == 0.0
    assert summary["near_le_50mm"]["valid_ratio"] == 1.0


def test_summary_includes_optional_display_only_gt_metrics() -> None:
    arrays = {
        "min_hand_object_distance_mm": np.asarray([10.0, 100.0], dtype=np.float32),
        "effect_rms_mm": np.asarray([2.0, 3.0], dtype=np.float32),
        "effective_effect_rms_mm": np.asarray([2.0, 0.0], dtype=np.float32),
        "oicm_sample_valid": np.asarray([True, False]),
        "gt_effect_rms_mm": np.asarray([1.0, 4.0], dtype=np.float32),
        "pred_gt_effect_epe_mm": np.asarray([0.5, 4.0], dtype=np.float32),
    }
    summary = _summary(arrays)
    assert summary["gt_object_flow_display_only"] is True
    assert summary["overall_gt_effect_rms_mean_mm"] == 2.5
    assert summary["overall_pred_gt_effect_epe_mm"] == 2.25


def test_effect_colors_are_rgb_uint8() -> None:
    colors = _effect_colors(np.asarray([0.0, 5.0, 10.0], dtype=np.float32), 10.0)
    assert colors.shape == (3, 3)
    assert colors.dtype == np.uint8
    assert not np.array_equal(colors[0], colors[-1])
