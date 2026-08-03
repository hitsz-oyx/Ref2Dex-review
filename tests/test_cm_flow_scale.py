from __future__ import annotations

import numpy as np

from src.task.Cm.compute_flow_scale import calibrate_flow_scale


def _write_stream(root, *, frames: int, points: int, candidate_mask: np.ndarray) -> None:
    stream = root / "sequence"
    stream.mkdir(parents=True)
    # x displacement equals frame index, which makes all expected RMS values
    # simple and independent of point identity.
    obj = np.zeros((frames, points, 3), dtype=np.float64)
    obj[..., 0] = np.arange(frames, dtype=np.float64)[:, None]
    np.savez(stream / "shared.npz", seq_id=np.asarray("sequence"), obj_points_world=obj)
    np.savez(stream / "left.npz", obj_candidate_mask_5cm=candidate_mask)


def test_calibration_uses_dataset_current_range_and_equal_strides(tmp_path) -> None:
    candidates = np.ones((5, 3), dtype=bool)
    _write_stream(tmp_path, frames=5, points=3, candidate_mask=candidates)
    result = calibrate_flow_scale(
        tmp_path, min_stride=1, max_stride=2, active_only=True, num_obj_points=512,
    )
    # Dataset currents are exactly [0, 1, 2], not the additional [3] that a
    # stride-1-only loop would include.  Each pair's norm is its stride.
    assert result["statistics_per_stride"]["1"]["num_pairs"] == 3
    assert result["statistics_per_stride"]["2"]["num_pairs"] == 3
    assert result["statistics_per_stride"]["1"]["flow_rms_m"] == 1.0
    assert result["statistics_per_stride"]["2"]["flow_rms_m"] == 2.0
    assert result["flow_target_rms_m"] == np.sqrt((1.0 + 4.0) / 2.0)


def test_calibration_caps_each_pair_weight_at_runtime_sample_count(tmp_path) -> None:
    candidates = np.zeros((3, 600), dtype=bool)
    candidates[0, :] = True
    candidates[1, :100] = True
    _write_stream(tmp_path, frames=3, points=600, candidate_mask=candidates)
    result = calibrate_flow_scale(
        tmp_path, min_stride=1, max_stride=1, active_only=True, num_obj_points=512,
    )
    # The two eligible current frames contribute 512 and 100 expected sampled
    # points respectively, rather than 600 and 100 raw candidates.
    assert result["statistics_num_points"] == 612
    assert result["statistics_per_stride"]["1"]["num_points"] == 612
    assert result["statistics_num_obj_points"] == 512
