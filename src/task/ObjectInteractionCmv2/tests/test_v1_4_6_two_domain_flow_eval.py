from __future__ import annotations

import math

import pytest
import torch

from src.task.ObjectInteractionCmv2.eval_two_domain_mano import FlowMetrics, _stable_choice


def test_flow_metrics_separate_static_points_and_angle():
    prediction = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    metrics = FlowMetrics(angle_epsilon_m=1e-6)
    metrics.update(prediction, target)
    result = metrics.summary()
    assert result["samples"] == 1 and result["object_points"] == 3
    assert result["flow_epe_micro_mm"] == pytest.approx(1000 * math.sqrt(2) / 3)
    assert result["prediction_flow_magnitude_micro_mm"] == pytest.approx(2000 / 3)
    assert result["gt_flow_magnitude_micro_mm"] == pytest.approx(2000 / 3)
    assert result["flow_angle_mean_deg"] == pytest.approx(45.0)
    assert result["angle_valid_points"] == 2 and result["angle_excluded_points"] == 1
    assert result["gt_static_points"] == 1 and result["prediction_static_points"] == 1
    assert result["both_static_points"] == 1


def test_stable_choice_is_reproducible_and_without_replacement():
    first = _stable_choice(100, 20, seed=42, domain="arctic", stride=5)
    assert first == _stable_choice(100, 20, seed=42, domain="arctic", stride=5)
    assert first != _stable_choice(100, 20, seed=42, domain="arctic", stride=6)
    assert len(first) == len(set(first)) == 20
    with pytest.raises(ValueError, match="only 3 transitions"):
        _stable_choice(3, 4, seed=42, domain="grab", stride=1)
