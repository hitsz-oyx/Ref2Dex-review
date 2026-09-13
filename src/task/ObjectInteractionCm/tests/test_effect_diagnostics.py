import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.task.ObjectInteractionCm.research.cross_source_effect.diagnose import (
    cluster_summary, error_attribution, hand_baselines, rigid_fit, token_diagnostics,
)


def test_rigid_fit_recovers_rotation_translation_and_not_reflection():
    rng = np.random.default_rng(42)
    points, query = rng.normal(size=(50, 3)), rng.normal(size=(15, 3))
    rotation = Rotation.from_rotvec([.2, -.4, .3])
    shift = np.array([.1, .02, -.3])
    target = rotation.apply(points) + shift
    result, fallback, residual = rigid_fit(points, target, query)
    np.testing.assert_allclose(result, rotation.apply(query) + shift, atol=1e-12)
    assert not fallback and residual < 1e-9
    reflected = points.copy()
    reflected[:, 0] *= -1
    _, _, residual = rigid_fit(points, reflected, query)
    assert residual > 1


@pytest.mark.parametrize("count", [1, 2, 3])
def test_degenerate_hand_points_use_translation(count):
    hand = np.arange(count)[:, None] * np.array([[1., 0., 0.]])
    query = np.array([[.1, .2, .3]])
    flow = np.broadcast_to([.01, -.02, .03], hand.shape)
    mean, rigid, fallback, residual = hand_baselines(query, hand, flow)
    np.testing.assert_allclose(mean, [[.01, -.02, .03]])
    np.testing.assert_allclose(rigid, mean)
    assert fallback and residual < 1e-10


def test_hand_baseline_has_no_object_gt_dependency():
    hand = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0.]])
    flow = np.broadcast_to([0, .003, .004], hand.shape)
    obj = np.array([[.03, .04, .05]])
    mean, rigid, fallback, _ = hand_baselines(obj, hand, flow)
    assert not fallback
    np.testing.assert_allclose(rigid, mean, atol=1e-12)
    assert np.linalg.norm(mean) * 1000 == pytest.approx(5)
    with pytest.raises(ValueError):
        rigid_fit(np.empty((0, 3)), np.empty((0, 3)), obj)


def test_noisy_rigid_fit_matches_independent_svd_solution():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(40, 3)) * .03
    y = Rotation.from_rotvec([.1, .2, -.3]).apply(x) + rng.normal(size=x.shape) * .001
    query = rng.normal(size=(12, 3)) * .05
    u, _, vh = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
    correction = np.diag([1., 1., np.linalg.det(u @ vh)])
    expected = (query - x.mean(0)) @ (u @ correction @ vh) + y.mean(0)
    fitted, fallback, _ = rigid_fit(x, y, query)
    assert not fallback
    np.testing.assert_allclose(fitted, expected, atol=1e-12)


def test_slot_effective_count_can_be_constant_even_when_tokens_collapse():
    weights = np.tile([.99, .01, 0.], (16, 1))
    usage = weights.mean(axis=-1)
    usage /= usage.sum()
    assert np.exp(-(usage * np.log(usage)).sum()) == pytest.approx(16)
    rank, spread, _ = token_diagnostics(np.ones((16, 32)), np.zeros((16, 3)), np.eye(3))
    assert rank == 0 and spread == 0


def test_cluster_draws_pair_baselines_within_sequences():
    rows = [dict(sequence_id="a", error=1., baseline=3.)] * 9
    rows += [dict(sequence_id="b", error=10., baseline=12.)]
    result = cluster_summary(rows, keys=("error", "baseline"))
    assert result["frame_micro"]["baseline"] - result["frame_micro"]["error"] == pytest.approx(2)
    np.testing.assert_allclose(np.array(result["frame_micro_ci95"]["baseline"])
                               - np.array(result["frame_micro_ci95"]["error"]), [2, 2])


def test_sparse_group_error_share_does_not_equal_error_excess():
    rows = [dict(source="grab", epe_mm=10., zero_epe_mm=9., mse_mm2=100., active_fraction=.01)]
    rows += [dict(source="grab", epe_mm=5., zero_epe_mm=20., mse_mm2=25., active_fraction=.5)] * 9
    low = error_attribution(rows)["grab"]["active_below_0.1"]
    assert low["zero_gate_improvement_total_mm"] == pytest.approx(.1)
    assert low["perfect_subset_improvement_upper_mm"] == pytest.approx(1.)
    assert low["full_epe_fraction"] == pytest.approx(10 / 55)
