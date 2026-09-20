import copy
import numpy as np
import pytest

from src.task.CmResidual.v121e_snapshot import PRODUCER, evaluate_state


def _record():
    roots = np.zeros((2, 3, 13), dtype=np.float64)
    roots[..., 6] = 1.0
    return {
        "state_id": "state", "candidate_action_sha256": "a" * 64,
        "task_indices_before": [0, 0, 0, 10], "object_actor_index": 1,
        "post_actor_root_state": roots.tolist(),
        "post_dof_state": np.zeros((2, 18, 2)).tolist(),
        "post_ig": np.zeros((2, 18)).tolist(), "physics_score": [1.0, 1.0],
        "score_producer": PRODUCER,
        "score_baseline": "shared_canonical_collected_pre_action",
    }


def test_snapshot_parity_accepts_identical_duplicate_arms():
    result = evaluate_state(_record(), _record())
    assert result["valid"]
    assert all(result["gates"].values())


def test_snapshot_parity_rejects_cross_method_object_drift():
    prefix, restore = _record(), _record()
    restore = copy.deepcopy(restore)
    roots = np.asarray(restore["post_actor_root_state"])
    roots[:, 1, 0] = 1e-3
    restore["post_actor_root_state"] = roots.tolist()
    result = evaluate_state(prefix, restore)
    assert not result["valid"]
    assert not result["gates"]["cross_object_ceiling"]


def test_snapshot_parity_requires_canonical_score_producer():
    restore = _record(); restore["score_producer"] = "offline_recompute"
    with pytest.raises(ValueError, match="producer"):
        evaluate_state(_record(), restore)
