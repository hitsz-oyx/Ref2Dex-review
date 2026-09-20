import copy
from pathlib import Path

import numpy as np
import pytest

from src.task.CmResidual.tools.run_v121e2_warmup_sweep import _arm_specs, _merge_snapshots
from src.task.CmResidual.v121e2_warmup import (
    METHODS, PRODUCER, WINDOWS, _centroid, canonical_snapshot_sha256,
    classify_sweep, evaluate_state, warmup_action_slice,
)


def _record(method="full"):
    roots = np.zeros((3, 13), dtype=np.float64)
    roots[:, 6] = 1.0
    return {
        "state_id": "state", "method": method, "repeat": 1, "num_envs": 1,
        "source_snapshot_sha256": "initial" if method == "full" else method * 64,
        "source_episode_sha256": "e" * 64, "candidate_action_sha256": "a" * 64,
        "task_indices_target": [0, 0, 0, 10], "object_actor_index": 1,
        "setter_input_max_abs_error": {"dof": 0.0, "root": 0.0},
        "setter_return_status": {"dof": True, "root": True},
        "pre_actor_root_state": roots.tolist(), "pre_dof_state": np.zeros((18, 2)).tolist(),
        "pre_ig": np.zeros(18).tolist(),
        "post_actor_root_state": roots.tolist(), "post_dof_state": np.zeros((18, 2)).tolist(),
        "post_ig": np.zeros(18).tolist(), "physics_score": 1.0,
        "score_producer": PRODUCER,
    }


def _method_records():
    result = {}
    for method in METHODS:
        first, second = _record(method), _record(method)
        second["repeat"] = 2
        result[method] = [first, second]
    return result


def _classification_result(flags, valid=True):
    return {"implementation_valid": valid,
            "methods": {str(window): {"pass": bool(flag)} for window, flag in zip(WINDOWS, flags)}}


def test_warmup_action_slice_is_exact_and_bounded():
    actions = np.arange(20).reshape(10, 2)
    assert np.array_equal(warmup_action_slice(actions, 7, 4), actions[3:7])
    assert warmup_action_slice(actions, 7, 0).shape == (0, 2)
    with pytest.raises(ValueError, match="slice"):
        warmup_action_slice(actions, 2, 4)


def test_snapshot_hash_is_content_and_identity_stable():
    values = (np.zeros((18, 2)), np.zeros((3, 13)), np.zeros(4), 0, 0,
              np.zeros(1), np.zeros(18))
    first = canonical_snapshot_sha256("state", 4, *values)
    assert first == canonical_snapshot_sha256("state", 4, *values)
    assert first != canonical_snapshot_sha256("state", 2, *values)


def test_snapshot_merge_requires_all_30_unique_keys_and_valid_hashes(tmp_path):
    selected = [{"state_id": f"state-{index}", "frame_id": 20} for index in range(6)]
    rows = []
    for state in selected:
        for window in WINDOWS:
            dof, roots = np.zeros((18, 2)), np.zeros((3, 13))
            indices, contact, ig = np.zeros(4), np.zeros(1), np.zeros(18)
            sha = canonical_snapshot_sha256(state["state_id"], window, dof, roots, indices,
                                            0, 0, contact, ig)
            rows.append({
                "state_id": state["state_id"], "window": window,
                "frame_id": state["frame_id"] - window,
                "snapshot_dof_state": dof, "snapshot_actor_root_state": roots,
                "snapshot_task_indices": indices, "snapshot_reset_buf": 0,
                "snapshot_terminate_buf": 0, "snapshot_contact_reset": contact,
                "snapshot_ig": ig, "snapshot_sha256": sha,
            })
    source = tmp_path / "source.npz"
    np.savez_compressed(source, **{name: np.asarray([row[name] for row in rows]) for name in rows[0]})
    merged = _merge_snapshots([source], selected, tmp_path / "merged.npz")
    assert len(merged) == 30
    rows[-1]["snapshot_sha256"] = "bad"
    np.savez_compressed(source, **{name: np.asarray([row[name] for row in rows]) for name in rows[0]})
    with pytest.raises(ValueError, match="content hash"):
        _merge_snapshots([source], selected, tmp_path / "invalid.npz")


def test_full_centroid_sign_aligns_quaternion():
    left = (np.zeros((1, 13)), np.zeros((1, 2)), np.zeros(18), None)
    left[0][0, 6] = 1.0
    right = tuple(value.copy() if hasattr(value, "copy") else value for value in left)
    right[0][0, 3:7] *= -1
    center = _centroid(left, right)
    assert np.allclose(center[0][0, 3:7], left[0][0, 3:7])


def test_evaluator_requires_pre_and_post_parity():
    records = _method_records()
    assert evaluate_state(records)["methods"]["2"]["pass"]
    drifted = copy.deepcopy(records)
    for arm in drifted["2"]:
        arm["post_actor_root_state"][1][0] = 1e-3
    result = evaluate_state(drifted)
    assert result["methods"]["2"]["pre"]["gates"]["object_ceiling"]
    assert not result["methods"]["2"]["post"]["gates"]["object_ceiling"]
    assert not result["methods"]["2"]["pass"]


def test_sweep_classification_requires_a_stable_passing_suffix():
    supported = classify_sweep([_classification_result([False, False, True, True, True]) for _ in range(6)])
    assert supported["conclusion"] == "SUPPORTED" and supported["minimal_supported_L"] == 2
    nonmonotonic = classify_sweep([_classification_result([False, False, True, False, True]) for _ in range(6)])
    assert nonmonotonic["conclusion"] == "INCONCLUSIVE"
    refuted = classify_sweep([_classification_result([False] * 5) for _ in range(6)])
    assert refuted == {"conclusion": "REFUTED", "minimal_supported_L": None,
                       "method_pass": {window: False for window in WINDOWS},
                       "conclusion_scope": "warmup<=8 insufficient"}
    invalid = classify_sweep([_classification_result([True] * 5, valid=False)] +
                             [_classification_result([True] * 5) for _ in range(5)])
    assert invalid["conclusion"] == "INVALID_IMPLEMENTATION"


def test_runner_has_72_independent_single_env_arms():
    assert len(_arm_specs()) == 12
    assert len(_arm_specs()) * 6 == 72
    assert _arm_specs()[:2] == (("full", 1), ("full", 2))


def test_generation_pass_is_separate_and_does_not_shortcut_episode_replay():
    source = Path("src/task/CmResidual/tools/v121e2_snapshot_generation_bootstrap.py").read_text()
    assert "for index, action in enumerate(episode[\"executed_action_history\"])" in source
    loop = source.split("for index, action in enumerate", 1)[1].split("expected =", 1)[0]
    assert "break" not in loop
    runner = Path("src/task/CmResidual/tools/run_v121e2_warmup_sweep.py").read_text()
    assert '"generation_passes_count_as_arms": False' in runner
    assert "snapshots_path.chmod(0o444)" in runner
    assert "canonical snapshot file changed after generation" in runner
