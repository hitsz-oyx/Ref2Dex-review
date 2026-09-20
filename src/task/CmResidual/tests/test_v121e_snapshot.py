import copy
import json
from pathlib import Path
import numpy as np
import pytest

from src.task.CmResidual.tools.run_v121e_snapshot_parity import _arm_specs, _parity_command, _source_rows
from src.task.CmResidual.v121e_snapshot import PHASE_QUOTAS, PRODUCER, evaluate_state


def _record():
    roots = np.zeros((3, 13), dtype=np.float64)
    roots[..., 6] = 1.0
    return {
        "state_id": "state", "candidate_action_sha256": "a" * 64,
        "task_indices_before": [0, 0, 0, 10], "object_actor_index": 1,
        "canonical_snapshot_sha256": "b" * 64,
        "setter_input_dof_sha256": "c" * 64,
        "setter_input_root_sha256": "d" * 64,
        "setter_input_max_abs_error": {"dof": 0.0, "root": 0.0},
        "setter_return_status": {"dof": True, "root": True},
        "num_envs": 1,
        "post_actor_root_state": roots.tolist(),
        "post_dof_state": np.zeros((18, 2)).tolist(),
        "post_ig": np.zeros(18).tolist(), "physics_score": 1.0,
        "score_producer": PRODUCER,
        "score_baseline": "shared_canonical_collected_pre_action",
    }


def test_snapshot_parity_accepts_identical_duplicate_arms():
    result = evaluate_state([_record(), _record()], [_record(), _record()])
    assert result["valid"]
    assert all(result["gates"].values())


def test_snapshot_parity_rejects_cross_method_object_drift():
    prefix = [_record(), _record()]
    restore = [_record(), _record()]
    restore = copy.deepcopy(restore)
    for record in restore:
        roots = np.asarray(record["post_actor_root_state"])
        roots[1, 0] = 1e-3
        record["post_actor_root_state"] = roots.tolist()
    result = evaluate_state(prefix, restore)
    assert not result["valid"]
    assert not result["gates"]["cross_object_ceiling"]


def test_snapshot_parity_requires_canonical_score_producer():
    restore = _record(); restore["score_producer"] = "offline_recompute"
    with pytest.raises(ValueError, match="producer"):
        evaluate_state([_record(), _record()], [restore, _record()])


def test_snapshot_parity_uses_four_independent_single_env_processes():
    assert PHASE_QUOTAS == {0: 2, 1: 2, 2: 2}
    assert _arm_specs() == (("prefix", 1), ("prefix", 2), ("restore", 1), ("restore", 2))
    command = _parity_command(5909)
    assert command[command.index("--num_envs") + 1] == "1"


def test_snapshot_runner_validates_and_reuses_frozen_v121e_artifacts(tmp_path):
    source = tmp_path / "frozen-v121e"
    (source / "episodes/0000").mkdir(parents=True)
    (source / "run_manifest.json").write_text(json.dumps({
        "run_id": source.name, "run_status": "COMPLETED",
    }))
    count = 16
    payload = {
        "state_id": np.asarray([f"state-{index:02d}" for index in range(count)]),
        "phase_id": np.asarray([0] * 6 + [1] * 6 + [2] * 4),
        "episode_id": np.zeros(count, dtype=np.int64),
        "seed": np.full(count, 5909, dtype=np.int64),
        "collection_batch_id": np.asarray(["frozen"] * count),
        "progress": np.arange(count, dtype=np.int64),
        "raw_obs": np.zeros((count, 1442), dtype=np.float32),
        "policy_mu": np.zeros((count, 18), dtype=np.float32),
        "snapshot_dof_state": np.zeros((count, 18, 2), dtype=np.float32),
        "snapshot_actor_root_state": np.zeros((count, 3, 13), dtype=np.float32),
        "snapshot_task_indices": np.zeros((count, 4), dtype=np.int64),
        "snapshot_reset_buf": np.zeros(count, dtype=np.int64),
        "snapshot_terminate_buf": np.zeros(count, dtype=np.int64),
        "snapshot_contact_reset": np.zeros((count, 1), dtype=np.float32),
        "canonical_current_ig": np.zeros((count, 18), dtype=np.float32),
    }
    np.savez_compressed(source / "selected_states.npz", **payload)
    np.savez_compressed(source / "episodes/0000/episode.npz",
                        initial_dof_state=np.zeros((18, 2)), initial_actor_root_state=np.zeros((3, 13)),
                        initial_task_indices=np.zeros(4, dtype=np.int64),
                        executed_action_history=np.zeros((16, 18)), done_history=np.zeros(16, dtype=bool))
    rows, identity = _source_rows(source)
    assert len(rows) == 16 and identity["run_id"] == source.name
    payload["phase_id"] = np.zeros(count, dtype=np.int64)
    np.savez_compressed(source / "selected_states.npz", **payload)
    with pytest.raises(ValueError, match="phase composition"):
        _source_rows(source)


def test_snapshot_setter_has_no_pre_simulate_refresh_or_observation_recompute():
    source = Path("src/task/CmResidual/tools/v121e_snapshot_parity_bootstrap.py").read_text()
    body = source.split("def _set_public", 1)[1].split("\ndef main", 1)[0]
    assert "_refresh_sim_tensors" not in body
    assert "_compute_observations" not in body
    assert "setter_input_dof_sha256" in body
    assert "setter_input_max_abs_error" in body


def test_snapshot_parity_rejects_nonidentical_setter_input():
    bad = _record()
    bad["setter_input_max_abs_error"]["dof"] = 1e-7
    with pytest.raises(ValueError, match="setter input"):
        evaluate_state([_record(), _record()], [bad, _record()])
