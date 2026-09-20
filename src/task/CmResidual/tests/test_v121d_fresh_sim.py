"""Contract tests for the V1.21d native-init fresh-simulator backend."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.task.CmResidual.v121c_prefix_replay import BRANCH_COUNT, PublicBranchState
from src.task.CmResidual.v121c_ranking import (
    PHASE_CONTACT,
    PHASE_MOVING,
    PHASE_PRECONTACT,
)
from src.task.CmResidual.v121d_artifacts import validate_fresh_sim_physics_records
from src.task.CmResidual.v121d_fresh_sim import (
    ALL_DUPLICATE_PAIR_COUNT,
    BACKEND_VERSION,
    BRANCH_BACKEND,
    all_duplicate_pairwise_divergence,
    calibrate_all_duplicate,
    replay_native_init_fresh_sim,
    select_gate_a_indices,
    select_gate_b_indices,
    validate_native_init_episode,
)


def _episode() -> dict[str, np.ndarray]:
    return {
        "episode_id": np.asarray(0, dtype=np.int64),
        "seed": np.asarray(5909, dtype=np.int64),
        "initial_dof_state": np.zeros((18, 2), dtype=np.float32),
        "initial_actor_root_state": np.zeros((2, 13), dtype=np.float32),
        "initial_task_indices": np.asarray([7, 11, 3, 0], dtype=np.int64),
        "executed_action_history": np.asarray(
            [[0.1] + [0.0] * 17, [0.2] + [0.0] * 17], dtype=np.float32
        ),
        "done_history": np.asarray([False, False]),
        "reference_index": np.asarray([11, 12], dtype=np.int64),
        "data_id": np.asarray(7, dtype=np.int64),
        "start_time": np.asarray(3, dtype=np.int64),
        "progress_history": np.asarray([0, 1], dtype=np.int64),
        "resolved_sim_config_sha256": np.asarray("d" * 64),
        "state_init_mode": np.asarray("Start"),
        "hybrid_init_prob": np.asarray(1.0),
        "motion_source_sha256": np.asarray("e" * 64),
    }


class _FakeNativeRuntime:
    def __init__(self, episode, *, initial_drift=0.0, prefix_drift=0.0, done=False):
        self.dof = np.repeat(episode["initial_dof_state"][None], BRANCH_COUNT, axis=0)
        self.roots = np.repeat(episode["initial_actor_root_state"][None], BRANCH_COUNT, axis=0)
        self.indices = np.repeat(episode["initial_task_indices"][None], BRANCH_COUNT, axis=0)
        self.dof[1, 0, 0] += initial_drift
        self.prefix_drift = prefix_drift
        self.done = done
        self.actions = []
        self.closed = False

    def public_state(self):
        return PublicBranchState(self.dof.copy(), self.roots.copy(), self.indices.copy())

    def step(self, actions):
        self.actions.append(np.asarray(actions).copy())
        if len(self.actions) == 1:
            self.dof[:, 0, 0] += actions[:, 0]
            self.dof[1, 1, 0] += self.prefix_drift
            self.indices[:, 1] = 12
            self.indices[:, 3] = 1
        return np.full(BRANCH_COUNT, self.done, dtype=np.bool_)

    def close(self):
        assert not self.closed
        self.closed = True


class _FakeFactory:
    def __init__(self, **runtime_kwargs):
        self.runtime_kwargs = runtime_kwargs
        self.created = []

    def create(self, episode, branch_count, sim_instance_id):
        assert branch_count == BRANCH_COUNT
        runtime = _FakeNativeRuntime(episode, **self.runtime_kwargs)
        self.created.append((sim_instance_id, runtime))
        return runtime


def test_native_init_replay_uses_fresh_lifecycle_prefix_and_8_plus_1_mapping():
    episode = _episode()
    candidates = np.zeros((8, 18), dtype=np.float32)
    candidates[:, 0] = np.arange(8, dtype=np.float32) / 8
    factory = _FakeFactory()
    result = replay_native_init_fresh_sim(
        factory, episode, 1, candidates, sim_instance_id=4
    )
    assert result.branch_backend == BRANCH_BACKEND
    assert result.native_initial_parity and result.pre_candidate_parity
    assert result.sim_create_ok and result.sim_destroy_ok
    assert result.duplicate_replica_count == 2
    assert factory.created[0][0] == 4
    runtime = factory.created[0][1]
    assert runtime.closed and len(runtime.actions) == 2
    np.testing.assert_array_equal(
        runtime.actions[0], np.repeat(episode["executed_action_history"][:1], 9, axis=0)
    )
    np.testing.assert_array_equal(runtime.actions[1][:8], candidates)
    np.testing.assert_array_equal(runtime.actions[1][8], candidates[0])


def test_initial_or_prefix_parity_failure_never_executes_candidate_and_still_destroys():
    candidates = np.zeros((8, 18), dtype=np.float32)
    initial_factory = _FakeFactory(initial_drift=2e-5)
    initial = replay_native_init_fresh_sim(
        initial_factory, _episode(), 1, candidates, sim_instance_id=0
    )
    assert not initial.native_initial_parity
    assert initial.duplicate_replica_count == 0
    assert initial.pre_candidate_state is None and initial.post_candidate_state is None
    assert initial_factory.created[0][1].actions == []
    assert initial_factory.created[0][1].closed

    prefix_factory = _FakeFactory(prefix_drift=2e-5)
    prefix = replay_native_init_fresh_sim(
        prefix_factory, _episode(), 1, candidates, sim_instance_id=1
    )
    assert prefix.native_initial_parity and not prefix.pre_candidate_parity
    assert prefix.duplicate_replica_count == 0
    assert prefix.post_candidate_state is None
    assert len(prefix_factory.created[0][1].actions) == 1
    assert prefix_factory.created[0][1].closed


def test_done_during_prefix_is_hard_failure_but_sim_is_destroyed():
    factory = _FakeFactory(done=True)
    with pytest.raises(RuntimeError, match="prefix triggered done/reset"):
        replay_native_init_fresh_sim(
            factory, _episode(), 1, np.zeros((8, 18), dtype=np.float32), sim_instance_id=0
        )
    assert factory.created[0][1].closed


def test_all_duplicate_mode_repeats_candidate_zero_nine_times():
    candidates = np.zeros((8, 18), dtype=np.float32)
    candidates[0] = 0.25
    candidates[1:] = -0.25
    factory = _FakeFactory()
    result = replay_native_init_fresh_sim(
        factory, _episode(), 1, candidates, sim_instance_id=0, all_duplicate=True
    )
    assert result.duplicate_replica_count == 9
    np.testing.assert_array_equal(
        factory.created[0][1].actions[-1], np.repeat(candidates[0:1], 9, axis=0)
    )


def test_gate_a_selection_is_phase_stratified_and_spans_prefix_quantiles():
    frames = np.tile(np.arange(6), 3)
    states = np.asarray([f"state-{index:02d}" for index in range(18)])
    phases = np.repeat([PHASE_MOVING, PHASE_CONTACT, PHASE_PRECONTACT], 6)
    selected = select_gate_a_indices(frames, states, phases)
    assert selected.tolist() == [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17]
    with pytest.raises(RuntimeError, match="INCONCLUSIVE"):
        select_gate_a_indices(frames[:11], states[:11], phases[:11])


def test_gate_b_uses_all_36_pairs_state_maxima_and_fixed_phase_quota():
    state_ids = np.asarray([f"state-{index:03d}" for index in range(80)][::-1])
    available_phases = np.asarray(
        [PHASE_MOVING] * 30 + [PHASE_CONTACT] * 30 + [PHASE_PRECONTACT] * 20
    )
    selected = select_gate_b_indices(state_ids, available_phases)
    assert selected.shape == (64,)
    assert [int(np.sum(available_phases[selected] == phase)) for phase in range(3)] == [24, 24, 16]
    pose = np.zeros((9, 7), dtype=np.float64)
    pose[:, 6] = 1.0
    pose[:, 0] = np.arange(9) * 1e-6
    ig = np.zeros((9, 18), dtype=np.float64)
    score = np.arange(9, dtype=np.float64) * 1e-7
    pairwise = all_duplicate_pairwise_divergence(pose, ig, score)
    assert pairwise.shape == (ALL_DUPLICATE_PAIR_COUNT, 4)
    divergence = np.repeat(pairwise[None], 64, axis=0)
    phases = np.asarray(
        [PHASE_MOVING] * 24 + [PHASE_CONTACT] * 24 + [PHASE_PRECONTACT] * 16
    )
    calibration = calibrate_all_duplicate(divergence, phases)
    assert calibration["pair_count"] == 64 * 36
    assert calibration["replica_outcome_count"] == 64 * 9
    assert calibration["epsilon_physx"] >= 1e-6
    divergence[0, 0, 0] = 6e-4
    with pytest.raises(RuntimeError, match="INVALID_IMPLEMENTATION"):
        calibrate_all_duplicate(divergence, phases)


def _legacy_physics(count=1):
    return {
        "physics_clone_valid": np.ones(count, dtype=bool),
        "physics_candidate_valid": np.ones((count, 8), dtype=bool),
        "physics_score": np.zeros((count, 8)),
        "physics_next_object_pose": np.zeros((count, 8, 7)),
        "physics_next_IG": np.zeros((count, 8, 18)),
        "duplicate_delta_object_pose": np.zeros(count),
        "duplicate_delta_IG": np.zeros(count),
        "duplicate_delta_score": np.zeros(count),
    }


def test_v121d_schema_rejects_legacy_and_accepts_complete_fresh_sim_provenance():
    records = _legacy_physics()
    with pytest.raises(ValueError, match="V1.21d physics records missing"):
        validate_fresh_sim_physics_records(records)
    divergence = np.full((1, ALL_DUPLICATE_PAIR_COUNT, 4), np.nan)
    divergence[0, 0] = 0.0
    records.update(
        {
            "branch_backend": np.asarray([BRANCH_BACKEND]),
            "backend_version": np.asarray([BACKEND_VERSION]),
            "sim_instance_id": np.asarray([3], dtype=np.int64),
            "native_initial_parity": np.asarray([True]),
            "native_initial_max_abs_error_by_field": np.zeros((1, 5)),
            "pre_candidate_parity": np.asarray([True]),
            "pre_candidate_max_abs_error_by_field": np.zeros((1, 5)),
            "prefix_length": np.asarray([1], dtype=np.int64),
            "task_indices_exact": np.asarray([True]),
            "sim_create_ok": np.asarray([True]),
            "sim_destroy_ok": np.asarray([True]),
            "duplicate_replica_count": np.asarray([2], dtype=np.int64),
            "duplicate_pair_count": np.asarray([1], dtype=np.int64),
            "duplicate_pairwise_divergence": divergence,
            "state_max_abs_pairwise_score_divergence": np.asarray([0.0]),
        }
    )
    assert validate_fresh_sim_physics_records(records) == 1
    records["physics_clone_valid"][0] = False
    records["physics_candidate_valid"][0] = False
    records["native_initial_parity"][0] = False
    records["pre_candidate_parity"][0] = False
    records["pre_candidate_max_abs_error_by_field"][0] = np.nan
    records["duplicate_replica_count"][0] = 0
    records["duplicate_pair_count"][0] = 0
    records["duplicate_pairwise_divergence"][0] = np.nan
    records["state_max_abs_pairwise_score_divergence"][0] = np.nan
    assert validate_fresh_sim_physics_records(records) == 1
    records["branch_backend"][0] = "tensor_restore"
    with pytest.raises(ValueError, match="legacy or unknown"):
        validate_fresh_sim_physics_records(records)


def test_episode_contract_and_runner_require_native_start_and_explicit_gpu():
    assert validate_native_init_episode(_episode()) == 2
    invalid = _episode()
    invalid["state_init_mode"] = np.asarray("Random")
    with pytest.raises(ValueError, match="must be Start"):
        validate_native_init_episode(invalid)
    entrypoint = Path("src/task/CmResidual/tools/run_v121d_native_init_smoke.py")
    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--physical-gpu" in completed.stdout
    assert "--execute" in completed.stdout
