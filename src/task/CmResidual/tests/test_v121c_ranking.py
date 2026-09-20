"""Pure-contract tests for the V1.21c native-action ranking gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.task.CmResidual.v121c_artifacts import (
    PHYSICS_SCORE_PRODUCER,
    validate_episode_replay,
    validate_physics_records,
    validate_state_records,
)
from src.task.CmResidual.v121c_collection import CollectionAccumulator, CollectionConfig
from src.task.CmResidual.v121c_prefix_replay import (
    BRANCH_COUNT,
    DExploreTaskPrefixRuntime,
    PHYSICS_SUBSTEPS,
    PublicBranchState,
    candidate_env_assignment,
    replay_prefix_branches,
)
from src.task.CmResidual.v121c_ranking import (
    CANDIDATE_COUNT,
    PHASE_CONTACT,
    PHASE_MOVING,
    PHASE_PRECONTACT,
    active_phase,
    branch_state_parity,
    calibration_reference_targets,
    calibrate_duplicate_anchor,
    candidate_seed,
    canonical_ig,
    canonical_ig_from_body_positions,
    canonical_state_id,
    cm_scores,
    cm_valid_mask,
    collect_active_phase_quota,
    compose_current_local_delta,
    episode_block_bootstrap,
    freeze_physx_epsilon,
    generate_candidate_actions,
    pairwise_metrics,
    physics_scores,
    pose_xyzw_to_matrix,
    next_state_cost,
    ranking_eligible,
    select_phase_quota,
    top1_metrics,
    CollectionInsufficient,
)
from src.task.CmResidual.dexplore_cm_geometry import dexplore_action_to_native_targets
from src.task.CmResidual.tools.run_v121c_prefix_smoke import smoke_command, smoke_gate_passed
from src.task.CmResidual.tools.run_v121c_duplicate_calibration import (
    DEXPLORE_PYTHON as DUPLICATE_DEXPLORE_PYTHON,
    _command as duplicate_calibration_command,
    _document_identity as duplicate_document_identity,
    _validate_collection as validate_calibration_collection,
)
from src.task.CmResidual.tools.run_v121c_calibration_collection import (
    DEXPLORE_PYTHON,
    _command as calibration_collection_command,
    _gpu_used_mib,
    _load_rows as load_calibration_rows,
    _select as select_calibration,
)


def test_candidate_generation_is_bitwise_reproducible_and_clipped():
    mu = torch.tensor([[1.4, -1.2] + [0.0] * 16, [0.1] * 18], dtype=torch.float32)
    sigma = torch.full_like(mu, 0.25)
    first, first_seeds = generate_candidate_actions(mu, sigma, "batch-1", ["state-a", "state-b"])
    second, second_seeds = generate_candidate_actions(mu, sigma, "batch-1", ["state-a", "state-b"])
    assert torch.equal(first, second)
    assert torch.equal(first_seeds, second_seeds)
    assert first.shape == (2, CANDIDATE_COUNT, 18)
    assert torch.equal(first[0, 0], mu[0].clamp(-1, 1))
    assert torch.all(first.abs() <= 1)
    assert first_seeds[0].item() != first_seeds[1].item()


def test_prefix_smoke_command_and_duplicate_candidates_are_frozen():
    command = smoke_command()
    assert "--num_envs" in command and command[command.index("--num_envs") + 1] == "9"
    assert "--seed" in command and command[command.index("--seed") + 1] == "5909"
    assert "--sim_device" in command and command[command.index("--sim_device") + 1] == "cuda:0"
    entrypoint = Path("src/task/CmResidual/tools/run_v121c_prefix_smoke.py")
    help_result = subprocess.run(
        [sys.executable, str(entrypoint), "--help"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert "--run-id" in help_result.stdout
    assert smoke_gate_passed(
        {"parity_valid": False, "task_indices_equal": True, "duplicate_valid": True}
    )
    assert not smoke_gate_passed(
        {"parity_valid": True, "task_indices_equal": False, "duplicate_valid": True}
    )


class _FakePhysicsGym:
    def __init__(self):
        self.simulate_count = 0
        self.fetch_count = 0

    def simulate(self, sim):
        assert sim == "sim"
        self.simulate_count += 1

    def fetch_results(self, sim, wait):
        assert sim == "sim" and wait is True
        self.fetch_count += 1


class _FakeDExploreStepTask:
    def __init__(self, control_freq_inv=PHYSICS_SUBSTEPS):
        self.num_envs = BRANCH_COUNT
        self.control_freq_inv = control_freq_inv
        self.device = "cpu"
        self.sim = "sim"
        self.gym = _FakePhysicsGym()
        self.pre_count = 0
        self.physics_count = 0
        self.post_count = 0

    def pre_physics_step(self, actions):
        assert tuple(actions.shape) == (BRANCH_COUNT, 18)
        self.pre_count += 1

    def _physics_step(self):
        self.physics_count += 1
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)

    def post_physics_step(self):
        self.post_count += 1


def test_dexplore_runtime_matches_one_30hz_control_step():
    task = _FakeDExploreStepTask()
    runtime = DExploreTaskPrefixRuntime(task, gymtorch_module=None)
    runtime.step(task, np.zeros((BRANCH_COUNT, 18), dtype=np.float32))
    assert task.pre_count == 1
    assert task.physics_count == 1
    assert task.gym.simulate_count == 2
    assert task.gym.fetch_count == 1
    assert task.post_count == 1

    invalid = _FakeDExploreStepTask(control_freq_inv=1)
    with pytest.raises(ValueError, match="control_freq_inv=2"):
        DExploreTaskPrefixRuntime(invalid, gymtorch_module=None).step(
            invalid, np.zeros((BRANCH_COUNT, 18), dtype=np.float32)
        )
    assert invalid.pre_count == 0 and invalid.gym.simulate_count == 0


def test_collection_accumulator_enforces_seed_horizon_uniqueness_and_quota():
    config = CollectionConfig("batch", target_unique=3, max_collection_episodes=2,
                              phase_quotas={PHASE_MOVING: 1, PHASE_CONTACT: 1, PHASE_PRECONTACT: 1})
    accumulator = CollectionAccumulator(config)
    base = {"state_id": "a"}
    assert not accumulator.add(state_id="too-late", episode_id=0, seed=5909, frame_id=4,
                               sequence_length=10, active=True, phase_id=PHASE_MOVING, record=base)
    assert accumulator.add(state_id="c", episode_id=0, seed=5909, frame_id=0,
                           sequence_length=10, active=True, phase_id=PHASE_PRECONTACT,
                           record={"state_id": "c"})
    assert accumulator.add(state_id="a", episode_id=0, seed=5909, frame_id=1,
                           sequence_length=10, active=True, phase_id=PHASE_MOVING,
                           record={"state_id": "a"})
    assert accumulator.add(state_id="b", episode_id=1, seed=5910, frame_id=0,
                           sequence_length=10, active=True, phase_id=PHASE_CONTACT,
                           record={"state_id": "b"})
    assert accumulator.complete
    assert [row["state_id"] for row in accumulator.finalize()] == ["a", "b", "c"]
    assert not accumulator.add(state_id="extra", episode_id=1, seed=5910, frame_id=1,
                               sequence_length=10, active=True, phase_id=PHASE_MOVING,
                               record={"state_id": "extra"})


def test_state_id_uses_canonical_float32_bytes():
    obs = np.linspace(-1, 1, 1442, dtype=np.float64)
    first = canonical_state_id("a" * 64, 5909, 0, 3, obs, "b" * 64)
    second = canonical_state_id("a" * 64, 5909, 0, 3, obs.astype(np.float32), "b" * 64)
    assert first == second
    assert first != canonical_state_id("a" * 64, 5909, 0, 4, obs, "b" * 64)


def test_active_formula_and_phase_priority():
    forces = torch.zeros(5, 5, 3)
    forces[0, 0, 0] = 0.11       # moving takes priority over contact
    forces[1, 0, 0] = 0.11       # contact phase
    tip_distance = torch.full((5, 5), 0.1)
    tip_distance[2, 0] = 0.03    # precontact
    tip_distance[3, 0] = 0.03    # inactive? it is active, intentionally
    ref_contact = torch.zeros(5, 5)
    ref_contact[1, 0] = 1.0
    ref_translation = torch.tensor([0.003, 0.0, 0.0, 0.0, 0.0])
    ref_rotation = torch.zeros(5)
    ref_near = torch.full((5, 5), 0.1)
    ref_near[0, 0] = 0.03
    ref_near[3, 0] = 0.03
    active, reason, phase = active_phase(
        forces, tip_distance, ref_contact, ref_translation, ref_rotation, ref_near)
    assert active.tolist() == [True, True, True, True, False]
    assert phase.tolist() == [PHASE_MOVING, PHASE_CONTACT, PHASE_PRECONTACT, PHASE_PRECONTACT, -1]
    assert int(reason[0]) & 1
    assert int(reason[1]) & 4
    assert int(reason[2]) & 2


def test_phase_quota_is_deterministic_and_never_backfills():
    ids = ["z", "a", "m", "b", "c"]
    phases = torch.tensor([PHASE_MOVING, PHASE_MOVING, PHASE_CONTACT, PHASE_PRECONTACT, -1])
    selected = select_phase_quota(ids, phases, {PHASE_MOVING: 1, PHASE_CONTACT: 1, PHASE_PRECONTACT: 1})
    assert selected.tolist() == [1, 2, 3]
    assert ranking_eligible(torch.tensor([[True, True, True, True, False, False, False, False]]))[0]
    with pytest.raises(CollectionInsufficient):
        collect_active_phase_quota(ids, torch.ones(5, dtype=torch.bool), phases,
                                   {PHASE_MOVING: 2, PHASE_CONTACT: 2, PHASE_PRECONTACT: 2})


def test_canonical_ig_and_local_right_composition():
    key = torch.zeros(1, 6, 3)
    key[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    surface = torch.zeros(1, 256, 3)
    heading_inverse = torch.eye(3).unsqueeze(0)
    ig = canonical_ig(key, surface, heading_inverse)
    assert ig.shape == (1, 18)
    torch.testing.assert_close(ig[0, :3], torch.tensor([1.0, 0.0, 0.0]))

    current = torch.eye(4).unsqueeze(0)
    angle = torch.tensor(torch.pi / 2)
    current[0, :3, :3] = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                        [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1]])
    current[0, 0, 3] = 2.0
    delta = torch.zeros(1, 1, 6)
    delta[0, 0, 0] = 1.0
    result = compose_current_local_delta(current, delta)
    torch.testing.assert_close(result[0, 0, :3, 3], torch.tensor([2.0, 1.0, 0.0]), atol=1e-5, rtol=0)
    full_body = torch.zeros(1, 18, 3)
    full_body[:, 0] = key[:, 0]
    full_body[:, 3] = key[:, 1]
    full_body[:, 6] = key[:, 2]
    full_body[:, 9] = key[:, 3]
    full_body[:, 12] = key[:, 4]
    full_body[:, 15] = key[:, 5]
    torch.testing.assert_close(canonical_ig_from_body_positions(full_body, surface, heading_inverse), ig)


def test_native_action_mapping_keeps_clamp_and_coupling_contract():
    action = torch.zeros(1, 18)
    action[0, 0] = 2.0
    action[0, 6] = -2.0
    action[0, 15] = 0.5
    current = torch.zeros(1, 18)
    lower = torch.full((18,), -2.0)
    upper = torch.full((18,), 2.0)
    targets = dexplore_action_to_native_targets(action, current, lower, upper)
    assert targets[0, 0].item() == pytest.approx(1.0)
    assert targets[0, 6].item() == pytest.approx(0.0)
    assert targets[0, 7].item() == pytest.approx(0.0)
    assert targets[0, 16].item() == pytest.approx(targets[0, 15].item() * 0.6)
    assert targets[0, 17].item() == pytest.approx(targets[0, 15].item() * 0.8)


def test_cm_score_uses_object_and_ig_progress():
    current = torch.eye(4).unsqueeze(0)
    goal = current.clone()
    goal[0, 0, 3] = 0.02
    delta = torch.zeros(1, 8, 6)
    delta[:, 0, 0] = 0.02
    current_ig = torch.zeros(1, 18)
    reference_ig = torch.zeros(1, 18)
    predicted_ig = torch.zeros(1, 8, 18)
    scores = cm_scores(current, goal, delta, current_ig, reference_ig, predicted_ig)
    assert scores["score"].shape == (1, 8)
    assert scores["score"][0, 0] > 0
    assert scores["score"][0, 1:].abs().max() < 1e-6


def test_duplicate_score_uses_t6_object_goal_and_t1_ig_reference():
    key_count = 16
    feature_count = 119 + key_count * 3 + 1 + 16 + key_count * 3
    hoi = torch.zeros(1, 10, feature_count)
    # t+1 deliberately disagrees with t+6 for the object target.
    hoi[0, 1, 106:113] = torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    hoi[0, 6, 106:113] = torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ig_start = 119 + key_count * 3 + 1 + 16
    reference_full = torch.arange(key_count * 3, dtype=torch.float32).reshape(key_count, 3)
    hoi[0, 1, ig_start:ig_start + key_count * 3] = reference_full.reshape(-1)
    goal, reference_ig = calibration_reference_targets(hoi, torch.tensor([0]), 0, key_count)
    assert goal[0, 0, 3].item() == pytest.approx(0.02)
    expected_ig = reference_full[[0, 3, 6, 9, 12, 15]].reshape(1, 18)
    torch.testing.assert_close(reference_ig, expected_ig)

    next_pose = goal[:, None].expand(1, 2, 4, 4).clone()
    next_ig = reference_ig[:, None].expand(1, 2, 18).clone()
    assert torch.equal(next_state_cost(goal, reference_ig, next_pose, next_ig), torch.zeros(1, 2))


def test_physics_score_uses_one_shared_canonical_collected_baseline():
    current_pose = torch.eye(4).unsqueeze(0)
    goal_pose = current_pose.clone()
    goal_pose[:, 0, 3] = 0.02
    current_ig = torch.zeros(1, 18)
    reference_ig = torch.zeros(1, 18)
    next_pose = current_pose[:, None].expand(1, 2, 4, 4).clone()
    next_pose[0, 0, 0, 3] = 0.01
    next_pose[0, 1, 0, 3] = 0.02
    next_ig = torch.zeros(1, 2, 18)
    result = physics_scores(
        current_pose, goal_pose, current_ig, reference_ig, next_pose, next_ig
    )
    assert result["score"].shape == (1, 2)
    assert result["score"][0, 1] > result["score"][0, 0]

    shifted_current = current_pose.clone()
    shifted_current[:, 1, 3] = 0.01
    shifted = physics_scores(
        shifted_current, goal_pose, current_ig, reference_ig, next_pose, next_ig
    )
    # Changing the one canonical baseline shifts every candidate equally.
    torch.testing.assert_close(
        shifted["score"] - result["score"],
        (shifted["baseline_cost"] - result["baseline_cost"])[:, None].expand(1, 2),
    )
    with pytest.raises(ValueError, match="shared"):
        physics_scores(
            current_pose[:, None].expand(1, 2, 4, 4), goal_pose,
            current_ig, reference_ig, next_pose, next_ig,
        )


def test_pose_xyzw_to_matrix_uses_xyzw_order():
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
    matrix = pose_xyzw_to_matrix(pose)
    torch.testing.assert_close(matrix[0, :3, :3], torch.eye(3))
    torch.testing.assert_close(matrix[0, :3, 3], pose[0, :3])


def test_cm_validity_requires_support_mass_and_finite_effect():
    candidate = torch.ones(1, 8, dtype=torch.bool)
    delta = torch.zeros(1, 8, 6)
    masks = torch.zeros(1, 8, 16, dtype=torch.bool)
    mass = torch.zeros(1, 8, 16)
    assert not cm_valid_mask(candidate, delta, masks, mass).any()
    masks[:, :4, 0] = True
    mass[:, :4, 0] = 1
    valid = cm_valid_mask(candidate, delta, masks, mass)
    assert valid[0, :4].all() and not valid[0, 4:].any()
    delta[0, 0, 0] = float("nan")
    assert not cm_valid_mask(candidate, delta, masks, mass)[0, 0]


def test_pairwise_state_weighting_ties_and_top1_crash_failure():
    cm = torch.zeros(2, 8)
    phys = torch.zeros(2, 8)
    cm[0, :3] = torch.tensor([3.0, 2.0, 1.0])
    phys[0, :3] = torch.tensor([3.0, 2.0, 1.0])
    cm[1, :2] = torch.tensor([1.0, 1.0])
    phys[1, :3] = torch.tensor([3.0, 2.0, 1.0])
    cm_valid = torch.zeros(2, 8, dtype=torch.bool)
    physics_valid = torch.zeros(2, 8, dtype=torch.bool)
    cm_valid[0, :3] = True
    cm_valid[1, :2] = True
    physics_valid[:, :3] = True
    pair = pairwise_metrics(cm, phys, cm_valid, physics_valid, 1e-6)
    torch.testing.assert_close(pair.state_accuracy, torch.tensor([1.0, 0.5]))
    assert pair.accuracy == pytest.approx(0.75)
    physics_valid[1, 0] = False
    top = top1_metrics(cm, phys, cm_valid, physics_valid, 1e-6)
    assert top.valid_state.tolist() == [True, False]
    assert top.selected_index.tolist() == [0, 0]


def test_episode_bootstrap_is_deterministic_and_epsilon_is_p99_based():
    values = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    episodes = np.array(["a", "a", "b", "b", "c", "c"])
    first = episode_block_bootstrap(values, episodes, draws=200, seed=9)
    second = episode_block_bootstrap(values, episodes, draws=200, seed=9)
    assert first == second
    assert first.episode_count == 3
    assert freeze_physx_epsilon([0.0] * 100) == 1e-6
    assert freeze_physx_epsilon(np.arange(100, dtype=np.float64)) > 400
    calibration = calibrate_duplicate_anchor(np.zeros(64), np.zeros(64), np.arange(64, dtype=np.float64))
    assert calibration["epsilon_physx"] > 0
    with pytest.raises(RuntimeError, match="INVALID_IMPLEMENTATION"):
        calibrate_duplicate_anchor([6e-4], [0.0], [0.0])


def test_branch_parity_checks_public_tensors_and_task_indices():
    reference = torch.zeros(4)
    branches = torch.zeros(2, 4)
    indices = torch.tensor([1, 2])
    branch_indices = torch.tensor([[1, 2], [1, 3]])
    branches[1, 0] = 2e-5
    assert branch_state_parity(reference, branches, indices, branch_indices).tolist() == [True, False]


class _FakePrefixRuntime:
    """Small deterministic stand-in proving the runtime-neutral replay call order."""

    def __init__(self, *, drift: float = 0.0, index_drift: bool = False):
        self.drift = drift
        self.index_drift = index_drift
        self.actions = []

    def make_envs(self, count):
        assert count == BRANCH_COUNT
        return {"dof": np.zeros((count, 18, 2), dtype=np.float32),
                "roots": np.zeros((count, 2, 13), dtype=np.float32),
                "indices": np.tile(np.array([7, 11]), (count, 1))}

    def restore_initial(self, envs, episode):
        envs["dof"][:] = episode["initial_dof_state"]
        envs["roots"][:] = episode["initial_actor_root_state"]
        envs["indices"][:] = episode["initial_task_indices"]

    def step(self, envs, actions):
        self.actions.append(actions.copy())
        envs["dof"][:, 0, 0] += actions[:, 0]

    def public_state(self, envs):
        dof = envs["dof"].copy()
        dof[1, 0, 0] += self.drift
        indices = envs["indices"].copy()
        if self.index_drift:
            indices[1, 0] += 1
        return PublicBranchState(dof, envs["roots"].copy(), indices)


def test_prefix_replay_uses_executed_history_then_candidates_and_duplicate_anchor():
    episode = {"episode_id": np.array([0]), "seed": np.array([5909]),
               "initial_dof_state": np.zeros((18, 2), dtype=np.float32),
               "initial_actor_root_state": np.zeros((2, 13), dtype=np.float32),
               "initial_task_indices": np.array([7, 11]),
               "executed_action_history": np.array([[0.1] + [0.0] * 17, [0.2] + [0.0] * 17], dtype=np.float32),
               "done_history": np.array([False, False]), "reference_index": np.array([0, 1]),
               "data_id": np.array(["s1_airplane_lift"]), "start_time": np.array([0.0]),
               "progress_history": np.array([0, 1]), "resolved_sim_config_sha256": np.array(["d" * 64])}
    candidates = np.zeros((8, 18), dtype=np.float32)
    candidates[:, 0] = np.arange(8, dtype=np.float32) / 8.0
    runtime = _FakePrefixRuntime()
    result = replay_prefix_branches(runtime, episode, 1, candidates, assignment_seed=5909)
    assert result.parity_valid and result.parity_max_abs_error == 0.0
    assert result.task_indices_equal
    assert sorted(result.env_candidate_ids.tolist()) == [0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert len(runtime.actions) == 2
    np.testing.assert_array_equal(runtime.actions[0], np.repeat(episode["executed_action_history"][:1], 9, axis=0))
    np.testing.assert_array_equal(runtime.actions[1], candidates[result.env_candidate_ids])
    assert result.post_candidate_state is not None


def test_candidate_environment_assignment_is_reproducible_and_balanced():
    first = candidate_env_assignment(5909)
    second = candidate_env_assignment(5909)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == [0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert not np.array_equal(first, candidate_env_assignment(5910))


def test_calibration_selection_is_phase_balanced_and_state_id_deterministic():
    rows = []
    for phase, count in ((0, 26), (1, 25), (2, 18)):
        for index in reversed(range(count)):
            rows.append({"phase_id": phase, "state_id": f"{phase}-{index:03d}"})
    selected = select_calibration(rows)
    assert selected is not None and len(selected) == 64
    assert [sum(row["phase_id"] == phase for row in selected) for phase in (0, 1, 2)] == [24, 24, 16]
    assert {row["state_id"] for row in selected if row["phase_id"] == 0} == {
        f"0-{index:03d}" for index in range(24)
    }
    assert select_calibration([row for row in rows if row["phase_id"] != 2]) is None


def test_calibration_gpu_lookup_rejects_unreported_device(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *args, **kwargs: "0, 12\n3, 7\n",
    )
    assert _gpu_used_mib(3) == 7
    with pytest.raises(RuntimeError, match="GPU9"):
        _gpu_used_mib(9)


def test_calibration_collection_uses_pinned_dexplore_runtime(tmp_path):
    command = calibration_collection_command(5909, tmp_path / "episode")
    assert command[0] == str(DEXPLORE_PYTHON)
    assert command[1].endswith("v121c_collect_episode_bootstrap.py")
    assert command[command.index("--num_envs") + 1] == "1"
    assert command[command.index("--seed") + 1] == "5909"
    entrypoint = Path("src/task/CmResidual/tools/run_v121c_calibration_collection.py").resolve()
    help_result = subprocess.run(
        [sys.executable, str(entrypoint), "--help"], cwd=tmp_path, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert "--gpu" in help_result.stdout


def test_calibration_collection_preserves_full_uint64_candidate_seed(tmp_path):
    batch_id = "batch-with-high-seed"
    state_id = "state-a"
    seed = candidate_seed(batch_id, state_id)
    path = tmp_path / "active.npz"
    np.savez_compressed(
        path,
        state_id=np.asarray([state_id]),
        frame_id=np.asarray([3]),
        reference_index=np.asarray([0]),
        progress=np.asarray([3]),
        phase_id=np.asarray([1]),
        active_reason_mask=np.asarray([2]),
        candidate_seed=np.asarray([seed], dtype=np.uint64),
        executed_action_prefix_sha256=np.asarray(["a" * 64]),
        raw_obs=np.zeros((1, 1442), dtype=np.float32),
        policy_mu=np.zeros((1, 18), dtype=np.float32),
        policy_sigma=np.ones((1, 18), dtype=np.float32),
    )
    rows = load_calibration_rows(path, episode_id=0, seed=5909, collection_batch_id=batch_id)
    assert rows[0]["candidate_seed"] == seed
    with np.load(path) as payload:
        broken = {name: payload[name] for name in payload.files}
    broken["candidate_seed"] = broken["candidate_seed"].astype(np.float64)
    np.savez_compressed(path, **broken)
    with pytest.raises(ValueError, match="must be uint64"):
        load_calibration_rows(path, episode_id=0, seed=5909, collection_batch_id=batch_id)


def _write_valid_frozen_calibration(tmp_path, phases=None):
    batch_id = "batch"
    state_ids = np.asarray([f"state-{index}" for index in range(64)])
    seeds = np.asarray([candidate_seed(batch_id, state) for state in state_ids], dtype=np.uint64)
    phases = np.asarray(
        [0] * 24 + [1] * 24 + [2] * 16 if phases is None else phases, dtype=np.uint8
    )
    actions = np.zeros((64, 18), dtype=np.float32)
    prefix_hashes = np.asarray([
        hashlib.sha256(np.asarray(actions[:frame], dtype="<f4").tobytes()).hexdigest()
        for frame in range(64)
    ])
    episodes = tmp_path / "episodes/0000"
    episodes.mkdir(parents=True)
    np.savez_compressed(
        episodes / "episode.npz",
        episode_id=np.asarray(0, dtype=np.int64),
        seed=np.asarray(5909, dtype=np.int64),
        initial_dof_state=np.zeros((18, 2), dtype=np.float32),
        initial_actor_root_state=np.zeros((2, 13), dtype=np.float32),
        initial_task_indices=np.asarray([0, 0, 0, 0], dtype=np.int64),
        executed_action_history=actions,
        done_history=np.zeros(64, dtype=np.bool_),
        reference_index=np.zeros(64, dtype=np.int64),
        data_id=np.asarray(0, dtype=np.int64),
        start_time=np.asarray(0, dtype=np.int64),
        progress_history=np.arange(64, dtype=np.int64),
        resolved_sim_config_sha256=np.asarray("d" * 64),
    )
    config = {"collection_batch_id": batch_id}
    (tmp_path / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    phase_counts = {str(phase): int(np.count_nonzero(phases == phase)) for phase in (0, 1, 2)}
    manifest = {
        "run_status": "COMPLETED", "selected_state_count": 64,
        "selected_phase_counts": phase_counts, "run_id": "collection", "git_commit": "abc",
        "config": str((tmp_path / "config.json").resolve()),
        "selected_states": str((tmp_path / "selected_states.npz").resolve()),
        "protocol": {"collection_batch_id": batch_id},
    }
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8",
    )
    np.savez_compressed(
        tmp_path / "selected_states.npz",
        state_id=state_ids,
        collection_batch_id=np.asarray([batch_id] * 64),
        episode_id=np.zeros(64, dtype=np.int64),
        seed=np.full(64, 5909, dtype=np.int64),
        frame_id=np.arange(64),
        reference_index=np.zeros(64, dtype=np.int64),
        progress=np.arange(64),
        phase_id=phases,
        candidate_seed=seeds,
        executed_action_prefix_sha256=prefix_hashes,
        raw_obs=np.zeros((64, 1442), dtype=np.float32),
        policy_mu=np.zeros((64, 18), dtype=np.float32),
        policy_sigma=np.ones((64, 18), dtype=np.float32),
    )
    return manifest


def test_duplicate_calibration_requires_exact_frozen_collection(tmp_path):
    _write_valid_frozen_calibration(tmp_path)
    manifest, episodes = validate_calibration_collection(tmp_path)
    assert manifest["run_id"] == "collection" and episodes.tolist() == [0]
    command = duplicate_calibration_command(5909)
    assert command[0] == str(DUPLICATE_DEXPLORE_PYTHON)
    assert command[command.index("--num_envs") + 1] == "9"
    documents = duplicate_document_identity()
    assert set(documents) == {
        "guidance_v121", "guidance_v121a", "guidance_v121c", "guidance_v121d",
        "plan_v121c", "plan_v121d",
    }
    assert all(len(identity["sha256"]) == 64 for identity in documents.values())
    broken = dict(np.load(tmp_path / "selected_states.npz"))
    broken["candidate_seed"] = broken["candidate_seed"].copy()
    broken["candidate_seed"][0] += np.uint64(1)
    np.savez_compressed(tmp_path / "selected_states.npz", **broken)
    with pytest.raises(ValueError, match="candidate_seed mismatch"):
        validate_calibration_collection(tmp_path)


@pytest.mark.parametrize("counts", ((64, 0, 0), (23, 25, 16)))
def test_duplicate_calibration_rejects_wrong_phase_composition(tmp_path, counts):
    phases = [0] * counts[0] + [1] * counts[1] + [2] * counts[2]
    _write_valid_frozen_calibration(tmp_path, phases)
    with pytest.raises(ValueError, match="phase composition"):
        validate_calibration_collection(tmp_path)


@pytest.mark.parametrize("defect", ("duplicate_state", "multiple_batch", "missing_episode", "frame_oob"))
def test_duplicate_calibration_rejects_provenance_drift(tmp_path, defect):
    _write_valid_frozen_calibration(tmp_path)
    selected_path = tmp_path / "selected_states.npz"
    with np.load(selected_path, allow_pickle=False) as payload:
        selected = {name: payload[name].copy() for name in payload.files}
    if defect == "duplicate_state":
        selected["state_id"][1] = selected["state_id"][0]
    elif defect == "multiple_batch":
        selected["collection_batch_id"][1] = "other"
    elif defect == "missing_episode":
        (tmp_path / "episodes/0000/episode.npz").unlink()
    elif defect == "frame_oob":
        selected["frame_id"][0] = 64
    np.savez_compressed(selected_path, **selected)
    with pytest.raises(ValueError):
        validate_calibration_collection(tmp_path)


def test_prefix_replay_records_numeric_drift_but_still_steps_candidates():
    episode = {"episode_id": np.array([0]), "seed": np.array([5909]),
               "initial_dof_state": np.zeros((18, 2), dtype=np.float32),
               "initial_actor_root_state": np.zeros((2, 13), dtype=np.float32),
               "initial_task_indices": np.array([7, 11]),
               "executed_action_history": np.zeros((1, 18), dtype=np.float32),
               "done_history": np.array([False]), "reference_index": np.array([0]),
               "data_id": np.array(["s1_airplane_lift"]), "start_time": np.array([0.0]),
               "progress_history": np.array([0]), "resolved_sim_config_sha256": np.array(["d" * 64])}
    runtime = _FakePrefixRuntime(drift=2e-5)
    result = replay_prefix_branches(
        runtime, episode, 0, np.zeros((8, 18), dtype=np.float32), assignment_seed=5909
    )
    assert not result.parity_valid
    assert result.task_indices_equal
    assert result.post_candidate_state is not None
    assert len(runtime.actions) == 1


def test_prefix_replay_stops_before_candidate_when_task_indices_differ():
    episode = {"episode_id": np.array([0]), "seed": np.array([5909]),
               "initial_dof_state": np.zeros((18, 2), dtype=np.float32),
               "initial_actor_root_state": np.zeros((2, 13), dtype=np.float32),
               "initial_task_indices": np.array([7, 11]),
               "executed_action_history": np.zeros((1, 18), dtype=np.float32),
               "done_history": np.array([False]), "reference_index": np.array([0]),
               "data_id": np.array(["s1_airplane_lift"]), "start_time": np.array([0.0]),
               "progress_history": np.array([0]),
               "resolved_sim_config_sha256": np.array(["d" * 64])}
    runtime = _FakePrefixRuntime(index_drift=True)
    result = replay_prefix_branches(
        runtime, episode, 0, np.zeros((8, 18), dtype=np.float32), assignment_seed=5909
    )
    assert result.parity_valid
    assert not result.task_indices_equal
    assert result.post_candidate_state is None
    assert runtime.actions == []


def test_replay_schema_validates_required_shapes():
    count = 2
    state = {name: np.zeros((count, *shape), dtype=np.float32) for name, shape in {
        "raw_obs": (1442,), "native_q": (18,), "native_dq": (18,), "object_pose": (7,),
        "object_twist": (6,), "contact_force": (5, 3), "policy_mu": (18,),
        "policy_sigma": (18,), "candidate_actions": (8, 18), "candidate_valid": (8,),
        "cm_delta_xi": (8, 6), "cm_valid": (8,), "cm_token_mask": (8, 16),
        "cm_token_mass": (8, 16), "cm_score": (8,),
    }.items()}
    for name in ("candidate_valid", "cm_valid", "cm_token_mask"):
        state[name] = state[name].astype(bool)
    state.update({"state_id": np.array(["a", "b"]), "collection_batch_id": np.array(["c", "c"]),
                  "episode_id": np.array([0, 1]), "seed": np.array([5909, 5910]),
                  "frame_id": np.array([0, 0]), "reference_index": np.array([0, 0]),
                  "checkpoint_sha256": np.array(["a" * 64] * count),
                  "motion_source_sha": np.array(["b" * 64] * count),
                  "obs_rms_snapshot_sha256": np.array(["c" * 64] * count),
                  "active_reason_mask": np.zeros(count, dtype=np.uint8), "phase_id": np.zeros(count, dtype=np.uint8),
                  "candidate_seed": np.zeros(count, dtype=np.uint64)})
    assert validate_state_records(state) == count
    physics = {"physics_clone_valid": np.ones(count, dtype=bool),
               "env_candidate_ids": np.tile(np.array([0, 0, 1, 2, 3, 4, 5, 6, 7]), (count, 1)),
               "physics_candidate_valid": np.ones((count, 8), dtype=bool),
               "physics_score": np.zeros((count, 8)), "physics_next_object_pose": np.zeros((count, 8, 7)),
               "physics_next_IG": np.zeros((count, 8, 18)), "duplicate_delta_object_pose": np.zeros(count),
               "duplicate_delta_IG": np.zeros(count), "duplicate_delta_score": np.zeros(count),
               "physics_score_producer": np.asarray([PHYSICS_SCORE_PRODUCER] * count),
               "physics_score_object_goal_offset": np.full(count, 6, dtype=np.int64),
               "physics_score_ig_reference_offset": np.full(count, 1, dtype=np.int64),
               "physics_score_shared_baseline": np.ones(count, dtype=np.bool_)}
    assert validate_physics_records(physics, count) == count
    wrong_clock = dict(physics)
    wrong_clock["physics_score_object_goal_offset"] = np.ones(count, dtype=np.int64)
    with pytest.raises(ValueError, match=r"t\+6"):
        validate_physics_records(wrong_clock, count)
    branch_local = dict(physics)
    branch_local["physics_score_shared_baseline"] = np.zeros(count, dtype=np.bool_)
    with pytest.raises(ValueError, match="shared canonical"):
        validate_physics_records(branch_local, count)

    episode = {"episode_id": np.array([0]), "seed": np.array([5909]),
               "initial_dof_state": np.zeros((18, 2)), "initial_actor_root_state": np.zeros((2, 13)),
               "initial_task_indices": np.array([0, 1]), "executed_action_history": np.zeros((3, 18)),
               "done_history": np.array([False, False, True]), "reference_index": np.arange(3),
               "data_id": np.array(["s1_airplane_lift"]), "start_time": np.array([0.0]),
               "progress_history": np.arange(3), "resolved_sim_config_sha256": np.array(["d" * 64])}
    assert validate_episode_replay(episode) == 3


def test_offline_runner_consumes_producer_score_without_recomputing(tmp_path):
    count = 2
    state = {name: np.zeros((count, *shape), dtype=np.float32) for name, shape in {
        "raw_obs": (1442,), "native_q": (18,), "native_dq": (18,), "object_pose": (7,),
        "object_twist": (6,), "contact_force": (5, 3), "policy_mu": (18,),
        "policy_sigma": (18,), "candidate_actions": (8, 18), "candidate_valid": (8,),
        "cm_delta_xi": (8, 6), "cm_valid": (8,), "cm_token_mask": (8, 16),
        "cm_token_mass": (8, 16), "cm_score": (8,),
    }.items()}
    state["candidate_valid"] = np.ones((count, 8), dtype=bool)
    state["cm_valid"] = np.ones((count, 8), dtype=bool)
    state["cm_token_mask"] = np.ones((count, 8, 16), dtype=bool)
    state["cm_token_mass"] = np.ones((count, 8, 16), dtype=np.float32)
    state["cm_score"] = np.tile(np.arange(8, dtype=np.float32), (count, 1))
    state.update({"state_id": np.array(["a", "b"]), "collection_batch_id": np.array(["c", "c"]),
                  "episode_id": np.array([0, 1]), "seed": np.array([5909, 5910]),
                  "frame_id": np.array([0, 0]), "reference_index": np.array([0, 0]),
                  "checkpoint_sha256": np.array(["a" * 64] * count),
                  "motion_source_sha": np.array(["b" * 64] * count),
                  "obs_rms_snapshot_sha256": np.array(["c" * 64] * count),
                  "active_reason_mask": np.zeros(count, dtype=np.uint8), "phase_id": np.zeros(count, dtype=np.uint8),
                  "candidate_seed": np.zeros(count, dtype=np.uint64)})
    physics = {"physics_clone_valid": np.ones(count, dtype=bool),
               "env_candidate_ids": np.tile(np.array([0, 0, 1, 2, 3, 4, 5, 6, 7]), (count, 1)),
               "physics_candidate_valid": np.ones((count, 8), dtype=bool),
               "physics_score": np.tile(np.arange(8, dtype=np.float32), (count, 1)),
               # Zero quaternions are deliberately not scoreable poses.  The
               # evaluator must still succeed because the producer-owned finite
               # physics_score is authoritative and poses are audit evidence only.
               "physics_next_object_pose": np.zeros((count, 8, 7)),
               "physics_next_IG": np.zeros((count, 8, 18)),
               "duplicate_delta_object_pose": np.zeros(count),
               "duplicate_delta_IG": np.zeros(count),
               "duplicate_delta_score": np.zeros(count),
               "physics_score_producer": np.asarray([PHYSICS_SCORE_PRODUCER] * count),
               "physics_score_object_goal_offset": np.full(count, 6, dtype=np.int64),
               "physics_score_ig_reference_offset": np.full(count, 1, dtype=np.int64),
               "physics_score_shared_baseline": np.ones(count, dtype=np.bool_)}
    input_path = tmp_path / "records.npz"
    np.savez_compressed(input_path, **state, **physics)
    output = tmp_path / "run"
    command = [sys.executable, "src/task/CmResidual/tools/eval_v121c_ranking.py",
               "--input", str(input_path), "--run-id", "test-v121c", "--output", str(output),
               "--epsilon-physx", "1e-6", "--bootstrap-draws", "20"]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[4],
                               check=True, capture_output=True, text=True)
    assert "INCONCLUSIVE" in completed.stdout
    assert (output / "run_manifest.json").is_file()
    assert (output / "metrics.jsonl").is_file()
