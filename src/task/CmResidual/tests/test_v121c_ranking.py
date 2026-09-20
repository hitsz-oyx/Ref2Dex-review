"""Pure-contract tests for the V1.21c native-action ranking gate."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.task.CmResidual.v121c_artifacts import (
    validate_episode_replay,
    validate_physics_records,
    validate_state_records,
)
from src.task.CmResidual.v121c_collection import CollectionAccumulator, CollectionConfig
from src.task.CmResidual.v121c_ranking import (
    CANDIDATE_COUNT,
    PHASE_CONTACT,
    PHASE_MOVING,
    PHASE_PRECONTACT,
    active_phase,
    branch_state_parity,
    calibrate_duplicate_anchor,
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
    ranking_eligible,
    select_phase_quota,
    top1_metrics,
    CollectionInsufficient,
)
from src.task.CmResidual.dexplore_cm_geometry import dexplore_action_to_native_targets


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
               "physics_candidate_valid": np.ones((count, 8), dtype=bool),
               "physics_score": np.zeros((count, 8)), "physics_next_object_pose": np.zeros((count, 8, 7)),
               "physics_next_IG": np.zeros((count, 8, 18)), "duplicate_delta_object_pose": np.zeros(count),
               "duplicate_delta_IG": np.zeros(count), "duplicate_delta_score": np.zeros(count)}
    assert validate_physics_records(physics, count) == count

    episode = {"episode_id": np.array([0]), "seed": np.array([5909]),
               "initial_dof_state": np.zeros((18, 2)), "initial_actor_root_state": np.zeros((2, 13)),
               "initial_task_indices": np.array([0, 1]), "executed_action_history": np.zeros((3, 18)),
               "done_history": np.array([False, False, True]), "reference_index": np.arange(3),
               "data_id": np.array(["s1_airplane_lift"]), "start_time": np.array([0.0]),
               "progress_history": np.arange(3), "resolved_sim_config_sha256": np.array(["d" * 64])}
    assert validate_episode_replay(episode) == 3


def test_offline_runner_writes_new_manifest_and_metrics(tmp_path):
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
               "physics_candidate_valid": np.ones((count, 8), dtype=bool),
               "physics_score": np.tile(np.arange(8, dtype=np.float32), (count, 1)),
               "physics_next_object_pose": np.zeros((count, 8, 7)),
               "physics_next_IG": np.zeros((count, 8, 18)),
               "duplicate_delta_object_pose": np.zeros(count),
               "duplicate_delta_IG": np.zeros(count),
               "duplicate_delta_score": np.zeros(count)}
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
