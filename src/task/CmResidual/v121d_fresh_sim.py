"""V1.21d native-init, fresh-simulator counterfactual replay contracts.

The module is intentionally independent of Isaac Gym imports.  A runtime factory
owns simulator construction and destruction; this protocol owns validation,
prefix replay, branch assignment, and the staged calibration helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from src.task.CmResidual.v121c_artifacts import validate_episode_replay
from src.task.CmResidual.v121c_prefix_replay import (
    ACTION_DIM,
    BRANCH_COUNT,
    CANDIDATE_COUNT,
    PARITY_TOLERANCE,
    PublicBranchState,
)
from src.task.CmResidual.v121c_ranking import (
    PHASE_CONTACT,
    PHASE_MOVING,
    PHASE_PRECONTACT,
    freeze_physx_epsilon,
)


BRANCH_BACKEND = "native_init_fresh_sim_9env"
BACKEND_VERSION = "v121d.1"
ALL_DUPLICATE_PAIR_COUNT = math.comb(BRANCH_COUNT, 2)
GATE_A_PHASE_COUNT = 4
GATE_B_PHASE_QUOTAS = {
    PHASE_MOVING: 24,
    PHASE_CONTACT: 24,
    PHASE_PRECONTACT: 16,
}
POSITION_CEILING_M = 5e-4
ROTATION_CEILING_RAD = 5e-3


class NativeInitRuntime(Protocol):
    """One newly-created nine-env task; it must never be reused for another state."""

    def public_state(self) -> PublicBranchState:
        """Return current q/dq, roots, and task indices for all nine envs."""

    def step(self, actions: np.ndarray) -> np.ndarray:
        """Execute one control step and return boolean done/reset flags shaped ``[9]``."""

    def close(self) -> None:
        """Destroy this task and simulator."""


class NativeInitRuntimeFactory(Protocol):
    """Create a new simulator instance for exactly one replay state."""

    def create(
        self,
        episode: Mapping[str, Any],
        branch_count: int,
        sim_instance_id: int,
    ) -> NativeInitRuntime:
        """Native-reset a fresh task matching the episode identity."""


@dataclass(frozen=True)
class ParityDiagnostics:
    dof_position: float
    dof_velocity: float
    root_position: float
    root_quaternion: float
    root_twist: float
    task_indices_exact: bool

    @property
    def max_abs_error(self) -> float:
        return max(
            self.dof_position,
            self.dof_velocity,
            self.root_position,
            self.root_quaternion,
            self.root_twist,
        )

    @property
    def error_vector(self) -> np.ndarray:
        return np.asarray(
            (
                self.dof_position,
                self.dof_velocity,
                self.root_position,
                self.root_quaternion,
                self.root_twist,
            ),
            dtype=np.float64,
        )

    def valid(self, tolerance: float) -> bool:
        return self.max_abs_error <= tolerance and self.task_indices_exact


@dataclass(frozen=True)
class FreshSimReplayResult:
    branch_backend: str
    backend_version: str
    sim_instance_id: int
    sim_create_ok: bool
    sim_destroy_ok: bool
    native_initial_parity: bool
    native_initial_diagnostics: ParityDiagnostics
    pre_candidate_parity: bool
    pre_candidate_diagnostics: ParityDiagnostics | None
    prefix_length: int
    task_indices_exact: bool
    duplicate_replica_count: int
    pre_candidate_state: PublicBranchState | None
    post_candidate_state: PublicBranchState | None


def validate_native_init_episode(record: Mapping[str, Any]) -> int:
    """Validate the V1.21c episode plus V1.21d native-init provenance."""
    horizon = validate_episode_replay(record)
    required = {
        "state_init_mode",
        "hybrid_init_prob",
        "motion_source_sha256",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"native-init episode missing fields: {sorted(missing)}")
    mode = str(np.asarray(record["state_init_mode"]).reshape(-1)[0])
    if mode != "Start":
        raise ValueError("state_init_mode must be Start")
    probability = float(np.asarray(record["hybrid_init_prob"]).reshape(-1)[0])
    if probability != 1.0:
        raise ValueError("hybrid_init_prob must equal 1.0")
    motion_sha = str(np.asarray(record["motion_source_sha256"]).reshape(-1)[0])
    if len(motion_sha) != 64:
        raise ValueError("motion_source_sha256 must be a SHA256 hex string")
    initial_indices = np.asarray(record["initial_task_indices"])
    if initial_indices.shape != (4,) or initial_indices.dtype.kind not in "iu":
        raise ValueError("initial_task_indices must be integer [data,reference,start,progress]")
    return horizon


def _validate_state_layout(state: PublicBranchState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dof = np.asarray(state.dof_state)
    roots = np.asarray(state.actor_root_state)
    indices = np.asarray(state.task_indices)
    if dof.ndim != 3 or dof.shape[0] != BRANCH_COUNT or dof.shape[-1] != 2:
        raise ValueError("dof_state must be [9,D,2]")
    if roots.ndim != 3 or roots.shape[0] != BRANCH_COUNT or roots.shape[-1] != 13:
        raise ValueError("actor_root_state must be [9,A,13]")
    if indices.shape != (BRANCH_COUNT, 4) or indices.dtype.kind not in "iu":
        raise ValueError("task_indices must be integer [9,4]")
    if not np.isfinite(dof).all() or not np.isfinite(roots).all():
        raise ValueError("public state must be finite")
    return dof, roots, indices


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def parity_against_episode_initial(
    state: PublicBranchState,
    episode: Mapping[str, Any],
) -> ParityDiagnostics:
    """Compare every native-reset env against the collector's initial snapshot."""
    dof, roots, indices = _validate_state_layout(state)
    reference_dof = np.asarray(episode["initial_dof_state"])
    reference_roots = np.asarray(episode["initial_actor_root_state"])
    reference_indices = np.asarray(episode["initial_task_indices"])
    if reference_dof.shape != dof.shape[1:] or reference_roots.shape != roots.shape[1:]:
        raise ValueError("episode initial public-state shape mismatch")
    return ParityDiagnostics(
        dof_position=_max_abs(dof[..., 0], reference_dof[None, ..., 0]),
        dof_velocity=_max_abs(dof[..., 1], reference_dof[None, ..., 1]),
        root_position=_max_abs(roots[..., :3], reference_roots[None, ..., :3]),
        root_quaternion=_max_abs(roots[..., 3:7], reference_roots[None, ..., 3:7]),
        root_twist=_max_abs(roots[..., 7:13], reference_roots[None, ..., 7:13]),
        task_indices_exact=bool(np.all(indices == reference_indices[None])),
    )


def parity_between_branches(
    state: PublicBranchState,
    expected_indices: np.ndarray,
) -> ParityDiagnostics:
    """Check branch equality and the collector's expected task indices."""
    dof, roots, indices = _validate_state_layout(state)
    expected = np.asarray(expected_indices)
    if expected.shape != (4,) or expected.dtype.kind not in "iu":
        raise ValueError("expected_indices must be integer [4]")
    return ParityDiagnostics(
        dof_position=_max_abs(dof[..., 0], dof[0:1, ..., 0]),
        dof_velocity=_max_abs(dof[..., 1], dof[0:1, ..., 1]),
        root_position=_max_abs(roots[..., :3], roots[0:1, ..., :3]),
        root_quaternion=_max_abs(roots[..., 3:7], roots[0:1, ..., 3:7]),
        root_twist=_max_abs(roots[..., 7:13], roots[0:1, ..., 7:13]),
        task_indices_exact=bool(
            np.all(indices == indices[0:1]) and np.all(indices == expected[None])
        ),
    )


def expected_task_indices(episode: Mapping[str, Any], frame_id: int) -> np.ndarray:
    """Return collector-observed indices after replaying ``frame_id`` actions."""
    initial = np.asarray(episode["initial_task_indices"], dtype=np.int64)
    if initial.shape != (4,):
        raise ValueError("initial_task_indices must have four entries")
    result = initial.copy()
    result[1] = int(np.asarray(episode["reference_index"])[frame_id])
    result[3] = int(np.asarray(episode["progress_history"])[frame_id])
    return result


def _validate_candidates(candidate_actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(candidate_actions, dtype=np.float32)
    if actions.shape != (CANDIDATE_COUNT, ACTION_DIM):
        raise ValueError("candidate_actions must be [8,18]")
    if not np.isfinite(actions).all() or (np.abs(actions) > 1.0).any():
        raise ValueError("candidate_actions must be finite native actions in [-1,1]")
    return np.ascontiguousarray(actions)


def replay_native_init_fresh_sim(
    factory: NativeInitRuntimeFactory,
    episode: Mapping[str, Any],
    frame_id: int,
    candidate_actions: np.ndarray,
    *,
    sim_instance_id: int,
    all_duplicate: bool = False,
    tolerance: float = PARITY_TOLERANCE,
) -> FreshSimReplayResult:
    """Create, use, and destroy one simulator for one counterfactual state."""
    horizon = validate_native_init_episode(episode)
    if not isinstance(frame_id, (int, np.integer)) or frame_id <= 0 or frame_id >= horizon:
        raise ValueError("V1.21d frame_id must select a state after a non-empty prefix")
    if not isinstance(sim_instance_id, (int, np.integer)) or sim_instance_id < 0:
        raise ValueError("sim_instance_id must be a non-negative integer")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    candidates = _validate_candidates(candidate_actions)
    history = np.asarray(episode["executed_action_history"], dtype=np.float32)
    done_history = np.asarray(episode["done_history"], dtype=bool)
    if done_history[:frame_id].any():
        raise ValueError("cannot prefix-replay past a collector terminal transition")

    runtime: NativeInitRuntime | None = None
    result: FreshSimReplayResult | None = None
    try:
        runtime = factory.create(episode, BRANCH_COUNT, int(sim_instance_id))
        initial_state = runtime.public_state()
        initial = parity_against_episode_initial(initial_state, episode)
        initial_valid = initial.valid(tolerance)
        if not initial_valid:
            result = FreshSimReplayResult(
                BRANCH_BACKEND,
                BACKEND_VERSION,
                int(sim_instance_id),
                True,
                False,
                False,
                initial,
                False,
                None,
                int(frame_id),
                initial.task_indices_exact,
                0,
                None,
                None,
            )
        else:
            for action in history[:frame_id]:
                done = np.asarray(
                    runtime.step(np.repeat(action[None], BRANCH_COUNT, axis=0))
                )
                if done.shape != (BRANCH_COUNT,) or done.dtype != np.bool_:
                    raise ValueError("runtime step must return boolean done flags [9]")
                if done.any():
                    raise RuntimeError("INVALID_IMPLEMENTATION: prefix triggered done/reset")
            before = runtime.public_state()
            pre = parity_between_branches(before, expected_task_indices(episode, int(frame_id)))
            pre_valid = pre.valid(tolerance)
            after = None
            if pre_valid:
                branch_actions = (
                    np.repeat(candidates[0:1], BRANCH_COUNT, axis=0)
                    if all_duplicate
                    else np.concatenate((candidates, candidates[0:1]), axis=0)
                )
                done = np.asarray(runtime.step(branch_actions))
                if done.shape != (BRANCH_COUNT,) or done.dtype != np.bool_:
                    raise ValueError("runtime step must return boolean done flags [9]")
                if done.any():
                    raise RuntimeError("INVALID_IMPLEMENTATION: candidate step triggered reset")
                after = runtime.public_state()
            result = FreshSimReplayResult(
                BRANCH_BACKEND,
                BACKEND_VERSION,
                int(sim_instance_id),
                True,
                False,
                True,
                initial,
                pre_valid,
                pre,
                int(frame_id),
                initial.task_indices_exact and pre.task_indices_exact,
                (BRANCH_COUNT if all_duplicate else 2) if pre_valid else 0,
                before,
                after,
            )
    finally:
        if runtime is not None:
            runtime.close()
    if result is None:
        raise RuntimeError("fresh simulator returned no replay result")
    return replace(result, sim_destroy_ok=True)


def select_gate_a_indices(
    frame_ids: Sequence[int] | np.ndarray,
    state_ids: Sequence[str] | np.ndarray,
    phase_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Select four deterministic prefix quantiles from each of the three phases."""
    frames = np.asarray(frame_ids).reshape(-1)
    states = np.asarray(state_ids).reshape(-1).astype(str)
    phases = np.asarray(phase_ids).reshape(-1)
    if not (frames.size == states.size == phases.size) or frames.size == 0:
        raise ValueError("frame_ids, state_ids, and phase_ids must have equal non-zero length")
    if len(set(states.tolist())) != states.size:
        raise ValueError("state_ids must be unique")
    selected: list[int] = []
    for phase in (PHASE_MOVING, PHASE_CONTACT, PHASE_PRECONTACT):
        candidates = sorted(
            (int(frames[index]), states[index], index)
            for index in np.flatnonzero(phases == phase)
        )
        if len(candidates) < GATE_A_PHASE_COUNT:
            raise RuntimeError("INCONCLUSIVE: Gate A requires four distinct states per phase")
        last = len(candidates) - 1
        positions = [int(math.floor(last * quantile + 0.5)) for quantile in (0, 1 / 3, 2 / 3, 1)]
        if len(set(positions)) != GATE_A_PHASE_COUNT:
            raise RuntimeError("INCONCLUSIVE: Gate A quantiles are not distinct")
        selected.extend(candidates[position][2] for position in positions)
    return np.asarray(selected, dtype=np.int64)


def select_gate_b_indices(
    state_ids: Sequence[str] | np.ndarray,
    phase_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Select the fixed 24/24/16 calibration states by canonical state-id order."""
    states = np.asarray(state_ids).reshape(-1).astype(str)
    phases = np.asarray(phase_ids).reshape(-1)
    if states.size == 0 or states.size != phases.size:
        raise ValueError("state_ids and phase_ids must have equal non-zero length")
    if len(set(states.tolist())) != states.size:
        raise ValueError("state_ids must be unique")
    selected: list[int] = []
    for phase, quota in GATE_B_PHASE_QUOTAS.items():
        candidates = sorted(
            (states[index], index) for index in np.flatnonzero(phases == phase)
        )
        if len(candidates) < quota:
            raise RuntimeError("INCONCLUSIVE: Gate B phase quota is insufficient")
        selected.extend(index for _, index in candidates[:quota])
    return np.asarray(selected, dtype=np.int64)


def all_duplicate_pairwise_divergence(
    object_pose_xyzw: np.ndarray,
    canonical_ig: np.ndarray,
    score: np.ndarray,
) -> np.ndarray:
    """Return 36 rows of position/rotation/IG/score divergence for nine replicas."""
    pose = np.asarray(object_pose_xyzw, dtype=np.float64)
    ig = np.asarray(canonical_ig, dtype=np.float64)
    scores = np.asarray(score, dtype=np.float64)
    if pose.shape != (BRANCH_COUNT, 7) or ig.shape != (BRANCH_COUNT, 18) or scores.shape != (BRANCH_COUNT,):
        raise ValueError("replica outcomes must be pose [9,7], IG [9,18], score [9]")
    if not np.isfinite(pose).all() or not np.isfinite(ig).all() or not np.isfinite(scores).all():
        raise ValueError("replica outcomes must be finite")
    quaternion_norm = np.linalg.norm(pose[:, 3:7], axis=1)
    if (quaternion_norm <= 0).any():
        raise ValueError("replica object quaternions must have non-zero norm")
    rows = []
    for left, right in itertools.combinations(range(BRANCH_COUNT), 2):
        q_left = pose[left, 3:7] / quaternion_norm[left]
        q_right = pose[right, 3:7] / quaternion_norm[right]
        rotation = 2.0 * np.arccos(np.clip(abs(float(np.dot(q_left, q_right))), 0.0, 1.0))
        rows.append(
            (
                np.linalg.norm(pose[left, :3] - pose[right, :3]),
                rotation,
                np.linalg.norm(ig[left] - ig[right]),
                abs(float(scores[left] - scores[right])),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def calibrate_all_duplicate(
    pairwise_divergence: np.ndarray,
    phase_ids: Sequence[int] | np.ndarray,
    *,
    position_ceiling_m: float = POSITION_CEILING_M,
    rotation_ceiling_rad: float = ROTATION_CEILING_RAD,
) -> dict[str, float]:
    """Validate Gate B's 64x36 outcomes and freeze epsilon from state maxima."""
    divergence = np.asarray(pairwise_divergence, dtype=np.float64)
    phases = np.asarray(phase_ids).reshape(-1)
    expected_states = sum(GATE_B_PHASE_QUOTAS.values())
    if divergence.shape != (expected_states, ALL_DUPLICATE_PAIR_COUNT, 4):
        raise ValueError("Gate B divergence must have shape [64,36,4]")
    if phases.shape != (expected_states,) or not np.isfinite(divergence).all():
        raise ValueError("Gate B phases/outcomes must be complete and finite")
    counts = {phase: int(np.sum(phases == phase)) for phase in GATE_B_PHASE_QUOTAS}
    if counts != GATE_B_PHASE_QUOTAS:
        raise ValueError(f"Gate B phase quota mismatch: {counts}")
    max_position = float(np.max(divergence[..., 0]))
    max_rotation = float(np.max(divergence[..., 1]))
    if max_position > position_ceiling_m or max_rotation > rotation_ceiling_rad:
        raise RuntimeError(
            "INVALID_IMPLEMENTATION: all-duplicate replica exceeds the PhysX hard ceiling "
            f"(position={max_position:g}, rotation={max_rotation:g})"
        )
    state_max_score = np.max(np.abs(divergence[..., 3]), axis=1)
    return {
        "state_count": float(expected_states),
        "replica_outcome_count": float(expected_states * BRANCH_COUNT),
        "pair_count": float(divergence.shape[0] * divergence.shape[1]),
        "position_max_m": max_position,
        "rotation_max_rad": max_rotation,
        "score_state_max_p99": float(np.percentile(state_max_score, 99.0)),
        "epsilon_physx": freeze_physx_epsilon(state_max_score),
    }
