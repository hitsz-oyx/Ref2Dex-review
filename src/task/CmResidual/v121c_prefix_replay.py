"""Runtime-neutral prefix replay for the V1.21c PhysX branch protocol.

The real DExplore adapter supplies the four methods in :class:`PrefixReplayRuntime`.
This module owns the protocol which is easy to accidentally vary between a duplicate
smoke and a full ranking run: every branch starts at the source episode initial state,
replays only the recorded executed native actions, is checked before its candidate
step, and then receives eight candidates plus a duplicate of candidate zero.  The
candidate-to-environment assignment is deterministically permuted per state so an
environment's small numerical drift cannot be confounded with one candidate forever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from src.task.CmResidual.v121c_artifacts import validate_episode_replay


ACTION_DIM = 18
CANDIDATE_COUNT = 8
BRANCH_COUNT = CANDIDATE_COUNT + 1
PARITY_TOLERANCE = 1e-5
PHYSICS_SUBSTEPS = 2


@dataclass(frozen=True)
class PublicBranchState:
    """The public state which must agree immediately before a candidate step.

    ``dof_state`` is normally q/dq and ``actor_root_state`` contains object/table
    roots.  Keeping them separate prevents a runtime adapter from omitting either
    category while still allowing its native Isaac Gym layout.
    """

    dof_state: np.ndarray
    actor_root_state: np.ndarray
    task_indices: np.ndarray

    def flattened_float_state(self) -> np.ndarray:
        arrays = (np.asarray(self.dof_state), np.asarray(self.actor_root_state))
        values = []
        for name, value in zip(("dof_state", "actor_root_state"), arrays):
            if value.ndim < 1 or value.shape[0] != BRANCH_COUNT:
                raise ValueError(f"{name} must have {BRANCH_COUNT} branch rows")
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite numeric state")
            values.append(np.ascontiguousarray(value).reshape(BRANCH_COUNT, -1))
        return np.concatenate(values, axis=1)

    def validate_task_indices(self) -> np.ndarray:
        indices = np.asarray(self.task_indices)
        if indices.ndim < 1 or indices.shape[0] != BRANCH_COUNT:
            raise ValueError(f"task_indices must have {BRANCH_COUNT} branch rows")
        if indices.dtype.kind not in "iu":
            raise ValueError("task_indices must be an integer array")
        return np.ascontiguousarray(indices).reshape(BRANCH_COUNT, -1)


class PrefixReplayRuntime(Protocol):
    """Minimal adapter needed to execute the frozen V1.21c replay protocol."""

    def make_envs(self, count: int) -> Any:
        """Create ``count`` validation environments without changing source data."""

    def restore_initial(self, envs: Any, episode: Mapping[str, Any]) -> None:
        """Restore all branches to this episode's recorded initial public state."""

    def step(self, envs: Any, actions: np.ndarray) -> None:
        """Run one native-action control step for every branch, with ``[B,18]`` actions."""

    def public_state(self, envs: Any) -> PublicBranchState:
        """Return q/dq, actor roots and task/reference indices for every branch."""

class DExploreTaskPrefixRuntime:
    """Thin adapter for an already-created nine-env DExplore task.

    ``gymtorch_module`` is injected by the caller after importing Isaac Gym.  This
    keeps this Task-local module importable in ordinary CPU contract tests.  The
    adapter deliberately does not construct, reset, or destroy a simulator: caller
    setup is responsible for the pinned config and input identities.
    """

    def __init__(self, task: Any, gymtorch_module: Any) -> None:
        self.task = task
        self.gymtorch = gymtorch_module

    def make_envs(self, count: int) -> Any:
        if count != BRANCH_COUNT or int(self.task.num_envs) != BRANCH_COUNT:
            raise ValueError("V1.21c prefix replay requires an already-created 9-env task")
        return self.task

    @staticmethod
    def _tensor(value: Any, *, device: torch.device, shape: tuple[int, ...], name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
        return tensor

    def restore_initial(self, envs: Any, episode: Mapping[str, Any]) -> None:
        if envs is not self.task:
            raise ValueError("adapter may only restore its constructed task")
        validate_episode_replay(episode)
        task = self.task
        dof = task._dof_state.view(BRANCH_COUNT, -1, 2)
        roots = task._root_states.view(BRANCH_COUNT, -1, 13)
        source_dof = self._tensor(episode["initial_dof_state"], device=dof.device,
                                  shape=tuple(dof.shape[1:]), name="initial_dof_state")
        source_roots = self._tensor(episode["initial_actor_root_state"], device=roots.device,
                                    shape=tuple(roots.shape[1:]), name="initial_actor_root_state")
        dof.copy_(source_dof.unsqueeze(0).expand_as(dof))
        roots.copy_(source_roots.unsqueeze(0).expand_as(roots))

        def scalar(name: str) -> int:
            value = np.asarray(episode[name]).reshape(-1)[0]
            return int(value)

        task.data_id.fill_(scalar("data_id"))
        task.ref_index.fill_(scalar("reference_index"))
        task.start_times.fill_(scalar("start_time"))
        task.progress_buf.fill_(scalar("progress_history"))
        task.gym.set_actor_root_state_tensor(task.sim, self.gymtorch.unwrap_tensor(task._root_states))
        task.gym.set_dof_state_tensor(task.sim, self.gymtorch.unwrap_tensor(task._dof_state))

    def step(self, envs: Any, actions: np.ndarray) -> None:
        if envs is not self.task:
            raise ValueError("adapter may only step its constructed task")
        if int(getattr(self.task, "control_freq_inv", 0)) != PHYSICS_SUBSTEPS:
            raise ValueError("DExplore native action requires control_freq_inv=2 (30 Hz)")
        native = torch.as_tensor(_validate_candidates_or_branches(actions), device=self.task.device)
        self.task.pre_physics_step(native)
        # Match DExplore BaseTask.step exactly: one action/control step owns two
        # 1/60 s PhysX substeps, followed by one reference/progress update.
        self.task._physics_step()
        if self.task.device == "cpu":
            self.task.gym.fetch_results(self.task.sim, True)
        self.task.post_physics_step()

    def public_state(self, envs: Any) -> PublicBranchState:
        if envs is not self.task:
            raise ValueError("adapter may only snapshot its constructed task")
        task = self.task
        indices = torch.stack((task.data_id, task.ref_index, task.start_times, task.progress_buf), dim=-1)
        return PublicBranchState(task._dof_state.view(BRANCH_COUNT, -1, 2).detach().cpu().numpy().copy(),
                                 task._root_states.view(BRANCH_COUNT, -1, 13).detach().cpu().numpy().copy(),
                                 indices.detach().cpu().numpy().copy())


@dataclass(frozen=True)
class PrefixBranchResult:
    """Protocol outcome with numeric drift retained as a diagnostic."""

    parity_valid: bool
    parity_max_abs_error: float
    task_indices_equal: bool
    env_candidate_ids: np.ndarray
    pre_candidate_state: PublicBranchState
    post_candidate_state: PublicBranchState | None


def _validate_candidates(candidate_actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(candidate_actions, dtype=np.float32)
    if actions.shape != (CANDIDATE_COUNT, ACTION_DIM):
        raise ValueError("candidate_actions must be [8,18]")
    if not np.isfinite(actions).all() or (actions < -1.0).any() or (actions > 1.0).any():
        raise ValueError("candidate_actions must be finite native actions in [-1,1]")
    return np.ascontiguousarray(actions)


def _validate_candidates_or_branches(actions: np.ndarray) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float32)
    if value.shape != (BRANCH_COUNT, ACTION_DIM):
        raise ValueError("actions must be [9,18]")
    if not np.isfinite(value).all() or (value < -1.0).any() or (value > 1.0).any():
        raise ValueError("actions must be finite native actions in [-1,1]")
    return np.ascontiguousarray(value)


def candidate_env_assignment(assignment_seed: int) -> np.ndarray:
    """Map nine env slots to candidates using a reproducible per-state permutation.

    Candidate zero occurs twice and candidates one through seven occur once.  The
    returned array is indexed by environment and contains the assigned candidate id.
    """
    if not isinstance(assignment_seed, (int, np.integer)) or int(assignment_seed) < 0:
        raise ValueError("assignment_seed must be a non-negative integer")
    candidate_ids = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(int(assignment_seed)))
    return np.ascontiguousarray(candidate_ids[generator.permutation(BRANCH_COUNT)])


def _parity(state: PublicBranchState, tolerance: float) -> tuple[bool, float, bool]:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    flattened = state.flattened_float_state()
    indices = state.validate_task_indices()
    max_error = float(np.max(np.abs(flattened - flattened[0:1])))
    equal_indices = bool(np.all(indices == indices[0:1]))
    return max_error <= tolerance, max_error, equal_indices


def replay_prefix_branches(
    runtime: PrefixReplayRuntime,
    episode: Mapping[str, Any],
    frame_id: int,
    candidate_actions: np.ndarray,
    *,
    assignment_seed: int,
    tolerance: float = PARITY_TOLERANCE,
) -> PrefixBranchResult:
    """Replay a source prefix then run a permuted candidate/duplicate multiset.

    ``frame_id`` is the candidate-action frame.  Thus the recorded prefix is exactly
    ``executed_action_history[:frame_id]``.  A terminal transition in that prefix is
    rejected: silently resetting or stepping after it would invalidate the branch.
    Finite public state and identical task/reference indices are hard requirements.
    Cross-environment q/dq/root numerical parity is recorded but does not block the
    candidate step: the deterministic assignment permutation and duplicate anchor
    make that simulator noise measurable without claiming exact counterfactual clones.
    """
    horizon = validate_episode_replay(episode)
    if not isinstance(frame_id, (int, np.integer)) or frame_id < 0 or frame_id >= horizon:
        raise ValueError("frame_id must select an action within the episode horizon")
    actions = _validate_candidates(candidate_actions)
    done = np.asarray(episode["done_history"], dtype=bool)
    if done[:int(frame_id)].any():
        raise ValueError("cannot prefix-replay past a terminal transition")

    envs = runtime.make_envs(BRANCH_COUNT)
    runtime.restore_initial(envs, episode)
    history = np.asarray(episode["executed_action_history"], dtype=np.float32)
    for action in history[:int(frame_id)]:
        runtime.step(envs, np.repeat(action[None], BRANCH_COUNT, axis=0))

    before = runtime.public_state(envs)
    parity_valid, max_error, indices_equal = _parity(before, tolerance)
    env_candidate_ids = candidate_env_assignment(assignment_seed)
    if not indices_equal:
        return PrefixBranchResult(
            parity_valid, max_error, False, env_candidate_ids, before, None
        )

    branch_actions = actions[env_candidate_ids]
    runtime.step(envs, branch_actions)
    return PrefixBranchResult(
        parity_valid, max_error, True, env_candidate_ids, before, runtime.public_state(envs)
    )
