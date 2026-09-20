"""Execute the V1.21d source-collect then fresh-sim native-init Gate 0 smoke."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

# Isaac Gym's Python 3.8 bindings still reference removed NumPy aliases.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401
import torch

from src.task.CmResidual.v121c_prefix_replay import PublicBranchState
from src.task.CmResidual.v121d_fresh_sim import (
    BRANCH_BACKEND,
    POSITION_CEILING_M,
    ROTATION_CEILING_RAD,
    replay_native_init_fresh_sim,
)


BRANCH_COUNT = 9
SMOKE_FRAME_ID = 1


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_indices(task: Any) -> np.ndarray:
    return torch.stack(
        (task.data_id, task.ref_index, task.start_times, task.progress_buf), dim=-1
    ).detach().cpu().numpy().copy()


def _public_state(task: Any) -> PublicBranchState:
    count = int(task.num_envs)
    return PublicBranchState(
        task._dof_state.view(count, -1, 2).detach().cpu().numpy().copy(),
        task._root_states.view(count, -1, 13).detach().cpu().numpy().copy(),
        _task_indices(task),
    )


def _step_task(task: Any, actions: np.ndarray) -> np.ndarray:
    native = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=task.device)
    if native.shape != (int(task.num_envs), 18):
        raise ValueError("native action batch does not match task env count")
    task.pre_physics_step(native)
    task.gym.simulate(task.sim)
    if task.device == "cpu":
        task.gym.fetch_results(task.sim, True)
    task.post_physics_step()
    return task.reset_buf.detach().cpu().numpy().astype(np.bool_, copy=True)


def _neutral_action(task: Any) -> np.ndarray:
    current_q = task._dof_pos[0].detach()
    scale = task._pd_action_scale.detach()
    action = torch.zeros(18, dtype=torch.float32, device=current_q.device)
    action[6:] = 2.0 * current_q[6:] / scale[6:] - 1.0
    return action.clamp(-1.0, 1.0).cpu().numpy()


def _candidate_actions(neutral: np.ndarray) -> np.ndarray:
    actions = np.repeat(np.asarray(neutral, dtype=np.float32)[None], 8, axis=0)
    for index in range(1, 8):
        actions[index, (index - 1) % 6] += np.float32(0.01 * index)
        actions[index, 6 + (index - 1) % 12] += np.float32(0.01)
    return np.clip(actions, -1.0, 1.0)


def _quat_geodesic_xyzw(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0)))


def _sim_identity(args: Any, cfg: Mapping[str, Any], seed: int, env_count: int) -> dict[str, Any]:
    return {
        "task": args.task,
        "seed": seed,
        "num_envs": env_count,
        "control_frequency_inv": int(cfg["env"]["controlFrequencyInv"]),
        "sim": cfg.get("sim", {}),
        "physics_engine": str(args.physics_engine),
        "sim_device": args.sim_device,
        "state_init_mode": "Start",
        "hybrid_init_prob": 1.0,
    }


def _destroy_task(task: Any) -> None:
    if task is not None and getattr(task, "sim", None) is not None:
        task.gym.destroy_sim(task.sim)
        task.sim = None


class _DExploreNativeRuntime:
    def __init__(self, task: Any) -> None:
        self.task = task
        self.closed = False

    def public_state(self) -> PublicBranchState:
        if self.closed:
            raise RuntimeError("simulator is already destroyed")
        return _public_state(self.task)

    def step(self, actions: np.ndarray) -> np.ndarray:
        if self.closed:
            raise RuntimeError("simulator is already destroyed")
        return _step_task(self.task, actions)

    def close(self) -> None:
        if self.closed:
            raise RuntimeError("simulator may only be destroyed once")
        _destroy_task(self.task)
        self.closed = True


class _DExploreFreshFactory:
    def __init__(
        self,
        args: Any,
        cfg: Mapping[str, Any],
        cfg_train: Mapping[str, Any],
        parse_sim_params: Any,
        parse_task: Any,
        set_seed: Any,
    ) -> None:
        self.args = args
        self.cfg = cfg
        self.cfg_train = cfg_train
        self.parse_sim_params = parse_sim_params
        self.parse_task = parse_task
        self.set_seed = set_seed
        self.create_count = 0
        self.last_object_actor_index: int | None = None
        self.last_sim_config_sha256: str | None = None

    def create(self, episode: Mapping[str, Any], branch_count: int, sim_instance_id: int) -> _DExploreNativeRuntime:
        if branch_count != BRANCH_COUNT or sim_instance_id != self.create_count:
            raise ValueError("fresh factory requires monotonic one-shot sim_instance_id values")
        cfg = copy.deepcopy(self.cfg)
        cfg_train = copy.deepcopy(self.cfg_train)
        cfg["env"]["numEnvs"] = branch_count
        seed = int(np.asarray(episode["seed"]).reshape(-1)[0])
        cfg_train["params"]["seed"] = self.set_seed(
            seed, cfg_train["params"].get("torch_deterministic", False)
        )
        self.last_sim_config_sha256 = _json_sha256(
            _sim_identity(self.args, cfg, seed, branch_count)
        )
        sim_params = self.parse_sim_params(self.args, cfg, cfg_train)
        task, env = self.parse_task(self.args, cfg, cfg_train, sim_params, distill=False)
        try:
            env.reset()
        except BaseException:
            _destroy_task(task)
            raise
        actor_count = task._root_states.view(branch_count, -1, 13).shape[1]
        self.last_object_actor_index = int(task._tar_actor_ids[0].item()) % actor_count
        self.create_count += 1
        return _DExploreNativeRuntime(task)


def _collect_source_episode(
    args: Any,
    cfg: Mapping[str, Any],
    cfg_train: Mapping[str, Any],
    parse_sim_params: Any,
    parse_task: Any,
    set_seed: Any,
    motion_sha256: str,
) -> tuple[dict[str, object], np.ndarray]:
    source_cfg = copy.deepcopy(cfg)
    source_train = copy.deepcopy(cfg_train)
    source_cfg["env"]["numEnvs"] = 1
    seed = int(source_train["params"].get("seed", 5909))
    source_train["params"]["seed"] = set_seed(
        seed, source_train["params"].get("torch_deterministic", False)
    )
    sim_params = parse_sim_params(args, source_cfg, source_train)
    task = None
    try:
        task, env = parse_task(args, source_cfg, source_train, sim_params, distill=False)
        env.reset()
        initial = _public_state(task)
        neutral = _neutral_action(task)
        initial_indices = initial.task_indices[0].copy()
        done = _step_task(task, neutral[None])
        if done.any():
            raise RuntimeError("source prefix unexpectedly reached done/reset")
        after_prefix_indices = _task_indices(task)[0]
        prefix = np.repeat(neutral[None], 2, axis=0)
        sim_sha = _json_sha256(_sim_identity(args, source_cfg, seed, 1))
        episode = {
            "episode_id": np.asarray(0, dtype=np.int64),
            "seed": np.asarray(seed, dtype=np.int64),
            "initial_dof_state": initial.dof_state[0].copy(),
            "initial_actor_root_state": initial.actor_root_state[0].copy(),
            "initial_task_indices": initial_indices.astype(np.int64, copy=False),
            "executed_action_history": prefix,
            "done_history": np.zeros(2, dtype=np.bool_),
            "reference_index": np.asarray(
                [initial_indices[1], after_prefix_indices[1]], dtype=np.int64
            ),
            "data_id": np.asarray(int(initial_indices[0]), dtype=np.int64),
            "start_time": np.asarray(int(initial_indices[2]), dtype=np.int64),
            "progress_history": np.asarray(
                [initial_indices[3], after_prefix_indices[3]], dtype=np.int64
            ),
            "resolved_sim_config_sha256": np.asarray(sim_sha),
            "state_init_mode": np.asarray("Start"),
            "hybrid_init_prob": np.asarray(1.0, dtype=np.float64),
            "motion_source_sha256": np.asarray(motion_sha256),
        }
        return episode, neutral
    finally:
        _destroy_task(task)


def _diagnostics(value: Any) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "dof_position_max_abs_error": value.dof_position,
        "dof_velocity_max_abs_error": value.dof_velocity,
        "root_position_max_abs_error": value.root_position,
        "root_quaternion_max_abs_error": value.root_quaternion,
        "root_twist_max_abs_error": value.root_twist,
        "task_indices_exact": value.task_indices_exact,
        "max_abs_error": value.max_abs_error,
    }


def main() -> int:
    from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task

    output_text = os.environ.get("REF2DEX_V121D_SMOKE_OUTPUT")
    motion_sha256 = os.environ.get("REF2DEX_V121D_MOTION_SHA256")
    if not output_text or not motion_sha256 or len(motion_sha256) != 64:
        raise RuntimeError("V1.21d output and motion SHA256 environment are required")
    output = Path(output_text).resolve()
    output.mkdir(parents=True, exist_ok=False)

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    if int(cfg["env"]["numEnvs"]) != BRANCH_COUNT:
        raise ValueError("V1.21d Gate 0 requires exactly 9 validation environments")
    cfg["env"]["stateInit"] = "Start"
    cfg["env"]["hybridInitProb"] = 1.0
    cfg["env"]["enableEarlyTermination"] = False
    cfg["env"]["rolloutLength"] = 8
    cfg["env"]["episodeLength"] = 8

    episode, neutral = _collect_source_episode(
        args, cfg, cfg_train, parse_sim_params, parse_task, set_seed, motion_sha256
    )
    factory = _DExploreFreshFactory(
        args, cfg, cfg_train, parse_sim_params, parse_task, set_seed
    )
    result = replay_native_init_fresh_sim(
        factory,
        episode,
        SMOKE_FRAME_ID,
        _candidate_actions(neutral),
        sim_instance_id=0,
    )

    episode_dir = output / "artifacts" / "episodes"
    episode_dir.mkdir(parents=True)
    np.savez_compressed(episode_dir / "0.npz", **episode)
    if result.pre_candidate_state is not None:
        np.savez_compressed(
            output / "pre_candidate_state.npz",
            dof_state=result.pre_candidate_state.dof_state,
            actor_root_state=result.pre_candidate_state.actor_root_state,
            task_indices=result.pre_candidate_state.task_indices,
        )

    position_divergence = None
    rotation_divergence = None
    dof_divergence = None
    duplicate_valid = False
    if result.post_candidate_state is not None:
        roots = np.asarray(result.post_candidate_state.actor_root_state)
        object_actor = factory.last_object_actor_index
        if object_actor is None:
            raise RuntimeError("fresh factory did not expose the object actor")
        object_zero = roots[0, object_actor]
        object_duplicate = roots[8, object_actor]
        position_divergence = float(np.linalg.norm(object_zero[:3] - object_duplicate[:3]))
        rotation_divergence = _quat_geodesic_xyzw(object_zero[3:7], object_duplicate[3:7])
        dof_divergence = float(
            np.max(
                np.abs(
                    np.asarray(result.post_candidate_state.dof_state)[0]
                    - np.asarray(result.post_candidate_state.dof_state)[8]
                )
            )
        )
        duplicate_valid = (
            np.isfinite(position_divergence)
            and np.isfinite(rotation_divergence)
            and position_divergence <= POSITION_CEILING_M
            and rotation_divergence <= ROTATION_CEILING_RAD
        )

    metrics = {
        "branch_backend": BRANCH_BACKEND,
        "sim_instance_id": result.sim_instance_id,
        "sim_create_ok": result.sim_create_ok,
        "sim_destroy_ok": result.sim_destroy_ok,
        "source_sim_create_count": 1,
        "source_sim_destroy_count": 1,
        "validation_sim_create_count": factory.create_count,
        "validation_sim_destroy_count": int(result.sim_destroy_ok),
        "source_sim_config_sha256": str(
            np.asarray(episode["resolved_sim_config_sha256"]).reshape(-1)[0]
        ),
        "validation_sim_config_sha256": factory.last_sim_config_sha256,
        "native_initial_parity": result.native_initial_parity,
        "native_initial_diagnostics": _diagnostics(result.native_initial_diagnostics),
        "pre_candidate_parity": result.pre_candidate_parity,
        "pre_candidate_diagnostics": _diagnostics(result.pre_candidate_diagnostics),
        "task_indices_exact": result.task_indices_exact,
        "prefix_steps": result.prefix_length,
        "candidate_steps": int(result.post_candidate_state is not None),
        "duplicate_valid": bool(duplicate_valid),
        "duplicate_object_position_divergence_m": position_divergence,
        "duplicate_object_rotation_divergence_rad": rotation_divergence,
        "duplicate_dof_max_abs_divergence": dof_divergence,
        "position_ceiling_m": POSITION_CEILING_M,
        "rotation_ceiling_rad": ROTATION_CEILING_RAD,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output / "smoke_result.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True, allow_nan=False))
    if not (
        result.native_initial_parity
        and result.pre_candidate_parity
        and result.sim_destroy_ok
        and duplicate_valid
    ):
        raise RuntimeError("V1.21d native-init/fresh-sim Gate 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
