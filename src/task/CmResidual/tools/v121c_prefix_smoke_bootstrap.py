"""Create a pinned DExplore task and execute the V1.21c prefix-branch smoke.

This file is launched from the read-only DExplore checkout.  It intentionally
does not load an rl_games player: the smoke validates simulator restore, prefix
replay, public-state diagnostics, and the identical-action duplicate anchor only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

# Isaac Gym's Python 3.8 bindings still reference removed NumPy aliases.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

# Isaac Gym must be imported before anything which imports torch.
import isaacgym  # noqa: F401
from isaacgym import gymtorch
import torch

from src.task.CmResidual.v121c_prefix_replay import (
    DExploreTaskPrefixRuntime,
    replay_prefix_branches,
)


BRANCH_COUNT = 9
SMOKE_FRAME_ID = 1
POSITION_CEILING_M = 5e-4
ROTATION_CEILING_RAD = 5e-3


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quat_geodesic_xyzw(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0)))


def _neutral_action(task) -> np.ndarray:
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


def _parity_diagnostics(state) -> dict[str, object]:
    dof = np.asarray(state.dof_state)
    roots = np.asarray(state.actor_root_state)
    indices = np.asarray(state.task_indices)

    def maximum(value: np.ndarray) -> float:
        return float(np.max(np.abs(value - value[0:1])))

    return {
        "dof_position_max_abs_error": maximum(dof[..., 0]),
        "dof_velocity_max_abs_error": maximum(dof[..., 1]),
        "root_position_max_abs_error_by_actor": [
            maximum(roots[:, actor, :3]) for actor in range(roots.shape[1])
        ],
        "root_quaternion_max_abs_error_by_actor": [
            maximum(roots[:, actor, 3:7]) for actor in range(roots.shape[1])
        ],
        "root_twist_max_abs_error_by_actor": [
            maximum(roots[:, actor, 7:13]) for actor in range(roots.shape[1])
        ],
        "task_indices_equal": bool(np.all(indices == indices[0:1])),
        "task_indices": indices.tolist(),
    }


def _episode_from_task(task, sim_config_sha256: str, neutral: np.ndarray) -> dict[str, object]:
    dof = task._dof_state.view(BRANCH_COUNT, -1, 2).detach().cpu().numpy()
    roots = task._root_states.view(BRANCH_COUNT, -1, 13).detach().cpu().numpy()
    indices = torch.stack(
        (task.data_id, task.ref_index, task.start_times, task.progress_buf), dim=-1
    ).detach().cpu().numpy()
    initial_progress = int(task.progress_buf[0].item())
    initial_reference = int(task.ref_index[0].item())
    prefix = np.repeat(np.asarray(neutral, dtype=np.float32)[None], 2, axis=0)
    return {
        "episode_id": np.asarray(0, dtype=np.int64),
        "seed": np.asarray(5909, dtype=np.int64),
        "initial_dof_state": dof[0].copy(),
        "initial_actor_root_state": roots[0].copy(),
        "initial_task_indices": indices[0].copy(),
        "executed_action_history": prefix,
        "done_history": np.zeros(2, dtype=np.bool_),
        "reference_index": np.asarray(
            [initial_reference, initial_reference], dtype=np.int64
        ),
        "data_id": np.asarray(int(task.data_id[0].item()), dtype=np.int64),
        "start_time": np.asarray(int(task.start_times[0].item()), dtype=np.int64),
        "progress_history": np.asarray(
            [initial_progress, initial_progress + 1], dtype=np.int64
        ),
        "resolved_sim_config_sha256": np.asarray(sim_config_sha256),
    }


def main() -> int:
    from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task

    output_text = os.environ.get("REF2DEX_V121C_SMOKE_OUTPUT")
    if not output_text:
        raise RuntimeError("REF2DEX_V121C_SMOKE_OUTPUT is required")
    output = Path(output_text).resolve()
    output.mkdir(parents=True, exist_ok=False)

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    if int(cfg["env"]["numEnvs"]) != BRANCH_COUNT:
        raise ValueError("V1.21c duplicate smoke requires exactly 9 environments")
    cfg["env"]["stateInit"] = "Start"
    cfg["env"]["hybridInitProb"] = 1.0
    cfg["env"]["enableEarlyTermination"] = False
    cfg["env"]["rolloutLength"] = 8
    cfg["env"]["episodeLength"] = 8
    cfg_train["params"]["seed"] = set_seed(
        cfg_train["params"].get("seed", 5909),
        cfg_train["params"].get("torch_deterministic", False),
    )
    sim_params = parse_sim_params(args, cfg, cfg_train)
    sim_identity = {
        "task": args.task,
        "seed": int(cfg_train["params"]["seed"]),
        "num_envs": BRANCH_COUNT,
        "control_frequency_inv": int(cfg["env"]["controlFrequencyInv"]),
        "sim": cfg.get("sim", {}),
        "physics_engine": str(args.physics_engine),
        "sim_device": args.sim_device,
    }
    sim_config_sha256 = _json_sha256(sim_identity)
    task = None
    try:
        task, env = parse_task(args, cfg, cfg_train, sim_params, distill=False)
        env.reset()
        neutral = _neutral_action(task)
        episode = _episode_from_task(task, sim_config_sha256, neutral)
        candidates = _candidate_actions(neutral)
        result = replay_prefix_branches(
            DExploreTaskPrefixRuntime(task, gymtorch),
            episode,
            SMOKE_FRAME_ID,
            candidates,
            assignment_seed=5909,
        )
        episode_dir = output / "artifacts" / "episodes"
        episode_dir.mkdir(parents=True)
        np.savez_compressed(episode_dir / "0.npz", **episode)
        np.savez_compressed(
            output / "pre_candidate_state.npz",
            dof_state=result.pre_candidate_state.dof_state,
            actor_root_state=result.pre_candidate_state.actor_root_state,
            task_indices=result.pre_candidate_state.task_indices,
        )
        parity_diagnostics = _parity_diagnostics(result.pre_candidate_state)
        if result.post_candidate_state is None:
            metrics = {
                "parity_valid": False,
                "parity_max_abs_error": result.parity_max_abs_error,
                "task_indices_equal": bool(result.task_indices_equal),
                "env_candidate_ids": result.env_candidate_ids.tolist(),
                "duplicate_valid": False,
                "sim_config_sha256": sim_config_sha256,
                "parity_diagnostics": parity_diagnostics,
            }
        else:
            post_roots = np.asarray(result.post_candidate_state.actor_root_state)
            object_actor = int(task._tar_actor_ids[0].item()) % post_roots.shape[1]
            duplicate_slots = np.flatnonzero(result.env_candidate_ids == 0)
            if duplicate_slots.shape != (2,):
                raise RuntimeError("candidate zero must occupy exactly two environment slots")
            first_slot, second_slot = map(int, duplicate_slots)
            object_zero = post_roots[first_slot, object_actor]
            object_duplicate = post_roots[second_slot, object_actor]
            position_divergence = float(
                np.linalg.norm(object_zero[:3] - object_duplicate[:3])
            )
            rotation_divergence = _quat_geodesic_xyzw(
                object_zero[3:7], object_duplicate[3:7]
            )
            dof_divergence = float(
                np.max(
                    np.abs(
                        np.asarray(result.post_candidate_state.dof_state)[first_slot]
                        - np.asarray(result.post_candidate_state.dof_state)[second_slot]
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
                "parity_valid": bool(result.parity_valid),
                "parity_max_abs_error": result.parity_max_abs_error,
                "task_indices_equal": bool(result.task_indices_equal),
                "env_candidate_ids": result.env_candidate_ids.tolist(),
                "candidate_zero_env_slots": duplicate_slots.tolist(),
                "duplicate_valid": bool(duplicate_valid),
                "duplicate_object_position_divergence_m": position_divergence,
                "duplicate_object_rotation_divergence_rad": rotation_divergence,
                "duplicate_dof_max_abs_divergence": dof_divergence,
                "position_ceiling_m": POSITION_CEILING_M,
                "rotation_ceiling_rad": ROTATION_CEILING_RAD,
                "prefix_steps": SMOKE_FRAME_ID,
                "candidate_steps": 1,
                "branch_count": BRANCH_COUNT,
                "sim_config_sha256": sim_config_sha256,
                "parity_diagnostics": parity_diagnostics,
            }
        (output / "smoke_result.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, sort_keys=True))
        if not metrics["task_indices_equal"] or not metrics["duplicate_valid"]:
            raise RuntimeError("V1.21c prefix/duplicate smoke gate failed")
        return 0
    finally:
        if task is not None and getattr(task, "sim", None) is not None:
            task.gym.destroy_sim(task.sim)


if __name__ == "__main__":
    raise SystemExit(main())
