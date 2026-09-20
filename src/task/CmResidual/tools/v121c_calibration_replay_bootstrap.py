"""Replay one episode's selected V1.21c calibration states in nine PhysX envs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import torch

from src.task.CmResidual.v121c_prefix_replay import (
    DExploreTaskPrefixRuntime,
    replay_prefix_branches,
)
from src.task.CmResidual.v121c_ranking import (
    calibration_reference_targets,
    generate_candidate_actions,
    next_state_cost,
    pose_xyzw_to_matrix,
)


POSITION_CEILING_M = 5e-4
ROTATION_CEILING_RAD = 5e-3
KEY_INDICES = (0, 3, 6, 9, 12, 15)


def _quat_geodesic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left / left.norm(dim=-1, keepdim=True)
    right = right / right.norm(dim=-1, keepdim=True)
    return 2.0 * torch.acos((left * right).sum(-1).abs().clamp(0.0, 1.0))


def _canonical_ig(task) -> torch.Tensor:
    from env.tasks.base_dexplore_task import compute_sdf
    from isaacgym.torch_utils import quat_rotate
    from utils import torch_utils

    key_selector = torch.tensor(KEY_INDICES, device=task.device, dtype=torch.long)
    key_positions = task._rigid_body_pos[:, task._key_body_ids[key_selector], :]
    local_points = task.object_points[task.object_id[task.data_id]] * task.ball_size
    object_rotation = task._target_states[:, 3:7]
    object_position = task._target_states[:, :3]
    rotations = object_rotation[:, None].expand(-1, local_points.shape[1], -1).reshape(-1, 4)
    world_points = torch_utils.quat_rotate(rotations, local_points.reshape(-1, 3)).view_as(local_points)
    world_points = world_points + object_position[:, None]
    displacement = compute_sdf(key_positions, world_points).view(-1, 3)
    heading = torch_utils.calc_heading_quat_inv(task._rigid_body_rot[:, 7, :])
    heading = heading[:, None].expand(-1, len(KEY_INDICES), -1).reshape(-1, 4)
    return quat_rotate(heading, displacement).view(task.num_envs, -1)


def _duplicate_next_cost(task, env_slots: np.ndarray, progress: int) -> torch.Tensor:
    """Score duplicate next states without any replay-env current baseline."""
    slots = torch.as_tensor(env_slots, device=task.device, dtype=torch.long)
    post_pose = pose_xyzw_to_matrix(task._target_states[slots, :7]).unsqueeze(0)
    data_ids = task.data_id[slots]
    key_count = len(task._key_body_ids)
    # The duplicate slots belong to the same collected state.  Their data id must
    # agree, and one canonical t+6/t+1 target is shared across both next states.
    if not torch.equal(data_ids, data_ids[:1].expand_as(data_ids)):
        raise RuntimeError("duplicate slots do not share one reference data id")
    goal_pose_t6, reference_ig_t1 = calibration_reference_targets(
        task.hoi_data, data_ids[:1], progress, key_count
    )
    actual_ig = _canonical_ig(task)[slots].unsqueeze(0)
    return next_state_cost(goal_pose_t6, reference_ig_t1, post_pose, actual_ig)[0]


def main() -> int:
    from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task

    collection_root = Path(os.environ["REF2DEX_V121C_COLLECTION_ROOT"]).resolve()
    output = Path(os.environ["REF2DEX_V121C_CALIBRATION_EPISODE_OUTPUT"]).resolve()
    episode_id = int(os.environ["REF2DEX_V121C_EPISODE_ID"])
    output.mkdir(parents=True, exist_ok=False)
    with np.load(collection_root / "selected_states.npz", allow_pickle=False) as selected:
        mask = selected["episode_id"] == episode_id
        states = {name: selected[name][mask] for name in selected.files}
    if states["state_id"].size == 0:
        raise ValueError(f"episode {episode_id} has no selected calibration states")
    with np.load(collection_root / f"episodes/{episode_id:04d}/episode.npz", allow_pickle=False) as source:
        episode = {name: source[name] for name in source.files}

    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    cfg["env"]["stateInit"] = "Start"
    cfg["env"]["hybridInitProb"] = 1.0
    cfg["env"]["enableEarlyTermination"] = False
    cfg["env"]["rolloutLength"] = int(np.asarray(episode["executed_action_history"]).shape[0] + 1)
    cfg["env"]["episodeLength"] = cfg["env"]["rolloutLength"]
    cfg_train["params"]["seed"] = set_seed(
        cfg_train["params"].get("seed", int(states["seed"][0])),
        cfg_train["params"].get("torch_deterministic", False),
    )
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task = None
    rows = []
    try:
        task, env = parse_task(args, cfg, cfg_train, sim_params, distill=False)
        env.reset()
        runtime = DExploreTaskPrefixRuntime(task, gymtorch)
        for index, state_id_value in enumerate(states["state_id"]):
            state_id = str(state_id_value)
            candidates, generated_seeds = generate_candidate_actions(
                torch.from_numpy(states["policy_mu"][index:index + 1]),
                torch.from_numpy(states["policy_sigma"][index:index + 1]),
                str(states["collection_batch_id"][index]),
                [state_id],
            )
            assignment_seed = int(states["candidate_seed"][index])
            if int(generated_seeds[0].item()) != assignment_seed:
                raise ValueError(f"candidate seed mismatch for {state_id}")
            result = replay_prefix_branches(
                runtime,
                episode,
                int(states["frame_id"][index]),
                candidates[0].numpy(),
                assignment_seed=assignment_seed,
            )
            if not result.task_indices_equal or result.post_candidate_state is None:
                raise RuntimeError(f"task/reference index mismatch for {state_id}")
            duplicate_slots = np.flatnonzero(result.env_candidate_ids == 0)
            first_slot, second_slot = map(int, duplicate_slots)
            post_roots = torch.from_numpy(result.post_candidate_state.actor_root_state)
            actor_count = post_roots.shape[1]
            object_actor = int(task._tar_actor_ids[0].item()) % actor_count
            first_object = post_roots[first_slot, object_actor]
            second_object = post_roots[second_slot, object_actor]
            position = float(torch.linalg.vector_norm(first_object[:3] - second_object[:3]).item())
            rotation = float(_quat_geodesic(first_object[3:7][None], second_object[3:7][None])[0].item())
            next_costs = _duplicate_next_cost(
                task, duplicate_slots, int(states["progress"][index])
            )
            # S_B-S_A = -(C_next_B-C_next_A); the shared collected baseline is
            # intentionally absent and therefore cannot become branch-local.
            score_delta = float((-(next_costs[1] - next_costs[0])).item())
            dof = np.asarray(result.post_candidate_state.dof_state)
            rows.append({
                "state_id": state_id,
                "episode_id": episode_id,
                "frame_id": int(states["frame_id"][index]),
                "phase_id": int(states["phase_id"][index]),
                "candidate_seed": assignment_seed,
                "env_candidate_ids": result.env_candidate_ids.tolist(),
                "candidate_zero_env_slots": duplicate_slots.tolist(),
                "numeric_parity_valid": bool(result.parity_valid),
                "numeric_parity_max_abs_error": result.parity_max_abs_error,
                "task_indices_equal": True,
                "duplicate_position_m": position,
                "duplicate_rotation_rad": rotation,
                "duplicate_dof_max_abs": float(np.max(np.abs(dof[first_slot] - dof[second_slot]))),
                "duplicate_score_delta": score_delta,
                "score_producer": "v121c_duplicate_calibration_next_cost.v1",
                "object_goal_offset": 6,
                "ig_reference_offset": 1,
                "score_baseline": "shared_canonical_collected_pre_action_cancelled",
                "duplicate_valid": bool(
                    np.isfinite(position) and np.isfinite(rotation)
                    and np.isfinite(score_delta)
                    and position <= POSITION_CEILING_M
                    and rotation <= ROTATION_CEILING_RAD
                ),
            })
        (output / "records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        summary = {
            "episode_id": episode_id,
            "state_count": len(rows),
            "duplicate_valid_count": sum(row["duplicate_valid"] for row in rows),
            "max_position_m": max(row["duplicate_position_m"] for row in rows),
            "max_rotation_rad": max(row["duplicate_rotation_rad"] for row in rows),
            "max_abs_score_delta": max(abs(row["duplicate_score_delta"]) for row in rows),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True))
        if summary["duplicate_valid_count"] != summary["state_count"]:
            raise RuntimeError("duplicate hard ceiling failed")
        return 0
    finally:
        if task is not None and getattr(task, "sim", None) is not None:
            task.gym.destroy_sim(task.sim)


if __name__ == "__main__":
    raise SystemExit(main())
