"""Produce one single-env prefix or direct-restore outcome for V1.21e.1."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
import isaacgym  # noqa: F401
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate
import torch

from src.task.CmResidual.v121c_ranking import (
    calibration_reference_targets, physics_scores, pose_xyzw_to_matrix,
)
from src.task.CmResidual.v121e_snapshot import PRODUCER

BRANCHES = 1
KEY_INDICES = (0, 3, 6, 9, 12, 15)


def _ig(task) -> torch.Tensor:
    from env.tasks.base_dexplore_task import compute_sdf
    from utils import torch_utils
    sel = torch.tensor(KEY_INDICES, device=task.device, dtype=torch.long)
    keys = task._rigid_body_pos[:, task._key_body_ids[sel], :]
    points = task.object_points[task.object_id[task.data_id]] * task.ball_size
    rotations = task._target_states[:, None, 3:7].expand(-1, points.shape[1], -1).reshape(-1, 4)
    world = torch_utils.quat_rotate(rotations, points.reshape(-1, 3)).view_as(points)
    world = world + task._target_states[:, None, :3]
    displacement = compute_sdf(keys, world).view(-1, 3)
    heading = torch_utils.calc_heading_quat_inv(task._rigid_body_rot[:, 7, :])
    heading = heading[:, None].expand(-1, len(KEY_INDICES), -1).reshape(-1, 4)
    return quat_rotate(heading, displacement).view(task.num_envs, -1)


def _step(task, action: np.ndarray) -> None:
    native = torch.as_tensor(action, device=task.device, dtype=torch.float32).repeat(BRANCHES, 1)
    task.pre_physics_step(native)
    task._physics_step()
    if task.device == "cpu": task.gym.fetch_results(task.sim, True)
    task.post_physics_step()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_snapshot_sha256(state) -> str:
    digest = hashlib.sha256()
    for name in ("snapshot_dof_state", "snapshot_actor_root_state", "raw_obs", "canonical_current_ig"):
        digest.update(np.ascontiguousarray(np.asarray(state[name], dtype="<f4")).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(state["snapshot_task_indices"], dtype="<i8")).tobytes())
    return digest.hexdigest()


def _setter_status(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    return str(value)


def _set_public(task, dof: np.ndarray, roots: np.ndarray, indices: np.ndarray, bookkeeping=None) -> dict:
    dofs = task._dof_state.view(BRANCHES, -1, 2)
    root_states = task._root_states.view(BRANCHES, -1, 13)
    source_dof = torch.as_tensor(dof, device=task.device, dtype=dofs.dtype)
    source_roots = torch.as_tensor(roots, device=task.device, dtype=root_states.dtype)
    dofs.copy_(source_dof.unsqueeze(0).expand_as(dofs))
    root_states.copy_(source_roots.unsqueeze(0).expand_as(root_states))
    for tensor, value in zip((task.data_id, task.ref_index, task.start_times, task.progress_buf), indices):
        tensor.fill_(int(value))
    if bookkeeping is not None:
        task.reset_buf.fill_(int(bookkeeping["reset_buf"]))
        task._terminate_buf.fill_(int(bookkeeping["terminate_buf"]))
        contact = torch.as_tensor(bookkeeping["contact_reset"], device=task.device)
        task.contact_reset.copy_(contact.unsqueeze(0).expand_as(task.contact_reset))
    dof_input = dofs[0].detach().cpu().numpy().copy()
    root_input = root_states[0].detach().cpu().numpy().copy()
    root_status = task.gym.set_actor_root_state_tensor(task.sim, gymtorch.unwrap_tensor(task._root_states))
    dof_status = task.gym.set_dof_state_tensor(task.sim, gymtorch.unwrap_tensor(task._dof_state))
    # GPU pipeline contract: do not refresh or recompute observation/IG between a
    # setter and the next simulate.  The copied setter input is the auditable state.
    return {
        "setter_input_dof_sha256": _sha256_array(dof_input),
        "setter_input_root_sha256": _sha256_array(root_input),
        "setter_input_max_abs_error": {
            "dof": float(np.max(np.abs(dof_input - np.asarray(dof, dtype=np.float32)))),
            "root": float(np.max(np.abs(root_input - np.asarray(roots, dtype=np.float32)))),
        },
        "setter_return_status": {"root": _setter_status(root_status), "dof": _setter_status(dof_status)},
        "setter_task_indices": [int(value) for value in indices],
    }


def main() -> int:
    from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task
    selected_path = Path(os.environ["REF2DEX_V121E_SELECTED"])
    episode_path = Path(os.environ["REF2DEX_V121E_EPISODE"])
    index = int(os.environ["REF2DEX_V121E_STATE_INDEX"])
    mode = os.environ["REF2DEX_V121E_MODE"]
    output = Path(os.environ["REF2DEX_V121E_OUTPUT"])
    with np.load(selected_path, allow_pickle=False) as src:
        state = {k: src[k][index] for k in src.files}
    with np.load(episode_path, allow_pickle=False) as src:
        episode = {k: src[k] for k in src.files}
    set_np_formatting(); args = get_args(); cfg, cfg_train, _ = load_cfg(args)
    if args.motion_file: cfg["env"]["motion_file"] = args.motion_file
    cfg["env"].update(stateInit="Start", hybridInitProb=1.0, enableEarlyTermination=False,
                      rolloutLength=int(len(episode["executed_action_history"]) + 1),
                      episodeLength=int(len(episode["executed_action_history"]) + 1))
    cfg_train["params"]["seed"] = set_seed(cfg_train["params"].get("seed", int(state["seed"])), False)
    task = None
    try:
        task, env = parse_task(args, cfg, cfg_train, parse_sim_params(args, cfg, cfg_train), distill=False)
        env.reset(); task._adaptive_kappa_enabled = False; task._enable_early_termination = False
        if int(task.num_envs) != BRANCHES or int(task.control_freq_inv) != 2:
            raise ValueError("V1.21e.1 producer requires one env at 30 Hz")
        if mode == "prefix":
            initial = np.asarray(episode["initial_task_indices"], dtype=np.int64)
            setter = _set_public(task, episode["initial_dof_state"], episode["initial_actor_root_state"], initial)
            for action in episode["executed_action_history"][:int(state["frame_id"])]: _step(task, action)
        elif mode == "restore":
            setter = _set_public(
                task, state["snapshot_dof_state"], state["snapshot_actor_root_state"],
                state["snapshot_task_indices"], {
                    "reset_buf": state["snapshot_reset_buf"],
                    "terminate_buf": state["snapshot_terminate_buf"],
                    "contact_reset": state["snapshot_contact_reset"],
                },
            )
        else: raise ValueError("mode must be prefix or restore")
        before = torch.stack((task.data_id, task.ref_index, task.start_times, task.progress_buf), -1)
        expected = torch.as_tensor(state["snapshot_task_indices"], device=task.device).expand_as(before)
        if not torch.equal(before, expected): raise RuntimeError("task/reference index mismatch before candidate")
        action = np.clip(np.asarray(state["policy_mu"], dtype=np.float32), -1, 1)
        _step(task, action)
        roots = task._root_states.view(BRANCHES, -1, 13)[0].detach().cpu().numpy().copy()
        dof = task._dof_state.view(BRANCHES, -1, 2)[0].detach().cpu().numpy().copy()
        post_ig = _ig(task)[0]
        actor_count = roots.shape[0]
        obj = int(task._tar_actor_ids[0].item()) % actor_count
        current_pose = torch.as_tensor(state["snapshot_actor_root_state"][obj, :7], device=task.device).view(1, 7)
        current_ig = torch.as_tensor(state["canonical_current_ig"], device=task.device).view(1, 18)
        goal, ref_ig = calibration_reference_targets(task.hoi_data, task.data_id[:1], int(state["progress"]), len(task._key_body_ids))
        next_pose = pose_xyzw_to_matrix(task._target_states[:, :7]).view(1, BRANCHES, 4, 4)
        score = physics_scores(pose_xyzw_to_matrix(current_pose), goal, current_ig, ref_ig,
                               next_pose, post_ig.view(1, BRANCHES, 18))["score"][0, 0]
        record = {
            "state_id": str(state["state_id"]), "mode": mode,
            "repeat": int(os.environ["REF2DEX_V121E_REPEAT"]), "num_envs": 1,
            "candidate_action_sha256": hashlib.sha256(np.asarray(action, dtype="<f4").tobytes()).hexdigest(),
            "task_indices_before": [int(x) for x in state["snapshot_task_indices"].tolist()],
            "canonical_snapshot_sha256": _canonical_snapshot_sha256(state),
            "canonical_raw_obs_sha256": _sha256_array(state["raw_obs"]),
            "canonical_current_ig_sha256": _sha256_array(state["canonical_current_ig"]),
            "object_actor_index": obj, "post_actor_root_state": roots.tolist(),
            "post_dof_state": dof.tolist(), "post_ig": post_ig.cpu().numpy().tolist(),
            "physics_score": float(score.detach().cpu().item()), "score_producer": PRODUCER,
            "score_baseline": "shared_canonical_collected_pre_action", "object_goal_offset": 6,
            "ig_reference_offset": 1,
        }
        record.update(setter)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"state_id": record["state_id"], "mode": mode}, sort_keys=True))
        return 0
    finally:
        if task is not None and getattr(task, "sim", None) is not None: task.gym.destroy_sim(task.sim)


if __name__ == "__main__": raise SystemExit(main())
