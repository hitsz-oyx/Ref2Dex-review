"""Produce one independent V1.21e.2 full-prefix or warm-up arm."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
import isaacgym  # noqa: F401
import torch

from src.task.CmResidual.tools.v121e_snapshot_parity_bootstrap import (
    _ig, _set_public, _sha256_array, _step,
)
from src.task.CmResidual.v121c_ranking import calibration_reference_targets, physics_scores, pose_xyzw_to_matrix
from src.task.CmResidual.v121e2_warmup import PRODUCER, WINDOWS, warmup_action_slice


def _episode_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task
    selected_path = Path(os.environ["REF2DEX_V121E2_SELECTED"])
    snapshots_path = Path(os.environ["REF2DEX_V121E2_SNAPSHOTS"])
    episode_path = Path(os.environ["REF2DEX_V121E2_EPISODE"])
    index = int(os.environ["REF2DEX_V121E2_STATE_INDEX"])
    method = os.environ["REF2DEX_V121E2_METHOD"]
    repeat = int(os.environ["REF2DEX_V121E2_REPEAT"])
    output = Path(os.environ["REF2DEX_V121E2_OUTPUT"])
    if method != "full" and int(method) not in WINDOWS: raise ValueError("invalid warm-up method")
    with np.load(selected_path, allow_pickle=False) as src:
        state = {name: src[name][index] for name in src.files}
    with np.load(episode_path, allow_pickle=False) as src:
        episode = {name: src[name] for name in src.files}
    with np.load(snapshots_path, allow_pickle=False) as src:
        snapshot_rows = [{name: src[name][i] for name in src.files} for i in range(len(src["state_id"]))]
    state_id = str(state["state_id"]); frame = int(state["frame_id"])
    lookup = {(str(row["state_id"]), int(row["window"])): row for row in snapshot_rows}
    target = lookup[(state_id, 0)]

    set_np_formatting(); args = get_args(); cfg, cfg_train, _ = load_cfg(args)
    if args.motion_file: cfg["env"]["motion_file"] = args.motion_file
    horizon = len(episode["executed_action_history"])
    cfg["env"].update(stateInit="Start", hybridInitProb=1.0, enableEarlyTermination=False,
                      rolloutLength=horizon + 1, episodeLength=horizon + 1)
    cfg_train["params"]["seed"] = set_seed(int(state["seed"]), False)
    task = None
    try:
        task, env = parse_task(args, cfg, cfg_train, parse_sim_params(args, cfg, cfg_train), distill=False)
        env.reset(); task._adaptive_kappa_enabled = False; task._enable_early_termination = False
        if int(task.num_envs) != 1 or int(task.control_freq_inv) != 2:
            raise ValueError("V1.21e.2 arm requires one env at 30 Hz")
        if method == "full":
            setter = _set_public(task, episode["initial_dof_state"], episode["initial_actor_root_state"],
                                 np.asarray(episode["initial_task_indices"], dtype=np.int64))
            for action in episode["executed_action_history"][:frame]: _step(task, action)
            source_sha = "initial:" + _sha256_array(episode["initial_dof_state"])
            pre_ig = _ig(task)[0]
        else:
            window = int(method); source = lookup[(state_id, window)]
            setter = _set_public(task, source["snapshot_dof_state"], source["snapshot_actor_root_state"],
                                 source["snapshot_task_indices"], {
                                     "reset_buf": source["snapshot_reset_buf"],
                                     "terminate_buf": source["snapshot_terminate_buf"],
                                     "contact_reset": source["snapshot_contact_reset"],
                                 })
            for action in warmup_action_slice(episode["executed_action_history"], frame, window):
                _step(task, action)
            source_sha = str(source["snapshot_sha256"])
            pre_ig = torch.as_tensor(source["snapshot_ig"], device=task.device) if window == 0 else _ig(task)[0]
        indices = torch.stack((task.data_id, task.ref_index, task.start_times, task.progress_buf), -1)[0]
        expected = torch.as_tensor(target["snapshot_task_indices"], device=task.device)
        if not torch.equal(indices, expected): raise RuntimeError("task/reference index mismatch before candidate")
        pre_roots = task._root_states.view(1, -1, 13)[0].detach().cpu().numpy().copy()
        pre_dof = task._dof_state.view(1, -1, 2)[0].detach().cpu().numpy().copy()
        action = np.clip(np.asarray(state["policy_mu"], dtype=np.float32), -1, 1)
        _step(task, action)
        post_roots = task._root_states.view(1, -1, 13)[0].detach().cpu().numpy().copy()
        post_dof = task._dof_state.view(1, -1, 2)[0].detach().cpu().numpy().copy()
        post_ig = _ig(task)[0]
        actor_count = post_roots.shape[0]
        obj = int(task._tar_actor_ids[0].item()) % actor_count
        current_pose = torch.as_tensor(state["snapshot_actor_root_state"][obj, :7], device=task.device).view(1, 7)
        current_ig = torch.as_tensor(state["canonical_current_ig"], device=task.device).view(1, 18)
        goal, ref_ig = calibration_reference_targets(task.hoi_data, task.data_id[:1], int(state["progress"]), len(task._key_body_ids))
        score = physics_scores(pose_xyzw_to_matrix(current_pose), goal, current_ig, ref_ig,
                               pose_xyzw_to_matrix(task._target_states[:, :7]).view(1, 1, 4, 4),
                               post_ig.view(1, 1, 18))["score"][0, 0]
        record = {
            "state_id": state_id, "method": method, "repeat": repeat, "num_envs": 1,
            "source_snapshot_sha256": source_sha, "source_episode_sha256": _episode_sha(episode_path),
            "candidate_action_sha256": hashlib.sha256(np.asarray(action, dtype="<f4").tobytes()).hexdigest(),
            "task_indices_target": [int(value) for value in target["snapshot_task_indices"].tolist()],
            "object_actor_index": obj,
            "pre_actor_root_state": pre_roots.tolist(), "pre_dof_state": pre_dof.tolist(),
            "pre_ig": pre_ig.detach().cpu().numpy().tolist(),
            "post_actor_root_state": post_roots.tolist(), "post_dof_state": post_dof.tolist(),
            "post_ig": post_ig.detach().cpu().numpy().tolist(), "physics_score": float(score.item()),
            "score_producer": PRODUCER, "score_baseline": "shared_canonical_collected_pre_action",
            "object_goal_offset": 6, "ig_reference_offset": 1,
        }
        record.update(setter)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"state_id": state_id, "method": method, "repeat": repeat}, sort_keys=True))
        return 0
    finally:
        if task is not None and getattr(task, "sim", None) is not None: task.gym.destroy_sim(task.sim)


if __name__ == "__main__": raise SystemExit(main())
