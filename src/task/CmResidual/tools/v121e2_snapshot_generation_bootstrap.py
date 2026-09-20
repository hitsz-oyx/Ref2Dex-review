"""Materialize immutable canonical t-L public snapshots for one frozen episode."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
import isaacgym  # noqa: F401
import torch

from src.task.CmResidual.tools.v121e_snapshot_parity_bootstrap import _ig, _set_public, _step
from src.task.CmResidual.v121e2_warmup import WINDOWS, canonical_snapshot_sha256


def main() -> int:
    from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task
    selected_path = Path(os.environ["REF2DEX_V121E2_SELECTED"])
    episode_path = Path(os.environ["REF2DEX_V121E2_EPISODE"])
    episode_id = int(os.environ["REF2DEX_V121E2_EPISODE_ID"])
    output = Path(os.environ["REF2DEX_V121E2_SNAPSHOT_OUTPUT"])
    with np.load(selected_path, allow_pickle=False) as src:
        selected = [{name: src[name][i] for name in src.files}
                    for i in range(len(src["state_id"])) if int(src["episode_id"][i]) == episode_id]
    with np.load(episode_path, allow_pickle=False) as src:
        episode = {name: src[name] for name in src.files}
    if not selected:
        raise ValueError(f"episode {episode_id} owns no selected states")
    targets = {}
    for state in selected:
        frame = int(state["frame_id"])
        for window in WINDOWS:
            if frame < window: raise ValueError("target frame is shorter than warm-up window")
            targets.setdefault(frame - window, []).append((state, window))

    set_np_formatting(); args = get_args(); cfg, cfg_train, _ = load_cfg(args)
    if args.motion_file: cfg["env"]["motion_file"] = args.motion_file
    horizon = len(episode["executed_action_history"])
    cfg["env"].update(stateInit="Start", hybridInitProb=1.0, enableEarlyTermination=False,
                      rolloutLength=horizon + 1, episodeLength=horizon + 1)
    seed = int(np.asarray(episode["seed"]).reshape(-1)[0])
    cfg_train["params"]["seed"] = set_seed(seed, False)
    task = None
    try:
        task, env = parse_task(args, cfg, cfg_train, parse_sim_params(args, cfg, cfg_train), distill=False)
        env.reset(); task._adaptive_kappa_enabled = False; task._enable_early_termination = False
        if int(task.num_envs) != 1 or int(task.control_freq_inv) != 2:
            raise ValueError("canonical snapshot generation requires one env at 30 Hz")
        _set_public(task, episode["initial_dof_state"], episode["initial_actor_root_state"],
                    np.asarray(episode["initial_task_indices"], dtype=np.int64))
        rows = []

        def capture(frame):
            dof = task._dof_state.view(1, -1, 2)[0].detach().cpu().numpy().copy()
            roots = task._root_states.view(1, -1, 13)[0].detach().cpu().numpy().copy()
            indices = torch.stack((task.data_id, task.ref_index, task.start_times, task.progress_buf), -1)[0].cpu().numpy().copy()
            contact = task.contact_reset[0].detach().cpu().numpy().copy()
            ig = _ig(task)[0].detach().cpu().numpy().copy()
            for state, window in targets[frame]:
                reset = int(task.reset_buf[0].item())
                terminate = int(task._terminate_buf[0].item())
                sha = canonical_snapshot_sha256(str(state["state_id"]), window, dof, roots, indices,
                                                reset, terminate, contact, ig)
                rows.append({"state_id": str(state["state_id"]), "window": window, "frame_id": frame,
                             "snapshot_dof_state": dof, "snapshot_actor_root_state": roots,
                             "snapshot_task_indices": indices, "snapshot_reset_buf": reset,
                             "snapshot_terminate_buf": terminate,
                             "snapshot_contact_reset": contact, "snapshot_ig": ig,
                             "snapshot_sha256": sha})

        if 0 in targets: capture(0)
        for index, action in enumerate(episode["executed_action_history"]):
            _step(task, action)
            frame = index + 1
            if frame in targets: capture(frame)
        expected = len(selected) * len(WINDOWS)
        if len(rows) != expected: raise RuntimeError(f"materialized {len(rows)}/{expected} snapshots")
        payload = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **payload)
        print(json.dumps({"episode_id": episode_id, "snapshot_count": len(rows)}, sort_keys=True))
        return 0
    finally:
        if task is not None and getattr(task, "sim", None) is not None: task.gym.destroy_sim(task.sim)


if __name__ == "__main__": raise SystemExit(main())
