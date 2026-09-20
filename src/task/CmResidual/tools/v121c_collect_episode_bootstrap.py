"""Collect one official-policy V1.21c episode through the DExplore player.

The outer launcher starts this file once per episode seed.  It monkey-patches only
the player ``run`` hook; vendor task, model, checkpoint restore and action wrapper
remain untouched.  The resulting artifact contains every active pre-action state
and the exact native actions actually passed to ``env.step``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import isaacgym  # noqa: F401  # must precede torch
import torch

from src.task.CmResidual.v121c_ranking import (
    H_REF,
    active_phase,
    candidate_seed,
    canonical_state_id,
)


CHECKPOINT_SHA256 = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"
TIP_KEY_INDICES = torch.tensor([3, 6, 9, 12, 15], dtype=torch.long)


def _prefix_sha256(actions: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for action in actions:
        digest.update(np.asarray(action, dtype="<f4").reshape(18).tobytes())
    return digest.hexdigest()


def _reference_features(task, frame_id: int) -> tuple[torch.Tensor, ...]:
    data_id = int(task.data_id[0].item())
    reference = task.hoi_data[data_id]
    current = reference[frame_id]
    future = reference[frame_id + H_REF]
    key_count = len(task._key_body_ids)
    contact_start = 119 + key_count * 3 + 1
    ig_start = contact_start + 16
    tip_indices = TIP_KEY_INDICES.to(reference.device)
    reference_contact = reference[frame_id:frame_id + H_REF + 1, contact_start:contact_start + 16]
    reference_contact = reference_contact[:, tip_indices].amax(dim=0, keepdim=True)
    reference_ig = reference[frame_id:frame_id + H_REF + 1, ig_start:ig_start + key_count * 3]
    reference_tip_distance = reference_ig.view(H_REF + 1, key_count, 3)[:, tip_indices].norm(dim=-1).amin(dim=0, keepdim=True)
    translation = (future[106:109] - current[106:109]).norm().reshape(1)
    rotation_dot = torch.abs(torch.dot(future[109:113], current[109:113])).clamp(0.0, 1.0)
    rotation = (2.0 * torch.acos(rotation_dot)).reshape(1)
    return reference_contact, translation, rotation, reference_tip_distance


def _active_record(task, frame_id: int) -> tuple[bool, int, int]:
    contact = task._contact_forces[:, task._contact_body_ids, :]
    tip_indices = TIP_KEY_INDICES.to(task.device)
    tips = task._rigid_body_pos[:, task._key_body_ids[tip_indices], :]
    object_points = task.curr_obj_points
    actual_tip_distance = torch.cdist(tips, object_points).amin(dim=-1)
    ref_contact, ref_translation, ref_rotation, ref_tip_distance = _reference_features(task, frame_id)
    active, reason, phase = active_phase(
        contact,
        actual_tip_distance,
        ref_contact,
        ref_translation,
        ref_rotation,
        ref_tip_distance,
    )
    return bool(active[0].item()), int(reason[0].item()), int(phase[0].item())


@torch.no_grad()
def _collect_run(player) -> None:
    output = Path(os.environ["REF2DEX_V121C_EPISODE_OUTPUT"]).resolve()
    batch_id = os.environ["REF2DEX_V121C_COLLECTION_BATCH_ID"]
    episode_id = int(os.environ["REF2DEX_V121C_EPISODE_ID"])
    seed = int(os.environ["REF2DEX_V121C_EPISODE_SEED"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("episode.npz", "active_states.npz", "summary.json")):
        raise FileExistsError(f"refusing to overwrite episode artifacts in {output}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    task = player.env.task
    if int(task.num_envs) != 1:
        raise ValueError("V1.21c collector requires exactly one environment")
    if int(task.control_freq_inv) != 2:
        raise ValueError("V1.21c collector requires 30 Hz / two physics substeps")
    task._adaptive_kappa_enabled = False
    task._enable_early_termination = False
    player.has_batch_dimension = True
    obs_dict = player.env_reset()
    raw_obs = obs_dict["obs"]
    if tuple(raw_obs.shape) != (1, 1442):
        raise ValueError(f"official raw observation must be [1,1442], got {tuple(raw_obs.shape)}")

    initial_dof = task._dof_state.view(1, -1, 2)[0].detach().cpu().numpy().copy()
    initial_roots = task._root_states.view(1, -1, 13)[0].detach().cpu().numpy().copy()
    initial_indices = torch.stack(
        (task.data_id, task.ref_index, task.start_times, task.progress_buf), dim=-1
    )[0].detach().cpu().numpy().copy()
    start_time = int(task.start_times[0].item())
    sequence_length = int(task.max_episode_length[task.data_id[0]].item())

    actions: list[np.ndarray] = []
    done_history: list[bool] = []
    reference_history: list[int] = []
    progress_history: list[int] = []
    records: list[dict[str, object]] = []
    max_actions = sequence_length - start_time
    for frame_id in range(max_actions):
        progress = int(task.progress_buf[0].item())
        if progress + H_REF <= sequence_length - 1:
            active, reason, phase = _active_record(task, progress)
        else:
            active, reason, phase = False, 0, -1

        prepared_obs = player._preproc_obs(raw_obs)
        model_input = {
            "is_train": False,
            "prev_actions": None,
            "obs": prepared_obs,
            "rnn_states": player.states,
        }
        result = player.model(model_input)
        player.states = result["rnn_states"]
        mu = result["mus"].detach()
        sigma = result["sigmas"].detach()
        action = result["actions"].detach().clamp(-1.0, 1.0)
        if not all(torch.isfinite(value).all() for value in (raw_obs, mu, sigma, action)):
            raise FloatingPointError(f"non-finite policy state at frame {frame_id}")

        if active:
            raw_np = raw_obs[0].detach().cpu().numpy().astype(np.float32, copy=True)
            prefix_sha = _prefix_sha256(actions)
            state_id = canonical_state_id(
                CHECKPOINT_SHA256, seed, episode_id, frame_id, raw_np, prefix_sha
            )
            records.append({
                "state_id": state_id,
                "frame_id": frame_id,
                "reference_index": int(task.ref_index[0].item()),
                "progress": progress,
                "phase_id": phase,
                "active_reason_mask": reason,
                "candidate_seed": candidate_seed(batch_id, state_id),
                "executed_action_prefix_sha256": prefix_sha,
                "raw_obs": raw_np,
                "policy_mu": mu[0].cpu().numpy().astype(np.float32, copy=True),
                "policy_sigma": sigma[0].cpu().numpy().astype(np.float32, copy=True),
            })

        executed = action[0].cpu().numpy().astype(np.float32, copy=True)
        next_obs, _, done, _ = player.env_step(player.env, action)
        actions.append(executed)
        is_done = bool(done.reshape(-1)[0].item())
        done_history.append(is_done)
        reference_history.append(int(task.ref_index[0].item()))
        progress_history.append(progress)
        raw_obs = player.obs_to_torch(next_obs)["obs"] if not isinstance(next_obs, dict) else next_obs["obs"]
        if is_done:
            break

    episode = {
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "initial_dof_state": initial_dof,
        "initial_actor_root_state": initial_roots,
        "initial_task_indices": initial_indices,
        "executed_action_history": np.asarray(actions, dtype=np.float32),
        "done_history": np.asarray(done_history, dtype=np.bool_),
        "reference_index": np.asarray(reference_history, dtype=np.int64),
        "data_id": np.asarray(int(initial_indices[0]), dtype=np.int64),
        "start_time": np.asarray(start_time, dtype=np.int64),
        "progress_history": np.asarray(progress_history, dtype=np.int64),
        "resolved_sim_config_sha256": np.asarray(os.environ["REF2DEX_V121C_SIM_CONFIG_SHA256"]),
    }
    np.savez_compressed(output / "episode.npz", **episode)
    if records:
        pool = {
            name: np.asarray([record[name] for record in records])
            for name in records[0]
        }
        # Mixed Python integers above/below int64 otherwise promote to float64
        # and silently destroy the low bits of the SHA-derived seed.
        pool["candidate_seed"] = np.asarray(
            [record["candidate_seed"] for record in records], dtype=np.uint64
        )
    else:
        pool = {
            "state_id": np.asarray([], dtype="<U64"),
            "frame_id": np.asarray([], dtype=np.int64),
            "phase_id": np.asarray([], dtype=np.int64),
        }
    np.savez_compressed(output / "active_states.npz", **pool)
    summary = {
        "episode_id": episode_id,
        "seed": seed,
        "horizon": len(actions),
        "active_state_count": len(records),
        "phase_counts": {
            str(phase): sum(int(record["phase_id"]) == phase for record in records)
            for phase in (0, 1, 2)
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


def main() -> int:
    import run as dexplore_run
    from learning.dexplore_players import DexplorePlayerContinuous

    DexplorePlayerContinuous.run = _collect_run
    dexplore_run.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
