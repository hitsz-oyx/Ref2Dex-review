"""Task-local provenance wrapper around IsaacGymEnvs and rl_games PPO."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import isaacgym  # Must precede torch imports in the Isaac Gym runtime.
from isaacgym import gymapi
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from src.task.CmDecoderv2.kinematics import InspireKinematics, QUERY_LINKS
from src.task.CmDecoderv2.rl.online_base import sha256
from src.task.CmDecoderv2.rl.residual_contract import actual_queries, matrix_pose, pose_matrix, sim_to_native

ROOT = Path(__file__).resolve().parents[5]
VENDOR = ROOT / "third_party/IsaacGymEnvs"
CODE_PATHS = [
    "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py",
    "third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml",
    "src/task/CmDecoderv2/rl/online_base.py",
    "src/task/CmDecoderv2/rl/residual_contract.py",
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def code_hashes():
    return {p: sha256(ROOT / p) for p in CODE_PATHS}


def make_config(args):
    with initialize_config_dir(config_dir=str(VENDOR / "isaacgymenvs/cfg"), version_base="1.1"):
        cfg = compose(config_name="config", overrides=[
            "task=CmResidualOnline", "train=CmResidualPPO", "headless=True", "force_render=False",
            f"num_envs={args.num_envs}", f"max_iterations={args.iterations}", "seed=42",
            "sim_device=cuda:0", "rl_device=cuda:0", "graphics_device_id=-1",
        ])
    result = OmegaConf.to_container(cfg, resolve=True)
    train = result["train"]["params"]["config"]
    train.update(device="cuda:0", train_dir=str(args.output.parent.resolve()),
                 full_experiment_name=args.output.name, name=args.output.name,
                 save_best_after=0, save_frequency=25, minibatch_size=min(2048, args.num_envs * 32))
    return result


def create_env(config):
    return isaacgym_task_map[config["task"]["name"]](
        cfg=config["task"], rl_device="cuda:0", sim_device="cuda:0", graphics_device_id=-1,
        headless=True, virtual_screen_capture=False, force_render=False)


@torch.inference_mode()
def rollout(env, output: Path, policy=None):
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.compute_observations()
    returns = torch.zeros(env.num_envs, device=env.device)
    peaks = torch.zeros_like(returns)
    finished = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    successes = torch.zeros_like(finished)
    wrist_track, object_track, actions_track = [], [], []
    with (output / "rollout_metrics.jsonl").open("w") as log:
        for step in range(env.max_episode_length):
            action = torch.zeros((env.num_envs, 12), device=env.device) if policy is None else policy(env.obs_buf)
            obs, reward, done, info = env.step(action)
            active = ~finished
            returns[active] += reward[active]
            # step() auto-resets done envs, so use terminal observations/episode stats.
            episode = info.get("episode")
            if episode:
                ids = done.nonzero(as_tuple=False).flatten()
                new = active[ids]
                successes[ids[new]] = episode["success"][new].bool()
                peaks[ids[new]] = episode["max_lift_m"][new]
            still = active & ~done.bool()
            peaks[still] = torch.maximum(peaks[still], env.episode_max_lift[still])
            finished |= done.bool()
            wrist_pose = info["terminal_wrist_pose"][0] if done[0] else env.actual_link_poses()[0, 0]
            object_pose = info["terminal_object_pose"][0] if done[0] else env.actor_root_state[env.object_indices[0].long(), :7]
            wrist_track.append(wrist_pose.cpu().numpy().copy())
            object_track.append(object_pose.cpu().numpy().copy())
            actions_track.append(action[0].cpu().numpy().copy())
            row = {"step": step + 1, "return_mean": returns.mean().item(), "max_lift_mean_m": peaks.mean().item(),
                   "finished": int(finished.sum()), "successes": int(successes.sum()),
                   "residual_rms": action.square().mean().sqrt().item()}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if finished.all():
                break
    if not finished.all():
        raise AssertionError("A full legal reference episode did not terminate")
    np.savez(output / "rollout.npz", wrist=np.stack(wrist_track), object_pose=np.stack(object_track), actions=np.stack(actions_track))
    result = {"episodes": env.num_envs, "control_steps": step + 1, "success_rate": successes.float().mean().item(),
              "return_mean": returns.mean().item(), "max_lift_mean_m": peaks.mean().item(),
              "max_lift_max_m": peaks.max().item(), "online_decoder_calls": env.base.calls,
              "success_definition": "instantaneous object lift > 0.08 m; not sustained grasp"}
    write_json(output / "evaluation.json", result)
    return result


@torch.inference_mode()
def validate(env, output):
    if env.num_envs < 2:
        raise ValueError("Reset isolation gate requires at least two environments")
    fk = InspireKinematics(ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")
    frozen_before = {k: v.clone() for k, v in env.base.model.state_dict().items()}
    initial_root = env.actor_root_state[env.hand_indices.long()].clone()
    initial = env.initial_links[0].clone()
    for _ in range(12):
        env.base_wrist[:] = initial
        env.base_q[:] = env.initial_native[[6, 8, 10, 12, 14, 15]]
        action = torch.zeros((env.num_envs, 12), device=env.device)
        action[0, 6] = 1
        env.step(action)
    target_effect = (env.actual_link_poses()[0, 0, :3, 3] - env.actual_link_poses()[1, 0, :3, 3]).norm().item()
    assert target_effect > 1e-4, f"Wrist residual did not actuate: {target_effect}"
    torch.testing.assert_close(env.actor_root_state[env.hand_indices.long()], initial_root)
    native = sim_to_native(env.dof_pos, env.sim_indices)
    max_fk_error = 0.
    max_query_error = 0.
    for i in range(min(4, env.num_envs)):
        q = native[i].cpu().numpy()
        expected = fk.link_transforms_native(q)
        links = env.actual_link_poses()[i]
        expected = torch.tensor(np.stack([expected[k] for k in QUERY_LINKS]), device=env.device, dtype=links.dtype)
        max_fk_error = max(max_fk_error, (expected[:, :3, 3] - links[:, :3, 3]).norm(dim=-1).max().item())
        obj = pose_matrix(env.actor_root_state[env.object_indices[i].long(), :7])
        _, observed = actual_queries(native[i:i+1], links[None], obj[None])
        query = torch.from_numpy(fk.query_features_native(q, obj.cpu().numpy())).to(env.device)
        max_query_error = max(max_query_error, (observed[0] - query).abs().max().item())
    assert max_fk_error < 1e-4, f"Simulator/FK coordinate mismatch: {max_fk_error} m"
    assert max_query_error < .002, f"Actual observation query mismatch: {max_query_error}"
    other_root = env.actor_root_state.clone()
    other_q = env.dof_state.clone()
    env.reset_idx(torch.tensor([0], device=env.device))
    torch.testing.assert_close(env.dof_state[1:], other_q[1:])
    torch.testing.assert_close(env.actor_root_state[3:], other_root[3:])
    torch.testing.assert_close(env.actor_root_state[env.object_indices[0].long()], env.initial_root_states[env.object_indices[0].long()])
    torch.testing.assert_close(sim_to_native(env.dof_pos[0], env.sim_indices), env.initial_native)
    assert env.dof_vel[0].eq(0).all()
    env.compute_observations()
    before = env.base_wrist.clone()
    links = env.actual_link_poses().clone()
    links[:, :, 0, 3] += .01
    obj = pose_matrix(env.actor_root_state[env.object_indices.long(), :7])
    _, changed = env.base.predict(env.progress_buf, sim_to_native(env.dof_pos, env.sim_indices), links, obj)
    feedback_effect = (changed - before).abs().max().item()
    assert feedback_effect > 1e-6
    baseline = rollout(env, output)
    for key, value in env.base.model.state_dict().items():
        assert torch.equal(value, frozen_before[key]), f"Frozen tensor changed: {key}"
    assert all(not p.requires_grad for p in env.base.model.parameters())
    result = {"passed": True, "wrist_actuation_effect_m": target_effect,
              "max_fk_position_error_m": max_fk_error, "max_query_feature_error": max_query_error,
              "feedback_target_effect": feedback_effect, "reset_isolation": True,
              "frozen_weights_equal": True, "zero_residual_baseline": baseline,
              "checkpoint_sha256": env.base.manifest["checkpoint_sha256"], "code_sha256": code_hashes()}
    write_json(output / "gate.json", result)
    return result


class JsonObserver(RLGPUAlgoObserver):
    def __init__(self, output):
        super().__init__()
        self.path = output / "metrics.jsonl"

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        row = {"step": frame, "epoch": epoch_num, "elapsed_seconds": total_time}
        for k, v in self.direct_info.items():
            value = v.item() if torch.is_tensor(v) else v
            if isinstance(value, (int, float)) and np.isfinite(value):
                row[k] = value
        if self.algo.game_rewards.current_size:
            row["ppo_episode_reward_mean"] = self.algo.game_rewards.get_mean().mean().item()
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def run_policy(config, env, output, checkpoint=None):
    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: env})
    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    observer = JsonObserver(output)
    runner = Runner(observer)
    runner.load(config["train"])
    runner.reset()
    if checkpoint is not None:
        player = runner.create_player()
        player.restore(str(checkpoint))
        player.get_batch_size(env.obs_buf, env.num_envs)
        result = rollout(env, output, lambda obs: player.get_action(obs, is_deterministic=True))
        result.update(policy_checkpoint=str(checkpoint), policy_checkpoint_sha256=sha256(checkpoint))
        write_json(output / "evaluation.json", result)
        return result
    runner.run({"train": True, "play": False, "checkpoint": None, "sigma": None})
    algo = observer.algo
    algo.set_eval()
    @torch.inference_mode()
    def policy(obs):
        return algo.get_action_values({"obs": obs})["mus"].clamp(-1, 1)
    result = rollout(env, output, policy)
    result.update(last_epoch=int(algo.epoch_num), last_step=int(algo.frame),
                  checkpoints=[str(p.relative_to(output)) for p in sorted((output / "nn").glob("*.pth"))])
    write_json(output / "training_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate", "train", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output / "train.log").open("w", buffering=1)
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    config = make_config(args)
    if args.source_manifest is not None:
        config["task"]["basePolicy"]["sourceManifest"] = str(args.source_manifest.resolve())
    write_json(args.output / "config.json", config)
    manifest = {"manifest_schema": "ref2dex.run.v1", "task": "CmDecoderv2", "mode": args.mode,
                "run_id": args.output.name, "modification_version": "V1.1.15", "operation_category": ["experiment", "operation"],
                "created_at": datetime.now(timezone.utc).isoformat(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                "command": sys.argv, "seed": 42, "config_snapshot": "config.json", "metadata_snapshot": "metadata.json",
                "initial_checkpoint": config["task"]["basePolicy"]["decoderCheckpoint"],
                "policy_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "source_manifest_override": str(args.source_manifest.resolve()) if args.source_manifest else None,
                "gate": str(args.gate) if args.gate else None, "code_sha256": code_hashes(),
                "output_dir": str(args.output)}
    write_json(args.output / "run_manifest.json", manifest)
    write_json(args.output / "metadata.json", {"source_manifest": config["task"]["basePolicy"]["sourceManifest"],
               "action_dim": 12, "observation_dim": 71, "online_closed_loop": True, "sequence": "s1/airplane_lift", "split": "train", "pilot_only": True})
    if args.mode in {"train", "evaluate"}:
        if args.gate is None:
            raise ValueError("Training requires a passing physics gate")
        gate = json.loads(args.gate.read_text())
        if not gate["passed"] or gate["code_sha256"] != code_hashes():
            raise ValueError("Physics gate missing or code changed since gate")
        if gate["checkpoint_sha256"] != config["task"]["basePolicy"]["decoderCheckpointSha256"]:
            raise ValueError("Gate and training use different decoder checkpoints")
    if args.mode == "evaluate" and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise ValueError("Evaluation requires an explicit existing policy checkpoint")
    if args.mode == "train" and args.checkpoint is not None:
        raise ValueError("Use evaluate for a saved policy; implicit resume is not supported")
    env = create_env(config)
    started = time.monotonic()
    try:
        result = validate(env, args.output) if args.mode == "validate" else run_policy(config, env, args.output, args.checkpoint)
        print(json.dumps({"result": result, "elapsed_seconds": time.monotonic() - started}), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
