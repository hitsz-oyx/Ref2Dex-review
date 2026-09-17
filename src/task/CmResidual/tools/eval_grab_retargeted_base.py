"""Evaluate zero-residual GRAB retargeting with the object placed out of contact."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "third_party/IsaacGymEnvs"
REFERENCE = ROOT / "data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/reference.npz"


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--steps", type=int, default=366)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modification-version", default="V1.13.1")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--wrist-stiffness", type=float, default=200.0)
    parser.add_argument("--wrist-damping", type=float, default=20.0)
    parser.add_argument("--object-protocol", choices=("out_of_contact", "physical"),
                        default="out_of_contact")
    args = parser.parse_args()
    if not 1 <= args.steps <= 366:
        raise ValueError("steps must be in [1, 366]")
    if (args.wrist_stiffness, args.wrist_damping) not in ((200.0, 20.0), (300.0, 30.0), (400.0, 40.0)):
        raise ValueError("V1.13 gain Gate only permits its three predeclared pairs")
    output = ROOT / "outputs/CmResidual" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    log_path = output / "eval.log"
    config_path = output / "config.json"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    reference_manifest = REFERENCE.with_name("manifest.json")
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1", "task": "CmResidual",
        "operation": "grab_retargeted_zero_residual_" + args.object_protocol,
        "run_id": args.run_id,
        "activity_id": args.activity_id, "run_status": "STARTED", "created_at": stamp(),
        "modification_version": args.modification_version,
        "operation_category": ["experiment", "operation"],
        "base_commit": commit,
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "command": [sys.executable, *sys.argv], "seed": args.seed,
        "physical_gpu": args.gpu, "wrist_stiffness": args.wrist_stiffness,
        "wrist_damping": args.wrist_damping, "object_protocol": args.object_protocol,
        "initial_checkpoint": None, "config_snapshot": str(config_path.resolve()),
        "metadata_snapshot": str(reference_manifest.resolve()),
        "input_references": [{"path": str(REFERENCE.resolve()), "sha256": digest(REFERENCE)},
                             {"path": str(reference_manifest.resolve()), "sha256": digest(reference_manifest)}],
        "output_dir": str(output.resolve()), "metrics": str(metrics_path.resolve()),
        "log": str(log_path.resolve()), "conclusion": "INCONCLUSIVE",
    }
    write(manifest_path, manifest)
    log_path.write_text("", encoding="utf-8")
    try:
        # Isaac Gym must be imported before torch.
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        sys.path[:0] = [str(VENDOR), str(ROOT)]
        import isaacgym  # noqa: F401
        import torch
        from isaacgym import gymtorch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        import isaacgymenvs
        overrides = [
                "task=CmResidualGrabRetargeted", "train=CmResidualPPO", "task.env.numEnvs=1",
                "headless=True", "task.env.terminateOnSuccess=false",
                f"task.env.wristStiffness={args.wrist_stiffness}",
                f"task.env.wristDamping={args.wrist_damping}", f"seed={args.seed}"]
        if args.gpu is None:
            overrides += ["pipeline=cpu", "sim_device=cpu", "rl_device=cpu",
                          "graphics_device_id=-1", "num_subscenes=1"]
            device = "cpu"
            graphics_device = -1
        else:
            overrides += ["pipeline=gpu", "sim_device=cuda:0", "rl_device=cuda:0",
                          "graphics_device_id=0", "num_subscenes=4",
                          "task.sim.physx.max_gpu_contact_pairs=8388608",
                          "task.sim.physx.default_buffer_size_multiplier=5.0"]
            device = "cuda:0"
            graphics_device = 0
        with initialize_config_dir(version_base="1.1", config_dir=str(VENDOR / "isaacgymenvs/cfg")):
            cfg = compose(config_name="config", overrides=overrides)
        write(config_path, {"task": OmegaConf.to_container(cfg.task, resolve=True),
                            "steps": args.steps, "object_protocol": args.object_protocol,
                            "overrides": overrides,
                            "residual_action": "all zeros"})
        env = isaacgymenvs.make(
            seed=args.seed, task="CmResidualGrabRetargeted", num_envs=1,
            sim_device=device, rl_device=device, graphics_device_id=graphics_device,
            headless=True, multi_gpu=False, virtual_screen_capture=False,
            force_render=False, cfg=cfg)
        env.reset()
        if args.object_protocol == "out_of_contact":
            object_ids = env.object_indices.contiguous()
            env.actor_root_state[object_ids.long(), 0] += 10.0
            env.actor_root_state[object_ids.long(), 7:13] = 0
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env.actor_root_state),
                gymtorch.unwrap_tensor(object_ids), len(object_ids))
            env.compute_observations()
        action = torch.zeros((1, 18), device=env.rl_device)
        rows = []
        with metrics_path.open("w", encoding="utf-8") as stream:
            for step in range(1, args.steps + 1):
                obs, _, done, info = env.step(action)
                terminal = bool(done[0])
                native = (info["terminal_observation"][:, :18] if terminal else
                          env.dof_pos.index_select(-1, env.sim_indices))
                target = env.reference.robot_q[step:step + 1]
                observed = info["terminal_observation"] if terminal else obs["obs"]
                actual_wrist = (info["terminal_wrist_pose"] if terminal else
                                env.actual_link_poses()[:, 0])
                reference_wrist = env.reference.wrist_pose[step:step + 1]
                row = {"step": step, "reference_index": step,
                       "finite": bool(torch.isfinite(observed).all()),
                       "actual_wrist_position_m": actual_wrist[0, :3, 3].detach().cpu().tolist(),
                       "reference_wrist_position_m": reference_wrist[0, :3, 3].detach().cpu().tolist(),
                       "wrist_error_m": float(info["reference_root_position_error_m"]),
                       "finger_q_error_rad": float((native[:, 6:] - target[:, 6:]).abs().mean()),
                       "tip_error_m": float(info["reference_tip_position_error_m"]),
                       "tip_distance_mean_m": float(info["tip_distance_mean"]),
                       "lift_mean_m": float(info["lift_mean"]),
                       "object_position_error_m": float(info["reference_object_position_error_m"]),
                       "contact_occupancy": float(info["contact_occupancy"]),
                       "residual_saturation_ratio": float(info["residual_saturation_ratio"]),
                       "residual_target_delta_max": float(info["residual_target_delta_max"]),
                       "done_count": int(done.sum())}
                if not row["finite"] or (args.object_protocol == "out_of_contact" and
                                          row["contact_occupancy"] != 0):
                    raise RuntimeError(f"Evaluation gate failed at step {step}: {row}")
                stream.write(json.dumps(row) + "\n")
                rows.append(row)
        summary = {key: max(row[key] for row in rows) for key in
                   ("wrist_error_m", "finger_q_error_rad", "tip_error_m", "contact_occupancy",
                    "residual_target_delta_max")}
        summary["steps"] = len(rows)
        manifest.update(run_status="COMPLETED", ended_at=stamp(), last_step=len(rows),
                        summary=summary, conclusion="INCONCLUSIVE",
                        engineering_wiring="SUPPORTED", tracking_gate="UNSET")
        write(manifest_path, manifest)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False))
    except Exception as exc:
        manifest.update(run_status="FAILED", ended_at=stamp(), failure_reason=str(exc),
                        conclusion="INVALID_IMPLEMENTATION")
        write(manifest_path, manifest)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
