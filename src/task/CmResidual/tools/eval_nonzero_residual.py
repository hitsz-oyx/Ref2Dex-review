"""Validate deterministic non-zero residuals against paired zero controls."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shlex
import sys
import traceback

from eval_zero_residual import (
    DEFAULT_DEXPLORE,
    DEFAULT_DEXPLORE_SOURCE,
    DEFAULT_OI_CM,
    DEFAULT_REFERENCE,
    REPOSITORY_ROOT,
    VENDOR_ROOT,
    Tee,
    _git,
    _input,
    _now,
    _scalar,
    _write_json,
)


ACTION_VALUE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--activity-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dexplore-checkpoint", type=Path, default=DEFAULT_DEXPLORE)
    parser.add_argument("--oi-cm-checkpoint", type=Path, default=DEFAULT_OI_CM)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--dexplore-source", type=Path, default=DEFAULT_DEXPLORE_SOURCE)
    parser.add_argument(
        "--isaac-gym-python", type=Path,
        default=Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python")))
    return parser.parse_args()


def _fixed_actions(torch, device):
    actions = torch.zeros((4, 18), dtype=torch.float32, device=device)
    actions[2] = ACTION_VALUE
    actions[3] = -ACTION_VALUE
    return actions


def _expected_requested(torch, actions, env, independent_indices):
    expected = torch.zeros_like(actions)
    expected[:, :3] = actions[:, :3] * env.residual_translation_scale
    expected[:, 3:6] = actions[:, 3:6] * env.residual_rotation_scale
    expected[:, list(independent_indices)] = (
        actions[:, list(independent_indices)] * env.residual_finger_scale)
    return expected


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    run_id = args.run_id or f"cmresidual_nonzero_v15_{_now().replace(':', '').replace('-', '')}"
    output = (args.output or REPOSITORY_ROOT / "outputs/CmResidual" / run_id).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    output.mkdir(parents=True)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    log_path = output / "eval.log"

    dexplore = _input(args.dexplore_checkpoint, "dexplore_checkpoint")
    oi_cm = _input(args.oi_cm_checkpoint, "oi_cm_checkpoint")
    reference = _input(args.reference, "reference")
    source = _input(args.dexplore_source, "dexplore_source_tensor")
    reference_manifest_path = Path(reference["path"]).with_name("manifest.json")
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    activity_id = args.activity_id or "ACT-CMRESIDUAL-V151-NONZERO"
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": _now(),
        "mode": "eval_nonzero_residual",
        "task": "CmResidual",
        "run_id": run_id,
        "activity_id": activity_id,
        "run_status": "STARTED",
        "modification_version": "V1.5.1",
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output),
        "command": command,
        "seed": args.seed,
        "base_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "config_snapshot": str(config_path),
        "metadata_snapshot": str(reference_manifest_path.resolve()),
        "input_references": [dexplore, oi_cm, reference, source],
        "reference_training_eligible": bool(reference_manifest.get("training_eligible", False)),
        "reference_allow_ineligible_for": "diagnostic",
        "initial_checkpoint": None,
        "checkpoint": None,
        "metrics": str(metrics_path),
        "log": str(log_path),
        "sim_cleanup": "process_exit_cleanup",
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)

    os.environ["DEXPLORE_CHECKPOINT"] = dexplore["path"]
    os.environ["OI_CM_CHECKPOINT"] = oi_cm["path"]
    os.environ["CMRESIDUAL_REFERENCE"] = reference["path"]
    os.environ["CMRESIDUAL_DEXPLORE_SOURCE"] = source["path"]
    for path in (args.isaac_gym_python.resolve(), REPOSITORY_ROOT, VENDOR_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    env = None
    rows = []
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
            with redirect_stdout(Tee(sys.stdout, log_stream)), redirect_stderr(Tee(sys.stderr, log_stream)):
                print(f"run_id={run_id} run_status=STARTED")
                import isaacgym  # noqa: F401 - must precede torch
                import torch
                from hydra import compose, initialize_config_dir
                from omegaconf import OmegaConf
                import isaacgymenvs
                from isaacgymenvs.tasks.cm_residual.contract import (
                    INDEPENDENT_NATIVE, MIMIC_NATIVE, MIMIC_SCALE, MIMIC_SOURCE)

                with initialize_config_dir(
                        version_base="1.1", config_dir=str(VENDOR_ROOT / "isaacgymenvs/cfg")):
                    cfg = compose(config_name="config", overrides=[
                        "task=CmResidual",
                        "task.env.numEnvs=4",
                        "pipeline=cpu",
                        "sim_device=cpu",
                        "rl_device=cpu",
                        "graphics_device_id=-1",
                        "headless=True",
                        "force_render=False",
                        "num_subscenes=1",
                        "task.env.terminateOnSuccess=false",
                        "task.reference.allowIneligibleFor=diagnostic",
                        f"seed={args.seed}",
                    ])
                resolved_task = OmegaConf.to_container(cfg.task, resolve=True)
                _write_json(config_path, {
                    "run_id": run_id,
                    "task": resolved_task,
                    "steps": args.steps,
                    "num_envs": 4,
                    "seed": args.seed,
                    "actions": {"env_0": 0.0, "env_1": 0.0,
                                "env_2": ACTION_VALUE, "env_3": -ACTION_VALUE},
                })
                env = isaacgymenvs.make(
                    seed=args.seed, task="CmResidual", num_envs=4,
                    sim_device="cpu", rl_device="cpu", graphics_device_id=-1,
                    headless=True, multi_gpu=False, virtual_screen_capture=False,
                    force_render=False, cfg=cfg)
                observation = env.reset()["obs"]
                actions = _fixed_actions(torch, env.rl_device)
                expected_requested = _expected_requested(
                    torch, actions, env, INDEPENDENT_NATIVE)
                if tuple(observation.shape) != (4, 2005):
                    raise RuntimeError(f"Unexpected observation shape: {tuple(observation.shape)}")

                # Directly exercise indexed reset isolation.  Comparing two
                # independent contact trajectories is not a valid isolation
                # test because parallel PhysX solves may diverge numerically.
                untouched = torch.tensor([0, 1, 2], device=env.device, dtype=torch.long)
                root_ids = torch.cat((
                    env.hand_indices[untouched], env.object_indices[untouched],
                    env.table_indices[untouched])).long()
                isolation_before = (
                    env.dof_state[untouched].clone(),
                    env.actor_root_state[root_ids].clone(),
                    env.native_targets[untouched].clone(),
                    env.prev_actions[untouched].clone(),
                    env.progress_buf[untouched].clone(),
                    env.reference_index[untouched].clone(),
                )
                env.reset_idx(torch.tensor([3], device=env.device, dtype=torch.long))
                isolation_after = (
                    env.dof_state[untouched],
                    env.actor_root_state[root_ids],
                    env.native_targets[untouched],
                    env.prev_actions[untouched],
                    env.progress_buf[untouched],
                    env.reference_index[untouched],
                )
                reset_isolation_error = max(
                    _scalar((before - after).abs().amax())
                    for before, after in zip(isolation_before, isolation_after))
                env.compute_observations()

                with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_stream:
                    for step in range(1, args.steps + 1):
                        observation_dict, reward, done, info = env.step(actions)
                        observation = observation_dict["obs"]
                        targets = env.native_targets
                        zero_pair_state_error = (env.dof_state[0] - env.dof_state[1]).abs().amax()
                        zero_pair_target_error = (targets[0] - targets[1]).abs().amax()
                        objects = env.actor_root_state[env.object_indices.long()]
                        mimic_error = torch.stack([
                            (targets[:, mimic] - targets[:, INDEPENDENT_NATIVE[source]] * scale).abs().amax()
                            for mimic, source, scale in zip(
                                MIMIC_NATIVE, MIMIC_SOURCE, MIMIC_SCALE)
                        ]).amax()
                        lower_error = (env.native_lower - targets[2:]).clamp_min(0).amax()
                        upper_error = (targets[2:] - env.native_upper).clamp_min(0).amax()
                        row = {
                            "step": step,
                            "reward_mean": _scalar(reward.mean()),
                            "done_count": int(done.sum().item()),
                            "observation_finite": bool(torch.isfinite(observation).all().item()),
                            "reward_finite": bool(torch.isfinite(reward).all().item()),
                            "target_finite": bool(torch.isfinite(targets).all().item()),
                            "object_pose_finite": bool(torch.isfinite(objects[:, :7]).all().item()),
                            "requested_delta_error_max": _scalar(
                                (env.residual_requested_delta - expected_requested).abs().amax()),
                            "ignored_mimic_requested_abs_max": _scalar(
                                env.residual_requested_delta[:, list(MIMIC_NATIVE)].abs().amax()),
                            "zero_target_delta_max": _scalar(
                                env.residual_applied_delta[:2].abs().amax()),
                            "nonzero_applied_delta_min": _scalar(
                                env.residual_applied_delta[2:].abs().amax(dim=1).min()),
                            "nonzero_applied_delta_max": _scalar(
                                env.residual_applied_delta[2:].abs().amax()),
                            "saturation_ratio": _scalar(env.residual_saturation[2:].mean()),
                            "mimic_error_max": _scalar(mimic_error),
                            "joint_limit_error_max": _scalar(torch.maximum(lower_error, upper_error)),
                            "reset_isolation_error_max": reset_isolation_error,
                            "zero_pair_state_error_max": _scalar(zero_pair_state_error),
                            "zero_pair_target_error_max": _scalar(zero_pair_target_error),
                            "zero_pair_object_pose_error_max": _scalar(
                                (objects[0, :7] - objects[1, :7]).abs().amax()),
                            "signed_response_separation_max": _scalar(
                                (env.dof_state[2, :, 0] - env.dof_state[3, :, 0]).abs().amax()),
                            "contact_occupancy": _scalar(info["contact_occupancy"]),
                            "lift_mean_m": _scalar(info["lift_mean"]),
                        }
                        metrics_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
                        if not all(row[name] for name in (
                                "observation_finite", "reward_finite", "target_finite",
                                "object_pose_finite")):
                            raise FloatingPointError(f"Non-finite gate value at step {step}")

                maxima = {
                    name: max(row[name] for row in rows)
                    for name in (
                        "requested_delta_error_max", "ignored_mimic_requested_abs_max",
                        "zero_target_delta_max", "saturation_ratio", "mimic_error_max",
                        "joint_limit_error_max", "zero_pair_state_error_max",
                        "reset_isolation_error_max", "zero_pair_target_error_max",
                        "zero_pair_object_pose_error_max")
                }
                summary = {
                    "steps": len(rows),
                    **maxima,
                    "min_nonzero_applied_delta": min(
                        row["nonzero_applied_delta_min"] for row in rows),
                    "max_nonzero_applied_delta": max(
                        row["nonzero_applied_delta_max"] for row in rows),
                    "final_signed_response_separation": rows[-1]["signed_response_separation_max"],
                    "reset_count": sum(row["done_count"] for row in rows),
                }
                gate_passed = (
                    summary["requested_delta_error_max"] <= 1e-7
                    and summary["ignored_mimic_requested_abs_max"] == 0.0
                    and summary["zero_target_delta_max"] == 0.0
                    and summary["mimic_error_max"] <= 1e-6
                    and summary["joint_limit_error_max"] <= 1e-7
                    and summary["reset_isolation_error_max"] == 0.0
                    and summary["min_nonzero_applied_delta"] > 0.0
                    and summary["final_signed_response_separation"] > 1e-6
                    and summary["reset_count"] == 0)
                manifest.update({
                    "run_status": "COMPLETED",
                    "completed_at": _now(),
                    "last_step": len(rows),
                    "last_epoch": None,
                    "best_metric": {
                        "name": "final_signed_response_separation",
                        "value": summary["final_signed_response_separation"],
                    },
                    "exit_reason": "Reached configured deterministic non-zero residual budget",
                    "gate_passed": gate_passed,
                    "conclusion": "SUPPORTED" if gate_passed else "INVALID_IMPLEMENTATION",
                    "scientific_conclusion": "INCONCLUSIVE",
                    "summary": summary,
                })
                _write_json(manifest_path, manifest)
                print(json.dumps({"run_id": run_id, "run_status": "COMPLETED",
                                  "gate_passed": gate_passed, **summary}, ensure_ascii=False))
                if not gate_passed:
                    raise RuntimeError(f"Non-zero residual gate failed: {summary}")
    except BaseException as error:
        if manifest.get("run_status") != "COMPLETED":
            traceback.print_exc()
            manifest.update({
                "run_status": "FAILED",
                "completed_at": _now(),
                "last_step": len(rows),
                "last_epoch": None,
                "best_metric": None,
                "checkpoint": None,
                "exit_reason": f"{type(error).__name__}: {error}",
                "conclusion": "INVALID_IMPLEMENTATION",
                "scientific_conclusion": "INCONCLUSIVE",
            })
            _write_json(manifest_path, manifest)
        raise
    finally:
        # Explicit destroy_sim is intentionally avoided because this Isaac Gym
        # build can block there; the isolated process owns and reclaims the sim.
        env = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
