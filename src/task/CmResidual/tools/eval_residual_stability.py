"""Evaluate zero or checkpoint mean residuals under V1.8/V1.9/V1.10 protocols."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import sys
import traceback

from eval_zero_residual import (
    DEFAULT_DEXPLORE,
    DEFAULT_DEXPLORE_SOURCE,
    DEFAULT_OI_CM,
    REPOSITORY_ROOT,
    VENDOR_ROOT,
    Tee,
    _git,
    _input,
    _now,
    _scalar,
    _write_json,
)


DEFAULT_REFERENCE = REPOSITORY_ROOT / (
    "data/processed_data/cm_residual/reference_tracking_v2/"
    "s1_airplane_lift/reference.npz")
PROTOCOL_STEPS = 367
PROTOCOL_ENVS = 64
PRESERVATION_RATIO = 0.80
V19_VARIANTS = {
    "control": "CmResidualSafeCriticControlPPO",
    "critic_cm": "CmResidualSafeCriticCmPPO",
}
LEGACY_MODIFICATION_VERSIONS = {
    "v18_no_cm": "V1.8",
    "control": "V1.9.2",
    "critic_cm": "V1.9.2",
}
V110_MODIFICATION_VERSION = "V1.10.1"


def _resolve_modification_version(variant: str, requested: str = "") -> str:
    legacy = LEGACY_MODIFICATION_VERSIONS[variant]
    modification_version = requested or legacy
    allowed = {legacy}
    if variant in V19_VARIANTS:
        allowed.add(V110_MODIFICATION_VERSION)
    if modification_version not in allowed:
        raise ValueError(
            f"Unsupported modification version {modification_version!r} for {variant}; "
            f"expected one of {sorted(allowed)}")
    return modification_version


def _install_isaacgym_numpy_compat() -> None:
    """Restore the one NumPy alias used by the pinned Isaac Gym release."""
    import numpy as np

    if "float" not in np.__dict__:
        np.float = float


def preservation_gate(summary: dict, baseline: dict, ratio: float = PRESERVATION_RATIO) -> dict:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("preservation ratio must be in (0, 1]")
    lift_base = float(baseline["mean_env_max_lift_m"])
    contact_base = float(baseline["mean_contact_occupancy"])
    if lift_base <= 0.0 or contact_base <= 0.0:
        raise ValueError("baseline lift and contact must both be positive")
    lift_ratio = float(summary["mean_env_max_lift_m"]) / lift_base
    contact_ratio = float(summary["mean_contact_occupancy"]) / contact_base
    return {
        "threshold": ratio,
        "lift_ratio": lift_ratio,
        "contact_ratio": contact_ratio,
        "passed": lift_ratio >= ratio and contact_ratio >= ratio,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--activity-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument(
        "--variant", choices=("v18_no_cm", *V19_VARIANTS), default="v18_no_cm")
    parser.add_argument(
        "--modification-version", default="",
        help="Explicit run provenance version; V1.10 experiments must pass V1.10.1")
    parser.add_argument("--steps", type=int, default=PROTOCOL_STEPS)
    parser.add_argument("--num-envs", type=int, default=PROTOCOL_ENVS)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dexplore-checkpoint", type=Path, default=DEFAULT_DEXPLORE)
    parser.add_argument("--oi-cm-checkpoint", type=Path, default=DEFAULT_OI_CM)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--dexplore-source", type=Path, default=DEFAULT_DEXPLORE_SOURCE)
    parser.add_argument(
        "--isaac-gym-python", type=Path,
        default=Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python")))
    return parser.parse_args()


def _load_policy(checkpoint: Path, params: dict, device, observation_dim: int):
    import torch
    from isaacgymenvs.learning.cm_models import ModelCmContinuous
    from isaacgymenvs.learning.cm_network_builder import CmBuilder

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
        for key, value in payload["model"].items()
    }
    builder = CmBuilder()
    builder.load(params["network"])
    model = ModelCmContinuous(builder).build({
        "input_shape": (observation_dim,),
        "actions_num": 18,
        "num_seqs": 1,
        "value_size": 1,
        "normalize_input": bool(params["config"]["normalize_input"]),
        "normalize_value": bool(params["config"]["normalize_value"]),
    })
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload


def main() -> int:
    args = parse_args()
    is_v19 = args.variant in V19_VARIANTS
    modification_version = _resolve_modification_version(
        args.variant, args.modification_version)
    train_config = V19_VARIANTS.get(args.variant, "CmResidualSafePPO")
    use_oi_cm_context = bool(is_v19)
    observation_dim = 2005 if is_v19 else 1442
    if (args.steps != PROTOCOL_STEPS or args.num_envs != PROTOCOL_ENVS
            or args.gpu != 5 or args.seed != 42):
        raise ValueError(
            f"{modification_version} protocol is locked to GPU5, 367 steps, 64 envs, and seed 42")
    if bool(args.checkpoint) != bool(args.baseline_manifest):
        raise ValueError("checkpoint evaluation requires --baseline-manifest, and baseline uses neither")
    mode = "checkpoint_mean" if args.checkpoint else "zero_residual"
    run_label = "v18" if not is_v19 else f"v19_{args.variant}"
    run_id = args.run_id or f"cmresidual_{run_label}_{mode}_{datetime.now():%Y%m%d_%H%M%S}"
    output = (args.output or REPOSITORY_ROOT / "outputs/CmResidual" / run_id).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.mkdir(parents=True)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    log_path = output / "eval.log"

    dexplore = _input(args.dexplore_checkpoint, "dexplore_checkpoint")
    oi_cm = _input(args.oi_cm_checkpoint, "oi_cm_checkpoint") if is_v19 else None
    reference = _input(args.reference, "reference")
    source = _input(args.dexplore_source, "dexplore_source_tensor")
    checkpoint = _input(args.checkpoint, "residual_checkpoint") if args.checkpoint else None
    reference_manifest_path = Path(reference["path"]).with_name("manifest.json")
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    baseline_manifest = None
    if args.baseline_manifest:
        baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    protocol = {
        "physical_gpu": args.gpu,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "steps": args.steps,
        "terminate_on_success": True,
        "use_oi_cm_context": use_oi_cm_context,
        "observation_dim": observation_dim,
        "policy_family": "v19_critic_only_ablation" if is_v19 else "v18_no_cm",
        "deterministic_mean_action": True,
        "preservation_ratio": PRESERVATION_RATIO,
        "preservation_metrics": {
            "lift": "mean_env_max_lift_m",
            "contact": "mean_contact_occupancy",
        },
        "input_sha256": {
            "dexplore_checkpoint": dexplore["sha256"],
            **({"oi_cm_checkpoint": oi_cm["sha256"]} if oi_cm else {}),
            "reference": reference["sha256"],
            "dexplore_source_tensor": source["sha256"],
        },
        "reward": {"lift": 20.0, "tip_distance": 2.0, "action_penalty": 0.01},
        "residual_scale": {"translation_m": 0.015, "rotation_rad": 0.20,
                           "finger_rad": 0.08},
    }
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": _now(),
        "mode": f"{run_label}_stability_{mode}",
        "task": "CmResidual",
        "run_id": run_id,
        "activity_id": args.activity_id or (
            f"ACT-CMRESIDUAL-{modification_version.replace('.', '')}-"
            f"{args.variant.upper()}-STABILITY-EVAL"),
        "run_status": "STARTED",
        "modification_version": modification_version,
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output),
        "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "base_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "config_snapshot": str(config_path),
        "metadata_snapshot": str(reference_manifest_path.resolve()),
        "input_references": [dexplore] + ([oi_cm] if oi_cm else []) + [reference, source]
                            + ([checkpoint] if checkpoint else []),
        "reference_training_eligible": bool(reference_manifest.get("training_eligible", False)),
        "initial_checkpoint": checkpoint,
        "baseline_manifest": str(args.baseline_manifest.resolve()) if args.baseline_manifest else None,
        "metrics": str(metrics_path),
        "log": str(log_path),
        "protocol": protocol,
        "variant": args.variant,
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)

    environment_paths = [args.isaac_gym_python.resolve(), REPOSITORY_ROOT, VENDOR_ROOT]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["DEXPLORE_CHECKPOINT"] = dexplore["path"]
    if oi_cm:
        os.environ["OI_CM_CHECKPOINT"] = oi_cm["path"]
    os.environ["CMRESIDUAL_REFERENCE"] = reference["path"]
    os.environ["CMRESIDUAL_DEXPLORE_SOURCE"] = source["path"]
    for path in environment_paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    env = None
    rows = []
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
            with redirect_stdout(Tee(sys.stdout, log_stream)), redirect_stderr(Tee(sys.stderr, log_stream)):
                _install_isaacgym_numpy_compat()
                import isaacgym  # noqa: F401 - must precede torch
                import torch
                from hydra import compose, initialize_config_dir
                from omegaconf import OmegaConf
                import isaacgymenvs

                overrides = [
                    "task=CmResidual", f"train={train_config}",
                    f"task.basePolicy.useOiCmContext={'true' if use_oi_cm_context else 'false'}",
                    f"task.reference.path={args.reference.resolve()}",
                    f"task.env.numEnvs={args.num_envs}",
                    "task.env.terminateOnSuccess=true", "pipeline=gpu",
                    "sim_device=cuda:0", "rl_device=cuda:0", "graphics_device_id=0",
                    "headless=True", "force_render=False",
                    "task.sim.physx.max_gpu_contact_pairs=8388608",
                    "task.sim.physx.default_buffer_size_multiplier=5.0",
                    f"seed={args.seed}",
                ]
                with initialize_config_dir(
                        version_base="1.1", config_dir=str(VENDOR_ROOT / "isaacgymenvs/cfg")):
                    cfg = compose(config_name="config", overrides=overrides)
                resolved = OmegaConf.to_container(cfg, resolve=True)
                task_cfg = resolved["task"]
                resolved_contract = {
                    "input_sha256": {
                        "dexplore_checkpoint": task_cfg["basePolicy"]["checkpointSha256"],
                        **({
                            "oi_cm_checkpoint": task_cfg["basePolicy"]["oiCmCheckpointSha256"],
                        } if is_v19 else {}),
                        "reference": task_cfg["reference"]["sha256"],
                        "dexplore_source_tensor": task_cfg["reference"]["sourceTensorSha256"],
                    },
                    "reward": {
                        "lift": float(task_cfg["env"]["liftRewardScale"]),
                        "tip_distance": float(task_cfg["env"]["distanceRewardScale"]),
                        "action_penalty": float(task_cfg["env"]["actionPenaltyScale"]),
                    },
                    "residual_scale": {
                        "translation_m": float(task_cfg["residual"]["translationScaleM"]),
                        "rotation_rad": float(task_cfg["residual"]["rotationScaleRad"]),
                        "finger_rad": float(task_cfg["residual"]["fingerScaleRad"]),
                    },
                }
                expected_contract = {name: protocol[name] for name in (
                    "input_sha256", "reward", "residual_scale")}
                if resolved_contract != expected_contract:
                    raise RuntimeError(
                        f"{modification_version} evaluation contract drift: {resolved_contract} != {expected_contract}")
                _write_json(config_path, {"run_id": run_id, "resolved": resolved,
                                          "runtime_overrides": overrides, "protocol": protocol})
                env = isaacgymenvs.make(
                    seed=args.seed, task="CmResidual", num_envs=args.num_envs,
                    sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0,
                    headless=True, multi_gpu=False, virtual_screen_capture=False,
                    force_render=False, cfg=cfg)
                observation = env.reset()["obs"]
                if tuple(observation.shape) != (args.num_envs, observation_dim):
                    raise RuntimeError(
                        f"Unexpected {args.variant} observation shape: {tuple(observation.shape)}")
                policy = payload = None
                env_max_lift = torch.zeros(args.num_envs, device=env.rl_device)
                completed_episode_returns = []
                if args.checkpoint:
                    policy, payload = _load_policy(
                        args.checkpoint.resolve(), resolved["train"]["params"],
                        env.rl_device, observation_dim)
                    manifest["checkpoint_epoch"] = int(payload.get("epoch", -1))
                    _write_json(manifest_path, manifest)

                with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_stream:
                    for step in range(1, args.steps + 1):
                        if policy is None:
                            actions = torch.zeros((args.num_envs, 18), device=env.rl_device)
                        else:
                            with torch.no_grad():
                                actions = policy({"obs": observation, "is_train": False})["mus"].clamp(-1, 1)
                        observation_dict, reward, done, info = env.step(actions)
                        observation = observation_dict["obs"]
                        if "terminal_object_pose" in info:
                            object_z_before_reset = info["terminal_object_pose"][:, 2]
                        else:
                            object_z_before_reset = env.actor_root_state[
                                env.object_indices.long(), 2]
                        lift_by_env = object_z_before_reset - env.initial_object_z
                        env_max_lift = torch.maximum(env_max_lift, lift_by_env)
                        step_episode_returns = []
                        if "episode" in info and "return" in info["episode"]:
                            step_episode_returns = [
                                float(value) for value in
                                info["episode"]["return"].detach().cpu().flatten().tolist()
                            ]
                            completed_episode_returns.extend(step_episode_returns)
                        row = {
                            "step": step,
                            "reward_mean": _scalar(reward.mean()),
                            "lift_mean_m": _scalar(info["lift_mean"]),
                            "mean_env_max_lift_m": _scalar(env_max_lift.mean()),
                            "tip_distance_mean_m": _scalar(info["tip_distance_mean"]),
                            "contact_occupancy": _scalar(info["contact_occupancy"]),
                            "success_fraction": _scalar(info["success_fraction"]),
                            "success_rate": _scalar(info["success_rate"]),
                            "done_count": int(done.sum().item()),
                            "completed_episode_return_mean": (
                                sum(step_episode_returns) / len(step_episode_returns)
                                if step_episode_returns else None),
                            "residual_action_rms": _scalar(actions.square().mean().sqrt()),
                            "residual_target_delta_max": _scalar(info["residual_target_delta_max"]),
                            "observation_finite": bool(torch.isfinite(observation).all().item()),
                            "target_finite": bool(torch.isfinite(env.native_targets).all().item()),
                        }
                        if not row["observation_finite"] or not row["target_finite"]:
                            raise FloatingPointError(f"Non-finite rollout at step {step}")
                        metrics_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
    except BaseException as error:
        traceback.print_exc()
        manifest.update({
            "run_status": "FAILED", "completed_at": _now(), "last_step": len(rows),
            "exit_reason": f"{type(error).__name__}: {error}",
            "conclusion": "INVALID_IMPLEMENTATION", "scientific_conclusion": "INCONCLUSIVE",
        })
        _write_json(manifest_path, manifest)
        raise

    summary = {
        "steps": len(rows),
        "mean_lift_m": sum(row["lift_mean_m"] for row in rows) / len(rows),
        "max_lift_m": max(row["lift_mean_m"] for row in rows),
        "mean_env_max_lift_m": rows[-1]["mean_env_max_lift_m"],
        "mean_contact_occupancy": sum(row["contact_occupancy"] for row in rows) / len(rows),
        "mean_tip_distance_m": sum(row["tip_distance_mean_m"] for row in rows) / len(rows),
        "max_success_fraction": max(row["success_fraction"] for row in rows),
        "final_success_rate": rows[-1]["success_rate"],
        "completed_episode_count": len(completed_episode_returns),
        "mean_completed_episode_return": (
            sum(completed_episode_returns) / len(completed_episode_returns)
            if completed_episode_returns else 0.0),
        "max_residual_target_delta": max(row["residual_target_delta_max"] for row in rows),
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise FloatingPointError(f"Non-finite summary: {summary}")
    if baseline_manifest is None:
        zero_residual_parity = summary["max_residual_target_delta"] == 0.0
        positive_baseline = (
            summary["mean_env_max_lift_m"] > 0.0
            and summary["mean_contact_occupancy"] > 0.0)
        gate = {
            "zero_residual_parity": zero_residual_parity,
            "positive_preservation_baseline": positive_baseline,
            "passed": zero_residual_parity and positive_baseline,
        }
    else:
        if baseline_manifest.get("protocol") != protocol:
            raise ValueError("Baseline protocol does not match checkpoint evaluation protocol")
        gate = preservation_gate(summary, baseline_manifest["summary"])
    manifest.update({
        "run_status": "COMPLETED", "completed_at": _now(), "last_step": len(rows),
        "last_epoch": manifest.get("checkpoint_epoch"),
        "best_metric": {"name": "mean_env_max_lift_m",
                        "value": summary["mean_env_max_lift_m"]},
        "checkpoint": checkpoint["path"] if checkpoint else None,
        "exit_reason": f"Reached {modification_version} {args.variant} deterministic evaluation budget",
        "gate": gate, "gate_passed": bool(gate["passed"]),
        "conclusion": "SUPPORTED" if gate["passed"] else "REFUTED",
        "scientific_conclusion": "INCONCLUSIVE", "summary": summary,
    })
    _write_json(manifest_path, manifest)
    print(json.dumps({"run_id": run_id, "run_status": "COMPLETED", **summary, "gate": gate},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
