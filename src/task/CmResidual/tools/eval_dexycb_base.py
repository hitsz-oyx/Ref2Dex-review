"""Evaluate frozen DExplore with strict zero residual on the V1.11 DexYCB scene."""
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
    DEFAULT_DEXPLORE, REPOSITORY_ROOT, VENDOR_ROOT, Tee, _git, _input, _now,
    _scalar, _write_json,
)


DEFAULT_ROOT = REPOSITORY_ROOT / (
    "data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--activity-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dexplore-checkpoint", type=Path, default=DEFAULT_DEXPLORE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_ROOT / "reference.npz")
    parser.add_argument("--dexplore-source", type=Path, default=DEFAULT_ROOT / "interaction_hand_inspire.pt")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT / "assets")
    parser.add_argument(
        "--isaac-gym-python", type=Path,
        default=Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.gpu, args.seed, args.num_envs, args.steps) != (5, 42, 4, 72):
        raise ValueError("V1.11.2 repair protocol is locked to GPU5, seed42, 4 envs, and 72 steps")
    if not (args.asset_root / "master_chef_can.urdf").is_file():
        raise FileNotFoundError(f"Missing generated asset bundle: {args.asset_root}")
    run_id = args.run_id or f"cmresidual_v111_dexycb_base_{datetime.now():%Y%m%d_%H%M%S}"
    output = (args.output or REPOSITORY_ROOT / "outputs/CmResidual" / run_id).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    log_path = output / "eval.log"

    dexplore = _input(args.dexplore_checkpoint, "dexplore_checkpoint")
    reference = _input(args.reference, "dexycb_reference")
    source = _input(args.dexplore_source, "dexplore_compatibility_source")
    metadata_path = args.reference.with_name("manifest.json").resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("evaluation_eligible") or metadata.get("training_eligible"):
        raise ValueError("DexYCB artifact must be evaluation-only and evaluation_eligible")
    protocol = {
        "physical_gpu": 5, "seed": 42, "num_envs": 4, "steps": 72,
        "task": "CmResidualDexYCBBase", "use_oi_cm_context": False,
        "observation_dim": 1442, "residual_action": "exact_zero",
        "terminate_on_success": False,
        "sequence_id": "subject-10/20201022_110806",
        "object_name": "002_master_chef_can",
    }
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": _now(),
        "mode": "eval_dexycb_frozen_dexplore_zero_residual",
        "task": "CmResidual",
        "run_id": run_id,
        "activity_id": args.activity_id or f"ACT-{datetime.now():%Y%m%d-%H%M%S}-CMRESIDUAL-V1111-DEXYCB",
        "run_status": "STARTED",
        "modification_version": "V1.11.2",
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output),
        "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "base_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "config_snapshot": str(config_path),
        "metadata_snapshot": str(metadata_path),
        "input_references": [dexplore, reference, source],
        "reference_training_eligible": False,
        "reference_evaluation_eligible": True,
        "initial_checkpoint": None,
        "metrics": str(metrics_path),
        "log": str(log_path),
        "protocol": protocol,
        "conclusion": "INCONCLUSIVE",
        "scientific_conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["DEXPLORE_CHECKPOINT"] = dexplore["path"]
    os.environ["CMRESIDUAL_DEXYCB_REFERENCE"] = reference["path"]
    os.environ["CMRESIDUAL_DEXYCB_REFERENCE_SHA256"] = reference["sha256"]
    os.environ["CMRESIDUAL_DEXYCB_SOURCE"] = source["path"]
    os.environ["CMRESIDUAL_DEXYCB_SOURCE_SHA256"] = source["sha256"]
    os.environ["CMRESIDUAL_DEXYCB_ASSET_ROOT"] = str(args.asset_root.resolve())
    for path in (args.isaac_gym_python.resolve(), REPOSITORY_ROOT, VENDOR_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    env = None
    rows: list[dict] = []
    initial_alignment: dict[str, float] = {}
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
            with redirect_stdout(Tee(sys.stdout, log_stream)), redirect_stderr(Tee(sys.stderr, log_stream)):
                import numpy as np
                if "float" not in np.__dict__:
                    np.float = float
                import isaacgym  # noqa: F401
                import torch
                from hydra import compose, initialize_config_dir
                from omegaconf import OmegaConf
                import isaacgymenvs

                overrides = [
                    "task=CmResidualDexYCBBase", "train=CmResidualSafePPO",
                    f"task.env.numEnvs={args.num_envs}",
                    "pipeline=gpu", "sim_device=cuda:0", "rl_device=cuda:0",
                    "graphics_device_id=0", "headless=True", "force_render=False",
                    "task.sim.physx.max_gpu_contact_pairs=8388608",
                    "task.sim.physx.default_buffer_size_multiplier=5.0",
                    f"seed={args.seed}",
                ]
                with initialize_config_dir(version_base="1.1", config_dir=str(VENDOR_ROOT / "isaacgymenvs/cfg")):
                    cfg = compose(config_name="config", overrides=overrides)
                resolved = OmegaConf.to_container(cfg, resolve=True)
                task_cfg = resolved["task"]
                if task_cfg["basePolicy"]["useOiCmContext"] or task_cfg["env"]["terminateOnSuccess"]:
                    raise RuntimeError("DexYCB base protocol drifted from no-Cm/non-terminating contract")
                if task_cfg["reference"]["sha256"] != reference["sha256"] or task_cfg["reference"]["sourceTensorSha256"] != source["sha256"]:
                    raise RuntimeError("Resolved input checksums differ from evaluated files")
                _write_json(config_path, {"run_id": run_id, "resolved": resolved,
                                          "runtime_overrides": overrides, "protocol": protocol})
                env = isaacgymenvs.make(
                    seed=args.seed, task="CmResidualDexYCBBase", num_envs=args.num_envs,
                    sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0,
                    headless=True, multi_gpu=False, virtual_screen_capture=False,
                    force_render=False, cfg=cfg)
                observation = env.reset()["obs"]
                if tuple(observation.shape) != (args.num_envs, 1442):
                    raise RuntimeError(f"Unexpected observation shape {tuple(observation.shape)}")
                initial_alignment = {
                    "native_q_abs_max": _scalar(env.reset_native_error_max.amax()),
                    "wrist_position_m": _scalar(env.reset_wrist_position_error_m.amax()),
                    "object_position_m": _scalar(env.reset_object_position_error_m.amax()),
                    "table_position_m": _scalar(env.reset_table_position_error_m.amax()),
                }
                manifest["initial_alignment_error"] = initial_alignment
                _write_json(manifest_path, manifest)
                actions = torch.zeros((args.num_envs, 18), dtype=torch.float32, device=env.rl_device)
                with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_stream:
                    for step in range(1, args.steps + 1):
                        observation_dict, reward, done, info = env.step(actions)
                        observation = observation_dict["obs"]
                        row = {
                            "step": step,
                            "reward_mean": _scalar(reward.mean()),
                            "lift_mean_m": _scalar(info["lift_mean"]),
                            "tip_distance_mean_m": _scalar(info["tip_distance_mean"]),
                            "contact_occupancy": _scalar(info["contact_occupancy"]),
                            "reference_contact_occupancy": _scalar(info["reference_contact_occupancy"]),
                            "reference_root_position_error_m": _scalar(info["reference_root_position_error_m"]),
                            "reference_object_position_error_m": _scalar(info["reference_object_position_error_m"]),
                            "reference_tip_position_error_m": _scalar(info["reference_tip_position_error_m"]),
                            "done_count": int(done.sum().item()),
                            "base_action_abs_max": _scalar(env.base_action.abs().amax()),
                            "residual_action_abs_max": _scalar(actions.abs().amax()),
                            "residual_target_delta_max": _scalar(info["residual_target_delta_max"]),
                            "observation_finite": bool(torch.isfinite(observation).all().item()),
                            "target_finite": bool(torch.isfinite(env.native_targets).all().item()),
                            "object_pose_finite": bool(torch.isfinite(
                                env.actor_root_state[env.object_indices.long(), :7]).all().item()),
                        }
                        if not all(row[name] for name in ("observation_finite", "target_finite", "object_pose_finite")):
                            raise FloatingPointError(f"Non-finite rollout state at step {step}")
                        if row["residual_action_abs_max"] != 0.0 or row["residual_target_delta_max"] != 0.0:
                            raise RuntimeError(f"Zero-residual parity failed at step {step}")
                        metrics_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
    except BaseException as error:
        traceback.print_exc()
        manifest.update({
            "run_status": "FAILED", "completed_at": _now(), "last_step": len(rows),
            "last_epoch": None, "best_metric": None, "checkpoint": None,
            "exit_reason": f"{type(error).__name__}: {error}",
            "conclusion": "INVALID_IMPLEMENTATION", "scientific_conclusion": "INCONCLUSIVE",
        })
        _write_json(manifest_path, manifest)
        raise

    summary = {
        "steps": len(rows),
        "mean_lift_m": sum(row["lift_mean_m"] for row in rows) / len(rows),
        "max_lift_m": max(row["lift_mean_m"] for row in rows),
        "mean_tip_distance_m": sum(row["tip_distance_mean_m"] for row in rows) / len(rows),
        "min_tip_distance_m": min(row["tip_distance_mean_m"] for row in rows),
        "mean_contact_occupancy": sum(row["contact_occupancy"] for row in rows) / len(rows),
        "mean_reference_contact_occupancy": sum(row["reference_contact_occupancy"] for row in rows) / len(rows),
        "mean_reference_root_position_error_m": sum(row["reference_root_position_error_m"] for row in rows) / len(rows),
        "max_reference_root_position_error_m": max(row["reference_root_position_error_m"] for row in rows),
        "mean_reference_object_position_error_m": sum(row["reference_object_position_error_m"] for row in rows) / len(rows),
        "max_reference_object_position_error_m": max(row["reference_object_position_error_m"] for row in rows),
        "mean_reference_tip_position_error_m": sum(row["reference_tip_position_error_m"] for row in rows) / len(rows),
        "max_reference_tip_position_error_m": max(row["reference_tip_position_error_m"] for row in rows),
        "reset_count": sum(row["done_count"] for row in rows),
        "max_residual_target_delta": max(row["residual_target_delta_max"] for row in rows),
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise FloatingPointError(f"Non-finite summary: {summary}")
    gate = {
        "completed_budget": len(rows) == args.steps,
        "initial_alignment": max(initial_alignment.values()) <= 1e-5,
        "zero_residual_parity": summary["max_residual_target_delta"] == 0.0,
        "finite": True,
    }
    gate["passed"] = all(gate.values())
    manifest.update({
        "run_status": "COMPLETED", "completed_at": _now(), "last_step": len(rows),
        "last_epoch": None,
        "best_metric": {"name": "min_tip_distance_m", "value": summary["min_tip_distance_m"]},
        "checkpoint": None,
        "exit_reason": "Reached approved 72-step DexYCB base-only evaluation budget",
        "gate": gate, "gate_passed": gate["passed"],
        "conclusion": "SUPPORTED" if gate["passed"] else "INVALID_IMPLEMENTATION",
        "scientific_conclusion": "INCONCLUSIVE", "summary": summary,
    })
    _write_json(manifest_path, manifest)
    print(json.dumps({"run_id": run_id, "run_status": "COMPLETED", "gate": gate,
                      "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
