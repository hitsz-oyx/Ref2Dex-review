"""Launch one auditable V1.18 reference PPO stage or implementation smoke."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from eval_zero_residual import _git, _input, _now, _write_json, REPOSITORY_ROOT, VENDOR_ROOT
from run_ppo_stability import _resolved_config


REFERENCE = REPOSITORY_ROOT / "data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/reference.npz"
SOURCE = REPOSITORY_ROOT / "data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/s1_airplane_lift/interaction_hand_inspire.pt"
REFERENCE_SHA = "a2d710b911cf8988750208450c411b3095e187ed2b2f05df459c24d5748812c1"
SOURCE_SHA = "19b110dc81c4928b4f3e6197d8549b011fd48a8157e75fd06668bd306dc396cf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--run-id", default=f"cmresidual_v118_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--work-version", default="V1.18")
    parser.add_argument("--stage", choices=("smoke", "a", "b", "c128", "c366"), required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Explicit smoke env count; defaults to 1 for V1.14a and 128 otherwise.")
    parser.add_argument("--checkpoint", default="", help="Pinned Stage-A checkpoint required for B/C.")
    parser.add_argument("--checkpoint-sha256", default="",
                        help="Required actor checkpoint digest for V1.14a Stage-B pilots.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--cmv2-v114", action="store_true",
                        help="Use the opt-in V1.14a shared-candidate adapter.")
    parser.add_argument("--cmv2-v114-checkpoint", default="",
                        help="Explicit V1.14a checkpoint; required for Stage B.")
    parser.add_argument("--cmv2-v114-sha256", default="",
                        help="Explicit V1.14a checkpoint digest; required for Stage B.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _stage_settings(stage: str) -> tuple[int, int, float]:
    if stage == "smoke":
        return 64, 1, 0.10
    if stage == "a":
        return 64, 50_000_000, 0.0
    if stage == "b":
        return 64, 50_000_000, 0.10
    if stage == "c128":
        return 128, 100_000_000, 0.10
    return 366, 100_000_000, 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_v114_checkpoint(path: Path, expected_sha256: str, *, require_inspire: bool) -> dict:
    import torch
    if not path.is_file() or len(expected_sha256) != 64 or _sha256(path) != expected_sha256:
        raise ValueError("V1.14a checkpoint path/SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_name") != "object_interaction_cmv2_shared_start_candidate_checkpoint_v2":
        raise ValueError("V1.14a checkpoint schema mismatch")
    groups = tuple(payload.get("active_groups") or ())
    if require_inspire and "grab/inspire_f1" not in groups:
        raise ValueError("Stage-B V1.14a checkpoint must include grab/inspire_f1")
    return {"path": str(path), "sha256": expected_sha256, "active_groups": list(groups),
            "run_id": payload.get("run_id"), "step": payload.get("step"), "epoch": payload.get("epoch")}


def main() -> None:
    args = parse_args()
    if args.gpu < 0:
        raise ValueError("--gpu must be a non-negative physical CUDA device index")
    if args.cmv2_v114 and args.stage not in ("smoke", "b"):
        raise ValueError("V1.14a Cmv2 integration is authorized only for smoke or bounded Stage B")
    num_envs = args.num_envs if args.num_envs is not None else (
        1 if args.cmv2_v114 and args.stage == "smoke" else 128)
    if num_envs <= 0 or (args.stage != "smoke" and num_envs != 128):
        raise ValueError("--num-envs must be positive and is adjustable only for smoke")
    window, budget_steps, lambda_cm = _stage_settings(args.stage)
    if args.stage in ("b", "c128", "c366") and not args.checkpoint:
        raise ValueError("Stage B/C requires an explicit pinned predecessor checkpoint")
    output = REPOSITORY_ROOT / "outputs/CmResidual" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    config_path, manifest_path = output / "config.json", output / "run_manifest.json"
    log_path, metrics_path = output / "train.log", output / "metrics.jsonl"
    reference, source = _input(REFERENCE, "grab_reference"), _input(SOURCE, "source_tensor")
    if reference["sha256"] != REFERENCE_SHA or source["sha256"] != SOURCE_SHA:
        raise ValueError("Pinned GRAB input SHA mismatch")
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Pinned predecessor checkpoint does not exist: {checkpoint}")
    checkpoint_identity = None
    if checkpoint is not None:
        checkpoint_digest = _sha256(checkpoint)
        if args.checkpoint_sha256 and checkpoint_digest != args.checkpoint_sha256:
            raise ValueError("Actor checkpoint SHA256 mismatch")
        if args.cmv2_v114 and args.stage == "b" and not args.checkpoint_sha256:
            raise ValueError("V1.14a Stage B requires --checkpoint-sha256")
        checkpoint_identity = {"path": str(checkpoint), "sha256": checkpoint_digest}
    v114_identity = None
    if args.cmv2_v114_checkpoint or args.cmv2_v114_sha256:
        if not (args.cmv2_v114_checkpoint and args.cmv2_v114_sha256):
            raise ValueError("V1.14a checkpoint path and SHA256 must be provided together")
        v114_path = Path(args.cmv2_v114_checkpoint).expanduser().resolve()
        v114_identity = _validate_v114_checkpoint(
            v114_path, args.cmv2_v114_sha256,
            require_inspire=args.stage == "b")
    elif args.cmv2_v114 and args.stage == "b":
        raise ValueError("V1.14a Stage B requires an explicit Inspire-compatible checkpoint")
    epochs = int(args.max_epochs) if args.max_epochs is not None else (1 if args.stage == "smoke" else 100000000)
    if epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    # The vendored CommonAgent stops on ``epoch > max_epochs``.  Interpret the
    # public launcher argument as the actual requested count instead.
    trainer_max_epochs = epochs - 1
    train_dir, cm_buffer = output / "train", output / "cm_buffer"
    task_name = "CmResidualGrabReferenceV118Cmv2V114" if args.cmv2_v114 else "CmResidualGrabReferenceV118"
    overrides = [
        f"task={task_name}", "train=CmResidualGrabReferenceV118PPO",
        "headless=True", "force_render=False", "pipeline=gpu", "sim_device=cuda:0", "rl_device=cuda:0",
        "graphics_device_id=0", f"num_envs={num_envs}", "num_subscenes=4", "seed=42",
        f"task.env.episodeLength={window}", f"task.referenceStart.windowLength={window}",
        f"task.cmBuffer.outputDir={cm_buffer}", f"train.params.config.cm_distill_coef={lambda_cm}",
        f"train.params.config.max_epochs={trainer_max_epochs}", f"+train.params.config.train_dir={train_dir}",
        f"+full_experiment_name=CmResidualGrabReferenceV118_{args.stage}",
        f"hydra.run.dir={output / 'hydra'}", "hydra.job.chdir=True",
    ]
    if args.stage == "smoke":
        overrides.append(f"train.params.config.minibatch_size={num_envs * 64}")
    if checkpoint is not None:
        overrides.append(f"checkpoint={checkpoint}")
    if v114_identity is not None:
        overrides.extend((f"task.basePolicy.cmv2Checkpoint={v114_identity['path']}",
                          f"task.basePolicy.cmv2CheckpointSha256={v114_identity['sha256']}"))
    resolved = _resolved_config(overrides)
    task, params = resolved["task"], resolved["train"]["params"]
    if (task["env"]["numObservations"] != 1442 or task["env"]["numActions"] != 18 or
            not task["basePolicy"]["useV118Planner"] or task["cmPlanner"]["candidates"] != 8 or
            task["cmBuffer"]["mode"] != "transition_only" or params["algo"]["name"] != "cm_planner_continuous"):
        raise RuntimeError("V1.18 task/PPO contract drift")
    if args.cmv2_v114 and (task["basePolicy"]["cmv2Schema"] != "cmv2_v114a_rigid_candidate_16x32_v1"
                           or int(task["basePolicy"]["cmHandPoints"]) != 2048):
        raise RuntimeError("V1.18c V1.14a Cmv2 contract drift")
    _write_json(config_path, {"run_id": args.run_id, "resolved": resolved, "runtime_overrides": overrides})
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "created_at": _now(), "task": "CmResidual",
        "mode": (f"v118c_v114a_reference_ppo_{args.stage}" if args.cmv2_v114
                 else f"v118_reference_ppo_{args.stage}"),
        "run_id": args.run_id, "activity_id": args.activity_id,
        "run_status": "STARTED", "work_version": args.work_version,
        "operation_category": ["experiment", "operation"], "output_dir": str(output), "seed": 42,
        "base_commit": _git("rev-parse", "HEAD"), "worktree_dirty": bool(_git("status", "--porcelain")),
        "config_snapshot": str(config_path), "metadata_snapshot": str(REFERENCE.with_name("manifest.json")),
        "input_references": [reference, source], "initial_checkpoint": checkpoint_identity,
        "cmv2_checkpoint": v114_identity,
        "metrics": str(metrics_path), "log": str(log_path), "cm_buffer": str(cm_buffer), "physical_gpu": args.gpu,
        "budget": {"envs": num_envs, "window": window,
                   "target_env_steps": (num_envs * window * epochs if args.max_epochs is not None else budget_steps),
                   "max_epochs": epochs,
                   "trainer_max_epochs": trainer_max_epochs,
                   "lambda_cm": lambda_cm}, "cmv2_variant": "v1.14a" if args.cmv2_v114 else "v1.3",
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)
    metrics_path.write_text("", encoding="utf-8")
    if args.dry_run:
        manifest.update(run_status="COMPLETED", completed_at=_now(), exit_reason="config preflight", conclusion="SUPPORTED")
        _write_json(manifest_path, manifest)
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["CMRESIDUAL_REFERENCE"] = str(REFERENCE)
    env["CMRESIDUAL_DEXPLORE_SOURCE"] = str(SOURCE)
    env["PYTHONPATH"] = os.pathsep.join(["/home2/wyy/isaac-gym/isaacgym/python", str(REPOSITORY_ROOT),
                                           str(VENDOR_ROOT), env.get("PYTHONPATH", "")])
    bootstrap = "import runpy, numpy as np; np.float=float; runpy.run_path(%r, run_name='__main__')" % str(VENDOR_ROOT / "isaacgymenvs/train.py")
    command = [sys.executable, "-c", bootstrap, *overrides]
    manifest.update(run_status="RUNNING", command=" ".join(shlex.quote(item) for item in command))
    _write_json(manifest_path, manifest)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"PPO exited {result.returncode}")
        manifest.update(run_status="COMPLETED", completed_at=_now(), conclusion="SUPPORTED",
                        scientific_conclusion="INCONCLUSIVE")
        _write_json(manifest_path, manifest)
    except BaseException as error:
        traceback.print_exc()
        manifest.update(run_status="FAILED", completed_at=_now(), exit_reason=f"{type(error).__name__}: {error}",
                        conclusion="INVALID_IMPLEMENTATION", scientific_conclusion="INCONCLUSIVE")
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
