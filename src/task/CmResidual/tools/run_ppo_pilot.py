"""Run and audit the fixed two-update CmResidual PPO wiring pilot."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from eval_zero_residual import (
    DEFAULT_DEXPLORE,
    DEFAULT_DEXPLORE_SOURCE,
    DEFAULT_OI_CM,
    DEFAULT_REFERENCE,
    REPOSITORY_ROOT,
    VENDOR_ROOT,
    _git,
    _input,
    _now,
    _write_json,
)


NUM_ENVS = 64
MAX_EPOCHS = 2
HORIZON_LENGTH = 32
MINIBATCH_SIZE = 2048
EXPERIMENT_NAME = "rlgames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--activity-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=5, help="Physical GPU index exposed as cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dexplore-checkpoint", type=Path, default=DEFAULT_DEXPLORE)
    parser.add_argument("--oi-cm-checkpoint", type=Path, default=DEFAULT_OI_CM)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--dexplore-source", type=Path, default=DEFAULT_DEXPLORE_SOURCE)
    parser.add_argument(
        "--isaac-gym-python", type=Path,
        default=Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python")))
    return parser.parse_args()


def _resolved_config(overrides: list[str]) -> dict:
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    compose_overrides = [value for value in overrides if not value.startswith((
        "+train.params.config.train_dir=", "+full_experiment_name=", "hydra."))]
    with initialize_config_dir(
            version_base="1.1", config_dir=str(VENDOR_ROOT / "isaacgymenvs/cfg")):
        cfg = compose(config_name="config", overrides=compose_overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _tensor_tree_finite(value) -> bool:
    import torch
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_tensor_tree_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tensor_tree_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _event_metrics(summary_dir: Path) -> tuple[list[dict], list[str]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = sorted(summary_dir.glob("events.out.tfevents*"))
    if len(event_files) != 1:
        raise RuntimeError(f"Expected one TensorBoard event file, got {event_files}")
    accumulator = EventAccumulator(str(event_files[0])).Reload()
    tags = accumulator.Tags().get("scalars", [])
    required = (
        "losses/a_loss", "losses/c_loss", "losses/entropy", "losses/bounds_loss",
        "info/kl", "info/last_lr", "info/epochs", "success_fraction/iter",
        "residual_rms/iter", "residual_saturation_ratio/iter",
    )
    missing = [tag for tag in required if tag not in tags]
    if missing:
        raise RuntimeError(f"Missing required TensorBoard scalars: {missing}; available={tags}")
    series = {tag: accumulator.Scalars(tag) for tag in tags}
    epoch_events = series["info/epochs"]
    if len(epoch_events) != MAX_EPOCHS:
        raise RuntimeError(f"Expected {MAX_EPOCHS} epochs, got {len(epoch_events)}")
    rows = []
    for index, epoch_event in enumerate(epoch_events):
        row = {"epoch": int(round(epoch_event.value)), "frame": int(epoch_event.step)}
        for tag in required:
            values = series[tag]
            if index >= len(values):
                raise RuntimeError(f"Scalar {tag} has only {len(values)} entries")
            row[tag] = float(values[index].value)
        rows.append(row)
    if not _tensor_tree_finite(rows):
        raise FloatingPointError(f"Non-finite TensorBoard metric: {rows}")
    return rows, tags


def _load_policy_action(checkpoint: Path, resolved: dict):
    import torch
    from isaacgymenvs.learning.cm_models import ModelCmContinuous
    from isaacgymenvs.learning.cm_network_builder import CmBuilder

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = resolved["train"]["params"]
    state = {
        key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
        for key, value in payload["model"].items()
    }

    def build():
        builder = CmBuilder()
        builder.load(params["network"])
        model = ModelCmContinuous(builder).build({
            "input_shape": (2005,),
            "actions_num": 18,
            "num_seqs": 1,
            "value_size": 1,
            "normalize_input": bool(params["config"]["normalize_input"]),
            "normalize_value": bool(params["config"]["normalize_value"]),
        })
        model.load_state_dict(state, strict=True)
        return model.eval()

    fixed = torch.linspace(-1.0, 1.0, 2005, dtype=torch.float32).unsqueeze(0)
    first, second = build(), build()
    with torch.no_grad():
        action_first = first({"obs": fixed, "is_train": False})["mus"]
        action_second = second({"obs": fixed, "is_train": False})["mus"]
    return payload, action_first, action_second


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cmresidual_ppo_pilot_v15_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = (args.output or REPOSITORY_ROOT / "outputs/CmResidual" / run_id).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot: {output}")
    output.mkdir(parents=True)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    train_log = output / "train.log"
    validation_path = output / "checkpoint_validation.json"

    dexplore = _input(args.dexplore_checkpoint, "dexplore_checkpoint")
    oi_cm = _input(args.oi_cm_checkpoint, "oi_cm_checkpoint")
    reference = _input(args.reference, "reference")
    source = _input(args.dexplore_source, "dexplore_source_tensor")
    reference_manifest_path = Path(reference["path"]).with_name("manifest.json")
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    activity_id = args.activity_id or "ACT-CMRESIDUAL-V153-PPO-PILOT"
    overrides = [
        "task=CmResidual",
        "train=CmResidualPilotPPO",
        "headless=True",
        f"num_envs={NUM_ENVS}",
        f"max_iterations={MAX_EPOCHS}",
        "pipeline=gpu",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
        "graphics_device_id=0",
        "force_render=False",
        "task.sim.physx.max_gpu_contact_pairs=8388608",
        "task.sim.physx.default_buffer_size_multiplier=5.0",
        "task.reference.allowIneligibleFor=ppo_pilot",
        f"seed={args.seed}",
        f"+train.params.config.train_dir={output}",
        f"+full_experiment_name={EXPERIMENT_NAME}",
        f"hydra.run.dir={output / 'hydra'}",
        "hydra.job.chdir=True",
    ]
    resolved = _resolved_config(overrides)
    _write_json(config_path, {
        "run_id": run_id,
        "physical_gpu": args.gpu,
        "cuda_visible_devices": str(args.gpu),
        "resolved": resolved,
        "runtime_overrides": overrides,
    })
    command_values = [sys.executable, str(VENDOR_ROOT / "isaacgymenvs/train.py"), *overrides]
    command = " ".join(shlex.quote(value) for value in command_values)
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": _now(),
        "mode": "ppo_wiring_pilot",
        "task": "CmResidual",
        "run_id": run_id,
        "activity_id": activity_id,
        "run_status": "STARTED",
        "modification_version": "V1.5.4",
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
        "reference_allow_ineligible_for": "ppo_pilot",
        "initial_checkpoint": None,
        "checkpoint": None,
        "metrics": str(metrics_path),
        "log": str(train_log),
        "physical_gpu": args.gpu,
        "budget": {
            "num_envs": NUM_ENVS,
            "max_epochs": MAX_EPOCHS,
            "horizon_length": HORIZON_LENGTH,
            "minibatch_size": MINIBATCH_SIZE,
        },
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["DEXPLORE_CHECKPOINT"] = dexplore["path"]
    environment["OI_CM_CHECKPOINT"] = oi_cm["path"]
    environment["CMRESIDUAL_REFERENCE"] = reference["path"]
    environment["CMRESIDUAL_DEXPLORE_SOURCE"] = source["path"]
    python_paths = [str(args.isaac_gym_python.resolve()), str(REPOSITORY_ROOT), str(VENDOR_ROOT)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    try:
        print(json.dumps({"run_id": run_id, "run_status": "STARTED",
                          "output": str(output), "physical_gpu": args.gpu}, ensure_ascii=False), flush=True)
        with train_log.open("w", encoding="utf-8", buffering=1) as stream:
            result = subprocess.run(
                command_values, cwd=REPOSITORY_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training subprocess exited with code {result.returncode}")

        experiment_dir = output / EXPERIMENT_NAME
        checkpoints = sorted((experiment_dir / "nn").glob("*.pth"), key=lambda path: path.stat().st_mtime_ns)
        if not checkpoints:
            raise RuntimeError(f"Pilot produced no checkpoint under {experiment_dir / 'nn'}")
        last_checkpoint = checkpoints[-1]
        best_checkpoint = experiment_dir / "nn/CmResidualPilot.pth"
        rows, event_tags = _event_metrics(experiment_dir / "summaries")
        with metrics_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        payload, action_first, action_second = _load_policy_action(last_checkpoint, resolved)
        reload_error = float((action_first - action_second).abs().amax().item())
        validation = {
            "checkpoint": str(last_checkpoint),
            "checkpoint_sha256": _input(last_checkpoint, "ppo_checkpoint")["sha256"],
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_frame": int(payload.get("frame", -1)),
            "model_finite": _tensor_tree_finite(payload.get("model", {})),
            "optimizer_finite": _tensor_tree_finite(payload.get("optimizer", {})),
            "normalizer_finite": _tensor_tree_finite({
                key: value for key, value in payload.get("model", {}).items()
                if "running_mean_std" in key or "value_mean_std" in key
            }),
            "action_finite": _tensor_tree_finite((action_first, action_second)),
            "deterministic_reload_action_max_abs_diff": reload_error,
            "event_scalar_tags": event_tags,
        }
        _write_json(validation_path, validation)
        completed_episode = any(tag in event_tags for tag in ("Episode/success", "rewards/iter"))
        passed = (
            validation["checkpoint_epoch"] >= MAX_EPOCHS
            and validation["model_finite"]
            and validation["optimizer_finite"]
            and validation["normalizer_finite"]
            and validation["action_finite"]
            and reload_error <= 1e-6
            and completed_episode
            and len(rows) == MAX_EPOCHS)
        manifest.update({
            "run_status": "COMPLETED",
            "completed_at": _now(),
            "last_step": validation["checkpoint_frame"],
            "last_epoch": validation["checkpoint_epoch"],
            "best_metric": {
                "name": "checkpoint_reload_action_max_abs_diff",
                "value": reload_error,
            },
            "checkpoint": str(last_checkpoint),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else None,
            "checkpoint_validation": str(validation_path),
            "exit_reason": "Reached fixed two-update PPO wiring budget",
            "gate_passed": passed,
            "conclusion": "SUPPORTED" if passed else "INVALID_IMPLEMENTATION",
            "scientific_conclusion": "INCONCLUSIVE",
            "summary": {
                "epochs": len(rows),
                "completed_episode": completed_episode,
                "checkpoint_count": len(checkpoints),
                "reload_action_max_abs_diff": reload_error,
                "final_metrics": rows[-1],
            },
        })
        _write_json(manifest_path, manifest)
        print(json.dumps({"run_id": run_id, "run_status": "COMPLETED",
                          "gate_passed": passed, **manifest["summary"]}, ensure_ascii=False), flush=True)
        if not passed:
            raise RuntimeError(f"PPO wiring gate failed: {manifest['summary']}")
    except BaseException as error:
        traceback.print_exc()
        if not metrics_path.exists():
            metrics_path.write_text(json.dumps({
                "status": "FAILED", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False) + "\n", encoding="utf-8")
        if manifest.get("run_status") != "COMPLETED":
            manifest.update({
                "run_status": "FAILED",
                "completed_at": _now(),
                "last_step": None,
                "last_epoch": None,
                "best_metric": None,
                "checkpoint": None,
                "exit_reason": f"{type(error).__name__}: {error}",
                "conclusion": "INVALID_IMPLEMENTATION",
                "scientific_conclusion": "INCONCLUSIVE",
            })
            _write_json(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
