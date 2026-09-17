"""Launch an explicit three-rank V1.16 Cmv2 actor PPO capacity/formal run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from eval_zero_residual import _git, _input, _now, _write_json, REPOSITORY_ROOT, VENDOR_ROOT
from run_ppo_formal import _event_metrics
from run_ppo_stability import _load_checkpoint, _resolved_config, _safe_contract


GPU_IDS = (0, 1, 2)
REFERENCE = REPOSITORY_ROOT / "data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/reference.npz"
SOURCE = REPOSITORY_ROOT / "data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/s1_airplane_lift/interaction_hand_inspire.pt"
REFERENCE_SHA = "a2d710b911cf8988750208450c411b3095e187ed2b2f05df459c24d5748812c1"
SOURCE_SHA = "19b110dc81c4928b4f3e6197d8549b011fd48a8157e75fd06668bd306dc396cf"
BOOTSTRAP = Path(__file__).with_name("run_cmv2_actor_bootstrap.py")


def _parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integers") from error
    if result != GPU_IDS:
        raise argparse.ArgumentTypeError("V1.16 requires physical GPUs exactly 0,1,2")
    return result


def _gpu_memory(gpus: tuple[int, ...]) -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE)
    values = {}
    for line in result.stdout.splitlines():
        index, used = (item.strip() for item in line.split(","))
        values[int(index)] = int(used)
    missing = set(gpus) - set(values)
    if missing:
        raise RuntimeError(f"nvidia-smi did not report GPUs {sorted(missing)}")
    return {gpu: values[gpu] for gpu in gpus}


def _torchrun_command(overrides: list[str]) -> list[str]:
    """Use the active Python's torchrun module and a real training script."""
    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(f"Missing DDP bootstrap script: {BOOTSTRAP}")
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=3", str(BOOTSTRAP), *overrides]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"cmresidual_v116_ddp_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--modification-version", default="V1.16")
    parser.add_argument("--gpus", type=_parse_gpus, required=True)
    parser.add_argument("--envs-per-rank", type=int, choices=(128, 256), required=True)
    parser.add_argument("--capacity-probe", action="store_true",
                        help="Run a bounded one-epoch capacity probe, not a formal training result")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Explicit formal-run epoch cap; omit for manual-stop operation")
    parser.add_argument("--max-used-mib", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_used_mib < 0:
        raise ValueError("--max-used-mib must be non-negative")
    if args.capacity_probe and args.max_epochs is not None:
        raise ValueError("capacity probe owns its one-epoch budget; omit --max-epochs")
    output = REPOSITORY_ROOT / "outputs/CmResidual" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    config_path, manifest_path = output / "config.json", output / "run_manifest.json"
    metrics_path, log_path = output / "metrics.jsonl", output / "train.log"
    reference, source = _input(REFERENCE, "grab_reference"), _input(SOURCE, "source_tensor")
    if reference["sha256"] != REFERENCE_SHA or source["sha256"] != SOURCE_SHA:
        raise ValueError("Pinned GRAB input SHA mismatch")
    if not json.loads(REFERENCE.with_name("manifest.json").read_text())["training_eligible"]:
        raise ValueError("GRAB reference is not training eligible")
    used_mib = _gpu_memory(args.gpus)
    if any(value > args.max_used_mib for value in used_mib.values()):
        raise RuntimeError(f"GPU capacity gate failed: used MiB={used_mib}, limit={args.max_used_mib}")
    cm_buffer_dir = output / "cm_buffer"
    train_dir = output / "train"
    experiment_name = "CmResidualGrabReferenceTransitionCmv2ActorV116_ddp"
    overrides = [
        "task=CmResidualGrabReferenceTransitionCmv2ActorV116",
        "train=CmResidualGrabReferenceTransitionCmv2ActorV116PPO",
        "headless=True", "force_render=False", "pipeline=gpu", "multi_gpu=True",
        f"num_envs={args.envs_per_rank}", "sim_device=cuda:0", "rl_device=cuda:0",
        "graphics_device_id=0", "num_subscenes=4", "seed=42",
        "task.sim.physx.max_gpu_contact_pairs=8388608",
        "task.sim.physx.default_buffer_size_multiplier=5.0",
        f"task.reference.path={REFERENCE}", f"task.reference.sourceTensor={SOURCE}",
        f"task.cmBuffer.outputDir={cm_buffer_dir}",
        f"+train.params.config.train_dir={train_dir}",
        f"+full_experiment_name={experiment_name}",
        f"hydra.run.dir={output / 'hydra'}", "hydra.job.chdir=True",
    ]
    if args.capacity_probe:
        overrides.append("train.params.config.max_epochs=1")
    elif args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError("--max-epochs must be positive")
        overrides.append(f"train.params.config.max_epochs={args.max_epochs}")
    resolved = _resolved_config(overrides)
    task, params = resolved["task"], resolved["train"]["params"]
    if (task["env"]["numObservations"] != 726 or task["cmBuffer"]["mode"] != "transition_only" or
            task["cmBuffer"]["schema"] != "cmresidual.cm_buffer.transition_only.v2" or
            task["basePolicy"]["useCmv2ActionEvaluator"] or not task["basePolicy"]["useCmv2ActorContext"]):
        raise RuntimeError("V1.16 task contract drift")
    contract = _safe_contract(params, 726)
    if not contract["passed"] or contract["actor_input_dim"] != 214 or contract["critic_input_dim"] != 68:
        raise RuntimeError(f"V1.16 model contract failed: {contract}")
    _write_json(config_path, {"run_id": args.run_id, "resolved": resolved,
                              "runtime_overrides": overrides, "model_contract": contract})
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "created_at": _now(), "task": "CmResidual",
        "mode": "v116_cmv2_actor_ddp_capacity" if args.capacity_probe else "v116_cmv2_actor_ddp_formal",
        "run_id": args.run_id, "activity_id": args.activity_id, "run_status": "STARTED",
        "modification_version": args.modification_version,
        "operation_category": ["experiment", "operation"], "output_dir": str(output),
        "seed": 42, "base_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")), "config_snapshot": str(config_path),
        "metadata_snapshot": str(REFERENCE.with_name("manifest.json")),
        "input_references": [reference, source], "metrics": str(metrics_path), "log": str(log_path),
        "physical_gpus": list(args.gpus), "gpu_used_mib_preflight": used_mib,
        "world_size": 3, "envs_per_rank": args.envs_per_rank,
        "total_envs": 3 * args.envs_per_rank, "cm_buffer": str(cm_buffer_dir),
        "rank_buffer_dirs": [str(cm_buffer_dir / f"rank_{rank:03d}") for rank in range(3)],
        "budget": "one epoch capacity probe" if args.capacity_probe else
                  (f"explicit {args.max_epochs} epochs" if args.max_epochs else "manual stop"),
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)
    metrics_path.write_text("", encoding="utf-8")
    if args.dry_run:
        manifest.update(run_status="COMPLETED", completed_at=_now(),
                        exit_reason="Preflight/config dry run", conclusion="SUPPORTED",
                        scientific_conclusion="INCONCLUSIVE")
        _write_json(manifest_path, manifest)
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    env["PYTHONPATH"] = os.pathsep.join([
        "/home2/wyy/isaac-gym/isaacgym/python", str(REPOSITORY_ROOT), str(VENDOR_ROOT),
        env.get("PYTHONPATH", "")])
    command = _torchrun_command(overrides)
    manifest.update(run_status="RUNNING", command=" ".join(shlex.quote(item) for item in command))
    _write_json(manifest_path, manifest)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"torchrun exited {result.returncode}")
        rank_manifests = [Path(path) / "manifest.json" for path in manifest["rank_buffer_dirs"]]
        if not all(path.is_file() for path in rank_manifests):
            raise RuntimeError(f"Missing rank buffer manifests: {rank_manifests}")
        expected_epochs = 1 if args.capacity_probe else args.max_epochs
        if expected_epochs is not None:
            rows, tags = _event_metrics(train_dir / experiment_name / "summaries", expected_epochs)
            with metrics_path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        else:
            tags = []
        checkpoint = sorted((train_dir / experiment_name / "nn").glob("*.pth"),
                            key=lambda path: path.stat().st_mtime_ns)[-1]
        _, validation = _load_checkpoint(checkpoint, params, 726)
        if not all(validation[key] for key in ("model_finite", "optimizer_finite", "action_finite")):
            raise RuntimeError("DDP checkpoint finite validation failed")
        validation_path = output / "checkpoint_validation.json"
        validation.update(checkpoint=str(checkpoint), checkpoint_sha256=_input(checkpoint, "ppo_checkpoint")["sha256"])
        _write_json(validation_path, validation)
        manifest.update(run_status="COMPLETED", completed_at=_now(), checkpoint=str(checkpoint),
                        checkpoint_validation=str(validation_path), last_step=validation["checkpoint_frame"],
                        last_epoch=validation["checkpoint_epoch"],
                        best_metric={"name": "checkpoint_reload_action_max_abs_diff",
                                     "value": validation["deterministic_reload_action_max_abs_diff"]},
                        event_tags=tags,
                        conclusion="SUPPORTED", scientific_conclusion="INCONCLUSIVE")
        _write_json(manifest_path, manifest)
    except BaseException as error:
        traceback.print_exc()
        manifest.update(run_status="STOPPED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                        completed_at=_now(), exit_reason=f"{type(error).__name__}: {error}",
                        conclusion="INVALID_IMPLEMENTATION", scientific_conclusion="INCONCLUSIVE")
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
