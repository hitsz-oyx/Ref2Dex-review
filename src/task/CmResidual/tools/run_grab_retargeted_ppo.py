"""Run the approved V1.13 exploratory GRAB PPO in two audited stages."""
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


REFERENCE = REPOSITORY_ROOT / "data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/reference.npz"
SOURCE = REPOSITORY_ROOT / "data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/s1_airplane_lift/interaction_hand_inspire.pt"
REFERENCE_SHA = "a2d710b911cf8988750208450c411b3095e187ed2b2f05df459c24d5748812c1"
SOURCE_SHA = "19b110dc81c4928b4f3e6197d8549b011fd48a8157e75fd06668bd306dc396cf"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"cmresidual_v1134_grab_ppo_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--gpu", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu != 5:
        raise ValueError("V1.13 exploratory PPO is approved only on GPU5")
    output = REPOSITORY_ROOT / "outputs/CmResidual" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    log_path = output / "train.log"
    validation_path = output / "checkpoint_validation.json"
    reference = _input(REFERENCE, "grab_reference")
    source = _input(SOURCE, "source_tensor")
    if reference["sha256"] != REFERENCE_SHA or source["sha256"] != SOURCE_SHA:
        raise ValueError("GRAB reference/source SHA mismatch")
    reference_manifest_path = REFERENCE.with_name("manifest.json")
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    if reference_manifest.get("split") != "train" or reference_manifest.get("training_eligible") is not True:
        raise ValueError("GRAB reference is not training eligible")

    common = [
        "task=CmResidualGrabRetargeted", "train=CmResidualGrabRetargetedPPO",
        "headless=True", "num_envs=64", "task.env.wristStiffness=400.0",
        "task.env.wristDamping=40.0", "pipeline=gpu", "sim_device=cuda:0",
        "rl_device=cuda:0", "graphics_device_id=0", "force_render=False",
        "num_subscenes=4", "task.sim.physx.max_gpu_contact_pairs=8388608",
        "task.sim.physx.default_buffer_size_multiplier=5.0", "seed=42",
        f"task.reference.path={REFERENCE}", f"task.reference.sourceTensor={SOURCE}",
        "train.params.config.horizon_length=32",
        "train.params.config.minibatch_size=2048",
    ]
    resolved = _resolved_config(common)
    task = resolved["task"]
    if (task["basePolicy"]["mode"] != "retargeted_reference" or
            task["basePolicy"]["useOiCmContext"] or
            task["basePolicy"].get("useCmv2Context", False) or
            task["env"]["numObservations"] != 68 or
            task["env"]["episodeLength"] != 366 or
            task["env"]["wristStiffness"] != 400.0 or
            task["env"]["wristDamping"] != 40.0 or
            resolved["checkpoint"]):
        raise RuntimeError("GRAB PPO task contract drift")
    contract = _safe_contract(resolved["train"]["params"], 68)
    if not contract["passed"] or contract["actor_input_dim"] != 68 or contract["critic_input_dim"] != 68:
        raise RuntimeError(f"GRAB PPO model contract failed: {contract}")
    _write_json(config_path, {"run_id": args.run_id, "physical_gpu": 5,
                              "resolved": resolved, "runtime_overrides": common,
                              "model_contract": contract})
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "created_at": _now(),
        "mode": "exploratory_grab_retargeted_ppo", "task": "CmResidual",
        "run_id": args.run_id, "activity_id": args.activity_id,
        "run_status": "STARTED", "modification_version": "V1.13.4",
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output), "seed": 42, "base_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "config_snapshot": str(config_path), "metadata_snapshot": str(reference_manifest_path),
        "input_references": [reference, source], "reference_training_eligible": True,
        "initial_checkpoint": None, "checkpoint": None,
        "metrics": str(metrics_path), "log": str(log_path), "physical_gpu": 5,
        "budget": {"num_envs": 64, "max_epochs": 10, "horizon_length": 32,
                   "minibatch_size": 2048, "total_env_steps": 20480},
        "model_contract": contract, "conclusion": "INCONCLUSIVE", "stages": [],
    }
    _write_json(manifest_path, manifest)
    metrics_path.write_text("", encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "5"
    environment["CMRESIDUAL_REFERENCE"] = str(REFERENCE)
    environment["CMRESIDUAL_DEXPLORE_SOURCE"] = str(SOURCE)
    python_paths = [str(Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python"))),
                    str(REPOSITORY_ROOT), str(VENDOR_ROOT)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    entry = VENDOR_ROOT / "isaacgymenvs/train.py"
    bootstrap = ("import runpy, numpy as np; np.float = float; "
                 f"runpy.run_path({str(entry)!r}, run_name='__main__')")
    checkpoint = None
    all_rows = []
    try:
        for stage, end_epoch, expected in (("stage2", 2, 2), ("stage10", 10, 8)):
            stage_dir = output / stage
            experiment = f"CmResidualGrabRetargetedPPO_{stage}"
            overrides = common + [f"train.params.config.max_epochs={end_epoch}",
                                  f"+train.params.config.train_dir={stage_dir}",
                                  f"+full_experiment_name={experiment}",
                                  f"hydra.run.dir={stage_dir / 'hydra'}",
                                  "hydra.job.chdir=True"]
            if checkpoint is not None:
                overrides.append(f"checkpoint={checkpoint}")
            command = [sys.executable, "-c", bootstrap, *overrides]
            manifest["run_status"] = "RUNNING"
            manifest["active_stage"] = stage
            manifest["command"] = " ".join(shlex.quote(arg) for arg in command)
            _write_json(manifest_path, manifest)
            with log_path.open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(f"\n=== {stage} ===\n")
                stream.flush()
                result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment,
                                        stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{stage} training subprocess exited {result.returncode}")
            experiment_dir = stage_dir / experiment
            rows, tags = _event_metrics(experiment_dir / "summaries", expected)
            if rows[-1]["epoch"] != end_epoch:
                raise RuntimeError(f"{stage} final epoch drift: {rows[-1]['epoch']}")
            checkpoints = sorted((experiment_dir / "nn").glob("*.pth"),
                                 key=lambda path: path.stat().st_mtime_ns)
            if not checkpoints:
                raise RuntimeError(f"{stage} produced no checkpoint")
            checkpoint = checkpoints[-1].resolve()
            _, validation = _load_checkpoint(checkpoint, resolved["train"]["params"], 68)
            if (validation["checkpoint_epoch"] != end_epoch or
                    not validation["model_finite"] or not validation["optimizer_finite"] or
                    not validation["action_finite"] or
                    validation["deterministic_reload_action_max_abs_diff"] > 1e-6):
                raise RuntimeError(f"{stage} checkpoint validation failed: {validation}")
            for row in rows:
                row["stage"] = stage
                all_rows.append(row)
            with metrics_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["stages"].append({"stage": stage, "last_epoch": end_epoch,
                                       "checkpoint": str(checkpoint), "metrics_count": len(rows),
                                       "event_tags": tags})
            manifest["checkpoint"] = str(checkpoint)
            manifest["last_epoch"] = end_epoch
            manifest["last_step"] = validation["checkpoint_frame"]
            _write_json(manifest_path, manifest)
            print(json.dumps({"run_id": args.run_id, "stage": stage,
                              "last_epoch": end_epoch, "checkpoint": str(checkpoint)}), flush=True)
            if any(row["residual_saturation_ratio/iter"] > 0.1 for row in rows):
                raise RuntimeError(f"{stage} residual target saturation exceeded 10%")
        if len(all_rows) != 10 or [row["epoch"] for row in all_rows] != list(range(1, 11)):
            raise RuntimeError("Expected continuous epoch 1..10 metrics")
        validation["checkpoint"] = str(checkpoint)
        validation["checkpoint_sha256"] = _input(checkpoint, "ppo_checkpoint")["sha256"]
        _write_json(validation_path, validation)
        manifest.update({"run_status": "COMPLETED", "completed_at": _now(),
                         "checkpoint_validation": str(validation_path),
                         "best_metric": {"name": "checkpoint_reload_action_max_abs_diff",
                                         "value": validation["deterministic_reload_action_max_abs_diff"]},
                         "best_checkpoint": None,
                         "exit_reason": "Reached approved 10-epoch exploratory budget",
                         "gate_passed": True, "conclusion": "SUPPORTED",
                         "scientific_conclusion": "INCONCLUSIVE",
                         "summary": {"epochs": 10, "final_metrics": all_rows[-1]}})
        _write_json(manifest_path, manifest)
    except BaseException as error:
        traceback.print_exc()
        manifest.update({"run_status": "FAILED", "completed_at": _now(),
                         "exit_reason": f"{type(error).__name__}: {error}",
                         "conclusion": "INVALID_IMPLEMENTATION",
                         "scientific_conclusion": "INCONCLUSIVE"})
        _write_json(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
