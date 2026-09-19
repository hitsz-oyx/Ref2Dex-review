"""Run one locked V1.14 GRAB reference-transition PPO curriculum stage."""
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
    parser.add_argument("--run-id", default=f"cmresidual_v1142_reftrack_w64_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--work-version", default="V1.14.4")
    parser.add_argument("--window-length", type=int, choices=(64, 128, 366), default=64)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--cm-buffer", action="store_true",
                        help="Record frozen Cmv2 executed-transition diagnostics without PPO updates to Cmv2")
    parser.add_argument("--cmv2-actor", action="store_true",
                        help="Use frozen zero-residual Cmv2 tokens/effects in actor only; also records CmBuffer")
    parser.add_argument("--v116", action="store_true",
                        help="Use the V1.16 GPU nominal-base path and transition-only buffer")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 1-env, 1-epoch implementation smoke instead of the locked curriculum")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu != 5:
        raise ValueError("V1.14 curriculum is approved only on GPU5")
    if args.v116 and not args.cmv2_actor:
        raise ValueError("--v116 requires --cmv2-actor")
    output = REPOSITORY_ROOT / "outputs/CmResidual" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    config, manifest_path = output / "config.json", output / "run_manifest.json"
    metrics, log, validation_path = output / "metrics.jsonl", output / "train.log", output / "checkpoint_validation.json"
    reference, source = _input(REFERENCE, "grab_reference"), _input(SOURCE, "source_tensor")
    if reference["sha256"] != REFERENCE_SHA or source["sha256"] != SOURCE_SHA:
        raise ValueError("Pinned GRAB input SHA mismatch")
    metadata = REFERENCE.with_name("manifest.json")
    if not json.loads(metadata.read_text())["training_eligible"]:
        raise ValueError("GRAB reference is not training eligible")
    horizon = 8 if args.smoke else 64 if args.window_length == 64 else 128
    if args.v116:
        task_name = "CmResidualGrabReferenceTransitionCmv2ActorV116"
        train_name = "CmResidualGrabReferenceTransitionCmv2ActorV116PPO"
    elif args.cmv2_actor:
        task_name = "CmResidualGrabReferenceTransitionCmv2Actor"
        train_name = "CmResidualGrabReferenceTransitionCmv2ActorPPO"
    else:
        task_name = "CmResidualGrabReferenceTransitionCmBuffer" if args.cm_buffer else "CmResidualGrabReferenceTransition"
        train_name = ("CmResidualGrabReferenceTransitionCmBufferPPO" if args.cm_buffer
                      else "CmResidualGrabReferenceTransitionPPO")
    common = [
        f"task={task_name}", f"train={train_name}",
        "headless=True", f"num_envs={1 if args.smoke else 64}", "pipeline=gpu", "sim_device=cuda:0", "rl_device=cuda:0",
        "graphics_device_id=0", "force_render=False", "num_subscenes=4", "seed=42",
        "task.sim.physx.max_gpu_contact_pairs=8388608", "task.sim.physx.default_buffer_size_multiplier=5.0",
        f"task.env.episodeLength={args.window_length}", f"task.referenceStart.windowLength={args.window_length}",
        f"train.params.config.horizon_length={horizon}", f"train.params.config.minibatch_size={8 if args.smoke else 2048}",
        f"task.reference.path={REFERENCE}", f"task.reference.sourceTensor={SOURCE}"]
    cm_buffer_dir = output / "cm_buffer"
    if args.cm_buffer or args.cmv2_actor:
        common += [f"task.cmBuffer.outputDir={cm_buffer_dir}", "task.cmBuffer.rank=0"]
        if args.smoke:
            common.append("task.cmBuffer.flushEvery=1")
    resolved = _resolved_config(common)
    task = resolved["task"]
    expected_observation_dim = 726 if args.cmv2_actor else 68
    if (task["env"]["numObservations"] != expected_observation_dim or task["referenceStart"]["mode"] != "random_uniform" or
            task["referenceStart"]["windowLength"] != args.window_length or
            task["referenceTrackingReward"]["mode"] != "reference_transition" or
            task["env"]["terminateOnSuccess"] or
            bool(task["basePolicy"].get("useCmv2ActionEvaluator", False)) !=
            (args.cm_buffer if not args.v116 else False)):
        raise RuntimeError("V1.14 curriculum task contract drift")
    observation_dim = 726 if args.cmv2_actor else 68
    contract = _safe_contract(resolved["train"]["params"], observation_dim)
    actor_dim = 214 if args.cmv2_actor else 68
    if (not contract["passed"] or contract["actor_input_dim"] != actor_dim or
            contract["critic_input_dim"] != 68):
        raise RuntimeError(f"PPO model contract failed: {contract}")
    _write_json(config, {"run_id": args.run_id, "resolved": resolved, "runtime_overrides": common,
                         "model_contract": contract})
    manifest = {"manifest_schema": "ref2dex.run.v1", "created_at": _now(), "task": "CmResidual",
                "mode": ("v116_cmv2_actor_transition_only_ppo" if args.v116 else
                         "v115_cmv2_actor_reference_transition_ppo" if args.cmv2_actor else
                         "v114_reference_transition_ppo"), "run_id": args.run_id, "activity_id": args.activity_id,
                "run_status": "STARTED", "work_version": args.work_version, "operation_category": ["experiment", "operation"],
                "output_dir": str(output), "seed": 42, "base_commit": _git("rev-parse", "HEAD"),
                "worktree_dirty": bool(_git("status", "--porcelain")), "config_snapshot": str(config),
                "metadata_snapshot": str(metadata), "input_references": [reference, source], "initial_checkpoint": None,
                "checkpoint": None, "metrics": str(metrics), "log": str(log), "physical_gpu": 5,
                "cm_buffer": str(cm_buffer_dir) if (args.cm_buffer or args.cmv2_actor) else None,
                "budget": {"num_envs": 1 if args.smoke else 64, "window_length": args.window_length, "horizon_length": horizon,
                           "minibatch_size": 8 if args.smoke else 2048,
                           "stages": [1] if args.smoke else [2, 10], "saturation_stop": 0.05},
                "conclusion": "INCONCLUSIVE", "stages": []}
    _write_json(manifest_path, manifest); metrics.write_text(""); log.write_text("")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "5"; env["CMRESIDUAL_REFERENCE"] = str(REFERENCE); env["CMRESIDUAL_DEXPLORE_SOURCE"] = str(SOURCE)
    env["PYTHONPATH"] = os.pathsep.join(["/home2/wyy/isaac-gym/isaacgym/python", str(REPOSITORY_ROOT), str(VENDOR_ROOT), env.get("PYTHONPATH", "")])
    bootstrap = "import runpy, numpy as np; np.float=float; runpy.run_path(%r, run_name='__main__')" % str(VENDOR_ROOT / "isaacgymenvs/train.py")
    checkpoint, rows = None, []
    try:
        stages = (("smoke", 1, 1),) if args.smoke else (("smoke", 2, 2), ("continuation", 10, 8))
        for label, end, expected in stages:
            stage_dir, name = output / label, f"CmResidualGrabReferenceTransition_{label}"
            overrides = common + [f"train.params.config.max_epochs={end}", f"+train.params.config.train_dir={stage_dir}",
                                  f"+full_experiment_name={name}", f"hydra.run.dir={stage_dir / 'hydra'}", "hydra.job.chdir=True"]
            if checkpoint: overrides.append(f"checkpoint={checkpoint}")
            command = [sys.executable, "-c", bootstrap, *overrides]
            manifest.update(run_status="RUNNING", active_stage=label, command=" ".join(shlex.quote(x) for x in command)); _write_json(manifest_path, manifest)
            with log.open("a", encoding="utf-8", buffering=1) as stream:
                result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f"{label} exited {result.returncode}")
            stage_rows, tags = _event_metrics(stage_dir / name / "summaries", expected)
            if stage_rows[-1]["epoch"] != end: raise RuntimeError("epoch contract drift")
            checkpoint = sorted((stage_dir / name / "nn").glob("*.pth"), key=lambda p: p.stat().st_mtime_ns)[-1].resolve()
            _, validation = _load_checkpoint(checkpoint, resolved["train"]["params"], observation_dim)
            if (not validation["model_finite"] or not validation["optimizer_finite"] or not validation["action_finite"] or
                    validation["deterministic_reload_action_max_abs_diff"] > 1e-6): raise RuntimeError("checkpoint validation failed")
            for row in stage_rows: row["stage"] = label; rows.append(row)
            with metrics.open("a", encoding="utf-8") as stream:
                for row in stage_rows: stream.write(json.dumps(row) + "\n")
            manifest["stages"].append({"stage": label, "last_epoch": end, "checkpoint": str(checkpoint), "event_tags": tags})
            manifest.update(checkpoint=str(checkpoint), last_epoch=end, last_step=validation["checkpoint_frame"]); _write_json(manifest_path, manifest)
            if any(row["residual_saturation_ratio/iter"] > .05 for row in stage_rows): raise RuntimeError("residual saturation exceeded 5%")
        validation.update(checkpoint=str(checkpoint), checkpoint_sha256=_input(checkpoint, "ppo_checkpoint")["sha256"]); _write_json(validation_path, validation)
        manifest.update(run_status="COMPLETED", completed_at=_now(), checkpoint_validation=str(validation_path),
                        best_metric={"name": "checkpoint_reload_action_max_abs_diff", "value": validation["deterministic_reload_action_max_abs_diff"]},
                        exit_reason="Reached 1-epoch smoke budget" if args.smoke else "Reached locked 2+8 epoch budget",
                        conclusion="SUPPORTED", scientific_conclusion="INCONCLUSIVE"); _write_json(manifest_path, manifest)
    except BaseException as error:
        traceback.print_exc(); manifest.update(run_status="FAILED", completed_at=_now(), exit_reason=f"{type(error).__name__}: {error}", conclusion="INVALID_IMPLEMENTATION", scientific_conclusion="INCONCLUSIVE"); _write_json(manifest_path, manifest); raise


if __name__ == "__main__":
    main()
