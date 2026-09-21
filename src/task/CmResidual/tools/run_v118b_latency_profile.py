"""Run the approved V1.18b cumulative interaction latency benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import traceback

from run_v118a_latency_profile import (
    ROOT, VENDOR, ISAAC_GYM, PYTHON, REFERENCE, SOURCE, CHECKPOINT,
    _now, _sha256, _write, _git, _gpu_used_mib, _inputs,
)


BOOTSTRAP = Path(__file__).resolve().with_name("v118b_latency_bootstrap.py")
OUTPUT_ROOT = ROOT / "src/task/CmResidual/research/v118b_sparse_interaction/output"
EXPECTED_STATE_SHA = "9f66e70fbb07aa66eed9b7da12cf7d1d64db02180240f690db9fec584b905e75"
EXPECTED_CANDIDATE_SHA = "0c3723c72e3996f0a5749474a3afcd28fb7bc9e3a92c3dfaf0f26fec12628381"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"cmresidual_v118b_latency_gpu6_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("invalid GPU or iteration counts")
    inputs = _inputs()
    used_mib = _gpu_used_mib(args.gpu)
    if used_mib > 1024:
        raise RuntimeError(f"GPU{args.gpu} capacity gate failed: {used_mib} MiB used")
    if args.dry_run:
        print(json.dumps({"preflight": "passed", "gpu": args.gpu, "gpu_used_mib": used_mib,
                          "warmup": args.warmup, "repeats": args.repeats, "inputs": inputs}, sort_keys=True))
        return 0
    if _git("status", "--porcelain"):
        raise RuntimeError("formal profile requires a clean worktree")
    output = (OUTPUT_ROOT / args.run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile_path, log_path = output / "profile.json", output / "run.log"
    overrides = [
        "task=CmResidualGrabReferenceV118", "train=CmResidualGrabReferenceV118PPO",
        "headless=True", "force_render=False", "pipeline=gpu", "sim_device=cuda:0", "rl_device=cuda:0",
        "graphics_device_id=0", "num_envs=128", "num_subscenes=4", "seed=42",
        "task.env.episodeLength=64", "task.referenceStart.windowLength=64",
        "train.params.config.cm_distill_coef=0.1", "train.params.config.max_epochs=0",
        f"checkpoint={CHECKPOINT}", f"task.cmBuffer.outputDir={output / 'unused_buffer'}",
        f"+train.params.config.train_dir={output / 'unused_train'}",
        "+full_experiment_name=CmResidualV118bLatency", f"hydra.run.dir={output / 'hydra'}",
        "hydra.job.chdir=True",
    ]
    command = [str(PYTHON), str(BOOTSTRAP), *overrides]
    manifest = {
        "manifest_schema": "ref2dex.cmresidual.v118b_latency_run.v1", "created_at": _now(),
        "task": "CmResidual", "work_version": "V1.18b", "run_id": args.run_id,
        "run_status": "RUNNING", "git_commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "physical_gpu": args.gpu, "logical_gpu": 0, "seed": 42, "num_envs": 128,
        "selection": "same first planner-active Phase-B state as V1.18a",
        "variants": ["baseline", "merge", "link_aabb", "link_sparse"],
        "candidate_count": 8, "warmup": args.warmup, "repeats": args.repeats,
        "inputs": inputs, "command": " ".join(shlex.quote(value) for value in command),
        "profile": str(profile_path), "log": str(log_path), "scientific_conclusion": "INCONCLUSIVE",
    }
    _write(output / "config.json", {"overrides": overrides, "variants": manifest["variants"],
                                     "profiler": {"warmup": args.warmup, "repeats": args.repeats}})
    _write(output / "run_manifest.json", manifest)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "CMRESIDUAL_REFERENCE": str(REFERENCE),
        "CMRESIDUAL_DEXPLORE_SOURCE": str(SOURCE), "REF2DEX_V118B_PROFILE": str(profile_path),
        "REF2DEX_V118B_WARMUP": str(args.warmup), "REF2DEX_V118B_REPEATS": str(args.repeats),
        "REF2DEX_V118B_VENDOR_TRAIN": str(VENDOR / "isaacgymenvs/train.py"),
        "PYTHONPATH": os.pathsep.join((str(ISAAC_GYM), str(ROOT), str(VENDOR), environment.get("PYTHONPATH", ""))),
    })
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode or not profile_path.is_file():
            raise RuntimeError(f"profile process exited {completed.returncode}; profile_exists={profile_path.is_file()}")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if (profile["state_sha256"] != EXPECTED_STATE_SHA or
                profile["candidate_actions_sha256"] != EXPECTED_CANDIDATE_SHA):
            raise RuntimeError("V1.18b did not reproduce the frozen V1.18a state/candidate identity")
        manifest.update(run_status="COMPLETED", completed_at=_now(), profile_sha256=_sha256(profile_path),
                        selected_state_sha256=profile["state_sha256"], conclusion="SUPPORTED")
        _write(output / "run_manifest.json", manifest)
        print(output)
        return 0
    except BaseException as error:
        traceback.print_exc()
        manifest.update(run_status="FAILED", completed_at=_now(),
                        exit_reason=f"{type(error).__name__}: {error}", conclusion="INVALID_IMPLEMENTATION")
        _write(output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
