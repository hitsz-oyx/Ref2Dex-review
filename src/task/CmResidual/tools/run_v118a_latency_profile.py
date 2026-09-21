"""Run the approved V1.18a single-state CUDA latency profile."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "third_party/IsaacGymEnvs"
ISAAC_GYM = Path("/home2/wyy/isaac-gym/isaacgym/python")
PYTHON = Path("/home2/wyy/miniconda3/envs/graspenv/bin/python")
BOOTSTRAP = Path(__file__).resolve().with_name("v118a_latency_bootstrap.py")
OUTPUT_ROOT = ROOT / "src/task/CmResidual/research/v118a_latency/output"
REFERENCE = ROOT / "data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/reference.npz"
SOURCE = ROOT / "data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/s1_airplane_lift/interaction_hand_inspire.pt"
CHECKPOINT = ROOT / "outputs/CmResidual/cmresidual_v121_phasea_refppo_gpu6_20260919/train/CmResidualGrabReferenceV118_a/nn/CmResidualGrabReferenceV118PPO_19-21-27-14.pth"
CMV2 = VENDOR / "isaacgymenvs/tasks/cm_residual/latest.pt"
EXPECTED = {
    "checkpoint": "14c2201eec1d98b817426a8280eef733236d9a5eabf573542d0142bf316af7c1",
    "cmv2": "371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591",
    "reference": "a2d710b911cf8988750208450c411b3095e187ed2b2f05df459c24d5748812c1",
    "source": "19b110dc81c4928b4f3e6197d8549b011fd48a8157e75fd06668bd306dc396cf",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _gpu_used_mib(gpu: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    values = {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.splitlines()}
    if gpu not in values:
        raise RuntimeError(f"GPU{gpu} is not reported by nvidia-smi")
    return values[gpu]


def _inputs() -> dict[str, dict[str, str]]:
    paths = {"checkpoint": CHECKPOINT, "cmv2": CMV2, "reference": REFERENCE, "source": SOURCE}
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != EXPECTED[name]:
            raise ValueError(f"{name} SHA256 mismatch: {digest}")
        result[name] = {"path": str(path.resolve()), "sha256": digest}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"cmresidual_v118a_latency_gpu6_{datetime.now():%Y%m%d_%H%M%S}")
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
    dirty = _git("status", "--porcelain")
    if dirty:
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
        "+full_experiment_name=CmResidualV118aLatency", f"hydra.run.dir={output / 'hydra'}",
        "hydra.job.chdir=True",
    ]
    command = [str(PYTHON), str(BOOTSTRAP), *overrides]
    manifest = {
        "manifest_schema": "ref2dex.cmresidual.v118a_latency_run.v1", "created_at": _now(),
        "task": "CmResidual", "work_version": "V1.18a", "run_id": args.run_id,
        "run_status": "RUNNING", "git_commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "physical_gpu": args.gpu, "logical_gpu": 0, "seed": 42, "num_envs": 128,
        "selection": "first planner-active pre-action state in Phase-B rollout",
        "candidate_counts": [1, 2, 4, 8], "warmup": args.warmup, "repeats": args.repeats,
        "inputs": inputs, "command": " ".join(shlex.quote(value) for value in command),
        "profile": str(profile_path), "log": str(log_path), "scientific_conclusion": "INCONCLUSIVE",
    }
    _write(output / "config.json", {"overrides": overrides, "profiler": {"warmup": args.warmup, "repeats": args.repeats}})
    _write(output / "run_manifest.json", manifest)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "CMRESIDUAL_REFERENCE": str(REFERENCE),
        "CMRESIDUAL_DEXPLORE_SOURCE": str(SOURCE), "REF2DEX_V118A_PROFILE": str(profile_path),
        "REF2DEX_V118A_WARMUP": str(args.warmup), "REF2DEX_V118A_REPEATS": str(args.repeats),
        "REF2DEX_V118A_VENDOR_TRAIN": str(VENDOR / "isaacgymenvs/train.py"),
        "PYTHONPATH": os.pathsep.join((str(ISAAC_GYM), str(ROOT), str(VENDOR), environment.get("PYTHONPATH", ""))),
    })
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode or not profile_path.is_file():
            raise RuntimeError(f"profile process exited {completed.returncode}; profile_exists={profile_path.is_file()}")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
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
