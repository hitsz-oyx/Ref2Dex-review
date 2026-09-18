"""Run the approved V1.17 external DExplore GRAB teacher capacity smoke."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEXPLORE_ROOT = Path("/home2/wyy/oyx_ws/dexplore")
DATA_ROOT = REPOSITORY_ROOT / "data/processed_data/inspire_rl_object_dexplore"
SEQUENCE = "s1_airplane_lift"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/Dexplore"
ENV_CONFIG = "dexplore/data/cfg/inspire.yaml"
TRAIN_CONFIG = "dexplore/data/cfg/train/rlg/inspire.yaml"
SMOKE_GPU = 5
SMOKE_LOGICAL_GPU = 0
SMOKE_HORIZON = 64
SMOKE_MINIBATCH = 256
SMOKE_ITERATIONS = 1
SMOKE_SEED = 42


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _gpu_used_mib(gpu: int) -> int:
    output = _run(["nvidia-smi", "--query-gpu=index,memory.used",
                   "--format=csv,noheader,nounits"], cwd=REPOSITORY_ROOT)
    values = {}
    for line in output.splitlines():
        index, used = (item.strip() for item in line.split(","))
        values[int(index)] = int(used)
    if gpu not in values:
        raise RuntimeError(f"nvidia-smi did not report GPU{gpu}")
    return values[gpu]


def _check_inputs() -> dict:
    manifest_path = DATA_ROOT / "manifest.json"
    if not DEXPLORE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing external DExplore checkout: {DEXPLORE_ROOT}")
    if not (DEXPLORE_ROOT / "dexplore/run.py").is_file():
        raise FileNotFoundError("Missing DExplore training entrypoint")
    if not (DEXPLORE_ROOT / "dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf").is_file():
        raise FileNotFoundError("Missing external DExplore Inspire hand asset")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing DExplore-compatible GRAB manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "dexplore_inspire_rl_simulated_object_v1"
            or manifest.get("num_sequences") != 660):
        raise ValueError("Unexpected DExplore GRAB manifest contract")
    source_samples = sorted(DATA_ROOT.glob("*/interaction_hand_inspire.pt"))
    if len(source_samples) != 660:
        raise ValueError(f"Expected 660 source DExplore GRAB tensors, found {len(source_samples)}")
    sample = DATA_ROOT / SEQUENCE / "interaction_hand_inspire.pt"
    if not sample.is_file():
        raise FileNotFoundError(f"Missing selected DExplore GRAB sequence: {sample}")
    import torch
    tensor = torch.load(sample, map_location="cpu", weights_only=True)
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != 598
            or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
        raise ValueError("DExplore GRAB sample must be a finite float32 (T,598) tensor")
    return {
        "data_manifest": str(manifest_path.resolve()),
        "data_manifest_sha256": _sha256(manifest_path),
        "source_sample_count": len(source_samples),
        "selected_sequence": SEQUENCE,
        "sample": {"path": str(sample.resolve()), "sha256": _sha256(sample),
                   "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
        "external_commit": _run(["git", "rev-parse", "HEAD"], cwd=DEXPLORE_ROOT),
        "external_worktree_dirty": bool(_run(["git", "status", "--porcelain"], cwd=DEXPLORE_ROOT)),
    }


def _check_runtime() -> dict:
    import torch
    rl_games_version = version("rl-games")
    if rl_games_version != "1.1.4":
        raise RuntimeError(
            "DExplore requires rl-games==1.1.4; run the launcher with its isolated runtime"
        )
    return {
        "python_executable": sys.executable,
        "rl_games_version": rl_games_version,
        "torch_version": torch.__version__,
    }


def _motion_input(output: Path) -> Path:
    return output / "motion_input"


def _materialize_motion_input(output: Path) -> dict:
    """Make a per-run directory view that excludes non-motion source folders."""
    motion_input = _motion_input(output)
    motion_input.mkdir()
    sequence_root = DATA_ROOT / SEQUENCE
    (motion_input / SEQUENCE).symlink_to(sequence_root.resolve(), target_is_directory=True)
    return {"path": str(motion_input.resolve()), "kind": "symlink_view", "sample_count": 1,
            "selected_sequence": SEQUENCE}


def _command(output: Path, *, num_envs: int) -> list[str]:
    return [
        sys.executable, "dexplore/run.py",
        "--task", "Dexplore_Inspire",
        "--cfg_env", ENV_CONFIG,
        "--cfg_train", TRAIN_CONFIG,
        "--motion_file", str(_motion_input(output)),
        "--output_path", str(output / "train"),
        "--headless",
        "--sim_device", f"cuda:{SMOKE_LOGICAL_GPU}",
        "--rl_device", f"cuda:{SMOKE_LOGICAL_GPU}",
        "--graphics_device_id", str(SMOKE_LOGICAL_GPU),
        "--num_envs", str(num_envs),
        "--horizon_length", str(SMOKE_HORIZON),
        "--minibatch_size", str(SMOKE_MINIBATCH),
        "--max_iterations", str(SMOKE_ITERATIONS),
        "--seed", str(SMOKE_SEED),
    ]


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--modification-version", required=True, choices=("V1.17",))
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    output = OUTPUT_ROOT / args.run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    inputs = _check_inputs()
    runtime = _check_runtime()
    runtime["cuda_visible_devices"] = str(SMOKE_GPU)
    gpu_used_mib = _gpu_used_mib(SMOKE_GPU)
    if gpu_used_mib > 512:
        raise RuntimeError(f"GPU{SMOKE_GPU} capacity gate failed: {gpu_used_mib} MiB used > 512 MiB")
    command = _command(output, num_envs=args.num_envs)
    config = {
        "external_source": str(DEXPLORE_ROOT),
        "runtime_environment": runtime,
        "environment_config": ENV_CONFIG,
        "training_config": TRAIN_CONFIG,
        "runtime_command": command,
        "runtime_contract": {
            "physical_gpu": SMOKE_GPU,
            "logical_gpu": SMOKE_LOGICAL_GPU,
            "num_envs": args.num_envs,
            "horizon_length": SMOKE_HORIZON,
            "minibatch_size": SMOKE_MINIBATCH,
            "max_iterations": SMOKE_ITERATIONS,
            "seed": SMOKE_SEED,
            "initial_checkpoint": None,
        },
    }
    if args.dry_run:
        print(json.dumps({"run_id": args.run_id, "preflight": "passed", "gpu_used_mib": gpu_used_mib,
                          "command": command}, ensure_ascii=False))
        return 0

    output.mkdir(parents=True)
    motion_input = _materialize_motion_input(output)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    log_path = output / "train.log"
    _write_json(config_path, config)
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": _now(),
        "task": "Dexplore",
        "mode": "capacity_smoke",
        "run_id": args.run_id,
        "activity_id": args.activity_id,
        "run_status": "STARTED",
        "modification_version": args.modification_version,
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output.resolve()),
        "config_snapshot": str(config_path.resolve()),
        "metadata_snapshot": inputs["data_manifest"],
        "input_references": inputs,
        "motion_input": motion_input,
        "runtime_environment": runtime,
        "initial_checkpoint": None,
        "physical_gpu": SMOKE_GPU,
        "logical_gpu": SMOKE_LOGICAL_GPU,
        "gpu_used_mib_preflight": gpu_used_mib,
        "gpu_peak_mib": None,
        "log": str(log_path.resolve()),
        "checkpoint": None,
        "tensorboard": None,
        "command": command,
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)
    gpu_peak_mib = gpu_used_mib
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(SMOKE_GPU)
            process = subprocess.Popen(command, cwd=DEXPLORE_ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, text=True, env=environment)
            while process.poll() is None:
                gpu_peak_mib = max(gpu_peak_mib, _gpu_used_mib(SMOKE_GPU))
                time.sleep(0.25)
            exit_code = process.wait()
        gpu_peak_mib = max(gpu_peak_mib, _gpu_used_mib(SMOKE_GPU))
        checkpoints = sorted((output / "train").rglob("*.pth"))
        events = sorted((output / "train").rglob("events.out.tfevents*"))
        if exit_code != 0:
            raise RuntimeError(f"DExplore subprocess exited with code {exit_code}")
        if not checkpoints or not events:
            raise RuntimeError("DExplore smoke did not produce both checkpoint and TensorBoard event")
        manifest.update({
            "run_status": "COMPLETED",
            "completed_at": _now(),
            "exit_code": exit_code,
            "gpu_peak_mib": gpu_peak_mib,
            "checkpoint": str(checkpoints[-1].resolve()),
            "tensorboard": str(events[-1].resolve()),
            "conclusion": "INCONCLUSIVE",
        })
        _write_json(manifest_path, manifest)
        return 0
    except BaseException as error:
        manifest.update({
            "run_status": "FAILED",
            "completed_at": _now(),
            "failure_reason": f"{type(error).__name__}: {error}",
            "gpu_peak_mib": gpu_peak_mib,
            "traceback": traceback.format_exc(),
            "conclusion": "INVALID_IMPLEMENTATION",
        })
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
