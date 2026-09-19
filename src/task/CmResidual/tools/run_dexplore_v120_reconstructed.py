"""Run the V1.20 reconstructed DExplore single-sequence smoke or formal training.

The launcher never alters the external DExplore checkout or the converted data.
It only runs a clean, pinned checkout against the isolated V1.20 tensor root.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[4]
DEXPLORE = Path("/home2/wyy/oyx_ws/_external/dexplore_official_v120")
DATA_ROOT = ROOT / "data/processed_data/dexplore_reconstructed_v120/converted_attempt2"
OUTPUT_ROOT = ROOT / "outputs/Dexplore"
ENV_CONFIG = "dexplore/data/cfg/inspire.yaml"
TRAIN_CONFIG = "dexplore/data/cfg/train/rlg/inspire.yaml"
SEQUENCE = "s1_airplane_lift"
HOROVOD_BOOTSTRAP = Path(__file__).resolve().with_name("dexplore_horovod_rank_bootstrap.py")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _gpu_used_mib(gpu: int) -> int:
    rows = _run(["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"], ROOT).splitlines()
    values = {int(a.strip()): int(b.strip()) for a, b in (row.split(",") for row in rows)}
    return values[gpu]


def _check_inputs() -> dict:
    if not (DEXPLORE / "dexplore/run.py").is_file():
        raise FileNotFoundError(f"missing clean DExplore checkout: {DEXPLORE}")
    if _run(["git", "diff", "--name-only"], DEXPLORE) or _run(
            ["git", "diff", "--cached", "--name-only"], DEXPLORE):
        raise RuntimeError("external DExplore tracked source differs from its pinned commit")
    untracked = set(filter(None, _run(["git", "ls-files", "--others", "--exclude-standard"], DEXPLORE).splitlines()))
    allowed_generated = {"data_processing/smplx_vert_segmentation.json",
                         "dexplore/data/assets/mjcf/objects/airplane/airplane.obj",
                         "dexplore/data/assets/mjcf/objects/table/table.obj"}
    if untracked - allowed_generated:
        raise RuntimeError(f"external DExplore has unexpected untracked files: {sorted(untracked - allowed_generated)}")
    manifest_path = DATA_ROOT.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classification") != "reconstructed_baseline":
        raise ValueError("V1.20 requires the explicitly classified reconstructed baseline")
    tensor = DATA_ROOT / SEQUENCE / "interaction_hand_inspire.pt"
    if not tensor.is_file():
        raise FileNotFoundError(f"missing converted sequence: {tensor}")
    import torch
    value = torch.load(tensor, map_location="cpu", weights_only=False)
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (432, 598)
            or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
        raise ValueError("converted tensor must be finite float32 (432,598)")
    return {"source_commit": _run(["git", "rev-parse", "HEAD"], DEXPLORE),
            "source_run_py_sha256": _sha256(DEXPLORE / "dexplore/run.py"),
            "source_untracked_generated": sorted(untracked),
            "data_manifest": str(manifest_path.resolve()),
            "data_manifest_sha256": _sha256(manifest_path),
            "tensor": {"path": str(tensor.resolve()), "sha256": _sha256(tensor),
                       "shape": list(value.shape), "dtype": str(value.dtype)}}


def _command(output: Path, *, num_envs: int, epochs: int, horovod: bool) -> list[str]:
    base = [sys.executable, str(HOROVOD_BOOTSTRAP) if horovod else "dexplore/run.py", "--task", "Dexplore_Inspire",
            "--cfg_env", ENV_CONFIG, "--cfg_train", TRAIN_CONFIG,
            "--motion_file", str(DATA_ROOT), "--output_path", str(output / "train"),
            "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
            "--graphics_device_id", "0", "--num_envs", str(num_envs),
            "--horizon_length", "64", "--minibatch_size", "16384" if horovod else "256",
            "--max_iterations", str(epochs), "--seed", "42"]
    if horovod:
        base.append("--horovod")
        return [str(Path(sys.executable).with_name("horovodrun")), "-np", "4", "-H", "localhost:4"] + base
    return base


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--horovod-4gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    physical_gpus = (0, 1, 3, 7) if args.horovod_4gpu else (args.gpu,)
    if args.horovod_4gpu and args.num_envs != 2048:
        raise ValueError("--horovod-4gpu requires --num-envs 2048 per rank")
    # Preserve the upstream training YAML default.  DExplore exits only after
    # ``epoch_num > max_epochs``, so this executes epochs 1..100001.
    epochs = 1 if args.mode == "smoke" else 100000
    output = OUTPUT_ROOT / args.run_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    inputs = _check_inputs()
    gpu_preflight = {gpu: _gpu_used_mib(gpu) for gpu in physical_gpus}
    if any(used > 1024 for used in gpu_preflight.values()):
        raise RuntimeError(f"GPU capacity gate failed: {gpu_preflight}")
    command = _command(output, num_envs=args.num_envs, epochs=epochs, horovod=args.horovod_4gpu)
    runtime = {"python": sys.executable, "cuda_visible_devices": ",".join(map(str, physical_gpus)),
               "physical_gpus": list(physical_gpus), "horovod": args.horovod_4gpu,
               "gpu_preflight_mib": gpu_preflight}
    if args.dry_run:
        print(json.dumps({"preflight": "passed", "command": command, "inputs": inputs}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True)
    _write(output / "config.json", {"command": command, "runtime": runtime,
                                     "input_references": inputs, "mode": args.mode,
                                     "num_envs": args.num_envs, "epochs": epochs})
    manifest = {"manifest_schema": "ref2dex.run.v1", "created_at": _now(),
                "run_id": args.run_id, "activity_id": args.activity_id,
                "work_version": "V1.20", "operation_category": ["experiment", "operation"],
                "run_status": "STARTED", "output_dir": str(output.resolve()), "command": command,
                "runtime": runtime, "input_references": inputs, "log": str((output / "train.log").resolve()),
                "conclusion": "INCONCLUSIVE"}
    manifest_path = output / "run_manifest.json"
    _write(manifest_path, manifest)
    peak = dict(gpu_preflight)
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = runtime["cuda_visible_devices"]
        env["REF2DEX_DEXPLORE_RUN"] = str((DEXPLORE / "dexplore/run.py").resolve())
        if args.horovod_4gpu:
            env["REF2DEX_GRAD_ACCUM_STEPS"] = "64"
        with (output / "train.log").open("w", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(command, cwd=DEXPLORE, stdout=log, stderr=subprocess.STDOUT,
                                       text=True, env=env)
            while process.poll() is None:
                for gpu in physical_gpus:
                    peak[gpu] = max(peak[gpu], _gpu_used_mib(gpu))
                time.sleep(0.25)
            code = process.wait()
        events = sorted((output / "train").rglob("events.out.tfevents*"))
        checkpoints = sorted((output / "train").rglob("*.pth"))
        if code or not events or not checkpoints:
            raise RuntimeError(f"training exit={code}, events={len(events)}, checkpoints={len(checkpoints)}")
        manifest.update({"run_status": "COMPLETED", "completed_at": _now(), "exit_code": code,
                         "gpu_peak_mib": peak, "tensorboard": str(events[-1].resolve()),
                         "checkpoint": str(checkpoints[-1].resolve())})
        _write(manifest_path, manifest)
        return 0
    except BaseException as error:
        manifest.update({"run_status": "FAILED", "completed_at": _now(), "gpu_peak_mib": peak,
                         "failure_reason": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc(), "conclusion": "INVALID_IMPLEMENTATION"})
        _write(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
