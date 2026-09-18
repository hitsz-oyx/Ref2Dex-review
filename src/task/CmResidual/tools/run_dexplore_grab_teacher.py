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
from typing import Optional

import yaml


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
FORMAL_ITERATIONS = 152
FORMAL_SAVE_FREQUENCY = 38
RESUME_EXTRA_EPOCHS = 1000
RESUME_CHECKPOINT = (
    REPOSITORY_ROOT
    / "outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215"
    / "train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000114.pth"
)
HOROVOD_PHYSICAL_GPUS = (0, 3)
HOROVOD_WORLD_SIZE = len(HOROVOD_PHYSICAL_GPUS)
HOROVOD_TORCH_VERSION = "2.0.1+cu118"
HOROVOD_GPU_CAPACITY_MIB = 4096
CUDA_LIBRARY_DIR = Path("/home2/wyy/CUDA/cuda-12.1/lib64")
HOROVOD_BOOTSTRAP = Path(__file__).resolve().with_name("dexplore_horovod_rank_bootstrap.py")


def _run_settings(
    mode: str,
    *,
    num_envs: int = 2048,
    world_size: int = 1,
    resume_epoch: Optional[int] = None,
    extra_epochs: Optional[int] = None,
) -> dict:
    if mode == "smoke":
        return {"max_iterations": SMOKE_ITERATIONS, "save_frequency": None,
                "target_env_steps": SMOKE_ITERATIONS * world_size * num_envs * SMOKE_HORIZON}
    if mode == "formal":
        if resume_epoch is not None:
            if extra_epochs is None or extra_epochs < 1:
                raise ValueError("resume formal runs require a positive extra epoch budget")
            # DExplore stops after epoch_num > max_epochs, so subtract one from
            # the desired final epoch.
            max_iterations = resume_epoch + extra_epochs - 1
            return {"max_iterations": max_iterations, "save_frequency": FORMAL_SAVE_FREQUENCY,
                    "target_env_steps": extra_epochs * world_size * num_envs * SMOKE_HORIZON,
                    "resume_epoch": resume_epoch, "extra_epochs": extra_epochs,
                    "target_final_epoch": resume_epoch + extra_epochs}
        # DExplore terminates only after epoch_num > max_epochs, so 152 yields
        # 153 completed PPO epochs and 20,054,016 env-steps at 2048x64.
        return {"max_iterations": FORMAL_ITERATIONS, "save_frequency": FORMAL_SAVE_FREQUENCY,
                "target_env_steps": (FORMAL_ITERATIONS + 1) * world_size * num_envs * SMOKE_HORIZON}
    raise ValueError(f"Unknown V1.17 run mode: {mode}")


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


def _checkpoint_info(path: Path) -> dict:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload: {path}")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int):
        raise ValueError(f"Resume checkpoint lacks integer epoch: {path}")
    frame = checkpoint.get("frame")
    last_mean_rewards = checkpoint.get("last_mean_rewards")
    if hasattr(last_mean_rewards, "item"):
        last_mean_rewards = float(last_mean_rewards.item())
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "epoch": epoch,
        "frame": int(frame) if isinstance(frame, int) else frame,
        "last_mean_rewards": last_mean_rewards,
    }


def _check_runtime(*, launcher: str) -> dict:
    import torch
    rl_games_version = version("rl-games")
    if rl_games_version != "1.1.4":
        raise RuntimeError(
            "DExplore requires rl-games==1.1.4; run the launcher with its isolated runtime"
        )
    runtime = {
        "python_executable": sys.executable,
        "rl_games_version": rl_games_version,
        "torch_version": torch.__version__,
    }
    if launcher == "horovod":
        if torch.__version__ != HOROVOD_TORCH_VERSION:
            raise RuntimeError(
                f"Horovod launcher requires torch=={HOROVOD_TORCH_VERSION}, got {torch.__version__}"
            )
        import horovod.torch as hvd
        required = {"mpi": hvd.mpi_built(), "cuda": hvd.cuda_built(), "nccl": hvd.nccl_built()}
        if not all(required.values()):
            raise RuntimeError(f"Horovod build lacks required backends: {required}")
        nccl_lib = Path(sys.prefix) / "lib/python3.8/site-packages/nvidia/nccl/lib"
        if not nccl_lib.is_dir():
            raise RuntimeError(f"Missing isolated NCCL runtime directory: {nccl_lib}")
        runtime.update({"horovod_version": version("horovod"), "horovod_backends": required,
                        "nccl_library_dir": str(nccl_lib)})
    return runtime


def _configure_horovod_library_path() -> str:
    """Expose the isolated NCCL and host CUDA runtime before importing Torch."""
    nccl_lib = Path(sys.prefix) / "lib/python3.8/site-packages/nvidia/nccl/lib"
    if not CUDA_LIBRARY_DIR.is_dir() or not nccl_lib.is_dir():
        raise RuntimeError(
            f"Missing Horovod runtime libraries: cuda={CUDA_LIBRARY_DIR}, nccl={nccl_lib}"
        )
    library_path = ":".join((str(CUDA_LIBRARY_DIR), str(nccl_lib),
                             os.environ.get("LD_LIBRARY_PATH", "")))
    os.environ["LD_LIBRARY_PATH"] = library_path
    return library_path


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


def _command(output: Path, *, num_envs: int, max_iterations: int,
             train_config: str = TRAIN_CONFIG, horovod: bool = False) -> list[str]:
    command = [
        sys.executable, str(HOROVOD_BOOTSTRAP) if horovod else "dexplore/run.py",
        "--task", "Dexplore_Inspire",
        "--cfg_env", ENV_CONFIG,
        "--cfg_train", train_config,
        "--motion_file", str(_motion_input(output)),
        "--output_path", str(output / "train"),
        "--headless",
        "--sim_device", f"cuda:{SMOKE_LOGICAL_GPU}",
        "--rl_device", f"cuda:{SMOKE_LOGICAL_GPU}",
        "--graphics_device_id", str(SMOKE_LOGICAL_GPU),
        "--num_envs", str(num_envs),
        "--horizon_length", str(SMOKE_HORIZON),
        "--minibatch_size", str(SMOKE_MINIBATCH),
        "--max_iterations", str(max_iterations),
        "--seed", str(SMOKE_SEED),
    ]
    if horovod:
        command.append("--horovod")
    return command


def _horovod_command(output: Path, *, num_envs: int, max_iterations: int,
                     train_config: str = TRAIN_CONFIG) -> list[str]:
    horovodrun = Path(sys.executable).with_name("horovodrun")
    return [str(horovodrun), "-np", str(HOROVOD_WORLD_SIZE), "-H",
            f"localhost:{HOROVOD_WORLD_SIZE}"] + _command(
                output, num_envs=num_envs, max_iterations=max_iterations,
                train_config=train_config, horovod=True,
            )


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_formal_train_config(output: Path, *, save_frequency: int,
                               resume_from: Optional[Path] = None) -> Path:
    source = DEXPLORE_ROOT / TRAIN_CONFIG
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    params = config["params"]["config"]
    params["save_frequency"] = save_frequency
    params["save_best_after"] = save_frequency
    if resume_from is not None:
        params["resume_from"] = str(resume_from.resolve())
    path = output / "train_config.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--modification-version", required=True,
                        choices=("V1.17", "V1.17.1", "V1.17.2", "V1.17.3", "V1.17.4", "V1.17.5"))
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--launcher", choices=("single", "horovod"), default="single")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--extra-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.mode == "formal" and args.num_envs != 2048:
        raise ValueError("The approved V1.17 formal run is fixed to 2048 environments")
    if args.resume_checkpoint is not None and args.mode != "formal":
        raise ValueError("--resume-checkpoint is only valid for formal runs")
    if args.extra_epochs is not None and args.resume_checkpoint is None:
        raise ValueError("--extra-epochs requires --resume-checkpoint")
    if args.launcher == "horovod":
        _configure_horovod_library_path()
    output = OUTPUT_ROOT / args.run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    inputs = _check_inputs()
    runtime = _check_runtime(launcher=args.launcher)
    physical_gpus = HOROVOD_PHYSICAL_GPUS if args.launcher == "horovod" else (SMOKE_GPU,)
    runtime["cuda_visible_devices"] = ",".join(str(gpu) for gpu in physical_gpus)
    gpu_used_mib = {gpu: _gpu_used_mib(gpu) for gpu in physical_gpus}
    capacity_limit = HOROVOD_GPU_CAPACITY_MIB if args.launcher == "horovod" else 512
    over_capacity = {gpu: used for gpu, used in gpu_used_mib.items() if used > capacity_limit}
    if over_capacity:
        raise RuntimeError(
            f"GPU capacity gate failed (limit {capacity_limit} MiB): {over_capacity}"
        )
    world_size = HOROVOD_WORLD_SIZE if args.launcher == "horovod" else 1
    initial_checkpoint = None
    if args.resume_checkpoint is not None:
        initial_checkpoint = _checkpoint_info(args.resume_checkpoint)
    settings = _run_settings(
        args.mode,
        num_envs=args.num_envs,
        world_size=world_size,
        resume_epoch=initial_checkpoint["epoch"] if initial_checkpoint else None,
        extra_epochs=(args.extra_epochs or RESUME_EXTRA_EPOCHS) if initial_checkpoint else None,
    )
    train_config = TRAIN_CONFIG
    if not args.dry_run and args.mode == "formal":
        output.mkdir(parents=True)
        train_config = str(_write_formal_train_config(
            output,
            save_frequency=settings["save_frequency"],
            resume_from=args.resume_checkpoint,
        ))
    command = (_horovod_command if args.launcher == "horovod" else _command)(
        output, num_envs=args.num_envs, max_iterations=settings["max_iterations"],
        train_config=train_config,
    )
    config = {
        "external_source": str(DEXPLORE_ROOT),
        "runtime_environment": runtime,
        "environment_config": ENV_CONFIG,
        "training_config": TRAIN_CONFIG,
        "runtime_command": command,
        "runtime_contract": {
            "launcher": args.launcher,
            "physical_gpus": list(physical_gpus),
            "logical_gpus": list(range(world_size)),
            "world_size": world_size,
            "num_envs_per_rank": args.num_envs,
            "total_num_envs": world_size * args.num_envs,
            "horizon_length": SMOKE_HORIZON,
            "minibatch_size": SMOKE_MINIBATCH,
            "mode": args.mode,
            "max_iterations": settings["max_iterations"],
            "target_env_steps": settings["target_env_steps"],
            "save_frequency": settings["save_frequency"],
            "seed": SMOKE_SEED,
            "initial_checkpoint": initial_checkpoint,
        },
    }
    if args.dry_run:
        print(json.dumps({"run_id": args.run_id, "preflight": "passed", "gpu_used_mib": gpu_used_mib,
                          "command": command}, ensure_ascii=False))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    motion_input = _materialize_motion_input(output)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    log_path = output / "train.log"
    _write_json(config_path, config)
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": _now(),
        "task": "Dexplore",
        "mode": f"v117_{args.launcher}_{args.mode}",
        "run_id": args.run_id,
        "activity_id": args.activity_id,
        "run_status": "STARTED",
        "modification_version": args.modification_version,
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output.resolve()),
        "config_snapshot": str(config_path.resolve()),
        "training_config_snapshot": train_config if args.mode == "formal" else None,
        "metadata_snapshot": inputs["data_manifest"],
        "input_references": inputs,
        "motion_input": motion_input,
        "runtime_environment": runtime,
        "initial_checkpoint": initial_checkpoint,
        "launcher": args.launcher,
        "physical_gpus": list(physical_gpus),
        "logical_gpus": list(range(world_size)),
        "world_size": world_size,
        "num_envs_per_rank": args.num_envs,
        "total_num_envs": world_size * args.num_envs,
        "gpu_used_mib_preflight": gpu_used_mib,
        "gpu_peak_mib": None,
        "log": str(log_path.resolve()),
        "checkpoint": None,
        "tensorboard": None,
        "command": command,
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)
    gpu_peak_mib = dict(gpu_used_mib)
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = runtime["cuda_visible_devices"]
            if args.launcher == "horovod":
                nccl_lib = runtime["nccl_library_dir"]
                environment["LD_LIBRARY_PATH"] = str(CUDA_LIBRARY_DIR) + ":" + nccl_lib + ":" + environment.get("LD_LIBRARY_PATH", "")
            process = subprocess.Popen(command, cwd=DEXPLORE_ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, text=True, env=environment)
            while process.poll() is None:
                for gpu in physical_gpus:
                    gpu_peak_mib[gpu] = max(gpu_peak_mib[gpu], _gpu_used_mib(gpu))
                time.sleep(0.25)
            exit_code = process.wait()
        for gpu in physical_gpus:
            gpu_peak_mib[gpu] = max(gpu_peak_mib[gpu], _gpu_used_mib(gpu))
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
