"""Build, inspect, and launch the isolated V1.20 torchrun DExplore command."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BOOTSTRAP = Path(__file__).resolve().with_name("dexplore_ddp_rank_bootstrap.py")
DEFAULT_DEXPLORE_RUN = REPOSITORY_ROOT / "third_party/DExplore/dexplore/run.py"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/Dexplore"


def parse_gpus(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if len(values) < 2 or len(set(values)) != len(values) or any(item < 0 for item in values):
        raise ValueError("--gpus must name at least two unique non-negative physical GPU indices")
    return values


def torchrun_command(*, gpus: Sequence[int], dexplore_run: Path, dexplore_args: Sequence[str]) -> list[str]:
    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(f"missing DDP bootstrap: {BOOTSTRAP}")
    if not dexplore_run.is_file():
        raise FileNotFoundError(f"missing DExplore entrypoint: {dexplore_run}")
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={len(gpus)}", str(BOOTSTRAP),
            "--dexplore-run", str(dexplore_run)] + list(dexplore_args)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def smoke_dexplore_args(*, motion_root: Path, output: Path, num_envs: int,
                        max_iterations: int, seed: int) -> list[str]:
    if num_envs < 1 or max_iterations < 1:
        raise ValueError("--num-envs and --max-iterations must be positive")
    return ["--task", "Dexplore_Inspire", "--cfg_env", "dexplore/data/cfg/inspire.yaml",
            "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
            "--motion_file", str(motion_root), "--output_path", str(output / "train"),
            "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
            "--graphics_device_id", "0", "--num_envs", str(num_envs),
            "--horizon_length", "64", "--minibatch_size", "256",
            "--max_iterations", str(max_iterations), "--seed", str(seed), "--horovod"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    parser.add_argument("--dexplore-run", type=Path, default=DEFAULT_DEXPLORE_RUN)
    parser.add_argument("--run-id", help="new output identity; required for --execute")
    parser.add_argument("--motion-root", type=Path, help="converted DExplore tensor root; required for --execute")
    parser.add_argument("--input-manifest", type=Path, help="motion-root provenance manifest; required for --execute")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="create a new run directory and execute the smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("dexplore_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        raise ValueError("--execute and --dry-run are mutually exclusive")
    gpus = parse_gpus(args.gpus)
    if args.execute:
        if not args.run_id or args.motion_root is None or args.input_manifest is None:
            raise ValueError("--execute requires --run-id, --motion-root, and --input-manifest")
        if args.dexplore_args:
            raise ValueError("--execute constructs the smoke arguments; do not append free-form DExplore arguments")
        motion_root = args.motion_root.resolve()
        manifest = args.input_manifest.resolve()
        if not motion_root.is_dir() or not manifest.is_file():
            raise FileNotFoundError("motion root or input manifest is missing")
        input_record = json.loads(manifest.read_text(encoding="utf-8"))
        if input_record.get("classification") != "reconstructed_baseline":
            raise ValueError("V1.20 smoke requires a reconstructed_baseline input manifest")
        output = (args.output_root / args.run_id).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
        dexplore_args = smoke_dexplore_args(motion_root=motion_root, output=output,
                                            num_envs=args.num_envs,
                                            max_iterations=args.max_iterations, seed=args.seed)
    else:
        dexplore_args = list(args.dexplore_args)
        if dexplore_args[:1] == ["--"]:
            dexplore_args = dexplore_args[1:]
        output = None
        input_record = None
        manifest = None
    command = torchrun_command(gpus=gpus, dexplore_run=args.dexplore_run.resolve(), dexplore_args=dexplore_args)
    source_root = args.dexplore_run.resolve().parents[1]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    environment["PYTHONPATH"] = ":".join((str(Path(__file__).resolve().parent),
                                            str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")))
    print("CUDA_VISIBLE_DEVICES=" + environment["CUDA_VISIBLE_DEVICES"])
    print(" ".join(command))
    if args.execute:
        output.mkdir(parents=True)
        runtime = {"python": sys.executable, "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                   "physical_gpus": list(gpus), "backend": "torch.distributed:nccl",
                   "horovod_package": False, "legacy_cli_facade": True}
        config = {"command": command, "runtime": runtime, "num_envs_per_rank": args.num_envs,
                  "max_iterations": args.max_iterations, "seed": args.seed,
                  "motion_root": str(args.motion_root.resolve()), "input_manifest": str(manifest)}
        _write_json(output / "config.json", config)
        manifest_path = output / "run_manifest.json"
        run_manifest = {
            "manifest_schema": "ref2dex.run.v1", "created_at": _timestamp(),
            "task": "CmResidual", "work_version": "V1.20", "run_id": args.run_id,
            "git_commit": _git_commit(REPOSITORY_ROOT), "external_source_commit": _git_commit(source_root),
            "run_status": "STARTED", "command": command, "runtime": runtime,
            "input_manifest": str(manifest), "input_classification": input_record["classification"],
            "output_dir": str(output), "config": str(output / "config.json"),
            "train_log": str(output / "train.log"), "seed": args.seed,
        }
        _write_json(manifest_path, run_manifest)
        try:
            with (output / "train.log").open("w", encoding="utf-8", buffering=1) as log:
                subprocess.run(command, cwd=source_root, env=environment, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
            events = sorted((output / "train").rglob("events.out.tfevents*"))
            checkpoints = sorted((output / "train").rglob("*.pth"))
            if not events or not checkpoints:
                raise RuntimeError(f"smoke is missing events={len(events)} checkpoints={len(checkpoints)}")
            run_manifest.update({"run_status": "COMPLETED", "completed_at": _timestamp(),
                                 "tensorboard": str(events[-1]), "checkpoint": str(checkpoints[-1])})
            _write_json(manifest_path, run_manifest)
        except BaseException as error:
            run_manifest.update({"run_status": "FAILED", "completed_at": _timestamp(),
                                 "failure_reason": f"{type(error).__name__}: {error}"})
            _write_json(manifest_path, run_manifest)
            raise
    elif not args.dry_run:
        subprocess.run(command, cwd=source_root, env=environment, check=True)


if __name__ == "__main__":
    main()
