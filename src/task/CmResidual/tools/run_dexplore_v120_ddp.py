"""Build, inspect, and launch the isolated V1.20 torchrun DExplore command."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BOOTSTRAP = Path(__file__).resolve().with_name("dexplore_ddp_rank_bootstrap.py")
DEFAULT_DEXPLORE_RUN = REPOSITORY_ROOT / "third_party/DExplore/dexplore/run.py"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/Dexplore"
RUNTIME_ASSETS = (
    (REPOSITORY_ROOT / "data/raw_data/GRAB/objects/airplane/mesh.obj",
     Path("dexplore/data/assets/mjcf/objects/airplane/airplane.obj"),
     "dcbb1cce38e65b3ee608e20f0846cbf93aa3b7bbf863a67582d9944f9a0d64f0"),
    (REPOSITORY_ROOT / "data/raw_data/GRAB/objects/table/mesh.obj",
     Path("dexplore/data/assets/mjcf/objects/table/table.obj"),
     "25c6fb8b774a04f5314a13a538b9c26886716d196d46a68979e817755cf0e383"),
)


def parse_gpus(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if len(values) < 2 or len(set(values)) != len(values) or any(item < 0 for item in values):
        raise ValueError("--gpus must name at least two unique non-negative physical GPU indices")
    return values


def torchrun_command(*, gpus: Sequence[int], dexplore_run: Path, dexplore_args: Sequence[str],
                     bootstrap: Path = BOOTSTRAP, bootstrap_args: Sequence[str] = ()) -> list[str]:
    if not bootstrap.is_file():
        raise FileNotFoundError(f"missing DDP bootstrap: {bootstrap}")
    if not dexplore_run.is_file():
        raise FileNotFoundError(f"missing DExplore entrypoint: {dexplore_run}")
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={len(gpus)}", str(bootstrap),
            "--dexplore-run", str(dexplore_run)] + list(bootstrap_args) + list(dexplore_args)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_runtime_assets(dexplore_run: Path) -> list[dict[str, str | bool]]:
    """Materialize DExplore's ignored mesh inputs from fixed raw GRAB sources.

    DExplore deliberately ignores this asset directory; copying is byte-for-byte,
    refuses unexpected existing content, and never follows a symlink.
    """
    source_root = dexplore_run.resolve().parents[1]
    records = []
    for raw_source, relative_target, expected_sha256 in RUNTIME_ASSETS:
        if not raw_source.is_file() or raw_source.is_symlink():
            raise FileNotFoundError(f"missing regular raw GRAB asset: {raw_source}")
        source_sha256 = _sha256(raw_source)
        if source_sha256 != expected_sha256:
            raise ValueError(f"raw GRAB asset hash mismatch: {raw_source}")
        target = source_root / relative_target
        copied = False
        if target.exists():
            if target.is_symlink() or not target.is_file() or _sha256(target) != expected_sha256:
                raise ValueError(f"refusing to replace unexpected runtime asset: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_source, target, follow_symlinks=False)
            copied = True
        records.append({"raw_source": str(raw_source), "target": str(target),
                        "sha256": expected_sha256, "materialized": copied})
    return records


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def smoke_dexplore_args(*, motion_root: Path, output: Path, num_envs: int,
                        horizon_length: int, minibatch_size: int,
                        max_iterations: int, seed: int) -> list[str]:
    if min(num_envs, horizon_length, minibatch_size, max_iterations) < 1:
        raise ValueError("--num-envs, --horizon-length, --minibatch-size, and --max-iterations must be positive")
    return ["--task", "Dexplore_Inspire", "--cfg_env", "dexplore/data/cfg/inspire.yaml",
            "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
            "--motion_file", str(motion_root), "--output_path", str(output / "train"),
            "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
            "--graphics_device_id", "0", "--num_envs", str(num_envs),
            "--horizon_length", str(horizon_length), "--minibatch_size", str(minibatch_size),
            "--max_iterations", str(max_iterations), "--seed", str(seed), "--horovod"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    parser.add_argument("--dexplore-run", type=Path, default=DEFAULT_DEXPLORE_RUN)
    parser.add_argument("--run-id", help="new output identity; required for --execute")
    parser.add_argument("--motion-root", type=Path, help="converted DExplore tensor root; required for --execute")
    parser.add_argument("--input-manifest", type=Path, help="motion-root provenance manifest; required for --execute")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rank-bootstrap", type=Path, default=BOOTSTRAP,
                        help="Task-local rank bootstrap; defaults to the V1.20 DDP facade")
    parser.add_argument("--cm-distill-coef", type=float,
                        help="only valid with the V1.21 Cm-off bootstrap and must be exactly zero")
    parser.add_argument("--actual-epochs", type=int,
                        help="exact epoch budget for the V1.21 Cm-off bootstrap")
    parser.add_argument("--work-version", default="V1.20")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon-length", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true", help="create a new run directory and execute the smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("dexplore_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        raise ValueError("--execute and --dry-run are mutually exclusive")
    gpus = parse_gpus(args.gpus)
    bootstrap = args.rank_bootstrap.resolve()
    bootstrap_args: list[str] = []
    if args.cm_distill_coef is not None:
        if bootstrap.name != "dexplore_cm_off_rank_bootstrap.py":
            raise ValueError("--cm-distill-coef requires dexplore_cm_off_rank_bootstrap.py")
        if args.cm_distill_coef != 0.0:
            raise ValueError("this launcher only supports Cm-off --cm-distill-coef 0")
        if args.actual_epochs is None or args.actual_epochs < 1:
            raise ValueError("Cm-off launcher requires positive --actual-epochs")
        bootstrap_args = ["--cm-distill-coef", "0", "--actual-epochs", str(args.actual_epochs)]
    elif args.actual_epochs is not None:
        raise ValueError("--actual-epochs requires --cm-distill-coef")
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
        runtime_assets = prepare_runtime_assets(args.dexplore_run)
        dexplore_args = smoke_dexplore_args(motion_root=motion_root, output=output,
                                            num_envs=args.num_envs,
                                            horizon_length=args.horizon_length,
                                            minibatch_size=args.minibatch_size,
                                            max_iterations=args.max_iterations, seed=args.seed)
    else:
        dexplore_args = list(args.dexplore_args)
        if dexplore_args[:1] == ["--"]:
            dexplore_args = dexplore_args[1:]
        output = None
        input_record = None
        manifest = None
        runtime_assets = []
    command = torchrun_command(gpus=gpus, dexplore_run=args.dexplore_run.resolve(), dexplore_args=dexplore_args,
                               bootstrap=bootstrap, bootstrap_args=bootstrap_args)
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
                  "horizon_length": args.horizon_length, "minibatch_size": args.minibatch_size,
                  "max_iterations": args.max_iterations, "seed": args.seed,
                  "rank_bootstrap": str(bootstrap), "cm_distill_coef": args.cm_distill_coef,
                  "actual_epochs": args.actual_epochs,
                  "motion_root": str(args.motion_root.resolve()), "input_manifest": str(manifest),
                  "runtime_assets": runtime_assets}
        _write_json(output / "config.json", config)
        manifest_path = output / "run_manifest.json"
        run_manifest = {
            "manifest_schema": "ref2dex.run.v1", "created_at": _timestamp(),
            "task": "CmResidual", "work_version": args.work_version, "run_id": args.run_id,
            "git_commit": _git_commit(REPOSITORY_ROOT), "external_source_commit": _git_commit(source_root),
            "run_status": "STARTED", "command": command, "runtime": runtime,
            "input_manifest": str(manifest), "input_classification": input_record["classification"],
            "runtime_assets": runtime_assets,
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
