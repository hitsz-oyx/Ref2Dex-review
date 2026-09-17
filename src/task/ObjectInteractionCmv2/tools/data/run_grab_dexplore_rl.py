#!/usr/bin/env python3
"""Run DExplore's teacher policy without modifying the external checkout.

DExplore resolves its Isaac Gym assets relative to the process working
directory.  This launcher builds a run-local asset overlay containing symlinks
to the checked-in robot assets and to this run's generated GRAB object meshes,
then starts one batch process per selected GPU.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path


MOTION_FILENAME = "interaction_hand_inspire.pt"


def _prepare_runtime(runtime_root: Path, dexplore_root: Path, object_root: Path) -> None:
    if runtime_root.exists():
        raise FileExistsError(f"Runtime root already exists: {runtime_root}")
    source_assets = dexplore_root / "dexplore" / "data" / "assets"
    target_assets = runtime_root / "dexplore" / "data" / "assets"
    target_assets.mkdir(parents=True)
    for source in sorted(source_assets.iterdir()):
        if source.name == "mjcf":
            continue
        (target_assets / source.name).symlink_to(source, target_is_directory=source.is_dir())
    target_mjcf = target_assets / "mjcf"
    target_mjcf.mkdir()
    for source in sorted((source_assets / "mjcf").iterdir()):
        if source.name == "objects" or source.is_dir():
            continue
        (target_mjcf / source.name).symlink_to(source)
    (target_mjcf / "objects").symlink_to(object_root, target_is_directory=True)


def _run_batch(
    batch_id: int,
    paths: list[Path],
    gpu: str,
    args: argparse.Namespace,
    batch_root: Path,
) -> int:
    batch_dir = batch_root / f"batch_{batch_id:04d}"
    batch_dir.mkdir(parents=True)
    for path in paths:
        (batch_dir / path.name).symlink_to(path, target_is_directory=True)
    log_path = args.output_root / "_logs" / f"batch_{batch_id:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_py = args.dexplore_root / "dexplore" / "run.py"
    command = [
        str(args.python),
        str(run_py),
        "--task",
        "Dexplore_Inspire",
        "--cfg_env",
        str(args.dexplore_root / "dexplore/data/cfg/inspire.yaml"),
        "--cfg_train",
        str(args.dexplore_root / "dexplore/data/cfg/train/rlg/inspire.yaml"),
        "--test",
        "--checkpoint",
        str(args.checkpoint),
        "--headless",
        "--motion_file",
        str(batch_dir),
        "--num_envs",
        str(len(paths)),
        "--seed",
        str(args.seed),
        "--export_rl",
        "--export_output_dir",
        str(args.output_root),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONPATH"] = str(args.dexplore_root / "dexplore")
    environment["PATH"] = str(args.python.parent) + os.pathsep + environment.get("PATH", "")
    environment_root = args.python.parent.parent
    environment["LD_LIBRARY_PATH"] = (
        str(environment_root / "lib")
        + os.pathsep
        + environment.get("LD_LIBRARY_PATH", "")
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=args.runtime_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometric-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--dexplore-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.geometric_root = args.geometric_root.resolve()
    args.output_root = args.output_root.resolve()
    args.object_root = args.object_root.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.dexplore_root = args.dexplore_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.python = args.python.resolve()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU index")
    all_paths = sorted(
        path
        for path in args.geometric_root.iterdir()
        if path.is_dir() and (path / MOTION_FILENAME).is_file()
    )
    if not all_paths:
        raise FileNotFoundError(f"No geometric trajectories below {args.geometric_root}")
    paths = list(all_paths)
    if args.runtime_root.exists() or args.output_root.exists():
        if not args.resume:
            raise FileExistsError(
                f"Runtime or RL output root already exists: {args.runtime_root}, {args.output_root}"
            )
        if not args.runtime_root.is_dir() or not args.output_root.is_dir():
            raise FileNotFoundError("--resume requires existing runtime and RL output directories")
        paths = [
            path
            for path in paths
            if not (args.output_root / path.name / MOTION_FILENAME).is_file()
        ]
    else:
        _prepare_runtime(args.runtime_root, args.dexplore_root, args.object_root)
        args.output_root.mkdir(parents=True)
    if not paths:
        print("All geometric trajectories already have RL exports")
        return
    batches = [paths[index:index + args.batch_size] for index in range(0, len(paths), args.batch_size)]
    batch_root = Path(tempfile.mkdtemp(prefix="cmv2-dexplore-batches-", dir=args.runtime_root))
    pending = list(enumerate(batches))
    active: list[tuple[threading.Thread, dict[str, int | None], int, str]] = []

    def launch(batch_id: int, batch: list[Path], gpu: str) -> None:
        result: dict[str, int | None] = {"code": None}

        def target() -> None:
            result["code"] = _run_batch(batch_id, batch, gpu, args, batch_root)

        thread = threading.Thread(target=target, daemon=False)
        thread.start()
        active.append((thread, result, batch_id, gpu))

    for gpu in gpus:
        if not pending:
            break
        batch_id, batch = pending.pop(0)
        launch(batch_id, batch, gpu)
    failed = []
    while active:
        finished = None
        for index, (thread, result, batch_id, gpu) in enumerate(active):
            if not thread.is_alive():
                thread.join()
                finished = (index, result, batch_id, gpu)
                break
        if finished is None:
            time.sleep(1.0)
            continue
        index, result, batch_id, gpu = finished
        active.pop(index)
        code = int(result["code"] if result["code"] is not None else -1)
        print(f"batch {batch_id} on GPU {gpu}: return code {code}", flush=True)
        if code != 0:
            failed.append(batch_id)
        elif pending:
            next_id, next_batch = pending.pop(0)
            launch(next_id, next_batch, gpu)
    if failed:
        raise RuntimeError(f"DExplore RL batches failed: {failed}")
    exported = sorted(args.output_root.glob(f"*/{MOTION_FILENAME}"))
    if len(exported) != len(all_paths):
        raise RuntimeError(f"Expected {len(all_paths)} RL exports, found {len(exported)}")
    print(f"Exported {len(exported)} RL trajectories")


if __name__ == "__main__":
    main()
