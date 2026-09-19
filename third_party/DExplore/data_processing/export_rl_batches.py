"""Run ``checkpoint/inspire.pth`` over geometric Inspire motions in batches.

The task already contains the Isaac Gym environment and policy player.  This
orchestrator only makes temporary per-batch motion directories and launches
one headless Dexplore process per available GPU; the player hook writes native
``interaction_hand_inspire.pt`` files to ``--output-root``.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def run_batch(batch_id, paths, gpu, args, temp_root):
    batch_dir = temp_root / f"batch_{batch_id:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        link = batch_dir / path.name
        if not link.exists():
            link.symlink_to(path, target_is_directory=True)
    log_path = Path(args.output_root) / "_logs" / f"batch_{batch_id:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        args.python, str(repo / "dexplore" / "run.py"),
        "--task", "Dexplore_Inspire",
        "--cfg_env", str(repo / "dexplore/data/cfg/inspire.yaml"),
        "--cfg_train", str(repo / "dexplore/data/cfg/train/rlg/inspire.yaml"),
        "--test", "--checkpoint", str(Path(args.checkpoint).resolve()), "--headless",
        "--motion_file", str(batch_dir), "--num_envs", str(len(paths)),
        "--export_rl", "--export_output_dir", str(Path(args.output_root).resolve()),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(repo / "dexplore")
    with open(log_path, "w") as log:
        return subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometric-root", required=True,
                        help="Directory containing geometric sequence subdirectories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", default="checkpoint/inspire.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", default="/home2/wyy/miniconda3/envs/graspenv/bin/python")
    args = parser.parse_args()

    geometric_root = Path(args.geometric_root)
    paths = sorted(p for p in geometric_root.iterdir()
                   if p.is_dir() and (p / "interaction_hand_inspire.pt").exists())
    if not paths:
        raise FileNotFoundError(f"No geometric Inspire motions under {geometric_root}")
    batches = [paths[i:i + args.batch_size] for i in range(0, len(paths), args.batch_size)]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device")
    temp_root = Path(tempfile.mkdtemp(prefix="dexplore_rl_batches-"))
    try:
        # Keep at most one Isaac Gym process per selected GPU.  Running a new
        # batch on the same GPU begins as soon as its previous one completes.
        # Keep one process per GPU and, when one finishes, reuse that same
        # device for the next batch.  Waiting for the first *started* batch
        # (rather than the first list entry) can otherwise leave a fast GPU
        # idle while a slower one is still compiling Isaac Gym.
        import threading
        pending = list(enumerate(batches))
        active = []

        def launch(batch_id, batch, gpu):
            result = {"code": None}
            thread = threading.Thread(
                target=lambda: result.update(code=run_batch(batch_id, batch, gpu, args, temp_root)),
                daemon=True)
            thread.start()
            active.append((thread, result, batch_id, gpu))

        for gpu in gpus:
            if not pending:
                break
            batch_id, batch = pending.pop(0)
            launch(batch_id, batch, gpu)

        while active:
            finished = None
            for i, (thread, result, batch_id, gpu) in enumerate(active):
                if not thread.is_alive():
                    thread.join()
                    finished = (i, result, batch_id, gpu)
                    break
            if finished is None:
                time.sleep(1.0)
                continue
            i, result, batch_id, gpu = finished
            active.pop(i)
            print(f"batch {batch_id} on GPU {gpu}: return code {result['code']}", flush=True)
            if result["code"] != 0:
                print(f"  see {Path(args.output_root) / '_logs' / f'batch_{batch_id:04d}.log'}", flush=True)
            if pending:
                next_id, next_batch = pending.pop(0)
                launch(next_id, next_batch, gpu)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
