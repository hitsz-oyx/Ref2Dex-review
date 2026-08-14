from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[5]
JOBS = ("pretrained_two_seq", "scratch_two_seq", "scratch_single_seq")


def main() -> None:
    parser = argparse.ArgumentParser(description="续训未在 3000 step 收敛的 PointWorldWAM 实验")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) < len(JOBS):
        raise ValueError(f"需要至少 {len(JOBS)} 张 GPU，实际为 {gpu_ids}")
    processes = []
    for gpu, name in zip(gpu_ids, JOBS):
        output_dir = ROOT / "output" / "exp" / "pointworld_wam_forward_long" / name
        config_path = output_dir / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["train"]["steps"] = args.steps
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        log_handle = open(output_dir / "continue.log", "w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = "."
        command = [
            sys.executable,
            "-m",
            "src.task.PointWorldWAM.train",
            "--config",
            str(config_path),
            "--resume",
            str(output_dir / "best.pt"),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((name, gpu, process, log_handle))
        print(f"continued {name} pid={process.pid} gpu={gpu}", flush=True)
    failures = []
    for name, gpu, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        print(f"finished {name} gpu={gpu} returncode={returncode}", flush=True)
        if returncode:
            failures.append((name, returncode))
    if failures:
        raise RuntimeError(f"续训失败: {failures}")


if __name__ == "__main__":
    main()
