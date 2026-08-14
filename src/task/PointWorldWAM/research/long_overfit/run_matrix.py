from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[5]
JOBS = (
    ("pretrained_two_seq", True, 2),
    ("scratch_two_seq", False, 2),
    ("pretrained_single_seq", True, 1),
    ("scratch_single_seq", False, 1),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="四卡并行运行 PointWorldWAM 3000-step overfit")
    parser.add_argument("--config", default="src/task/PointWorldWAM/configs/grab_forward_long.yaml")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) < len(JOBS):
        raise ValueError(f"需要至少 {len(JOBS)} 张 GPU，实际为 {gpu_ids}")
    with open(ROOT / args.config, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    processes = []
    for gpu, (name, pretrained, max_sequences) in zip(gpu_ids, JOBS):
        cfg = copy.deepcopy(base)
        cfg["model"]["pretrained"] = pretrained
        cfg["data"]["max_sequences"] = max_sequences
        output_dir = ROOT / "output" / "exp" / "pointworld_wam_forward_long" / name
        cfg["train"]["output_dir"] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        log_handle = open(output_dir / "train.log", "w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = "."
        command = [
            sys.executable,
            "-m",
            "src.task.PointWorldWAM.train",
            "--config",
            str(config_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((name, gpu, process, log_handle))
        print(f"started {name} pid={process.pid} gpu={gpu} log={output_dir / 'train.log'}", flush=True)

    failures = []
    for name, gpu, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        print(f"finished {name} gpu={gpu} returncode={returncode}", flush=True)
        if returncode:
            failures.append((name, returncode))
    if failures:
        raise RuntimeError(f"训练失败: {failures}")


if __name__ == "__main__":
    main()
