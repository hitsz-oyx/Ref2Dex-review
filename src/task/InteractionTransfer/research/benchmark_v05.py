"""V0.5 轻量 launcher：两个子进程退出后由 OS 回收 PTv3/native 内存。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--train-sequence", action="append", required=True)
    parser.add_argument("--val-sequence", action="append", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-transitions", type=int, default=16)
    parser.add_argument("--train-max-transitions", type=int)
    parser.add_argument("--val-max-transitions", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="output/exp/interaction_transfer_v05")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    common = ["--root", args.root, "--steps", str(args.steps), "--max-transitions",
              str(args.max_transitions), "--device", args.device]
    if args.train_max_transitions:
        common += ["--train-max-transitions", str(args.train_max_transitions)]
    if args.val_max_transitions:
        common += ["--val-max-transitions", str(args.val_max_transitions)]
    for sequence in args.train_sequence:
        common += ["--train-sequence", sequence]
    for sequence in args.val_sequence:
        common += ["--val-sequence", sequence]
    paths = {}
    for model in ("cm", "direct_edge"):
        path = output / f"{model}.json"
        command = [sys.executable, "-m", "src.task.InteractionTransfer.research.train_eval_v05",
                   "--model", model, *common, "--output", str(path)]
        subprocess.run(command, check=True)
        paths[model] = path
    cm = json.loads(paths["cm"].read_text(encoding="utf-8"))
    direct = json.loads(paths["direct_edge"].read_text(encoding="utf-8"))
    summary = {"cm": cm, "direct_edge": direct,
               "cm_direct_edge_ratio": cm["eval"]["gt"]["point_error_m"] /
               max(direct["eval"]["gt"]["point_error_m"], 1e-12)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
