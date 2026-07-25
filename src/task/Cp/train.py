"""Train one Cp stage using the shared BaseRunner lifecycle."""
from __future__ import annotations
import argparse
from src.base import cleanup_distributed, load_config
from .runner import CpHumanClosureRunner

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",default="src.task.Cp.config:Config"); p.add_argument("--data"); p.add_argument("--output-dir"); p.add_argument("--device"); p.add_argument("--set",action="append",default=[]); a=p.parse_args()
    cfg=load_config(a.config,overrides=a.set)
    if a.data: cfg.data.train_path=a.data
    if a.output_dir: cfg.train.output_dir=a.output_dir
    if a.device: cfg.train.device=a.device
    if not cfg.data.train_path: raise ValueError("Pass --data or set data.train_path")
    try: CpHumanClosureRunner(cfg,mode="train").run()
    finally: cleanup_distributed()
if __name__ == "__main__": main()
