from __future__ import annotations
import argparse
from src.base import cleanup_distributed, load_checkpoint, task_config_from_dict
from .runner import CpHumanClosureRunner
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--device",default="auto"); a=p.parse_args()
    cfg=task_config_from_dict(load_checkpoint(a.checkpoint,map_location="cpu")["config"]); cfg.train.device=a.device
    try: CpHumanClosureRunner(cfg,mode="eval",checkpoint=a.checkpoint).run()
    finally: cleanup_distributed()
if __name__ == "__main__": main()
