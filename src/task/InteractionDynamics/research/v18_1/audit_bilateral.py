"""审计 V18 event 在 t_g 附近是否同时存在另一只手接触。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.task.InteractionDynamics.dataset_grasp_v18 import load_events, sequence_interaction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world, local = (int(os.getenv(name, default)) for name, default in
                          (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if world > 1: torch.cuda.set_device(local); dist.init_process_group("nccl")
    events = load_events(args.events); rows = []
    for index in range(rank, len(events), world):
        event = events[index]
        other = event.path.with_name("left.npz" if event.path.stem == "right" else "right.npz")
        interaction = sequence_interaction(other, device=torch.device("cuda", local))
        start, end = max(0, event.grasp_frame - 2), min(len(interaction["contact_count"]), event.grasp_frame + 3)
        other_contact = int(interaction["contact_count"][start:end].max())
        rows.append({"event_index": index, "path": str(event.path),
                     "grasp_frame": event.grasp_frame, "other_contact_max": other_contact,
                     "group": "single_hand_clean" if other_contact < 4 else "bilateral"})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
    if world > 1: dist.barrier()
    if rank == 0:
        merged = sum((json.loads((args.output / f"rank{i}.json").read_text())
                      for i in range(world)), [])
        merged.sort(key=lambda row: row["event_index"])
        counts = {group: sum(row["group"] == group for row in merged)
                  for group in ("single_hand_clean", "bilateral")}
        result = {"events": len(merged), "counts": counts,
                  "ratios": {key: value / len(merged) for key, value in counts.items()}}
        (args.output / "events.json").write_text(json.dumps(merged, indent=2) + "\n")
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
