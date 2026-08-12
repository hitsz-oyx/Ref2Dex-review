"""四卡构造 V18 stable-grasp tensor event shards。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.task.InteractionDynamics.dataset_grasp_v18 import (
    GraspV18Dataset, load_events)
from src.task.InteractionDynamics.dataset_v17 import split_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world, local = (int(os.getenv(name, default)) for name, default in
                          (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if world > 1: torch.cuda.set_device(local); dist.init_process_group("nccl")
    events = load_events(args.events)
    paths = sorted({event.path for event in events})
    split = split_sequences(paths, 42)
    split_names = {path: name for name in ("train", "val", "test")
                   for path in getattr(split, name)}
    selected = [(index, event) for index, event in enumerate(events) if index % world == rank]
    for local_index, (global_index, event) in enumerate(selected):
        dataset = GraspV18Dataset([event], device=torch.device("cuda", local))
        rows = [dataset[i] for i in range(len(dataset))]
        if not rows: continue
        payload = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        split_name = split_names[event.path]
        folder = args.output / split_name; folder.mkdir(parents=True, exist_ok=True)
        torch.save(payload, folder / f"event_{global_index:05d}.pt")
        if local_index % 50 == 0:
            print(json.dumps({"rank": rank, "events": local_index + 1}), flush=True)
    if world > 1: dist.barrier()
    if rank == 0:
        summary = {name: {"shards": len(list((args.output / name).glob("*.pt")))}
                   for name in ("train", "val", "test")}
        for name in summary:
            summary[name]["samples"] = sum(len(torch.load(path, map_location="cpu")["state"])
                                            for path in (args.output / name).glob("*.pt"))
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
