"""Portable DDP runner for the V1.14a model and V1.14b active-group contract."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .mixed_training import REPO_ROOT, sha256_file, utc_now, write_json
from .part_se3 import collate_part_se3, part_se3_v112_loss
from .part_se3_v114_training import (
    CHECKPOINT_SCHEMA,
    build_part_se3_dataset,
    initialize_random_v114_model,
    load_v114_config,
    normalized_batch_counts,
)


def _move(samples, device):
    batch = collate_part_se3(samples)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def sample_batch(datasets, config, generator, step):
    counts = normalized_batch_counts(
        config["active_groups"], config["group_weights"],
        int(config["training"]["batch_size_per_rank"]), step)
    samples = []
    for group in config["active_groups"]:
        dataset = datasets[group]
        for index in torch.randint(len(dataset), (counts[group],), generator=generator).tolist():
            samples.append({**dataset[index], "hand_variant": group.split("/")[1]})
    return samples, counts


def _emit(output: Path, manifest: dict, rank: int, record: dict) -> None:
    if rank:
        return
    record = {"timestamp": utc_now(), **record}
    line = json.dumps(record, allow_nan=False)
    for name in ("metrics.jsonl", "train.log"):
        with (output / name).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    for key in ("last_step", "last_epoch", "phase", "best_metric"):
        if key in record:
            manifest[key] = record[key]
    manifest["updated_at"] = record["timestamp"]
    write_json(output / "run_manifest.json", manifest)
    print(line, flush=True)


@torch.inference_mode()
def _evaluate(model, datasets, config, device, rank, world_size, smoke):
    model.eval()
    metrics = {}
    for group in config["active_groups"]:
        stride_losses = []
        for stride in config["data"]["validation_strides"]:
            dataset = datasets[f"{group}/stride{stride}"]
            limit = min(len(dataset), 4) if smoke else len(dataset)
            indices = list(range(rank, limit, world_size))
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            for start in range(0, len(indices), 32):
                chosen = indices[start:start + 32]
                if not chosen:
                    continue
                batch = _move([dataset[index] for index in chosen], device)
                loss = part_se3_v112_loss(model(batch), batch)["total"]
                totals += torch.tensor([len(chosen), float(loss) * len(chosen)],
                                       dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            if int(totals[0]) != limit or not limit:
                raise ValueError(f"validation coverage mismatch: {group}/stride{stride}")
            value = float(totals[1] / totals[0])
            metrics[f"{group}/stride{stride}"] = {"samples": limit, "loss": value}
            stride_losses.append(value)
        metrics[group] = sum(stride_losses) / len(stride_losses)
    metrics["selection_metric"] = sum(
        config["group_weights"][group] * metrics[group] for group in config["active_groups"])
    return metrics


def _checkpoint(output, name, model, optimizer, config, manifest, step, epoch, best, rank):
    if rank == 0:
        temporary = output / f"{name}.tmp"
        torch.save({"schema_name": CHECKPOINT_SCHEMA, "work_version": "V1.14",
                    "architecture_version": model.architecture_version,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "run_id": manifest["run_id"], "git_commit": manifest["git_commit"],
                    "active_groups": config["active_groups"], "group_weights": config["group_weights"],
                    "step": step, "epoch": epoch, "best_metric": best}, temporary)
        os.replace(temporary, output / name)
    dist.barrier()


def worker(config_path: Path, output: Path, run_id: str, smoke_steps: int) -> None:
    config = load_v114_config(config_path, allow_approved_run=not bool(smoke_steps))
    rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"]); local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != int(config.get("training", {}).get("world_size", world_size)):
        raise ValueError("torchrun world size differs from V1.14b config")
    torch.set_num_threads(int(config.get("training", {}).get("cpu_threads_per_rank", 8)))
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    manifest = json.loads((output / "run_manifest.json").read_text())
    step = epoch = 0; best = None
    try:
        if sha256_file(config_path) != manifest["config_sha256"]:
            raise ValueError("resolved V1.14b config changed after preparation")
        for path, expected in manifest["input_sha256"].items():
            if sha256_file(path) != expected:
                raise ValueError(f"V1.14b input changed after preparation: {path}")
        seed = int(config.get("training", {}).get("seed", 42))
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        train = {}; validation = {}; sequence_limit = 1 if smoke_steps else None
        for group in config["active_groups"]:
            train[group] = build_part_se3_dataset(config, group, "train", max_sequences=sequence_limit)
            for stride in config["data"]["validation_strides"]:
                validation[f"{group}/stride{stride}"] = build_part_se3_dataset(
                    config, group, "val", fixed_stride=stride, max_sequences=sequence_limit)
        batch_size = int(config.get("training", {}).get("batch_size_per_rank", max(1, len(train))))
        steps_per_epoch = smoke_steps or math.ceil(sum(len(value) for value in train.values()) / (batch_size * world_size))
        epochs = 1 if smoke_steps else int(config["training"]["epochs"])
        model = initialize_random_v114_model(config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("training", {}).get("learning_rate", 1e-3)))
        wrapped = DistributedDataParallel(model, device_ids=[local_rank])
        generator = torch.Generator().manual_seed(seed + rank * 100003)
        if rank == 0:
            manifest.update(run_status="RUNNING", started_at=utc_now(), train_rows={k: len(v) for k, v in train.items()},
                            steps_per_epoch=steps_per_epoch, planned_steps=steps_per_epoch * epochs)
            write_json(output / "run_manifest.json", manifest)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            wrapped.train()
            for _ in range(steps_per_epoch):
                samples, counts = sample_batch(train, config, generator, step)
                batch = _move(samples, device); optimizer.zero_grad(set_to_none=True)
                losses = part_se3_v112_loss(wrapped(batch), batch)
                if not torch.isfinite(losses["total"]):
                    raise ValueError(f"non-finite V1.14b loss at step {step + 1}")
                losses["total"].backward(); optimizer.step(); step += 1
                if smoke_steps or step == 1 or step % int(config["training"]["log_interval"]) == 0:
                    value = losses["total"].detach(); dist.all_reduce(value); value /= world_size
                    _emit(output, manifest, rank, {"phase": "training", "last_step": step,
                          "last_epoch": epoch, "loss": float(value), "batch_counts_per_rank": counts,
                          "elapsed_s": time.perf_counter() - started})
            result = _evaluate(model, validation, config, device, rank, world_size, bool(smoke_steps))
            improved = best is None or result["selection_metric"] < best
            if improved:
                best = result["selection_metric"]
                _checkpoint(output, "best.pt", model, optimizer, config, manifest, step, epoch, best, rank)
            _checkpoint(output, "latest.pt", model, optimizer, config, manifest, step, epoch, best, rank)
            _emit(output, manifest, rank, {"phase": "epoch_complete", "last_step": step,
                  "last_epoch": epoch, "best_metric": best, "validation": result})
        if rank == 0:
            manifest.update(run_status="COMPLETED", finished_at=utc_now(), conclusion="INCONCLUSIVE",
                            last_step=step, last_epoch=epoch, best_metric=best)
            write_json(output / "run_manifest.json", manifest)
    except BaseException as error:
        write_json(output / f"rank{rank}_error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        if rank == 0:
            manifest.update(run_status="FAILED", finished_at=utc_now(), conclusion="INCONCLUSIVE", error=repr(error))
            write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        dist.destroy_process_group()


def prepare(config_path: Path, output: Path, run_id: str, smoke_steps: int) -> dict:
    config = load_v114_config(config_path, allow_approved_run=not bool(smoke_steps))
    if output.exists() or output.name != run_id:
        raise ValueError("V1.14b output must be a new directory named by run_id")
    if not smoke_steps and subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip():
        raise ValueError("formal V1.14b training requires a clean committed worktree")
    split_root = Path(config["split_root"])
    identity_paths = [Path(config["articulation_metadata"]), split_root / "index.json",
                      split_root / "cache_manifest.json"]
    for source in config["sources"].values():
        identity_paths.extend((Path(source["index"]), Path(source["manifest"])))
    if config.get("oakink2_parts"):
        adapter = Path(config["oakink2_parts"]["adapter_root"])
        identity_paths.extend((adapter / "index.json", adapter / "cache_manifest.json"))
    missing = [str(path) for path in identity_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V1.14b inputs are incomplete: {missing}")
    split_index = json.loads((split_root / "index.json").read_text())
    if split_index.get("active_groups") != config["active_groups"]:
        raise ValueError("V1.14b split active_groups differ from the run config")
    input_sha256 = {str(path.resolve()): sha256_file(path) for path in identity_paths}
    output.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    resolved = output / "config.json"; write_json(resolved, config)
    manifest = {"schema_name": "ref2dex_run_manifest_v1", "task": "ObjectInteractionCmv2",
                "work_version": "V1.14", "git_commit": commit, "run_id": run_id,
                "run_status": "STARTED", "created_at": utc_now(),
                "operation": "v114b_active_group_ddp_smoke" if smoke_steps else "v114b_active_group_ddp_train",
                "config": str(resolved), "config_sha256": sha256_file(resolved),
                "input_sha256": input_sha256,
                "active_groups": config["active_groups"], "group_weights": config["group_weights"],
                "hand_sampling": config["hand_sampling"], "coordinates": "object/root reference; metres; seconds; radians",
                "initialization": "random", "initial_checkpoint": None, "smoke_steps": int(smoke_steps),
                "outputs": {"directory": str(output), "metrics": "metrics.jsonl", "latest": "latest.pt", "best": "best.pt"},
                "conclusion": "INCONCLUSIVE"}
    write_json(output / "run_manifest.json", manifest)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker")); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args(); output = args.output.resolve()
    if args.action == "worker":
        worker(args.config, output, args.run_id, args.smoke_steps); return
    config = prepare(args.config, output, args.run_id, args.smoke_steps)
    gpus = config.get("resources", {}).get("physical_gpus") or list(range(int(config.get("training", {}).get("world_size", 1))))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
               "-m", "src.task.ObjectInteractionCmv2.train_part_se3_v114_ddp", "worker",
               "--config", str(output / "config.json"), "--output", str(output), "--run-id", args.run_id,
               "--smoke-steps", str(args.smoke_steps)]
    raise SystemExit(subprocess.call(command, cwd=REPO_ROOT, env=environment))


if __name__ == "__main__":
    main()
