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
from typing import Mapping

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


def advance_sample_generator(generator, dataset_lengths: Mapping[str, int], config, steps: int) -> None:
    """Restore the V1.14b replacement-sampler position without reading samples."""
    for step in range(int(steps)):
        counts = normalized_batch_counts(
            config["active_groups"], config["group_weights"],
            int(config["training"]["batch_size_per_rank"]), step)
        for group in config["active_groups"]:
            torch.randint(int(dataset_lengths[group]), (counts[group],), generator=generator)


def validate_resume_contract(payload: Mapping, source_manifest: Mapping, config: Mapping) -> int:
    """Reject implicit or cross-contract checkpoint reuse before a retry starts."""
    identity = (payload.get("schema_name"), payload.get("work_version"),
                payload.get("architecture_version"))
    expected = (CHECKPOINT_SCHEMA, "V1.14", config["model"]["architecture_version"])
    if identity != expected:
        raise ValueError("V1.14e.1 resume checkpoint identity mismatch")
    if (payload.get("active_groups") != config["active_groups"]
            or payload.get("group_weights") != config["group_weights"]):
        raise ValueError("V1.14e.1 resume active-group contract mismatch")
    if (source_manifest.get("task") != "ObjectInteractionCmv2"
            or source_manifest.get("work_version") != "V1.14"
            or source_manifest.get("run_status") != "FAILED"
            or source_manifest.get("active_groups") != config["active_groups"]
            or source_manifest.get("group_weights") != config["group_weights"]):
        raise ValueError("V1.14e.1 source run manifest mismatch")
    step = int(payload.get("step", -1))
    planned = int(source_manifest.get("planned_steps", -1))
    if (int(config["training"]["epochs"]) != 1 or int(payload.get("epoch", -1)) != 1
            or step <= 0 or planned <= step):
        raise ValueError("V1.14e.1 only resumes an incomplete single-epoch checkpoint")
    if "model" not in payload or "optimizer" not in payload:
        raise ValueError("V1.14e.1 resume checkpoint is incomplete")
    return step


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


def _checkpoint(output, name, model, optimizer, generator, config, manifest, step, epoch, best, rank):
    if rank == 0:
        temporary = output / f"{name}.tmp"
        torch.save({"schema_name": CHECKPOINT_SCHEMA, "work_version": "V1.14",
                    "architecture_version": model.architecture_version,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "run_id": manifest["run_id"], "git_commit": manifest["git_commit"],
                    "active_groups": config["active_groups"], "group_weights": config["group_weights"],
                    "step": step, "epoch": epoch, "best_metric": best,
                    "sample_generator_state": generator.get_state().cpu(),
                    "torch_rng_state": torch.get_rng_state().cpu(),
                    "cuda_rng_state": torch.cuda.get_rng_state().cpu()}, temporary)
        os.replace(temporary, output / name)
    dist.barrier()


def worker(config_path: Path, output: Path, run_id: str, smoke_steps: int,
           resume_checkpoint: Path | None = None) -> None:
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
        checkpoint_interval = int(config["training"]["checkpoint_interval"])
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        model = initialize_random_v114_model(config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("training", {}).get("learning_rate", 1e-3)))
        wrapped = DistributedDataParallel(model, device_ids=[local_rank])
        generator = torch.Generator().manual_seed(seed + rank * 100003)
        if resume_checkpoint is not None:
            expected = manifest.get("initial_checkpoint") or {}
            if (str(resume_checkpoint.resolve()) != expected.get("path")
                    or sha256_file(resume_checkpoint) != expected.get("sha256")):
                raise ValueError("V1.14e.1 resume checkpoint changed after preparation")
            restored = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            step = validate_resume_contract(restored, manifest["source_run"], config)
            model.load_state_dict(restored["model"], strict=True)
            optimizer.load_state_dict(restored["optimizer"])
            best = restored.get("best_metric")
            if "sample_generator_state" in restored:
                generator.set_state(restored["sample_generator_state"].cpu())
            else:
                advance_sample_generator(generator, {key: len(value) for key, value in train.items()}, config, step)
            if "torch_rng_state" in restored:
                torch.set_rng_state(restored["torch_rng_state"].cpu())
            if "cuda_rng_state" in restored:
                torch.cuda.set_rng_state(restored["cuda_rng_state"].cpu(), device)
        if rank == 0:
            manifest.update(run_status="RUNNING", started_at=utc_now(), train_rows={k: len(v) for k, v in train.items()},
                            steps_per_epoch=steps_per_epoch, planned_steps=steps_per_epoch * epochs,
                            resumed_step=step if resume_checkpoint is not None else None)
            write_json(output / "run_manifest.json", manifest)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            wrapped.train()
            epoch_start = step if resume_checkpoint is not None else 0
            for _ in range(epoch_start, steps_per_epoch):
                samples, counts = sample_batch(train, config, generator, step)
                batch = _move(samples, device); optimizer.zero_grad(set_to_none=True)
                losses = part_se3_v112_loss(wrapped(batch), batch)
                if not torch.isfinite(losses["total"]):
                    raise ValueError(f"non-finite V1.14b loss at step {step + 1}")
                losses["total"].backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                finite = torch.stack([torch.isfinite(value).all() for value in gradients]).all().to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise ValueError(f"non-finite V1.14b gradient at step {step + 1}")
                optimizer.step(); step += 1
                if smoke_steps or step == 1 or step % int(config["training"]["log_interval"]) == 0:
                    value = losses["total"].detach(); dist.all_reduce(value); value /= world_size
                    _emit(output, manifest, rank, {"phase": "training", "last_step": step,
                          "last_epoch": epoch, "loss": float(value), "batch_counts_per_rank": counts,
                          "elapsed_s": time.perf_counter() - started})
                if not smoke_steps and step % checkpoint_interval == 0:
                    _checkpoint(output, "latest.pt", model, optimizer, generator, config, manifest,
                                step, epoch, best, rank)
                    _emit(output, manifest, rank, {"phase": "checkpoint", "last_step": step,
                          "last_epoch": epoch, "best_metric": best})
            result = _evaluate(model, validation, config, device, rank, world_size, bool(smoke_steps))
            improved = best is None or result["selection_metric"] < best
            if improved:
                best = result["selection_metric"]
                _checkpoint(output, "best.pt", model, optimizer, generator, config, manifest,
                            step, epoch, best, rank)
            _checkpoint(output, "latest.pt", model, optimizer, generator, config, manifest,
                        step, epoch, best, rank)
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


def prepare(config_path: Path, output: Path, run_id: str, smoke_steps: int,
            resume_checkpoint: Path | None = None,
            resume_run_manifest: Path | None = None) -> dict:
    config = load_v114_config(config_path, allow_approved_run=not bool(smoke_steps))
    if (resume_checkpoint is None) != (resume_run_manifest is None) or (smoke_steps and resume_checkpoint):
        raise ValueError("V1.14e.1 resume requires checkpoint + source manifest and forbids smoke")
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
    initial_checkpoint = None
    source_run = None
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        resume_run_manifest = resume_run_manifest.resolve()
        if not resume_checkpoint.is_file() or not resume_run_manifest.is_file():
            raise FileNotFoundError("V1.14e.1 resume inputs are incomplete")
        source_run = json.loads(resume_run_manifest.read_text())
        source_config = json.loads(Path(source_run["config"]).read_text())
        if source_config != config or source_run.get("input_sha256") != input_sha256:
            raise ValueError("V1.14e.1 source config or input identity changed")
        restored = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        resumed_step = validate_resume_contract(restored, source_run, config)
        initial_checkpoint = {
            "path": str(resume_checkpoint), "sha256": sha256_file(resume_checkpoint),
            "source_run_manifest": str(resume_run_manifest),
            "source_run_manifest_sha256": sha256_file(resume_run_manifest),
            "source_run_id": source_run["run_id"], "source_git_commit": source_run["git_commit"],
            "step": resumed_step,
        }
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
                "initialization": "random", "initial_checkpoint": initial_checkpoint,
                "source_run": source_run, "smoke_steps": int(smoke_steps),
                "outputs": {"directory": str(output), "metrics": "metrics.jsonl", "latest": "latest.pt", "best": "best.pt"},
                "conclusion": "INCONCLUSIVE"}
    write_json(output / "run_manifest.json", manifest)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker")); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-run-manifest", type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    if args.action == "worker":
        worker(args.config, output, args.run_id, args.smoke_steps, args.resume_checkpoint); return
    config = prepare(args.config, output, args.run_id, args.smoke_steps,
                     args.resume_checkpoint, args.resume_run_manifest)
    gpus = config.get("resources", {}).get("physical_gpus") or list(range(int(config.get("training", {}).get("world_size", 1))))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
               "-m", "src.task.ObjectInteractionCmv2.train_part_se3_v114_ddp", "worker",
               "--config", str(output / "config.json"), "--output", str(output), "--run-id", args.run_id,
               "--smoke-steps", str(args.smoke_steps)]
    if args.resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(args.resume_checkpoint.resolve())))
    raise SystemExit(subprocess.call(command, cwd=REPO_ROOT, env=environment))


if __name__ == "__main__":
    main()
