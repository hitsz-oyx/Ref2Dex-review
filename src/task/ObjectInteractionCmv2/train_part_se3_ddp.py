"""Frozen-source two-GPU training for the approved V1.12 direct-part-SE(3) model."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .mixed_training import (
    GROUPS,
    GROUP_WEIGHTS,
    REPO_ROOT,
    batch_counts,
    sample_batch,
    sha256_file,
    utc_now,
    write_json,
)
from .part_se3 import PART_SE3_VERSION, collate_part_se3, part_se3_v112_loss
from .part_se3_training import (
    CHECKPOINT_SCHEMA,
    build_part_se3_dataset,
    initialize_random_model,
    load_part_se3_training_config,
)


def validation_indices(count: int, rank: int, world_size: int, limit: int | None = None) -> list[int]:
    return list(range(rank, min(count, limit) if limit else count, world_size))


def move_batch(samples, device):
    batch = collate_part_se3(samples)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def emit(output: Path, manifest: dict, rank: int, record: dict) -> None:
    if rank:
        return
    record = {"timestamp": utc_now(), **record}
    line = json.dumps(record, allow_nan=False)
    for name in ("metrics.jsonl", "train.log"):
        with (output / name).open("a") as handle:
            handle.write(line + "\n")
    for key in ("last_step", "last_epoch", "phase", "best_metric"):
        if key in record:
            manifest[key] = record[key]
    manifest["updated_at"] = record["timestamp"]
    write_json(output / "run_manifest.json", manifest)
    print(line, flush=True)


def save_checkpoint(output: Path, name: str, model, optimizer, generator, config, manifest,
                    step: int, epoch: int, best: float | None, rank: int, world_size: int) -> None:
    state = {"sampler": generator.get_state(), "cpu": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state()}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, state)
    if rank == 0:
        temporary = output / (name + ".tmp")
        torch.save({
            "schema_name": CHECKPOINT_SCHEMA,
            "work_version": config["work_version"],
            "architecture_version": PART_SE3_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "run_id": manifest["run_id"],
            "git_commit": manifest["git_commit"],
            "source_sha256": manifest["source_sha256"],
            "input_sha256": manifest["input_sha256"],
            "step": step,
            "epoch": epoch,
            "best_metric": best,
            "seed": config["training"]["seed"],
            "rng_states": gathered,
            "world_size": world_size,
            "batch_size_per_rank": config["training"]["batch_size_per_rank"],
            "selection_metric": "weighted_five_group_mean_of_three_stride_total_losses",
        }, temporary)
        temporary.replace(output / name)
    dist.barrier()


def _reduce(values: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(values)
    return values


@torch.inference_mode()
def evaluate_group(model, dataset, indices, device) -> dict[str, float | int | dict]:
    totals = torch.zeros(10, dtype=torch.float64, device=device)
    for offset in range(0, len(indices), 64):
        chosen = indices[offset:offset + 64]
        if not chosen:
            continue
        batch = move_batch([dataset[index] for index in chosen], device)
        prediction = model(batch)
        losses = part_se3_v112_loss(prediction, batch)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise ValueError("non-finite V1.12 validation loss")
        epe = torch.linalg.vector_norm(prediction["obj_flow_pred"] - batch["obj_flow_gt"], dim=-1)
        part_count = batch["part_valid_mask"].sum(1)
        size = len(chosen)
        totals[0] += size
        totals[1] += losses["total"].to(torch.float64) * size
        totals[2] += losses["translation"].to(torch.float64) * size
        totals[3] += losses["rotation"].to(torch.float64) * size
        totals[4] += losses["flow"].to(torch.float64) * size
        totals[5] += epe.sum().to(torch.float64)
        totals[6] += epe.numel()
        totals[7] += (part_count == 1).sum()
        totals[8] += (part_count == 2).sum()
        totals[9] += (part_count >= 3).sum()
    values = _reduce(totals).cpu().tolist()
    samples = int(round(values[0]))
    if samples <= 0 or values[6] <= 0:
        raise ValueError("empty V1.12 validation shard")
    return {
        "samples": samples,
        "loss": values[1] / samples,
        "translation_loss": values[2] / samples,
        "rotation_geodesic_rad": values[3] / samples,
        "flow_loss": values[4] / samples,
        "flow_epe_mm": 1000.0 * values[5] / values[6],
        "part_count_groups": {"one": int(round(values[7])), "two": int(round(values[8])),
                              "three_plus": int(round(values[9]))},
    }


@torch.inference_mode()
def evaluate(model, datasets, config, device, rank, world_size, output, manifest, smoke):
    model.eval()
    result = {}
    for group in GROUPS:
        for stride in config["data"]["validation_strides"]:
            key = f"{group}/stride{stride}"
            dataset = datasets[key]
            indices = validation_indices(len(dataset), rank, world_size, 4 if smoke else None)
            emit(output, manifest, rank, {"phase": "validation", "group": key,
                                          "local_transitions": len(indices)})
            metrics = evaluate_group(model, dataset, indices, device)
            expected = min(len(dataset), 4) if smoke else len(dataset)
            if metrics["samples"] != expected:
                raise RuntimeError(f"validation coverage mismatch for {key}: {metrics['samples']} != {expected}")
            result[key] = metrics
            emit(output, manifest, rank, {"phase": "validation", "group": key, **metrics})
    result["selection_metric"] = sum(
        GROUP_WEIGHTS[group] * sum(result[f"{group}/stride{stride}"]["loss"] for stride in (1, 2, 3)) / 3
        for group in GROUPS)
    return result


def check_input_identity(output: Path, manifest: dict) -> None:
    if sha256_file(output / "config.json") != manifest["config_sha256"]:
        raise ValueError("resolved V1.12 configuration changed after preparation")
    for path, expected in manifest["input_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"input changed after V1.12 run preparation: {path}")
    for relative, expected in manifest["source_sha256"].items():
        if sha256_file(output / "source_snapshot" / relative) != expected:
            raise ValueError(f"V1.12 source snapshot changed: {relative}")


def run_worker(output_value: str | Path) -> None:
    output = Path(output_value).resolve()
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "run_manifest.json").read_text())
    rank, world_size, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    physical_gpus = tuple(config["resources"]["physical_gpus"])
    if world_size != 2 or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, physical_gpus)):
        raise ValueError("V1.12 formal resources are exactly physical GPUs 0 and 2 with two ranks")
    torch.set_num_threads(config["training"]["cpu_threads_per_rank"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.mem_get_info(device)[0] < config["training"]["minimum_free_memory_gib"] * 2**30:
        raise RuntimeError("insufficient free GPU memory; refusing to alter V1.12 batch or precision")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    step, epoch, best = 0, 0, None
    try:
        check_input_identity(output, manifest)
        torch.manual_seed(config["training"]["seed"])
        torch.cuda.manual_seed_all(config["training"]["seed"])
        smoke = manifest["smoke_steps"] > 0
        manifest.update(run_status="RUNNING", phase="loading_datasets", started_at=utc_now())
        emit(output, manifest, rank, {"phase": "loading_datasets", "last_step": 0, "last_epoch": 0})
        train, validation = {}, {}
        sequence_limit = 1 if smoke and config.get("data_backend", "reference") == "reference" else None
        for group in GROUPS:
            train[group] = build_part_se3_dataset(
                config, group, "train", max_sequences=sequence_limit)
            emit(output, manifest, rank, {"phase": "loading_datasets", "group": group,
                                          "train_rows": len(train[group])})
            for stride in config["data"]["validation_strides"]:
                validation[f"{group}/stride{stride}"] = build_part_se3_dataset(
                    config, group, "val", fixed_stride=stride, max_sequences=sequence_limit)
        steps_per_epoch = math.ceil(sum(len(dataset) for dataset in train.values()) / 128)
        if smoke:
            steps_per_epoch = manifest["smoke_steps"]
        epochs = 1 if smoke else config["training"]["epochs"]
        model = initialize_random_model(config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
        wrapped = DistributedDataParallel(model, device_ids=[local_rank])
        generator = torch.Generator().manual_seed(config["training"]["seed"] + 100003 * rank)
        manifest.update(
            train_rows={group: len(dataset) for group, dataset in train.items()},
            val_rows={group: len(dataset) for group, dataset in validation.items()},
            dropped_train_timeline_transitions={
                group: getattr(dataset, "dropped_timeline_transitions", 0)
                for group, dataset in train.items()},
            steps_per_epoch=steps_per_epoch,
            planned_steps=steps_per_epoch * epochs,
            initialization="random; optimizer/step/epoch/best start from zero",
        )
        emit(output, manifest, rank, {"phase": "training", "steps_per_epoch": steps_per_epoch,
                                      "planned_steps": steps_per_epoch * epochs,
                                      "optimizer_state_entries_at_start": len(optimizer.state)})
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            wrapped.train()
            epoch_started = time.perf_counter()
            for position in range(steps_per_epoch):
                batch = move_batch(sample_batch(train, generator, step), device)
                optimizer.zero_grad(set_to_none=True)
                losses = part_se3_v112_loss(wrapped(batch), batch)
                finite = torch.isfinite(losses["total"]).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise ValueError(f"non-finite V1.12 training loss at step {step + 1}")
                losses["total"].backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                finite = torch.stack([torch.isfinite(value).all() for value in gradients]).all().to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise ValueError(f"non-finite V1.12 gradient at step {step + 1}")
                optimizer.step()
                step += 1
                if step == 1 or step % config["training"]["log_interval"] == 0 or smoke:
                    values = torch.stack([value.detach() for value in losses.values()])
                    dist.all_reduce(values)
                    memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                                           torch.cuda.max_memory_reserved(device) / 2**30], device=device)
                    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                    emit(output, manifest, rank, {
                        "phase": "training", "last_step": step, "last_epoch": epoch,
                        "position_in_epoch": position + 1, "elapsed_s": time.perf_counter() - started,
                        "epoch_elapsed_s": time.perf_counter() - epoch_started,
                        "losses": dict(zip(losses, (values / world_size).cpu().tolist())),
                        "batch_counts_per_rank": batch_counts(step - 1),
                        "peak_allocated_gib": float(memory[0]), "peak_reserved_gib": float(memory[1]),
                    })
                if step == 1 or step % config["training"]["checkpoint_interval"] == 0:
                    save_checkpoint(output, "latest.pt", model, optimizer, generator, config, manifest,
                                    step, epoch, best, rank, world_size)
                del losses, batch
            validation_result = evaluate(
                model, validation, config, device, rank, world_size, output, manifest, smoke)
            improved = best is None or validation_result["selection_metric"] < best
            if improved:
                best = validation_result["selection_metric"]
                save_checkpoint(output, "best.pt", model, optimizer, generator, config, manifest,
                                step, epoch, best, rank, world_size)
            save_checkpoint(output, "latest.pt", model, optimizer, generator, config, manifest,
                            step, epoch, best, rank, world_size)
            emit(output, manifest, rank, {"phase": "epoch_complete", "last_step": step,
                                          "last_epoch": epoch, "best_metric": best,
                                          "improved": improved, "validation": validation_result})
        if smoke:
            restored = torch.load(output / "latest.pt", map_location="cpu")
            if restored.get("schema_name") != CHECKPOINT_SCHEMA:
                raise RuntimeError("V1.12 smoke checkpoint schema mismatch")
            for name, value in model.state_dict().items():
                if not torch.equal(value.cpu(), restored["model"][name]):
                    raise RuntimeError(f"V1.12 checkpoint roundtrip mismatch: {name}")
            gathered = [None] * world_size
            dist.all_gather_object(gathered, sha256_file(output / "latest.pt"))
            if len(set(gathered)) != 1:
                raise RuntimeError("V1.12 ranks observed different checkpoint bytes")
            emit(output, manifest, rank, {"phase": "smoke_verified", "checkpoint_roundtrip": True,
                                          "ranks_observed_same_checkpoint": True})
        manifest.update(run_status="COMPLETED", finished_at=utc_now(), last_step=step,
                        last_epoch=epoch, best_metric=best)
        emit(output, manifest, rank, {"phase": "completed"})
    except BaseException as error:
        write_json(output / f"rank{rank}_error.json", {
            "timestamp": utc_now(), "error": repr(error), "traceback": traceback.format_exc()})
        if rank == 0:
            manifest.update(run_status="FAILED", finished_at=utc_now(), last_step=step,
                            last_epoch=epoch, best_metric=best, error=repr(error),
                            conclusion="INCONCLUSIVE")
            write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        dist.destroy_process_group()


def prepare(config_path: str | Path, run_id: str, smoke_steps: int) -> Path:
    if Path(run_id).name != run_id or not run_id or smoke_steps < 0:
        raise ValueError("invalid V1.12 run ID or smoke budget")
    config = load_part_se3_training_config(config_path)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    if status:
        raise ValueError("V1.12 run preparation requires a clean committed worktree")
    split_root = Path(config["split_root"])
    split_manifest = json.loads((split_root / "cache_manifest.json").read_text())
    inputs = dict(split_manifest["source_sha256"])
    for path, expected in inputs.items():
        if sha256_file(path) != expected:
            raise ValueError(f"source index/cache manifest changed since split: {path}")
    adapter_root = Path(config["oakink2_parts"]["adapter_root"])
    required = [adapter_root / name for name in ("cache_manifest.json", "index.json", "run_manifest.json")]
    adapter_manifest, adapter_index, adapter_run = (json.loads(path.read_text()) for path in required)
    if (adapter_manifest.get("schema_name") != "ref2dex_cmv2_oakink2_part_adapter_v1"
            or adapter_index.get("schema_name") != "ref2dex_cmv2_oakink2_part_adapter_v1"
            or adapter_run.get("run_status") != "COMPLETED"
            or int(adapter_manifest.get("validation", {}).get("bad_count", 1) or 0) != 0):
        raise ValueError("OakInk2 part adapter validation failed")
    identity_paths = [config["articulation_metadata"], str(split_root / "index.json"),
                      str(split_root / "cache_manifest.json"), *(str(path) for path in required)]
    for source in config["sources"].values():
        identity_paths.extend((source["index"], source["manifest"]))
    for path in identity_paths:
        inputs[path] = sha256_file(path)
    if config.get("data_backend") == "compact":
        from .compact_endpoint import CACHE_SCHEMA
        compact_root = Path(config["compact_cache_root"])
        compact_manifest_path = compact_root / "manifest.json"
        compact_index_path = compact_root / "index.bin"
        compact_manifest = json.loads(compact_manifest_path.read_text())
        if (compact_manifest.get("schema_name") != CACHE_SCHEMA
                or compact_manifest.get("work_version") != "V1.13"
                or int(compact_manifest.get("validation", {}).get("bad_count", 1)) != 0
                or sha256_file(compact_index_path) != compact_manifest.get("index_sha256")):
            raise ValueError("V1.13 compact cache identity or validation failed")
        for path, expected in compact_manifest.get("source_sha256", {}).items():
            if sha256_file(path) != expected:
                raise ValueError(f"V1.13 compact cache source changed: {path}")
        inputs[str(compact_manifest_path)] = sha256_file(compact_manifest_path)
        inputs[str(compact_index_path)] = compact_manifest["index_sha256"]
    output = Path(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    source_hashes = {}
    source_paths = list(Path(__file__).parent.glob("*.py"))
    source_paths += [parent / "__init__.py" for parent in list(Path(__file__).parents)[1:4]
                     if (parent / "__init__.py").is_file()]
    for source in source_paths:
        relative = source.relative_to(REPO_ROOT)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(destination)
    write_json(output / "config.json", config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1", "task": "ObjectInteractionCmv2",
        "work_version": config["work_version"], "run_id": run_id, "run_status": "STARTED",
        "created_at": utc_now(),
        "operation": (("five_source_compact_endpoint_part_se3_ddp_smoke" if smoke_steps
                       else "five_source_compact_endpoint_part_se3_ddp_train")
                      if config.get("data_backend") == "compact" else
                      ("five_source_endpoint_part_se3_ddp_smoke" if smoke_steps
                       else "five_source_endpoint_part_se3_ddp_train")),
        "git_commit": commit,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        "implementation_identity": "clean git_commit plus frozen source_snapshot SHA256",
        "source_sha256": source_hashes, "input_sha256": inputs, "config": "config.json",
        "config_sha256": sha256_file(output / "config.json"), "smoke_steps": smoke_steps,
        "seed": config["training"]["seed"], "physical_gpus": config["resources"]["physical_gpus"],
        "nofile_limit": config["resources"]["nofile_limit"],
        "batch_size_per_rank": 64, "global_batch_size": 128, "group_weights": GROUP_WEIGHTS,
        "schema": ("V1.13 compact endpoint edges; V1.12 model/GT/loss semantics"
                   if config.get("data_backend") == "compact" else
                   "V1.12 endpoint KNN; whole-object part sampling; direct per-part SE(3)"),
        "coordinates": "current object/root reference frame; metres; seconds; radians",
        "split": str(split_root / "index.json"), "initial_checkpoint": None,
        "initialization": "random", "selection_metric": "weighted five-group mean of stride1/2/3 total loss",
        "approval_basis": config.get(
            "approval_basis",
            "user approved stopping V1.11.1 and reusing its formal budget on physical GPUs 0 and 2"),
        "outputs": {"directory": str(output), "metrics": "metrics.jsonl", "log": "train.log",
                    "service_log": "service.log", "latest": "latest.pt", "best": "best.pt"},
        "conclusion": "INCONCLUSIVE",
    }
    write_json(output / "run_manifest.json", manifest)
    return output


def supervise(output_value: str | Path) -> None:
    output = Path(output_value).resolve()

    def stop(signum, frame):
        manifest = json.loads((output / "run_manifest.json").read_text())
        manifest.update(run_status="STOPPED", finished_at=utc_now(), stop_signal=signum)
        write_json(output / "run_manifest.json", manifest)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=2", "-m", "src.task.ObjectInteractionCmv2.train_part_se3_ddp",
               "worker", "--output", str(output)]
    status = subprocess.call(command, cwd=output / "source_snapshot")
    manifest = json.loads((output / "run_manifest.json").read_text())
    if status or manifest["run_status"] != "COMPLETED":
        manifest.update(run_status="FAILED", finished_at=utc_now(), launcher_exit_code=status)
        write_json(output / "run_manifest.json", manifest)
        raise SystemExit(status or 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker", "supervise"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        run_worker(args.output)
        return
    if args.action == "supervise":
        supervise(args.output)
        return
    output = prepare(args.config, args.run_id, args.smoke_steps)
    unit = "ref2dex-" + args.run_id.replace("_", "-")
    config = json.loads((output / "config.json").read_text())
    resources = config["resources"]
    visible_gpus = ",".join(map(str, resources["physical_gpus"]))
    command = [
        "systemd-run", "--user", "--unit", unit, "--property=Restart=no",
        "--property=KillMode=control-group", "--property=TimeoutStopSec=60",
        f"--property=LimitNOFILE={resources['nofile_limit']}",
        f"--property=WorkingDirectory={output / 'source_snapshot'}",
        f"--property=StandardOutput=append:{output / 'service.log'}",
        f"--property=StandardError=append:{output / 'service.log'}",
        f"--setenv=CUDA_VISIBLE_DEVICES={visible_gpus}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=PYTHONUNBUFFERED=1", "--setenv=OMP_NUM_THREADS=8",
        "--setenv=NCCL_ASYNC_ERROR_HANDLING=1", f"--setenv=TMPDIR={output / 'tmp'}",
        sys.executable, "-m", "src.task.ObjectInteractionCmv2.train_part_se3_ddp",
        "supervise", "--output", str(output),
    ]
    manifest = json.loads((output / "run_manifest.json").read_text())
    manifest.update(service_unit=unit + ".service", launch_command=command)
    write_json(output / "run_manifest.json", manifest)
    try:
        subprocess.run(command, check=True)
    except Exception as error:
        manifest.update(run_status="FAILED", error=repr(error), finished_at=utc_now())
        write_json(output / "run_manifest.json", manifest)
        raise
    print(json.dumps({"output": str(output), "service_unit": unit + ".service"}))


if __name__ == "__main__":
    main()
