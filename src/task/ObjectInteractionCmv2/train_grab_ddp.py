"""Two-GPU GRAB V1.3.4 calibration and four-epoch DDP training."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import load_grab_config
from .grab import GrabManoTransitions, sha256_file
from .model import ObjectInteractionCmv2V13Model, object_interaction_v13_loss
from .train_grab import utc_now, write_json


def _batch(dataset, indices, device):
    samples = [dataset[int(index)] for index in indices]
    return {
        key: torch.stack([sample[key] for sample in samples]).to(device)
        for key in samples[0] if torch.is_tensor(samples[0][key])
    }


def shard_epoch_indices(count, seed, epoch, rank, world_size):
    """One deterministic epoch without padding or duplicated samples."""
    order = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed + epoch))
    return order[rank::world_size]


def _model_and_optimizer(cfg, device):
    model = ObjectInteractionCmv2V13Model(SimpleNamespace(**cfg["model"])).to(device)
    wrapped = DistributedDataParallel(
        model, device_ids=[device.index], find_unused_parameters=True)
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=cfg["training"]["learning_rate"])
    return wrapped, optimizer


def _save_checkpoint(path, model, optimizer, step, epoch, position, cfg, index_sha256,
                     device, rank, world_size):
    local_rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
    rng_states = [None] * world_size
    dist.all_gather_object(rng_states, local_rng)
    if rank:
        return
    value = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "architecture_version": model.module.architecture_version,
        "modification_version": cfg["modification_version"],
        "seed": cfg["training"]["seed"],
        "index_sha256": index_sha256,
        "world_size": world_size,
        "batch_size_per_gpu": cfg["training"]["batch_size_per_gpu"],
        "epochs": cfg["training"]["epochs"],
        "step": step,
        "epoch": epoch,
        "position": position,
        "rng_states": rng_states,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _run_calibration(dataset, cfg, output, device, rank, world_size, manifest):
    target = float(cfg["training"]["target_memory_gib"])
    manifest.update(
        run_status="RUNNING", valid_pairs=len(dataset),
        dropped_pairs=dataset.dropped_pairs)
    if rank == 0:
        write_json(output / "run_manifest.json", manifest)
    results = []
    selected = None
    for batch_size in cfg["training"]["batch_candidates"]:
        torch.cuda.empty_cache()
        torch.manual_seed(cfg["training"]["seed"])
        model, optimizer = _model_and_optimizer(cfg, device)
        indices = list(range(rank * batch_size, (rank + 1) * batch_size))
        batch = _batch(dataset, indices, device)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        losses = object_interaction_v13_loss(model(batch), batch)
        if not torch.isfinite(losses["total"]):
            raise ValueError(f"Nonfinite calibration loss at batch {batch_size}")
        losses["total"].backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        local = torch.tensor([
            torch.cuda.max_memory_allocated(device) / 2**30,
            torch.cuda.max_memory_reserved(device) / 2**30,
        ], device=device)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        if rank == 0:
            row = {
                "batch_size_per_gpu": batch_size,
                "allocated_gib_per_gpu": [float(x[0]) for x in gathered],
                "reserved_gib_per_gpu": [float(x[1]) for x in gathered],
                "loss_finite": True,
            }
            results.append(row)
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if max(row["reserved_gib_per_gpu"]) <= target:
                selected = batch_size
        dist.barrier()
        del losses, batch, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        decision = torch.tensor(
            [int(rank == 0 and selected != batch_size)], device=device)
        dist.broadcast(decision, src=0)
        if decision.item():
            break
    if rank == 0:
        if selected is None:
            raise ValueError("No approved batch fits the 24 GiB reserved-memory target")
        manifest.update(
            run_status="COMPLETED", finished_at=utc_now(),
            selected_batch_size_per_gpu=selected,
            selected_global_batch=selected * world_size,
            peak_reserved_gib=max(next(
                row for row in results if row["batch_size_per_gpu"] == selected
            )["reserved_gib_per_gpu"]),
            calibration_results=results, last_step=len(results), last_epoch=None,
            best_metric=None, conclusion="INCONCLUSIVE")
        write_json(output / "run_manifest.json", manifest)
        print(json.dumps({"selected_batch_size_per_gpu": selected, "results": results}))


def _run_training(dataset, cfg, output, device, rank, world_size, manifest, resume):
    training = cfg["training"]
    batch_size = training["batch_size_per_gpu"]
    rank_pairs = len(dataset) // world_size
    if len(dataset) % world_size or rank_pairs != 163361:
        raise ValueError("Approved DDP shard size mismatch")
    steps_per_epoch = math.ceil(rank_pairs / batch_size)
    planned_steps = steps_per_epoch * training["epochs"]
    torch.manual_seed(training["seed"])
    model, optimizer = _model_and_optimizer(cfg, device)
    step, start_epoch, start_position = 0, 0, 0
    if resume is not None:
        saved = torch.load(resume, map_location=device, weights_only=False)
        required = {
            "architecture_version": model.module.architecture_version,
            "index_sha256": manifest["input"]["index_sha256"],
            "seed": training["seed"], "world_size": world_size,
            "batch_size_per_gpu": batch_size, "epochs": training["epochs"],
        }
        if any(saved.get(key) != value for key, value in required.items()):
            raise ValueError("DDP resume contract mismatch")
        model.module.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, start_epoch, start_position = (
            saved["step"], saved["epoch"], saved["position"])
        torch.set_rng_state(saved["rng_states"][rank]["cpu"].cpu())
        torch.cuda.set_rng_state(saved["rng_states"][rank]["cuda"].cpu(), device)
    run_started = time.perf_counter()
    manifest.update(
        run_status="RUNNING", planned_steps=planned_steps,
        steps_per_epoch=steps_per_epoch,
        batch_size_per_gpu=batch_size, global_batch=batch_size * world_size,
        valid_pairs=len(dataset), dropped_pairs=dataset.dropped_pairs,
        initial_checkpoint=str(resume.resolve()) if resume else None)
    if rank == 0:
        write_json(output / "run_manifest.json", manifest)
    dist.barrier()
    metrics = (output / "metrics.jsonl").open("a") if rank == 0 else None
    log = (output / "train.log").open("a") if rank == 0 else None
    stopped = False
    completed_epochs = start_epoch
    pairs_seen = start_epoch * len(dataset) + start_position * world_size
    try:
        for epoch in range(start_epoch, training["epochs"]):
            local_order = shard_epoch_indices(
                len(dataset), training["seed"], epoch, rank, world_size)
            position = start_position if epoch == start_epoch else 0
            while position < rank_pairs:
                indices = local_order[position:position + batch_size]
                batch = _batch(dataset, indices, device)
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                losses = object_interaction_v13_loss(prediction, batch)
                if not torch.isfinite(losses["total"]):
                    raise ValueError(f"Nonfinite DDP loss at step {step + 1}")
                losses["total"].backward()
                optimizer.step()
                torch.cuda.synchronize(device)
                step += 1
                position += len(indices)
                pairs_seen += len(indices) * world_size
                epe = torch.linalg.vector_norm(
                    prediction["obj_flow_pred"].detach() - batch["obj_flow_gt"], dim=-1).mean()
                reduced = torch.stack([
                    losses["total"].detach(), losses["translation"].detach(),
                    losses["rotation"].detach(), losses["flow"].detach(), epe])
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced /= world_size
                resources = torch.tensor([
                    time.perf_counter() - tick,
                    torch.cuda.max_memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_reserved(device) / 2**30,
                ], device=device)
                dist.all_reduce(resources, op=dist.ReduceOp.MAX)
                if rank == 0:
                    record = {
                        "step": step, "epoch": epoch,
                        "pairs_seen": pairs_seen,
                        "loss": float(reduced[0]),
                        "translation_loss": float(reduced[1]),
                        "rotation_loss": float(reduced[2]),
                        "flow_loss": float(reduced[3]),
                        "flow_epe_mm": float(reduced[4] * 1000),
                        "step_seconds": float(resources[0]),
                        "gpu_peak_allocated_gib": float(resources[1]),
                        "gpu_peak_reserved_gib": float(resources[2]),
                    }
                    line = json.dumps(record) + "\n"
                    metrics.write(line)
                    metrics.flush()
                    log.write(line)
                    log.flush()
                next_epoch = epoch + 1 if position == rank_pairs else epoch
                next_position = 0 if position == rank_pairs else position
                time_limit = torch.tensor([
                    int(rank == 0 and time.perf_counter() - run_started >= training["max_duration_s"])
                ], device=device)
                dist.broadcast(time_limit, src=0)
                save_due = (
                    step % training["checkpoint_interval"] == 0
                    or next_epoch > epoch or bool(time_limit.item()))
                if save_due:
                    _save_checkpoint(
                        output / "latest.pt", model, optimizer, step, next_epoch,
                        next_position, cfg, manifest["input"]["index_sha256"],
                        device, rank, world_size)
                if next_epoch > epoch:
                    completed_epochs = next_epoch
                    if rank == 0:
                        source = output / "latest.pt"
                        (output / f"epoch_{next_epoch}.pt").write_bytes(source.read_bytes())
                if time_limit.item() and next_epoch < training["epochs"]:
                    stopped = True
                    break
                del batch, prediction, losses
            if stopped:
                break
        if rank == 0:
            manifest.update(
                run_status="STOPPED" if stopped else "COMPLETED",
                stop_reason="max_duration_s" if stopped else None,
                finished_at=utc_now(), last_step=step,
                last_epoch=completed_epochs, epochs_completed=completed_epochs,
                pairs_seen=pairs_seen, latest_checkpoint="latest.pt",
                best_metric=None, conclusion="INCONCLUSIVE")
            write_json(output / "run_manifest.json", manifest)
            print(json.dumps({
                "run_status": manifest["run_status"],
                "last_step": step, "epochs_completed": completed_epochs,
                "pairs_seen": pairs_seen}))
    finally:
        if metrics is not None:
            metrics.close()
        if log is not None:
            log.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    cfg = load_grab_config(args.config)
    if cfg["training"].get("mode") not in ("ddp_calibration", "ddp_formal"):
        raise ValueError("Expected V1.3.4 DDP configuration")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise ValueError("V1.3.4 requires physical GPUs 2 and 3")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != cfg["training"]["world_size"] or local_rank not in (0, 1):
        raise ValueError("Expected exactly two DDP workers")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    output = Path(cfg["output_root"]) / args.run_id
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", cfg)
    dist.barrier()
    manifest = {
        "task": "ObjectInteractionCmv2",
        "modification_version": cfg["modification_version"],
        "run_id": args.run_id, "run_status": "STARTED",
        "created_at": utc_now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": "config.json",
        "seed": cfg["training"]["seed"],
        "architecture_version": ObjectInteractionCmv2V13Model.architecture_version,
        "initial_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "input": {
            "index": cfg["source"]["index"],
            "index_sha256": sha256_file(Path(cfg["source"]["index"])),
            "manifest": cfg["source"]["manifest"],
            "manifest_sha256": sha256_file(Path(cfg["source"]["manifest"])),
        },
        "input_schema": "ref2dex_object_interaction_cm_bilateral_mano_v1_4",
        "coordinate_frame": "object_pose_t",
        "device_ids": [2, 3], "world_size": world_size,
        "outputs": (
            {"metrics": "metrics.jsonl"}
            if cfg["training"]["mode"] == "ddp_calibration"
            else {"metrics": "metrics.jsonl", "train_log": "train.log",
                  "latest_checkpoint": "latest.pt"}),
        "conclusion": "INCONCLUSIVE",
    }
    if rank == 0:
        write_json(output / "run_manifest.json", manifest)
    try:
        dataset = GrabManoTransitions(
            cfg["source"]["index"], cfg["source"]["manifest"],
            "train", direct_pose_gt=True)
        if (len(dataset.sequences) != cfg["training"]["expected_train_sequences"]
            or len(dataset) != cfg["training"]["expected_train_pairs"]
            or dataset.dropped_pairs != 0):
            raise ValueError("DDP GRAB input differs from approved contract")
        if cfg["training"]["mode"] == "ddp_calibration":
            _run_calibration(dataset, cfg, output, device, rank, world_size, manifest)
        else:
            _run_training(dataset, cfg, output, device, rank, world_size, manifest, args.resume)
    except Exception as error:
        if rank == 0:
            manifest.update(
                run_status="FAILED", finished_at=utc_now(),
                error=repr(error), conclusion="INVALID_IMPLEMENTATION")
            write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
