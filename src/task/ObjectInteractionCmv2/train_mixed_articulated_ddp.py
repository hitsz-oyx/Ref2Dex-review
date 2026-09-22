"""Frozen-source, model-only warm-start training for the approved V1.11g run."""
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
from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .articulated import ArticulatedObjectInteractionCmv2V15Model, articulated_v15_loss, collate_articulated
from .eval_two_domain_mano import FlowMetrics
from .mixed_training import (
    GROUPS, GROUP_WEIGHTS, REPO_ROOT, WORK_VERSION, batch_counts, build_dataset,
    load_config, sample_batch, sha256_file, utc_now, write_json,
)
from .oakink2_parts import OakInkPartTransitions


def initialize_model(config, device):
    model = ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(**config["model"])).to(device)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_version") != model.architecture_version:
        raise ValueError("initial checkpoint architecture mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    return model, optimizer, {key: checkpoint.get(key) for key in ("epoch", "step", "best_metric")}


def validation_indices(count, rank, world_size, limit=None):
    return list(range(rank, min(count, limit) if limit else count, world_size))


def move_batch(samples, device):
    batch = collate_articulated(samples)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def emit(output, manifest, rank, record):
    if rank:
        return
    record = {"timestamp": utc_now(), **record}
    line = json.dumps(record, allow_nan=False)
    for name in ("metrics.jsonl", "train.log"):
        with (output / name).open("a") as handle:
            handle.write(line + "\n")
    manifest.update({key: record[key] for key in ("last_step", "last_epoch", "phase", "best_metric") if key in record})
    manifest["updated_at"] = record["timestamp"]
    write_json(output / "run_manifest.json", manifest)
    print(line, flush=True)


def save_checkpoint(output, name, model, optimizer, generator, config, manifest, step, epoch, best, rank, world_size):
    state = {"sampler": generator.get_state(), "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    gathered = [None] * world_size
    dist.all_gather_object(gathered, state)
    if rank:
        return
    temporary = output / (name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "architecture_version": model.architecture_version, "work_version": WORK_VERSION,
                "run_id": manifest["run_id"], "base_commit": manifest["base_commit"],
                "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
                "step": step, "epoch": epoch, "best_metric": best, "seed": config["training"]["seed"],
                "rng_states": gathered, "world_size": world_size, "batch_size_per_rank": 64,
                "selection_metric": "weighted_five_group_mean_of_three_stride_validation_losses"}, temporary)
    temporary.replace(output / name)


def _metric_names():
    return [field.name for field in fields(FlowMetrics) if field.name != "angle_epsilon_m"]


def _metric_stat_names():
    return [name for name in _metric_names() if name != "samples"]


def _per_component_loss(prediction, batch):
    flow = F.smooth_l1_loss(prediction["obj_flow_pred"] / 0.02, batch["obj_flow_gt"] / 0.02, reduction="none").mean((1, 2))
    translation = F.smooth_l1_loss(prediction["delta_xi_root"][:, :3] / 0.02, batch["delta_xi_root_gt"][:, :3] / 0.02, reduction="none").mean(1)
    rotation = F.smooth_l1_loss(prediction["delta_xi_root"][:, 3:], batch["delta_xi_root_gt"][:, 3:], reduction="none").mean(1)
    return flow + translation + rotation


def _per_component_flow_fields(prediction, target):
    flow = prediction.detach()
    target = target.detach()
    if not torch.isfinite(flow).all() or not torch.isfinite(target).all():
        raise ValueError("non-finite component flow metrics")
    pred_norm = torch.linalg.vector_norm(flow, dim=-1)
    gt_norm = torch.linalg.vector_norm(target, dim=-1)
    epe = torch.linalg.vector_norm(flow - target, dim=-1)
    valid = (pred_norm >= 1e-6) & (gt_norm >= 1e-6)
    cosine = (flow * target).sum(-1) / (pred_norm * gt_norm).clamp_min(1e-12)
    angle = torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))
    return {
        "points": torch.full((flow.shape[0],), flow.shape[1], dtype=torch.float64, device=flow.device),
        "epe_sum_m": epe.sum(1).to(torch.float64),
        "pred_norm_sum_m": pred_norm.sum(1).to(torch.float64),
        "gt_norm_sum_m": gt_norm.sum(1).to(torch.float64),
        "angle_sum_deg": torch.where(valid, angle, torch.zeros_like(angle)).sum(1).to(torch.float64),
        "angle_valid_points": valid.sum(1).to(torch.float64),
        "gt_static_points": (gt_norm < 1e-6).sum(1).to(torch.float64),
        "prediction_static_points": (pred_norm < 1e-6).sum(1).to(torch.float64),
        "both_static_points": ((gt_norm < 1e-6) & (pred_norm < 1e-6)).sum(1).to(torch.float64),
    }


def _reduce_validation(metrics, loss_sum, component_samples, device, world_size):
    names = _metric_names()
    totals = torch.tensor([loss_sum, component_samples] + [getattr(metrics, name) for name in names], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    values = totals.cpu().tolist()
    for name, value in zip(names, values[2:]):
        setattr(metrics, name, int(round(value)) if name in {"samples", "points"} else value)
    return values[0], values[1]


def _evaluate_standard_group(model, dataset, indices, device):
    metrics, loss_sum = FlowMetrics(), 0.0
    for offset in range(0, len(indices), 64):
        chosen = indices[offset:offset + 64]
        batch = move_batch([dataset[index] for index in chosen], device)
        prediction = model(batch)
        loss = articulated_v15_loss(prediction, batch)["total"]
        if not torch.isfinite(loss):
            raise ValueError("non-finite validation loss")
        loss_sum += float(loss.cpu()) * len(chosen)
        metrics.update(prediction["obj_flow_pred"], batch["obj_flow_gt"])
    return metrics, loss_sum, metrics.samples


def _evaluate_part_group(model, dataset, indices, device):
    metrics, loss_sum, component_samples = FlowMetrics(), 0.0, 0
    samples, spans = [], []

    def consume():
        nonlocal samples, spans, loss_sum, component_samples
        if not samples:
            return
        batch = move_batch(samples, device)
        prediction = model(batch)
        losses = _per_component_loss(prediction, batch)
        if not torch.isfinite(losses).all():
            raise ValueError("non-finite OakInk2 component validation loss")
        fields = {name: value.cpu().tolist() for name, value in _per_component_flow_fields(prediction["obj_flow_pred"], batch["obj_flow_gt"]).items()}
        loss_values = losses.detach().cpu().tolist()
        for start, stop in spans:
            components = stop - start
            loss_sum += sum(loss_values[start:stop]) / components
            metrics.samples += 1
            for name in _metric_stat_names():
                setattr(metrics, name, getattr(metrics, name) + sum(fields[name][start:stop]) / components)
            component_samples += components
        samples, spans = [], []

    for transition_index in indices:
        component_samples_for_transition = [dataset.sample_component(transition_index, component)
                                            for component in range(dataset.component_count(transition_index))]
        if samples and len(samples) + len(component_samples_for_transition) > 64:
            consume()
        start = len(samples)
        samples.extend(component_samples_for_transition)
        spans.append((start, len(samples)))
    consume()
    return metrics, loss_sum, component_samples


@torch.inference_mode()
def evaluate(model, datasets, config, device, rank, world_size, output, manifest, smoke):
    model.eval()
    result = {}
    for group in GROUPS:
        for stride in config["data"]["validation_strides"]:
            key = f"{group}/stride{stride}"
            dataset = datasets[key]
            indices = validation_indices(len(dataset), rank, world_size, 4 if smoke else None)
            emit(output, manifest, rank, {"phase": "validation", "group": key, "local_transitions": len(indices)})
            if isinstance(dataset, OakInkPartTransitions):
                metrics, loss_sum, component_samples = _evaluate_part_group(model, dataset, indices, device)
                aggregation = "transition_mean_over_components"
            else:
                metrics, loss_sum, component_samples = _evaluate_standard_group(model, dataset, indices, device)
                aggregation = "sample"
            loss_sum, component_samples = _reduce_validation(metrics, loss_sum, component_samples, device, world_size)
            expected = min(len(dataset), 4) if smoke else len(dataset)
            if metrics.samples != expected:
                raise RuntimeError(f"validation coverage mismatch for {key}: {metrics.samples} != {expected}")
            result[key] = {"loss": loss_sum / metrics.samples, **metrics.summary(), "aggregation": aggregation,
                           "component_samples": component_samples,
                           "mean_components_per_transition": component_samples / metrics.samples}
            emit(output, manifest, rank, {"phase": "validation", "group": key, **result[key]})
    result["selection_metric"] = sum(
        GROUP_WEIGHTS[group] * sum(result[f"{group}/stride{stride}"]["loss"] for stride in (1, 2, 3)) / 3
        for group in GROUPS)
    return result


def check_input_identity(output, manifest):
    if sha256_file(output / "config.json") != manifest["config_sha256"]:
        raise ValueError("resolved configuration changed after preparation")
    for path, expected in manifest["input_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"input changed after run preparation: {path}")
    for relative, expected in manifest["source_sha256"].items():
        if sha256_file(output / "source_snapshot" / relative) != expected:
            raise ValueError(f"source snapshot changed: {relative}")


def run_worker(output):
    output = Path(output).resolve()
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "run_manifest.json").read_text())
    rank, world_size, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    resources = config.get("resources", {})
    physical_gpus = tuple(int(value) for value in resources.get("physical_gpus", (1, 3)))
    visible_gpus = ",".join(str(value) for value in physical_gpus)
    if world_size != 2 or os.environ.get("CUDA_VISIBLE_DEVICES") != visible_gpus:
        raise ValueError(f"approved resources are exactly physical GPUs {list(physical_gpus)} with two ranks")
    torch.set_num_threads(config["training"]["cpu_threads_per_rank"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.mem_get_info(device)[0] < config["training"]["minimum_free_memory_gib"] * 2**30:
        raise RuntimeError("insufficient free GPU memory; refusing to change batch or precision")
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
        for group in GROUPS:
            train[group] = build_dataset(config, config["split_root"], group, "train", max_sequences=1 if smoke else None)
            emit(output, manifest, rank, {"phase": "loading_datasets", "group": group, "train_rows": len(train[group])})
            for stride in (1, 2, 3):
                validation[f"{group}/stride{stride}"] = build_dataset(
                    config, config["split_root"], group, "val", fixed_stride=stride, max_sequences=1 if smoke else None)
        steps_per_epoch = math.ceil(sum(len(dataset) for dataset in train.values()) / 128)
        if smoke:
            steps_per_epoch = manifest["smoke_steps"]
        epochs = 1 if smoke else config["training"]["epochs"]
        model, optimizer, initial_metadata = initialize_model(config, device)
        wrapped = DistributedDataParallel(model, device_ids=[local_rank])
        generator = torch.Generator().manual_seed(config["training"]["seed"] + 100003 * rank)
        manifest.update(train_rows={group: len(dataset) for group, dataset in train.items()},
                        val_rows={group: len(dataset) for group, dataset in validation.items()},
                        dropped_train_timeline_transitions={group: getattr(dataset, "dropped_timeline_transitions", 0)
                                                            for group, dataset in train.items()},
                        steps_per_epoch=steps_per_epoch, planned_steps=steps_per_epoch * epochs,
                        initial_checkpoint_metadata=initial_metadata, initialization="model_only; optimizer/step/epoch/best reset")
        emit(output, manifest, rank, {"phase": "training", "steps_per_epoch": steps_per_epoch,
                                     "planned_steps": steps_per_epoch * epochs, "optimizer_state_entries_at_start": len(optimizer.state)})
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            wrapped.train()
            epoch_started = time.perf_counter()
            for position in range(steps_per_epoch):
                batch = move_batch(sample_batch(train, generator, step), device)
                optimizer.zero_grad(set_to_none=True)
                losses = articulated_v15_loss(wrapped(batch), batch)
                finite = torch.isfinite(losses["total"]).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise ValueError(f"non-finite training loss at step {step + 1}")
                losses["total"].backward()
                finite = torch.stack([torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                                      if parameter.grad is not None]).all().to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise ValueError(f"non-finite training gradient at step {step + 1}")
                optimizer.step()
                step += 1
                if step == 1 or step % config["training"]["log_interval"] == 0 or smoke:
                    values = torch.stack([value.detach() for value in losses.values()])
                    dist.all_reduce(values)
                    memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                                           torch.cuda.max_memory_reserved(device) / 2**30], device=device)
                    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                    emit(output, manifest, rank, {"phase": "training", "last_step": step, "last_epoch": epoch,
                         "position_in_epoch": position + 1, "elapsed_s": time.perf_counter() - started,
                         "epoch_elapsed_s": time.perf_counter() - epoch_started,
                         "losses": dict(zip(losses, (values / world_size).cpu().tolist())),
                         "batch_counts_per_rank": batch_counts(step - 1), "peak_allocated_gib": float(memory[0]),
                         "peak_reserved_gib": float(memory[1])})
                if step == 1 or step % config["training"]["checkpoint_interval"] == 0:
                    save_checkpoint(output, "latest.pt", model, optimizer, generator, config, manifest, step, epoch, best, rank, world_size)
                del losses, batch
            validation_result = evaluate(model, validation, config, device, rank, world_size, output, manifest, smoke)
            improved = best is None or validation_result["selection_metric"] < best
            if improved:
                best = validation_result["selection_metric"]
                save_checkpoint(output, "best.pt", model, optimizer, generator, config, manifest, step, epoch, best, rank, world_size)
            save_checkpoint(output, "latest.pt", model, optimizer, generator, config, manifest, step, epoch, best, rank, world_size)
            emit(output, manifest, rank, {"phase": "epoch_complete", "last_step": step, "last_epoch": epoch,
                                         "best_metric": best, "improved": improved, "validation": validation_result})
        if smoke:
            dist.barrier()
            restored = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
            for name, value in model.state_dict().items():
                if not torch.equal(value.cpu(), restored["model"][name]):
                    raise RuntimeError(f"checkpoint roundtrip / cross-rank parameter mismatch: {name}")
            emit(output, manifest, rank, {"phase": "smoke_verified", "checkpoint_roundtrip": True, "rank_parameters_identical": True})
        manifest.update(run_status="COMPLETED", finished_at=utc_now(), last_step=step, last_epoch=epoch, best_metric=best)
        emit(output, manifest, rank, {"phase": "completed"})
    except BaseException as error:
        write_json(output / f"rank{rank}_error.json", {"timestamp": utc_now(), "error": repr(error), "traceback": traceback.format_exc()})
        if rank == 0:
            manifest.update(run_status="FAILED", finished_at=utc_now(), last_step=step, last_epoch=epoch,
                            best_metric=best, error=repr(error), conclusion="INCONCLUSIVE")
            write_json(output / "run_manifest.json", manifest)
        raise
    finally:
        dist.destroy_process_group()


def prepare(config_path, run_id, smoke_steps):
    if Path(run_id).name != run_id or not run_id or smoke_steps < 0:
        raise ValueError("invalid run ID or smoke budget")
    config = load_config(config_path)
    split_root = Path(config["split_root"])
    split_manifest = json.loads((split_root / "cache_manifest.json").read_text())
    inputs = dict(split_manifest["source_sha256"])
    for path, expected in inputs.items():
        if sha256_file(path) != expected:
            raise ValueError(f"source index/cache manifest changed since split: {path}")
    adapter_root = Path(config["oakink2_parts"]["adapter_root"])
    adapter_manifest_path, adapter_index_path = adapter_root / "cache_manifest.json", adapter_root / "index.json"
    adapter_run_manifest_path = adapter_root / "run_manifest.json"
    if not adapter_manifest_path.is_file() or not adapter_index_path.is_file() or not adapter_run_manifest_path.is_file():
        raise FileNotFoundError("OakInk2 part adapter is not complete")
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    adapter_index = json.loads(adapter_index_path.read_text())
    adapter_run = json.loads(adapter_run_manifest_path.read_text())
    if (adapter_manifest.get("schema_name") != "ref2dex_cmv2_oakink2_part_adapter_v1"
            or adapter_index.get("schema_name") != "ref2dex_cmv2_oakink2_part_adapter_v1"
            or adapter_run.get("run_status") != "COMPLETED"
            or int(adapter_manifest.get("validation", {}).get("bad_count", 1) or 0) != 0):
        raise ValueError("OakInk2 part adapter validation failed")
    for path in (config["checkpoint"], config["articulation_metadata"], str(split_root / "index.json"), str(split_root / "cache_manifest.json"),
                 str(adapter_manifest_path), str(adapter_index_path), str(adapter_run_manifest_path)):
        inputs[path] = sha256_file(path)
    output = Path(config["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    source_hashes = {}
    source_paths = list(Path(__file__).parent.glob("*.py"))
    source_paths += [parent / "__init__.py" for parent in list(Path(__file__).parents)[1:4] if (parent / "__init__.py").is_file()]
    for source in source_paths:
        relative = source.relative_to(REPO_ROOT)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(destination)
    write_json(output / "config.json", config)
    manifest = {"schema_name": "ref2dex_run_manifest_v1", "task": "ObjectInteractionCmv2", "work_version": WORK_VERSION,
                "run_id": run_id, "run_status": "STARTED", "created_at": utc_now(), "operation": "five_source_part_ddp_smoke" if smoke_steps else "five_source_part_ddp_train",
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
                "implementation_identity": "base_commit plus frozen source_snapshot SHA256; uncommitted task changes",
                "source_sha256": source_hashes, "input_sha256": inputs, "config": "config.json", "config_sha256": sha256_file(output / "config.json"),
                "smoke_steps": smoke_steps, "seed": config["training"]["seed"],
                "physical_gpus": config["resources"]["physical_gpus"],
                "nofile_limit": config["resources"]["nofile_limit"],
                "batch_size_per_rank": 64, "global_batch_size": 128, "group_weights": GROUP_WEIGHTS,
                "schema": "V1.4 hand geometry; OakInk2 V1.11h component adapter; V1.5 articulated model", "coordinates": "current component/root local; metres; seconds; radians",
                "split": str(split_root / "index.json"), "initial_checkpoint": config["checkpoint"],
                "outputs": {"directory": str(output), "metrics": "metrics.jsonl", "log": "train.log", "service_log": "service.log", "latest": "latest.pt", "best": "best.pt"},
                "conclusion": "INCONCLUSIVE"}
    write_json(output / "run_manifest.json", manifest)
    return output


def supervise(output):
    output = Path(output).resolve()
    def stop(signum, frame):
        manifest = json.loads((output / "run_manifest.json").read_text())
        manifest.update(run_status="STOPPED", finished_at=utc_now(), stop_signal=signum)
        write_json(output / "run_manifest.json", manifest)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2",
               "-m", "src.task.ObjectInteractionCmv2.train_mixed_articulated_ddp", "worker", "--output", str(output)]
    status = subprocess.call(command, cwd=output / "source_snapshot")
    manifest = json.loads((output / "run_manifest.json").read_text())
    if status or manifest["run_status"] != "COMPLETED":
        manifest.update(run_status="FAILED", finished_at=utc_now(), launcher_exit_code=status)
        write_json(output / "run_manifest.json", manifest)
        raise SystemExit(status or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker", "supervise"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        run_worker(args.output)
    elif args.action == "supervise":
        supervise(args.output)
    else:
        output = prepare(args.config, args.run_id, args.smoke_steps)
        unit = "ref2dex-" + args.run_id.replace("_", "-")
        resolved_config = json.loads((output / "config.json").read_text())
        resources = resolved_config["resources"]
        visible_gpus = ",".join(str(value) for value in resources["physical_gpus"])
        command = ["systemd-run", "--user", "--unit", unit, "--property=Restart=no", "--property=KillMode=control-group",
                   "--property=TimeoutStopSec=60", f"--property=LimitNOFILE={resources['nofile_limit']}",
                   f"--property=WorkingDirectory={output / 'source_snapshot'}", f"--property=StandardOutput=append:{output / 'service.log'}",
                   f"--property=StandardError=append:{output / 'service.log'}", f"--setenv=CUDA_VISIBLE_DEVICES={visible_gpus}",
                   "--setenv=PYTHONDONTWRITEBYTECODE=1", "--setenv=PYTHONUNBUFFERED=1", "--setenv=OMP_NUM_THREADS=8",
                   "--setenv=NCCL_ASYNC_ERROR_HANDLING=1", f"--setenv=TMPDIR={output / 'tmp'}", sys.executable,
                   "-m", "src.task.ObjectInteractionCmv2.train_mixed_articulated_ddp", "supervise", "--output", str(output)]
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
