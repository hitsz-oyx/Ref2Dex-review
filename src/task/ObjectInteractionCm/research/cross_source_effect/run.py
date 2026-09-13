"""Frozen V1.3 cross-source effect diagnostic; never trains or changes caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.base.run_manifest import build_run_manifest, write_run_manifest
from src.task.ObjectInteractionCm.dataset import (
    ObjectInteractionCmDataset, _collate_object_interaction_cm, _stable_seed,
)
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


ROOT = Path(__file__).resolve().parents[5]
CHECKPOINT = ROOT / "outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt"
DIGEST = "3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283"
SOURCE_NAMES = {"grab": "MANO", "inspire_f1": "Inspire RL"}
SOURCES = tuple(SOURCE_NAMES)
MOTION_BINS = [0, 1, 2, 4, 8, 16, 32, 64, 128, float("inf")]
CONTACT_BINS = [0, .01, .05, .1, .25, .5, 1.000001]
MODEL_KEYS = ("obj_points", "obj_normals", "obj_valid_mask", "hand_points",
              "hand_normals", "hand_flow", "hand_valid_mask", "knn_edge_indices",
              "knn_edge_valid_mask")
METRIC_KEYS = (
    "epe_mm", "mse_mm2", "zero_epe_mm", "zero_mse_mm2", "shuffle_epe_mm",
    "zero_token_epe_mm", "gain_over_zero_mm", "shuffle_penalty_mm",
    "zero_token_penalty_mm", "hand_rms_mm", "object_rms_mm",
    "active_fraction", "sampled_active_fraction", "shuffle_input_change_rms_mm",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def vector_errors(prediction, target, valid):
    squared = (prediction - target).square().sum(dim=-1)
    weights = valid.to(squared.dtype)
    denominator = weights.sum(dim=-1)
    if (denominator == 0).any():
        raise ValueError("Object error requires at least one valid point")
    return ((squared.sqrt() * weights).sum(dim=-1) / denominator * 1000,
            (squared * weights).sum(dim=-1) / denominator * 1e6)


def masked_rms(flow, valid):
    weights = valid.to(flow.dtype)
    return ((flow.square().sum(dim=-1) * weights).sum(dim=-1)
            / weights.sum(dim=-1).clamp_min(1)).sqrt() * 1000


def shuffled_flow(flow, valid, seeds):
    result = flow.clone()
    for i, seed in enumerate(seeds):
        indices = torch.nonzero(valid[i], as_tuple=False).flatten()
        permutation = np.random.default_rng(seed).permutation(len(indices))
        order = torch.as_tensor(permutation, device=flow.device, dtype=torch.long)
        result[i, indices] = flow[i, indices[order]]
    return result


def decode(model, batch, result, *, zero_tokens=False):
    tokens = result["cm_tokens"]
    prediction, _ = model.object_decoder(
        batch["obj_points"].float(), F.normalize(batch["obj_normals"].float(), dim=-1, eps=1e-6),
        torch.zeros_like(tokens) if zero_tokens else tokens,
        result["cm_anchor_pos"], result["cm_anchor_normal"],
    )
    return prediction * model.object_flow_target_scale


def match_rows(rows, seed=42):
    strata = defaultdict(lambda: {source: [] for source in SOURCES})
    for row in rows:
        if not row["primary_valid"]:
            continue
        key = (row["object_name"],
               int(np.searchsorted(MOTION_BINS, row["hand_rms_mm"], side="right") - 1),
               int(np.searchsorted(MOTION_BINS, row["object_rms_mm"], side="right") - 1),
               int(np.searchsorted(CONTACT_BINS, row["active_fraction"], side="right") - 1))
        strata[key][row["source"]].append(row["sample_index"])
    rng = np.random.default_rng(seed)
    selected, records = [], []
    for key, members in sorted(strata.items()):
        count = min(len(members[source]) for source in SOURCES)
        if not count:
            continue
        chosen = {source: sorted(int(i) for i in rng.choice(members[source], count, replace=False))
                  for source in SOURCES}
        selected.extend(i for values in chosen.values() for i in values)
        records.append({"stratum": list(key), "available": {s: len(members[s]) for s in SOURCES},
                        "per_source_count": count, "selected": chosen})
    return set(selected), records


def cluster_estimates(rows, repeats=2000, seed=42):
    """Paired control differences, but independent clusters across sources."""
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["sequence_id"]].append([row[key] for key in METRIC_KEYS])
    if not clusters:
        return None, None
    arrays = [np.asarray(values, dtype=np.float64) for _, values in sorted(clusters.items())]
    counts = np.array([len(array) for array in arrays], dtype=np.float64)
    sums = np.stack([array.sum(axis=0) for array in arrays])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(arrays), np.full(len(arrays), 1 / len(arrays)), size=repeats)
    boot = weights @ sums / (weights @ counts)[:, None]

    def metrics(values):
        result = dict(zip(METRIC_KEYS, (float(x) for x in values)))
        result["rmse_mm"] = float(np.sqrt(result.pop("mse_mm2")))
        result["zero_rmse_mm"] = float(np.sqrt(result.pop("zero_mse_mm2")))
        result["improvement_over_zero_fraction"] = (
            result["gain_over_zero_mm"] / result["zero_epe_mm"] if result["zero_epe_mm"] > 0 else None)
        return result

    intervals = {}
    for i, key in enumerate(METRIC_KEYS):
        values = boot[:, i]
        if key.endswith("mse_mm2"):
            values = np.sqrt(values)
            key = key.replace("mse_mm2", "rmse_mm")
        intervals[key] = np.percentile(values, [2.5, 97.5]).tolist()
    return {"samples": int(counts.sum()), "sequences": len(arrays),
            "frame_micro": metrics(sums.sum(axis=0) / counts.sum()),
            "sequence_macro": metrics((sums / counts[:, None]).mean(axis=0)),
            "frame_micro_cluster_ci95": intervals}, boot


def summarize(rows, repeats=2000):
    result, bootstraps = {}, {}
    for i, source in enumerate(SOURCES):
        subset = [row for row in rows if row["source"] == source]
        result[source], bootstraps[source] = cluster_estimates(subset, repeats, 42 + i)
    if all(result.values()):
        a, b = (result[s]["frame_micro"]["epe_mm"] for s in SOURCES)
        ba, bb = (bootstraps[s][:, 0] for s in SOURCES)
        result["source_gap"] = {
            "definition": "MANO minus Inspire RL; no equivalence margin or equivalence test",
            "epe_difference_mm": a - b, "epe_ratio": a / b if b > 0 else None,
            "difference_ci95_mm": np.percentile(ba - bb, [2.5, 97.5]).tolist(),
            "ratio_ci95": np.percentile(ba / bb, [2.5, 97.5]).tolist() if np.all(bb > 0) else None,
            "equal_source_epe_mm": (a + b) / 2,
        }
    result["pooled"], _ = cluster_estimates(rows, repeats, 44)
    return result


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    keys = ["epe_mm", "zero_epe_mm", "shuffle_epe_mm", "zero_token_epe_mm"]
    labels = ["Full Cm", "Zero flow", "Shuffled flow", "Zero tokens"]
    for axis, section, title in zip(axes, ("primary", "matched"), ("All valid val", "Matched subset")):
        for i, source in enumerate(SOURCES):
            stats = summary[section].get(source)
            if stats is None:
                continue
            values = np.array([stats["frame_micro"][key] for key in keys])
            ci = np.array([stats["frame_micro_cluster_ci95"][key] for key in keys]).T
            axis.bar(np.arange(4) + (i - .5) * .36, values, .36, label=SOURCE_NAMES[source],
                     color=("#247b8d", "#bd4856")[i])
            axis.errorbar(np.arange(4) + (i - .5) * .36, (ci[0] + ci[1]) / 2,
                          yerr=(ci[1] - ci[0]) / 2, fmt="none", color="#252525", capsize=3)
        axis.set(xticks=np.arange(4), xticklabels=labels, ylabel="Object EPE (mm)", title=title)
        axis.tick_params(axis="x", labelrotation=18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args):
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be one directory name")
    os.chdir(ROOT)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(CHECKPOINT) != DIGEST:
        raise ValueError("Approved checkpoint SHA256 mismatch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    meta, data = cfg["meta"], cfg["data"]
    assert cfg["modification_version"] == "V1.3"
    for key, value in {"num_obj_points": 1024, "num_obj_pool": 4096, "knn_k": 32,
                       "hand_stream_mode": "unique_knn_edges", "interaction_radius_m": .02,
                       "hand_supervision_radius_m": .02, "coordinate_frame": "object_pose_t"}.items():
        assert meta[key] == value, (key, meta[key])
    assert data["eval_stride"] == 2 and cfg["train"]["seed"] == 42
    index_path, scale_path = ROOT / data["index_path"], ROOT / meta["scale_manifest_path"]
    assert index_path.is_file() and scale_path.is_file()
    index = json.loads(index_path.read_text())
    entries = index["sequences"]["val"]
    assert len(entries) == 58
    parent_sets = {split: {e["parent_seq_id"] for e in index["sequences"][split]}
                   for split in ("train", "val", "test")}
    assert not (parent_sets["val"] & (parent_sets["train"] | parent_sets["test"]))
    assert {s: sum(e["source"] == s for e in entries) for s in SOURCES} == {"grab": 28, "inspire_f1": 30}
    if args.smoke:
        entries = [next(e for e in entries if e["source"] == source) for source in SOURCES]
    protected = {str(p): sha256(p) for p in (CHECKPOINT, index_path, scale_path)}
    model = ObjectInteractionCmModel(SimpleNamespace(meta=SimpleNamespace(**meta), modification_version="V1.3"))
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval().to(args.device)
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = Path(__file__).parent / "output" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    config = {"modification_version": "V1.3.1", "operation_category": ["diagnostic", "experiment"],
              "train": {"seed": 42}, "checkpoint": str(CHECKPOINT), "data": data, "meta": meta,
              "arguments": vars(args), "evaluation_partition": "val", "dataset_epoch": 0,
              "loader_active_only": False, "primary_mask": "full_active_count > 0 and sample_valid",
              "bootstrap_repeats": 2000, "motion_bins_mm": MOTION_BINS[:-1] + ["inf"],
              "contact_fraction_bins": CONTACT_BINS, "command": sys.argv,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    metadata = {"schema_name": index["schema_name"], "schema_version": index["schema_version"],
                "coordinate_frame": meta["coordinate_frame"], "num_obj_pool": 4096, "num_obj_points": 1024,
                "checkpoint_epoch": payload["epoch"], "checkpoint_step": payload["step"],
                "checkpoint_best_metric": payload["best_metric"], "scales": model.scale_values,
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "python_version": sys.version, "device": args.device,
                "device_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
                "entries": entries, "protected_digests": list(protected.items()), "sequence_files": [],
                "scientific_limitations": ["mixed-source trained, val-selected checkpoint", "unpaired source interactions",
                                           "zero tokens is OOD and preserves motion-dependent anchors",
                                           "spatial shuffle preserves flow multiset and rigid translation"]}
    files = [Path(__file__), ROOT / "src/task/ObjectInteractionCm/dataset.py",
             ROOT / "src/task/ObjectInteractionCm/model.py", ROOT / "src/task/ObjectInteractionCm/decoder.py",
             ROOT / "src/task/ObjectInteractionCm/slot_attention.py", Path(__file__).with_name("experiment.yaml")]
    metadata["code_digests"] = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    for entry in entries:
        directory = index_path.parent / entry["path"] / "geometry"
        metadata["sequence_files"].extend({"path": str(p), "size_bytes": p.stat().st_size,
                                           "mtime_ns": p.stat().st_mtime_ns,
                                           **({"sha256": sha256(p)} if p.suffix == ".json" else {})}
                                          for p in sorted(directory.iterdir()) if p.is_file())
    write_json(out / "config.json", config)
    write_json(out / "metadata.json", metadata)
    manifest = build_run_manifest(task="ObjectInteractionCm", run_name=args.run_id, output_dir=out,
                                  mode="diagnostic", config=config, metadata={k: metadata[k] for k in
                                  ("schema_name", "schema_version", "coordinate_frame", "num_obj_pool", "num_obj_points")},
                                  config_source=Path(__file__).with_name("experiment.yaml"),
                                  initial_checkpoint=CHECKPOINT, repo_root=ROOT)
    write_run_manifest(out / "run_manifest.json", manifest)
    rows, artifacts, max_parity = [], [], 0.0

    def log(message):
        line = datetime.now().astimezone().isoformat(timespec="seconds") + " " + message
        print(line, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"START run_id={args.run_id} smoke={args.smoke} sequences={len(entries)}")
    with torch.inference_mode(), (out / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for sequence_number, entry in enumerate(entries):
            path = (index_path.parent / entry["path"]).resolve()
            dataset = ObjectInteractionCmDataset(
                ROOT, sequence_entries=[{"path": str(path), "source": entry["source"]}],
                num_obj_points=meta["num_obj_points"], num_hand_points=meta["num_hand_points"],
                max_hand_points=meta["max_hand_points"], max_knn_hand_points=meta["max_knn_hand_points"],
                hand_stream_mode=meta["hand_stream_mode"], knn_k=meta["knn_k"],
                hand_supervision_radius_m=meta["hand_supervision_radius_m"], base_seed=42,
                min_stride=data["min_stride"], max_stride=data["max_stride"], fixed_stride=data["eval_stride"],
                active_only=False, source_stride_values={"grab": data["grab_stride_values"],
                                                        "inspire_f1": data["inspire_stride_values"]})
            selection = list(range(len(dataset)))
            if args.smoke:
                active = [i for i, (_, frame) in enumerate(dataset.rows) if dataset.sequences[0].candidate_active(frame)]
                selection = sorted(set([0] + active[:7]))
            loader = DataLoader(Subset(dataset, selection), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=_collate_object_interaction_cm,
                                pin_memory=args.device.startswith("cuda"))
            saved = defaultdict(list)
            cursor = 0
            for cpu in loader:
                if time.monotonic() - started > 1800:
                    raise TimeoutError("Approved 30-minute budget exceeded")
                batch = {key: cpu[key].to(args.device, non_blocking=True) for key in MODEL_KEYS}
                gt = cpu["obj_flow_gt"].to(args.device)
                result = model(batch)
                pred = result["pred_obj_flow"]
                replay = decode(model, batch, result)
                parity = float((replay - pred).abs().max())
                max_parity = max(max_parity, parity)
                torch.testing.assert_close(replay, pred, rtol=1e-5, atol=1e-7)
                token_zero = decode(model, batch, result, zero_tokens=True)
                seeds = [_stable_seed(42, entry["parent_seq_id"], int(raw), "spatial_flow_shuffle")
                         for raw in cpu["raw_frame_id"]]
                control_batch = dict(batch)
                control_batch["hand_flow"] = shuffled_flow(batch["hand_flow"], batch["hand_valid_mask"], seeds)
                control = model(control_batch)
                assert torch.equal(result["sample_valid"], control["sample_valid"])
                shuffled = control["pred_obj_flow"]
                valid = batch["obj_valid_mask"]
                epe, mse = vector_errors(pred, gt, valid)
                zepe, zmse = vector_errors(torch.zeros_like(gt), gt, valid)
                sepe, _ = vector_errors(shuffled, gt, valid)
                tepe, _ = vector_errors(token_zero, gt, valid)
                values = {"epe_mm": epe, "mse_mm2": mse, "zero_epe_mm": zepe, "zero_mse_mm2": zmse,
                          "shuffle_epe_mm": sepe, "zero_token_epe_mm": tepe,
                          "gain_over_zero_mm": zepe - epe, "shuffle_penalty_mm": sepe - epe,
                          "zero_token_penalty_mm": tepe - epe,
                          "hand_rms_mm": masked_rms(batch["hand_flow"], batch["hand_valid_mask"]),
                          "object_rms_mm": zmse.sqrt(),
                          "active_fraction": cpu["full_active_count"].float() / 4096,
                          "sampled_active_fraction": result["sampled_active_count"].float() / 1024,
                          "shuffle_input_change_rms_mm": masked_rms(control_batch["hand_flow"] - batch["hand_flow"], batch["hand_valid_mask"])}
                values = {k: v.cpu().numpy() for k, v in values.items()}
                size = len(gt)
                ids = np.arange(len(rows), len(rows) + size)
                arrays = {"sample_index": ids, "obj_points": batch["obj_points"], "obj_normals": batch["obj_normals"],
                          "gt_obj_flow": gt, "pred_obj_flow": pred, "shuffled_obj_flow": shuffled,
                          "zero_token_obj_flow": token_zero,
                          **{k: result[k] for k in ("cm_tokens", "cm_anchor_pos", "cm_anchor_normal", "sample_valid", "sampled_active_count")}}
                for key, value in arrays.items():
                    array = value.cpu().numpy() if torch.is_tensor(value) else value
                    if not np.isfinite(array).all():
                        raise ValueError(f"Nonfinite {key} in {entry['id']}")
                    saved[key].append(array)
                for j in range(size):
                    position = selection[cursor + j]
                    sample_valid = bool(result["sample_valid"][j])
                    row = {"sample_index": int(ids[j]), "sequence_id": entry["parent_seq_id"],
                           "sequence_number": sequence_number, "source": entry["source"],
                           "variant": entry["variant"], "object_name": entry["object_name"],
                           "action_name": entry["action_name"], "frame_index": dataset.rows[position][1],
                           "raw_frame_id": int(cpu["raw_frame_id"][j]), "next_raw_frame_id": int(cpu["next_raw_frame_id"][j]),
                           "stride": int(cpu["stride"][j]), "delta_time_s": float(cpu["delta_time_s"][j]),
                           "sample_valid": sample_valid, "primary_valid": sample_valid and bool(cpu["full_active_count"][j] > 0),
                           "hand_valid_points": int(cpu["hand_valid_points"][j]),
                           "full_active_count": int(cpu["full_active_count"][j]), "shuffle_seed": seeds[j],
                           **{k: float(v[j]) for k, v in values.items()}}
                    metrics.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                cursor += size
            artifact = out / f"effects_{sequence_number:03d}.npz"
            np.savez_compressed(artifact, **{k: np.concatenate(v) for k, v in saved.items()})
            artifacts.append({"file": artifact.name, "sequence_id": entry["parent_seq_id"], "samples": cursor})
            metrics.flush()
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > 3 * 1024 ** 3:
                raise RuntimeError("Approved 3 GiB output budget exceeded")
            log(f"sequence={sequence_number + 1}/{len(entries)} source={entry['source']} id={entry['id']} samples={cursor} total={len(rows)}")
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(original_state[k], v.cpu()) for k, v in model.state_dict().items())
    assert all(sha256(Path(path)) == digest for path, digest in protected.items())
    assert all(Path(f["path"]).stat().st_size == f["size_bytes"] and
               Path(f["path"]).stat().st_mtime_ns == f["mtime_ns"] for f in metadata["sequence_files"])
    primary = [row for row in rows if row["primary_valid"]]
    selected, strata = match_rows(rows)
    matched = [row for row in rows if row["sample_index"] in selected]
    write_json(out / "matched_selection.json", {"sample_indices": sorted(selected), "strata": strata})
    summary = {"engineering_checks": {"finite": True, "state_unchanged": True, "inputs_unchanged": True,
                                      "decoder_replay_max_abs_m": max_parity, "gt_excluded_from_model": True},
               "smoke_only": args.smoke, "elapsed_seconds": time.monotonic() - started,
               "primary": summarize(primary), "matched": summarize(matched), "coverage": {},
               "artifacts": artifacts, "matched_strata": len(strata),
               "conclusion": "INCONCLUSIVE", "conclusion_scope": "Unseen-hand transfer and effect equivalence not tested"}
    for source in SOURCES:
        subset = [r for r in rows if r["source"] == source]
        count, active, valid = len(subset), sum(r["full_active_count"] > 0 for r in subset), sum(r["primary_valid"] for r in subset)
        summary["coverage"][source] = {"rows": count, "full_pool_active": active, "primary_valid": valid,
            "sample_valid_fraction_all_rows": valid / count if count else None,
            "sample_valid_fraction_full_pool_active": valid / active if active else None,
            "matched_samples": sum(r["sample_index"] in selected for r in subset)}
    write_json(out / "effect_summary.json", summary)
    plot_summary(summary, out / "effect_comparison.png")
    log(f"COMPLETED rows={len(rows)} valid={len(primary)} matched={len(matched)} elapsed_s={time.monotonic() - started:.1f}")
    print(json.dumps({"output": str(out), "primary": summary["primary"], "matched": summary["matched"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    run(args)


if __name__ == "__main__":
    main()
