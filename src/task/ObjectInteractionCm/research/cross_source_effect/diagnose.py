"""Mechanism diagnostics from archived predictions; no model training or GT-conditioned baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import torch

from src.base.run_manifest import build_run_manifest, write_run_manifest
from src.task.ObjectInteractionCm.dataset import ObjectInteractionCmDataset
from .run import ROOT, SOURCES, sha256, write_json


DEFAULT_SOURCE = Path(__file__).parent / "output/cross_source_effect_v1_3_1_val_20260912_112900"
KEYS = ("full_epe_mm", "zero_epe_mm", "mean_hand_epe_mm", "rigid_hand_epe_mm",
        "projected_epe_mm", "gt_rigid_residual_mm", "pred_nonrigid_residual_mm",
        "full_minus_mean_hand_mm", "full_minus_rigid_hand_mm", "projection_gain_mm",
        "hand_rigid_residual_mm", "token_centered_rank", "token_slot_spread_ratio",
        "anchor_object_spread_ratio")


def epe(prediction, target):
    return float(np.linalg.norm(prediction - target, axis=-1).mean() * 1000)


def rigid_fit(source, target, query):
    """Fit a proper SE(3) with SciPy, falling back to mean translation when rank < 2."""
    source, target, query = (np.asarray(x, dtype=np.float64) for x in (source, target, query))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
        raise ValueError("Rigid fitting requires nonempty matching [N,3] point arrays")
    if not all(np.isfinite(x).all() for x in (source, target, query)):
        raise ValueError("Nonfinite rigid-fit input")
    center_x, center_y = source.mean(axis=0), target.mean(axis=0)
    x, y = source - center_x, target - center_y
    fallback = len(source) < 3
    if not fallback:
        fallback = any(np.linalg.svd(a, compute_uv=False)[1] <= 1e-8 for a in (x, y))
    rotation = Rotation.identity()
    if not fallback:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            rotation, _ = Rotation.align_vectors(y, x)
        if caught:
            fallback, rotation = True, Rotation.identity()
    fitted = rotation.apply(query - center_x) + center_y
    residual = epe(rotation.apply(x) + center_y, target)
    return fitted, fallback, residual


def hand_baselines(object_points, hand_points, hand_flow):
    """Only the same hand geometry/flow available to Cm enters these predictions."""
    mean_flow = np.broadcast_to(hand_flow.mean(axis=0), object_points.shape)
    fitted, fallback, residual = rigid_fit(hand_points, hand_points + hand_flow, object_points)
    return mean_flow, fitted - object_points, fallback, residual


def token_diagnostics(tokens, anchors, object_points):
    tokens = np.asarray(tokens, dtype=np.float64)
    centered = tokens - tokens.mean(axis=0)
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    if energy.sum() <= 1e-24:
        rank = 0.0
    else:
        mass = energy / energy.sum()
        positive = mass[mass > 0]
        rank = float(np.exp(-(positive * np.log(positive)).sum()))
    token_ratio = float(np.sqrt((centered ** 2).mean()) / max(np.sqrt((tokens ** 2).mean()), 1e-12))
    a = np.asarray(anchors, dtype=np.float64)
    p = np.asarray(object_points, dtype=np.float64)
    spread = np.sqrt(((a - a.mean(axis=0)) ** 2).sum(axis=-1).mean())
    radius = np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=-1).mean())
    return rank, token_ratio, float(spread / max(radius, 1e-12))


def cluster_summary(rows, keys=KEYS, seed=42, repeats=2000):
    if not rows:
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[row["sequence_id"]].append([row[key] for key in keys])
    arrays = [np.asarray(v, dtype=np.float64) for _, v in sorted(groups.items())]
    counts = np.array([len(a) for a in arrays], dtype=np.float64)
    sums = np.stack([a.sum(axis=0) for a in arrays])
    weights = np.random.default_rng(seed).multinomial(len(arrays), np.full(len(arrays), 1 / len(arrays)), repeats)
    draws = (weights @ sums) / (weights @ counts)[:, None]
    return {"samples": len(rows), "sequences": len(groups),
            "frame_micro": dict(zip(keys, (sums.sum(axis=0) / counts.sum()).tolist())),
            "sequence_macro": dict(zip(keys, (sums / counts[:, None]).mean(axis=0).tolist())),
            "frame_micro_ci95": dict(zip(keys, np.percentile(draws, [2.5, 97.5], axis=0).T.tolist()))}


def error_attribution(rows):
    result = {}
    for source in SOURCES:
        selected = [r for r in rows if r["source"] == source]
        if not selected:
            continue
        full = np.array([r["epe_mm"] for r in selected])
        zero = np.array([r["zero_epe_mm"] for r in selected])
        squared = np.array([r["mse_mm2"] for r in selected])
        contact = np.array([r["active_fraction"] for r in selected])

        def contribution(mask):
            return {"count": int(mask.sum()), "sample_fraction": float(mask.mean()),
                    "full_epe_fraction": float(full[mask].sum() / full.sum()),
                    "full_squared_error_fraction": float(squared[mask].sum() / squared.sum()),
                    "mean_epe_mm": float(full[mask].mean()) if mask.any() else None,
                    "zero_gate_improvement_total_mm": float((full[mask] - zero[mask]).sum() / len(full)),
                    "perfect_subset_improvement_upper_mm": float(full[mask].sum() / len(full))}

        record = {"quantiles_mm": dict(zip(("p50", "p90", "p95", "p99"), np.percentile(full, [50, 90, 95, 99]).tolist()))}
        for threshold in (.01, .05, .1, .25):
            record[f"active_below_{threshold}"] = contribution(contact < threshold)
        for fraction in (.01, .05, .1):
            mask = np.zeros(len(full), dtype=bool)
            mask[np.argsort(full)[-max(1, int(np.ceil(len(full) * fraction))):]] = True
            record[f"top_error_{fraction}"] = contribution(mask)
        record["near_static_gt_below_1mm"] = contribution(zero < 1)
        result[source] = record
    return result


def training_history(path):
    history = [json.loads(line) for line in path.open()]
    train = [r for r in history if "train_epoch/obj/flow_epe_mm" in r]
    val = [r for r in history if "val/obj/flow_epe_mm" in r]
    keys = ("epoch", "step", "train_epoch/obj/flow_epe_mm", "train_epoch/hand/flow_epe_knn_unique_mm",
            "val/obj/flow_epe_mm", "val/grab/obj/flow_epe_mm", "val/inspire_f1/obj/flow_epe_mm")
    compress = lambda r: {k: r[k] for k in keys if k in r}
    return {"first_train": compress(train[0]), "last_train": compress(train[-1]),
            "best_val": compress(min(val, key=lambda r: r["val/obj/flow_epe_mm"])),
            "last_val": compress(val[-1]),
            "warning": "Train random stride 1..10 vs val fixed 2: not a matched train/val generalization gap",
            "slot_metric_warning": "Normalized slot weights sum to one per slot; their mean-based effective_count is tautologically S"}


def run(args):
    if Path(args.run_id).name != args.run_id or args.run_id in {".", "..", ""}:
        raise ValueError("run-id must be a single new directory name")
    torch.set_num_threads(2)
    os.chdir(ROOT)
    started = time.monotonic()
    source = Path(args.source).resolve()
    config = json.loads((source / "config.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    archive_summary = json.loads((source / "effect_summary.json").read_text())
    if archive_summary["smoke_only"]:
        raise ValueError("Use the full frozen run as diagnostic input")
    records = [json.loads(line) for line in (source / "metrics.jsonl").open()]
    by_id = {r["sample_index"]: r for r in records}
    entries = metadata["entries"]
    artifacts = archive_summary["artifacts"]
    if args.smoke:
        chosen = [next(i for i, e in enumerate(entries) if e["source"] == s) for s in SOURCES]
        artifacts = [artifacts[i] for i in chosen]
    index_path = ROOT / config["data"]["index_path"]
    for path, digest in metadata["protected_digests"]:
        assert sha256(path) == digest, path
    for path, digest in metadata["code_digests"].items():
        assert sha256(ROOT / path) == digest, path
    for record in metadata["sequence_files"]:
        stat = Path(record["path"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (record["size_bytes"], record["mtime_ns"])
    out = Path(__file__).parent / "output" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    config_snapshot = {"work_version": "V1.3.2", "operation_category": ["diagnostic", "experiment"],
                       "train": {"seed": 42}, "source_output_path": str(source),
                       "index_path": str(index_path), "arguments": vars(args), "evaluation_partition": "val",
                       "bootstrap_repeats": 2000, "meta": config["meta"],
                       "source_run_id": source.name, "command": sys.argv, "no_learning": True,
                       "diagnostic_only_gt_use": "fit GT rigid residual and offline statistics; never hand baseline inputs"}
    input_files = [source / name for name in ("config.json", "metadata.json", "metrics.jsonl", "effect_summary.json", "run_manifest.json")]
    input_files += [source / item["file"] for item in artifacts]
    history_path = Path(config["checkpoint"]).parent.parent / "metrics.jsonl"
    input_files.append(history_path)
    input_hashes = {str(p): sha256(p) for p in input_files}
    snapshot = {"coordinate_frame": "object_pose_t", "units": "m", "num_obj_points": 1024,
                "source_metadata_path": str(source / "metadata.json"), "scipy_version": scipy.__version__,
                "numpy_version": np.__version__, "code_sha256": sha256(Path(__file__)),
                "input_digests": list(input_hashes.items()), "archived_weights_sha256": metadata["protected_digests"][0][1]}
    write_json(out / "config.json", config_snapshot)
    write_json(out / "metadata.json", snapshot)
    manifest = build_run_manifest(task="ObjectInteractionCm", run_name=args.run_id, output_dir=out,
                                  mode="diagnostic", config=config_snapshot,
                                  metadata={"coordinate_frame": "object_pose_t", "num_obj_points": 1024},
                                  config_source=Path(__file__).with_name("diagnostic.yaml"),
                                  initial_checkpoint=config["checkpoint"], repo_root=ROOT)
    assert all(r["exists"] for r in manifest["input_references"])
    write_run_manifest(out / "run_manifest.json", manifest)

    def log(message):
        line = datetime.now().astimezone().isoformat(timespec="seconds") + " " + message
        print(line, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    rows, fallbacks = [], 0
    log(f"START {args.run_id} sequences={len(artifacts)} smoke={args.smoke}")
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for artifact in artifacts:
            entry = next(e for e in entries if e["parent_seq_id"] == artifact["sequence_id"])
            dataset = ObjectInteractionCmDataset(
                ROOT, sequence_entries=[{"path": str((index_path.parent / entry["path"]).resolve()), "source": entry["source"]}],
                **{k: config["meta"][k] for k in ("num_obj_points", "num_hand_points", "max_hand_points", "max_knn_hand_points",
                                                  "hand_stream_mode", "knn_k", "hand_supervision_radius_m")},
                base_seed=42, min_stride=1, max_stride=10, fixed_stride=2, active_only=False)
            with np.load(source / artifact["file"], allow_pickle=False) as archive:
                ids = archive["sample_index"].tolist()
                selected = [(j, by_id[i]) for j, i in enumerate(ids) if by_id[i]["primary_valid"]]
                if args.smoke:
                    selected = selected[:3]
                points, targets, predictions = (archive[k] for k in ("obj_points", "gt_obj_flow", "pred_obj_flow"))
                tokens, anchors = archive["cm_tokens"], archive["cm_anchor_pos"]
                for j, row in selected:
                    if time.monotonic() - started > 900:
                        raise TimeoutError("15-minute CPU budget exceeded")
                    sample = dataset[row["frame_index"]]
                    assert int(sample["raw_frame_id"]) == row["raw_frame_id"]
                    assert int(sample["next_raw_frame_id"]) == row["next_raw_frame_id"]
                    assert int(sample["stride"]) == row["stride"] == 2
                    np.testing.assert_array_equal(sample["obj_points"].numpy(), points[j])
                    np.testing.assert_array_equal(sample["obj_flow_gt"].numpy(), targets[j])
                    p, gt, pred = (a[j].astype(np.float64) for a in (points, targets, predictions))
                    mask = sample["hand_valid_mask"].numpy()
                    assert mask.sum() == row["hand_valid_points"] > 0
                    hp = sample["hand_points"].numpy()[mask].astype(np.float64)
                    hf = sample["hand_flow"].numpy()[mask].astype(np.float64)
                    mean, rigid, fallback, hand_residual = hand_baselines(p, hp, hf)
                    fallbacks += int(fallback)
                    fitted_pred, _, pred_nonrigid = rigid_fit(p, p + pred, p)
                    _, _, gt_residual = rigid_fit(p, p + gt, p)
                    rank, slot_spread, anchor_spread = token_diagnostics(tokens[j], anchors[j], p)
                    full_error, mean_error, rigid_error, projected = (epe(x, gt) for x in (pred, mean, rigid, fitted_pred - p))
                    np.testing.assert_allclose(full_error, row["epe_mm"], atol=1e-4, rtol=2e-6)
                    values = dict(zip(KEYS, (full_error, row["zero_epe_mm"], mean_error, rigid_error, projected,
                        gt_residual, pred_nonrigid, full_error - mean_error, full_error - rigid_error,
                        full_error - projected, hand_residual, rank, slot_spread, anchor_spread)))
                    assert np.isfinite(list(values.values())).all()
                    record = {**row, **values, "rigid_hand_fallback": fallback}
                    rows.append(record)
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            log(f"sequence={entry['parent_seq_id']} samples={len(selected)} total={len(rows)}")
    assert len({r["sample_index"] for r in rows}) == len(rows)
    if not args.smoke:
        assert {r["sample_index"] for r in rows} == {r["sample_index"] for r in records if r["primary_valid"]}
    result = {"smoke_only": args.smoke, "source_run_id": source.name, "conclusion": "INCONCLUSIVE",
              "conclusion_scope": "Mechanism diagnostics, not cross-hand equivalence or learned ablation",
              "sources": {}, "strata": {}, "error_attribution": error_attribution(rows),
              "training_history": training_history(history_path), "rigid_hand_fallbacks": fallbacks,
              "engineering_checks": {"sample_ids_unique": True, "sampling_and_gt_replay_exact": True,
                                     "finite": True, "hand_baselines_use_no_object_gt": True}}
    for i, source_name in enumerate(SOURCES):
        subset = [r for r in rows if r["source"] == source_name]
        result["sources"][source_name] = cluster_summary(subset, seed=42 + i)
        result["strata"][source_name] = {name: cluster_summary([r for r in subset if predicate(r)], seed=42 + i)
            for name, predicate in (("low_contact_lt_0.1", lambda r: r["active_fraction"] < .1),
                                    ("high_contact_ge_0.25", lambda r: r["active_fraction"] >= .25),
                                    ("near_static_gt_lt_1mm", lambda r: r["zero_epe_mm"] < 1),
                                    ("moving_gt_ge_1mm", lambda r: r["zero_epe_mm"] >= 1))}
    assert all(sha256(p) == digest for p, digest in input_hashes.items())
    assert all(sha256(p) == digest for p, digest in metadata["protected_digests"])
    assert all((Path(r["path"]).stat().st_size, Path(r["path"]).stat().st_mtime_ns) ==
               (r["size_bytes"], r["mtime_ns"]) for r in metadata["sequence_files"])
    result["engineering_checks"]["inputs_unchanged"] = True
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "diagnosis.json", result)
    assert sum(p.stat().st_size for p in out.iterdir()) < 100 * 1024 ** 2
    log(f"COMPLETED samples={len(rows)} elapsed_s={time.monotonic()-started:.1f} fallbacks={fallbacks}")
    print(json.dumps({"output": str(out), "sources": result["sources"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
