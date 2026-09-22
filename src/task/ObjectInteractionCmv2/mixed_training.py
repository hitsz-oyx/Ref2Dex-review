"""Frozen five-source splits and balanced batches for V1.11h/i training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .articulated import ArticulatedTransitions
from .multi_domain import ThreeDomainTransitions, normalize_hand_variant, sha256_file
from .oakink2_parts import OakInkPartTransitions


WORK_VERSION = "V1.11.1"
CONFIG_SCHEMAS = {
    "object_interaction_cmv2_mixed_ddp_v1_11h": (1, 3),
    "object_interaction_cmv2_mixed_ddp_v1_11i": (0, 2),
}
DEFAULT_NOFILE_LIMIT = 262144
GROUPS = ("grab/mano", "arctic/mano", "oakink2/mano", "grab/inspire_f1", "oakink2/inspire_f1")
GROUP_WEIGHTS = dict(zip(GROUPS, (1 / 6, 1 / 6, 1 / 6, 1 / 4, 1 / 4)))
REPO_ROOT = Path(__file__).resolve().parents[3]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def normalize_resources(config):
    schema = config.get("schema_name")
    expected_gpus = CONFIG_SCHEMAS.get(schema)
    if expected_gpus is None:
        raise ValueError("unsupported mixed training config")
    resources = config.get("resources") or {}
    physical_gpus = tuple(int(value) for value in resources.get("physical_gpus", expected_gpus))
    if physical_gpus != expected_gpus:
        raise ValueError(f"{schema} requires physical GPUs {list(expected_gpus)}")
    nofile_limit = int(resources.get("nofile_limit", DEFAULT_NOFILE_LIMIT))
    if nofile_limit < DEFAULT_NOFILE_LIMIT:
        raise ValueError("nofile_limit is below the approved memmap-loading minimum")
    config["resources"] = {"physical_gpus": list(physical_gpus), "nofile_limit": nofile_limit}
    return config["resources"]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    normalize_resources(config)
    if config.get("work_version") != WORK_VERSION or set(config["sources"]) != set(GROUPS):
        raise ValueError("work_version or five-source contract mismatch")
    training, data = config["training"], config["data"]
    required = {"batch_size_per_rank": 64, "world_size": 2, "epochs": 16,
                "seed": 42, "learning_rate": 0.001, "checkpoint_interval": 200}
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("unapproved training budget")
    if (data["train_strides"] != [1, 2, 3] or data["validation_strides"] != [1, 2, 3]
            or data["num_obj_points"] != 1024 or data["oakink2_val_fraction"] != 0.1):
        raise ValueError("unapproved data contract")
    expected_model = {"architecture_version": "v1_5_articulated_fk", "hidden_width": 128, "knn_k": 32,
                      "interaction_radius_m": 0.02, "feature_scale_m": 0.02, "frame_dt_s": 1 / 30}
    if config["model"] != expected_model:
        raise ValueError("unapproved model contract")
    part_config = config.get("oakink2_parts")
    if set(part_config or {}) != {"adapter_root", "selection_index", "annotation_root", "stage3_root"}:
        raise ValueError("approved OakInk2 part-adapter inputs are required")
    for key in ("checkpoint", "articulation_metadata", "output_root", "split_root"):
        candidate = Path(config[key])
        config[key] = str((candidate if candidate.is_absolute() else REPO_ROOT / candidate).resolve())
    for key, value in part_config.items():
        candidate = Path(value)
        part_config[key] = str((candidate if candidate.is_absolute() else REPO_ROOT / candidate).resolve())
    for source in config["sources"].values():
        for key in ("index", "manifest"):
            source[key] = str(Path(source[key]).resolve())
    return config


def recording_id(sequence_id):
    parts = sequence_id.split("/")
    if len(parts) != 3 or parts[0] != "oakink2" or not parts[1] or not parts[2]:
        raise ValueError(f"unrecognized OakInk2 recording/segment ID: {sequence_id}")
    return "/".join(parts[:2])


def oakink_holdout(sequence_ids, seed=42, fraction=0.1):
    recordings = sorted({recording_id(value) for value in sequence_ids})
    if len(recordings) < 2 or not 0 < fraction < 1:
        raise ValueError("holdout needs at least two recordings and a valid fraction")
    ordered = sorted(recordings, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())
    count = max(1, math.ceil(len(ordered) * fraction))
    if count >= len(ordered):
        raise ValueError("holdout would leave no training recordings")
    return set(ordered[:count])


def combine_entries(grouped, seed=42, fraction=0.1):
    if set(grouped) != set(GROUPS):
        raise ValueError("exactly five groups are required")
    identities = {}
    for group, splits in grouped.items():
        seen = {}
        for split in ("train", "val", "test"):
            for entry in splits[split]:
                sequence_id = entry["id"]
                if sequence_id in seen:
                    raise ValueError(f"duplicate or leaking ID: {group}/{sequence_id}")
                seen[sequence_id] = split
        identities[group] = seen
    for domain in ("grab", "oakink2"):
        if identities[f"{domain}/mano"] != identities[f"{domain}/inspire_f1"]:
            raise ValueError(f"{domain}: variants disagree on sequence identities or existing split")
    for group in ("oakink2/mano", "oakink2/inspire_f1"):
        if grouped[group]["val"] or grouped[group]["test"]:
            raise ValueError("OakInk2 input is no longer the approved all-train cache")
    holdout = oakink_holdout(identities["oakink2/mano"], seed, fraction)
    output = {split: [] for split in ("train", "val", "test")}
    for group in GROUPS:
        domain, variant = group.split("/")
        for source_split, entries in grouped[group].items():
            for entry in entries:
                split = source_split
                if domain == "oakink2":
                    split = "val" if recording_id(entry["id"]) in holdout else "train"
                output[split].append({**entry, "dataset": domain, "hand_variant": variant,
                                      "split": split, "source_split": source_split, "group": group})
    for entries in output.values():
        entries.sort(key=lambda entry: (entry["group"], entry["id"]))
    for split in ("train", "val"):
        if {entry["group"] for entry in output[split]} != set(GROUPS):
            raise ValueError(f"missing train/validation group in {split}")
    return output, sorted(holdout)


def create_split(config, output, run_id):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    grouped, fingerprints = {}, {}
    for group in GROUPS:
        domain, variant = group.split("/")
        source = config["sources"][group]
        for key in ("index", "manifest"):
            fingerprints[source[key]] = sha256_file(source[key])
        manifest = json.loads(Path(source["manifest"]).read_text())
        if int(manifest.get("validation", {}).get("bad_count", 0) or 0):
            raise ValueError(f"invalid source cache: {group}")
        index = json.loads(Path(source["index"]).read_text())
        grouped[group] = {}
        for split in ("train", "val", "test"):
            selected = []
            for row in index["sequences"][split]:
                if row.get("dataset") != domain:
                    continue
                actual_variant = normalize_hand_variant(row.get("hand_variant", row.get("variant", row.get("source"))))
                if actual_variant != variant:
                    continue
                path = Path(row["path"])
                if not path.is_absolute():
                    path = Path(source["index"]).parent / path
                if not (path / "geometry" / "manifest.json").is_file():
                    raise FileNotFoundError(path)
                selected.append({**row, "path": str(path.resolve())})
            grouped[group][split] = selected
    sequences, holdout = combine_entries(grouped, config["training"]["seed"], config["data"]["oakink2_val_fraction"])
    counts = {split: dict(Counter(row["group"] for row in rows)) for split, rows in sequences.items()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "index.json", {"schema_name": "ref2dex_cmv2_mixed_index_v1", "work_version": WORK_VERSION,
               "sequences": sequences, "counts": counts, "oakink2_val_recordings": holdout,
               "split_policy": "existing GRAB/ARCTIC; seeded 10% OakInk2 recording-group holdout shared by variants"})
    write_json(output / "cache_manifest.json", {"schema_name": "ref2dex_cmv2_mixed_split_v1", "work_version": WORK_VERSION,
               "knn_hand_points_per_stream": {"mano": 4096, "inspire_f1": 20270}, "knn_k": 32,
               "validation": {"bad_count": 0}, "source_sha256": fingerprints, "counts": counts})
    write_json(output / "run_manifest.json", {"task": "ObjectInteractionCmv2", "work_version": WORK_VERSION,
               "run_id": run_id, "operation": "frozen_five_source_split", "run_status": "COMPLETED",
               "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
               "seed": config["training"]["seed"], "config": "config.json", "schema": "V1.4 geometry unchanged",
               "coordinates": "geometry unchanged; training uses current object/root local, metres, seconds, radians",
               "finished_at": utc_now(), "source_sha256": fingerprints,
               "outputs": {"index": str(output / "index.json"), "cache_manifest": str(output / "cache_manifest.json")},
               "conclusion": "N/A"})
    return counts


def build_dataset(config, split_root, group, split, fixed_stride=None, max_sequences=None):
    domain, variant = group.split("/")
    split_root = Path(split_root)
    if domain == "oakink2":
        dataset = ThreeDomainTransitions(
            [{"name": domain, "hand_variant": variant, "index": str(split_root / "index.json"),
              "manifest": str(split_root / "cache_manifest.json")}], split,
            num_obj_points=config["data"]["num_obj_points"], train_stride_values={domain: config["data"]["train_strides"]},
            fixed_stride=fixed_stride, active_only=False, base_seed=config["training"]["seed"],
            max_sequences_per_domain=max_sequences, allow_manifest_split_override=True)
        return OakInkPartTransitions(dataset, config["oakink2_parts"]["adapter_root"])
    index = json.loads((split_root / "index.json").read_text())
    articulation = json.loads(Path(config["articulation_metadata"]).read_text())["articulation"]
    rows = [row for row in index["sequences"][split] if row["group"] == group]
    if max_sequences:
        rows = rows[:max_sequences]
    specs = [{"name": domain, "hand_variant": variant, "path": row["path"], "id": row["id"],
              "articulation": articulation if domain == "arctic" else {"num_links": 1, "joints": []}} for row in rows]
    return ArticulatedTransitions(specs, split, num_obj_points=config["data"]["num_obj_points"],
                                 stride_values=[fixed_stride] if fixed_stride else config["data"]["train_strides"],
                                 base_seed=config["training"]["seed"])


def batch_counts(step, batch_size=64):
    if batch_size != 64 or step < 0:
        raise ValueError("approved per-rank batch is 64 and step must be nonnegative")
    counts = dict(zip(GROUPS, (10, 10, 10, 16, 16)))
    for extra in range(2):
        counts[GROUPS[(step + extra) % 3]] += 1
    return counts


def sample_batch(datasets, generator, step):
    samples = []
    for group, count in batch_counts(step).items():
        dataset = datasets[group]
        for index in torch.randint(len(dataset), (count,), generator=generator).tolist():
            if isinstance(dataset, OakInkPartTransitions):
                component = int(torch.randint(dataset.component_count(index), (1,), generator=generator).item())
                sample = dataset.sample_component(index, component)
            else:
                sample = dataset[index]
            samples.append({**sample, "hand_variant": group.split("/")[1]})
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    configuration = load_config(args.config)
    print(json.dumps(create_split(configuration, configuration["split_root"], args.run_id), indent=2))
