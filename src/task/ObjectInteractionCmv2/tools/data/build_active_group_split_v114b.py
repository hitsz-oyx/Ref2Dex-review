#!/usr/bin/env python3
"""Build a V1.14b split from an explicit subset of the six bilateral groups."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from src.task.ObjectInteractionCmv2.mixed_training import oakink_holdout, sha256_file, utc_now, write_json
from src.task.ObjectInteractionCmv2.multi_domain import normalize_hand_variant
from src.task.ObjectInteractionCmv2.part_se3_v114_training import load_v114_config


def build(config_path: Path, output: Path, run_id: str) -> dict:
    config = load_v114_config(config_path)
    if output.exists():
        raise FileExistsError(output)
    grouped = {}; fingerprints = {}
    for group in config["active_groups"]:
        domain, variant = group.split("/")
        source = config["sources"][group]
        for key in ("index", "manifest"):
            fingerprints[source[key]] = sha256_file(source[key])
        manifest = json.loads(Path(source["manifest"]).read_text())
        if int(manifest.get("validation", {}).get("bad_count", 0) or 0):
            raise ValueError(f"invalid source manifest: {group}")
        index = json.loads(Path(source["index"]).read_text())
        grouped[group] = {split: [] for split in ("train", "val", "test")}
        for split in grouped[group]:
            for row in index.get("sequences", {}).get(split, []):
                if str(row.get("dataset")) != domain:
                    continue
                actual = normalize_hand_variant(row.get("hand_variant", row.get("variant", row.get("source"))))
                if actual != variant:
                    continue
                path = Path(row["path"])
                if not path.is_absolute():
                    path = Path(source["index"]).parent / path
                if not (path / "geometry" / "manifest.json").is_file():
                    raise FileNotFoundError(path)
                grouped[group][split].append({**row, "path": str(path.resolve())})
    for domain in ("grab", "arctic", "oakink2"):
        variants = [group for group in config["active_groups"] if group.startswith(domain + "/")]
        if len(variants) < 2:
            continue
        identities = [{split: {str(row["id"]) for row in grouped[group][split]}
                       for split in ("train", "val", "test")} for group in variants]
        if any(value != identities[0] for value in identities[1:]):
            raise ValueError(f"{domain}: active hand variants disagree on IDs or source split")
    oakink_groups = [group for group in config["active_groups"] if group.startswith("oakink2/")]
    holdout = set()
    if oakink_groups:
        ids = [str(row["id"]) for row in grouped[oakink_groups[0]]["train"]]
        holdout = oakink_holdout(ids, int(config.get("training", {}).get("seed", 42)),
                                 float(config["data"]["oakink2_val_fraction"]))
    sequences = {split: [] for split in ("train", "val", "test")}
    for group in config["active_groups"]:
        domain, variant = group.split("/")
        for source_split, rows in grouped[group].items():
            for row in rows:
                split = source_split
                if domain == "oakink2":
                    recording = "/".join(str(row["id"]).split("/")[:2])
                    split = "val" if recording in holdout else "train"
                sequences[split].append({**row, "dataset": domain, "hand_variant": variant,
                                         "split": split, "source_split": source_split, "group": group})
    for split in ("train", "val"):
        present = {row["group"] for row in sequences[split]}
        if present != set(config["active_groups"]):
            raise ValueError(f"{split}: missing active groups {set(config['active_groups']) - present}")
        sequences[split].sort(key=lambda row: (config["active_groups"].index(row["group"]), str(row["id"])))
    counts = {split: dict(Counter(row["group"] for row in rows)) for split, rows in sequences.items()}
    output.mkdir(parents=True)
    write_json(output / "index.json", {"schema_name": "ref2dex_cmv2_active_group_index_v114b",
               "work_version": "V1.14", "active_groups": config["active_groups"],
               "group_weights": config["group_weights"], "sequences": sequences, "counts": counts,
               "oakink2_val_recordings": sorted(holdout)})
    write_json(output / "cache_manifest.json", {"schema_name": "ref2dex_cmv2_active_group_split_v114b",
               "work_version": "V1.14", "active_groups": config["active_groups"],
               "hand_sampling_contract": config["hand_sampling"]["contract"], "knn_k": 32,
               "source_sha256": fingerprints, "validation": {"bad_count": 0}, "counts": counts})
    write_json(output / "run_manifest.json", {"schema_name": "ref2dex_data_run_manifest_v1",
               "task": "ObjectInteractionCmv2", "work_version": "V1.14", "run_id": run_id,
               "run_status": "COMPLETED", "operation": "v114b_active_group_split",
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "finished_at": utc_now(), "inputs": fingerprints,
               "outputs": {"index": str(output / "index.json"),
                           "cache_manifest": str(output / "cache_manifest.json")}, "conclusion": "N/A"})
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.output, args.run_id), indent=2))


if __name__ == "__main__":
    main()
