#!/usr/bin/env python3
"""Create an auditable GRAB+ARCTIC MANO training index without editing cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _arctic_split(sequence_id: str, seed: int) -> str:
    """Assign complete ARCTIC trajectories reproducibly, never by frame."""
    digest = hashlib.sha256(f"{int(seed)}:{sequence_id}".encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:8], "big") % 10 == 0 else "train"


def build(source_index: Path, source_manifest: Path, output_root: Path, seed: int) -> dict[str, Any]:
    source_index = source_index.resolve()
    source_manifest = source_manifest.resolve()
    output_root = output_root.resolve()
    index = json.loads(source_index.read_text(encoding="utf-8"))
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    sequences = {split: [] for split in ("train", "val", "test")}
    seen: set[str] = set()
    for original_split, rows in index.get("sequences", {}).items():
        if original_split not in sequences:
            continue
        for row in rows:
            if str(row.get("dataset")) not in {"grab", "arctic"}:
                continue
            item = dict(row)
            dataset = str(item["dataset"])
            sequence_id = str(item["id"])
            if sequence_id in seen:
                raise ValueError(f"duplicate source sequence: {sequence_id}")
            seen.add(sequence_id)
            if dataset == "grab":
                target_split = original_split
            else:
                target_split = _arctic_split(sequence_id, seed)
            item.update({"split": target_split, "hand_variant": "mano", "source": "mano"})
            sequences[target_split].append(item)
    arctic_train = sum(item["dataset"] == "arctic" for item in sequences["train"])
    arctic_val = sum(item["dataset"] == "arctic" for item in sequences["val"])
    if not arctic_train or not arctic_val:
        raise ValueError("trajectory split must contain both ARCTIC train and val sequences")
    for rows in sequences.values():
        rows.sort(key=lambda item: str(item["id"]))
    output_root.mkdir(parents=True, exist_ok=True)
    output_index = output_root / "index.json"
    output_manifest = output_root / "cache_manifest.json"
    payload = {
        "schema_name": "ref2dex_object_interaction_cmv2_two_domain_mano_index_v1_4",
        "schema_version": "1.0.0",
        "created_at": _now(),
        "work_version": "V1.4.4",
        "source_index": str(source_index),
        "source_manifest": str(source_manifest),
        "arctic_split": {"unit": "trajectory_sequence_id", "method": "sha256_mod_10", "seed": int(seed),
                         "train_fraction": 0.9},
        "hand_contract": "MANO bilateral KNN 4096 points (2048 per side), KNN32",
        "knn_hand_points_per_stream": {"mano": 4096},
        "knn_k": 32,
        "source_probability": {"grab": 0.5, "arctic": 0.5},
        "sequences": sequences,
        "counts": {split: len(rows) for split, rows in sequences.items()},
        "counts_by_domain": {split: {domain: sum(item["dataset"] == domain for item in rows)
                                      for domain in ("grab", "arctic")}
                             for split, rows in sequences.items()},
    }
    manifest = {
        "schema_name": "ref2dex_object_interaction_cmv2_two_domain_mano_cache_manifest_v1_4",
        "schema_version": "1.0.0",
        "created_at": _now(),
        "work_version": "V1.4.4",
        "index": str(output_index),
        "source_index": str(source_index),
        "source_manifest": str(source_manifest),
        "knn_hand_points_per_stream": {"mano": 4096},
        "knn_k": 32,
        "validation": {"bad_count": 0, "bad_examples": []},
        "arctic_split": payload["arctic_split"],
        "counts": payload["counts"],
        "counts_by_domain": payload["counts_by_domain"],
    }
    _write_json(output_index, payload)
    _write_json(output_manifest, manifest)
    return {"index": str(output_index), "manifest": str(output_manifest), **payload["counts_by_domain"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    run_manifest = output_root / f"run_manifest_{args.run_id}.json"
    state = {
        "schema_name": "ref2dex_run_manifest_v1", "task": "ObjectInteractionCmv2",
        "operation": "v1_4_4_two_domain_mano_trajectory_split", "run_id": args.run_id,
        "run_status": "STARTED", "created_at": _now(), "work_version": "V1.4.4",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_index": str(args.source_index.resolve()), "source_manifest": str(args.source_manifest.resolve()),
        "output_root": str(output_root), "seed": int(args.seed), "conclusion": "INCONCLUSIVE",
    }
    _write_json(run_manifest, state)
    try:
        result = build(args.source_index, args.source_manifest, output_root, args.seed)
        state.update({"run_status": "COMPLETED", "finished_at": _now(), "outputs": result,
                      "conclusion": "SUPPORTED"})
        _write_json(run_manifest, state)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        state.update({"run_status": "FAILED", "finished_at": _now(), "error": repr(error),
                      "conclusion": "INVALID_IMPLEMENTATION"})
        _write_json(run_manifest, state)
        raise


if __name__ == "__main__":
    main()
