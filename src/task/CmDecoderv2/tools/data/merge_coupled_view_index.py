"""Merge the new coupled train/val cache with the preserved MANO test cache."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupled-index", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, default=ROOT / "data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json")
    parser.add_argument("--old-cache", type=Path, default=ROOT / "data/processed_data/object_interaction_cm_dexplore_rl_v1_3")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    coupled = json.loads(args.coupled_index.read_text(encoding="utf-8"))
    old = json.loads(args.old_index.read_text(encoding="utf-8"))
    sequences = {"train": [], "val": [], "test": []}
    for split in ("train", "val"):
        for item in coupled["sequences"][split]:
            value = dict(item)
            value["path"] = str((args.coupled_index.parent / item["path"]).resolve())
            sequences[split].append(value)
    for item in old["sequences"]["test"]:
        value = dict(item)
        value["path"] = str((args.old_cache / item["path"]).resolve())
        sequences["test"].append(value)
    output = {**coupled, "schema_name": "ref2dex_object_interaction_cm_index_v1_2", "experiment_schema": "ref2dex_coupled_geometric_view_source_v1", "sequences": sequences,
              "split_contract": {"train": "inspire_rl_coupled_geometric", "val": "inspire_rl_coupled_geometric", "test": "mano_qualitative_only"},
              "counts": {key: len(value) for key, value in sequences.items()}}
    args.output_root.mkdir(parents=True)
    (args.output_root / "index.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema_name": "ref2dex_run_manifest_v1", "task": "CmDecoderv2", "operation": "merge_coupled_view_index", "run_id": args.output_root.name, "run_status": "COMPLETED", "work_version": "V1.1.13", "operation_category": ["data", "operation"], "created_at": datetime.now().astimezone().isoformat(timespec="seconds"), "base_commit": commit, "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)), "command": [sys.executable, *sys.argv], "counts": output["counts"], "outputs": {"index": str((args.output_root / "index.json").resolve())}, "conclusion": "SUPPORTED"}
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
