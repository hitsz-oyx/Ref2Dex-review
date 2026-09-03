"""Build a lightweight GRAB + Inspire-F1 index for ObjectInteractionCm V1.1.

The index references existing mmap/layered cache directories and does not copy
the large point arrays.  It intentionally fails when a split entry cannot be
resolved, so missing or damaged sequences cannot silently enter training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_GRAB_ROOT = REPO_ROOT / "data/processed_data/cm_object_v2_surface512_object_pose_20260830"
DEFAULT_GRAB_SPLIT = DEFAULT_GRAB_ROOT / "splits_seed42/split.json"
DEFAULT_REPAIR_ROOT = REPO_ROOT / "data/processed_data/object_interaction_cm_grab_repairs_20260903"
DEFAULT_INSPIRE_ROOT = REPO_ROOT / "data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4"
DEFAULT_INSPIRE_SELECTION = DEFAULT_INSPIRE_ROOT / "selection_all_object_disjoint_seed42.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/processed_data/object_interaction_cm_v1_1/index.json"


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _grab_entries(root: Path, split_path: Path, repair_root: Path) -> dict[str, list[dict[str, str]]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, str]]] = {}
    for name in ("train", "val", "test"):
        file_name = split.get(f"{name}_split")
        if not file_name:
            raise ValueError(f"GRAB split is missing {name}_split")
        split_file = split_path.parent / file_name
        entries = []
        for item in _read_lines(split_file):
            sequence = Path(item)
            original = (root / sequence).resolve()
            repaired = (repair_root / sequence).resolve()
            path = repaired if (repaired / "shared/meta.json").is_file() else original
            if not (path / "shared/meta.json").is_file():
                raise FileNotFoundError(f"GRAB split entry has no cache: {path}")
            entries.append({"source": "grab", "id": sequence.as_posix(), "path": str(path)})
        result[name] = entries
    return result


def _inspire_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.parts and relative.parts[0] == root.name:
        relative = Path(*relative.parts[1:])
    return (root / relative).resolve()


def _inspire_entries(root: Path, selection_path: Path) -> dict[str, list[dict[str, str]]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cache_dirs = selection.get("cache_dirs", {})
    result: dict[str, list[dict[str, str]]] = {}
    for name in ("train", "val", "test"):
        entries = []
        for episode_id in selection.get("splits", {}).get(name, []):
            if episode_id not in cache_dirs:
                raise KeyError(f"Inspire split entry missing cache_dirs mapping: {episode_id}")
            path = _inspire_path(root, str(cache_dirs[episode_id]))
            if not (path / "geometry/manifest.json").is_file():
                raise FileNotFoundError(f"Inspire split entry has no geometry manifest: {path}")
            entries.append({"source": "inspire_f1", "id": str(episode_id), "path": str(path)})
        result[name] = entries
    return result


def _relativize(entries: dict[str, list[dict[str, str]]], base: Path) -> dict[str, list[dict[str, str]]]:
    return {
        split: [
            {**entry, "path": os.path.relpath(entry["path"], base)}
            for entry in values
        ]
        for split, values in entries.items()
    }


def build_index(*, grab_root: Path, grab_split: Path, repair_root: Path, inspire_root: Path,
                inspire_selection: Path, output: Path) -> Path:
    grab = _grab_entries(grab_root.resolve(), grab_split.resolve(), repair_root.resolve())
    inspire = _inspire_entries(inspire_root.resolve(), inspire_selection.resolve())
    merged = {
        split: grab[split] + inspire[split]
        for split in ("train", "val", "test")
    }
    merged = _relativize(merged, output.resolve().parent)
    payload: dict[str, Any] = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_1",
        "schema_version": "1.1.0",
        "created_at": "2026-09-03",
        "source_probability": {"grab": 0.5, "inspire_f1": 0.5},
        "stride_policy": {"grab": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "inspire_f1": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]},
        "object_pool_points": 4096,
        "model_object_points": 1024,
        "hand_points_per_stream": 1538,
        "max_union_hand_points": 3076,
        "split_policy": {"grab": "existing sequence split", "inspire_f1": "existing object-disjoint split"},
        "source_roots": {"grab": str(grab_root.resolve()), "inspire_f1": str(inspire_root.resolve()), "grab_repairs": str(repair_root.resolve())},
        "sequences": merged,
        "counts": {split: {source: sum(1 for item in merged[split] if item["source"] == source) for source in ("grab", "inspire_f1")} for split in merged},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grab-root", default=str(DEFAULT_GRAB_ROOT))
    parser.add_argument("--grab-split", default=str(DEFAULT_GRAB_SPLIT))
    parser.add_argument("--repair-root", default=str(DEFAULT_REPAIR_ROOT))
    parser.add_argument("--inspire-root", default=str(DEFAULT_INSPIRE_ROOT))
    parser.add_argument("--inspire-selection", default=str(DEFAULT_INSPIRE_SELECTION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    path = build_index(
        grab_root=Path(args.grab_root), grab_split=Path(args.grab_split), repair_root=Path(args.repair_root),
        inspire_root=Path(args.inspire_root), inspire_selection=Path(args.inspire_selection), output=Path(args.output),
    )
    print(path)


if __name__ == "__main__":
    main()
