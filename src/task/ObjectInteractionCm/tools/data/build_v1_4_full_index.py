#!/usr/bin/env python3
"""Build a V1.4 multi-domain geometry index.

The formal index keeps the existing GRAB val/test split, puts all ARCTIC and
OakInk2 entries in train, and gives each present train domain an explicit
source name. ``--smoke`` creates a tiny, clearly marked index with one
sequence per domain reused across the three split keys; it is only for loader
and training wiring checks and must not be used as a scientific split.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SOURCE_STRIDES = tuple(range(1, 11))


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_complete_sequence(sequence: Path) -> bool:
    """Reject exporter temporary directories while accepting final geometry."""
    return ".partial" not in sequence.name and (sequence / "geometry" / "manifest.json").is_file()


def _entry(*, source: str, dataset: str, sequence: Path, identifier: str) -> dict[str, str]:
    return {
        "source": source,
        "id": identifier,
        "path": str(sequence.resolve()),
        "dataset": dataset,
    }


def _collect_tree(root: Path, *, source: str, dataset: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for manifest in sorted(root.glob("**/geometry/manifest.json")):
        sequence = manifest.parent.parent
        if not _is_complete_sequence(sequence):
            continue
        relative = sequence.relative_to(root).as_posix()
        entries.append(
            _entry(
                source=source,
                dataset=dataset,
                sequence=sequence,
                identifier=f"{dataset}/{relative}",
            )
        )
    return entries


def _collect_grab(split_root: Path, grab_root: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    entries = {split: [] for split in ("train", "val", "test")}
    missing: list[str] = []
    for split in entries:
        split_file = split_root / f"{split}.txt"
        seen: set[str] = set()
        for sequence_id in _lines(split_file):
            raw_path = Path(sequence_id)
            if raw_path.suffix == ".npz":
                raw_path = raw_path.parent
            candidates = [grab_root / raw_path]
            parts = raw_path.parts
            if len(parts) >= 2 and parts[0].lower() == "grab" and parts[1] == "ds4_full":
                candidates.append(grab_root.joinpath(*parts[2:]))
            sequence = next((candidate for candidate in candidates if _is_complete_sequence(candidate)), None)
            if sequence is not None:
                key = str(sequence.resolve())
                if key in seen:
                    continue
                seen.add(key)
                relative_id = sequence.relative_to(grab_root).as_posix()
                entries[split].append(
                    _entry(
                        source="grab",
                        dataset="grab",
                        sequence=sequence,
                        identifier=f"grab/{relative_id}",
                    )
                )
            else:
                missing.append(f"grab/{sequence_id}")
    return entries, missing


def _smoke_entries(
    entries: dict[str, list[dict[str, str]]],
    *,
    per_source: int,
) -> dict[str, list[dict[str, str]]]:
    if per_source <= 0:
        raise ValueError("--smoke-sequences-per-source must be positive")
    all_by_source: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        for item in entries[split]:
            all_by_source.setdefault(str(item["source"]), []).append(item)
    selected: dict[str, list[dict[str, str]]] = {}
    for source, values in sorted(all_by_source.items()):
        unique: dict[str, dict[str, str]] = {str(item["path"]): item for item in values}
        selected[source] = [unique[path] for path in sorted(unique)[:per_source]]
    required = {"grab", "arctic", "oakink2"}
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"Smoke index requires all three domains; missing={missing}")
    chosen = [item for source in sorted(required) for item in selected[source]]
    return {split: [dict(item) for item in chosen] for split in ("train", "val", "test")}


def _source_probabilities(entries: dict[str, list[dict[str, str]]]) -> dict[str, float]:
    sources = sorted({str(item["source"]) for item in entries["train"]})
    if not sources:
        raise ValueError("The index has no train sources")
    probability = 1.0 / len(sources)
    return {source: probability for source in sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grab-root", type=Path, required=True)
    parser.add_argument("--arctic-root", type=Path, required=True)
    parser.add_argument("--oak-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-sequences-per-source", type=int, default=1)
    args = parser.parse_args()

    entries, missing = _collect_grab(args.split_root, args.grab_root)
    if missing:
        raise SystemExit(f"missing GRAB geometry {len(missing)}; first={missing[:3]}")
    entries["train"].extend(_collect_tree(args.arctic_root, source="arctic", dataset="arctic"))
    entries["train"].extend(_collect_tree(args.oak_root, source="oakink2", dataset="oakink2"))
    if args.smoke:
        entries = _smoke_entries(entries, per_source=int(args.smoke_sequences_per_source))

    probabilities = _source_probabilities(entries)
    payload = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
        "schema_version": "1.2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_kind": "three_domain_smoke" if args.smoke else "three_domain_formal",
        "knn_k": 32,
        "object_pool_points": 4096,
        "model_object_points": 1024,
        "hand_points_per_stream": 3076,
        "max_union_hand_points": 3076,
        "source_domains": sorted(probabilities),
        "source_probability_policy": "equal",
        "source_probability": probabilities,
        "stride_policy": {source: list(SOURCE_STRIDES) for source in sorted(probabilities)},
        "split_policy": (
            {
                "smoke": "one completed sequence per domain reused across train/val/test; engineering only",
                "grab": "smoke representative",
                "arctic": "smoke representative",
                "oakink2": "smoke representative",
            }
            if args.smoke
            else {
                "grab": "existing InteractionTransfer grab_seed42 split; val/test unchanged",
                "arctic": "all train",
                "oakink2": "single-object geometry only, train",
            }
        ),
        "sequences": entries,
        "counts": {split: len(values) for split, values in entries.items()},
        "domain_counts": {
            split: {
                source: sum(1 for item in values if item["source"] == source)
                for source in sorted(probabilities)
            }
            for split, values in entries.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": payload["counts"], "domain_counts": payload["domain_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
