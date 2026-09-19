"""Export a Dexplore-compatible view of native Inspire motion directories.

The full Inspire export intentionally contains every GRAB sequence.  The
normal Dexplore loader, however, excludes sequences with any left-hand contact
and skips names containing ``doorknob`` before loading them.  This utility
materializes that exact filtered view without modifying the full export.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch


CONTACT_LEFT_START = 206
CONTACT_LEFT_END = 222
MOTION_FILENAME = "interaction_hand_inspire.pt"


def _load_motion(path: Path) -> torch.Tensor:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] < CONTACT_LEFT_END:
        raise ValueError(f"Invalid Inspire tensor: {path} (shape={getattr(data, 'shape', None)})")
    return data


def _sequence_names(root: Path) -> List[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Motion root does not exist: {root}")
    return sorted(
        path.parent.name
        for path in root.glob(f"*/{MOTION_FILENAME}")
        if path.is_file()
    )


def _classify(root: Path, names: List[str]) -> Dict[str, object]:
    kept_no_left: List[str] = []
    left_contact: List[str] = []
    for index, name in enumerate(names, 1):
        motion = _load_motion(root / name / MOTION_FILENAME)
        has_left_contact = bool(
            (motion[:, CONTACT_LEFT_START:CONTACT_LEFT_END] > 0).any().item()
        )
        if has_left_contact:
            left_contact.append(name)
        else:
            kept_no_left.append(name)
        if index % 100 == 0 or index == len(names):
            print(f"Scanned {index}/{len(names)} motions", flush=True)
    doorknob_no_left = [name for name in kept_no_left if "doorknob" in name]
    return {
        "kept_no_left": kept_no_left,
        "left_contact": left_contact,
        "doorknob_no_left": doorknob_no_left,
    }


def _materialize(source_root: Path, output_root: Path, names: List[str]) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists; remove it explicitly before rerunning: {output_root}"
        )
    output_root.mkdir(parents=True)
    for index, name in enumerate(names, 1):
        source = source_root / name / MOTION_FILENAME
        destination = output_root / name / MOTION_FILENAME
        destination.parent.mkdir()
        shutil.copy2(source, destination)
        if index % 100 == 0 or index == len(names):
            print(f"Copied {index}/{len(names)} motions to {output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometric-root", type=Path, required=True,
        help="Full geometric Inspire motion root",
    )
    parser.add_argument(
        "--rl-root", type=Path, required=True,
        help="Full RL Inspire motion root",
    )
    parser.add_argument(
        "--output-geometric-root", type=Path, required=True,
        help="Output root for filtered geometric motions",
    )
    parser.add_argument(
        "--output-rl-root", type=Path, required=True,
        help="Output root for filtered RL motions",
    )
    args = parser.parse_args()

    geometric_names = _sequence_names(args.geometric_root)
    rl_names = _sequence_names(args.rl_root)
    if geometric_names != rl_names:
        only_geometric = sorted(set(geometric_names) - set(rl_names))
        only_rl = sorted(set(rl_names) - set(geometric_names))
        raise ValueError(
            "Geometric/RL sequence sets differ: "
            f"only geometric={only_geometric[:5]}, only RL={only_rl[:5]}"
        )

    stats = _classify(args.geometric_root, geometric_names)
    no_left = list(stats["kept_no_left"])
    doorknob_no_left = list(stats["doorknob_no_left"])
    kept = [name for name in no_left if "doorknob" not in name]
    if not kept:
        raise RuntimeError("No motions remain after Dexplore-compatible filtering")

    _materialize(args.geometric_root, args.output_geometric_root, kept)
    _materialize(args.rl_root, args.output_rl_root, kept)

    manifest = {
        "format": "dexplore_inspire_filtered_v1",
        "source_geometric_root": str(args.geometric_root.resolve()),
        "source_rl_root": str(args.rl_root.resolve()),
        "filter": {
            "exclude_any_left_hand_contact": True,
            "left_contact_columns": [CONTACT_LEFT_START, CONTACT_LEFT_END],
            "exclude_name_contains": ["doorknob"],
        },
        "num_source": len(geometric_names),
        "num_no_left_contact": len(no_left),
        "num_left_contact_excluded": len(stats["left_contact"]),
        "num_doorknob_excluded_after_contact_filter": len(doorknob_no_left),
        "num_kept": len(kept),
        "excluded_left_contact_examples": list(stats["left_contact"])[:20],
        "excluded_doorknob_examples": doorknob_no_left,
    }
    for output_root in (args.output_geometric_root, args.output_rl_root):
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
