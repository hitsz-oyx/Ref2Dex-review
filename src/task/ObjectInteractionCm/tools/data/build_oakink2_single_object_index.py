#!/usr/bin/env python3
"""Build a semantic OakInk2 train index containing only single-instance segments.

Object parts are collapsed through OakInk2's official ``object_part_tree``;
parts of one physical instance do not turn a segment into a multi-object one.
This index is metadata only.  It does not claim that every listed frame passes
the later motion or 2 cm geometric filters.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _root_map(tree: dict[str, list[str]]) -> dict[str, str]:
    parent = {child: node for node, children in tree.items() for child in children}
    roots: dict[str, str] = {}
    for node in set(parent) | set(tree):
        current = node
        while current in parent:
            current = parent[current]
        roots[node] = current
    return roots


def build(annotation_root: Path, object_root: Path, output: Path) -> dict:
    program_root = object_root / "program" / "program_info"
    tree = json.loads((object_root / "object_affordance" / "object_part_tree.json").read_text())
    roots = _root_map(tree)
    rows = []
    excluded_segments = 0
    for program_path in sorted(program_root.glob("*.json")):
        sequence = program_path.stem
        program = json.loads(program_path.read_text())
        for frame_def, item in program.items():
            object_ids = [str(x) for x in item.get("obj_list", [])]
            instance_ids = sorted({roots.get(obj, obj) for obj in object_ids})
            if len(instance_ids) != 1:
                excluded_segments += 1
                continue
            ranges = ast.literal_eval(frame_def)
            left_range, right_range = ranges
            rows.append({
                "sequence": sequence,
                "frame_range_def": frame_def,
                "frame_range_lh": list(left_range) if left_range is not None else None,
                "frame_range_rh": list(right_range) if right_range is not None else None,
                "frame_range": [min(x[0] for x in ranges if x is not None),
                                max(x[1] for x in ranges if x is not None)],
                "primitive": item.get("primitive"),
                "interaction_mode": item.get("interaction_mode"),
                "object_ids": object_ids,
                "instance_id": instance_ids[0],
                "obj_list_lh": item.get("obj_list_lh"),
                "obj_list_rh": item.get("obj_list_rh"),
            })
    payload = {
        "schema_name": "ref2dex_oakink2_single_object_segment_index_v1",
        "source": "OakInk2 program_info + object_part_tree",
        "annotation_root": str(annotation_root.resolve()),
        "object_root": str(object_root.resolve()),
        "selection": "exactly one object_part_tree root per primitive",
        "excluded_multi_object_segments": excluded_segments,
        "segment_count": len(rows),
        "segments": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return {k: payload[k] for k in ("schema_name", "segment_count", "excluded_multi_object_segments")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.annotation_root, args.object_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
