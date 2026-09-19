#!/usr/bin/env python3
"""Select OakInk2 frames for active-tool-only ObjectInteractionCm samples.

The command writes a compact selection index.  It does not copy the large
Stage3 arrays.  Object part IDs are collapsed through OakInk2's official
``object_part_tree``; among distinct instances, exactly one active tool is
required.  Motion is measured before the 2 cm hand/object filter.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Names that denote an actively operated tool in the OakInk2 object catalog.
# Objects outside this set are still allowed when a primitive has only one
# object instance (e.g. opening a laptop), but multi-instance rows need a
# unique member from this set.
TOOL_NAMES = {
    "scissors", "orange cleaver knife", "white cleaver knife", "white fruit knife",
    "spoon", "glass rod", "rod", "fork", "tongs", "sponge", "toothbrush",
    "stapler", "pen", "whiteboard pen", "pencil", "pencil sharpener",
    "ignitor", "tape", "power plug", "usb stick", "mouse",
}


# OakInk2 has six known primitives whose object list is empty or broader than
# the natural-language primitive.  Keep the correction table keyed by the
# immutable sequence/range pair so the producer never guesses from names.
SEGMENT_OBJECT_OVERRIDES = {
    (
        "scene_02__A001++seq__cebe99032fc48d6e5bd9__2023-04-15-19-59-32",
        "((2784, 4713), None)",
    ): {
        "object_ids": ["O02@0053@00001"],
        "hand_sides": ["left"],
        "reason": "open_laptop_lid_missing_object_list",
    },
    (
        "scene_02__A002++seq__4635d47c6f3886b08da2__2023-04-17-14-51-43",
        "((360, 1241), (773, 1241))",
    ): {
        "object_ids": ["O02@0053@00001"],
        "hand_sides": ["left", "right"],
        "reason": "close_laptop_lid_missing_object_list",
    },
    (
        "scene_02__A004++seq__3efc8765c5fecbe7385e__2023-04-26-20-45-15",
        "((716, 1092), None)",
    ): {
        "object_ids": ["O02@0019@00002"],
        "hand_sides": ["left"],
        "reason": "pull_out_drawer_missing_object_list",
    },
    (
        "scene_02__A004++seq__9cf448429878c58036ef__2023-05-07-18-47-31",
        "((1465, 1873), None)",
    ): {
        "object_ids": ["O02@0053@00001"],
        "hand_sides": ["left"],
        "reason": "close_laptop_lid_excludes_incidental_gamecontroller",
    },
    (
        "scene_02__A007++seq__9a0f56cc00f8f61c02f8__2023-04-19-10-01-54",
        "((622, 1640), (633, 1652))",
    ): {
        "object_ids": ["O02@0060@00001", "O02@0060@00002"],
        "hand_sides": ["left", "right"],
        "reason": "open_book_missing_object_list",
    },
    (
        "scene_03__A007++seq__3680e9111f1211099630__2023-04-20-15-47-28",
        "((697, 1183), None)",
    ): {
        "object_ids": ["O02@0205@00001"],
        "hand_sides": ["left"],
        "reason": "place_asbestos_mesh_missing_object_list",
    },
}


def _roots(tree: dict[str, list[str]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    parent = {child: node for node, children in tree.items() for child in children}
    root_of: dict[str, str] = {}
    for node in set(parent) | set(tree):
        cur = node
        while cur in parent:
            cur = parent[cur]
        root_of[node] = cur
    members: dict[str, list[str]] = {}
    for node, root in root_of.items():
        members.setdefault(root, []).append(node)
    return root_of, members


def _rot_angle(a: np.ndarray, b: np.ndarray) -> float:
    r = a.T @ b
    cosine = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _motion_span(poses: dict[int, np.ndarray], frame_ids: list[int],
                 translation_m: float, rotation_deg: float) -> tuple[int, int] | None:
    usable = [fid for fid in frame_ids if fid in poses]
    if len(usable) < 2:
        return None
    moving = []
    for prev, cur in zip(usable[:-1], usable[1:]):
        p0, p1 = poses[prev], poses[cur]
        dt = float(np.linalg.norm(p1[:3, 3] - p0[:3, 3]))
        da = math.degrees(_rot_angle(p0[:3, :3], p1[:3, :3]))
        if dt > translation_m or da > rotation_deg:
            moving.extend((prev, cur))
    if not moving:
        return None
    return min(moving), max(moving)


def _frame_distances(stage3_by_key: dict[tuple[str, str, str], Path],
                     sequence: str, object_ids: list[str], frame_ids: list[int],
                     sides: tuple[str, ...],
                     cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] | None = None,
                     ) -> dict[int, float]:
    """Return the minimum hand-to-object distance over parts and both hands."""
    result = {int(fid): float("inf") for fid in frame_ids}
    wanted = set(result)
    for object_id in object_ids:
        for side in sides:
            path = stage3_by_key.get((sequence, object_id, side))
            if path is None:
                continue
            key = (sequence, object_id, side)
            cached = cache.get(key) if cache is not None else None
            if cached is None:
                with np.load(path, allow_pickle=False) as data:
                    ids = np.asarray(data["raw_frame_id"], dtype=np.int64)
                    dist = np.asarray(data["hand_to_obj_min_dist"], dtype=np.float32)
                if cache is not None:
                    cache[key] = (ids, dist)
            else:
                ids, dist = cached
            for i, fid in enumerate(ids.tolist()):
                if int(fid) in wanted:
                    value = float(np.min(dist[i]))
                    if value < result[int(fid)]:
                        result[int(fid)] = value
    return result


def _load_stage3_map(stage3_root: Path) -> dict[tuple[str, str, str], Path]:
    mapping: dict[tuple[str, str, str], Path] = {}
    stats_files = sorted(stage3_root.glob("oakink2_stage3_stats_*.json"))
    if not stats_files:
        raise FileNotFoundError(f"No oakink2_stage3_stats_*.json under {stage3_root}")
    for stats_path in stats_files:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        for row in payload.get("outputs", []):
            seq_id = str(row["seq_id"])
            parts = seq_id.rsplit("/", 2)
            if len(parts) != 3:
                continue
            sequence, object_id, side = parts
            path = Path(str(row["file"]))
            if not path.is_file():
                path = stage3_root / path.name
            if path.is_file():
                mapping[(sequence, object_id, side)] = path
    return mapping


def _resolve_active_roots(
    *,
    sequence: str,
    frame_def: str,
    item: dict,
    root_of: dict[str, str],
    members: dict[str, list[str]],
    names: dict[str, str],
) -> dict:
    """Resolve one primitive without silently choosing between active roots."""
    original_object_ids = [str(x) for x in item.get("obj_list", [])]
    override = SEGMENT_OBJECT_OVERRIDES.get((sequence, frame_def))
    object_ids = list(override["object_ids"]) if override else original_object_ids
    roots = sorted({root_of.get(x, x) for x in object_ids})
    mode = str(item.get("interaction_mode") or "")
    side_roots = {
        side: sorted({root_of.get(str(x), str(x)) for x in (item.get(f"obj_list_{side[:1]}h") or [])})
        for side in ("left", "right")
    }
    if override:
        for side in override["hand_sides"]:
            side_roots[side] = list(roots)
    if mode.startswith("rh"):
        main_ids = [str(x) for x in (item.get("obj_list_rh") or [])]
    elif mode.startswith("lh"):
        main_ids = [str(x) for x in (item.get("obj_list_lh") or [])]
    else:
        main_ids = list(object_ids)
    main_roots = sorted({root_of.get(x, x) for x in main_ids})
    tool_roots = sorted({
        root for root in roots
        if any(names.get(member) in TOOL_NAMES for member in members.get(root, [root]))
    })

    if override:
        selected_roots = roots
        selection_reason = "semantic_object_override"
        resolution_reason = str(override["reason"])
    elif len(roots) == 1:
        selected_roots = roots
        selection_reason = "single_instance_after_part_tree"
        resolution_reason = None
    elif len(tool_roots) == 1:
        selected_roots = tool_roots
        selection_reason = "unique_active_tool"
        resolution_reason = None
    elif len(main_roots) == 1:
        selected_roots = main_roots
        selection_reason = "unique_main_hand_object"
        resolution_reason = None
    elif mode == "bh_main" and len(main_roots) > 1 and set(main_roots) == set(roots):
        selected_roots = main_roots
        selection_reason = "bilateral_multi_active_split"
        resolution_reason = "one_single_object_candidate_per_active_root"
    else:
        selected_roots = []
        selection_reason = None
        resolution_reason = None
    return {
        "original_object_ids": original_object_ids,
        "object_ids": object_ids,
        "roots": roots,
        "main_roots": main_roots,
        "tool_roots": tool_roots,
        "side_roots": side_roots,
        "selected_roots": selected_roots,
        "selection_reason": selection_reason,
        "resolution_reason": resolution_reason,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_snapshot(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "sha256": _sha256(path)}


def _git_info() -> tuple[str, bool]:
    repo_root = Path(__file__).resolve().parents[5]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return commit, dirty


def build(annotation_root: Path, object_root: Path, stage3_root: Path,
          output: Path, translation_m: float = 0.001,
          rotation_deg: float = 1.0, run_id: str | None = None,
          work_version: str = "V1.4.3") -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    part_tree_path = object_root / "object_affordance/object_part_tree.json"
    descriptions_path = object_root / "object_raw/obj_desc.json"
    tree = json.loads(part_tree_path.read_text())
    root_of, members = _roots(tree)
    descriptions = json.loads(descriptions_path.read_text())
    names = {str(k): str(v.get("obj_name", k)) for k, v in descriptions.items()}
    stage3 = _load_stage3_map(stage3_root)
    rows: list[dict] = []
    uncertain: list[dict] = []
    stats = Counter()
    for annotation_path in sorted(annotation_root.glob("*.pkl")):
        # OakInk2 annotations use pickle for tensor-bearing values.  No output
        # arrays are written here; loading one sequence at a time bounds RAM.
        import pickle
        with annotation_path.open("rb") as stream:
            anno = pickle.load(stream)
        sequence = annotation_path.stem
        raw = anno.get("raw_mano", {})
        frame_ids = sorted(int(x) for x in raw)
        distance_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        if not frame_ids:
            stats["empty_sequence"] += 1
            continue
        for frame_def, item in json.loads(
                (object_root / "program/program_info" / (sequence + ".json")).read_text()).items():
            ranges = ast.literal_eval(frame_def)
            bounds = [x for x in ranges if x is not None]
            if not bounds:
                stats["empty_segment"] += 1
                continue
            start, end = min(x[0] for x in bounds), max(x[1] for x in bounds)
            segment_frames = [fid for fid in frame_ids if start <= fid <= end]
            resolution = _resolve_active_roots(
                sequence=sequence, frame_def=frame_def, item=item,
                root_of=root_of, members=members, names=names,
            )
            object_ids = resolution["object_ids"]
            roots = resolution["roots"]
            if not roots:
                uncertain.append({"sequence": sequence, "frame_range_def": frame_def,
                                  "primitive": item.get("primitive"), "reason": "no_objects"})
                stats["uncertain_no_objects"] += 1
                continue
            mode = str(item.get("interaction_mode") or "")
            selected_roots = resolution["selected_roots"]
            if not selected_roots:
                uncertain.append({"sequence": sequence, "frame_range_def": frame_def,
                                  "primitive": item.get("primitive"), "object_ids": object_ids,
                                  "root_ids": roots, "root_names": [names.get(r, r) for r in roots],
                                  "main_root_ids": resolution["main_roots"],
                                  "main_root_names": [names.get(r, r) for r in resolution["main_roots"]],
                                  "tool_root_ids": resolution["tool_roots"],
                                  "reason": "active_tool_not_unique"})
                stats["uncertain_active_tool"] += 1
                continue
            if resolution["selection_reason"] == "bilateral_multi_active_split":
                stats["resolved_bilateral_multi_active_segments"] += 1
                stats["resolved_bilateral_multi_active_candidates"] += len(selected_roots)
            elif resolution["selection_reason"] == "semantic_object_override":
                stats["resolved_semantic_override_segments"] += 1
                stats["resolved_semantic_override_candidates"] += len(selected_roots)
            for selected_root in selected_roots:
                selected_ids = sorted(x for x in object_ids if root_of.get(x, x) == selected_root)
                pose_by_id = {
                    object_id: {int(fid): np.asarray(anno["obj_transf"][object_id][int(fid)], dtype=np.float32)
                                for fid in frame_ids if object_id in anno.get("obj_transf", {}) and int(fid) in anno["obj_transf"][object_id]}
                    for object_id in selected_ids
                }
                spans = [_motion_span(pose_by_id[obj], segment_frames, translation_m, rotation_deg)
                         for obj in selected_ids]
                spans = [span for span in spans if span is not None]
                if not spans:
                    stats["static_selected_tool"] += 1
                    continue
                motion_start, motion_end = min(x[0] for x in spans), max(x[1] for x in spans)
                motion_frames = [fid for fid in segment_frames if motion_start <= fid <= motion_end]
                distances = _frame_distances(
                    stage3, sequence, selected_ids, motion_frames, ("left", "right"),
                    cache=distance_cache,
                )
                kept = [fid for fid in motion_frames if distances.get(fid, float("inf")) < 0.02]
                if not kept:
                    stats["no_2cm_frame"] += 1
                    continue
                rows.append({
                    "sequence": sequence, "frame_range_def": frame_def,
                    "primitive": item.get("primitive"), "interaction_mode": mode,
                    "selected_root_id": selected_root,
                    "selected_object_ids": selected_ids,
                    "selected_object_name": names.get(selected_root, selected_root),
                    "selection_reason": resolution["selection_reason"],
                    "resolution_reason": resolution["resolution_reason"],
                    "original_object_ids": resolution["original_object_ids"],
                    "selected_hand_sides": [
                        side for side, side_values in resolution["side_roots"].items()
                        if selected_root in side_values
                    ],
                    "motion_frame_range": [motion_start, motion_end],
                    "selected_frame_ids": kept,
                    "selected_frame_count": len(kept),
                    "min_hand_object_distance_m": min(distances[fid] for fid in kept),
                    "source_stage3_keys": [f"{sequence}/{obj}/left" for obj in selected_ids]
                        + [f"{sequence}/{obj}/right" for obj in selected_ids],
                })
                stats["selected_segments"] += 1
    payload = {
        "schema_name": "ref2dex_oakink2_active_tool_frame_selection_v1",
        "schema_version": "1.1.0",
        "source": "OakInk2 program_info + object_part_tree + Stage3 hand_to_obj_min_dist",
        "annotation_root": str(annotation_root.resolve()),
        "object_root": str(object_root.resolve()),
        "stage3_root": str(stage3_root.resolve()),
        "selection_policy": {
            "active_object": "unique active tool; single instance fallback; bilateral main roots split into one candidate per object",
            "semantic_overrides": "six sequence/range keyed corrections for missing or over-inclusive object lists",
            "motion_first": True,
            "translation_motion_threshold_m": translation_m,
            "rotation_motion_threshold_deg": rotation_deg,
            "hand_object_distance_strict_lt_m": 0.02,
            "hand_sides": "left+right union, no side ID",
            "object_parts": "object_part_tree members are one physical instance",
        },
        "segment_count": len(rows), "uncertain_count": len(uncertain),
        "stats": dict(stats), "segments": rows, "uncertain": uncertain,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(json.dumps({k: payload[k] for k in (
        "schema_name", "schema_version", "source", "selection_policy", "segment_count",
        "uncertain_count", "stats")}, indent=2, ensure_ascii=False) + "\n")
    base_commit, worktree_dirty = _git_info()
    stats_paths = sorted(stage3_root.glob("oakink2_stage3_stats_*.json"))
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "oakink2_active_tool_frame_selection",
        "run_id": run_id or output.name,
        "run_status": "COMPLETED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "work_version": work_version,
        "base_commit": base_commit,
        "worktree_dirty": worktree_dirty,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "annotation_root": str(annotation_root.resolve()),
            "annotation_file_count": len(list(annotation_root.glob("*.pkl"))),
            "object_root": str(object_root.resolve()),
            "object_part_tree": _file_snapshot(part_tree_path),
            "object_descriptions": _file_snapshot(descriptions_path),
            "stage3_root": str(stage3_root.resolve()),
            "stage3_stats": [_file_snapshot(path) for path in stats_paths],
        },
        "parameters": payload["selection_policy"],
        "seed": None,
        "checkpoint": None,
        "metadata_snapshot": str((output / "manifest.json").resolve()),
        "outputs": {
            "index": str((output / "index.json").resolve()),
            "manifest": str((output / "manifest.json").resolve()),
        },
        "counts": {
            "selected_candidates": len(rows),
            "uncertain": len(uncertain),
            **dict(stats),
        },
        "conclusion": "SUPPORTED" if not uncertain else "INCONCLUSIVE",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return {"segment_count": len(rows), "uncertain_count": len(uncertain), "stats": dict(stats)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation-motion-threshold-m", type=float, default=0.001)
    parser.add_argument("--rotation-motion-threshold-deg", type=float, default=1.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--work-version", default="V1.4.3")
    args = parser.parse_args()
    print(json.dumps(build(args.annotation_root.resolve(), args.object_root.resolve(),
                           args.stage3_root.resolve(), args.output.resolve(),
                           args.translation_motion_threshold_m,
                           args.rotation_motion_threshold_deg,
                           args.run_id, args.work_version), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
