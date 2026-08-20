"""Validate OakInk wrist centering against its calibrated camera extrinsics."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.task.correspondence_ptv3_v2.research.oakink_conversion.convert_oakink_pilot import (
    _mesh,
    _transform,
)


def _load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a @ b.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _rms_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))) * 1000.0)


def _collect_multiview_groups(hand_dir: Path, limit: int) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with os.scandir(hand_dir) as entries:
        for entry in entries:
            if not entry.name.endswith(".pkl"):
                continue
            parts = Path(entry.name).stem.split("__")
            if len(parts) != 5:
                continue
            groups["__".join(parts[:4])].append(entry.name)
            ready = [sorted(names) for names in groups.values() if len(names) >= 4]
            if len(ready) >= limit:
                return ready[:limit]
    return [sorted(names) for names in groups.values() if len(names) >= 2][:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oakink-root", required=True)
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.oakink_root).resolve()
    anno = root / "downloads" / "image" / "anno"
    obj_root = root / "downloads" / "image" / "obj"
    requested_groups = int(args.groups)
    groups = _collect_multiview_groups(anno / "hand_v", requested_groups * 8)
    if not groups:
        raise RuntimeError("No multi-view OakInk frames were found.")

    rows: list[dict[str, float | str | int]] = []
    mesh_cache: dict[str, np.ndarray] = {}
    successful_groups = 0
    for names in groups:
        obj_id = names[0].split("__", 1)[0]
        mesh_id = obj_id.split("_", 1)[0]
        mesh_path = obj_root / f"{mesh_id}.obj"
        if not mesh_path.is_file():
            continue
        if successful_groups >= requested_groups:
            break
        successful_groups += 1
        reference: dict[str, np.ndarray] | None = None
        for view_idx, name in enumerate(names):
            hand_cam = np.asarray(_load(anno / "hand_v" / name), dtype=np.float32)
            joints_cam = np.asarray(_load(anno / "hand_j" / name), dtype=np.float32)
            obj_cam_from_canonical = np.asarray(
                _load(anno / "obj_transf" / name), dtype=np.float32
            )
            general = _load(anno / "general_info" / name)
            cam_from_world = np.asarray(general["cam_extr"], dtype=np.float32)
            world_from_cam = np.linalg.inv(cam_from_world).astype(np.float32)

            wrist_cam = joints_cam[0]
            hand_centered_cam = hand_cam - wrist_cam
            hand_world = _transform(hand_cam, world_from_cam)
            wrist_world = _transform(joints_cam[[0]], world_from_cam)[0]
            hand_centered_world = hand_world - wrist_world

            if mesh_id not in mesh_cache:
                mesh_cache[mesh_id] = _mesh(mesh_path)[0]
            canonical = mesh_cache[mesh_id]
            object_cam = _transform(canonical, obj_cam_from_canonical)
            object_world = _transform(object_cam, world_from_cam)
            distance_cam = cKDTree(object_cam).query(hand_cam, workers=1)[0]
            distance_world = cKDTree(object_world).query(hand_world, workers=1)[0]

            world_from_object = world_from_cam @ obj_cam_from_canonical
            state = {
                "hand_centered_cam": hand_centered_cam,
                "hand_centered_world": hand_centered_world,
                "distance_cam": distance_cam,
                "distance_world": distance_world,
                "object_centered_cam": obj_cam_from_canonical[:3, 3] - wrist_cam,
                "object_centered_world": world_from_object[:3, 3] - wrist_world,
                "object_rotation_cam": obj_cam_from_canonical[:3, :3],
                "object_rotation_world": world_from_object[:3, :3],
            }
            if reference is None:
                reference = state
                continue
            assert reference is not None
            rows.append(
                {
                    "group": "__".join(Path(name).stem.split("__")[:4]),
                    "view": view_idx,
                    "hand_centered_camera_rms_mm": _rms_mm(
                        state["hand_centered_cam"], reference["hand_centered_cam"]
                    ),
                    "hand_centered_world_rms_mm": _rms_mm(
                        state["hand_centered_world"], reference["hand_centered_world"]
                    ),
                    "object_centered_camera_translation_mm": float(
                        np.linalg.norm(
                            state["object_centered_cam"] - reference["object_centered_cam"]
                        )
                        * 1000.0
                    ),
                    "object_centered_world_translation_mm": float(
                        np.linalg.norm(
                            state["object_centered_world"] - reference["object_centered_world"]
                        )
                        * 1000.0
                    ),
                    "object_camera_rotation_deg": _rotation_error_deg(
                        state["object_rotation_cam"], reference["object_rotation_cam"]
                    ),
                    "object_world_rotation_deg": _rotation_error_deg(
                        state["object_rotation_world"], reference["object_rotation_world"]
                    ),
                    "contact_distance_cross_view_mae_mm": float(
                        np.mean(np.abs(state["distance_cam"] - reference["distance_cam"]))
                        * 1000.0
                    ),
                    "contact_distance_camera_world_mae_mm": float(
                        np.mean(np.abs(state["distance_cam"] - state["distance_world"]))
                        * 1000.0
                    ),
                }
            )

    metric_names = [key for key in rows[0] if key not in {"group", "view"}]
    summary = {
        "num_groups": successful_groups,
        "num_view_pairs": len(rows),
        "metrics": {
            key: {
                "mean": float(np.mean([float(row[key]) for row in rows])),
                "p95": float(np.quantile([float(row[key]) for row in rows], 0.95)),
                "max": float(np.max([float(row[key]) for row in rows])),
            }
            for key in metric_names
        },
        "interpretation": (
            "Wrist subtraction preserves hand-object distances but leaves camera rotation. "
            "Inverse camera extrinsics align the same frame across views."
        ),
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
