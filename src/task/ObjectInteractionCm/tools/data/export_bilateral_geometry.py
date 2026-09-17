#!/usr/bin/env python3
"""Export a side-free bilateral geometric hand sequence.

This adapter is intentionally agnostic to the retargeter.  A retargeter (for
example Dexplore's MANO->Inspire path) writes one ``left`` and one ``right``
stream; this command copies the shared object arrays and concatenates the two
retargeted streams into ``geometry/hand_*_world.npy``.  Offline KNN files are
not copied: they must be recomputed on the merged point cloud.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

try:  # package import (tests and ``python -m``)
    from .merge_dual_hand_stream import merge_bilateral_hand_arrays
except ImportError:  # direct script invocation from this directory
    from merge_dual_hand_stream import merge_bilateral_hand_arrays


_SHARED_FILES = (
    "obj_points_pool_world.npy", "obj_normals_pool_world.npy",
    "obj_pose_world.npy", "source_frame_id.npy", "frame_time.npy",
)


def _find(root: Path, side: str, stem: str) -> Path:
    candidates = (
        root / side / f"{stem}.npy",
        root / f"{side}_{stem}.npy",
        root / "geometry" / side / f"{stem}.npy",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {side} {stem} under {root}")


def _optional_find(root: Path, side: str, stem: str) -> Path | None:
    try:
        return _find(root, side, stem)
    except FileNotFoundError:
        return None


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array))


def export_bilateral_geometry(input_root: str | Path, output_root: str | Path) -> dict:
    """Merge left/right retargeted arrays and write a cache-ready sequence.

    Required side files are ``hand_points_world.npy`` and
    ``hand_normals_world.npy``.  Optional ``candidate_active_5cm.npy`` masks
    are unioned; if neither side provides one, no candidate mask is emitted.
    Existing output is rejected to avoid silently replacing an experiment.
    """

    source = Path(input_root).resolve()
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing output: {target}")
    target.mkdir(parents=True)
    geometry = target / "geometry"
    geometry.mkdir()

    for name in _SHARED_FILES:
        candidates = (source / "shared" / name, source / "geometry" / name, source / name)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing shared geometry file {name} under {source}")
        shutil.copyfile(path, geometry / name)

    left_points = np.load(_find(source, "left", "hand_points_world"), mmap_mode="r")
    right_points = np.load(_find(source, "right", "hand_points_world"), mmap_mode="r")
    left_normals = np.load(_find(source, "left", "hand_normals_world"), mmap_mode="r")
    right_normals = np.load(_find(source, "right", "hand_normals_world"), mmap_mode="r")
    points = merge_bilateral_hand_arrays(left_points, right_points, name="hand_points")
    normals = merge_bilateral_hand_arrays(left_normals, right_normals, name="hand_normals")
    if normals.shape != points.shape:
        raise ValueError(f"Point/normal shape mismatch: {points.shape} vs {normals.shape}")
    _save(geometry / "hand_points_world.npy", points)
    _save(geometry / "hand_normals_world.npy", normals)

    masks = []
    for side in ("left", "right"):
        path = _optional_find(source, side, "candidate_active_5cm")
        if path is not None:
            value = np.asarray(np.load(path, mmap_mode="r"), dtype=bool)
            if value.shape != (points.shape[0],):
                raise ValueError(f"{path}: expected [{points.shape[0]}], got {value.shape}")
            masks.append(value)
    if masks:
        _save(geometry / "obj_candidate_mask_5cm.npy", np.logical_or.reduce(masks))

    meta = {
        "schema_name": "ref2dex_object_interaction_cm_bilateral_geometry_v1",
        "schema_version": "1.0.0",
        "coordinate_frame": "object_pose_t",
        "hand_side": "bilateral_merged_left_then_right",
        "merged_hand_sides": True,
        "hand_points": int(points.shape[1]),
        "left_hand_points": int(left_points.shape[1]),
        "right_hand_points": int(right_points.shape[1]),
        "offline_knn": "recompute_after_merge",
        "source_root": str(source),
    }
    (target / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_bilateral_geometry(args.input_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
