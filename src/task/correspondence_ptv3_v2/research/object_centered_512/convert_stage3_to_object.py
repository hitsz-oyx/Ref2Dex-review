"""Convert hand-root Stage 3 NPZ files to the object coordinate frame.

This is an offline, lossless geometry conversion: world-space poses and MANO /
robot descriptors are preserved, while ``obj_points``, ``obj_normals``,
``hand_points`` and ``hand_normals`` are transformed per frame by
``inv(obj_root_pose_world)``.  The source tree is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[..., :3, :3].transpose(0, 2, 1) + pose[..., None, :3, 3]


def _to_object(points: np.ndarray, hand_pose: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
    world = _world(points, hand_pose)
    inv_obj = np.linalg.inv(obj_pose)
    return _world(world, inv_obj)


def _normals_to_object(normals: np.ndarray, hand_pose: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
    rotation = np.einsum("tij,tjk->tik", obj_pose[..., :3, :3].transpose(0, 2, 1), hand_pose[..., :3, :3])
    out = np.einsum("tni,tji->tnj", normals, rotation)
    out /= np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)
    return out.astype(np.float32)


def convert_file(source: Path, destination: Path) -> None:
    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    frame = str(np.asarray(payload.get("coordinate_frame", "hand_root")).item())
    if frame == "object":
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **payload)
        return
    if frame != "hand_root":
        raise ValueError(f"{source}: unsupported coordinate_frame={frame!r}")
    for key in ("obj_points", "obj_normals", "hand_points", "hand_normals", "hand_root_pose", "obj_root_pose_world"):
        if key not in payload:
            raise KeyError(f"{source}: missing {key}")
    hand_pose = np.asarray(payload["hand_root_pose"], dtype=np.float64)
    obj_pose = np.asarray(payload["obj_root_pose_world"], dtype=np.float64)
    payload["obj_points"] = _to_object(np.asarray(payload["obj_points"], dtype=np.float64), hand_pose, obj_pose).astype(np.float32)
    payload["hand_points"] = _to_object(np.asarray(payload["hand_points"], dtype=np.float64), hand_pose, obj_pose).astype(np.float32)
    payload["obj_normals"] = _normals_to_object(np.asarray(payload["obj_normals"], dtype=np.float64), hand_pose, obj_pose)
    payload["hand_normals"] = _normals_to_object(np.asarray(payload["hand_normals"], dtype=np.float64), hand_pose, obj_pose)
    payload["coordinate_frame"] = np.asarray("object")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = sorted(args.source_root.glob("**/*.npz"))
    if not files:
        raise ValueError(f"No NPZ files under {args.source_root}")
    converted = 0
    for source in files:
        destination = args.destination_root / source.relative_to(args.source_root)
        if destination.exists() and not args.overwrite:
            continue
        convert_file(source, destination)
        converted += 1
    meta = {"schema": "ref2dex_object_centered_stage3_v1", "coordinate_frame": "object", "source_root": str(args.source_root), "num_files": len(files)}
    args.destination_root.mkdir(parents=True, exist_ok=True)
    (args.destination_root / "meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(f"source_files={len(files)} converted={converted} destination={args.destination_root}")


if __name__ == "__main__":
    main()
