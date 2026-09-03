"""Convert the existing object-only Cm Stage4 NPZ cache to mmap V2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA_NAME = "ref2dex_cm_object_v2"
SCHEMA_VERSION = "1.0.0"


def _ragged(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = [np.flatnonzero(row).astype(np.uint32) for row in np.asarray(mask, bool)]
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(row) for row in rows], dtype=np.int64)
    indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint32)
    return offsets, indices


def convert_sequence(source: Path, output: Path, *, overwrite: bool = False) -> None:
    shared_path = source / "shared.npz"
    if not shared_path.is_file():
        raise FileNotFoundError(shared_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}")
    with np.load(shared_path, allow_pickle=False) as shared:
        required = {"raw_frame_id", "obj_points_world", "obj_normals_world", "obj_point_id", "ds_rate", "source_fps"}
        missing = required.difference(shared.files)
        if missing:
            raise KeyError(f"{shared_path}: missing {sorted(missing)}")
        frames, pool = shared["obj_points_world"].shape[:2]
        if pool != 4096 or shared["obj_points_world"].shape != (frames, pool, 3):
            raise ValueError(f"{shared_path}: expected [T,4096,3]")
        output.mkdir(parents=True, exist_ok=True)
        shared_out = output / "shared"
        shared_out.mkdir(exist_ok=True)
        for key in ("raw_frame_id", "obj_points_world", "obj_normals_world", "obj_point_id"):
            np.save(shared_out / f"{key}.npy", np.asarray(shared[key]))
        if "obj_pose_world" in shared.files:
            np.save(shared_out / "obj_pose_world.npy", np.asarray(shared["obj_pose_world"]))
        meta = {key: str(np.asarray(shared[key]).item()) for key in ("dataset_name", "seq_id", "subject_id", "seq_name", "object_name") if key in shared.files}
        meta.update({"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
                     "ds_rate": int(np.asarray(shared["ds_rate"]).item()),
                     "source_fps": float(np.asarray(shared["source_fps"]).item()),
                     "coordinate_frame": "world", "num_obj_pool": 4096,
                     "num_hand_points": 1538})
        (shared_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for side in ("left", "right"):
        side_path = source / f"{side}.npz"
        if not side_path.is_file():
            continue
        with np.load(side_path, allow_pickle=False) as hand:
            out = output / side
            out.mkdir(exist_ok=True)
            for key in ("hand_points_world", "hand_normals_world", "hand_root_pose_world"):
                if key not in hand.files:
                    raise KeyError(f"{side_path}: missing {key}")
                np.save(out / f"{key}.npy", np.asarray(hand[key]))
            if "hand_mesh_vertices_world" in hand.files and "hand_mesh_faces" in hand.files:
                np.save(out / "hand_mesh_vertices_world.npy", np.asarray(hand["hand_mesh_vertices_world"]))
                np.save(out / "hand_mesh_faces.npy", np.asarray(hand["hand_mesh_faces"]))
            offsets, indices = _ragged(hand["obj_candidate_mask_5cm"])
            np.save(out / "candidate_offsets.npy", offsets)
            np.save(out / "candidate_indices.npy", indices)
            raw = np.asarray(hand["raw_frame_id"], dtype=np.int32) if "raw_frame_id" in hand.files else np.arange(len(offsets) - 1, dtype=np.int32)
            np.save(out / "raw_frame_id.npy", raw)


def convert_root(source_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, int]:
    stats = {"sequences": 0, "converted": 0, "skipped": 0, "failed": 0}
    for shared in sorted(source_root.glob("**/shared.npz")):
        source = shared.parent
        relative = source.relative_to(source_root)
        target = output_root / relative
        stats["sequences"] += 1
        try:
            convert_sequence(source, target, overwrite=overwrite)
            stats["converted"] += 1
        except FileExistsError:
            stats["skipped"] += 1
        except Exception:
            stats["failed"] += 1
            raise
    (output_root / "meta.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps({"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "stats": stats}, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage4 NPZ -> Cm object-only mmap V2")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(convert_root(Path(args.input_root), Path(args.output_root), overwrite=args.overwrite))


if __name__ == "__main__":
    main()
