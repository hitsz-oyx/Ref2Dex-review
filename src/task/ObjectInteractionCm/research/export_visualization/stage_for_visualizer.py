"""Stage exported ARCTIC/OakInk2 NPZ files for the existing Viser viewer.

The staged files are a read-only visualization projection.  They are not an
OI-Cm training cache: object points are tiled to the viewer's 4096-point
display pool and the object pose is identity because the exported points are
already in world coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


SCHEMA = "ref2dex_object_interaction_cm_dexplore_rl_v1"
INDEX_SCHEMA = "ref2dex_object_interaction_cm_index_v1_1"


def _pool(value: np.ndarray, count: int = 4096) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected [T,N,3], got {value.shape}")
    if value.shape[1] == count:
        return value
    ids = np.arange(count, dtype=np.int64) % value.shape[1]
    return value[:, ids]


def _normals(points: np.ndarray) -> np.ndarray:
    return np.zeros_like(points, dtype=np.float32)


def _write_sequence(root: Path, sequence_id: str, source: str, variant: str,
                    object_name: str, object_points: np.ndarray,
                    hand_points: np.ndarray, source_frame: np.ndarray) -> None:
    sequence = root / "sequences" / "train" / source / sequence_id.replace("/", "_")
    geometry = sequence / "geometry"
    geometry.mkdir(parents=True, exist_ok=True)
    objects = _pool(object_points)
    hands = _pool(hand_points, 1538)
    frames = objects.shape[0]
    if hands.shape[0] != frames or len(source_frame) != frames:
        raise ValueError(f"frame mismatch for {sequence_id}: object={frames}, hand={hands.shape[0]}, frame={len(source_frame)}")
    pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    distances = np.linalg.norm(objects[:, :, None] - hands[:, None, :], axis=-1).min(axis=(1, 2))
    active = distances <= 0.05
    arrays = {
        "obj_points_pool_world.npy": objects,
        "obj_normals_pool_world.npy": _normals(objects),
        "obj_pose_world.npy": pose,
        "source_frame_id.npy": np.asarray(source_frame, dtype=np.int32),
        "hand_points_world.npy": hands,
        "hand_normals_world.npy": _normals(hands),
        "obj_candidate_mask_5cm.npy": active,
    }
    for name, value in arrays.items():
        np.save(geometry / name, value)
    manifest = {
        "schema_name": SCHEMA,
        "sequence_id": sequence_id,
        "split": "train",
        "source": source,
        "variant": variant,
        "world_frame": "dexplore_native_object_pose_world",
        "coordinate_frame": "object_pose_t",
        "hand_side": "right",
        "hand_points": 1538,
        "effective_fps": 30.0,
        "visualization_projection": True,
        "source_note": "world-space exported points; identity pose is used only by the viewer projection",
    }
    (geometry / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def stage_oakink2(root: Path, oak_dir: Path) -> list[dict]:
    entries = []
    for path in sorted(oak_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            sequence_id = f"{path.stem}"
            _write_sequence(root, sequence_id, "oakink2", "inspire_rl",
                            sequence_id, data["object_sample_vertices_world"].reshape(len(data["source_frame_id"]), -1, 3),
                            data["right_mano_vertices_world"], data["source_frame_id"])
        entries.append({"id": sequence_id, "path": f"sequences/train/oakink2/{sequence_id}",
                        "source": "oakink2", "variant": "inspire_rl", "object_name": sequence_id, "split": "train"})
    return entries


def stage_arctic(root: Path, raw_sequence: Path, max_frames: int) -> list[dict]:
    from process.ARCTIC.raw import ArcticRawAdapter

    adapter = ArcticRawAdapter(num_obj_points=4096, device="cpu", max_frames=max_frames,
                               nn_batch_size=4, mano_batch_size=16)
    fields = adapter.process_sequence(str(raw_sequence))
    sequence_id = str(fields["seq_id"])
    _write_sequence(root, sequence_id, "arctic", "inspire_rl", str(fields["object_name"]),
                    fields["obj_points_world"], fields["right_hand_points_world"], fields["raw_frame_id"])
    return [{"id": sequence_id, "path": f"sequences/train/arctic/{sequence_id.replace('/', '_')}",
             "source": "arctic", "variant": "inspire_rl", "object_name": str(fields["object_name"]), "split": "train"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oak-dir", type=Path, required=True)
    parser.add_argument("--arctic-sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-arctic-frames", type=int, default=48)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = stage_oakink2(args.output, args.oak_dir)
    entries += stage_arctic(args.output, args.arctic_sequence, args.max_arctic_frames)
    index = {"schema_name": INDEX_SCHEMA, "knn_k": 32, "model_object_points": 1024,
             "sequences": {"train": entries, "val": [], "test": []},
             "visualization_projection": True}
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (args.output / "stage_manifest.json").write_text(json.dumps({"entries": entries, "source_oakink2": str(args.oak_dir.resolve()),
        "source_arctic": str(args.arctic_sequence.resolve()), "max_arctic_frames": args.max_arctic_frames}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"index": str(args.output / "index.json"), "entries": len(entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
