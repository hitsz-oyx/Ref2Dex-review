"""Build the auditable, stride-expanded V1.2 pair index without copying geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


STRIDES = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source_manifest: Path, output_root: Path, checkpoint: Path) -> dict:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("schema") != "cmdecoder_object_pose_geometry_v1":
        raise ValueError(f"Unsupported source manifest schema: {source.get('schema')!r}")
    if tuple(source.get("stride_values", ())) != STRIDES:
        raise ValueError(f"Source stride contract mismatch: {source.get('stride_values')!r}")
    output_root.mkdir(parents=True, exist_ok=True)
    split_summary: dict[str, dict] = {}
    for split, episodes in source["splits"].items():
        episode_names: list[str] = []
        episode_counts: list[int] = []
        episode_index: list[int] = []
        current_frame: list[int] = []
        stride: list[int] = []
        target_frame: list[int] = []
        source_frame_id: list[int] = []
        target_source_frame_id: list[int] = []
        by_stride = {str(value): 0 for value in STRIDES}
        for epi_idx, episode in enumerate(episodes):
            # cache_dirs are rooted at the cache root (the parent of the v4
            # manifest directory), e.g. ``v4/episodes/<id>``.
            geometry = source_manifest.parent.parent / source["cache_dirs"][episode] / "geometry"
            geometry = geometry.resolve()
            source_ids = np.load(geometry / "source_frame_id.npy", mmap_mode="r", allow_pickle=False)
            frame_time = np.load(geometry / "frame_time.npy", mmap_mode="r", allow_pickle=False)
            hand_points = np.load(geometry / "hand_points_world.npy", mmap_mode="r", allow_pickle=False)
            hand_normals = np.load(geometry / "hand_normals_world.npy", mmap_mode="r", allow_pickle=False)
            if hand_points.ndim != 3 or hand_points.shape[1:] != (1538, 3):
                raise ValueError(f"{episode}: expected hand points [T,1538,3], got {hand_points.shape}")
            if hand_normals.shape != hand_points.shape:
                raise ValueError(f"{episode}: hand normals shape mismatch: {hand_normals.shape}")
            if frame_time.shape[0] != source_ids.shape[0]:
                raise ValueError(f"{episode}: frame_time/source_frame_id length mismatch")
            episode_names.append(episode)
            count = 0
            for frame in range(len(source_ids)):
                for value in STRIDES:
                    target = frame + value
                    if target >= len(source_ids):
                        continue
                    if int(source_ids[target]) - int(source_ids[frame]) != value:
                        continue
                    dt = float(frame_time[target] - frame_time[frame])
                    if not dt > 0.0:
                        raise ValueError(f"{episode}: non-positive delta_time at {frame}->{target}")
                    episode_index.append(epi_idx)
                    current_frame.append(frame)
                    stride.append(value)
                    target_frame.append(target)
                    source_frame_id.append(int(source_ids[frame]))
                    target_source_frame_id.append(int(source_ids[target]))
                    by_stride[str(value)] += 1
                    count += 1
            episode_counts.append(count)
        arrays_path = output_root / f"pairs_{split}.npz"
        np.savez_compressed(
            arrays_path,
            episode_index=np.asarray(episode_index, dtype=np.int32),
            frame=np.asarray(current_frame, dtype=np.int32),
            stride=np.asarray(stride, dtype=np.int16),
            target_frame=np.asarray(target_frame, dtype=np.int32),
            source_frame_id=np.asarray(source_frame_id, dtype=np.int64),
            target_source_frame_id=np.asarray(target_source_frame_id, dtype=np.int64),
        )
        split_summary[split] = {
            "episode_count": len(episodes),
            "pair_count": int(len(stride)),
            "by_stride": by_stride,
            "episodes": [
                {"name": name, "pair_count": int(count)}
                for name, count in zip(episode_names, episode_counts)
            ],
            "arrays": arrays_path.name,
            "arrays_sha256": sha256(arrays_path),
        }
    result = {
        "schema": "cmdecoder_pair_index_v1_2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "object_pose_t",
        "hand_flow_frame": "object_pose_t",
        "stride_values": list(STRIDES),
        "pair_policy": "all_even_strides_expanded",
        "object_pool_points": 4096,
        "model_object_points": 1024,
        "real_hand_points": 1538,
        "padded_hand_points": 3076,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256(source_manifest),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "splits": split_summary,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_manifest = {
        "schema": "cmdecoder_data_run_manifest_v1",
        "run_id": output_root.name,
        "run_status": "COMPLETED",
        "run_type": "pair_index_build",
        "modification_version": "V1.2.0",
        "operation_category": "data",
        "output_root": str(output_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "created_at": result["created_at"],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.source_manifest, args.output_root, args.checkpoint)
    print(json.dumps({split: value["pair_count"] for split, value in result["splits"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
