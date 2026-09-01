"""Create a lightweight object-pose cache view over an existing world geometry cache.

The geometry is intentionally not copied.  A new manifest and a symlinked
episode tree provide an auditable coordinate contract while the Decoder builds
``(t, t+stride)`` pairs from mmap geometry at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "cmdecoder_object_pose_geometry_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    source_manifest_path = args.source_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("schema") != "cmdecoder_layered_v4":
        raise ValueError(f"Unexpected source schema: {source.get('schema')!r}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output cache: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_episodes = source_root / "v4" / "episodes"
    if not source_episodes.is_dir():
        raise FileNotFoundError(source_episodes)
    episodes_link = output_root / "v4" / "episodes"
    episodes_link.parent.mkdir(parents=True, exist_ok=True)
    episodes_link.symlink_to(source_episodes, target_is_directory=True)

    for split, episodes in source["splits"].items():
        for episode in episodes:
            relative = source["cache_dirs"][episode]
            geometry = output_root / relative / "geometry"
            if not (geometry / "obj_pose_world.npy").is_file():
                raise FileNotFoundError(f"Missing object pose for {episode}: {geometry}")
            geometry_manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8"))
            if geometry_manifest.get("schema") != "cmdecoder_layered_v4":
                raise ValueError(f"Unexpected geometry schema for {episode}: {geometry_manifest.get('schema')!r}")

    manifest = {
        "schema": SCHEMA,
        "source_schema": source["schema"],
        "coordinate_frame": "object_pose_t",
        "hand_flow_frame": "object_pose_t",
        "pair_policy": "runtime_even_stride_values",
        "stride_values": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "source_root": str(source_root),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "episode_count": source["episode_count"],
        "splits": source["splits"],
        "split_objects": source.get("split_objects"),
        "cache_dirs": source["cache_dirs"],
        "geometry_contract": {
            "num_hand_points": 1538,
            "num_obj_points": 512,
            "num_obj_pool": 4096,
            "candidate_semantics": "all_object_surface_points",
            "surface_sampling": "deterministic_face_barycentric_with_runtime_pool_selection",
        },
    }
    output_manifest = output_root / "v4" / "selection_all_object_disjoint_seed42.json"
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {output_manifest}")
    print(f"episodes: {manifest['episode_count']} ({len(manifest['splits']['train'])}/{len(manifest['splits']['val'])}/{len(manifest['splits']['test'])})")


if __name__ == "__main__":
    main()
