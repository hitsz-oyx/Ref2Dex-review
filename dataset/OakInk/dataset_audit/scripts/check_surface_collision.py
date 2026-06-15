#!/usr/bin/env python
from __future__ import annotations

import argparse

from _script_utils import add_common_args, bootstrap, load_adapter


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.mesh_io import load_cache, save_cache, write_csv, write_json
    from dataset_audit.common.surface_collision import audit_surface_collision_cache

    parser = argparse.ArgumentParser(description="Check lightweight triangle-surface collision proxy metrics.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-json", default="outputs/dataset_audit/surface_collision_summary.json")
    parser.add_argument("--out-csv", default="outputs/dataset_audit/surface_collision_frames.csv")
    parser.add_argument("--eps-mm", type=float, default=1.0)
    add_common_args(parser)
    args = parser.parse_args()
    cache_path = repo_root / args.cache
    cache = load_cache(cache_path)
    if "faces_object" not in cache:
        adapter = load_adapter(repo_root)
        cache = adapter.enrich_cache(repo_root, cache_path, cache)
        if "faces_object" in cache:
            enriched_path = repo_root / "outputs/dataset_audit/cache" / f"{cache_path.stem}_audit_enriched.npz"
            save_cache(enriched_path, cache)
            cache_path = enriched_path
    rows, summary = audit_surface_collision_cache(
        cache_path,
        stride=args.stride,
        start=args.start,
        end=args.end,
        unit_scale_to_mm=args.unit_scale_to_mm,
        eps_mm=args.eps_mm,
    )
    write_csv(repo_root / args.out_csv, rows)
    write_json(repo_root / args.out_json, summary)
    print(f"Wrote surface-collision frames: {repo_root / args.out_csv}")
    print(f"Wrote surface-collision summary: {repo_root / args.out_json}")
    print(f"surface_collision_frame_ratio={summary['surface_collision_frame_ratio']:.6f}")


if __name__ == "__main__":
    main()

