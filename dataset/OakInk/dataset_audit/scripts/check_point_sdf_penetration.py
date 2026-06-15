#!/usr/bin/env python
from __future__ import annotations

import argparse

from _script_utils import add_common_args, bootstrap, load_adapter


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.mesh_io import load_cache, save_cache, write_csv, write_json
    from dataset_audit.common.point_sdf import audit_point_sdf_cache

    parser = argparse.ArgumentParser(description="Check point-SDF negative-distance penetration metrics.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-json", default="outputs/dataset_audit/penetration_summary.json")
    parser.add_argument("--out-csv", default="outputs/dataset_audit/penetration_frames.csv")
    parser.add_argument("--surface-mode", choices=["vertices", "face_centers", "vertices_and_face_centers"], default="vertices")
    parser.add_argument("--eps-mm", type=float, default=1.0)
    parser.add_argument("--minor-depth-mm", type=float, default=2.0)
    parser.add_argument("--minor-inside-ratio", type=float, default=0.002)
    parser.add_argument("--significant-depth-mm", type=float, default=5.0)
    parser.add_argument("--significant-inside-ratio", type=float, default=0.01)
    parser.add_argument("--significant-inside-count", type=float, default=20)
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
    thresholds = {
        "eps_mm": args.eps_mm,
        "minor_depth_mm": args.minor_depth_mm,
        "minor_inside_ratio": args.minor_inside_ratio,
        "significant_depth_mm": args.significant_depth_mm,
        "significant_inside_ratio": args.significant_inside_ratio,
        "significant_inside_count": args.significant_inside_count,
    }
    rows, summary = audit_point_sdf_cache(
        cache_path,
        stride=args.stride,
        start=args.start,
        end=args.end,
        surface_mode=args.surface_mode,
        unit_scale_to_mm=args.unit_scale_to_mm,
        thresholds=thresholds,
    )
    write_csv(repo_root / args.out_csv, rows)
    write_json(repo_root / args.out_json, summary)
    print(f"Wrote point-SDF frames: {repo_root / args.out_csv}")
    print(f"Wrote point-SDF summary: {repo_root / args.out_json}")
    print(f"max_frame_max_negative_depth_mm={summary['max_frame_max_negative_depth_mm']:.6f}")


if __name__ == "__main__":
    main()

