#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from _script_utils import bootstrap


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.mesh_io import read_csv, write_csv, write_json
    from dataset_audit.common.point_sdf import audit_point_sdf_cache
    from dataset_audit.common.statistics import aggregate_sequence_summaries

    parser = argparse.ArgumentParser(description="Sample multiple caches and aggregate negative-distance statistics.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--cache-glob", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--frames-per-seq", type=int, default=None)
    parser.add_argument("--out-json", default="outputs/dataset_audit/sample_penetration_statistics.json")
    parser.add_argument("--out-csv", default="outputs/dataset_audit/sample_penetration_statistics.csv")
    parser.add_argument("--surface-mode", choices=["vertices", "face_centers", "vertices_and_face_centers"], default="vertices")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--unit-scale-to-mm", type=float, default=1000.0)
    args = parser.parse_args()
    cache_paths: list[Path] = []
    if args.cache_glob:
        cache_paths.extend(Path(path) for path in sorted(glob.glob(str(repo_root / args.cache_glob))))
    if args.manifest:
        for row in read_csv(repo_root / args.manifest):
            cache_path = row.get("cache_path") or row.get("unified_cache_path")
            if cache_path:
                path = Path(cache_path)
                cache_paths.append(path if path.is_absolute() else repo_root / path)
    if args.sample_size is not None:
        cache_paths = cache_paths[: args.sample_size]
    rows = []
    failures = 0
    for cache_path in cache_paths:
        try:
            end = args.frames_per_seq if args.frames_per_seq is not None else None
            _, summary = audit_point_sdf_cache(
                cache_path,
                stride=args.stride,
                end=end,
                surface_mode=args.surface_mode,
                unit_scale_to_mm=args.unit_scale_to_mm,
            )
            rows.append(summary)
        except Exception as exc:
            failures += 1
            rows.append({"status": "failed", "cache_path": str(cache_path), "error": f"{type(exc).__name__}: {exc}"})
    aggregate = aggregate_sequence_summaries(rows, failures=0)
    aggregate["sequence_summaries"] = rows
    write_csv(repo_root / args.out_csv, rows)
    write_json(repo_root / args.out_json, aggregate)
    print(f"Wrote sample statistics CSV: {repo_root / args.out_csv}")
    print(f"Wrote sample statistics JSON: {repo_root / args.out_json}")


if __name__ == "__main__":
    main()

