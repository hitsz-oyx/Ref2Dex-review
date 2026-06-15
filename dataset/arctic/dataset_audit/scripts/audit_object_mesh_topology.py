#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _script_utils import bootstrap, load_adapter


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.mesh_io import load_obj_mesh, write_csv, write_json
    from dataset_audit.common.topology import mesh_topology

    parser = argparse.ArgumentParser(description="Audit object mesh topology for watertightness and proxy suitability.")
    parser.add_argument("--out-csv", default="outputs/dataset_audit/object_mesh_topology.csv")
    parser.add_argument("--out-json", default="outputs/dataset_audit/object_mesh_topology_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    adapter = load_adapter(repo_root)
    mesh_paths = adapter.object_mesh_paths(repo_root, limit=args.limit)
    rows = []
    for mesh_path in mesh_paths:
        path = Path(mesh_path)
        try:
            vertices, faces = load_obj_mesh(path)
            row = {"mesh_path": str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path)}
            row.update(mesh_topology(vertices, faces))
            row["recommended_collision_mesh_candidate"] = "yes" if row["is_watertight"] else "needs_proxy_or_heuristic"
        except Exception as exc:
            row = {
                "mesh_path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "reliability": "unreadable",
                "recommended_collision_mesh_candidate": "no",
            }
        rows.append(row)
    summary = {
        "dataset_name": getattr(adapter, "DATASET_NAME", repo_root.name),
        "num_meshes_checked": len(rows),
        "num_watertight": sum(1 for row in rows if row.get("is_watertight") is True),
        "num_unreadable": sum(1 for row in rows if row.get("reliability") == "unreadable"),
        "output_csv": args.out_csv,
    }
    write_csv(repo_root / args.out_csv, rows)
    write_json(repo_root / args.out_json, summary)
    print(f"Wrote topology CSV: {repo_root / args.out_csv}")
    print(f"Wrote topology summary: {repo_root / args.out_json}")


if __name__ == "__main__":
    main()

