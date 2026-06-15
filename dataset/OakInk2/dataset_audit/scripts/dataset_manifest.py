#!/usr/bin/env python
from __future__ import annotations

import argparse

from _script_utils import bootstrap, load_adapter


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.mesh_io import write_csv, write_json

    parser = argparse.ArgumentParser(description="Generate a lightweight dataset audit manifest.")
    parser.add_argument("--out", default="outputs/dataset_audit/manifest.csv")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    adapter = load_adapter(repo_root)
    rows, summary = adapter.build_manifest(repo_root, limit=args.limit)
    out_path = repo_root / args.out
    write_csv(out_path, rows)
    summary_path = repo_root / args.summary_json if args.summary_json else out_path.with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"Wrote {len(rows)} manifest rows: {out_path}")
    print(f"Wrote manifest summary: {summary_path}")


if __name__ == "__main__":
    main()

