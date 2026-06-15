#!/usr/bin/env python
from __future__ import annotations

import argparse

from _script_utils import bootstrap, load_adapter


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.mesh_io import write_json

    parser = argparse.ArgumentParser(description="Inspect static contact/proximity/affordance information.")
    parser.add_argument("--out-json", default="outputs/dataset_audit/contact_info_summary.json")
    parser.add_argument("--max-hits", type=int, default=80)
    args = parser.parse_args()
    adapter = load_adapter(repo_root)
    summary = adapter.inspect_contact_info(repo_root, max_hits=args.max_hits)
    write_json(repo_root / args.out_json, summary)
    print(f"Wrote contact info summary: {repo_root / args.out_json}")


if __name__ == "__main__":
    main()

