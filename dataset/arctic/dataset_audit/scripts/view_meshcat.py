#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from _script_utils import bootstrap


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.meshcat_viewer import headless_check, play_meshcat

    parser = argparse.ArgumentParser(description="View a unified mesh sequence cache in MeshCat.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--show-centers", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--highlight-points", default=None, help="Reserved for future penetration-point highlighting.")
    parser.add_argument("--headless-check", action="store_true")
    args = parser.parse_args()
    cache_path = repo_root / args.cache
    if args.headless_check:
        print(json.dumps(headless_check(cache_path), indent=2, sort_keys=True))
        return
    play_meshcat(
        cache_path,
        fps=args.fps,
        stride=args.stride,
        start=args.start,
        end=args.end,
        loop=args.loop,
        show_centers=args.show_centers,
    )


if __name__ == "__main__":
    main()

