#!/usr/bin/env python
from __future__ import annotations

import argparse

from _script_utils import bootstrap, load_adapter


def main() -> None:
    repo_root = bootstrap(__file__)
    from dataset_audit.common.mesh_io import create_toy_cache, write_json

    parser = argparse.ArgumentParser(description="Prepare a unified mesh-sequence cache for audit tools.")
    parser.add_argument("--item", default=None, help="Dataset-specific manifest item id.")
    parser.add_argument("--manifest", default="outputs/dataset_audit/manifest.csv")
    parser.add_argument("--out-cache", default=None)
    parser.add_argument("--toy", action="store_true", help="Create a small synthetic hand-object cache for smoke tests.")
    args = parser.parse_args()
    if args.toy:
        out_cache = repo_root / (args.out_cache or "outputs/dataset_audit/toy_cache/toy_hand_object.npz")
        payload = create_toy_cache(out_cache)
        print(f"Wrote toy cache: {out_cache}")
        print(f"Frames: {payload['verts_object'].shape[0]}")
        return
    if not args.item:
        raise SystemExit("Provide --item from the manifest, or use --toy for a synthetic smoke-test cache.")
    adapter = load_adapter(repo_root)
    result = adapter.prepare_sequence_cache(
        repo_root,
        item=args.item,
        manifest_path=repo_root / args.manifest,
        out_cache=None if args.out_cache is None else repo_root / args.out_cache,
    )
    if "summary_path" in result:
        write_json(result["summary_path"], result)
    print(result.get("message", result))


if __name__ == "__main__":
    main()

