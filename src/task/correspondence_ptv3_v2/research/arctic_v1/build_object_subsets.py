from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-object symlink subsets from the ARCTIC V1 manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="/tmp/arctic_eval_stratified_v1_by_object")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    by_object: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in payload["selected_files"]:
        by_object[item["object"]].append(item)

    if output_root.exists() or output_root.is_symlink():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "objects": [],
    }

    for obj, items in sorted(by_object.items()):
        obj_root = output_root / obj
        for item in items:
            src = Path(item["path"]).resolve()
            rel = src.relative_to(Path(payload["source_root"]).resolve())
            dst = obj_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
        summary["objects"].append(
            {
                "object": obj,
                "num_sequences": len(items),
                "paths": [item["path"] for item in items],
            }
        )

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
