"""Summarize contact/geometry distributions from one or more Stage 3 roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root: Path, max_files: int) -> dict[str, object]:
    files = sorted(root.glob("**/*.npz"))
    if max_files > 0:
        files = files[:max_files]
    distances = []
    obj_norm = []
    hand_norm = []
    frames = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            d = np.asarray(data["hand_to_obj_min_dist"], dtype=np.float32)
            o = np.asarray(data["obj_points"], dtype=np.float32)
            h = np.asarray(data["hand_points"], dtype=np.float32)
        distances.append(d.reshape(-1))
        obj_norm.append(np.linalg.norm(o.reshape(-1, 3), axis=1))
        hand_norm.append(np.linalg.norm(h.reshape(-1, 3), axis=1))
        frames += int(d.shape[0])
    if not distances:
        raise ValueError(f"No Stage 3 files under {root}")
    d = np.concatenate(distances)
    on = np.concatenate(obj_norm)
    hn = np.concatenate(hand_norm)
    return {
        "root": str(root.resolve()), "files": len(files), "frames": frames,
        "distance_mean_m": float(d.mean()),
        "distance_quantiles_m": {str(q): float(np.quantile(d, q)) for q in (.01, .05, .25, .5, .75, .95, .99)},
        "contact_fraction": {f"lt_{int(mm)}mm": float((d < mm / 1000.0).mean()) for mm in (5, 10, 20, 30, 50)},
        "obj_radius_quantiles_m": {str(q): float(np.quantile(on, q)) for q in (.05, .5, .95)},
        "hand_radius_quantiles_m": {str(q): float(np.quantile(hn, q)) for q in (.05, .5, .95)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = [summarize(Path(root), args.max_files) for root in args.roots]
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
