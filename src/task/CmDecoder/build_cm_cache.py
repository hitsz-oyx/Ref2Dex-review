"""Precompute frozen Cm tokens into an episode-local mmap sidecar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.base import load_config
from src.task.CmDecoder.model import CmDecoderModel


FIELDS = ("hand_points", "hand_normals", "hand_flow", "obj_points", "obj_normals", "obj_valid_mask", "delta_time_s")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src.task.CmDecoder.config:Config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    manifest_path = args.manifest or Path(cfg.data.cache_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_root = Path(cfg.meta.cache_root).resolve()
    episodes = sum((manifest["splits"][name] for name in ("train", "val", "test")), [])
    episodes = episodes[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    model = CmDecoderModel(cfg).to(device).eval()
    checkpoint_sha = _sha256(Path(cfg.meta.cm_checkpoint))
    with torch.no_grad():
        for done, episode in enumerate(episodes, 1):
            episode_root = cache_root / manifest["cache_dirs"][episode]
            task_root = episode_root / "task"
            task_manifest_path = task_root / "manifest.json"
            task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
            output_root = episode_root / "cm"
            output_root.mkdir(parents=True, exist_ok=True)
            cm_manifest_path = output_root / "manifest.json"
            signature = {"schema": "cmdecoder_cm_tokens_v1", "task_manifest_sha256": _sha256(task_manifest_path),
                         "cm_checkpoint_sha256": checkpoint_sha, "samples": int(task_manifest["samples"]),
                         "shape": [int(model.cm.num_cm_tokens), int(model.cm.cm_dim)]}
            if cm_manifest_path.exists() and json.loads(cm_manifest_path.read_text(encoding="utf-8")) == signature:
                print(f"[{done:03d}/{len(episodes):03d}] cached {episode}", flush=True)
                continue
            arrays = {field: np.load(task_root / f"{field}.npy", mmap_mode="r", allow_pickle=False) for field in FIELDS}
            count = int(task_manifest["samples"])
            output = np.lib.format.open_memmap(output_root / "cm_tokens.npy", mode="w+", dtype=np.float32,
                                               shape=(count, int(model.cm.num_cm_tokens), int(model.cm.cm_dim)))
            for start in range(0, count, args.batch_size):
                stop = min(start + args.batch_size, count)
                batch = {field: torch.from_numpy(np.array(value[start:stop], copy=True)).to(device) for field, value in arrays.items()}
                output[start:stop] = model.encode_cm(batch).float().cpu().numpy()
            output.flush()
            cm_manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
            print(f"[{done:03d}/{len(episodes):03d}] built {episode}", flush=True)


if __name__ == "__main__":
    main()
