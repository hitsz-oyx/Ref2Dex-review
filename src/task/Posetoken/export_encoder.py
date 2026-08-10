"""从训练 checkpoint 导出不含 reconstruction decoder 的 PoseEncoder 权重。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model"]
    decoder_prefixes = ("local_decoder.", "global_decoder.")
    encoder = {key: value for key, value in state.items()
               if not key.startswith(decoder_prefixes)}
    config = checkpoint.get("config", {})
    meta = config.get("meta", {})
    payload = {
        "pose_encoder": encoder,
        "config": {
            "num_patches": meta.get("num_patches"),
            "patch_size": meta.get("patch_size"),
            "model_dim": meta.get("model_dim"),
            "global_dim": meta.get("global_dim"),
            "version": "pose_token_v1",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"导出 {len(encoder)} 个 encoder tensors 到 {args.output}")


if __name__ == "__main__":
    main()
