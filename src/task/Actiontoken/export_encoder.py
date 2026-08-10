import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    is_v2 = any(key.startswith("flow_action.") for key in checkpoint["model"])
    if is_v2:
        state = {key[len("flow_action.encoder."):]: value
                 for key, value in checkpoint["model"].items()
                 if key.startswith("flow_action.encoder.")}
    else:
        state = {key[len("dynamic."):]: value for key, value in checkpoint["model"].items()
                 if key.startswith("dynamic.") and not key.startswith("dynamic.flow_decoder.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = checkpoint.get("config", {}).get("meta", {})
    if is_v2:
        payload = {"action_encoder": state, "version": "action_token_v2_flow",
                   "num_patches": int(meta.get("num_patches", 64)),
                   "patch_size": int(meta.get("patch_size", 32)),
                   "model_dim": int(meta.get("model_dim", 384)),
                   "motion_scale": float(meta.get("motion_scale", 100.0)), "fps": 30.0}
    else:
        payload = {"dynamic_action_encoder": state, "version": "action_token_v1",
                   "config": meta}
    torch.save(payload, args.output)
    print(f"导出 {len(state)} 个 tensors 到 {args.output}")


if __name__ == "__main__":
    main()
