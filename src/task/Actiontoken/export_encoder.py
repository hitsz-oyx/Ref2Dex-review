import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = {key[len("dynamic."):]: value for key, value in checkpoint["model"].items()
             if key.startswith("dynamic.") and not key.startswith("dynamic.flow_decoder.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dynamic_action_encoder": state, "version": "action_token_v1",
                "config": checkpoint.get("config", {}).get("meta", {})}, args.output)
    print(f"导出 {len(state)} 个 tensors 到 {args.output}")


if __name__ == "__main__":
    main()
