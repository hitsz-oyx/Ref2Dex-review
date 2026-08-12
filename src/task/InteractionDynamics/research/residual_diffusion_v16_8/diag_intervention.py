"""读取 V16.8 checkpoint 的 controlled baseline 与 Goal intervention。"""
import argparse
import json
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
args = parser.parse_args()
result = torch.load(args.checkpoint, map_location="cpu")
print(json.dumps({"v16_7_deterministic": result["v16_7_deterministic"],
                  "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
