"""读取 V16.8 checkpoint 的 sampler stability 与四 seed 诊断。"""
import argparse
import json
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
args = parser.parse_args()
result = torch.load(args.checkpoint, map_location="cpu")
print(json.dumps({"sampling_stability": result["sampling_stability"],
                  "four_seed_dynamic": result["four_seed_dynamic"]},
                 ensure_ascii=False, indent=2))
