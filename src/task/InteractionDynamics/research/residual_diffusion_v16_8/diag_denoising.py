"""读取 V16.8 checkpoint 的 oracle denoising 结果。"""
import argparse
import json
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint")
args = parser.parse_args()
result = torch.load(args.checkpoint, map_location="cpu")
print(json.dumps(result["oracle_denoising_rmse_norm"], ensure_ascii=False, indent=2))
