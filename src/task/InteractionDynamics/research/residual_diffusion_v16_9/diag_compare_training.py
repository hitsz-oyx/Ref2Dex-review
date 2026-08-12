"""汇总 V16.8、V16.9 uniform-10k 与 half-high-10k 的关键指标。"""
import argparse
import json
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoints", nargs=3)
args = parser.parse_args()
output = {}
for path in args.checkpoints:
    value = torch.load(path, map_location="cpu")
    output[path] = {"timestep_sampling": value.get("timestep_sampling", "uniform"),
                    "dynamic": value["metrics"]["correct"]["dynamic"],
                    "oracle": value["oracle_denoising_rmse_norm"],
                    "four_seed_dynamic": value["four_seed_dynamic"]}
print(json.dumps(output, ensure_ascii=False, indent=2))
