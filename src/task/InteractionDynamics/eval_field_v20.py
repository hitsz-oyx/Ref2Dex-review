"""V20 normal/mean/shuffle 与 permutation-equivariance 评估。"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.field_dynamics_v20 import FieldDynamicsTransition
from src.task.InteractionDynamics.train_field_v20 import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--split",default="train");args=p.parse_args()
    device=torch.device("cuda");payload=torch.load(args.checkpoint,map_location="cpu");saved=payload["args"]
    model=FieldDynamicsTransition(dim=saved["dim"],layers=saved["layers"]).to(device)
    model.load_state_dict(payload["model"]);model.eval();mean=payload["mean"].to(device);std=payload["std"].to(device)
    dataset=CachedFieldV20Dataset(args.cache,args.split);loader=DataLoader(dataset,8)
    result={name:evaluate(model,loader,mean,std,device,name) for name in ("normal","mean","shuffle")}
    batch=next(iter(loader));batch={k:v.to(device) for k,v in batch.items()};permutation=torch.randperm(128,device=device)
    normal=model(batch["current_y"],batch["anchors_cm"],batch["object_patches"])
    permuted=model(batch["current_y"][:,permutation],batch["anchors_cm"][:,permutation],batch["object_patches"][:,permutation])
    inverse=torch.argsort(permutation);result["permutation_max_abs"]=float((normal-permuted[:,inverse]).abs().max())
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
