"""V20 field-native deterministic transition 训练与 controlled gate。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.field_dynamics_v20 import FieldDynamicsTransition


def channel_statistics(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat([dataset[i]["delta_y"].reshape(-1, 8) for i in range(len(dataset))])
    valid = torch.cat([dataset[i]["p_valid"].expand(128, 8).reshape(-1) for i in range(len(dataset))])
    mean, std = values.mean(0), values.std(0).clamp_min(.01)
    if not bool(valid.any()): mean[7] = 0; std[7] = 1
    return mean.reshape(1, 1, 1, 8), std.reshape(1, 1, 1, 8)


def loss_terms(prediction, target, p_valid, mean, std):
    error = ((prediction * std + mean - target) / std).square()
    groups = {"r": error[..., :3].mean(), "d": error[..., 3].mean(),
              "v": error[..., 4:7].mean()}
    p_mask = p_valid[:, None, None].expand_as(error[..., 7])
    groups["p"] = error[..., 7][p_mask].mean() if p_mask.any() else error[..., 7].sum() * 0
    return groups, groups["r"] + groups["d"] + groups["v"] + .2 * groups["p"]


@torch.no_grad()
def evaluate(model, loader, mean, std, device, intervention="normal"):
    absolute = torch.zeros(4, device=device); count = torch.zeros(4, device=device)
    persistence = torch.zeros(4, device=device); spatial = []
    for raw in loader:
        batch = {k: v.to(device) for k, v in raw.items()}
        pred = model(batch["current_y"], batch["anchors_cm"], batch["object_patches"], intervention)
        pred = pred * std + mean; target = batch["delta_y"]
        for i, value in enumerate((pred[..., :3]-target[..., :3], pred[..., 3]-target[..., 3],
                                   pred[..., 4:7]-target[..., 4:7], pred[..., 7]-target[..., 7])):
            if i == 3:
                mask = batch["p_valid"][:, None, None].expand_as(value); value = value[mask]
            absolute[i] += value.abs().sum(); count[i] += value.numel()
        for i, value in enumerate((target[..., :3], target[..., 3], target[..., 4:7], target[..., 7])):
            persistence[i] += value.abs().sum()
        spatial.append(pred.var(1).mean())
    names=("r_mae_cm","d_mae_cm","v_mae_cm","p_mae_cm")
    result={name:float(absolute[i]/count[i].clamp_min(1)) for i,name in enumerate(names)}
    result.update({"persistence_"+name:float(persistence[i]/count[i].clamp_min(1)) for i,name in enumerate(names)})
    result["output_spatial_variance"] = float(torch.stack(spatial).mean())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--steps",type=int,default=1500)
    parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--dim",type=int,default=128); parser.add_argument("--layers",type=int,default=4)
    parser.add_argument("--controlled32",action="store_true"); parser.add_argument("--eval-every",type=int,default=250)
    args=parser.parse_args(); torch.manual_seed(42); device=torch.device("cuda")
    full=CachedFieldV20Dataset(args.cache,"train"); train=Subset(full,range(min(32,len(full)))) if args.controlled32 else full
    mean,std=(x.to(device) for x in channel_statistics(train)); model=FieldDynamicsTransition(dim=args.dim,layers=args.layers).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr); loader=DataLoader(train,args.batch_size,shuffle=True)
    validation=DataLoader(train,args.batch_size); iterator=iter(loader); args.output.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        try: raw=next(iterator)
        except StopIteration: iterator=iter(loader);raw=next(iterator)
        batch={k:v.to(device) for k,v in raw.items()}; pred=model(batch["current_y"],batch["anchors_cm"],batch["object_patches"])
        terms,loss=loss_terms(pred,batch["delta_y"],batch["p_valid"],mean,std)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            row={"step":step,"loss":float(loss),**{k:float(v) for k,v in terms.items()},
                 "normal":evaluate(model.eval(),validation,mean,std,device)};model.train();print(json.dumps(row),flush=True)
    torch.save({"model":model.state_dict(),"mean":mean.cpu(),"std":std.cpu(),"args":vars(args)},args.output/"latest.pt")


if __name__=="__main__":main()
