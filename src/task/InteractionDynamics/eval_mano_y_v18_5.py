"""评估 V18.5 MANO-constrained Y-supervised checkpoint。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.eval_grasp_v18 import subset_masks, trajectory_statistics
from src.task.InteractionDynamics.mano_hand_transition import CachedManoHDataset, ManoHandTransition
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import decode_mano_y
from src.task.InteractionDynamics.train_mano_h_v18_4 import evenly_spaced
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move


def contact_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction.reshape(-1, 8, 7)[..., 6] < 2
    gt = target.reshape(-1, 8, 7)[..., 6] < 2
    tp = (pred & gt).sum().float(); precision = tp / pred.sum().clamp_min(1)
    recall = tp / gt.sum().clamp_min(1)
    return float(2 * precision * recall / (precision + recall).clamp_min(1e-12))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--split",choices=("train","val","test"),required=True); parser.add_argument("--samples",type=int)
    parser.add_argument("--first",action="store_true")
    parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    parser.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR))
    args=parser.parse_args(); device=torch.device("cuda"); checkpoint=torch.load(args.checkpoint,map_location="cpu")
    saved=checkpoint["args"]; model=ManoHandTransition(dim=saved["dim"],layers=saved["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval(); delta_stats={k:v.to(device) for k,v in checkpoint["delta_stats"].items()}
    full=CachedManoHDataset(args.cache,args.split)
    dataset=(Subset(full,range(min(args.samples,len(full)))) if args.first and args.samples is not None
             else evenly_spaced(full,args.samples))
    loader=DataLoader(dataset,args.batch_size,shuffle=False,num_workers=0); layers=ManoLayers(args.grab_root,args.mano_path,device)
    names=("overall","formation","transition","maintenance")
    totals={name:{"n":0,"sq":torch.zeros(3),"success":0,"pred":[],"gt":[],"contacts":[]} for name in names}
    normalized_abs=[]
    for raw in loader:
        batch=move(raw,device); normalized=model(batch["state"],batch["anchors_cm"],batch["object_patches"],batch["current_h"])
        prediction,_=decode_mano_y(normalized*delta_stats["delta_std"]+delta_stats["delta_mean"],batch,layers)
        normalized_abs.append(normalized.abs().cpu()); stable=trajectory_statistics(prediction)
        for name,mask in subset_masks(batch).items():
            n=int(mask.sum());
            if not n: continue
            row=totals[name]; row["n"]+=n; error=(prediction[mask]-batch["future"][mask]).reshape(-1,8,7)
            row["sq"]+=torch.tensor([error[...,:3].square().sum().cpu(),error[...,3:6].square().sum().cpu(),error[...,6].square().sum().cpu()])
            row["success"]+=int(stable["success"][mask].sum()); row["contacts"].append(stable["contact"][mask].cpu())
            row["pred"].append(prediction[mask].cpu()); row["gt"].append(batch["future"][mask].cpu())
    result={"split":args.split,"samples":len(dataset),"normalized_h":{"mean_abs":float(torch.cat(normalized_abs).mean()),"max_abs":float(torch.cat(normalized_abs).max()),"outside_3sigma_rate":float((torch.cat(normalized_abs)>3).float().mean())},"groups":{}}
    for name,row in totals.items():
        if not row["n"]: continue
        pred=torch.cat(row["pred"]); gt=torch.cat(row["gt"]); elements=row["n"]*128*8
        rmse=(row["sq"]/torch.tensor([elements*3,elements*3,elements])).sqrt(); contacts=torch.cat(row["contacts"]).float()
        margin=pred.reshape(-1,8,7); violation=(margin[...,3:6].norm(dim=-1)>margin[...,6]+1e-6).float().mean()
        result["groups"][name]={"samples":row["n"],"u_rmse_cm":float(rmse[0]),"r_rmse_cm":float(rmse[1]),"d_rmse_cm":float(rmse[2]),"stable_success_rate":row["success"]/row["n"],"contact_f1":contact_f1(pred,gt),"terminal_contact":{"mean":float(contacts.mean()),"median":float(contacts.median()),"p10":float(torch.quantile(contacts,.1)),"p90":float(torch.quantile(contacts,.9))},"rd_violation_rate":float(violation)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__": main()
