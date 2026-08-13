"""V20.3 Direct-H 的 split 指标、shortcut、行为与穿透评估。"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from process.GRAB.raw import DEFAULT_GRAB_ROOT,DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.direct_h_dynamics_v20_3 import DirectHDynamicsV20_3
from src.task.InteractionDynamics.eval_mano_projector_v20_1 import terminal_penetration
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_direct_h_v20_3 import apply_direct_ablation,evaluate
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers,move


def behavior_metrics(prediction:torch.Tensor,target:torch.Tensor,current_y:torch.Tensor)->dict:
    def stable(field):
        terminal=field[:,:,-3:];contact=(terminal[...,3]<2).sum(1)
        return (contact>=4).all(1)&(terminal[...,4:7].square().mean((1,2,3)).sqrt()<.3)
    pred_stable,gt_stable=stable(prediction),stable(target);formation=(current_y[...,3]<2).sum(1)<4
    pred_contact=prediction[...,3]<2;gt_contact=target[...,3]<2
    tp=(pred_contact&gt_contact).sum();precision=tp/(pred_contact.sum().clamp_min(1));recall=tp/(gt_contact.sum().clamp_min(1))
    return {"stable_rate":float(pred_stable.float().mean()),"gt_stable_rate":float(gt_stable.float().mean()),
        "formation_samples":int(formation.sum()),"formation_stable_rate":float(pred_stable[formation].float().mean()) if formation.any() else 0.,
        "formation_gt_stable_rate":float(gt_stable[formation].float().mean()) if formation.any() else 0.,
        "contact_f1":float(2*precision*recall/(precision+recall).clamp_min(1e-8)),
        "terminal_contact_mean":float(pred_contact[:,:,-1].sum(1).float().mean()),
        "gt_terminal_contact_mean":float(gt_contact[:,:,-1].sum(1).float().mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("train","val","test"),default="test");p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--penetration-samples",type=int,default=12)
    p.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    p.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR));args=p.parse_args();device=torch.device("cuda")
    payload=torch.load(args.checkpoint,map_location="cpu");saved=payload["args"]
    model=DirectHDynamicsV20_3(dim=saved["dim"],temporal_layers=saved["temporal_layers"]).to(device)
    model.load_state_dict(payload["model"]);model.eval();h_std=payload["h_std"].to(device);field_std=payload["field_std"].to(device)
    dataset=CachedFieldV20Dataset(args.cache,args.split);loader=DataLoader(dataset,args.batch_size,shuffle=False,num_workers=0)
    layers=ManoLayers(args.grab_root,args.mano_path,device)
    result=evaluate(model,loader,h_std,field_std,layers,device);predictions=[];targets=[];currents=[];penetrations=[]
    with torch.no_grad():
        for raw in loader:
            batch=move(raw,device);z=model(batch["current_y"],batch["anchors_cm"],batch["object_patches"],batch["current_h"])
            y_h,surface=decode_mano_field(z*h_std,batch,layers);predictions.append(y_h.cpu());targets.append(batch["future_y"][...,:7].cpu());currents.append(batch["current_y"].cpu())
            for local in range(min(len(y_h),max(0,args.penetration_samples-len(penetrations)))):
                single={k:([v[local]] if isinstance(v,list) else v[local:local+1]) for k,v in batch.items()}
                penetrations.append(terminal_penetration(surface[local:local+1],single,args.grab_root))
    result["behavior"]=behavior_metrics(torch.cat(predictions),torch.cat(targets),torch.cat(currents))
    valid=[x for x in penetrations if x["valid"]];result["penetration"]={"samples":len(valid),
        "max_mm_mean":sum(x["max_mm"] for x in valid)/max(len(valid),1),"mean_mm_mean":sum(x["mean_mm"] for x in valid)/max(len(valid),1),
        "ratio_mean":sum(x["ratio"] for x in valid)/max(len(valid),1)}
    first=move(next(iter(loader)),device);order=torch.randperm(first["current_y"].shape[1],device=device)
    with torch.no_grad():
        normal=model(first["current_y"],first["anchors_cm"],first["object_patches"],first["current_h"])
        permuted=model(first["current_y"][:,order],first["anchors_cm"][:,order],first["object_patches"][:,order],first["current_h"])
    result["permutation_max_abs"]=float((normal-permuted).abs().max());args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
