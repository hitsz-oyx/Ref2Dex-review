"""V20.2 Stage B：评估 free/realized 场、shortcut、stable 与终点穿透。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.eval_mano_projector_v20_1 import stable, terminal_penetration
from src.task.InteractionDynamics.field_h_realizer_v20_2 import FieldHRealizerV20_2
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_h_realizer_v20_2 import (apply_ablation, error_metrics,
    load_field, move, predict_field)
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--field-checkpoint",type=Path,required=True);p.add_argument("--realizer-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--samples",type=int,default=4)
    p.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    p.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR));args=p.parse_args()
    device=torch.device("cuda");field,mean,std=load_field(args.field_checkpoint,device)
    payload=torch.load(args.realizer_checkpoint,map_location="cpu");saved=payload["args"]
    realizer=FieldHRealizerV20_2(dim=saved["dim"],temporal_layers=saved["temporal_layers"]).to(device)
    realizer.load_state_dict(payload["model"]);realizer.eval();h_std=payload["h_std"].to(device)
    dataset=CachedFieldV20Dataset(args.cache,"train");layers=ManoLayers(args.grab_root,args.mano_path,device);rows=[]
    for index,raw in enumerate(DataLoader(dataset,batch_size=1,shuffle=False)):
        if index>=args.samples:break
        batch=move(raw,device);predicted=predict_field(field,batch,mean,std);gt=batch["future_y"]
        shuffled_raw=next(iter(DataLoader(torch.utils.data.Subset(dataset,[(index+1)%min(args.samples,len(dataset))]),batch_size=1)))
        shuffled_batch=move(shuffled_raw,device);shuffled_predicted=predict_field(field,shuffled_batch,mean,std)
        results={};normal_surface=None;normal_y=None;normal_z=None
        for mode in ("normal","no_y","shuffle_y","no_object","no_current_h"):
            inputs=apply_ablation(predicted,batch["anchors_cm"],batch["object_patches"],batch["current_h"],mode)
            if mode=="shuffle_y":inputs=(shuffled_predicted,*inputs[1:])
            with torch.no_grad():
                z=realizer(*inputs);realized,surface=decode_mano_field(z*h_std,batch,layers)
            results[mode]=error_metrics(realized,gt[...,:7])
            if mode=="normal":normal_surface,normal_y,normal_z=surface,realized,z
        _,gt_surface=decode_mano_field(batch["future_delta_h"],batch,layers)
        row={"index":index,"free_vs_gt":error_metrics(predicted[...,:7],gt[...,:7]),
             "realized_vs_gt":results,"realized_vs_free":error_metrics(normal_y,predicted[...,:7]),
             "free_stable":stable(predicted),"realized_stable":stable(normal_y),"gt_stable":stable(gt),
             "realized_penetration":terminal_penetration(normal_surface,batch,args.grab_root),
             "gt_penetration":terminal_penetration(gt_surface,batch,args.grab_root),
             "normalized_h_abs_max":float(normal_z.abs().max()),
             "normalized_h_over_3_ratio":float((normal_z.abs()>3).float().mean())}
        rows.append(row);print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
