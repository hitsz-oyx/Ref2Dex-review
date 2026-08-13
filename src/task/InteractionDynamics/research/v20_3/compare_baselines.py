"""在同一 V20 test split 上统一比较 Free-Y、Direct-H 与 Y→H realizer。"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from process.GRAB.raw import DEFAULT_GRAB_ROOT,DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.direct_h_dynamics_v20_3 import DirectHDynamicsV20_3
from src.task.InteractionDynamics.eval_direct_h_v20_3 import behavior_metrics
from src.task.InteractionDynamics.field_h_realizer_v20_2 import FieldHRealizerV20_2
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_h_realizer_v20_2 import error_metrics,load_field,predict_field
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers,move


def load_direct(path,device):
    payload=torch.load(path,map_location="cpu");a=payload["args"]
    model=DirectHDynamicsV20_3(dim=a["dim"],temporal_layers=a["temporal_layers"]).to(device)
    model.load_state_dict(payload["model"]);return model.eval(),payload["h_std"].to(device)


def load_realizer(path,device):
    payload=torch.load(path,map_location="cpu");a=payload["args"]
    model=FieldHRealizerV20_2(dim=a["dim"],temporal_layers=a["temporal_layers"]).to(device)
    model.load_state_dict(payload["model"]);return model.eval(),payload["h_std"].to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--field",type=Path,required=True);p.add_argument("--direct",type=Path,required=True)
    p.add_argument("--realizer",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    p.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR));args=p.parse_args();device=torch.device("cuda")
    field,mean,std=load_field(args.field,device);direct,direct_std=load_direct(args.direct,device);realizer,realizer_std=load_realizer(args.realizer,device)
    layers=ManoLayers(args.grab_root,args.mano_path,device);loader=DataLoader(CachedFieldV20Dataset(args.cache,"test"),8,shuffle=False)
    values={k:[] for k in ("free","direct_h","y_to_h","gt")};currents=[];parity_max=0.
    with torch.no_grad():
        for raw in loader:
            batch=move(raw,device);free=predict_field(field,batch,mean,std)
            direct_y,_=decode_mano_field(direct(batch["current_y"],batch["anchors_cm"],batch["object_patches"],batch["current_h"])*direct_std,batch,layers)
            realized_y,_=decode_mano_field(realizer(free,batch["anchors_cm"],batch["object_patches"],batch["current_h"])*realizer_std,batch,layers)
            parity,_=decode_mano_field(batch["future_delta_h"],batch,layers);gt=batch["future_y"][...,:7]
            parity_max=max(parity_max,float((parity-gt).abs().max()));currents.append(batch["current_y"].cpu())
            for key,value in (("free",free[...,:7]),("direct_h",direct_y),("y_to_h",realized_y),("gt",gt)):values[key].append(value.cpu())
    gt=torch.cat(values.pop("gt"));current=torch.cat(currents);result={"samples":len(gt),"gt_parity_max_abs_cm":parity_max,"methods":{}}
    for key,rows in values.items():
        prediction=torch.cat(rows);result["methods"][key]={"error":error_metrics(prediction,gt),"behavior":behavior_metrics(prediction,gt,current)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":main()
