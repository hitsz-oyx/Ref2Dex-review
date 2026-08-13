"""V20.2 Stage C：冻结 learned H realizer，微调 V20 field predictor。"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset

from process.GRAB.raw import DEFAULT_GRAB_ROOT,DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.field_dynamics_v20 import FieldDynamicsTransition
from src.task.InteractionDynamics.field_h_realizer_v20_2 import FieldHRealizerV20_2
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_field_v20 import loss_terms
from src.task.InteractionDynamics.train_h_realizer_v20_2 import error_metrics,move,normalized_field_loss
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers


def load_models(field_path,realizer_path,device):
    field_payload=torch.load(field_path,map_location="cpu");a=field_payload["args"]
    field=FieldDynamicsTransition(dim=a["dim"],layers=a["layers"]).to(device);field.load_state_dict(field_payload["model"])
    realizer_payload=torch.load(realizer_path,map_location="cpu");a=realizer_payload["args"]
    realizer=FieldHRealizerV20_2(dim=a["dim"],temporal_layers=a["temporal_layers"]).to(device)
    realizer.load_state_dict(realizer_payload["model"]);realizer.requires_grad_(False).eval()
    return (field,realizer,field_payload["mean"].to(device),field_payload["std"].to(device),
            realizer_payload["h_std"].to(device),field_payload["args"])


@torch.no_grad()
def evaluate(field,realizer,loader,mean,std,h_std,layers,device):
    free=[];realized=[];gap=[]
    for raw in loader:
        batch=move(raw,device);delta=field(batch["current_y"],batch["anchors_cm"],batch["object_patches"])*std+mean
        predicted=batch["current_y"][:,:,None]+delta;z=realizer(predicted,batch["anchors_cm"],batch["object_patches"],batch["current_h"])
        y_h,_=decode_mano_field(z*h_std,batch,layers);free.append(predicted[...,:7]-batch["future_y"][...,:7])
        realized.append(y_h-batch["future_y"][...,:7]);gap.append(y_h-predicted[...,:7])
    zero=lambda x:torch.zeros_like(torch.cat(x));return {"free_vs_gt":error_metrics(torch.cat(free),zero(free)),
        "realized_vs_gt":error_metrics(torch.cat(realized),zero(realized)),"realized_vs_free":error_metrics(torch.cat(gap),zero(gap))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--field-checkpoint",type=Path,required=True);p.add_argument("--realizer-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,default=500)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--lr",type=float,default=3e-5)
    p.add_argument("--eval-every",type=int,default=100);p.add_argument("--controlled32",action="store_true")
    p.add_argument("--feasible-weight",type=float,default=.2);p.add_argument("--consistency-weight",type=float,default=.05)
    p.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    p.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR));args=p.parse_args()
    torch.manual_seed(42);device=torch.device("cuda");full=CachedFieldV20Dataset(args.cache,"train")
    train=Subset(full,range(min(32,len(full)))) if args.controlled32 else full
    validation_dataset=train if args.controlled32 else CachedFieldV20Dataset(args.cache,"val")
    loader=DataLoader(train,args.batch_size,shuffle=True,num_workers=0);validation=DataLoader(validation_dataset,args.batch_size,shuffle=False,num_workers=0)
    field,realizer,mean,std,h_std,field_args=load_models(args.field_checkpoint,args.realizer_checkpoint,device)
    layers=ManoLayers(args.grab_root,args.mano_path,device);optimizer=torch.optim.AdamW(field.parameters(),lr=args.lr)
    iterator=iter(loader);args.output.mkdir(parents=True,exist_ok=True);best=float("inf")
    for step in range(1,args.steps+1):
        try:raw=next(iterator)
        except StopIteration:iterator=iter(loader);raw=next(iterator)
        batch=move(raw,device);normalized=field(batch["current_y"],batch["anchors_cm"],batch["object_patches"])
        delta=normalized*std+mean;predicted=batch["current_y"][:,:,None]+delta
        z=realizer(predicted,batch["anchors_cm"],batch["object_patches"],batch["current_h"])
        y_h,_=decode_mano_field(z*h_std,batch,layers);_,field_loss=loss_terms(normalized,batch["delta_y"],batch["p_valid"],mean,std)
        _,feasible=normalized_field_loss(y_h,batch["future_y"],std)
        consistency=((predicted[...,:7]-y_h.detach())/std[...,:7]).square().mean()
        loss=field_loss+args.feasible_weight*feasible+args.consistency_weight*consistency
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(field.parameters(),1.);optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            result=evaluate(field.eval(),realizer,validation,mean,std,h_std,layers,device);field.train()
            print(json.dumps({"step":step,"loss":float(loss),"field_loss":float(field_loss),"feasible":float(feasible),"consistency":float(consistency),"validation":result}),flush=True)
            saved_args={**vars(args),"dim":field_args["dim"],"layers":field_args["layers"]}
            payload={"model":field.state_dict(),"mean":mean.cpu(),"std":std.cpu(),"args":saved_args,"step":step,"validation":result}
            torch.save(payload,args.output/"latest.pt");score=sum(result["free_vs_gt"][k] for k in ("r_rmse_cm","d_rmse_cm","v_rmse_cm"))
            if score<best:best=score;torch.save(payload,args.output/"best.pt")


if __name__=="__main__":main()
