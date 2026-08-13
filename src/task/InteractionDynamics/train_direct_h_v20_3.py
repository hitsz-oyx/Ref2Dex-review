"""V20.3 Direct-H baseline 训练；只通过 causal r/d/v 监督 MANO trajectory。"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset

from process.GRAB.raw import DEFAULT_GRAB_ROOT,DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.direct_h_dynamics_v20_3 import DirectHDynamicsV20_3
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_h_realizer_v20_2 import delta_h_std,error_metrics,normalized_field_loss
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers,move


def apply_direct_ablation(current_y,anchors,patches,current_h,mode):
    if mode=="normal":return current_y,anchors,patches,current_h
    if mode=="no_y":return torch.zeros_like(current_y),anchors,patches,current_h
    if mode=="shuffle_y":return current_y.roll(1,0),anchors,patches,current_h
    if mode=="no_object":return current_y,torch.zeros_like(anchors),torch.zeros_like(patches),current_h
    if mode=="no_current_h":return current_y,anchors,patches,torch.zeros_like(current_h)
    raise ValueError(mode)


@torch.no_grad()
def evaluate(model,loader,h_std,field_std,layers,device):
    modes=("normal","no_y","shuffle_y","no_object","no_current_h");errors={k:[] for k in modes};zs=[]
    for raw in loader:
        batch=move(raw,device)
        for mode in modes:
            z=model(*apply_direct_ablation(batch["current_y"],batch["anchors_cm"],batch["object_patches"],batch["current_h"],mode))
            y_h,_=decode_mano_field(z*h_std,batch,layers);errors[mode].append(y_h-batch["future_y"][...,:7])
            if mode=="normal":zs.append(z)
    def aggregate(values):
        value=torch.cat(values);return error_metrics(value,torch.zeros_like(value))
    z=torch.cat(zs);return {"realized_vs_gt":{k:aggregate(v) for k,v in errors.items()},
        "normalized_h_abs_max":float(z.abs().max()),"normalized_h_over_3_ratio":float((z.abs()>3).float().mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--field-checkpoint",type=Path,required=True,help="只读取 train-only field std")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,default=4000)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--dim",type=int,default=192);p.add_argument("--temporal-layers",type=int,default=2)
    p.add_argument("--eval-every",type=int,default=250);p.add_argument("--controlled32",action="store_true")
    p.add_argument("--prior-weight",type=float,default=1e-4)
    p.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    p.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR));args=p.parse_args()
    torch.manual_seed(42);device=torch.device("cuda");full=CachedFieldV20Dataset(args.cache,"train")
    train=Subset(full,range(min(32,len(full)))) if args.controlled32 else full
    val=train if args.controlled32 else CachedFieldV20Dataset(args.cache,"val")
    loader=DataLoader(train,args.batch_size,shuffle=True,num_workers=0);validation=DataLoader(val,args.batch_size,shuffle=False,num_workers=0)
    field_payload=torch.load(args.field_checkpoint,map_location="cpu");field_std=field_payload["std"].to(device)
    h_std=delta_h_std(train).to(device);model=DirectHDynamicsV20_3(dim=args.dim,temporal_layers=args.temporal_layers).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    layers=ManoLayers(args.grab_root,args.mano_path,device);iterator=iter(loader);args.output.mkdir(parents=True,exist_ok=True);best=float("inf")
    for step in range(1,args.steps+1):
        try:raw=next(iterator)
        except StopIteration:iterator=iter(loader);raw=next(iterator)
        batch=move(raw,device);z=model(batch["current_y"],batch["anchors_cm"],batch["object_patches"],batch["current_h"])
        y_h,_=decode_mano_field(z*h_std,batch,layers);terms,rdv=normalized_field_loss(y_h,batch["future_y"],field_std)
        prior=torch.relu(z.abs()-3).square().mean();loss=rdv+args.prior_weight*prior
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1:
            gradients=[v.grad for v in model.parameters() if v.grad is not None]
            print(json.dumps({"gradient_gate":{"finite":all(torch.isfinite(v).all() for v in gradients),"nonzero":any(float(v.abs().sum())>0 for v in gradients)}}),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            result=evaluate(model.eval(),validation,h_std,field_std,layers,device);model.train()
            print(json.dumps({"step":step,"loss":float(loss),"terms":{k:float(v) for k,v in terms.items()},"prior":float(prior),"validation":result}),flush=True)
            payload={"model":model.state_dict(),"h_std":h_std.cpu(),"field_std":field_std.cpu(),"args":vars(args),"step":step,"validation":result}
            torch.save(payload,args.output/"latest.pt");metrics=result["realized_vs_gt"]["normal"]
            score=sum(metrics[k] for k in ("r_rmse_cm","d_rmse_cm","v_rmse_cm"))
            if score<best:best=score;torch.save(payload,args.output/"best.pt")


if __name__=="__main__":main()
