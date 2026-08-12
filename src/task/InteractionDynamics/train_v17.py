"""V17 full-frame deterministic residual 与 v-diffusion DDP 训练。"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np, torch, yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.task.InteractionDynamics.dataset_v17 import CachedV17Dataset, V17Dataset, dominant_sequences, split_sequences
from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
from src.task.InteractionDynamics.residual_interaction_diffusion import ResidualInteractionDiffusion, make_v_target, sample_residual_v
from src.task.InteractionDynamics.residual_interaction_regression import ResidualInteractionRegressor, persistence_future

def stats_for(dataset, path):
    if path.exists(): return torch.load(path,map_location="cpu")
    sums={k:None for k in ["state","goal","residual"]}; squares={k:None for k in sums}; counts={k:0 for k in sums}
    for i in range(len(dataset)):
        row=dataset[i]
        for k,dims in [("state",(0,)),("goal",None),("residual",(0,))]:
            x=row[k].double()
            s=x if dims is None else x.sum(dims); q=x.square() if dims is None else x.square().sum(dims)
            n=1 if dims is None else x.numel()//s.numel()
            sums[k]=s if sums[k] is None else sums[k]+s; squares[k]=q if squares[k] is None else squares[k]+q; counts[k]+=n
        if (i+1)%1000==0: print(json.dumps({"normalization_samples":i+1}),flush=True)
    out={}
    for k in sums:
        mean=sums[k]/counts[k]; var=(squares[k]/counts[k]-mean.square()).clamp_min(0)
        shape=(1,1,-1) if k!="goal" else (1,-1)
        out[k+"_mean"]=mean.float().reshape(shape); out[k+"_std"]=var.sqrt().float().reshape(shape).clamp_min(.01 if k=="goal" else .05)
    path.parent.mkdir(parents=True,exist_ok=True); torch.save(out,path); return out

def evaluate(det_model,diff_model,loader,stats,device,sampling_steps):
    totals={m:{k:0. for k in ["u","r","d","n","tp","pp","gp"]} for m in ["persistence","deterministic","diffusion"]}
    for batch in loader:
        state=batch["state"].to(device); future=batch["future"].to(device); residual=batch["residual"].to(device)
        anchors=batch["anchors_cm"].to(device); patches=batch["object_patches"].to(device); goal=batch["goal"].to(device)
        sn=(state-stats["state_mean"])/stats["state_std"]; gn=(goal-stats["goal_mean"])/stats["goal_std"]
        persist=persistence_future(state,4)
        pred_det=det_model(sn,anchors,patches,gn)*stats["residual_std"]+stats["residual_mean"]+persist
        noise=torch.randn(residual.shape,device=device)
        pred,_=sample_residual_v(diff_model,sn,anchors,patches,gn,100,initial_noise=noise,sampling_steps=sampling_steps)
        pred_diff=pred*stats["residual_std"]+stats["residual_mean"]+persist
        dynamic=residual.square().mean((1,2)).sqrt()>=.2
        for name,value in [("persistence",persist),("deterministic",pred_det),("diffusion",pred_diff)]:
            p=value.reshape(*value.shape[:-1],4,7)[dynamic]; t=future.reshape(*future.shape[:-1],4,7)[dynamic]
            if not len(p): continue
            z=totals[name]; z["u"]+=(p[...,:3]-t[...,:3]).square().sum().item(); z["r"]+=(p[...,3:6]-t[...,3:6]).square().sum().item(); z["d"]+=(p[...,6]-t[...,6]).square().sum().item(); z["n"]+=p[...,6].numel()
            pc=torch.exp(-p[...,6].square()/2)>.5; gc=torch.exp(-t[...,6].square()/2)>.5; z["tp"]+=(pc&gc).sum().item(); z["pp"]+=pc.sum().item(); z["gp"]+=gc.sum().item()
    out={}
    for name,z in totals.items():
        precision=z["tp"]/max(z["pp"],1); recall=z["tp"]/max(z["gp"],1)
        out[name]={"dynamic_u_rmse_cm":(z["u"]/(z["n"]*3))**.5,"dynamic_r_rmse_cm":(z["r"]/(z["n"]*3))**.5,"dynamic_d_rmse_cm":(z["d"]/z["n"])**.5,"dynamic_contact_f1":2*precision*recall/max(precision+recall,1e-8)}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="src/task/InteractionDynamics/configs/v17_full.yaml"); p.add_argument("--output",type=Path,required=True); p.add_argument("--max-optimizer-steps",type=int); p.add_argument("--per-gpu-batch",type=int); p.add_argument("--lr",type=float); p.add_argument("--wandb",action="store_true"); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); rank=int(os.getenv("RANK",0)); world=int(os.getenv("WORLD_SIZE",1)); local=int(os.getenv("LOCAL_RANK",0)); distributed=world>1
    if a.per_gpu_batch: cfg["training"]["per_gpu_batch"]=a.per_gpu_batch
    if a.lr: cfg["training"]["lr"]=a.lr
    if distributed: torch.cuda.set_device(local); dist.init_process_group("nccl")
    device=torch.device("cuda",local); seed=cfg["seed"]+rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    paths=dominant_sequences(cfg["data"]["root"],cfg["data"]["dominant_hand_manifest"]); split=split_sequences(paths,cfg["data"]["split_seed"])
    cache=cfg["data"].get("cache_root")
    train=(CachedV17Dataset(cache,"train") if cache and (Path(cache)/"train").exists() else V17Dataset(cfg["data"]["root"],split.train))
    val=(CachedV17Dataset(cache,"val") if cache and (Path(cache)/"val").exists() else V17Dataset(cfg["data"]["root"],split.val))
    test=(CachedV17Dataset(cache,"test") if cache and (Path(cache)/"test").exists() else V17Dataset(cfg["data"]["root"],split.test))
    stats_path=a.output/"train_statistics.pt"
    if rank==0: stats=stats_for(train,stats_path)
    if distributed: dist.barrier(); stats=torch.load(stats_path,map_location="cpu")
    stats={k:v.to(device) for k,v in stats.items()}
    m=cfg["model"]; det=ResidualInteractionRegressor(4,4,m["dim"],m["heads"],m["layers"]).to(device); diffusion=ResidualInteractionDiffusion(4,4,m["dim"],m["heads"],m["layers"]).to(device)
    if distributed: det=DDP(det,device_ids=[local]); diffusion=DDP(diffusion,device_ids=[local])
    parameters=list(det.parameters())+list(diffusion.parameters()); opt=torch.optim.AdamW(parameters,lr=cfg["training"]["lr"],weight_decay=cfg["training"]["weight_decay"])
    sampler=DistributedSampler(train,world,rank,shuffle=True) if distributed else None
    loader=DataLoader(train,batch_size=cfg["training"]["per_gpu_batch"],sampler=sampler,shuffle=sampler is None,num_workers=cfg["training"]["num_workers"],pin_memory=True,persistent_workers=cfg["training"]["num_workers"]>0)
    val_loader=DataLoader(val,batch_size=cfg["training"]["per_gpu_batch"],shuffle=False,num_workers=2,pin_memory=True)
    run=None
    if rank==0 and a.wandb:
        import wandb; run=wandb.init(project=cfg["wandb"]["project"],group=cfg["wandb"]["group"],name=cfg["wandb"]["name"],mode="online",config=cfg)
    _,alpha_bar=cosine_schedule(100,device); global_step=0; best=float("inf"); a.output.mkdir(parents=True,exist_ok=True)
    for epoch in range(cfg["training"]["epochs"]):
        if sampler: sampler.set_epoch(epoch)
        det.train(); diffusion.train(); begin=time.time()
        for batch in loader:
            state=batch["state"].to(device,non_blocking=True); residual=batch["residual"].to(device,non_blocking=True); anchors=batch["anchors_cm"].to(device,non_blocking=True); patches=batch["object_patches"].to(device,non_blocking=True); goal=batch["goal"].to(device,non_blocking=True)
            sn=(state-stats["state_mean"])/stats["state_std"]; rn=(residual-stats["residual_mean"])/stats["residual_std"]; gn=(goal-stats["goal_mean"])/stats["goal_std"]
            t=torch.randint(100,(len(state),),device=device); noise=torch.randn_like(rn); noisy,v=make_v_target(rn,noise,alpha_bar[t][:,None,None]); perm=torch.stack([torch.randperm(128,device=device) for _ in range(len(state))])
            def g(x): return x.gather(1,perm.reshape(*perm.shape,*([1]*(x.ndim-2))).expand_as(x))
            with torch.autocast("cuda",dtype=torch.bfloat16):
                d=det(g(sn),g(anchors),g(patches),gn); pv=diffusion(g(noisy),g(sn),g(anchors),g(patches),gn,t); loss_det=torch.nn.functional.mse_loss(d,g(rn)); loss_v=torch.nn.functional.mse_loss(pv,g(v)); loss=loss_det+loss_v
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.); opt.step(); global_step+=1
            if rank==0 and global_step%20==0:
                row={"train/loss":float(loss),"train/deterministic_mse":float(loss_det),"train/v_mse":float(loss_v),"global_step":global_step,"epoch":epoch+1}; print(json.dumps(row),flush=True); run.log(row,step=global_step) if run else None
            if a.max_optimizer_steps and global_step>=a.max_optimizer_steps: break
        if distributed: dist.barrier()
        if rank==0:
            det_raw=det.module if distributed else det; diff_raw=diffusion.module if distributed else diffusion; det_raw.eval(); diff_raw.eval()
            with torch.no_grad(): metrics=evaluate(det_raw,diff_raw,val_loader,stats,device,cfg["diffusion"]["sampling_steps"])
            score=metrics["diffusion"]["dynamic_r_rmse_cm"]+metrics["diffusion"]["dynamic_d_rmse_cm"]; summary={"epoch":epoch+1,"global_step":global_step,"seconds":time.time()-begin,"val":metrics}; print(json.dumps(summary,ensure_ascii=False),flush=True)
            if run: run.log({f"val/{model}/{key}":value for model,row in metrics.items() for key,value in row.items()},step=global_step)
            checkpoint={"deterministic":det_raw.state_dict(),"diffusion":diff_raw.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch+1,"global_step":global_step,"stats":{k:v.cpu() for k,v in stats.items()},"split":{"train":list(map(str,split.train)),"val":list(map(str,split.val)),"test":list(map(str,split.test))},"config":cfg,"val":metrics}
            torch.save(checkpoint,a.output/"latest.pt")
            if score<best: best=score; torch.save(checkpoint,a.output/"best.pt")
        if distributed: dist.barrier()
        if a.max_optimizer_steps and global_step>=a.max_optimizer_steps: break
    if run: run.finish()
    if distributed: dist.destroy_process_group()
if __name__=="__main__": main()
