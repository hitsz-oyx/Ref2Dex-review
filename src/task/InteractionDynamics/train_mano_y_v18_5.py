"""V18.5 仅用 Y residual 监督 MANO-H structured decoder。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.inverse_mano import build_mano
from src.task.InteractionDynamics.mano_hand_transition import CachedManoHDataset, ManoHandTransition
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import decode_mano_y
from src.task.InteractionDynamics.eval_grasp_v18 import trajectory_statistics
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future
from src.task.InteractionDynamics.train_mano_h_v18_4 import evenly_spaced, statistics


class ManoLayers:
    def __init__(self, grab_root: Path, mano_path: Path, device: torch.device):
        self.grab_root, self.mano_path, self.device = grab_root, mano_path, device; self.layers = {}
    def __call__(self, key: str, side: str, source: str):
        if key not in self.layers:
            layer, _, _ = build_mano(self.grab_root / source, self.grab_root,
                                     side, self.mano_path, self.device)
            self.layers[key] = layer.requires_grad_(False)
        return self.layers[key]


def residual_std(dataset) -> torch.Tensor:
    values = torch.cat([dataset[i]["residual"] for i in range(len(dataset))], 0).double()
    return values.std(0).float().clamp_min(.05).reshape(1, 1, -1)


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def y_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = (prediction - target).reshape(-1, 8, 7)
    values = {"u_rmse_cm": error[..., :3], "r_rmse_cm": error[..., 3:6], "d_rmse_cm": error[..., 6]}
    return {key: float(value.square().mean().sqrt()) for key, value in values.items()}


@torch.no_grad()
def evaluate(model, loader, delta_stats, r_std, layers, device, max_batches=None):
    squares = torch.zeros(3, device=device); elements = torch.zeros(3, device=device); losses=[]; pred_contact=[]; gt_contact=[]
    for number, raw in enumerate(loader):
        if max_batches is not None and number >= max_batches: break
        batch = move(raw, device); normalized = model(batch["state"], batch["anchors_cm"],
            batch["object_patches"], batch["current_h"])
        delta = normalized * delta_stats["delta_std"] + delta_stats["delta_mean"]
        future, _ = decode_mano_y(delta, batch, layers)
        pred_residual = future - persistence_future(batch["state"], 8)
        losses.append(float(((pred_residual - batch["residual"]) / r_std).square().mean()))
        error = (future - batch["future"]).reshape(-1, 8, 7)
        pred_contact.append(trajectory_statistics(future)["contact"])
        gt_contact.append(trajectory_statistics(batch["future"])["contact"])
        for i, value in enumerate((error[..., :3], error[..., 3:6], error[..., 6])):
            squares[i] += value.square().sum(); elements[i] += value.numel()
    rmse = (squares / elements).sqrt()
    pred_mean=float(torch.cat(pred_contact).mean()); gt_mean=float(torch.cat(gt_contact).mean())
    return {"loss_y": sum(losses)/len(losses), "u_rmse_cm": float(rmse[0]),
            "r_rmse_cm": float(rmse[1]), "d_rmse_cm": float(rmse[2]),
            "terminal_contact_mean":pred_mean,"gt_terminal_contact_mean":gt_mean,
            "contact_mean_abs_error":abs(pred_mean-gt_mean)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--controlled32",action="store_true"); parser.add_argument("--train-samples",type=int)
    parser.add_argument("--val-samples",type=int); parser.add_argument("--steps",type=int,default=1500)
    parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--lr",type=float,default=2e-4)
    parser.add_argument("--dim",type=int,default=256); parser.add_argument("--layers",type=int,default=6)
    parser.add_argument("--prior-weight",type=float,default=0.); parser.add_argument("--eval-every",type=int,default=250)
    parser.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    parser.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR))
    args=parser.parse_args(); device=torch.device("cuda"); torch.manual_seed(42)
    full=CachedManoHDataset(args.cache,"train")
    train=Subset(full,range(min(32,len(full)))) if args.controlled32 else evenly_spaced(full,args.train_samples)
    val=train if args.controlled32 else evenly_spaced(CachedManoHDataset(args.cache,"val"),args.val_samples)
    delta_stats={k:v.to(device) for k,v in statistics(train).items()}; r_std=residual_std(train).to(device)
    model=ManoHandTransition(dim=args.dim,layers=args.layers).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    layers=ManoLayers(args.grab_root,args.mano_path,device)
    loader=DataLoader(train,args.batch_size,shuffle=True,num_workers=0); val_loader=DataLoader(val,args.batch_size,shuffle=False,num_workers=0)
    iterator=iter(loader); args.output.mkdir(parents=True,exist_ok=True); best=float("inf")
    for step in range(1,args.steps+1):
        try: raw=next(iterator)
        except StopIteration: iterator=iter(loader); raw=next(iterator)
        batch=move(raw,device); normalized=model(batch["state"],batch["anchors_cm"],batch["object_patches"],batch["current_h"])
        delta=normalized*delta_stats["delta_std"]+delta_stats["delta_mean"]
        future,_=decode_mano_y(delta,batch,layers); pred_residual=future-persistence_future(batch["state"],8)
        loss_y=((pred_residual-batch["residual"])/r_std).square().mean()
        prior=torch.relu(normalized.abs()-3).square().mean(); loss=loss_y+args.prior_weight*prior
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if step==1:
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            print(json.dumps({"gradient_gate":{"finite":all(torch.isfinite(g).all() for g in grads),"nonzero":any(float(g.abs().sum())>0 for g in grads)}}),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            metrics=evaluate(model.eval(),val_loader,delta_stats,r_std,layers,device); model.train()
            row={"step":step,"train_loss_y":float(loss_y),"prior":float(prior),"normalized_abs_max":float(normalized.abs().max()),"val":metrics}; print(json.dumps(row),flush=True)
            payload={"model":model.state_dict(),"delta_stats":{k:v.cpu() for k,v in delta_stats.items()},"residual_std":r_std.cpu(),"args":vars(args),"step":step,"val":metrics}
            torch.save(payload,args.output/"latest.pt"); score=(metrics["r_rmse_cm"]+metrics["d_rmse_cm"]+metrics["u_rmse_cm"]
                +.02*metrics["contact_mean_abs_error"])
            if score<best: best=score; torch.save(payload,args.output/"best.pt")


if __name__=="__main__": main()
