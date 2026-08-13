"""V20.2 Stage B：冻结 V20，只训练 learned MANO-H realizer。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.field_dynamics_v20 import FieldDynamicsTransition
from src.task.InteractionDynamics.field_h_realizer_v20_2 import FieldHRealizerV20_2
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move


def delta_h_std(dataset) -> torch.Tensor:
    values = torch.cat([dataset[i]["future_delta_h"].reshape(-1, 30)
                        for i in range(len(dataset))])
    return values.std(0).clamp_min(1e-3).reshape(1, 1, 30)


def normalized_field_loss(prediction: torch.Tensor, target: torch.Tensor,
                          field_std: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    error = ((prediction - target[..., :7]) / field_std[..., :7]).square()
    terms = {"r": error[..., :3].mean(), "d": error[..., 3].mean(),
             "v": error[..., 4:7].mean()}
    return terms, sum(terms.values())


def load_field(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu"); saved = payload["args"]
    model = FieldDynamicsTransition(dim=saved["dim"], layers=saved["layers"]).to(device)
    model.load_state_dict(payload["model"]); model.requires_grad_(False).eval()
    return model, payload["mean"].to(device), payload["std"].to(device)


@torch.no_grad()
def predict_field(model, batch, mean, std):
    delta = model(batch["current_y"], batch["anchors_cm"], batch["object_patches"])
    return batch["current_y"][:, :, None] + delta * std + mean


def apply_ablation(predicted_y, anchors, patches, current_h, mode):
    if mode == "normal": return predicted_y, anchors, patches, current_h
    if mode == "no_y": return torch.zeros_like(predicted_y), anchors, patches, current_h
    if mode == "shuffle_y":
        order = torch.roll(torch.arange(len(predicted_y), device=predicted_y.device), 1)
        return predicted_y[order], anchors, patches, current_h
    if mode == "no_object": return predicted_y, torch.zeros_like(anchors), torch.zeros_like(patches), current_h
    if mode == "no_current_h": return predicted_y, anchors, patches, torch.zeros_like(current_h)
    raise ValueError(mode)


def error_metrics(prediction, target):
    error = prediction - target[..., :7]; result = {}
    for name, value in (("r", error[..., :3]), ("d", error[..., 3]), ("v", error[..., 4:7])):
        result[name + "_mae_cm"] = float(value.abs().mean())
        result[name + "_rmse_cm"] = float(value.square().mean().sqrt())
    return result


@torch.no_grad()
def evaluate(realizer, field, loader, h_std, field_mean, field_std, layers, device):
    modes = ("normal", "no_y", "shuffle_y", "no_object", "no_current_h")
    errors = {mode: [] for mode in modes}; free = []; gaps = []; normalized = []
    for raw in loader:
        batch = move(raw, device); predicted = predict_field(field, batch, field_mean, field_std)
        free.append(predicted[..., :7] - batch["future_y"][..., :7])
        for mode in modes:
            inputs = apply_ablation(predicted, batch["anchors_cm"], batch["object_patches"],
                                    batch["current_h"], mode)
            z = realizer(*inputs); realized, _ = decode_mano_field(z * h_std, batch, layers)
            errors[mode].append(realized - batch["future_y"][..., :7])
            if mode == "normal":
                gaps.append(realized - predicted[..., :7]); normalized.append(z)
    def aggregate(values): return error_metrics(torch.cat(values), torch.zeros_like(torch.cat(values)))
    z = torch.cat(normalized)
    return {"free_vs_gt": aggregate(free), "realized_vs_gt": {k: aggregate(v) for k, v in errors.items()},
            "realized_vs_free": aggregate(gaps), "normalized_h_abs_max": float(z.abs().max()),
            "normalized_h_over_3_ratio": float((z.abs() > 3).float().mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--field-checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--dim", type=int, default=192)
    p.add_argument("--temporal-layers", type=int, default=2); p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--controlled32", action="store_true"); p.add_argument("--prior-weight", type=float, default=1e-4)
    p.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    p.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR)); args = p.parse_args()
    torch.manual_seed(42); device = torch.device("cuda")
    full = CachedFieldV20Dataset(args.cache, "train")
    train = Subset(full, range(min(32, len(full)))) if args.controlled32 else full
    loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=0)
    validation = DataLoader(train, args.batch_size, shuffle=False, num_workers=0)
    h_std = delta_h_std(train).to(device); field, mean, std = load_field(args.field_checkpoint, device)
    realizer = FieldHRealizerV20_2(dim=args.dim, temporal_layers=args.temporal_layers).to(device)
    optimizer = torch.optim.AdamW(realizer.parameters(), lr=args.lr, weight_decay=1e-4)
    layers = ManoLayers(args.grab_root, args.mano_path, device); iterator = iter(loader)
    args.output.mkdir(parents=True, exist_ok=True); best = float("inf")
    for step in range(1, args.steps + 1):
        try: raw = next(iterator)
        except StopIteration: iterator = iter(loader); raw = next(iterator)
        batch = move(raw, device)
        with torch.no_grad(): predicted = predict_field(field, batch, mean, std)
        z = realizer(predicted, batch["anchors_cm"], batch["object_patches"], batch["current_h"])
        realized, _ = decode_mano_field(z * h_std, batch, layers)
        terms, realize_loss = normalized_field_loss(realized, batch["future_y"], std)
        prior = torch.relu(z.abs() - 3).square().mean(); loss = realize_loss + args.prior_weight * prior
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(realizer.parameters(), 1.)
        optimizer.step()
        if step == 1:
            gradients = [v.grad for v in realizer.parameters() if v.grad is not None]
            print(json.dumps({"gradient_gate": {"finite": all(torch.isfinite(v).all() for v in gradients),
                "nonzero": any(float(v.abs().sum()) > 0 for v in gradients)}}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(realizer.eval(), field, validation, h_std, mean, std, layers, device); realizer.train()
            print(json.dumps({"step": step, "loss": float(loss), "terms": {k: float(v) for k, v in terms.items()},
                              "prior": float(prior), "validation": result}), flush=True)
            payload = {"model": realizer.state_dict(), "h_std": h_std.cpu(), "field_mean": mean.cpu(),
                       "field_std": std.cpu(), "args": vars(args), "step": step, "validation": result}
            torch.save(payload, args.output / "latest.pt")
            score = sum(result["realized_vs_gt"]["normal"][k] for k in
                        ("r_rmse_cm", "d_rmse_cm", "v_rmse_cm"))
            if score < best: best = score; torch.save(payload, args.output / "best.pt")


if __name__ == "__main__": main()
