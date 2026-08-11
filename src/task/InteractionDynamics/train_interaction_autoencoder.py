"""训练最小 coordinate-conditioned Y→C→Y_hat slot autoencoder。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.InteractionDynamics.eval_interaction_compression import (
    make_dataset, reconstruction_metrics, sequence_disjoint_indices)
from src.task.InteractionDynamics.interaction_autoencoder import InteractionAutoencoder
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.uni3d import deterministic_fps


def materialize(dataset, indices: list[int], device: torch.device,
                tau_m: float) -> dict[str, torch.Tensor]:
    rows = {name: [] for name in ["anchors", "relative_geometry",
                                  "relative_distance", "relative_motion"]}
    for index in indices:
        sample = dataset[index]
        hand = torch.as_tensor(sample["action_hand_points_object_sequence"], device=device)
        obj = torch.as_tensor(sample["world_obj_points_object"], device=device)
        anchors = obj[deterministic_fps(obj[None], 128)[0]]
        y = build_interaction_y(hand, anchors, tau_m)
        rows["anchors"].append((100 * anchors).cpu())
        for name in ["relative_geometry", "relative_distance", "relative_motion"]:
            rows[name].append((100 * y[name]).cpu())
    return {name: torch.stack(values) for name, values in rows.items()}


def model_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(torch.nn.functional.mse_loss(output[name], batch[name])
               for name in ["relative_geometry", "relative_distance", "relative_motion"])


@torch.no_grad()
def evaluate(model: InteractionAutoencoder, data: dict[str, torch.Tensor],
             batch_size: int, device: torch.device) -> dict[str, float]:
    predictions = {name: [] for name in ["relative_geometry", "relative_distance", "relative_motion"]}
    for start in range(0, len(data["anchors"]), batch_size):
        batch = {name: value[start:start + batch_size].to(device) for name, value in data.items()}
        output = model(**{
            "anchors_cm": batch["anchors"],
            "relative_geometry_cm": batch["relative_geometry"],
            "relative_distance_cm": batch["relative_distance"],
            "relative_motion_cm": batch["relative_motion"],
        })
        for name in predictions:
            predictions[name].append(output[name].cpu())
    prediction = {name: torch.cat(value) for name, value in predictions.items()}
    result = {}
    for short, name in [("r", "relative_geometry"), ("d", "relative_distance"),
                        ("u", "relative_motion")]:
        result[f"{short}_rmse_cm"] = float(torch.nn.functional.mse_loss(
            prediction[name], data[name]).sqrt())
    pred_contact = torch.exp(-prediction["relative_distance"].square() / 2) > .5
    gt_contact = torch.exp(-data["relative_distance"].square() / 2) > .5
    tp = (pred_contact & gt_contact).sum().float()
    precision, recall = tp / pred_contact.sum().clamp_min(1), tp / gt_contact.sum().clamp_min(1)
    result["contact_f1"] = float(2 * precision * recall / (precision + recall).clamp_min(1e-8))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=512)
    parser.add_argument("--max-eval", type=int, default=128)
    parser.add_argument("--max-train-per-sequence", type=int, default=32)
    parser.add_argument("--max-eval-per-sequence", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = make_dataset(args.config)
    train_indices, eval_indices = sequence_disjoint_indices(
        dataset, args.seed, args.max_train, args.max_eval,
        args.max_train_per_sequence, args.max_eval_per_sequence)
    train = materialize(dataset, train_indices, device, args.tau_m)
    validation = materialize(dataset, eval_indices, device, args.tau_m)
    model = InteractionAutoencoder(args.slots, args.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train["anchors"]), generator=generator)
        for start in range(0, len(permutation), args.batch_size):
            index = permutation[start:start + args.batch_size]
            batch = {name: value[index].to(device) for name, value in train.items()}
            output = model(batch["anchors"], batch["relative_geometry"],
                           batch["relative_distance"], batch["relative_motion"])
            loss = model_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        row = {"epoch": epoch, **evaluate(model, validation, args.batch_size, device)}
        if epoch == args.epochs:
            row.update({f"train_{name}": value for name, value in
                        evaluate(model, train, args.batch_size, device).items()})
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "slots": args.slots, "dim": args.dim,
                "history": history, "train_indices": train_indices,
                "eval_indices": eval_indices}, args.output)


if __name__ == "__main__":
    main()
