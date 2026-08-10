"""冻结 Action Encoder，以线性探针诊断 global/local hand action 信息。"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from src.base import build_runner_from_checkpoint
from src.base.base_runner import move_to_device
from src.base.checkpoint import unwrap_model
from src.task.InteractionDynamics.runner import patch_motion_target


@dataclass
class ProbeData:
    tokens: torch.Tensor
    full: torch.Tensor
    global_motion: torch.Tensor
    local: torch.Tensor


def decompose_patch_motion(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N,T,P,3] -> global [N,T,3], zero-mean local [N,T,P,3]."""
    global_motion = target.mean(2)
    return global_motion, target - global_motion[:, :, None]


@torch.no_grad()
def extract(model: torch.nn.Module, loader, device: torch.device,
            motion_scale: float) -> ProbeData:
    token_rows, full_rows = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        prediction = model(batch)
        target = patch_motion_target(batch["hand_disp_chunk"].float(),
                                     prediction["hand_knn_idx"], motion_scale)
        token_rows.append(prediction["action_context_tokens"].float().cpu())
        full_rows.append(target.cpu())
    tokens, full = torch.cat(token_rows), torch.cat(full_rows)
    global_motion, local = decompose_patch_motion(full)
    return ProbeData(tokens=tokens, full=full, global_motion=global_motion, local=local)


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Return affine ridge weights [D+1,O], leaving the intercept unregularized."""
    x = x.float()
    y = y.float()
    design = torch.cat([x, torch.ones(x.shape[0], 1)], 1)
    gram = design.T @ design
    regularizer = torch.eye(gram.shape[0], dtype=gram.dtype) * ridge
    regularizer[-1, -1] = 0
    return torch.linalg.solve(gram + regularizer, design.T @ y)


def predict_ridge(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    design = torch.cat([x.float(), torch.ones(x.shape[0], 1)], 1)
    return design @ weight


def probe_shapes(data: ProbeData, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    if target == "global":
        return data.tokens.mean(1), data.global_motion.flatten(1)
    y = data.full if target == "full" else data.local
    return data.tokens.flatten(0, 1), y.permute(0, 2, 1, 3).flatten(0, 1).flatten(1)


def evaluate_probe(data: ProbeData, target: str, weight: torch.Tensor) -> dict[str, float]:
    x, y = probe_shapes(data, target)
    prediction = predict_ridge(x, weight)
    mse = (prediction - y).square().mean()
    zero_mse = y.square().mean()
    return {"rmse_cm": float(mse.sqrt()), "zero_rmse_cm": float(zero_mse.sqrt()),
            "relative_improvement": float(1.0 - mse.sqrt() / zero_mse.sqrt())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ridge-grid", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(args.checkpoint, config=args.config,
                                          mode="eval", device=args.device)
    model = unwrap_model(runner.model).eval()
    model.requires_grad_(False)
    loaders = {"train": runner.train_loader, "val": runner.val_loader, "test": runner.test_loader}
    if any(loader is None for loader in loaders.values()):
        raise RuntimeError("Action probe requires train/val/test loaders")
    data = {name: extract(model, loader, runner.device, float(runner.cfg.meta.motion_scale))
            for name, loader in loaders.items()}
    train = data["train"]
    centered = train.tokens - train.tokens.mean(1, keepdim=True)
    normalized = torch.nn.functional.normalize(train.tokens, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    off_diagonal = ~torch.eye(train.tokens.shape[1], dtype=torch.bool)
    result: dict[str, object] = {
        "token_spatial_variance": float(centered.square().mean()),
        "token_pairwise_cosine": float(similarity[:, off_diagonal].mean()),
    }
    ridge_grid = [float(value) for value in args.ridge_grid.split(",")]
    for target in ("full", "global", "local"):
        train_x, train_y = probe_shapes(train, target)
        candidates = []
        for ridge in ridge_grid:
            weight = fit_ridge(train_x, train_y, ridge)
            val_metrics = evaluate_probe(data["val"], target, weight)
            candidates.append((val_metrics["rmse_cm"], ridge, weight, val_metrics))
        _, best_ridge, weight, val_metrics = min(candidates, key=lambda row: row[0])
        result[target] = {"selected_ridge": best_ridge,
                          "train": evaluate_probe(data["train"], target, weight),
                          "val": val_metrics,
                          "test": evaluate_probe(data["test"], target, weight)}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
