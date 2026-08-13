"""V20.4：冻结 Free-Y/Direct-H 后诊断 channel 选择与连续融合。"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.direct_h_dynamics_v20_3 import DirectHDynamicsV20_3
from src.task.InteractionDynamics.eval_direct_h_v20_3 import behavior_metrics
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_h_realizer_v20_2 import error_metrics, load_field, predict_field
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move


CHANNELS = {"r": slice(0, 3), "d": slice(3, 4), "v": slice(4, 7)}


def fuse_channels(y_free: torch.Tensor, y_h: torch.Tensor, code: str) -> torch.Tensor:
    """按 Y/H 三字符编码融合 r/d/v；p 始终保留 Free-Y。"""
    if len(code) != 3 or any(source not in "YH" for source in code):
        raise ValueError(f"非法 fusion code: {code}")
    result = y_free.clone()
    for source, channel in zip(code, CHANNELS.values()):
        if source == "H":
            result[..., channel] = y_h[..., channel]
    return result


def interpolate_velocity(y_free: torch.Tensor, y_h: torch.Tensor, alpha: float) -> torch.Tensor:
    result = y_free.clone()
    result[..., 4:7] = (1 - alpha) * y_free[..., 4:7] + alpha * y_h[..., 4:7]
    return result


def sample_channel_rmse(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: (prediction[..., index] - target[..., index]).square().flatten(1).mean(1).sqrt()
            for name, index in CHANNELS.items()}


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double() - a.double().mean(); b = b.double() - b.double().mean()
    denominator = a.square().sum().sqrt() * b.square().sum().sqrt()
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def complementarity(y_free: torch.Tensor, y_h: torch.Tensor, target: torch.Tensor) -> dict:
    free = sample_channel_rmse(y_free, target); hand = sample_channel_rmse(y_h, target)
    result = {}
    for name in CHANNELS:
        oracle = torch.minimum(free[name], hand[name])
        best_endpoint = min(float(free[name].mean()), float(hand[name].mean()))
        result[name] = {
            "h_win_rate": float((hand[name] < free[name]).float().mean()),
            "error_correlation": correlation(free[name], hand[name]),
            "free_sample_rmse_mean_cm": float(free[name].mean()),
            "h_sample_rmse_mean_cm": float(hand[name].mean()),
            "oracle_sample_rmse_mean_cm": float(oracle.mean()),
            "oracle_improvement_vs_best_endpoint": (
                (best_endpoint - float(oracle.mean())) / best_endpoint if best_endpoint > 0 else 0.0
            ),
        }
    return result


def evaluate_prediction(prediction: torch.Tensor, target: torch.Tensor,
                        current_y: torch.Tensor) -> dict:
    return {"error": error_metrics(prediction[..., :7], target),
            "behavior": behavior_metrics(prediction[..., :7], target, current_y)}


def load_direct(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu"); saved = payload["args"]
    model = DirectHDynamicsV20_3(dim=saved["dim"], temporal_layers=saved["temporal_layers"]).to(device)
    model.load_state_dict(payload["model"]); model.requires_grad_(False).eval()
    return model, payload["h_std"].to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--field-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-h-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    args = parser.parse_args(); device = torch.device("cuda")
    field, mean, std = load_field(args.field_checkpoint, device)
    direct, h_std = load_direct(args.direct_h_checkpoint, device)
    layers = ManoLayers(args.grab_root, args.mano_path, device)
    loader = DataLoader(CachedFieldV20Dataset(args.cache, args.split), args.batch_size,
                        shuffle=False, num_workers=0)
    free_rows, hand_rows, target_rows, current_rows = [], [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            y_free = predict_field(field, batch, mean, std)
            delta_h = direct(batch["current_y"], batch["anchors_cm"],
                             batch["object_patches"], batch["current_h"]) * h_std
            y_h, _ = decode_mano_field(delta_h, batch, layers)
            free_rows.append(y_free.cpu()); hand_rows.append(y_h.cpu())
            target_rows.append(batch["future_y"][..., :7].cpu())
            current_rows.append(batch["current_y"].cpu())
    y_free = torch.cat(free_rows); y_h = torch.cat(hand_rows)
    target = torch.cat(target_rows); current = torch.cat(current_rows)
    combinations = {}
    for sources in itertools.product("YH", repeat=3):
        code = "".join(sources)
        combinations[code] = evaluate_prediction(fuse_channels(y_free, y_h, code), target, current)
    interpolation = {}
    for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        interpolation[str(alpha)] = evaluate_prediction(
            interpolate_velocity(y_free, y_h, alpha), target, current)
    result = {"split": args.split, "samples": len(target), "combinations": combinations,
              "velocity_interpolation": interpolation,
              "complementarity": complementarity(y_free[..., :7], y_h, target)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
