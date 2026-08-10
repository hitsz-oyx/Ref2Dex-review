"""V1.1：定位 dense flow 误差来自 PoseToken 还是 Action/temporal readout。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from src.task.Actiontoken.config import Config
from src.task.Actiontoken.generator import SyntheticManoActionDataset
from src.task.Actiontoken.model import ActionTokenModel
from src.task.InteractionDynamics.uni3d import gather_points
from src.task.Posetoken.model import StaticPoseEncoder


class OddProbe(nn.Module):
    """共享 forward/reverse 权重并取奇对称分量。"""
    def __init__(self, input_dim: int, hidden: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.GELU()])
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, forward: torch.Tensor, reverse: torch.Tensor) -> torch.Tensor:
        return .5 * (self.network(forward) - self.network(reverse))


def pose_meta(meta):
    return SimpleNamespace(model_dim=meta.model_dim, global_dim=meta.global_dim,
                           attention_heads=meta.attention_heads, patch_size=meta.patch_size,
                           num_patches=meta.num_patches, num_hand_points=meta.num_hand_points)


def make_batch(device: torch.device):
    cfg = Config()
    meta, data = cfg.meta, cfg.data
    dataset = SyntheticManoActionDataset(
        mano_path=meta.mano_path, side=meta.side, num_samples=32, seed=42,
        num_pca_comps=meta.num_pca_comps, num_patches=meta.num_patches,
        patch_size=meta.patch_size, chunk_len=meta.chunk_len, generation_batch_size=128,
        velocity_decay=data.velocity_decay, velocity_std=data.velocity_std)
    batch = {key: torch.stack([dataset[index][key] for index in range(len(dataset))]).to(device)
             for key in ("hand_points_root_sequence", "hand_cano_points", "patch_knn_idx")}
    sequence = batch["hand_points_root_sequence"]
    patches = torch.stack([gather_points(sequence[:, frame], batch["patch_knn_idx"])
                           for frame in range(sequence.shape[1])], 1)
    target = (patches[:, 1:] - patches[:, :-1]) * 100.0
    model = ActionTokenModel(SimpleNamespace(meta=meta)).to(device).eval()
    with torch.no_grad():
        local, global_pose = model.encode_pose_sequence(batch)
    return cfg, batch, patches, target, local, global_pose


def train_probe(mode: str, patches: torch.Tensor, target: torch.Tensor,
                local: torch.Tensor, global_pose: torch.Tensor, steps: int, lr: float):
    device = target.device
    if mode == "no_temporal":
        before, after = local[:, :-1], local[:, 1:]
        gb = global_pose[:, :-1, None].expand(-1, -1, local.shape[2], -1)
        ga = global_pose[:, 1:, None].expand_as(gb)
        forward = torch.cat([before, after, after - before, gb, ga], -1)
        reverse = torch.cat([after, before, before - after, ga, gb], -1)
        probe = OddProbe(forward.shape[-1], (768, 384), 32 * 3).to(device)
    elif mode == "pose_pair":
        before, after = local[:, :-1], local[:, 1:]
        forward, reverse = torch.cat([before, after], -1), torch.cat([after, before], -1)
        probe = OddProbe(forward.shape[-1], (1024, 1024), 32 * 3).to(device)
    elif mode == "surface_pair":
        before, after = patches[:, :-1], patches[:, 1:]
        forward = torch.cat([before, after], -1).flatten(-2)
        reverse = torch.cat([after, before], -1).flatten(-2)
        probe = OddProbe(forward.shape[-1], (1024, 1024), 32 * 3).to(device)
    else:
        raise ValueError(mode)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(42)
    for step in range(steps):
        index = torch.randint(0, target.shape[0], (8,), generator=generator, device=device)
        prediction = probe(forward[index], reverse[index]).reshape_as(target[index])
        loss = torch.nn.functional.mse_loss(prediction, target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = probe(forward, reverse).reshape_as(target)
    return statistics(prediction, target)


def pose_reconstruction_difference(cfg, batch, target, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder = StaticPoseEncoder(SimpleNamespace(meta=pose_meta(cfg.meta))).to(target.device)
    encoder.load_state_dict(checkpoint["model"])
    encoder.eval()
    points = batch["hand_points_root_sequence"]
    batch_size, frames, count, _ = points.shape
    flat = {"hand_points_root": points.reshape(batch_size * frames, count, 3),
            "hand_cano_points": batch["hand_cano_points"][:, None].expand(
                -1, frames, -1, -1).reshape(batch_size * frames, count, 3),
            "patch_knn_idx": batch["patch_knn_idx"][:, None].expand(
                -1, frames, -1, -1).reshape(batch_size * frames, 64, 32)}
    with torch.no_grad():
        deformation = encoder(flat)["pred_patch_deformation_internal"].reshape(
            batch_size, frames, 64, 32, 3)
    return statistics(deformation[:, 1:] - deformation[:, :-1], target)


def statistics(prediction, target):
    residual = prediction - target
    rmse = residual.square().mean().sqrt()
    zero = target.square().mean().sqrt()
    result = {"rmse_cm": rmse.item(), "zero_rmse_cm": zero.item(),
              "improvement": (1 - rmse / zero).item(),
              "epe_mm": (residual / 100.0).norm(dim=-1).mean().mul(1000).item()}
    speed = target.norm(dim=-1)
    thresholds = torch.quantile(speed.flatten(), torch.tensor([1 / 3, 2 / 3], device=speed.device))
    for name, mask in (("slow", speed <= thresholds[0]),
                       ("medium", (speed > thresholds[0]) & (speed <= thresholds[1])),
                       ("fast", speed > thresholds[1])):
        result[f"{name}_rmse_cm"] = residual.square().sum(-1)[mask].mean().sqrt().item()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no_temporal", "pose_pair", "surface_pair", "pose_diff"),
                        required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pose-checkpoint", type=Path,
        default=Path("outputs/posetoken/pose_token_20260810_205909/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)
    cfg, batch, patches, target, local, global_pose = make_batch(device)
    result = (pose_reconstruction_difference(cfg, batch, target, args.pose_checkpoint)
              if args.mode == "pose_diff"
              else train_probe(args.mode, patches, target, local, global_pose, args.steps, args.lr))
    payload = {"mode": args.mode, "steps": 0 if args.mode == "pose_diff" else args.steps, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
