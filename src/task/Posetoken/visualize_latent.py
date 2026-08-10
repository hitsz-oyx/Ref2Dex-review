"""导出 global PoseToken 的二维 PCA，用于检查 synthetic pose latent。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.base import build_runner_from_checkpoint
from src.base.checkpoint import unwrap_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("output/Posetoken/research/global_pose_pca.npz"))
    args = parser.parse_args()
    runner = build_runner_from_checkpoint(
        args.checkpoint, config=args.config, mode="eval", device=args.device)
    loader = (runner.val_loaders if args.split == "val" else runner.test_loaders)[
        f"{args.split}/"]
    model = unwrap_model(runner.model)
    latents, poses, betas, groups, ids = [], [], [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = runner.prepare_batch(raw)
            latents.append(model(batch)["global_pose_token"].float().cpu())
            poses.append(raw["mano_pose"].float())
            betas.append(raw["mano_beta"].float())
            groups.append(raw["pose_group_id"].long())
            ids.append(raw["sample_id"].long())
            if sum(len(value) for value in latents) >= args.samples:
                break
    latent = torch.cat(latents)[:args.samples]
    centered = latent - latent.mean(0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=2)
    pca = centered @ components[:, :2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, pca=pca.numpy(), global_pose_token=latent.numpy(),
                        mano_pose=torch.cat(poses)[:args.samples].numpy(),
                        mano_beta=torch.cat(betas)[:args.samples].numpy(),
                        pose_group_id=torch.cat(groups)[:args.samples].numpy(),
                        sample_id=torch.cat(ids)[:args.samples].numpy())
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6, 5))
    color = torch.cat(poses)[:args.samples, 0].numpy()
    scatter = axis.scatter(pca[:, 0], pca[:, 1], c=color, s=12, cmap="coolwarm")
    axis.set(xlabel="PoseToken PCA-1", ylabel="PoseToken PCA-2",
             title=f"Synthetic MANO global PoseToken ({args.split})")
    figure.colorbar(scatter, ax=axis, label="MANO PCA pose dim 0")
    figure.tight_layout()
    plot_path = args.output.with_suffix(".png")
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    print(f"导出 {len(latent)} 个 latent 到 {args.output} 和 {plot_path}")


if __name__ == "__main__":
    main()
