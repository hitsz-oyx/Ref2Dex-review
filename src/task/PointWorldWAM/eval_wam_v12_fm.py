from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset_wam_chunk import build_wam_chunk_dataset
from .train import move_batch
from .train_wam_v2 import sample_forward, sample_inverse
from .wam_v2 import ChunkJointWAM


@torch.no_grad()
def evaluate_mode(model, loader, device, mode, steps, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    values = {}
    for batch in loader:
        batch = move_batch(batch, device)
        target = model.normalized_targets(batch)
        if mode == "inverse":
            initial = {
                side: torch.randn(
                    target[side].shape,
                    device=device,
                    dtype=target[side].dtype,
                    generator=generator,
                )
                for side in ("left", "right")
            }
            generated = sample_inverse(
                model, batch, initial["left"], initial["right"], steps
            )
            generated_error, identity_error = [], []
            for side in ("left", "right"):
                points = model.hand_points_from_normalized_action(
                    batch, side, generated[side]
                )
                generated_error.append(
                    torch.linalg.vector_norm(
                        points - batch[f"{side}_hand_chunk"], dim=-1
                    ).mean(dim=(-1, -2))
                )
                identity_error.append(
                    torch.linalg.vector_norm(
                        batch[f"{side}_hand_points"][:, None]
                        - batch[f"{side}_hand_chunk"],
                        dim=-1,
                    ).mean(dim=(-1, -2))
                )
            values.setdefault("inverse_hand_mm", []).append(
                500.0 * sum(generated_error)
            )
            values.setdefault("identity_hand_mm", []).append(
                500.0 * sum(identity_error)
            )
        else:
            initial = torch.randn(
                target["world"].shape,
                device=device,
                dtype=target["world"].dtype,
                generator=generator,
            )
            generated = sample_forward(model, batch, target, initial, steps)
            zero_action = {
                side: model.normalize_action(
                    side, torch.zeros_like(batch[f"{side}_action_chunk"])
                )
                for side in ("left", "right")
            }
            without_action = sample_forward(
                model, batch, zero_action, initial.clone(), steps
            )
            for key, world in (
                ("forward_world_mm", generated),
                ("zero_action_world_mm", without_action),
            ):
                values.setdefault(key, []).append(
                    1000.0
                    * torch.linalg.vector_norm(
                        model.denormalize_world(world) - batch["world_chunk"], dim=-1
                    ).mean(dim=(-1, -2))
                )
            values.setdefault("zero_flow_world_mm", []).append(
                1000.0
                * torch.linalg.vector_norm(batch["world_chunk"], dim=-1).mean(
                    dim=(-1, -2)
                )
            )
    result = {key: float(torch.cat(parts).mean()) for key, parts in values.items()}
    result["finite"] = all(torch.isfinite(torch.cat(parts)).all().item() for parts in values.values())
    if mode == "inverse":
        result["passes_gate"] = result["inverse_hand_mm"] < result["identity_hand_mm"]
    else:
        result["uses_action"] = result["forward_world_mm"] < result["zero_action_world_mm"]
        result["passes_gate"] = (
            result["forward_world_mm"] < result["zero_flow_world_mm"]
            and result["uses_action"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="复算 V1.2 单任务 FM 门禁")
    parser.add_argument("--mode", choices=("inverse", "forward"), required=True)
    parser.add_argument(
        "--config", default="src/task/PointWorldWAM/configs/grab_wam_v12_fm.yaml"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.train.device)
    dataset = build_wam_chunk_dataset(cfg.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint_path = Path(cfg.train.output_dir) / args.mode / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ChunkJointWAM(cfg.model, dataset.statistics(), dataset.subjects).to(device)
    model.load_state_dict(checkpoint["model"])
    result = {
        "mode": args.mode,
        "step": int(checkpoint["step"]),
        "windows": len(dataset),
        "evaluation": evaluate_mode(
            model,
            loader,
            device,
            args.mode,
            cfg.flow_matching.inference_steps,
            cfg.seed + 100000,
        ),
    }
    output = Path("output/research/pointworld_wam/wam_v12")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.mode}_fm_eval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
