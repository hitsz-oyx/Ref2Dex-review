from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.base import build_runner_from_checkpoint
from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2
from src.task.correspondence_ptv3_v2.losses import contact_target_from_distance


@dataclass
class ViewerState:
    frame_idx: int = 0
    epoch: int = 0
    selected_rank: int = 0
    show_gt: bool = True
    show_cross: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize correspondence_ptv3_v2 predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _ensure_v2_runner(runner) -> None:
    if getattr(runner.cfg, "name", None) != "correspondence_ptv3_v2":
        raise ValueError("visualize.py only supports correspondence_ptv3_v2 checkpoints.")


def _load_sequence(path: Path, runner) -> CorrStaticDatasetV2:
    meta = runner.cfg.meta
    return CorrStaticDatasetV2(
        path,
        num_obj_points=int(meta.num_obj_points),
        num_hand_points=int(meta.num_hand_points),
        k_cross=int(meta.k_cross),
        k_ctx=int(meta.k_ctx),
        ctx_radius=float(meta.ctx_radius),
        num_supervision_edges=int(meta.num_supervision_edges),
        contact_radius=float(meta.contact_radius),
        base_seed=int(runner.cfg.train.seed),
        augment=False,
        apply_hand_perturb=False,
        eval_sampling_epoch=0,
    )


def _sample(dataset: CorrStaticDatasetV2, frame_idx: int, epoch: int) -> dict[str, torch.Tensor]:
    dataset.set_epoch(epoch)
    return dataset[frame_idx]


def _select_valid_obj(batch: dict[str, torch.Tensor], rank: int) -> int:
    valid_idx = torch.nonzero(batch["runtime_obj_valid_mask"].bool(), as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return 0
    return int(valid_idx[int(rank) % int(valid_idx.numel())].item())


def _dense_cross_prob(runner, batch: dict[str, torch.Tensor], obj_idx: int) -> torch.Tensor:
    model = runner.model
    if model is None:
        raise RuntimeError("Runner model is not initialized.")
    prepared = runner.prepare_batch({key: value.unsqueeze(0) if torch.is_tensor(value) and value.dim() > 0 and key not in {"num_obj_points", "num_hand_points"} else value for key, value in batch.items()})
    with torch.no_grad():
        outputs = model(prepared)
        obj_token = outputs["obj_dense_tokens"][:, obj_idx : obj_idx + 1]
        hand_token = outputs["hand_dense_tokens"]
        edge_input = torch.cat(
            [
                obj_token.unsqueeze(2).expand(-1, -1, hand_token.shape[1], -1),
                hand_token.unsqueeze(1),
            ],
            dim=-1,
        )
        edge_shared = model.edge_shared_backbone(edge_input)
        dense_logits = model.cross_edge_head(edge_shared).squeeze(0).squeeze(-1).squeeze(0)
        return torch.sigmoid(dense_logits).cpu()


def _gt_cross_prob(batch: dict[str, torch.Tensor], obj_idx: int, contact_radius: float) -> torch.Tensor:
    gt_points = batch["gt_points"].float()
    num_obj = int(batch["num_obj_points"])
    obj_point = gt_points[obj_idx]
    hand_points = gt_points[num_obj : num_obj + int(batch["num_hand_points"])]
    distance = torch.norm(hand_points - obj_point.unsqueeze(0), dim=-1)
    return contact_target_from_distance(distance, contact_radius=contact_radius).cpu()


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(
        args.checkpoint,
        mode="eval",
        device=args.device,
        build_data=False,
    )
    _ensure_v2_runner(runner)
    dataset = _load_sequence(Path(args.input), runner)
    state = ViewerState(frame_idx=args.frame, epoch=args.epoch)
    sample = _sample(dataset, state.frame_idx, state.epoch)
    selected_obj = _select_valid_obj(sample, state.selected_rank)
    gt_obj_heat = sample["contact_target"].cpu()
    eval_batch = {key: value.unsqueeze(0) if torch.is_tensor(value) and value.dim() > 0 and key not in {"num_obj_points", "num_hand_points"} else value for key, value in sample.items()}
    with torch.no_grad():
        pred = runner.inference(runner.model, eval_batch)
    eval_obj_heat = pred["pred_obj_contact_prob"].squeeze(0).cpu()
    gt_cross = _gt_cross_prob(sample, selected_obj, float(runner.cfg.meta.contact_radius))
    eval_cross = _dense_cross_prob(runner, sample, selected_obj)
    if args.check_only:
        print(
            {
                "frame": state.frame_idx,
                "epoch": state.epoch,
                "selected_obj": selected_obj,
                "gt_obj_heat_shape": tuple(gt_obj_heat.shape),
                "eval_obj_heat_shape": tuple(eval_obj_heat.shape),
                "gt_cross_shape": tuple(gt_cross.shape),
                "eval_cross_shape": tuple(eval_cross.shape),
            }
        )
        return
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for interactive visualization. Use --check-only for validation.") from exc

    print("A/D or arrows: frame, [/]: epoch, G: GT/Eval, C: heatmap/cross, ,/.: object, R: reset")
    print("Interactive v2 viewer is not fully implemented in this environment. Use --check-only for data/prediction validation.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="correspondence_ptv3_v2")
    vis.destroy_window()


if __name__ == "__main__":
    main()
