"""用冻结的 InteractionDynamics C 监督优化未来 MANO 动作。"""
from __future__ import annotations

import argparse
import json
import os.path as op
from pathlib import Path

import numpy as np
import torch
import trimesh
from smplx import MANO

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABSeqData
from src.base import build_runner_from_checkpoint
from src.base.checkpoint import unwrap_model
from src.base.base_runner import move_to_device


def world_to_frame(points: torch.Tensor, pose_frame_to_world: torch.Tensor) -> torch.Tensor:
    return (points - pose_frame_to_world[:3, 3]) @ pose_frame_to_world[:3, :3]


def face_centers(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    return vertices[:, faces].mean(2)


def c_from_future_points(model: torch.nn.Module, batch: dict[str, torch.Tensor],
                         future_world: torch.Tensor, hand_pose: torch.Tensor,
                         object_pose: torch.Tensor) -> torch.Tensor:
    """Encode candidate future hand points; future object GT never enters model forward."""
    if future_world.ndim == 3:
        future_world = future_world.unsqueeze(0)
    candidate = dict(batch)
    hand_frame = world_to_frame(future_world, hand_pose)
    object_frame = world_to_frame(future_world, object_pose)
    candidate["hand_disp_chunk"] = hand_frame - batch["action_hand_points_hand"]
    candidate["action_hand_disp_chunk_object"] = (
        object_frame - batch["world_hand_points_object"])
    prediction = model(candidate)
    return prediction["se3_interaction_field"].mean(2)


def build_mano(raw_path: Path, raw_root: Path, side: str, mano_path: Path, device: torch.device
               ) -> tuple[MANO, GRABSeqData, dict[str, np.ndarray]]:
    sequence = GRABSeqData(str(raw_path))
    params = sequence.get_hand_params(side)
    vtemp_path = raw_root / sequence.get_hand_vtemp_relpath(side)
    kwargs = {"is_rhand": side == "right", "use_pca": True, "num_pca_comps": 24,
              "flat_hand_mean": True}
    if vtemp_path.exists():
        kwargs["v_template"] = trimesh.load(vtemp_path, process=False).vertices.astype(np.float32)
    layer = MANO(str(mano_path), **kwargs).to(device)
    layer.requires_grad_(False)
    return layer, sequence, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    default_raw_root = Path(DEFAULT_GRAB_ROOT) / "data"
    parser.add_argument("--grab-root", type=Path, default=default_raw_root)
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("output/InteractionDynamics/mano_inverse/result.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(args.checkpoint, config=args.config,
                                          mode="eval", device=args.device)
    loader = runner.test_loader if args.split == "test" else runner.val_loader
    if loader is None:
        raise RuntimeError(f"No {args.split} loader")
    dataset = loader.dataset
    hand_cache_path, current = dataset.sample_location(args.sample_index)
    item = dataset[args.sample_index]
    device = runner.device
    batch = move_to_device({key: value.unsqueeze(0) for key, value in item.items()}, device)
    model = unwrap_model(runner.model).eval()
    model.requires_grad_(False)

    with np.load(hand_cache_path.parent / "shared.npz", allow_pickle=False) as shared:
        source_raw_file = str(shared["source_raw_file"].item())
        raw_frame_ids = np.asarray(shared["raw_frame_id"], dtype=np.int64)
        object_pose_np = np.asarray(shared["obj_root_pose_world"][current], dtype=np.float32)
    with np.load(hand_cache_path, allow_pickle=False) as hand_cache:
        side = str(hand_cache["side"].item())
        hand_pose_np = np.asarray(hand_cache["hand_root_pose_world"][current], dtype=np.float32)
    raw_path = args.grab_root / source_raw_file
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到 cache 对应的 GRAB raw sequence: {raw_path}")
    mano, sequence, params = build_mano(raw_path, args.grab_root, side, args.mano_path, device)
    del sequence
    future_cache_ids = np.arange(current + 1, current + 9)
    raw_ids = raw_frame_ids[np.concatenate([[current], future_cache_ids])]
    current_raw, future_raw = int(raw_ids[0]), raw_ids[1:]

    def parameter(name: str, width: int) -> torch.Tensor:
        value = np.asarray(params[name][current_raw], dtype=np.float32).reshape(1, width)
        return torch.nn.Parameter(torch.from_numpy(value).to(device).expand(8, -1).clone())

    global_orient = parameter("global_orient", 3)
    hand_pose = parameter("hand_pose", 24)
    translation = parameter("transl", 3)
    betas_np = np.asarray(params["betas"], dtype=np.float32)
    if betas_np.ndim > 1:
        betas_np = betas_np[current_raw]
    betas = torch.from_numpy(betas_np).to(device)[None].expand(8, -1)
    faces = torch.from_numpy(np.asarray(mano.faces, dtype=np.int64)).to(device)
    hand_root_pose = torch.from_numpy(hand_pose_np).to(device)
    object_root_pose = torch.from_numpy(object_pose_np).to(device)

    with torch.no_grad():
        gt_future_hand = batch["action_hand_points_hand"] + batch["hand_disp_chunk"]
        gt_future_world = (gt_future_hand @ hand_root_pose[:3, :3].T
                           + hand_root_pose[:3, 3])
        teacher_c = c_from_future_points(model, batch, gt_future_world,
                                         hand_root_pose, object_root_pose).detach()

    optimizer = torch.optim.Adam([global_orient, hand_pose, translation], lr=args.lr)
    history = []
    initial_points = None
    for step in range(args.steps + 1):
        output = mano(global_orient=global_orient, hand_pose=hand_pose,
                      betas=betas, transl=translation)
        candidate_world = face_centers(output.vertices, faces)
        candidate_c = c_from_future_points(model, batch, candidate_world,
                                           hand_root_pose, object_root_pose)
        c_loss = torch.nn.functional.mse_loss(candidate_c, teacher_c)
        pose_sequence = torch.cat([hand_pose[:1].detach(), hand_pose], 0)
        orient_sequence = torch.cat([global_orient[:1].detach(), global_orient], 0)
        trans_sequence = torch.cat([translation[:1].detach(), translation], 0)
        smooth = (torch.diff(pose_sequence, dim=0).square().mean()
                  + torch.diff(orient_sequence, dim=0).square().mean()
                  + (100.0 * torch.diff(trans_sequence, dim=0)).square().mean())
        loss = c_loss + args.smooth_weight * smooth
        candidate_hand = world_to_frame(candidate_world, hand_root_pose)
        error_mm = (candidate_hand - gt_future_hand.squeeze(0)).norm(dim=-1) * 1000.0
        if step == 0:
            initial_points = candidate_world.detach().clone()
        if step % 20 == 0 or step == args.steps:
            row = {"step": step, "loss": float(loss.detach()),
                   "c_rmse": float(c_loss.detach().sqrt()),
                   "hand_ade_mm": float(error_mm.mean()),
                   "hand_fde_mm": float(error_mm[-1].mean())}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert initial_points is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_raw_file=np.asarray(source_raw_file), side=np.asarray(side),
        current_raw_frame=np.asarray(current_raw), future_raw_frames=future_raw,
        global_orient=global_orient.detach().cpu().numpy(),
        hand_pose=hand_pose.detach().cpu().numpy(), translation=translation.detach().cpu().numpy(),
        initial_points_world=initial_points.cpu().numpy(),
        optimized_points_world=candidate_world.detach().cpu().numpy(),
        gt_points_world=gt_future_world.squeeze(0).cpu().numpy(),
        history_json=np.asarray(json.dumps(history, ensure_ascii=False)),
    )
    print(f"已保存 {args.output}")


if __name__ == "__main__":
    main()
