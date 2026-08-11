"""用 V16.1 解析 interaction field 或 ActionToken 监督执行 MANO self-inverse。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.base import load_config
from src.task.Actiontoken.model import FlowActionEncoder
from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.inverse_mano import build_mano, face_centers
from src.task.InteractionDynamics.uni3d import deterministic_fps, gather_points


TIP_VERTEX_INDEX = torch.tensor([744, 320, 443, 555, 672])


def points_to_frames(points_world: torch.Tensor, poses_frame_to_world: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "tni,tij->tnj", points_world - poses_frame_to_world[:, None, :3, 3],
        poses_frame_to_world[:, :3, :3])


def root_feature(global_orient: torch.Tensor, wrist_world: torch.Tensor,
                 object_poses: torch.Tensor) -> torch.Tensor:
    object_rotation = object_poses[:, :3, :3]
    relative_rotation = object_rotation.transpose(-1, -2) @ axis_angle_to_matrix(global_orient)
    relative_translation = torch.einsum(
        "ti,tij->tj", wrist_world - object_poses[:, :3, 3], object_rotation)
    return torch.cat([100.0 * relative_translation, matrix_to_axis_angle(relative_rotation)], -1)


def load_action_encoder(checkpoint: str, dim: int, device: torch.device) -> FlowActionEncoder:
    payload = torch.load(checkpoint, map_location="cpu")
    encoder = FlowActionEncoder(dim, 32).to(device)
    state = payload.get("action_encoder")
    if state is None:
        state = {key[len("flow_action.encoder."):]: value
                 for key, value in payload.get("model", {}).items()
                 if key.startswith("flow_action.encoder.")}
    encoder.encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).eval()
    return encoder


def action_tokens(encoder: FlowActionEncoder, hand_local: torch.Tensor,
                  patch_knn_idx: torch.Tensor) -> torch.Tensor:
    flow = torch.diff(hand_local, dim=0) * 100.0
    patch_flow = torch.stack([gather_points(flow[step:step + 1], patch_knn_idx)[0]
                              for step in range(flow.shape[0])])
    return encoder.encode(patch_flow)


def binary_contact_metrics(pred_r: torch.Tensor, gt_r: torch.Tensor,
                           sigma_m: float = .01) -> dict[str, float]:
    pred = torch.exp(-pred_r.square().sum(-1) / (2 * sigma_m ** 2)) > .5
    target = torch.exp(-gt_r.square().sum(-1) / (2 * sigma_m ** 2)) > .5
    tp = (pred & target).sum().float()
    precision = tp / pred.sum().clamp_min(1)
    recall = tp / target.sum().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {"contact_precision": float(precision), "contact_recall": float(recall),
            "contact_f1": float(f1),
            "contact_pred_active_fraction": float(pred.float().mean()),
            "contact_target_active_fraction": float(target.float().mean())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v16_object_contact_diverse_overfit.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target", choices=["r", "u", "interaction", "full", "action"],
                        default="interaction")
    parser.add_argument("--init", choices=["noise", "repeat"], default="noise")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate-beta-offset", type=float, default=0.0,
        help="沿固定随机 MANO beta 方向偏移 candidate morphology；0 表示 self-inverse")
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.data
    dataset = InteractionDynamicsDataset(
        data_cfg.train_path, dominant_hand_manifest=data_cfg.dominant_hand_manifest,
        hand_side=data_cfg.hand_side, num_effect_points=cfg.meta.num_effect_points,
        chunk_len=cfg.meta.chunk_len, temporal_stride=cfg.meta.temporal_stride,
        base_seed=cfg.train.seed, max_samples=data_cfg.max_train_samples,
        max_samples_per_sequence=getattr(data_cfg, "max_samples_per_sequence", None),
        min_object_effect_norm=data_cfg.min_object_effect_norm)
    if args.split != "train":
        raise ValueError("当前 V16.1 最小实现只使用显式 controlled train dataset")
    hand_cache_path, current = dataset.sample_location(args.sample_index)
    device = torch.device(args.device)

    with np.load(hand_cache_path.parent / "shared.npz", allow_pickle=False) as shared:
        source_raw_file = str(shared["source_raw_file"].item())
        raw_frame_ids = np.asarray(shared["raw_frame_id"], dtype=np.int64)
        object_pose_np = np.asarray(shared["obj_root_pose_world"][current:current + 9], np.float32)
        object_points_np = np.asarray(shared["obj_points_world"][current], np.float32)
    with np.load(hand_cache_path, allow_pickle=False) as hand_cache:
        side = str(hand_cache["side"].item())
        action_patch_knn = torch.from_numpy(
            np.asarray(dataset[args.sample_index]["action_patch_knn_idx"], np.int64))[None].to(device)
    raw_path = args.grab_root / source_raw_file
    mano, _, params = build_mano(raw_path, args.grab_root, side, args.mano_path, device)
    raw_ids = raw_frame_ids[current:current + 9]

    def raw_tensor(name: str, width: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(params[name][raw_ids], np.float32).reshape(9, width)).to(device)

    gt_orient = raw_tensor("global_orient", 3)
    gt_pose = raw_tensor("hand_pose", 24)
    gt_translation = raw_tensor("transl", 3)
    beta_np = np.asarray(params["betas"], np.float32)
    if beta_np.ndim > 1:
        beta_np = beta_np[raw_ids]
    else:
        beta_np = np.broadcast_to(beta_np, (9, beta_np.shape[-1])).copy()
    teacher_betas = torch.from_numpy(beta_np).to(device)
    beta_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    beta_direction = torch.randn(
        teacher_betas.shape[-1], generator=beta_generator, device=device)
    beta_direction = beta_direction / beta_direction.square().mean().sqrt()
    candidate_betas = (teacher_betas + args.candidate_beta_offset * beta_direction).clamp(-2, 2)
    object_poses = torch.from_numpy(object_pose_np).to(device)
    object_points_world = torch.from_numpy(object_points_np).to(device)
    object_points = points_to_frames(object_points_world[None], object_poses[:1])[0]
    anchor_index = deterministic_fps(object_points[None], 128)[0]
    anchors = object_points[anchor_index]
    faces = torch.from_numpy(np.asarray(mano.faces, np.int64)).to(device)

    with torch.no_grad():
        gt_output = mano(global_orient=gt_orient, hand_pose=gt_pose,
                         betas=teacher_betas, transl=gt_translation)
        gt_surface_world = face_centers(gt_output.vertices, faces)
        gt_surface_object = points_to_frames(gt_surface_world, object_poses)
        gt_y = build_interaction_y(gt_surface_object, anchors, args.tau_m)
        gt_root = root_feature(gt_orient, gt_output.joints[:, 0], object_poses)
        gt_tips = gt_output.vertices[:, TIP_VERTEX_INDEX.to(device)]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    if args.init == "noise":
        pose_init = gt_pose[1:] + torch.randn(
            gt_pose[1:].shape, generator=generator, device=device) * .01
        orient_init = gt_orient[1:] + (2 * torch.rand(
            gt_orient[1:].shape, generator=generator, device=device) - 1) * (5 * torch.pi / 180)
        translation_init = gt_translation[1:] + (2 * torch.rand(
            gt_translation[1:].shape, generator=generator, device=device) - 1) * .005
    else:
        pose_init = gt_pose[:1].expand(8, -1).clone()
        orient_init = gt_orient[:1].expand(8, -1).clone()
        translation_init = gt_translation[:1].expand(8, -1).clone()
    pose = torch.nn.Parameter(pose_init)
    orient = torch.nn.Parameter(orient_init)
    translation = torch.nn.Parameter(translation_init)

    action_encoder = None
    gt_action = None
    if args.target == "action":
        action_encoder = load_action_encoder(cfg.meta.action_encoder_checkpoint,
                                             cfg.meta.model_dim, device)
        gt_hand_rotation = axis_angle_to_matrix(gt_orient)
        gt_hand_local = torch.einsum(
            "tni,tij->tnj", gt_surface_world - gt_output.joints[:, None, 0], gt_hand_rotation)
        with torch.no_grad():
            gt_action = action_tokens(action_encoder, gt_hand_local, action_patch_knn).detach()

    optimizer = torch.optim.Adam([pose, orient, translation], lr=args.lr)
    history: list[dict[str, float]] = []
    initial_surface = None
    for step in range(args.steps + 1):
        all_pose = torch.cat([gt_pose[:1], pose], 0)
        all_orient = torch.cat([gt_orient[:1], orient], 0)
        all_translation = torch.cat([gt_translation[:1], translation], 0)
        output = mano(global_orient=all_orient, hand_pose=all_pose,
                      betas=candidate_betas, transl=all_translation)
        surface_world = face_centers(output.vertices, faces)
        surface_object = points_to_frames(surface_world, object_poses)
        pred_y = build_interaction_y(surface_object, anchors, args.tau_m)
        loss_r = torch.nn.functional.mse_loss(
            100 * pred_y["relative_geometry"], 100 * gt_y["relative_geometry"])
        loss_u = torch.nn.functional.mse_loss(
            100 * pred_y["relative_motion"], 100 * gt_y["relative_motion"])
        pred_root = root_feature(all_orient, output.joints[:, 0], object_poses)
        loss_g = torch.nn.functional.mse_loss(pred_root[1:], gt_root[1:])
        if args.target == "r":
            loss = loss_r
        elif args.target == "u":
            loss = loss_u
        elif args.target == "interaction":
            loss = loss_r + loss_u
        elif args.target == "full":
            loss = loss_r + loss_u + loss_g
        else:
            assert action_encoder is not None and gt_action is not None
            hand_rotation = axis_angle_to_matrix(all_orient)
            hand_local = torch.einsum(
                "tni,tij->tnj", surface_world - output.joints[:, None, 0], hand_rotation)
            loss = torch.nn.functional.mse_loss(
                action_tokens(action_encoder, hand_local, action_patch_knn), gt_action)
        if step == 0:
            initial_surface = surface_world.detach().clone()
        error_mm = (surface_world[1:] - gt_surface_world[1:]).norm(dim=-1) * 1000
        joint_error_mm = (output.joints[1:] - gt_output.joints[1:]).norm(dim=-1) * 1000
        tip_error_mm = (output.vertices[1:, TIP_VERTEX_INDEX.to(device)] - gt_tips[1:]).norm(dim=-1) * 1000
        if step % 50 == 0 or step == args.steps:
            row = {
                "step": step, "loss": float(loss.detach()),
                "r_rmse_cm": float(loss_r.detach().sqrt()),
                "u_rmse_cm": float(loss_u.detach().sqrt()),
                "g_rmse": float(loss_g.detach().sqrt()),
                "surface_ade_mm": float(error_mm.mean()),
                "surface_fde_mm": float(error_mm[-1].mean()),
                "joint_mpjpe_mm": float(joint_error_mm.mean()),
                "wrist_error_mm": float(joint_error_mm[:, 0].mean()),
                "tip_mpjpe_mm": float(tip_error_mm.mean()),
                **binary_contact_metrics(pred_y["relative_geometry"], gt_y["relative_geometry"]),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert initial_surface is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, source_raw_file=np.asarray(source_raw_file), side=np.asarray(side),
        current_raw_frame=np.asarray(raw_ids[0]), target=np.asarray(args.target),
        initialization=np.asarray(args.init), anchors_object=anchors.detach().cpu().numpy(),
        candidate_beta_offset=np.asarray(args.candidate_beta_offset, np.float32),
        teacher_betas=teacher_betas.detach().cpu().numpy(),
        candidate_betas=candidate_betas.detach().cpu().numpy(),
        initial_points_world=initial_surface.cpu().numpy(),
        optimized_points_world=surface_world.detach().cpu().numpy(),
        gt_points_world=gt_surface_world.cpu().numpy(),
        history_json=np.asarray(json.dumps(history, ensure_ascii=False)))
    print(f"已保存 {args.output}")


if __name__ == "__main__":
    main()
