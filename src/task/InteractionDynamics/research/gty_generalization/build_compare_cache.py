"""构建 GT MANO / GT-Y Allegro / Direct-H MANO 的离线 comparison cache。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABSeqData, load_object_canonical_mesh
from src.task.InteractionDynamics.dataset_grasp_v18 import _rigid_world_to_reference, _transform
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.direct_h_dynamics_v20_3 import DirectHDynamicsV20_3
from src.task.InteractionDynamics.inverse_mano import face_centers
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import build_causal_rdv_batched, decode_mano_field
from src.task.InteractionDynamics.penetration import penetration_from_mesh
from src.task.InteractionDynamics.research.gty_generalization.optimize_robot_from_gty import (
    optimize_robot_from_gty, representative_frames,
)
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move
from src.task.InteractionDynamics.viewer_gty.robot_backend import UrdfHandBackend, scan_robot_assets
from src.task.InteractionDynamics.viewer_v2.prediction_v18_5 import decode_mano_vertices


def load_direct(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu"); saved = payload["args"]
    model = DirectHDynamicsV20_3(dim=saved["dim"], temporal_layers=saved["temporal_layers"]).to(device)
    model.load_state_dict(payload["model"]); model.requires_grad_(False).eval()
    return model, payload["h_std"].to(device)


def single_batch(item: dict, device: torch.device) -> dict:
    return move({key: ([value] if isinstance(value, str) else value[None])
                 for key, value in item.items()}, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--direct-h-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--robot-assets", type=Path, default=Path("data/raw_data/robot_hands"))
    parser.add_argument("--robot", default="Allegro")
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--optimize-mode", choices=("full", "root", "finger"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    args = parser.parse_args(); device = torch.device(args.device)
    assets = scan_robot_assets(args.robot_assets)
    if args.robot not in assets:
        raise ValueError(f"未找到 {args.robot}，已扫描到 {sorted(assets)}")
    backend = UrdfHandBackend(assets[args.robot], device)
    dataset = CachedFieldV20Dataset(args.cache, args.split); model, h_std = load_direct(args.direct_h_checkpoint, device)
    layers = ManoLayers(args.grab_root, args.mano_path, device); args.output.mkdir(parents=True, exist_ok=True)
    seen = set(); written = 0
    for index in range(len(dataset)):
        item = dataset[index]; source = item["source_raw_file"]
        if source in seen:
            continue
        seen.add(source); batch = single_batch(item, device)
        with torch.no_grad():
            z = model(batch["current_y"], batch["anchors_cm"], batch["object_patches"], batch["current_h"])
            pred_y, _ = decode_mano_field(z * h_std, batch, layers)
            pred_world = decode_mano_vertices(z * h_std, batch, layers)[0]
            layer = layers(item["mano_key"], item["side"], item["source_raw_file"])
            current_future_world = layer(global_orient=item["global_orient"].to(device),
                hand_pose=item["hand_pose"].to(device), betas=item["betas"].to(device),
                transl=item["transl"].to(device)).vertices
        sequence = GRABSeqData(str(args.grab_root / source)); mesh = load_object_canonical_mesh(
            sequence.obj_name, str(args.grab_root), "m")
        raw_ids = item["raw_frame_ids"].numpy(); object_params = sequence.get_object_params()
        with np.load(Path(item["source_hand_cache"]).parent / "shared.npz", allow_pickle=False) as shared:
            reference_raw_ids = np.asarray(shared["raw_frame_id"], np.int64)
            object_world = torch.from_numpy(np.asarray(shared["obj_points_world"], np.float32)).to(device)
        raw_to_index = {int(raw): i for i, raw in enumerate(reference_raw_ids)}
        current_index = raw_to_index[int(raw_ids[0])]
        if current_index == 0:
            raise ValueError("current frame 前没有 causal previous state")
        previous_raw_id = int(reference_raw_ids[current_index - 1])
        hand_params = sequence.get_hand_params(item["side"])
        def raw_parameter(name: str, width: int):
            return torch.from_numpy(np.asarray(hand_params[name][previous_raw_id], np.float32).reshape(1, width)).to(device)
        beta = np.asarray(hand_params["betas"], np.float32)
        beta = beta[previous_raw_id] if beta.ndim > 1 else beta
        previous_world = layer(global_orient=raw_parameter("global_orient", 3),
            hand_pose=raw_parameter("hand_pose", 24),
            betas=torch.from_numpy(np.asarray(beta, np.float32)[None]).to(device),
            transl=raw_parameter("transl", 3)).vertices
        gt_world = torch.cat((previous_world, current_future_world[:8]), 0)
        pred_world = torch.cat((previous_world, pred_world[:8]), 0)
        reference_raw_id = int(reference_raw_ids[0])
        canonical_rotation = axis_angle_to_matrix(torch.from_numpy(np.asarray(
            object_params["global_orient"][reference_raw_id], np.float32))).to(device)
        canonical_translation = torch.from_numpy(np.asarray(
            object_params["transl"][reference_raw_id], np.float32)).to(device)
        reference_rotation, reference_translation = _rigid_world_to_reference(object_world)
        selected_indices = torch.as_tensor([current_index-1, *range(current_index, current_index+8)],
                                           dtype=torch.long, device=device)
        gt_reference = _transform(gt_world, reference_rotation[selected_indices],
                                  reference_translation[selected_indices])
        pred_reference = _transform(pred_world, reference_rotation[selected_indices],
                                    reference_translation[selected_indices])
        gt_object = (gt_reference - canonical_translation[None, None]) @ canonical_rotation
        pred_object = (pred_reference - canonical_translation[None, None]) @ canonical_rotation
        anchors_reference = batch["anchors_cm"][0] / 100
        anchors_canonical = (anchors_reference - canonical_translation) @ canonical_rotation
        gt_y = torch.cat((batch["current_y"][0, :, None, :7],
                          batch["future_y"][0, :, :7, :7]), 1).clone()
        pred_y = torch.cat((batch["current_y"][0, :, None, :7], pred_y[0, :, :7]), 1)
        for value in (gt_y, pred_y):
            value[..., :3] = value[..., :3] @ canonical_rotation
            value[..., 4:7] = value[..., 4:7] @ canonical_rotation
        mano_faces = torch.as_tensor(np.asarray(layer.faces, np.int64), dtype=torch.long, device=device)
        gt_surface = face_centers(gt_object, mano_faces)
        recomputed_y = build_causal_rdv_batched(gt_surface[None], anchors_canonical[None])[0]
        parity_error = (recomputed_y - gt_y).abs()
        parity = {name: {"max_abs_cm": float(parity_error[..., channel].max()),
                         "mean_abs_cm": float(parity_error[..., channel].mean())}
                  for name, channel in (("r", slice(0, 3)), ("d", slice(3, 4)), ("v", slice(4, 7)))}
        if any(row["mean_abs_cm"] >= 1e-3 or row["max_abs_cm"] >= 1e-2 for row in parity.values()):
            raise RuntimeError(f"canonical parity Gate 失败: {parity}")
        optimization = optimize_robot_from_gty(backend, gt_y, 100 * anchors_canonical,
            gt_object, steps=args.steps, optimize_mode=args.optimize_mode)
        object_vertices = np.asarray(mesh.vertices, np.float32); object_faces = np.asarray(mesh.faces, np.int32)
        representatives = representative_frames(gt_y)
        # 优化器产生并缓存完整的 previous+8 九帧轨迹；代表帧仅用于昂贵的 penetration 诊断。
        available = torch.ones(9, dtype=torch.bool)
        robot_vertices = optimization["vertices"].cpu()
        robot_metrics = {"representative_frames": representatives,
            "start_loss": optimization["start_loss"], "final_loss": optimization["final_loss"],
            "y_residual": [None] + optimization["frame_error"].cpu().tolist(),
            "joint_limit_margin": optimization["joint_margin"].cpu().tolist(),
            "penetration_mm": [None] * 9, "contact_count": [None] +
                (optimization["prediction"][..., 3] < 2).sum(0).cpu().tolist()}
        for frame in representatives:
            diagnostic = penetration_from_mesh(robot_vertices[frame].numpy(), object_vertices,
                                               object_faces, anchors_canonical.cpu().numpy())
            robot_metrics["penetration_mm"][frame] = diagnostic.max_penetration_mm if diagnostic.valid else None
        pred_error = (pred_y - gt_y).square().mean((0, 2)).sqrt()
        pred_penetration = []
        for frame in range(9):
            diagnostic = penetration_from_mesh(pred_object[frame].cpu().numpy(), object_vertices,
                                               object_faces, anchors_canonical.cpu().numpy())
            pred_penetration.append(diagnostic.max_penetration_mm if diagnostic.valid else None)
        payload = {"source_raw_file": source, "object_name": sequence.obj_name,
            "subject": item["mano_key"].split(":")[0], "raw_frame_ids": item["raw_frame_ids"],
            "object_vertices": torch.from_numpy(object_vertices), "object_faces": torch.from_numpy(object_faces),
            "gt_vertices_object": gt_object.cpu(), "pred_vertices_object": pred_object.cpu(),
            "mano_faces": mano_faces.cpu(), "anchors": anchors_canonical.cpu(),
            "gt_y": gt_y.cpu(), "pred_y": pred_y.cpu(),
            "canonical_parity": parity,
            "pred_metrics": {"field_error": [None] + pred_error.cpu().tolist(),
                "penetration_mm": pred_penetration,
                "contact_count": [None] + (pred_y[..., 3] < 2).sum(0).cpu().tolist()},
            "robot_results": {args.robot: {"vertices_object": robot_vertices,
                "initial_vertices_object": optimization["initial_vertices"].cpu(),
                "faces": torch.from_numpy(backend.faces), "available": available,
                "surface_sample_count": backend.surface_sample_count,
                "surface_sample_seed": backend.surface_sample_seed,
                "q_initial": optimization["q_initial"].cpu(), "q_optimized": optimization["q"].cpu(),
                "root_translation_initial": optimization["root_translation_initial"].cpu(),
                "root_translation_optimized": optimization["root_translation"].cpu(),
                "root_rotation_initial": optimization["root_rotation_initial"].cpu(),
                "root_rotation_optimized": optimization["root_rotation"].cpu(),
                "gradient_gate": optimization["gradient_gate"],
                "optimization_history": optimization["optimization_history"],
                "metrics": {**robot_metrics, **optimization["final_metrics"]}}}}
        torch.save(payload, args.output / f"trajectory_{written:04d}.pt")
        print({"trajectory": source, "object": sequence.obj_name, "robot": args.robot,
               "dof": backend.dof, "visual_triangles": backend.visual_triangle_count,
               "representative_frames": representatives,
               "surface_samples": backend.surface_sample_count, "parity": parity,
               "gradient_gate": optimization["gradient_gate"],
               "loss": [optimization["start_loss"], optimization["final_loss"]]}, flush=True)
        written += 1
        if written >= args.max_trajectories:
            break
    if written == 0:
        raise RuntimeError("没有构建任何 comparison trajectory")


if __name__ == "__main__":
    main()
