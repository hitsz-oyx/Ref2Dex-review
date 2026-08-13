"""V18.5/V18.6 checkpoint → 通用 VisualizationSample provider。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABSeqData, load_object_canonical_mesh
from src.task.InteractionDynamics.eval_grasp_v18 import trajectory_statistics
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import build_interaction_y_batched
from src.task.InteractionDynamics.mano_hand_transition import CachedManoHDataset, ManoHandTransition
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import decode_mano_y
from src.task.InteractionDynamics.penetration import penetration_from_mesh
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move
from src.task.InteractionDynamics.viewer_v2.backends import ManoBackend, ObjectTrajectory, VisualizationSample
from src.task.InteractionDynamics.viewer_v2.stable import detect_latch


def decode_mano_vertices(delta_h: torch.Tensor, batch: dict, layer_for) -> torch.Tensor:
    """按训练 decoder 的同一坐标链恢复 world MANO 顶点，仅供可视化。"""
    device = delta_h.device
    vertices: list[torch.Tensor | None] = [None] * len(delta_h)
    for key in sorted(set(batch["mano_key"])):
        indices = [i for i, value in enumerate(batch["mano_key"]) if value == key]
        index = torch.as_tensor(indices, device=device)
        layer = layer_for(key, batch["side"][indices[0]], batch["source_raw_file"][indices[0]])
        q = batch["object_rotation"][index]
        t = batch["object_translation"][index]
        current_rotation = q[:, :1].transpose(-1, -2) @ axis_angle_to_matrix(
            batch["global_orient"][index, :1])
        relative = axis_angle_to_matrix(delta_h[index, :, 3:6]) @ current_rotation
        world_rotation = q[:, 1:] @ relative
        object_position = (batch["current_h"][index, None, :3] + delta_h[index, :, :3]) / 100
        world_translation = (object_position[..., None, :] - t[:, 1:]) @ q[:, 1:].transpose(-1, -2)
        future_pose = batch["hand_pose"][index, :1] + delta_h[index, :, 6:]
        current_vertices = layer(
            global_orient=batch["global_orient"][index, 0],
            hand_pose=batch["hand_pose"][index, 0],
            betas=batch["betas"][index, 0],
            transl=batch["transl"][index, 0],
        ).vertices
        future_vertices = layer(
            global_orient=matrix_to_axis_angle(world_rotation).flatten(0, 1),
            hand_pose=future_pose.flatten(0, 1),
            betas=batch["betas"][index, 1:].flatten(0, 1),
            transl=world_translation[..., 0, :].flatten(0, 1),
        ).vertices.reshape(len(index), 8, -1, 3)
        world_vertices = torch.cat([current_vertices[:, None], future_vertices], dim=1)
        for local, original in enumerate(indices):
            vertices[original] = world_vertices[local]
    return torch.stack([value for value in vertices if value is not None])


class V18_5PredictionProvider:
    def __init__(self, checkpoint: str | Path, cache: str | Path, split: str,
                 grab_root: str | Path = Path(DEFAULT_GRAB_ROOT) / "data",
                 mano_path: str | Path = DEFAULT_MANO_MODEL_DIR,
                 device: str = "cuda", post_latch_frames: int = 60) -> None:
        self.device = torch.device(device); self.dataset = CachedManoHDataset(cache, split)
        self.grab_root = Path(grab_root); self.mano_path = Path(mano_path)
        payload = torch.load(checkpoint, map_location="cpu"); saved = payload["args"]
        self.model = ManoHandTransition(dim=saved["dim"], layers=saved["layers"]).to(self.device)
        self.model.load_state_dict(payload["model"]); self.model.eval()
        self.delta_stats = {key: value.to(self.device) for key, value in payload["delta_stats"].items()}
        self.layers = ManoLayers(self.grab_root, self.mano_path, self.device)
        self.backend = ManoBackend(); self.split = split
        self.post_latch_frames = post_latch_frames

    def __len__(self) -> int: return len(self.dataset)

    @torch.inference_mode()
    def __call__(self, index: int) -> VisualizationSample:
        item = self.dataset[index]
        batch = move({key: ([value] if isinstance(value, str) else value[None])
                      for key, value in item.items()}, self.device)
        normalized = self.model(batch["state"], batch["anchors_cm"],
                                batch["object_patches"], batch["current_h"])
        delta = normalized * self.delta_stats["delta_std"] + self.delta_stats["delta_mean"]
        future, predicted_surface_reference = decode_mano_y(delta, batch, self.layers)
        predicted_vertices_world = decode_mano_vertices(delta, batch, self.layers)
        layer = self.layers(item["mano_key"], item["side"], item["source_raw_file"])
        faces = np.asarray(layer.faces, np.int32)
        gt_vertices_world = layer(
            global_orient=item["global_orient"].to(self.device),
            hand_pose=item["hand_pose"].to(self.device), betas=item["betas"].to(self.device),
            transl=item["transl"].to(self.device)).vertices
        sequence = GRABSeqData(str(self.grab_root / item["source_raw_file"]))
        mesh = load_object_canonical_mesh(sequence.obj_name, str(self.grab_root), "m")
        raw_ids = item["raw_frame_ids"].numpy(); params = sequence.get_object_params()
        with np.load(Path(item["source_hand_cache"]).parent / "shared.npz",
                     allow_pickle=False) as shared:
            full_raw_ids = np.asarray(shared["raw_frame_id"], np.int64)
        raw_to_cache = {int(raw): cache for cache, raw in enumerate(full_raw_ids)}
        cache_start = raw_to_cache[int(raw_ids[0])]
        available_raw_ids = full_raw_ids[cache_start:]
        object_rotation = axis_angle_to_matrix(torch.from_numpy(np.asarray(
            params["global_orient"][available_raw_ids], np.float32))).numpy()
        object_translation = np.asarray(params["transl"][available_raw_ids], np.float32)
        poses = np.tile(np.eye(4, dtype=np.float32), (len(available_raw_ids), 1, 1))
        poses[:, :3, :3] = object_rotation; poses[:, :3, 3] = object_translation

        predicted_y = build_interaction_y_batched(
            predicted_surface_reference, batch["anchors_cm"] / 100)
        distance_cm = (100 * predicted_y["relative_distance"][0]).cpu().numpy()
        motion_cm = (100 * predicted_y["relative_motion"][0]).cpu().numpy()
        canonical_vertices = np.asarray(mesh.vertices, np.float32)
        canonical_faces = np.asarray(mesh.faces, np.int32)

        def diagnostic(vertices_world: np.ndarray, frame: int):
            vertices_object = ((vertices_world - object_translation[frame])
                               @ object_rotation[frame])
            return penetration_from_mesh(vertices_object, canonical_vertices, canonical_faces,
                                         item["anchors_cm"].numpy() / 100)

        # p 第一版不参与 gate；只在预测末帧做 Pred/GT 同口径诊断，避免高面数 mesh
        # 的精确逐面距离拖慢交互加载。
        pred_diag = diagnostic(predicted_vertices_world[0, -1].cpu().numpy(), 8)
        gt_diag = diagnostic(gt_vertices_world[-1].cpu().numpy(), 8)
        latch_frame, stable_states = detect_latch(distance_cm, motion_cm)

        def object_follow(frames: list, latch: int | None) -> list:
            if latch is None: return frames
            end = min(len(poses), latch + 1 + self.post_latch_frames)
            local = ((frames[latch].vertices - object_translation[latch])
                     @ object_rotation[latch])
            return frames[:latch + 1] + [type(frames[latch])(
                local @ object_rotation[frame].T + object_translation[frame],
                frames[latch].faces) for frame in range(latch + 1, end)]

        prediction_frames = self.backend.decode_prediction(
            item, (predicted_vertices_world[0].cpu().numpy(), faces))
        gt_frames = self.backend.decode_gt({"gt_mesh": (gt_vertices_world.cpu().numpy(), faces)})
        prediction_frames = object_follow(prediction_frames, latch_frame)
        gt_frames = object_follow(gt_frames, latch_frame) if gt_frames is not None else None
        poses = poses[:len(prediction_frames)]
        proxy = bool(trajectory_statistics(future)["success"][0])
        pred_valid = pred_diag.valid; gt_valid = gt_diag.valid
        return VisualizationSample(
            ObjectTrajectory(canonical_vertices, canonical_faces, poses), prediction_frames,
            gt_frames, proxy, collision=(None if not pred_valid else
                                         pred_diag.penetrating_point_ratio > 0),
            max_penetration_mm=(pred_diag.max_penetration_mm if pred_valid else None),
            mean_penetration_mm=(pred_diag.mean_penetration_mm if pred_valid else None),
            gt_max_penetration_mm=(gt_diag.max_penetration_mm if gt_valid else None),
            gt_mean_penetration_mm=(gt_diag.mean_penetration_mm if gt_valid else None),
            latch_frame=latch_frame, stable_states=tuple(stable_states),
            label=f"{self.split} sample {index}")
