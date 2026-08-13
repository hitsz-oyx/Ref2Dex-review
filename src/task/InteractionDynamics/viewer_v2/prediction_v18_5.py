"""V18.5/V18.6 checkpoint → 通用 VisualizationSample provider。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABSeqData, load_object_canonical_mesh
from src.task.InteractionDynamics.eval_grasp_v18 import trajectory_statistics
from src.task.InteractionDynamics.mano_hand_transition import CachedManoHDataset, ManoHandTransition
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import decode_mano_y
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move
from src.task.InteractionDynamics.viewer_v2.backends import ManoBackend, ObjectTrajectory, VisualizationSample


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
                 device: str = "cuda") -> None:
        self.device = torch.device(device); self.dataset = CachedManoHDataset(cache, split)
        self.grab_root = Path(grab_root); self.mano_path = Path(mano_path)
        payload = torch.load(checkpoint, map_location="cpu"); saved = payload["args"]
        self.model = ManoHandTransition(dim=saved["dim"], layers=saved["layers"]).to(self.device)
        self.model.load_state_dict(payload["model"]); self.model.eval()
        self.delta_stats = {key: value.to(self.device) for key, value in payload["delta_stats"].items()}
        self.layers = ManoLayers(self.grab_root, self.mano_path, self.device)
        self.backend = ManoBackend(); self.split = split

    def __len__(self) -> int: return len(self.dataset)

    @torch.inference_mode()
    def __call__(self, index: int) -> VisualizationSample:
        item = self.dataset[index]
        batch = move({key: ([value] if isinstance(value, str) else value[None])
                      for key, value in item.items()}, self.device)
        normalized = self.model(batch["state"], batch["anchors_cm"],
                                batch["object_patches"], batch["current_h"])
        delta = normalized * self.delta_stats["delta_std"] + self.delta_stats["delta_mean"]
        future, _ = decode_mano_y(delta, batch, self.layers)
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
        object_rotation = axis_angle_to_matrix(torch.from_numpy(
            np.asarray(params["global_orient"][raw_ids], np.float32))).numpy()
        object_translation = np.asarray(params["transl"][raw_ids], np.float32)
        poses = np.tile(np.eye(4, dtype=np.float32), (9, 1, 1))
        poses[:, :3, :3] = object_rotation; poses[:, :3, 3] = object_translation
        prediction_frames = self.backend.decode_prediction(
            item, (predicted_vertices_world[0].cpu().numpy(), faces))
        gt_frames = self.backend.decode_gt({"gt_mesh": (gt_vertices_world.cpu().numpy(), faces)})
        proxy = bool(trajectory_statistics(future)["success"][0])
        return VisualizationSample(
            ObjectTrajectory(np.asarray(mesh.vertices, np.float32),
                             np.asarray(mesh.faces, np.int32), poses),
            prediction_frames, gt_frames, proxy, label=f"{self.split} sample {index}")
