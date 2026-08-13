"""V18.5/V18.6 checkpoint → 通用 VisualizationSample provider。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABSeqData, load_object_canonical_mesh
from src.task.InteractionDynamics.eval_grasp_v18 import trajectory_statistics
from src.task.InteractionDynamics.inverse_mano import face_centers
from src.task.InteractionDynamics.mano_hand_transition import CachedManoHDataset, ManoHandTransition
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import decode_mano_y
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move
from src.task.InteractionDynamics.viewer_v2.backends import ManoBackend, ObjectTrajectory, VisualizationSample


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
        future, predicted_surface_object = decode_mano_y(delta, batch, self.layers)
        predicted_surface_world = torch.einsum(
            "btpi,btji->btpj",
            predicted_surface_object - batch["object_translation"][:, :, None, 0],
            batch["object_rotation"])
        layer = self.layers(item["mano_key"], item["side"], item["source_raw_file"])
        faces = np.asarray(layer.faces, np.int32)
        gt_surface = face_centers(layer(
            global_orient=item["global_orient"].to(self.device),
            hand_pose=item["hand_pose"].to(self.device), betas=item["betas"].to(self.device),
            transl=item["transl"].to(self.device)).vertices,
            torch.as_tensor(faces, dtype=torch.long, device=self.device))
        sequence = GRABSeqData(str(self.grab_root / item["source_raw_file"]))
        mesh = load_object_canonical_mesh(sequence.obj_name, str(self.grab_root), "m")
        raw_ids = item["raw_frame_ids"].numpy(); params = sequence.get_object_params()
        object_rotation = axis_angle_to_matrix(torch.from_numpy(
            np.asarray(params["global_orient"][raw_ids], np.float32))).numpy()
        object_translation = np.asarray(params["transl"][raw_ids], np.float32)
        poses = np.tile(np.eye(4, dtype=np.float32), (9, 1, 1))
        poses[:, :3, :3] = object_rotation; poses[:, :3, 3] = object_translation
        prediction_frames = self.backend.decode_prediction(
            item, (predicted_surface_world[0].cpu().numpy(), faces))
        gt_frames = self.backend.decode_gt({"gt_mesh": (gt_surface.cpu().numpy(), faces)})
        proxy = bool(trajectory_statistics(future)["success"][0])
        return VisualizationSample(
            ObjectTrajectory(np.asarray(mesh.vertices, np.float32),
                             np.asarray(mesh.faces, np.int32), poses),
            prediction_frames, gt_frames, proxy, label=f"{self.split} sample {index}")
