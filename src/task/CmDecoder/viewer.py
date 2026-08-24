"""Viser viewer for comparing current, GT, and CmDecoder-predicted hands.

Example:
    PYTHONPATH=. python -m src.task.CmDecoder.viewer \
      --object apple --scene 2 --port 8095

The default checkpoint is the full object-disjoint 3 Hz ``qt_cm`` model. Pass
``--decoder-checkpoint`` to visualize another compatible CmDecoder checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import viser

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.base.utils import import_from_path
from src.task.CmDecoder.dataset import _load_hrdex_io, _robot_hand_mesh, _to_world
from src.task.CmDecoder.q_optimizer import (
    DifferentiableInspireHand,
    optimize_q_from_hand_points,
    rotvec_to_matrix,
)


DEFAULT_DECODER_CHECKPOINT = Path(
    "outputs/cmdecoder/cm_decoder_20260822_143843/checkpoints/best.pt"
)
TASK_FIELDS = (
    "hand_points",
    "hand_normals",
    "hand_flow",
    "obj_points",
    "obj_normals",
    "obj_valid_mask",
    "q_t",
    "q_next",
    "delta_time_s",
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_indices(source_frame_ids: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``build_horizon_cache.py`` source/target pair selection."""
    source_frame_ids = np.asarray(source_frame_ids, dtype=np.int64).reshape(-1)
    if stride < 1:
        raise ValueError(f"horizon stride must be positive, got {stride}")
    current = np.arange(0, len(source_frame_ids) - stride, dtype=np.int64)
    target = current + stride
    keep = (source_frame_ids[target] - source_frame_ids[current]) == stride
    return current[keep], target[keep]


def _compose_hand_q(q_full_current: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
    """Build the hand articulation in the current wrist frame.

    The relative wrist transform is applied to the resulting mesh separately;
    target arm q/FK is intentionally never used.
    """
    result = np.asarray(q_full_current, dtype=np.float32).copy()
    hand_q = np.asarray(hand_q, dtype=np.float32).reshape(-1)
    if result.shape != (12,) or hand_q.shape != (6,):
        raise ValueError(f"Expected q_full [12] and hand_q [6], got {result.shape} and {hand_q.shape}")
    result[6:] = hand_q
    return result


def _geometry_dir(cache_root: Path, cache_relative: str, episode: str) -> Path:
    cache_id = Path(cache_relative).name
    candidates = [cache_root / cache_relative / "geometry"]
    candidates.extend(cache_root.parent.glob(f"*/v4/episodes/{cache_id}/geometry"))
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("episode") == episode:
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate the 30 Hz geometry cache for {episode!r} (cache id {cache_id}). "
        f"Searched beside {cache_root}."
    )


class CmDecoderViewerData:
    """Load one cached episode and run a CmDecoder checkpoint on demand."""

    def __init__(
        self,
        checkpoint: Path,
        object_name: str,
        scene: str,
        *,
        dataset_root: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        payload = load_checkpoint(self.checkpoint, map_location="cpu")
        self.cfg = task_config_from_dict(payload["config"])
        self.decoder_input = str(self.cfg.meta.decoder_input)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_cfg = copy.copy(self.cfg.model)
        model_cfg.meta = self.cfg.meta
        model_cls = import_from_path(model_cfg.class_path)
        self.model = model_cls(model_cfg)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.decoder_input = str(getattr(self.model, "decoder_input", self.decoder_input))

        self.object_name = object_name
        self.scene = str(scene)
        self.episode_id = f"inspire_f1/{object_name}/{self.scene}"
        manifest_path = _resolve_path(self.cfg.data.cache_manifest)
        self.selection_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.episode_id not in self.selection_manifest["cache_dirs"]:
            raise KeyError(
                f"Episode {self.episode_id!r} is not present in checkpoint cache manifest {manifest_path}"
            )
        self.split = next(
            (name for name, episodes in self.selection_manifest["splits"].items() if self.episode_id in episodes),
            "unknown",
        )
        cache_root = _resolve_path(self.cfg.meta.cache_root)
        cache_relative = self.selection_manifest["cache_dirs"][self.episode_id]
        self.cache_episode = cache_root / cache_relative
        self.task_dir = self.cache_episode / "task"
        self.task_manifest = json.loads((self.task_dir / "manifest.json").read_text(encoding="utf-8"))
        self.task_arrays = {
            name: np.load(self.task_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in TASK_FIELDS
        }
        self.sample_count = int(self.task_manifest["samples"])

        self.geometry_dir = _geometry_dir(cache_root, cache_relative, self.episode_id)
        self.q_full = np.load(self.geometry_dir / "q_full.npy", mmap_mode="r", allow_pickle=False)
        self.source_frame_ids = np.load(
            self.geometry_dir / "source_frame_id.npy", mmap_mode="r", allow_pickle=False
        )
        self.hand_points_world = np.load(
            self.geometry_dir / "hand_points_world.npy", mmap_mode="r", allow_pickle=False
        )
        self.obj_points_world = np.load(
            self.geometry_dir / "obj_points_world.npy", mmap_mode="r", allow_pickle=False
        )
        self.wrist_pose_world = np.load(
            self.geometry_dir / "wrist_pose_world.npy", mmap_mode="r", allow_pickle=False
        )
        stride = int(self.task_manifest.get("horizon_stride", 1))
        self.horizon_stride = stride
        self.current_indices, self.target_indices = _pair_indices(self.source_frame_ids, stride)
        if len(self.current_indices) != self.sample_count:
            raise ValueError(
                f"Task/geometry pair mismatch for {self.episode_id}: "
                f"task={self.sample_count}, geometry={len(self.current_indices)}"
            )

        root_value = dataset_root if dataset_root is not None else Path(self.cfg.meta.dataset_root)
        self.dataset_root = _resolve_path(root_value)
        episode = self.dataset_root / self.episode_id
        c2r_path = episode / "C2R.npy"
        self.c2r = (
            np.asarray(np.load(c2r_path, allow_pickle=False), dtype=np.float32)
            if c2r_path.exists()
            else np.eye(4, dtype=np.float32)
        )
        robot_urdf = _resolve_path(self.cfg.meta.robot_urdf)
        self.io = _load_hrdex_io(self.dataset_root.parent)
        self.urdf = self.io.parse_urdf(robot_urdf)
        self.mesh_cache: dict[Any, Any] = {}

        self.cm_tokens: np.ndarray | None = None
        token_path = self.cache_episode / "cm" / "cm_tokens.npy"
        token_manifest_path = self.cache_episode / "cm" / "manifest.json"
        if self.decoder_input != "qt_only" and token_path.is_file() and token_manifest_path.is_file():
            token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
            cm_checkpoint = _resolve_path(self.cfg.meta.cm_checkpoint)
            if token_manifest.get("cm_checkpoint_sha256") == _sha256(cm_checkpoint):
                self.cm_tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
        self._predictions: dict[int, np.ndarray] = {}
        self._predicted_flows: dict[int, np.ndarray] = {}
        self._wrist_predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._q_fitter: DifferentiableInspireHand | None = None

    def __len__(self) -> int:
        return self.sample_count

    def _batch(self, index: int) -> dict[str, torch.Tensor]:
        batch = {
            name: torch.from_numpy(np.array(array[index], copy=True)).unsqueeze(0).to(self.device)
            for name, array in self.task_arrays.items()
        }
        if self.cm_tokens is not None:
            batch["cm_tokens"] = torch.from_numpy(
                np.array(self.cm_tokens[index], copy=True)
            ).unsqueeze(0).to(self.device)
        return batch

    @torch.inference_mode()
    def _model_output(self, index: int) -> dict[str, torch.Tensor]:
        return self.model(self._batch(index))

    def predict_q(self, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, len(self) - 1))
        if index not in self._predictions:
            with torch.inference_mode():
                output = self._model_output(index)
            if "pred_q_next" in output:
                q = output["pred_q_next"]
                if "pred_wrist_delta_translation" in output:
                    self._wrist_predictions[index] = (
                        output["pred_wrist_delta_translation"][0].detach().cpu().numpy().astype(np.float32),
                        output["pred_wrist_delta_rotvec"][0].detach().cpu().numpy().astype(np.float32),
                    )
            elif "pred_hand_flow" in output:
                predicted_flow = output["pred_hand_flow"].detach()
                self._predicted_flows[index] = predicted_flow[0].cpu().numpy().astype(np.float32)
                if self._q_fitter is None:
                    self._q_fitter = DifferentiableInspireHand(
                        self.cfg.meta.robot_urdf,
                        num_hand_points=int(self.cfg.meta.num_hand_points),
                        sample_seed=int(self.cfg.meta.sample_seed),
                        device=self.device,
                    )
                batch = self._batch(index)
                fit = optimize_q_from_hand_points(
                    self._q_fitter,
                    q_t=batch["q_t"],
                    current_hand_points=batch["hand_points"],
                    target_hand_points=batch["hand_points"] + predicted_flow,
                    steps=int(getattr(self.cfg.meta, "q_fit_steps", 100)),
                    lr=float(getattr(self.cfg.meta, "q_fit_lr", 0.05)),
                    prior_weight=float(getattr(self.cfg.meta, "q_fit_prior_weight", 1e-4)),
                    wrist_prior_weight=float(
                        getattr(self.cfg.meta, "wrist_fit_prior_weight", 1e-6)
                    ),
                )
                q = fit["q"]
                self._wrist_predictions[index] = (
                    fit["wrist_delta_translation"][0].cpu().numpy().astype(np.float32),
                    fit["wrist_delta_rotvec"][0].cpu().numpy().astype(np.float32),
                )
            else:
                raise KeyError("CmDecoder model must output pred_q_next or pred_hand_flow")
            self._predictions[index] = q[0].detach().cpu().numpy().astype(np.float32)
        return self._predictions[index]

    def frame(self, index: int) -> dict[str, Any]:
        index = int(np.clip(index, 0, len(self) - 1))
        source_index = int(self.current_indices[index])
        target_index = int(self.target_indices[index])
        q_t = np.asarray(self.task_arrays["q_t"][index], dtype=np.float32)
        q_gt = np.asarray(self.task_arrays["q_next"][index], dtype=np.float32)
        q_pred = self.predict_q(index)
        q_full_current = np.asarray(self.q_full[source_index], dtype=np.float32)
        wrist_current = np.asarray(self.wrist_pose_world[source_index], dtype=np.float32)
        wrist_target = np.asarray(self.wrist_pose_world[target_index], dtype=np.float32)
        wrist_gt_relative = np.linalg.inv(wrist_current) @ wrist_target
        pred_translation, pred_rotvec = self._wrist_predictions.get(
            index, (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        )
        pred_rotation = rotvec_to_matrix(torch.from_numpy(pred_rotvec)[None])[0].numpy()
        wrist_pred_relative = np.eye(4, dtype=np.float32)
        wrist_pred_relative[:3, :3] = pred_rotation
        wrist_pred_relative[:3, 3] = pred_translation

        meshes = {}
        mesh_specs = (
            ("current", q_t, np.eye(4, dtype=np.float32)),
            ("gt", q_gt, wrist_gt_relative),
            ("predicted", q_pred, wrist_pred_relative),
        )
        for name, hand_q, wrist_relative in mesh_specs:
            q_full = _compose_hand_q(q_full_current, hand_q)
            vertices, faces = _robot_hand_mesh(self.io, self.urdf, q_full, self.mesh_cache)
            vertices_world = _to_world(vertices, self.c2r)
            vertices_current_wrist = (
                vertices_world - wrist_current[:3, 3]
            ) @ wrist_current[:3, :3]
            vertices_moved = (
                vertices_current_wrist @ wrist_relative[:3, :3].T + wrist_relative[:3, 3]
            )
            vertices_world = (
                vertices_moved @ wrist_current[:3, :3].T + wrist_current[:3, 3]
            )
            meshes[name] = (vertices_world.astype(np.float32), faces)
        return {
            "meshes": meshes,
            "hand_points": np.asarray(self.hand_points_world[source_index], dtype=np.float32),
            "object_points": np.asarray(self.obj_points_world[source_index], dtype=np.float32),
            "flow": np.asarray(
                self.hand_points_world[target_index] - self.hand_points_world[source_index], dtype=np.float32
            ),
            "q_t": q_t,
            "q_gt": q_gt,
            "q_pred": q_pred,
            "wrist_gt_relative": wrist_gt_relative,
            "wrist_pred_relative": wrist_pred_relative,
            "delta_time_s": float(self.task_arrays["delta_time_s"][index]),
            "source_frame_id": int(self.source_frame_ids[source_index]),
            "target_frame_id": int(self.source_frame_ids[target_index]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_DECODER_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--object", dest="object_name", required=True)
    parser.add_argument("--scene", default="0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    data = CmDecoderViewerData(
        _resolve_path(args.decoder_checkpoint),
        args.object_name,
        args.scene,
        dataset_root=args.dataset_root,
        device=args.device,
    )
    server = viser.ViserServer(host=args.host, port=args.port)
    frame = server.gui.add_slider("Pair", min=0, max=len(data) - 1, step=1, initial_value=0)
    play = server.gui.add_button("Play")
    stop = server.gui.add_button("Stop")
    show_current = server.gui.add_checkbox("Current hand", True)
    show_gt = server.gui.add_checkbox("GT hand", True)
    show_predicted = server.gui.add_checkbox("Predicted hand", True)
    show_object = server.gui.add_checkbox("Show Object Samples", True)
    show_points = server.gui.add_checkbox("Show Hand Samples", True)
    show_flow = server.gui.add_checkbox("Show Hand Flow", True)
    hand_point_size = server.gui.add_slider(
        "Hand Point Size", min=0.001, max=0.020, step=0.001, initial_value=0.005
    )
    object_point_size = server.gui.add_slider(
        "Object Point Size", min=0.001, max=0.020, step=0.001, initial_value=0.004
    )
    flow_line_width = server.gui.add_slider(
        "Flow Line Width", min=0.5, max=10.0, step=0.5, initial_value=2.0
    )
    flow_scale = server.gui.add_slider(
        "Flow Length Scale", min=0.0, max=20.0, step=0.5, initial_value=1.0
    )
    status = server.gui.add_markdown("")
    state = {"playing": False}
    handles = []
    lock = threading.Lock()

    def render() -> None:
        nonlocal handles
        sample = data.frame(int(frame.value))
        with server.atomic():
            for handle in handles:
                handle.remove()
            handles = []
            if show_object.value:
                handles.append(
                    server.scene.add_point_cloud(
                        "/world/object/samples",
                        sample["object_points"],
                        colors=(175, 175, 175),
                        point_size=float(object_point_size.value),
                    )
                )
            mesh_specs = (
                (show_current.value, "current", (65, 145, 255), 0.28),
                (show_gt.value, "gt", (55, 210, 120), 0.38),
                (show_predicted.value, "predicted", (255, 120, 45), 0.48),
            )
            for visible, name, color, opacity in mesh_specs:
                if visible:
                    vertices, faces = sample["meshes"][name]
                    handles.append(
                        server.scene.add_mesh_simple(
                            f"/world/hand/{name}", vertices, faces, color=color, opacity=opacity
                        )
                    )
            if show_points.value:
                handles.append(
                    server.scene.add_point_cloud(
                        "/world/hand/samples",
                        sample["hand_points"],
                        colors=(235, 235, 235),
                        point_size=float(hand_point_size.value),
                    )
                )
            if show_flow.value:
                starts = sample["hand_points"][::8]
                ends = (sample["hand_points"] + sample["flow"] * float(flow_scale.value))[::8]
                segments = np.stack([starts, ends], axis=1)
                handles.append(
                    server.scene.add_line_segments(
                        "/world/hand/flow",
                        segments,
                        colors=(220, 70, 210),
                        line_width=float(flow_line_width.value),
                    )
                )
        mae_deg = float(np.abs(sample["q_pred"] - sample["q_gt"]).mean() * 180.0 / np.pi)
        identity_deg = float(np.abs(sample["q_t"] - sample["q_gt"]).mean() * 180.0 / np.pi)
        wrist_translation_error_mm = float(
            np.linalg.norm(
                sample["wrist_pred_relative"][:3, 3] - sample["wrist_gt_relative"][:3, 3]
            ) * 1000.0
        )
        wrist_rotation_error = (
            sample["wrist_pred_relative"][:3, :3].T @ sample["wrist_gt_relative"][:3, :3]
        )
        wrist_rotation_error_deg = float(
            np.degrees(
                np.arccos(np.clip((np.trace(wrist_rotation_error) - 1.0) * 0.5, -1.0, 1.0))
            )
        )
        status.content = (
            f"**{data.episode_id}** ({data.split})  \n"
            f"Pair: **{int(frame.value)}/{len(data)-1}**, source frames: "
            f"**{sample['source_frame_id']} → {sample['target_frame_id']}**, "
            f"Δt: **{sample['delta_time_s']:.3f} s**  \n"
            f"Decoder input: **{data.decoder_input}**, q MAE: **{mae_deg:.3f}°**, "
            f"identity: **{identity_deg:.3f}°**  \n"
            f"Wrist error: **{wrist_translation_error_mm:.2f} mm / "
            f"{wrist_rotation_error_deg:.2f}°**  \n"
            f"q current: `{np.array2string(sample['q_t'], precision=4)}`  \n"
            f"q GT: `{np.array2string(sample['q_gt'], precision=4)}`  \n"
            f"q predicted: `{np.array2string(sample['q_pred'], precision=4)}`  \n"
            "GT/predicted meshes use relative wrist SE(3); future arm FK is not used."
        )

    def render_locked() -> None:
        with lock:
            render()

    frame.on_update(lambda _: render_locked())
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    for control in (
        show_current,
        show_gt,
        show_predicted,
        show_object,
        show_points,
        show_flow,
        hand_point_size,
        object_point_size,
        flow_line_width,
        flow_scale,
    ):
        control.on_update(lambda _: render_locked())
    render_locked()
    print(f"CmDecoder viewer: http://localhost:{args.port}", flush=True)
    while True:
        if state["playing"]:
            frame.value = (int(frame.value) + 1) % len(data)
        time.sleep(1.0 / max(args.fps, 1e-3))


if __name__ == "__main__":
    main()
