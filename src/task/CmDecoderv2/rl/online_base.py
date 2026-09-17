"""Frozen Cm windows plus online decoder queries from simulator state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.base import load_config
from src.task.CmDecoderv2.model import CmDecoderV2
from src.task.CmDecoderv2.pointflow import make_relative_transform
from .residual_contract import actual_queries


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_decoder(config_path: str, checkpoint_path: str, expected_sha: str, device: str) -> CmDecoderV2:
    if sha256(checkpoint_path) != expected_sha:
        raise ValueError("Frozen decoder checkpoint SHA256 mismatch")
    cfg = load_config(config_path)
    cfg.model.meta = cfg.meta
    model = CmDecoderV2(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False).eval()
    return model


class OnlineCmBase:
    def __init__(self, config: dict, device: str):
        self.manifest_path = Path(config["sourceManifest"]).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["schema_name"] != "ref2dex_cm_online_source_v1":
            raise ValueError("Not an online Cm source artifact")
        source_path = self.manifest_path.parent / self.manifest["data_file"]
        if sha256(source_path) != self.manifest["data_sha256"]:
            raise ValueError("Cm source artifact checksum mismatch")
        if self.manifest["checkpoint_sha256"] != config["decoderCheckpointSha256"]:
            raise ValueError("Cm source/checkpoint identity mismatch")
        self.model = load_frozen_decoder(config["decoderConfig"], config["decoderCheckpoint"], config["decoderCheckpointSha256"], device)
        with np.load(source_path, allow_pickle=False) as data:
            self.data = {k: torch.as_tensor(data[k].copy(), device=device) for k in data.files}
        self.window_count = int(self.data["cm_tokens"].shape[0])
        self.calls = 0

    @torch.inference_mode()
    def predict(self, frame: torch.Tensor, native_q: torch.Tensor,
                link_world: torch.Tensor, object_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state, queries = actual_queries(native_q, link_world, object_world)
        if not torch.isfinite(state).all() or not torch.isfinite(queries).all():
            raise FloatingPointError(
                f"Non-finite online input: native={bool(torch.isfinite(native_q).all())}, "
                f"links={bool(torch.isfinite(link_world).all())}, object={bool(torch.isfinite(object_world).all())}"
            )
        index = frame.clamp(0, self.window_count - 1).long()
        output = self.model.core(self.data["cm_tokens"][index], self.data["anchor_pos"][index],
                                 self.data["anchor_normal"][index], state, queries)
        q = state[:, :6] + output["pred_q_delta"][:, 0]
        wrist = link_world[:, 0] @ make_relative_transform(output["pred_wrist_rotvec"][:, 0], output["pred_wrist_translation"][:, 0])
        if not torch.isfinite(q).all() or not torch.isfinite(wrist).all():
            raise FloatingPointError("Non-finite online decoder target")
        self.calls += 1
        return q, wrist
