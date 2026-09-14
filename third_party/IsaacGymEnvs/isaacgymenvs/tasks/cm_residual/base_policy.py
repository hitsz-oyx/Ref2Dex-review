"""DExplore Inspire teacher contract used by CmResidual."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib

import torch
from torch import nn

OBSERVATION_DIM = 1442
ACTION_DIM = 18


class InspireDExplorePolicy(nn.Module):
    """Frozen DExplore teacher with the published Inspire MLP layout."""

    def __init__(self, checkpoint: str, device: torch.device | str = "cpu",
                 expected_sha256: Optional[str] = None):
        super().__init__()
        self.device = torch.device(device)
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DExplore checkpoint not found: {path}")
        if expected_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_sha256:
                raise ValueError(f"DExplore checkpoint SHA256 mismatch: {digest} != {expected_sha256}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = payload.get("model", payload)
        required = {
            "a2c_network.actor_mlp.0.weight": (1024, OBSERVATION_DIM),
            "a2c_network.actor_mlp.6.weight": (512, 1024),
            "a2c_network.mu.weight": (ACTION_DIM, 512),
        }
        for key, shape in required.items():
            if key not in model or tuple(model[key].shape) != shape:
                raise ValueError(f"Unsupported DExplore checkpoint key/shape: {key}")
        self.actor = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, 1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.ELU(),
            nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512, ACTION_DIM),
        )
        self.actor[0].load_state_dict({"weight": model["a2c_network.actor_mlp.0.weight"], "bias": model["a2c_network.actor_mlp.0.bias"]})
        self.actor[2].load_state_dict({"weight": model["a2c_network.actor_mlp.2.weight"], "bias": model["a2c_network.actor_mlp.2.bias"]})
        self.actor[4].load_state_dict({"weight": model["a2c_network.actor_mlp.4.weight"], "bias": model["a2c_network.actor_mlp.4.bias"]})
        self.actor[6].load_state_dict({"weight": model["a2c_network.actor_mlp.6.weight"], "bias": model["a2c_network.actor_mlp.6.bias"]})
        self.actor[8].load_state_dict({"weight": model["a2c_network.mu.weight"], "bias": model["a2c_network.mu.bias"]})
        stats = payload.get("running_mean_std", {})
        mean = torch.as_tensor(stats.get("running_mean", torch.zeros(OBSERVATION_DIM)), dtype=torch.float32)
        var = torch.as_tensor(stats.get("running_var", torch.ones(OBSERVATION_DIM)), dtype=torch.float32)
        if tuple(mean.shape) != (OBSERVATION_DIM,) or tuple(var.shape) != (OBSERVATION_DIM,):
            raise ValueError("DExplore running_mean_std must be 1442-dimensional")
        self.register_buffer("running_mean", mean)
        self.register_buffer("running_var", var.clamp_min(1e-6))
        self.to(self.device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.checkpoint = str(path)
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != OBSERVATION_DIM:
            raise ValueError(f"DExplore observation must be {OBSERVATION_DIM}D, got {obs.shape[-1]}")
        normalized = (obs.to(self.device) - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return self.actor(normalized).clamp(-1.0, 1.0)

    act = forward
