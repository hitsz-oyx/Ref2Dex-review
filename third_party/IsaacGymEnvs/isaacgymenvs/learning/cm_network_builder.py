"""rl_games network consuming frozen, simulator-generated OI-Cm context."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from rl_games.algos_torch import network_builder


class CmBuilder(network_builder.A2CBuilder):
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.cm_feature_dim = int(params.get("cm_feature_dim", 32))
            super().__init__(params, **kwargs)
            obs_dim = kwargs.get("input_shape")[0]
            checkpoint = str(params.get("oi_cm_checkpoint", "") or "").strip()
            if not checkpoint:
                raise ValueError("CmResidual requires network.oi_cm_checkpoint for the frozen OI-Cm model")
            self.oi_cm = self._load_oi_cm(Path(checkpoint).expanduser().resolve())
            self.cm_feature_dim = int(getattr(self.oi_cm, "feature_dim", 32))
            for parameter in self.oi_cm.parameters():
                parameter.requires_grad_(False)
            # The task appends the frozen OI-Cm context after the 1442-D teacher
            # observation.  Keep the full policy observation width unchanged here.
            self._replace_first(self.actor_mlp, obs_dim)
            if self.separate:
                self._replace_first(self.critic_mlp, obs_dim)

        @staticmethod
        def _load_oi_cm(path: Path):
            if not path.is_file():
                raise FileNotFoundError(f"OI-Cm checkpoint not found: {path}")
            from types import SimpleNamespace
            from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel
            payload = torch.load(path, map_location="cpu", weights_only=False)
            cfg = payload.get("config", {})
            meta = SimpleNamespace(**cfg.get("meta", {}))
            meta.modification_version = cfg.get("modification_version", "V1.3")
            model = ObjectInteractionCmModel(meta)
            model.load_state_dict(payload["model"], strict=True)
            return model.eval()

        @staticmethod
        def _replace_first(module: nn.Sequential, input_dim: int):
            for i, layer in enumerate(module):
                if isinstance(layer, nn.Linear):
                    module[i] = nn.Linear(input_dim, layer.out_features).to(layer.weight.device)
                    return
            raise RuntimeError("CmBuilder requires a non-empty MLP")

        def _cm_input(self, obs):
            # Geometry and frozen OI-Cm inference happen in the simulator task;
            # the resulting context is carried after the 1442-D teacher prefix.
            return obs

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            return self.eval_actor(obs) + (self.eval_critic(obs), states)

        def eval_actor(self, obs):
            a_out = self.actor_cnn(self._cm_input(obs)).contiguous().view(obs.shape[0], -1)
            a_out = self.actor_mlp(a_out)
            if self.is_discrete:
                return self.logits(a_out)
            if self.is_multi_discrete:
                return [logit(a_out) for logit in self.logits]
            mu = self.mu_act(self.mu(a_out))
            if self.space_config['fixed_sigma']:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))
            return mu, sigma

        def eval_critic(self, obs):
            c_out = self.critic_cnn(self._cm_input(obs)).contiguous().view(obs.shape[0], -1)
            c_out = self.critic_mlp(c_out)
            return self.value_act(self.value(c_out))

        def get_aux_loss(self):
            # OI-Cm is frozen; there is no auxiliary online/target loss.
            return {}

    def build(self, name, **kwargs):
        return CmBuilder.Network(self.params, **kwargs)
