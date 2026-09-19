"""rl_games network consuming frozen, simulator-generated OI-Cm context."""
from __future__ import annotations

import torch
from torch import nn
from rl_games.algos_torch import network_builder


class CmBuilder(network_builder.A2CBuilder):
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.cm_feature_dim = int(params.get("cm_feature_dim", 32))
            observation_dim = int(kwargs.get("input_shape")[0])
            self.actor_input_dim = int(params.get("actor_input_dim", observation_dim))
            self.critic_input_dim = int(params.get("critic_input_dim", observation_dim))
            self.critic_candidate_input_dims = tuple(
                int(value) for value in params.get("critic_candidate_input_dims", ()))
            for name, value in (
                    ("actor_input_dim", self.actor_input_dim),
                    ("critic_input_dim", self.critic_input_dim)):
                if value <= 0 or value > observation_dim:
                    raise ValueError(
                        f"{name} must be in [1, {observation_dim}], got {value}")
            if self.actor_input_dim != self.critic_input_dim and not bool(params.get("separate", False)):
                raise ValueError("Asymmetric actor/critic inputs require separate=true")
            super().__init__(params, **kwargs)
            self.learn_sigma = bool(self.space_config.get("learn_sigma", True))
            if self.fixed_sigma and not self.learn_sigma:
                self.sigma.requires_grad_(False)
            # The task appends the frozen OI-Cm context after the 1442-D teacher
            # observation.  V1.9 can expose that suffix to the critic only.
            self._replace_first(self.actor_mlp, self.actor_input_dim)
            if self.separate:
                self._replace_first(
                    self.critic_mlp, self.critic_input_dim,
                    candidate_input_dims=self.critic_candidate_input_dims)

        @staticmethod
        def _replace_first(module: nn.Sequential, input_dim: int,
                           candidate_input_dims=()):
            for i, layer in enumerate(module):
                if isinstance(layer, nn.Linear):
                    candidate_dims = tuple(candidate_input_dims) or (input_dim,)
                    if len(set(candidate_dims)) != len(candidate_dims):
                        raise ValueError(
                            f"candidate_input_dims must be unique, got {candidate_dims}")
                    if input_dim not in candidate_dims:
                        raise ValueError(
                            f"input_dim {input_dim} missing from candidate_input_dims "
                            f"{candidate_dims}")
                    # Construct every declared candidate in the same order so matched
                    # ablations leave the global RNG in the same post-build state.
                    candidates = {
                        dim: nn.Linear(dim, layer.out_features).to(layer.weight.device)
                        for dim in candidate_dims
                    }
                    module[i] = candidates[input_dim]
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
            actor_obs = self._cm_input(obs)[..., :self.actor_input_dim]
            a_out = self.actor_cnn(actor_obs).contiguous().view(obs.shape[0], -1)
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
            critic_obs = self._cm_input(obs)[..., :self.critic_input_dim]
            c_out = self.critic_cnn(critic_obs).contiguous().view(obs.shape[0], -1)
            c_out = self.critic_mlp(c_out)
            return self.value_act(self.value(c_out))

        def get_aux_loss(self):
            # OI-Cm is frozen; there is no auxiliary online/target loss.
            return {}

    def build(self, name, **kwargs):
        return CmBuilder.Network(self.params, **kwargs)


class CmEffectBuilder(CmBuilder):
    """V1.15 actor-only Cmv2 token pooling over a raw 68+16×40+18 transport."""

    class Network(CmBuilder.Network):
        base_dim = 68
        token_count = 16
        token_dim = 40
        effect_dim = 18
        pooled_dim = 128
        raw_actor_input_dim = base_dim + token_count * token_dim + effect_dim
        effective_actor_input_dim = base_dim + pooled_dim + effect_dim

        def __init__(self, params, **kwargs):
            configured_actor_dim = int(params.get("actor_input_dim", self.raw_actor_input_dim))
            configured_critic_dim = int(params.get("critic_input_dim", self.base_dim))
            if configured_actor_dim != self.raw_actor_input_dim:
                raise ValueError(
                    f"Cmv2 actor transport must be {self.raw_actor_input_dim}, got {configured_actor_dim}")
            if configured_critic_dim != self.base_dim:
                raise ValueError(f"Cmv2 critic input must remain {self.base_dim}, got {configured_critic_dim}")
            if not bool(params.get("separate", False)):
                raise ValueError("Cmv2 actor-only context requires separate=true")
            super().__init__(params, **kwargs)
            self.raw_actor_input_dim = configured_actor_dim
            self.actor_input_dim = self.effective_actor_input_dim
            self._replace_first(self.actor_mlp, self.actor_input_dim)
            self.token_encoder = nn.Sequential(
                nn.Linear(self.token_dim, self.pooled_dim), nn.ELU(),
                nn.Linear(self.pooled_dim, self.pooled_dim), nn.ELU())
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.pooled_dim, nhead=4, dim_feedforward=self.pooled_dim * 2,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=False)
            self.token_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.effect_query = nn.Sequential(
                nn.Linear(self.base_dim + self.effect_dim, self.pooled_dim), nn.ELU(),
                nn.Linear(self.pooled_dim, self.pooled_dim))
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.pooled_dim, num_heads=4, dropout=0.0, batch_first=True)

        def _cm_input(self, obs):
            if obs.ndim != 2 or obs.shape[-1] < self.raw_actor_input_dim:
                raise ValueError(
                    f"Cmv2 actor observation must be [B,>={self.raw_actor_input_dim}], got {tuple(obs.shape)}")
            raw = obs[..., :self.raw_actor_input_dim]
            if not torch.isfinite(raw).all():
                raise FloatingPointError("Non-finite Cmv2 actor observation")
            base = raw[..., :self.base_dim]
            tokens = raw[..., self.base_dim:self.base_dim + self.token_count * self.token_dim].reshape(
                raw.shape[0], self.token_count, self.token_dim)
            effects = raw[..., -self.effect_dim:]
            valid = tokens[..., -1] > 0.5
            has_valid = valid.any(dim=1)
            # MHA disallows an all-masked sequence.  A zero dummy token preserves
            # a deterministic all-invalid representation while contributing no Cmv2 data.
            safe_valid = valid.clone()
            safe_valid[~has_valid, 0] = True
            encoded = self.token_encoder(tokens)
            encoded = encoded.masked_fill(~valid[..., None], 0.0)
            encoded = self.token_attention(encoded, src_key_padding_mask=~safe_valid)
            encoded = encoded.masked_fill(~valid[..., None], 0.0)
            query = self.effect_query(torch.cat((base, effects), dim=-1)).unsqueeze(1)
            pooled, _ = self.attention_pool(query, encoded, encoded,
                                            key_padding_mask=~safe_valid, need_weights=False)
            pooled = torch.where(has_valid[:, None], pooled[:, 0], torch.zeros_like(pooled[:, 0]))
            return torch.cat((base, pooled, effects), dim=-1)

    def build(self, name, **kwargs):
        return CmEffectBuilder.Network(self.params, **kwargs)
