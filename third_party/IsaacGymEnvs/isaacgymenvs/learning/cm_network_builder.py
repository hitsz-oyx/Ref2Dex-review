"""rl_games network consuming frozen, simulator-generated OI-Cm context."""
from __future__ import annotations

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
