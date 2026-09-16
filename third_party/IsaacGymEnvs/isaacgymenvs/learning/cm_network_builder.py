"""rl_games network consuming frozen, simulator-generated OI-Cm context."""
from __future__ import annotations

from torch import nn
from rl_games.algos_torch import network_builder


class CmBuilder(network_builder.A2CBuilder):
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.cm_feature_dim = int(params.get("cm_feature_dim", 32))
            super().__init__(params, **kwargs)
            self.learn_sigma = bool(self.space_config.get("learn_sigma", True))
            if self.fixed_sigma and not self.learn_sigma:
                self.sigma.requires_grad_(False)
            obs_dim = kwargs.get("input_shape")[0]
            # The task appends the frozen OI-Cm context after the 1442-D teacher
            # observation.  Keep the full policy observation width unchanged here.
            self._replace_first(self.actor_mlp, obs_dim)
            if self.separate:
                self._replace_first(self.critic_mlp, obs_dim)

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
