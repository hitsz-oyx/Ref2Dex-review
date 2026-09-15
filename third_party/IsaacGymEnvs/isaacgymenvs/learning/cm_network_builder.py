"""rl_games network with a trainable Cm online feature and frozen target."""
from __future__ import annotations

import copy
import torch
from torch import nn
from rl_games.algos_torch import network_builder


class CmBuilder(network_builder.A2CBuilder):
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.cm_feature_dim = int(params.get("cm_feature_dim", 128))
            super().__init__(params, **kwargs)
            obs_dim = kwargs.get("input_shape")[0]
            self.cm_online = nn.Sequential(nn.Linear(obs_dim, self.cm_feature_dim), nn.ELU(), nn.Linear(self.cm_feature_dim, self.cm_feature_dim))
            self.cm_target = copy.deepcopy(self.cm_online)
            for parameter in self.cm_target.parameters():
                parameter.requires_grad_(False)
            self._replace_first(self.actor_mlp, obs_dim + self.cm_feature_dim * 2)
            if self.separate:
                self._replace_first(self.critic_mlp, obs_dim + self.cm_feature_dim * 2)

        @staticmethod
        def _replace_first(module: nn.Sequential, input_dim: int):
            for i, layer in enumerate(module):
                if isinstance(layer, nn.Linear):
                    module[i] = nn.Linear(input_dim, layer.out_features).to(layer.weight.device)
                    return
            raise RuntimeError("CmBuilder requires a non-empty MLP")

        @torch.no_grad()
        def _ema_target(self, decay=0.995):
            for target, online in zip(self.cm_target.parameters(), self.cm_online.parameters()):
                target.mul_(decay).add_(online, alpha=1.0 - decay)

        def _cm_input(self, obs):
            online = self.cm_online(obs)
            with torch.no_grad():
                self._ema_target()
                target = self.cm_target(obs)
            return torch.cat((obs, online, target.detach()), dim=-1)

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
            # The online feature receives PPO gradients through actor and critic;
            # target synchronization is explicit and has no extra reward term.
            return {}

    def build(self, name, **kwargs):
        return CmBuilder.Network(self.params, **kwargs)
