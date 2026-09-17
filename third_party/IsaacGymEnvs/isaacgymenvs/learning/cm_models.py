import torch

from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class ModelCmContinuous(ModelA2CContinuousLogStd):
    def build(self, config):
        net = self.network_builder.build('cm_actor_critic', **config)
        return ModelCmContinuous.Network(net, obs_shape=config['input_shape'],
                                         normalize_value=config.get('normalize_value', False),
                                         normalize_input=config.get('normalize_input', False),
                                         value_size=config.get('value_size', 1))


class ModelCmEffectContinuous(ModelA2CContinuousLogStd):
    """Cmv2 actor model whose RunningMeanStd is restricted to the 68-D base."""

    class Network(ModelA2CContinuousLogStd.Network):
        base_observation_dim = 68

        def __init__(self, a2c_network, **kwargs):
            # Do not construct the inherited full 726-D normalizer: token mask and
            # Cmv2 geometry channels have fixed semantics and must remain untouched.
            normalize_input = bool(kwargs.pop("normalize_input", False))
            super().__init__(a2c_network, normalize_input=False, **kwargs)
            self.normalize_input = normalize_input
            if self.normalize_input:
                self.running_mean_std = torch.jit.script(
                    RunningMeanStd((self.base_observation_dim,)))

        def norm_obs(self, observation):
            if not self.normalize_input:
                return observation
            if observation.ndim != 2 or observation.shape[-1] < self.base_observation_dim:
                raise ValueError("Cmv2 observation must contain the 68-D base prefix")
            with torch.no_grad():
                base = self.running_mean_std(observation[..., :self.base_observation_dim])
                return torch.cat((base, observation[..., self.base_observation_dim:]), dim=-1)

    def build(self, config):
        net = self.network_builder.build('cm_effect_actor_critic', **config)
        return ModelCmEffectContinuous.Network(
            net, obs_shape=config['input_shape'],
            normalize_value=config.get('normalize_value', False),
            normalize_input=config.get('normalize_input', False),
            value_size=config.get('value_size', 1))
