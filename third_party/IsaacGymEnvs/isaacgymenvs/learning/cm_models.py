from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelCmContinuous(ModelA2CContinuousLogStd):
    def build(self, config):
        net = self.network_builder.build('cm_actor_critic', **config)
        return ModelCmContinuous.Network(net, obs_shape=config['input_shape'],
                                         normalize_value=config.get('normalize_value', False),
                                         normalize_input=config.get('normalize_input', False),
                                         value_size=config.get('value_size', 1))
