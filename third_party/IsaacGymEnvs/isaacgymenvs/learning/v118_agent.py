"""PPO agent with a fixed rollout-time Cm teacher target for V1.18."""
from __future__ import annotations

import torch
from torch import nn

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

from .common_agent import CommonAgent


class V118PlannerAgent(CommonAgent):
    """Keep planner targets in the rollout buffer and distil only into actor μ."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        # CommonAgent is a vendored A2C variant and does not initialize this
        # alias although its bootstrap critic path consumes it.  Keep the
        # standard rl_games normalizer selection local to V1.18.
        self.value_mean_std = (self.central_value_net.model.value_mean_std
                               if self.has_central_value else self.model.value_mean_std)

    def _load_config_params(self, config):
        super()._load_config_params(config)
        # This vendored CommonAgent predates rl-games renaming ``seq_len`` to
        # ``seq_length``.  Preserve its AMP dataset call without changing the
        # shared agent implementation.
        self.seq_len = self.seq_length
        self.cm_distill_coef = float(config.get("cm_distill_coef", 0.0))
        if self.cm_distill_coef < 0:
            raise ValueError("cm_distill_coef must be non-negative")

    def init_tensors(self):
        super().init_tensors()
        actions = self.experience_buffer.tensor_dict["actions"]
        values = self.experience_buffer.tensor_dict["values"]
        self.experience_buffer.tensor_dict["cm_teacher_actions"] = torch.zeros_like(actions)
        self.experience_buffer.tensor_dict["cm_teacher_weights"] = torch.zeros_like(values)
        self.tensor_list += ["cm_teacher_actions", "cm_teacher_weights"]

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        # rl_games only forwards its standard PPO fields into the dataset.
        # Teacher tensors are rollout-time fixed targets and must follow the
        # same shuffled minibatch indices as observations/actions.
        self.dataset.values_dict["cm_teacher_actions"] = batch_dict["cm_teacher_actions"]
        self.dataset.values_dict["cm_teacher_weights"] = batch_dict["cm_teacher_weights"]

    def _task(self):
        task = getattr(self.vec_env, "env", None)
        if task is None or not hasattr(task, "v118_teacher"):
            raise RuntimeError("V1.18 PPO requires the direct CmResidual V1.18 task wrapper")
        return task

    def play_steps(self):
        self.set_eval()
        epinfos = []
        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            for key in self.update_list:
                self.experience_buffer.update_data(key, n, res_dict[key])
            if self.cm_distill_coef > 0:
                teacher = self._task().v118_teacher(res_dict["mus"])
            else:
                teacher = {
                    "teacher_action": res_dict["mus"].detach(),
                    "teacher_weight": torch.zeros(res_dict["mus"].shape[0], device=res_dict["mus"].device,
                                                  dtype=res_dict["mus"].dtype),
                }
            self.experience_buffer.update_data("cm_teacher_actions", n, teacher["teacher_action"])
            self.experience_buffer.update_data("cm_teacher_weights", n, teacher["teacher_weight"].unsqueeze(-1))
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            self.experience_buffer.update_data("rewards", n, self.rewards_shaper(rewards))
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)
            terminated = infos.get("terminate") if isinstance(infos, dict) else None
            if terminated is None:
                terminated = self.dones
            terminated = terminated.float().unsqueeze(-1)
            next_vals = self._eval_critic(self.obs) * (1.0 - terminated)
            self.experience_buffer.update_data("next_values", n, next_vals)
            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)
            not_dones = 1.0 - self.dones.float()
            self.current_rewards *= not_dones.unsqueeze(1)
            self.current_lengths *= not_dones
        returns = self.discount_values(self.experience_buffer.tensor_dict["dones"].float(),
                                       self.experience_buffer.tensor_dict["values"],
                                       self.experience_buffer.tensor_dict["rewards"],
                                       self.experience_buffer.tensor_dict["next_values"])
        batch = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch["returns"] = a2c_common.swap_and_flatten01(returns + self.experience_buffer.tensor_dict["values"])
        batch["played_frames"] = self.batch_size
        return batch

    def calc_gradients(self, input_dict):
        self.set_train()
        obs = self._preproc_obs(input_dict["obs"])
        teacher_action = input_dict["cm_teacher_actions"].detach()
        teacher_weight = input_dict["cm_teacher_weights"].detach()
        batch = {"is_train": True, "prev_actions": input_dict["actions"], "obs": obs}
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch.update(rnn_states=input_dict["rnn_states"], seq_length=self.seq_len)
        clip = self.e_clip
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            result = self.model(batch)
            action_log_probs, values, entropy, mu, sigma = (result["prev_neglogp"], result["value"], result["entropy"],
                                                             result["mu"], result["sigma"])
            actor_info = self._actor_loss(input_dict["old_logp_actions"], action_log_probs,
                                          input_dict["advantages"], clip)
            actor_loss = actor_info["actor_loss"]
            critic_info = self._critic_loss(input_dict["old_values"], values, clip, input_dict["returns"], self.clip_value)
            critic_loss = critic_info["critic_loss"]
            bounds_loss = self.bound_loss(mu)
            distill_per_sample = teacher_weight.squeeze(-1) * (mu - teacher_action).square().mean(dim=-1)
            losses, _ = torch_ext.apply_masks([actor_loss.unsqueeze(1), critic_loss, entropy.unsqueeze(1),
                                                bounds_loss.unsqueeze(1), distill_per_sample.unsqueeze(1)], rnn_masks)
            actor_loss, critic_loss, entropy, bounds_loss, distill_loss = losses
            loss = (actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy +
                    self.bounds_loss_coef * bounds_loss + self.cm_distill_coef * distill_loss)
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for parameter in self.model.parameters():
                parameter.grad = None
        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer); self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer); self.scaler.update()
        else:
            self.scaler.step(self.optimizer); self.scaler.update()
        with torch.no_grad():
            kl = torch_ext.policy_kl(mu.detach(), sigma.detach(), input_dict["mu"], input_dict["sigma"], reduce_kl=not self.is_rnn)
        self.train_result = {"entropy": entropy, "kl": kl, "last_lr": self.last_lr, "lr_mul": 1.0,
                             "b_loss": bounds_loss, "cm_distill_loss": distill_loss,
                             "cm_teacher_weight": teacher_weight.mean()}
        self.train_result.update(actor_info)
        self.train_result.update(critic_info)
