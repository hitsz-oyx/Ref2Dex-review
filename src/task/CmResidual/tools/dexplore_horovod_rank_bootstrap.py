"""Run external DExplore with Isaac Gym's implicit tensor device bound to the Horovod rank."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

import numpy as np


# Isaac Gym's Python 3.8 bindings and DExplore still use these aliases.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int


def _patch_common_agent_train() -> None:
    """Restore synchronized Horovod shutdown without editing external DExplore."""
    import learning.common_agent as common_agent

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.frame = 0

        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        model_output_file = os.path.join(self.nn_dir, self.config["name"])

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        self._init_train()

        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info["total_time"]
            total_time += sum_time
            frame = self.frame
            if self.multi_gpu:
                self.hvd.sync_stats(self)

            should_exit = False
            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info["play_time"]
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        "epoch_num:{}".format(epoch_num),
                        "mean_rewards:{}".format(self._get_mean_rewards()),
                        f"fps step: {fps_step:.1f} fps total: {fps_total:.1f}",
                    )

                self.writer.add_scalar("performance/total_fps", curr_frames / scaled_time, frame)
                self.writer.add_scalar("performance/step_fps", curr_frames / scaled_play_time, frame)
                self.writer.add_scalar("info/epochs", epoch_num, frame)
                self._log_train_info(train_info, frame)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar("rewards{0}/frame".format(i), mean_rewards[i], frame)
                        self.writer.add_scalar("rewards{0}/iter".format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar("rewards{0}/time".format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar("episode_lengths/frame", mean_lengths, frame)
                    self.writer.add_scalar("episode_lengths/iter", mean_lengths, epoch_num)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if self.save_freq > 0 and epoch_num % self.save_freq == 0:
                    self.save(model_output_file)

                    if self._save_intermediate:
                        int_model_output_file = model_output_file + "_" + str(epoch_num).zfill(8)
                        self.save(int_model_output_file)

                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print("MAX EPOCHS NUM!")
                    should_exit = True

            if self.multi_gpu:
                should_exit_t = common_agent.torch.tensor(should_exit).float()
                self.hvd.broadcast_value(should_exit_t, "should_exit")
                should_exit = should_exit_t.bool().item()

            if should_exit:
                return self.last_mean_rewards, epoch_num

    common_agent.CommonAgent.train = train


def _patch_gradient_accumulation() -> None:
    """Accumulate 64 microbatches into one logical 16384-sample PPO update."""
    steps = int(os.environ.get("REF2DEX_GRAD_ACCUM_STEPS", "1"))
    if steps == 1:
        return
    if steps != 64:
        raise ValueError("V1.20 only supports 64x256 accumulation for logical minibatch 16384")
    import horovod.torch as hvd
    import torch
    from rl_games.distributed.hvd_wrapper import HorovodWrapper
    from torch import nn
    from rl_games.algos_torch import torch_ext
    import learning.dexplore_agent as dexplore_agent

    def setup_algo(self, algo):
        hvd.broadcast_parameters(algo.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(algo.optimizer, root_rank=0)
        algo.optimizer = hvd.DistributedOptimizer(
            algo.optimizer, named_parameters=algo.model.named_parameters(),
            backward_passes_per_step=steps,
        )
        self.sync_stats(algo)
    HorovodWrapper.setup_algo = setup_algo

    def calc_gradients(self, input_dict):
        self.set_train()
        total = input_dict['actions'].shape[0]
        if total != 16384:
            raise ValueError(f"expected logical minibatch 16384, got {total}")
        micro = total // steps
        mask_total = input_dict['rand_action_mask'].sum().clamp_min(1.0)
        self.optimizer.zero_grad()
        result = None
        for start in range(0, total, micro):
            end = start + micro
            part = {k: (v[start:end] if hasattr(v, 'shape') and v.ndim > 0 and v.shape[0] == total else v)
                    for k, v in input_dict.items()}
            values_old, logp_old = part['old_values'], part['old_logp_actions']
            advantage, returns, actions = part['advantages'], part['returns'], part['actions']
            old_mu, old_sigma = part['mu'], part['sigma']
            obs = self._preproc_obs(part['obs'])
            mask = part['rand_action_mask']
            batch = {'is_train': True, 'prev_actions': actions, 'obs': obs}
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                out = self.model(batch)
                a_info = self._actor_loss(logp_old, out['prev_neglogp'], advantage, self.e_clip)
                c_info = self._critic_loss(values_old, out['values'], self.e_clip, returns, self.clip_value)
                actor = (mask * a_info['actor_loss']).sum() / mask_total
                critic = c_info['critic_loss'].sum() / total
                entropy = (mask * out['entropy']).sum() / mask_total
                bound = (mask * self.bound_loss(out['mus'])).sum() / mask_total
                loss = actor + self.critic_coef * critic + self.bounds_loss_coef * bound
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite accumulated PPO loss at microbatch {start // micro}")
            self.scaler.scale(loss).backward()
            with torch.no_grad():
                kl = torch_ext.policy_kl(out['mus'].detach(), out['sigmas'].detach(), old_mu, old_sigma, True)
                result = {'entropy': entropy.detach(), 'kl': kl.detach(), 'last_lr': self.last_lr,
                          'lr_mul': 1.0, 'b_loss': bound.detach(), 'actor_loss': actor.detach(),
                          'actor_clip_frac': (mask * a_info['actor_clipped'].float()).sum().div(mask_total).detach(),
                          'critic_loss': critic.detach()}
        self.scaler.unscale_(self.optimizer)
        self.optimizer.synchronize()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        with self.optimizer.skip_synchronize():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.train_result = result

    dexplore_agent.DexploreAgent.calc_gradients = calc_gradients


def main() -> None:
    local_rank = int(os.environ.get(
        "HOROVOD_LOCAL_RANK", os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    ))

    # Import Isaac Gym before Torch.  Its stock helper defaults device='cuda:0',
    # which crosses devices on nonzero Horovod ranks.  Explicit device callers
    # retain their requested device; only omitted devices become rank-local.
    from isaacgym import torch_utils

    torch = torch_utils.torch
    torch.cuda.set_device(local_rank)
    original_to_torch = torch_utils.to_torch

    def rank_local_to_torch(x, dtype=torch.float, device=None, requires_grad=False):
        if device is None:
            device = f"cuda:{local_rank}"
        return original_to_torch(x, dtype=dtype, device=device, requires_grad=requires_grad)

    torch_utils.to_torch = rank_local_to_torch
    dexplore_run = Path(os.environ.get(
        "REF2DEX_DEXPLORE_RUN",
        "/home2/wyy/oyx_ws/dexplore/dexplore/run.py",
    ))
    sys.path.insert(0, str(dexplore_run.parent))
    _patch_common_agent_train()
    _patch_gradient_accumulation()
    runpy.run_path(str(dexplore_run), run_name="__main__")


if __name__ == "__main__":
    main()
