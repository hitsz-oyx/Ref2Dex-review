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
    dexplore_run = Path("/home2/wyy/oyx_ws/dexplore/dexplore/run.py")
    sys.path.insert(0, str(dexplore_run.parent))
    _patch_common_agent_train()
    runpy.run_path(str(dexplore_run), run_name="__main__")


if __name__ == "__main__":
    main()
