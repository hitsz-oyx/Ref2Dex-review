"""Launch unmodified DExplore through the Task-local torch.distributed facade."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

import numpy as np


if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int


DEFAULT_DEXPLORE_RUN = str(Path(__file__).resolve().parents[4] / "third_party/DExplore/dexplore/run.py")


def _patch_synchronized_shutdown() -> None:
    """Make the external rank-0 termination decision visible to every rank."""
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
                scaled_play_time = train_info["play_time"]
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    print("epoch_num:{}".format(epoch_num), "mean_rewards:{}".format(self._get_mean_rewards()),
                          f"fps step: {curr_frames / scaled_play_time:.1f} fps total: {curr_frames / sum_time:.1f}")
                self.writer.add_scalar("performance/total_fps", curr_frames / sum_time, frame)
                self.writer.add_scalar("performance/step_fps", curr_frames / scaled_play_time, frame)
                self.writer.add_scalar("info/epochs", epoch_num, frame)
                self._log_train_info(train_info, frame)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()
                    for index in range(self.value_size):
                        self.writer.add_scalar("rewards{0}/frame".format(index), mean_rewards[index], frame)
                        self.writer.add_scalar("rewards{0}/iter".format(index), mean_rewards[index], epoch_num)
                        self.writer.add_scalar("rewards{0}/time".format(index), mean_rewards[index], total_time)
                    self.writer.add_scalar("episode_lengths/frame", mean_lengths, frame)
                    self.writer.add_scalar("episode_lengths/iter", mean_lengths, epoch_num)
                    if self.has_self_play_config:
                        self.self_play_manager.update(self)
                if self.save_freq > 0 and epoch_num % self.save_freq == 0:
                    self.save(model_output_file)
                    if self._save_intermediate:
                        self.save(model_output_file + "_" + str(epoch_num).zfill(8))
                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print("MAX EPOCHS NUM!")
                    should_exit = True

            if self.multi_gpu:
                flag = common_agent.torch.tensor(should_exit, dtype=common_agent.torch.float32)
                self.hvd.broadcast_value(flag, "ref2dex_should_exit")
                should_exit = bool(flag.item())
            if should_exit:
                return self.last_mean_rewards, epoch_num

    common_agent.CommonAgent.train = train


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dexplore-run", default=os.environ.get("REF2DEX_DEXPLORE_RUN", DEFAULT_DEXPLORE_RUN))
    args, passthrough = parser.parse_known_args(argv)
    dexplore_run = Path(args.dexplore_run).resolve()
    if not dexplore_run.is_file():
        raise FileNotFoundError(f"missing DExplore entrypoint: {dexplore_run}")

    # Isaac Gym must import Torch first in this runtime.  The facade lazily
    # imports torch only after this point.
    from isaacgym import torch_utils

    local_rank = int(os.environ["LOCAL_RANK"])
    torch_utils.torch.cuda.set_device(local_rank)
    original_to_torch = torch_utils.to_torch

    def rank_local_to_torch(x, dtype=torch_utils.torch.float, device=None, requires_grad=False):
        if device is None:
            device = f"cuda:{local_rank}"
        return original_to_torch(x, dtype=dtype, device=device, requires_grad=requires_grad)

    torch_utils.to_torch = rank_local_to_torch
    from dexplore_ddp_compat import cleanup, initialize_from_env, install_horovod_facade

    initialize_from_env()
    install_horovod_facade()
    sys.path.insert(0, str(dexplore_run.parent))
    _patch_synchronized_shutdown()
    sys.argv = [str(dexplore_run)] + passthrough
    try:
        runpy.run_path(str(dexplore_run), run_name="__main__")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
