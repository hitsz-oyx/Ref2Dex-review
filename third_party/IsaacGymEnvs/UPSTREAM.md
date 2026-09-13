# IsaacGymEnvs 依赖说明

本目录是 Isaac Sim 官方 `IsaacGymEnvs` 的 vendor snapshot，作为 Ref2Dex 的第三方依赖维护。

- upstream: `https://github.com/isaac-sim/IsaacGymEnvs`
- release: `release/1.5.1`
- pinned commit: `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`
- import mode: squashed snapshot，未携带上游 `.git` 目录
- local task integration: `isaacgymenvs/tasks/cm_residual.py`、`isaacgymenvs/tasks/__init__.py`、`isaacgymenvs/cfg/task/CmResidual*.yaml`、`isaacgymenvs/cfg/train/CmResidualPPO.yaml`

`runs/`、日志、缓存和训练输出由上游 `.gitignore` 及 Ref2Dex 运行规范排除，不应提交到主仓库。上游同步时先在外部 mirror 中完成版本核对，再更新本目录并重新运行 CmResidual smoke。
