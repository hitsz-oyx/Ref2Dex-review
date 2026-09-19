# V1.20 DDP facade 与 DExplore vendor snapshot 接入

- timestamp: 2026-09-19 17:14:54 +0800
- activity_id: ACT-20260919-171454-CMRESIDUAL-V120-DDP-VENDOR-INTEGRATION
- work_version: V1.20
- mode: governance → change
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认 V1.20a 的无 Horovod DDP 两阶段路线，并确认将 DExplore 固定为 `third_party/DExplore` vendor snapshot。
- branch: ai/cmresidual/ddp-adapter-plan
- base_commit: b7f0baec6e5d7e8f75465242c866379805dafa2f

## Scope

- [V1.20a DDP plan](../plan/V1.20a.md) — Task-local PyTorch/NCCL facade、两 rank 工程 smoke 与后续四 rank 门。
- [V1.20b vendor plan](../plan/V1.20b.md) — DExplore vendor snapshot 的来源、保护项与回滚。
- [DDP facade](../../tools/dexplore_ddp_compat.py)、[rank bootstrap](../../tools/dexplore_ddp_rank_bootstrap.py)、[launcher](../../tools/run_dexplore_v120_ddp.py)、[targeted tests](../../tests/test_dexplore_v120_ddp_compat.py)。
- [DExplore provenance](../../../../../third_party/DExplore/UPSTREAM.md) — source fork commit `c31f57f186409ce5f0de47ced2d347abffe45d06`、license 与 exclusions。

## Changed

- 新增只在 bootstrap 进程内注册的 Horovod-compatible facade，底层为 PyTorch `torch.distributed`/NCCL；不安装或导入 Horovod wheel，不修改 DExplore 或 `rl_games`。
- 新增 rank-local CUDA 映射、同步正常退出、rank-0 output 与显式 `torchrun` command builder；launcher 的真正执行模式要求明确 run id、motion root、input manifest 和 GPU 列表。
- 新增 `third_party/DExplore` squashed snapshot（约 76 MiB）；保留 NVIDIA license/NOTICE 与 tracked source/assets，不带 nested `.git`、未跟踪 `outputs/` 或 checkpoint。上游 tracked LFS checkpoint 已显式排除。
- launcher 默认 source 改为 repo-local `third_party/DExplore/dexplore/run.py`，但仍允许显式覆盖；尚未创建任何 DDP 运行目录或占用 GPU。

## Protected

`/home2/wyy/oyx_ws/dexplore` 外部 checkout 及其未跟踪 `outputs/`、现有 Python/Horovod 环境、数据、cache、checkpoint 和历史 outputs 均未移动、删除或修改。

## Verification

- `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_v120_ddp_compat.py src/task/CmResidual/tests/test_dexplore_v120_motion_input.py`：9 passed。
- DDP 两 rank CPU/Gloo 受控更新与单进程拼接 global batch 的 SGD 更新一致；launcher dry-run 解析为 repo-local vendor 的 `torch.distributed.run` 命令。
- `py_compile`、non-vendor `git diff --check`、`python3 tools/verify.py --changed` 通过；vendor 的所有保留 tracked 文件以 Git blob hash 对照固定 source commit（排除 vendor-local `.gitignore`），并确认无 `.git`、`outputs/`、`.pt` 或 `.pth`。
- vendor 原样保留上游 CRLF/尾随空白；对整个 vendor 运行 `git diff --check` 会报告 inherited formatting，故没有格式化或修改 snapshot 内容。

## Result and next step

工程实现为 `SUPPORTED`（仅 facade API、受控通信和 vendor provenance）。真实两 rank Isaac Gym smoke 尚未开始，科学结论为 `INCONCLUSIVE`。启动前仍需在 V1.20a 范围内锁定合法 motion root/input manifest、显式 checkpoint 策略和 GPU 资源快照。

## Rollback

回退本实现和 vendor 提交即可撤销 Task-local DDP 文件、vendor snapshot、计划与本 Activity；外部 checkout、环境、输入数据与历史输出不受影响。
