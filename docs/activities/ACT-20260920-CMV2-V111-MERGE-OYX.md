# ACT-20260920-CMV2-V111-MERGE-OYX — 集成 Cmv2 V1.11 工作

timestamp: 2026-09-20T13:06:06+00:00
activity_id: ACT-20260920-CMV2-V111-MERGE-OYX
work_version: V1.2h
base_commit: e9cda6c6daf19aff8881d5a226c7575476876526
git_commit: 1803098925840ebb3197f8a9e50b1ad0b2c1cb3b
branch: oyx
scope: ObjectInteractionCm/ObjectInteractionCmv2 implementation and stable-branch integration
change_level: L3
approval: user-approved
approval_basis: 用户明确要求提交当前全部更改，合并到 `oyx`，再从稳定基线创建 V1.12 开发分支。

## Changes

- 将当前工作树 29 个文件提交为 `77569c7`，包括 OakInk2 MANO viewer 兼容、V1.11 五组混合训练入口、配置、测试、plan、Activity 和实验卡。
- 使用 `--no-ff` 将 `cmv2` 合并到稳定分支 `oyx`，生成 merge commit `1803098`；无文本冲突。
- 将用户新建的 `src/task/ObjectInteractionCmv2/docs/指导/V1.12.md` 作为后续开发输入纳入稳定基线；当前 Task 指针仍为 `V1.11.1`。

## Protected

- 未停止、重启或改写任何训练服务、NAS output、cache 或 checkpoint。
- 未更改 observation、action、GT、坐标/单位、split、metric 或 checkpoint 解释。

## Verification

- `/home/wbcd/miniconda3/envs/graspenv/bin/python tools/verify.py --changed`：`VERIFY PASS`。
- ObjectInteractionCm viewer 和 ObjectInteractionCmv2 混合训练定向测试：`21 passed`。
- `git diff --check`：通过；合并时无未解决冲突。

## Rollback

在保留后续工作的前提下反转 merge commit `1803098`可撤销本次稳定分支集成；运行产物不受 Git 回滚影响。
