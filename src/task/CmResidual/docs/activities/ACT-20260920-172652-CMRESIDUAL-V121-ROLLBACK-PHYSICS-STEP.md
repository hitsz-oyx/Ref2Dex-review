# V1.21d 整体回退与 DExplore 30 Hz replay step 修复

- timestamp: 2026-09-20T17:26:52+08:00
- activity_id: ACT-20260920-172652-CMRESIDUAL-V121-ROLLBACK-PHYSICS-STEP
- work_version: V1.21
- mode: change
- change_level: L2（删除已定稿 V1.21d 路线与运行记录）；30 Hz Task-local bug fix 本身为 L1
- approval: user-approved
- approval_basis: 用户明确要求 V1.21d 计划、实现、运行文档和输出一起删除并整体回退，只修 physics step，不再处理多环境不同步或继续该 ranking 路线
- branch: ai/cmresidual/v121-cm-actor
- base_commit: 5be0988

## Scope and rollback

通过可审计的 Git revert 撤销以下三个已推送提交，未改写分支历史：

- `ac0dc20`：V1.21d Gate 0 Activity、experiment 与 README 状态；
- `4607422`：fresh-sim backend、schema、runner、tests、实现 Activity 与 V1.21d 定稿变化；
- `4b7bd0a`：V1.21d plan 草案和文档入口。

对应 revert commits 为 `62743c8`、`6257fc5`、`5be0988`。当前树不再包含
`plan/V1.21d.md`、任何 `v121d_*` 代码/runner/test，或 V1.21d Activity/experiment。

唯一 V1.21d run 目录
`outputs/CmResidual/cmresidual_v121d_native_init_gate0_gpu3_20260920_1645/`（约 `52K`）已按用户明确
授权移入系统回收站，当前输出路径不存在；没有删除或覆盖 V1.21c 及更早 outputs。

## Physics step fix

修正 `DExploreTaskPrefixRuntime.step()`：不再直接调用一次 `gym.simulate()`，而是校验
`control_freq_inv == 2` 后调用官方 DExplore `BaseTask._physics_step()`。官方方法为同一个 native action
执行两次 `1/60 s` simulate，随后 adapter 仅 fetch 一次、调用一次 `post_physics_step()`，因此一个 action
与 reference/progress 均保持 `30 Hz`，和 Cm `dt=1/30` 一致。

该修复不改变 PPO 的官方 step 路径；PPO 原本就使用 `BaseTask.step()` / `_physics_step()`。没有修改
observation、action mapping、reward、GT、坐标、单位、split、metric、checkpoint 或 frozen Cmv2。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py`：`18 passed`；新增合同测试固定
  one pre / two simulate / one fetch / one post，并拒绝 `control_freq_inv != 2`。
- `python3 -m py_compile src/task/CmResidual/v121c_prefix_replay.py`：通过。
- `python3 tools/verify.py --changed`：通过。
- `git diff --check`：通过。
- 未运行 GPU/PhysX、collector、ranking 或 PPO。

## Scientific conclusion and next step

这是实现时间尺度修复和路线回退，不产生新的物理或 Cm 证据。既有 V1.21c 9-env smoke 只作为历史失败
事实保留，不再把跨 env exact parity 当作需要解决的 PPO 问题，也不继续当前 Cm ranking validation。
Cm ranking、抓取效果与策略结论均保持 `INCONCLUSIVE`。下一步回到用户另行指定的研究任务。

## Rollback

如需恢复被撤销的 tracked V1.21d 内容，可在新审批下 revert `5be0988`、`6257fc5`、`62743c8`；
physics step 修复可单独回退本 Activity 对应实现提交。V1.21d ignored run 只能在系统回收站尚未清理时
人工恢复，Git 不包含该输出。
