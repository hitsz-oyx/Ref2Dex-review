# V1.21c episode-prefix replay branch adapter

- timestamp: 2026-09-20 15:36:00 +0800
- activity_id: ACT-20260920-153600-CMRESIDUAL-V121C-PREFIX-REPLAY
- work_version: V1.21
- mode: change
- change_level: L1 (Task-local implementation under the approved V1.21c L2 plan)
- approval: user-approved
- approval_basis: V1.21c FINAL plan approval plus the user's explicit “继续做”.
- branch: ai/cmresidual/v121-cm-actor
- base_commit: 5fe0b9f8a0c38ead8b07101ef37bcd9b45b2df79

## Scope

新增不依赖 Isaac Gym 的 prefix-replay protocol adapter；实际 DExplore runtime 只需提供创建
9 个 validation env、从 episode initial state 恢复、批量 native-action step 和公开状态快照。
adapter 固定执行 recorded `executed_action_history[:frame_id]`，在 candidate step 前检查
q/dq、actor root states 和 task/reference indices 的公开状态 parity，并随后执行 candidates
`0..7` 加 candidate-0 duplicate anchor。

## Changed

- 新增 [v121c_prefix_replay.py](../../v121c_prefix_replay.py)：runtime protocol、公开状态
  schema、9-branch prefix replay 与 `1e-5` parity gate；parity 失败会返回 invalid，且不会执行
  candidate step 或隐式 reset。
- 扩展 [test_v121c_ranking.py](../../tests/test_v121c_ranking.py)：fake runtime 验证 executed
  prefix 的重放顺序、8+1 candidate 排列/duplicate，以及 parity failure 的停止边界。
- 更新 Task README 与 Activity index；没有创建 outputs、cache、数据或运行 manifest。

## Protected

保持 DExplore 1442-D observation、(T,598) motion source、18-D native action、H_ref=6、坐标、
单位、GT、split、checkpoint 解释和 frozen Cmv2 语义不变；未修改 vendor、`src/base`、数据、cache、
checkpoint、既有 outputs、用户指导或最终计划。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py src/task/CmResidual/tests/test_dexplore_cm_geometry.py src/task/CmResidual/tests/test_cmv2_adapter.py`: `22 passed`。
- `python3 -m py_compile src/task/CmResidual/v121c_prefix_replay.py`: 通过。
- `git diff --check`: 通过。
- 未运行 `python3 tools/verify.py --changed`：该统一入口在上一 Activity 已确认会因用户保留的
  未跟踪指导 `指导/V1.21a.md`、`指导/V1.21c.md` 的链接检查失败；本次没有改变它们或该已知条件。

## Scientific conclusion and next step

这是 prefix replay/branch 接线和 fake-tensor smoke，不构成 PhysX ranking 证据，科学结论仍为
`INCONCLUSIVE`。下一步是在不改变本 protocol 的条件下实现官方 DExplore runtime adapter，并在
单 episode / 1-state / 9-env duplicate smoke 通过后，另行申请正式 V1.21c ranking run。

## Rollback

回退本 Activity 对应的 Task-local 提交即可移除 adapter 与测试；不删除或覆盖用户指导、vendor、
数据、cache、checkpoint、既有 outputs 或历史记录。
