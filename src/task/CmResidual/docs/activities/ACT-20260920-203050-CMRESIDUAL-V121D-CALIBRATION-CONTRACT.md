# V1.21d duplicate calibration contract correction

- timestamp: 2026-09-20T20:30:50+08:00
- activity_id: ACT-20260920-203050-CMRESIDUAL-V121D-CALIBRATION-CONTRACT
- work_version: V1.21
- base_commit: `f8af204ed123eaba447f45293629b51eb6d31864`
- branch: `ai/cmresidual/v121-cm-actor`
- mode: change
- change_level: L2
- approval: covered by the user-approved [V1.21d FINAL plan](../plan/V1.21d.md)

## Scope and changed

按 V1.21d 修正 V1.21c duplicate calibration 的实现合同，但未启动 PhysX run：

- 在 ranking core 中集中实现 `t+6` object goal、`t+1` IG reference 的 next-state cost；正式
  PhysX score API 只接受一份无 candidate 维的 canonical collected pre-action pose/IG，禁止使用
  replay env 各自分叉后的 branch-local current state；
- 现有 64-state calibration artifact 不含完整 canonical `T_t/IG_t`，因此 duplicate calibration
  只比较统一 next-state cost，并以 `S_B-S_A=-(C_next_B-C_next_A)` 消去共享 baseline；
- calibration bootstrap 删除独立 `_loss()`，固定目标时钟、score producer 与 baseline metadata；
- duplicate launcher 固定使用 `dexplore_v117` symlink，manifest 同时记录 executable 原文、resolved
  target 与 target SHA256；
- frozen collection validation 提升为运行前 hard gate，覆盖 64 unique states、single batch、
  `24/24/16` phase composition、exact uint64 candidate seed、episode identity、frame/progress/reference
  对齐及 executed-action prefix SHA；
- physics record schema 新增 producer、`t+6/t+1` clock 和 shared-baseline provenance；offline evaluator
  只消费/校验 producer 写出的 `physics_score`，不从 pose/IG 重新评分。

旧 run `cmresidual_v121c_duplicate_calibration_gpu3_20260920_2000` 原样保留为
`FAILED / INVALID_IMPLEMENTATION` 证据；本次不把其 hard-ceiling failure 归因于 score bug、Python
runtime 或任何未隔离单一因素。

## Protected

30 Hz / two-substep `_physics_step()`、candidate-env deterministic permutation、numeric parity diagnostic、
task/reference index hard gate、duplicate position/rotation ceiling、epsilon 公式、candidate、Cmv2、坐标、
单位、GT、split、checkpoint 解释均未改变。vendor、`src/base/`、数据、cache、checkpoint 与既有 outputs
未修改或删除。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py`：`34 passed`；
- `python3 -m py_compile`：5 个受影响 Python 入口通过；
- frozen collection
  `cmresidual_v121c_calibration_collection_gpu3_seedfix_20260920_1950` 只读 hard-gate validation：
  `64` states、`24/24/16`、episodes `[0,1,2,3,4,5]`，通过；
- 用户随后明确授权将 V1.21d 指导、计划、Activity、实现与测试纳入同一提交；暂存完整提交范围后
  `python3 tools/verify.py --changed`：`VERIFY PASS`；
- `git diff --check`：通过。

## Conclusion, next step and rollback

本 Activity 只证明 corrected contract 与输入 validator 的工程接线；未运行 simulator，Cm 科学结论仍为
`INCONCLUSIVE`。下一步是在实现提交与 GPU3 资源预检后单独取得 run 确认，再以新 run_id 执行一次
corrected 64-state calibration。若再次超过任一 hard ceiling，接受 `INVALID_IMPLEMENTATION` 并停止。

回滚只撤回 V1.21d score/validator/runtime/schema/test 与本文档提交；不回退 30 Hz 修复、随机 env
分配、frozen collection、旧 run 或 V1.21c 既有证据。
