# V1.21c multi-env noise allocation 合同修订

- timestamp: 2026-09-20T18:53:00+08:00
- activity_id: ACT-20260920-185300-CMRESIDUAL-V121C-NOISE-ALLOCATION
- work_version: V1.21
- git_commit: `c80b0a4350a2e3c257b6784d4c0709eb49552d5e`（核心合同）；`8e738999cc11515d33c5e1d33ef1e1b950abd966`（runner hard-gate 对齐）
- base_commit: `8cacd303f6f6c8f6cfeb473c6caa5dc11df13e00`
- branch: `ai/cmresidual/v121-cm-actor`
- mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户明确允许直接修改已 FINAL 的 V1.21c plan，并确认继续 Plan C、将多环境小差异视为已处理；该例外不构成以后覆盖 FINAL plan 的通用授权

## Scope

在已修复的一 action = 30 Hz = 两个 `1/60 s` PhysX substeps 合同上，V1.21c 不再把 9 个
并行 env 的 `q/dq/root` exact parity 当成 candidate step 的 hard gate。数值差异仍按原阈值完整
记录为 diagnostic；finite 与 task/reference indices 逐项相等仍是 hard gate。

每个 state 使用 `candidate_seed` 驱动固定 PCG64 permutation，把 candidate multiset
`[0,0,1,2,3,4,5,6,7]` 分配到 9 个 env。`env_candidate_ids[9]` 写入 physics replay schema，
candidate 0 所在的两个随机 slot 继续提供 duplicate anchor。这样避免 candidate identity 永久绑定
固定 env slot，但不声称解决了 PhysX 多环境数值分叉，也不把并行 env 称为 exact clones。

用户明确授权本次直接修订 `plan/V1.21c.md`；plan 内已记录例外范围、结论上限、门禁、schema、
验证和停止条件。没有恢复或新增 V1.21d。

## Protected

- observation、native action、reward、GT、坐标、单位、split、metric 与 checkpoint 解释不变；
- 30 Hz/two-substep 修复保持不变；
- 64-state calibration、duplicate object hard ceiling 与 `epsilon_PhysX` 公式不变；
- PPO、collector、512-state ranking、P=3、Cmv2 更新和旧 outputs 均未启动或修改。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py`：`20 passed`；
- `python3 -m py_compile`：prefix replay、artifact schema、smoke bootstrap 和 offline evaluator 通过；
- `python3 tools/verify.py --changed`：`VERIFY PASS`；
- `git diff --check`：通过。

本 Activity 只记录实现合同，没有运行 PhysX，因此 `run_status=N/A`、`conclusion=N/A`。

## Rollback

回退本 Activity 对应提交即可恢复此前固定 env 0..7/8 和 numeric-parity hard gate；不得回退
`8cacd30` 的 30 Hz physics-step 修复。旧 V1.21c parity-failure run 保留为历史证据。
