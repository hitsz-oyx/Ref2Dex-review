# V1.21e.2 canonical snapshot 与短 warm-up sweep 实现

- timestamp: 2026-09-20T23:14:50+08:00
- activity_id: ACT-20260920-231450-CMRESIDUAL-V121E2-WARMUP-SWEEP
- work_version: V1.21
- base_commit: `5264603e18cb2cb2e92f0c00a479e226f2733d44`
- branch: `ai/cmresidual/v121e-snapshot-restore`
- mode: change
- change_level: L2
- approval: 用户确认 [V1.21e.2 FINAL 微补充](../plan/V1.21e.2.md)，包括 canonical
  snapshot-generation、full-prefix centroid target、稳定通过后缀判定和有限 REFUTED 边界

## Scope

实现单独的 canonical snapshot-generation pass：按 episode 从 frozen initial state 完整 replay 已保存
actions，在 6 个既有 state 的 `t-8/t-4/t-2/t-1/t` 截取 30 个 public snapshots，逐项计算 content
hash，聚合后设为只读；72 个 parity arms 只能消费该文件，runner 在每个 arm 后复核文件 SHA。
generation passes 不计入 arms，manifest 单独记录其 commit、命令、episode/action SHA 与每个 snapshot SHA。

每个 state 的 `full/0/1/2/4/8` 各运行两个独立 fresh single-env subprocess。producer 同时保存
candidate 前后 q/dq、actor roots、IG 及 candidate 后 physics score；`L>0` 只执行精确的
`actions[t-L:t]`，所有方法执行相同 `clip(policy_mu_t,-1,1)`。离线 evaluator 用两个 full arms 的
sign-aligned quaternion centroid/其它量算术 centroid 作为 target，candidate 前后均应用既有 noise floor
与 object hard ceiling。只有某个 L 及所有更长窗口在 6/6 states 均通过才报告最小可用 L；非单调结果为
`INCONCLUSIVE`，全失败只反驳 `warmup<=8 sufficient`。

未修改 Cm、observation、action、reward、GT、坐标、单位、split、score clock、shared canonical baseline、
阈值或 checkpoint 解释；未启动 512-state ranking、PPO 或 PhysX private-state 工作。

## Verification and rollback

- pinned DExplore runtime pytest：`15 passed`（V1.21e.1 + V1.21e.2 定向合同）；
- `py_compile`：V1.21e.2 evaluator、两个 bootstrap、runner 与 tests 通过；
- `python3 tools/verify.py --changed`：`VERIFY PASS`；
- `git diff --check`：通过。

当前只完成工程合同，scientific conclusion 为 `INCONCLUSIVE`。正式 GPU3 run 必须绑定本实现的干净
commit 与新 run_id；终态另写 Activity/experiment card。回滚为 revert 对应 Task-local implementation
commit，不删除或改写旧 outputs、source episodes、selected states、数据、cache 或 checkpoint。
