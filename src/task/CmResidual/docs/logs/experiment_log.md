# CmResidual 实验记录

- scope: `task:CmResidual`
- related: [任务入口](../README.md)、[V1.3 最终计划](../plan/V1.3.md)、[活动记录](activity_log.md)

## 2026-09-15 — V1.3.2 zero-residual gate

- modification_version: `V1.3.2`
- operation_category: `experiment`、`operation`
- approval: `user-approved`
- run_id: `cmresidual_zero_v13_20260915_164405`
- run_status: `FAILED`
- base_commit: `95d5aea25a8d75306982d4266d1040911fd25efa`
- seed / budget: `42` / `4 envs × 367 control steps`，CPU PhysX
- initial_checkpoint: DExplore `inspire.pth` 与冻结 OI-Cm `best.pt`，确切路径及 SHA256 见
  [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/run_manifest.json)
- checkpoint: `null`（本次不构造或训练 PPO）
- last_step: `367`
- best_metric: `max_lift_m=0.0885452628`（仅为 rollout 观测值，不是有效抓取成绩）

**问题与冻结条件**

本 gate 检查冻结 DExplore 在严格全零 residual 下能否保持 base-only physical target 映射、完成有限的
Isaac Gym rollout，并产生可解释的接触与物体行为。保持 `s1/airplane_lift` reference、30 Hz、reward、
OI-Cm tokens/anchors/effect、DExplore checkpoint 和全部模型参数不变。reference manifest 明确为
`training_eligible=false`，本次只允许诊断性评估。

**结果**

- 367 行逐步 [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/metrics.jsonl)
  已完整写入；所有 observation 与 physical target finite。
- `residual_action_abs_max=0`、`residual_target_delta_max=0`，三组 physical residual RMS 均为 0，
  `target_saturation_ratio=0`。这支持 zero-residual wiring 精确退化到 frozen base target 的局部工程不变量。
- mean / max lift 为 `0.0697425 / 0.0885453 m`，累计 reset 数为 4，最终内部 success rate 为 1.0；但
  contact occupancy 全程为 0，mean tip distance 为 `1.421636 m`。无手部接触时的 apparent lift/success
  不能解释为抓取，反而说明初始化、reference 对齐或物体/地面动力学仍存在阻断性问题。
- rollout 步数完成后，Isaac Gym `destroy_sim` 超过 7 分钟仍未返回并持续异常占用 CPU；进程最终以
  `SIGTERM` 停止。终态 manifest 由停止操作补记为 `FAILED`，不把完整 metrics 等同于正常退出。

**结论**

- `SUPPORTED`：单测与 rollout 都支持“zero residual 不改变 DExplore physical target”，且本次状态张量 finite。
- `INVALID_IMPLEMENTATION`：完整 gate 未满足正常退出和合理接触行为；不能据 apparent lift/success 宣称
  DExplore 抓取有效。
- `INCONCLUSIVE`：Cm residual 的可学习性、PPO 收敛、抓取效果及科研假设均未检验。

按 [V1.3 最终计划](../plan/V1.3.md) 的停止条件，本次没有启动 PPO。下一步需先经用户确认，将初始化/reference
对齐与 `destroy_sim` 清理分别作为诊断范围，再决定是否修正和复跑 gate。运行日志见
[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/eval.log)。
