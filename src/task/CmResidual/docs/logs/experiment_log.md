# CmResidual 实验记录

- scope: `task:CmResidual`
- related: [任务入口](../README.md)、[V1.4 最终计划](../plan/V1.4.md)、[V1.3 最终计划](../plan/V1.3.md)、[活动记录](activity_log.md)

## 2026-09-15 — V1.4.2 DExplore parity 与 zero-residual gate

- modification_version: `V1.4.2`
- operation_category: `experiment`、`operation`
- approval: `user-approved`（[V1.4 最终计划](../plan/V1.4.md)）
- activity_id: `ACT-20260915-191707-CMRESIDUAL-V142-ZERO-FINAL`
- run_id: `cmresidual_zero_v14_final_20260915_191707`
- run_status: `COMPLETED`
- base_commit: `b1dac1c7955c9b960481279425f9e4e2e2269a98`
- seed / budget: `42` / `4 envs × 367 control steps`，CPU PhysX
- frozen inputs: DExplore `inspire.pth`、OI-Cm `best.pt`、legacy source tensor 与 corrected reference 的路径和 SHA256 见
  [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/run_manifest.json)
- initial_checkpoint: `null`（评估不恢复 CmResidual PPO）
- checkpoint: `null`（本次不构造或训练 PPO）
- last_step: `367`
- last_epoch: `null`
- best_metric: `max_lift_m=0.2123541832`（gate 行为观测量，不是独立抓取 benchmark）

**假设与冻结条件**

本 gate 只检验：按发布源码重建 DExplore actor、721D observation、legacy reference/reset、simulator 参数和
base target 后，严格全零 residual 能否退化为可运行的 frozen DExplore baseline。corrected reference 继续保持
`training_eligible=false`；reward 数值、OI-Cm、residual 权限、模型权重与 checkpoint 均冻结，并关闭
success-triggered reset 以完整覆盖 367 帧。不检验 Cm residual 的可学习性，也不启动 PPO。

**实现诊断与中间失败**

- 首次正式运行 `cmresidual_zero_v14_20260915_184916` 虽完成 367 步、finite/reset/zero-target 条件，但在
  source 第 16 个接触帧没有近物或接触，判为 `INVALID_IMPLEMENTATION`；证据保留在
  [失败运行 manifest](../../../../../outputs/CmResidual/cmresidual_zero_v14_20260915_184916/run_manifest.json)。
- 对未经修改的 DExplore 首帧 state/reference 做真实输入对照，定位到 provider 错误解释源码
  `body_pos[..., 6:]`：末维只有 3，因此该切片实际为空且应为 no-op。错误实现却沿 body 维对 10 个关键点
  取 remainder，生成最高 `188.4956 m/s` 的伪 reference 速度并破坏 object-surface IG。
- 修正后 20 步单环境 [reference smoke](../../../../../outputs/CmResidual/cmresidual_zero_v14_reference_smoke_20260915_191608/run_manifest.json)
  在第 16 步得到 tip distance=`0.045783 m`、实测 contact occupancy=`0.6`，随后才执行最终正式 gate。

**最终结果**

- 初始 native q/wrist/object/table 最大误差分别为 `0`、`2.09e-7 m`、`1.30e-7 m`、`7.45e-8 m`；
  367 步无 reset，observation、target 与 object pose 全程 finite。
- `residual_target_delta=0`、`target_saturation_ratio=0`。第 16 步 tip distance=`0.045808 m`、实测/reference
  contact occupancy=`0.65/0.4`；全程平均/最大 contact occupancy=`0.480790/0.75`，最小 tip distance=
  `0.037769 m`。
- `success_fraction` 从第 34 步出现，最大为 `1.0`、末步为 `0.75`；由于 gate 禁用 success reset、没有完成
  episode，manifest 内 legacy `final_success_rate=0` 不代表逐环境成功状态为零。
- 完整证据：[config.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/config.json)、
  [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/metrics.jsonl)、
  [eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/eval.log)。

**结论**

- `SUPPORTED`（工程）：真实 source 对照、reset、zero-target、finite、接触行为和正常进程终态共同支持
  “不加 residual 时，当前实现可正常执行 frozen DExplore baseline”。
- `INCONCLUSIVE`（科研）：mean/max lift=`0.079841/0.212354 m` 只是同一 gate 内的行为观测，不是独立
  benchmark；Cm residual 增益、PPO 收敛和抓取效果尚未检验。

按 [V1.4 最终计划](../plan/V1.4.md) 的边界，本轮没有启动 PPO，也没有修改 corrected reference、reward、
checkpoint 或外部 DExplore 源码。

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
