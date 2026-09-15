# CmResidual Task 入口

CmResidual 研究以几何重定向轨迹为 reference 的物理反馈 tracker，以及冻结该 tracker 后训练的独立
interaction residual。canonical owner 为 `src/task/CmResidual/`；IsaacGym vendor 路径只允许保留薄 adapter
和 Hydra 注册，不作为研究文档、数据合同或运行状态的事实源。

## 当前状态

- `modification_version`: `V1.3.2`
- 阶段：V1.3 contact/physical residual 修正已实现，zero-residual gate 未通过
- 计划状态：`final`（[V1.3 计划](plan/V1.3.md)）
- 执行审批：用户已批准 V1.3 代码修正和 CPU zero-residual gate；PPO pilot、并行训练和长训不在批准范围内
- 当前实现：两个注册入口均使用 32-D 冻结 OI-Cm、真实 checkpoint/reference 和 2005-D observation；
  DExplore contact 来自 net contact force；PPO residual 在 DExplore target 转换后施加
  0.015 m / 0.20 rad / 0.08 rad 的有界物理修正
- 当前 gate：4 env × 367 步 rollout 的 observation/target finite 且 zero residual target delta 为零，但
  `destroy_sim` 超过 7 分钟未返回，同时 contact occupancy 为 0、平均 tip distance 为 1.421636 m；
  run 标记为 `FAILED / INVALID_IMPLEMENTATION`，没有启动 PPO
- 执行边界：reference manifest 仍为 `training_eligible=false`；后续先诊断初始化/对齐与仿真清理问题，
  不把本次 apparent lift/success 或工程 smoke 解释为抓取效果

## 文档入口

- [V1.1 用户指导](指导/V1.1.md)
- [V1.1 最终执行计划](plan/V1.1.md)
- [V1.3 用户指导](指导/V1.3.md)
- [V1.3 最终执行计划](plan/V1.3.md)
- [活动记录](logs/activity_log.md)
- [实验记录](logs/experiment_log.md)

`D0` 已确认旧 reference 不能直接作为训练 GT：tensor columns `54:57` 的右腕 exp-map 与权威 global quaternion
不等价，且旧 18-DOF retarget 允许 wrist 逐帧补偿手指误差。计划现已改为读取 columns `51:54` + `309:313`
（`xyzw`），用整段唯一 `X_HB` 映射 wrist、只约束优化 q6 并生成新版本产物。主训练区间已固定为 indices
`[44,410]` 的 367 帧，完整 432 帧仅作诊断。当前执行仅限 D1；P0 及后续阶段仍待单独审批。
