# CmResidual Task 入口

CmResidual 研究以几何重定向轨迹为 reference 的物理反馈 tracker，以及冻结该 tracker 后训练的独立
interaction residual。canonical owner 为 `src/task/CmResidual/`；IsaacGym vendor 路径只允许保留薄 adapter
和 Hydra 注册，不作为研究文档、数据合同或运行状态的事实源。

## 当前状态

- `modification_version`: `V1.4.2`
- 阶段：V1.4 DExplore actor/observation/reference/reset/simulator parity 已实现，zero-residual gate 通过
- 计划状态：`final`（[V1.4 计划](plan/V1.4.md)）
- 执行审批：用户已批准 V1.4 上游基线修正和 CPU zero-residual gate；PPO pilot、并行训练和长训不在批准范围内
- 当前实现：两个注册入口均使用 32-D 冻结 OI-Cm、真实 checkpoint/reference 和 2005-D observation；
  DExplore contact 来自 net contact force；PPO residual 在 DExplore target 转换后施加
  0.015 m / 0.20 rad / 0.08 rad 的有界物理修正
- 当前 gate：4 env × 367 步 rollout 正常完成，reset 最大位置误差 `2.09e-7 m`，zero residual target delta
  始终为 0；第 16 个 source 接触帧平均 tip distance=`0.045808 m`、contact occupancy=`0.65`，全程平均
  contact occupancy=`0.48079`，工程门禁为 `SUPPORTED`
- 已确认 V1.3 主因是把原 DExplore 的空切片 `body_pos[..., 6:]` 误写成 body 维 remainder，制造了最高
  `188.4956 m/s` 的 reference 伪速度；同时 V1.4 对齐了 ReLU/RMS、scene reset 与 simulator 参数
- 执行边界：corrected reference manifest 仍为 `training_eligible=false`；本次结果只支持 frozen DExplore
  基线工程可用，不代表 Cm、PPO 或抓取科研效果成立

## 文档入口

- [V1.1 用户指导](指导/V1.1.md)
- [V1.1 最终执行计划](plan/V1.1.md)
- [V1.3 用户指导](指导/V1.3.md)
- [V1.3 最终执行计划](plan/V1.3.md)
- [V1.4 用户指导](指导/V1.4.md)
- [V1.4 最终执行计划](plan/V1.4.md)
- [活动记录](logs/activity_log.md)
- [实验记录](logs/experiment_log.md)

V1.1 的 `D0` 已确认旧 reference 不能直接作为 corrected research GT：tensor columns `54:57` 的右腕 exp-map 与权威 global quaternion
不等价，且旧 18-DOF retarget 允许 wrist 逐帧补偿手指误差。计划现已改为读取 columns `51:54` + `309:313`
（`xyzw`），用整段唯一 `X_HB` 映射 wrist、只约束优化 q6 并生成新版本产物。主区间固定为 indices
`[44,410]` 的 367 帧，完整 432 帧仅作诊断。V1.4 只为发布 checkpoint parity 读取 legacy source，不改变该
corrected-reference 结论；PPO pilot 及后续阶段仍待单独审批。
