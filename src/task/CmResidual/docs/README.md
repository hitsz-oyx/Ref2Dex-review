# CmResidual Task 入口

CmResidual 研究以几何重定向轨迹为 reference 的物理反馈 tracker，以及冻结该 tracker 后训练的独立
interaction residual。canonical owner 为 `src/task/CmResidual/`；IsaacGym vendor 路径只允许保留薄 adapter
和 Hydra 注册，不作为研究文档、数据合同或运行状态的事实源。

## 当前状态

- `modification_version`: `V1.0.1`
- 阶段：`D1` reference artifact 与离线合同
- 计划状态：`final`
- 方向审批：用户已确认 canonical owner、`s1/airplane_lift` 单轨迹范围、累计 base residual、联合 RSI、
  gate/pilot 优先和长训另行批准；并确认旧 sidecar 不覆盖、使用固定 wrist SE(3) 映射且逐帧只优化 q6 的
  constrained retarget 新 reference 方向
- 执行审批：用户已批准 367 帧主区间、plan 定稿及 D1 builder/new artifact/offline validation
- 当前状态：已确认采用锁定 URDF 的 pinky mimic `1.18`；legacy `1.05` 仅保留历史兼容，D1 artifact 已生成；因 raw contact flag 缺失，整体结论为 INCONCLUSIVE，尚不可进入物理 gate
- 执行边界：尚未批准 Isaac Gym、P0/P1/P2、PPO、pilot 或长训；新 reference 只有 D1 全门通过后才可申请进入
  物理 gate

## 文档入口

- [V1.0 用户指导](指导/V1.0.md)
- [V1.0 最终执行计划](plan/V1.0.md)
- [活动记录](logs/activity_log.md)

`D0` 已确认旧 reference 不能直接作为训练 GT：tensor columns `54:57` 的右腕 exp-map 与权威 global quaternion
不等价，且旧 18-DOF retarget 允许 wrist 逐帧补偿手指误差。计划现已改为读取 columns `51:54` + `309:313`
（`xyzw`），用整段唯一 `X_HB` 映射 wrist、只约束优化 q6 并生成新版本产物。主训练区间已固定为 indices
`[44,410]` 的 367 帧，完整 432 帧仅作诊断。当前执行仅限 D1；P0 及后续阶段仍待单独审批。
