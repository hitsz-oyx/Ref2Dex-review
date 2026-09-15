# CmResidual Task 入口

CmResidual 研究以几何重定向轨迹为 reference 的物理反馈 tracker，以及冻结该 tracker 后训练的独立
interaction residual。canonical owner 为 `src/task/CmResidual/`；IsaacGym vendor 路径只允许保留薄 adapter
和 Hydra 注册，不作为研究文档、数据合同或运行状态的事实源。

## 当前状态

- `modification_version`: `V1.6`
- 阶段：V1.5 Gate A 与 V1.5.4 PPO wiring pilot 通过；V1.6 完成 corrected reference 训练准入
- 计划状态：`final`（[V1.6 计划](plan/V1.6.md)；历史 V1.5 计划保留）
- 执行审批：用户批准删除 `num_subscenes=1`，并完成 GPU5、seed42、64 env × 2 updates 单次 pilot
- 当前实现：32-D 冻结 OI-Cm、2005-D observation、真实输入校验、net contact force，以及
  `0.015 m / 0.20 rad / 0.08 rad` 有界 residual 保持不变；pilot 使用 canonical GPU PhysX subscene 配置
- 当前 gate：V1.4 zero-residual、V1.5 fixed nonzero residual 和 V1.5.4 PPO wiring 均为工程 `SUPPORTED`
- 当前诊断：`num_subscenes=1` 会在 GPU `prepare_sim` 触发 SIGSEGV；删除 override 后 1-env smoke 和 64 env × 2 updates 均正常
- 当前结论上限：pilot 只证明 PPO wiring、checkpoint 保存/重载和数值 finite；residual 增益、收敛和抓取科研效果仍为 `INCONCLUSIVE`
- 已确认 V1.3 主因是把原 DExplore 的空切片 `body_pos[..., 6:]` 误写成 body 维 remainder，制造了最高
  `188.4956 m/s` 的 reference 伪速度；同时 V1.4 对齐了 ReLU/RMS、scene reset 与 simulator 参数
- corrected reference 已切换到 `reference_tracking_v2/s1_airplane_lift`；source raw contact 与固定 URDF 表面距离门
  通过，manifest 为 `training_eligible=true`。旧 v1 保留为历史诊断证据
- 当前只证明 reference/data contract 具备训练准入；正式 PPO 的收敛、Cm 增益和抓取科研效果仍需独立训练与评估

## 文档入口

- [V1.1 用户指导](指导/V1.1.md)
- [V1.1 最终执行计划](plan/V1.1.md)
- [V1.3 用户指导](指导/V1.3.md)
- [V1.3 最终执行计划](plan/V1.3.md)
- [V1.4 用户指导](指导/V1.4.md)
- [V1.4 最终执行计划](plan/V1.4.md)
- [V1.5 用户指导](指导/V1.5.md)
- [V1.5 最终执行计划](plan/V1.5.md)
- [V1.6 用户指导](指导/V1.6.md)
- [V1.6 最终执行计划](plan/V1.6.md)
- [活动记录](logs/activity_log.md)
- [实验记录](logs/experiment_log.md)

V1.1 的 `D0` 已确认旧 reference 不能直接作为 corrected research GT：tensor columns `54:57` 的右腕 exp-map 与权威 global quaternion
不等价，且旧 18-DOF retarget 允许 wrist 逐帧补偿手指误差。计划现已改为读取 columns `51:54` + `309:313`
（`xyzw`），用整段唯一 `X_HB` 映射 wrist、只约束优化 q6 并生成新版本产物。主区间固定为 indices
`[44,410]` 的 367 帧，完整 432 帧仅作诊断。V1.4 只为发布 checkpoint parity 读取 legacy source，不改变该
corrected-reference 结论；V1.5 只允许显式 `ppo_pilot` 使用 legacy source，且失败未产生科研结论。
