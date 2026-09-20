# ObjectInteractionCmv2 文档入口

- [V1.0 指导](指导/V1.0.md)：用户确认的研究目标、边界与禁止事项；
- [V1.0 架构](architecture/V1.0.md)：当前版本冻结的输入、结构化交互路径、输出与不变量；
- [V1.0 计划](plan/V1.0.md)：实现边界、阶段、验证与回滚；当前为 `final`，已作为 V1.0 实现与小规模 pilot 的审批闸门；
- [V1.2 指导](指导/V1.2.md)：swept 手点流交互的研究方向；本轮用户修订见对应计划。
- [V1.2 最终计划](plan/V1.2.md)：GRAB-only、MANO-only、固定 stride=1 的实施和有界验证范围。
- [V1.3 指导](指导/V1.3.md)：空间交互融合与直接刚体监督的研究方向。
- [V1.3 最终计划](plan/V1.3.md)：仅 GRAB/MANO；V1.3.2 不包含 `p_effect` 预测分支。
- [V1.4 指导](指导/V1.4.md)：用户确认的 GRAB、ARCTIC、OakInk2 三域刚体训练范围。
- [V1.4 最终计划](plan/V1.4.md)：三域 cache 合同、OakInk2 原始帧回填、等概率 sampler 与 smoke 闸门。
- [铰接对象架构设计](architecture/articulated_object_design.md)：ARCTIC two-link / GRAB rigid-fallback 张量流、joint graph 与解析 FK；V1.5.1 工程实现已就绪，但真实 cache FK replay 尚未通过，训练仍被阻断。
- [V1.5 指导](指导/V1.5.md)：用户确认的 ARCTIC two-link / GRAB rigid-fallback 首阶段范围。
- [V1.5 最终计划](plan/V1.5.md)：articulated loader、graph、analytic FK 与定向 smoke 的冻结实施边界。
- [V1.5 ARCTIC 运动学 metadata](../configs/active/arctic_articulation_v1_5.json)：由 producer 约定及全 cache FK replay 验证的 two-link revolute contract。
- [V1.6 最终计划](plan/V1.6.md)：随机初始化 V1.5 的 GPU1 batch calibration 与 8-step smoke。
- [V1.6 指导](指导/V1.6.md)：用户确认的随机初始化、双域平衡和受控校准范围。
- [V1.7 最终计划](plan/V1.7.md)：GRAB stride=1、ARCTIC stride=5..10 的 16-epoch 从零正式训练。
- [V1.8 指导](指导/V1.8.md) 与 [V1.8 最终计划](plan/V1.8.md)：保留未完成 B16 run，以 B64 calibration 闸门后从随机初始化重启相同的 16-epoch 二域训练。
- [V1.9 指导](指导/V1.9.md) 与 [V1.9 最终计划](plan/V1.9.md)：当前 ARCTIC test 缺失时，使用 V1.8 best checkpoint 做 GRAB stride=1 / ARCTIC stride=5..10 的 validation diagnostic。
- [V1.10 指导](指导/V1.10.md) 与 [V1.10 最终计划](plan/V1.10.md)：以纯软链接导航层规范 Cmv2 数据与 cache；不迁移 NAS 实体，不改变训练配置。
- [V1.11 指导](指导/V1.11.md) 与 [V1.11 最终计划](plan/V1.11.md)：OakInk2 MANO 全量导出与随机初始化的 GRAB/ARCTIC stride=1..3 二域正式训练。
- [V1.11a 最终计划](plan/V1.11a.md)：恢复高分辨率 OakInk2 MANO、GRAB/ARCTIC Inspire cache，并迁移本次新 manifest 的版本字段。
- [V1.11b 最终计划](plan/V1.11b.md)：用 V1.9 固定 validation 口径评估 V1.11 最优二域 checkpoint。
- [V1.3 GRAB 正式训练配置](../configs/active/grab_mano_v1_3_formal.yaml)：用户批准的单轮全量 train 范围与停止条件。
- [V1.3 双卡显存校准配置](../configs/active/grab_mano_v1_3_ddp_calibration.yaml)：GPU2/3 的 batch 显存校准入口。
- [V1.3 双卡四轮正式配置](../configs/active/grab_mano_v1_3_ddp_formal.yaml)：每卡 batch 160、全局 batch 320、四轮从头训练。
- [Task 活动记录](logs/activity_log.md)：本 Task 的治理、文档、实现和运行时间线。
- [当前 Activities](activities/README.md)：合并后治理合同下的长期运行与诊断记录。

本 Task 与旧 `ObjectInteractionCm` 独立。旧 Task 的代码、数据、cache、checkpoint、输出和未提交改动均不在本 Task 范围内。
