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
- [铰接对象架构设计](architecture/articulated_object_design.md)：待确认的 ARCTIC two-link / GRAB rigid-fallback 张量流、joint graph 与解析 FK 设计；尚未实现或冻结。
- [V1.3 GRAB 正式训练配置](../configs/active/grab_mano_v1_3_formal.yaml)：用户批准的单轮全量 train 范围与停止条件。
- [V1.3 双卡显存校准配置](../configs/active/grab_mano_v1_3_ddp_calibration.yaml)：GPU2/3 的 batch 显存校准入口。
- [V1.3 双卡四轮正式配置](../configs/active/grab_mano_v1_3_ddp_formal.yaml)：每卡 batch 160、全局 batch 320、四轮从头训练。
- [Task 活动记录](logs/activity_log.md)：本 Task 的治理、文档、实现和运行时间线。

本 Task 与旧 `ObjectInteractionCm` 独立。旧 Task 的代码、数据、cache、checkpoint、输出和未提交改动均不在本 Task 范围内。
