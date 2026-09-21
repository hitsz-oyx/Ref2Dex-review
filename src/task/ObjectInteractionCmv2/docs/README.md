# ObjectInteractionCmv2 文档入口

## 当前工作状态

- work_version：`V1.12`；Task-local 实现、CPU/真实只读门禁、单 GPU smoke 及 GPU0+2 B64 双卡 smoke 已完成。V1.12 正式 run 已按用户要求在 step 200、epoch 1 受控停止，终态为 `STOPPED`；下一步在独立 V1.13 分支诊断并优化数据 I/O，当前不据此形成科学结论。
- 当前 V1.12 架构为 `v1_12_endpoint_part_se3`：固定起始物体 query 的端点 KNN union-rerank、整体多部件 1024 点分层采样、grounded surface-token soft routing、part self-attention 和共享逐部件直接 SE(3) head；part ID 仅用于 grouping/routing。
- V1.12 独立入口：[模型](../part_se3.py)、[数据适配](../part_se3_data.py)、[构造与 checkpoint 合同](../part_se3_training.py)、[非正式运行配置](../configs/active/mixed_part_se3_v1_12.yaml)、[实现与 smoke Activity](activities/ACT-20260920-CMV2-V112-ENDPOINT-PART-SE3.md)。
- V1.11.1 正式五组混合 run `cmv2_v111i_mixed_ddp_formal_20260920T114242Z` 已按用户要求于 step 18040、epoch 2 受控停止；其 `latest.pt`、`best.pt`、metrics 与 source snapshot 均保留。
- V1.12 正式 run `cmv2_v112_part_se3_ddp_formal_20260921T030019Z` 已于 `2026-09-21T03:31:14+00:00` 受控停止；`latest.pt`、manifest、metrics 与日志均保留，尚未完成首轮 validation，故无 `best.pt`。
- [V1.11g FINAL](plan/V1.11g.md)：五组、GPU1+3、每卡64、stride1..3、16epochs、15组验证，ARCTIC Inspire不等待也不动态纳入；资源切换见 [V1.11i FINAL](plan/V1.11i.md)。
- V1.11.1 历史入口：[DDP launcher](../train_mixed_articulated_ddp.py)、[部件级固定配置](../configs/active/mixed_articulated_v1_11h_ddp.yaml)、[部件 adapter builder](../tools/data/build_oakink2_part_adapter_v1_11h.py)、[Activity 与原始运行证据](activities/ACT-20260920-CMV2-V111G-MIXED-DDP-GATE.md)、[实验卡](experiments/EXP-20260920-V111G-MIXED-WARMSTART.md)。

## 指导与历史方案入口

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
- [V1.11e 最终计划](plan/V1.11e.md)：隔离空 `.partial` 残留并以 user systemd 托管 GRAB/ARCTIC Inspire 续跑。
- [V1.11f 最终计划](plan/V1.11f.md)：在 GPU2/GPU1 独立服务上并行调度 GRAB/ARCTIC Inspire，避免同一输出根双写。
- [V1.11i 最终计划补充](plan/V1.11i.md)：按用户要求将本轮混合训练资源切换为 GPU0+GPU2，其他研究变量不变。
- [V1.11b 最终计划](plan/V1.11b.md)：用 V1.9 固定 validation 口径评估 V1.11 最优二域 checkpoint。
- [V1.11c 最终计划](plan/V1.11c.md)：按 stride 1/2/3 分开统计二域 validation 的 EPE 与流模长。
- [V1.11d 最终计划](plan/V1.11d.md)：在 OakInk2 loader 中排除跨 `frame_time` discontinuity 的短 stride transition，不重导 cache。
- [V1.12 指导](指导/V1.12.md) 与 [V1.12 最终计划](plan/V1.12.md)：固定起始物体 query 的端点 KNN union-rerank、整体多部件采样、surface-token soft routing 与逐部件直接 SE(3)。
- [V1.3 GRAB 正式训练配置](../configs/active/grab_mano_v1_3_formal.yaml)：用户批准的单轮全量 train 范围与停止条件。
- [V1.3 双卡显存校准配置](../configs/active/grab_mano_v1_3_ddp_calibration.yaml)：GPU2/3 的 batch 显存校准入口。
- [V1.3 双卡四轮正式配置](../configs/active/grab_mano_v1_3_ddp_formal.yaml)：每卡 batch 160、全局 batch 320、四轮从头训练。
- [Task 活动记录](logs/activity_log.md)：本 Task 的治理、文档、实现和运行时间线。
- [当前 Activities](activities/README.md)：合并后治理合同下的长期运行与诊断记录。

本 Task 与旧 `ObjectInteractionCm` 独立。旧 Task 的代码、数据、cache、checkpoint、输出和未提交改动均不在本 Task 范围内。
