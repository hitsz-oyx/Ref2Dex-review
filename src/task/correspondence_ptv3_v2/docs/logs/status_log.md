# correspondence_ptv3_v2 当前状态

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-09-02
- last_verified: 2026-09-02 UTC
- related: [架构](architecture_log.md) / [仓库记忆](repo_memory.md) / [实验](experiment_log.md) / [修改](modification_log.md) / [当前指导](../指导/V1.md)

## 当前状态

- 当前阶段 / 指导: feature 分支内容已合入 `oyx` 并通过回归测试；ARCTIC 外部评估仍遵循 V1。
- 当前进行中: OakInk2 object-centered Stage3、object-disjoint train/val 划分以及 ARCTIC/HRDexDB 四手型 object-frame cache 均已完成；GPU 2/3 七域训练正常运行至 step 143,700 / 21,030,400（epoch 1），机器人重建 batch 正常，最近一次 5000-step checkpoint 为 step 140,000。
- 最近可靠结论: OakInk 新产物含 2596 个 NPZ、252,172 帧，使用官方 root quaternion 做完整手根旋转规范化，并通过严格 MANO loader、MANO 重建、接触距离和四视角一致性校验。ContactPose 原始 `use` 数据有 1181 个序列目录，均带 `mano_fits_15.json`；其中 1477 个有效 hand fits（18 维：global axis-angle + 15 PCA），说明 MANO 并非数据集缺失，而是当前 `use_stage3_v2` 的 885 个 v2.0 NPZ 没有把 MANO 字段导出。现有 ContactPose exporter 导出的单序列 v2.1 pilot 已在 loader/runner 中以零噪声重建，hand face-center 平均绝对误差 `2.92e-9 m`、最大 `2.98e-8 m`。HRDexDB human `human/apple/0` 的 257 帧使用 `MANO_RIGHT + flat_hand_mean=True` 重建，vertex RMSE 为 `6.33e-8 m`，hand-root face-center RMSE 为 `6.66e-8 m`；当前 MANO reconstruction/axis-angle 扰动链路已可用。当前三域 mixed 的训练 loss 仍下降，但验证 QFL 在约 step 45k–68k 后整体进入平台并出现轻微回升：GRAB clean QFL 从最佳 0.000388 上升到最近 0.000540，ContactPose 从 0.000724 上升到 0.000791，OakInk 在约 0.000224–0.000250 间波动；因此更接近“训练目标收敛、验证指标早熟/轻微过拟合”，不能把最终 step 当作最佳模型。协议 E 中 H80 best 的 hand-only 最终 QFL / recovery 最好，H50 best 的三条件等权 perturbed QFL 最低；历史两域 mixed 明显落后。HOCap subject_1 的纯 GRAB noPCA 外部基线也已完成。相邻存储帧运动诊断显示当前 object-frame GRAB/F1 手点 RMS 均值约 6.19/2.52 mm，但 GRAB Stage 3 的 raw frame 间隔恒为 4，不能直接解释为原始视频 stride=1；F1 stride=2/3 时分别升至 4.29/5.93 mm，stride=3 更接近 GRAB 幅度。
- 阻塞 / 风险: 七域续训最终 clean macro QFL 为 `0.000223831`，未超过历史 `best.pt` 的 step 34907 / epoch 1、`0.000148572`，后续评估优先使用 `best.pt`。HOCap 原始 Stage 3 已核验为合法 MANO PCA45；补充 HOCap PCA45 profile 并重导出后，三种外部评测均正常完成，无 traceback/OOM。
- 下一步: 按 5000 step 继续保存 checkpoint；当前日志 ETA 约 4441 小时（约 185 天），需要用户决定是否停止并改为固定 step 上限。
- 证据与相关文档: [EXP-013](experiment_log.md#exp-013--oakink-true-hand-root-全量重导出与-mano-重建校验)；合并后全量 pytest `282 passed, 3 skipped`；[EXP-012](experiment_log.md#exp-012--纯-grab-nopca-在-hocap-subject_1-的外部测试)；[协议 E 结果](../../result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md)。
