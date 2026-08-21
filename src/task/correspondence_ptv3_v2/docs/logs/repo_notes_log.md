# correspondence_ptv3_v2 repo notes

## 当前状态

任务目标是在 hand/object correspondence 输入上学习 clean contact target，并分析 object pose 扰动下的恢复能力。当前 OakInk 方向是“无 PCA 手部扰动，但保留 runtime object resampling”；GRAB 方向已完成 5 mm no-PCA runtime run 的整理。V1 的 ARCTIC 外部评估已完成，分层子集和 object-macro 结果落在 `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.md`。

当前指导版本为 [`../指导/V1.md`](../指导/V1.md)：在启动三数据集混训前，先用全量 ARCTIC 或覆盖全部物体类别的确定性分层子集评估现有纯 GRAB 与 GRAB+ContactPose checkpoint。

- 主架构：[`../架构.md`](../架构.md)
- 当前指导：[`../指导/V1.md`](../指导/V1.md)
- 实验日志：[`experiment_log.md`](experiment_log.md)
- 修改记录：[`modification_log.md`](modification_log.md)
- 设计决策：[`decision_log.md`](decision_log.md)
- GRAB 数据根：`/mnt/ugreen_nas/storage/Ref2Dex_storage/GRAB/data/GRAB`
- OakInk 全量 Stage 3：`/mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk/processed/stage3_corr_oakink_mano_face_handroot_20260818`
- 处理后中间数据：`data/processed_data -> /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data`
- 带 MANO 的 ARCTIC min11 评测 Stage 3：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1`；仅 11 个物体、4175 帧，用于小规模 hand/object/joint robustness 对照。

OakInk 导出没有 MANO pose/betas，但有 4096 物点、1538 手面中心和 clean 最短距离。当前导出是 wrist-centered camera frame，不是真实 MANO rotation-canonicalized hand-root。
