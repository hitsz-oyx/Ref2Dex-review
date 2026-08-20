# correspondence_ptv3_v2 repo notes

## 当前状态

任务目标是在 hand/object correspondence 输入上学习 clean contact target，并分析 object pose 扰动下的恢复能力。当前 OakInk 方向是“无 PCA 手部扰动，但保留 runtime object resampling”；GRAB 方向正在续跑一条“5 mm 配置 + runtime sampling + no-PCA hand perturb”的 run。GRAB 的大体积数据目录已切到 NAS 软链，`dataset/GRAB/data -> /mnt/ugreen_nas/storage/Ref2Dex_storage/GRAB/data/GRAB`，本地只保留代码和脚本。

- 主架构：[`../架构.md`](../架构.md)
- 实验日志：[`experiment_log.md`](experiment_log.md)
- 修改记录：[`modification_log.md`](modification_log.md)
- 设计决策：[`decision_log.md`](decision_log.md)
- GRAB 数据根：`/mnt/ugreen_nas/storage/Ref2Dex_storage/GRAB/data/GRAB`
- OakInk 全量 Stage 3：`/mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk/processed/stage3_corr_oakink_mano_face_handroot_20260818`
- 处理后中间数据：`data/processed_data -> /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data`

OakInk 导出没有 MANO pose/betas，但有 4096 物点、1538 手面中心和 clean 最短距离。当前导出是 wrist-centered camera frame，不是真实 MANO rotation-canonicalized hand-root。
