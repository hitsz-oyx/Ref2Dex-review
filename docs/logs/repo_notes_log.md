# Repo notes

## 当前状态

本仓库的大体积只读数据优先放 NAS。当前已将 `dataset/GRAB/data` 切到软链，指向 `/mnt/ugreen_nas/storage/Ref2Dex_storage/GRAB/data/GRAB`；其余数据根也统一按 NAS 路径引用。

- GRAB 数据根：`/mnt/ugreen_nas/storage/Ref2Dex_storage/GRAB/data/GRAB`
- ARCTIC 数据根：`/mnt/ugreen_nas/storage/Ref2Dex_storage/arctic/data/arctic_data`
- ContactPose 数据根：`/mnt/ugreen_nas/storage/Ref2Dex_storage/ContactPose/dataset/ContactPose`
- OakInk 全量 Stage 3：`/mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk/processed/stage3_corr_oakink_mano_face_handroot_20260818`
- 处理后中间数据：`data/processed_data -> /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data`

本地仓库保留代码、配置、日志和小体积子集；训练输出仍在 `outputs/`，需要时再按任务单独处理。
