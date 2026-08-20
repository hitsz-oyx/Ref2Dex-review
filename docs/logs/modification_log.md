# Modification log

## 2026-08-20 — 将 GRAB 大体积数据入口切到 NAS

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: 跨 task / 全局

**文件**

- `dataset/GRAB/data` — 软链到 NAS 上的 GRAB 数据根。
- `docs/logs/repo_notes_log.md` — 记录仓库级 NAS 数据根路径。

**改动原因**

本地根分区已接近满盘，而 GRAB 的大体积只读数据在 NAS 上已有完整副本。把入口改到 NAS 可以释放本地空间，并让后续任务统一引用同一份数据根。

**对应指导**

无

**影响范围**

跨 task / 全局

## 2026-08-20 — 将 processed_data 迁到 NAS 并补结果配置摘要

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: 跨 task / 全局

**文件**

- `data/processed_data` — 迁移到 NAS 路径 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data` 的软链。
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` — 补充三模型对比对应的训练配置摘要。
- `docs/logs/repo_notes_log.md` — 记录仓库级 processed_data NAS 路径。

**改动原因**

用户要求把本地 `processeddata` 也放到 NAS，并把当前对比结果里对应的训练配置写清楚，方便后续复现和检查配置差异。

**影响范围**

跨 task / 全局
