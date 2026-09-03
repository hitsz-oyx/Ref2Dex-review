# correspondence_ptv3_v2 仓库记忆

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-08-25
- last_verified: 2026-08-25
- related: [当前状态](status_log.md)、[架构](architecture_log.md)、[实验](experiment_log.md)

本文只记录可随仓库迁移的长期事实。NAS 根、软链目标、解释器和 GPU 等写入同目录下被 Git 忽略的 `machine_memory.md`。

## 2026-08-24 — 数据路径使用仓库相对入口

- category: path / convention
- status: active
- last_verified: 2026-08-24
- fact: 配置、脚本和受版本管理文档应优先使用 `data/processed_data/...` 等仓库相对入口；本机可将该入口映射到 NAS，但不得把挂载点固化为跨机器合同。
- source / anchor: `AGENTS.md`、根 `.gitignore`。

## 2026-08-20 — OakInk 无 MANO 字段边界

- category: convention / pitfall
- status: deprecated
- last_verified: 2026-08-24
- fact: 2026-08-18 的 OakInk Stage 3 有 4096 物点、1538 手面中心和 clean 最短距离，但没有 MANO pose/betas，且只做 wrist-centered camera-frame 平移中心化。该边界只用于复现旧无手扰动实验，不能代表 2026-08-24 true-hand-root 产物。
- source / anchor: `architecture_log.md`、EXP-001、EXP-013。

## 2026-08-24 — OakInk true-hand-root 与 MANO translation 语义

- category: convention / pitfall
- status: active
- last_verified: 2026-08-24
- fact: 当前 OakInk 规范产物使用官方 root quaternion 将手物几何转换到 rotation-canonicalized `hand_root`，并携带 axis-angle45 MANO 参数。OakInk `hand_tsl` 表示 wrist joint 世界位置；写入训练 schema 的 `mano_transl` 必须减去 shape-dependent MANO wrist template offset，不能直接复制 `hand_tsl`。
- source / anchor: `research/oakink_conversion/convert_oakink_pilot.py`、EXP-013。

## 2026-08-23 — ARCTIC min11 只用于小规模对照

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: `data/processed_data/stage3/arctic_min11_mano_v1` 只覆盖 11 个物体和 4175 帧，用于 hand/object/joint robustness 小规模对照；不能替代全量 ARCTIC 外部评估。
- source / anchor: `result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md`。

## 2026-08-25 — HRDexDB minimal allhands 与 correspondence 适配边界

- category: path / pitfall
- status: active
- last_verified: 2026-08-25
- fact: `output/HRDexDB_correspondence_minimal_allhands_20260825.tar.gz` 覆盖四类手型的 2104 个 episode，C2R-valid 子集为 2088 个；它是 minimal raw 包，不含 camera/grasp/video 和 episode 内 object pose 文件。object pose 位于 `v0_nonvideo/object_6d_pose_v1|v2/<hand>/<object>_<seq>.npz`，物体网格位于 `v0_nonvideo/assets/mesh_v2`，与当前 `process/HRDexDB/*_stage3.py` 直接读取的目录合同不同。
- source / anchor: 2026-08-25 archive listing 与代表性 human/robot 文件 shape 核验。

## 2026-08-25 — ContactPose MANO 字段边界

- category: convention / path
- status: active
- last_verified: 2026-08-25
- fact: ContactPose 原始 `data/contactpose_data/full*_{use,handoff}/<object>/` 序列普遍带 `mano_fits_15.json`（18 维 pose：3 维 global axis-angle + 15 维 PCA）和 `betas/mTc`；当前 NAS `processed_data/stage3/data/ContactPose/use_stage3_v2` 是 schema 2.0.0，仅存几何字段，不能开启 MANO reconstruction。`process/ContactPose/stage3_export.py` 已具备 v2.1 MANO 导出路径，单序列 pilot 的零噪声 hand face-center 重建误差小于 `3e-8 m`。
- source / anchor: ContactPose raw file audit、`process/ContactPose/stage3_export.py`、`contactpose_mano_pilot_20260825`。
