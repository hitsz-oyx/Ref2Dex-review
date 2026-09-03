# correspondence_ptv3_v2 仓库记忆

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [活动记录](activity_log.md)、[架构](architecture_log.md)、[实验](experiment_log.md)

## V1.2.12 — 数据入口回到根空间

- category: governance / operation
- status: active
- last_verified: 2026-09-01
- fact: correspondence_ptv3_v2 不再维护 Task-local `components/`、`data/` 或 `registry/`。训练/评估配置直接引用根级 `data/processed_data/`、`dataset/`、`third_party/`；Task 内校准 JSON 仍作为研究配置输入保留。
- source / anchor: `config.py`、`docs/plan/V1.md`、根 `.gitignore`。

本文只记录可随仓库迁移的长期事实。NAS 根、软链目标、解释器和 GPU 等写入同目录下被 Git 忽略的 `machine_memory.md`。

## 2026-08-24 — 数据路径使用仓库相对入口

- category: path / convention
- status: active
- last_verified: 2026-08-24
- fact: 配置、脚本和受版本管理文档应优先使用 `data/processed_data/...` 等仓库相对入口；本机可将该入口映射到 NAS，但不得把挂载点固化为跨机器合同。
- source / anchor: `AGENTS.md`、根 `.gitignore`。

## 2026-08-20 — OakInk 无 MANO 字段边界

- category: convention / pitfall
- status: active
- last_verified: 2026-08-24
- fact: 当前 OakInk Stage 3 有 4096 物点、1538 手面中心和 clean 最短距离，但没有 MANO pose/betas。存储几何使用 wrist-centered camera frame，不是 MANO rotation-canonicalized hand-root；不得在缺少 MANO 字段时启用 MANO reconstruction。
- source / anchor: `architecture_log.md`、OakInk conversion diagnostics。

## 2026-08-23 — ARCTIC min11 只用于小规模对照

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: `data/processed_data/stage3/arctic_min11_mano_v1` 只覆盖 11 个物体和 4175 帧，用于 hand/object/joint robustness 小规模对照；不能替代全量 ARCTIC 外部评估。
- source / anchor: `result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md`。
