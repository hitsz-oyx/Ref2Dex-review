# CmDecoder 仓库记忆

- scope: task:CmDecoder
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [当前状态](status_log.md)、[架构](architecture_log.md)、[实验](experiment_log.md)

本文记录可随仓库迁移的 CmDecoder 长期事实。绝对数据路径、NAS 映射和本机资产计数写入同目录下被 Git 忽略的 `machine_memory.md`。

## 2026-08-23 — 全量非视频数据规范入口

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: 四手型 builder 的仓库内规范数据根为 `dataset/HRDexDB/v0_nonvideo`，机器人 URDF 根为 `dataset/HRDexDB/assets/robots`，读取 helper 为 `dataset/HRDexDB/hrdexdb_contact_heatmaps/hrdexdb_io.py`。`dataset/HRDexDB/` 被 Git 忽略；任何仓库外旧路径只能作为本机兼容 symlink。
- source / anchor: `src/task/CmDecoder/config.py`、`src/task/CmDecoder/build_cache.py`、根 `.gitignore`。

## 2026-08-22 — 20-episode 3 Hz token cache 绑定新 Cm

- category: path / convention
- status: active
- last_verified: 2026-08-22
- fact: `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/episodes/*/cm` 的 20 个 token sidecar 绑定 `cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt`，checkpoint SHA256 为 `64737bf4453b4ed37de105ea4c17a5e235e63c8bf2df7338c4c88a9d0817ebb5`。重训旧 Cm 对照时必须重建相应 token sidecar。
- source / anchor: 20 个 `cm/manifest.json`。

## 2026-08-22 — 全量 object-disjoint 3 Hz cache

- category: path / convention
- status: active
- last_verified: 2026-08-22
- fact: 30 Hz manifest 为 `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_576_seed42.json`，含 576 个 episode、362027 帧。3 Hz manifest 为 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/selection_576_object_disjoint_seed42.json`，保留 568 个 episode、284414 pairs；train/val/test 为 448/58/62 episode，object 交集为空。
- source / anchor: selection、geometry、task 与 token manifests。

## 2026-08-22 — wrist-aware 3 Hz v2 cache

- category: path / convention / pitfall
- status: active
- last_verified: 2026-08-22
- fact: wrist-aware 点监督必须使用 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v2/selection_576_object_disjoint_seed42.json`。manifest 必须标记 `cache_version=v2`、`hand_flow_frame=current_wrist`。v1 只保留历史复现，不能用于联合 wrist+q point loss；dataset 应通过 `required_hand_flow_frame=current_wrist` fail-fast。
- source / anchor: v2 manifests、EXP-015。
