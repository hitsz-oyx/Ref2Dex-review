# CmDecoder 接手记忆

- scope: task:CmDecoder
- last_updated: 2026-08-23
- last_verified: 2026-08-23
- related: [当前状态](status_log.md), [架构](architecture_log.md)

## 2026-08-22 — HRDexDB Inspire F1 可用资产

- category: path / environment
- status: active
- last_verified: 2026-08-22
- fact: `/home2/wyy/oyx_ws/HRDexDB/v0` 中有592个 Inspire F1 episode 目录；591组 arm position/time、591组 compact v1 pose、555组 compact v2 pose、576组 C2R。arm 数组均可加载且为 `[T,6]` position + `[T]` time。完整模态最大交集为576组，其中539组有 v2 pose，另37组需要 v1 fallback。
- source / anchor: 2026-08-22 本地全量扫描；官方 Hugging Face main。

本次白名单下载没有新增视频；本地原有4119个 Inspire F1 MP4。compact pose 位于 `object_6d_pose_v1/inspire_f1/<object>_<scene>.npz` 和 `object_6d_pose_v2/inspire_f1/<object>_<scene>.npz`，每个 NPZ 以 `frame_<index>` 保存 `float32 [4,4]` pose。当前 `build_cache.py` 尚未消费该根级 NPZ 布局。

## 2026-08-23 — 全量非视频数据规范路径

- category: path / environment
- status: active
- last_verified: 2026-08-23
- fact: 四手型全量 builder 的规范数据根为 `/home2/wyy/oyx_ws/Ref2Dex/dataset/HRDexDB/v0_nonvideo`，机器人 URDF 根为 `dataset/HRDexDB/assets/robots`，读取 helper 为 `dataset/HRDexDB/hrdexdb_contact_heatmaps/hrdexdb_io.py`。`dataset/HRDexDB/` 被 Git 忽略；旧外部 `v0_nonvideo` 是迁移期间的兼容 symlink。
- source / anchor: `src/task/CmDecoder/config.py`、`src/task/CmDecoder/build_cache.py`、根 `.gitignore`。

## 2026-08-22 — 20-episode 3 Hz token cache 绑定新 Cm

- category: path / convention
- status: active
- last_verified: 2026-08-22
- fact: `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/episodes/*/cm` 的20个 token sidecar 当前全部绑定 `cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt`，checkpoint SHA256 为 `64737bf4453b4ed37de105ea4c17a5e235e63c8bf2df7338c4c88a9d0817ebb5`。旧 Cm 实验 checkpoint 本身仍可读取，但若要重新训练旧 Cm 对照，必须重建相应 token sidecar。
- source / anchor: 20个 `cm/manifest.json` 的 `cm_checkpoint_sha256`。

## 2026-08-22 — 全量 object-disjoint 3 Hz cache

- category: path / convention
- status: active
- last_verified: 2026-08-22
- fact: 30 Hz geometry/task manifest 为 `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_576_seed42.json`，包含576个 episode、362,027帧；539组使用 compact v2 pose，37组回退 compact v1。3 Hz manifest 为 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/selection_576_object_disjoint_seed42.json`，连续 stride=10 条件保留568个 episode、284,414 pairs；train/val/test 为448/58/62 episode和228,977/28,391/27,046 pairs。
- source / anchor: 两个 manifest 及576个 geometry、568个 task/token manifest 的全量检查。

object-disjoint 的 train/val/test 分别覆盖70/10/9个物体，交集为空。568个 token sidecar 全部绑定 SHA256 `64737bf4453b4ed37de105ea4c17a5e235e63c8bf2df7338c4c88a9d0817ebb5`；30 Hz cache 约44 GB，3 Hz task+token cache 约23 GB。

## 2026-08-22 — wrist-aware 3 Hz v2 cache

- category: path / convention / pitfall
- status: active
- last_verified: 2026-08-22
- fact: wrist-aware 点监督必须使用 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v2/selection_576_object_disjoint_seed42.json`。该版本将目标点表示在当前腕坐标系，manifest 标记 `cache_version=v2`、`hand_flow_frame=current_wrist`；共有568个 episode、284,414 pairs，split 与 v1 完全一致。568个 token sidecar 均绑定 Cm checkpoint SHA256 `64737bf4453b4ed37de105ea4c17a5e235e63c8bf2df7338c4c88a9d0817ebb5`，task+token 约23 GB，全量 schema/hash/语义扫描为0错误。
- source / anchor: v2 selection/task/token manifest 全量扫描；EXP-015。

v1 cache 保留用于 EXP-010—014 等历史实验复现，但其目标点在目标腕系，不能用于联合 wrist+q 的 point loss。dataset 可通过 `required_hand_flow_frame=current_wrist` 对错误版本 fail-fast。
