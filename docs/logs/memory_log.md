# Ref2Dex 接手记忆

- scope: root
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [仓库常识](repo_notes_log.md)

## 2026-08-22 — HRDexDB 非视频资产下载状态

- category: environment / path
- status: active
- last_verified: 2026-08-22
- fact: HRDexDB 数据根为 `/home2/wyy/oyx_ws/HRDexDB/v0`。Inspire F1 已有 591 组 arm position/time、591 组 compact v1 object pose、555 组 compact v2 object pose和576组 C2R；arm 数组全部通过 NumPy shape 检查。CmDecoder 所需完整模态的最大交集为576个 episode。
- source / anchor: Hugging Face dataset `HRDexDB/HRDexDB` main；本地文件扫描。

下载约定：只下载 C2R、arm/hand/timestamp、`assets/mesh_v2/*.obj` 和 `object_6d_pose_v{1,2}/inspire_f1/*.npz`；不下载视频、相机、tactile。稳定路径是官方 `https://huggingface.co` 配合 `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7897`，并按本地 HEAD 精确列出缺失文件后调用 `hf_hub_download`。当前 `snapshot_download` 会因约80万条仓库路径的递归枚举长时间阻塞；`hf-mirror.com` 重定向资源还会触发当前客户端的 URL 校验问题。

## 2026-08-22 — DexYCB Stage4 修复版 cache

- category: path / legacy / pitfall
- status: active
- last_verified: 2026-08-22
- fact: `data/processed_data/stage4/data/dexycb` 已用修复 adapter 重建为 subject-10/right 的 50 条序列、2853 帧，active-frame ratio 80.30%；正式评测 split 为 `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json`。修复前 489M cache 仍在系统回收站；旧 `dexycb_v1` split 和旧评测输出不得复用。
- source / anchor: `process/DexYCB/raw.py`、`process/DexYCB/stage4_cm.py`、Cm EXP-010。

## 2026-08-23 — HRDexDB 全量非视频原始数据副本

- category: path / data
- status: stale
- last_verified: 2026-08-23
- fact: 独立 worktree `/home2/wyy/oyx_ws/HRDexDB/v0_nonvideo` 已完成 HRDexDB 主分支的非视频数据下载，包含 `human`（MANO）、`allegro_v5`、`inspire_dftp`、`inspire_f1`、`assets`、`object_6d_pose_v1/v2` 和 `metadata`；共约 753,048 个文件，LFS pointer 余数为 0，视频文件数为 0。完整仓库视频体量约 1.15 TB，本副本只保留 Cm cache 构建所需模态。
- source / anchor: 官方 HF 数据集 `HRDexDB/HRDexDB`；`process/HRDexDB/lfs_batch_proxy.py`；最终 `git lfs status`。

## 2026-08-23 — HRDexDB 规范非视频路径迁入 Ref2Dex

- category: path / data
- status: active
- last_verified: 2026-08-23
- fact: HRDexDB 非视频数据的规范根为 `/home2/wyy/oyx_ws/Ref2Dex/dataset/HRDexDB/v0_nonvideo`；机器人资产和读取 helper 分别位于 `dataset/HRDexDB/assets`、`dataset/HRDexDB/hrdexdb_contact_heatmaps`。该目录整体被 Git 忽略。旧外部 `v0_nonvideo` 路径是迁移期间为在途 cache builder 保留的兼容 symlink。
- source / anchor: `.gitignore`、`src/task/CmDecoder/config.py`、`src/task/CmDecoder/build_cache.py`。

## 2026-08-23 — HRDexDB geometry cache 的 Cm 输入边界

- category: convention / pitfall
- status: active
- last_verified: 2026-08-23
- fact: 新版 `cmdecoder_layered_v4` geometry layer 需要同时保存 legacy decoder 的 512 点字段和 Cm 使用的 `obj_points_pool_world [T,4096,3]`、`obj_normals_pool_world`、`obj_candidate_mask_5cm [T,4096]`。缺少 candidate mask 的旧 512 点 smoke cache 不能直接用于 Cm 训练；DenseToken feature 仍禁止写入该层。
- source / anchor: `src/task/CmDecoder/build_cache.py`、`src/task/Cm/dataset_hrdexdb.py`、Cm V1.2 candidate contract。

## 2026-08-23 — CmDecoder 多手型 cache builder

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: `src/task/CmDecoder/build_cache.py` 支持 `human,allegro_v5,inspire_dftp,inspire_f1`；全量非视频源根为 `dataset/HRDexDB/v0_nonvideo`，机器人 URDF 根为 `dataset/HRDexDB/assets/robots`。用 `--robot-types human,allegro_v5,inspire_dftp,inspire_f1 --episodes 0` 构建全量，建议输出到独立 cache root。
- source / anchor: 2026-08-23 四类 smoke cache；`src/task/CmDecoder/build_cache.py`。

## 2026-08-24 — HRDexDB 全量 geometry cache 完成

- category: path / convention
- status: active
- last_verified: 2026-08-24
- fact: `data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json` 是四手型 HRDexDB 的正式 object-disjoint manifest，共 2088 个有效 episode，train/val/test=`1642/232/214`。缺少机器人 `C2R.npy` 的 16 个 episode 被 selector 排除；cache 不包含 DenseToken 输出。
- source / anchor: builder resume log、正式 manifest、`src/task/CmDecoder/build_cache.py`。

## 2026-08-24 — GitHub 远程推送代理

- category: environment / convention
- status: active
- last_verified: 2026-08-24
- fact: 当前环境向 GitHub 远程推送需要使用本地 HTTP 代理 `127.0.0.1:7897`。命令入口前设置 `http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY` 均为 `http://127.0.0.1:7897`；未设置代理时向 `origin` 推送会返回 HTTP 403。
- source / anchor: `git push origin oyx` 在 2026-08-24 的失败/代理重试记录。
