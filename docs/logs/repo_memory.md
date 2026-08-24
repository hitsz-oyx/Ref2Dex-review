# Ref2Dex 仓库记忆

- scope: root
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [当前状态](status_log.md)、[架构记录](architecture_log.md)、[修改记录](modification_log.md)

本文只记录可随仓库迁移的长期事实。解释器绝对路径、GPU、代理、NAS 挂载点和仓库软链的本机目标写入同目录下被 Git 忽略的 `machine_memory.md`。

## 2026-08-24 — 仓库与机器记忆分离

- category: convention
- status: active
- last_verified: 2026-08-24
- fact: 根级和 Task 级记忆统一拆为受版本管理的 `repo_memory.md` 与本机私有的 `machine_memory.md`。凡可用仓库相对路径表达的入口、schema、命名约定、历史遗留和兼容性事实必须留在 repo memory；只有随机器变化的绝对路径、环境和挂载映射进入 machine memory。
- source / anchor: `AGENTS.md`、`.gitignore`。

## 2026-08-22 — DexYCB Stage4 修复版 cache

- category: path / legacy / pitfall
- status: active
- last_verified: 2026-08-22
- fact: `data/processed_data/stage4/data/dexycb` 已用修复 adapter 重建为 subject-10/right 的 50 条序列、2853 帧，active-frame ratio 80.30%；正式评测 split 为 `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json`。修复前 cache、旧 `dexycb_v1` split 和旧评测输出不得复用。
- source / anchor: `process/DexYCB/raw.py`、`process/DexYCB/stage4_cm.py`、Cm EXP-010。

## 2026-08-23 — HRDexDB 仓库内规范入口

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: HRDexDB 非视频数据的仓库内规范根为 `dataset/HRDexDB/v0_nonvideo`；机器人资产和读取 helper 分别位于 `dataset/HRDexDB/assets`、`dataset/HRDexDB/hrdexdb_contact_heatmaps`。`dataset/HRDexDB/` 整体被 Git 忽略。旧的仓库外路径只允许作为本机兼容 symlink，不是配置或文档的规范入口。
- source / anchor: `.gitignore`、`src/task/CmDecoder/config.py`、`src/task/CmDecoder/build_cache.py`。

## 2026-08-23 — HRDexDB geometry cache 的 Cm 输入边界

- category: convention / pitfall
- status: active
- last_verified: 2026-08-23
- fact: `cmdecoder_layered_v4` geometry layer 需要同时保存 legacy decoder 的 512 点字段和 Cm 使用的 `obj_points_pool_world [T,4096,3]`、`obj_normals_pool_world`、`obj_candidate_mask_5cm [T,4096]`。缺少 candidate mask 的旧 512 点 smoke cache 不能直接用于 Cm 训练；DenseToken feature 禁止写入该层。
- source / anchor: `src/task/CmDecoder/build_cache.py`、`src/task/Cm/dataset_hrdexdb.py`、Cm V1.2 candidate contract。

## 2026-08-23 — CmDecoder 多手型 cache builder

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: `src/task/CmDecoder/build_cache.py` 支持 `human,allegro_v5,inspire_dftp,inspire_f1`；数据入口为 `dataset/HRDexDB/v0_nonvideo`，机器人 URDF 根为 `dataset/HRDexDB/assets/robots`。全量构建使用 `--robot-types human,allegro_v5,inspire_dftp,inspire_f1 --episodes 0`，输出应写到独立 cache root。
- source / anchor: `src/task/CmDecoder/build_cache.py`。

## 2026-08-24 — HRDexDB 全量 geometry cache manifest

- category: path / convention
- status: active
- last_verified: 2026-08-24
- fact: `data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json` 是四手型 HRDexDB 的正式 object-disjoint manifest，共 2088 个有效 episode，train/val/test=`1642/232/214`。缺少机器人 `C2R.npy` 的 16 个 episode 被 selector 排除；cache 不包含 DenseToken 输出。
- source / anchor: `src/task/CmDecoder/build_cache.py`、正式 manifest。

## 2026-08-20 — 共享 GRAB raw asset 约定

- category: convention / pitfall
- status: active
- last_verified: 2026-08-24
- fact: `process/GRAB/raw.py` 被多个 Task 共用。GRAB sequence root 与 `tools/subject_meshes` 可能位于不同层级，调用方不得假设所有相对 asset 都直接位于传入 root 下。正式 Cm/V2 Stage4 强制使用 subject-specific MANO template；缺失时必须显式失败，旧流程只有通过兼容开关才允许回退。
- source / anchor: `process/GRAB/raw.py`、`process/GRAB/stage4_cm.py`。
