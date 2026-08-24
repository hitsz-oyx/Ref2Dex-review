# Cm 接手记忆

- scope: task:Cm
- last_updated: 2026-08-23
- last_verified: 2026-08-23
- related: [当前状态](status_log.md)、[架构记录](architecture_log.md)、[实验记录](experiment_log.md)

## 运行环境

- Python: `/home2/wyy/miniconda3/envs/graspenv/bin/python`
- torchrun: `/home2/wyy/miniconda3/envs/graspenv/bin/torchrun`
- 数据 / 缓存根路径: `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data`
- 必须使用的命令入口: 从仓库根目录运行 `python -m src.task.Cm.<module>`；系统 `/usr/bin/python` 无法导入仓库的 `src.task.Cm`。

## 2026-08-23 — mixed 续训日志存在 step 回退边界

- category: pitfall / convention
- status: active
- last_verified: 2026-08-23
- fact: 两条 mixed run 于 2026-08-23 19:42 CST 从最近完整 checkpoint 原目录续训并保持 global batch 144。C=256 的 `metrics.jsonl` 曾写到 step 121770，恢复点为 118080；C=64 曾写到 step 90500，恢复点为 84870。因此 JSONL 在迁移点后 step 非单调，后续曲线统计必须按时间/恢复段处理，不能把同一 step 的多段记录直接混合。当前 resolved `config.json` 为 world size 2 对应的 per-device batch 72；源 YAML 仍保留原 3 GPU × 48 默认入口。
- source / anchor: `output/exp/cm_v121/*_2gpu_bs72_resume_20260823.log`、两条 run 的 `metrics.jsonl`。

## 2026-08-22 — 新旧 ObjectV2 路径与兼容边界

- category: path / pitfall
- status: active
- last_verified: 2026-08-22
- fact: 新版 mixed 逻辑根目录是 `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820`。它用 sequence 级相对软链接复用 `cm_object_v2_subject_template_20260820/grab` 和 `cm_object_v2/arctic`，不能在归档源实体 cache 时单独保留。
- source / anchor: `src/task/Cm/configs/object_v2_grab_arctic_subject_template_20260820.yaml`、同根 `splits_seed42/splits.json` 与 `metadata.json`。

## 2026-08-22 — 旧 mixed cache 只用于历史运行

- category: legacy / pitfall
- status: deprecated
- last_verified: 2026-08-22
- fact: `data/processed_data/cm_object_v2` 的 ARCTIC 可继续复用，但其中旧 GRAB 由错误 root 解析静默回退到平均 MANO template；当前仍在运行的 mixed/gate warm-up 依赖该旧数据，其结果必须标记数据有效性风险。
- source / anchor: `process/GRAB/raw.py` 的 subject asset 修复、`data/processed_data/cm_object_v2_subject_template_20260820/grab` 的重建记录。

## 2026-08-22 — 新版 mixed metadata 口径

- category: convention
- status: active
- last_verified: 2026-08-22
- fact: split 使用 seed42、按 dataset 分层的 sequence 划分，计数为 train/val/test=`1308/164/164`；calibration 只读 train split，对 stride 1--10 等权，`flow_target_rms_m=0.07328625889337191`、scale=`13.645122770626802`。训练覆盖 1--10 stride，验证只固定报告 1/5/10。
- source / anchor: 新版 mixed 根目录的 `splits_seed42/splits.json`、`metadata.json` 与训练 YAML。

## 2026-08-22 — DexYCB 修复版评估 cache

- category: path / legacy / pitfall
- status: active
- last_verified: 2026-08-22
- fact: `data/processed_data/stage4/data/dexycb` 已由修复 adapter 重建为 subject-10/right 的 50 条序列、2853 帧；正式 split 为 `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json`。旧 489M cache 仍在系统回收站，旧 `dexycb_v1` split 与 `outputs/cm/cm_eval_dexycb_subject10` 数字均不得复用。
- source / anchor: `process/DexYCB/raw.py`、[EXP-010](experiment_log.md#exp-010--dexycb-subject-10-修复版正式评估)。

## 2026-08-23 — HRDexDB 非视频原始数据已完成下载

- category: path / convention
- status: stale
- last_verified: 2026-08-23
- fact: 原始 HRDexDB 全量非视频副本位于 `/home2/wyy/oyx_ws/HRDexDB/v0_nonvideo`，包含 human/MANO、Allegro-V5、Inspire-DFTP、Inspire-F1、assets、两个 object-pose 版本和 metadata；不含视频。后续先用对应 cache builder 生成统一 `hand_points_world [T,1538,3]` / `obj_points_world [T,512,3]` geometry，再接入 `dataset_hrdexdb.py`。
- source / anchor: `process/HRDexDB/lfs_batch_proxy.py`、最终 pointer/视频完整性检查。

## 2026-08-23 — HRDexDB 非视频数据迁入仓库 dataset 目录

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: Cm/CmDecoder 使用的 HRDexDB 非视频根已迁移为 `/home2/wyy/oyx_ws/Ref2Dex/dataset/HRDexDB/v0_nonvideo`；机器人 URDF 根为同级 `dataset/HRDexDB/assets/robots`，自包含读取 helper 位于 `dataset/HRDexDB/hrdexdb_contact_heatmaps`。`dataset/HRDexDB/` 整体由根 `.gitignore` 排除。旧 `/home2/wyy/oyx_ws/HRDexDB/v0_nonvideo` 仅为兼容正在运行的 cache builder 的 symlink，不再是规范入口。
- source / anchor: `.gitignore`、`src/task/CmDecoder/config.py`、`src/task/CmDecoder/build_cache.py`。

## 2026-08-23 — 新 C=32 GRAB ablation 使用离线 W&B

- category: environment / pitfall
- status: active
- last_verified: 2026-08-23
- fact: 当前系统时间晚于 W&B 服务端证书有效期，在线初始化报 `x509: certificate has expired` 并在首个训练 step 前退出。C=32 geometry-only/no-time run 改为 `wandb.mode=offline`，本地指标与 checkpoint 不受影响；不要把首次在线启动失败视为实现或科研结果。
- source / anchor: `output/exp/cm_v121/cm_object_v2_grab_gate_cm32_geometry_only_no_time_gpu7.log`、新实验 YAML。

## 2026-08-23 — HRDexDB Cm 微调禁止 DenseToken 输出缓存

- category: convention / pitfall
- status: active
- last_verified: 2026-08-23
- fact: 允许使用只含 world geometry、wrist pose、时间和帧映射的共享 geometry cache；`data.use_dense_cache` 必须为 false。DenseToken 输出缓存会切断解冻 DenseToken 的反向梯度，不能用于 HRDexDB Cm fine-tune。
- source / anchor: `src/task/Cm/dataset_hrdexdb.py`、`src/task/Cm/model.py`、方案 A 配置。
