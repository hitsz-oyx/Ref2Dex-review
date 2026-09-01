# Cm 仓库记忆

- scope: task:Cm
- last_updated: 2026-08-31
- last_verified: 2026-08-31
- related: [当前状态](status_log.md)、[架构记录](architecture_log.md)、[实验记录](experiment_log.md)

## V1.2.11 — Cm Task-local 资产入口

- category: governance / operation
- status: active
- last_verified: 2026-08-31
- fact: Cm 使用的外部预训练模型、body model 和机器人资产统一放在被忽略的 `src/task/Cm/assets/`；根 `assets` 仅是兼容软链接。当前 `src/task/Cm/assets/checkpoints/densetoken` 指向历史 `src/task/Cm/densetoken_ckpt`，真实大型文件未移动。
- source / anchor: `src/task/Cm/src/config.py`、`.gitignore`、`src/task/Cm/docs/logs/architecture_log.md`。

本文记录可随仓库迁移的 Cm 长期事实。本机解释器、GPU、绝对数据路径和服务异常写入同目录下被 Git 忽略的 `machine_memory.md`。

## 运行入口

- 数据与 cache 的仓库相对根为 `data/processed_data`。
- 必须从仓库根目录运行 `python -m src.task.Cm.<module>`；具体解释器路径属于 machine memory。

## 2026-08-23 — mixed 续训日志存在 step 回退边界

- category: pitfall / convention
- status: active
- last_verified: 2026-08-23
- fact: 两条 mixed run 于 2026-08-23 从最近完整 checkpoint 原目录续训并保持 global batch 144。C=256 的 `metrics.jsonl` 曾写到 step 121770，恢复点为 118080；C=64 曾写到 step 90500，恢复点为 84870。因此迁移点后 step 非单调，曲线统计必须按时间和恢复段处理。
- source / anchor: `output/exp/cm_v121/` 日志、两条 run 的 `metrics.jsonl`。

## 2026-08-22 — 新旧 ObjectV2 路径与兼容边界

- category: path / pitfall
- status: active
- last_verified: 2026-08-22
- fact: 新版 mixed 逻辑根目录是 `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820`。它用 sequence 级相对软链接复用 `cm_object_v2_subject_template_20260820/grab` 和 `cm_object_v2/arctic`，不能在归档源实体 cache 时单独保留。
- source / anchor: 当前 mixed 配置、同根 split 与 metadata。

## 2026-08-22 — 旧 mixed cache 只用于历史运行

- category: legacy / pitfall
- status: deprecated
- last_verified: 2026-08-22
- fact: `data/processed_data/cm_object_v2` 的 ARCTIC 可继续复用，但其中旧 GRAB 由错误 root 解析静默回退到平均 MANO template；依赖旧数据的结果必须标记数据有效性风险。
- source / anchor: `process/GRAB/raw.py` 修复、subject-template GRAB cache。

## 2026-08-22 — 新版 mixed metadata 口径

- category: convention
- status: active
- last_verified: 2026-08-22
- fact: split 使用 seed42、按 dataset 分层的 sequence 划分，train/val/test=`1308/164/164`；calibration 只读 train split，对 stride 1--10 等权，`flow_target_rms_m=0.07328625889337191`、scale=`13.645122770626802`。训练覆盖 stride 1--10，验证固定报告 1/5/10。
- source / anchor: 新版 mixed 根目录的 `splits_seed42/splits.json`、`metadata.json` 与训练 YAML。

## 2026-08-22 — DexYCB 修复版评估 cache

- category: path / legacy / pitfall
- status: active
- last_verified: 2026-08-22
- fact: `data/processed_data/stage4/data/dexycb` 为修复后的 subject-10/right cache；正式 split 为 `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json`。旧 cache、旧 split 和旧评测数字不得复用。
- source / anchor: `process/DexYCB/raw.py`、[EXP-010](experiment_log.md#exp-010--dexycb-subject-10-修复版正式评估)。

## 2026-08-23 — HRDexDB Cm 微调禁止 DenseToken 输出缓存

- category: convention / pitfall
- status: active
- last_verified: 2026-08-23
- fact: 允许使用只含 world geometry、wrist pose、时间和帧映射的共享 geometry cache；`data.use_dense_cache` 必须为 false。DenseToken 输出缓存会切断解冻 DenseToken 的反向梯度，不能用于 HRDexDB Cm fine-tune。
- source / anchor: `src/task/Cm/dataset/hrdexdb.py`、`src/task/Cm/src/model.py`、方案 A 配置。
