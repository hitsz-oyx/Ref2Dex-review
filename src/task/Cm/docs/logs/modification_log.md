# Cm AI 修改记录

## 2026-08-18 — 建立 Cm 规范日志入口

- branch: 当前工作分支
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/docs/logs/repo_notes_log.md` — 建立 Cm 任务入口、状态和代码索引。
- `src/task/Cm/docs/logs/architecture_log.md` — 汇总 V1.2 pipeline、数据合同、评估和缓存路线。
- `src/task/Cm/docs/logs/decision_log.md` — 迁移现有 V1.2 cache 转换器决策。
- `src/task/Cm/docs/logs/experiment_log.md` — 见同目录实验历史摘要。

**改动原因**

按 `AGENTS.md` 将 Cm 现有 `log.md`、`decision_log.md`、框架和 Pipeline 文档整理到规范入口；保留原文档作为兼容/历史参考。

**影响范围**

仅 Cm 文档，不改变代码和实验定义。

## 2026-08-18 — 支持联合 object-v2 root 统计

- branch: 当前工作分支
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/dataset_object_v2.py` — 允许 `CmObjectV2Dataset` 读取包含 `grab/`、`arctic/` 子目录的联合 object-v2 root。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 EXP-001 full-data cache/E1 证据。

**改动原因**

V1.2 联合根目录可被 Runner 的 `_sequence_dirs()` 识别，但 E1 统计入口无法读取，导致 `compute_object_v2_stats --root cm_object_v2` 失败。修复后统计入口与 mixed config 的 root 语义一致。

**影响范围**

仅 Cm 数据读取和文档；不改变模型、loss 或 GT 定义。

## 2026-08-19 — 修正 Cm viewer 的 Python 3.8 兼容性

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/viewer/server.py` — 将 Python 3.9+ 的 `str.removesuffix()` 改为兼容 Python 3.8 的 `Path.stem`。

**改动原因**

仓库推荐环境为 Python 3.8.20；Scene Cache viewer 改动中的 `removesuffix()` 会在 legacy NPZ 推理路径运行时报错。

**影响范围**

仅 Cm viewer 的 side 名称解析，不改变推理结果。
