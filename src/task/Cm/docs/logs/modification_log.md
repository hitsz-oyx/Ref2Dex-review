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

## 2026-08-19 — 推进 V1.2.1 数据-only 混合实验链路

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/dataset_object_v2.py` — 用共享 epoch 和 LRU 打开缓存修复多进程读取稳定性，并支持固定 split。
- `src/task/Cm/compute_flow_scale.py` — 增加 object-v2 train-only flow calibration 入口。
- `src/task/Cm/build_object_v2_splits.py` — 生成 sequence 级固定 split 与 `splits.json`。
- `src/task/Cm/configs/object_v2_grab_arctic.yaml` — 固定为 no-gate + time condition，并接入固定 split 与校准标记。
- `src/task/Cm/runner.py` — 识别联合 `grab/` + `arctic/` root 的 object-v2 数据目录。
- `src/task/Cm/docs/指导/V1.2.1.md` — 研究指导版本，作为本轮实现与实验依据。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 更新任务入口状态与主要入口索引。
- `src/task/Cm/docs/logs/architecture_log.md` — 记录固定 split、校准和联合 root 的数据流约束。
- `src/task/Cm/docs/logs/decision_log.md` — 记录共享 epoch、LRU cache 和确定性 split 的实现选择。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 V1.2.1 混合短训实验结果与结论。

**改动原因**

按 V1.2.1 指导修复 worker epoch、cache 泄漏和训练 split，并按用户确认采用只改数据链路的 no-gate + time condition 方案，随后用 2-step smoke 与 3-seed 300-step 短训验证实现闭环。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 代码、配置、文档和短训验证都已完成
- 剩余: 下一轮对照实验
- 续接点: GRAB-only / ARCTIC-only / mixed 同预算比较

**项目阶段进度**（可选）

- 阶段: V1.2.1 mixed data-only baseline
- 进度: 1/2 个主要步骤完成
- 本次对应阶段中的第几步: 1

**下一步打算做什么**（可选）

继续跑 GRAB-only、ARCTIC-only 和 mixed 对照，必要时再加长训练步数。
