# 全局 AI 修改记录

## 2026-08-20 — 接入外部 HRDexDB Inspire F1 数据路径

- branch: working tree
- post-commit: HEAD
- 范围: 跨 task

**文件**

- `docs/logs/repo_notes_log.md` — 记录 HRDexDB 数据根、Inspire F1 URDF 和 LFS 下载环境约定。
- `src/task/CmDecoder/` — 新建 frozen-Cm 动作重建任务（详细改动见任务级 modification log）。

**改动原因**

用户要求利用外部 HRDexDB Inspire F1 数据和既有 GRAB-only Cm checkpoint 训练关节角 Decoder；外部数据路径与下载约定具有仓库级复用意义。

## 2026-08-20 — 修正共享 GRAB raw asset 解析

- branch: `oyx`
- post-commit: `HEAD`
- 范围: 跨 task

**文件**

- `process/GRAB/raw.py` — 统一 sequence root 与 subject `v_template` 的相对路径解析，并支持严格失败。
- `process/GRAB/stage4_cm.py` — 正式 Cm Stage4 默认禁止缺失 subject template，写入可追溯 metadata。
- `tests/test_cm_sequence_dataset.py` — 增加 raw layout 回归测试。

**改动原因**

共享 GRAB adapter 的旧拼接逻辑会在 `dataset/GRAB` root 下找错 `tools/subject_meshes`，静默回退平均 MANO；这会污染 Cm/V2 hand geometry。

**影响范围**

共享 GRAB 数据处理；Cm/V2 需重建受影响 cache。

## 2026-08-18 — 建立根级规范文档入口

- branch: 当前工作分支
- post-commit: `HEAD`
- 范围: 全局

**文件**

- `docs/logs/architecture_log.md` — 补充仓库架构、共享数据流和 Task/Cm 关系。
- `docs/logs/repo_notes_log.md` — 归档运行环境、数据路径和文档索引。
- `docs/logs/decision_log.md` — 建立全局决策入口。
- `docs/logs/modification_log.md` — 建立全局修改记录入口。

**改动原因**

按仓库 `AGENTS.md` 将原有单一项目总览整理为根级五类文档入口；不改变代码、数据或实验定义。

**影响范围**

全局文档结构。
