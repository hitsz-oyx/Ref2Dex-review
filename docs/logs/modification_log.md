# Modification log

## 2026-08-23 — 启动全量 ARCTIC MANO 数据导出

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: 跨 task / 全局

**文件 / 数据**

- `docs/logs/status_log.md` — 新建根级状态入口，记录 ARCTIC 导出与现有 mixed 训练的并行状态。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 记录全量 ARCTIC Stage 2/3 导出正在运行及完成后的核验要求。
- `output/research/arctic_full_mano_v21_20260823/run_export.sh` — 后台顺序执行 Stage 2、Stage 3 的被忽略启动脚本。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage2/arctic_full_mano_v21_20260823` — 全量 ARCTIC MANO Stage 2 输出目录。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_full_mano_v21_20260823` — Stage 2 成功后自动生成的 v2.1 hand-root Stage 3 目录。

**改动原因**

用户确认先把全量 ARCTIC 左右手、stride 1、含 MANO 参数的数据导出到 NAS，并要求后台运行。任务使用物理 GPU 3 的 tmux `arctic_full_mano_v21_20260823`，不覆盖旧 ARCTIC 数据，也不干扰 GPU 1、2 上的 mixed 训练。

## 2026-08-21 — 将六类日志规则统一收回文档模板区

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree (未提交)`
- 范围: 全局

**文件**

- `AGENTS.md` — 删除独立的实验、决策和修改记录章节，将其边界与模板统一收回六类文档职责说明，并顺延后续章节编号。
- `docs/logs/modification_log.md` — 记录本次规范统一。

**改动原因**

用户确认六类文档应采用统一结构，避免实验、决策和修改记录被单独拎出形成重复且过强的规范。

**对应指导**

无

**影响范围**

全局；未修改任何 task 级具体日志或架构内容。

## 2026-08-21 — 统一六类日志模板并收敛实验规范

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree (未提交)`
- 范围: 全局

**文件**

- `AGENTS.md` — 为六类日志加入统一元信息外壳和最小模板；将运行环境并入 `memory_log.md`，将自主决策和修改记录模板集中到文档职责说明；裁剪 `experiment_log.md` 的必填字段并将详细观察、解释和下一步改为可选。
- `docs/logs/modification_log.md` — 记录本次规范收敛。

**改动原因**

按用户确认降低模板的强制性，减少重复规则，保留科研追溯所需的最小信息，同时给不同 Task 保留正文结构自主性。

**对应指导**

无

**影响范围**

全局；未修改任何 task 级具体日志或架构内容。

## 2026-08-21 — 将递归日志规范拆分为状态、记忆与完整架构事实源

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree (未提交)`
- 范围: 全局

**文件**

- `AGENTS.md` — 将 AI 维护文档规范从五类调整为六类，新增 `status_log.md` 与 `memory_log.md`，并规定 `architecture_log.md` 为不依赖 `docs/架构.md` 的完整架构事实源。
- `docs/logs/modification_log.md` — 记录本次全局规范修改。

**改动原因**

根据用户确认，分离当前状态与接手记忆，明确环境/路径/历史遗留和架构事实的归属，避免状态、memory、架构和修改记录互相混写。

**对应指导**

无

**影响范围**

全局；仅修改文档规范，未迁移任何 task 的具体日志内容或架构内容。

## 2026-08-20 — 将 GRAB 大体积数据入口切到 NAS

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: 跨 task / 全局

**文件**

- `dataset/GRAB/data` — 软链到 NAS 上的 GRAB 数据根。
- `docs/logs/repo_notes_log.md` — 记录仓库级 NAS 数据根路径。

**改动原因**

本地根分区已接近满盘，而 GRAB 的大体积只读数据在 NAS 上已有完整副本。把入口改到 NAS 可以释放本地空间，并让后续任务统一引用同一份数据根。

**对应指导**

无

**影响范围**

跨 task / 全局

## 2026-08-20 — 将 processed_data 迁到 NAS 并补结果配置摘要

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: 跨 task / 全局

**文件**

- `data/processed_data` — 迁移到 NAS 路径 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data` 的软链。
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` — 补充三模型对比对应的训练配置摘要。
- `docs/logs/repo_notes_log.md` — 记录仓库级 processed_data NAS 路径。

**改动原因**

用户要求把本地 `processeddata` 也放到 NAS，并把当前对比结果里对应的训练配置写清楚，方便后续复现和检查配置差异。

**影响范围**

跨 task / 全局
## 2026-08-24 — 新增 HOCap 外部测试数据转换入口

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: 跨 task / 全局

**文件 / 数据**

- `process/HOCap/__init__.py` — 新增 HOCap 处理包入口。
- `process/HOCap/stage3_export.py` — 新增 annotation-only HOCap 到 Ref2Dex Stage 3 的转换器；输出按序列、物体实例和有效手拆分。
- NAS `processed_data/stage3/hocap_subject1_annotation_v1` — 启动 subject_1 外部测试数据转换，不加入训练。

**改动原因**

HOCap 是新增外部数据域，影响仓库级数据处理路径；转换器固定 HOCap MANO PCA45、标准 1538 hand face points 和 hand-root 坐标契约，同时保留原始 HOCap 数据不变。
