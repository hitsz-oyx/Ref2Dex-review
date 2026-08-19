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

## 2026-08-19 — 补齐 V1.2.1 单数据集对照配置与计数修正

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/compute_flow_scale.py` — object-v2 校准的 sequence 计数改为按完整路径去重，避免同名 sequence 被误合并。
- `src/task/Cm/configs/object_v2_grab_only.yaml` — GRAB-only 的 no-gate + time condition 对照配置。
- `src/task/Cm/configs/object_v2_arctic_only.yaml` — ARCTIC-only 的 no-gate + time condition 对照配置。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 GRAB-only / ARCTIC-only 小规模短训结果。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 更新当前研究状态和入口索引。
- `src/task/Cm/docs/logs/architecture_log.md` — 记录单数据集对照已经纳入 object-v2 管线。

**改动原因**

用户要求继续推进 V1.2.1，但又明确关心 seed 数量与训练规模；因此把后续工作收敛为同预算的单数据集对照，并修正校准元数据里的序列计数口径。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 单数据集对照配置已补齐，已完成 GRAB-only / ARCTIC-only 的小规模实验整理
- 剩余: 若要进一步判断假设，需要提高预算而不是增加重复 seed
- 续接点: 更长预算的 mixed / single-dataset 对照

**项目阶段进度**（可选）

- 阶段: V1.2.1 mixed + single-dataset baseline
- 进度: 2/2 个主要小规模验证步骤完成
- 本次对应阶段中的第几步: 2

**下一步打算做什么**（可选）

如果继续推进研究，应把 300-step 短训升级到更长预算；当前不建议再堆同级别 seed。


## 2026-08-19 — 记录 V1.2.1 吞吐 benchmark 与 DDP 兼容配置

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/configs/object_v2_grab_arctic.yaml` — 为 object-v2 mixed/full training 打开 `find_unused_parameters=true`。
- `src/task/Cm/configs/object_v2_grab_only.yaml` — 为 GRAB-only full training 打开 `find_unused_parameters=true`。
- `src/task/Cm/configs/object_v2_arctic_only.yaml` — 为 ARCTIC-only full training 打开 `find_unused_parameters=true`。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 记录 train-only 吞吐 benchmark 与推荐的 3 GPU DDP。
- `src/task/Cm/docs/logs/decision_log.md` — 记录 3 GPU DDP 的选型理由。

**改动原因**

用户要求开始 full-data 训练前先统计吞吐并选最合适的多卡方案；同时 2 GPU DDP 已经暴露 `no-gate` 路径上的 unused-parameter 问题，需要显式打开 `find_unused_parameters=true`。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 吞吐 benchmark 已完成，配置已适配 DDP
- 剩余: 等用户确认 full-training 具体范围后启动在线 wandb 长训
- 续接点: GRAB-only / ARCTIC-only / mixed 的正式 run

**项目阶段进度**（可选）

- 阶段: V1.2.1 full-data long-run prep
- 进度: benchmark + DDP 修正完成
- 本次对应阶段中的第几步: 1

**下一步打算做什么**（可选）

等待用户确认正式长训范围，然后用 3 GPU 启动在线 wandb 训练。


## 2026-08-19 — 记录 mixed full run 与 batch-size sweep 结论

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/docs/logs/repo_notes_log.md` — 记录 mixed 3 GPU batch-size sweep 的吞吐结果与正式推荐 batch。
- `src/task/Cm/docs/logs/decision_log.md` — 记录为什么正式长训选 3 GPU + batch_size=48。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 mixed full run（batch 24）完成事实与结果。

**改动原因**

用户要求按“多卡 + batch size 极限吞吐”选正式训练口径；因此先完成 train-only sweep，再把结果写回任务入口和自主决策记录。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 吞吐 sweep 已完成，mixed full run 已完成
- 剩余: 等待用户确认是否立即按 batch_size=48 继续长训 GRAB-only / ARCTIC-only / mixed
- 续接点: 正式长训启动

**项目阶段进度**（可选）

- 阶段: V1.2.1 long-run prep
- 进度: batch-size sweep 完成
- 本次对应阶段中的第几步: 2

**下一步打算做什么**（可选）

如果用户确认，就用 3 GPU + batch_size=48 开正式长训。

## 2026-08-19 — 将正式长训入口切到 batch 48 与 online wandb

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/configs/object_v2_grab_arctic.yaml` — mixed/full 训练默认 batch 调到 48，`wandb` 改为 online。
- `src/task/Cm/configs/object_v2_grab_only.yaml` — GRAB-only full 训练默认 batch 调到 48，`wandb` 改为 online。
- `src/task/Cm/configs/object_v2_arctic_only.yaml` — ARCTIC-only full 训练默认 batch 调到 48，`wandb` 改为 online。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 将正式长训入口描述更新为 3 GPU + batch 48。
- `src/task/Cm/docs/logs/decision_log.md` — 将正式长训决策锚定到 batch 48 的吞吐峰值。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 throughput sweep / mixed full pilot 的正式 EXP 记录。

**改动原因**

用户明确说明吞吐 benchmark 关注的是多卡 + batch size 的极限点，而不是单纯卡数；当前 sweep 已验证 3 GPU + batch 48 为 mixed 路线峰值，因此将正式长训入口统一收口到该设置，并把 pilot 与正式选择分开记录。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 正式长训入口已切到 48，下一步可直接启动在线训练
- 剩余: 等长训产出
- 续接点: mixed / GRAB-only / ARCTIC-only formal run

**项目阶段进度**（可选）

- 阶段: V1.2.1 full-data long-run prep
- 进度: 入口收口完成
- 本次对应阶段中的第几步: 3

**下一步打算做什么**（可选）

启动 3 GPU + batch 48 的正式长训。

## 2026-08-19 — 启动 mixed 正式长训

- branch: `oyx`
- post-commit: `3e32de8`
- 范围: task 内部

**文件**

- `output/exp/cm_v121/cm_v121_mixed_3gpu_bs48_full_20260819_165238.log` — mixed full-data 训练日志（运行产物，不纳入版本管理）。
- `src/task/Cm/docs/logs/experiment_log.md` — 将吞吐 EXP-004 锚定到提交 `3e32de8`。

**改动原因**

按 V1.2.1 的数据-only 约束和 EXP-004 吞吐结论，启动 3 GPU、per-device batch 48、global batch 144、online wandb 的 mixed 全量训练。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: torchrun 已启动，3 个 worker 正在加载 Frozen DenseToken 与 object-v2 数据。
- 剩余: 等待 10000-step 训练、验证和 checkpoint 完成。
- 续接点: 检查日志中的 `train_setup`、step 进度、W&B 初始化和最终 metrics。

**项目阶段进度**（可选）

- 阶段: V1.2.1 full-data long-run
- 进度: mixed 1/3 个正式数据路线已启动；GRAB-only / ARCTIC-only 待 mixed 完成后按相同吞吐设置启动。

**下一步打算做什么**（可选）

按分钟级检查 mixed 运行状态；完成后记录最终指标并启动两个 single-dataset 对照。
