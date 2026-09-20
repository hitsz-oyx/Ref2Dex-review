---
name: research-experiment-workflow
description: 设计并记录可复现的训练、评估、benchmark 和数据处理实验，分离运行状态与科研证据。
metadata:
  short-description: 让运行可复现且结论可审计
---

# 科研实验工作流

本 Skill 用于 run 模式。它不修改研究代码或变量；需要修改时先切换到 change 并经过相应审批。长任务的启动、等待、停止与恢复使用 `long-running-tasks`。

## 运行前固定合同

读取 AGENTS、current_versions、Task README、完整指导谱系、适用最终 plan、实验定义和必要输入 manifest。确认 work_version、可复现 git_commit、Task、hypothesis、变量、对照、数据 split、GT、坐标/单位、初始 checkpoint、metric、预算、资源和停止条件。任何研究语义变化先停止并请求用户确认。

同一数值基线必须读取无后缀指导和全部已确认 `a`–`z` 补充指导；跨文件冲突以后字母覆盖，同一文件冲突则停止并按 plan 草案/用户确认流程处理。计划补充只在后续指导改变已执行计划的实施合同时存在，不与每个指导补充一一对应。

## 身份、产物与证据

work_version 是唯一版本术语；git_commit 与 run_id 分别标识实现和运行。正式训练、评估、benchmark 和数据处理生成小型 run_manifest.json，至少记录 task、work_version、git_commit、run_id、配置快照、输入/cache manifest、schema、坐标/单位、split、seed、初始 checkpoint 和输出目录。旧 modification_version 只在读取历史产物时兼容；新产物不得写它。

路径、manifest 与测试归属使用 `directory-and-artifacts`。不得覆盖旧运行、静默选择“最新 checkpoint”，或提交 cache、checkpoint、原始数据和大型生成结果。

重要运行生命周期终态和人工检查写入最近作用域 `docs/activities/` 的独立 Activity；科学证据写入 `docs/experiments/` 的 experiment card。Activity 回答“做了什么”，experiment 回答“知道了什么”。`logs/activity_log.md` 与 `logs/experiment_log.md` 仅是历史审计，不能作为当前状态入口或新写入位置。

使用：

    run_status: STARTED | RUNNING | COMPLETED | FAILED | STOPPED | UNKNOWN
    conclusion: SUPPORTED | REFUTED | INCONCLUSIVE | INVALID_IMPLEMENTATION | N/A

run_status 只说明进程状态；conclusion 由 hypothesis、metric 和证据决定。终态 Activity 至少链接运行目录、manifest、config、实际生成的 metrics.jsonl/train.log、最后 step/epoch、best metric、最佳/最近 checkpoint（存在时）和失败/停止原因。

## 交接

实际启动、等待、监控、停止和恢复遵循 `long-running-tasks`。本 Skill 负责在运行前固定
科学变量，并在运行后按 hypothesis、metric 和证据给出独立 conclusion；运行状态本身不构成
科学结论。

统一机器门禁为：

    python tools/verify.py --changed

最终回复给出 Task、work_version、git_commit、run_id、run_status、输出目录、manifest、metrics/log、checkpoint（存在时）、工程验证和独立 scientific conclusion。smoke 只能证明 wiring 或可运行性，不能证明研究效果。
