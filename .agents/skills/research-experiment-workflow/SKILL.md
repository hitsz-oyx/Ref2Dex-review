---
name: research-experiment-workflow
description: 运行并记录可复现的训练、评估、benchmark、数据处理和长任务，分离运行状态与科研证据。
metadata:
  short-description: 让运行可复现且结论可审计
---

# 科研实验工作流

本 Skill 用于 run 模式。它不修改研究代码或变量；需要修改时先切换到 change 并经过相应审批。

## 1. 运行前固定合同

读取最小上下文：AGENTS.md、docs/current_versions.yaml、Task README、相关指导谱系/plan、实验定义、架构和必要的输入
manifest。确认：

- research_version、可复现的 git_commit 和 Task；
- hypothesis、变量、对照、数据 split、GT、坐标/单位和 checkpoint 初始值；
- metric、预算、资源、停止条件和输出目录；
- 研究语义是否保持不变。任何语义变化先停止并请求用户确认。

同一数值基线必须读取无后缀指导和所有已确认的 `a`–`z` 补充指导。不同文件的明确冲突由较后字母覆盖；同一文件
内部冲突、无法同时满足或语义不充分时，不得启动运行，必须请用户确认并先修订计划草案。用户确认后才可更新指导、
定稿计划和运行合同。

运行前固定代码和配置，生成独立 run_id。不得覆盖旧运行、静默选择“最新 checkpoint”或用机器私有事实替代数据合同。

## 2. 运行身份和产物

正式训练、评估、benchmark 和数据处理都生成小型 run_manifest.json，至少记录：

    task
    research_version
    git_commit
    run_id
    config snapshot
    input/cache manifest
    schema、坐标/单位和 split 引用
    seed
    initial checkpoint
    output directory

数据合同通过 manifest 或 metadata snapshot 引用，不把完整 metadata 再复制进 run manifest。运行输出按 docs/目录规范.md
写入 outputs/<Task>/<run_id>/ 或 Task research output；cache、checkpoint、原始数据和大型生成结果不提交。

## 3. 状态和证据

长期 Activity 记录重要生命周期终态和人工检查，不维护额外实时 heartbeat JSON；旧 activity_log 仅作历史审计。使用：

    run_status: STARTED | RUNNING | COMPLETED | FAILED | STOPPED | UNKNOWN
    conclusion: SUPPORTED | REFUTED | INCONCLUSIVE | INVALID_IMPLEMENTATION | N/A

run_status 只说明进程状态；conclusion 由 hypothesis、metric 和证据决定。运行成功不等于科研假设成立。终态 activity 至少
链接运行目录、manifest、config、实际生成的 metrics.jsonl/train.log、最后 step/epoch、best metric、最佳/最近 checkpoint
（存在时）和失败/停止原因。

正式实验使用 `experiments/` 下独立 experiment card，记录 hypothesis、configuration、runs、evidence、conclusion 和 limitations。实验索引
只做导航；重要性标签不代表结果成立。

## 4. 长任务

启动前确认资源和停止条件，启动后按分钟级等待。只有新输出、错误、资源冲突或会改变下一步决策的中间信息才提前检查。
出现未批准的范围变化、安全风险或输入漂移时停止并记录原因。不要覆盖已有运行或 checkpoint。

## 5. 验证和交接

实验入口在第二阶段接入统一门禁：

    python tools/verify.py --changed

第一阶段工具尚不存在时，记录实际运行命令、配置快照、输入引用、定向 smoke/测试和 manifest 检查，不得称为 VERIFY PASS。
smoke 只能证明 wiring 或可运行性，不能证明研究效果。

最终回复必须给出 Task、research_version、git_commit、run_id、run_status、输出目录、manifest、metrics/log、checkpoint
（存在时）、工程验证和独立科研 conclusion。运行终态由最近作用域 Activity 作为唯一状态入口；历史 summary 可保留，新 BaseRunner
不自动生成标准 summary。
