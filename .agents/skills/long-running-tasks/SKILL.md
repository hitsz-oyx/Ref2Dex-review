---
name: long-running-tasks
description: 启动、等待、监控、停止或恢复训练、仿真、数据处理等长任务时使用；不决定实验方法或科学结论。
metadata:
  short-description: 让长任务等待与状态记录保持受控
---

# 长任务操作

本 Skill 适用于训练、仿真、编译、数据处理、模型下载、视频渲染、benchmark 及其他预计
运行超过几十秒的命令。它只处理运行操作与状态记录；实验变量、科学证据和 conclusion
遵循 `research-experiment-workflow`，路径与产物遵循 `directory-and-artifacts`。

启动前确认运行已经具备固定的代码提交、work_version、run_id、输入、资源、预算和停止
条件。若发现输入漂移、未批准范围变化、安全风险或研究变量变化，停止运行并转回相应的
change/审批流程。

## 等待与观察

避免以高频 polling 反复唤醒模型。让工具等待，只有新输出、错误、资源冲突或可能改变下
一步决策的信息才提前检查；预计数分钟的任务按分钟级而非秒级检查。

纯状态查询时：

- 对空 `write_stdin` 查询，`yield_time_ms` 不低于 `180000`；不需要中间输出时优先
  `300000`。
- 对异步 `functions.wait`，使用不低于 `180000` 的等待时间。
- 若外层 `functions.exec` 包含内部等待，外层 `@exec yield_time_ms` 至少比最长内部等待多
  `30000 ms`。

实际交互输入（例如确认、换行或 Ctrl-C）应立即发送；长等待规则只适用于不发送内容的
状态查询。

## 状态、停止与恢复

使用以下运行状态，且不要把它们当成科学结论：

```text
run_status: STARTED | RUNNING | COMPLETED | FAILED | STOPPED | UNKNOWN
```

- 启动后记录命令、run_id、输出目录、manifest、资源和当前状态。
- 终态或需要人工决策时记录最后 step/epoch、best metric、关键 checkpoint（存在时）、
  `metrics.jsonl`/`train.log` 入口及失败或停止原因。
- 恢复前确认同一 work_version、代码提交、配置、输入/cache manifest 和 checkpoint 解释仍然
  一致；否则停止并请用户确认，不得静默续跑或选择“最新 checkpoint”。
- 运行终态及重要人工检查写入最近作用域的 Activity；科学 hypothesis、evidence 和
  conclusion 写 experiment card。`run_status` 与 conclusion 必须分开。

工程 smoke 只能证明接线或可运行性，不能证明研究效果。运行结束后按
`research-experiment-workflow` 解释证据并交接。
