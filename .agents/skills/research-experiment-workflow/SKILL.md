---
name: research-experiment-workflow
description: 运行并记录可复现的科研实验、评估、数据处理和长时任务，分离状态、证据与生成产物。
metadata:
  short-description: 让实验可复现且可追溯
---

# 科研实验工作流

训练、评估、benchmark、数据处理、可视化、模型下载、编译、仿真，或预计超过短命令的任务，都使用本 Skill。

## 运行前

- 阅读项目当前状态、架构、记忆，以及相关 experiment/decision 记录。遵循项目路径和 schema，不用机器私有假设替代它们。
- 明确假设、变量、数据划分、checkpoint 初始值、指标、预算和停止条件。如果其中任何一项改变科研语义，先请用户确认。
- 将正式实验证据与当前状态、实现历史分开记录。

## 运行中与运行后

- 新实验使用明确的新输出目录；不要覆盖基线，也不要静默选择“最新 checkpoint”。
- 缓存、checkpoint、原始数据和大型生成结果不纳入版本控制。
- 记录确切命令、配置快照、代码版本、输入/缓存标识和定量结果。结论标记为 `SUPPORTED`、`REFUTED`、`INCONCLUSIVE` 或 `INVALID_IMPLEMENTATION`。
- smoke test 只能证明 wiring 正常，不能作为科研结果。

## 长时任务等待规则

完整规则见 [long-running-tasks.md](references/long-running-tasks.md)。核心要求是：让执行工具等待，不要让模型每几秒重新推理一次来查询状态。

- 启动长任务后按分钟级等待；只有中间输出会改变下一步决策时才提前检查。
- 出现错误、资源冲突或未批准的范围变化时安全停止。
- 空的状态查询与实际交互输入必须区分：前者长等待，后者立即发送。
