# Ref2Dex Agent 规范

本文件只保留 Ref2Dex 的本地事实和必须始终生效的安全规则。通用修改治理、实验流程和可组合组件规则分别由仓库内的 Skill 提供，避免每次任务加载一份过长的总说明。

## 0. 需求确认与高风险边界

开始修改代码、配置、数据处理逻辑或实验实现前，先确认目标、范围、保持不变的内容和科研约束。涉及坐标系、单位、GT、数据 split、cache/schema、评估指标、checkpoint 解释、公共 `src/base`、仓库治理、数据迁移、破坏性操作或长时间任务时，必须先向用户说明方案并获得确认。

不确定改动等级时按高等级处理。普通实现细节可以自主选择，但不得擅自改变研究目标、数据语义或关键 invariant。

## 1. 默认语言

项目文档、研究日志、实验记录和 Git 提交信息默认使用中文。代码标识符、接口名、命令、配置字段和兼容 schema 字段保留英文。

## 2. Skill 路由

这些 Skill 位于 `.agents/skills/`，内容应保持任务无关，未来可整体复制到其他仓库：

- 修改代码、配置或文档：使用 `research-change-control`。
- 训练、评估、数据处理、benchmark 或长任务：使用 `research-experiment-workflow`。
- Component、Artifact、Contract、Pipeline 或 adapter：使用 `modular-component-runtime`。

Skill 负责通用流程；本文件负责下面的 Ref2Dex 特殊路径、日志和科学事实。

## 3. 六类递归日志

日志按最近作用域维护，不为形式批量创建空文件：

| 文档 | 文件名 | 用途 |
|---|---|---|
| status | `status_log.md` | 当前阶段、进行中、阻塞和下一步快照 |
| memory | `repo_memory.md` / `machine_memory.md` | 可迁移事实 / 本机环境事实 |
| architecture | `architecture_log.md` | Pipeline、数据流、schema、张量、指标和 invariant |
| experiment | `experiment_log.md` | 正式实验假设、结果和结论 |
| decision | `decision_log.md` | 需求明确后的非平凡自主选择 |
| modification | `modification_log.md` | AI 实际修改的文件、原因和验证 |

根级路径为 `docs/logs/`，Task 级路径为 `src/task/<Task>/docs/logs/`。根级只记录跨 Task 或全仓库事实；Task 级记录局部事实。`machine_memory.md` 必须被 Git 忽略，`repo_notes_log.md` 已废弃且禁止重新创建。

修改记录必须包含 `change_level`、`approval`、分支、scope、文件、原因和验证。文档自身的修改也只记录在 `modification_log.md`，不建立 changelog。

## 4. 开始修改 Task 前的入口

按需读取：

```text
docs/logs/status_log.md
docs/logs/architecture_log.md
docs/logs/repo_memory.md
docs/logs/machine_memory.md（存在时）
src/task/<Task>/docs/logs/status_log.md
src/task/<Task>/docs/logs/architecture_log.md
src/task/<Task>/docs/logs/repo_memory.md
src/task/<Task>/docs/logs/machine_memory.md（存在时）
相关 experiment/decision/modification 日志和指导文档
相关代码
```

涉及实验读取 `experiment_log.md`；涉及已有设计选择读取 `decision_log.md`；修改、回归或历史行为读取相关 `modification_log.md`。文档用于导航，代码和可复现实验结果是事实最终来源；发现不一致时先调查再更新文档。

## 5. Ref2Dex 特殊路径与运行约定

- `src/base/`：共享配置、Runner、数据划分、checkpoint、分布式和训练基础设施。
- `process/`：数据集 adapter 及预处理入口。
- `src/task/<Task>/`：Task 实现；诊断脚本放在 `src/task/<Task>/research/<实验名>/`。
- `components/`：通用 manifest、registry 和示例；入口由 `tools/researchctl.py` 发现与校验。
- `output/exp/`、`output/research/`、训练输出、cache、原始数据和 checkpoint 默认不提交。

公共 Component 约定是 `Component / Artifact / Contract / Pipeline / Experiment`；`encoder`、`decoder`、MANO、数据集和具体 Task 只能作为 capability、tag 或合同约束，不作为框架一级抽象。组件发现、list、describe 和 dry-run 不得启动训练或加载真实权重。

## 6. Git 与用户改动保护

提交前检查：

```bash
git status
git diff
git diff --cached
```

只 `git add` 当前任务相关的显式路径。用户已有的修改、未跟踪实验脚本、数据、cache、output 和 checkpoint 不得被覆盖、reset、revert 或带入提交；不确定归属时先询问用户。提交信息默认中文且描述实际研究/实现结果。

## 7. 子任务

简单问题由当前 Agent 完成；只有独立调查、独立模块、独立实验或需要单独 review 时才拆分子任务，不为形式上的 Agent 化创建层层代理。
