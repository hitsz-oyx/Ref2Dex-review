# Ref2Dex Agent 规范

本文件只保留 Ref2Dex 的本地事实和必须始终生效的安全规则。通用修改治理和实验流程由仓库内的 Skill 提供，避免每次任务加载一份过长的总说明。

## 0. 需求确认与高风险边界

开始修改代码、配置、数据处理逻辑或实验实现前，先确认目标、范围、保持不变的内容和科研约束。涉及坐标系、单位、GT、数据 split、cache/schema、评估指标、checkpoint 解释、公共 `src/base`、仓库治理、数据迁移、破坏性操作或长时间任务时，必须先向用户说明方案并获得确认。

不确定改动等级时按高等级处理。普通实现细节可以自主选择，但不得擅自改变研究目标、数据语义或关键 invariant。

## 1. 默认语言

项目文档、研究日志、实验记录和 Git 提交信息默认使用中文。代码标识符、接口名、命令、配置字段和兼容 schema 字段保留英文。

## 2. Skill 路由

这些 Skill 位于 `.agents/skills/`，内容应保持任务无关，未来可整体复制到其他仓库：

- 修改代码、配置或文档：使用 `research-change-control`。
- 训练、评估、数据处理、benchmark 或长任务：使用 `research-experiment-workflow`。
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

### 4.1 指导与执行计划

任务研究方向由用户提供的 `src/task/<Task>/docs/指导/V<n>.md` 定义；具体代码修改前，
必须与用户协商形成同版本的 `src/task/<Task>/docs/plan/V<n>.md`。两者一一对应：

- `指导/V<n>.md`：研究目标、科学假设、希望探索的方向和明确的禁止事项；用户负责提供和修订；
- `plan/V<n>.md`：将指导转成可执行的文件范围、接口/schema、保持不变的不变量、修改等级、
  审批边界、验证命令和回滚方式；由 Agent 起草，和用户商讨后定稿；
- 计划在协商期间允许反复修改，只有明确标记为最终版本后，Agent 才能按该计划修改代码、配置、
  数据处理或实验流程；计划未定稿时只能阅读、分析和提出方案；
- 计划版本号必须与指导版本号一致。若实现过程中发现最终计划无法执行或科研语义需要变化，
  必须暂停并重新协商，不得静默偏离；
- 协商中的 plan 草稿变化不写入 `modification_log.md`。定稿后的 plan 版本和实际实现结果，
  在执行完成时一并写入对应的最终修改记录；plan 不是实验记录、架构事实源或修改流水账。

开始 Task 工作时，除读取指导文档外，还应检查同版本 `docs/plan/V<n>.md` 是否存在、是否已定稿。
没有定稿计划时，先进入计划协商，不得直接实现。

涉及实验读取 `experiment_log.md`；涉及已有设计选择读取 `decision_log.md`；修改、回归或历史行为读取相关 `modification_log.md`。文档用于导航，代码和可复现实验结果是事实最终来源；发现不一致时先调查再更新文档。

## 5. 通用交接与版本规范

### 5.1 完成汇报与规范反馈

每次完成代码、配置、数据处理、实验、诊断或文档工作后，Agent 的最终汇报必须包含：

- 完成结果、实际修改范围、未修改的保护边界和可回滚入口；
- 使用的版本线/细分版本、plan 或指导版本、操作类别和审批状态；
- 验证命令、关键结果、输出/manifest 入口，以及结果属于 `SUPPORTED`、`REFUTED`、`INCONCLUSIVE` 还是 `INVALID_IMPLEMENTATION`；
- 明确区分工程 smoke 证据与科研结论，不把实现可运行误报为效果成立；
- 本次是否遇到格式、目录、版本、日志或审批规则的阻碍。

如果现有格式或规范迫使 Agent 采取不自然、重复、易错或无法审计的做法，Agent 必须在完成汇报中单独列出“规范反馈”，说明：

1. 具体受限场景和当前 workaround；
2. 可能造成的风险或维护成本；
3. 建议新增/修改的规则、适用范围和兼容影响；
4. 是否需要用户确认后才能修改 `AGENTS.md`、Skill 或公共合同。

Agent 可以提出规范改进建议，但不得因为建议本身擅自改变用户指导、版本线、架构快照或公共合同。只有用户明确要求或确认后，才把建议写入 `AGENTS.md`、Skill 或对应版本文档；规则修改本身也要记录在最近作用域的 `modification_log.md`。

### 5.2 版本线与 AI 行为

仓库按 `Vn/Vn.m/Vn.m.k` 版本线工作：`Vn` 由用户确认创建，`Vn.m` 必须有同版本最终 plan，
`Vn.m.k` 用于该 plan 下的具体实现、并行实验和状态变化，不单独创建 plan。指导文档只由用户
更新或在用户明确指令下更新；架构快照只有用户明确要求时才更新。每次实质改动都必须在相关
status、experiment、decision 或 modification 记录中写完整细分版本和操作类别。

操作类别固定为 `governance`、`architecture`、`code`、`data`、`experiment`、`diagnostic`、
`operation`、`documentation`；无法归类时必须在回复中说明并补充类别。完整规则见
[`docs/版本线与AI行为分类.md`](docs/版本线与AI行为分类.md)。

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

## 8. Ref2Dex 仓库特有约定

### 8.1 路径、运行和目录路由

- `src/base/`：共享配置、Runner、数据划分、checkpoint、分布式和训练基础设施。
- `process/`：数据集 adapter 及预处理入口。
- `src/task/<Task>/`：Task 实现；诊断脚本放在 `src/task/<Task>/research/<实验名>/`。
- `outputs/`、`src/task/*/research/*/output/`、训练输出、cache、原始数据和 checkpoint 默认不提交；根 `output/` 只作为历史兼容目录，不再新增内容。

Agent 不需要用户为每次小改动指定目录。开始工作时先按内容自动归类；只有会改变科研语义、公共合同或共享基础设施的边界才向用户确认。

| 内容 | 默认位置 |
|---|---|
| 共享运行时、配置、分布式和 checkpoint 基础设施 | `src/base/` |
| Task 模型、Runner、Dataset 和局部工具 | `src/task/<Task>/src/`、`dataset/` |
| Task 配置 | `src/task/<Task>/configs/active/` 或 `archive/` |
| 诊断脚本与实验定义 | `src/task/<Task>/research/<experiment>/`（必须包含 `README.md`、`experiment.yaml`；不得在 `research/` 根目录放散脚本） |
| 指导与执行计划 | `src/task/<Task>/docs/指导/`、`docs/plan/` |
| 状态、架构、实验、决策和修改记录 | 最近作用域的 `docs/logs/` |
| 训练 / 评估运行及其追溯文件 | `outputs/<Task>/<run_id>/`（遵循 BaseRunner） |
| 诊断运行及其产物 | `src/task/<Task>/research/<experiment>/output/<run_id>/` |
| 数据和 cache | 配置声明的 `data/` 或 cache root，并在其旁保存数据 manifest |
| Task 专属预训练模型、body model、URDF 等外部资产 | `src/task/<Task>/assets/<kind>/<asset_id>/`；训练生成的 checkpoint 仍在 `outputs/`；根 `assets/` 仅允许历史兼容软链接 |

训练、评估、benchmark 或正式数据处理运行必须生成 `run_manifest.json`（位于
`outputs/<Task>/<run_id>/` 或对应 research 实验的 `output/<run_id>/`），至少记录 Task、实验、指导/plan
版本、Git 提交、配置快照、输入数据或 cache manifest 及其文件基本信息、schema、坐标系、seed、初始 checkpoint
和输出入口。`config.json` 或 `metadata.json` 可以作为组成部分，
但不能替代这份统一追溯清单。

`experiment.yaml` 是实验定义；数据/cache 由配置直接声明，Task 专属外部资产放在被忽略的
`src/task/<Task>/assets/` 并由配置直接引用；一次具体运行使用 `run_manifest.json`。
这些文件职责不同，不得互相替代。
