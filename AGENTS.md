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
- 按任务模式选择闸门：`read-only/diagnostic`、`run-only/operation`、`change`、`governance`；只读检查不因缺少 plan 被阻塞，运行已有实验不默认修改代码或研究变量。

每次请求开始时先在交接中声明一个主任务模式；如果执行中从只读检查转为运行或变更，必须重新确认范围并切换相应闸门：

| 模式 | 允许动作 | 必须留下的记录 |
| --- | --- | --- |
| `read-only/diagnostic` | 阅读、扫描、解释、只读查询和诊断 | 若产生新证据，记录 activity；不得改代码、配置、数据或科研结论 |
| `run-only/operation` | 按既有 final plan/experiment 运行、等待、停止、恢复和查询 | activity 的 `run_id`、`run_status`、命令、`base_commit`、输出和检查依据；正式实验另写 experiment |
| `change` | 按 final plan 修改代码、配置、数据处理或测试 | activity 的等级、审批、文件、原因、验证和回滚入口 |
| `governance` | 修改 AGENTS、Skill、目录、版本或公共记录合同 | L3、用户批准、最终 plan、完整 diff 审计和规范反馈 |

模式本身不改变科研目标；任何模式都不得覆盖用户已有改动、旧运行或不可逆产物。
Skill 负责通用流程；本文件负责下面的 Ref2Dex 特殊路径、日志和科学事实。

### 2.1 文档导航与加载规则

`AGENTS.md` 是仓库级常驻指令；普通 `docs/*.md` 不会因为存在而自动加载。以下文件按任务条件主动读取：

- [`docs/README.md`](docs/README.md)：文档入口和职责导航，所有任务先读。
- [`docs/current_versions.yaml`](docs/current_versions.yaml)：当前各作用域的 `modification_version` 指针，所有会产生记录或运行的任务先读。
- [`docs/modification_policy.md`](docs/modification_policy.md)：版本线、`modification_version` 和操作类别；涉及版本判断、plan/指导、activity 归档、治理或文档规范修改时必读。
- [`docs/目录规范.md`](docs/目录规范.md)：数据、cache、资产、research、运行目录和 manifest 的放置规则；创建、移动或判断这些产物路径时必读。
- [`docs/ai_task_checklist.md`](docs/ai_task_checklist.md)：开始/完成交接的摘要清单；复杂交接、长任务或用户要求复核时读取。它不新增或覆盖其他规范。

文档权威性按以下顺序解释：本文件的安全与审批规则优先；具体路径和产物职责以 `docs/目录规范.md` 为准；版本和操作分类以 `docs/modification_policy.md` 为准；交接清单仅作检查表。发现冲突时暂停并记录，不静默选择。

## 3. 递归记录与唯一时间线

记录按最近作用域维护，不为形式批量创建空文件。`activity_log.md` 是从切换点开始的唯一工作活动时间线；历史 `modification_log.md` 只作只读审计记录，不再新增条目：

| 文档 | 文件名 | 用途 |
|---|---|---|
| activity | `activity_log.md` | 代码、配置、数据、训练、评估、诊断、启动/停止和产物的摘要时间线 |
| memory | `repo_memory.md` / `machine_memory.md` | 可迁移事实 / 本机环境事实 |
| architecture | `architecture_log.md`（现有）或 `architecture.md`（目标命名） | Pipeline、数据流、schema、张量、指标和 invariant |
| experiment | `experiment_log.md` | 正式实验假设、结果和结论 |
| decision | `decision_log.md` | 需求明确后的非平凡自主选择 |

根级路径为 `docs/logs/`，Task 级路径为 `src/task/<Task>/docs/logs/`。根级只记录跨 Task 或全仓库事实；Task 级记录局部事实。`machine_memory.md` 必须被 Git 忽略，`repo_notes_log.md` 已废弃且禁止重新创建。`status_log.md` 从切换点起不再作为规范入口。

活动条目按事件类型记录必要字段：修改事件包含 `change_level`、`approval`、分支、scope、文件、原因和验证；运行事件包含 `run_id`、`run_status`、命令、输出和证据入口。所有活动条目都包含精确到秒的 `timestamp`、`activity_id`、`modification_version` 和 `base_commit`。

## 4. 开始修改 Task 前的入口

所有任务先读取文档入口和当前版本指针，再按 §2.1 的条件读取规范文件；随后读取最近作用域的事实记录和相关代码：

```text
docs/README.md
docs/current_versions.yaml
docs/logs/activity_log.md
docs/logs/architecture_log.md
docs/logs/repo_memory.md
docs/logs/machine_memory.md（存在时）
src/task/<Task>/docs/logs/activity_log.md
src/task/<Task>/docs/logs/architecture_log.md
src/task/<Task>/docs/logs/repo_memory.md
src/task/<Task>/docs/logs/machine_memory.md（存在时）
相关 experiment/decision/activity 历史、指导和最终 plan
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
- 协商中的 plan 草稿变化不写入 `activity_log.md`。定稿后的 plan 版本和实际实现结果，
  在执行完成时一并写入对应的最终修改记录；plan 不是实验记录、架构事实源或修改流水账。

开始 `change` 或改变研究变量的 `run-only/operation` 工作时，除读取指导文档外，还应检查同数字后缀的 `docs/plan/V<n>.md` 是否存在、是否已定稿；没有定稿计划时，先进入计划协商，不得直接实现或运行。
`read-only/diagnostic` 可以在没有新 plan 时执行，但不得写入代码、配置、数据、实验结论或架构事实。

涉及实验读取 `experiment_log.md`；涉及已有设计选择读取 `decision_log.md`；修改、回归或历史行为读取相关 `activity_log.md` 和历史 `modification_log.md`。文档用于导航，代码和可复现实验结果是事实最终来源；发现不一致时先调查再更新文档。

## 5. 通用交接与版本规范

### 5.1 完成汇报与规范反馈

每次完成代码、配置、数据处理、实验、诊断或文档工作后，Agent 的最终汇报必须包含：

- 完成结果、实际修改范围、未修改的保护边界和可回滚入口；
- 使用的 `modification_version`、操作类别和审批状态；如涉及 plan/指导，提供其文档链接而不是新增版本字段；
- 验证命令、关键结果、输出/manifest 入口，以及结果属于 `SUPPORTED`、`REFUTED`、`INCONCLUSIVE` 还是 `INVALID_IMPLEMENTATION`；
- 明确区分工程 smoke 证据与科研结论，不把实现可运行误报为效果成立；
- 只要产生或更新 plan、activity、实验产物、运行目录或 manifest，必须提供可点击的 Markdown 路径；运行状态同时给出 `run_id` 和 `run_status`；
- 上述本地链接的显示文本使用仓库根目录相对路径；仓库文档中的 target 按该 Markdown 文件位置使用
  相对链接，聊天回复中的 target 使用当前客户端支持的本地绝对文件路径，不把文档中的 `../` 原样
  复制到回复；回复中的本地路径或路径链接与前后中文、标点之间留半角空格，例如
  `活动记录： [docs/logs/activity_log.md](/workspace/Ref2Dex/docs/logs/activity_log.md) 。`，独立列表项的
  行首和行尾除外；
- 活动记录至少导航到本次输出目录和关键产物；交接前使用 `audit_diff.py --check-links` 校验最新活动
  条目的本地链接。运行中尚未生成的链接只允许在 `run_status` 为 `STARTED` / `RUNNING` 且同一行标记
  `PENDING` 时保留；终态不得留下失效链接；
- 本次是否遇到格式、目录、版本、日志或审批规则的阻碍。

如果现有格式或规范迫使 Agent 采取不自然、重复、易错或无法审计的做法，Agent 必须在完成汇报中单独列出“规范反馈”，说明：

1. 具体受限场景和当前 workaround；
2. 可能造成的风险或维护成本；
3. 建议新增/修改的规则、适用范围和兼容影响；
4. 是否需要用户确认后才能修改 `AGENTS.md`、Skill 或公共合同。

Agent 可以提出规范改进建议，但不得因为建议本身擅自改变用户指导、版本线、架构快照或公共合同。只有用户明确要求或确认后，才把建议写入 `AGENTS.md`、Skill 或对应版本文档；规则修改本身也要记录在最近作用域的 `activity_log.md`。

### 5.2 版本线与 AI 行为

仓库按 `Vn/Vn.m/Vn.m.k` 组织 `modification_version`：`Vn` 由用户确认创建，`Vn.m` 是工作/计划边界，
`Vn.m.k` 用于该边界下的具体实现、并行实验和状态变化，不单独创建 plan。指导文档只由用户
更新或在用户明确指令下更新；架构快照只有用户明确要求时才更新。每次实质改动都必须在相关
activity、experiment、decision 或 architecture 记录中写完整 `modification_version` 和操作类别。
指导/plan 的数字后缀只用于文档配对，不作为运行 JSON 的额外版本字段。

操作类别固定为 `governance`、`architecture`、`code`、`data`、`experiment`、`diagnostic`、
`operation`、`documentation`；无法归类时必须在回复中说明并补充类别。完整规则见
[`docs/modification_policy.md`](docs/modification_policy.md)。

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

子任务继承根 `AGENTS.md`、根项目总览和父 Task 的研究指导，不复制项目总览。只有当子任务形成独立研究/实现边界时才创建自己的 `docs/README.md`；其 `docs/logs/` 按事件按需创建 `activity_log.md`、`experiment_log.md`、`decision_log.md` 或局部架构记录，没有内容不建空文件。子任务的活动记录必须能回链父 Task 的 `activity_id`、实验或运行目录。

## 8. Ref2Dex 仓库特有约定

### 8.1 路径、运行和目录路由

共享运行时在 `src/base/`，数据预处理入口在 `process/`，Task 实现在 `src/task/<Task>/`，诊断实验在
`src/task/<Task>/research/<experiment>/`。具体目录、递归文档、运行产物、数据/cache、资产和各类
manifest 的唯一规范见 [`docs/目录规范.md`](docs/目录规范.md)；Agent 不需要用户为每次小改动指定目录，
应按该规范自动归类。

`outputs/`、研究 output、训练输出、cache、原始数据和 checkpoint 默认不提交；根 `output/` 只作为历史
兼容目录，不再新增内容。训练、评估、benchmark 和正式数据处理仍必须生成 `run_manifest.json`，并在
activity 中登记 `activity_id`、`run_id`、精确到秒的时间戳、`base_commit` 和可点击产物路径；字段和路径
细节以目录规范及适用 Skill 为准。
