---
name: research-change-control
description: 在科研或软件仓库中进行代码、配置、数据处理和文档修改时，使用透明的影响等级、最小差异、审批闸门和文档与差异一致性检查。
metadata:
  short-description: 让科研修改保持简洁且可审计
---

# 科研修改控制

当你要修改仓库代码、配置、数据处理逻辑、实验流程或项目文档时使用本 Skill。

## 核心行为

- 先确认目标，保留用户已有的未提交改动。
- 优先采用最小语义差异：不要顺手重构无关代码、整文件格式化或引入无关抽象。
- 在项目的 modification log 中声明影响等级。这是供用户复核的 AI 判断，不能假装脚本能够推断科研影响。
- 如果改动会影响科研语义、公共合同、共享基础设施、破坏性状态或长时外部任务，在实现前说明方案并请用户确认。
- 在同一次交接中同时给出实现、验证和文档；不能把 smoke test 或实现错误冒充科研结论。

## 指导与执行计划

当仓库提供成对的 `docs/指导/V<n>.md` 与 `docs/plan/V<n>.md` 时，遵循以下流程；路径中的
`Task`、版本号和领域名称不属于本 Skill 的固定假设：

1. 用户提供或修订 `指导/V<n>.md`，说明研究目标、假设、边界和禁止事项。
2. Agent 先阅读指导以及相关状态、架构、决策和修改记录，起草同版本的 `plan/V<n>.md`。
3. Agent 与用户讨论计划中的实现范围、文件、接口/schema、不变量、修改等级、验证和回滚方式；
   协商期间可以反复修改 plan。
4. 只有用户和 Agent 明确将 plan 标记为最终版本后，才开始代码、配置、数据处理或实验流程修改。
   plan 未定稿时，Agent 只能分析、提问和修改 plan 本身。
5. 若实现中需要偏离最终 plan，先暂停并重新协商；不得用代码修改替代计划确认。

指导与 plan 的版本号必须一致。协商期间的草稿变化不追加到 `modification_log.md`；完成实现时，
只在最终修改记录中引用定稿 plan 版本，并记录实际 diff 和验证结果。plan 不替代 architecture、
experiment、decision 或 modification log。

## 等级与审批

分级时阅读 [change-levels.md](references/change-levels.md)。默认规则如下：

- **L0**：语义不变的配置、文档、测试或诊断，可自动完成。
- **L1**：保持公共数据/指标/checkpoint 合同的任务内实现，可自动完成并做定向验证。
- **L2**：修改数据字段、坐标系、GT、数据划分、缓存、指标、checkpoint 解释或公共接口合同；编辑前请用户确认。
- **L3**：修改共享框架行为、仓库治理、依赖、迁移、破坏性操作或长时任务；编辑前请用户确认，并先给出范围、风险、回滚和验证计划。

拿不准时选择更高等级。按真实对话记录 `approval: auto`、`pending` 或 `user-approved`，不要把用户沉默当作批准。

## 修改记录

使用离改动最近的 modification log。项目没有既定格式时，使用以下简洁条目：

```markdown
## YYYY-MM-DD — <摘要>

- change_level: L0 / L1 / L2 / L3
- approval: auto / pending / user-approved
- skills_used: 本次实际采用的仓库 Skill 名称列表
- branch:
- post-commit:
- scope:

**文件**
- `path/to/file` — 做了什么

**原因**
...

**验证**
...
```

交接前运行项目审计，或运行随附脚本：

```bash
python .agents/skills/research-change-control/scripts/audit_diff.py \
  --log <project-modification-log> --staged
```

脚本只检查最新记录是否客观列出了所选 Git 差异中的文件，并包含等级、审批、范围、原因和验证段；它不会评判等级是否正确。能力边界见 [audit-contract.md](references/audit-contract.md)。

`skills_used` 用于让用户复核 Agent 是否加载了与变更类型匹配的 Skill。它是透明记录，不是
脚本能够证明的模型内部状态；缺少该字段时不得声称流程已完整执行。

## 验证与交接

- 先运行最窄且有意义的测试；共享代码或合同变更再扩大验证范围。
- 在修改记录中写出确切的验证命令和结果。
- 提交前检查 `git status`、`git diff` 和 staged diff。显式 stage 目标路径，绝不带入无关的用户改动。
- 除非用户明确要求，不提交生成数据、缓存、checkpoint 和输出。
