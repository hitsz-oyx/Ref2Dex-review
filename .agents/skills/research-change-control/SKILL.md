---
name: research-change-control
description: 控制 Ref2Dex 的代码、配置、测试、文档和治理修改，保持最小差异、明确审批和可审计交接。
metadata:
  short-description: 让仓库修改保持简洁且可回滚
---

# 科研修改控制

本 Skill 用于 change 和 governance 模式。它只负责“怎么修改”，不负责目录归属、
研究结论或长时运行状态；分别使用 `directory-and-artifacts`、
`research-experiment-workflow` 和 `long-running-tasks`。

## 最小上下文

先声明 Task、工作模式、范围和保护边界，再读取 AGENTS、current_versions、当前 Task README、当前指导谱系、相关代码和测试。涉及路径、产物、manifest 或测试归属时，使用 `directory-and-artifacts`。目录合同不能决定位置时，停止该路径上的写操作并请用户选择；发现已发生的目录例外时，优先报告并询问是否提升为目录合同，不把事后报告当作追认。同一数值基线的指导必须读取无后缀文件及按 `a`–`z` 排序的全部已确认补充文件，不能只读取最后一个字母文件。

跨文件的明确冲突由较后字母覆盖冲突部分；同一文件内部冲突、无法同时满足或语义不充分时，停止相关实施并请用户确认。确认前在 plan 草案列出冲突原文、候选解释、拟定措辞、保护项、风险、验证和回滚；确认后才可修改指导和定稿计划。只有历史兼容、冲突或证据追溯时，才搜索 Activity、experiment、decision 和 Git 历史；不要为了形式完整读取旧日志。开始前检查 git status、git diff 和 git diff --cached，保留用户已有改动。

## 等级、计划与版本

按 [change-levels.md](references/change-levels.md) 判断 L0–L3。L0/L1 可执行并在交接中报告。L2/L3 编辑前必须说明改变什么、保持什么不变、风险、验证和回滚并获得用户确认。修改 AGENTS、Skill 和公共治理合同固定为 L3；记录真实 approval 与 approval_basis，不把沉默视为批准。

`work_version` 是唯一版本术语；git_commit 和 run_id 是独立身份。新记录和新 manifest 只写 work_version，旧的 modification_version/research_version/governance_version 只作为历史读取兼容，不能回写为新规范。

只有新 work_version 边界或 L2/L3 方案需要最终 plan。指导补充不自动建立 plan，也不要求一一对应：已有 `a`、`b`、`c` 指导后才协商的计划仍为 `plan/V<n>.md`。状态不是 `FINAL` 的 plan 可原地修订；一旦 `FINAL`，后续任何改变实施合同的修改都不得静默覆盖，通常必须建立 `plan/V<n><letter>.md`。由同基线指导补充触发时使用该指导字母；没有对应指导时使用 plan 谱系下一个未使用字母。

已执行的字母 plan 若在同一分支、同一研究问题和保护边界内进行有界纠偏或探索重试，且用户明确选择不新建分支，可建立 `plan/V<n><letter>.<k>.md` 点数字微补充。`k` 从 1 开始、无前导零、在同一父 plan 下连续递增；它不推进 work_version、不用于指导命名，也不替代独立问题所需的下一字母 plan。按基线、字母顺序以及每个字母后的整数 `.1`、`.2` 顺序读取；后项只覆盖明确冲突。微补充须写明复用分支理由、改变与保持项、风险、验证、回滚和真实审批，且不降低 L2/L3 或 run 门禁。开始实施前读取基线 plan 及全部适用补充。

## 分支、记录和验证

`oyx` 是稳定集成分支；独立任务使用 `ai/<task>/<description>`。只编辑当前任务的显式文件，不整文件格式化或顺手重构。不得 reset、覆盖或带入用户已有修改；outputs、cache、checkpoint、原始数据和大型日志不纳入提交。

开始分支工作前检查现有 worktree 和 Git 状态。没有真实并行需求时复用一个干净的 AI worktree，不为每个顺序
分支新增 worktree；只有并行、既有 worktree 有待保护修改，或用户要求长期隔离时才新增，并记录原因。已合并
worktree 仅在干净、无未跟踪内容且用户明确授权后可移除；未知、用户拥有、含修改或临时 worktree 不得触碰。
移除 worktree 不隐含删除分支，分支删除须单独确认。

长期工作单元写入最近作用域 `docs/activities/` 的独立 Activity，索引为 `activities/README.md`。Activity 至少包含精确时间、activity_id 或 work_version、work_version、git/base commit、branch、scope、等级、审批、原因、文件、验证和回滚入口。普通命令、逐步进度和一次性 smoke 不写 Activity。科学假设、证据和结论写入 `docs/experiments/` 的 experiment card。`logs/activity_log.md` 和 `logs/experiment_log.md` 仅为历史审计，不能作为新记录目标或当前状态入口。

修改后执行：

    python tools/verify.py --changed

第一阶段工具尚不存在时，记录实际定向检查和 git diff --check，不得声称 VERIFY PASS。
`verify.py` 只检查适合机器判断的语法、结构、链接、版本合同和 hermetic 测试；Activity
字段、提交范围和变更理由仍须由人审阅，不以逐路径脚本冒充完整审计。

## 交接

最终回复按 Task / mode / impact level / work_version / branch / git_commit / research semantics / changed / protected / verification / scientific conclusion / next 交接。工程 smoke 只能说明接线或可运行性，不能作为科研效果。
