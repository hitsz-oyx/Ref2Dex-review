---
name: research-change-control
description: 控制 Ref2Dex 的代码、配置、测试、文档和治理修改，保持最小差异、明确审批和可审计交接。
metadata:
  short-description: 让仓库修改保持简洁且可回滚
---

# 科研修改控制

本 Skill 用于 change 和 governance 模式。它只负责“怎么修改”，不负责研究结论或长时运行状态；运行请使用
research-experiment-workflow。

## 1. 最小上下文

先声明 Task、工作模式、范围和保护边界，再读取：

    AGENTS.md
    docs/current_versions.yaml
    src/task/<Task>/docs/README.md
    当前指导谱系（若任务涉及研究目标）
    相关代码和相关测试

同一数值基线的指导必须读取无后缀文件及按 `a`–`z` 排序的全部已确认补充文件，不能只读取最后一个字母文件。
跨文件的明确冲突由较后字母覆盖冲突部分；同一文件内部冲突、无法同时满足或语义不充分时，停止相关实施并请用户确认。
确认前起草或修订 plan，列出冲突原文、候选解释、拟定措辞、保护项、风险、验证和回滚；确认后才可修改指导和定稿计划。
只有出现历史兼容、冲突或证据追溯时，才搜索 activity archive、experiment、decision 和 Git 历史；不要为了形式完整
读取所有历史日志。开始前检查 git status、git diff 和 git diff --cached，保留用户已有改动。

## 2. 等级和审批

按 [change-levels.md](references/change-levels.md) 判断 L0–L3：

- L0：文档、格式、测试、只读诊断和不改变运行变量的元数据；
- L1：Task 内实现，observation/action/reward/GT/坐标/单位/split/metric/checkpoint 合同不变；
- L2：研究语义或公共接口合同变化；
- L3：AGENTS、Skill、治理、共享 src/base、依赖、迁移、破坏性操作或长任务。

L0/L1 可执行并在交接中报告。L2/L3 编辑前必须说明改变什么、保持什么不变、风险、验证和回滚并获得用户确认。
修改 AGENTS、Skill 和公共治理合同固定为 L3。记录真实的 approval 与 approval_basis，不把沉默视为批准。

## 3. plan 和研究版本

只有新 research_version 或 L2/L3 方案需要最终 plan/V<n>.md。L0/L1 修复不创建新的 plan，也不递增 research version。
指导由用户维护；只有用户确认同文件冲突的修订措辞后，Agent 才可修改相应指导。治理任务可以把用户明确的治理指导作为范围依据，不把 plan 草稿冒充实现事实。

如果实现中发现需要改变最终范围、研究变量、公共合同或保护边界，立即暂停并重新确认；不得用代码或配置差异代替计划确认。

## 4. 分支和实施

oyx 是稳定集成分支。独立任务使用 ai/<task>/<description> 分支；实验参数变化不创建分支。按最小语义差异实施，
只编辑当前任务的显式文件，不整文件格式化，不顺手重构无关代码。不得 reset、覆盖或带入用户已有修改；outputs、
cache、checkpoint、原始数据和大型日志不纳入提交。

## 5. 验证和记录

按 docs/目录规范.md 选择最窄且有意义的测试：Task 内改动优先 Task 测试，公共接口才扩大到 shared/integration，
治理修改做文档、YAML/JSON、脚本语法和链接检查。

第二阶段建立后，唯一机器门禁为：

    python tools/verify.py --changed

第一阶段工具尚不存在时，必须记录实际执行的定向检查和 git diff --check，不得声称 VERIFY PASS。交接前可用：

    python .agents/skills/research-change-control/scripts/audit_diff.py \
      --log <activity_log> --staged --check-links

长期 Activity 存在当前作用域 `activities/`，索引为 `activities/README.md`；旧 `activity_log.md` 仅作历史审计。
Activity 至少包含精确时间、activity_id 或 work_version、governance_version 或 research_version、git_commit、branch、scope、等级、
审批、原因、文件、验证和回滚入口。重要事件才写 activity；普通逐步进度和一次性 smoke 不写科研时间线。

## 6. 交接

最终回复按以下字段交接：

    Task / mode / impact level
    branch / git_commit
    research semantics: CHANGED 或 UNCHANGED
    changed / protected
    verification
    scientific conclusion: SUPPORTED / REFUTED / INCONCLUSIVE / INVALID_IMPLEMENTATION / N/A
    next

工程 smoke 只能说明接线或可运行性，不能作为科研效果。产生或更新 plan、activity、实验产物、运行目录或 manifest
时，回复必须给出可点击路径；路径显示使用仓库根相对路径，聊天目标使用客户端支持的本地绝对路径。

完整能力边界见 [audit-contract.md](references/audit-contract.md)。历史记录只读兼容，不批量回填。
