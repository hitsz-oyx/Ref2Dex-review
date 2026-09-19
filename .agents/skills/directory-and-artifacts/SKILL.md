---
name: directory-and-artifacts
description: 规定 Ref2Dex 代码、数据、cache、资产、运行产物、manifest 与测试的目录职责；处理路径或产物时使用。
metadata:
  short-description: 让路径与产物职责保持可追溯
---

# 目录与产物合同

处理路径、数据、cache、资产、运行产物、manifest 或测试归属时使用本 Skill。它只决定
“放在哪里、什么可提交、由谁负责”；变更审批使用 `research-change-control`，实验方法和
科学结论使用 `research-experiment-workflow`，长时运行操作使用 `long-running-tasks`。

## 产物归类

| 内容 | 规范位置 | 是否提交 |
| --- | --- | --- |
| 共享运行时、Contract、Runner、Manifest | `src/base/` | 是 |
| Task 模型、Dataset、Task 工具 | `src/task/<Task>/` | 是 |
| 独立研究实验 | `src/task/<Task>/research/<experiment>/` | 定义文件是，产物否 |
| 原始数据 | `data/raw_data/` 或外部 `dataset/<name>/data/` | 否 |
| 派生 cache、split、统计 | `data/processed_data/` | 否 |
| 预训练模型、MANO、SMPL-X、URDF | `src/task/<Task>/assets/` | 外部文件忽略；说明可提交 |
| 训练、评估、正式 benchmark | `outputs/<task>/<run_id>/` | 否 |
| 研究诊断、可视化、小型分析 | `research/<experiment>/output/<run_id>/` | 否 |
| 长期工作与科学结论 | 最近作用域 `docs/activities/`、`docs/experiments/` | 是 |

根 `output/` 仅为历史兼容，不新增；不创建 `results/` 或 `result/`。实际数据和 cache
不得复制到 Task 目录。

## 未决位置与目录例外

只有当前目录合同无法决定新内容的位置时，才触发本节；表中已有的合规位置不重复询问。
在未决时，停止该路径上的创建、移动和写入，向用户确认位置，并给出候选位置、每个位置的
提交/追溯影响和需要保存的内容，再等待用户选择。不得为“以后可能会用”而预创建目录、临时文件、
cache 或产物。

若发现内容已被先行创建、移动或写入到未确认位置，发现后的下一次回复必须优先报告：

- 精确路径、内容类型与规模；
- 发生原因、与本合同的偏离及可能影响；
- 当前 Git 跟踪/忽略状态；
- 保留、迁移或删除的可选处理方式。

报告不等于用户批准，也不能继续扩大该例外。报告时必须询问用户是否应把该例外提升为本
Skill 的通用规则；只有用户明确确认后，才通过已批准的治理变更修改本合同。个案位置决定
本身不自动成为长期规范。

## 研究与文档结构

```text
src/task/<Task>/research/<experiment>/
├── README.md
├── experiment.yaml
├── run.py 或其他实验脚本
└── output/<run_id>/
    ├── run_manifest.json
    └── 生成的图、表、NPZ、日志

src/task/<Task>/docs/
├── README.md
├── 指导/V<n>.md
├── plan/V<n>.md
├── activities/README.md
├── experiments/README.md
└── architecture/V<n>.md
```

research 根目录不放散脚本、结果或 checkpoint；实验目录使用稳定 lower_snake_case。Task
目录只保留代码、配置、研究文档和索引。指导谱系和 plan 补充的读取、命名与冲突规则以
根 `AGENTS.md` 为准。

## 数据、运行与 manifest

- `src/task/<Task>/dataset/` 放运行时读取器、schema 与 Dataset；
  `src/task/<Task>/tools/data/` 放 Task cache、split、校准与统计；根 `tools/` 放跨 Task 工具。
- 训练、评估和带 checkpoint 的正式运行使用 `outputs/<task>/<run_id>/`；诊断、可视化和
  局部 benchmark 使用 `research/<experiment>/output/<run_id>/`；可复用 cache 使用
  `data/processed_data/<producer>/<schema>/<version>/`。
- `experiment.yaml` 记录实验目的、入口、适用指导/plan 与默认产物位置；cache 根的
  `manifest.json` 记录数据来源、schema、split、坐标与统计；`asset_manifest.json` 记录外部
  资产来源、版本和校验。
- 每个运行目录的 `config.json` 是解析后的配置快照；`run_manifest.json` 是 task、
  work_version、git_commit、run_id、输入、seed、checkpoint 与输出入口的轻量运行身份；
  `metrics.jsonl` 是机器读取的逐 step/epoch 曲线。完整 metadata 通过引用保留，不复制进
  run manifest。
- 运行终态、最佳 checkpoint 与失败/停止原因写独立 Activity；experiment card 记录
  hypothesis、protocol、runs、evidence、conclusion 和 limitations。两者不使用旧日志替代。

## 测试归属

现有根 `tests/` 是 legacy；新测试按所有权放置：

| 变化范围 | 默认验证范围 |
| --- | --- |
| 单一 Task 代码、配置或局部工具 | `src/task/<Task>/tests/`；公共接口再加 `tests/shared/` |
| `src/base/`、公共配置、Runner 或 Manifest | `tests/shared/` + 受影响 Task 的最小 smoke |
| 跨 Task、checkpoint 或数据格式 | `tests/integration/` + 所有受影响 Task |
| `process/` adapter 或 Stage | `process/tests/` + 消费方 Task 定向测试 |
| 治理文档、Skill、目录和链接合同 | `tests/governance/` + Activity/Skill 链接审计 |

物理迁移完成前，pytest.ini 与根 tests/conftest.py 保持不变；全量回归可运行根 `pytest`。
统一机器门禁为 `python tools/verify.py --changed`，其测试选择只针对当前分支实际变更。
