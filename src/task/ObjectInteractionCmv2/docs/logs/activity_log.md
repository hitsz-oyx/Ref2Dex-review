# ObjectInteractionCmv2 活动记录

## 2026-09-15 03:50:09 UTC — 建立 V1.0 Task 文档与版本边界

- activity_id: `ACT-20260915-035009`
- timestamp: `2026-09-15 03:50:09 +0000`
- modification_version: `V1.0`
- type: `governance / architecture / documentation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户逐项确认独立 Task、V1.0 指针、rigid smoke 范围、articulation 后置及文档集合。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `85e70edffa85d8d1698a3e8adb22e111033cb892`
- worktree_dirty: `true`（存在本任务前的用户未提交改动）
- scope: 仅创建 ObjectInteractionCmv2 文档入口、V1.0 架构/计划草案、活动记录，并新增根版本指针。

**文件**

- [`指导/V1.0.md`](../指导/V1.0.md) — 用户提供的指导，仅作读取依据，未修改。
- [`README.md`](../README.md) — 新建 Task 文档导航。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 新增 `ObjectInteractionCmv2: V1.0`。
- [`docs/README.md`](../README.md) — 新建 Task 文档导航。
- [`docs/architecture/V1.0.md`](../architecture/V1.0.md) — 新建 V1.0 架构快照。
- [`docs/plan/V1.0.md`](../plan/V1.0.md) — 新建待协商的执行计划草案。

**原因**

按用户确认的 V1.0 边界，为结构化 interaction/effect 架构建立可审计入口；旧 ObjectInteractionCm 及其用户改动保持不变。

**验证**

- 已读取指导、仓库版本/目录/修改政策及 `research-change-control` Skill。
- 已检查新 Task 原先仅包含 `指导/V1.0.md`。
- 待计划定稿后执行文档链接审计；本条记录的链接目标需在交接前复核。

**结论**

本轮仅完成文档与治理边界，未运行训练、评估或数据处理；无科研效果结论。

## 2026-09-15 03:56:00 UTC — V1.0 计划定稿

- activity_id: `ACT-20260915-035600`
- timestamp: `2026-09-15 03:56:00 +0000`
- modification_version: `V1.0`
- type: `governance / architecture / documentation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认五项边界后要求“继续”；据此冻结 V1.0 的 synthetic-first smoke、K=16、hidden=128、residual 默认开启和同预算对照参数。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `85e70edffa85d8d1698a3e8adb22e111033cb892`
- worktree_dirty: `true`（存在本任务前的用户未提交改动）
- scope: 定稿 V1.0 plan 与 architecture 状态；不开始代码、数据、cache、checkpoint 或训练。

**文件**

- [`指导/V1.0.md`](../指导/V1.0.md) — 用户指导，仅作依据，未修改。
- [`README.md`](../README.md) — Task 文档导航，保持入口。
- [`architecture/V1.0.md`](../architecture/V1.0.md) — 将架构状态标为 `frozen snapshot`。
- [`plan/V1.0.md`](../plan/V1.0.md) — 冻结 V1.0 默认参数并标为 `final`。
- [`logs/activity_log.md`](activity_log.md) — 登记本次定稿活动。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 保持 `ObjectInteractionCmv2: V1.0` 指针；该文件中的旧 Task 变更属于用户既有改动。

**原因**

为后续实现提供已确认的 V1.0 执行闸门，避免在代码阶段隐式改变 token、数据或验证边界。

**验证**

- 计划与架构文件存在，且 V1.0 配对、状态和默认参数一致。
- 待本次文档 diff 完成后运行 `audit_diff.py --worktree --scope-prefix ... --check-links`。

**结论**

计划定稿只证明治理与工程边界已明确；尚无训练、评估或科研效果证据。

## 2026-09-15 04:50:22 UTC — V1.0.1 rigid synthetic contract 实现

- activity_id: `ACT-20260915-045022`
- timestamp: `2026-09-15 04:50:22 +0000`
- modification_version: `V1.0.1`
- type: `code / diagnostic`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: V1.0 final plan 已定稿，用户连续要求继续；实现严格限定在 Task-local rigid/synthetic smoke 范围。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `85e70edffa85d8d1698a3e8adb22e111033cb892`
- worktree_dirty: `true`（存在本任务前的用户未提交改动）
- scope: 新增结构化 rigid effect 模型、局部 2 cm interaction、grounded tokens、FK flow、synthetic batch 与 Task-local tests；未接入真实 cache/ARCTIC/旧 checkpoint/正式训练。

**文件**

- [`__init__.py`](../../__init__.py) — 暴露 V1.0 模型与 loss。
- [`model.py`](../../model.py) — 实现 geometry/local interaction、K=16 grounded tokens、root SE(3)、可选 residual 与 loss。
- [`synthetic.py`](../../synthetic.py) — 提供 seed=42 的 synthetic rigid contract。
- [`tests/test_v1_0_smoke.py`](../../tests/test_v1_0_smoke.py) — 覆盖 FK、forward/backward 与空接触 mask。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 将 Task 指针推进为 `V1.0.1`。
- [`README.md`](../README.md) — 保持 Task 文档入口。
- [`architecture/V1.0.md`](../architecture/V1.0.md) — 作为实现合同依据，未改变研究边界。
- [`plan/V1.0.md`](../plan/V1.0.md) — final 计划，未改变执行范围。
- [`指导/V1.0.md`](../指导/V1.0.md) — 用户指导，仅作依据，未修改。
- [`logs/activity_log.md`](activity_log.md) — 登记实现与验证。

**原因**

先以独立 synthetic rigid contract 验证结构化 effect/FK 路径，降低接入真实 cache 前的工程风险。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest src/task/ObjectInteractionCmv2/tests -q`：`3 passed`。
- 结果属于 `SUPPORTED` 的工程 smoke 证据；不构成真实数据效果、articulation 或跨手型迁移结论。

**回滚**

删除本条新增的 Task-local Python/test 文件并将 `ObjectInteractionCmv2` 指针恢复为 `V1.0`；旧 Task 和用户既有改动不触碰。

## 2026-09-15 07:29:42 +0000 — OakInk2 小规模 rigid pilot 完成

- activity_id: `ACT-20260915-072942`
- timestamp: `2026-09-15 07:29:42 +0000`
- modification_version: `V1.0.2`
- type: `data / experiment / operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确要求补充闭环并使用一部分 OakInk2 试跑，不进行全量训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `85e70edffa85d8d1698a3e8adb22e111033cb892`
- worktree_dirty: `true`（存在本任务前的用户未提交改动）
- run_id: `oakink2_pilot_20260915T0505Z`
- run_status: `COMPLETED`
- scope: 只读 OakInk2 Stage3 单物体 NPZ；2 个文件、每条最多 4 个相邻帧训练 2 steps；1 个文件、2 帧评估；未生成全量 cache、未修改 NAS 输入。

**文件与产物**

- `src/task/ObjectInteractionCmv2/` — 本次 Task-local 实现与文档范围内的全部文件。
- [`model.py`](../../model.py)、[`oakink2.py`](../../oakink2.py)、[`train.py`](../../train.py)、[`eval.py`](../../eval.py) — V1.0.2 闭环入口与只读 adapter。
- [`outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/`](../../../../../outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/) — 运行目录。
- [`outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/run_manifest.json`](../../../../../outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/run_manifest.json) — 运行合同与输入入口。
- [`outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/metrics.jsonl`](../../../../../outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/metrics.jsonl) — 2 steps 训练曲线。
- [`outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/latest.pt`](../../../../../outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/latest.pt) — 最近 checkpoint。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — Task 指针更新为 `V1.0.2`。

**命令与结果**

```text
/home/wbcd/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCmv2.train --data-root /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/oakink2_object_centered_v1 --max-files 2 --max-frames 4 --epochs 1 --max-steps 2 --output outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z
/home/wbcd/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCmv2.eval --checkpoint outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/latest.pt --data-root /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/oakink2_object_centered_v1 --max-files 1 --max-frames 2
```

训练 loss 从 `0.08002` 降至 `0.07584`；评估 `total=0.07250`、`flow=0.00158`、`effect=0.70910`、`residual=0.00230`，均为 finite。工程闭环结论：`SUPPORTED`；科研效果结论：`INCONCLUSIVE`，样本和步数不足以支持泛化或收益判断。

**原因**

按用户要求用少量 OakInk2 数据验证 Cmv2 的读取、训练、checkpoint 和评估闭环，不启动全量训练。

**验证**

- 训练与评估命令均成功退出，manifest、metrics 和 checkpoint 均已生成。
- Task-local pytest：`3 passed`。
- 结果属于工程 smoke/pilot 证据，不构成科研效果结论。

**验证与回滚**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest src/task/ObjectInteractionCmv2/tests -q`：`3 passed`。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python .agents/skills/research-change-control/scripts/audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCmv2 --scope-prefix docs/current_versions.yaml --check-links`：通过。
- 回滚：删除独立运行目录和新增闭环文件，将版本指针恢复为 `V1.0.1`；NAS 原始数据、旧 Task 和既有输出不动。
