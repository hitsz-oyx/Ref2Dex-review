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

## 2026-09-15 08:21:48 +0000 — V1.0.2 新分支提交并推送

- activity_id: `ACT-20260915-082148`
- timestamp: `2026-09-15 08:21:48 +0000`
- modification_version: `V1.0.2`
- type: `operation / governance`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确要求新建分支、提交当前工作并使用指定 HTTP/HTTPS 代理推送，随后连续要求继续。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `710d2ca1426013bd5d928e03595451d265804bdb`
- worktree_dirty: `true`（保留本任务之外的 `ObjectInteractionCm` / `Cm` 用户改动）
- run_id: `git-push-objectinteractioncmv2-v1.0.2-20260915T082148Z`
- run_status: `COMPLETED`
- scope: 只提交 `ObjectInteractionCmv2` Task 和 `docs/current_versions.yaml` 中该 Task 的两行指针，并推送到新的 origin 分支；不提交、覆盖或回滚其他工作区改动。

**提交与远端**

- implementation_commit: `710d2ca1426013bd5d928e03595451d265804bdb`（`实现 ObjectInteractionCmv2 小规模训练闭环`）。
- remote_ref: `origin/feature/objectinteractioncmv2-v1.0.2`。
- [`ObjectInteractionCmv2 Task`](../../) — 本次实现、指导、架构、final plan、测试与活动记录。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 本提交只包含 `ObjectInteractionCmv2: V1.0.2`；`ObjectInteractionCm: V1.4.2` 仍作为未暂存用户改动保留。
- [`run_manifest.json`](../../../../../outputs/objectinteractioncmv2/oakink2_pilot_20260915T0505Z/run_manifest.json) — 已完成 OakInk2 pilot 的运行合同；运行产物被 Git 忽略，未纳入提交。

**原因**

按用户要求为已完成的 V1.0.2 工作建立独立、可审阅且不夹带其他 Task 改动的远端分支。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest src/task/ObjectInteractionCmv2/tests -q`：`3 passed`。
- `audit_diff.py --staged --check-links`：首个提交前通过，activity 与 12 个受审计变更路径一致，9 个本地链接有效。
- `git push -u origin feature/objectinteractioncmv2-v1.0.2`：在用户指定的四个代理环境变量下成功，新建远端分支并设置 upstream。
- `git ls-remote --heads origin refs/heads/feature/objectinteractioncmv2-v1.0.2`：远端首次推送 SHA 与 `710d2ca1426013bd5d928e03595451d265804bdb` 一致。

**保护边界、结论与回滚**

- 未提交 `ObjectInteractionCm`、`Cm`、旧输出、OakInk2 原始数据或被忽略的 checkpoint/metrics；未执行 reset、revert、删除或 force push。
- Git 分支、提交与远端可达性属于工程操作 `SUPPORTED`；OakInk2 pilot 的科研效果仍为 `INCONCLUSIVE`。
- 回滚入口为父提交 `85e70edffa85d8d1698a3e8adb22e111033cb892`；若需删除远端分支或回退提交，必须另行确认，不能影响当前保留的用户改动。

## 2026-09-15 09:19:29 +0000 — GRAB 与配对 DExplore 训练前只读核对

- activity_id: `ACT-20260915-091929-CMV2-GRAB-DEXPLORE-PREFLIGHT`
- timestamp: `2026-09-15 09:19:29 +0000`
- modification_version: `V1.0.2`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问先用 GRAB 与配对 DExplore 数据训练、保持之前配置并比较性能是否存在歧义；本条只读核对现有数据、配置、历史指标和运行能力。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（存在本 Task 之外的 ObjectInteractionCm/Cm 用户改动和进行中的 OakInk2 index 运行）
- scope: 只读识别可比 baseline、数据入口、训练预算、指标和当前 Cmv2 runner 缺口；不修改 plan、代码、配置、split、cache、checkpoint 或实验结论，不启动训练。

**文件与证据**

- [`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml`](../../../ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml) — 历史可比配置：GRAB/MANO 与 DExplore RL-Inspire 源概率 `0.5/0.5`、seed 42、batch 32、KNN=32、2 cm、train stride 1..10、eval stride 2、202300 steps、以 `val/obj/flow_epe_mm` 选 best。
- [`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`](../../../ObjectInteractionCm/docs/logs/experiment_log.md) — V1.3 历史 best equal-source object EPE `6.516671 mm`（epoch 180 / step 132480），GRAB `7.290740 mm`、DExplore RL-Inspire `5.742603 mm`。
- [`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_v1_3_cache.py`](../../../ObjectInteractionCm/tools/data/build_dexplore_rl_v1_3_cache.py) — V1.3 cache 依赖旧 V1.2.5 index/cache，并仍包含旧机器 `/home2/wyy/...` URDF 默认入口。
- [`src/task/CmDecoderv2/tools/data/prepare_mano_actual_finetune_view.py`](../../../CmDecoderv2/tools/data/prepare_mano_actual_finetune_view.py) — 历史“paired view”实际是 MANO source 与 DExplore actual-Inspire target 的 decoder 微调数据合同，不等同于 V1.3 OI-Cm 的 source-balanced mixed index。
- [`src/task/ObjectInteractionCmv2/train.py`](../../train.py) 与 [`src/task/ObjectInteractionCmv2/eval.py`](../../eval.py) — 当前仅支持 batch=2 的 synthetic/OakInk2 小型入口；尚无 YAML config、source-balanced split/stride、validation/best checkpoint、object EPE mm、DDP 或 offline-KNN cache adapter。
- [`src/task/ObjectInteractionCmv2/docs/plan/V1.0.md`](../plan/V1.0.md) — final plan 允许后续 V1.0.k 真实 rigid cache 与同预算对照，但改变数据/运行语义和启动正式训练仍需明确本次范围。

**发现与待确认项**

- 历史路径 `data/processed_data/object_interaction_cm_dexplore_rl_v1_3/`、其 V1.2.5 source cache、paired decoder view 以及 V1.3 训练输出/checkpoint 当前均不存在；NAS 未找到迁移副本。当前本机 DExplore checkout 只有代码与 Inspire 资产，没有实际 RL trajectory 数据，旧 `/home2` 入口也不存在。
- V1.3 mixed index 不是同一 object state/effect 的逐样本配对实验；真正的 paired decoder view 历史计数为 train 254、val 30、test 0，且当前数据目录同样缺失。需要用户明确“配对好的 DExplore”具体指哪一份现存数据路径。
- “配置一致”可保持共同数据/优化合同，但 Cmv2 没有旧模型的 hand decoder/loss；公平主比较应限定为同 split、seed、source probability、stride、batch/step 预算和 `val/obj/flow_epe_mm`，不能直接比较两种模型的 total loss。
- 若采用旧全预算，历史三卡运行量级约 202300 steps、数小时；建议先完成 adapter 与 2-step/小验证集 smoke，再从随机初始化执行完整同预算训练。当前另有 `oakink2_active_tool_segments_v1_1_20260915_090003` CPU 数据任务运行中，不应停止或覆盖。

**原因**

避免把 source-balanced mixed 数据误称为逐样本配对，或在 cache/checkpoint 已缺失、Cmv2 runner 尚不具备公平评估合同的情况下直接启动不可复现训练。

**验证**

- 只读核对 V1.0 final plan、Cmv2 train/eval 实现、V1.3 config/plan/activity/experiment、paired-view producer 和当前文件系统路径。
- 检查 NAS `processed_data`、当前 DExplore checkout、旧 `/home2` 路径、GPU/进程状态；未找到历史 DExplore cache 或 paired view，未产生运行产物。
- 结论：训练前歧义与缺失输入证据为 `SUPPORTED`；新模型相对旧 V1.3 的性能仍为 `INCONCLUSIVE`。

**回滚**

删除本条 activity 增量即可；本轮没有代码、配置、数据或运行产物需要恢复。

## 2026-09-15 09:35:58 +0000 — DExplore RL checkpoint 重导出前只读核对

- activity_id: `ACT-20260915-093558-CMV2-DEXPLORE-RL-EXPORT-PREFLIGHT`
- timestamp: `2026-09-15 09:35:58 +0000`
- modification_version: `V1.0.2`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户提出使用 DExplore 的 RL checkpoint 重新导出并询问是否有歧义；本条仅核对 checkpoint、原始输入、运行环境和旧数据口径。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留本 Task 之外的 ObjectInteractionCm/Cm 用户改动和进行中的 OakInk2 index 运行）
- scope: 只读识别 canonical RL checkpoint、重导出依赖、配对样本口径、设备与可复现性边界；不修改 DExplore/Ref2Dex 代码、plan、配置、资产或数据，不启动 pilot 或全量导出。

**文件与证据**

- DExplore 仓库 `/home/wbcd/workspace/oyx_ws/dexplore` 的官方 GRAB Inspire 导出说明与 `data_processing/export_rl_batches.py` 均以 teacher checkpoint `checkpoint/inspire.pth` 启动 RL rollout；该文件 SHA256 为 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`，不同于 `inspire_distill.pth`。
- RL player 的 export 路径使用 deterministic action，将实际模拟 Inspire DOF 写入列 `373:391`、实际 object state 写入列 `198:205`，其余 reference/contact/graph 字段沿用输入 canonical tensor。
- [`src/task/ObjectInteractionCmv2/docs/plan/V1.0.md`](../plan/V1.0.md) 允许后续真实 rigid cache 对照；本次数据语义与长任务仍需另行确认。
- [`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml`](../../../ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml) 与 [`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`](../../../ObjectInteractionCm/docs/logs/experiment_log.md) 是旧 V1.3 配置和结果口径入口。

**发现与待确认项**

- checkpoint 选择可明确为 teacher `inspire.pth`；DExplore `dexplore-4090` 环境在补充其 conda `lib` 到 `LD_LIBRARY_PATH` 后可导入 Isaac Gym、PyTorch CUDA 和 `rl_games`，GPU 1–3 可用于后续运行，GPU 0 上存在外部进程，不应占用。
- 原始 GRAB 1335 个序列与 subject/object 工具资产存在，但 canonical full-body tensor 尚未生成；DExplore 所需 object mesh 资产目录也缺失，可以从 GRAB 重建。
- 当前机器和 NAS 均未找到 licensed SMPL-X model 文件；canonical 转换必须获得用户提供的 `SMPLX_NEUTRAL.npz` 所在模型目录，这是启动导出的实际阻塞项。
- 旧 DExplore-compatible 口径会排除任何左手接触和 doorknob，得到 660 对；若导出全部 1335 条，则不再与旧 V1.3 数据口径等价。旧 cache/index 已缺失，无法保证逐条复原原 split，只能在确认后以 parent/sequence、seed 42 重建无 pair leakage 的确定性 split。
- DExplore export 入口没有显式传递 seed，若执行应由 Task-local wrapper 固定 `--seed 42`，并在新目录写 checkpoint hash、输入清单、过滤规则和 manifest；不得覆盖旧路径或修改 DExplore 仓库。

**原因**

重导出会重建 canonical/asset 数据并启动 GPU 长任务，且过滤范围、split 和 checkpoint 解释会直接影响科研可比性；在缺少 SMPL-X 模型目录和用户确认前不能安全启动。

**验证**

- 只读检查 DExplore checkpoint payload/hash、GRAB export/conversion/filter 实现、原始数据与资产目录、conda 环境、GPU/进程状态和旧 Ref2Dex V1.3 配置。
- 没有创建 run_id、运行目录、cache、asset 或 manifest；没有启动导出。
- 结论：teacher checkpoint 与导出链路识别为 `SUPPORTED`；性能差异仍为 `INCONCLUSIVE`。

**回滚**

删除本条 activity 增量即可；本轮没有代码、配置、数据或运行产物需要恢复。

## 2026-09-15 09:40:34 +0000 — 本机 SMPL-X 模型定位与加载验证

- activity_id: `ACT-20260915-094034-CMV2-SMPLX-LOCATE`
- timestamp: `2026-09-15 09:40:34 +0000`
- modification_version: `V1.0.2`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求先自行寻找本机已有 SMPL-X；本条只读搜索并验证模型可加载。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留本 Task 之外的 ObjectInteractionCm/Cm 用户改动和既有诊断日志增量）
- scope: 只读定位 SMPL-X 模型、解析已有软链接并在 CPU 上加载 male/female/neutral；不复制、移动、解压或修改模型，不启动数据转换和导出。

**文件**

- 模型父目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models`；DExplore `convert_grab.py --smplx_model_dir` 应传该父目录，由 `smplx.create(..., model_type="smplx")` 追加 `smplx/`。
- 模型目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/smplx`，包含 male/female/neutral 的 `.npz` 与 `.pkl`，版本文件声明 SMPL-X `Version 1.0`。
- 已有入口：`/home/wbcd/workspace/dex/retarget/InterAct/models/smplx` 是指向上述模型目录的可解析软链接；当前 `/home/wbcd/workspace/oyx_ws/InterAct` 尚未创建对应链接。
- [`src/task/ObjectInteractionCmv2/docs/plan/V1.0.md`](../plan/V1.0.md) — 后续真实数据对照的现有 final plan 边界。

**原因**

确认此前报告的“缺少 SMPL-X”是搜索范围不足，而不是模型资产实际缺失；避免重复下载 licensed asset 或要求用户再次提供。

**验证**

- 精确文件搜索确认 `SMPLX_{MALE,FEMALE,NEUTRAL}.{npz,pkl}` 六个文件存在；三份 `.pkl` 大小约 `544 MB`，三份 `.npz` 大小约 `109 MB`。
- 使用 `/home/wbcd/miniconda3/envs/dexplore-data/bin/python` 从模型父目录分别执行 `smplx.create`，male/female/neutral 均成功加载为 `SMPLX`，`v_template=(10475, 3)`、`NUM_JOINTS=54`、`num_betas=10`、`num_expression_coeffs=10`。
- 未创建 run_id 或任何运行/数据产物；SMPL-X 依赖可用性结论为 `SUPPORTED`，重导出和性能结论仍为 `INCONCLUSIVE`。

**回滚**

删除本条 activity 增量即可；本轮没有模型、代码、配置或数据变更需要恢复。

## 2026-09-15 12:34:26 +0000 — GRAB→DExplore teacher RL 配对数据 pilot 启动

- activity_id: `ACT-20260915-123426-CMV2-GRAB-DEXPLORE-PILOT-START`
- timestamp: `2026-09-15 12:34:26 +0000`
- modification_version: `V1.0.3`
- type: `code / data / operation`
- task_mode: `change`，pilot 运行开始后切换为 `run-only/operation`
- change_level: `L2 / L3`
- approval: `user-approved`
- approval_basis: 用户确认使用 DExplore teacher `inspire.pth`、seed 42、先 10 条 pilot 再全量导出、写入新目录，并在上一会话末要求继续；本会话按该已批准口径恢复执行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留本 Task 之外的 `ObjectInteractionCm` / `Cm` 用户改动；本 Task 尚有此前诊断记录增量）
- run_id: `grab-dexplore-rl-pilot10-20260915T122717Z`
- run_status: `STARTED`
- scope: 新增 Task-local GRAB canonical 构建与配对验证工具；只读使用原始 GRAB、共享 SMPL-X、DExplore `c31f57f` 和 teacher checkpoint；先处理固定 10 条 pilot，验证后才启动全量。不得修改外部 DExplore/InterAct、旧 `ObjectInteractionCm`、`src/base`、原始数据或已有运行。

**文件与产物**

- [`src/task/ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py`](../../tools/data/build_grab_dexplore_export.py) — 确定性 selection、GRAB canonical 构建、资产生成和 geometric/RL 配对验证。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 将本 Task 指针推进为 `V1.0.3`；同文件中旧 Task 的既有改动不属于本次范围。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/selection.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/selection.json) — 固定输入清单；当前规则实际得到 656 条，而历史 README 记录为 660 条，差异已显式保留。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/) — pilot 运行目录。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/run_manifest.json) — `STARTED` 运行合同。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/canonical/` — PENDING。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/geometric/` — PENDING。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/rl/` — PENDING。

**原因**

历史 660 计数无法由当前固定代码和原始数据复现；按 DExplore 实际左手 segment/LBS 映射重新扫描 1335 条后得到 656 条、排除 662 条左手接触与 17 条 doorknob。固定真实 selection 后再运行 pilot，可避免为匹配旧文档数字而改变数据语义。

**验证**

- `/home/wbcd/miniconda3/envs/dexplore-data/bin/python -m py_compile src/task/ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py`：通过。
- selection 完整扫描 `1335/1335`；`656 + 662 + 17 = 1335`，s1 男性 SMPL-X 与当前 DExplore 映射得到 778 个左手顶点。
- checkpoint SHA256：`8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- 工程状态：运行中；科研性能结论：`INCONCLUSIVE`。

**回滚**

停止尚在运行的本 run_id 后，删除本次新增 Task-local 工具与独立 `V1.0.3` 数据目录，并将 `ObjectInteractionCmv2` 指针恢复为 `V1.0.2`；不触碰外部仓库、原始数据、旧 Task 或其他运行。

## 2026-09-15 12:41:01 +0000 — 首次 GRAB→DExplore pilot 因坐标检查失败而终止

- activity_id: `ACT-20260915-124101-CMV2-GRAB-DEXPLORE-PILOT-FAIL`
- timestamp: `2026-09-15 12:41:01 +0000`
- modification_version: `V1.0.3`
- type: `data / operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续 `ACT-20260915-123426-CMV2-GRAB-DEXPLORE-PILOT-START` 的已批准 pilot；按 final plan 的坐标异常停止条件终止，未进入 RL rollout。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `grab-dexplore-rl-pilot10-20260915T122717Z`
- run_status: `FAILED`
- scope: 终止当前独立 pilot，保留失败产物用于审计；不启动 teacher RL、全量 canonical 或全量 geometric 导出。

**文件与产物**

- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/) — 失败运行目录。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/run_manifest.json) — `FAILED` 终态与失败原因。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/commands.log`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T122717Z/commands.log) — canonical、geometric 与检查命令输出。
- [`src/task/ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py`](../../tools/data/build_grab_dexplore_export.py) — 已移除错误的一次 forward 快捷假设，改为按 InterAct 语义重新执行变换后 SMPL-X forward。

**原因**

首次实现把 SMPL-X global orientation 当作绕世界原点的刚体旋转，导致 canonical body translation 错误；离线检查在接触帧得到 `min_tip_dist=0.491 m`。对照旧单序列链路后确认 object pose 一致而 human translation 最大差 `0.388889 m`，定位为实现错误。

**验证**

- canonical 10/10 与 geometric 10/10 均生成，但 `visualize_inspire_trajectory.py --check-only` 触发坐标警告。
- 未生成 `rl/`，未启动 checkpoint rollout；工程结论：`INVALID_IMPLEMENTATION`，科研性能结论：`INCONCLUSIVE`。
- 修正后必须在新的 run_id 中重新构建，不覆盖本失败目录。

**回滚**

失败目录保留为审计入口；如用户后续要求清理，可删除该独立目录，不影响原始 GRAB、外部仓库或其他运行。

## 2026-09-15 12:41:37 +0000 — 修正后 GRAB→DExplore pilot 重启

- activity_id: `ACT-20260915-124137-CMV2-GRAB-DEXPLORE-PILOT-RESTART`
- timestamp: `2026-09-15 12:41:37 +0000`
- modification_version: `V1.0.3`
- type: `data / operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 在原批准范围内修复首次 pilot 的实现错误后，以新目录重新执行同一固定 10 条序列；不改变 checkpoint、seed、selection 或数据合同。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `grab-dexplore-rl-pilot10-20260915T124102Z`
- run_status: `STARTED`
- scope: 重建修正后的 canonical 和 geometric 数据，坐标检查通过后才运行 teacher RL；仍只使用 GPU 3，不占用 GPU 1、2 的外部训练。

**文件与产物**

- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/) — 新 pilot 运行目录。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/run_manifest.json) — `STARTED` 运行合同。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/canonical/` — PENDING。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/geometric/` — PENDING。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/rl/` — PENDING。

**原因**

用新 run_id 保留失败证据，同时验证修正后的 SMPL-X 根变换是否恢复 hand/object 对齐。

**验证**

- 修正脚本再次通过 `py_compile`。
- 运行状态：`STARTED`；工程与科研结论均待 pilot 终态。

**回滚**

可停止该 run_id 并删除独立新目录；失败 pilot、原始输入、外部仓库和其他运行不受影响。

## 2026-09-15 12:50:39 +0000 — 修正后 GRAB→DExplore pilot 验证完成

- activity_id: `ACT-20260915-125039-CMV2-GRAB-DEXPLORE-PILOT-COMPLETE`
- timestamp: `2026-09-15 12:50:39 +0000`
- modification_version: `V1.0.3`
- type: `code / data / operation`
- task_mode: `change` 与 `run-only/operation`
- change_level: `L2 / L3`
- approval: `user-approved`
- approval_basis: 延续 `ACT-20260915-124137-CMV2-GRAB-DEXPLORE-PILOT-RESTART` 的已批准范围；pilot 验证通过后按约定进入全量导出。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留本 Task 之外的 `ObjectInteractionCm` / `Cm` 用户改动）
- run_id: `grab-dexplore-rl-pilot10-20260915T124102Z`
- run_status: `COMPLETED`
- scope: 完成固定 10 条序列的 canonical、geometric 与 teacher RL 配对导出；补充终态中的 URDF FK 手物对齐检查，不改变 checkpoint、seed、selection、外部 DExplore 或原始数据。

**文件与产物**

- [`src/task/ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py`](../../tools/data/build_grab_dexplore_export.py) — 修复 SMPL-X 根变换并加入终态配对与手物对齐验证。
- [`src/task/ObjectInteractionCmv2/tools/data/run_grab_dexplore_rl.py`](../../tools/data/run_grab_dexplore_rl.py) — 使用 run-local 资产覆盖层、显式 seed 和可恢复 batch 执行 teacher rollout。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/) — 259 MiB pilot 运行目录。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/run_manifest.json) — `COMPLETED` 终态与依赖快照。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/manifest.json) — 逐序列配对统计和 schema 合同。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/validation.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/validation.json) — 10 条、3335 帧终态验证结果。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/commands.log`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/pilot10_20260915T124102Z/commands.log) — 全部执行命令及输出。

**原因**

pilot 需要先证明修正后的 canonical 坐标、确定性 geometric retarget 与 teacher rollout 能形成同序列、同帧且受保护字段不变的配对样本，才能安全扩大到全量。

**验证**

- `python -m py_compile ...build_grab_dexplore_export.py ...run_grab_dexplore_rl.py`：通过。
- canonical、geometric、RL 均为 `10/10`；总计 3335 帧，shape 均为 `[T, 598]`，数值有限。
- 10/10 序列的 object pose 与 Inspire DOF 均被 teacher rollout 更新；受保护列最大绝对差为 `0.0`。
- `s1_airplane_fly_1` 接触帧 153 的最小指尖到物体中心距离为 `0.009514 m`，低于 `0.15 m` 阈值。
- 工程数据合同结论：`SUPPORTED`；该 smoke 只证明导出实现有效，不构成模型效果结论，科研性能结论仍为 `INCONCLUSIVE`。

**回滚**

删除独立 pilot 目录和本次新增 Task-local 工具，并将 `ObjectInteractionCmv2` 指针恢复为 `V1.0.2`；不触碰失败审计目录、原始 GRAB、外部 DExplore、旧 Task 或其他运行。

## 2026-09-15 12:51:48 +0000 — 全量 GRAB→DExplore 配对导出开始

- activity_id: `ACT-20260915-125148-CMV2-GRAB-DEXPLORE-FULL-START`
- timestamp: `2026-09-15 12:51:48 +0000`
- modification_version: `V1.0.3`
- type: `data / operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户批准先执行 10 条 pilot、通过后执行全量；`ACT-20260915-125039-CMV2-GRAB-DEXPLORE-PILOT-COMPLETE` 已记录 pilot 为 `SUPPORTED`。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留本 Task 之外的 `ObjectInteractionCm` / `Cm` 用户改动）
- run_id: `grab-dexplore-rl-full-20260915T125105Z`
- run_status: `STARTED`
- scope: 按已固定 selection 导出全部 656 条 canonical、geometric 和 teacher RL 配对轨迹；seed 42、teacher checkpoint 与 pilot 相同。GPU 1、2 被外部训练占用，本次仅使用空闲 GPU 3，不停止或抢占其他进程。

**文件与产物**

- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/selection.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/selection.json) — 固定 656 条输入清单。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/) — 全量运行目录。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json) — `STARTED` 运行合同。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/canonical/` — PENDING。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/geometric/` — PENDING。
- `data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/rl/` — PENDING。

**原因**

pilot 已验证坐标、shape、配对列保护与 teacher 导出合同，因此在不改变数据语义的前提下扩大到固定的完整兼容集合。

**验证**

- 初始化 manifest 声明 `num_selected=656`、`seed=42`、`devices=3`，checkpoint SHA256 为 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- 运行状态：`STARTED`；工程与科研结论待全量终态。

**回滚**

停止该 run_id 并删除独立全量运行目录；pilot、selection、原始输入、外部仓库和其他运行不受影响。

## 2026-09-15 15:11:47 +0000 — 全量 GRAB→DExplore 配对导出完成

- activity_id: `ACT-20260915-151147-CMV2-GRAB-DEXPLORE-FULL-COMPLETE`
- timestamp: `2026-09-15 15:11:47 +0000`
- modification_version: `V1.0.3`
- type: `code / data / operation`
- task_mode: `change` 与 `run-only/operation`
- change_level: `L2 / L3`
- approval: `user-approved`
- approval_basis: 延续 `ACT-20260915-125148-CMV2-GRAB-DEXPLORE-FULL-START` 的已批准全量导出；实现层修复未改变 selection、数据语义、checkpoint、seed 或输出 schema。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留本 Task 之外的 `ObjectInteractionCm` / `Cm` 用户改动；运行期间其他工作将 `ObjectInteractionCm` 指针推进至 `V1.4.5`，不属于本次范围）
- run_id: `grab-dexplore-rl-full-20260915T125105Z`
- run_status: `COMPLETED`
- last_step: `N/A`（数据导出）
- last_epoch: `N/A`（数据导出）
- best_metric: `N/A`（数据导出）
- checkpoint: DExplore teacher `/home/wbcd/workspace/oyx_ws/dexplore/checkpoint/inspire.pth`，SHA256 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`
- scope: 完成固定 656 条 GRAB 序列的 canonical、geometric 与 teacher RL 配对导出和终态验证；只修改 Task-local 工具、当前版本指针和 Task activity，外部 DExplore、原始 GRAB、共享 `src/base`、旧 `ObjectInteractionCm` 与其他运行保持不变。

**文件与产物**

- [`src/task/ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py`](../../tools/data/build_grab_dexplore_export.py) — canonical 构建、selection、资产生成、配对验证和终态 manifest。
- [`src/task/ObjectInteractionCmv2/tools/data/run_grab_dexplore_rl.py`](../../tools/data/run_grab_dexplore_rl.py) — run-local 资产覆盖层、显式环境路径、seed 42、单 GPU batch 与断点恢复。
- [`src/task/ObjectInteractionCmv2/tools/__init__.py`](../../tools/__init__.py) 与 [`src/task/ObjectInteractionCmv2/tools/data/__init__.py`](../../tools/data/__init__.py) — Task-local 数据工具包入口。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 本 Task 指针为 `V1.0.3`；同文件其他 Task 的既有增量不属于本次修改。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/) — 7.4 GiB 全量运行目录。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json) — `COMPLETED` 终态、依赖版本和输出入口。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/manifest.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/manifest.json) — 656 条逐序列统计和配对 schema。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/validation.json`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/validation.json) — 165423 帧终态验证。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/commands.log`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/commands.log) — canonical、geometric、RL 和验证命令及退出状态。
- [`data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/rl/_logs/`](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/rl/_logs/) — 21 个 teacher rollout batch 日志。

**原因**

ObjectInteractionCmv2 的真实 rigid 训练需要同一 GRAB 参考轨迹下的 geometric retarget 与 teacher-policy 实际轨迹配对；全量导出把已通过 pilot 的合同扩展到当前代码可复现的完整 DExplore 兼容集合。

**验证**

- canonical `656/656`、geometric `656/656`、teacher RL `656/656`，阶段均为 0 个数据失败；共 165423 帧。
- 21/21 RL batch 日志均记录 seed 42 和指定 checkpoint，未发现 traceback、OOM 或 segmentation fault。
- 每条 geometric/RL tensor shape 均为 `[T, 598]` 且数值有限；两侧序列集合与 canonical 精确一致。
- 656/656 序列的 object pose `[198:205]` 和 Inspire DOF `[373:391]` 均发生 teacher 更新；其余受保护列最大绝对差为 `0.0`。
- `s1_airplane_fly_1` 接触帧 153 的最小指尖到物体中心距离为 `0.017122 m`，低于 `0.15 m` 阈值。
- 首次全量 RL 启动因子进程缺少 `LD_LIBRARY_PATH` 而在生成轨迹前停止；Task-local launcher 补齐 conda `lib/` 后从 0 条恢复，最终 21 个 batch 全部成功。
- 首次全量验证因 selection 与扁平目录的字典序不同而误报集合不等；统一排序后对同一 656 元素集合完成逐条验证。
- 工程数据合同结论：`SUPPORTED`；本结果证明数据导出实现和配对合同成立，不是 ObjectInteractionCmv2 模型效果证据，科研性能结论仍为 `INCONCLUSIVE`。

**回滚**

删除独立 `full_20260915T125105Z` 运行目录和本次新增 Task-local 工具，并将 `ObjectInteractionCmv2` 指针恢复为 `V1.0.2`；不触碰 selection 之外的原始数据、外部 DExplore、旧 `ObjectInteractionCm`、共享 `src/base`、其他 Task 或其他运行。

## 2026-09-16 15:59:45 +0000 — V1.2 定稿前 GRAB/MANO 输入只读核对

- activity_id: `ACT-20260916-155945-CMV2-GRAB-INPUT-DIAGNOSTIC`
- timestamp: `2026-09-16 15:59:45 +0000`
- modification_version: `V1.0.3`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求读取前序会话后继续；读取已批准 cache 产物与前序指令，无代码实现或新实验。V1.2 仍为草稿，不推进 Task 指针。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- scope: 读取 1335 条 GRAB 的逐序列 manifest、frame_time、source_frame_id 和 split；每个 split 各抽一条序列核对刚体坐标一致性。仅追加本 activity；不修改代码、配置、GT、split、数据、架构或指导。
- conclusion: `SUPPORTED`（检查覆盖内的数据兼容性）；训练效果 `INCONCLUSIVE`。

**原因**

前会话已将首轮范围明确为 GRAB-only、MANO-only、无铰接，实施前需要确认现有 cache 中可选出对应数据且不改变既有 split。此记录只保存新诊断证据，不记录协商中的 plan 草稿修改。

**文件与证据入口**

- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本次诊断。
- [src/task/ObjectInteractionCmv2/docs/plan/V1.2.md](../plan/V1.2.md) — 待定稿执行边界。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](../../../ObjectInteractionCm/docs/logs/activity_log.md) — 独立 cache 运行的状态入口。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/) — 输入逐序列 cache。
- [data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/index.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/index.json) — 仅用于核对继承的 GRAB split，不读取 Inspire 几何。

**验证**

解释器为 `/home/wbcd/miniconda3/envs/graspenv/bin/python`；通过只读内联 Python 遍历 `sequences/**/geometry/manifest.json`，仅筛选 `dataset=grab`，逐条读取 `frame_time.npy` 与 `source_frame_id.npy`，校验 `np.diff(source_frame_id)==4`、`np.isclose(np.diff(frame_time.astype(float64)), 1/30, atol=1e-5, rtol=0)`，并将 sequence→split 映射与原 index 精确比较。结果：

| split | 序列 | 帧 | 连续 stride=1 训练对候选 |
| --- | ---: | ---: | ---: |
| train | 1068 | 327790 | 326722 |
| val | 134 | 40896 | 40762 |
| test | 133 | 37578 | 37445 |
| 合计 | 1335 | 406264 | 404929 |

- 时间差范围 `0.03333282470703125..0.033336639404296875 s`；404929 个原始帧号增量全部为 4，无缺失配对。
- split missing/extra 均为 0；逐序列 `source=mano`、`object_representation=rigid_se3`、左右手顺序和每侧 2048 点均符合声明，失败 0。固定 surface correspondence 的左右 SHA256 组合只有 1 种，`cross_frame_fixed=true`。
- 每个 split 的首条序列在首/中/末帧计算 `(obj_points_world - pose[:3,3]) @ pose[:3,:3]`；test/train/val 的 canonical 一致性最大误差分别为 `1.2066312e-7 / 1.0141586e-7 / 1.2148050e-7 m`。这只是 3 条坐标抽检，不是全量刚体几何证明。
- 独立 cache finalize 负责全量 shape、finite、KNN 范围及覆盖；其结果单独记录在生产 Task。
- 不运行训练，不生成 checkpoint、metrics.jsonl 或 train.log，不把数据兼容性当作模型效果证据。

**回滚**

仅撤销本次诊断 activity；源 cache、split、指导和架构无需恢复。

## 2026-09-17 02:18:36 +0000 — V1.2 GRAB/MANO 有界训练 smoke 启动

- activity_id: `ACT-20260917-021836-CMV2-GRAB-V12-SMOKE-START`
- timestamp: `2026-09-17 02:18:36 +0000`
- modification_version: `V1.2.1`
- type: `experiment / operation`
- task_mode: `run-only/operation`（V1.2 已按用户确认从 change 切换到有界验证）
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确将 [src/task/ObjectInteractionCmv2/docs/plan/V1.2.md](../plan/V1.2.md) 定为最终版；该计划第 5 节授权至多 3 条 train 序列、8 optimizer steps、batch size 2 的真实训练 smoke。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `grab-mano-v1-2-smoke-20260917T021836Z`
- run_status: `STARTED`
- scope: 只读使用已完成的 GRAB/MANO cache 前 3 条 train 序列，GPU 1、seed 42、最多 8 steps；输出写 NAS 独立目录，不改变 cache、split、GT 或其他运行。

**文件与产物**

- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_2_smoke.yaml](../../configs/active/grab_mano_v1_2_smoke.yaml) — 固定 smoke 边界。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — 输入 cache 终态。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本次运行状态入口。
- `outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-20260917T021836Z/`：PENDING（NAS 路径；run_status=STARTED）。

**原因**

按最终计划验证真实数据到 swept 模型、优化器和运行 manifest 的接线；不把 smoke 当科研效果结论。

**命令与验证**

- `REF2DEX_GRAB_MANO_CACHE=data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4 REF2DEX_CMV2_OUTPUT_ROOT=/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2 /home/wbcd/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCmv2.train --config src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_2_smoke.yaml --run-id grab-mano-v1-2-smoke-20260917T021836Z`。
- 定向 pytest 7/7 已通过；真实运行尚待终态，工程结论 `INCONCLUSIVE`。

**回滚**

停止本 run 的进程并保留独立 NAS 输出和日志；不触碰输入 cache 或旧 checkpoint。

## 2026-09-17 02:22:24 +0000 — V1.2 GRAB-only swept 交互实现

- activity_id: `ACT-20260917-022224-CMV2-V12-IMPLEMENT`
- timestamp: `2026-09-17 02:22:24 +0000`
- modification_version: `V1.2.1`
- type: `code / architecture / documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户于 2026-09-17 明确确认 [src/task/ObjectInteractionCmv2/docs/plan/V1.2.md](../plan/V1.2.md) 为最终版；仅在该计划的 GRAB-only/MANO-only、刚体和有界验证边界实施。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- scope: Task-local V1.2 swept KNN32、GRAB/MANO stride=1 adapter、配置化训练/评估和定向测试；未修改旧 ObjectInteractionCm、公共 src/base、输入 cache、GT、split、指导、架构快照或正式训练配置。
- conclusion: `SUPPORTED`（实现与有界工程验证）；科研效果 `INCONCLUSIVE`。

**文件**

- [src/task/ObjectInteractionCmv2/docs/plan/V1.2.md](../plan/V1.2.md) — 用户确认后由 draft 标为 final。
- [src/task/ObjectInteractionCmv2/model.py](../../model.py) — 完整有效手点流的分块 swept top-32、硬 2 cm mask、时间/距离边特征；旧静态模式保留供旧入口读取 checkpoint。
- [src/task/ObjectInteractionCmv2/grab.py](../../grab.py) — GRAB/MANO 刚体 manifest/index 校验、原 split、固定 `(t,t+1)` 和当前物体坐标系监督。
- [src/task/ObjectInteractionCmv2/config.py](../../config.py) 与 [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_2_smoke.yaml](../../configs/active/grab_mano_v1_2_smoke.yaml) — 配置化路径及 V1.2 硬约束、有界 smoke 参数。
- [src/task/ObjectInteractionCmv2/train.py](../../train.py)、[src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py)、[src/task/ObjectInteractionCmv2/eval.py](../../eval.py)、[src/task/ObjectInteractionCmv2/eval_grab.py](../../eval_grab.py) — 新配置入口、旧入口兼容、运行快照、checkpoint/resume 与固定 stride 评估。
- [src/task/ObjectInteractionCmv2/tests/test_v1_2_swept.py](../../tests/test_v1_2_swept.py) — 中途掠过、零流、tie、分块精确性、严格阈值、全 padding 和未来 GT 不进入模型输入。
- [src/task/ObjectInteractionCmv2/docs/README.md](../README.md)、[src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 计划导航、实施记录和本 Task 指针 `V1.2.1`。

**原因**

首轮需要用候选 MANO 手点流线段相对当前物体的最短距离构造因果局部交互，同时保证对象全局 1024 点和 GRAB 既有刚体监督、split 不变。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：8 passed。
- `git diff --check -- src/task/ObjectInteractionCmv2 docs/current_versions.yaml`：通过。
- 真实 cache 每个 split 各 1 条读取检查：train/val/test 均为 `[1024,3]` 物体、`[4096,3]` MANO 手，`delta_time_s≈0.033333335`，检查范围内无缺失配对或非有限值；前 3 条 train 序列共有 1002 个有效配对、排除 0。
- 后续 8-step smoke 及 resume 仅证明接线、梯度和重现性；评估结果不得作为性能成立证据。

**回滚**

仅撤销上述 V1.2 Task-local 新文件及局部修改，并把 ObjectInteractionCmv2 的版本指针恢复为 `V1.0.3`；保留既有用户改动、旧代码历史、输入 cache 和独立运行目录。

## 2026-09-17 02:22:25 +0000 — V1.2 有界训练、恢复与评估终态

- activity_id: `ACT-20260917-022225-CMV2-V12-SMOKE-COMPLETE`
- timestamp: `2026-09-17 02:22:25 +0000`
- modification_version: `V1.2.1`
- type: `experiment / operation / diagnostic`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: V1.2 最终计划第 5 节只批准至多 3 条 train 序列、8 steps、batch size 2 的有界工程验证；未启动正式训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `grab-mano-v1-2-smoke-timed-20260917T022100Z`（主 smoke）；`grab-mano-v1-2-resume-fixed-20260917T022100Z`（恢复）；`grab-mano-v1-2-eval-smoke-20260917T022200Z`（验证评估）。
- run_status: `COMPLETED`（以上三项）；`grab-mano-v1-2-resume-20260917T022100Z` 为 `FAILED`，原因是初版恢复把 CPU RNG 状态随 `map_location` 映射到 GPU，已修复并保留失败证据。初次 `grab-mano-v1-2-smoke-20260917T021836Z` 为 `COMPLETED`，后由同步计时的同参数运行复核。
- conclusion: `SUPPORTED`（有界工程接线、终态与断点恢复）；GRAB 模型效果 `INCONCLUSIVE`。
- scope: 只使用 GRAB/MANO cache 前 3 条 train 序列及 val 的 1 条、8 个配对；GPU 1，seed 42，所有输出写 NAS 独立目录，不覆盖旧运行。

**命令与产物**

- 训练：`REF2DEX_GRAB_MANO_CACHE=data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4 REF2DEX_CMV2_OUTPUT_ROOT=/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2 /home/wbcd/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCmv2.train --config src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_2_smoke.yaml --run-id grab-mano-v1-2-smoke-timed-20260917T022100Z`。
- 恢复命令在上述命令基础上增加 `--resume outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/step_4.pt`，改用独立 `run_id=grab-mano-v1-2-resume-fixed-20260917T022100Z`；失败的首次恢复同样独立留档。
- 评估：`python -m src.task.ObjectInteractionCmv2.eval --config src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_2_smoke.yaml --checkpoint outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/latest.pt --split val --max-sequences 1 --max-pairs 8 --run-id grab-mano-v1-2-eval-smoke-20260917T022200Z`，沿用上述环境变量及 graspenv 解释器。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/) — 主 smoke NAS 输出；[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/train.log)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/latest.pt) 与 [step_4.pt](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-timed-20260917T022100Z/step_4.pt)。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-fixed-20260917T022100Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-fixed-20260917T022100Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-fixed-20260917T022100Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-fixed-20260917T022100Z/train.log)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-fixed-20260917T022100Z/latest.pt) — 恢复运行终态。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-20260917T022100Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-20260917T022100Z/run_manifest.json) 与 [train.log](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-resume-20260917T022100Z/train.log) — 初次恢复失败证据。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-eval-smoke-20260917T022200Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-eval-smoke-20260917T022200Z/run_manifest.json) 与 [metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-eval-smoke-20260917T022200Z/metrics.jsonl) — val 有界评估。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-20260917T021836Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-20260917T021836Z/run_manifest.json) 与 [metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-20260917T021836Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-20260917T021836Z/train.log)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-smoke-20260917T021836Z/latest.pt) — 第一次 8-step smoke 终态。

**原因**

按最终计划的有界工程验证范围检查真实 GRAB 数据接线、运行产物、固定 stride 评估和断点恢复；首次恢复错误需要保留失败证据并修复。

**验证**

- 主 smoke `last_step=8`、`last_epoch=0`、`best_metric=N/A`；1002 个候选配对、0 个排除，8 条训练指标均有限；同步计时首步 0.184 s、后续约 0.026 s/step，PyTorch GPU 峰值已分配显存约 270 MB。没有正式验证选择最佳 checkpoint；仅有 `latest.pt` 和第 4 步恢复点。
- 从第 4 步 checkpoint 继续到第 8 步，连续运行与恢复运行的最终模型参数逐项 `torch.equal`，最大差 0.0；恢复运行 `last_step=8`，`best_metric=N/A`。
- val 工程评估只读 1 条序列的 8 个配对：EPE 25.287 mm、RMSE 25.646 mm、zero-flow EPE 0.0261 mm；模型明显差于该基线，但训练仅 8 步，科研效果结论为 `INCONCLUSIVE`，不据此选择 checkpoint 或修改计划。
- 初次恢复 `FAILED` 的 `last_step=0`（尚未训练）、无 metrics/checkpoint；错误为 `TypeError: RNG state must be a torch.ByteTensor`，修复 CPU RNG 恢复后重试成功。数据/GT/既有 checkpoint 未受影响。

**回滚**

停止相应 run（均已终止），按 `run_id` 独立保留或移除 NAS smoke/评估目录；不删除输入 cache、旧运行或用户已有修改。正式训练预算仍待单独确认。

## 2026-09-17 02:24:04 +0000 — 显式 swept 配置的 V1.2 主 smoke 完成

- activity_id: `ACT-20260917-022404-CMV2-V12-FINAL-SMOKE`
- timestamp: `2026-09-17 02:24:04 +0000`
- modification_version: `V1.2.1`
- type: `experiment / operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认的 V1.2 最终计划第 5 节授权相同上限的工程 smoke；本次使用显式 `interaction_mode: swept` 的配置快照。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `grab-mano-v1-2-final-smoke-20260917T022400Z`
- run_status: `COMPLETED`
- last_step: `8`
- last_epoch: `0`
- best_metric: `N/A`（无正式验证选择）
- latest_checkpoint: [outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/latest.pt](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/latest.pt)
- conclusion: `SUPPORTED`（8-step 工程接线）；科研效果 `INCONCLUSIVE`。
- scope: 同一已完成 GRAB/MANO 输入、前 3 条 train 序列、GPU 1、seed 42、8 steps、batch size 2；NAS 独立新目录，未改变源 cache、GT、split 或旧运行。

**文件与产物**

- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_2_smoke.yaml](../../configs/active/grab_mano_v1_2_smoke.yaml) — 显式 swept 配置。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/) — NAS 输出目录。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/run_manifest.json) — 终态和输入 hash。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/config.json](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/config.json) — 解析后配置快照。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/metrics.jsonl) 与 [outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/train.log](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/train.log) — 8 步指标和日志。
- [outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/step_4.pt](../../../../../outputs/ObjectInteractionCmv2/grab-mano-v1-2-final-smoke-20260917T022400Z/step_4.pt) — 恢复点；最新 checkpoint 见上。

**原因**

此前主 smoke 的快照依靠当时模型默认模式推断 swept；显式配置后复跑，使运行快照独立表达交互模式。

**验证**

- 配置快照包含 `interaction_mode=swept`；`run_manifest.json` 为 `COMPLETED`，8 条指标均有限，`valid_pairs=1002`、`dropped_pairs=0`。
- 新 checkpoint 与前一次同步计时的 8-step smoke 在全部模型参数上逐项 `torch.equal`；后者从第 4 步恢复的最终参数也完全相同。
- 仅是工程 smoke；不把 val 上差于 zero-flow 的前序 8-pair 评估视为模型效果已成立。

**回滚**

按本 `run_id` 保留或移除独立 NAS smoke 目录；不修改输入 cache、其他运行或历史 checkpoint。

## 2026-09-17 02:25:43 +0000 — GRAB 时间与来源隔离合同补验

- activity_id: `ACT-20260917-022543-CMV2-V12-GRAB-CONTRACT-TEST`
- timestamp: `2026-09-17 02:25:43 +0000`
- modification_version: `V1.2.1`
- type: `code / diagnostic`
- task_mode: `change`
- change_level: `L0`（仅增加 Task-local 数据合同测试及模型说明文字）
- approval: `auto`
- approval_basis: V1.2 最终计划已批准本 Task 的定向测试；测试不改变训练行为或科研变量。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- scope: 为 V1.2 GRAB adapter 加合成 cache 合同测试，检查缺帧排除、GRAB/MANO 来源隔离、左右手顺序拒绝和当前物体坐标系监督；源 cache、模型参数和运行目录不变。
- conclusion: `SUPPORTED`（定向工程合同）；科研性能仍 `INCONCLUSIVE`。

**文件**

- [src/task/ObjectInteractionCmv2/tests/test_v1_2_grab_contract.py](../../tests/test_v1_2_grab_contract.py) — 合成四帧 cache 的缺帧、来源和坐标测试。
- [src/task/ObjectInteractionCmv2/model.py](../../model.py) — 文档字符串注明静态和 swept 两种模式。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本条验证记录。

**原因**

正式训练前需要把数据来源隔离和缺帧配对限制变成可重复检查，而不只依赖真实 cache 的人工抽检。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：9 passed。
- `git diff --check -- src/task/ObjectInteractionCmv2 docs/current_versions.yaml`：通过。
- 合成测试没有写入正式 GRAB cache；真实 3 条 train 和 val/test 各 1 条读取检查仍见 `ACT-20260917-022224-CMV2-V12-IMPLEMENT`。

**回滚**

仅移除本次 Task-local 测试和模型说明文字；正式 cache、checkpoint、split、GT 和其他运行无需恢复。

## 2026-09-17 02:26:33 +0000 — 有界 smoke 的零值参数保护

- activity_id: `ACT-20260917-022633-CMV2-V12-SMOKE-BOUNDS`
- timestamp: `2026-09-17 02:26:33 +0000`
- modification_version: `V1.2.1`
- type: `code / diagnostic`
- task_mode: `change`
- change_level: `L1`
- approval: `auto`
- approval_basis: V1.2 final plan 已授权 Task-local 有界工程验证；本次仅收紧 smoke 参数校验，不改变科学变量、数据合同或训练上限。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- scope: smoke 模式下 train/eval 的序列、步数、batch 和配对数必须为正且不超过批准上限；避免零值被解释为全量读取。
- conclusion: `SUPPORTED`（参数边界检查）；科研效果 `INCONCLUSIVE`。

**文件**

- [src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py) — smoke 训练预算正数与上限校验。
- [src/task/ObjectInteractionCmv2/eval_grab.py](../../eval_grab.py) — smoke 评估序列和配对数正数与上限校验。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本次记录。

**原因**

评估循环原先把 `max_pairs=0` 当作无上限，需要显式拒绝该输入以维护有界验证授权。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：9 passed。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/{config,grab,model,train_grab,eval_grab,train,eval}.py`：通过。
- `git diff --check -- src/task/ObjectInteractionCmv2 docs/current_versions.yaml`：通过。
- 已完成 smoke 参数均为正且在上限内；不需要重跑已成功的 run。

**回滚**

仅撤销本次两处 Task-local 参数校验；不修改已完成的运行目录、cache 或 checkpoint。

## 2026-09-17 02:28:16 +0000 — 正式训练提案与 GRAB 全 split 只读扫描

- activity_id: `ACT-20260917-022816-CMV2-V12-FORMAL-PROPOSAL`
- timestamp: `2026-09-17 02:28:16 +0000`
- modification_version: `V1.2.1`
- type: `code / documentation / diagnostic`
- task_mode: `change`（提案配置与 checkpoint 间隔）；全量索引检查为 `read-only/diagnostic`
- change_level: `L2`（正式训练预算仅提出，不执行）
- approval: `user-approved`（V1.2 实施范围）；正式训练 `pending`
- approval_basis: 用户确认 V1.2 最终计划及有界验证；该计划第 5 节要求正式训练另行确认，故提案配置位于 `configs/proposed/`，没有启动该配置。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- scope: 提出 GRAB 全 train、最多 10000 steps、batch 2、GPU 1、seed 42、每 1000 steps 保存最新 checkpoint、训练后完整 val 固定 stride=1 评估的首轮探索性预算；只读扫描完整 train/val/test index，不启动正式训练或评估。
- conclusion: `SUPPORTED`（当前 adapter 的全 split 输入合同）；正式训练效果 `INCONCLUSIVE`。

**文件与证据入口**

- [src/task/ObjectInteractionCmv2/configs/proposed/grab_mano_v1_2.yaml](../../configs/proposed/grab_mano_v1_2.yaml) — 待用户确认的可执行提案；尚非 active 配置。
- [src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py) — 允许按配置间隔更新 latest checkpoint，smoke 默认每步保存。
- [src/task/ObjectInteractionCmv2/docs/plan/V1.2.md](../plan/V1.2.md) — 正式训练单独确认边界。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json) 与 [run_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — 只读输入。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本次记录。

**原因**

有界 smoke 已证明工程接线；用户需要在批准长时训练前看到明确变量、资源、停止上限及完整输入覆盖，而不应由 Agent 默认为全量训练。

**验证**

- `load_grab_config` 对 proposed YAML 解析成功：只接受 GRAB/MANO、stride=1、swept KNN32、硬 2 cm；提案为 10000 steps、batch 2、GPU 1、seed 42、每 1000 steps checkpoint。
- 只读构造 `GrabManoTransitions` 全量：train 1068 条、326722 对、排除 0；val 134 条、40762 对、排除 0；test 133 条、37445 对、排除 0。全量索引扫描约 5.58/0.87/0.56 s；没有加载训练张量或修改数据。
- 10000 steps × batch 2 只覆盖约 6.1% 的 train 配对，故该预算仅是首轮探索性训练提案，不能作为完整一轮训练或最终模型结论。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：9 passed；`py_compile`、`git diff --check` 通过。

**回滚**

删除 proposed YAML 并撤销可配置 checkpoint 间隔即可；输入 cache、已完成 smoke、checkpoint、正式训练状态均无需恢复。

## 2026-09-17 02:38:56 +0000 — V1.2 GRAB swept 实现代码单独提交

- activity_id: `ACT-20260917-023856-CMV2-V12-CODE-COMMIT`
- timestamp: `2026-09-17 02:38:56 +0000`
- modification_version: `V1.2.1`
- type: `operation / documentation`
- task_mode: `change`
- change_level: `L0`（仅按用户要求提交已验证代码）
- approval: `user-approved`
- approval_basis: 用户明确要求“先把代码提交一下”；仅提交本 Task 本轮 V1.2 的 10 个代码、配置和测试文件，不带入已有未提交计划、指导、记录、其他 Task 或产物。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a186aff3968ff9dcfe09182182f20a958d1b59f4`（本次提交后 HEAD）
- worktree_dirty: `true`（既有文档、版本指针和其他 Task 改动仍未提交）
- commit: `a186aff` — `实现 ObjectInteractionCmv2 GRAB 手流 swept 训练入口`
- scope: 提交 `config.py`、`configs/active/grab_mano_v1_2_smoke.yaml`、`eval.py`、`eval_grab.py`、`grab.py`、`model.py`、`tests/test_v1_2_grab_contract.py`、`tests/test_v1_2_swept.py`、`train.py`、`train_grab.py`；不推送远端。
- conclusion: `SUPPORTED`（代码提交与工程测试）；科研效果 `INCONCLUSIVE`。

**文件与入口**

- [src/task/ObjectInteractionCmv2/model.py](../../model.py)、[src/task/ObjectInteractionCmv2/grab.py](../../grab.py)、[src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py)、[src/task/ObjectInteractionCmv2/eval_grab.py](../../eval_grab.py) — 本次提交主要实现。
- [src/task/ObjectInteractionCmv2/docs/plan/V1.2.md](../plan/V1.2.md) 与 [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 工作区内已批准计划和时间线；此次代码提交没有暂存这些既有未提交文件。

**原因**

用户要求先将实现代码单独固定为可引用提交，同时保护工作区中大量并行任务改动。

**验证**

- 提交前 `git diff --cached --name-only` 仅列上述 10 个显式路径；`git diff --cached --check` 通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：9 passed。
- `git show --stat --oneline --summary HEAD`：确认提交 `a186aff` 只有 10 个 V1.2 实现路径；分支相对 origin ahead 1，未推送。

**回滚**

如需回退本次代码提交，以 `a186aff` 的父提交为代码回滚入口；操作时仍须保留工作区现有未提交文件和其他 Task 改动，不执行全工作区 reset。

## 2026-09-17 03:47:18 +0000 — V1.3 指导与 GRAB 位姿 GT 只读核对

- activity_id: `ACT-20260917-034718-CMV2-V13-PREFLIGHT`
- timestamp: `2026-09-17 03:47:18 +0000`
- modification_version: `V1.2.1`（V1.3 计划仍为草案，Task 指针未推进）
- type: `diagnostic / documentation`
- task_mode: `read-only/diagnostic`（仅另写协商中的 plan 草稿）
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求按 V1.3 指导继续并明确 DexYCB 暂不参与；只读核对现有合同并起草同版本计划，不修改代码、配置、数据或研究结论。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a186aff3968ff9dcfe09182182f20a958d1b59f4`
- worktree_dirty: `true`
- scope: 核对 V1.3 指导、V1.2 模型和 GRAB cache pose/point correspondence；只读抽查各 split 两条、每条首/中/末相邻帧，并确认当前没有 ObjectInteractionCmv2 可直接消费的 DexYCB 双手 MANO2048 cache。V1.3 草稿变化不作为实施记录。
- conclusion: `SUPPORTED`（所抽帧的直接 SE(3) GT 可重建点流）；全量 GT、噪声阈值和模型收益 `INCONCLUSIVE`。

**文件与证据入口**

- [src/task/ObjectInteractionCmv2/docs/指导/V1.3.md](../指导/V1.3.md) — 用户指定的新研究方向，未修改。
- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 协商中的 draft，尚无代码实施授权。
- [src/task/ObjectInteractionCmv2/model.py](../../model.py) 与 [src/task/ObjectInteractionCmv2/grab.py](../../grab.py) — 现有 V1.2 模型、坐标变换和监督来源。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/) — 只读 GRAB/MANO cache；[run_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) 为已完成输入。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本次诊断记录。

**原因**

V1.3 直接监督刚体位姿会新增 GT 合同；在协商前需确认缓存里的 `obj_pose_world` 与逐点流真实一致，并分清 DexYCB 其他 Task 的 Stage4 cache 与本 Task 输入合同。

**验证**

- 只读 NumPy 检查 `R_t^T R_(t+1)`、`R_t^T(p_(t+1)-p_t)` 重建下一帧当前物体坐标系物体点；train/val/test 各前两条，每条首/中/末相邻帧，最大误差分别小于 `1.23e-7 / 1.33e-7 / 1.22e-7 m`。此为 18 帧对抽检，不是全量证明。
- `data/processed_data` 顶层没有本 Task 可直接使用的 DexYCB MANO2048 cache；根级 repo memory 所述 Stage4 DexYCB 属于另一数据合同，本轮用户已排除 DexYCB。
- `git diff --check` 对本 Task 草稿与活动记录通过；未启动训练或正式评估。

**回滚**

如需撤销本次工作，仅删除 V1.3 plan 草稿和本活动条目；指导、cache、V1.2 代码/提交与既有运行无需恢复。

## 2026-09-17 03:55:43 +0000 — GRAB train 刚体位移噪声筛查与 V1.3 阈值提案

- activity_id: `ACT-20260917-035543-CMV2-V13-EFFECT-THRESHOLD-DIAGNOSTIC`
- timestamp: `2026-09-17 03:55:43 +0000`
- modification_version: `V1.2.1`（V1.3 plan 仍为待定稿草案）
- type: `diagnostic / documentation`
- task_mode: `read-only/diagnostic`（只更新协商中的 V1.3 plan 草稿）
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户明确要求从刚体位移筛查 `p_effect` 阈值；只读扫描现有 GRAB train pose/点，不修改 cache、代码、配置或训练结果。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a186aff3968ff9dcfe09182182f20a958d1b59f4`
- worktree_dirty: `true`
- scope: 对已完成 GRAB/MANO cache 的 1068 条 train 序列、326722 个相邻帧对计算 pose 推导的物体表面 RMS 位移；用固定 seed 的 train-only 对数分布拟合提出 effect 标签阈值。val/test 未参与拟合，未启动优化器。
- conclusion: `SUPPORTED`（计算可行、阈值估计对分半抽样稳定）；“低运动群等于真实静止”仍 `INCONCLUSIVE`。

**文件与证据入口**

- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json) 与 [run_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — 只读输入。
- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 只在草稿中提出 `0.28 mm` 固定标签阈值和 train-only 估计方法，尚无实施授权。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本次证据与回滚入口。

**原因**

指导要求避免把 `1e-6 m` 的数值非零当作真实效果；用户建议直接扫描刚体位移，因此需从训练集的 pose 运动分布提出可重复、训练前冻结的候选阈值。

**验证**

- 只读脚本对每条序列首帧固定 1024 物体点求均值/协方差，再由相邻 `T_t^-1 T_(t+1)` 精确计算这些点在刚体运动下的 RMS 位移；train 全量 `326722` 对，耗时约 `2.23 s`。中位数 `1.096624 mm`、25% 分位 `0.066373 mm`。
- seed 42 抽 50000 对 `log10(d_rigid)` 拟合：2 分量 BIC `130346.88`，3 分量 BIC `127822.8`；3 分量最低运动群与下一群的后验交点 `0.27561 mm`。按序列随机分半重算交点 `0.2726/0.2811 mm`；草案拟冻结 `0.28 mm`，约 `41.0%` train 配对被标为 no-effect。
- 第一次 sklearn 拟合因本机 OpenBLAS 默认线程数超限而退出（exit 139），没有写入仓库或 cache；以 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` 对同一只读计算重试成功。
- 此阈值来源于无人工静止标签的混合分布，是工程标签定义的提案；不宣称低运动群均为真实静止，也不据此判断模型效果。

**回滚**

仅撤销本条诊断记录与 V1.3 plan 草稿中的阈值提案；源 cache、已提交 V1.2 代码、旧运行及版本指针不变。

## 2026-09-17 04:07:47 +0000 — V1.3 空间融合与直接刚体监督有界验证

- timestamp: `2026-09-17 04:07:47 +0000`
- activity_id: `ACT-20260917-040747-CMV2-V13-IMPLEMENTATION`
- modification_version: `V1.3.1`
- type: `code, data, experiment, diagnostic, documentation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户先以“可以”确认 V1.3 实施，随后明确“先不要加阈值”；最终计划已按后者修订。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a186aff3968ff9dcfe09182182f20a958d1b59f4`
- worktree_dirty: `true`
- scope: 仅 ObjectInteractionCmv2 的 GRAB/MANO V1.3 代码、配置、测试、文档及独立 NAS smoke；DexYCB 和既有 cache、旧运行不变。

**文件**

- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 标记最终版，并删除新增 effect 阈值的标签与闸门。
- [src/task/ObjectInteractionCmv2/model.py](../../model.py) — V1.3 attention pooling、竞争 token、空间 cross-attention、稳定旋转与直接 SE(3) loss；保留 V1.2 模型路径。
- [src/task/ObjectInteractionCmv2/grab.py](../../grab.py) — V1.3 可选 pose GT 与重建 fail-fast，旧 reader 默认保持原合同。
- [src/task/ObjectInteractionCmv2/config.py](../../config.py)、[src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py)、[src/task/ObjectInteractionCmv2/eval_grab.py](../../eval_grab.py) — 新架构校验、checkpoint 隔离和运行接线。
- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_smoke.yaml](../../configs/active/grab_mano_v1_3_smoke.yaml) — 3 序列、8 步、batch 2 的独立配置，无新增 effect 阈值。
- [src/task/ObjectInteractionCmv2/tests/test_v1_3_spatial.py](../../tests/test_v1_3_spatial.py)、[src/task/ObjectInteractionCmv2/tests/test_v1_3_grab_pose.py](../../tests/test_v1_3_grab_pose.py) — token、空接触、梯度与 pose GT 回归。
- [src/task/ObjectInteractionCmv2/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 导航、版本指针与证据。

**原因**

按 V1.3 指导保留 swept hard 2 cm/32 邻居和 GRAB 30 Hz/stride 1，同时让接触位置经 token anchor、normal、mass 参与刚体预测；直接监督当前物体坐标系中的平移与旋转。用户取消新增 `p_effect` 阈值，因此沿用原 `1e-6 m` 数值非零标签；先前 0.28 mm 扫描仅是诊断，不进入训练。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：13 passed；`git diff --check` 通过。
- 全量 cache 只读 pose/point 重建：train 1068 序列/326722 对、val 134/40762、test 133/37445；失配 0；最大误差分别为 `2.566e-7`、`1.756e-7`、`2.852e-7 m`。全量 pose 正交与行列式检查：三个 split 共 406264 帧，失败 0。
- 训练终态：`run_id=cmv2_v13_smoke_final_20260917T040800Z`，`run_status=COMPLETED`，`last_step=8`，`last_epoch=0`，`best_metric=null`；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_smoke_final_20260917T040800Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_smoke_final_20260917T040800Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_smoke_final_20260917T040800Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_smoke_final_20260917T040800Z/train.log)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_smoke_final_20260917T040800Z/latest.pt)。第 8 步 flow EPE `18.04 mm`，token anchor 平均两两距离 `0.258 mm`；空间分离仍需正式训练验证。
- val 终态：`run_id=cmv2_v13_val_final_20260917T040800Z`，`run_status=COMPLETED`，8 对、`last_step=null`、`best_metric=null`；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_val_final_20260917T040800Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_val_final_20260917T040800Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_val_final_20260917T040800Z/metrics.jsonl)，EPE `10.50 mm`。
- test 终态：`run_id=cmv2_v13_test_final_20260917T040800Z`，`run_status=COMPLETED`，8 对、`last_step=null`、`best_metric=null`；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_test_final_20260917T040800Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_test_final_20260917T040800Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_test_final_20260917T040800Z/metrics.jsonl)，EPE `10.82 mm`。
- 开发中早期 smoke `cmv2_v13_smoke_20260917T040258Z`、`cmv2_v13_smoke_20260917T040600Z`、`cmv2_v13_smoke_final_20260917T041000Z` 及各自 val/test 运行均为 `COMPLETED`、`INCONCLUSIVE`，已被最终运行取代；产物保留在独立目录。
- 结论：`INCONCLUSIVE`。上述 8 步训练和 8 对评估仅是工程 smoke，不能支持模型优于基线或 token 已避免 collapse 的科研结论。

**回滚**

以当前 `base_commit` 的 V1.2 代码和 [V1.2 配置](../../configs/active/grab_mano_v1_2_smoke.yaml) 为入口；撤销 V1.3 Task-local 增量与本 Task 版本指针即可。旧 cache、checkpoint 和运行目录没有覆盖。

## 2026-09-17 04:30:48 +0000 — V1.3.2 移除 p_effect 预测分支

- timestamp: `2026-09-17 04:30:48 +0000`
- activity_id: `ACT-20260917-043048-CMV2-V132-NO-EFFECT`
- modification_version: `V1.3.2`
- type: `code, experiment, documentation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确提出“不要预测这个 p_effect 了，网络架构先不加这个”；同一 V1.3 最终计划已按该指示修订。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a186aff3968ff9dcfe09182182f20a958d1b59f4`
- worktree_dirty: `true`
- scope: 仅 V1.3 GRAB/MANO 模型与训练合同；旧 V1.2 模型和历史运行保留。

**文件**

- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 最终计划删除 `p_effect` 输出、参数和 BCE 项，明确新旧 V1.3 checkpoint 隔离。
- [src/task/ObjectInteractionCmv2/model.py](../../model.py) — V1.3 融合层更名为 `fusion_head`，移除 `effect_logit`、概率输出和 effect loss；V1.2 路径不变。
- [src/task/ObjectInteractionCmv2/config.py](../../config.py)、[src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_smoke.yaml](../../configs/active/grab_mano_v1_3_smoke.yaml) — 冻结 `v1_3_rigid_only` 架构标识，拒绝 effect 分支配置。
- [src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py)、[src/task/ObjectInteractionCmv2/eval_grab.py](../../eval_grab.py) — V1.3.2 运行版本、checkpoint 校验和三项损失日志。
- [src/task/ObjectInteractionCmv2/tests/test_v1_3_spatial.py](../../tests/test_v1_3_spatial.py) — 断言无 effect 参数、输出或 loss 项。
- [src/task/ObjectInteractionCmv2/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 导航、Task 指针与本条记录。

**原因**

当前研究阶段只学习结构化刚体位移；用户明确决定暂不建模交互发生概率。先前 `0.28 mm` 只读统计仍仅是历史诊断，不进入 V1.3.2 配置或损失。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：13 passed；`git diff --check` 通过。
- 对照旧 V1.3.1 checkpoint 的 `architecture_version=v1_3`，新 checkpoint 为 `v1_3_rigid_only` 且 state dict 中 effect 参数数为 0；新训练日志无 `effect_loss`。旧 checkpoint 不能按新架构标识恢复。
- 训练：`run_id=cmv2_v13_no_effect_smoke_20260917T043014Z`，`run_status=COMPLETED`，`last_step=8`，`last_epoch=0`，`best_metric=null`；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_smoke_20260917T043014Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_smoke_20260917T043014Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_smoke_20260917T043014Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_smoke_20260917T043014Z/train.log)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_smoke_20260917T043014Z/latest.pt)。第 8 步 flow EPE `18.04 mm`。
- val：`run_id=cmv2_v13_no_effect_val_20260917T043014Z`，`run_status=COMPLETED`，8 对，`last_step=null`，`best_metric=null`；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_val_20260917T043014Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_val_20260917T043014Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_val_20260917T043014Z/metrics.jsonl)，EPE `10.50 mm`。
- test：`run_id=cmv2_v13_no_effect_test_20260917T043014Z`，`run_status=COMPLETED`，8 对，`last_step=null`，`best_metric=null`；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_test_20260917T043014Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_test_20260917T043014Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_test_20260917T043014Z/metrics.jsonl)，EPE `10.82 mm`。
- 结论：`INCONCLUSIVE`；8 步工程 smoke 仅证明新合同可运行，不构成效果结论。

**回滚**

已提交的 V1.2 基线为 `a186aff3968ff9dcfe09182182f20a958d1b59f4`；撤销本 Task 的 V1.3 增量与版本指针可返回该基线。V1.3.1 运行及 checkpoint 留在独立目录，没有被覆盖。

## 2026-09-17 04:36:08 +0000 — GRAB 正式训练只读预检

- timestamp: `2026-09-17 04:36:08 +0000`
- activity_id: `ACT-20260917-043608-CMV2-GRAB-TRAIN-PREFLIGHT`
- modification_version: `V1.3.2`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问是否可以开始 GRAB 训练；本条仅只读核查，不启动正式训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a186aff3968ff9dcfe09182182f20a958d1b59f4`
- worktree_dirty: `true`
- scope: ObjectInteractionCmv2 V1.3.2 GRAB 正式训练准备状态。

**文件与证据入口**

- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 当前最终计划仅批准三序列、八步 smoke；正式训练需单独批准范围、设备、停止条件与评估。
- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_smoke.yaml](../../configs/active/grab_mano_v1_3_smoke.yaml) — 当前唯一 V1.3 活跃配置是有界 smoke。
- [src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py) 与 [src/task/ObjectInteractionCmv2/eval_grab.py](../../eval_grab.py) — 全量训练可通过新配置调用；当前训练仅写 latest checkpoint，未在线验证或选择 best。
- [outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_smoke_20260917T043014Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_no_effect_smoke_20260917T043014Z/run_manifest.json) — V1.3.2 有界运行完成；不等于全数据训练授权。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本条诊断与交接入口。

**原因**

正式 GRAB 训练为 L3 长任务。需要先冻结全量 train 范围、batch/step、checkpoint、验证范围、设备和停止条件，并使运行 manifest 的 base_commit 可指向已提交的 V1.3.2 代码。

**验证**

- 输入 cache 已有 train 1068 序列/326722 有效相邻对、val 134/40762、test 133/37445；若 batch=16，单次覆盖 train 需 `ceil(326722/16)=20421` optimizer steps。
- GPU1 当前显存占用 0 MiB，NAS `outputs/ObjectInteractionCmv2` 所在卷可用约 52 TiB；资源状态会在实际启动前复核。
- V1.3.2 八步 smoke `run_status=COMPLETED`；当前 Task 代码仍未提交，HEAD 仅为 V1.2 基线 `a186aff`。没有启动正式训练，也没有正式训练 `run_id`。
- 结论：`INCONCLUSIVE`（仅具备初步运行条件，尚未验证 batch=16 吞吐、全量训练效果或选择最佳 checkpoint）。

**回滚**

本次只读查询仅新增本条活动记录；无需撤销数据或运行产物。

## 2026-09-17 04:40:52 +0000 — V1.3.3 GRAB 正式训练配置与终止机制

- timestamp: `2026-09-17 04:40:52 +0000`
- activity_id: `ACT-20260917-044052-CMV2-V133-FORMAL-PREP`
- modification_version: `V1.3.3`
- type: `code, experiment, documentation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户对单次全量 GRAB train、batch 16、GPU1、四小时上限、终态完整 val 及启动前定向提交的具体方案回复“可以”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a186aff3968ff9dcfe09182182f20a958d1b59f4`
- worktree_dirty: `true`
- scope: 仅 ObjectInteractionCmv2 的 V1.3 模型/读取器/训练评估入口、校准与正式配置及测试；不含其他 Task、用户指导文件和运行产物。

**文件**

- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 增补已批准的 §7 训练预算、停止和评估范围。
- [src/task/ObjectInteractionCmv2/model.py](../../model.py)、[src/task/ObjectInteractionCmv2/grab.py](../../grab.py)、[src/task/ObjectInteractionCmv2/config.py](../../config.py) — V1.3.2 刚体模型和 GT 读取实现及 V1.3.3 配置闸门。
- [src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py)、[src/task/ObjectInteractionCmv2/eval_grab.py](../../eval_grab.py) — 从配置记录版本、检查全量 split、四小时优雅停止并保存终态 checkpoint。
- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_smoke.yaml](../../configs/active/grab_mano_v1_3_smoke.yaml)、[src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_calibration.yaml](../../configs/active/grab_mano_v1_3_calibration.yaml)、[src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_formal.yaml](../../configs/active/grab_mano_v1_3_formal.yaml) — 分离既有 smoke、batch16 校准和全量正式配置。
- [src/task/ObjectInteractionCmv2/tests/test_v1_3_spatial.py](../../tests/test_v1_3_spatial.py)、[src/task/ObjectInteractionCmv2/tests/test_v1_3_grab_pose.py](../../tests/test_v1_3_grab_pose.py) — V1.3 合同回归。
- [src/task/ObjectInteractionCmv2/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 导航、版本指针与记录；这些含先前未提交内容的文件本次不整体暂存。

**原因**

当前训练入口只有八步 smoke 配置。单轮全量训练需冻结 326722 对/20421 步合同，并在四小时预算耗尽时写出可恢复的 checkpoint 与 `STOPPED` 终态。用户已有其他未提交改动，故提交仅选当前 Task 的代码、配置、测试和本计划。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：13 passed。
- 新配置解析：calibration `V1.3.3/3 序列/8 步`，formal `V1.3.3/全序列/20421 步`；`py_compile`、`git diff --check` 通过。
- 尚未启动校准或正式训练；运行状态将另记 activity。结论为 `INCONCLUSIVE`。

**回滚**

本次改动提交后以该提交为 GRAB 训练代码锚点；返回既有 V1.2 基线可从 `a186aff3968ff9dcfe09182182f20a958d1b59f4` 重新检出。其他 Task 工作区与历史输出不动。

## 2026-09-17 04:42:36 +0000 — batch16 校准完成并启动全量 GRAB 训练

- timestamp: `2026-09-17 04:42:36 +0000`
- activity_id: `ACT-20260917-044236-CMV2-V133-GRAB-RUNNING`
- modification_version: `V1.3.3`
- type: `experiment, operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户批准 [V1.3 最终计划 §7](../plan/V1.3.md) 的校准、全量训练与 val 范围。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a927ab19035194aac8165e39b3a55ccb27612812`
- worktree_dirty: `true`（其他 Task、导航和活动记录仍有未提交改动；运行代码已定向提交）
- scope: GRAB MANO V1.3.3 资源校准与全量 train；本阶段不读取 test。

**命令与状态**

- 校准命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .../graspenv/bin/python -m src.task.ObjectInteractionCmv2.train_grab --config src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_calibration.yaml --run-id cmv2_v13_grab_b16_cal_20260917T044133Z`。
- 校准：`run_id=cmv2_v13_grab_b16_cal_20260917T044133Z`，`run_status=COMPLETED`，`last_step=8`，`last_epoch=0`，`best_metric=null`，latest checkpoint 存在；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_b16_cal_20260917T044133Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_b16_cal_20260917T044133Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_b16_cal_20260917T044133Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_b16_cal_20260917T044133Z/train.log)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_b16_cal_20260917T044133Z/latest.pt)。8 步均 finite，计算单步 `0.045–0.197 s`，GPU 峰值 `1.89 GiB`。
- 正式命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .../graspenv/bin/python -m src.task.ObjectInteractionCmv2.train_grab --config src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_formal.yaml --run-id cmv2_v13_grab_full_20260917T044156Z`。
- 正式训练：`run_id=cmv2_v13_grab_full_20260917T044156Z`，`run_status=RUNNING`；已核对 `valid_pairs=326722`、`dropped_pairs=0`，随机初始化且 `initial_checkpoint=null`。[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/train.log)。首个 checkpoint 在 step 1000，当前 `latest.pt` PENDING。

**原因**

用户已批准在完成资源校准后从随机初始化运行一轮全量 GRAB train；先提交代码，保证运行 manifest 的 `base_commit` 可追溯。

**验证**

- 提交 `a927ab19035194aac8165e39b3a55ccb27612812` 仅包含本 Task 的 11 个代码/配置/测试/计划文件，暂存差异与链接审计通过；其他 Task 工作区未带入提交。
- 正式运行截至记录时已写出 step 183，loss finite；当前仅为进度证据，科研结论 `INCONCLUSIVE`。

**回滚**

运行代码锚点为 `a927ab19035194aac8165e39b3a55ccb27612812`。正式运行使用独立 NAS 目录；不会覆盖校准、V1.3.2 smoke 或输入 cache。

## 2026-09-17 05:20:08 +0000 — 按用户要求停止 GRAB 训练并诊断曲线

- timestamp: `2026-09-17 05:20:08 +0000`
- activity_id: `ACT-20260917-052008-CMV2-V133-STOP-AND-DIAGNOSE`
- modification_version: `V1.3.3`
- type: `operation, diagnostic`
- change_level: `L0`
- approval: `user-approved`
- approval_basis: 用户明确指示“先到此为止”，并询问 loss 与指标；已终止正在运行的进程，不启动 val。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a927ab19035194aac8165e39b3a55ccb27612812`
- worktree_dirty: `true`
- scope: GRAB 正式训练终止与已有训练日志的只读统计；未改训练代码、输入 cache、val/test。

**命令与状态**

- 原训练命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .../graspenv/bin/python -m src.task.ObjectInteractionCmv2.train_grab --config src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_formal.yaml --run-id cmv2_v13_grab_full_20260917T044156Z`。
- `run_id=cmv2_v13_grab_full_20260917T044156Z`；`run_status=STOPPED`；`stop_reason=user_requested_stop`；`last_step=16644`，`last_epoch=0`，`best_metric=null`，最近可恢复 checkpoint 为 step 16000。进程已退出；step 16001–16644 只见于日志，未保存为 checkpoint。
- [运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/train.log)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/latest.pt)。

**原因**

用户要求此刻停止并检查 loss 是否仍下降。现有训练入口没有信号处理器；发送 SIGINT 后进程退出，再根据实际 metrics 尾行和 checkpoint 内 step，将运行 manifest 原子更新为 `STOPPED`，保留停止原因与可恢复步数。

**验证**

- 已核对训练进程退出，metrics 共 16644 行、step 连续到 16644，全部检查的 loss/EPE 为 finite；checkpoint 内 `step=16000`、`position=256000`，架构标识为 `v1_3_rigid_only`。
- 前 1000 步平均 total loss `0.118314`、训练 flow EPE `4.9671 mm`；最后 1000 个已记录 step 分别为 `0.089886`、`2.8460 mm`。但 step 10001–13000 与 13001–16000 的平均 loss 为 `0.090496` 与 `0.090440`，训练 EPE 为 `3.0594` 与 `3.0422 mm`，显示后段近乎平台，不能称为持续稳定下降。
- 最后 1000 步平均旋转项 `0.076076`，约占 total loss 的大部分；平移项 `0.005593`，点流项 `0.008217`。这些均为训练数据上的归一化损失，EPE 单位为 mm。
- 本轮未运行完整 val 或 test，`conclusion=INCONCLUSIVE`；训练曲线不能证明泛化或模型已优于零流基线。

**回滚**

没有改动已提交代码；保留独立运行目录和最近 checkpoint 以供后续明确授权的恢复或评估。无需删除历史产物。

## 2026-09-17 05:28:51 +0000 — 双卡大规模 GRAB 训练只读预检

- timestamp: `2026-09-17 05:28:51 +0000`
- activity_id: `ACT-20260917-052851-CMV2-DUAL-GPU-PREFLIGHT`
- modification_version: `V1.3.3`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户提出 GPU2/3、增大 batch 至半显存并询问歧义；本条仅核查现状，未启动新运行或修改训练语义。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a927ab19035194aac8165e39b3a55ccb27612812`
- worktree_dirty: `true`
- scope: ObjectInteractionCmv2 V1.3 GRAB 双卡训练可行性。

**文件与证据入口**

- [src/task/ObjectInteractionCmv2/train_grab.py](../../train_grab.py) — 当前单进程/单设备训练，没有 DDP 或双卡 sampler。
- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_formal.yaml](../../configs/active/grab_mano_v1_3_formal.yaml) 与 [V1.3 最终计划](../plan/V1.3.md) — 已批准范围为 GPU1、batch16、单轮训练；新双卡预算尚未定稿。
- [outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/run_manifest.json) 和 [latest.pt](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/latest.pt) — 已停止运行，最近可恢复 checkpoint 为 step 16000。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本条预检入口。

**原因**

双卡和目标半显存会改变训练实现与全局 batch；是否从 step 16000 续训、总训练轮数会改变数据曝光与 checkpoint 解释，必须先冻结在最终计划中。

**验证**

- GPU2、GPU3 各约 49140 MiB，总使用量当前 0 MiB；目标一半约 24570 MiB/卡，只能以短校准测量实际显存后选择 batch。
- 现有 batch16 单卡校准峰值约 1.89 GiB，但不能据此外推双卡大 batch 的真实峰值。
- 当前代码无双卡 DDP 路径；本次没有启动训练、评估或显存校准。结论 `INCONCLUSIVE`。

**回滚**

仅新增只读预检记录，无模型、配置、数据或运行产物变更。

## 2026-09-17 05:33:19 +0000 — step 16000 的数据轮次与耗时核算

- timestamp: `2026-09-17 05:33:19 +0000`
- activity_id: `ACT-20260917-053319-CMV2-STEP16000-AUDIT`
- modification_version: `V1.3.3`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问现有 checkpoint 到底训练了多少轮、耗时多久；只读核算。
- skills_used: `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a927ab19035194aac8165e39b3a55ccb27612812`
- worktree_dirty: `true`
- scope: GRAB V1.3.3 已停止运行的 checkpoint 轮次和墙钟耗时。

**文件与证据入口**

- [outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/metrics.jsonl)、[latest.pt](../../../../../outputs/ObjectInteractionCmv2/cmv2_v13_grab_full_20260917T044156Z/latest.pt) — 原始状态、逐步记录和 checkpoint。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本条诊断。

**原因**

区分 optimizer step 与完整数据轮次，并以运行创建时间及 checkpoint 文件实际保存时间计算墙钟耗时。

**验证**

- batch 16，train 有 326722 对；step 16000 对应 `16000×16=256000` 对，约 `0.78354` 个 epoch，完整一轮需 `ceil(326722/16)=20421` step。因此该 checkpoint 仍在第 1 轮，完整轮数为 0。
- 运行 manifest 创建于 `2026-09-17T04:42:03+00:00`；step 16000 checkpoint 修改时间约 `05:17:58+00:00`，墙钟历时 `2155.7 s≈35 分 56 秒`。逐步计时字段只覆盖计算段，不能代替墙钟耗时。
- 结论：`SUPPORTED`（轮次与耗时核算）；不涉及模型效果结论。

**回滚**

仅新增只读诊断记录，无训练进程或数据修改。

## 2026-09-17 05:42:24 +0000 — V1.3.4 双卡 GRAB 训练入口与显存校准配置

- timestamp: `2026-09-17 05:42:24 +0000`
- activity_id: `ACT-20260917-054224-CMV2-V134-DDP-IMPLEMENTATION`
- modification_version: `V1.3.4`
- type: `code, experiment, documentation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户指定 GPU2/3、约半显存 batch，并明确四轮从新初始化训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `a927ab19035194aac8165e39b3a55ccb27612812`
- worktree_dirty: `true`
- scope: ObjectInteractionCmv2 V1.3 Task-local DDP 训练、显存校准与合同测试；DexYCB、旧运行和其他 Task 不变。

**文件**

- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 已批准 §8，冻结双卡、四轮、从头训练与显存校准规则。
- [src/task/ObjectInteractionCmv2/config.py](../../config.py) — V1.3.4 DDP 配置闸门。
- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_ddp_calibration.yaml](../../configs/active/grab_mano_v1_3_ddp_calibration.yaml) — GPU2/3、batch 候选与 24 GiB 峰值上限。
- [src/task/ObjectInteractionCmv2/train_grab_ddp.py](../../train_grab_ddp.py) — 两卡 NCCL 训练、无重复分片、显存校准、checkpoint 和运行 manifest。
- [src/task/ObjectInteractionCmv2/tests/test_v1_3_ddp_contract.py](../../tests/test_v1_3_ddp_contract.py) — 分片不重复与物理 GPU 配置回归。
- [src/task/ObjectInteractionCmv2/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 导航、版本指针和活动记录；含已有未提交内容，暂不整体暂存。

**原因**

旧入口只支持单 GPU、batch16，不能直接满足两卡或显存目标。新入口按轮次拆分每个 GRAB train 配对，不补齐采样；先测真实峰值再冻结 per-GPU batch。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/`：15 passed。
- `py_compile`、`git diff --check` 通过；GPU2/3 各约 48 GiB，校准/正式运行尚未启动。
- 结论 `INCONCLUSIVE`；下一步以真实两卡校准验证 NCCL、峰值显存和配置选择。

**回滚**

本次新入口与旧单卡入口分离；撤销 V1.3.4 Task-local 增量可回到已提交的 `a927ab19035194aac8165e39b3a55ccb27612812`。其他 Task 工作区、旧 checkpoint 和输入 cache 不动。

## 2026-09-17 07:05:42 +0000 — 双卡显存校准终态与正式 batch 冻结

- timestamp: `2026-09-17 07:05:42 +0000`
- activity_id: `ACT-20260917-070542-CMV2-V134-BATCH-CALIBRATED`
- modification_version: `V1.3.4`
- type: `experiment, code, documentation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户批准 GPU2/3、约半显存 batch、随机初始化四轮；batch 根据已批准计划 §8 的峰值 reserved 上限选择。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `42341bdc1fa371c800e75b62a9c293ec7c32ab3a`
- worktree_dirty: `true`
- scope: GRAB 双卡显存校准与 V1.3.4 正式配置；校准权重未用于正式训练。

**文件与证据入口**

- [src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_ddp_formal.yaml](../../configs/active/grab_mano_v1_3_ddp_formal.yaml) — 冻结每卡 batch160、全局 batch320、四轮与 GPU2/3。
- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md)、[src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_ddp_calibration.yaml](../../configs/active/grab_mano_v1_3_ddp_calibration.yaml) — 规则和校准输入。
- 校准 `run_id=cmv2_v134_ddp_cal_20260917T054306Z`，`run_status=COMPLETED`，`last_step=6` 个候选、`last_epoch=null`、`best_metric=null`，无 checkpoint；[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_ddp_cal_20260917T054306Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_ddp_cal_20260917T054306Z/run_manifest.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_ddp_cal_20260917T054306Z/metrics.jsonl)。
- [src/task/ObjectInteractionCmv2/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 导航、指针与活动；含先前未提交内容，本次不整体暂存。

**原因**

用户要求按两卡约半显存设置较大 batch；显存需在真实张量的两卡 optimizer step 上测量，不能单凭 batch16 线性推算。仅提交当前 Task 的正式配置，使运行的代码和 batch 可追溯。

**验证**

- GPU2/3 batch160 的峰值 allocated 均 `18.60 GiB`、reserved 均 `21.72 GiB`；下一候选 batch192 的 reserved 均 `26.06 GiB`，超过 24 GiB 上限。因此按既定规则选 `160`；距离 22 GiB 目标下沿约 `0.28 GiB`。
- 解析正式配置为 `V1.3.4`、两卡、每卡 batch160、四轮；每轮 `ceil(163361/160)=1022` optimizer steps，四轮共 `4088` 步。
- 结论 `SUPPORTED`（显存选择），模型效果仍 `INCONCLUSIVE`。

**回滚**

校准输出和新正式配置独立，旧 checkpoint 与输入 cache 未覆盖；代码锚点为 `42341bdc1fa371c800e75b62a9c293ec7c32ab3a`。

## 2026-09-17 07:07:32 +0000 — GPU2/3 四轮 GRAB 训练启动

- timestamp: `2026-09-17 07:07:32 +0000`
- activity_id: `ACT-20260917-070732-CMV2-V134-DDP-START`
- modification_version: `V1.3.4`
- type: `experiment, operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户指定 GPU2/3、半显存目标、四轮、随机初始化；校准按最终计划 §8 冻结 batch160。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `0cd030831307ea1d97223a00e69f37129acdae44`
- worktree_dirty: `true`（其他 Task 与导航记录仍有未提交内容，运行代码/配置已定向提交）
- scope: GRAB MANO train 全量、DDP GPU2/3、从新初始化四轮；本阶段不读 val/test。

**命令与状态**

- `CUDA_VISIBLE_DEVICES=2,3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .../graspenv/bin/torchrun --standalone --nproc_per_node=2 -m src.task.ObjectInteractionCmv2.train_grab_ddp --config src/task/ObjectInteractionCmv2/configs/active/grab_mano_v1_3_ddp_formal.yaml --run-id cmv2_v134_grab_ddp4_20260917T070638Z`，输入 cache/output root 环境变量见解析配置快照。
- `run_id=cmv2_v134_grab_ddp4_20260917T070638Z`，`run_status=RUNNING`，`initial_checkpoint=null`，计划 `4×1022=4088` 步；已核对 `valid_pairs=326722`、`dropped_pairs=0`。[运行目录](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z)、[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/run_manifest.json)、[config.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/config.json)、[metrics.jsonl](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/metrics.jsonl)、[train.log](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/train.log)。首个 `latest.pt` 在 step 200，当前 PENDING。

**原因**

按用户确认的 V1.3.4 计划从随机初始化运行四轮，使用两张指定 GPU 和已校准的每卡 batch160。

**验证**

- 启动后 manifest 显示 base commit `0cd0308`、两卡、全局 batch320；step17 的显存峰值 reserved 约 `21.71 GiB/卡`，与校准一致；loss finite。训练结果尚未完成，`conclusion=INCONCLUSIVE`。

**回滚**

独立 NAS 运行目录；不覆盖旧单卡 run、旧 checkpoint 或源 cache。代码与配置入口是 `0cd030831307ea1d97223a00e69f37129acdae44`。

## 2026-09-17 07:32:51 +0000 — 归档提交 Cmv2 历史导出工具与研究文档

- timestamp: `2026-09-17 07:32:51 +0000`
- activity_id: `ACT-20260917-073251-CMV2-ARCHIVE-COMMIT`
- modification_version: `V1.3.4`
- type: `operation, documentation`
- change_level: `L0`（仅整理和提交既有工作区内容；原实现影响等级见各历史活动）
- approval: `user-approved`
- approval_basis: 用户明确要求将其他代码分门别类提交。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `337b054`
- worktree_dirty: `true`
- scope: 归档本 Task 既有 GRAB→DExplore 数据导出工具、指导、计划草稿/最终计划、拟议配置和文档导航；不执行草稿计划或改变正在运行的四轮训练。

**文件**

- `src/task/ObjectInteractionCmv2/` — 本次提交的 Task 内数据工具和文档；逐项路径以提交 diff 为准。
- [src/task/ObjectInteractionCmv2/docs/plan/V1.3.md](../plan/V1.3.md) — 当前最终计划；V1.1 草稿仍保留草稿状态。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 历史操作与训练状态入口。
- [outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/run_manifest.json) — 正在运行的训练清单；未纳入 Git。

**原因**

按用户要求将先前已实现但尚未入库的 Task-local 工具与计划文档独立归档；训练继续使用启动时的 `0cd0308` 代码锚点。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py src/task/ObjectInteractionCmv2/tools/data/run_grab_dexplore_rl.py`：通过。
- Task 测试与暂存差异链接审计在提交前核对；本次提交不产生新的模型效果结论，仍为 `INCONCLUSIVE`。

**回滚**

本次提交作为独立 Git commit，可按提交范围反向应用；正在运行的独立训练目录、输入 cache 与旧 checkpoint 不变。

## 2026-09-17 12:41:44 +0000 — Cmv2 与三域 OICm 训练入口范围核对

- timestamp: `2026-09-17 12:41:44 +0000`
- activity_id: `ACT-20260917-124144-CMV2-OICM-SCOPE-CHECK`
- modification_version: `V1.3.4`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `user-requested`
- approval_basis: 用户询问“目前 OICmv2 可以开始全量训练了吗、接口是否好了”，并要求继续补全 OakInk2 失败帧。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `7afcb4605988e04f44326bb0395a886a78c9be80`
- worktree_dirty: `false`
- scope: 只读区分 `ObjectInteractionCmv2` 与新三域 `ObjectInteractionCm` 的训练接口和运行状态；未启动训练、未修改代码、cache 或 checkpoint。
- conclusion: `INCONCLUSIVE`（任务名存在范围歧义；三域正式训练仍需先完成 OakInk2 回填）。

**原因**

`ObjectInteractionCmv2` 当前 V1.3 接口在配置层固定为 GRAB/MANO、stride=1；新三域等概率接口属于独立的 `ObjectInteractionCm` Task。两者不能按同一个全量训练入口解释。

**验证**

- [src/task/ObjectInteractionCmv2/config.py](../../config.py) 强制 `source.name=grab`、`hand_source=mano`、`stride=1`，没有 ARCTIC/OakInk2 三域 sampler。
- `run_id=cmv2_v134_grab_ddp4_20260917T070638Z` 的 manifest 为 `COMPLETED`、`last_step=4088`、`last_epoch=4`；这是 GRAB-only DDP 训练，未运行完整 val/test，效果结论仍为 `INCONCLUSIVE`。[run_manifest.json](../../../../../outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/run_manifest.json)
- 新三域 `ObjectInteractionCm` 的 loader/等概率接口和 2-step smoke 已通过，但 OakInk2 全量 cache 仍因 `193` 条 partial frame coverage 失败；补全前不能开始三域正式训练。

**待确认边界**

用户所说的“ OICmv2 ”需要在以下两者中确认其一：

1. `ObjectInteractionCmv2`：继续 GRAB-only V1.3；其四轮全量 GRAB DDP 已完成。
2. `ObjectInteractionCm`：继续 GRAB/ARCTIC/OakInk2 三域；下一步读取 OakInk2 原始 annotation/object cache，补齐失败 segment 后再启动正式训练。

**保护与回滚**

本次仅新增诊断记录；删除本条 activity 即可回滚。训练、cache、checkpoint 和现有接口均未改变。

## 2026-09-17 13:40:05 +0000 — V1.4 三域高分辨率接口与回填器实现

- timestamp: `2026-09-17 13:40:05 +0000`
- activity_id: `ACT-20260917-134005-CMV2-V142-THREE-DOMAIN-IMPLEMENTATION`
- modification_version: `V1.4.2`
- type: `code, data, documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户于 2026-09-17 明确确认 Cmv2 V1.4 三域范围，并明确 MANO 每侧 2048、Inspire 每侧 10135 的训练手点合同。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `a7449d716b0185514e77a1fa698cccd9fc09de2c`
- worktree_dirty: `true`
- scope: 在 Cmv2 新增三域高分辨率 loader、等概率 sampler、变长 batch、2-step smoke 入口与 OakInk2 原始 annotation 回填器；不改 V1.3 GRAB 入口、`src/base`、旧 ObjectInteractionCm、已有 cache/输出/checkpoint。

**文件**

- [V1.4 最终计划](../plan/V1.4.md) — 已批准的范围、点数、2 cm、KNN32、split 与 smoke 闸门。
- [V1.4 指导](../指导/V1.4.md)、[src/task/ObjectInteractionCmv2/docs/README.md](../README.md) 与 [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 用户确认的范围、Task 导航和 `V1.4.2` 版本指针。
- [src/task/ObjectInteractionCmv2/config.py](../../config.py) — V1.4 三域配置和不变量验证。
- [src/task/ObjectInteractionCmv2/multi_domain.py](../../multi_domain.py) — MANO 4096 / Inspire 20270 高分辨率流、直接 SE(3) GT、stride、padding 与 sampler。
- [src/task/ObjectInteractionCmv2/train_multi_domain.py](../../train_multi_domain.py) 与 [src/task/ObjectInteractionCmv2/configs/active/three_domain_v1_4_smoke.yaml](../../configs/active/three_domain_v1_4_smoke.yaml) — 独立 smoke 入口。
- [src/task/ObjectInteractionCmv2/tools/data/backfill_oakink2_inspire_v1_4.py](../../tools/data/backfill_oakink2_inspire_v1_4.py) — 从原始 annotation pose 和既有 Stage3 静态几何重建独立 OakInk2 高分辨率 cache。
- [src/task/ObjectInteractionCmv2/tests/test_v1_4_three_domain.py](../../tests/test_v1_4_three_domain.py) — variant、padding、stride、GT 与严格域均衡回归。

**原因**

此前 OakInk2 3076 点 export 和 1656 个旧 complete 目录都不符合 V1.4 高分辨率训练合同，不能复用为训练输入。新 loader 强制将 3076 限为 decoder 兼容流；高分辨率 KNN 流分别严格为双手 MANO 4096 或 Inspire 20270，旧 cache 仅保留审计用途。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/config.py src/task/ObjectInteractionCmv2/multi_domain.py src/task/ObjectInteractionCmv2/train_multi_domain.py src/task/ObjectInteractionCmv2/tools/data/backfill_oakink2_inspire_v1_4.py`：通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_4_three_domain.py`：`4 passed`。
- 只读实证核对：GRAB/ARCTIC 真实 `knn_hand_points_world.npy` 为双手 4096；旧 OakInk2 pilot 为 3076，故被 loader 拒绝作为 `inspire_f1` 训练流。高分辨率 OakInk2 raw-frame smoke 尚未启动，结论为 `INCONCLUSIVE`。

**回滚**

本次仅为 Task-local V1.4 增量；回滚入口是本次实现提交的父提交与 [V1.4 最终计划](../plan/V1.4.md) §6。不会删除或覆盖旧 OakInk2 full、partial 或 pilot 目录。

## 2026-09-17 13:43:00 +0000 — OakInk2 高分辨率原始帧回填全量启动

- timestamp: `2026-09-17 13:43:00 +0000`
- activity_id: `ACT-20260917-134300-CMV2-V142-OAKINK2-HIGHRES-START`
- modification_version: `V1.4.2`
- type: `data, operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确要求从原始 OakInk2 数据补齐失败帧，并已确认 V1.4 三域范围、高分辨率手点合同和全量 cache 前置步骤。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `3243a8ed7eb053fe5ee47b7a052ff80c1d5a7ff1`
- worktree_dirty: `false`
- run_id: `cmv2_highres_full_20260917T134300Z`
- run_status: `STARTED`
- scope: 在 GPU2 从原始 OakInk2 annotation 的 `obj_transf` 和 Stage3 静态几何重建 1849 个 selected segment 的 Inspire 20270 KNN cache；V1.3、旧 Task 及旧 OakInk2 3076 full/partial/pilot 输出保持不变。
- conclusion: `INCONCLUSIVE`（运行中；仅此前 1-segment 高分辨率 smoke 已证明工程链路）。

**命令与状态**

- 命令：`PYTHONPATH=/home/wbcd/workspace/oyx_ws/Ref2Dex /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCmv2.tools.data.backfill_oakink2_inspire_v1_4 --selection-index …/oakink2_inspire_selection_v1_4_23/full_20260917T091507Z/index.json --annotation-root …/OakInk-v2/anno_preview --stage3-root …/stage3/oakink2_object_centered_v1 --existing-root …/full_20260917T091507Z --output-root …/cmv2_highres_full_20260917T134300Z --mano-root …/mano --dex-root …/dex-retargeting --device cuda:2 --mano-batch-size 128 --knn-frame-batch 2 --knn-object-chunk 512 --run-id cmv2_highres_full_20260917T134300Z`。
- [高分辨率运行目录](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z)、[run_manifest.json](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest.json)、[实时日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/cmv2_highres_full_20260917T134300Z.log)。
- 启动状态：`expected_segments=1849`、`reused_segments=0`、`backfilled_segments=0`、`completed_frames=0`、`failures=[]`；预计只有全量成功时才会生成正式 `index.json` 和 `cache_manifest.json`，当前 PENDING。

**原因**

先前 full export 的 1656 个完成段及 193 个 partial 段均为 3076 高分辨率错误合同或失败残留，不能拼接为本轮训练输入。因此新根目录从头重建所有 1849 段；输入选择 index、原始数据和旧输出全部只读保留。

**验证**

- `run_id=cmv2_highres_backfill_smoke_20260917T134100Z` 的失败段 smoke 已 `COMPLETED`：637 帧、`failures=[]`、KNN hand shape `[637,20270,3]`、decoder compatibility shape `[637,3076,3]`、KNN32 最大 index `20268`；Cmv2 `InspireSequenceView(..., hand_variant="inspire_f1")` 实际读取为 20270 点。
- NAS 可用空间约 52 TB；该 637 帧 high-resolution smoke 占约 567 MB。GPU2 启动后进程存活，当前正式运行尚未生成效果结论。

**回滚**

停止该独立 PID/运行即可；新运行根与旧 full/partial/pilot 根隔离，不覆盖原始 annotation、Stage3、旧 cache、训练输出或 checkpoint。终态将以本 activity 为唯一状态入口补充。

## 2026-09-17 13:54:26 +0000 — 部分 OakInk2 高分辨率三域 smoke 终态

- timestamp: `2026-09-17 13:54:26 +0000`
- activity_id: `ACT-20260917-135426-CMV2-V142-PARTIAL-THREE-DOMAIN-SMOKE`
- modification_version: `V1.4.2`
- type: `experiment, operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确要求“拿一部分来试试看”；范围限定为三域各一条 sequence、固定 seed、2 optimizer steps。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `196ade2d32e1047c6ecab32b4debaa31b129cac7`
- worktree_dirty: `false`
- run_id: `cmv2_v142_three_domain_partial_highres_smoke_20260917T135300Z`
- run_status: `COMPLETED`
- scope: 只读使用 GRAB/ARCTIC MANO 4096 cache 和全量回填中已完成的一条 OakInk2 Inspire 20270 cache，验证三域 loader、变长 collate、V1.3 刚体模型前反向与 loss；不修改正式 cache、split、模型、V1.3 或旧 Task。
- conclusion: `SUPPORTED`（工程 smoke）；科研效果结论 `INCONCLUSIVE`。

**命令与状态**

- 以 [V1.4 smoke 配置](../../configs/active/three_domain_v1_4_smoke.yaml) 解析同一 2-step 合同，因四张 GPU 均有外部任务或 OakInk2 回填占用，运行设备改为 CPU；这不验证 GPU 性能。
- [临时运行目录](../../../../../../../../../../tmp/ref2dex-cmv2-three-domain-smoke.7VugWN/cmv2_v142_three_domain_partial_highres_smoke_20260917T135300Z)、[run_manifest.json](../../../../../../../../../../tmp/ref2dex-cmv2-three-domain-smoke.7VugWN/cmv2_v142_three_domain_partial_highres_smoke_20260917T135300Z/run_manifest.json)、[metrics.jsonl](../../../../../../../../../../tmp/ref2dex-cmv2-three-domain-smoke.7VugWN/cmv2_v142_three_domain_partial_highres_smoke_20260917T135300Z/metrics.jsonl)、[train.log](../../../../../../../../../../tmp/ref2dex-cmv2-three-domain-smoke.7VugWN/cmv2_v142_three_domain_partial_highres_smoke_20260917T135300Z/train.log) 与 [latest.pt](../../../../../../../../../../tmp/ref2dex-cmv2-three-domain-smoke.7VugWN/cmv2_v142_three_domain_partial_highres_smoke_20260917T135300Z/latest.pt)。
- runner 终态：`last_step=2`、`last_epoch=1`、`best_metric=null`、`source_counts={grab: 4}`；replacement sampler 的两个随机 batch 恰好均抽到 GRAB，因此这些 step loss 不能作为三域效果样本。

**原因**

正式 OakInk2 20270 cache 尚在运行，用户要求先用已完成的部分验证接口。为了不抢占 GPU2 或干扰其他 GPU 任务，采用 CPU 运行。NAS canonical `outputs/ObjectInteractionCmv2` 对当前用户没有写权限，两个启动尝试均在创建运行目录前退出，故使用本机临时目录保存本次小型 smoke；没有写入或覆盖 NAS 正式 output/cache。

**验证**

- dataset 已读取 `grab=219`、`arctic=593`、`oakink2=1236` 个有效 transition，合计 2048；variant 行数为 `mano=812`、`inspire_f1=1236`。
- 两个 runner step 的 loss 均 finite：`1.8631`、`1.4577`；checkpoint 记录 `v1_3_rigid_only`、`step=2`。
- 另行构造一个必含 `grab/arctic/oakink2` 的真实 batch 并执行前向、loss、反向：hand batch shape `[3,20270,3]`，valid hand counts `[4096,4096,20270]`，loss `4.4040` finite。这直接覆盖了 MANO/Inspire 混合 padding 的模型路径。
- 该结果只证明数据/模型接线；不验证泛化、跨域效果或 GPU 吞吐。全量 OakInk2 回填仍 `STARTED`，当时 `30/1849` 段、`16567` 帧、`failures=[]`。

**回滚**

本次为独立本机临时 smoke；移除其临时目录即可回收小型 checkpoint/日志，不影响 NAS cache。正式训练仍须等待高分辨率 OakInk2 cache 完成并另行确定 GPU、预算、验证指标。

## 2026-09-17 13:57:02 +0000 — NAS Cmv2 output 写入状态更正

- timestamp: `2026-09-17 13:57:02 +0000`
- activity_id: `ACT-20260917-135702-CMV2-OUTPUT-WRITE-STATUS-CORRECTION`
- modification_version: `V1.4.2`
- type: `diagnostic, documentation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户追问此前为什么没有写权限；本条只读复核权限、挂载和历史输出归属。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `c480f6afa2cb142788a22bbdda67ec849f9d45cf`
- worktree_dirty: `false`
- scope: 更正部分三域 smoke 记录中关于 NAS canonical output 不可写的未证实归因；不修改 NAS 权限、运行、cache 或 checkpoint。
- conclusion: `SUPPORTED`（权限状态复核）；与模型科研结论无关。

**原因**

此前两个后台 smoke 启动均未留下输出目录或 launcher log，但没有保留可证明 `permission denied` 的 stderr；不能据此断定 ACL 拒绝写入。

**验证**

- `/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2` 的表面归属为 `lsj:uucp`、模式 `775`，本机身份为 `wbcd`（UID 1006），且当前 group 不含 `uucp`；单看 POSIX mode 会推断无写权限。
- 实际 NFSv3 挂载为 `rw,sec=sys`，Shell `test -w` 对该目录及相邻 output/data run root 均返回 writable，说明服务端实际授权或映射与本机 group 列表不同。
- 系统未安装 `getfacl`，且失败 launcher 没有日志，故不能追溯其确切退出原因。此前“没有写权限”的说法撤回；后续正式运行应先保留一次创建结果/错误，再做归因。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](activity_log.md) — 本次更正与原 smoke 状态的唯一入口。

**回滚**

本条仅追加诊断更正；不涉及权限变更或数据写入。临时 smoke 输出仍按上一条记录保留，不能据此推断正式 NAS output 不可用。

## 2026-09-17 14:03:02 +0000 — OakInk2 高分辨率回填吞吐瓶颈诊断

- timestamp: `2026-09-17 14:03:02 +0000`
- activity_id: `ACT-20260917-140302-CMV2-HIGHRES-BACKFILL-BOTTLENECK`
- modification_version: `V1.4.2`
- type: `diagnostic, operation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问全量 cache 导出耗时原因；只读检查运行进程、GPU、I/O 计数和 exporter hot path。
- skills_used: `research-experiment-workflow`
- branch: `oyx`
- base_commit: `ae3d6ab4e10bafa1357d85dd28ad493c283a95ee`
- worktree_dirty: `false`
- run_id: `cmv2_highres_full_20260917T134300Z`
- run_status: `RUNNING`
- scope: 解释高分辨率 OakInk2 cache 的吞吐瓶颈；不改变 exporter、GPU 分配、参数、输入或输出。
- conclusion: `SUPPORTED`（工程性能诊断）；与模型科研结论无关。

**原因**

每个 selected frame 必须生成双手 Inspire 20270 点训练流、KNN32、2 cm/5 cm object masks 和 hand supervision mask。相较旧 3076 点流，高分辨率手点数约为 6.6 倍；exporter 又按 segment 串行处理，逐帧 Python retarget、FK、chunked pairwise distance 与 NAS memmap 写出不能重叠。

**验证**

- 当时完成 `60/1849` segment、`27899` frame、`failures=[]`；进程墙钟约 19 分 41 秒，RSS 约 3.5 GiB、CPU 约 70%。[run_manifest.json](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest.json) 与 [实时日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/cmv2_highres_full_20260917T134300Z.log)。
- 每帧 object→hand KNN 需要 `4096×20270≈8300 万` 距离；反向 hand→object supervision 再做一次 chunked distance。GPU2 显存约 675 MiB、采样瞬时利用率为 0%，符合 KNN 短 burst 与 CPU 串行 retarget/FK 交替的行为，而非 GPU 持续饱和。
- 内核累计 I/O 约读取 11.9 GB、写入 42.8 GB，输出约 26 GB；`pidstat` 采样写入约 25 MB/s、无 iodelay，说明 NAS 写入有成本但不是已观测的唯一或主导饱和点。
- 当前实现不会跨 segment 缓存重叠帧的 retarget/KNN 结果；优化该点需要停止/重启并改变数据处理实现，本次未做。

**回滚**

只读诊断，无需回滚；当前导出继续使用已批准的参数与独立输出根。

## 2026-09-17 14:10:22 +0000 — 高分辨率 cache 可复用性与并行化预检

- timestamp: `2026-09-17 14:10:22 +0000`
- activity_id: `ACT-20260917-141022-CMV2-HIGHRES-PARALLEL-PREFLIGHT`
- modification_version: `V1.4.2`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问能否尝试批量并行以及既有结果是否可复用；本条只读核查当前 completed cache、partial 状态与 selection 重叠。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `a76f23603c24eae529f2f0eb573c21ce83684505`
- worktree_dirty: `false`
- run_id: `cmv2_highres_full_20260917T134300Z`
- run_status: `RUNNING`
- scope: 判断在不覆盖当前高分辨率输出的前提下实施并行 producer 的可行性；未停止、修改或重启导出。
- conclusion: `SUPPORTED`（可复用性诊断）；不涉及模型效果。

**原因**

并行化需要停止/恢复当前 L3 数据任务并修改 producer，必须先确认已完成段是否符合 V1.4 高分辨率合同，以及跨 segment 重叠能带来多少缓存收益。

**验证**

- 当时 run manifest 记录 `90/1849` 完成、`34950` 帧、`failures=[]`；磁盘扫描发现 94 个 complete geometry manifest、`0` 个 `.partial` 目录。逐段校验 `hand_variant=inspire_f1`、`knn_hand_points=20270`、`knn_points_per_side=10135`、实际 KNN points shape 与 KNN indices `<20270`，全部通过。已完成段可原样复用。
- selection 共有 379591 个 segment-frame occurrence、359857 个唯一 `(sequence, raw_frame_id)`，重复 19734 个（约 5.20%）；333 个 sequence 有多个 segment，单 sequence 最多 28 个。跨段 frame cache 有帮助但其理论上限约为 5%，不是主瓶颈。
- 当前 producer 只从旧 3076 `existing_root` 复用，若改造必须增加“先校验当前 `output_root` 中已完成的 20270 段再跳过”的 resume 行为；不能让新 worker 与当前 writer 并发写同一 root。
- [run_manifest.json](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest.json) 与 [实时日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/cmv2_highres_full_20260917T134300Z.log) — 当前运行状态入口。

**回滚**

只读预检，无代码或数据改动。后续若获得批准，先安全停止当前 writer、保留/验证 complete 段、使用独立并行实现恢复未完成段；旧 full/partial 与当前 complete cache 均不覆盖。

## 2026-09-17 14:15:54 +0000 — 高分辨率回填安全恢复与 KNN 批处理 smoke 准备

- timestamp: `2026-09-17 14:15:54 +0000`
- activity_id: `ACT-20260917-141554-CMV2-HIGHRES-RESUME-BATCH-SMOKE-PREP`
- modification_version: `V1.4.3`
- type: `code, data, operation, documentation`
- task_mode: `change`，随后切换为 `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确回复“可以，你继续吧”，批准停止单 writer、复用已完成段并尝试批量恢复 smoke。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e1966df71bab1d807be0a969538b4f054ff18407`
- worktree_dirty: `true`（仅本条列出的 V1.4.3 文件尚未提交）
- run_id: `cmv2_highres_full_20260917T134300Z`
- run_status: `STOPPED`
- scope: 停止已批准的单 writer 基线；保留同一高分辨率输出根并逐段校验复用；为两个未完成段的 GPU2 KNN batch smoke 增加独立 manifest。未改变 selection、split、坐标、2 cm、KNN32、MANO 4096 或 Inspire KNN 20270 合同。
- conclusion: `INCONCLUSIVE`（工程恢复准备；不是模型效果结论）。

**文件与产物**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — Cmv2 当前指针更新为 `V1.4.3`。
- [docs/plan/V1.4.md](../plan/V1.4.md) — 已定稿的批处理恢复边界：先单 writer 的 frame batch smoke，不在未测量前加入多进程 GPU/写入竞争。
- [tools/data/backfill_oakink2_inspire_v1_4.py](../../tools/data/backfill_oakink2_inspire_v1_4.py) — 同一 `output_root` 优先校验复用完整 20270 点段；恢复 run 不覆盖已暂停基线 manifest；限量 smoke 不发布局部正式 index/cache manifest，并登记实际 batch 参数。
- [已暂停基线 run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest.json) — `STOPPED`，`100` 个计数已写入段、磁盘逐段校验发现 `102` 个 complete geometry 段、`39885` 已计帧、无记录 failure。

**原因**

基线只能从旧 3076 点 root 查找可复用项，恢复时会错过同一 high-resolution root 中已完成的有效段，且会覆盖基线 `run_manifest.json`。该最小修订先消除这两个恢复风险；GPU2 的 frame batch 从默认 `2` 提至 smoke 的 `8`，以测量 KNN 短 burst 的批处理收益，但不引入未经测量的多进程写入或新的数据语义。

**验证**

- `ps -p 370520 -o pid=,stat=,etime=,cmd=` 无输出，GPU2 空闲 `48509 MiB`，并扫描到 `0` 个 `.partial` 目录；未删除、移动或覆盖已有 sequence。
- 已逐段检查 102 个 geometry manifest 及其 KNN shape/index/finite 合同；旧 `existing_root` 的 3076 点条目仍只作只读审计源，不能进入 V1.4 训练流。
- 待提交前运行 `py_compile`、Task 定向测试、`git diff --check` 和 `audit_diff.py --check-links`；恢复 smoke 的结果另记独立 `run_id`/manifest。

**回滚**

停止新的限量恢复 run 即可；基线 manifest、102 个已验证高分辨率段与旧 3076 审计 root 均保留。撤销代码只需恢复本条所列三个受版本控制文件，绝不删除 NAS cache。

## 2026-09-17 14:17:56 +0000 — GPU2 KNN batch=8 恢复 smoke 已启动

- timestamp: `2026-09-17 14:17:56 +0000`
- activity_id: `ACT-20260917-141756-CMV2-HIGHRES-RESUME-BATCH8-SMOKE-STARTED`
- modification_version: `V1.4.3`
- type: `operation, data`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确回复“可以，你继续吧”，批准批量恢复 smoke；运行参数遵循已定稿 [docs/plan/V1.4.md](../plan/V1.4.md)。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `c1b501ef6e7b4ffd72d5308d8a47e7df5ed1e338`
- worktree_dirty: `false`（启动时）
- run_id: `cmv2_highres_resume_batch8_smoke_20260917T142000Z`
- run_status: `RUNNING`
- scope: selection offset `102` 的两个连续未完成 OakInk2 段；同一高分辨率输出根，GPU2 单 writer，`knn_frame_batch=8`、`knn_object_chunk=512`、`mano_batch_size=128`。不生成局部正式 index/cache manifest。
- conclusion: `INCONCLUSIVE`（工程 smoke 运行中；不是模型效果结论）。

**原因**

先测量 frame batch 对实际高分辨率 KNN 阶段的影响，再决定是否值得引入独立 CPU 预取队列；这样不会同时改变并发模型与 GPU batch，且不会让多个进程争用 GPU2 或同一 sequence 写入。

**命令与产物**

- 命令：`PYTHONPATH=/home/wbcd/workspace/oyx_ws/Ref2Dex /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ObjectInteractionCmv2/tools/data/backfill_oakink2_inspire_v1_4.py ... --device cuda:2 --mano-batch-size 128 --knn-frame-batch 8 --knn-object-chunk 512 --offset 102 --limit 2 --run-id cmv2_highres_resume_batch8_smoke_20260917T142000Z`。
- [恢复 smoke manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest_cmv2_highres_resume_batch8_smoke_20260917T142000Z.json) — 当前 `STARTED`，独立于已暂停基线 manifest。
- [恢复 smoke 日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/cmv2_highres_resume_batch8_smoke_20260917T142000Z.log) — `RUNNING`。
- [首次启动失败日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/cmv2_highres_resume_batch8_smoke_20260917T141800Z.log) — `ModuleNotFoundError: src`，在创建 manifest/cache 前退出；已保留并用显式 `PYTHONPATH` 的新 run id 重试。

**验证**

- 启动 4 秒后 PID `433153` 存活；GPU2 为 `417 MiB / 48092 MiB free`。manifest 已记录两个待处理段、batch/chunk/MANO 参数且 `failures=[]`。
- 若任一段出现 schema、finite、KNN index、OOM 或写入错误，停止该 smoke，保留已完成基线并将终态写回本活动记录；不自动启动全量恢复。

**回滚**

仅终止此 run；不删除其已有输出，也不修改基线 `run_manifest.json` 或任何已验证段。

## 2026-09-17 14:19:14 +0000 — GPU2 KNN batch=8 恢复 smoke 完成

- timestamp: `2026-09-17 14:19:14 +0000`
- activity_id: `ACT-20260917-141914-CMV2-HIGHRES-RESUME-BATCH8-SMOKE-COMPLETED`
- modification_version: `V1.4.3`
- type: `operation, data, diagnostic`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续用户对 V1.4.3 限量批处理恢复 smoke 的明确批准。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `b55afe93d8ad3fb5fb93bc9f8c8c6394e28c15c8`
- worktree_dirty: `false`（终态检查时）
- run_id: `cmv2_highres_resume_batch8_smoke_20260917T142000Z`
- run_status: `COMPLETED`
- last_step: `2/2 segments`
- best_metric: `N/A`（数据回填，无训练指标）
- checkpoint: `N/A`（数据回填，无 checkpoint）
- scope: GPU2 单 writer 对 selection offset `102:104` 的两个未完成段执行 `knn_frame_batch=8` 限量恢复；不启动全量恢复、不改变并发模型，也不发布局部正式 index/cache manifest。
- conclusion: `SUPPORTED`（cache 恢复与高分辨率合同 smoke）；模型效果结论为 `INCONCLUSIVE`。

**原因**

该终态用于验证同 root 的完整段复用设计和 KNN frame batch 参数能在真实未完成段上写出合规数据，再决定是否启动受同一计划约束的余下恢复；两段样本不足以量化全量吞吐收益，也不能据此证明 CPU worker 的价值。

**产物与结果**

- [恢复 smoke manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest_cmv2_highres_resume_batch8_smoke_20260917T142000Z.json) — `COMPLETED`，`backfilled_segments=2`、`completed_frames=389`、`reused_segments=0`、`failures=[]`，从 `2026-09-17T14:17:46+00:00` 至 `14:18:20+00:00`。
- [恢复 smoke 日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/cmv2_highres_resume_batch8_smoke_20260917T142000Z.log) — writer 输出 `segments=2`、`backfilled=2`、`failures=0`。
- [共享高分辨率输出根](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z) — 基线的 102 个加本次 2 个，共 104 个 complete geometry 段；基线仍由其独立 manifest 标记 `STOPPED`。

**验证**

- 对新段 `0102`（89 帧）和 `0103`（300 帧）逐段执行 `_validate_v1_4_geometry`：`hand_variant=inspire_f1`、KNN bilateral `20270` 点、KNN32 index 上界、数组 shape 与 sampled finite 均通过。
- 输出根无 `.partial` 目录；`index.json` 和 `cache_manifest.json` 均不存在，确认限量 smoke 没有发布误导性的局部正式 cache。
- 运行结束后 GPU2 恢复空闲（`0 MiB / 48509 MiB free`）；未生成 `metrics.jsonl`、`train.log` 或 checkpoint，因为本运行是数据回填而非训练/评估。
- smoke 总墙钟约 34 秒（含模型/retarget 初始化），约 11.4 frame/s；样本段长度与单 writer 基线不同，因此仅作为可运行证据，不报告相对加速比。

**回滚**

无需删除任何 cache；若不继续，保持基线 `STOPPED` 和这 104 个 complete 段即可。若继续，恢复入口会先校验并跳过它们；若出现错误，停止新的 run 并保留所有已完成段。

## 2026-09-17 14:23:05 +0000 — OakInk2 高分辨率 batch=8 全量恢复已启动

- timestamp: `2026-09-17 14:23:05 +0000`
- activity_id: `ACT-20260917-142305-CMV2-HIGHRES-RESUME-BATCH8-FULL-STARTED`
- modification_version: `V1.4.3`
- type: `operation, data`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户在限量 smoke 成功后明确要求“那你继续吧”；运行遵循已定稿 [docs/plan/V1.4.md](../plan/V1.4.md) 的同 root 恢复边界。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `3d590398fcb7459ae01b7ac22895960c2bf08d3c`
- worktree_dirty: `false`（启动时）
- run_id: `cmv2_highres_resume_batch8_full_20260917T142300Z`
- run_status: `RUNNING`
- scope: 对完整 1849 个 selected OakInk2 segment 恢复；先校验并跳过 current root 中已完成的 104 段，再用 GPU2 单 writer 回填余下段。参数固定为 `mano_batch_size=128`、`knn_frame_batch=8`、`knn_object_chunk=512`，不改变 cache schema、selection、split、坐标、KNN32、2 cm、MANO 4096 或 Inspire KNN 20270 合同。
- conclusion: `INCONCLUSIVE`（工程数据运行中；不是模型效果结论）。

**原因**

限量 smoke 已证明 batch=8 的恢复路径能在未完成真实段生成合规高分辨率 cache；全量运行沿用同一 writer 与参数，以避免把未测量的 CPU 多进程预取和正式数据回填混为一次变量变化。

**命令与产物**

- 命令：`PYTHONPATH=/home/wbcd/workspace/oyx_ws/Ref2Dex /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ObjectInteractionCmv2/tools/data/backfill_oakink2_inspire_v1_4.py ... --device cuda:2 --mano-batch-size 128 --knn-frame-batch 8 --knn-object-chunk 512 --run-id cmv2_highres_resume_batch8_full_20260917T142300Z`。
- [全量恢复 manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest_cmv2_highres_resume_batch8_full_20260917T142300Z.json) — 当前 `STARTED`，将记录复用/回填段数、frames 和失败清单。
- [全量恢复日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/cmv2_highres_resume_batch8_full_20260917T142300Z.log) — `RUNNING`。
- [共享高分辨率输出根](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z) — 只追加缺失 sequence；不覆盖 104 个已验证 complete 段或已暂停基线 manifest。

**验证**

- 启动约 4 秒后 PID `443044` 存活，manifest 的 `expected_segments=1849`；前 100 个已完成段已校验复用、`failures=[]`，GPU2 已分配 `417 MiB`。
- 每完成 10 段刷新运行 manifest；终态前将检查 failures、formal `index.json`/`cache_manifest.json`、完整段数、schema/KNN 合同和 `.partial` 残留。未生成 `metrics.jsonl` 或 checkpoint，因为是数据回填。

**回滚**

终止此 `run_id` 即可；其此前已经完成的完整段仍可被后续 resume 校验复用。不得删除、覆盖或移动基线及本次生成的 cache。

## 2026-09-17 14:59:50 +0000 — V1.4.4 双域 MANO 续训与 cache 队列实现

- timestamp: `2026-09-17 14:59:50 +0000`
- activity_id: `ACT-20260917-145950-CMV2-V144-TWO-DOMAIN-TRAIN-CACHE-QUEUE-IMPLEMENTATION`
- modification_version: `V1.4.4`
- type: `code, data, experiment, operation, documentation`
- task_mode: `change`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确确认以 4-epoch `latest.pt` 作模型权重初始化、自行 trajectory 划分 ARCTIC、约 8 小时预算、保存 `best.pt`，并授权 GPU0。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `0a4d6ba1922ab63d8215017d8d74a5609c2104ce`
- worktree_dirty: `true`（仅本条列出的 V1.4.4 实现尚未提交）
- scope: 新增 GRAB+ARCTIC MANO 双域权重初始化训练、ARCTIC trajectory split、best/latest checkpoint 规则和通用顺序 cache queue；不修改旧 4-epoch checkpoint、原始 MANO cache、旧 Task 或正在 GPU2 上运行的 OakInk2 Inspire 回填。
- conclusion: `INCONCLUSIVE`（实现已完成、尚未启动正式训练；不是跨域效果结论）。

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) 与 [docs/plan/V1.4.md](../plan/V1.4.md) — Cmv2 指针与已定稿 V1.4.4 范围。
- [config.py](../../config.py)、[configs/active/two_domain_mano_v1_4_4_8h.yaml](../../configs/active/two_domain_mano_v1_4_4_8h.yaml)、[train_two_domain_mano.py](../../train_two_domain_mano.py) — GPU0、MANO-4096 双域 `1/2:1/2` 训练，旧权重仅初始化，8 小时/16 epoch 上限，GRAB 与 ARCTIC 验证等权选择 `best.pt`，同时保留 `latest.pt`。
- [tools/data/build_two_domain_mano_split_v1_4.py](../../tools/data/build_two_domain_mano_split_v1_4.py) — 不改原 cache 的 ARCTIC trajectory 级 90/10 split 和独立 index/cache/run manifest。
- [tools/data/run_cache_queue_v1_4_4.py](../../tools/data/run_cache_queue_v1_4_4.py) — 顺序、可恢复且以 producer 成功 manifest 为完成条件的后台 cache queue，不把旧 3076/KNN8 Inspire cache 误判为合格。
- [tests/test_v1_4_4_two_domain_split.py](../../tests/test_v1_4_4_two_domain_split.py) — trajectory 稳定性与 GRAB split 保留测试。

**原因**

旧 `train_grab_ddp.py` 的 optimizer/RNG resume 合同严格绑定 GRAB-only index、两卡 world size 和 epoch 数，不能在双域数据合同下直接恢复。新入口仅导入经批准的模型权重；ARCTIC 验证以完整 trajectory 划分，令 `best.pt` 有可审计且不泄漏的选择指标。cache queue 只编排明确的 producer command/成功 manifest，避免失败 cache 被静默继续使用。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/config.py src/task/ObjectInteractionCmv2/train_two_domain_mano.py src/task/ObjectInteractionCmv2/tools/data/build_two_domain_mano_split_v1_4.py src/task/ObjectInteractionCmv2/tools/data/run_cache_queue_v1_4_4.py` 通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_4_three_domain.py src/task/ObjectInteractionCmv2/tests/test_v1_4_4_two_domain_split.py`：`5 passed`。
- `git diff --check` 通过。GPU2 OakInk2 Inspire run 仍为 `RUNNING`、`backfilled_segments=96`、`failures=[]`；本实现未接触其进程或输出。

**回滚**

恢复本条所列受版本控制文件并把 Cmv2 指针回退到 `V1.4.3`；不删除任何 cache 或旧 checkpoint。尚未产生的新 split/run 输出可通过停止其独立 run 保留审计后不再使用。

## 2026-09-17 15:01:08 +0000 — 修正双域 split 首次运行的 manifest 父目录创建

- timestamp: `2026-09-17 15:01:08 +0000`
- activity_id: `ACT-20260917-150108-CMV2-V144-SPLIT-MANIFEST-PARENT-FIX`
- modification_version: `V1.4.4`
- type: `code, data, operation`
- task_mode: `change`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 属于用户已批准的 V1.4.4 trajectory split 正式运行前的最小实现修正。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `b418c6b02b97bdbf1d769d2e60e92a5f72816921`
- worktree_dirty: `true`（仅本条所列修正和记录尚未提交）
- run_id: `cmv2_two_domain_mano_split_20260917T150100Z`
- run_status: `FAILED`
- scope: 修正 split 工具在 output root 尚不存在时首次写 run manifest 的父目录创建顺序；未写入 split、未修改原 cache、未启动训练。
- conclusion: `INVALID_IMPLEMENTATION`（首次启动包装错误；数据/模型合同未执行）。

**文件**

- [tools/data/build_two_domain_mano_split_v1_4.py](../../tools/data/build_two_domain_mano_split_v1_4.py) — `_write_json` 先创建父目录，再执行原子临时文件替换。

**原因**

首次命令在 `run_manifest_*.json.tmp` 写入前调用 `build()`，因此 output root 尚未创建并触发 `FileNotFoundError`。错误发生在读取/处理 source index 之前，未产生可误用的 split 产物。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/tools/data/build_two_domain_mano_split_v1_4.py` 通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_4_4_two_domain_split.py`：`1 passed`。
- 下一步以同一独立 output root 和新 run id 重跑；只有 `COMPLETED` manifest/index/cache manifest 同时存在才可用于训练。

**回滚**

撤销该单行目录创建修正即可；没有需要删除或恢复的 cache/split 数据。

## 2026-09-17 15:02:16 +0000 — GRAB+ARCTIC MANO 权重初始化训练已启动

- timestamp: `2026-09-17 15:02:16 +0000`
- activity_id: `ACT-20260917-150216-CMV2-V144-TWO-DOMAIN-MANO-TRAIN-STARTED`
- modification_version: `V1.4.4`
- type: `experiment, operation, data`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户批准 GPU0，并确认只以今日 4-epoch `latest.pt` 初始化模型、ARCTIC trajectory split、8 小时预算和 `best.pt` 保存规则。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `e26934a85d9c3b90a8f8cba784578dc41c8e4c28`
- worktree_dirty: `false`（启动时）
- run_id: `cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z`
- run_status: `RUNNING`
- scope: GPU0 上的 GRAB/ARCTIC MANO-4096 `1/2:1/2` source-balanced 训练；从 GRAB-only 4-epoch checkpoint 加载模型权重但重建 optimizer/sampler，最大 16 epoch 或 28800 秒，以先达到者为准。GPU2 OakInk2 Inspire cache 回填继续独立运行。
- conclusion: `INCONCLUSIVE`（训练运行中；不是跨域效果结论）。

**原因**

当前两域 MANO cache 已完整，且独立 ARCTIC trajectory split 已生成；以新 output root 训练可以保留旧 run，并按 GRAB/ARCTIC val loss 等权选择新 `best.pt`。

**命令与产物**

- 命令：`PYTHONPATH=/home/wbcd/workspace/oyx_ws/Ref2Dex REF2DEX_CMV2_TWO_DOMAIN_MANO_INDEX=.../two_domain_mano_split_seed42/index.json REF2DEX_CMV2_TWO_DOMAIN_MANO_MANIFEST=.../cache_manifest.json REF2DEX_CMV2_OUTPUT_ROOT=.../outputs/ObjectInteractionCmv2 python -m src.task.ObjectInteractionCmv2.train_two_domain_mano --config src/task/ObjectInteractionCmv2/configs/active/two_domain_mano_v1_4_4_8h.yaml --run-id cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z --initial-checkpoint .../cmv2_v134_grab_ddp4_20260917T070638Z/latest.pt`。
- [双域 split run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_4/two_domain_mano_split_seed42/run_manifest_cmv2_two_domain_mano_split_20260917T150200Z.json) — `COMPLETED`；ARCTIC `266` train / `35` val trajectory，GRAB 既有 split 保持不变。
- [训练运行目录](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z) — `RUNNING`；[run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z/run_manifest.json)、[launcher log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z.launcher.log)，`metrics.jsonl`、`train.log`、`latest.pt` 与 `best.pt` 在生成前为 `PENDING`。

**验证**

- 启动 12 秒后 PID `509488` 存活；GPU0 `9145 MiB` 已用、约 `39364 MiB` 空闲，满足 35 GiB 启动门槛；manifest 已记录批准的初始 checkpoint。
- 数据集构建/首步完成后将核对双域 transition 数、loss finite、GPU 内存、`latest.pt` 和首个完整 epoch 的 `best.pt`；任何 OOM、非有限 loss、缺少任一 val 域或影响 GPU0 既有进程都会停止此 run，而不停止 GPU2 cache。

**回滚**

终止此独立 run 即可；旧 4-epoch checkpoint、split、GPU2 cache 回填和其他用户 GPU0 进程不被覆盖。

## 2026-09-17 15:04:57 +0000 — 修正 ARCTIC 派生验证 split 的严格标签冲突

- timestamp: `2026-09-17 15:04:57 +0000`
- activity_id: `ACT-20260917-150457-CMV2-V144-ARCTIC-DERIVED-SPLIT-OVERRIDE`
- modification_version: `V1.4.4`
- type: `code, experiment, data, operation, documentation`
- task_mode: `change`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确授权 Agent 自行按 trajectory 划分 ARCTIC 验证集；本修正仅使该已批准的派生 split 可被 V1.4.4 双域 runner 读取。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `0f11a4c9d3036128056fb890ef71e6ddde8740ba`
- worktree_dirty: `true`（仅本条列出的修正、V1.4.4 plan 补充和记录尚未提交）
- run_id: `cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z`
- run_status: `FAILED`
- scope: 记录并修正首次双域启动对 ARCTIC source geometry 原始 `split=train` 与派生 index `split=val` 的严格冲突；未训练、未执行 optimizer step、未生成 checkpoint，也未修改源 cache。
- conclusion: `INVALID_IMPLEMENTATION`（首次 loader split 标签错误）；修正后的 runner 尚待新 run 验证。

**文件与产物**

- [multi_domain.py](../../multi_domain.py) — 默认仍严格验证 manifest split；增加显式 `allow_manifest_split_override`，只由 V1.4.4 双域 runner 传入。
- [train_two_domain_mano.py](../../train_two_domain_mano.py)、[tests/test_v1_4_4_two_domain_split.py](../../tests/test_v1_4_4_two_domain_split.py) 与 [docs/plan/V1.4.md](../plan/V1.4.md) — 将该覆盖限制为用户批准的 ARCTIC trajectory 派生 split，并验证默认严格/显式覆盖两条路径。
- [训练失败 run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z/run_manifest.json) — `FAILED` 于 dataset validation，`last_step=null`；[launcher log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150300Z.launcher.log)。

**原因**

ARCTIC 原始 cache 的 geometry manifest 正确标记其来源处理 split 为 `train`，而新的、独立的 trajectory-level index 把其中 35 条用于模型验证。原 loader 将两者一律当作错误；此处不能重写 source geometry manifest，故在新 runner 中以显式开关声明 index 为训练 split 的权威来源。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/multi_domain.py src/task/ObjectInteractionCmv2/train_two_domain_mano.py` 通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_4_three_domain.py src/task/ObjectInteractionCmv2/tests/test_v1_4_4_two_domain_split.py`：`6 passed`；新增用例确认默认仍拒绝 mismatch，只有显式 override 才允许。
- `git diff --check` 通过；GPU0 训练进程已退出，GPU2 OakInk2 Inspire cache 回填未被停止。

**回滚**

撤销该显式 override 和新 run 即可；源 geometry manifest、ARCTIC split cache、旧 checkpoint 与失败运行目录均保留审计。

## 2026-09-17 15:06:03 +0000 — 修正后 GRAB+ARCTIC MANO 权重初始化训练已启动

- timestamp: `2026-09-17 15:06:03 +0000`
- activity_id: `ACT-20260917-150603-CMV2-V144-TWO-DOMAIN-MANO-TRAIN-RESTARTED`
- modification_version: `V1.4.4`
- type: `experiment, operation, data`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续用户对 GPU0、旧 `latest.pt` 模型权重初始化、ARCTIC trajectory val、8 小时预算与 `best.pt` 的明确批准；此前失败为已修正的实现错误。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `808b777cf1d1b5d9770172c86b195ffa7410f6b2`
- worktree_dirty: `false`（启动时）
- run_id: `cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z`
- run_status: `RUNNING`
- scope: GPU0 上的 GRAB/ARCTIC MANO-4096 等概率双域训练；旧 4-epoch checkpoint 仅初始化模型，重建 optimizer/sampler，最多 16 epoch 或 28800 秒。ARCTIC 派生 val 只通过 V1.4.4 显式 loader override 读取；GPU2 OakInk2 Inspire cache 回填继续独立运行。
- conclusion: `INCONCLUSIVE`（训练运行中；不是跨域效果结论）。

**原因**

首次运行在尚未训练前被严格 source geometry split 检查拒绝；修正后仍不改变 source cache，而由已审计的派生 index 定义 ARCTIC 模型验证归属。

**命令与产物**

- [训练运行目录](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z) — [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/run_manifest.json) 与 [launcher log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z.launcher.log)；`metrics.jsonl`、`train.log`、`latest.pt`、`best.pt` 在生成前为 `PENDING`。
- 初始权重：[cmv2_v134_grab_ddp4 latest.pt](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v134_grab_ddp4_20260917T070638Z/latest.pt)；split：[run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_4/two_domain_mano_split_seed42/run_manifest_cmv2_two_domain_mano_split_20260917T150200Z.json)。

**验证**

- 启动约 9 秒后 PID `516837` 存活、GPU0 `9145 MiB` 已用/约 `39364 MiB` 空闲，满足 35 GiB 启动门槛；新 manifest 已记录输入 checkpoint。
- 首次完成 dataset 构建/epoch 后检查 train/val 两域 transition、finite loss、`latest.pt` 与按等权 val metric 写出的 `best.pt`；出现 OOM、非有限 loss 或资源异常则停止此独立 run，绝不停止 GPU2 cache 回填。

**回滚**

终止此 run 即可；不覆盖旧 checkpoint、失败 run、split、缓存或 GPU0 既有进程。

## 2026-09-17 15:21:18 +0000 — V1.4.4 cache queue producer 审计

- timestamp: `2026-09-17 15:21:18 +0000`
- activity_id: `ACT-20260917-152118-CMV2-V144-CACHE-QUEUE-PRODUCER-AUDIT`
- modification_version: `V1.4.4`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求先审计，确认无问题后才让后台 queue 持续导出其他 cache；本条只读检查 producer、schema、当前运行和 GPU 归属。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `175d3ec60f0505e8c73dc22bd3d2de908de4f8af`
- worktree_dirty: `false`
- scope: 审计 GRAB/ARCTIC Inspire-20270、OakInk2 MANO-4096 是否已有可由 V1.4.4 queue 安全调用的 producer；不启动/停止 queue 或 cache run。
- conclusion: `REFUTED`（当前不能安全启动完整 queue）；与模型效果无关。

**原因**

queue 的完成条件是 producer 写出 `COMPLETED` success manifest；在将长期后台任务交给它前，必须确认每个 job 的输出已经满足 V1.4 的 4096/20270、KNN32 和 manifest 合同，不能用名称相似的历史 cache 代替。

**验证**

- 当前 OakInk2 Inspire producer [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest_cmv2_highres_resume_batch8_full_20260917T142300Z.json) 正在 GPU2 运行，`backfilled_segments=156`、`completed_frames=81201`、`failures=[]`；这是唯一可直接延续的高分辨率 queue job。
- GRAB/ARCTIC MANO-4096 已有完整 producer/cache；但旧 Inspire [cache manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/cache_manifest.json) 明确为 bilateral `3076` 点、`KNN=8`，且没有完整 geometry manifest，不能升级为 20270/KNN32 或加入训练。
- 当前仓库没有 GRAB/ARCTIC Inspire-20270/KNN32 producer，也没有 OakInk2 MANO-4096 producer；`run_cache_queue_v1_4_4.py` 是安全编排器，但尚无三个可审计的 job command/success manifest 可入队。
- GPU2 由 OakInk2 cache 使用；GPU0 的 V1.4.4 双域 MANO 训练仍在运行，其他 GPU 也有既有进程。queue 不应擅自占用或和现有 producer 并发写入。

**回滚**

纯诊断，无需回滚；保持当前 OakInk2 Inspire run 与双域训练不变。后续需先实现并小规模验证三个缺失高分辨率 producer，才生成 queue JSON 并启动后台连续任务。

## 2026-09-17 15:31:58 +0000 — 实现 V1.4.5 缺失高分辨率 cache producer

- timestamp: `2026-09-17 15:31:58 +0000`
- activity_id: `ACT-20260917-153158-CMV2-V145-HIGHRES-PRODUCER-IMPLEMENTATION`
- modification_version: `V1.4.5`
- type: `code, data, documentation`
- task_mode: `change`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户在 producer 审计后明确回复“可以，你先实现吧”；范围限定为新增 GRAB/ARCTIC Inspire-20270/KNN32 与 OakInk2 MANO-4096/KNN32 producer。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `2c3df4df64ddd66a49f3696cc3121848b4245864`
- worktree_dirty: `true`（本条所列 V1.4.5 实现和记录尚未提交）
- scope: 新 producer 只写 Cmv2 新 output root，保存 decoder 3076 compatibility stream 和独立高分辨率 KNN stream；不修改 `ObjectInteractionCm`、旧 3076/KNN8 cache、原 MANO cache、正在运行的 GPU0 训练或 GPU2 OakInk2 Inspire 回填。
- conclusion: `INCONCLUSIVE`（实现和无 GPU contract smoke 已通过；真实数据 GPU smoke 尚未运行，不能作为数据或效果结论）。

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) 与 [docs/plan/V1.4.md](../plan/V1.4.md) — 指向已定稿的 V1.4.5 producer/queue 边界。
- [multi_domain.py](../../multi_domain.py) — 仅新增两种 Cmv2 producer schema 的显式 reader 白名单；保持既有 schema 和点数检查不变。
- [tools/data/build_stage4_inspire_highres_v1_4.py](../../tools/data/build_stage4_inspire_highres_v1_4.py) — 以只读 Stage4 MANO、固定 Inspire visual surface sample（每侧 10135）和 KNN32 生成 GRAB/ARCTIC Inspire cache；来源 split 逐项沿用现有 MANO index。
- [tools/data/build_oakink2_mano_highres_v1_4.py](../../tools/data/build_oakink2_mano_highres_v1_4.py) — 以官方 OakInk2 annotation、固定 MANO triangle/barycentric correspondence（每侧 2048）和 KNN32 生成 train segment cache。
- [tests/test_v1_4_5_highres_producers.py](../../tests/test_v1_4_5_highres_producers.py) — 验证双侧 high-res→3076 decoder 派生、两 producer schema、4096/20270 shape 与 reader 接受路径。
- [docs/logs/activity_log.md](activity_log.md) — 本次实现的唯一活动时间线。

**原因**

审计确认旧 GRAB/ARCTIC Inspire 为 3076/KNN8，且 OakInk2 MANO producer 缺失；它们不能被重新标记为新合同。新实现将 KNN 永远建立在 `knn_hand_points_world.npy` 的 4096/20270 点流上，decoder 仅由该流确定性下采样，避免 3076 点意外成为训练输入。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/multi_domain.py src/task/ObjectInteractionCmv2/tools/data/build_stage4_inspire_highres_v1_4.py src/task/ObjectInteractionCmv2/tools/data/build_oakink2_mano_highres_v1_4.py` 通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_4_three_domain.py src/task/ObjectInteractionCmv2/tests/test_v1_4_4_two_domain_split.py src/task/ObjectInteractionCmv2/tests/test_v1_4_5_highres_producers.py`：`8 passed`。
- `git diff --check` 通过。真实 cache smoke 遵循 [docs/plan/V1.4.md](../plan/V1.4.md) 的 GPU2 串行闸门，当前 OakInk2 Inspire writer 完成前不与之并发；因此三个新 producer 均未入 queue。

**回滚**

恢复本条列出的版本控制文件并把 Cmv2 指针回退至 `V1.4.4` 即可；不删除任何已有 cache、checkpoint、输出或运行进程。后续 smoke/full run 使用独立 root，停止时保留其 manifest 和日志审计。

## 2026-09-17 15:35:55 +0000 — 双域 MANO 训练状态检查

- timestamp: `2026-09-17 15:35:55 +0000`
- activity_id: `ACT-20260917-153555-CMV2-V144-TWO-DOMAIN-MANO-STATUS`
- modification_version: `V1.4.4`
- type: `operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续用户已批准的 GPU0、8 小时 GRAB/ARCTIC MANO 双域训练；本条仅查询状态。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `a6c3f2062b6fc567da05cde5cf96810bcc60900c`
- worktree_dirty: `false`（查询前）
- run_id: `cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z`
- run_status: `RUNNING`
- scope: 查询既有 GPU0 训练和独立 GPU2 OakInk2 Inspire cache 回填；不停止、重启或改写任何运行。
- conclusion: `INCONCLUSIVE`（训练仍在进行，当前指标不是最终跨域结论）。

**原因**

用户询问训练是否仍在运行；需以 PID、GPU 和实际 epoch 指标区分“进程存在”与“有有效训练进度”。

**状态与证据**

- 训练 PID `516837` 仍存活；已完成 epoch `2`、step `4550`。当前 `selection_metric=0.1935872248433341`（GRAB val `0.3062595267893915`，ARCTIC val `0.08091492289727668`），优于 epoch 1，因此已有新的 [best.pt](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/best.pt) 和 [latest.pt](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/latest.pt)。
- [训练 run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/run_manifest.json)、[metrics.jsonl](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/metrics.jsonl) 与 [train.log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/train.log) 已存在。
- GPU2 OakInk2 Inspire backfill PID `443044` 也仍存活；其 manifest 当前累计 `reused_segments=104`、`backfilled_segments=516`、`completed_frames=182654`、`failures=[]`，尚未完成，故 V1.4.5 smoke/queue 仍不启动。

**验证**

- `ps -p 516837,443044` 确认两个 PID 均为 `Rsl`；`nvidia-smi` 显示训练 GPU0 利用率 `100%`。
- 仅检查 [metrics.jsonl](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/metrics.jsonl) 的最新两条完整 epoch 记录，未作额外训练或 cache 操作。

## 2026-09-18 02:12:18 +0000 — V1.4.5 高分辨率 producer 串行真实 smoke 启动

- timestamp: `2026-09-18 02:12:18 +0000`
- activity_id: `ACT-20260918-021218-CMV2-V145-HIGHRES-PRODUCER-SMOKE-STARTED`
- modification_version: `V1.4.5`
- type: `data, operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续用户已确认的 V1.4.5 producer/持续 queue 范围；V1.4 最终计划 §9 要求当前 OakInk2 Inspire 成功完成后，先对三条新 producer 做单条真实数据 smoke，再串行启动 full queue。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `cbc4a798b8c50f81def0181bfe14f5a37fd0eadd`
- worktree_dirty: `false`（启动前）
- run_id: `cmv2_v145_highres_producer_smoke_20260918T021218Z`
- run_status: `STARTED`
- scope: GPU2 串行运行 GRAB Inspire-20270、ARCTIC Inspire-20270、OakInk2 MANO-4096 各一条真实 source smoke；已有 OakInk2 Inspire-20270 全量回填已 `COMPLETED`，双域训练也已完成。smoke 仅写新的独立 root，不触及旧 cache、训练 checkpoint 或其他 GPU 进程。
- conclusion: `INCONCLUSIVE`（真实 producer 合同尚待 smoke 结果；不能据此得出模型效果结论）。

**前序状态与计划产物**

- OakInk2 Inspire 前序 [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest_cmv2_highres_resume_batch8_full_20260917T142300Z.json) 已 `COMPLETED`，`1849/1849` segment、`379591` frame、`failures=[]`；其 [cache manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/cache_manifest.json) 已存在。
- GRAB/ARCTIC Inspire smoke 与 OakInk2 MANO smoke 的独立 run root、run manifest 和日志将在产生前标记为 `PENDING`；任一 smoke `FAILED` 即不创建 full queue，已产生的独立 smoke 目录保留审计。

**验证与回滚**

- 启动前确认 GPU2 空闲，且没有 `build_stage4_inspire_highres_v1_4`、`build_oakink2_mano_highres_v1_4` 或 cache queue 进程；所有输入 index、Stage4、annotation、Stage3、MANO 和旧 cache 均按只读输入使用。
- 回滚入口为终止当前独立 smoke；不删除任何已完成 cache 或 checkpoint。只有三个 smoke 都以 `COMPLETED` manifest 结束，才按 [V1.4 最终计划 §9](../plan/V1.4.md) 创建并启动串行 full queue。

## 2026-09-18 02:19:57 +0000 — V1.4.5 高分辨率 full cache queue 已启动；双域 MANO 训练已完成

- timestamp: `2026-09-18 02:19:57 +0000`
- activity_id: `ACT-20260918-021957-CMV2-V145-CACHE-QUEUE-STARTED-V144-TRAIN-COMPLETED`
- modification_version: `V1.4.5`（queue）；`V1.4.4`（已完成训练）
- type: `operation, data, experiment, documentation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续用户对 GPU2 串行持续导出和 V1.4.4 双域 MANO 训练的明确批准；V1.4 §9 的三项真实 smoke 均已成功完成，满足 full queue 启动条件。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `cbc4a798b8c50f81def0181bfe14f5a37fd0eadd`
- worktree_dirty: `true`（仅本次活动记录尚未提交；full cache producer 使用同一已提交代码）
- run_id: `cmv2_v145_highres_queue_20260918T022000Z` / `cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z`
- run_status: `RUNNING`（queue）/ `COMPLETED`（训练）
- scope: queue 在 GPU2 串行导出 GRAB Inspire-20270、ARCTIC Inspire-20270、OakInk2 MANO-4096；每项仅在自身 success manifest 为 `COMPLETED` 时才解锁下一项。双域训练的既有独立 output 已封存；不启动新的模型训练，不覆盖旧 cache、checkpoint 或其他 GPU 进程。
- conclusion: `INCONCLUSIVE`（cache queue 仍在运行；训练只提供优化状态，尚无 held-out 跨域效果结论）。

**原因**

此前 queue 缺少三条可审计 producer，因而不能启动；实现后的三个独立真实 smoke 已分别验证 GRAB/ARCTIC Inspire-20270 和 OakInk2 MANO-4096 的数据合同。前序 OakInk2 Inspire 回填也已成功终态，故按计划启动同一 GPU 的单 writer 队列，而不与其他 cache 或训练并发写入。

**真实 smoke 与 queue 证据**

- GRAB Inspire smoke [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_smokes/cmv2_v145_grab_inspire_smoke_20260918T021300Z/run_manifest_cmv2_v145_grab_inspire_smoke_20260918T021300Z.json)：`COMPLETED`，`279` frame，KNN 最大 index `20251`。
- ARCTIC Inspire smoke [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_smokes/cmv2_v145_arctic_inspire_smoke_20260918T021500Z/run_manifest_cmv2_v145_arctic_inspire_smoke_20260918T021500Z.json)：`COMPLETED`，`732` frame，KNN 最大 index `20269`。
- OakInk2 MANO smoke [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_smokes/cmv2_v145_oakink2_mano_smoke_20260918T021800Z/run_manifest_cmv2_v145_oakink2_mano_smoke_20260918T021800Z.json)：`COMPLETED`，`1246` frame，KNN 最大 index `4093`；三项均 `failures=[]`。
- queue 的 [配置](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/queue.json)、[状态](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/queue_state.json) 与 [启动日志](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/queue.launcher.log) 已存在；当前 GRAB job 为 `RUNNING`，其 [job log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/grab_inspire_20270_knn32.log) 与 [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/grab_inspire_20260918T022000Z/run_manifest_cmv2_v145_grab_inspire_full_20260918T022000Z.json) 为实时入口。

**双域训练终态**

- [训练 run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/run_manifest.json) 为 `COMPLETED`：last epoch `16`、last step `36400`，`best_metric=0.18120233068706187`（epoch 13）；[best.pt](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/best.pt)、[latest.pt](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/latest.pt)、[metrics.jsonl](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/metrics.jsonl) 与 [train.log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/train.log) 均保留。

**验证**

- 三项真实 smoke 都由其独立 `COMPLETED` manifest 和 `failures=[]` 证明 producer、shape、KNN index 上界、30 Hz 及原子输出链路可用；它们是工程 smoke 证据，不是模型效果结论。
- queue 启动后 PID `897032` 及其 GRAB producer 子进程均存活，GPU2 处于使用状态；queue state 已持久化首项的命令、manifest 和日志入口。

**验证与回滚**

- 启动后 PID `897032`（queue）及其 GRAB producer 子进程均存活；GPU2 处于使用状态。queue 遇任一非零退出或 success manifest 不为 `COMPLETED` 会写 `FAILED` 并停止，不启动后续 job。
- 回滚入口是终止 queue/当前 job；已完成的独立 full cache 及 smoke 保留审计，不删除、移动或覆盖；恢复时用相同 [queue 配置](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/queue.json) 重新执行，producer 的 `--resume` 只验证并复用完整段。

## 2026-09-18 02:23:11 +0000 — 双域 MANO 训练终态与 ARCTIC stride 核验

- timestamp: `2026-09-18 02:23:11 +0000`
- activity_id: `ACT-20260918-022311-CMV2-V144-TWO-DOMAIN-TRAIN-STRIDE-DIAGNOSTIC`
- modification_version: `V1.4.4`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问既有 GRAB/ARCTIC 双域训练状态及 ARCTIC temporal stride；本条仅读取已完成 run 的 manifest、配置、metrics 和实现。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `241e6d3f372b53335ba7bea066d29d9d3014ec58`
- worktree_dirty: `false`（查询前）
- run_id: `cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z`
- run_status: `COMPLETED`
- scope: 核验 V1.4.4 GRAB/ARCTIC MANO 双域 run 的终态、最佳 selection metric 和实际训练/验证 stride；不启动、停止或修改任何训练、queue、配置、cache、checkpoint。
- conclusion: `INCONCLUSIVE`（工程优化运行已完成；没有新的 held-out 跨域效果评估）。

**原因**

运行状态和 temporal stride 会改变对 checkpoint 与验证指标的解释，必须以该 run 的冻结 `config.json` 和 runner 实现为准，而非从数据的 30 Hz 输入或口头范围推断。

**状态与证据**

- [训练 run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/run_manifest.json)：`COMPLETED`，last epoch `16`、last step `36400`，`best_metric=0.18120233068706187`；最佳发生在 epoch `13`，GRAB val loss `0.28988706171464246`，ARCTIC val loss `0.07251759965948128`。
- 冻结 [config.json](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/config.json) 声明 `train_stride_values.arctic=[1,2,3,4,5,6,7,8,9,10]`，与 GRAB 相同；并非只使用 `[5,6,7,8,9,10]`。
- [train_two_domain_mano.py](../../train_two_domain_mano.py) 在 train split 将上述 domain-specific stride values 交给 transition dataset；[multi_domain.py](../../multi_domain.py) 以稳定 seed 对每条 `(sequence, current frame)` 在允许集合中选一个 stride，故训练样本使用 `1..10`。验证通过 `fixed_stride=2`，即两域验证都是 stride `2`。
- [metrics.jsonl](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/metrics.jsonl)、[train.log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/train.log)、[best.pt](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/best.pt) 与 [latest.pt](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/latest.pt) 均存在。

**验证**

- 读取 run manifest 的 `run_status/last_epoch/last_step/best_metric` 与 metrics 最后 16 条 epoch 记录一致；epoch 13 的 selection metric 为全程最低值。
- 配置校验与源码均要求 V1.4.4 的 GRAB、ARCTIC `train_stride_values` 恰为 `1..10`、`eval_stride=2`；本次没有执行训练或评估。

**回滚**

纯只读诊断；不涉及运行状态、数据或代码变更，无需回滚。

## 2026-09-18 02:26:54 +0000 — V1.4.4 双域 MANO run 的 EPE 产物核验

- timestamp: `2026-09-18 02:26:54 +0000`
- activity_id: `ACT-20260918-022654-CMV2-V144-TWO-DOMAIN-EPE-DIAGNOSTIC`
- modification_version: `V1.4.4`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求浏览既有 GRAB/ARCTIC 双域训练的 EPE；本条仅读取运行产物、训练入口和指标实现，不运行新的评估。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `25af8cbe37aeae7e1411101ca8e30d935ac9c29b`
- worktree_dirty: `false`（查询前）
- run_id: `cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z`
- run_status: `COMPLETED`
- scope: 确认该 run 是否计算/保存 object 或 flow EPE；不将 V1.4.4 selection loss、历史 V1.3 smoke EPE 或其他 Task 的 EPE 误归因给本次双域训练。
- conclusion: `INCONCLUSIVE`（本 run 未产生 EPE，不能据现有产物报告 GRAB、ARCTIC 或均值 EPE）。

**原因**

EPE 是以物理单位解释预测误差的指标，不能由训练 loss 或跨版本/跨数据合同的历史 EPE 替代。必须先确认本次 run 的 metrics schema 和训练/评估入口是否实际实现该指标。

**状态与证据**

- [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/run_manifest.json) 仅将 selection metric 定义为 `mean(val_grab_loss,val_arctic_loss)`；没有 EPE output 或离线评估产物。
- [metrics.jsonl](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/metrics.jsonl) 的 16 条 epoch 记录只含 `val_grab_loss`、`val_arctic_loss`、`selection_metric`、`best_metric`、source counts 与 elapsed time；没有 `epe`/`flow_epe_mm`/`object_epe_mm` 字段。对应 [train.log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v144_grab_arctic_mano_init8h_20260917T150600Z/train.log) 相同。
- [train_two_domain_mano.py](../../train_two_domain_mano.py) 的验证仅聚合两域 loss 并选择 `best.pt`；源码和本 run 目录均未发现 EPE 计算或保存。历史 V1.3 smoke 的 EPE 属于不同 run/工程 smoke，不可用于本次 V1.4.4 结果。

**验证**

- 对本 run 输出目录、metrics、train log、V1.4.4 training/evaluation 入口进行只读 `rg` 检索；目标 run 的 EPE 匹配结果为空，metrics schema 与 manifest 的 selection metric 一致。
- 未执行 checkpoint 载入、验证或测试，因此没有产生新的 EPE，也没有改变当前 cache queue 的资源占用。

**回滚**

纯只读诊断；无运行、代码、配置或数据修改，无需回滚。

## 2026-09-18 02:41:06 +0000 — V1.4.5 串行高分辨率 cache queue 状态核验

- timestamp: `2026-09-18 02:41:06 +0000`
- activity_id: `ACT-20260918-024106-CMV2-V145-CACHE-QUEUE-STATUS`
- modification_version: `V1.4.5`
- type: `diagnostic, operation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 GRAB/ARCTIC Inspire 与 OakInk2 cache 是否已完整导出；本条只检查 queue state、producer manifest、实际 geometry 写入、日志与 GPU/进程状态。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `29a847d0b8b0edf2a2c2ae34fcf2b27e34760cf7`
- worktree_dirty: `false`（查询前）
- run_id: `cmv2_v145_highres_queue_20260918T022000Z`
- run_status: `RUNNING`
- scope: 核验既有 GPU2 串行 queue；不停止、重启、调整或并行化 producer，不更改 cache、训练 checkpoint、输入 index 或其他用户 GPU 进程。
- conclusion: `INCONCLUSIVE`（queue 尚未全部完成；当前没有失败证据）。

**原因**

queue 的顺序合同要求只有当前 GRAB Inspire 完整成功才允许 ARCTIC Inspire，随后才允许 OakInk2 MANO；必须区分已经完成的 OakInk2 Inspire、正在导出的 GRAB 和尚未获解锁的下游 job，不能将同为“Inspire”或同为“OakInk2”的不同 variant 混为完整。

**状态与证据**

- 先前 OakInk2 Inspire-20270 [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/run_manifest_cmv2_highres_resume_batch8_full_20260917T142300Z.json) 已 `COMPLETED`：`1849/1849` segment、`379591` frame、`failures=[]`；正式 [index.json](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/index.json) 与 [cache manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/cmv2_highres_full_20260917T134300Z/cache_manifest.json) 已存在。
- 当前 [queue state](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/queue_state.json) 显示 GRAB Inspire-20270/KNN32 job `RUNNING`；其 [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/grab_inspire_20260918T022000Z/run_manifest_cmv2_v145_grab_inspire_full_20260918T022000Z.json) 为 `STARTED`，已完成 `131/1335` sequence、`49397` frame、`failures=[]`。最近 geometry manifest 写入为 `2026-09-18 02:40:57 +0000`；有 `131` 个完整 geometry 和 `1` 个 `.partial`，符合原子写入中的单项状态。
- ARCTIC Inspire-20270 与 OakInk2 MANO-4096 的 full root/manifest 尚未出现，因为 queue 尚未完成 GRAB 前序；这不是失败，亦不能称为已导出完成。
- queue PID `897032` 等待当前 GRAB producer PID `897162`；producer 为 `Rl` 状态、持续 CPU 工作，GPU2 分配约 `651 MiB`。其 [job log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/grab_inspire_20270_knn32.log) 连续写入成功记录。

**验证**

- 对 queue state、所有 full run manifest/cache manifest、job log、完成 geometry 数和 `.partial` 数做只读核验；状态、输出计数、最近写入时间与存活进程相互一致。
- 未将 `STARTED` GRAB output 当作可用完整 index，未运行任何训练或评估；也没有修改 queue/producer 状态。

**回滚**

纯只读状态核验；运行控制权保持原 queue。若用户明确要求停止，可终止 queue/当前 job；已完成 geometry 及其 manifest 保留，恢复时使用原 queue 的 `--resume` 合同。

## 2026-09-18 02:43:14 +0000 — GRAB Inspire-20270 queue 完成时间估算

- timestamp: `2026-09-18 02:43:14 +0000`
- activity_id: `ACT-20260918-024314-CMV2-V145-GRAB-INSPIRE-ETA-DIAGNOSTIC`
- modification_version: `V1.4.5`
- type: `diagnostic, operation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前 GRAB Inspire-20270 全量导出预计还需多久；本条仅用 queue start timestamp 与当前 manifest 完成计数估算。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d315ede795440e4d401b444254d908af18e13791`
- worktree_dirty: `false`（查询前）
- run_id: `cmv2_v145_grab_inspire_full_20260918T022000Z`
- run_status: `RUNNING`
- scope: 只读估算当前 GRAB producer 完成时间；不调整 GPU、并发度、batch/chunk 参数、queue 顺序或任何数据。
- conclusion: `INCONCLUSIVE`（预测而非终态；实际耗时受每条序列帧数、NAS I/O 和 GPU 吞吐影响）。

**原因**

queue 必须串行完成 GRAB 后才启动 ARCTIC 和 OakInk2 MANO，故当前 job 的实测吞吐是后续可用时间的直接约束；不能以单条 smoke 或固定“每 epoch”时间替代全量序列的真实吞吐。

**状态与证据**

- [queue state](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/queue_state.json) 显示 GRAB job started at `2026-09-18T02:19:05+00:00`；其 [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/grab_inspire_20260918T022000Z/run_manifest_cmv2_v145_grab_inspire_full_20260918T022000Z.json) 于本次核验时为 `141/1335` sequence、`53860` frame、`failures=[]`。
- 从已运行约 `24.0` 分钟估得平均吞吐约 `5.885` sequence/min；线性外推余下 `1194` 条约需 `202.9` 分钟，预计完成约 `2026-09-18 06:05:55 +0000`。
- producer PID `897162` 仍为 `Rl`，最新 [job log](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z/grab_inspire_20270_knn32.log) 已记录第 `141` 条成功，当前无失败条目。

**验证**

- 以当前 UTC 时间减 queue 的 `started_at` 得实际 elapsed，再用 manifest 的 completed/expected sequence 计数计算 `sequence/min` 和线性剩余时间；没有执行运行控制或数据操作。

**回滚**

纯只读预测；不涉及需回滚的变更。

## 2026-09-18 02:55:26 +0000 — 实现 V1.4.6 双域 MANO checkpoint 离线 flow 评估器

- timestamp: `2026-09-18 02:55:26 +0000`
- activity_id: `ACT-20260918-025526-CMV2-V146-TWO-DOMAIN-FLOW-EVAL-IMPLEMENTATION`
- modification_version: `V1.4.6`
- type: `code, experiment, documentation`
- task_mode: `change`
- change_level: `L2`（新增离线指标的聚合与静止角度合同；不改变模型、GT、split、cache 或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户于 2026-09-18 确认：使用 `best.pt`、GRAB stride=1 抽 12000、ARCTIC stride=5--10 各抽 2000，并报告 point-micro EPE、预测/GT 模长与有效点夹角。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `1c355730091047de2d5743cd3cd91f12896eb0c7`
- worktree_dirty: `false`（实现前）
- scope: 新增 Task-local 离线 evaluator 和纯指标测试，更新 V1.4 final plan 与版本指针；不启动完整评估，不触碰 GPU2 cache queue、cache、训练配置或任何 checkpoint。
- conclusion: `INCONCLUSIVE`（实现与 CPU 测试通过；尚未完成 checkpoint/dataset smoke 或正式 24000-transition 评估，不能报告模型数值）。

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 将本 Task 当前指针更新为 `V1.4.6`。
- [docs/plan/V1.4.md](../plan/V1.4.md) — 追加已定稿的 §10，冻结 checkpoint、抽样、度量、GPU1 资源门槛与回滚边界。
- [eval_two_domain_mano.py](../../eval_two_domain_mano.py) — 新增确定性抽样、严格 checkpoint/schema/显存检查、微平均 flow 指标及 run manifest 输出。
- [tests/test_v1_4_6_two_domain_flow_eval.py](../../tests/test_v1_4_6_two_domain_flow_eval.py) — 覆盖 EPE/模长/夹角/静态点统计和确定性无放回抽样。
- [logs/activity_log.md](activity_log.md) — 登记本次实现边界与验证证据。

**原因**

已完成的 V1.4.4 双域训练只记录 validation loss，未产生用户所需 EPE、flow 模长或方向误差。离线评估必须固定为用户指定的 best checkpoint、transition 范围和点级单位合同，同时分离静止流的未定义角度，避免以训练 loss 或历史 run 指标替代本次数值。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCmv2/eval_two_domain_mano.py`：通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_4_6_two_domain_flow_eval.py src/task/ObjectInteractionCmv2/tests/test_v1_4_4_two_domain_split.py`：`4 passed`。
- `git diff --check`：通过。提交前还将运行 scoped `audit_diff.py --check-links`；真实 checkpoint/dataset smoke 和完整评估各自以独立 run manifest 记录。

**回滚**

回滚为不使用或删除本次未提交的 evaluator/测试/计划增量；不会删除或覆盖外部 checkpoint、数据 index、cache 或正在运行的 GPU2 queue。后续运行若失败，只保留其独立 output manifest 作为诊断证据。
