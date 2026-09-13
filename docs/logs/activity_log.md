# Ref2Dex 活动记录

- scope: root
- last_updated: 2026-09-12
- current_pointer: [docs/current_versions.yaml](../current_versions.yaml)
- historical_audit: [modification_log.md](modification_log.md)

## 2026-09-13 10:29:45 +0800 — IsaacGymEnvs vendor 迁移与路径 smoke

- activity_id: ACT-20260913-102945-ISAACGYM-VENDOR-MIGRATION
- timestamp: 2026-09-13 10:29:45 +0800
- modification_version: V1.2.15
- type: governance / code / operation / documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认将 `/home2/wyy/oyx_ws/IsaacGymEnvs` 纳入 Ref2Dex，并采用 `third_party/IsaacGymEnvs` 的 squashed vendor 方案。
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- import_commit: e1aabc1897bd27c793a765ec58e6145fb47ee02f
- upstream: `https://github.com/isaac-sim/IsaacGymEnvs`, `release/1.5.1`, `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3`
- scope: 导入 upstream snapshot 到 `third_party/IsaacGymEnvs/`，不携带嵌套 `.git`；迁移现有 CmResidual 本地修改；新增 upstream provenance 说明；将 vendor smoke 的 object asset 路径切到仓库内；根 `runs/` 加入忽略。外部 checkout 保留为只读 mirror，不删除、不覆盖。
- files: `./.gitignore`; `third_party/IsaacGymEnvs/UPSTREAM.md`; `third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py`; `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py`; `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml`; `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDecoderBank.yaml`; `third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPPO.yaml`。
- cumulative_dirty_paths: `./.gitignore`; `third_party/IsaacGymEnvs/UPSTREAM.md`; `third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py`; `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py`; `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml`; `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDecoderBank.yaml`; `third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPPO.yaml`。
- run_id: `cm_residual_decoder_bank_vendor_smoke_20260913_102945`
- run_status: COMPLETED
- command: `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home2/wyy/oyx_ws/Ref2Dex:/home2/wyy/oyx_ws/Ref2Dex/third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python third_party/IsaacGymEnvs/isaacgymenvs/train.py task=CmResidualDecoderBank train=CmResidualPPO headless=True num_envs=4 max_iterations=1 pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 train.params.config.minibatch_size=128`
- output: [runs/CmResidual_13-10-29-45](../../runs/CmResidual_13-10-29-45/)、[run_manifest.json](../../runs/CmResidual_13-10-29-45/run_manifest.json)。vendor 路径下单 PPO epoch 完成，action `(12,)`、observation `(71,)`；未完成 episode，`rew=-inf` 仍是 smoke 统计伪影。
- conclusion: SUPPORTED（vendor snapshot、路径、CmResidualDecoderBank alias 和 RL smoke 可运行）；INCONCLUSIVE（正式RL训练和物理抓取）。

**原因**

外部 checkout 本身已有独立 Git，但 Ref2Dex 主工作区无法直接显示其 diff。将固定 upstream 基线和当前本地任务修改纳入主仓库后，VSCode 可以在同一仓库中追踪 vendor 与本地改动，同时保留外部 mirror 作为回滚/对照来源。

**验证**

- `git -C /home2/wyy/oyx_ws/IsaacGymEnvs log` 确认原仓库基线及 origin；vendor `UPSTREAM.md` 记录来源、commit 和本地修改边界。
- `py_compile`、Ref2Dex 定向测试和 vendor 路径 IsaacGymEnvs GPU smoke 通过；配置中的 object asset 已解析到 `third_party/IsaacGymEnvs/assets`。
- vendor `runs/` 和根 `runs/` 不进入 Git；外部仓库未删除或 reset。

**产物与回滚**

- [third_party/IsaacGymEnvs/UPSTREAM.md](../../third_party/IsaacGymEnvs/UPSTREAM.md)
- [vendor RL smoke](../../runs/CmResidual_13-10-29-45/)、[run_manifest.json](../../runs/CmResidual_13-10-29-45/run_manifest.json)
- 回滚入口：保留外部 mirror；如需撤销 vendor 基线，可回到父提交 `5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a`，不触碰当前 Ref2Dex dirty 研究文件。

## 2026-09-12 13:39:26 +0800 — 跨手 Cm 路线的代码与既有证据核对

- activity_id: ACT-20260912-133926-CROSS-HAND-REVIEW
- timestamp: 2026-09-12 13:39:26 +0800
- modification_version: V1.2.15（沿用根级指针，只登记跨 Task 只读诊断）
- task_versions_reviewed: ObjectInteractionCm V1.3.2；CmDecoderv2 V1.1.7
- type: diagnostic / documentation
- task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求自主浏览仓库并判断如何通过 Cm 实现跨手动作表征；本轮只读代码与既有产物，追加诊断记录。
- skills_used: research-change-control
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true（开始时已有根 AGENTS 删除、版本指针、ObjectInteractionCm 计划/日志及未跟踪诊断/测试）
- scope: 跨 Task 代码与既有证据核对、下一步建议；不运行新训练/模型评估，不修改科研结论、模型、数据、split、cache、checkpoint、指导、计划或架构快照。
- conclusion: SUPPORTED（所列代码事实和归档数值的复核）；INCONCLUSIVE（跨手可复用性、未见手型泛化和下面提出的机制假设）。

**文件**

- [docs/logs/activity_log.md](activity_log.md) — 仅新增本条；没有其他本轮写入。

**原因**

用户的目标是跨手动作表征，因此需要分别检验 effect 预测、目标手对 Cm 的依赖、跨源交换和递归执行。
现有 object EPE 或来源分类实验不能独立替代后面三项。

1. 当前相关主线是 ObjectInteractionCm → CmDecoderv2。前者由物体几何、邻近手几何和 endpoint hand flow
   生成 `16 × 32` tokens、动态 anchor position/normal；后者冻结前者，以 `K=4` 的完整 Cm 和目标手当前
   q/wrist/link geometry 预测运动。实际接口包含 anchors，不能只把 tokens 当作整个动作通道。
   见 [src/task/ObjectInteractionCm/model.py](../../src/task/ObjectInteractionCm/model.py)、
   [src/task/CmDecoderv2/model.py](../../src/task/CmDecoderv2/model.py)。
2. ObjectInteractionCm 同时监督物体 flow 和源手 flow，默认权重均为 1；没有跨手交换/一致性损失。
   源手重建可能要求保留 embodiment-specific motion，是待做消融的机制假设，不能仅凭代码认定它有害；
   hand decoder 也接收手几何，部分形态信息可能由该条件提供。
   见 [src/task/ObjectInteractionCm/config.py](../../src/task/ObjectInteractionCm/config.py)、
   [src/task/ObjectInteractionCm/runner.py](../../src/task/ObjectInteractionCm/runner.py)。
3. 归档的 7670 个有效样本复算得到如下 frame-micro object EPE；这支持存在预测增益，不支持已经实现跨手迁移。

   | 方法 / mm | MANO | Inspire RL |
   | --- | ---: | ---: |
   | 平均接触手点位移 | 8.442019 | 7.163211 |
   | 接触手点刚体拟合 | 7.772431 | 7.902781 |
   | 当前 Cm 模型 | 7.290740 | 5.742603 |

   原有 paired sequence-bootstrap 显示 MANO 上相对刚体拟合的改善 CI 跨零；不是等价证明。
   证据：[src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/](../../src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/)、
   [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json](../../src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json)、
   [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl](../../src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl)。
4. 当前 best checkpoint 的实际 SHA256 与 V1.3 source classifier manifest 一致，均为
   `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。
   有效 held-out 样本 88 个，分类准确率 `65/88=73.8636%`，AUROC `0.7531`。
   这是 source 可分性的诊断，仍混合动作、接触、采样密度和数据来源差异；分类达到随机水平亦不能证明动作表征有效。
   证据：[src/task/CmDecoderv2/research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/metrics.json](../../src/task/CmDecoderv2/research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/metrics.json)、
   [src/task/CmDecoderv2/research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/run_manifest.json](../../src/task/CmDecoderv2/research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/run_manifest.json)。
5. V1.3 index 的 630 个 parent sequence 全部唯一，train 为 MANO/Inspire `254/255`、val `28/30`、
   test `63/0`。当前缓存不是同交互的双手配对；正式 test 也不能直接提供对称的双源评估。
   同名动作或原始 retargeting 来源不保证修正后的实际物体 effect 相同。
   证据：[data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json](../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json)、
   [src/task/ObjectInteractionCm/docs/plan/V1.2.5.md](../../src/task/ObjectInteractionCm/docs/plan/V1.2.5.md)。
6. 既有 `mouse_lift` Inspire 自身递归诊断中，GT 接触段 hand-position EPE 均值 `87.909 mm`，
   hand-flow EPE `2.363 mm`；这说明该序列存在递归状态漂移，不能把所有 rollout 失败归因于换手。
   物体位姿仍来自参考轨迹；OICM self-effect 重建并不是物理环境闭环成功的证据。
   证据：[src/task/CmDecoderv2/research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/diagnostic_tables.json](../../src/task/CmDecoderv2/research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/diagnostic_tables.json)。

**下一步建议（讨论方案，不是新的 final plan，也未执行）**

- 首先固定 Inspire 目标手当前状态，验证 decoder 的动作条件依赖：正确 Cm、相同物体/相近接触条件下
  另一真实动作的完整 Cm window、保持状态不动基线；置零完整 Cm 只作 OOD 辅助。真实 Cm 交换应把
  tokens 与对应 anchors 一起交换并保持时间顺序，以免留下未消融的动作通道。先比较单步和短时递归，
  用错误动作条件是否导致可解释的输出变化、正确条件是否提高误差指标判断。若仍有歧义，再训练相同预算的
  state-only decoder；不能把“推理时置零”称为已训练 state-only baseline。
- 随后建设小规模、可审计的跨手交换评估：目标手初始状态固定，分别输入同源 Cm、经物体状态/时间跨度/
  实际 effect/接触可达性审查的 MANO Cm，以及不同 effect 的负对照。物体坐标系与 anchor 变换必须明确，
  不以同名 sequence 自动建立正配对，不改正式 split。跨手成功以目标手可实现的运动、接触和物体任务效果
  判断，目标手未来 GT 不作为输入；同一个冻结 OICM 的评分仅作诊断，最终需要独立仿真/真实执行验证。
- 模型改动先用相同数据、训练预算与评估 stride 比较三项：现有 joint-loss Cm、去掉源手重建损失的 Cm、
  同一局部交互 encoder 绕过 Cm 压缩的 object-effect head。前两项判断原手重建监督的作用；后者只定位
  压缩/解码损失，不作为跨手表征候选。只有结果支持时再扩展成完整 2×2 消融或 effect/private 分支。
  object EPE 改善需要进一步通过交换评估验证，不能自动认定迁移提高。
- 表征目标建议为物体条件下的交互动作：保留任务 effect 与必要接触信息，把目标手运动学交给 decoder。
  不预设每个 slot 跨样本一一对应，不直接对未对齐 slot 做 L2；也不把低 source-classification accuracy
  或单步物体 flow 作为唯一目标，静止时的不同接触构型仍可能影响后续可执行动作。

**验证**

- `rg`/定向源码读取：核对 encoder、anchors、双重监督、Temporal-D2 输入及 loss；未修改运行实现。
- `python3` 标准库只读复核：读取上述 `metrics.jsonl`，按 source 使用 `statistics.fmean` 重算四个
  EPE 列并以 `math.isclose(abs_tol=1e-9)` 对照 `diagnosis.json`；全部通过，样本数 `3610/4060`。
  同次调用使用 `Counter` 核验 split/variant，断言 `len(parent_ids)==len(set(parent_ids))==630`；
  用 `hashlib.sha256` 分块读取 checkpoint 并与 classifier manifest 比较，通过；从 confusion matrix
  复算 `65/88`，与归档 accuracy 一致。没有执行新模型评估或建立新实验 run。
- `git diff --check -- docs/logs/activity_log.md` 和
  `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log docs/logs/activity_log.md --worktree --scope-prefix docs/logs/activity_log.md --check-links`：通过，最新条目 13 个本地链接均可导航。
  审计器按实现排除日志自身，报告 0 个其他变更路径；本轮日志差异另以 `git diff` 人工检查。
- 数值核对属于既有证据复核；不将其记为新的科研实验或更新 Task experiment 结论。

**回滚与规范反馈**

删除本活动条目即可回滚；所有开始时已有的工作区差异保留。本轮只读诊断与 L0 记录未遇到审批阻碍；
后续训练、配对评估与监督变体仍是建议，尚未形成新的执行计划。本轮未修改治理规则。

## 2026-09-03 22:03:39 +0800 — V1.2.15 共享运行追溯与测试治理提交边界

- activity_id: ACT-20260903-220339
- timestamp: 2026-09-03 22:03:39 +0800
- modification_version: V1.2.15
- type: governance / code / documentation / operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求将当前更改划定范围并分次提交；BaseRunner/manifest 收缩和测试目录治理此前均已逐项确认。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 2c4256d927a6c93b97ca3eb0d14df63227142ca9
- worktree_dirty: true（根目录运行快照和 CmDecoder 历史 modification 重复条目不纳入提交）
- scope: 共享 BaseRunner 运行产物职责、run manifest 精简、测试目录归属和相关治理文档；不修改科研模型、数据、GT、坐标系、split、checkpoint、output 或正在运行的训练进程。

**文件**

- [`.agents/`](../../.agents/) — 同步修改治理和实验运行 Skill 的 summary、manifest、测试范围及交接规则。
- [`AGENTS.md`](../../AGENTS.md) — 固化任务模式、测试归属、终态 activity 和路径导航合同。
- [`docs/`](../) — 更新文档导航、目录规范、交接清单、根架构/记忆和 V1.2.15 最终计划。
- [`src/base/`](../../src/base/) — 取消 BaseRunner 标准 `summary.json` 自动生成，并精简新 `run_manifest.json`。
- [`tests/`](../../tests/) — 增加根 legacy 测试归类清单，并补充 manifest/BaseRunner 回归断言。
- [`docs/logs/activity_log.md`](activity_log.md) — 登记共享变更、验证和本次提交边界。

**原因**

将已经批准并验证的共享治理/基础设施更改与 Task 实现、实验活动和生成产物分开提交，保证每个提交可独立审计和回滚；根目录运行快照不属于源码，废弃的 `modification_log.md` 不再接收重复新记录。

**验证**

- `python3 -m py_compile src/base/base_runner.py src/base/run_manifest.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q`：`320 passed, 3 skipped`；仅有既有弃用/兼容警告。
- `audit_diff.py --staged --check-links`：通过，覆盖 `14` 个 staged 变更路径，最新条目 `6` 个本地链接可导航；`git diff --cached --check`：通过。
- 工程回归不构成科研效果证据；`conclusion: N/A`。

**回滚**

可独立回退本批共享治理/BaseRunner 提交；此前的 CmDecoder 活动提交和 ObjectInteractionCm V1.2.1 Task 提交不在该回滚范围内，运行产物与训练进程无需处理。

## 2026-09-03 11:34:30 +0800 — V1.2.15 收缩运行追溯与取消 BaseRunner 标准 summary

- activity_id: ACT-20260903-113430
- timestamp: 2026-09-03 11:34:30 +0800
- modification_version: V1.2.15
- type: governance / code / documentation / diagnostic
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认取消 `summary.json`、保留 `metadata.json`、移除 manifest 中完整 `dataset_metadata`，并允许继续处理 `src/base/`。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（工作区另有用户/运行生成的未纳入本次范围的改动）
- run_status: COMPLETED
- conclusion: N/A（工程追溯与治理变更，不构成科研效果证据）
- scope: 共享 BaseRunner 运行产物职责、run manifest 精简、metadata 字段归属说明和相关回归测试；不修改模型、数据、GT、坐标系、split、checkpoint 内容或历史运行产物。

**文件**

- [`src/base/base_runner.py`](../../src/base/base_runner.py) — 移除 BaseRunner 对标准 `summary.json` 的自动写入，保留配置、metadata、manifest、metrics 和 train.log 既有流程。
- [`src/base/run_manifest.py`](../../src/base/run_manifest.py) — 新 manifest 不再内嵌完整 `dataset_metadata` 或 `dataset_split`，仅保留轻量静态 `contract`；旧 `write_run_summary` API 保留给历史/Task 专属调用方。
- [`tests/test_run_manifest.py`](../../tests/test_run_manifest.py) — 验证新 manifest 不重复展开 metadata，兼容旧 summary writer 测试继续保留。
- [`tests/test_overfit_diagnosis.py`](../../tests/test_overfit_diagnosis.py) — 增加 BaseRunner 新运行不产生 `summary.json` 的回归断言。
- [`AGENTS.md`](../../AGENTS.md)、[`.agents/skills/research-change-control/SKILL.md`](../../.agents/skills/research-change-control/SKILL.md)、[`.agents/skills/research-experiment-workflow/SKILL.md`](../../.agents/skills/research-experiment-workflow/SKILL.md) — 固化终态 activity、metadata_snapshot 和 summary 兼容边界。
- [`docs/目录规范.md`](../目录规范.md)、[`docs/ai_task_checklist.md`](../ai_task_checklist.md)、[`docs/plan/V1.2.15.md`](../plan/V1.2.15.md) — 同步 JSON 职责、终态导航、字段归属和已确认决定。
- [`docs/logs/architecture_log.md`](architecture_log.md)、[`docs/logs/repo_memory.md`](repo_memory.md) — 更新共享运行追溯事实。
- [`src/task/ObjectInteractionCm/docs/architecture/V1.1.md`](../../src/task/ObjectInteractionCm/docs/architecture/V1.1.md)、[`src/task/ObjectInteractionCm/docs/plan/V1.1.md`](../../src/task/ObjectInteractionCm/docs/plan/V1.1.md) — 清理当前 Task 规范中对标准 summary 的过时要求。
- [`docs/logs/activity_log.md`](activity_log.md) — 登记本次变更及验证入口。

**原因**

`summary.json` 与 activity、metrics、train.log 和 checkpoint 产生终态信息重复，且完整
`dataset_metadata` 与 `metadata.json` 重复。新运行将终态集中到 activity，manifest 只做小型 provenance
索引并引用 metadata snapshot；历史输出和显式 Task summary 不回写、不删除。

**验证**

- `python3 -m py_compile src/base/base_runner.py src/base/run_manifest.py`：通过。
- `python3 -m pytest -q tests/test_run_manifest.py tests/test_overfit_diagnosis.py`：`24 passed`。
- `python3 -m pytest -q tests/test_framework_contracts.py tests/test_base_runner_max_steps.py tests/test_base_runner_validation.py tests/test_checkpoint_compat.py tests/test_audit_diff.py`：`25 passed`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q`：`320 passed, 3 skipped`；保留环境弃用警告，不影响本次回归。
- `git diff --check`：通过；`audit_diff.py --worktree --check-links`：通过，覆盖 `14` 个本次变更路径，最新条目 `15` 个本地链接可导航。

**回滚**

回退本条列出的共享代码、规范文档和定向测试即可恢复旧的 BaseRunner summary 写入与 manifest 展开逻辑；不触碰已有运行目录、数据、cache 或 checkpoint。

## 2026-09-03 10:21:27 +0800 — V1.2.15 测试目录归属规范与根测试 legacy 分类

- activity_id: ACT-20260903-102127
- timestamp: 2026-09-03 10:21:27 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认采用“只建立规范和分类清单，暂不物理迁移旧 tests”的方案
- skills_used: research-change-control
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（CmDecoder ObjectInteractionCm 另有用户未提交改动，本次不纳入）
- scope: 测试目录所有权、验证范围和现有根 tests 的 legacy 分类；不移动、删除或改写任何现有测试文件，不改变 pytest.ini、tests/conftest.py 或科研语义

**文件**

- [tests/README.md](../../tests/README.md) — 记录未来 Task/process/shared/integration/governance 目录和当前 63 个根测试的逐文件归类。
- [docs/目录规范.md](../目录规范.md) — 增加测试目录、验证范围和根 legacy 规则。
- [docs/README.md](../README.md) — 增加测试规范导航。
- [AGENTS.md](../../AGENTS.md) — 增加测试规范加载入口、测试所有权和按影响范围选择验证的常驻要求。
- [.agents/skills/research-change-control/SKILL.md](../../.agents/skills/research-change-control/SKILL.md) — 要求按测试归属选择最窄验证集。
- [docs/plan/V1.2.15.md](../plan/V1.2.15.md) — 记录用户确认的测试分类边界。
- [docs/logs/activity_log.md](activity_log.md) — 记录本次治理事件。

**原因**

根 `tests/` 同时混合 Task、共享基础设施、数据处理和跨 Task 测试，导致 Task 小改动默认触发无关测试。
本次先建立所有权和验证范围规范，保留旧文件路径以避免迁移引入 import、pytest 收集和 fixture 风险。

**验证**

- 逐一核对根目录现有 `63` 个 `test_*.py`，分类清单无遗漏或重复。
- 未修改 `pytest.ini`、`tests/conftest.py` 和任何根测试内容；现有 `pytest` 行为保持不变。
- 分类核对：`63` 个根测试无遗漏、无重复。
- `audit_diff.py --worktree --check-links`（仅本次治理路径）：覆盖 `6` 个变更路径，`7` 个本地链接可导航；
  `git diff --check`：通过。
- 本次为治理/文档规范，不构成科研效果证据；`conclusion: N/A`。

## 2026-09-02 15:30:36 +0800 — V1.2.15 提交治理与运行入口改动

- activity_id: ACT-20260902-153036
- timestamp: 2026-09-02 15:30:36 +0800
- modification_version: V1.2.15
- type: governance / code / documentation / operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求提交本轮我们共同完成的相关改动；仅提交已确认属于本轮治理和 Cm/CmDecoder 入口的路径
- skills_used: research-change-control, research-experiment-workflow
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: Ref2Dex 治理闭环、递归活动记录、配置/运行追溯、Cm/CmDecoder 研究入口和链接审计；不提交被忽略的 output、cache、数据和 checkpoint

**文件**

- [.agents/](../../.agents/) — 治理与实验 Skill、审计合同和审计脚本。
- [AGENTS.md](../../AGENTS.md) — 仓库常驻规则、任务模式、交接和路径导航合同。
- [docs/](../) — 文档入口、版本指针、目录/版本规范、plan、日志迁移和兼容入口删除。
- [src/base/](../../src/base/) — 配置版本字段、运行 manifest 和终态 summary 追溯支持。
- [src/task/Cm/](../../src/task/Cm/) — Cm 配置、研究入口和局部活动/事实记录迁移。
- [src/task/CmDecoder/](../../src/task/CmDecoder/) — CmDecoder 配置、rollout 入口、实验记录和活动导航。
- [src/task/correspondence_ptv3_v2/](../../src/task/correspondence_ptv3_v2/) — Task 活动/事实记录迁移。
- [tests/](../../tests/) — run manifest 回归和 activity 链接审计测试。

旧中文入口和历史 `status_log.md` 的删除属于上述 `docs/`、Task 日志迁移范围；没有把其他用户未确认的
实验产物、数据、cache、checkpoint 或 output 加入提交。

**原因**

将本轮已经确认的治理、运行追溯、Task 入口和链接导航改动形成可回滚的 Git 提交，保留工作区中不属于
本轮范围的内容不被带入。

**验证**

- 提交前检查 `git status`、完整 unstaged diff 和 staged diff；仅使用显式路径 stage。
- `python3 -m pytest -q tests/test_audit_diff.py`：`5 passed`；审计脚本 `py_compile` 通过。
- `audit_diff.py --staged --check-links`：根级本次提交范围的本地链接均可导航；`git diff --cached --check` 通过。
- 本次为治理/工程提交，不构成科研效果证据；`conclusion: N/A`。

## 2026-09-02 11:43:51 +0800 — V1.2.15 activity Markdown 链接诊断

- activity_id: ACT-20260902-114351
- timestamp: 2026-09-02 11:43:51 +0800
- modification_version: V1.2.15
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户请求诊断根 activity 的 Ctrl+Click 导航；本次不修改链接或科研内容
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 只读解析 [docs/logs/activity_log.md](activity_log.md) 的 Markdown 本地链接；不改变代码、配置、数据、产物或运行状态

**文件**

- [docs/logs/activity_log.md](activity_log.md) — 作为被检查的根级 activity 文档。
- [.agents/skills/research-change-control/scripts/audit_diff.py](../../.agents/skills/research-change-control/scripts/audit_diff.py) — 使用既有链接审计器核对最新条目。

**原因**

用户反馈根 activity 中的导航无法 Ctrl+Click，需要区分 Markdown target 错误与 IDE 链接服务问题。

**验证**

- 按文档所在目录解析相对 target；当前文件中的本地链接均指向仓库内已存在文件或目录，未发现失效 target。
- 例如 `../../AGENTS.md` 实际解析为仓库根 [AGENTS.md](../../AGENTS.md)；CmDecoder 的当天产物链接也通过 `--check-links`。
- 结论：Markdown 写法和 target 解析正常；若连 ASCII 链接都不能点击，优先检查 IDE 的 Markdown 插件、项目根打开方式和链接导航设置。
- 本次为环境/编辑器诊断，不构成科研结论；`conclusion: N/A`。

## 2026-09-02 11:32:42 +0800 — V1.2.15 产物导航与链接审计

- activity_id: ACT-20260902-113242
- timestamp: 2026-09-02 11:32:42 +0800
- modification_version: V1.2.15
- type: governance / documentation / tooling
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确批准补充链接规则，并要求 AI 回复中的路径与前后文本留空格
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 最新 activity 的本地链接审计、产物导航合同和 CmDecoder 当日产物链接修正；不涉及模型、配置、数据、GT、坐标、split、checkpoint 内容、训练进程或科研结论

**文件**

- [AGENTS.md](../../AGENTS.md) — 固化仓库相对显示路径、可点击导航、回复路径空格和交接链接校验要求。
- [.agents/skills/research-change-control/SKILL.md](../../.agents/skills/research-change-control/SKILL.md) — 将链接格式和 `--check-links` 纳入通用交接流程。
- [.agents/skills/research-change-control/references/audit-contract.md](../../.agents/skills/research-change-control/references/audit-contract.md) — 说明最新条目选择、链接检查范围和 `PENDING` 例外。
- [.agents/skills/research-change-control/scripts/audit_diff.py](../../.agents/skills/research-change-control/scripts/audit_diff.py) — 按 `timestamp` 选择最新活动并检查本地 Markdown 目标。
- [tests/test_audit_diff.py](../../tests/test_audit_diff.py) — 覆盖乱序活动、有效/失效链接、运行中待生成目标和外部链接。
- [docs/目录规范.md](../目录规范.md) — 定义各类运行必须提供的关键产物导航。
- [docs/ai_task_checklist.md](../ai_task_checklist.md) — 增加交接命令和回复空格检查项。
- [docs/plan/V1.2.15.md](../plan/V1.2.15.md) — 记录用户对链接审计与回复格式的补充批准。
- [src/task/CmDecoder/docs/logs/activity_log.md](../../src/task/CmDecoder/docs/logs/activity_log.md) — 修正当天 rollout 少一级 `..` 的目标，并补全仓库相对产物标签、目录、manifest、summary 和 checkpoint 导航。
- [docs/logs/activity_log.md](activity_log.md) — 记录本次治理事件和验证入口。

**原因**

现有合同虽然要求可点击路径，但缺少自动检查，且部分 CmDecoder 产物只写简称或使用错误的相对层级，导致 IDE 无法跳转。将显示路径与 Markdown 目标分工并加入本地审计，可让用户从 activity 递归进入运行目录和关键证据。

**验证**

- `python3 -m py_compile .agents/skills/research-change-control/scripts/audit_diff.py`：通过。
- `python3 -m pytest -q tests/test_audit_diff.py`：`5 passed`。
- 根 activity 定向 `--worktree --check-links`：覆盖 `8` 个变更路径，`10` 个本地链接均可导航。
- CmDecoder 最新 activity 定向 `--check-links`：`7` 个本地链接均可导航；8 份 rollout manifest/summary JSON 可解析。
- `git diff --check`：通过。
- 工程文档与审计工具修改不构成科研效果证据；`conclusion: N/A`。

**回滚**

仅需回退本条“文件”列出的治理文档、审计脚本、定向测试和 CmDecoder 活动链接；不需要处理任何运行产物或训练状态。

## 2026-09-01 22:53:59 +0800 — V1.2.15 删除中文兼容入口

- activity_id: ACT-20260901-225359
- timestamp: 2026-09-01 22:53:59 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求删除 `docs/版本线与AI行为分类.md` 和 `docs/AI交接清单.md`
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 删除根级中文兼容文档入口并清理现行导航；不涉及代码、数据、模型、GT、坐标、split、checkpoint 或运行产物

**文件**
- `docs/版本线与AI行为分类.md`、`docs/AI交接清单.md` — 删除中文兼容跳转页；canonical 规范分别保留在 `docs/modification_policy.md` 和 `docs/ai_task_checklist.md`。
- `docs/README.md`、`docs/logs/activity_log.md` — 移除现行兼容入口并记录本次删除事件；历史活动中的旧文件名改为非链接历史说明。
- `docs/` — 其他既有根级文档和历史审计记录未做语义迁移。

**原因**
用户确认不再保留两个中文兼容入口，仓库从此只使用 ASCII canonical 文件名。现行导航不再指向已删除路径，历史记录保留文件名以便理解过去的目录状态。

**验证**
- `git diff --check`：通过。
- `docs/README.md`、`AGENTS.md` 和根级活动记录的 Markdown 链接扫描：缺失 `0`。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log docs/logs/activity_log.md --worktree --scope-prefix docs --scope-prefix AGENTS.md`：通过。

<a id="act-20260901-221137"></a>

## 2026-09-01 22:11:37 +0800 — V1.2.15 文档路由与 canonical 命名

- activity_id: ACT-20260901-221137
- timestamp: 2026-09-01 22:11:37 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认补充 AGENTS 文档触发路由，并要求重新命名版本政策和 AI 交接文档
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 仓库文档加载规则、规范权威层级和根级说明文档 canonical 路径；不涉及代码、数据、模型、GT、坐标、split、checkpoint 或运行产物

**文件**
- [`AGENTS.md`](../../AGENTS.md) — 增加规范文档的条件式加载、权威层级和 canonical 路径导航；收窄重复的目录/manifest 说明。
- [`docs/modification_policy.md`](../modification_policy.md)、[`docs/ai_task_checklist.md`](../ai_task_checklist.md) — 使用清晰的 ASCII canonical 文件名并更新职责说明。
- `docs/版本线与AI行为分类.md`、`docs/AI交接清单.md` — 当时的中文兼容入口，已由本次删除操作移除。
- [`docs/README.md`](../README.md)、[`docs/项目总览.md`](../项目总览.md)、[`docs/plan/V1.2.15.md`](../plan/V1.2.15.md)、[`docs/logs/repo_memory.md`](repo_memory.md) — 更新导航、计划和当前事实引用。
- [`docs/`](../) — 本次变更涉及的其他根级治理文档引用保持历史记录可追溯。

**原因**
将“AI 什么时候必须读取哪份规范”写入常驻的 `AGENTS.md`，同时把版本政策、目录规范和交接清单分层，避免普通文档未加载造成规则遗漏，也避免 `AGENTS.md` 与政策正文重复漂移。新路径遵循 ASCII canonical 命名；旧路径保留跳转以兼容历史日志。

**验证**
- `git diff --check`：通过。
- 旧路径引用扫描：仅保留兼容入口和历史审计记录；当前导航与规范均指向 canonical 路径。
- Markdown 相对链接扫描：根级文档和兼容入口均可解析，缺失链接 `0`。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log docs/logs/activity_log.md --worktree --scope-prefix docs --scope-prefix AGENTS.md`：通过，覆盖 `8` 个变更路径。

<a id="act-20260901-211226"></a>

## 2026-09-01 21:12:26 +0800 — V1.2.15 治理闭环切换

- activity_id: ACT-20260901-211226
- timestamp: 2026-09-01 21:12:26 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认“按照你想的来，直接修改吧”，并批准任务模式、current pointer、activity 时间线、链接合同和命名收口方案
- skills_used: research-change-control, research-experiment-workflow
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 根级治理合同、活动记录迁移、Cm 重点入口和审计工具；不涉及模型、数据、GT、坐标、split、checkpoint 或训练变量

**文件**
- [`AGENTS.md`](../../AGENTS.md) — 增加任务模式、activity 唯一时间线、current pointer、运行反馈和可点击路径合同。
- [`docs/current_versions.yaml`](../current_versions.yaml) — 新增各作用域的 `modification_version` 指针。
- [`docs/plan/V1.2.15.md`](../plan/V1.2.15.md) — 将治理方案定稿并记录已确认决定。
- [`docs/README.md`](../README.md) — 新增 ASCII 文档目录稳定入口，保留中文说明文件兼容路径。
- [`docs/ai_task_checklist.md`](../ai_task_checklist.md)、[`docs/目录规范.md`](../目录规范.md)、[`docs/modification_policy.md`](../modification_policy.md)、[`docs/项目总览.md`](../项目总览.md) — 收口入口、目录、版本和机器记忆边界；中文旧名仅为当时的历史文件名。
- [`docs/logs/repo_memory.md`](repo_memory.md) — 更新根级活动入口。
- [`.agents/skills/research-change-control/SKILL.md`](../../.agents/skills/research-change-control/SKILL.md)、[`.agents/skills/research-change-control/references/audit-contract.md`](../../.agents/skills/research-change-control/references/audit-contract.md)、[`.agents/skills/research-change-control/references/change-levels.md`](../../.agents/skills/research-change-control/references/change-levels.md)、[`.agents/skills/research-change-control/scripts/audit_diff.py`](../../.agents/skills/research-change-control/scripts/audit_diff.py)、[`.agents/skills/research-experiment-workflow/SKILL.md`](../../.agents/skills/research-experiment-workflow/SKILL.md) — 同步任务模式、活动字段、manifest/summary 职责和差异审计。
- [`docs/logs/activity_log.md`](activity_log.md)、[`src/task/Cm/docs/logs/activity_log.md`](../../src/task/Cm/docs/logs/activity_log.md)、[`src/task/CmDecoder/docs/logs/activity_log.md`](../../src/task/CmDecoder/docs/logs/activity_log.md)、[`src/task/correspondence_ptv3_v2/docs/logs/activity_log.md`](../../src/task/correspondence_ptv3_v2/docs/logs/activity_log.md) — 将 `status_log.md` 从切换点移交为 activity 时间线，保留旧历史内容。
- [`docs/logs/machine_memory.md`](machine_memory.md)、[`src/task/Cm/docs/README.md`](../../src/task/Cm/docs/README.md)、[`src/task/Cm/research/README.md`](../../src/task/Cm/research/README.md) — 明确机器事实和 Cm 入口。
- [`src/task/Cm/docs/logs/repo_memory.md`](../../src/task/Cm/docs/logs/repo_memory.md)、[`src/task/Cm/docs/logs/experiment_log.md`](../../src/task/Cm/docs/logs/experiment_log.md)、[`src/task/Cm/docs/logs/decision_log.md`](../../src/task/Cm/docs/logs/decision_log.md) — 更新 Cm 相关导航。
- [`src/task/CmDecoder/docs/logs/repo_memory.md`](../../src/task/CmDecoder/docs/logs/repo_memory.md)、[`src/task/CmDecoder/docs/logs/experiment_log.md`](../../src/task/CmDecoder/docs/logs/experiment_log.md) — 更新 CmDecoder 相关导航。
- [`src/task/correspondence_ptv3_v2/docs/logs/repo_memory.md`](../../src/task/correspondence_ptv3_v2/docs/logs/repo_memory.md) — 更新 correspondence 相关导航。
- [`docs/logs/architecture_log.md`](architecture_log.md)、[`src/task/Cm/docs/logs/architecture_log.md`](../../src/task/Cm/docs/logs/architecture_log.md)、[`src/task/CmDecoder/docs/logs/architecture_log.md`](../../src/task/CmDecoder/docs/logs/architecture_log.md)、[`src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md`](../../src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md) — 将架构记录头部收窄为更新时间。
- [`src/base/base_config.py`](../../src/base/base_config.py)、[`src/base/run_manifest.py`](../../src/base/run_manifest.py)、[`src/base/base_runner.py`](../../src/base/base_runner.py)、[`src/base/__init__.py`](../../src/base/__init__.py) — 将运行版本字段收口到 `modification_version`，并为 train/eval 增加终态 `summary.json`。
- [`src/task/Cm/src/config.py`](../../src/task/Cm/src/config.py)、[`src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml`](../../src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml)、[`src/task/Cm/research/dense_cache_v1_1_1/experiment.yaml`](../../src/task/Cm/research/dense_cache_v1_1_1/experiment.yaml)、[`src/task/Cm/research/hand_flow_decoder/experiment.yaml`](../../src/task/Cm/research/hand_flow_decoder/experiment.yaml)、[`src/task/Cm/research/tsne_slots/experiment.yaml`](../../src/task/Cm/research/tsne_slots/experiment.yaml)、[`src/task/Cm/research/tsne_slots/run.py`](../../src/task/Cm/research/tsne_slots/run.py) — 当前 Cm 配置/实验定义只使用 `modification_version`。
- [`src/task/CmDecoder/config.py`](../../src/task/CmDecoder/config.py)、[`src/task/CmDecoder/current_cm_point_config.py`](../../src/task/CmDecoder/current_cm_point_config.py)、[`src/task/correspondence_ptv3_v2/config.py`](../../src/task/correspondence_ptv3_v2/config.py) — 当前 Task 配置只使用 `modification_version`。
- [`tests/test_run_manifest.py`](../../tests/test_run_manifest.py) — 覆盖旧 guidance/plan/operation 字段不进入新 manifest，以及终态 summary 合同。

**原因**
将任务按只读诊断、已有实验运行、实际变更和治理变更分流；以 activity 记录事件、以 `current_versions.yaml` 提供当前指针，并让每次交接都能跳转到计划、运行和产物。旧 `modification_log.md` 仅作为历史审计，不回写旧条目。

**验证**
- `git diff --check`：通过。
- `python3 -m py_compile .agents/skills/research-change-control/scripts/audit_diff.py src/base/base_config.py src/base/run_manifest.py src/base/base_runner.py`：通过。
- `python3` + PyYAML `docs/current_versions.yaml` 结构校验：通过，只有 `modification_version` 字段。
- `audit_diff.py --worktree`（根级及 Cm/CmDecoder/correspondence 三个作用域）：通过，分别覆盖 `39/12/5/3` 个变更路径。
- 变更文档 Markdown 相对链接扫描：`14 files; missing 0`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q`：`311 passed, 3 skipped`。
- 系统 `python3` 的全量收集因环境缺少 `smplx`/`viser` 失败；已切换项目 `graspenv` 完成完整回归，属于环境差异而非仓库测试回归。
- 工程验证不代表任何科研结论：`conclusion: N/A`。

## V1.2.13 — 删除根 Component 目录（2026-09-01）

- category: `governance`、`operation`、`documentation`
- 状态: 根 `components/` 已彻底删除；仓库不再提供默认 manifest 根或通用示例，Task 继续直接使用各自 Python 实现。
- 兼容边界: `src/base/` 的通用协议代码和 `tools/researchctl.py` 暂时保留；工具只接受显式传入的外部 manifest 根，不再猜测仓库路径。
- 保护边界: 未修改 Task 训练配置、数据、cache、checkpoint、output 或运行进程。
- 验证: 根目录/活动引用扫描无残留；定向测试 `18 passed`；全量 pytest `310 passed, 3 skipped`；`git diff --check` 通过。

## V1.2.11 — Task-local 外部资产入口（2026-08-31）

- category: `governance`、`operation`
- 状态: 根 `assets/` 不再承载新资产；Cm 的外部资产入口下沉到 `src/task/Cm/assets/`，根目录只保留被忽略的兼容软链接。
- 保护边界: 未移动大型 checkpoint、数据、cache 或运行中的进程；DenseToken 真实文件仍在旧 Cm 目录。
- 验证: 新旧 DenseToken 路径解析到同一文件，Cm 配置导入和全量 pytest 通过。

## V1.2.12 — 撤出两个 Task 的 Component、data 和 registry 入口（2026-09-01）

- category: `governance`、`operation`、`documentation`
- 状态: 已删除 CmDecoder 与 correspondence_ptv3_v2 的 Task-local `components/`、`data/`、`registry/` 及根级兼容软链接；两个配置恢复为直接选择各自 Python Task 实现，并继续使用根级 `data/`、`dataset/`、`third_party/`。
- 保护边界: 未移动或删除真实数据、cache、checkpoint、output 或运行进程；历史日志和用户指导保留。
- 文档: 当前目录规范、交接清单和根架构入口不再把 Task Component/registry 作为现行结构；历史日志条目不改写。
- 验证: Task 目录入口扫描、配置导入、定向测试、全量 pytest 与 `git diff --check`。

## V1.2.10 — 取消 Cm 路径索引（2026-08-31）

- category: `governance`、`data`、`documentation`
- 状态: Cm 不再维护 Task 内路径 registry 或便捷数据软链接；配置直接声明外部数据、cache 和资产路径，实体数据未移动。
- 验证: Cm 配置/loader 回归和全量 pytest 通过。

## V1.2.9 — 撤销隔离试验线（2026-08-31）

- category: `governance`、`code`、`documentation`
- 状态: 已删除 Cm 隔离试验目录、Cm Task 的清单/适配器入口，并清理活动文档中的失效引用；数据、cache、checkpoint、output、outputs 和既有训练实现保留。
- 规则: AGENTS.md 与保留 Skill 已恢复为不含该试验线的通用维护规则；后续历史条目仍保留原始记录，避免重写审计历史。
- 验证: AGENTS/Skill 文本扫描无相关术语；Cm 配置和 loader 定向回归通过。

## V1.2.8 — 声明层与运行层闭合（2026-08-31）

- category: `architecture`、`code`、`governance`、`experiment`、`documentation`
- 状态: 已完成；`check-task-config` 强制核验 Task config↔`components.json`↔manifest，CmComponent 启动核验 Python Pipeline 与 `pipeline.yaml` 拓扑，YAML 保持声明/启动断言定位。
- 运行时: `TrainableComponent` 提供 build、参数组前缀和 `id@version` checkpoint 命名空间；首个 batch 做 materialized Tensor shape/dtype/finite/batch 对齐检查，后续步骤走轻量 schema 路径。
- 清单修正: Cm registry 补登记 `ref2dex.correspondence.ptv3_v2@2.1.0/dense_encoder` 并与旧配置的 `cm_task_runner` role 对齐。
- 验证: 全量 pytest `329 passed, 3 skipped, 22 warnings`；两种坐标配置 CPU max_steps=2 train/eval 通过；证据为 `outputs/cmcomponent/cm_component_smoke_20260831_172705/` 与 `outputs/cmcomponent/cm_component_smoke_object_pose_20260831_172708/`；两个最新 run manifest 均记录 `V1.2.8`、递归 component_tree 和 parameter_prefix。
- 结论: `SUPPORTED: declaration/runtime audit and trainable lifecycle smoke`；仅为工程 smoke，不构成真实数据或坐标策略科研结论。用户指导与架构快照未修改。

## V1.2.7 — Component 晋升与内部模块边界（2026-08-31）

- category: `governance`、`architecture`、`documentation`
- 状态: 已完成；明确 `helper`、Task-local module、可发现 Component 三层边界，补充小开关的适用条件和必须晋升 Component 的语义/生命周期触发条件。
- 规则: Component 不是文件拆分单位；内部可拆多个类/模块而不增加 manifest。只有出现独立替换、版本、资源、生命周期、跨 Task 复用、并行回滚或公共 Contract/数据语义边界时才建立 Component。
- 保护: 用户指导、架构快照、运行代码、数据/cache/checkpoint/output 和运行中的旧 Cm 进程未修改。
- 结论: `SUPPORTED: explicit component-promotion boundary and anti-over-splitting rule`；本轮无科研实验结论。

## V1.2.6 — 根节点唯一声明与递归配置解耦（2026-08-31）

- category: `governance`、`architecture`、`code`、`experiment`、`documentation`
- 状态: 已完成；CmComponent 配置只声明 `coordinate_transform` 与 `pipeline` 两个根 Component，encoder/head 仅在 `pipeline.children` 中选择；外层 Pipeline 通过父 role 解析子组件。
- 追溯: `component_tree` 成为嵌套选择的规范视图，`run_manifest.components` 由树自动生成 child-first 扁平兼容视图，配置不再重复登记子组件。
- 验证: 两种坐标配置均完成 CPU `max_steps=2` smoke train/eval；定向 pytest `19 passed`；全量测试 `325 passed, 3 skipped`；最新证据为 `outputs/cmcomponent/cm_component_smoke_20260831_143911/` 与 `outputs/cmcomponent/cm_component_smoke_object_pose_20260831_143914/`。
- 结论: `SUPPORTED: nested component selection, stable outer Pipeline wiring, and provenance`; smoke 不构成真实数据效果结论。

## V1.2.5 — 递归 Component 选择与运行追溯（2026-08-31）

- category: `governance`、`architecture`、`code`、`experiment`、`documentation`
- 状态: 已完成；`children` 选择、父 role 解析和递归 `component_tree` 已接入 CmComponent，未更新用户指导或架构快照。
- 验证: 两种坐标配置均完成 CPU `max_steps=2` smoke train/eval；CmComponent + run manifest 定向测试 `18 passed`；全量测试 `324 passed, 3 skipped`。
- 证据: `outputs/cmcomponent/cm_component_smoke_20260831_131246/` 与 `outputs/cmcomponent/cm_component_smoke_object_pose_20260831_131249/`。
- 结论: `SUPPORTED: recursive component selection/wiring and provenance`；不构成真实数据效果结论。

## V1.2.1 — 版本线与 Task Component 治理收口（2026-08-30）

- category: `governance`、`architecture`、`documentation`
- 状态: 已完成本轮目录、清单、运行追溯和交接规则改造；没有修改用户指导文档，也没有创建或覆盖架构快照。
- 版本关系: `V1.2` 为已定稿执行计划；`V1.2.1` 为本次细分操作。后续并行实验沿用同一 `V1.2` plan，并分配新的 `V1.2.k`。
- Component: Task 专属 manifest 已迁移至 `src/task/<Task>/components/`，旧 `components/ref2dex/*` 保留相对软链接；各 Task 新增 `components.json` 清单。
- 数据路径: 各 Task 新增 `registry/data_paths.json` 和被忽略的 `external_paths.json` 覆盖入口，Task 内 `data/` 仅保留相对软链接，不复制外部数据。
- 运行追溯: 新运行的 `run_manifest.json` 记录版本线、plan/guide、细分操作、操作类别、Component registry 和解析后的 Component 清单；历史 manifest 不回写。
- 架构规则: `architecture_log.md` 继续作为过渡导航和历史事实；只有用户明确发出架构更新指令时才建立/更新 `docs/architecture/Vn.m.md`。
- 下一步: 运行中的 Cm 训练完成后，按用户指令决定是否创建下一小版本或架构快照；普通状态刷新不改变 `k`。

## V1.2.2 — CmComponent 隔离 Task 建立（2026-08-30）

- category: `architecture`、`code`、`governance`
- 状态: 已建立独立的 `src/task/CmComponent/`，作为 Component Pipeline 的实验边界；旧 Cm、数据/cache、checkpoint 和运行进程保持不变。
- 当前计划: `src/task/CmComponent/docs/plan/V1.2.md`（final）；本次操作使用 `V1.2.2`，不创建新的大版本。
- 组件: Dense encoder 和 Cm head 已实现为真实 Component，Pipeline/Runner 负责显式组合；当前采用 adapter-first，下一步先做固定 batch parity。
- 风险: 新 Task 仍复用旧 Cm 的 dataset/loss/metric 生命周期，若要进一步拆分 head 内部，必须继续在新 Task 内进行并补充兼容测试。

## V1.2.3 — CmComponent 独立插拔式训练 smoke（2026-08-31）

- category: `architecture`、`code`、`experiment`
- 状态: 已完成独立 Component、Task-local synthetic 数据契约和 BaseRunner 训练/评估闭环；定向测试 `9 passed`。
- 组件: `PointFeatureEncoderComponent`、`SlotFlowHeadComponent`、`CmComponentPipeline` 均为独立 `Component`，清单版本统一为 `2.0.0`；Pipeline 拓扑为 `dense_encoder -> cm_head`。
- 训练证据: CPU `max_steps=2` smoke 已生成 `outputs/cmcomponent/cm_component_smoke_20260831_111831/`，包含 run manifest、配置、metadata、metrics 和 latest/best checkpoint；使用 latest checkpoint 的 test eval 已通过。
- 兼容边界: CmComponent Python/config 不再导入或继承旧 `src/task/Cm`；新 checkpoint 不承诺旧 Cm checkpoint 兼容。旧 Cm 代码、数据、cache、checkpoint、output 和训练进程未改动。
- 结论: `SUPPORTED: wiring/training lifecycle`；smoke 不代表真实数据上的科学效果。下一步需另建小版本 plan 接入真实数据或进行并行组件实验。

## V1.2.4 — 坐标策略 Component 化（2026-08-31）

- category: `governance`、`architecture`、`code`、`data`、`experiment`
- 状态: 已完成；`hand_root_t` 与 `object_pose_t` 已成为同一 `coordinate_transform` 角色下的两个独立 Component。
- 规则: 已将“出现第二种候选实现即提升为 Component”“每个运行 role 恰好选择一个实现”“可发现 manifest 统一使用 `component.yaml`”“新增/替换组件必须有清单与合同测试”“耦合处只依赖稳定接口”“YAML/JSON 只选择不执行”和“完成后报告规范限制”写入 `AGENTS.md`。
- 验证: 两种配置均完成 CPU smoke train/eval；CmComponent 定向测试 `14 passed`；全量测试 `323 passed, 3 skipped`；最新两个 run manifest 均记录坐标 Component、版本、pose source、source/output frame 和完整 Component 类别。
- 结论: `SUPPORTED: coordinate-component wiring/training lifecycle`；不构成真实数据效果比较。旧 Cm 代码、数据/cache/checkpoint/output 和进程未修改。

## 当前状态

- 当前阶段 / 指导: 已以 `oyx` 为主完成 `feature/hand-pca-perturbation` 的内容整合、递归 repo/machine memory 重构和全量回归验证。
- 当前进行中: Cm 混合 GRAB+Inspire-F1 手流重建当前运行于 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_185013`（torchrun world size=3）；旧 epoch 29 / step `124410` 仍作为历史恢复锚点。CmDecoder 三卡 EXP-022 运行于 GPU0/1/2，当前约 step `86500` / epoch 19，validation best=`1.4249 mm`（epoch 15），预计剩余约 `5.5 h`；该 best 已在 held-out test episode 完成 EXP-025 rollout，point EPE=`6.896/11.415 mm`（mean/final），短程反馈泛化尚可但仍有累积误差。Cm 已完成破坏性目录迁移：真实实现位于 `src/`、`dataset/`、`visualization/`，配置位于 `configs/{active,archive}`，旧顶层 Python/YAML 入口已移除。
- 最近可靠结论: feature 分支的 HOCap subject_1 Stage 3 已生成 28 个文件、23,896 个 frame samples并通过 loader 核验；Cm mixed 两条 run 已分别保留 epoch 48/40 的最近完整 checkpoint，详细指标见 Cm 状态与实验日志。
- 阻塞 / 风险: ARCTIC 全量 Stage 2/3 的最终产物状态仍需独立核验；HOCap 仅是外部测试集。合并代码已通过全量测试，但尚未在真实全量外部数据上重新导出。
- 下一步: 等待 CmDecoder GRAB→ARCTIC MANO 泛化结果，同时等待 Cm C=32/C=64 完成可比的中长程 validation，再决定是否恢复 correspondence 训练。
- 证据与相关文档: 全量 pytest `311 passed, 3 skipped`；[历史修改记录](modification_log.md)、[架构记录](architecture_log.md)、[correspondence 活动](../../src/task/correspondence_ptv3_v2/docs/logs/activity_log.md)、[Cm 活动](../../src/task/Cm/docs/logs/activity_log.md)。
- 2026-08-30 框架更新: `BaseRunner` 在 train/eval 启动时生成不会覆盖历史尝试的 `run_manifest.json`（续跑/评估使用时间戳文件）；`ref2dex.run.v1` 记录 Git 状态、输入 manifest/cache 引用、文件基本信息、数据合同、seed 和初始 checkpoint，不计算加密 hash。Cm loader 对坐标根做全量一致性检查。

## 2026-08-30 目录规范兼容迁移

- Cm 的 t-SNE 诊断已迁入 `src/task/Cm/research/tsne_slots/`；旧 `tsne_slots.py` 和模块调用方式保留为软链接/兼容入口。
- 新增统一的研究实验包格式、`experiment.yaml`、研究输出规则、Manifest 职责说明和 AI 交接清单。
- 新增 `assets/checkpoints/densetoken` 过渡软链接，实际 checkpoint 文件仍保留在旧目录，未影响当前运行。
- 本次没有移动或删除 `data/`、`output/`、`outputs/`、checkpoint 或历史 `results/`；根 `output/` 仅标记为历史兼容路径。
