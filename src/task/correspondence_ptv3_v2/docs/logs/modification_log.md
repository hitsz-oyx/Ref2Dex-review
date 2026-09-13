# Modification log

## 2026-09-01 — V1.2.12 撤出 Task-local Component/data/registry

- change_level: L3（破坏性目录治理与数据入口迁移）
- approval: user-approved（用户明确确认彻底删除 Component、Task data/registry，并使用根空间）
- skills_used: `research-change-control`
- branch: `feature/modular-component-runtime`
- version: `V1.2.12`（plan: `docs/plan/V1.md`, final）
- category: `governance`、`operation`、`documentation`
- post-commit: 未提交；真实数据、cache、checkpoint、output 和评估产物未触碰
- scope: `src/task/correspondence_ptv3_v2/components/`、`data/`、`registry/`、配置和当前目录说明

**文件**

- 删除 Task-local Component 清单、数据软链接、路径 registry 及根级兼容软链接。
- `config.py` 移除 Component 选择并更新细分版本；新增 V1 执行计划。
- 状态、记忆和根/Task 文档同步当前入口，历史实验记录保留。
- `src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md` — 标记当前入口边界并区分历史架构描述。

**验证**

- 入口扫描无残留；correspondence 配置导入成功。
- 全量 pytest 与 `git diff --check` 通过。

## 2026-08-31 — 校准 correspondence Task component 声明

- change_level: L2（公共声明合同；未改变训练、数据、GT、坐标系或 checkpoint 语义）
- approval: user-approved（用户要求修理 `correspondence_ptv3_v2` component）
- skills_used: `research-change-control`
- branch: `feature/modular-component-runtime`
- post-commit: 未提交
- scope: task 内部组件清单与合同测试
- version: `V1.2.1` 操作上下文；组件声明版本 `2.1.1`

**文件**

- `src/task/correspondence_ptv3_v2/components/component.yaml` — 将过时的 token/prior 输出改为当前模型实际返回的 cross-edge 与 hand-contact 预测端口，补充法向输入、Stage 3 schema 和可用坐标系声明。
- `src/task/correspondence_ptv3_v2/components/components.json` — 将角色从 `dense_correspondence_encoder` 校准为 `correspondence_predictor`，并登记组件版本 `2.1.1`。
- `src/task/correspondence_ptv3_v2/config.py` — 与 Task 选择清单同步组件版本和角色。
- `tests/test_component_registry.py` — 增加 correspondence 输出合同和可选 hand-contact head 的回归断言。

**原因**

原 manifest 仍声明 `object_tokens`、`hand_tokens` 和 `hand_contact_prior`，但 live `StaticHOCPTv3V2.forward()` 已返回 `pred_cross_random_*`、`pred_cross_contact_aux_*` 和可选 `pred_hand_contact_*`；同时角色名只描述 encoder，无法反映实际预测任务。此次只修正文档/清单合同，不新增组件运行时，也不改变模型实现。

**验证**

- `PYTHONPATH=. python3 tools/researchctl.py check-task-config src/task/correspondence_ptv3_v2/config.py` — 通过。
- `PYTHONPATH=. python3 tools/researchctl.py check src/task/correspondence_ptv3_v2/components/component.yaml --resolve-entrypoints` — 通过。
- `PYTHONPATH=. python3 -m pytest -q tests/test_component_registry.py tests/test_framework_contracts.py tests/test_correspondence_ptv3_v2.py` — `39 passed, 3 warnings`。

## 2026-08-24 — 合入 hand PCA 分支并建立递归 memory

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 跨 task 数据处理

**文件**

- `src/task/correspondence_ptv3_v2/`、`process/{ARCTIC,ContactPose,GRAB,HOCap,common}/` 及相关测试 — 从 feature 分支合入 hand PCA perturbation、MANO reconstruction、外部数据转换和评估实现。
- `docs/logs/repo_memory.md` — 提炼跨机器成立的数据入口与历史边界。
- `docs/logs/machine_memory.md` — 保存当前服务器 NAS 映射并由 Git 忽略。
- `docs/logs/repo_notes_log.md` — 删除；当前状态保留在 `status_log.md`，架构事实保留在 `architecture_log.md`。
- `tests/test_hand_pca_perturbation.py`、`tests/test_stage3_mano_reconstruction.py` — 将三个过时断言对齐当前架构合同：仅在启用 hand perturb 时重建 MANO；legacy runtime resampling 需要显式 stored-hand proxy；schema version 是标识，完整 MANO 字段才是能力门槛。

**改动原因**

用户要求以 `oyx` 为主合入 feature 分支，并把 repo notes 拆入可提交的仓库记忆和本机私有记忆。

**验证**

- hand PCA/MANO/Stage3 定向测试：`50 passed, 3 skipped`。
- 仓库全量 pytest：`282 passed, 3 skipped`。
- 相关目录 `compileall` 通过。

## 2026-08-24 — 完成纯 GRAB noPCA 的 HOCap subject_1 外部测试

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件 / 产物**

- `output/research/hocap_subject1_grab_nopca_20260824/smoke_object_only.json` — 保存单文件 676 帧兼容性小测结果。
- `output/research/hocap_subject1_grab_nopca_20260824/full_object_only.json` — 保存 subject_1 全量 28 文件 / 23,896 帧的 clean 与 object perturb 指标。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 新增 EXP-012，记录 checkpoint、HOCap 协议、指标和结论边界。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 将 HOCap 从转换完成更新为首条外部基线已完成。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次评估与文档同步。

**改动原因**

响应用户要求，选取训练时关闭手部扰动的纯 GRAB noPCA checkpoint，在新转换的 HOCap subject_1 上先做实际验证。

**验证**

- checkpoint step 317800 / epoch 35，配置 `apply_hand_perturb=false`；
- 评估端手扰动关闭、runtime resampling 关闭，object perturb 固定 10°/10 mm；
- smoke 与全量均成功结束，无 schema、坐标系、CUDA 或有限值错误。

## 2026-08-23 — 后台启动全量 ARCTIC MANO Stage 2/3 导出

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: 跨 task / 全局

**文件 / 数据**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 增加全量 ARCTIC 导出运行状态、风险和后续核验步骤。
- `docs/logs/status_log.md` — 同步跨 task 的后台数据处理状态。
- `output/research/arctic_full_mano_v21_20260823/run_export.sh` — 在 GPU 3 顺序运行 ARCTIC Stage 2 与 Stage 3。
- NAS `processed_data/stage2/arctic_full_mano_v21_20260823`、`processed_data/stage3/arctic_full_mano_v21_20260823` — 新建独立输出，不覆盖旧数据。

**改动原因**

为后续 GRAB / ContactPose / OakInk / ARCTIC 四域训练准备全量、带 MANO 描述的 hand-root Stage 3 数据。后台进程已成功加载 MANO 与物体模板，并写出首批 Stage 2 文件。

## 2026-08-23 — 将历史 GRAB+ContactPose 两域模型加入协议 E

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py` — 为缺少 `meta.mano_model_dir` 的历史 v2.0 checkpoint 使用当前 Task 默认 MANO 路径。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md` — 加入历史 GRAB+ContactPose latest 的三条件、ΔQFL 和等权汇总。
- `src/task/correspondence_ptv3_v2/result/comprehensive_comparison_20260823.md` — 同步协议 E 五模型表和结论边界。
- `src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md` — 记录旧 checkpoint 的 MANO 路径兼容规则。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 扩展 EXP-011 的 checkpoint、结果与解释。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新最近可靠结论。
- `output/research/arctic_min11_mano_protocol_e_10mm_20260823/grab_contactpose_latest_*` — 保存三条件 JSON / log。

**改动原因**

用户明确选择历史 GRAB+ContactPose 两域 mixed latest.pt 加入协议 E。旧 checkpoint config 早于 MANO 字段，首次 hand-only 在缺少 `mano_model_dir` 时 fail-fast；补当前默认只读路径后重跑成功。

**验证**

- object-only、hand-only、hand+object 均完成 11 序列 / 4,175 帧；
- hand 条件协议字段为 10 mm、概率 1.0，changed-edge fraction 与协议 E 其他模型一致；
- 最终三个日志无 traceback 或错误。

## 2026-08-23 — 新增协议 E 并修复评估端手扰动概率漂移

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py` — hand-only / hand+object 评估显式固定 `hand_perturb_prob=1.0`，输出增加实际手扰动概率字段。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md` — 新增协议 E 的三条件分项、ΔQFL、等权汇总和限制。
- `src/task/correspondence_ptv3_v2/result/comprehensive_comparison_20260823.md` — 增加协议 E，并把旧协议 D hand/joint 结果标记为 `INVALID_IMPLEMENTATION`。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_h80_h50_bestpt_compare_20260823.md` — 增加旧评估失效说明。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_v1_compare_20260821.md` — 增加旧评估失效说明。
- `src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md` — 固化评估条件不得继承训练 exposure 的不变量和 ΔQFL 定义。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 新增 EXP-011，并更正 EXP-009/010 的有效性。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新最近可靠结论、风险和下一步。
- `output/research/arctic_min11_mano_protocol_e_10mm_20260823/` — 保存 8 个修正后 hand 条件 JSON / log；object-only 复用协议 D 的有效产物。

**改动原因**

用户要求用协议 E 在 10 mm hand perturb 下评估四个 checkpoint，并保留实际退化量。首次运行的一致性核验发现旧 evaluator 会继承 checkpoint 的 H80/H50 训练门控比例，导致不同模型收到不同 fraction 的手扰动；为实现固定评估协议，统一覆盖为 100% 后重新运行。

**验证**

- 8 个新 JSON 均为 4,175 帧，协议字段为 hand 10 mm / probability 1.0；
- 四个 checkpoint 的 hand-only changed-edge fraction 均为 0.003200，hand+object 均为 0.004462；
- 8 个日志未发现 traceback、ERROR 或协议失败。
- evaluator 语法检查通过；`tests/test_hand_pca_perturbation.py` 为 9 passed / 1 skipped / 2 个既有失败，失败分别来自测试 fixture 缺少 `apply_hand_perturb` 和旧错误消息正则，与本次 evaluator 改动无关。

## 2026-08-23 — 补充综合报告 A/B/C/D 协议口径

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/comprehensive_comparison_20260823.md` — 分别补充协议 A/B/C/D 的数据来源和选择规则、checkpoint、clean/perturbed 输入、hand/object 扰动、聚合方式、可回答问题与限制。

**改动原因**

用户要求报告明确说明 A/B/C/D 各自的评测口径，使后续读者能判断哪些结果可以直接横向比较，以及每套结果实际回答的研究问题。

**影响范围**

仅综合报告说明；未修改结果数字、原始 JSON、评测代码或训练。

## 2026-08-23 — 统一当前报告的总体 recovery 主指标

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/comprehensive_comparison_20260823.md` — 增加 `pseudo_recovery_brier` 定义，并将其设为唯一总体 recovery 主指标；fake/missed/projection 降级为诊断指标，不再参与主表排名。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_h80_h50_bestpt_compare_20260823.md` — 同步 H80/H50 直接对比的报告口径。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 同步 EXP-010 主结果表和关键观察的指标名。

**改动原因**

用户要求使用合并 fake-contact 与 missed-contact changed edges 的综合 `pseudo_recovery_brier` 报道当前结果，避免把两类 failure-mode 分解或 projection 当作总体模型排名依据。

**影响范围**

仅报告表达与实验日志；没有重新计算指标，也未修改原始 JSON、evaluator、模型或训练。

## 2026-08-23 — 汇总 result 目录的全方位对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/comprehensive_comparison_20260823.md` — 按四套评测协议汇总全部现有结果，补充 checkpoint/预算索引、协议内排名、跨协议一致性、目标导向选模建议和待补实验。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 将综合结果页更新为当前结果入口。

**改动原因**

用户要求整理 `result/` 当前结果并形成全方位对比。汇总时保持不同数据子集、MANO 能力、扰动条件和 checkpoint 口径的边界，避免把不可直接比较的绝对指标合并成单一排名；同时明确区分历史 GRAB+ContactPose mixed 与正在训练的三域等比例 mixed。

**影响范围**

仅 Task 级结果汇总和状态入口；未修改原始结果、训练代码、评测代码或正在运行的训练。

## 2026-08-23 — 完成 5 mm H80/O20 与 H50/O50 best.pt 三条件对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_h80_h50_bestpt_compare_20260823.md` — 新增两条 best.pt 的统一协议结果表、分解指标和限制说明。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新 EXP-010 的完成状态、三条件结果、解释和下一步。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新最近可靠结论和研究风险。
- `output/research/arctic_min11_mano_bestpt_h80_h50_20260823/` — 保存六个被忽略的评测 JSON 与日志。

**改动原因**

用户要求沿用上一版 5 mm 的评测方法，并进一步指定新旧版本都使用各自 `best.pt`。因此在同一 ARCTIC MANO min11、三种扰动条件、相同 batch/seed/evaluator 下重跑 H80/O20 与 H50/O50，避免把旧 latest.pt 结果与新 best.pt 混用。

**影响范围**

仅 Task 级评测产物和研究文档；未修改 evaluator、模型、训练配置或正在运行的 mixed 训练。

## 2026-08-22 — 在 GPU 1、2 恢复三域等比例 mixed 训练

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 新建 Task 当前状态入口，记录 mixed 续训进度、运行资源和风险。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新 EXP-007 的中断证据、恢复入口和新 W&B run。
- `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_resume_gpu12_20260822_050212/` — 从 step 22748 checkpoint 建立独立续训输出并启动双卡 DDP；该目录属于被忽略的实验产物。

**改动原因**

用户确认使用空闲物理 GPU 1、2 继续三域等比例 mixed baseline，并要求创建新的 W&B run ID。为避免新旧 run 的指标文件混写，将旧 `latest.pt` 复制到独立输出目录后恢复；训练已确认从 step 22748 继续写出，新 W&B run ID 为 `d5yvor4z`。

**对应指导**

当前三域混训方案；ARCTIC 外部评估继续遵循 `docs/指导/V1.md`。

**影响范围**

仅 Task 实验运行与状态文档；未修改训练代码、采样协议或模型配置。

## 2026-08-21 — 固定 GRAB H50/O50 互斥扰动配置并核对三域混训中断

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_h50_o50_exclusive_ddp2.yaml` — 新增 H50/O50 互斥扰动训练配置。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录三域混训的实际停止证据及 H50/O50 实验入口。

**改动原因**

用户确认采用方案 A：在保持 5 mm hand noise 和既有 object pose perturbation 的前提下，严格互斥地以 50% hand-only、50% object-only 训练，禁止 hand+object compound 样本。

同时复核已停止的三域混训目录：日志最后写到 step 27980，无 Python traceback、CUDA OOM、kernel OOM 或 checkpoint 临时文件；现有 checkpoint 只到 step 22748，支持“外部停止/会话结束发生在保存周期之间”的判断。

补充证据：W&B 本地日志 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_20260820_141453/wandb/run-20260820_141505-div3m72u/files/output.log` 同样以 step 27980 结束，未出现终止原因。

**对应指导**（如有）

当前三域混训方案；ARCTIC 外部评估遵循 `docs/指导/V1.md`。

**影响范围**

仅 task 内部配置和实验记录；未修改三域 loader 或旧 checkpoint。

## 2026-08-21 — 导出 ARCTIC MANO min11 并扩展三条件评估入口

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件 / 数据**

- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage2/arctic_min11_mano_v1` — 从 raw ARCTIC MANO 生成 11 个 Stage 2 文件。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1` — 生成 11 个带 MANO 的 Stage 3 `.npz`，共 4175 帧。
- `src/task/correspondence_ptv3_v2/config.py` — 增加 validation object perturb 开关，保留默认行为。
- `src/task/correspondence_ptv3_v2/dataset.py` — validation perturbed loader 可按 condition 关闭 object perturb。
- `src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py` — 增加 `object_only` / `hand_only` / `hand_object` 条件和 ARCTIC axis-angle45 5mm calibration。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_v1_manifest.json` — 固化 NAS 数据清单和 MANO schema。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_v1_compare_20260821.md` — 记录三条件结果。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md`、`decision_log.md`、`repo_notes_log.md` — 同步实验、决策和路径。

**改动原因**

响应用户要求先导出部分含 MANO 的 ARCTIC 数据，补齐 hand-only / hand+object 公平评估；不覆盖旧无 MANO缓存或全量数据。

**验证**

- 11/11 文件均含 `mano_pose (T,45)`、`mano_global_orient`、`mano_transl`、`mano_betas`。
- 全部标记 `mano_pose_repr=axis_angle`、`mano_use_pca=false`。
- evaluator `py_compile` 通过；三条件六个 JSON 结果均生成。

**影响范围**

仅 task 内部数据、评估入口和实验记录；默认 object-only evaluator 语义保持兼容。

## 2026-08-21 — 修正 min11 recovery 结果方向并确认互斥暴露差异

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_compare_20260821.md` — 将 recovery 指标按“越高越好”修正为 noPCA 优于 5mm，并补充 object exposure 解释。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 修正 EXP-008 的结论和下一步。

**改动原因**

复核 `pseudo_recovery_brier = (E_pseudo - E_model) / E_pseudo` 后，发现前一版结果表错误地把 5mm 的较低 recovery 标成了更优。代码指标方向本身正确；差异主要与 5mm 互斥门控的约 20% object-only 暴露、noPCA 的 100% object-only 暴露有关。

**影响范围**

仅修正科研记录，不修改模型、训练代码或评测实现。

## 2026-08-21 — 新增 ARCTIC min11 快速评估结果记录

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_manifest.json` — 固化覆盖 11 个物体的最小评估文件清单。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_compare_20260821.md` — 记录 noPCA / 5mm 的统一 object-only 协议、结果和限制。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 新增 EXP-008，记录方向性证据和后续 matched evaluation。

**改动原因**

响应用户“先评估、控制规模但覆盖类别”的要求；使用 11 个物体各一个文件的快速筛查，避免全量 ARCTIC 评测耗时过长。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

**影响范围**

仅 task 内部评测结果和科研记录；未修改模型、loss、训练代码或 evaluator。

## 2026-08-20 — 新增 GRAB/ContactPose/OakInk 等比例从头混训入口

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/dataset.py` — 新增 `DomainConcatDataset` 和 `DomainBalancedSampler`；支持按域独立切分 train/val，并在每个 batch 固定 1/3 域比例；`use_mano_reconstruction=false` 时不再把 GRAB 的 MANO 字段混入无 MANO 域的 batch。
- `src/task/correspondence_ptv3_v2/config.py` — 增加 task 级 `data.domain_paths` 配置字段。
- `src/task/correspondence_ptv3_v2/configs/mixed_grab_contactpose_oakink_equal.yaml` — 新增三域从随机初始化开始的 baseline 配置；显式关闭 runtime object resampling、MANO/hand perturb 和 scale augmentation，保留 object pose perturbation。
- `src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md` — 记录三域 loader、采样和 validation 数据流。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录等比例 sampler、batch size 48 和独立 validation loader 的自主选择。

**改动原因**

落实用户确认的三域从头训练协议：GRAB、ContactPose、OakInk 各 1/3，使用 domain sampling，不开启 runtime object resampling，不加入 scale augmentation。

**对应指导**

当前三域混训方案；V1 的 ARCTIC 外部评估指导仍作为启动前评测依据。

**验证**

- 配置成功加载；
- 实际 NAS 数据 Dataset smoke 成功，三个域 train/val 均构造完成；
- 首个 batch 为 `16 grab + 16 contactpose + 16 oakink`；
- batch 不含 runtime full-pool 字段，且无 MANO schema collate 冲突；
- `py_compile` 通过。

## 2026-08-21 — 记录并停止 no-PCA runtime 续训

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部实验状态

**文件**

- `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614` — 保留 no-PCA runtime run 的日志和 checkpoint，停止前 step 331300 / epoch 37，最新保存 checkpoint 为 step 317800 / epoch 35。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录 run 路径、停止状态及后续 object-perturb exposure 对照方向。

**改动原因**

用户要求保留 no-PCA 版本路径并停止当前训练，转向验证 5 mm exclusive hand/object perturbation 是否降低了 object perturb 暴露比例。

**影响范围**

仅 task 内部实验进程和记录；未修改模型或数据。

## 2026-08-20 — 完成 V1 ARCTIC 分层外部评估与 object-macro 汇总

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/research/arctic_v1/build_stratified_subset.py` — 构造覆盖全部 object / subject / action 的确定性分层软链子集。
- `src/task/correspondence_ptv3_v2/research/arctic_v1/build_object_subsets.py` — 为每个 object 构造单独的软链子集，便于按 object 汇总 macro。
- `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.md` — 写入 V1 ARCTIC micro / object-macro 对比结果。
- `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.json` — 保存 V1 汇总 JSON。

**改动原因**

用户要求按照 V1 指导实现：在不训练 ARCTIC 的前提下，用 held-out ARCTIC 评估纯 GRAB 与 GRAB+ContactPose。由于现有 evaluator 只输出全局 micro，需要额外构造 object 子集来计算 object-macro。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

**影响范围**

仅 task 内部实验、结果和研究脚本，不改变训练代码、模型、loss 或 evaluator 协议。

## 2026-08-20 — 新增 ARCTIC 全量/分层外部评估 V1 指导

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/指导/V1.md` — 规定现有纯 GRAB 与 GRAB+ContactPose checkpoint 的全量 ARCTIC 或全类别分层子集评测流程。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 将 V1 与 ARCTIC 外部评估更新为当前任务入口和研究近况。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录时间受限时采用 subject/object/action 确定性分层的自主选择。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次指导文档新增。

**改动原因**

响应用户要求，将启动三数据集混训前首先要完成的 ARCTIC 外部评估固化为可执行指导，修正旧 20 文件子集仅含 `box_*` 的类别偏置，并明确统一协议、结果统计和后续决策门。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

**影响范围**

仅 task 内部研究指导与记录，不修改代码、checkpoint、训练进程或评测实现。

## 2026-08-20 — 重新拉起 GRAB 5 mm no-PCA runtime 训练

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614` — 从 `latest.pt` 重新恢复同一训练 run。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录训练已在 tmux `grab5mm_no_pca_runtime_resume_20260820` 中重新拉起。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次恢复动作。

**改动原因**

上一轮续跑因本地根分区满盘而中断。NAS 软链释放空间后，按原 checkpoint 和原配置重新拉起同一训练 run。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 训练已重新启动并进入 DDP 恢复流程。
- 剩余: 继续观察是否能稳定跑到下一个 checkpoint。

**影响范围**

仅 task 内部训练进程状态，不改变模型、数据或评测协议。

## 2026-08-20 — 将 GRAB 大体积数据切到 NAS 软链

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `dataset/GRAB/data` — 改为指向 NAS 上 `GRAB/data/GRAB` 的软链。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 记录 GRAB 数据根的 NAS 路径。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录训练中断原因是根分区满盘，并注明已释放空间。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次修改。

**改动原因**

本地根分区已满，GRAB 的 `data` 目录是当前最大的可迁移只读数据。NAS 上已有对应数据，因此将本地入口切换到 NAS 软链，释放本地空间并保持代码路径不变。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 软链已切换完成，本地空间已释放。
- 剩余: 如需，后续继续把其他只读大数据目录也切到 NAS。

**影响范围**

仅 task 内部路径与文档，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 迁移 processed_data 到 NAS 并补结果配置摘要

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `data/processed_data` — 迁移到 NAS 上的 `processed_data` 目录软链。
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` — 补充 OakInk / GRAB / GRAB+ContactPose 的训练配置摘要。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 记录 task 级 processed_data NAS 路径。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次迁移和结果文档更新。

**改动原因**

响应用户要求，将本地 processed_data 迁到 NAS，减少本地盘占用，并把当前三模型对比结果中的训练配置补齐，便于以后直接核对协议差异。

**影响范围**

仅 task 内部路径与文档，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 完成三模型 ARCTIC 子集对比并修正结果口径

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` — 写入三模型 ARCTIC 子集对比表，并修正 recovery 高优标注。
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.json` — 保存结构化汇总结果。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 将当前状态改为已完成，并新增 EXP-005。
- `src/task/correspondence_ptv3_v2/research/log.md` — 修正 ARCTIC 结果解读中关于 recovery 方向的旧表述。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次修改。

**改动原因**

响应用户要求，完成 OakInk / GRAB / GRAB+ContactPose 在同一 ARCTIC 子集上的对比汇总，并把旧文档里和实际指标不一致的 recovery 口径修正为任务架构定义下的正确方向。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 三模型对比结果已落盘，相关实验和研究日志已同步。
- 剩余: 无。

**影响范围**

仅 task 内部文档与结果摘要，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 记录三模型 ARCTIC 对比的当前接续点

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录 OakInk、纯 GRAB、GRAB+ContactPose 在统一 ARCTIC 子集上的当前对比目标、协议、已有结果和待完成产物。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档同步。

**改动原因**

响应用户要求，先把当前正在进行的三模型 ARCTIC 对比写入任务文档，避免后续中断后丢失评测口径和接续点。已有纯 GRAB / GRAB+ContactPose 结果将复用，OakInk 已按相同协议完成子集评测，下一步生成时间戳结果表。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 对比定义和评测协议已记录，OakInk 子集评测已完成。
- 剩余: 汇总三方 JSON/历史 JSON，生成 `src/task/correspondence_ptv3_v2/result/` 下的时间戳 Markdown/JSON 结果。
- 续接点: 使用 `/tmp/oakink_arctic_eval_20260820.json`，并读取已有 `arctic_pure_20260817_132348.json`、`arctic_mixed_20260817_132348.json`。

**影响范围**

仅 task 内部文档和研究结果汇总，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 续跑 GRAB 5 mm no-PCA runtime 训练

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 更新任务入口状态，说明 GRAB no-PCA runtime run 已进入续跑。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新当前状态：原进程停在日志 step 36320 / epoch 4，已从 `latest.pt` 的 step 27240 / epoch 3 续跑。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次训练续接。

**改动原因**

用户要求继续训练当前 GRAB “5 mm 配置 + runtime sampling + no-PCA hand perturb” run。原进程已停止且无 Python traceback；`latest.pt` 是最近完整 checkpoint，因此从该 checkpoint 恢复。

训练续接信息：

- tmux: `grab5mm_no_pca_runtime_resume_20260820`
- checkpoint: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614/checkpoints/latest.pt`
- resume step / epoch: `27240` / `3`
- W&B run id: `zeqrhxyk`

**影响范围**

仅 task 内部训练进程与文档状态；不改变模型、数据、配置或评测协议。

## 2026-08-19 — 新增 5 mm no-PCA runtime 配置

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_no_pca_runtime_ddp2.yaml` — 以 5 mm 配置为基线，仅关闭 `apply_hand_perturb`，保留 runtime object resampling。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 更新任务入口状态，说明正在启动新的 GRAB no-PCA runtime run。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新当前状态，记录正在启动新的 no-PCA runtime 训练。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录把 5 mm 作为 no-PCA runtime 新 run 的基线、只关闭 hand perturb 的选择。

**改动原因**

用户要求训练一版“无 PCA 扰动，但用 runtime 采样，配置和 5 mm 一致”的 GRAB 训练。

训练已通过 `torchrun` 拉起，当前 run 目录为 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614`。

**影响范围**

仅 task 内部配置与实验记忆，不改变现有训练代码或评测协议。

## 2026-08-19 — 切换后续 no-PCA 默认对照并补数据集口径说明

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 新增后续 no-PCA 对照默认使用 mid-Aug compact repro 的决策。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 将当前状态中的 no-PCA 默认口径改为 mid-Aug compact repro。

**改动原因**

用户明确要求以后不再使用 `finetune_day_cosine_restart` 的续训口径作为默认 no-PCA 对照，而改用 8 月中旬训练的那条结果更接近的 no-PCA run。

**影响范围**

仅 task 内部文档口径，不改变训练、验证或模型代码。

## 2026-08-19 — 续接 OakInk 无手扰动训练并补齐 baseline 决策记录

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新 OakInk 正式训练当前状态：原进程日志停在 step 255300，已从 `latest.pt` 的 step 211410 续跑。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 补充 `finetune_day_cosine_restart` 作为 no-PCA canonical 参考的口径选择。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新与训练续接。

**改动原因**

继续 01a01514-d636-7d50-8a45-4eff274a7f03 会话中断后的任务：完成 GRAB 5 mm 与 canonical no-PCA baseline 对比的文档收尾，并发现 OakInk 单卡训练不在进程列表中后，按原 50 epoch 需求从 `latest.pt` 续跑。`latest.pt` 对应 step 211410，因此日志中 step 211410 到 255300 的未 checkpoint 训练进度不会被保留为模型状态。

**影响范围**

仅 task 内部。训练续接使用原 run 目录和原配置，只额外设置 `train.resume=.../checkpoints/latest.pt`；不改变模型、数据或评测协议。

## 2026-08-19 — 记录 GRAB 5 mm 与 `finetune_day_cosine_restart` 复现对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加 GRAB 5 mm PCA 互斥扰动版与 `finetune_day_cosine_restart` / 复现 run 的当前 best.pt 对比实验记录，并更新当前状态。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录 `finetune_day_cosine_restart` 作为 canonical no-PCA 参考的选择依据。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户澄清后的比较对象，对当前 5 mm 版本与几天前那条 no-PCA 复现链路做同口径对比，并将结果沉淀为可追溯实验记录。

**影响范围**

仅补充 task 实验记忆，不改变代码、训练配置或评测协议。

## 2026-08-19 — 记录 GRAB 5 mm 与旧 no-PCA legacy run 的方向性对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加 GRAB 5 mm PCA 互斥扰动版与旧 no-PCA legacy GRAB run 的方向性对比实验记录，并更新当前状态。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户要求，对当前 5 mm 加 PCA 扰动版本与之前 no-PCA 版本做进一步比较，并明确这次对比的协议差异与适用范围。

**影响范围**

仅补充 task 实验记忆，不改变代码、训练配置或评测协议。

## 2026-08-19 — 记录 GRAB 5 mm 与两条 9 mm best.pt 的当前指标对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加 GRAB 5 mm PCA 互斥扰动版与两条 9 mm baseline 的当前 best.pt 对比实验记录，并更新当前状态。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户要求，对未训练完的 GRAB 5 mm 加 PCA 扰动版本与此前 9 mm 版本做当前 best.pt 性能对比，并将结果沉淀为可追溯实验记录。

**影响范围**

仅补充 task 实验记忆，不改变代码、训练配置或评测协议。

## 2026-08-19 — 记录 OakInk/GRAB 训练吞吐与四数据集规模核验

- branch: `feature/hand-pca-perturbation`
- post-commit: 未提交（基于 `d5510e0`）
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加运行进度、cache/吞吐诊断及 GRAB、OakInk、ContactPose、ARCTIC 的统一帧样本统计。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户对 OakInk 无 PCA 训练耗时、cache 影响和四数据集帧数/训练样本关系的追问。统计为只读核验，不改变实验定义、训练配置或代码。

**影响范围**

仅补充 task 实验记忆，供后续评估训练预算与数据集组合时引用。

## 2026-08-18 — 支持 OakInk 无 MANO 手部扰动但保留 runtime sampling

- branch: `feature/hand-pca-perturbation`
- post-commit: 未提交（基于 `d5510e0`）
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/runner.py` — 加载并校验存储手点 proxy；无 MANO 路径使用该 proxy 完成 full-pool near/global 采样。
- `src/task/correspondence_ptv3_v2/config.py` — 新增 `stored_hand_proxy_indices_path`。
- `src/task/correspondence_ptv3_v2/calibration/oakink_stored_hand_fps256.json` — OakInk 存储手点 FPS-256 版本化索引。
- `src/task/correspondence_ptv3_v2/configs/oakink_50ep_no_hand_perturb_runtime.yaml` — OakInk 无手部扰动候选训练配置。
- `src/task/correspondence_ptv3_v2/research/oakink_conversion/validate_wrist_centering.py` — OakInk 多视角 wrist/camera/world 几何验证脚本。
- `README.md` 和 `docs/架构.md` — 同步路径、坐标系和无 MANO runtime 约定。

**改动原因**

用户选择方案 B：不重建 MANO、不加 PCA 手部扰动，但仍保留 runtime object resampling。

**验证**

- OakInk 8 个多视角组、24 个 view pair：只减 wrist 的手点跨视角 RMS 平均 163.14 mm；用 `cam_extr^{-1}` 后为 0.000080 mm；接触距离跨视角 MAE 为 0.000035 mm。
- stored proxy 与 MANO proxy 的 top-1024 near-pool Jaccard 平均 0.948。
- 无 MANO 模型目录下 dataset/runtime smoke、PTv3 forward、loss 和反向传播均通过。
## 2026-08-24 — 新增 HOCap annotation-only Stage 3 转换器并启动 subject_1 导出

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部 / 外部数据处理

**文件 / 数据**

- `process/HOCap/__init__.py` — 新增 HOCap 处理包入口。
- `process/HOCap/stage3_export.py` — 读取 HOCap `meta.yaml`、`poses_m.npy`、`poses_o.npy`、MANO betas 和 clean object mesh，按序列/物体/手生成 Ref2Dex Stage 3 v2.1。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hocap_subject1_annotation_v1_smoke*` — 20 帧 smoke 输出，已通过 schema 和几何检查。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hocap_subject1_annotation_v1` — subject_1 完整转换目标目录，CPU tmux `hocap_subject1_stage3_20260824` 后台运行。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/HOCap/logs/subject_1_stage3_export_20260824.log` — 转换日志。

**改动原因**

HOCap 原始数据不是当前 loader 直接接受的 Stage 3 schema；先完成单序列 smoke，再后台转换 subject_1，保持原始 tar、annotation-only metadata 和现有训练配置不变。

## 2026-08-24 — HOCap subject_1 Stage 3 转换完成并通过全量核验

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: task 内部 / 外部数据处理

**验证**

- 7 个序列、28 个序列/物体/手 NPZ、23,896 个 frame samples，转换失败数为 0。
- 所有文件包含 Stage3 必需字段；物体池 `(N,4096,3)`、手点 `(N,1538,3)`、距离 `(N,1538)`、hand-root pose `(N,4,4)` 均通过 shape 和 finite 检查。
- 物体和手部法线单位长度检查通过。
- `CorrStaticDatasetV2` clean loader 扫描 28 个文件并成功读取样本；coordinate frame 为 `hand_root`，dataset id 为 `hocap`。
## 2026-09-03 — 新增 GRAB+HRDexDB 5 cm object-centered 训练协议

- branch: `oyx`
- post-commit: working tree
- scope: task 内部

**文件**
- `src/task/correspondence_ptv3_v2/dataset.py` — 生成并传递 `hand_valid_mask`；支持加权 domain-balanced batch sampler。
- `src/task/correspondence_ptv3_v2/model.py` — 双向几何特征和手点特征尊重手点有效 mask。
- `src/task/correspondence_ptv3_v2/runner.py` — runtime 重采样、随机监督边和 contact 辅助候选应用手点 mask。
- `src/task/correspondence_ptv3_v2/config.py` — 增加 domain sampling weights 配置字段。
- `src/task/correspondence_ptv3_v2/configs/mixed_grab_hrdexdb_object_centered_5cm.yaml` — 新训练配置：1024 object points、5 cm 手点、无 contact loss、4:4:2 扰动。
- `src/task/correspondence_ptv3_v2/docs/指导/V2.md`、`docs/plan/V2.md` — 记录本轮研究指导和 final plan。
- `docs/current_versions.yaml` 及 Task logs — 更新版本、架构、决策和修改记录。

**改动原因**

落实用户确认的 GRAB + HRDexDB 五域 object-centered 训练语义，同时保持旧配置和旧 checkpoint 接口兼容。

**验证**

配置/NAS 首 batch smoke：五域可加载，domain quotas 为 `[20,5,5,5,5]`，batch shape 为 `(40,2562,3)`，手点有效数随帧变化；correspondence 定向测试 `59 passed`。
