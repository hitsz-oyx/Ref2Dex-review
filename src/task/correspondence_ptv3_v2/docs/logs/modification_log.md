# Modification log

## 2026-08-24 — 核验三域 mixed 在线训练状态

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部状态记录

**文件**

- `docs/logs/status_log.md` — 更新 DDP 在线进程、step、GPU、当前数据版本及 checkpoint 落后风险。

**改动原因**

响应用户对三域混合训练当前进度的查询。核验确认任务仍在 GPU 1/2 运行，当前约 step 142,480 / 154,670；在线配置尚未切换到新 OakInk true-hand-root 目录。

## 2026-08-24 — 补充三域 mixed 验证收敛判断

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部状态记录

**文件**

- `docs/logs/status_log.md` — 记录训练 loss 下降而验证 QFL 在早期最好、后期平台/回升的事实边界。

**改动原因**

根据用户追问，对 step 45,496、68,244、90,992、113,740、136,488 的逐域验证记录做趋势核对，避免把在线训练完成误判为验证性能仍在改善。

## 2026-08-24 — 重导出 OakInk true-hand-root 并修正 MANO translation

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 外部数据处理

**文件 / 数据**

- `research/oakink_conversion/convert_oakink_pilot.py` — 从官方 `general_info` 读取 root quaternion、local pose、shape 和 wrist translation；对手物共同执行完整 hand-root 变换，并输出 MANO 重建字段。
- `research/oakink_conversion/repair_oakink_mano_transl.py` — 原子修复首轮新产物的 `mano_transl`，把 OakInk wrist-position 语义转换为 `smplx.MANO` translation 语义。
- `docs/架构.md`、`docs/logs/{architecture_log,repo_memory,status_log,experiment_log,decision_log}.md` — 将旧 wrist-centered OakInk 边界标记为历史，记录新数据合同、验证证据和后续配置切换风险。
- NAS `OakInk/processed/stage3_corr_oakink_true_handroot_20260824` — 新增 2596 个 NPZ / 252,172 帧，不覆盖 2026-08-18 旧目录。
- 根级架构、状态、记忆、修改日志和 root/task `machine_memory.md` — 同步跨数据集坐标合同与本机实际路径。

**改动原因**

用户要求重新导出 OakInk，并保证转换到包含旋转规范化的真正手根坐标系，同时为后续手姿态扰动保留可用 MANO 参数。

**验证**

- 转换遍历 3168 个视角组；2596 成功，572 因缺 object mesh 明确跳过；
- 严格 Dataset MANO 路径可加载；抽样 MANO 重建最大误差 ≤0.000131 mm；
- KD-tree 距离缓存最大误差 ≤1.50×10⁻⁸ m；四视角 hand/object RMS ≤0.000069 mm；
- 2596 个文件无 `.repairing.npz` 残留，转换/修复脚本通过 `py_compile`。
- MANO/hand perturbation 定向回归：`16 passed, 3 skipped`。

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
## 2026-08-25 — 新增 HRDexDB-human 单 episode Stage 3 smoke exporter

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 外部数据处理

**文件**

- `process/HRDexDB/stage3_export.py` — 新增 HRDexDB-human 单 episode 转换器；读取 MANO OBJ、MANO 参数、object 6D pose 和 clean object mesh，输出当前 correspondence Stage 3 v2.1 geometry。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_human_smoke_v1/` — 生成 `human/apple/0` 的 257 帧 smoke NPZ 和 `meta.json`。

**改动原因**

响应先用一部分 HRDexDB 打通小链路的需求。当前 smoke 只验证 clean geometry，不改变现有 mixed 训练配置。

**验证**

- NPZ shape：object `(257,4096,3)`，hand `(257,1538,3)`，距离 `(257,1538)`；法向单位长度和 finite 检查通过。
- `CorrStaticDatasetV2` clean loader 成功读取 257 个 frame，`hand_root` 校验通过，`prepare_batch` 输出 `(2,2050,3)` 且 finite。
- `ref2dex-grab` 环境缺少 `spconv`，因此改用当前 mixed 训练所用的 `graspenv`；在空闲 GPU 3 上完成 PTv3 forward 和单步 loss/backward smoke：输出 logits/prob 全部 finite，loss=`0.154691`，457 个参数梯度张量 finite。

## 2026-08-25 — 核验 HRDexDB MANO reconstruction、hand-root 和扰动链路

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 外部数据处理

**文件 / 产物**

- `src/task/correspondence_ptv3_v2/research/hrdexdb_conversion/diagnose_hrdexdb.py` — 新增 HRDexDB human MANO forward、hand-root 数值核验和 world/hand-root 可视化脚本。
- `process/HRDexDB/stage3_export.py` — 增加 `--include-mano`，将已验证的 HRDexDB 3×3 旋转转换为 axis-angle45 并写入 Stage3 MANO 字段。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/research/hrdexdb_human_apple0_reconstruction_v1/` — 257 帧重建报告和 frame 0/100/200 的 world、hand-root PNG。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_human_smoke_v2_mano/` — 带 MANO 字段的 257 帧 Stage3 验证版本。

**验证**

- `MANO_RIGHT + flat_hand_mean=True`、直接 global rotation、原始 `transl/betas` 重建 257 帧 OBJ：vertex RMSE `6.33e-8 m`，P95 `1.18e-7 m`，最大 `2.78e-7 m`。
- `hand_root = R_global_orient.T @ (x_world - joints[0])`：root joint 最大绝对坐标 `0`，旋转 determinant 最小 `0.999998`，Stage3 face-center RMSE `6.66e-8 m`。
- 带 MANO 字段的 loader clean reconstruction 与存储 hand 点差为 `0`；axis-angle 扰动 smoke 的平均手点位移约 `0.965 mm`，PTv3 单步 loss=`0.153930`，backward 梯度 finite。
- GRAB 当前 mixed 目录包含 `1269` 个 right、`694` 个 left Stage3 文件；GRAB 转换器本身会同时生成两侧，当前训练 loader 没有 side 过滤。
- Inspire DFTP FK smoke：12 维 qpos、URDF 和稳定 1538 点 link/barycentric binding 可用；q hand joints 加 `0.01/0.03 rad` 高斯扰动时，平均手点位移约 `0.14/0.42 mm`，最大约 `2.66/8.00 mm`。

## 2026-08-25 — 接入 HRDexDB Inspire DFTP 的 URDF/FK q-space 扰动

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 外部数据处理

**文件 / 产物**

- `process/HRDexDB/robot_stage3_export.py` — 新增 `inspire_dftp/mug_holder/2` robot Stage 3 smoke exporter；固定 4096 个物体点和 1538 个 link/barycentric 手点，并保存 qpos、关节限位、C2R、hand-root pose 和 URDF 绑定元数据。
- `src/task/correspondence_ptv3_v2/robot_recon.py` — 新增独立机器人 FK 重建函数；只对 hand q joints `[6..11]` 加高斯噪声并按 URDF limits clip，arm/base qpos 保持不变。
- `src/task/correspondence_ptv3_v2/dataset.py` — 将 robot q-space 输入扰动接入 `CorrStaticDatasetV2` 及普通/domain-balanced dataloader；clean hand geometry 保留为 `gt_points/gt_normals`。
- `src/task/correspondence_ptv3_v2/config.py` — 增加 `use_robot_reconstruction`、`apply_robot_perturb`、`robot_perturb_prob` 配置字段。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_robot_smoke_v1/` — 64 帧 robot smoke 输出。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/research/hrdexdb_robot_smoke_v1/frame_00032_clean_vs_qperturb.png` — clean 与 q-space 扰动后的 hand-root 几何可视化。

**验证**

- FK `apply_perturb=False` 与存储 clean hand points 的 RMSE 约 `2.3e-8 m`；法向单位长度误差小于 `6e-8`，跨帧法向重建误差约 `3e-5` 以内。
- dataloader/runner 单 batch（GPU 3）输入手点相对 clean GT 的平均位移 `0.279 mm`、最大 `12.82 mm`；PTv3 loss=`0.147442`，反向传播梯度全部 finite。
- 当前 mixed 训练配置没有修改；机器人分支仍需批量 episode 导出和单独训练配置后再进入训练。

**限制**

机器手不能把 robot qpos 塞进 MANO 字段，也不应直接沿用 MANO PCA 噪声；本次已接入 correspondence runner 的在线 robot-q 分支，但目前仅完成 Inspire DFTP 单 episode smoke，尚未批量导出或加入 mixed 训练。

## 2026-08-25 — 核验 HRDexDB minimal allhands archive

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 外部数据只读核验

**文件**

- `docs/logs/repo_memory.md`、`docs/logs/status_log.md` — 记录 archive 的 four-hand coverage、C2R-valid 子集和当前 Stage 3 适配差异。

**验证**

- archive listing 完整可读、无重复/不安全路径；2104 episodes 的人手帧文件成对，机器人必需 raw 文件齐全，除 Inspire F1 已知 16 个缺失 `C2R.npy` episode。
- v1/v2 集中式 object pose 并集覆盖全部 episode；mesh_v2 覆盖所有使用对象；robot URDF 引用 mesh 全部可解析。

## 2026-08-25 — 新增 minimal allhands 到 Stage3 的统一适配器并启动全量转换

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 外部数据处理

**文件 / 产物**

- `process/HRDexDB/correspondence_minimal_adapter.py` — 支持 human、Inspire DFTP、Inspire F1、Allegro V5；解析集中式 v1/v2 object pose、`mesh_v2`，输出 MANO 或 robot-FK Stage3 v2.1。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/HRDexDB_minimal_allhands_20250825/` — archive 解压 staging 根。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_smoke_v2/` — 四手型 8 帧 smoke，全部通过。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_v1/` — 全量输出目标；四个 tmux 会话已启动，分别处理 human=441、Inspire DFTP=618、Inspire F1=576、Allegro V5=453。

**验证**

- human smoke 含 `mano_pose (8,45)` 等 reconstruction 字段；三类 robot smoke 的 clean FK 与存储 hand points 最大绝对误差均小于 `2e-7 m`。
- smoke dataloader 已验证 human MANO 分支和三类 robot q-space 分支。

## 2026-08-25 — 完成 minimal allhands 全量 Stage3 转换

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 外部数据处理

**产物**

- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_v1/` — 2088 个 Stage3 NPZ，共 1,265,425 帧。
- `manifest_human.jsonl`：441 文件 / 113,680 帧。
- `manifest_inspire_dftp.jsonl`：618 文件 / 533,402 帧。
- `manifest_inspire_f1.jsonl`：576 文件 / 362,027 帧。
- `manifest_allegro_v5.jsonl`：453 文件 / 256,316 帧。

**验证**

- 四个 manifest 的输出文件均存在，四个 errors 文件均为空，summary 均为 `failed=0`。
- 随机抽样四手型文件通过 schema、`[T,4096,3]` / `[T,1538,3]` shape、finite、MANO/robot 字段检查。

## 2026-08-25 — HRDexDB human/robot 独立 domain 配置与 PTv3 链路 smoke

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 跨数据域配置

**文件**

- `src/task/correspondence_ptv3_v2/configs/hrdexdb_human_domain_smoke.yaml` — 指向 NAS 全量 human Stage3，开启 MANO reconstruction 与 axis-angle 扰动。
- `src/task/correspondence_ptv3_v2/configs/hrdexdb_robot_domain_smoke.yaml` — 等比例接入 Inspire DFTP/F1/Allegro V5，开启 robot FK/q-space 扰动。

**验证**

- GPU 3 上真实执行 DataLoader → `prepare_batch` → PTv3 forward → loss backward；human batch `(2,2050,3)`，hand perturb 平均/最大位移 `0.556/5.514 mm`，loss `0.156940`，梯度 finite。
- robot batch `(3,2050,3)`，hand perturb 平均/最大位移 `0.417/6.510 mm`，loss `0.154295`，梯度 finite；三个 robot domain 的 clean/perturbed validation loader 均建立成功。

**限制**

当前 `meta.use_mano_reconstruction` 与 `meta.use_robot_reconstruction` 是全局开关，不能把 MANO 与 robot 文件直接放入同一 domain-balanced loader；因此先保留两套配置，未改变正在运行的三域 mixed。

## 2026-08-25 — 核验 ContactPose 原始 MANO 并完成单序列 v2.1 pilot

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 外部数据处理核验

**验证**

- ContactPose 原始 `use` 目录统计为 1181 个序列，均存在 `mano_fits_15.json`；有效 hand fits 共 1477 个，pose 维度统一为 18（global axis-angle + 15 PCA）。
- 当前 `use_stage3_v2/meta.json` 为 schema `2.0.0`，885 个 NPZ 均缺少 MANO 字段；因此之前判断“ContactPose 不能手扰动”是当前导出版本的限制，不是原始数据集能力限制。
- 使用现有 `process/ContactPose/stage3_export.py` 导出 `full47_use/mug` 单序列 pilot 到 NAS；在 `CorrStaticDatasetV2` + runner 中关闭实际噪声后重建，hand face-center 平均绝对误差 `2.92e-9 m`、最大 `2.98e-8 m`，finite 通过。

**产物**

- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/contactpose_mano_pilot_20260825/` — 单序列 v2.1 pilot，不替换现有 v2.0 数据。
## 2026-08-25 — 七域 MANO/机器人混合扰动训练链路

- branch: working tree
- post-commit: 未提交
- scope: 跨数据域配置与 task 内部实现

**文件**

- `src/task/correspondence_ptv3_v2/config.py` — 增加 mixed MANO/robot dispatch、robot domain 噪声尺度和 5 cm 交互帧过滤配置。
- `src/task/correspondence_ptv3_v2/dataset.py` — 支持 MANO/robot contract 的逐文件校验、过滤非交互帧、robot FK 扰动尺度和 mixed default-collate 占位字段。
- `src/task/correspondence_ptv3_v2/robot_recon.py` — 增加按 domain 传入的 q-space 噪声乘数。
- `src/task/correspondence_ptv3_v2/runner.py` — 将 MANO 重建改为 mixed batch 的逐样本子 batch dispatch，并回写重建手点。
- `process/ContactPose/stage3_export.py` — 写出 v2.1 MANO 字段并在导出阶段过滤 clean 最近距离超过 5 cm 的帧。
- `src/task/correspondence_ptv3_v2/configs/mixed_seven_domain_mano_robot_10mm_5cm.yaml` — 新增七域等比例、随机初始化、W&B online、GPU 2/3 训练配置。
- `src/task/correspondence_ptv3_v2/docs/logs/{status_log,architecture_log,modification_log}.md` — 同步当前架构和验证状态。

**验证**

- ContactPose 全量 `use`：1477 个 NPZ、621514 帧，MANO contract 完整；5 cm 过滤后训练帧 554608。
- 七域 loader 初始化成功，七域训练帧分别为 294562、554608、177249、72511、280032、183531、130887；默认 batch `(56,2050,3)`，MANO/robot flags 与 finite 检查通过。
- mixed MANO/robot batch 的 `prepare_batch`、PTv3 forward、loss backward 均 finite；GRAB 手部扰动 RMS 约 11.0 mm，Inspire DFTP FK 扰动 RMS 约 11.0 mm。

## 2026-08-26 — 修正七域训练的选择指标、runtime 采样与扰动比例

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部；同时修正共享 HOCap 导出统计

**文件**

- `src/task/correspondence_ptv3_v2/runner.py` — 将多域 `val_clean/cross_edge_random_qfl` 聚合为等权宏平均；runtime 改为完整受扰物体池上的均匀 512 点采样；记录机器人 FK 扰动 RMS。
- `src/task/correspondence_ptv3_v2/dataset.py` — 增加稳定的 4:4:2 hand/object/clean gate，runtime 忽略 clean candidate mask，修正 DDP epoch 步数和 OakInk view 分组，并传递 domain cache 路径。
- `src/task/correspondence_ptv3_v2/config.py` / `configs/mixed_seven_domain_mano_robot_10mm_5cm.yaml` — 固化 40%/40%/20% 协议；机器人 q-space 乘数按真实坐标-RMS 复核为 DFTP/F1/Allegro = 25/15/9，并保留实际 RMS 指标。
- `process/HOCap/stage3_export.py` — 元数据 frames 改为按输出文件计数的 frame samples。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录本次采样与增强协议决策。

**改动原因**

修复原 best metric 在多域验证中缺失导致 checkpoint 保存失败的问题，避免 OakInk 多视角泄漏，避免 DDP epoch 重复遍历数据；runtime 采样不再使用 clean-GT 近邻先验或 MANO/robot FPS proxy，训练噪声严格互斥且包含 20% clean 样本。

**验证**

- `compileall` 通过。
- 目标回归测试 `24 passed`。
- 真实 HRDexDB robot loader 200 帧 gate 计数为 hand 72、object 81、clean 47；runtime 输出 `(512, 3)`，并产生 `robot_perturb_rms_m`。

## 2026-08-26 — 建立七域未压缩 NPZ sidecar cache

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / NAS IO cache

**文件与产物**

- `src/task/correspondence_ptv3_v2/research/io_cache/build_uncompressed_npz_cache.py` — 新增可恢复、保留相对路径和 schema 的 sidecar 构建器。
- `src/task/correspondence_ptv3_v2/dataset.py` — 增加 `array_cache_path` / `array_cache_required`，在 schema 校验、过滤和读取阶段优先使用 sidecar，缺失时可回退源 NPZ。
- `src/task/correspondence_ptv3_v2/config.py` / `configs/mixed_seven_domain_mano_robot_10mm_5cm.yaml` — 增加并启用七域 cache 路径。
- NAS `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/cache/correspondence_ptv3_v2_uncompressed_v1/` — 完成七域全量 sidecar，共 8124 个文件、约 329 GB；原始 Stage3 未修改。

**验证**

- 七域各抽取中间文件逐 array 对比，source/cache 完全一致。
- Dataset source 与 cache sample tensor 完全一致；缺失 cache 的 required 模式按预期 fail-fast。
- 真实文件 benchmark：完整七文件读取由约 `1.223 s` 降至约 `0.209 s`，读取阶段约 `5.8×`；实际训练端到端收益待重启新配置后测量。

## 2026-08-26 — 停止旧 run 并启动逐 epoch 保存的新协议训练

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 训练运行

**文件与运行状态**

- `src/task/correspondence_ptv3_v2/configs/mixed_seven_domain_mano_robot_10mm_5cm.yaml` — 设置 `eval_every_epochs=1`、`save_every_epochs=1`；正式 run 使用源压缩 NPZ，并将 DataLoader 调整为 8 workers / prefetch 2。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新旧 run 停止、新 run 启动、cache 回退和当前性能状态。
- 旧 tmux `mixed_seven_mano_robot_20260825_gpu23` / W&B run `lcmssbqq` 已停止。
- 新 tmux `mixed_seven_mano_robot_20260826_gpu23` 在 GPU 2、3 运行；W&B run `0gp9pzjg`，输出目录为 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_seven_domain_mano_robot_10mm_5cm_20260826_061027`。

**改动原因**

新协议包含修正后的 runtime 均匀采样、4:4:2 互斥扰动 gate、DDP epoch 长度和多域 best metric，需要停止不会热加载这些变更的旧进程。每个 epoch 保存一次便于按验证结果回溯模型。

**验证与 cache 结论**

- 序列化 `config.json` 确认 `array_cache_path=null`、`num_workers=8`、`prefetch_factor=2`、`eval_every_epochs=1`、`save_every_epochs=1`。
- 新 run 已越过 step 80；step 40/60/80 的 step time 约为 1.59/2.07/1.54 秒，data wait 约为 0.85/1.24/0.45 秒。
- 直接启用约 329 GB 未压缩 NPZ sidecar 的首次尝试在冷 NAS 上超过 6 分钟仍未返回首个 batch，worker 阻塞于文件系统读取；该无有效 step 的 run 已停止。先前 `5.8x` 是热 page-cache 的单文件读取结果，不能代表 DDP 冷随机读取。sidecar 产物保留，但当前正式训练不启用。

## 2026-08-26 — 更新新 run 性能状态

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 状态记录

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新新 run 至 step 16,860，并记录 data-wait 与模型计算占比。

**验证**

- 最近 100 step data-wait 占总 step time 约 7%，最近 500 step 约 24%，全区间约 17%；当前约 1.76 秒/step，瓶颈主要是模型计算而非 datawait。

## 2026-08-27 — 七域训练切换为每 5000 step 保存

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 训练运行

**文件与运行状态**

- `src/task/correspondence_ptv3_v2/configs/mixed_seven_domain_mano_robot_10mm_5cm.yaml` — 设置 `save_every_steps=5000`、`save_every_epochs=null`；保留 `eval_every_epochs=1`。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新首个 epoch checkpoint、恢复 run 和新保存策略。
- 原 tmux `mixed_seven_mano_robot_20260826_gpu23` 在约 step 39,560 停止；从完整的 `step_000034907_epoch_000001.pt` 恢复，未保存的约 4,653 step 被舍弃。
- 恢复 tmux 为 `mixed_seven_mano_robot_step5000_20260827_gpu23`，W&B run 为 `bt3bcb9l`；沿用原输出目录和 optimizer/scheduler 状态。

**改动原因**

单个七域 epoch 约 34,907 step、约 17 小时；只按 epoch 保存会使意外中断时的最大回退过大。每 5,000 step 保存将正常运行时的最大回退压缩到约 2.4 小时。

**验证**

- `latest.pt` 在恢复前可完整加载，记录 step 34,907 / epoch 1，model、optimizer 和 scheduler 状态均存在。
- 恢复日志确认从 step 34,907 加载；序列化配置确认 `save_every_steps=5000`、`save_every_epochs=null`。
- `step_000035000_epoch_000002.pt` 已实际写出，约 1.02 GB，并成为新的 `latest.pt`；训练进程继续运行。

## 2026-08-28 — 核验七域训练运行状态

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 状态记录

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新当前进度、checkpoint 和 data-wait 风险。

**验证**

- tmux `mixed_seven_mano_robot_step5000_20260827_gpu23` 和两 rank 进程仍在运行，无 traceback、OOM、ChildFailedError 或 SIGTERM/SIGKILL。
- checkpoint 已滚动保存到 step 80,000；普通 checkpoint 保持 5 个，`latest.pt` 正常更新。
- 当前约 step 84,300 / epoch 3；最近 500 step data-wait 约占 38%，最新记录约 55–66%，但进程持续推进，属于数据读取抖动而非训练崩溃。
- epoch 2 七域 clean macro QFL `1.79797e-4`，高于 epoch 1 的 `1.48572e-4`，暂记为轻微验证回升，等待后续 epoch 判断趋势。

## 2026-08-29 — 生成 object-frame 512 点随机采样可视化

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 诊断脚本与可视化

**文件与产物**

- `src/task/correspondence_ptv3_v2/research/object_centered_512/visualize.py` — 新增诊断脚本：从 Stage 3 的 `hand_root` 通过 `hand_root_pose` 和 `obj_root_pose_world` 转到 object frame，随机抽取 512 个物体点，并与完整 4096 点池及手点对比绘图。
- `output/research/object_centered_512/` — 生成 GRAB 与 HRDexDB Inspire DFTP 真实样本的 512 点可视化（输出目录被 gitignore）。

**验证**

- GRAB `airplane_fly_1_right` frame 100：object-frame 与 canonical pool RMSE `2.37e-8 m`；4096/512 点最近邻平均间距 `1.23/3.11 mm`。
- HRDexDB Inspire DFTP `attached_container_0_right` frame 0：RMSE `1.92e-8 m`；4096/512 点最近邻平均间距 `2.44/5.70 mm`。
- 当前 OakInk true-hand-root NPZ 不含 `obj_root_pose_world` 或 canonical object pool，暂不能在不补充导出字段的情况下做严格 object-frame 转换。

## 2026-08-29 — 新增 GRAB + Inspire F1 object-centered root/pose 训练协议

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 数据坐标与训练增强

**文件**

- `src/task/correspondence_ptv3_v2/sampling.py` — 新增以手腕根为中心的局部 SE(3) 扰动。
- `src/task/correspondence_ptv3_v2/dataset.py` — 支持 hand-root / hand-pose / clean 的 4:4:2 互斥采样；支持将既有 hand-root Stage3 在读取时转换到 object frame；传递 object pose 给 MANO/robot 重建。
- `src/task/correspondence_ptv3_v2/runner.py`、`robot_recon.py` — MANO 与 robot FK 输出按 object frame 正确回投。
- `src/task/correspondence_ptv3_v2/config.py` — 增加手腕根扰动参数。
- `src/task/correspondence_ptv3_v2/configs/mixed_grab_inspire_f1_object_centered_root_pose.yaml` — 新实验配置，仅 GRAB 与 Inspire F1，不含 OakInk/OakInk。
- `src/task/correspondence_ptv3_v2/research/object_centered_512/convert_stage3_to_object.py` — 可选离线转换脚本；默认训练采用惰性转换，避免复制约 72 GB NAS 数据。

**改动原因**

物体坐标系下物体侧不加扰动，所有位姿误差等效施加到手侧；手腕根刚体扰动与 MANO/robot 姿态扰动互斥，训练比例固定为 40%/40%/20%。对象池仍从完整 4096 点运行时均匀抽样 512 点。

**验证**

- 新配置成功加载；两个域共 479,440 个训练帧，生成 `val_clean`、`val_perturbed`（姿态）和 `val_root_perturbed` 三类验证 loader。
- GRAB 单文件惰性转换与离线转换结果 object-frame/canonical RMSE 约 `2.4e-8 m`；根扰动 smoke 的采样计数符合约 4:4:2，扰动 RMS 为毫米级至厘米级。

## 2026-08-29 — 启动 object-centered GRAB + Inspire F1 训练

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 训练运行

**运行**

- 停止旧 tmux `mixed_seven_mano_robot_step5000_20260827_gpu23`，释放 GPU 2、3；未触碰 GPU 0、1 上的其他任务。
- 新 tmux：`mixed_grab_inspire_f1_object_centered_20260829_gpu23`。
- W&B：`fjhdnx97`；配置使用 `mixed_grab_inspire_f1_object_centered_root_pose.yaml`。

**验证**

- DDP world size 2、每卡 batch 56、global batch 112、总步数 264,300 初始化成功。
- 首个训练诊断已完成至 step 20，无 traceback、OOM 或 DDP 异常；GPU 2、3 均已进入计算状态。

## 2026-08-29 — 构建 object-frame sidecar 并降低 NAS 读取并发

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 数据 I/O 优化

**文件**

- `src/task/correspondence_ptv3_v2/research/io_cache/build_uncompressed_npz_cache.py` — sidecar 支持直接写入 object-frame 数组；`build_cache` 的坐标系参数保持向后兼容。
- `src/task/correspondence_ptv3_v2/configs/mixed_grab_inspire_f1_object_centered_root_pose.yaml` — 启用 GRAB/F1 完整 sidecar，训练 worker 改为每 rank 2 个、`prefetch_factor=1`。

**缓存与验证**

- GRAB `1963/1963` 文件、Inspire F1 `576/576` 文件已完成；缓存格式 `ref2dex_uncompressed_npz_v2`，实际占用约 `92 GB`，源树未修改。
- 缓存强制读取的 dataset smoke 通过；2 workers、batch 56 的 5 batch benchmark data-wait 均值约 `2.12 s`（首次 batch `2.55 s`），8 workers 在 NAS 上出现大量 D-state NFS 等待，因此采用 2 workers + 单批预取。
- `tests/test_correspondence_array_cache.py`：`2 passed`。

**改动原因**

训练 data-wait 的主要瓶颈是随机跨文件访问、压缩 NPZ 解码和整序列 object-frame 转换叠加；sidecar 消除后两项，降低读取并发避免 NAS 饱和。

## 2026-08-29 — 域均衡 sampler 增加序列局部性

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 采样顺序优化

**文件**

- `src/task/correspondence_ptv3_v2/dataset.py` — `DomainBalancedSampler` 在保留每 batch 域配额的前提下，按 NPZ 文件顺序生成域内 stream；通过 `sequence_locality_shuffle` 控制，关闭时保持原随机行为。
- `src/task/correspondence_ptv3_v2/configs/mixed_grab_inspire_f1_object_centered_root_pose.yaml` — 已启用该局部性配置。

**改动原因**

原域均衡 sampler 对每个域做全局 frame permutation，导致相邻样本几乎总来自不同文件，抵消 worker 的单文件 cache。现在每个域先随机文件顺序、再随机文件内帧顺序，仍保持 4:4:2 扰动和 GRAB/F1 等比例，只改变 IO 友好的样本顺序。

## 2026-08-29 — 重启缓存与序列局部性训练

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 训练运行

**运行**

- 停止仅完成约 40 step 的无局部性试跑；未生成需保留的 checkpoint。
- 新 tmux：`mixed_grab_inspire_f1_object_centered_cache_locality_20260829`，GPU 2、3，W&B run 名为 `grab_inspire_f1_object_centered_cache_locality_20260829`。

**验证**

- 新 run 已推进至 step 80，无 traceback、OOM 或 DDP 异常；两卡均在计算。
- step 80 的 `perf/step_ms=677.953`、`perf/data_wait_ms=0.234`、`perf/data_wait_ratio=0.000345`、`perf/eta_hours=49.758`；相较原 run 的约 3.9 s/step、data-wait 80–85%，当前估计总训练时间约 50 小时。

## 2026-08-30 — 统计各数据集 stride=1 相邻帧手部运动幅度

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 诊断脚本

**文件**

- `src/task/correspondence_ptv3_v2/research/hand_motion_stats.py` — 新增全量诊断脚本，统计相邻存储帧的 1538 个手点平均位移、RMS 位移和手腕根平移；有 `obj_root_pose_world` 的数据同时计算 object-frame 结果。

**验证**

- 扫描当前 NAS 上 GRAB、ContactPose、OakInk、HRDexDB human、Inspire DFTP/F1、Allegro V5 全部 Stage 3 NPZ。
- 结果单位为 mm/相邻存储帧；未改变数据或训练配置。GRAB 的 `raw_frame_id` 间隔恒为 4，OakInk 有 24.1% 非连续 raw id，因此两者不能直接等同于原始视频 stride=1。

## 2026-08-30 — 对比 Inspire F1 时间 stride 对运动幅度的影响

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 诊断脚本

**文件**

- `src/task/correspondence_ptv3_v2/research/compare_temporal_stride.py` — 新增按指定 temporal stride 计算 hand-point displacement 的诊断脚本。

**验证**

- Inspire F1 object-frame RMS：stride=1 为 `2.515 mm` 均值，stride=2 为 `4.291 mm`，stride=3 为 `5.933 mm`。
- 当前 GRAB Stage 3（raw frame 间隔恒为 4）的 object-frame RMS 均值为 `6.190 mm`；因此 F1 stride=3 比 stride=2 更接近 GRAB 的幅度。

## 2026-08-30 — 刷新当前训练状态快照

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 状态记录

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 将缓存与序列局部性训练状态更新至 step 87500 / epoch 17，并记录当前速度、data-wait、ETA 与 checkpoint 进度。

**改动原因**

根据 2026-08-30 UTC 最新训练日志和 GPU 检查结果刷新状态快照，便于后续接手和判断训练是否异常。

## 2026-08-30 — 更新训练收敛状态

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 状态记录

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新至 step 100000 / epoch 19，并记录验证集最佳点和平台期判断。

**改动原因**

最新 epoch 汇总显示训练 loss 仍缓慢下降，但 clean validation QFL 的最佳 checkpoint 停留在 step 68718 / epoch 13，后续指标主要波动，需区分“优化仍在进行”和“验证性能已收敛”。

## 2026-08-31 — 修复外部评估器对混合 checkpoint 的数据根切换

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 评估入口

**文件**

- `src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py` — 外部单数据集评估覆盖 `data.val_path` 时显式清空 checkpoint 携带的 `data.domain_paths`。

**改动原因**

混合训练 checkpoint 的 `domain_paths` 会优先进入域均衡 dataloader 分支，使 ARCTIC 外部 `val_path` 被忽略并报无评估数据；清空该字段后才能按评估器指定的单一测试根构建 loader，不改变 checkpoint 权重或训练配置。

补充修复：外部 ARCTIC 是单一 human 数据根，显式关闭从混合训练 checkpoint 继承的 `mixed_hand_reconstruction`、robot reconstruction 和 robot perturbation，避免其被错误要求同时具备 MANO/robot 字段合同。

再次补充：当 checkpoint 的目标坐标系为 `object` 时，评估器启用已有的 hand-root→object 惰性转换；该转换要求测试树提供 `hand_root_pose` 和 `obj_root_pose_world`，因此本次改用带完整 root pose 的 `arctic_full_mano_v21_20260823`，不把 hand-root 数据强行当作 object frame。

## 2026-08-31 — 完成当前 best checkpoint 的 ARCTIC min11 快速评估

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 外部评估

**文件**

- `output/research/arctic_v1/stratified/object_centered_best_{object_only,hand_only,hand_object}_20260831.json` — 记录 11 序列、4175 帧三种扰动条件的评估结果。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新训练至 step 183500 / epoch 35，并记录 best 已刷新至 step 148008 / epoch 28。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 新增 EXP-015，记录 ARCTIC min11 快速外部评估及其限制。

**改动原因**

用户要求检查当前模型效果；全量 ARCTIC 逐帧评估 I/O 过慢，因此采用已有 min11 子集获得方向性结果，并明确标注为 INCONCLUSIVE。

补充修复：外部评估器不再继承训练期 `filter_non_interacting_frames`，确保 HOCap 评估使用完整 23,896 帧，和既有 HOCap 基线保持一致。

## 2026-08-31 — 完成当前 best checkpoint 的 HOCap 全量 object-only 评估

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 外部评估

**文件**

- `output/research/hocap_subject1_object_centered_20260831/object_centered_best_object_only_full.json` — 记录 HOCap subject_1 的 28 个序列、23896 帧 clean/object-only 评估结果。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 标记 HOCap 全量 object-only 外部评估完成。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 新增 EXP-016，记录结果、baseline 对照和限制。

**改动原因**

用户要求使用 HOCap 验证当前 checkpoint；在关闭训练期交互帧过滤并启用 hand-root→object 转换后完成全量评估。

## 2026-08-31 — 刷新训练收敛状态

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 状态记录

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新至 step 203180 / epoch 39，并记录 epoch 28 后验证 best 未刷新、当前进入实用平台期。

**改动原因**

根据最新训练 epoch 汇总和 checkpoint 元数据复核收敛情况：训练 loss 仍缓慢下降，但验证 clean QFL 已连续约 11 个 epoch 未超过 step 148008 / epoch 28 的 best。

## 2026-08-31 — 修复七域混合 batch 的可选 object pose 字段契约

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 数据加载

**文件**

- `src/task/correspondence_ptv3_v2/dataset.py` — 对缺失 `obj_root_pose_world` 的 hand-root 样本写入 identity 占位；object-frame 仍对缺失字段 fail-fast。

**改动原因**

七域续训从 `latest.pt`（step 145000）启动后，混合 batch 中部分域缺少该可选字段，PyTorch default collate 抛出 `KeyError`。统一 batch schema 后 hand-root 训练不改变数值语义；object-frame 训练仍不会掩盖输入数据缺失。

**验证**

- `tests/test_correspondence_ptv3_v2.py` 与 `tests/test_correspondence_array_cache.py`：22 passed。

## 2026-09-02 — 修正扩展训练 cache 路径并启动七域训练

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 训练运行

**文件**

- `src/task/correspondence_ptv3_v2/configs/mixed_grab_arctic_hrdexdb_oakink2_object_centered.yaml` — 将 HRDexDB 四域 cache 路径与实际生成目录 `hrdex_*` 对齐。
- `/tmp/launch_expanded_training_20260901.sh` — 同步修正自动编排脚本的 cache manifest 检查路径。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新 OakInk2/cache 完成及训练启动状态。

**改动原因**

cache 构建器已成功生成 `hrdex_human`、`hrdex_inspire_dftp`、`hrdex_inspire_f1`、`hrdex_allegro_v5`，而配置原使用 `hrdexdb_*` 前缀，导致编排脚本在训练前 fail-fast。修正后七域训练于 2026-09-02 UTC 在 GPU 2/3 从头启动。

**验证**

- 五个 object-frame cache manifest 均存在；配置成功加载七个 domain。
- 训练进程（torchrun + 两个 rank）持续运行，启动约 5 分钟时仍在读取首批 NAS cache 文件，未出现 traceback/OOM。

## 2026-09-02 — 修复 DataLoader worker 的 HRDexDB 导入冲突

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 机器人重建运行时

**文件**

- `src/task/correspondence_ptv3_v2/robot_recon.py` — 当 worker 将任务内 `dataset.py` 错误解析为顶层 `dataset` 时，改用文件路径动态加载自包含的 HRDexDB IO 模块。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 记录首次训练失败原因及重启状态。

**改动原因**

首次七域训练在首个机器人样本进入 `perturb_robot_hand_from_frame` 时抛出 `ModuleNotFoundError: No module named 'dataset.HRDexDB'; 'dataset' is not a package`。该问题仅发生在 DataLoader worker 的模块名冲突，不是数据字段或模型错误；动态 fallback 不依赖 PYTHONPATH 顺序。

**验证**

- 在人为注入顶层 `dataset` 冲突的环境中，fallback 成功加载 `xarm_inspire_f1_right.urdf` 并完成 URDF 解析。

## 2026-09-03 — 核验扩展七域训练进度

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部状态记录

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 更新当前 step、checkpoint 和 ETA。

**改动原因**

响应训练状态查询。训练已从首个 batch 正常推进至 step 143,700，data-wait ratio 约 `0.00027`，GPU 2/3 利用率约 100%，无异常退出。

## 2026-09-01 — 设置扩展数据准备完成后的自动校验与训练启动

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 异步运行编排

**文件 / 运行**

- `/tmp/launch_expanded_training_20260901.sh` — 等待 OakInk2/ARCTIC/HRDexDB 后台任务结束，执行 OakInk2 object-disjoint split，检查五个 cache manifest，然后用 GPU 2、3 从头启动新配置。
- tmux `expanded_object_centered_prepare_and_train_20260901` — 后台等待与训练编排会话。

**改动原因**

数据转换和 cache 构建均为长任务；集中编排可避免在数据未完整时误启动训练，并保证 cache 缺失时 fail-fast。

## 2026-09-01 — 完成七域混合续训并更新状态

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 训练运行状态

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 记录七域续训完成、最终 step 及 best/latest checkpoint 选择建议。

**改动原因**

七域续训已从 step 145000 正常完成到配置的 `max_steps=154670`；最终 clean macro QFL 为 `0.000223831`，历史 `best.pt` 仍为 step 34907、`0.000148572`。

## 2026-09-01 — 完成七域 best 的 HOCap 全量 object-only 评测

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 外部评估

**文件**

- `output/research/hocap_subject1_seven_domain_20260901/object_only.json` — 记录七域 best 在 HOCap subject_1 全部 28 序列、23896 帧上的 object-only 结果。
- `src/task/correspondence_ptv3_v2/docs/logs/{status_log,experiment_log}.md` — 同步评测结果及 HOCap MANO 元信息限制。

**改动原因**

响应用户要求，对七域续训后的最佳 checkpoint 进行 HOCap 全域评测。object-only 已正常完成；尝试 hand-only / hand-object 时确认 HOCap Stage 3 将 axis-angle45 错标为 `mano_use_pca=True`，因此暂停这两种条件，避免生成错误科研结果。

## 2026-09-01 — 补齐 HOCap PCA45 profile、重导出并完成三条件评测

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / HOCap 派生数据与外部评估

**文件**

- `src/task/correspondence_ptv3_v2/calibration/hocap_pca45_target_9mm.json` — 新增 HOCap 左右手 PCA45 几何噪声标定。
- `src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py` — HOCap 自动选择 `hocap` dataset ID、PCA45 profile，并在结果协议中记录 `pca45`。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hocap_subject1_annotation_v1` — 使用 `hocopt` 环境完整重导出 7 个序列 / 28 文件 / 23896 帧。
- `output/research/hocap_subject1_seven_domain_reexport_20260901/{object_only,hand_only,hand_object}.json` — 保存三种条件评测结果。
- `src/task/correspondence_ptv3_v2/docs/logs/{status_log,experiment_log}.md` — 同步 EXP-018 和当前限制。

**改动原因**

此前 hand-only / hand-object 评测因缺少 HOCap PCA45 profile 失败。复核确认 HOCap 原始参数和导出器均使用 PCA45，故保留正确表示，仅补标定 profile 与评测入口兼容逻辑，并重新导出以验证链路完整性。

**验证**

- HOCap 重导出日志：7 个序列、28 个文件、23896 帧，`failed=0`；字段签名统一为 `(mano_use_pca=True, mano_num_pca_comps=45, mano_pose_repr='pca')`，finite 检查通过。
- `tests/test_correspondence_ptv3_v2.py` 与 `tests/test_correspondence_array_cache.py`：22 passed。

## 2026-09-01 — 准备扩展 object-centered 多域训练与 OakInk2 转换

- branch: oyx
- post-commit: working tree（未提交）
- scope: task 内部 / 数据处理与训练配置

**文件**

- `src/task/correspondence_ptv3_v2/research/oakink2_conversion/convert_oakink2_stage3.py` — 将 OakInk2 preview MANO 四元数、物体位姿和 canonical 点缓存转换为 object-centered Stage3；支持并行分片和静态 object pool 存储。
- `src/task/correspondence_ptv3_v2/research/oakink2_conversion/split_stage3_by_object.py` — 按 object ID 建立确定性的 object-disjoint train/val 软链接及 manifest。
- `src/task/correspondence_ptv3_v2/dataset.py` — 读取 `[4096,3]` 静态 object pool 时广播为逐帧视图，保持下游张量契约。
- `src/task/correspondence_ptv3_v2/configs/mixed_grab_arctic_hrdexdb_oakink2_object_centered.yaml` — 新增从头训练配置，包含 GRAB、全量 ARCTIC、HRDexDB 四手型和 OakInk2，沿用 object-centered 4:4:2/512 点/每 5000 step 保存协议。
- NAS `processed_data/stage3/oakink2_object_centered_v1`、`processed_data/stage3_cache/object_centered_expanded_v1` — 后台生成 OakInk2 Stage3 与 ARCTIC/HRDexDB object-frame cache。

**改动原因**

用户确认扩大 object-centered 训练数据集并从头训练。OakInk2 原始 annotation 没有本项目 Stage3，故补充转换器；静态 canonical pool 避免逐帧重复写入约 TB 级冗余，同时不改变 Dataset/runner 的 4096→512 运行时采样语义。

**验证**

- OakInk2 单序列 smoke：6 个输出、62694 帧，CorrStaticDatasetV2 object-frame 读取和 MANO contract 通过。
- `tests/test_correspondence_ptv3_v2.py` 与 `tests/test_correspondence_array_cache.py`：22 passed。
