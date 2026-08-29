# Cm AI 修改记录

## 2026-08-29 — 完成 Cm 实现迁移并移除兼容壳

- branch: `feature/modular-component-runtime`
- post-commit: 本提交（以 `git log` 为准）
- scope: task 内部 / 破坏性结构迁移
- change_level: L3（破坏性结构迁移）
- approval: 用户已明确确认完全搬迁，并授权纳入当前未提交修改

**文件**

- `src/task/Cm/` — 从顶层迁入唯一真实运行实现、数据、可视化、配置和任务文档。
- `src/task/CmDecoder/`、`src/task/InteractionDynamics/`、`components/ref2dex/cm/`、`process/GRAB/`、`tests/`、`tools/`、`docs/logs/` — 同步迁移后的入口和当前已授权实验改动。

**改动原因**

用户明确要求完全搬迁，不再维护旧壳；当前未提交的科研改动一并作为迁移内容纳入本次提交。这样新增组件和配置只需进入规范目录，不会继续产生重复入口。

**验证**

- 41 个配置文件解析通过，新模块导入通过；完整测试 `300 passed, 3 skipped`；旧顶层入口不存在。

## 2026-08-28 — 增加 GRAB/Inspire-F1 混合手流重建训练

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 模型、数据与训练实验

**文件**

- `src/task/Cm/src/config.py`、`model.py`、`runner.py` — 增加几何手流 decoder、独立 scale/loss、source/stride 手流指标及旧 checkpoint 兼容加载。
- `src/task/Cm/dataset/hrdexdb.py` — 泛化 base source 与 HRDex bucket 命名/概率，支持严格 GRAB+Inspire-F1 两源，并清理混合 collate 的 source-only 字段。
- `src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml` — 新增冻结 DenseToken、两源等权、3 卡 global batch 96、50 epoch 的正式入口。
- `src/task/Cm/research/hand_flow_decoder/calibrate_hand_flow_scale.py` — 新增 train-only、source/stride 等权的完整手流 RMS 校准。
- `tests/test_cm_slot_attention.py`、`tests/test_cm_hrdexdb.py` — 覆盖 decoder shape/输入合同、手流指标与两源 collate。
- `src/task/Cm/docs/logs/architecture_log.md`、`experiment_log.md`、`decision_log.md`、`status_log.md`、`modification_log.md` — 同步架构、实验、决策和当前状态。

**改动原因**

用户要求停止 CmDecoder 训练，改为从当前 Inspire-F1 `latest.pt` 开始 GRAB/Inspire-F1 等权混合 Cm 训练，并新增不直接混入 `z_hand` 的手流重建辅助目标。

**验证 / 产物**

- 当前无活跃 CmDecoder/Cm train 进程，无需杀进程；旧 checkpoint 原样保留。
- 完整校准输出：`output/research/cm_hand_flow_decoder/hand_flow_scale.json`，RMS=`0.06986298856554198 m`。
- checkpoint 加载、两源 loader、真实混合 batch 2/32 GPU 前反向均通过；batch 32 峰值 reserved=`2300 MiB`。
- Cm slot/HRDexDB/ObjectV2/flow-scale 相关 pytest：`21 passed`；`compileall`、`git diff --check` 通过。
- 正式 DDP 已在 GPU 0/1/2 启动，output=`outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324`；step 500 无 OOM/NaN。

## 2026-08-29 — 同步 EXP-021 epoch 21 训练状态

- change_level: L0
- approval: auto
- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 状态与实验记录

**文件**

- `src/task/Cm/docs/logs/status_log.md` — 更新至 epoch 21 validation / epoch 22 training，并记录最新 best 与预计完成时间。
- `src/task/Cm/docs/logs/experiment_log.md` — 补充 EXP-021 当前 source 分项 EPE 和 hand/object 相对 epoch 1 的变化。

**原因**

响应训练状态核查，避免状态快照停留在 step 500。

**验证**

- 从 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324/metrics.jsonl` 读取 21 条完整 validation；训练进程仍存活，最新 step=`94380`。

## 2026-08-28 — 新增 Inspire-F1 微调 Cm 的 slot-wise t-SNE 诊断

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / latent visualization

**文件**

- `src/task/Cm/research/tsne_slots.py` — 新增统一加载 GRAB test 与 Inspire-F1 test、固定 stride=5、等量抽样、逐 slot t-SNE、原始空间 silhouette 统计和 PNG/NPZ/metadata 输出。
- `src/task/Cm/docs/logs/status_log.md`、`decision_log.md` — 记录诊断入口、参数选择和当前产物。

**改动原因**

响应用户对 Inspire-F1 微调后 Cm 表征是否与 GRAB 聚合的可视化需求；保持模型和训练流程不变，仅增加可复现的离线诊断入口。

**验证 / 产物**

- `py_compile`、`git diff --check` 通过。
- smoke：每边 8 条 transition 成功生成 16-slot 图。
- 正式诊断：每边 512 条 test transition，固定 stride=5，使用 decoder-only `latest.pt`（step 71850 / epoch 30）成功生成 `output/research/cm_tsne_inspire_f1_latest_stride5/{slot_tsne.png,slot_tsne.npz,metadata.json}`。

## 2026-08-27 — 同步 decoder-only continuation 的 epoch 14 验证状态

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 实验记录

**文件**

- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md` — 更新 Inspire-F1 decoder-only 至 step `34500`，补充 epoch 9--14 validation 对比及收敛判断。

**改动原因**

响应训练状态检查；当前验证最优为 epoch 14，但曲线仍有波动，实验状态保持 `INCONCLUSIVE`。

## 2026-08-27 — 切换 Inspire-F1 续训为冻结 DenseToken 的 decoder-only

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / HRDexDB 微调

**文件**

- `src/task/Cm/src/config.py` — 增加 checkpoint 是否保留 DenseToken 及 decoder-only resume 配置字段。
- `src/task/Cm/src/model.py` — 冻结阶段 continuation checkpoint 保留已适配的 DenseToken 权重。
- `src/task/Cm/src/runner.py` — decoder-only resume 仅注册可训练 decoder 参数，跳过源 DenseToken optimizer/scaler 状态并接续 scheduler/global step。
- `src/task/Cm/configs/active/hrdexdb_inspire_f1_decoder_only_resume.yaml` — 新增从 Inspire-F1 epoch 8 / step 19160 冻结 DenseToken 续训入口。
- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md`、`decision_log.md` — 同步运行状态、实验边界和决策。

**改动原因**

用户要求停止全量 DenseToken 微调，使用当前最新权重冻结 DenseToken，继续训练 decoder；保持在线 DenseToken（不使用 cache）及原有数据、loss、global batch 和 step 语义。

**验证 / 运行**

- 初次启动发现继承配置中的 GRAB `init_checkpoint` 会覆盖 Inspire-F1 source，已在实际训练前停止错误进程并将该字段置空；随后重新启动并确认日志只加载 Inspire-F1 source。
- `py_compile`、`git diff --check` 通过；当前 run 无 OOM/NaN。

## 2026-08-26 — 新增 Inspire-F1-only additive 微调入口并启动训练

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / HRDexDB 数据入口

**文件**

- `src/task/Cm/src/config.py` — 增加 HRDexDB-only/source-prefix 和 fresh `init_checkpoint` 配置字段。
- `src/task/Cm/src/runner.py` — 增加从初始化 checkpoint 加载模型权重、重置 fine-tune optimizer/step/epoch 的入口。
- `src/task/Cm/dataset/hrdexdb.py` — 支持按 manifest episode prefix 过滤，并提供 Inspire-F1-only loader（446/67/63）。
- `src/task/Cm/configs/active/hrdexdb_inspire_f1_finetune_cm64_additive.yaml` — 新增 C=64 additive、DenseToken 解冻、Inspire-F1-only、strict fresh budget 配置。
- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md`、`decision_log.md` — 同步微调假设、验证和运行状态。

**改动原因**

用户选择 Inspire-F1 专门微调，并确认采用 global batch=64、50 epoch、202300 steps 的预算，同时允许解冻 DenseToken。

**验证 / 运行**

- `py_compile`、`git diff --check` 通过；
- model checkpoint load smoke：head 权重无 missing/unexpected，DenseToken frozen-stage 缺失键按设计忽略；
- loader smoke：train/val/test=`446/67/63` episodes；

## 2026-08-26 — 更新 Inspire-F1 微调运行状态

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md` — 记录正式 run 已推进至 epoch 7 / step `16765` 及 epoch 6 validation 指标。

**改动原因**

响应训练状态检查，保持实验文档与实际产物同步。

## 2026-08-26 — 同步各 GPU 训练状态

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md` — 补充五组 DDP 任务的 GPU 占用、进度及 Inspire-F1 epoch 7 validation。

**改动原因**

响应 GPU 级训练状态检查，避免将不同实验的进度混在一起。

## 2026-08-26 — 将 C=32 additive 从 GPU 0/7 迁移到 GPU 0/2

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 运行资源调整

**文件**

- `src/task/Cm/docs/logs/status_log.md` — 记录 C=32 additive 的 DDP 资源迁移和恢复点。

**改动原因**

GPU 4/6 的 C=64 additive 已完成并退出；按用户要求保持原配置、global batch 和两卡 DDP，仅将 C=32 additive 的第二个 rank 从 GPU 7 改到空闲显存较多的 GPU 2。迁移从 `latest.pt`（epoch 27 / step 109242）恢复，随后已完成 epoch 28 validation 并进入 epoch 29；GPU 1/3 上的其他实验未停止。

## 2026-08-27 — 解释 Inspire-F1 zero-flow 指标并同步 epoch 8

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md` — 更新 Inspire-F1 至 epoch 8，并记录 stride 分解后的 zero-flow 指标解释。

**改动原因**

最新验证中 stride 1 的流幅度仅约 `1.07 mm`，其相对 zero-flow 退化会显著拉低等 stride 相对指标；stride 5/10 已优于 zero-flow，不能将总负值解释为模型在所有 stride 上均失败。
- 正式 run 在 GPU 1/3 启动，step 100--200 无 OOM/NaN。

## 2026-08-26 — 更新严格 C=32 hard-gate 与 C=16 additive 至 epoch 5

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/experiment_log.md` — 补充两条新 run 的 epoch 1--5 validation、objective/capacity 对照和 slot 使用诊断。
- `src/task/Cm/docs/logs/status_log.md` — 更新四条严格预算 run 的当前进度、风险和下一步。

**改动原因**

用户询问当前训练状态；同步首批能够比较 hard-gate ramp 前表现和 C=16 容量影响的正式验证证据。

## 2026-08-26 — 启动 C=32 hard-gate 严格对照与 C=16 additive

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 正式实验

**文件**

- `src/task/Cm/configs/active/object_v2_grab_gate_cm32_geometry_only_no_time_budget50_bs64.yaml` — 新增与 C=32 additive 对齐 global batch/steps 的 hard-gate objective-only 配置。
- `src/task/Cm/configs/active/object_v2_grab_additive_cm16_geometry_only_no_time_budget50.yaml` — 新增只将 Cm 宽度降为 16 的 additive 容量配置。
- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md`、`decision_log.md` — 记录实验边界、GPU 共用选择和启动状态。

**改动原因**

用户要求消除旧 C=32 hard-gate 与 additive 的训练预算混杂，并在显存允许时增加 C=16 additive 容量对照。

**验证 / 运行**

- 两份 output resolved config 已核验：均为 global batch=64、50 epoch、202300 steps、AdamW/cosine、seed42；目标与维度覆盖符合预期。
- C=32 hard-gate 至少 step 1000、C=16 additive 至少 step 200，无 OOM/NaN；GPU 2/3 每卡合计约 3.3 GB。

## 2026-08-25 — 记录 C=32 与 C=64 hard-gate 对齐结果

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/experiment_log.md` — 补充 C=32 epoch 15 验证结果、单 slot 坍缩证据，以及与 C=64 的同 epoch/同 optimizer-step 对齐比较。
- `src/task/Cm/docs/logs/status_log.md` — 更新 C=32 中途停止状态和当前严格预算 additive 进度。

**改动原因**

用户要求比较此前 C=32 与 C=64 版本；将可复现的日志证据同步到任务文档。

## 2026-08-25 — 澄清 C=32/C=64 hard-gate 配置差异

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/experiment_log.md` — 明确 C=32 与旧 C=64 共享模型/数据语义，但实际 global batch、DDP、每 epoch step 数和 cosine 总步数不同，不能称为纯 C-only ablation。

**改动原因**

用户询问两版是否为同一训练配置；根据原始 YAML 和 resolved config 核对后同步准确边界。

## 2026-08-25 — 记录 strict additive 前两轮 validation

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/experiment_log.md` — 补充 C=64 additive strict 的 epoch 1/2 指标和同 epoch 对比表。
- `src/task/Cm/docs/logs/status_log.md` — 更新严格预算 additive 已完成 epoch 2 validation 的状态。

**改动原因**

用户要求按相同 epoch 对比当前训练；同步实际 validation 证据并标注 additive 与 hard-gate 目标不同。

## 2026-08-25 — 更新 strict additive 至 epoch 8

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/experiment_log.md` — 补充 strict additive epoch 8 validation 与 C=64 hard-gate 同 epoch 对照。
- `src/task/Cm/docs/logs/status_log.md` — 更新三条训练的最新进度和风险判断。

**改动原因**

用户询问当前训练状态；同步最新可复现 validation 和 slot collapse 诊断。

## 2026-08-25 — 新增并启动 C=32 additive 严格预算对照

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 正式实验

**文件**

- `src/task/Cm/configs/active/object_v2_grab_additive_cm32_geometry_only_no_time_budget50.yaml` — 新增仅覆盖 `cm_dim=32` 的严格 additive 对照配置。
- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md`、`decision_log.md` — 记录实验假设、GPU 选择和启动状态。

**改动原因**

用户要求在与 C=64 additive 相同配置下启动 C=32 additive，检验降低 Cm 宽度后是否仍能避免坍缩并改善 hard-gate C=32。

**验证 / 运行**

- 启动命令使用 GPU 0/7、DDP world size=2；resolved setup 确认 global batch=64、total_steps=202300；step 100--300 无 OOM/NaN。

## 2026-08-25 — 更新 C=32 additive 至 epoch 6

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/docs/logs/experiment_log.md` — 补充 C=32 additive epoch 1--6 validation、slot 使用和同 epoch对照。
- `src/task/Cm/docs/logs/status_log.md` — 更新 C=32/C=64 additive 进度与 C=64 hard-gate 完成状态。

**改动原因**

用户询问当前训练表现；同步最新验证曲线和容量对照证据。

## 2026-08-25 — 启动 additive strict-budget 对照

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 正式实验

**文件**

- `src/task/Cm/configs/active/object_v2_grab_additive_cm64_geometry_only_no_time_budget50.yaml` — 新增 global batch=64、50 epoch、202300 steps 的严格预算配置。
- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md`、`decision_log.md` — 记录预算对齐选择和新 run 状态。

**改动原因**

用户要求消除 batch size、epoch 和 scheduler budget 混杂，重新训练 additive C=64。

**验证 / 运行**

- GPU 4/6、world size=2，启动日志确认 global batch=64、total_steps=202300；已稳定运行至至少 step 700。

## 2026-08-25 — 更新修正版 additive pilot 的 epoch 9 证据

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 正式实验

**文件**

- `src/task/Cm/docs/logs/status_log.md` — 更新到 epoch 10 运行状态及同 epoch 对照。
- `src/task/Cm/docs/logs/experiment_log.md` — 记录 epoch 1--9 EPE、zero-flow improvement 和 contribution usage 趋势。

**改动原因**

修正版已形成连续九个 validation 结果，需要区分 aggregate 性能改善与 effective-slot 压缩力度。

## 2026-08-24 — 增加 candidate-level mixture flow pilot

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 正式实验

**文件**

- `src/task/Cm/src/config.py` — 增加 `loss_candidate_mixture_weight` 与 `candidate_mixture_temperature`，默认关闭以保持旧 checkpoint 目标兼容。
- `src/task/Cm/src/model.py` — 暴露未 hard-mask 的 per-object-point routing weights，供 candidate mixture responsibility 计算。
- `src/task/Cm/src/runner.py` — 增加 `-tau logsumexp(log(pi)-candidate_loss/tau)` 的 candidate-level Smooth-L1 mixture loss 和 responsibility 诊断指标。
- `src/task/Cm/configs/active/object_v2_grab_mixture_cm64_geometry_only_no_time.yaml` — 新增 C=64、GRAB object-only、geometry-only/no-time、无 gate/count、tau=0.05、10 epoch 等效 pilot 配置。

**改动原因**

用户确认检验 candidate-level mixture objective 是否能让不同 slot 根据 object point 的 candidate loss 形成责任分工；首轮不引入 hard gate、count loss 或额外 balance loss。

**验证 / 运行**

- `py_compile` 与 `git diff --check` 通过。
- 首次启动的自动解析日志显示 `total_steps=80920`，核对 object-v2 loader 后确认这正是约 10 epoch；随后一次显式 `max_steps=16184` 试运行被停止，未完成首 epoch，不纳入实验结论。
- 正式 pilot 将在 GPU 7 重新启动，显式 `max_steps=80920`；前一条错误预算 output `outputs/cm/cm_object_v2_grab_mixture_cm64_geometry_only_no_time_20260824_185257` 仅保留为无效启动证据。
- 正式 pilot 已在 GPU 7 以 `outputs/cm/cm_object_v2_grab_mixture_cm64_geometry_only_no_time_20260824_190444` 启动并运行至至少 step 1700；当前无 OOM/NaN，尚无 validation 结论。
- mixture pilot 完成 epoch 1 validation 后，更新状态/实验日志记录首轮指标；修正 `runner.py` 中 responsibility effective-branch diagnostic 的熵公式（仅影响日志，不影响当前已加载进程的训练 loss）。
- 按用户确认新增 additive contribution pilot：`model.py` 增加每-slot 3-D contribution 求和路径，`runner.py` 增加 aggregate supervision 下的 group sparsity 和 contribution usage 诊断，`config.py` 增加 additive/group-sparsity 字段，新增 `object_v2_grab_additive_cm64_geometry_only_no_time.yaml`；candidate mixture 进程已停止，新 pilot 已在 GPU 7 启动并运行至至少 step 800。
- additive pilot 继续运行至至少 step 3500；补充状态/实验记录，当前只观察到高噪声 step-level EPE 和瞬时 contribution usage 波动，未提前形成性能结论。
- 修正 additive group-sparsity：由错误的 batch 总有效点归一化改为逐样本有效点均值后沿 slot 求和、再取 batch 均值；新增 batch-size invariant 单元测试。首 run 已停止并标记 `INVALID_IMPLEMENTATION`，准备从头启动修正版。
- 修正版 additive run 已在 GPU 7 从头启动，output 为 `outputs/cm/cm_object_v2_grab_additive_cm64_geometry_only_no_time_20260824_230244`，运行至至少 step 800，无 OOM/NaN。

## 2026-08-24 — 拆分 Cm 仓库记忆与机器记忆

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 全局文档规范联动

**文件**

- `repo_memory.md` — 由旧 `memory_log.md` 和 `repo_notes_log.md` 提炼可迁移的 cache、校准、兼容性和运行入口事实。
- `machine_memory.md` — 保存当前服务器的解释器、绝对路径和 W&B 服务异常，并由 Git 忽略。
- `status_log.md`、`architecture_log.md`、`experiment_log.md` — 更新接手记忆链接。
- `repo_notes_log.md` — 删除。

**改动原因**

遵循新的递归 memory 规范，避免 Cm 公共文档绑定单台服务器路径。

## 2026-08-24 — 完成 HRDexDB 全量 geometry cache 并修复坏 episode 预筛选

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 跨 task 共享 cache

**文件**

- `src/task/CmDecoder/build_cache.py` — 机器人 episode 预筛选现在要求 `C2R.npy` 存在；增加 `--reuse-schema-cache`，并为 `episodes=0` 的 object-disjoint 导出生成 `selection_all_object_disjoint_seed42.json`。
- `data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json` — 生成 2088 episode 的 train/val/test object-disjoint manifest（1642/232/214）。
- `src/task/Cm/docs/logs/status_log.md`、`src/task/CmDecoder/docs/logs/status_log.md` — 更新 cache 完成状态。

**改动原因**

首次全量 parent 在 `inspire_f1/banana/4` 缺少 `C2R.npy` 时退出；此前已完成的 2088 个 v4 cache 保持有效，修复后通过 schema reuse 复用，不重复计算。

## 2026-08-24 — 删除无逻辑 Cm 工具兼容 wrapper

- branch: working tree
- post-commit: 未提交
- scope: task 内部

**文件**

- 删除 `src/task/Cm/build_dense_cache.py`、`build_object_sampling_bank.py`、`build_object_v2_splits.py`、`build_sampling_bank.py`、`compute_flow_scale.py`、`compute_object_v2_stats.py` 六个薄 wrapper。
- `tests/test_cm_scene.py`、`tests/test_cm_object_v2.py`、`tests/test_cm_flow_scale.py`、当前指导/项目说明 — 统一改用 `src.task.Cm.tools.data.*`。

**改动原因**

用户确认不需要旧入口兼容；保留唯一真实实现，减少根目录重复入口。

**验证**

- Cm scene/object-v2/flow-scale/slot tests：30 passed。
- `py_compile` 与 `git diff --check` 通过。

- scope: task:Cm
- last_updated: 2026-08-23
- related: [当前状态](status_log.md)、[架构记录](architecture_log.md)、[实验记录](experiment_log.md)、[决策记录](decision_log.md)

## 2026-08-23 — 增加 C=32 geometry-only/no-time gate 实验并整理数据工具

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 正式实验

**文件**

- `src/task/Cm/src/model.py`、`config.py` — 新增向后兼容的 `use_object_context`；关闭时移除 `z_obj/object_context_encoder`，物体侧只保留 raw point/normal 与 Cm-relative geometry。
- `src/task/Cm/configs/active/object_v2_grab_gate_cm32_geometry_only_no_time.yaml` — 新增 GRAB object-only C=32、无时间、Hard-Concrete gate loss + 5+5 warmup 的从头训练配置；W&B 因证书问题使用 offline。
- `src/task/Cm/tools/data/` — 归类 sampling bank、split、dense cache 和统计/calibration 的独立工具实现；删除根级无逻辑 wrapper。
- `tests/test_cm_slot_attention.py` — 验证 geometry-only decoder 对 `z_obj` 扰动严格不变。
- `src/task/Cm/docs/logs/{architecture,status,memory,experiment,modification}_log.md` — 同步模型合同、运行状态、环境问题和 EXP-011。

**改动原因**

用户要求禁止 DenseToken object context 绕过 Cm 瓶颈，不使用时间条件，并用 GRAB 从头训练 C=32 gate 模型；随后要求整理仓库内数据工具。

**验证 / 运行**

- 相关 Cm/CmDecoder tests：36 passed；新工具模块 CLI 和配置入口通过。
- 正式训练在 GPU 7、session 22800 中运行，output 为 `outputs/cm/cm_object_v2_grab_gate_cm32_geometry_only_no_time_20260823_235356`；已稳定到至少 step 800，无 OOM/NaN。
- 首次 online W&B 启动在 step 0 前因过期 TLS 证书失败，offline 重启成功。

## 2026-08-23 — 修正 HRDexDB Cm candidate 语义并补齐 source×stride 评估

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 跨 task 共享 cache schema

**文件**

- `src/task/CmDecoder/build_cache.py` — geometry layer 增加稳定 4096 object pool、法向和逐帧 5cm candidate mask；legacy task layer 继续保存 512 点字段。
- `src/task/Cm/dataset/hrdexdb.py` — 训练时从 candidate pool 在线采样 512 点并正确构造 `obj_valid_mask`；缺少 candidate geometry 时 fail-fast；验证/测试按 source×stride 建立 loader。
- `src/task/Cm/src/runner.py` — 支持 source-qualified stride 指标，并恢复 `val/mean_stride_epe_mm` 的 checkpoint 选择入口。
- `src/task/Cm/configs/active/hrdexdb_finetune_cm64.yaml` — 指向待生成的全 embodiment cache/manifest，验证和测试固定报告 stride 1/5/10，保留 base flow scale。
- `tests/test_cm_hrdexdb.py` — 增加 candidate padding mask 与 source×stride 汇总回归测试。
- `docs/logs/architecture_log.md`、`src/task/Cm/docs/logs/{architecture,status,decision,modification}_log.md`、`src/task/CmDecoder/docs/logs/architecture_log.md` — 同步共享 geometry、评估和固定 scale 语义。

**改动原因**

此前 HRDexDB smoke cache 的全表面 512 点被错误地直接视为 Cm 输入，未遵守现有 5cm candidate 合同；同时单一验证 stride 无法生成 Cm 的最佳指标。此次只修正入口和评估框架，不导出全量 cache、不启动微调，也不改变用户确认的 flow scale。

**验证**

- `tests/test_cm_hrdexdb.py`、`tests/test_cmdecoder_build_cache.py`、`tests/test_cm_object_v2.py`：8 passed。
- `py_compile`：Cm HRDexDB loader、runner、CmDecoder builder 通过。

## 2026-08-23 — 两条 mixed 长训迁移到四张 GPU

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验运行状态与文档

**文件 / 运行状态**

- C=256：停止 GPU 2/3/4 的 3-rank 进程，从 step 118080 checkpoint 恢复到 GPU 2/3，per-device batch `48→72`。
- C=64：停止 GPU 1/6/7 的 3-rank 进程，从 step 84870 checkpoint 恢复到 GPU 1/6，per-device batch `48→72`。
- `output/exp/cm_v121/*_2gpu_bs72_resume_20260823.log` — 新增两条恢复 launcher 日志。
- `src/task/Cm/docs/logs/{status_log,memory_log,experiment_log,decision_log,modification_log}.md` — 记录 GPU 分配、恢复点、global batch、日志 step 回退和未落盘进度边界。

**改动原因**

用户要求两版训练合计从六张卡缩减到四张卡，同时保持 global batch 不变并释放两张卡给其他使用者。

**验证**

- 两版均为 world size 2、per-device batch 72、global batch 144；optimizer/scheduler 从 checkpoint 恢复。
- C=256 GPU 2/3 单卡约 `7.6--7.9 GB`；C=64 GPU 1/6 单卡约 `3.5--3.6 GB`；首步无 OOM/NaN。
- GPU 4/7 利用率 0%，各仅约 `0.27 GB` 驱动上下文。

## 2026-08-23 — Cm HRDexDB 微调强制在线 DenseToken

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/Cm/dataset/hrdexdb.py` — HRDexDB fine-tune loader 对 `data.use_dense_cache=true` fail-fast，明确只允许 geometry cache，DenseToken 特征在线计算。
- `src/task/Cm/src/model.py` — DenseToken 可训练时若 batch 意外携带 `cached_z_*`，立即 fail-fast，防止静默绕过在线 DenseToken。
- `src/task/Cm/configs/active/hrdexdb_finetune_cm64.yaml` — 显式设置 `data.use_dense_cache=false`。
- `src/task/Cm/docs/logs/{architecture,status,memory,modification}_log.md` — 同步 DenseToken 解冻时的 cache 边界和当前状态。

**改动原因**

用户确认采用“共享 geometry cache、DenseToken 在线训练”的方案，避免预提取 token 阻断 DenseToken 梯度。

**验证**

- 使用 MANO 与机器人 smoke geometry cache 构造 `HrdexdbGeometryDataset`，输出 `[1538,3]` hand、`[512,3]` object flow 和随机 stride 样本通过。

## 2026-08-22 — 重建 DexYCB cache 并完成 C=256 正式评测

- branch: `oyx`
- post-commit: 未提交
- scope: 跨 task 数据处理与 Cm 实验

**文件 / 数据 / 产物**

- `data/processed_data/stage4/data/dexycb` — 用修复 adapter 重建 subject-10/right 的 50 条序列、2853 帧。
- `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/` — 新建全量 50-stream test split 和显式占位 train metadata。
- `src/task/Cm/configs/active/eval_dexycb_subject10_c256_fixed_20260822.yaml` — 新增 C=256 固定 checkpoint 评测入口。
- `outputs/cm/cm_eval_dexycb_subject10_c256_fixed_20260822/eval.log` — 全量 stride 1/5/10 评测产物。
- `src/task/Cm/docs/logs/{status_log,memory_log,architecture_log,experiment_log,decision_log,modification_log}.md` — 同步路径、数据合同、EXP-010、当前状态与占位 split 决策。

**改动原因**

用户授权在修复 DexYCB adapter 后重建正式 cache 并测试，以判断旧泛化异常是否来自实现错误。

**验证**

- cache 50/50 sequences，0 skipped/failed，active-frame ratio 80.30%，刚体漂移 max `4.06e-5 mm`；
- C=256 step 36900 评测平均 EPE `14.92 mm`，相对 zero-flow 改善 `68.27%`；
- 实际 Runner 完成全部三个 stride 并正常退出。

## 2026-08-22 — 移除 DexYCB 失效 cache

- branch: `oyx`
- post-commit: 未提交
- scope: 跨 task 数据与 Cm 状态

**文件 / 数据**

- `data/processed_data/stage4/data/dexycb` — 489M 失效 cache 已移入系统回收站，可恢复。
- `src/task/Cm/docs/logs/{status_log,architecture_log,memory_log,modification_log}.md` — 记录删除状态、重建阻塞和当前 C=256/C=64 checkpoint 进度。

**改动原因**

防止修复前 DexYCB cache 被再次误用于泛化评估。

## 2026-08-22 — 修复 DexYCB 评估数据适配

- branch: `oyx`
- post-commit: 未提交
- scope: 跨 task（共享 `process/DexYCB`，Cm 为直接使用者）

**文件**

- `process/DexYCB/{raw.py,stage4_cm.py}` — 修复 object quaternion/SE(3)、MANO reference-camera/PCA/non-flat mean 解码和 right-hand capture 筛选。
- `tests/test_dexycb_raw.py` — 新增 6 个针对性回归测试。
- `src/task/Cm/docs/logs/{status_log,architecture_log,memory_log,experiment_log,modification_log}.md` — 记录旧评估 `INVALID_IMPLEMENTATION`、旧 cache 禁用和新数据合同。

**改动原因**

旧 DexYCB cache 的手物坐标不一致，且物体姿态和旋转应用方向均错误，无法用于 Cm 跨数据集评估。

**验证**

- `tests/test_dexycb_raw.py` + `tests/test_cm_sequence_dataset.py`: 10 passed。
- 官方 reference-camera `joint_3d` 对齐：修复后 21 关节平均 `0.78 mm`，旧解码约 `12.69 mm`。
- 真实 subject-10 sequence 不落盘 smoke 通过；未重建 cache，未重跑评估。

## 2026-08-21 — 补充 Cm 当前架构与张量规格

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部文档

**文件**

- `src/task/Cm/docs/logs/architecture_log.md` — 补充当前 DenseToken、Dataset、Slot Attention、gate、object flow decoder 的张量形状、单位、输入边界和 warm-up 行为。
- `src/task/Cm/docs/logs/modification_log.md` — 记录本次架构文档更新。

**改动原因**

用户要求将 Cm 当前架构和张量规格沉淀到任务级 architecture log，便于后续分析 gate、slot routing 和数据/模型接口时直接对照实际代码。

**影响范围**

仅 task 内部文档，不改变代码、配置或实验定义。

## 2026-08-20 — 重建并整理 subject-template 修复版 GRAB cache

- branch: `oyx`
- post-commit: `2842637`
- 范围: task 内部数据产物与文档记录

**文件 / 产物**

- `data/processed_data/cm_object_v2_subject_template_20260820/grab` — 新生成的 1335-sequence object-v2 GRAB cache，含 2670 个 sampling bank、固定 split 和 calibration metadata。
- `data/processed_data/cm_stage4_v2_subject_template_20260820` — 已在转换、统计和校验完成后删除的 Stage4 中间产物。
- 旧版 `data/processed_data/cm_stage4_v2`、`data/processed_data/cm_object_v2` — 保持不动。

**改动原因**

按 V2 数据脚本修复后的 subject-specific `v_template` 重新生成 GRAB cache，避免旧 cache 的平均 MANO 语义；为控制磁盘占用，转换完成后删除新 Stage4 中间目录。

**验证**

- Stage4 生成：1335 sequences / 2670 hand streams / 0 failed。
- object-v2 转换：1335 converted / 0 failed。
- train-only flow RMS：`0.0932494331 m`，train sequences `1067`。

## 2026-08-20 — 修正 V2 GRAB subject template 解析

- branch: `oyx`
- post-commit: `HEAD`
- 范围: 跨 task（共享 `process/GRAB`，Cm/V2 为直接使用者）

**文件**

- `process/GRAB/raw.py` — 支持两种 GRAB root 布局解析 subject asset，并为正式调用提供严格 template 检查。
- `process/GRAB/stage4_cm.py` — V2 Stage4 默认启用严格 subject template，记录解析 root 和 template policy，提供显式兼容开关。
- `tests/test_cm_sequence_dataset.py` — 覆盖 `dataset/GRAB` 布局下 `v_template` 路径解析。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 记录 V2 数据合同及旧 cache 重建要求。

**改动原因**

实测发现原始 GRAB 的 `vtemp` 相对路径位于 `dataset/GRAB/data/tools/...`，旧脚本直接拼接 `dataset/GRAB/tools/...` 后静默使用平均 MANO，导致手几何和 candidate mask 偏离 subject-specific 语义。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**下一步打算做什么**

用新输出目录重建 GRAB Stage4/object-v2 cache，再重新进行正式训练比较。

## 2026-08-18 — 建立 Cm 规范日志入口

- branch: 当前工作分支
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/docs/logs/repo_notes_log.md` — 建立 Cm 任务入口、状态和代码索引。
- `src/task/Cm/docs/logs/architecture_log.md` — 汇总 V1.2 pipeline、数据合同、评估和缓存路线。
- `src/task/Cm/docs/logs/decision_log.md` — 迁移现有 V1.2 cache 转换器决策。
- `src/task/Cm/docs/logs/experiment_log.md` — 见同目录实验历史摘要。

**改动原因**

按 `AGENTS.md` 将 Cm 现有 `log.md`、`decision_log.md`、框架和 Pipeline 文档整理到规范入口；保留原文档作为兼容/历史参考。

**影响范围**

仅 Cm 文档，不改变代码和实验定义。

## 2026-08-18 — 支持联合 object-v2 root 统计

- branch: 当前工作分支
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/dataset/object_v2.py` — 允许 `CmObjectV2Dataset` 读取包含 `grab/`、`arctic/` 子目录的联合 object-v2 root。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 EXP-001 full-data cache/E1 证据。

**改动原因**

V1.2 联合根目录可被 Runner 的 `_sequence_dirs()` 识别，但 E1 统计入口无法读取，导致 `compute_object_v2_stats --root cm_object_v2` 失败。修复后统计入口与 mixed config 的 root 语义一致。

**影响范围**

仅 Cm 数据读取和文档；不改变模型、loss 或 GT 定义。

## 2026-08-19 — 修正 Cm viewer 的 Python 3.8 兼容性

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/visualization/server.py` — 将 Python 3.9+ 的 `str.removesuffix()` 改为兼容 Python 3.8 的 `Path.stem`。

**改动原因**

仓库推荐环境为 Python 3.8.20；Scene Cache viewer 改动中的 `removesuffix()` 会在 legacy NPZ 推理路径运行时报错。

**影响范围**

仅 Cm viewer 的 side 名称解析，不改变推理结果。

## 2026-08-19 — 推进 V1.2.1 数据-only 混合实验链路

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/dataset/object_v2.py` — 用共享 epoch 和 LRU 打开缓存修复多进程读取稳定性，并支持固定 split。
- `src/task/Cm/compute_flow_scale.py` — 增加 object-v2 train-only flow calibration 入口。
- `src/task/Cm/build_object_v2_splits.py` — 生成 sequence 级固定 split 与 `splits.json`。
- `src/task/Cm/configs/active/object_v2_grab_arctic.yaml` — 固定为 no-gate + time condition，并接入固定 split 与校准标记。
- `src/task/Cm/src/runner.py` — 识别联合 `grab/` + `arctic/` root 的 object-v2 数据目录。
- `src/task/Cm/docs/指导/V1.2.1.md` — 研究指导版本，作为本轮实现与实验依据。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 更新任务入口状态与主要入口索引。
- `src/task/Cm/docs/logs/architecture_log.md` — 记录固定 split、校准和联合 root 的数据流约束。
- `src/task/Cm/docs/logs/decision_log.md` — 记录共享 epoch、LRU cache 和确定性 split 的实现选择。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 V1.2.1 混合短训实验结果与结论。

**改动原因**

按 V1.2.1 指导修复 worker epoch、cache 泄漏和训练 split，并按用户确认采用只改数据链路的 no-gate + time condition 方案，随后用 2-step smoke 与 3-seed 300-step 短训验证实现闭环。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 代码、配置、文档和短训验证都已完成
- 剩余: 下一轮对照实验
- 续接点: GRAB-only / ARCTIC-only / mixed 同预算比较

**项目阶段进度**（可选）

- 阶段: V1.2.1 mixed data-only baseline
- 进度: 1/2 个主要步骤完成
- 本次对应阶段中的第几步: 1

**下一步打算做什么**（可选）

继续跑 GRAB-only、ARCTIC-only 和 mixed 对照，必要时再加长训练步数。

## 2026-08-19 — 补齐 V1.2.1 单数据集对照配置与计数修正

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/compute_flow_scale.py` — object-v2 校准的 sequence 计数改为按完整路径去重，避免同名 sequence 被误合并。
- `src/task/Cm/configs/active/object_v2_grab_only.yaml` — GRAB-only 的 no-gate + time condition 对照配置。
- `src/task/Cm/configs/active/object_v2_arctic_only.yaml` — ARCTIC-only 的 no-gate + time condition 对照配置。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 GRAB-only / ARCTIC-only 小规模短训结果。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 更新当前研究状态和入口索引。
- `src/task/Cm/docs/logs/architecture_log.md` — 记录单数据集对照已经纳入 object-v2 管线。

**改动原因**

用户要求继续推进 V1.2.1，但又明确关心 seed 数量与训练规模；因此把后续工作收敛为同预算的单数据集对照，并修正校准元数据里的序列计数口径。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 单数据集对照配置已补齐，已完成 GRAB-only / ARCTIC-only 的小规模实验整理
- 剩余: 若要进一步判断假设，需要提高预算而不是增加重复 seed
- 续接点: 更长预算的 mixed / single-dataset 对照

**项目阶段进度**（可选）

- 阶段: V1.2.1 mixed + single-dataset baseline
- 进度: 2/2 个主要小规模验证步骤完成
- 本次对应阶段中的第几步: 2

**下一步打算做什么**（可选）

如果继续推进研究，应把 300-step 短训升级到更长预算；当前不建议再堆同级别 seed。


## 2026-08-19 — 记录 V1.2.1 吞吐 benchmark 与 DDP 兼容配置

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/configs/active/object_v2_grab_arctic.yaml` — 为 object-v2 mixed/full training 打开 `find_unused_parameters=true`。
- `src/task/Cm/configs/active/object_v2_grab_only.yaml` — 为 GRAB-only full training 打开 `find_unused_parameters=true`。
- `src/task/Cm/configs/active/object_v2_arctic_only.yaml` — 为 ARCTIC-only full training 打开 `find_unused_parameters=true`。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 记录 train-only 吞吐 benchmark 与推荐的 3 GPU DDP。
- `src/task/Cm/docs/logs/decision_log.md` — 记录 3 GPU DDP 的选型理由。

**改动原因**

用户要求开始 full-data 训练前先统计吞吐并选最合适的多卡方案；同时 2 GPU DDP 已经暴露 `no-gate` 路径上的 unused-parameter 问题，需要显式打开 `find_unused_parameters=true`。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 吞吐 benchmark 已完成，配置已适配 DDP
- 剩余: 等用户确认 full-training 具体范围后启动在线 wandb 长训
- 续接点: GRAB-only / ARCTIC-only / mixed 的正式 run

**项目阶段进度**（可选）

- 阶段: V1.2.1 full-data long-run prep
- 进度: benchmark + DDP 修正完成
- 本次对应阶段中的第几步: 1

**下一步打算做什么**（可选）

等待用户确认正式长训范围，然后用 3 GPU 启动在线 wandb 训练。


## 2026-08-19 — 记录 mixed full run 与 batch-size sweep 结论

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/docs/logs/repo_notes_log.md` — 记录 mixed 3 GPU batch-size sweep 的吞吐结果与正式推荐 batch。
- `src/task/Cm/docs/logs/decision_log.md` — 记录为什么正式长训选 3 GPU + batch_size=48。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 mixed full run（batch 24）完成事实与结果。

**改动原因**

用户要求按“多卡 + batch size 极限吞吐”选正式训练口径；因此先完成 train-only sweep，再把结果写回任务入口和自主决策记录。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 吞吐 sweep 已完成，mixed full run 已完成
- 剩余: 等待用户确认是否立即按 batch_size=48 继续长训 GRAB-only / ARCTIC-only / mixed
- 续接点: 正式长训启动

**项目阶段进度**（可选）

- 阶段: V1.2.1 long-run prep
- 进度: batch-size sweep 完成
- 本次对应阶段中的第几步: 2

**下一步打算做什么**（可选）

如果用户确认，就用 3 GPU + batch_size=48 开正式长训。

## 2026-08-19 — 将正式长训入口切到 batch 48 与 online wandb

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/configs/active/object_v2_grab_arctic.yaml` — mixed/full 训练默认 batch 调到 48，`wandb` 改为 online。
- `src/task/Cm/configs/active/object_v2_grab_only.yaml` — GRAB-only full 训练默认 batch 调到 48，`wandb` 改为 online。
- `src/task/Cm/configs/active/object_v2_arctic_only.yaml` — ARCTIC-only full 训练默认 batch 调到 48，`wandb` 改为 online。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 将正式长训入口描述更新为 3 GPU + batch 48。
- `src/task/Cm/docs/logs/decision_log.md` — 将正式长训决策锚定到 batch 48 的吞吐峰值。
- `src/task/Cm/docs/logs/experiment_log.md` — 追加 throughput sweep / mixed full pilot 的正式 EXP 记录。

**改动原因**

用户明确说明吞吐 benchmark 关注的是多卡 + batch size 的极限点，而不是单纯卡数；当前 sweep 已验证 3 GPU + batch 48 为 mixed 路线峰值，因此将正式长训入口统一收口到该设置，并把 pilot 与正式选择分开记录。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 正式长训入口已切到 48，下一步可直接启动在线训练
- 剩余: 等长训产出
- 续接点: mixed / GRAB-only / ARCTIC-only formal run

**项目阶段进度**（可选）

- 阶段: V1.2.1 full-data long-run prep
- 进度: 入口收口完成
- 本次对应阶段中的第几步: 3

**下一步打算做什么**（可选）

启动 3 GPU + batch 48 的正式长训。

## 2026-08-19 — 启动 mixed 正式长训

- branch: `oyx`
- post-commit: `3e32de8`
- 范围: task 内部

**文件**

- `output/exp/cm_v121/cm_v121_mixed_3gpu_bs48_full_20260819_165238.log` — mixed full-data 训练日志（运行产物，不纳入版本管理）。
- `src/task/Cm/docs/logs/experiment_log.md` — 将吞吐 EXP-004 锚定到提交 `3e32de8`。

**改动原因**

按 V1.2.1 的数据-only 约束和 EXP-004 吞吐结论，启动 3 GPU、per-device batch 48、global batch 144、online wandb 的 mixed 全量训练。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: torchrun 已启动，3 个 worker 正在加载 Frozen DenseToken 与 object-v2 数据。
- 剩余: 等待 10000-step 训练、验证和 checkpoint 完成。
- 续接点: 检查日志中的 `train_setup`、step 进度、W&B 初始化和最终 metrics。

**项目阶段进度**（可选）

- 阶段: V1.2.1 full-data long-run
- 进度: mixed 1/3 个正式数据路线已启动；GRAB-only / ARCTIC-only 待 mixed 完成后按相同吞吐设置启动。

**下一步打算做什么**（可选）

按分钟级检查 mixed 运行状态；完成后记录最终指标并启动两个 single-dataset 对照。

## 2026-08-19 — 将 mixed 正式预算延长到 50 epochs

- branch: `oyx`
- post-commit: `HEAD`
- 范围: task 内部

**文件**

- `src/task/Cm/configs/active/object_v2_grab_arctic.yaml` — 将 mixed 正式总预算设为 50 epochs、184650 steps。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 标明 10000 steps 是 pilot，正式入口为 50 epochs/184650 steps。
- `src/task/Cm/docs/logs/experiment_log.md` — 更新当前研究状态，记录从 10000-step checkpoint 续训。

**改动原因**

用户确认 10000 steps 仅作为 pilot 不足以代表 full long training，要求继续完成 50 epochs 的全量 mixed 训练。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 预算配置已改为 184650 steps，准备从 `outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/step_000010000_epoch_000003.pt` 续训。
- 剩余: 启动续训并确认新总步数、checkpoint 恢复和 cosine 学习率状态。
- 续接点: 检查新 run 的 `train_setup`、`Loaded checkpoint` 和首个 `perf/` 记录。

**项目阶段进度**（可选）

- 阶段: V1.2.1 mixed full-data long-run
- 进度: 10000-step pilot 已完成；50-epoch 正式长训待启动。

**下一步打算做什么**（可选）

用 3 GPU + batch 48 + online wandb 启动续训，并在完成后追加 EXP-005 的正式结果。

## 2026-08-19 — 启动 50-epoch mixed continuation

- branch: `oyx`
- post-commit: `815d55e`
- 范围: task 内部

**文件**

- `output/exp/cm_v121/cm_v121_mixed_3gpu_bs48_50ep_resume_20260819_195242.log` — 50-epoch mixed continuation 日志（运行产物，不纳入版本管理）。
- `output/exp/cm_v121/pilot_artifacts/cm_v121_mixed_3gpu_bs48_10k_pilot_step_10000.pt` — 10000-step pilot checkpoint 备份，避免续训的 checkpoint 保留策略覆盖 pilot 证据。
- `src/task/Cm/docs/logs/modification_log.md` — 记录续训入口、恢复点和 W&B run。

**改动原因**

用户确认将 10000-step pilot 延长到 50 epochs/184650 steps；从已有 step 10000 checkpoint 恢复，保持 3 GPU、batch 48、no-gate + time condition 和 online wandb。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 已验证 `total_steps=184650`、`Loaded checkpoint ... at step 10000`，首个续训 step 10100 的学习率为 `2.9779e-4`；W&B run 为 `cqzdih3m`。
- 剩余: 继续运行至 step 184650，并追加 EXP-005 的完整结果。
- 续接点: 检查 `output/exp/cm_v121/cm_v121_mixed_3gpu_bs48_50ep_resume_20260819_195242.log` 的 epoch/step 和最终 val/test 指标。

**项目阶段进度**（可选）

- 阶段: V1.2.1 mixed full-data long-run
- 进度: 50-epoch continuation 已启动；当前约 step 10100/184650。

**下一步打算做什么**（可选）

按分钟级检查续训状态；完成后记录最终指标、W&B 链接和 checkpoint，并决定是否启动 single-dataset 对照。

## 2026-08-19 — 新增 GRAB gate+cm64 独立训练配置

- branch: `oyx`
- post-commit: 待提交
- 范围: task 内部

**文件**

- `src/task/Cm/configs/active/object_v2_grab_gate_cm64.yaml` — 新增 GRAB-only、time-conditioned、slot gate、`cm_dim=64` 的 50-epoch 候选配置。
- `src/task/Cm/docs/logs/decision_log.md` — 记录保持 time condition、按 epoch 控制预算和重新 benchmark batch 的理由。
- `src/task/Cm/docs/logs/modification_log.md` — 记录本次配置与文档修改。

**改动原因**

用户要求并行准备一版只使用 GRAB、用 gate 限制 slot、把 slot embedding 维数从 256 降到 64 的训练。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`；这是用户明确追加的模型候选，超出 V1.2.1 原先“只改数据”的主对照，但不覆盖主配置。

**单次修改进度**（可选）

- 当前: 独立配置加载和模型构造已通过，确认 `cm_dim=64`、gate/time condition 均开启、可训练参数量 94373；`tests/test_cm_slot_attention.py` 为 9 passed。
- 剩余: 在真正空闲的多卡上重新 sweep batch size，再启动 50 epochs online wandb 正式训练。
- 续接点: `src/task/Cm/configs/active/object_v2_grab_gate_cm64.yaml`。

**项目阶段进度**（可选）

- 阶段: V1.2.1 composite candidate
- 进度: 配置准备完成，GPU benchmark/训练待执行。

**下一步打算做什么**（可选）

验证配置与模型接口；持续观察 GPU，出现不与其他任务冲突的 3 卡组合后执行 batch sweep。

## 2026-08-19 — 在共享 GPU 1、5 启动 GRAB gate+cm64 长训

- branch: `oyx`
- post-commit: `99d32d5`
- 范围: task 内部

**文件**

- `output/exp/cm_v121/cm_v121_grab_gate_cm64_2gpu_bs48_50ep_20260819_234923.log` — 共享两卡正式训练 stdout/stderr（运行产物，不纳入版本管理）。
- `outputs/cm/cm_object_v2_grab_gate_cm64_20260819_234926/` — config、metadata、metrics、checkpoint 与 W&B 本地目录（运行产物，不纳入版本管理）。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 记录共享卡吞吐不可与独占 benchmark 直接比较。
- `src/task/Cm/docs/logs/experiment_log.md` — 更新 mixed 与 GRAB gate+cm64 均在运行中的当前状态。

**改动原因**

用户确认允许在 mixed 已占用的 GPU 1、5 上共享启动 GRAB gate+cm64 候选，以显存可容纳为前提接受吞吐下降。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`；本候选为用户追加的复合模型实验。

**单次修改进度**（可选）

- 当前: 2 GPU DDP 已启动；global batch 96，50 epochs 自动解析为 134900 steps；W&B run `z2t2b5mi`。step 100–200 初始吞吐约 111 samples/s，预计约 32.4 小时。
- 剩余: 持续训练并观察共享对 mixed 吞吐的影响；完成后追加正式 EXP。
- 续接点: 训练日志与 `outputs/cm/cm_object_v2_grab_gate_cm64_20260819_234926/metrics.jsonl`。

**项目阶段进度**（可选）

- 阶段: V1.2.1 composite candidate
- 进度: GRAB gate+cm64 50-epoch 正式训练已启动。

**下一步打算做什么**（可选）

确认两个任务在共享卡下持续前进且无 OOM/NaN；完成后分别整理 mixed 和复合候选的正式 EXP。

## 2026-08-20 — 将 GRAB gate+cm64 从共享 GPU 迁移到 GPU 6、7

- branch: `oyx`
- post-commit: `878da54`
- 范围: task 内部

**文件**

- `output/exp/cm_v121/cm_v121_grab_gate_cm64_2gpu_bs48_50ep_gpu67_resume_20260820_090438.log` — GPU 6、7 续训日志（运行产物，不纳入版本管理）。
- `output/exp/cm_v121/migration_artifacts/grab_gate_cm64_step_16188_epoch6.pt` — 迁移前安全 checkpoint 备份。
- `src/task/Cm/docs/logs/repo_notes_log.md` — 更新迁移后 GPU 与吞吐状态。
- `src/task/Cm/docs/logs/experiment_log.md` — 更新当前运行位置。

**改动原因**

用户确认将 GRAB gate+cm64 从共享的 GPU 1、5 移到已空闲的 GPU 6、7，以恢复吞吐并降低对 mixed 的资源干扰。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**（可选）

- 当前: 原 rank 已停止；从 step 16188/epoch 6 恢复到 GPU 6、7，W&B 新 run 为 `o6zlc1nu`，首个完整 step 吞吐约 270 samples/s。
- 剩余: 重跑 epoch 7 并继续至 step 134900；验证迁移后 mixed 吞吐是否恢复。
- 续接点: `output/exp/cm_v121/cm_v121_grab_gate_cm64_2gpu_bs48_50ep_gpu67_resume_20260820_090438.log`。

**项目阶段进度**（可选）

- 阶段: V1.2.1 composite candidate
- 进度: GRAB gate+cm64 已完成资源迁移，继续正式长训。

**下一步打算做什么**（可选）

按分钟级观察 GPU 6、7 上的稳定吞吐和 mixed 恢复情况；完成后追加正式 EXP 结果。

## 2026-08-20 — 增加 gate warm-up 配置与训练策略

- branch: `oyx`
- post-commit: `453806a`
- 范围: task 内部

**文件**

- `src/task/Cm/src/config.py` — 增加 gate warm-up 开关、全开 epoch 和渐进 epoch 配置。
- `src/task/Cm/src/model.py` — 支持按 epoch 动态设置 gate threshold，并在 warm-up 阶段强制所有 slot 参与 decoder。
- `src/task/Cm/src/runner.py` — 实现 5 epoch 全开、5 epoch threshold/count weight 线性 ramp 的调度与指标记录。
- `src/task/Cm/configs/active/object_v2_grab_gate_cm64_warmup.yaml` — 新增 GRAB-only gate+cm64 warm-up 正式训练入口。
- `tests/test_cm_slot_attention.py` — 增加 warm-up 全 slot 路径和调度测试。

**改动原因**

原 gate+cm64 训练出现 effective branch count 约 1、global top-1 usage 约 1 的严重 slot collapse。用户确认采用 warm-up 后重新训练。

**对应指导**（如有）

`src/task/Cm/docs/指导/V1.2.1.md`

**单次修改进度**

- 当前: 代码、配置、单测、配置调度 smoke 和独立正式训练启动均已完成；W&B run `8grohy8u`，GPU 0、1 共享运行。
- 剩余: 持续观察第 1、5、10 个 epoch 的 slot 使用和 EPE，确认渐进 gate 是否避免 collapse。
- 续接点: `output/exp/cm_v121/cm_v121_grab_gate_cm64_warmup_2gpu_bs48_50ep_gpu01_20260820_111403.log`。

**项目阶段进度**

- 阶段: V1.2.1 gate warm-up candidate
- 进度: 代码实现完成，实验尚未开始。
## 2026-08-22 — 新增长程训练结果对比表

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/results/object_v2_training_comparison.md` — 汇总 mixed、GRAB gate+cm64 和 warm-up 长程 run 的关键变量、最佳指标、收敛判断、数据有效性风险及绝对路径，并单列未纳入的短训。
- `src/task/Cm/docs/logs/decision_log.md` — 记录主比较采用至少 20 个完整 epoch 的筛选口径。

**改动原因**

用户要求把当前 experiment log 中训练预算足够的结果放在同一文档中进行表格化比较，并记录配置、cache、metrics、checkpoint 和日志路径。

## 2026-08-22 — 补充旧 Stage4 GRAB、C=256 长训对照

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/results/object_v2_training_comparison.md` — 补入旧 Stage4 GRAB 上 C=256 的原始缩放 gate/no-gate、校准缩放 no-gate，以及单手筛选 gate/no-gate 五条 20-epoch 长训；增加变量、指标、routing、收敛与绝对路径表。

**改动原因**

首次汇总错误地把范围收窄到 object-v2 主线，遗漏了用户指出的旧 GRAB 数据 C=256 gate/no-gate 正式长训。此次从实际 `config.json` 和 `metrics.jsonl` 重新核验后补齐。

## 2026-08-22 — 建立 subject-template 修复版 mixed cache 与训练入口

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部

**文件 / 产物**

- `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/` — 新建 sequence 级相对软链接 cache，复用修复版 GRAB 和既有 ARCTIC；生成 seed42 split、全量 statistics 与 train-only calibration。
- `src/task/Cm/configs/active/object_v2_grab_arctic_subject_template_20260820.yaml` — 新版 mixed 独立训练入口，未启动训练。
- `src/task/Cm/docs/logs/status_log.md` — 新建任务当前状态入口。
- `src/task/Cm/docs/logs/memory_log.md` — 新建环境、数据路径、旧 cache 风险和 metadata 口径记录。
- `src/task/Cm/docs/logs/architecture_log.md` — 补充统一元信息外壳、链接 cache 架构、stride 语义和当前路径表。
- `src/task/Cm/docs/logs/experiment_log.md` — 记录 EXP-006 cache/metadata implementation gate。
- `src/task/Cm/docs/logs/decision_log.md` — 记录 sequence 级链接树的选择与影响。
- `src/task/Cm/docs/logs/modification_log.md` — 记录本次修改。

**改动原因**

旧 mixed cache 的 GRAB 使用平均 MANO template，不能继续作为新版正式训练数据。用户确认使用软链接复用修复版 GRAB 与既有 ARCTIC、重新生成元数据，并只建立训练入口而暂不启动训练。

**验证**

- 1636 sequences；split train/val/test=`1308/164/164`，全覆盖且互斥。
- 全量 samples：GRAB=`327798`、ARCTIC=`334248`。
- train-only RMS=`0.07328625889337191 m`，scale=`13.645122770626802`，与 YAML 精确一致。
- 4908 个相对软链接、0 个断链；配置可加载，链接 cache 自身约 24M。
- 未启动训练。

## 2026-08-22 — 启动 subject-template 修复版 mixed 正式长训

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部实验与文档

**文件 / 产物**

- `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_20260822_125835/` — 新版 mixed 正式 run 的 config、metadata、metrics、checkpoint 和 W&B 本地目录（运行产物，不纳入版本管理）。
- `output/exp/cm_v121/cm_object_v2_grab_arctic_subject_template_3gpu_bs48_50ep_gpu234_20260822_125824.log` — launcher/stdout 日志（运行产物，不纳入版本管理）。
- `src/task/Cm/docs/logs/status_log.md` — 更新新版训练运行状态、吞吐和风险。
- `src/task/Cm/docs/logs/experiment_log.md` — 新增 EXP-007 正式长训记录。
- `src/task/Cm/docs/logs/modification_log.md` — 记录本次启动与文档更新。

**改动原因**

用户要求在修复版 mixed cache 上复用此前测得的最大吞吐配置启动训练；此前峰值为 3 GPU DDP、per-device batch 48。

**启动检查**

- GPU 2/3/4，world size 3，global batch 144，184650 steps；resolved config 指向新版 cache 和新版 calibration。
- step 100--1000 无 OOM/NaN；10 个记录点吞吐约 `283--328 samples/s`、均值约 `302 samples/s`，低于旧独占条件峰值 `366.7 samples/s`；data wait 约 `0.07%--0.12%`，训练保持运行。

## 2026-08-22 — 停止旧 GRAB gate warm-up 并启动新版 mixed C=64

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部实验与配置

**文件 / 产物**

- `src/task/Cm/configs/active/object_v2_grab_arctic_subject_template_20260820_cm64.yaml` — 新增仅将 `cm_dim` 改为 64 的新版 mixed 对照配置。
- `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_cm64_20260822_190435/` — C=64 正式 run 输出（运行产物，不纳入版本管理）。
- `output/exp/cm_v121/cm_object_v2_grab_arctic_subject_template_cm64_3gpu_bs48_50ep_gpu167_20260822_190414.log` — C=64 launcher 日志（运行产物，不纳入版本管理）。
- `src/task/Cm/docs/logs/status_log.md` — 更新两个新版 mixed run 与旧 warm-up 停止状态。
- `src/task/Cm/docs/logs/experiment_log.md` — 新增 EXP-008，并补齐 EXP-005 停止结论。
- `src/task/Cm/docs/logs/modification_log.md` — 记录本次进程与文档变更。

**改动原因**

用户判断旧 GRAB gate warm-up 已收敛，要求停止并将资源用于新版 mixed C=64 对照。

**执行与验证**

- 对旧 warm-up 两个进程组发送 `SIGTERM`，所有匹配子进程正常退出；保留 step 94430 / epoch 35 的 best/latest checkpoint，停止前最新状态为 step 102524 / epoch 38。
- C=64 resolved config 除名称、`cm_dim` 和标签外与 C=256 完全一致；使用 GPU 1/6/7、3 GPU × batch 48 启动。
- step 100--200 无 OOM/NaN，初始约 `237 samples/s`，训练保持运行。

## 2026-08-23 — 增加 HRDexDB 全量微调数据接口与可训练 DenseToken

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/Cm/dataset/hrdexdb.py` — 新增统一 HRDexDB geometry cache 读取、随机 stride、在线 object-flow 构造，以及 GRAB/ARCTIC/HRDexDB 三 bucket 概率混合和分开验证 loader。
- `src/task/Cm/src/dense_token.py` — 保留历史类名，增加 `freeze=False` 的可训练 DenseToken 路径。
- `src/task/Cm/src/model.py`、`src/task/Cm/src/config.py` — 增加解冻开关；冻结 checkpoint 继续省略 DenseToken，微调 checkpoint 保存可恢复的 DenseToken 参数。
- `src/task/Cm/src/runner.py` — 增加 `finetune_mode: hrdexdb` 数据派发。
- `src/task/Cm/configs/active/hrdexdb_finetune_cm64.yaml` — 新增方案 A 的 C=64 微调配置骨架。
- `src/task/Cm/docs/logs/status_log.md`、`architecture_log.md`、`decision_log.md`、`modification_log.md` — 同步当前入口、数据/张量契约、决策和改动记录。

**改动原因**

用户确认 HRDexDB 需包含 MANO 与所有机器人手型，内部不再额外分层；GRAB/ARCTIC 作为 base 后按三 source bucket 均衡概率微调，Cm 语义和随机 stride 保持不变，并在微调阶段解冻 DenseToken。当前仅完成框架和现有 Inspire-F1 smoke 验证，未启动正式微调。

## 2026-08-23 — 优化 HRDexDB candidate cache 构建并启动全量导出

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部 / 跨 task 共享 cache

**文件**

- `src/task/CmDecoder/build_cache.py` — 用 `scipy.spatial.cKDTree` 替换逐帧全量距离张量，candidate 语义仍为 object pool 到当前手点的 5 cm 半径内存在邻居。
- `src/task/Cm/docs/logs/status_log.md` — 记录后台全量 cache 运行状态。

**改动原因**

单 episode 探针从约 192.6 s 降至约 37.4 s；随后在持久 PTY session 74655 中以 4 workers、单线程 BLAS 启动全量 cache，日志为 `output/research/hrdexdb_cache/build_all_v1.log`。

## 2026-08-23 — 完成 HRDexDB 非视频原始数据下载

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部 / 外部数据

**文件**

- `/home2/wyy/oyx_ws/HRDexDB/v0_nonvideo` — 已落盘完整非视频 HRDexDB 原始模态；未下载视频。
- `src/task/Cm/docs/logs/memory_log.md`、`status_log.md` — 记录原始数据路径和下一步 geometry cache 构建。

**改动原因**

完成用户要求的其余数据下载；后续需将 human/MANO 与三类机器人手型统一转换为 Cm 的 1538/512 geometry schema。

## 2026-08-24 — 停止 mixed 平台期并启动双卡 C=64 geometry-only/no-time 对照

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部实验与配置

**文件 / 产物**

- `src/task/Cm/configs/active/object_v2_grab_gate_cm64_geometry_only_no_time.yaml` — 增加双卡对照的 per-device/validation batch=32，保持 C=32 基础配置的其余实验条件不变。
- `outputs/cm/cm_object_v2_grab_gate_cm64_geometry_only_no_time_20260824_155338/` — 双卡 C=64 新 run（运行产物，不纳入版本管理）。
- `output/exp/cm_v121/cm_object_v2_grab_gate_cm64_geometry_only_no_time_offline_gpu12_bs32.log` — DDP launcher 日志（运行产物，不纳入版本管理）。
- `src/task/Cm/docs/logs/status_log.md`、`experiment_log.md`、`decision_log.md`、`modification_log.md` — 同步停止原因、新对照和当前状态。

**改动原因**

用户要求 C=64 对照改用双卡并适当增大 batch；两条新版 mixed 训练已在验证平台期，继续运行的边际信息有限。

**执行与验证**

- mixed C=256/C=64 进程树已发送 SIGTERM 并退出，最近完整 checkpoint 分别为 epoch 48/40，未删除任何 checkpoint。
- 新 run 使用 GPU 1/2、world size=2，启动日志报告 `per_device_batch=32`、`global_batch=64`、`total_steps=202300`，两个 rank 初始化完成。

## 2026-08-29 — 完成 Cm 实现迁移并移除兼容壳

- branch: `feature/modular-component-runtime`
- post-commit: 待提交
- scope: task 内部 / 破坏性结构迁移
- change_level: L3（破坏性结构迁移）
- approval: 用户已明确确认完全搬迁，并授权纳入当前未提交修改

**文件**

- `src/task/Cm/` — 从顶层迁入唯一真实运行实现、数据、可视化、配置和任务文档。
- `src/task/CmDecoder/`、`src/task/InteractionDynamics/`、`components/ref2dex/cm/`、`process/GRAB/`、`tests/`、`tools/`、`docs/logs/` — 同步迁移后的入口和当前已授权实验改动。
- `src/task/Cm/dataset/{stage4,object_v2,scene,hrdexdb,cache_schema}.py` — 统一数据实现和 schema 入口。
- `src/task/Cm/visualization/{visualize,server,viewer.html}` — 统一可视化实现。
- `src/task/Cm/configs/{active,archive}/` — 迁入新配置并删除顶层软链接。
- `tests/test_cm_structure.py` — 验证规范入口可用且旧入口不存在。

**改动原因**

用户明确要求完全搬迁，不再维护旧壳；当前未提交的科研改动一并作为迁移内容纳入本次提交。这样新增组件和配置只需进入规范目录，不会继续产生重复入口。

**验证**

- 41 个配置文件解析通过，新模块导入通过；完整测试将在提交前运行。
