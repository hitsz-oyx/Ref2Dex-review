# ObjectInteractionCm 活动记录

- scope: task:ObjectInteractionCm
- last_updated: 2026-09-03
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [任务入口](../README.md)、[执行计划](../plan/V1.1.md)、[架构快照](../architecture/V1.1.md)、[指导](../指导/V1.1.md)

## 2026-09-02 23:44:00 +0800 — V1.1 ObjectInteractionCm 架构首版实现与 smoke

- activity_id: ACT-20260902-234400-OBJECTINTERACTIONCM
- timestamp: 2026-09-02 23:44:00 +0800
- run_id: object_interaction_cm_20260902_233920
- modification_version: V1.1.1
- type: architecture / code / data / documentation / operation
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 V1.1 方案并明确要求开始执行、只用 hand flow、1024 点采样、完整 4096 点计算 3 cm mask、左右流严格对齐，且暂不运行实验矩阵
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true
- run_status: COMPLETED
- conclusion: N/A（工程 smoke 仅验证链路与张量合同，不代表科研效果）
- scope: 新建独立 ObjectInteractionCm Task 的 V1.1 架构、双手数据合同、Cm/flow 模型、训练/评估/抽取入口和最小测试；不修改旧 Cm/CmDecoder、src/base、原始数据、cache 或既有 checkpoint

**文件**

- [src/task/ObjectInteractionCm/](../../) — 新 Task 实现目录，包含配置、mmap/NPZ 数据读取、严格左右手校验、4096→1024 物体采样、全池 3 cm 监督掩码、局部 hand-flow interaction、SlotAttention、双 flow decoder，以及 train/eval/extract 入口。
- [src/task/ObjectInteractionCm/docs/README.md](../README.md)、[src/task/ObjectInteractionCm/docs/plan/V1.1.md](../plan/V1.1.md)、[src/task/ObjectInteractionCm/docs/指导/V1.1.md](../指导/V1.1.md)、[src/task/ObjectInteractionCm/docs/architecture/V1.1.md](../architecture/V1.1.md) — V1.1 指导、最终计划、架构冻结快照和文档导航。
- [tests/test_object_interaction_cm.py](../../../../../tests/test_object_interaction_cm.py) — 双手严格对齐、当前帧筛选/采样、模型输出和空 mask 损失测试。
- [outputs/objectinteractioncm/object_interaction_cm_20260902_233920/](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/) — 一步 CPU BaseRunner smoke 运行目录；关键产物为 [run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/summary.json)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/metrics.jsonl) 和 [checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/checkpoints/latest.pt)。

**原因**

将用户确认的 V1.1 研究语义落成可执行的独立 Task：输入仅包含显式 object intrinsic 与左右 hand flow，左右手必须同时存在且帧级对齐；物体候选先按当前帧 5 cm 过滤，在完整 4096 点池上计算 3 cm mask，再采样 1024 点进入 Cm；Cm 固定为 16×32，并同时输出 object flow 与共享 hand flow。实验矩阵按用户要求留待后续，不在本活动中运行。

**验证**

- `python3 -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`。
- `python3 -m pytest -q tests/test_cm_object_v2.py tests/test_cm_slot_attention.py`：`16 passed`，未触及旧 Task 实现。
- `python3 -m py_compile src/task/ObjectInteractionCm/*.py`：通过。
- 真实有效序列 `s1/scissors_offhand_1` 读取与前向 smoke：object `[1,1024,3]`、Cm `[1,16,32]`、hand `[1,3076,3]`；不完整左右手序列按已确认合同拒绝。
- `python3 -m src.task.ObjectInteractionCm.train --set data.root=/tmp/oi_cm_smoke.oTkI5G --set data.train_path=/tmp/oi_cm_smoke.oTkI5G/grab/s1/toy --set data.val_split=0 --set data.test_fraction=0 --set data.batch_size=1 --set data.num_workers=0 --set data.persistent_workers=false --set data.min_stride=1 --set data.max_stride=1 --set train.max_steps=1 --set train.epochs=1 --set train.log_every_steps=1 --set train.eval_every_epochs=null --set train.save_every_epochs=null --set train.save_every_steps=null --set train.device=cpu`：`run_id=object_interaction_cm_20260902_233920`，`run_status=COMPLETED`；manifest/summary 可解析，`kept_frame_ratio=1.0`。该结果仅为工程 wiring 证据，科研结论为 `N/A`。
- 先前 `run_id=object_interaction_cm_20260902_233825` 也完成了一步 smoke，但发现其 `kept_frame_ratio` 分母实现有误；已修正并以 `233920` 重跑。该旧目录 [outputs/objectinteractioncm/object_interaction_cm_20260902_233825/](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233825/) 保留作审计，不作为验证证据。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_cm_sequence_dataset.py`：`4 passed`；系统 Python 的同一测试因缺少既有依赖 `smplx` 无法收集，但未修改其代码或数据。

**回滚**

删除本活动列出的新 Task 文件和测试即可回滚代码/文档；smoke 输出位于 `outputs/`，默认不纳入提交，可单独清理。旧 Task、共享 base、数据和既有运行产物无需回退。

## 2026-09-02 23:58:35 +0800 — V1.1 GRAB/Inspire cache 兼容性诊断

- activity_id: ACT-20260902-235835-OBJECTINTERACTIONCM-CACHE-DIAGNOSTIC
- timestamp: 2026-09-02 23:58:35 +0800
- modification_version: V1.1.2
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问是否可以开始 GRAB/Inspire 训练及是否需要导出 cache；本次只读检查，不写入 cache 或启动训练
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true
- run_status: NOT_STARTED
- conclusion: N/A
- scope: 只读核对 ObjectInteractionCm V1.1 loader 与现有 GRAB/Inspire cache 的 schema、双手流、点数、object pose 和文件完整性；未修改代码、配置、数据或运行状态

**文件与证据**

- [ObjectInteractionCm Task](../../) — 本次诊断所针对的 V1.1 loader、plan 和架构合同；仅作关联入口，未修改实现。
- [定向测试](../../../../../tests/test_object_interaction_cm.py) — 用于复核当前 loader/model 合同的既有测试，未修改。
- [GRAB cache](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/) — 1255 个 `dataset_name=grab` 序列，完整 object pool 为 4096 点、每侧 hand 为 1538 点并带 `obj_pose_world`；split 入口为 [split.json](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/splits_seed42/split.json)。
- [GRAB cache meta](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/meta.json) — 记录 source GRAB、4096 点池、1538 hand 点和 30 Hz effective FPS。
- [Inspire-F1 fast cache](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/) — 576 episode 的 layered v4 cache；每 episode 有 4096 点 object pool、1538 点单一 Inspire hand、`obj_pose_world`、hand flow 和 5 cm candidate mask，但没有 `left/` 与 `right/` 双流目录。
- [Inspire-F1 selection](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/selection_all_object_disjoint_seed42.json) — 现有 object-disjoint 选择清单。
- [两个损坏文件](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s7/headphones_lift/right/hand_points_world.npy)、[两个损坏文件](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s2/mug_drink_2/right/hand_points_world.npy) — NPY header 声明的帧数据大于实际文件大小；严格 loader 会失败。

**原因**

确认“GRAB + Inspire”目前不能直接作为 V1.1 双手 Task 的混合训练输入：GRAB cache 是双手格式但有两个损坏序列；Inspire cache 是单一机器人手格式，虽具备 4096 点池和 pose/flow，却不满足已确认的左右手必须同时存在且严格对齐合同。导出或适配前必须先明确是否允许把 Inspire 单手样本纳入双手合同；不允许用复制、零填充或隐式 fallback 伪造另一只手。

**验证**

- 只读扫描确认 GRAB cache 的 1255 个序列中缺失/不完整的双手序列为 `s7/headphones_lift` 和 `s2/mug_drink_2`；两者均是 right-hand 文件不完整。
- 只读检查 Inspire-F1 fast cache 的 episode geometry/task 字段：`hand_points_world`/`hand_flow` 均为单流 `[T,1538,3]`，object pool 为 `[T,4096,3]`，不存在左右 side 文件。
- 未执行训练、cache 导出、数据迁移或实验矩阵；现有 cache、split、checkpoint 和 outputs 保持不变。

**待用户确认**

1. “GRAB 和 Inspire 手”是要混合训练两种手型，还是先分别训练/评估？
2. 若混合训练，是否要把当前单一 Inspire 手流纳入 V1.1？若是，需要批准将双手合同改为“单手或双手 union”并重新定稿 plan/architecture；若否，则只能先修复 GRAB 的两个损坏序列，Inspire 暂不进入该 Task。

## 2026-09-03 00:40:47 +0800 — V1.1.3 available-hand union cache/index、smoke 与全量训练启动

- activity_id: ACT-20260903-004047-OBJECTINTERACTIONCM-V113
- timestamp: 2026-09-03 00:40:47 +0800
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948（全量训练）；object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601（真实数据 smoke）
- modification_version: V1.1.3
- type: architecture / code / data / documentation / operation
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 GRAB+Inspire 同训、所有真实可用手点 union、修复两条 GRAB 损坏序列、先 smoke 再无人值守全量训练，并允许本次按建议直接执行
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留既有 CmDecoder 日志改动）
- run_status: RUNNING
- conclusion: N/A（smoke 仅为工程链路证据；全量训练尚未形成科研结论）
- scope: 仅修改/新增 ObjectInteractionCm Task、V1.1 计划/架构/指导修订、定向测试和 cache index 工具；不改旧 Cm/CmDecoder、src/base、原始 cache、旧 split、checkpoint 或既有训练进程；全量训练仅使用 GPU4 单卡

**合同与数据产物**

- [V1.1 指导](../指导/V1.1.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 架构](../architecture/V1.1.md) — 冻结 available-hand union、GRAB/Inspire 0.5/0.5 混训、GRAB stride 1..10、Inspire 偶数 2..20、16×32 Cm 和 4096→1024 object sampling 合同。
- [混合 cache index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 1459 train / 184 val / 188 test 序列，其中 GRAB 1004/126/125、Inspire-F1 455/58/63；路径只引用现有 cache。
- [GRAB 修复 cache](../../../../../data/processed_data/object_interaction_cm_grab_repairs_20260903/) — 重新导出 `s7/headphones_lift` 与 `s2/mug_drink_2`，未覆盖原损坏目录。

**原因**

用户已确认将原先严格左右手合同调整为“所有真实可用手点 union”：GRAB 使用真实左右手，Inspire-F1 使用真实单手并以 valid mask 批处理；因此需要在不改原 cache 的前提下建立 source adapter/index，修复两条损坏 GRAB 序列，并用同一 16×32 架构启动混合训练。

**验证**

- `python3 -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`；`python3 -m compileall -q src/task/ObjectInteractionCm`：通过。
- 真实 GRAB/Inspire 前向+反向 smoke：两个 source 均有限值、`hand_points=[1,3076,3]`，GRAB 有效手点 3076、Inspire 有效手点 1538，batch32 一步 GPU smoke 峰值约 3.6 GB，未发生 OOM。
- [真实数据 smoke 运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601/) — `run_id=object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601`，`run_status=COMPLETED`；[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601/summary.json)；等 source object-flow EPE = 7.645 mm，属于工程 smoke 指标，不代表科研效果。
- [全量训练运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/) — `run_id=object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948`，`run_status=RUNNING`，检查时已推进至约 step 3500；[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl) 和 [train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/train.log) 已生成；`summary.json`、`checkpoints/latest.pt` 当前为 `PENDING`。训练命令为 `CUDA_VISIBLE_DEVICES=4 setsid nohup python3 -u -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/grab_inspire_f1_v1_1.yaml`。

**回滚与保护**

- 代码/文档回滚入口为删除本次新增的 `src/task/ObjectInteractionCm/`、`tests/test_object_interaction_cm.py` 及本条活动对应修改；原始 cache 和 repair 源均可独立保留。
- 未停止或修改 GPU0-2 上已有 CmDecoder 训练，未写入 `src/base/`、旧 Task 或仓库外路径；全量训练使用独立输出目录与 GPU4。

## 2026-09-03 04:56:22 +0800 — V1.1.3 GRAB/Inspire 全量训练完成

- activity_id: ACT-20260903-045622-OBJECTINTERACTIONCM-TRAIN-COMPLETED
- timestamp: 2026-09-03 04:56:22 +0800
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- modification_version: V1.1.3
- type: experiment / operation / diagnostic
- change_level: L3
- approval: user-approved
- approval_basis: 用户已批准按 V1.1 最终计划执行 GRAB+Inspire 全量无人值守训练；本条仅记录该既有运行的终态与只读指标核查
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留既有 CmDecoder 日志与 ObjectInteractionCm 未提交实现）
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（训练链路稳定完成，但尚无 held-out test 评估或基线对照，不能据此判定架构科研效果）
- scope: 只读核对全量训练进程、终态 summary、逐 epoch 验证指标、最佳/最终 checkpoint 和 NaN/错误信号；不运行新评估，不改模型、配置、cache、checkpoint 或其他进程

**文件**

- [docs/README.md](../README.md)、[docs/logs/activity_log.md](activity_log.md) — 增加实验记录导航并闭合训练终态。
- [全量训练运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/) — 完整训练产物。
- [config.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)、[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 配置、输入合同与终态摘要。
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/train.log) — 逐步/逐 epoch 指标和日志。
- [checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt)、[checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/latest.pt) — 等 source object-flow EPE 最优 checkpoint 与最终 checkpoint。
- [experiment_log.md](experiment_log.md) — 本次正式训练的假设、结果和结论。

**原因**

全量训练已自然达到 `max_steps=202300`，需要将此前 `RUNNING` 启动事件闭合为终态，并区分工程完成证据与尚未充分验证的科研效果。

**验证**

- 终态 `summary.json`：`run_status=COMPLETED`、`global_step=202300`、`epoch=18`，训练耗时约 `04:16:24`；原训练 PID 已退出，GPU4 已释放。
- `metrics.jsonl` 共 18 次 epoch validation；所有数值字段均为有限值，`train.log` 未发现 traceback、error、NaN 或 Inf。
- `best.pt`：step 193953 / epoch 17，等 source object-flow EPE `4.089469 mm`；`latest.pt`：step 202300 / epoch 18，保存的 best metric 同为 `4.089469 mm`。
- 最终 epoch validation：GRAB object/hand EPE `5.851712 / 2.728593 mm`，Inspire-F1 object/hand EPE `2.343177 / 0.774676 mm`，等 source object EPE `4.097444 mm`。
- 未执行 held-out test 或对照实验；因此工程运行结果有效，科研结论保持 `INCONCLUSIVE`。

**回滚**

本条仅增加 Task-local 文档记录；删除本条 activity 与对应 experiment 条目即可回滚记录，不影响训练产物和 checkpoint。

## 2026-09-03 08:42:06 +0800 — V1.1.3 训练收敛与 batch/GPU 等效性核查

- activity_id: ACT-20260903-084206-OBJECTINTERACTIONCM-CONVERGENCE-CHECK
- timestamp: 2026-09-03 08:42:06 +0800
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问既有训练的收敛性、步数、batch size、GPU 数和与旧 Cm 的等效性；本次只读核对
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（当前训练在既定 cosine schedule 下已进入平台，但未进行 held-out test 或严格同算力对照）
- scope: 只读检查 ObjectInteractionCm 终态配置/曲线与旧混合 Cm、旧 CmDecoder 的实际 train_setup；未启动、停止或修改任何运行、模型、配置、cache 或 checkpoint

**文件**

- [ObjectInteractionCm config.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 当前训练设置、18 个 epoch 验证曲线和终态。
- [旧混合 Cm train.log](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222/train.log) — 旧 Cm 实际 world size 与 global batch 入口。
- [旧 CmDecoder train.log](../../../../../outputs/cmdecoder/cm_decoder_20260901_151052/train.log) — 旧 CmDecoder 实际 world size 与 global batch 入口。
- [本活动记录](activity_log.md) — 本次核查记录。

**原因**

需要区分“每卡 micro-batch 相同”“全局 batch 相同”“optimizer step 数相同”和“总样本暴露量相同”，这些条件不能互相替代。

**验证**

- 当前 ObjectInteractionCm：`world_size=1`、GPU4、`per_device_batch=32`、`global_batch=32`、`total_steps=202300`；配置的 `epochs=50` 是安全上限，实际由 `max_steps` 在 epoch 18 结束。
- 旧混合 Cm：实际 `world_size=3`、每卡 batch `32`、`global_batch=96`、`total_steps=213150`；因此当前 run 仅与其每卡 batch 相同，不与其 global batch 或单步样本量等效。
- 旧 CmDecoder：实际 `world_size=3`、每卡 batch `16`、`global_batch=48`；当前 run 也不与该运行的 global batch 等效。
- 当前验证等权 object EPE：epoch 1 `7.3993 mm`，epoch 17 最佳 `4.0895 mm`，epoch 18 `4.0974 mm`；最后 5 个 epoch 均值 `4.1216 mm`、标准差 `0.0271 mm`，最终仅比最佳高 `0.0080 mm`。epoch 18 learning rate `4.20e-7`，已接近 cosine schedule 终点。
- 当前训练无 NaN/Inf 或 traceback；最佳 checkpoint 为 epoch 17 / step 193953，最终 checkpoint 为 epoch 18 / step 202300。
- 当前步数是旧混合 Cm 的 `95.7%`，但按 optimizer step 不能视作同等样本预算：当前约 `202300×32=6.47M` 样本实例，旧混合 Cm 约 `213150×96=20.46M` 样本实例，且两者数据/模型合同不同。

**回滚**

本条只增加诊断记录；删除本条即可回滚，不影响任何运行产物。

## 2026-09-03 11:06:35 +0800 — V1.1.3 ObjectInteractionCm 全 stride 离线评估

- activity_id: `ACT-20260903-110635-OBJECTINTERACTIONCM-STRIDE-EVAL`
- timestamp: 2026-09-03 11:06:35 +0800
- modification_version: V1.1.3
- type: diagnostic / experiment
- change_level: L0（只读 checkpoint 与现有 val/test cache，生成独立评估产物）
- approval: user-requested
- approval_basis: 用户要求在代码提交后离线核查 Cm 统计是否为各 stride 均值
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `991518adc7fa60600007111a3ecd76d609bb577c`
- worktree_dirty: true（保留根级治理文档与既有 Task-local 日志改动）
- scope: `src/task/ObjectInteractionCm/` 的既有 checkpoint/cache 离线评估；不修改模型、配置或训练进程
- run_id: `objectinteractioncm_stride_eval_20260903_104821`
- run_status: `COMPLETED`
- command: `CUDA_VISIBLE_DEVICES=4 /home2/wyy/miniconda3/envs/graspenv/bin/python -u <inline full-stride evaluator>`（batch=128，num_workers=16）
- conclusion: `INCONCLUSIVE`（评估证据有效；本条回答统计口径，不构成新架构优劣结论）

**范围与口径**

- checkpoint：`outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt`，SHA256=`146cbfc87fa9a17987ab544d53b47cf51efd3a505790ac94e0c031865cd660f4`。
- val/test 全量遍历 40 个组合：GRAB stride `1..10`，Inspire-F1 stride `2,4,...,20`；每个组合固定 stride、active-only、1024 object points。
- 同时记录 object-flow EPE、zero-flow EPE、3 cm 手点 EPE、有效点数和样本数；未修改 cache、checkpoint 或训练进程。

**原因**

- 需要确认既有 `experiment_log.md` 的 EPE 是否代表各 stride 均值，并补充 GRAB/Inspire-F1 全跨度的可复核证据。

**关键发现**

- `experiment_log.md` 中 best epoch 17 的 GRAB `5.823295 mm`、Inspire-F1 `2.355643 mm` 与本次 **val stride=2** 的 `5.8233/2.3556 mm` 完全对应；它们不是各自所有 stride 的均值。
- 因此，现有训练日志/实验表的 EPE 口径是：固定 `eval_stride=2`，并按 source loader（GRAB、Inspire-F1）分别统计；不是 GRAB 1–10 或 Inspire-F1 2–20 的 stride 聚合均值。
- 全 stride 结果显示 object EPE 随跨度增加：test/GRAB `2.9515→27.8845 mm`（stride1→10），test/Inspire-F1 `2.9558→17.9355 mm`（stride2→20）；zero-flow 基线和手 EPE 也随跨度增加。

**产物与验证**

- `src/task/ObjectInteractionCm/docs/logs/` — Task-local 活动、实验和历史记录目录（保留既有未提交记录）。
- [评估结果](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/results.json)
- [run manifest](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/run_manifest.json)
- [summary](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/summary.json)
- 40/40 组合正常完成，所有 EPE 为有限值；独立 GPU4 运行，GPU0–2 上的 CmDecoder V1.2 全量训练未停止。

**验证**

- 结果文件包含 val/test × GRAB 10 个 stride × Inspire-F1 10 个 stride，共 40 条记录；`run_manifest.json` 与 `summary.json` 均为 `COMPLETED`。
- `experiment_log.md` 中的 best epoch 17 source 指标与 val stride=2 结果逐项对应，确认其统计口径不是各 stride 均值。

**回滚**

- 本条仅增加诊断记录；评估目录为独立生成产物，不影响代码、cache、checkpoint 和训练运行。

## 2026-09-03 10:38:55 +0800 — V1.1.3 运行 JSON 职责与重复字段诊断

- activity_id: ACT-20260903-103855-OBJECTINTERACTIONCM-JSON-AUDIT
- timestamp: 2026-09-03 10:38:55 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问既有训练运行目录中 JSON 是否存在职责重叠；本次只读比较文件结构、字段和值，不修改运行产物、配置或 checkpoint
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留用户既有 CmDecoder 与 ObjectInteractionCm 未提交改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（本条只诊断追溯 JSON 结构，不新增科研效果证据）
- scope: `config.json`、`metadata.json`、`run_manifest.json`、`summary.json` 的职责边界、精确重复字段和嵌套重复；未运行新实验

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/) — 本次只读检查范围。
- [config.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)、[metadata.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metadata.json)、[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 被比较的四个 JSON。

**原因**

当前运行目录同时保存配置、数据合同、运行追溯和终态指标；需要核对重复是否超出各自职责，避免后续收缩 JSON 时破坏 checkpoint/inference 兼容性。

**验证**

- `metadata.json` 的 22 个字段及其值被 `run_manifest.json.dataset_metadata` 完整复制；`run_manifest.json.contract` 另重复其中 6 个核心合同字段。
- `run_manifest.json` 与 `summary.json` 精确重复 `task`、`run_name`、`mode`、`output_dir`、`modification_version`；`summary.json` 另以 `run_id==run_name` 再记录一次运行身份。
- `config.json` 未被完整展开进 manifest，仅重复 `component_registry`、`components`、`modification_version`、`operation_category`；`summary.json` 仅重复 `modification_version` 并引用关键产物。
- 当前四个 JSON 均可解析；未修改任何产物。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响运行目录。
- 建议后续新 schema 让 manifest 引用 metadata 快照而不再内嵌完整 `dataset_metadata`；在审计 checkpoint/inference 消费者前，不删除运行目录 `metadata.json`。

## 2026-09-03 10:47:12 +0800 — V1.1.3 summary.json 消费者审计

- activity_id: ACT-20260903-104712-OBJECTINTERACTIONCM-SUMMARY-CONSUMERS
- timestamp: 2026-09-03 10:47:12 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问 summary.json 的实际读取者；本次只读搜索生成器、消费者和测试引用，不修改运行产物或代码
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留用户既有 CmDecoder 与 ObjectInteractionCm 未提交改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（本条只审计 JSON 消费关系，不新增科研效果证据）
- scope: 仓库内 `summary.json` 的生成调用、实际读取调用、测试和文档引用；重点核对 BaseRunner 运行目录

**文件**

- [BaseRunner 终态摘要写入](../../../../../src/base/base_runner.py) — train/eval 终态调用 `write_run_summary`。
- [summary.json 写入实现](../../../../../src/base/run_manifest.py) — 定义 `ref2dex.run_summary.v1` 结构。
- [运行目录 summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 本次被审计的实际文件。
- [运行追溯与目录规范](../../../../../docs/目录规范.md) — 将 summary 定义为用户可读终态入口。

**原因**

需要确认 summary 是 BaseRunner 的机器依赖，还是仅作为用户/审计入口，从而判断它是否可以与其他运行 JSON 合并。

**验证**

- 对本次 ObjectInteractionCm 的 BaseRunner 运行，仓库内没有发现任何代码读取该 `summary.json`；BaseRunner 只负责写入，恢复训练读取 checkpoint/history，不读取 summary。
- `tests/test_run_manifest.py` 只读取临时目录中的 summary 以验证写入合同，不消费实际训练产物。
- 发现的实际 summary 读取者仅是旧 `PointWorldWAM` 自定义训练/诊断脚本，用于其自身 resume 或分析，不适用于 ObjectInteractionCm 的 BaseRunner 输出。
- `summary.json` 的当前定位因此是用户查看、activity/experiment 导航和外部审计入口，而不是训练循环的必需状态文件。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响运行目录。
- 若未来确认没有外部归档工具依赖，可考虑将“终态摘要”改为可选产物，但不建议删除；它应继续保留为稳定的人类/审计接口。

## 2026-09-03 10:47:12 +0800 — V1.1.3 config/meta、dataset_split 与 checkpoint 消费关系核对

- activity_id: ACT-20260903-104712-OBJECTINTERACTIONCM-CONTRACT-AUDIT
- timestamp: 2026-09-03 10:47:12 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问 dataset_split 是否来自 config/meta，以及 config、metadata 与 checkpoint 的区别；本次只读核对配置、数据 index 和共享 checkpoint 代码
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留用户既有 CmDecoder 与 ObjectInteractionCm 未提交改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（本条只核对追溯合同，不新增科研效果证据）
- scope: ObjectInteractionCm `config.json`/`meta`、运行时 `metadata.json`、输入 index `counts`、checkpoint 内部 config/metadata 及 BaseRunner 消费路径

**文件**

- [ObjectInteractionCm 配置定义](../../../../../src/task/ObjectInteractionCm/config.py)、[active 配置](../../../../../src/task/ObjectInteractionCm/configs/active/grab_inspire_f1_v1_1.yaml) — 静态配置意图和模型/数据参数。
- [ObjectInteractionCm dataloader](../../../../../src/task/ObjectInteractionCm/dataset.py) — 根据实际 index entries 生成 `dataset_split`/`source_split`。
- [输入 index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 保存 sequences 和 source-level counts。
- [checkpoint 管理](../../../../../src/base/checkpoint.py)、[BaseRunner](../../../../../src/base/base_runner.py) — 保存并消费 checkpoint 内的 config/metadata。

**原因**

需要区分静态配置、实际数据事实和 checkpoint 自包含信息，才能决定 manifest 中哪些字段可安全删除而不丢失追溯能力。

**验证**

- 本次运行的 `config.json`/`meta` 不包含解析后的 `dataset_split`；只包含 `data.index_path`、split 策略参数和静态形状/阈值。
- `dataset_split`/`source_split` 在 dataloader 读取 index 后按实际 entries 计算；输入 index 的 `counts` 保存同一 source-level 事实。
- `CheckpointManager.save` 将运行时 `metadata` 和完整 `config` 一起写入 `.pt`；`build_runner_from_checkpoint` 用 checkpoint 的 `config` 重建配置，`setup_inference` 用 checkpoint 的 `metadata` 配置数据。
- 输出目录的 `metadata.json` 是 checkpoint metadata 的外部快照；当前代码没有从该输出 JSON 反向加载 checkpoint。删除外部快照不会等价于删除 checkpoint 内部 metadata，但会损失独立查看入口。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响运行目录。
- 若用户确认精简 manifest，优先移除完整 `dataset_metadata` 展开；`dataset_split` 是否保留为最小 contract 字段，再根据是否需要 manifest 单文件快速审计决定。
