# ObjectInteractionCm 活动记录

- scope: task:ObjectInteractionCm
- last_updated: 2026-09-04
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [任务入口](../README.md)、[执行计划](../plan/V1.1.md)、[架构快照](../architecture/V1.1.md)、[指导](../指导/V1.1.md)

## 2026-09-04 15:27:04 +0800 — HRDexDB MANO 配对与可适配数据集诊断

- activity_id: `ACT-20260904-152704-OICM-HR-MANO-DATASET-SEARCH`
- timestamp: 2026-09-04 15:27:04 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读数据与公开资料核查；未改变代码、配置、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求搜索可适配数据集，核对 HRDexDB 是否含 MANO 配对数据及其运动幅度
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: HRDexDB `episodes/pairings/objects` 元数据、human MANO OBJ/JSON、compact object pose、当前 Inspire-F1 选集连接、同一 object-pose/wrist-frame 的运动抽样，以及公开数据集的标注与许可证入口；未修改代码、配置、数据、cache、checkpoint 或运行目录
- conclusion: SUPPORTED（HRDexDB 完整发布目录含 human MANO 与经验证 robot→human pairing；当前 ObjectInteractionCm index 只喂 Inspire-F1，未喂 human MANO；human MANO 增大手部运动但没有改变 HR 近静止的物体运动分布。候选数据集的最终收益仍需受控实验，故因果结论为 INCONCLUSIVE）

**文件与证据**

- [HRDexDB README](../../../../../dataset/HRDexDB/v0_nonvideo/README.md) — 官方元数据说明：`train` 是完整 catalog 而非 benchmark split，`pairings` 只收录显式验证的 robot-to-human 链接。
- [episodes.parquet](../../../../../dataset/HRDexDB/v0_nonvideo/metadata/episodes.parquet)、[pairings.parquet](../../../../../dataset/HRDexDB/v0_nonvideo/metadata/pairings.parquet) — 本地目录统计：2104 条 episode、441 条 human episode、1335 条已验证 pairing；Inspire-F1 pairing 392 条。
- [当前 ObjectInteractionCm index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json)、[Inspire-F1 选集](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/selection_all_object_disjoint_seed42.json) — 当前训练实际使用 576 条 Inspire-F1 episode；与 pairing 连接后有 378 条 robot episode、377 条 unique human episode，但 human 记录尚未进入当前 index。
- [HRDexDB 全量 geometry manifest](../../../../../data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json) — 仓库已有四手型 `cmdecoder_layered_v4` cache：2088 条有效 episode（含 human 441 条），但该旧 manifest 仍是独立 CmDecoder cache，不能替代当前 ObjectInteractionCm 的 `object_pose_t` index。
- [HR cache builder](../../../../../src/task/CmDecoder/build_cache.py) — human 分支读取 `hand/mano/*.obj` 与 `mano_params/*.json`，以 1538 个 MANO face-center 保持 hand 点合同，并把 `q_semantics` 标记为 `unavailable_mano`；这说明 human MANO 可转成现有 surface/cache 表示，但不是可直接解释的机器人关节 q。
- [model.py](../../model.py) — 工作区已有用户改动，本次诊断未修改。

**原因**

- 需要区分“HRDexDB 是否拥有可用的 paired MANO 证据”和“当前训练是否真的看到了 human source”；同时用与现有 object-pose/wrist-frame 一致的时间跨度比较 hand/object flow，避免把配对关系误当作逐帧同步或把近静止物体目标误判成高动态监督。

**验证**

- 使用 `fastwam` 环境的 `pyarrow` 读取 parquet：所有 HR 元数据 `split=train`；pairings 的 `source` 均为 `grasp_result.json:human_paired_episode` 且 `verified=True`。当前选中的 377 个 unique human episode 中，MANO OBJ 数、MANO 参数数、compact object-pose 帧数与 episode `num_frames` 均一致。
- 对全部 440 个可连接 human episode 及当前选集对应的 377 个 episode 做只读抽样：解析 MANO face-center、object 4×4 pose、wrist（`joints[0]`/`global_orient[0]`），在当前 pipeline 的 object-pose/wrist frame 中按 30 Hz 计算 stride=2（约 66.7 ms）和 stride=10（约 333 ms）flow。数值是随机表面点诊断抽样，不是官方 benchmark。
- 当前选集对应 human 的 stride=2：hand flow 中位数 `5.44 mm`、p95 `15.33 mm`；object flow 中位数 `1.07 mm`、p95 `10.44 mm`、`>20 mm` 约 `0.06%`。stride=10：hand 中位数 `26.44 mm`，object 中位数 `3.34 mm`、p90 `40.95 mm`。
- 同口径既有诊断中，Inspire-F1 stride=2 的 hand/object 中位数约 `1.85/1.01 mm`，GRAB 约 `13.46/9.11 mm`；因此 human MANO 主要补手部 articulation/contact，object motion 仍接近 HR robot，远小于 GRAB。
- 公开资料核查了 GigaHands（完整物体运动与 MANO-derived hand）、HOT3D（刚性物体 6DoF + MANO）、OakInk/OakInk2（MANO + object SE(3)）、DexYCB（MANO + object 6D）、ARCTIC（高动态双手但含 articulated object）、HOGraspNet（接触/抓取标注）等候选；没有据此启动训练或改变研究变量。

**边界与回滚**

- 配对是“同对象/同任务语义”的 human↔robot 关联，不应默认逐帧同步；官方采集协议允许机器人操作者观察人类动作后按自身形态重现。因此 human MANO 更适合作为独立 source 或 contact/hand 预训练信号，不能直接当作 robot flow 的逐帧 GT。
- 仅新增本活动条目；删除本条即可回滚文档差异。代码、配置、数据、cache、checkpoint 和既有运行均未改动。

## 2026-09-04 12:25:51 +0800 — GRAB 与 Inspire-F1（HRDexDB 子集）分布诊断

- activity_id: `ACT-20260904-122551-OICM-DATA-DISTRIBUTION`
- timestamp: 2026-09-04 12:25:51 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读分布核查；未改变代码、配置、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求探查 GRAB 与 HRDexDB 的分布差异及其对效果的可能影响
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 读取当前 ObjectInteractionCm index、两套 cache/geometry manifest、scale manifest、HRDexDB episodes metadata 和既有验证结果；固定 stride=2 的接触/运动抽样，以及训练 stride policy 的对照。工作区已有的 [model.py](../../model.py) 未在本次诊断中修改。
- conclusion: SUPPORTED（存在足以解释混合训练效果受限的显著域差，主差异在运动/接触/embodiment 与 split；“域差是唯一原因”仍为 INCONCLUSIVE）

**文件与证据**

- [index.json](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 当前实际混合合同：GRAB 1004/126/125 条序列，Inspire-F1 455/58/63 条序列；source probability 0.5/0.5，GRAB stride 1..10、Inspire-F1 stride 2..20，split 规则不同。
- [GRAB cache meta.json](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/meta.json)、[HRDexDB selection](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/selection_all_object_disjoint_seed42.json)、[HRDexDB metadata README](../../../../../dataset/HRDexDB/v0_nonvideo/README.md) — 当前 HRDexDB 不是全库 2104 条，而是 Inspire-F1 的 576 条 object-disjoint 子集。
- [scales_train.json](../../../../../data/processed_data/object_interaction_cm_v1_2_1/scales_train.json) — 全局 flow scale 与 source/stride group RMS。
- [当前 metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl) — epoch 52 source 指标及 interaction coverage；[旧 stride 诊断](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/results.json) — 同 cache 合同下的 zero-flow 对照。

**原因**

- 需要把“物体侧表示更细”与“两个 source 是否提供同一种可学习信号”分开；只看混合 EPE 会把近静止 HR 目标、双手/单手输入和不同 split 难度混在一起。

**验证**

- 只读统计得到：固定 stride=2 的 object flow median 约 GRAB `9.11 mm`、Inspire-F1 `1.01 mm`；`>20 mm` 比例约 `33.5%` 对 `0.56%`。hand flow median 约 `13.46 mm` 对 `1.85 mm`。
- 同一 active transition 抽样中，object 点 `<5 cm` 接触覆盖约 `73.1%` 对 `40.5%`，KNN 有效邻居约 `5.8` 对 `3.1`；当前全量 val 也显示 sampled active points `747.9` 对 `429.0`。
- object bbox diagonal median 约 `172` 对 `217 mm`，geometry scale 差异中等；但 stride=2 group RMS flow 为 object `32.30` 对 `5.51 mm`、hand `35.87` 对 `4.63 mm`（约 5.9–7.8 倍）。
- 当前全量 val source object EPE 为 GRAB `5.073 mm`、Inspire-F1 `2.225 mm`；旧同 cache stride=2 zero-flow 对照中 Inspire-F1 `2.338 mm`、模型 `2.356 mm`，说明 HR 的绝对指标本身已接近“保持不动”下限。
- 本次未启动新训练、未运行测试集、未修改研究变量；统计结果用于诊断，不等同于因果实验。

**回滚与边界**

- 仅新增本活动条目；删除本条即可回滚文档差异。代码、配置、数据、cache、checkpoint 和既有运行均未改动。

## 2026-09-04 11:27:03 +0800 — V1.2.1 前缀手点 KNN 邻域搜索优化

- activity_id: `ACT-20260904-112703-OICM-KNN-PREFIX-OPT`
- timestamp: 2026-09-04 11:27:03 +0800
- modification_version: V1.2.1.4
- type: code / diagnostic
- change_level: L1（局部实现优化；不改参数、坐标系、GT、radius=5cm、K=8 或 checkpoint schema）
- approval: user-requested
- approval_basis: 用户明确要求先修改邻域搜索并在 GPU0/1/2 做吞吐短训
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: `LocalHandInteraction` 在前缀有效 mask（CmDecoder 的 1538/3076 contract）上改用 PyTorch3D `knn_points(K=8)`，再执行 5cm radius mask；非连续 mask 和全无效情况回退原 `cdist+topk`；未实现持久化邻域 cache
- conclusion: SUPPORTED（数值 parity 与独立 kernel benchmark 通过；端到端短训吞吐已测，科研效果尚未评估）

**原因**

- 之前的 `cdist` 在 mask 之前对 3,076 个手点全部计算；CmDecoder 实际只有前 1,538 个有效点。先 compact 前缀再做 KNN 可减少显存和距离计算，同时保留 5cm/K=8 语义。

**验证**

- 随机 `[B=2,N_obj=32,N_hand=3076]` parity：新旧 `edge_indices` 和 `edge_valid_mask` 完全一致，交互输出最大绝对差 `1.49e-8`，距离最大差 `5.03e-8`。
- GPU1 邻域 kernel（`B=16,N_obj=1024,N_hand=3076,K=8`）基准：旧 `cdist+topk=3.94 ms`，新 `knn_points=0.576 ms`，约 `6.84x` 加速；该结果只代表邻域子图，不等于完整训练加速。
- 验证命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/model.py`；另以同一随机权重分别执行旧 `cdist+topk` 参考路径和新路径，比较上述输出/诊断张量。
- 修改文件：[model.py](../../model.py)。回滚入口为恢复该文件本次 diff；没有改动 checkpoint、数据或公共 `src/base`。

**运行关联**

- 端到端短训见 CmDecoder 活动 `ACT-20260904-112703-CMDECODER-KNN-012-THROUGHPUT`；该运行使用本 Task 的 frozen best Cm，不改变 Cm 权重。

## 2026-09-04 08:45:46 +0800 — V1.2.1 三卡训练完成与收敛性核查

- activity_id: `ACT-20260904-084546-OBJECTINTERACTIONCM-V121-CONVERGENCE-FINAL`
- timestamp: 2026-09-04 08:45:46 +0800
- modification_version: V1.2.1.3
- type: diagnostic / operation
- change_level: L0（只读核对终态日志、metrics、checkpoint 与验证曲线）
- approval: user-requested
- approval_basis: 用户询问 Cm 训练是否收敛；本次不启动、停止或修改训练与评估口径
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: ObjectInteractionCm V1.2.1 三卡续训终态；不修改代码、配置、数据、cache 或 checkpoint
- run_id: `object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806`（resume）
- run_status: COMPLETED
- last_step: `202300`
- last_epoch: `52`
- best_metric: `3.649306 mm`（`val/obj/flow_epe_mm`，epoch 52）
- best_checkpoint: [best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt)（step `202300` / epoch `52`）
- latest_checkpoint: [latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt)（step `202300` / epoch `52`）
- practical_status: 基本收敛（object 指标已进入平台，最后 5 个 epoch 仅改善约 `0.066%`）
- conclusion: INCONCLUSIVE（未做 held-out test，且单次训练不能证明统计意义上的最终收敛）

**原因**

- 用户要求确认 ObjectInteractionCm 是否收敛；本次只读检查完整训练终态和最后若干 epoch 的验证曲线，区分“训练完成”“实用平台”和“科研意义上的严格收敛”。

**验证**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 日志末尾为 `Training finished at step 202300 in 08:11:05.`，epoch 52 validation 已写入。
- 总体 `val/obj/flow_epe_mm`：epoch 44=`3.669525 mm`、epoch 48=`3.651703 mm`、epoch 52=`3.649306 mm`；最后 5 个 epoch 均值=`3.650447 mm`、标准差=`0.000969 mm`，首尾改善约 `0.066%`。
- 总体 `val/loss` 最后 5 个 epoch 均值=`0.00307428`、标准差=`1.41e-6`；epoch 52 学习率=`3.30e-9`，优化步长已接近零。
- source 分支：GRAB object EPE 最终=`5.073422 mm`（全程 best）；Inspire-F1 object EPE 最优为 epoch 15 的 `2.216799 mm`，末期稳定在约 `2.22–2.25 mm`，说明后者早已进入平台。
- hand 3 cm EPE 最终=`2.404122 mm`，epoch 51 为 `2.404022 mm`，末期无实质变化；`train.log` 未发现 traceback、CUDA OOM、NCCL、NaN 或 Inf，训练进程已退出。
- 本次只读检查未停止或修改其他 GPU 任务；`git diff --check` 与 `audit_diff.py --check-links` 通过。

## 2026-09-04 00:06:29 +0800 — V1.2.1 各 source/stride 验证集快速评估

- activity_id: ACT-20260904-000629-OBJECTINTERACTIONCM-V121-STRIDE-EVAL
- timestamp: 2026-09-04 00:06:29 +0800
- modification_version: V1.2.1.3
- type: operation / diagnostic
- change_level: L0（用户请求的只读评估；仅生成独立运行产物）
- approval: user-requested
- approval_basis: 用户要求先用其他 GPU 评估各 stride；不停止或修改 GPU0/1/2 上的训练，不抢占 GPU4–7 上的既有任务
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留既有 ObjectInteractionCm 实现、训练和治理改动）
- scope: 使用训练运行的 `latest.pt`（step=129302、epoch=32）在验证集分别评估 GRAB stride `1..10` 与 Inspire-F1 stride `2..20`；每个 source/stride 均独立 forward，每组均匀抽取 512 个样本；不改代码、配置、cache、训练进程或其他 GPU 任务
- run_id: objectinteractioncm_stride_eval_subset_20260904_000408
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（这是每组 512 样本的快速诊断，不替代全验证集和最终 parity 结论）

**文件与证据**

- [运行 manifest](../../../../../outputs/research/objectinteractioncm_stride_eval_subset_20260904_000408/run_manifest.json)、[stride 结果](../../../../../outputs/research/objectinteractioncm_stride_eval_subset_20260904_000408/results.json) — 设备、checkpoint SHA256、采样合同、source/stride 的 object/hand EPE 和有效样本率。
- [当前训练运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt) — 评估输入及仍在运行的三卡训练。

**原因**

剩余 GPU 中 4–7 卡有高负载既有任务，3 卡有足球任务但显存余量足够，因此只在物理 GPU3 以小 batch 共存运行。先用每组 512 个均匀覆盖样本快速判断 stride 趋势；模型 interaction 分支使用 `hand_flow`，故每个 stride 必须独立 forward，不能复用其他 stride 的预测。

**验证**

- GPU3 评估期间总显存约 `11.4/24 GiB`，未发生 OOM；GPU0/1/2 训练进程和吞吐未被停止或改动。
- GRAB object EPE 从 stride1 的 `2.788 mm` 增至 stride10 的 `24.124 mm`；Inspire-F1 从 stride2 的 `2.314 mm` 增至 stride20 的 `10.935 mm`。对应 hand 3 cm EPE 分别为 `1.710→14.763 mm` 与 `0.730→3.205 mm`。
- stride2 的 source-mean object EPE 为 `3.799 mm`，与同 checkpoint 既有完整验证日志的 `3.776 mm` 接近，说明评估口径和 target 生成一致；快速子集结果仍不能视为最终全量指标。
- `results.json`、`run_manifest.json` 已写入 `run_status=COMPLETED`；未运行测试集、未修改研究变量。

**回滚**

删除本条活动记录即可回滚文档差异；评估输出为独立、可保留的只读产物，训练和其他 GPU 任务未受影响。

## 2026-09-03 23:54:32 +0800 — V1.2.1 首次 stride 复用评估作废

- activity_id: ACT-20260903-235432-OBJECTINTERACTIONCM-V121-STRIDE-EVAL-INVALID
- timestamp: 2026-09-03 23:54:32 +0800
- modification_version: V1.2.1.3
- type: operation / diagnostic
- change_level: L0（失败的只读评估尝试）
- approval: user-requested
- approval_basis: 用户要求先评估各 stride；该尝试未改变训练、代码或配置
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true
- scope: 评估 `latest.pt` 的 all-stride 运行；在 GRAB 完成后发现错误地假设不同 stride 可复用 forward，随后在 Inspire-F1 阶段中断
- run_id: objectinteractioncm_all_stride_eval_20260903_235432
- run_status: FAILED
- conclusion: INVALID_IMPLEMENTATION（模型显式读取 `hand_flow`，不同 stride 的输入不同；该次产生的 GRAB 数字不得引用）

**文件与证据**

- [失败运行 manifest](../../../../../outputs/research/objectinteractioncm_all_stride_eval_20260903_235432/run_manifest.json)、[残留结果](../../../../../outputs/research/objectinteractioncm_all_stride_eval_20260903_235432/results.json) — 已标记失败，仅作为审计证据。

**原因与回滚**

错误假设在继续运行前被发现并纠正；未修改训练进程。保留失败产物以避免把无效结果误当有效结果，后续有效评估见本活动上一条记录。

## 2026-09-03 21:59:31 +0800 — V1.2.1 三卡续训 epoch 进度核查

- activity_id: ACT-20260903-215931-OBJECTINTERACTIONCM-V121-EPOCH-STATUS
- timestamp: 2026-09-03 21:59:31 +0800
- modification_version: V1.2.1.3
- type: diagnostic / operation
- change_level: L0（只读核查既有训练，仅追加活动证据）
- approval: user-requested
- approval_basis: 用户询问当前训练状态及是否已经产生完整 epoch；本次不启动、停止或调整训练
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留 ObjectInteractionCm 既有未提交实现、配置与文档）
- scope: [ObjectInteractionCm Task](../../) 的既有 V1.2.1 三卡续训进程、日志、指标和 checkpoint；不修改代码、配置、cache、模型状态或 GPU 进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（resume）
- run_status: RUNNING
- conclusion: INCONCLUSIVE（训练与逐 epoch 验证链路正常，但正式训练尚未完成，不能形成架构效果结论）

**文件与证据**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[resume manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/run_manifest_resume_20260903_190509.json) — 既有三卡恢复运行及其追溯入口。
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 当前 step、epoch、训练吞吐、ETA 与逐 epoch 验证指标。
- [checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt)、[checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt) — 最近完整 epoch 与当前最佳 object-flow checkpoint。

**原因**

区分日志中的“正在训练的 epoch”和已经完成训练、验证及 checkpoint 保存的 epoch，并确认从 GPU4 迁移到 GPU0/1/2 后的续训仍在正常推进。

**验证**

- `ps -p 768731 -o pid,stat,etime,cmd`：torchrun 父进程存活；rank `768852/768853/768854` 分别占用 GPU0/1/2 约 `9.4 GiB`。
- 21:59:31 日志已推进至 step `85100 / 202300`，当前为 `epoch=21`；吞吐约 `774.7 samples/s`，日志 ETA 约 `4.03 h`。
- `metrics.jsonl` 已包含 20 条完整 `train_epoch` 和 20 条 validation：最近完成 `epoch=20` / step `83666`，`val/loss=0.00379970`、object-flow EPE `4.02003 mm`、3 cm hand-flow EPE `3.48386 mm`。
- `best.pt` 为 `epoch=15` / step `64651`，按既定 object-flow 指标取得当前最佳 `3.84406 mm`；`latest.pt` 为 `epoch=20` / step `83666`。
- `train.log` 未检出 traceback、CUDA OOM、NCCL error 或 NaN；本次只读查询未改变训练状态。

**回滚**

删除本条活动记录即可回滚文档差异；训练进程与运行产物未被修改。

## 2026-09-03 22:05:52 +0800 — V1.2.1 三卡续训收敛性核查

- activity_id: ACT-20260903-220552-OBJECTINTERACTIONCM-V121-CONVERGENCE-STATUS
- timestamp: 2026-09-03 22:05:52 +0800
- modification_version: V1.2.1.3
- type: diagnostic / operation
- change_level: L0（只读分析既有训练曲线，仅追加活动证据）
- approval: user-requested
- approval_basis: 用户询问当前训练是否收敛；本次不改变训练、配置、checkpoint 或评估口径
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留 ObjectInteractionCm 既有未提交实现、配置与文档）
- scope: [ObjectInteractionCm Task](../../) 的 V1.2.1 三卡续训曲线和运行状态；不修改代码、数据、cache、模型状态或 GPU 进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（resume）
- run_status: RUNNING
- conclusion: INCONCLUSIVE（已进入较稳定的验证区间，但训练仍在进行，不能宣称最终收敛）

**文件与证据**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 逐步、逐 epoch 训练/验证曲线及运行日志。
- [checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt)、[checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt) — 最近完整 epoch 和当前最佳 checkpoint。

**原因**

区分“训练损失还在下降”“验证指标进入平台”“最终收敛已被证明”三个层次，避免因单个 epoch 的改善过早停止三卡训练。

**验证**

- 22:05:52 时 epoch `21` 已完成训练和验证，当前已进入 epoch `22`（step `87600`）；torchrun 及 GPU0/1/2 仍正常运行。
- epoch 15–21 的 `val/obj/flow_epe_mm` 为 `3.844–4.052 mm`，整体围绕约 `3.93 mm` 窄幅波动；epoch 21 为 `3.865 mm`，接近当前最佳 epoch 15 的 `3.844 mm`，说明 object 分支已接近平台，但尚未严格单调收敛。
- epoch 15–21 的 `val/loss` 从 `0.0041135` 降至 `0.0036606`，epoch 21 为目前最低；`val/hand/flow_epe_3cm_mm` 从 `4.172 mm` 降至 `3.289 mm`，hand 分支仍有改善。
- 最近 5 个 validation 的均值/标准差：loss `0.003855 / 0.000114`，object EPE `3.944 / 0.078 mm`，hand EPE `3.611 / 0.197 mm`；因此判断为“已进入稳定下降/平台区”，不是“完全收敛”。
- `train.log` 未检出 traceback、CUDA OOM、NCCL error 或 NaN；未执行早停、重启或任何科研变量修改。

**回滚**

删除本条活动记录即可回滚文档差异；训练进程与运行产物未被修改。

## 2026-09-03 22:16:22 +0800 — V1.2.1 与旧 hand-root Cm 的收敛速度对齐诊断

- activity_id: ACT-20260903-221622-OBJECTINTERACTIONCM-V121-CONVERGENCE-COMPARE-CM
- timestamp: 2026-09-03 22:16:22 +0800
- modification_version: V1.2.1.3
- type: diagnostic / experiment
- change_level: L0（只读比较既有 run；不启动新评估、不修改运行）
- approval: user-requested
- approval_basis: 用户要求判断当前 ObjectInteractionCm 是否相较旧 `hand_root_t` Cm 收敛异常偏快
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留 ObjectInteractionCm 与 Cm 既有未提交改动）
- scope: [ObjectInteractionCm Task](../../) 与历史 [Cm Task](../../../Cm/) 的训练日志、配置和指标；只读，不改变任何代码、配置、cache、checkpoint 或进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（当前 resume）；cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324（旧 hand-root）；cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401（旧 object-pose 近似对照）
- run_status: RUNNING（当前 OI；旧 hand-root 已 STOPPED）
- conclusion: INCONCLUSIVE（墙钟训练确实更快，且在同坐标/短 stride 近似对照下有一定优化优势，但 hand-root 数值不能直接比较）

**文件与证据**

- [当前 OI config](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/config.json)、[当前 OI metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[当前 OI train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — `object_pose_t`、eval stride 2、当前 epoch/step 和吞吐。
- [旧 hand-root Cm config](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324/config.json)、[旧 hand-root metrics](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324/metrics.jsonl)、[旧 Cm 实验记录](../../../Cm/docs/logs/experiment_log.md) — `hand_root_t`、多 stride mean-EPE 及资源竞争记录。
- [旧 object-pose Cm config](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/config.json)、[旧 object-pose metrics](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/metrics.jsonl) — 与当前同为 object-pose、stride 2 的近似架构对照。

**原因**

用户观察到当前 OI 在约 21 个 epoch 时已达到低 EPE，需要拆分墙钟吞吐、坐标/stride 指标口径、有效监督难度、架构差异和 optimizer batch 动力学，判断是否是真实的优化加速。

**验证**

- 墙钟：当前 OI epoch 1→21 validation 间隔约 `3.19 h`；旧 hand-root Cm epoch 1→21 约 `14.53 h`。旧 run 曾与 CmDecoder 争用 GPU1，记录吞吐约 `146–148 samples/s`；当前三卡恢复后约 `773 samples/s`，因此墙钟约 `4.5×` 更快主要是资源状态改变。
- 优化预算：当前 OI epoch 21 为 step `87469`，旧 hand-root epoch 21 为 step `90090`；当前累计样本实例约 `7.67M`，旧 run 约 `8.65M`，当前不是因为看过更多样本才更低。
- 指标口径：旧 hand-root 的 `val/mean_stride_epe_mm` 同时平均 stride 1/5/10；epoch 21 三个 stride 的 source-mean object EPE 约为 `2.274/8.014/14.989 mm`，其 mean=`8.426 mm`。当前 OI 只评估 stride 2，epoch 21 object EPE=`3.865 mm`；二者不能直接比较。
- 坐标/幅度：旧 hand-root config 的 object-flow target RMS=`0.09319 m`，当前 object-pose train scale `s_obj_flow=0.05789 m`，后者约低 `38%`；object-pose frame 移除了 hand-root 运动带来的大幅相对位移，任务本身更容易。
- 近似公平对照：旧 Cm 的同为 `object_pose_t`、stride 2 run 在 epoch 1/21 为 `4.673/4.367 mm`；当前 OI 为 `4.579/3.865 mm`。在更接近的口径下，当前确有一定优化优势，但旧 run 使用 C=64、不同 decoder/scale，不能归因于单一结构因素。
- 额外混杂：当前 OI 首个有效 epoch 先以 global batch=`32` 单卡运行 `11409` steps，随后才切换 global batch=`96` 三卡；旧 hand-root 从头就是 global batch=`96`。因此 epoch 编号也不代表完全相同的 optimizer dynamics。
- 未检出当前 OI 的 traceback、OOM、NCCL error 或 NaN；本次没有停止、重启或改动任何训练变量。

**回滚**

删除本条活动记录即可回滚文档差异；所有训练进程与既有产物保持不变。

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

## 2026-09-03 18:28:06 +0800 — V1.2.1 架构实现、scale/miss 诊断、smoke 完成并启动全量训练

- activity_id: ACT-20260903-182806-OBJECTINTERACTIONCM-V121
- timestamp: 2026-09-03 18:28:06 +0800
- modification_version: V1.2.1.1
- type: architecture / code / data / documentation / operation
- change_level: L2（Task 内模型、loss、配置和训练流程；不修改共享 base、坐标/GT、split 或 cache schema）
- approval: user-approved
- approval_basis: 用户明确要求按指导/V1.2.1 修改架构，直接写最终 plan，复用可用 cache，先 smoke 后全量并开始执行
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留仓库既有治理、CmDecoder 和历史 ObjectInteractionCm 修改）
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806
- run_status: RUNNING
- conclusion: INCONCLUSIVE（smoke 只证明工程链路；全量尚未完成，不能判定科研效果）
- scope: [ObjectInteractionCm V1.2.1 Task](../../)；不修改旧 V1.1 输出、旧 checkpoint、原始 cache、split、Cm/CmDecoder 或共享 `src/base`

**文件**

- [ObjectInteractionCm Task 全部本次实现范围](../../) — 新增/更新 V1.2.1 架构、最终 plan、D128/C32 模型、双路 interaction、mask-before-top-k、sample-level mask、masked SlotAttn、additive object/hand decoder、runner、配置和 scale/miss 工具。
- [V1.2.1 架构](../architecture/V1.2.1.md)、[V1.2.1 最终 plan](../plan/V1.2.1.md)、[V1.2.1 指导](../指导/V1.2.1.md) — 文档合同和执行范围。
- [train-only scales](../../../../../data/processed_data/object_interaction_cm_v1_2_1/scales_train.json) — 复用现有 `object_interaction_cm_v1_1/index.json` 的训练 split 统计，未导出 cache；`s_geo=0.0321234m`、`s_hand_flow=0.0701309m`、`s_obj_flow=0.0578926m`。
- [sampling miss](../../../../../data/processed_data/object_interaction_cm_v1_2_1/sampling_miss.json) — 512 个抽样样本，`N_active_4096` 均值 1759.42、`N_active_1024` 均值 439.99，条件 miss `0.0%`，因此 `sampling_retry_attempts=0`。
- [smoke run](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/) — `run_status=COMPLETED`；两步 CUDA smoke 的 [manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/metrics.jsonl) 和 [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/checkpoints/latest.pt) 已生成。
- [full run](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/) — `run_status=RUNNING`；[manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) 已生成；终态 checkpoint/summary 为 `PENDING`。
- [full launcher log](../../../../../outputs/objectinteractioncm/v1_2_1_full_launcher.log) — 全量命令输出入口。

**原因**

落实 V1.2.1 的延迟压缩和物体侧聚集设计：interaction/hand flow 在 D=128 保留，SlotAttn 后才投影到 16×32 Cm；object decoder 仅使用原始 point/normal + Cm + anchor geometry；两个 decoder 都沿 slot contribution 直接求和。空采样样本保持 B 维并用 `sample_valid` 屏蔽监督，dummy 仅保持 SlotAttn 数值合法。现有混合 index/schema 已被 loader 直接接受，因此不重复导出 cache。

**验证**

- `python3 -m py_compile src/task/ObjectInteractionCm/*.py src/task/ObjectInteractionCm/tools/data/*.py`：通过。
- `python3 -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`。
- V1.1 best checkpoint 严格加载：通过兼容 legacy model path，未改变旧 checkpoint。
- scale/miss 两个只读脚本正常完成；统计产物见上方链接，条件 miss 低于启用 retry 的阈值。
- CUDA smoke `run_id=object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708`：`COMPLETED`，2 steps，forward/backward 无 NaN，sample mask、两路 additive decoder、manifest 和 checkpoint 均正常；这是工程 smoke 证据，不是科研结论。
- 全量命令：`setsid env CUDA_VISIBLE_DEVICES=4 nohup /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/grab_inspire_f1_v1_2_1.yaml`；启动时约 `279 samples/s`，估计约 6.4 小时，当前仍为 `RUNNING`。

**回滚**

删除/恢复本条列出的 V1.2.1 Task 代码、配置、文档和诊断脚本即可回滚实现；scale/miss JSON 与 `outputs/` 运行目录为新增可删除产物，旧 V1.1 cache/checkpoint/output 不受影响。全量运行可安全停止后保留已写入 manifest/log/checkpoint。

## 2026-09-03 18:36:00 +0800 — V1.2.1 空样本数值安全与 provenance 补强

- activity_id: ACT-20260903-183600-OBJECTINTERACTIONCM-V121-SAFETY
- timestamp: 2026-09-03 18:36:00 +0800
- modification_version: V1.2.1.2
- type: code / documentation
- change_level: L1
- approval: auto
- approval_basis: 按 V1.2.1 已批准架构补齐 dummy token 的零输入和 scale manifest 元数据记录；不改变有效 interaction 路径或研究变量
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806
- run_status: RUNNING
- conclusion: INCONCLUSIVE（仅数值安全/可追溯性补强，不产生科研结论）
- scope: [ObjectInteractionCm V1.2.1 Task](../../)；当前全量 run 继续在 GPU4 运行，旧代码/cache/checkpoint 不改

**文件**

- [ObjectInteractionCm Task](../../) — SlotAttn 对全空样本强制零 dummy 输入，runner provenance 记录 scale values，训练入口说明更新。
- [全量运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/) — 运行保持 `RUNNING`，不重启、不覆盖已有产物。

**原因**

避免空样本的无效 token 传播非零值，并让后续新运行的 metadata 明确记录 `s_geo/s_hand_flow/s_obj_flow`。该补强只影响 `sample_valid=0` 的数值安全分支；当前抽样 miss 统计为 0%，不改变有效样本的 forward 或 loss 语义。

**验证**

- `python3 -m py_compile src/task/ObjectInteractionCm/model.py src/task/ObjectInteractionCm/runner.py src/task/ObjectInteractionCm/slot_attention.py`：通过。
- 直接构造全空 mask 的 SlotAttn：slots/assignment/slot_weights 全 finite，dummy_used=True，active slot weight=0。
- `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过，最新条目链接可导航。
- 全量运行检查：PID 710877 仍存活，metrics 已推进至约 step 2600，无 NaN/traceback；该运行的 manifest 已在补强前生成，scale 数值仍可由 config 指向的 JSON 复核。

**回滚**

恢复本条涉及的 Task 文件即可回滚补强；不停止或删除正在运行的全量任务及其产物。

## 2026-09-03 19:04:57 +0800 — ObjectInteractionCm 从 GPU4 恢复到 GPU0/1/2 三卡

- activity_id: ACT-20260903-190457-OBJECTINTERACTIONCM-V121-MIGRATE
- timestamp: 2026-09-03 19:04:57 +0800
- modification_version: V1.2.1.3
- type: operation / diagnostic
- change_level: L3（停止并恢复长时训练，改变 world size；训练变量保持配置不变）
- approval: user-approved
- approval_basis: 用户明确要求停止 GPU0/1/2 decoder 训练并将 ObjectInteractionCm 放到三张卡
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true
- scope: [ObjectInteractionCm Task](../../)；不修改 cache、旧 checkpoint 或其他 GPU 进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（resume）
- run_status: RUNNING
- conclusion: INCONCLUSIVE（训练尚未完成）

**文件**

- [原 GPU4 run 目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/) — 保留 step `11409` 的 `latest.pt/best.pt`，作为恢复入口。
- [三卡 resume launcher log](../../../../../outputs/objectinteractioncm/v1_2_1_full_3gpu_resume_launcher.log) — `CUDA_VISIBLE_DEVICES=0,1,2`、torchrun 3 ranks 的启动输出。
- [resume manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/run_manifest_resume_20260903_190509.json) — 记录本次显式 resume 和 global batch=96。
- [当前 metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[当前 train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 恢复后继续追加。

**原因**

GPU0/1/2 的 decoder 已按用户指令停止且 GPU1/2 释放；GPU0 剩余 viewer 不占用训练合同。ObjectInteractionCm 原单卡运行在 step `11409` 已保存 checkpoint，因此不从零开始，而是显式从该 checkpoint 恢复到 3 卡，保持模型、数据 split、scale manifest、max_steps 和 loss 不变，仅将 world size 改为 3（per-device batch=32，global batch=96）。

**验证**

- 三卡启动成功：world size=3，rank 0/1/2 均完成 checkpoint load，日志显示 `Loaded checkpoint ... at step 11409`。
- 恢复后已推进至约 step `11700`，三张卡显存分别约 12.0/9.4/9.4 GiB；当前无 OOM、NaN、NCCL 或 traceback。
- GPU4 已释放到约 7 MiB；0 卡 viewer 进程保持运行，未误杀。
- `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过。

**回滚**

停止当前三卡 run 后可从原始 step `11409` checkpoint 重新选择设备恢复；本次未删除任何 decoder/ObjectInteractionCm 运行产物或 cache。

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

## 2026-09-03 11:35:41 +0800 — ObjectInteractionCm 物体预测劣化原因诊断

- activity_id: ACT-20260903-113541-OBJECTINTERACTIONCM-OBJECT-REGRESSION-DIAGNOSIS
- timestamp: 2026-09-03 11:35:41 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求比较 ObjectInteractionCm 与 Cm 的物体预测；本次仅阅读代码、配置、日志和既有运行产物，不修改实现、数据或研究变量
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户既有治理、CmDecoder 与 ObjectInteractionCm 文档改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948；comparison_run_id: cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（确认一个 KNN padding 实现缺陷及多项强烈的基线不等价因素；尚未形成研究效果结论）
- scope: ObjectInteractionCm 物体侧 KNN、padding mask、loss/target scaling、数据采样与旧 Cm C64 基线的训练初始化、batch 和指标口径

**原因**

需要解释 ObjectInteractionCm 的物体 EPE 为什么看起来不如原始 Cm，同时区分实现缺陷、训练条件不等价和物体侧聚集设计本身的影响。审计时发现工作区已有用户改动 [ObjectInteractionCm 架构文档](../architecture/V1.1.md) 与 [ObjectInteractionCm 计划文档](../plan/V1.1.md)，本次未读取其 diff 作为科学改动，也未修改它们。

**文件**

- [ObjectInteractionCm KNN 实现](../../model.py) — `LocalHandInteraction.forward` 在 `topk` 前未将无效 hand padding 距离置为无穷大。
- [ObjectInteractionCm 数据集](../../dataset.py) — Inspire hand 1538 个有效点后追加 1538 个零 padding；物体点从完整池均匀采样且全部计入 object loss。
- [已有架构文档改动](../architecture/V1.1.md)、[已有计划文档改动](../plan/V1.1.md) — 工作区既存用户改动，保留且未纳入本次诊断结论。
- [Cm 模型](../../../Cm/src/model.py)、[Cm runner](../../../Cm/src/runner.py) — 旧路径使用 DenseToken hand 特征；C64 基线启用 target scaling 与稠密 hand 监督。
- [ObjectInteractionCm 运行指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl) — 记录 source-level 邻居数、交互物体点比例和验证 EPE。
- [ObjectInteractionCm 全 stride 评估](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/results.json) — 检查不同 stride 下的测试行为。
- [Cm C64 基线运行](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/metrics.jsonl) — closest apples-to-apples 旧基线。

**关键证据**

- Inspire 首个验证 batch 的 raw top-8 KNN 平均仅 4.31 个有效点，半径内有效邻居平均 0.12；先 mask 无效点再 top-k 时分别为 8.00 和 0.39。raw top-k 中约 41.8% 的物体点没有任何有效邻居。GRAB 无 padding，因此不受该缺陷影响。
- ObjectInteractionCm 最佳验证 equal-source object EPE 为 4.0895 mm；旧 Cm C64 为 4.0367 mm，差约 0.0527 mm（1.31%）。source-level 上 OI 的 GRAB 略好（约 0.0716 mm），但 Inspire 差约 0.1771 mm，劣化并非两个 source 对称发生。
- 旧 Cm C64 使用 global batch 96、约 153540 steps，并从 adapted DenseToken checkpoint 初始化；OI 使用 global batch 32、约 202300 steps、随机初始化。按样本实例数估算，OI 约为旧基线的 44%，因此 step 数不能视为训练量相当。
- OI 为 C32、1024 object points、16 slots（约 64 点/slot）；旧 C64 为 512 object points、16 slots（约 32 点/slot），OI 的 token 表征容量和每 slot 压缩比更差。
- OI 的 hand loss 只在 3 cm 接触区域生效（全体约 18.4%，Inspire 约 6.95%）；旧 C64 hand loss 对有效 hand 点提供稠密监督。OI 同时合并左右手为一个共享集合，旧 Cm 分侧处理，监督和任务语义也不完全相同。
- 旧 Cm object-v2 loader 的语义是按 5 cm candidate mask 产生 `obj_valid_mask`，而 OI 将采样点全部标为 valid；但抽查/汇总当前旧 GRAB cache 的 764737 帧 candidate 数均为 4096，因此该项在这次具体 C64 run 中不是主要差异，仍应在复现实验时显式对齐。

**验证**

- 只读检查上述源码、配置快照、数据 index、既有 metrics/summary 和全 stride results；未改代码或运行产物。
- KNN 对比使用同一 batch 分别执行原始 top-k 与 invalid-distance=`inf` 的 masked top-k，结论只用于实现诊断，不替代独立修复后的回归实验。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响代码、数据或已有运行。
- 若进入 change，首要修复是 KNN 前 mask padding；随后在相同初始化、global batch、采样/分侧语义、target scaling 和评估 stride 下重跑，才能判断物体侧聚集设计本身是否劣化。

## 2026-09-03 14:28:37 +0800 — V1.2 指导补充实现不变量与公平比较合同

- activity_id: ACT-20260903-142837-OBJECTINTERACTIONCM-V12-GUIDANCE-SUPPLEMENT
- timestamp: 2026-09-03 14:28:37 +0800
- modification_version: V1.2.1
- type: documentation
- change_level: L2
- approval: user-approved
- approval_basis: 用户明确要求浏览 V1.2 指导并将上一轮诊断结论补充进去；本次只修改指导文档，不修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户既有治理、CmDecoder 以及 ObjectInteractionCm 架构/计划文档改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948；comparison_run_id: cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（补充已确认的 padding KNN 缺陷、比较合同和诊断边界；不宣称 object-side 方案效果成立）
- scope: `src/task/ObjectInteractionCm/docs/指导/V1.2.md`；补充 KNN mask-before-topk、无交互 object token 策略、baseline parity、处理宽度/尺度、监督/评估口径和配置阈值可追溯性

**文件**

- [V1.2 指导](../指导/V1.2.md) — 新增“本轮诊断补充”章节，将 padding KNN 缺陷升格为 P0 不变量，补充 object-side 稀疏交互、旧 Cm C64 公平比较、训练预算、source/stride 指标口径、slot collapse 排除项及配置合同要求。
- [已有 V1.1 架构文档改动](../architecture/V1.1.md)、[已有 V1.1 计划文档改动](../plan/V1.1.md) — 工作区既存用户改动，仅为 scope 审计列出，本次未修改。

**原因**

上一轮只读诊断确认 `LocalHandInteraction` 在 `topk` 前未排除 Inspire-F1 的 zero padding，且 OI 与旧 Cm 在初始化、global batch、DenseToken、Cm/processing width、监督密度和评估口径上均不完全等价。V1.2 若不把这些事项写成冻结合同，后续实现或实验会继续混入不可解释变量。

**验证**

- 只读复核 [ObjectInteractionCm model.py](../../model.py)、[dataset.py](../../dataset.py)、[runner.py](../../runner.py)、旧 [Cm model.py](../../../Cm/src/model.py) 与 [Cm runner.py](../../../Cm/src/runner.py)，以及 OI/旧 Cm 的 metrics、train.log、full-stride results。
- `git diff --check`：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 3 个 scope 变更路径一致，本地链接可导航。
- 未运行训练、评估、测试或数据处理；新增内容属于研究指导和比较合同，不构成效果证据。

**回滚与后续**

- 删除 [V1.2 指导](../指导/V1.2.md) 末尾新增章节即可回滚本次文档补充，不影响代码、数据、checkpoint 或已有运行。
- 进入 `change` 前仍需为 V1.2 起草并与用户定稿同版本 `plan/V1.2.md`；KNN 修复和 parity 实验不得在 plan 未定稿前执行。

## 2026-09-03 16:33:30 +0800 — V1.2 执行计划草案与架构提案

- activity_id: ACT-20260903-163330-OBJECTINTERACTIONCM-V12-PLAN-ARCHITECTURE
- timestamp: 2026-09-03 16:33:30 +0800
- modification_version: V1.2.2
- type: architecture / documentation
- change_level: L2
- approval: user-approved（仅批准起草计划和架构；实现仍待计划定稿）
- approval_basis: 用户明确要求先写一版 V1.2 计划和详细 architecture；本次只新增文档，不修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（设计提案和执行草案，不构成科研效果结论）
- scope: V1.2 的 hard interaction object mask、KNN mask-before-topk、128 维 late compression、interaction/hand-flow 与 object decoder 的信息隔离、masked SlotAttn、loss/评估合同和回滚边界

**文件**

- [V1.2 执行计划草案](../plan/V1.2.md) — 分阶段文件范围、张量合同、验证命令、parity 实验冻结项和审批闸门；状态为 `draft-for-review`。
- [V1.2 架构提案](../architecture/V1.2.md) — 详细的数据流、`[B,N,128]` 局部特征、`[B,16,128]→[B,16,32]` late projection、硬 mask 语义、decoder/loss 输入和诊断指标；状态为 `proposal-for-review`。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 架构](../architecture/V1.1.md)、[V1.1 计划](../plan/V1.1.md) — 作为本提案的上游/历史文档；工作区既有内容保留，未在本活动中重写。

**原因**

将用户确认的结构意图落成可评审的 V1.2 设计：物体 intrinsic 独立直连 object decoder；interaction + hand flow 先以 128 维保留，再经过只对交互物体点生效的 SlotAttn，最后压缩为 32 维 Cm。计划同时冻结 padding KNN、fallback、loss 覆盖范围和公平比较条件，防止实现阶段混入新的研究变量。

**验证**

- 新文档与现有指导、V1.1 计划/架构的相对链接已按 Task 目录结构写入。
- `git diff --check` 和新增文件的空树 diff 检查：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，7 个本地链接可导航。
- 未运行训练、评估、测试或数据处理；本条只记录文档设计，不产生工程 smoke 或科研证据。

**回滚与后续**

- 删除 [V1.2 执行计划草案](../plan/V1.2.md) 和 [V1.2 架构提案](../architecture/V1.2.md) 及本条记录即可回滚本次文档新增，不影响 V1.1、代码、cache、checkpoint 或旧输出。
- 用户确认并将 plan 标记为 `final` 后，才进入阶段 A-D 的 `change`；若需要改动数据语义、公共 `src/base` 或 cache/schema，必须重新确认范围。

## 2026-09-03 16:50:44 +0800 — KNN 半径 2 cm 交互物体点抽样统计

- activity_id: ACT-20260903-165044-OBJECTINTERACTIONCM-RADIUS-2CM-STATS
- timestamp: 2026-09-03 16:50:44 +0800
- modification_version: V1.2.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求只读统计 KNN 半径改为 2 cm 时的交互物体点数量；未修改代码、配置、数据或运行产物
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有工作区改动）
- run_id: object_interaction_cm_radius2cm_sample_20260903_165044
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（描述交互点密度，不代表模型效果结论）
- scope: ObjectInteractionCm V1.1 index 的 GRAB/Inspire-F1 train/val/test rows；每个 source/split 抽样最多 2000 帧，固定 stride=2、1024 物体点、base_seed=42、K=8，使用有效 hand 点计算 2 cm 半径

**结果摘要**

- val：GRAB 平均 `338/1024=33.0%` 个物体点参与交互，平均有效邻居 `2.52`；Inspire-F1 平均 `134/1024=13.1%`，平均有效邻居 `0.85`。
- train：GRAB `339/1024=33.1%`，Inspire-F1 `133/1024=13.0%`。
- test：GRAB `353/1024=34.5%`，Inspire-F1 `105/1024=10.3%`。
- val 中每帧至少有一个交互物体点的比例约为 GRAB `94.7%`、Inspire-F1 `82.9%`；因此若 SlotAttn 只收硬 mask 点，Inspire-F1 约 `17%` 抽样帧会触发空池 fallback。
- val 完整 4096 点池的近似交互点数为 GRAB `1338`、Inspire-F1 `540`；1024 点采样后的比例基本一致。

**文件**

- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.2 计划草案](../plan/V1.2.md)、[V1.1 架构](../architecture/V1.1.md)、[V1.2 架构提案](../architecture/V1.2.md) — 统计所依据的 Task 研究指导、计划和架构合同；这些文档中的工作区改动均保留，未因本次统计重写。

**原因**

需要在决定 KNN 半径前量化硬 mask 的稀疏度，特别是确认 Inspire-F1 在 2 cm 下是否会出现大量空 SlotAttn 池，以及 1024 点采样后每帧实际有多少物体点进入交互聚集。

**验证**

- 统计脚本以 `hand_valid_mask` 过滤 padding，并在有效手点上执行 cKDTree `K=8` 查询；没有使用当前实现中可能受 padding 影响的 raw top-k 结果。
- 命令：`PYTHONPATH=/home2/wyy/oyx_ws/Ref2Dex python3 /tmp/oi_radius_stats_fast.py`。
- 结果为抽样估计，不是全量逐帧扫描；抽样帧和统计口径固定，可按同一脚本复核。

## 2026-09-03 17:01:48 +0800 — 重排 V1.2 architecture 为端到端张量流

- activity_id: ACT-20260903-170148-OBJECTINTERACTIONCM-V12-ARCHITECTURE-REORDER
- timestamp: 2026-09-03 17:01:48 +0800
- modification_version: V1.2.4
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户指出原 architecture 缺少先总览后分模块的结构，并要求统一张量维度；本次仅重排和细化 V1.2 architecture，不改代码、配置、数据或 plan
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（architecture 文档重排，不构成工程或科研效果证据）
- scope: V1.2 architecture 的端到端总览、统一符号表、阶段输入/输出表、KNN→intrinsic→interaction→hard mask→SlotAttn→late projection→双 decoder 的顺序和模块张量尺寸

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 改为先给完整数据流和统一尺寸，再按执行顺序介绍数据整理、KNN、intrinsic、edge interaction、fusion、hard mask、SlotAttn、anchor、两个 decoder、loss 和诊断。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.2 计划](../plan/V1.2.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史合同，保留未改。

**原因**

原文虽然包含各模块约束，但总流程、模块顺序和中间 tensor shape 分散在不同章节，无法快速检查 interaction 是否绕过 Cm。重排后统一使用 `N=1024`、`H=3076`、`K=8`、`S=16`、`D=128`、`C=32`，并在总览表中列出每个阶段的输入、输出和消费者。

**验证**

- `git diff --no-index --check /dev/null src/task/ObjectInteractionCm/docs/architecture/V1.2.md`：通过。
- 逐段检查总览图、阶段表和模块公式的维度一致性；确认唯一 `D→C` 投影位于 SlotAttn 输出后，object decoder 只接收 `Ofeat` 和 `Cm`。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，5 个本地链接可导航。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:48:21 +0800 — 按 V1.2 指导收敛双路 additive 与 hand-flow 一级信息

- activity_id: ACT-20260903-174821-OBJECTINTERACTIONCM-V12-FLOW-VALUE-ADDITIVE
- timestamp: 2026-09-03 17:48:21 +0800
- modification_version: V1.2.9
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户要求按 V1.2 指导继续修改；落实真正 additive 聚合、hand-flow 独立 value/statistic、5 cm scaling、sample drop 和采样漏交互诊断；未修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（架构/计划合同更新，不构成工程或科研效果证据）
- scope: V1.2 architecture/plan 的 5 cm 固定半径、raw object decoder、空样本 drop、`Egeo/Eflow` 分离 value、`vbar/sbar` 显式 flow statistic、object/hand additive 求和与 sampling-induced-drop 指标

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 修正为 key 可合并而 value 分路；fusion 输入明确为 388 维；object/hand decoder 均直接求和 slot contribution；补充 5 cm 特征尺度和 sampling-induced-drop 诊断。
- [V1.2 plan](../plan/V1.2.md) — 同步双流 additive、独立 flow value/statistic、raw object decoder、drop_sample 和验证合同。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史文档，保留未改。

**原因**

V1.2 指导指出原设计的 object `softmax(candidate_flow)` 仍是 mixture，且 `Egeo+Eflow→V` 会在局部聚合前丢失 hand-flow 的独立影响。本次将两种 decoder 都改为真正 additive，并让 hand-flow 通过独立 value 和未投影的 `vbar/sbar` 保持一级输入信息。

**验证**

- 逐段检查 `D=128`、`C=32`、fusion `388→128`、object decoder `44→128→128`、hand decoder `166→128→128` 和 `B→B'` 维度。
- 确认唯一 `128→32` projection 仍位于 SlotAttn 后；flow contribution usage 只作 diagnostic，不参与前向求和。
- `git diff --check`、新增/未跟踪文档空树 diff 和 `audit_diff.py --check-links` 在本条 activity 完成后执行。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:47:14 +0800 — 按 V1.2 指导收敛双流 additive 与独立 hand-flow value

- activity_id: ACT-20260903-174714-OBJECTINTERACTIONCM-V12-ADDITIVE-FLOW-DESIGN
- timestamp: 2026-09-03 17:47:14 +0800
- modification_version: V1.2.8
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户要求按 V1.2 指导继续修改；同步落实真正 additive 的 object/hand flow、独立 hand-flow value/statistic、5 cm 和 sample drop 约束；未修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（架构/计划合同更新，不构成工程或科研效果证据）
- scope: V1.2 architecture/plan 的真实 additive 聚合、object/hand flow 双路 value、显式 flow statistic、5 cm 特征尺度、sample-level drop 与 sampling-induced-drop 诊断

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — object/hand decoder 改为逐 slot contribution 直接求和；local interaction 使用 `Igeo/Iflow/vbar/sbar` 四路 flow-preserving 输出；补充 5 cm 几何/hand-flow scaling 和采样漏交互指标。
- [V1.2 plan](../plan/V1.2.md) — 同步双流 additive、独立 value、显式 flow statistic、drop_sample 和诊断合同。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史文档，保留未改。

**原因**

指导指出原 architecture 的 `softmax(candidate_flow)` 仍是 mixture，并且 `Egeo+Eflow→V` 会在 local aggregation 前丢失 hand-flow 的独立影响。现在将两处都改为真正 additive，同时保留未投影的 flow statistic；这使 hand-flow 仍是 object-side representation 的一级信息，并与旧 Cm 的 additive 语义一致。

**验证**

- 逐段检查 object/hand flow 的 forward 公式均为 `Σ_s contribution_s`，usage 只作范数 diagnostic。
- 检查 local interaction 的 key/value 分离、`[B,N,388]→[B,N,128]` fusion、`D=128→C=32` 唯一投影位置及 5 cm scaling 公式。
- `git diff --check` 和 `audit_diff.py --check-links` 在本条 activity 完成后执行。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:30:04 +0800 — 核对 ObjectInteractionCm 当前 slot flow 聚合方式

- activity_id: ACT-20260903-173004-OBJECTINTERACTIONCM-AGGREGATION-DIAGNOSTIC
- timestamp: 2026-09-03 17:30:04 +0800
- modification_version: V1.2.7
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问当前 object/hand flow 是否为 additive；只读核对实际代码和旧 Cm 聚合分支，未修改实现
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有工作区改动）
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（确认实现语义，不代表模型效果结论）
- scope: 当前 ObjectInteractionCm 的 object/hand decoder slot aggregation，以及旧 Cm `use_additive_slot_contributions` 和 hand-flow decoder 的对照实现

**文件**

- [ObjectInteractionCm decoder](../../decoder.py) — 当前 object/hand decoder 均先生成 per-slot candidate flow，再用 `softmax(dim=slots)` 加权求和。
- [ObjectInteractionCm model](../../model.py) — 当前模型无 `use_additive_slot_contributions` 配置，并将两个 decoder 都按 mixture 路径调用。
- [旧 Cm model](../../../Cm/src/model.py) — 旧 C64 对照运行启用 additive object contributions；hand-flow decoder 在启用时直接对 slot contributions 求和。
- [V1.2 architecture](../architecture/V1.2.md)、[V1.2 plan](../plan/V1.2.md)、[V1.2 指导](../指导/V1.2.md)、[V1.1 architecture](../architecture/V1.1.md)、[V1.1 plan](../plan/V1.1.md) — 当前设计合同、上游指导和历史对照；未因本次只读核对改写。

**原因**

需要区分“当前 OI 已实现的聚合方式”和“V1.2 architecture 计划采用的 additive 方式”，避免把旧 Cm C64 运行配置或设计草案误认为当前 OI 已经具备。

**验证**

- 当前 OI object decoder：`weights=softmax(logits, dim=slots)`，`prediction=Σ weights×candidate_flow`。
- 当前 OI hand decoder：同样使用 `softmax` slot weights 后加权求和。
- 旧 Cm object decoder：`use_additive_slot_contributions=True` 时直接对 slot flow contributions 求和；旧 hand-flow decoder 在启用时也直接求和。
- 只读检查源码，未运行训练、评估、测试或数据处理。

## 2026-09-03 17:23:17 +0800 — V1.2 固定 5 cm、移除 null-Cm 并定义空样本丢弃

- activity_id: ACT-20260903-172317-OBJECTINTERACTIONCM-V12-EMPTY-SAMPLE-POLICY
- timestamp: 2026-09-03 17:23:17 +0800
- modification_version: V1.2.6
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户明确要求按 V1.2 指导继续修改架构，首版使用 5 cm，不使用 null-Cm，采样物体点无交互时跳过该样本训练；同步更新配对 plan 以避免合同不一致
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（架构/计划合同更新，不构成工程或科研效果证据）
- scope: V1.2 的 5 cm KNN、raw object point/normal decoder 输入、`sample_keep`、`B→B'` batch compaction、无交互样本 drop、空 batch optimizer 处理和跳过样本指标

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 固定 `r=0.05 m`；object decoder 使用 raw point/normal + Cm + anchor geometry；移除 null-Cm，定义 `sample_keep` 和有效子 batch `B'`。
- [V1.2 plan](../plan/V1.2.md) — 同步张量表、5 cm 默认、raw decoder 输入、drop_sample、全 batch drop 和验证标准。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史文档，保留未改。

**原因**

2 cm 统计显示 Inspire-F1 的交互点过稀；用户决定首版固定 5 cm，并明确不训练 null-Cm 路径。若 1024 个采样物体点没有任何有效邻居，则该样本从 SlotAttn、decoder、loss 和指标中移除，而不是伪造一个 Cm。

**验证**

- 逐段检查 architecture 总览图、阶段表和模块公式的 batch 维：KNN 前为 `B`，compaction 后统一为 `B'`。
- `git diff --check`、新增/未跟踪文档的空树 diff 检查：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，5 个本地链接可导航。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:21:21 +0800 — V1.2 固定 5 cm 并改为空交互样本丢弃

- activity_id: ACT-20260903-172121-OBJECTINTERACTIONCM-V12-DROP-EMPTY-SAMPLES
- timestamp: 2026-09-03 17:21:21 +0800
- modification_version: V1.2.5
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户明确要求 V1.2 首版继续使用 5 cm KNN、禁止 null-Cm，并在采样物体点没有任何交互点时跳过该样本训练；本次只同步 architecture 和 plan，不改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（设计合同更新，不构成工程或科研效果证据）
- scope: V1.2 的 KNN radius、object decoder 原始 geometry 输入、sample-level `B→B'` compaction、空交互样本 drop、训练/评估跳过口径及相关诊断指标

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 固定 `r=0.05 m`，object decoder 改为 raw point/normal + Cm + anchor geometry，移除 null-Cm，定义 `sample_keep` 和 `B'` 子 batch。
- [V1.2 plan](../plan/V1.2.md) — 同步 object decoder 输入、5 cm 默认值、空样本 drop、全 batch drop、验证和 parity 指标。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史合同，保留未改。

**原因**

2 cm 抽样统计显示 Inspire-F1 的交互物体点过稀；用户决定 V1.2 首版固定 5 cm，并明确不训练 null-Cm 路径。若 1024 个采样物体点没有任何有效 interaction，则将该样本从 SlotAttn、decoder、loss 和指标中移除，避免用伪造 Cm 或零流污染训练。

**验证**

- 逐段检查 architecture 的总览图、阶段表、`B'` 维度和 raw object decoder 的 44 维输入是否一致。
- `git diff --check`、新增 architecture 的空树 diff 检查：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，5 个本地链接可导航。
- 未运行训练、评估、测试或数据处理。
