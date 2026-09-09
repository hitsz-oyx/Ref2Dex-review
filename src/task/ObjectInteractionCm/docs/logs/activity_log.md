# ObjectInteractionCm 活动记录

- scope: task:ObjectInteractionCm
- last_updated: 2026-09-09
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [任务入口](../README.md)、[执行计划](../plan/V1.1.md)、[架构快照](../architecture/V1.1.md)、[指导](../指导/V1.1.md)

## 2026-09-09 22:34:07 +0800 — V1.2.18 估算离线 KNN 构建耗时

- activity_id: `ACT-20260909-223407-OBJECTINTERACTIONCM-OFFLINE-KNN-RUNTIME-DIAGNOSTIC`
- timestamp: `2026-09-09 22:34:07 +0800`
- modification_version: `V1.2.18`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问离线对每帧完整 4096 物体点计算 K=16 最近手点是否会很久；本次只读 benchmark 当前正式/预览 cache，不生成离线索引文件。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_id: `object_interaction_cm_offline_knn_runtime_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（耗时估算有效；正式离线索引生成仍需新的 cache/schema 变体和用户确认）
- scope: 使用当前 V1.2.5 正式 1538 点 Inspire cache 与 V1.2.15 10135 点预览 cache，评估 `4096 object × K=16 hand` 离线 KNN 的 CPU/GPU 量级、active-only 节省和 IO 写入边界；未修改模型、配置、cache、GT、split 或 checkpoint。

**结果**

- CPU `scipy.spatial.cKDTree` 单进程实测：1538 点约 `13.6–14.0 ms/frame`；10135 点约 `17.9–20.8 ms/frame`。按当前 Inspire-RL `72935` 帧线性外推，10135 点约 `21.7–25.2 min`；按全量 `159476` 帧约 `47.5–55.2 min`。
- GPU PyTorch3D batched 实测（RTX 3090，含 NumPy→GPU transfer，不含磁盘写索引）：1538 点约 `0.093 ms/frame`；10135 点约 `0.348 ms/frame`。纯计算外推 Inspire-RL 约 `0.42 min`，全量约 `0.93 min`。
- 实际端到端 builder 还要读 geometry、转 `uint16`、写 `.npy` 和写 manifest。对于 10135 点 Inspire，若只写 `[T,4096,16] uint16` 索引，输出约 `8.90 GiB`；IO 通常会把端到端时间拉到数分钟级。
- 当前 V1.2.5 `active_only=true`。Inspire-RL 共 `72935` 帧，其中 active `39554` 帧（`54.23%`）；GRAB 共 `86541` 帧，active `52278` 帧（`60.41%`）。若索引只为训练实际使用的 active 帧计算/保存，Inspire 输出约从 `8.90 GiB` 降到 `4.83 GiB`，计算时间也约降到 `54%`。
- 因此粗略判断：GPU batched builder 做 Inspire-only active 帧，应是几分钟量级；CPU 单进程是十几分钟量级，CPU 多进程可压到几分钟到十分钟。全量 all-source 索引也不像训练那样要数小时，更可能是十几分钟到一小时内，取决于 IO 和并行策略。

**实现建议**

- 若只是给 10135 Inspire 加速，优先做 Inspire-only active-frame 索引 cache；MANO 1538 点 KNN 本身较小，可先不做。
- builder 应按 sequence 分文件保存，便于 resume，不要一次写全局巨型数组；每个 geometry manifest 记录 `knn_index_file`、`knn_k`、`hand_point_count`、`object_pool_points`、`active_only` 和点集指纹。
- 训练读取时必须保留当前 object 随机抽样语义：Dataset 从 4096 中抽 1024 个点后，再从离线 `[4096,16]` 中 gather 对应行。
- 若后续想进一步提速，3 cm `hand_supervision_mask` 也应离线保存；否则 10135 点会把 Dataset 侧 CPU KDTree 成本放大。

**原因**

需要判断离线 KNN 是否会变成长时间预处理，并区分 compute、IO 写入和 active-only 策略的成本。该判断会影响后续是否值得为 10135 点 Inspire cache 新增 KNN 索引字段。

**文件与证据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)：更新 ObjectInteractionCm 当前指针为 `V1.2.18`。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)：记录本次离线 KNN 构建耗时诊断。
- [V1.2.5 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)
- [面积比例手点轨迹预览 index](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/index.json)
- [Dataset](../../dataset.py)
- [ObjectInteractionCm model](../../model.py)

**验证**

- CPU benchmark：对正式 1538 点 Inspire sequence 与预览 10135 点 Inspire sequence 各取最多 48 帧，执行 `cKDTree(hand).query(object_pool, k=16)`。
- GPU benchmark：对同两类 sequence 各取最多 96 帧，用 PyTorch3D `knn_points` batched 计算 `4096×K16`，chunk 分别为 16 和 8。
- active frame 统计：只读扫描 V1.2.5 index 下所有 sequence 的 `obj_candidate_mask_5cm.npy`。
- 未生成离线索引、未运行训练或评估；`git diff --check` 与本条 activity 的本地链接审计在交接前执行。

**回滚入口**

- 删除本条活动记录并将 [当前版本指针](../../../../../docs/current_versions.yaml) 恢复为 `V1.2.17`；正式 cache、模型、训练配置和既有预览不受影响。

## 2026-09-09 22:15:21 +0800 — V1.2.17 评估离线 KNN 索引加速方案

- activity_id: `ACT-20260909-221521-OBJECTINTERACTIONCM-OFFLINE-KNN-DIAGNOSTIC`
- timestamp: `2026-09-09 22:15:21 +0800`
- modification_version: `V1.2.17`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 KNN 是否可以加速以及是否可以离线预计算；本次只读检查当前 Dataset/模型语义、估算索引存储并在空闲 GPU 上进行小型 KNN/gather 基准，未修改代码、cache、配置或运行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_id: `object_interaction_cm_offline_knn_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（离线保存完整物体池的 KNN 索引在当前刚体坐标语义下可行；实际整步训练收益仍需 pilot parity/throughput 实验确认，不构成科研效果结论）
- scope: 当前 V1.2.5 ObjectInteractionCm 的 `4096` 物体池、运行时随机抽取 `1024` 点、`K=16` KNN、`5 cm` edge mask、`3 cm` hand supervision mask、变长 Inspire 手点的离线索引设计和 GPU 微基准；正式 cache、模型、训练配置、GT、split 和 checkpoint 未修改。

**结论与推荐方案**

- 可以离线预计算。为保持当前每个 epoch 从 `4096` 个物体点随机抽取 `1024` 个的语义，应为每帧完整 `4096` 点物体池保存 `K=16` 个最近手点的索引，而不是只保存一次运行时抽到的 `1024` 点结果。
- 推荐 cache 保存 `knn_indices`，形状 `[T,4096,16]`、`uint16`；`10135 < 65535`，索引类型足够。Dataset 在随机抽样后 gather 对应的 `[1024,16]` 索引，模型只对这 16 个邻居重新计算距离并施加 `distance <= 0.05 m`，因此不必保存距离或逐 edge mask。
- 当前 object/hand 都经过同一个 `object_pose_t` 刚体变换；刚体变换不改变邻居索引和距离，所以索引可在 world 坐标或 object frame 离线计算。它只依赖当前帧几何，不依赖未来 stride 或 hand flow。
- 若将 Inspire 改为 `10135` 点，必须用新手点集合重建索引；若改变 `K` 或 object/hand 点池，也必须重建。仅改变半径时，保留 top-16 索引并在运行时重算 16 个距离即可继续使用。

**存储量**

- 每帧完整索引为 `4096×16×2 = 131072 B`，约 `128 KiB`。
- 当前 `72935` 个 Inspire-RL 帧需要约 `8.90 GiB`；当前全部 `159476` 帧都保存索引需要约 `19.47 GiB`。
- 仅保存索引比同时保存 `float16` 距离或逐 edge bool 更省；若再保存每物体点的有效邻居数，额外只需约 `0.28 GiB`（Inspire）或 `0.61 GiB`（全量）。
- 结合上一条 V1.2.16 估算，10135 Inspire 手点使当前 cache 约为 `34.11 GiB`；再加 Inspire-only KNN 索引约 `43.01 GiB`，全量 source 索引约 `53.58 GiB`。当前文件系统可用空间约 `815 GiB`，容量不是立即阻塞，但 `/home2` 已使用约 `97%`，应避免产生多份全量副本。

**运行时与实现边界**

- GPU 微基准（RTX 3090，`B=8`、`N_obj=1024`、`K=16`）中，PyTorch3D KNN 从 `1538` 手点的 `0.584 ms/batch` 增至 `10135` 手点的 `2.116 ms/batch`，约 `3.63x`。
- 使用离线索引后，gather 16 个邻居并重算 16 个距离约 `0.068 ms/batch`；相对 `10135` 点 KNN，KNN 选择阶段约 `31.2x` 降低。该基准只覆盖邻居选择，不代表整个训练 step 同比例加速。
- 离线 KNN 不会消除 `SharedHandFlowDecoder` 对全部手点和 `16` 个 slot 的计算；若手点变为 `10135`，该部分仍约按点数线性增大。若 hand decoder 只用于 `3 cm` mask 下的辅助 loss，可进一步考虑只在监督点上计算，但这属于额外的模型/损失实现变更，需要单独确认。
- 变长 MANO/Inspire 混合 batch 仍需处理。当前不同 valid point count 会绕过 prefix fast path，退回 `cdist+topk`；应采用 source/point-count homogeneous batch，或实现按有效点数分组的 KNN 路径。
- 当前 Dataset 每个样本还会用 CPU `cKDTree` 计算 `3 cm` hand supervision mask。若使用 `10135` 点，建议同时离线保存逐手点 bool mask；Inspire train/val 约增加 `0.69 GiB`，可移除该 CPU 重复计算。

**文件与证据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)：更新 ObjectInteractionCm 当前指针为 `V1.2.17`。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)：记录本次离线 KNN 方案诊断。
- [V1.2.5 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)
- [Dataset](../../dataset.py)
- [ObjectInteractionCm model](../../model.py)
- [ObjectInteractionCm decoder](../../decoder.py)
- [面积比例手点预览 index](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/index.json)

**原因**

需要区分“离线缓存能否消除全量手点搜索”和“10135 点仍会带来的 decoder/padding 成本”。完整物体池索引可以保留当前 object 随机采样，而只把每次前向的 KNN 搜索替换为固定邻居 gather。

**验证**

- 只读检查 [Dataset](../../dataset.py) 的随机 object sampling、刚体坐标转换、hand supervision mask 和 [model](../../model.py) 的 PyTorch3D/fallback KNN 分支。
- 扫描 V1.2.5 index 得到当前 Inspire-RL `285` 条序列、`72935` 帧和全量 `159476` 帧，并计算 `uint16` 索引及可选 mask/count/distance 存储量。
- 使用 `CUDA_VISIBLE_DEVICES=1` 在 NVIDIA GeForce RTX 3090 上完成 `B=8,N_obj=1024,K=16` 的 1538/10135 KNN 与 16-neighbor gather 微基准；未接触正在使用的 GPU 7 任务。
- 使用 3 条 10135 点 Inspire 预览序列做稀疏性抽查：active object fraction 为 `0.4286–0.8160`，平均有效邻居为 `6.571–13.018`；样本量不足以据此替代全量存储设计。
- `git diff --check` 与本条 activity 的本地链接审计在交接前执行。

**回滚入口**

- 删除本条活动记录并将 [当前版本指针](../../../../../docs/current_versions.yaml) 恢复为 `V1.2.16`；正式 cache、模型、训练配置和既有预览不受影响。

## 2026-09-09 21:58:14 +0800 — V1.2.16 评估 Inspire 10135 手点 cache 存储开销

- activity_id: `ACT-20260909-215814-OBJECTINTERACTIONCM-CACHE-STORAGE-10135-DIAGNOSTIC`
- timestamp: `2026-09-09 21:58:14 +0800`
- modification_version: `V1.2.16`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问将 Inspire 手点从 1538 增加到 10135 后的 cache 存储开销；本次只读统计现有 V1.2.5 cache 并估算线性存储/邻域搜索规模，不重建或覆盖 cache。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_id: `object_interaction_cm_cache_storage_10135_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（存储和张量规模估算有效；是否值得接入训练仍需单独实验，不构成效果结论）
- scope: 当前 `object_interaction_cm_dexplore_rl_v1_2_5` cache 的 Inspire-RL train/val geometry、1538→10135 手点存储估算、混合 batch padding 和 `LocalHandInteraction` KNN 输入规模；正式 cache、训练配置、模型、GT、split 和 checkpoint 未修改。

**结果**

- 当前 Inspire-RL 为 `285` 条序列、`72935` 帧；完整 cache 约 `9.19 GiB`，其中 `hand_points_world.npy` 与 `hand_normals_world.npy` 合计约 `2.51 GiB`。
- 每帧手部两份 `float32 [N,3]` 数组从 `36912 B`（1538 点）增至 `243240 B`（10135 点），手部部分增加 `6.59x`，每帧多约 `201.5 KiB`。
- 仅替换 Inspire-RL 的手点数组，预计新增约 `14.02 GiB`；Inspire-RL 子 cache 约从 `9.19 GiB` 增至 `23.21 GiB`，整个当前 `20.09 GiB` cache 约增至 `34.11 GiB`，总量约 `1.70x`。
- 若物体点池仍为 `4096` 点，单帧 object+hand 几何从约 `132.0 KiB` 增至 `333.5 KiB`，约 `2.53x`；这还未计入 loader 临时张量。
- 当前 Dataset/模型使用统一的 `num_hand_points=1538` 和 `max_hand_points=3076`。若 Inspire 直接使用 `10135` 点，混合 GRAB/Inspire batch 需要扩大统一 padding 或实现变长 batch；否则会触发 shape/schema 不兼容。
- 若前向也使用全部 `10135` 点，物体 `1024` 点对手点的邻域搜索候选量约为原来的 `6.59x`。混合不同有效点数时，当前前缀 mask 快路径还可能退回完整 `cdist` 路径，运行时成本会高于单纯的磁盘增长。

**文件与证据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)：更新 ObjectInteractionCm 当前指针为 `V1.2.16`。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)：记录本次只读存储/运行时开销诊断。
- [V1.2.5 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)
- [示例 Inspire geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s5_mouse_pass_1/geometry/manifest.json)
- [Dataset](../../dataset.py)
- [ObjectInteractionCm model](../../model.py)

**原因**

需要将“cache 多占多少磁盘”和“训练前向是否变重”分开判断。10135 点主要改变手点数组、统一 batch padding 和 KNN 搜索规模；它不会自动改变物体池 `4096` 点或运行时物体抽样 `1024` 点。

**验证**

- 只读扫描 `index.json` 中所有 train/val Inspire-RL sequence，并按实际 `.npy` 文件大小汇总当前 cache。
- 按 `float32`、两份 `[T,N,3]` 手点/法线数组和 `1538→10135` 的点数比例估算新存储；当前 `.npy` header 对总量影响可忽略。
- 对照 [Dataset](../../dataset.py) 的统一点数/padding 合同和 [model](../../model.py) 的 prefix fast path、KNN 输入；未运行训练、评估、数据重建或前向 benchmark。
- `git diff --check` 与本条 activity 的本地链接审计在交接前执行。

**回滚入口**

- 删除本条活动记录并将 [当前版本指针](../../../../../docs/current_versions.yaml) 恢复为 `V1.2.15`；正式 cache、训练配置、模型和既有预览不受影响。

## 2026-09-09 20:57:40 +0800 — V1.2.15 面积比例手点轨迹预览与 viewer 兼容

- activity_id: `ACT-20260909-205740-OBJECTINTERACTIONCM-VARIABLE-HAND-TRAJECTORY-PREVIEW`
- timestamp: `2026-09-09 20:57:40 +0800`
- modification_version: `V1.2.15`
- type: `code / diagnostic / data / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认按最新完整 mesh 面积比例方案继续：MANO 2048 点、Inspire 10135 点，生成若干条数据集轨迹并使用 `visualize_grab.py` 可视化。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)（用户确认后追加轨迹预览扩展）
- run_id: `variable_hand_trajectory_preview_20260909`
- run_status: `COMPLETED`; viewer_run_id: `visualize_grab_8100_20260909`，viewer `run_status: RUNNING`，监听 `0.0.0.0:8100`
- conclusion: `SUPPORTED`（工程预览合同、点数/法线/帧对齐和 viewer 读取均通过；是否解决局部视觉空带仍需用户在交互界面中判断，不构成训练效果结论）
- scope: 从正式 V1.2.5 index 的 train split 确定性选择 MANO 和 Inspire-RL 各 3 条序列；沿用 4096 点物体池、物体姿态、帧号和 provenance，仅按 canonical triangle+barycentric 对应重建 MANO 2048 / Inspire 10135 手点，并逐帧重算 flat normal 与 5 cm activity mask。viewer 读取端允许 manifest 声明的变长手点数。
- protection: 未修改正式 `data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/` cache、训练配置、GT、split 定义、坐标系、模型或 checkpoint。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)
- [轨迹预览构建脚本](../../research/hand_region_sampling/build_trajectory_preview.py)
- [viewer 兼容修改](../../visualize_grab.py)
- [预览 index](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/index.json)
- [assignment](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/assignment.json)
- [run manifest](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/run_manifest.json)
- [预览输出目录](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/)

**选中的序列**

- MANO：`s1/airplane_fly_1`（279 帧，2048 点，223 帧 5 cm active）、`s1/airplane_pass_1`（219 帧，2048 点，130 active）、`s1/alarmclock_lift`（510 帧，2048 点，450 active）。
- Inspire-RL：`s1/airplane_lift`（432 帧，10135 点，358 active）、`s1/alarmclock_see_1`（254 帧，10135 点，135 active）、`s1/apple_pass_1`（177 帧，10135 点，109 active）。

**原因**

最新 PLY 采样合同只给出了 canonical 静态点云，无法直接观察手点在真实物体轨迹中的覆盖。此次用同一 face/barycentric 对应跨帧重建手点，并让 viewer 读取 manifest 中的手点数，从而在不重建正式 cache 的前提下检查 2048/10135 点采样的动态空间分布。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/build_trajectory_preview.py src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- 6 条序列数组检查：物体池均为 `[T,4096,3]`；MANO 手点为 `[T,2048,3]`；Inspire 手点为 `[T,10135,3]`；全部 finite；法线最大范数误差 MANO `1.79e-7`、Inspire `5.96e-8`。
- viewer `--check-only`：MANO `s1/airplane_fly_1` 与 Inspire `s1/airplane_lift` 均通过，KNN 1/4/8/16/32/64 和 mesh provenance 可读。
- 交互 viewer 已启动：`http://localhost:8100`（服务监听 `0.0.0.0:8100`，当前初始序列为 MANO `s1/airplane_fly_1`，界面可切换 6 条轨迹）。
- 首次生成中发现 Inspire 极小三角面的法线下限设置问题，已修正为 `1e-12` 并用同一 seed 完整重跑；错误中间目录已移至 `/tmp/variable_hand_trajectory_preview_20260909_bad_normals`，未进入最终预览目录。
- `git diff --check`：通过；针对本次 3 个变更路径执行 `audit_diff.py --worktree --check-links`：通过（10 个本地链接可导航）。完整 Task worktree 审计仍会包含此前用户遗留的其它未提交文件，故未将其误归入本条 activity；未运行训练或评估。

**回滚入口**

删除 [轨迹预览输出目录](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/)、[轨迹预览构建脚本](../../research/hand_region_sampling/build_trajectory_preview.py) 和 viewer 对变长手点的兼容改动，并将当前版本指针恢复到 `V1.2.14`；正式 cache 与既有 PLY 预览不受影响。

## 2026-09-09 16:27:23 +0800 — V1.2.14 核对 cache 点存储与 KNN16 计算位置

- activity_id: `ACT-20260909-162723-OBJECTINTERACTIONCM-CACHE-KNN-RUNTIME-DIAGNOSTIC`
- timestamp: `2026-09-09 16:27:23 +0800`
- modification_version: `V1.2.14`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前 cache 是保存点后前向计算 KNN16，还是 cache 预计算 KNN16/5 cm 阈值；本次只读检查 cache 文件、Dataset、模型和 V1.2.5 manifest。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（cache 保存每帧完整的 4096 点物体池、1538 点手表面及其法线；5 cm 只预计算了帧级 candidate active 标记，KNN16 与逐物体点的 5 cm edge mask 在前向动态计算）
- scope: 只读核对 V1.2.5 cache 的 geometry 文件、cache builder、ObjectInteractionCm Dataset 和 LocalHandInteraction；未修改代码、配置、cache、split、GT、checkpoint 或训练产物。

**文件与证据**

- `docs/current_versions.yaml`：更新 ObjectInteractionCm 当前版本指针为 `V1.2.14`。
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)：记录 `surface_points=1538`、`surface_seed=2024` 和已完成 cache conversion。
- [示例 Inspire geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s5_mouse_pass_1/geometry/manifest.json)：记录 `object_pool_points=4096`、`hand_points=1538`、`candidate_threshold_m=0.05`、`candidate_mask_shape=[167]`。
- [ObjectInteractionCm Dataset](../../dataset.py)：读取 `obj_points_pool_world.npy [T,4096,3]`、`hand_points_world.npy [T,1538,3]`，在 `__getitem__` 中动态随机抽取 1024 个物体点，并由当前/未来帧点计算 flow；同时用 full 4096 物体池动态计算 3 cm hand supervision mask。
- [ObjectInteractionCm model](../../model.py)：按配置的 `knn_k=16` 对运行时 1024 个物体点和全部有效手点做 KNN，再对 KNN 结果施加 `interaction_radius_m=0.05` 的逐 edge 阈值。
- [V1.2.5 active config](../../configs/active/dexplore_rl_v1_2_5.yaml)：明确 `num_obj_pool=4096`、`num_obj_points=1024`、`num_hand_points=1538`、`knn_k=16`、`interaction_radius_m=0.05`。

**原因**

当前 cache 的目标是保存可复用的几何和轨迹基础数据，不绑定某一次运行时的物体子采样或 KNN 配置。因而 KNN16 没有写入 cache；训练前向可以根据配置改变 K 值，而不必重建几何 cache。

**验证**

- V1.2.5 示例 cache 文件形状核对：`obj_points_pool_world.npy=(T,4096,3)`、`obj_normals_pool_world.npy=(T,4096,3)`、`hand_points_world.npy=(T,1538,3)`、`hand_normals_world.npy=(T,1538,3)`、`obj_candidate_mask_5cm.npy=(T,)`。
- Dataset `active_only=True` 只用一维 candidate 标记筛掉整帧；该标记不携带每个 object point 的 KNN 索引、距离或 16 邻居 mask。
- Model V1.2.5 前向顺序是：运行时选 1024 object points → KNN `k=16` → `distance <= 0.05 m` edge mask → attention/Cm；因此不是 cache 预存 KNN16。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未运行训练、评估或数据重建。

## 2026-09-09 13:24:57 +0800 — V1.2.13 按 mesh 表面积比例生成 MANO/Inspire 预览

- activity_id: `ACT-20260909-132457-OBJECTINTERACTIONCM-AREA-RATIO-UNIFORM-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 13:24:57 +0800`
- modification_version: `V1.2.13`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认使用按 mesh 表面积换算的点数进行均匀采样；MANO 固定 2048 点，Inspire 使用 10135 点。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `uniform_surface_ratio_2048_10135_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（MANO 2048 与 Inspire 10135 的完整 mesh 表面积均匀采样 PLY 已生成，实测点密度比为 `1.000055`；视觉适用性仍需用户检查，不构成训练效果结论）
- scope: 新增 `uniform_surface_ratio` 预览 profile；MANO/Inspire 均无 Region/link quota，分别按完整 mesh 面积比例采样；不覆盖既有 2048/2048 预览。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.13`。
- [预览脚本](../../research/hand_region_sampling/run.py)：支持 MANO 2048、Inspire 10135 的独立均匀采样点数和 manifest 记录。
- [实验 README](../../research/hand_region_sampling/README.md) 与 [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记 `uniform_surface_ratio` profile。
- [area-ratio 输出目录](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/)。
- [MANO 2048 PLY](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/mano_uniform_2048.ply) 与 [Inspire 10135 PLY](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/inspire_uniform_10135.ply)。
- [采样统计](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/run_manifest.json)。

**原因**

前一版两者都使用 2048 点，但完整 mesh 表面积相差约 4.94846 倍。为保持单位表面积点密度一致，本次固定 MANO 2048 点，并按 `2048 × A_Inspire/A_MANO = 10134.44` 向上取整为 Inspire 10135 点。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile uniform_surface_ratio --modification-version V1.2.13 --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909`：完成。
- MANO PLY 为 2048 点、Inspire PLY 为 10135 点；两者均 finite，统一灰色，最大单位法线误差分别约 `4.0e-8` 和 `4.4e-8`。
- 实测点密度：MANO `50250.20 points/m²`，Inspire `50252.98 points/m²`，密度比 `1.000055`。
- 同 seed 重跑后两份 PLY SHA-256 完全一致；未修改既有 2048/2048 PLY、正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未运行训练和评估。
- 回滚入口：删除 `uniform_surface_ratio` profile 的代码/文档改动、[area-ratio 输出目录](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/) 及版本指针更新；既有预览不需回滚。

## 2026-09-09 13:19:26 +0800 — V1.2.12 统计 MANO/Inspire mesh 表面积与等密度点数

- activity_id: `ACT-20260909-131926-OBJECTINTERACTIONCM-MESH-AREA-RATIO-DIAGNOSTIC`
- timestamp: `2026-09-09 13:19:26 +0800`
- modification_version: `V1.2.12`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问两份 canonical mesh 的表面积比例，以及 MANO 2048 点对应的 Inspire 等表面密度点数；本次只读统计已有 uniform-surface sampling summary。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（按当前 canonical 坐标解释，两份 mesh 面积比约 4.94846；相同表面点密度下 MANO 2048 点对应 Inspire 约 10134 点，取不低于该密度可用 10135 点）
- scope: 只读汇总 `uniform_surface_2048_preview_20260909/sampling_summary.json` 中 MANO/Inspire 的完整 mesh 三角面面积；未修改采样、PLY、cache 或训练数据。

**文件与证据**

- [uniform surface sampling summary](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/sampling_summary.json)：提供两种 canonical mesh 的总三角面面积和资产指纹。
- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.12`。

**原因**

用户希望按 mesh 表面积比例保持 MANO/Inspire 的点密度一致。计算使用 `N_inspire = 2048 × A_inspire / A_mano`，不改变已有 2048 点 PLY。

**验证**

- MANO 总表面积 `0.0407560583 m² = 407.5606 cm²`。
- Inspire 总表面积 `0.2016795812 m² = 2016.7958 cm²`。
- 面积比 `A_inspire/A_mano = 4.94845649`；换算点数 `2048 × ratio = 10134.4389`，最近整数为 `10134`，向上取整为 `10135`。
- 结果属于当前 canonical mesh/坐标口径的几何密度换算，不代表两种手的训练效果或跨形态泛化结论。

## 2026-09-09 13:10:34 +0800 — V1.2.11 MANO/Inspire 完整 mesh 均匀表面采样预览

- activity_id: `ACT-20260909-131034-OBJECTINTERACTIONCM-UNIFORM-SURFACE-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 13:10:34 +0800`
- modification_version: `V1.2.11`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求重新在 MANO/Inspire 各自完整 mesh 上均匀采样 2048 点并生成 PLY；本次将“均匀”固定为单位表面积采样概率相同，不使用 Region、link、segment 或 tip quota。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `uniform_surface_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（两份完整 mesh 表面积均匀采样 PLY 已生成，点数、finite、法线、无配额和确定性检查通过；视觉适用性仍由用户检查，不构成正式训练效果结论）
- scope: 在现有预览入口新增 `uniform_surface` profile；两种手各自全局按三角形面积选面并在面内作均匀 barycentric 采样，总数 2048，统一灰色显示；不覆盖旧预览。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.11`。
- [预览脚本](../../research/hand_region_sampling/run.py)：新增完整 mesh 表面积均匀采样、灰色 PLY 和 sampling method manifest 字段。
- [实验 README](../../research/hand_region_sampling/README.md) 与 [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记 `uniform_surface` profile 及无配额合同。
- [uniform surface 输出目录](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/)。
- [MANO uniform PLY](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/mano_uniform_2048.ply) 与 [Inspire uniform PLY](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/inspire_uniform_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/run_manifest.json)。

**原因**

前序 Region、tip-heavy 和 per-link 预览都显式改变了局部点密度，无法直接观察完整 mesh 在统一面积密度下的自然分布。本次取消所有语义配额，MANO 与 Inspire 各自在完整 canonical mesh 上独立生成 2048 点表面均匀基线。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile uniform_surface --modification-version V1.2.11 --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/uniform_surface_2048_preview_20260909`：完成。
- 两份 PLY 均为 2048 点、全字段 finite、RGB 全部为 `(190,190,190)`，最大单位法线误差小于 `4.4e-8`；summary 中 `region_quotas=null`、`stratum_quotas=null`。
- Inspire 按 source visual ID 的自然计数为 `1491,14,68,44,37,61,34,57,52,67,40,56,27`；`hand_base_link` 占 1491 点来自其约 73.8% 的总表面积，是全局表面积均匀采样的预期结果，不是额外权重。
- 同 seed 重跑后 MANO/Inspire PLY SHA-256 完全一致；balanced profile 重跑也与 V1.2.6 基线 PLY 哈希一致。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未修改正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint，未运行训练和评估。
- 回滚入口：删除 `uniform_surface` profile 的代码/文档改动、[uniform surface 输出目录](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/) 及版本指针更新；既有 balanced/tip-heavy/link-stratified 输出不需回滚。

## 2026-09-09 11:50:27 +0800 — V1.2.10 核对 MANO mesh 与空带来源

- activity_id: `ACT-20260909-115027-OBJECTINTERACTIONCM-MANO-MESH-STRUCTURE-DIAGNOSTIC`
- timestamp: `2026-09-09 11:50:27 +0800`
- modification_version: `V1.2.10`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 MANO 是否存在对应 link mesh，以及 MANO PLY 空带是否由形态缺少 mesh 造成；本次只读检查模型资产字段和现有采样代码。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（MANO 每个形态由同一 778 顶点/1538 三角面拓扑经 `shapedirs` 和蒙皮权重变形得到整体 mesh；不存在 Inspire 式独立 link mesh。空带属于采样/分区覆盖问题，不是形态没有 mesh）
- scope: 只读核对 `MANO_RIGHT.pkl` 的 `v_template`、`shapedirs`、`J_regressor`、`weights`、`f` 字段及 GRAB/MANO 当前 face-center 与预览采样路径；未修改代码、配置、cache 或产物。

**文件与证据**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.10`。
- [MANO 资产](../../../../../dataset/arctic/data/body_models/mano/MANO_RIGHT.pkl)：包含 `v_template (778,3)`、`shapedirs (778,3,10)`、`posedirs (778,3,135)`、`J_regressor (16,778)`、`weights (778,16)` 和 `f (1538,3)`；形状和姿态共享同一 mesh topology。
- [GRAB MANO 预处理](../../../../../process/GRAB/raw.py)：正式手点以每帧 MANO 变形顶点和固定 `faces` 的 face center 生成，保留跨帧 face 对应。
- [MANO 预览脚本](../../research/hand_region_sampling/run.py)：预览从同一 MANO faces 构造三角面池；`link_stratified` 对应的是 joint-segment 语义，不是 URDF link mesh。

**原因**

MANO 的 `β` 只改变同一模板 mesh 的顶点位置，不能提供 `thumb_distal`、`index_tip` 之类的独立刚体网格。当前 MANO 空带更可能来自面中心/有放回采样、按最近关节和 PCA 末端划分造成的空间覆盖不足；若要修复，应基于 MANO 关节位置、顶点 skinning weights 或 geodesic 邻域定义 joint-centered 区域，而不是寻找不存在的 link mesh。

**验证**

- 使用 NumPy 兼容 shim 成功读取 `MANO_RIGHT.pkl` 字段和形状；未发现独立 link mesh 字段。
- 只读检查 `process/GRAB/raw.py` 与预览脚本的 face/triangle 采样路径；未运行训练、评估或数据重建。
- 结论不代表新的采样方案已经验证，仅确认 MANO 几何表示和空带问题的来源类别。

## 2026-09-09 11:43:35 +0800 — V1.2.9 严格 per-link mesh 采样预览

- activity_id: `ACT-20260909-114335-OBJECTINTERACTIONCM-LINK-STRATIFIED-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 11:43:35 +0800`
- modification_version: `V1.2.9`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认改为每个有 mesh 的 Inspire link 独立采样，并为 MANO 使用对应 joint-segment 分层；总点数保持 2048，不覆盖既有预览和正式 cache。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `link_stratified_v2_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（已验证 Inspire 每个非空 visual link 获得独立 quota 且点来自该 link 自己的 mesh；但按 URDF `*_tip` frame 统计，小指仍无 15 mm 内点，局部 tip 空带是否消失需用户查看 PLY，不能仅据此宣称已解决）
- scope: 在 V1.2.6 预览入口新增严格 `link_stratified` profile；Inspire 按 13 个非空 visual link 分层，MANO 按 11 个 joint-segment 分层；不改变 balanced/tip_heavy 基线行为。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.9`。
- [预览脚本](../../research/hand_region_sampling/run.py)：新增 link/segment strata、独立面积加权采样和严格 link quota 映射。
- [实验 README](../../research/hand_region_sampling/README.md)：记录 `link_stratified` profile 及其保护边界。
- [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记 link-stratified quota profile。
- [link-stratified 输出目录](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/)。
- [MANO link-stratified PLY](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/mano_weighted_2048.ply) 与 [Inspire link-stratified PLY](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/inspire_weighted_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/run_manifest.json)。

**原因**

之前的 Region/tip-heavy 方案虽然读取了 link mesh，但同一根手指的多个 link 仍共享 Region quota，不能保证每个 link 都有点。本次将 hand base 固定为 512 点；五个末端父 link（`thumb_distal`、`index_intermediate`、`middle_intermediate`、`ring_intermediate`、`pinky_intermediate`）合计 1024 点；其余七个 finger link 合计 512 点。每个 link 内仍按自身三角面面积加权并作 barycentric surface sampling。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile link_stratified --modification-version V1.2.9 --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909`：完成。
- MANO/Inspire PLY 均为 2048 点，点和法线 finite，最大单位法线误差约 `4.0e-8`。
- Inspire `source_visual_id` 计数严格为 `512,74,73,73,205,73,205,73,205,73,205,73,204`，对应 13 个非空 visual link；所有点均由对应 link 的 mesh 三角面抽取。
- MANO segment Region 计数为 `512,103,205,103,205,102,205,102,205,102,204`。
- 基线 `balanced` profile 重新生成后 MANO/Inspire PLY SHA-256 与 V1.2.6 基线完全一致；未改变既有 profile。
- Inspire tip frame 15 mm 内点数为 thumb `56`、index `73`、middle `17`、ring `68`、pinky `0`；说明 per-link quota 不等于 tip-frame 邻域覆盖，后续若仍有局部空带需另做 joint-centered/collar 规则。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未修改正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint，未运行训练和评估。
- 回滚入口：删除 `link_stratified` 代码/文档改动、[link-stratified 输出目录](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/) 及版本指针更新，即可恢复既有 V1.2.6 预览行为；balanced/tip-heavy 输出不需回滚。

## 2026-09-09 11:12:46 +0800 — V1.2.7 tip-heavy 2048 点预览对比

- activity_id: `ACT-20260909-111246-OBJECTINTERACTIONCM-TIP-HEAVY-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 11:12:46 +0800`
- modification_version: `V1.2.7`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认保持总点数 2048，只增加 tip 配额以检查指尖关节处空带是否由比例不足造成；不覆盖基线、不接入正式 cache。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `tip_heavy_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（tip 配额翻倍后拇指/食指/中指/无名指 tip 关节附近覆盖明显增加，但小指因 `pinky_tip` frame 与末端 mesh 不对齐仍无 15 mm 内点；仅增加比例不能完全解决该问题）
- scope: 在现有 V1.2.6 预览脚本中新增 tip-heavy 配额 profile；总点数、三角面采样、Region 几何划分、seed、姿态和正式 cache/训练合同保持不变。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.7`。
- [预览脚本](../../research/hand_region_sampling/run.py)：新增 `balanced`/`tip_heavy` quota profile，并将 quota 显式传入采样与统计函数。
- [实验 README](../../research/hand_region_sampling/README.md)：记录 tip-heavy 对照的变量和命令。
- [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记默认 profile 与 tip-heavy profile。
- [tip-heavy 输出目录](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/)。
- [MANO tip-heavy PLY](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/mano_weighted_2048.ply) 与 [Inspire tip-heavy PLY](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/inspire_weighted_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/run_manifest.json)。

**原因**

当前 balanced profile 每根手指约 102 个 tip 点，且这些点分散在末端 20% 表面。为隔离“tip 比例是否不足”这一变量，本次保持所有几何和采样规则不变，只将 `palm/body/tip` 从 `512/1024/512` 调整为 `512/512/1024`，即每根手指 tip 约 205 点。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile tip_heavy --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/tip_heavy_2048_preview_20260909`：完成。
- MANO/Inspire PLY 均为 2048 点，Region 计数为 `512,103,205,103,205,102,205,102,205,102,204`，点和法线 finite，最大单位法线误差约 `4.2e-8`。
- 同 seed 重跑后两份 PLY 的 SHA-256 完全一致；统计 JSON 仅因输出目录绝对路径不同而不同。
- Inspire tip frame 15 mm 邻域计数（balanced → tip-heavy）：thumb `55→119`、index `95→183`、middle `50→83`、ring `93→179`、pinky `0→0`；说明增加配额对前四指有效，但不能修复小指 frame/mesh 对齐问题。
- `git diff --check`：通过；未修改正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint，未运行训练和评估。
- 回滚入口：删除新增 profile、文档改动和 [tip-heavy 输出目录](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/)，即可恢复 V1.2.6 预览行为；基线输出不受影响。

## 2026-09-09 10:26:31 +0800 — V1.2.6 完成 MANO/Inspire 2048 点 Region 加权采样预览

- activity_id: `ACT-20260909-102631-OBJECTINTERACTIONCM-REGION-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 10:26:31 +0800`
- modification_version: `V1.2.6`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认按默认方案执行：MANO/Inspire 共用 11 Region、总点数 2048、`palm/body/tip=25%/50%/25%`，Region 内保留面积加权。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的 ObjectInteractionCm 转换器/配置/查看器、CmDecoderv2 改动及其它用户修改）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `region_weighted_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（预览脚本、两份 2048 点 PLY、Region 配额、法线和确定性检查通过；PLY 的视觉适用性等待用户人工确认，正式训练效果尚未评估）
- scope: 新增 Task-local 研究预览脚本和定义，生成 MANO/Inspire canonical PLY；不修改 V1.2.5 正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint。

**文件与产物**

- [V1.2.6 计划](../plan/V1.2.6.md)：锁定 11 Region、2048 配额、20% 末端 tip、Region 内面积加权和预览保护边界。
- [Task 文档入口](../README.md) 与 [当前版本指针](../../../../../docs/current_versions.yaml)：加入/指向 V1.2.6 预览计划；未改其它 Task 版本。
- [预览脚本](../../research/hand_region_sampling/run.py)：实现 MANO/Inspire canonical triangle pool、Region 划分、固定配额采样和 PLY/manifest 输出。
- [实验定义](../../research/hand_region_sampling/experiment.yaml) 与 [实验 README](../../research/hand_region_sampling/README.md)：声明 exploratory、只输出预览、不接入正式 cache。
- [MANO 2048 点 PLY](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/mano_weighted_2048.ply)。
- [Inspire 2048 点 PLY](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/inspire_weighted_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/run_manifest.json)。

**原因**

现有 V1.2.5 的全局面积加权 1538 点使 Inspire 指尖覆盖不足。本次先用独立预览验证“固定 2048 点、显式 palm/body/tip 配额、Region 内仍按面积加权”的视觉分布，再决定是否改变正式数据语义。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/region_weighted_2048_preview_20260909`：完成，生成 MANO/Inspire 两份 PLY、统计和 run manifest。
- PLY 读取校验：两份均为 2048 vertices、11 Region 计数严格符合预设配额、点和法线 finite，最大单位法线误差小于 `4e-8`。
- 采样确定性校验：同一 seed=2024 下 MANO/Inspire 两次生成的点数组逐元素一致。
- 初次运行发现的 `smplx/chumpy` 与新 NumPy 别名兼容问题已通过预览入口本地 shim 解决；未改动生产模块。
- `git diff --check` 与 `audit_diff.py --staged --check-links`：通过；未运行训练、评估或全量数据处理。
- 回滚入口：删除新增的 [预览脚本](../../research/hand_region_sampling/run.py)、[实验定义](../../research/hand_region_sampling/experiment.yaml)、[计划](../plan/V1.2.6.md)、README 和对应 `output/region_weighted_2048_preview_20260909/`；V1.2.5 cache/config/checkpoint 不需回滚。

## 2026-09-09 00:08:25 +0800 — V1.2.5 核对 Inspire 三角面中心点数量

- activity_id: `ACT-20260909-000825-OBJECTINTERACTIONCM-INSPIRE-TRIANGLE-CENTER-COUNT`
- timestamp: `2026-09-09 00:08:25 +0800`
- modification_version: `V1.2.5`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户追问 1538 点总预算对指尖稀疏的影响，以及每个三角面中心作为点时的数量；本次只读复现正式 URDF mesh 的有效三角面计数。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅追加本诊断记录；保留工作树中已有改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（确认当前 1538 总预算显著压缩了三角面候选；“每面中心”会产生约 24.7 万点，但其三角剖分密度并不等于均匀几何采样）
- scope: 只读统计当前 Inspire URDF 的 visual mesh 三角面、退化面和每 link 分布；未修改代码、配置、cache、split、GT、训练或评估产物。

**文件与证据**

- [cache builder](../../tools/data/build_dexplore_rl_cache.py)：验证当前实现将所有有效三角面合并后按面积有放回抽取固定 1538 点。
- [V1.2.5 正式 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)：确认正式 cache 使用 `surface_points=1538`、`surface_seed=2024`。
- [本活动记录](activity_log.md)：登记本次只读诊断和保护边界。

**原因**

13 个 visual mesh 共 247,412 个三角面，其中 8 个退化面被当前代码过滤，得到 247,404 个有效三角面。因此保留每个三角面中心会产生 `247,404×3` 的点坐标数组，约为当前 1538 点的 160.9 倍。全量中心点还会继承 STL 三角剖分密度，不能直接视为均匀表面采样。

**验证**

- 有效三角面按大区统计：`hand_base=68,832`、`thumb=38,624`、`index=35,248`、`middle=34,006`、`ring=35,248`、`pinky=35,446`。
- 若保留所有面中心，五个末端 mesh 的点数分别为 `15,718/14,846/13,604/14,846/15,044`（拇/食/中/无名/小指），几何最末 10 mm 分别约有 `7,328/5,053/5,039/5,085/5,803` 个面中心；这说明全量中心点会显著改善末端覆盖，但也会带来约 2.97 MB 的单帧 float32 坐标，仅计算量和存储量就不再等同于当前训练合同。
- 未运行训练、评估或数据处理，未生成或修改 cache；回滚入口仅为删除本条 activity。

## 2026-09-08 23:58:38 +0800 — V1.2.5 为 Viser 查看器增加未来帧叠加

- activity_id: `ACT-20260908-235838-OBJECTINTERACTIONCM-VISER-FUTURE-OVERLAY`
- timestamp: `2026-09-08 23:58:38 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确确认未来步长 `delta` 范围为 0–10，0 表示不显示未来，其余按已协商方案叠加 `t+delta` 的手和物体；改动仅限 Task-local 只读可视化。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的鼠标 GUI、未来点云/mesh 场景节点及只读检查输出；不改变 cache、schema、GT、split、坐标系、采样、KNN、距离定义、训练或评估实现。
- conclusion: `SUPPORTED`（未来帧索引、末帧夹取及 MANO/Inspire-RL 服务端渲染通过工程验证；未进行浏览器端人工点击与主观观感回归，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：在“播放”区域加入 `未来 Δ（cache 帧）` 鼠标滑块，范围为 0–10；0 隐藏未来层，1–10 叠加 `min(t+Δ,T-1)` 的手和物体。未来物体使用绿色、未来手使用紫色，不绘制连线；未来点云和 mesh 分别服从既有 `点云显示`、`Mesh 显示` 控件，当前帧的累计距离阈值、最近距离和物体→手 KNN 语义保持不变；新增 `--future-delta` 启动/检查参数。
- [本活动记录](activity_log.md)：登记审批、边界、验证和回滚入口。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

需要在同一视图对比当前状态和指定未来 cache 帧的整体手物运动。实现复用训练 index 中同一条轨迹的逐帧点云与 mesh pose，不插值、不连接轨迹、不计算未来层的距离/KNN；在序列尾部显式夹到最后一帧并在状态区提示，避免越界或循环到序列开头。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py` 与 `_future_frame`/parser 断言：通过；确认 `Δ=0/1/6/10`、末帧夹取及参数范围。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --frame 0 --future-delta 3`：通过；MANO 当前 cache 帧 0 映射未来 cache 帧 3、原始帧 12，未夹取。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split val --sequence s1/torussmall_lift --frame 488 --future-delta 10`：通过；Inspire-RL 的 489 帧序列在末帧正确夹到 cache 帧 488、原始帧 1952，并报告 `future_clamped=true`。
- `timeout 12s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --frame 0 --future-delta 3 --point-display both --mesh-display both --host 127.0.0.1 --port 8134 --fps 1`：Viser 使用当前 V1.2.5 index 完成 MANO 当前/未来点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `timeout 15s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split val --sequence s1/torussmall_lift --frame 0 --future-delta 10 --point-display both --mesh-display both --host 127.0.0.1 --port 8135 --fps 1`：Viser 完成 Inspire-RL 当前/未来点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `timeout 7s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --future-delta 0 --point-display both --mesh-display both --host 127.0.0.1 --port 8136 --fps 1`：`Δ=0` 路径完成初始渲染，未来点云不可见且未来 mesh 不加载；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 7 个变更路径一致，4 个本地链接可导航。
- 回滚入口：仅恢复 [查看器](../../visualize_grab.py) 中未来帧 helper、GUI、四个未来 scene group 和 `--future-delta` 参数，并删除本条 activity；数据/cache 无需回滚。

## 2026-09-08 23:56:22 +0800 — V1.2.5 诊断 Inspire 指尖表面点稀疏

- activity_id: `ACT-20260908-235622-OBJECTINTERACTIONCM-INSPIRE-TIP-SAMPLING-DIAGNOSTIC`
- timestamp: `2026-09-08 23:56:22 +0800`
- modification_version: `V1.2.5`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练数据中 Inspire 手点的采样方式，并反馈可视化中指尖手点很少；本次仅只读核对 cache builder、URDF visual mesh、正式 cache manifest，并复现固定采样的区域计数。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅追加本诊断记录；保留工作树中已有的 ObjectInteractionCm 转换器、配置、查看器、训练记录及 CmDecoderv2 改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（确认当前 Inspire 固定表面采样确实造成指尖低覆盖；尚未做重采样或性能消融，不能据此断言它对指标的因果影响）
- scope: 只读检查 V1.2.5 Inspire 手点的 URDF visual surface sampling、固定点随 FK 变换的实现和 seed=2024 的逐 link/末端区域计数；未修改代码、配置、cache、split、GT、训练或评估产物。

**文件与证据**

- [cache builder](../../tools/data/build_dexplore_rl_cache.py)：从 13 个 URDF visual mesh 合并全部有效三角形，按三角形面积全局有放回抽取 1538 个三角形，再作重心采样；固定样本只在每帧随各 link FK 变换。
- [V1.2.5 正式 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json) 与 [示例 Inspire geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)：确认 `surface_seed=2024`、`surface_points=1538`、`method=area_weighted_triangle_barycentric`。
- [V1.2.5 训练配置](../../configs/active/dexplore_rl_v1_2_5.yaml)：确认运行时手点数 1538、物体→手 KNN=16、交互半径 5 cm。
- [本活动记录](activity_log.md)：登记本次只读诊断、定量结果和保护边界。

**原因**

当前策略没有 per-link 最小配额、指尖加权或全局 FPS。`hand_base_link` visual mesh 的面积为 `1488.759 cm²`，占全部 visual surface 的 `73.818%`，因此固定 seed 实际将 1538 点中的 1109 点分配给掌/腕 base，五个含指尖表面的末端 mesh 合计只有 139 点。用户看到的指尖稀疏不是单帧渲染偶发，而是全部 Inspire 序列共享的固定采样布局。

**验证**

- 使用正式 builder 的 `InspireUrdfModel.surface_samples(1538, 2024)` 复现采样：掌/腕 base 1109 点；拇指各 link 合计 126 点（末端 28），食指 74（末端 27），中指 86（末端 40），无名指 78（末端 26），小指 65（末端 18），总计 1538。
- 沿各末端 mesh 轴向统计几何最末 10 mm：拇/食/中/无名/小指分别仅 `4/5/9/5/4` 点；该计数支持“指尖覆盖稀疏”，但不是模型性能消融。
- URDF 的五个 `*_tip` link 均为空 link、没有 visual mesh，采样器不会为 tip link 单独留点；另发现 `pinky_tip` fixed frame 与末端 mesh 的局部轴向明显不一致，但该 frame 未参与当前 surface sampling，因而不是本次稀疏的直接原因。
- 未运行训练、评估或数据处理，未生成或修改 cache；回滚入口仅为删除本条 activity。

## 2026-09-08 22:13:29 +0800 — V1.2.5 缩小点云 sprite 并改用 float32 圆点渲染

- activity_id: `ACT-20260908-221329-OBJECTINTERACTIONCM-VISER-FINE-POINT-RENDERING`
- timestamp: `2026-09-08 22:13:29 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 在只读诊断确认 Inspire 小物体的 4096 点密度、1 mm 最小 sprite 和 Viser 默认 float16 量化后，Agent 提议将滑块下限降至 0.1 mm 并使用 `float32/circle/flat`，用户明确回复“可以，你继续吧”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的 Viser 点云显示参数；不改变点云数组、采样数量、坐标系、KNN、距离、mesh、训练或评估合同。
- conclusion: `SUPPORTED`（静态检查和 Inspire-RL 小物体服务端 smoke；未进行浏览器端主观观感回归，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：手点和物体点滑块由最小 `0.001 m`/步长 `0.001 m` 改为最小 `0.0001 m`/步长 `0.0001 m`；两类点云均显式使用 `precision="float32"`、`point_shape="circle"`、`point_shading="flat"`，避免世界坐标约 1 m 时的 float16 毫米级量化和方形渐变 sprite 造成视觉膨胀。
- [本活动记录](activity_log.md)：登记审批、验证和回滚入口。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

V1.2.5 的 Inspire 小物体可小至约 40 mm，而 4096 点的典型间距约 0.44 mm；旧的最小 1 mm sprite 会明显重叠。Viser 1.0.30 默认还会把点坐标转换为 float16，在当前约 1 m 的世界坐标处量化粒度接近 1 mm。此次只收细渲染参数，不通过缩放坐标或改写采样来掩盖显示问题。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过；静态核对两个 slider 均为 `min=0.0001, step=0.0001`，两个 `add_point_cloud` 均传入 `float32/circle/flat`。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split val --sequence s1/torussmall_lift --point-size 0.0001 --point-display both --mesh-display off --host 127.0.0.1 --port 8133 --fps 1`：当前 V1.2.5 index 的 Inspire-RL 小物体轨迹以 0.1 mm 点大小完成 Viser 初始渲染；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；未生成或修改数据/cache/运行产物。
- 回滚入口：仅恢复 [查看器](../../visualize_grab.py) 的两个 slider 范围和两处 `add_point_cloud` 渲染参数；数据/cache 无需回滚。

## 2026-09-08 22:02:49 +0800 — V1.2.5 诊断 Inspire 点云看起来偏大的原因

- activity_id: `ACT-20260908-220249-OBJECTINTERACTIONCM-VISER-POINT-SIZE-DIAGNOSTIC`
- timestamp: `2026-09-08 22:02:49 +0800`
- modification_version: `V1.2.5`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问查看器物体点是否为采样点，并反馈 Inspire 视图在最小点大小下仍显得偏大；本条仅做只读数组、尺度和 Viser 渲染语义核对。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（只追加本诊断记录；保留工作树已有的训练、转换器、配置和查看器改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 只读检查 [近期 V1.2.5 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)、indexed geometry、缓存构建器和本地 Viser 1.0.30 的 `point_size` 实现；未修改数据、代码或渲染参数。
- conclusion: `SUPPORTED`（采样来源和单位诊断得到工程证据；尚未实施点大小修正，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：只读核对其加载完整 `obj_points_pool_world.npy`、点数和 `point_size` 传递。
- [V1.2.5 训练 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json) 与 [示例 geometry](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)：核对采样点池 shape/dtype、来源 manifest 和坐标单位。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

需要区分“数据点本身过大”和“渲染点 sprite/点密度相对物体尺寸显得过大”。本次检查同时对照 cache builder 的输入输出和 Viser 的本地渲染实现，避免未经证据就修改坐标或采样语义。

**验证**

- `obj_points_pool_world.npy` 在 MANO 和 Inspire-RL 序列均为 `[T,4096,3]`、`float32`，示例帧 4096 个点全部唯一；不是直接使用 mesh 顶点。近期 V1.2.5 index 的运行时模型点数为 1024，但查看器故意显示完整 4096 点池。
- [cache builder](../../tools/data/build_dexplore_rl_cache.py) 先读取父 cache 的 `obj_points_world.npy [T,4096,3]`，再按当前物体姿态写出 `obj_points_pool_world.npy`；Inspire 手点则由 1538 个固定面积加权表面样本生成。两者均为采样点云，不是原始 mesh 顶点。
- 对 V1.2.5 Inspire 示例核对：普通 `airplane` 物体尺寸约 `138×155×44 mm`；`cubesmall`/`torussmall` 等小物体约 `40 mm` 尺寸，4096 点的典型最近邻间距约 `0.44 mm`（torussmall）。因此查看器最小 `point_size=0.001 m` 仍是 1 mm，在小物体上会发生明显 sprite 重叠。
- 本地 Viser 1.0.30 的 `add_point_cloud` 文档和客户端 shader 均按场景单位解释 `point_size`，查看器当前物体/手点 slider 下限为 `0.001 m`，且默认 `point_shape="square"`、`point_shading="gradient"`；这会使密集 Inspire 点云看起来比 MANO 大，但不表示坐标单位错误。
- 未修改 `visualize_grab.py` 或任何 cache；若要改善观感，后续可在用户确认后把最小点大小降到 `0.0001 m`、改用 `circle/flat`，并可选择显示 1024 点可视化子集而保持 KNN 在完整池上计算。

## 2026-09-08 21:37:48 +0800 — V1.2.5 为点云与 Mesh 增加独立鼠标显示切换

- activity_id: `ACT-20260908-213748-OBJECTINTERACTIONCM-VISER-GEOMETRY-TOGGLES`
- timestamp: `2026-09-08 21:37:48 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认采用两个独立鼠标下拉控件，分别切换点云和 Mesh 的关闭、仅物体、仅手、物体+手显示状态。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的显示 GUI 与点云 handle 可见性；不改变数据、KNN 定义、距离计算、mesh 几何、训练或评估实现。
- conclusion: `SUPPORTED`（工程 smoke 证据；未进行浏览器端人工点击回归，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：在同一“点云与 Mesh 显示”区域加入独立的 `点云显示` 和 `Mesh 显示` 下拉框；二者均支持关闭、仅物体、仅手和物体+手。点云切换只更新既有 point-cloud handle 的 `visible`，Mesh 仍按需加载，不重建点云数据或改变着色结果；同时新增可选的 `--point-display off|object|hand|both` 启动参数。
- [本活动记录](activity_log.md)：登记显示组合、验证和保护边界。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

此前点云始终显示，只有 Mesh 可以切换，无法快速在纯点云、纯 Mesh 和叠加视图之间比较。两个独立鼠标控件让物体和手的点云/mesh 可自由组合，同时保持轨迹、播放、距离阈值和 KNN 控件不变。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过；`--help` 显示 `--point-display {off,object,hand,both}` 与既有 mesh 参数。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split train --sequence s1/airplane_fly_1 --point-display off --mesh-display both --host 127.0.0.1 --port 8130 --fps 1`：Viser 启动并完成“关闭点云、显示物体+手 Mesh”的 MANO 初始渲染；timeout 主动退出，退出码 124。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split val --sequence s1/banana_lift --point-display hand --mesh-display off --host 127.0.0.1 --port 8131 --fps 1`：Viser 启动并完成“仅手点、关闭 Mesh”的 Inspire-RL 初始渲染；timeout 主动退出，退出码 124。
- `timeout 8s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --point-display off --mesh-display both --host 127.0.0.1 --port 8132 --fps 1`：使用当前 V1.2.5 正式训练 index（630 条序列）再次通过纯 Mesh 初始渲染；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；未生成或修改数据/cache/运行产物。

## 2026-09-08 20:36:03 +0800 — V1.2.5 为 Viser 查看器增加物体→手 KNN 高亮

- activity_id: `ACT-20260908-203603-OBJECTINTERACTIONCM-VISER-KNN-HIGHLIGHT`
- timestamp: `2026-09-08 20:36:03 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求增加 KNN 显示，提供 `1/4/8/16/32/64` 档位；该改动只增加 task-local 可视化，不改变训练、GT、cache、split 或指标合同。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有的训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的鼠标 GUI 和只读 KNN 诊断；不绘制连线、不写入数据/cache/output。
- conclusion: `SUPPORTED`（工程与数据语义 smoke 证据；不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：新增 `物体→手 KNN` 下拉控件，档位为关闭、`n=1`、`n=4`、`n=8`、`n=16`、`n=32`、`n=64`；每帧对完整 4096 点物体池查询最近 n 个右手点，并将被任一物体点选中的手点取并集后以黄色高亮。距离阈值仍为红色，KNN 黄色在重叠点上优先显示；不创建线段。
- [本活动记录](activity_log.md)：登记 KNN 方向、颜色语义、验证和保护边界。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

用户需要快速查看“物体附近的整体手点区域”，而不是逐条显示 KNN 连线。实现与当前 Cm 模型的 KNN 方向保持一致：物体点作为 query、手点作为候选邻居，对各物体点的最近 n 个手点取并集并着色；因此 n=1 表示每个物体点的第一近手点的并集，而非强行只保留一个手点。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split train --sequence s1/airplane_fly_1`：通过；KNN 并集手点数依次为 `n=1:1`、`4:6`、`8:10`、`16:22`、`32:37`、`64:73`，随 n 单调增加。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split val --sequence s1/banana_lift`：通过；KNN 并集手点数依次为 `n=1:1`、`4:4`、`8:9`、`16:18`、`32:33`、`64:70`，确认 Inspire-RL 轨迹同样使用物体→手方向。
- `timeout 12s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split train --sequence s1/airplane_fly_1 --host 127.0.0.1 --port 8129 --fps 1`：Viser HTTP/WebSocket 启动并完成初始点云渲染；timeout 主动退出，退出码 124。KNN 控件通过同一 GUI callback 接入，未绘制连线。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；未生成或修改数据/cache/运行产物。

## 2026-09-08 09:46:19 +0800 — V1.2.5 修正轨迹、KNN=16 的 Cm 正式训练完成

- activity_id: `ACT-20260908-094619-OICM-CORRECTED-KNN16-TRAIN-COMPLETED`
- timestamp: `2026-09-08 09:46:19 +0800`
- modification_version: `V1.2.5`
- type: `experiment / operation`
- change_level: `L0`（既有已批准正式 run 的终态核对与记录，不改变科研变量或运行产物）
- approval: `user-approved`
- approval_basis: 用户已批准 V1.2.5 修正轨迹、KNN=16 的正式训练；本次用户询问当前状态，只读核对已结束的既有 run 并收口终态记录。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 查看器和 V1.2.5 未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 只读核对 V1.2.5 正式 run 的进程、metrics、train log 与 checkpoint 元数据，并更新 Task activity/experiment 终态；不修改任何运行产物。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606`
- run_status: `COMPLETED`
- command: `CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml --distributed`
- last_step: `202300`
- last_epoch: `262`
- duration: `06:30:21`
- best_metric: `val/obj/flow_epe_mm=6.4225044410`（epoch 143 / step 110682，MANO `7.2434972881 mm`，RL-Inspire `5.6015115938 mm`）
- final_metric: `val/obj/flow_epe_mm=6.7061753591`（epoch 262 / step 202300，MANO `7.7338374220 mm`，RL-Inspire `5.6785132961 mm`）
- conclusion: `SUPPORTED`（修正数据与 KNN=16 的正式训练稳定完成）；下游 CmDecoderV2/rollout 效果仍为 `INCONCLUSIVE`

**产物与证据**

- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/config.json)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/metrics.jsonl) 与 [train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/train.log)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/best.pt)：epoch 143 / step 110682，SHA256 `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。
- [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/latest.pt)：epoch 262 / step 202300，SHA256 `f8bcd466021eebf3b5ee9f10e0e36091eb38d60a4a469feb5b381f1ab1db1899`。
- [实验记录](experiment_log.md) 与 [执行计划](../plan/V1.2.5.md)

**原因**

用户询问正式训练进展。进程已自然退出，需要区分 best 与 final validation，并将启动条目的运行中状态收口为唯一终态证据，避免后续 decoder 错误选择最后一轮 checkpoint。

**验证**

- `ps` 未发现该 run 的 torchrun、训练 rank 或 DataLoader 进程；GPU 0/1/2 利用率均为 0%。
- `train.log` 末尾为 `Training finished at step 202300 in 06:30:21.`；未检出 traceback、exception、NaN、OOM、NCCL failure 或 killed 记录。
- 解析 `metrics.jsonl` 共 262 次 validation：最小等权 source object EPE 位于 epoch 143 / step 110682；最终 validation 位于 epoch 262 / step 202300。
- 直接读取 best/latest checkpoint 元数据，确认 step、epoch 和共同保存的 `best_metric=6.422504440981543`；SHA256 已记录。
- 未修改 checkpoint、cache、配置或训练输出；旧错误轨迹 run 和所有既有用户改动均保留。

## 2026-09-07 23:58:32 +0800 — V1.2.5 full cache、KNN=16 smoke 与正式重训启动

- activity_id: `ACT-20260907-235832-OICM-CORRECTED-KNN16-FULL-TRAIN-START`
- timestamp: `2026-09-07 23:58:32 +0800`
- modification_version: `V1.2.5`
- type: `data / code / experiment / operation / documentation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认使用 630 条修正相交集重新训练 Cm，并追加要求 `KNN=16`；此前已说明独立 cache、scale、smoke、正式三卡训练和旧产物保护边界。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 查看器及旧 cache/output/checkpoint）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 新建修正轨迹 cache/index/scale、V1.2.5 配置和新训练 run；转换器只补充真实 RL root/version provenance；不修改旧训练变量之外的模型/loss、公共 `src/base`、旧数据或旧 checkpoint。
- cache_run_id: `oicm-dexplore-rl-full-20260907-231239`
- cache_run_status: `COMPLETED`
- smoke_run_id: `object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511`
- smoke_run_status: `COMPLETED`
- run_id: `object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606`
- run_status: `RUNNING`
- process: torchrun PID `2889418`，GPU `0,1,2`，DDP world size 3
- last_step: `4100`（人工检查于 2026-09-08 00:04:23 +0800；最近完整 checkpoint 为 step 3870 / epoch 5）
- last_epoch: `6`（运行中）
- best_metric: `val/obj/flow_epe_mm=7.9856659836`（epoch 4 / step 3096）
- conclusion: `INCONCLUSIVE`（正式训练运行中；cache 和 smoke 工程证据为 `SUPPORTED`）

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml) 与 [Task 入口](../README.md)
- [执行计划](../plan/V1.2.5.md)、[正式配置](../../configs/active/dexplore_rl_v1_2_5.yaml)、[smoke 配置](../../configs/active/dexplore_rl_v1_2_5_smoke.yaml) 与 [cache 转换器](../../tools/data/build_dexplore_rl_cache.py)
- [full cache 目录](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/)、[cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)、[index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)、[validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/validation_summary.json) 与 [KNN=16 scale](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/scales_train_v1_2_5.json)
- [smoke 运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/)、[smoke run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/run_manifest.json)、[smoke metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/metrics.jsonl)、[smoke train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/train.log) 与 [smoke latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/checkpoints/latest.pt)
- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/config.json)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/train.log)、[best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/best.pt) 与 [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/latest.pt)
- [实验记录](experiment_log.md)

**原因**

旧 OICM cache 使用 GRAB/reference object trajectory，不能监督 DExplore rollout 的实际物体运动。此次从修正后的 hand/object 联合轨迹重建全部 geometry 和 flow GT；`KNN=16` 同时进入 scale calibration 与模型局部 interaction，避免模型/统计口径不一致。

**验证**

- full cache `COMPLETED`：630 条互斥 parent sequence，train/val/test=`509/58/63`，train MANO/Inspire=`254/255`、val=`28/30`、test=`63/0`；index 与 run manifest 的 RL root 均为 `inspire_rl_object_dexplore`。
- scale calibration：train-only、seed 42、两 source stride 1..10、`knn_k=16`；`s_geo=0.0296379011 m`、`s_hand_flow=0.1148334428 m`、`s_obj_flow=0.0911242272 m`。
- Dataset probe：train/val/test frame samples=`74246/8232/9334`，batch object `[2,1024,3]`、hand `[2,1538,3]`，metadata/config 均为 KNN=16。
- 三卡 2-step smoke `COMPLETED`：global batch 6，loss/gradient finite，sample valid ratio 1，产生 config、manifest、metrics、train log 和 checkpoint；属于工程 `SUPPORTED`。
- 正式训练从随机初始化启动：global batch 96、202300 steps；截至 epoch 4 / step 3096，best equal-source object EPE `7.985666 mm`（MANO `8.277207`、RL-Inspire `7.694125`），best/latest checkpoint 已生成；运行未终止，科研结论保持 `INCONCLUSIVE`。
- `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' ... pytest -q tests/test_object_interaction_cm.py`：`3 passed`；属于 Task 现有 legacy 定向回归。
- `git diff --check -- docs/current_versions.yaml src/task/ObjectInteractionCm`：通过。
- `audit_diff.py --worktree --scope-prefix <本次 8 个受控路径> --check-links`：通过；审计 7 个相对 HEAD 的变更路径及最新 activity 的 25 个现有本地链接。
- 回滚入口：停止该新 run 并移除 V1.2.5 新 cache/config/output；旧 cache、旧 checkpoint 和旧运行均未覆盖、可继续只读复核。

## 2026-09-07 23:03:24 +0800 — V1.2.5 修正物体轨迹、KNN=16 方案与 pilot

- activity_id: `ACT-20260907-230324-OICM-CORRECTED-TRAJECTORY-KNN16-PILOT`
- timestamp: `2026-09-07 23:03:24 +0800`
- modification_version: `V1.2.5`
- type: `data / code / experiment`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认使用 630 条修正数据相交集重新训练 Cm，并明确要求 `KNN=16`。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 查看器、旧 cache、旧 output 和旧 checkpoint）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 新增 V1.2.5 cache/config/plan；转换器只增加实际 `rl_root` provenance 和 modification-version 参数；不修改旧 cache、旧配置、旧 checkpoint 或公共运行时。
- run_id: `oicm-dexplore-rl-pilot-20260907-230232`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（pilot 数据合同和修正轨迹接入通过；full cache、scale 和正式训练尚未完成）

**文件与产物**

- [执行计划](../plan/V1.2.5.md)
- [full 配置](../../configs/active/dexplore_rl_v1_2_5.yaml) 与 [smoke 配置](../../configs/active/dexplore_rl_v1_2_5_smoke.yaml)：新训练使用 `knn_k=16` 和新 index/scale 路径。
- [cache 转换器](../../tools/data/build_dexplore_rl_cache.py)：index 的 `source_roots.inspire_rl` 改为记录实际 `--rl-root`；run manifest 的版本号由命令行锁定。
- [pilot run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5_pilot/run_manifest.json)、[pilot contact check](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5_pilot/pilot_contact_check.json) 和 [pilot geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5_pilot/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)。

**原因与保护边界**

旧 cache 使用旧参考物体轨迹；修正根的 actual object pose 必须重新进入 object geometry、flow GT 和后续 Cm。`KNN=16` 是本轮用户确认的新模型变量，因此同步用于模型与 train-only scale calibration。旧数据、旧运行和依赖旧 checkpoint 的结果均保留，可直接回滚到旧入口。

**验证**

- `py_compile`：转换器与 calibration 脚本通过。
- pilot command：`build_dexplore_rl_cache.py --mode pilot --sequence s1/airplane_lift --variant inspire_rl --rl-root data/processed_data/inspire_rl_object_dexplore --refresh-rl-candidates`，通过；parent intersection `630`，pilot frames `432`，object pool `4096`，hand `1538`。
- pilot contact check：export 与 geometry active frames 均 `358`，disagreement `0`。
- pilot 数值核对：新 cache translation 与 corrected tensor 最大误差 `0`；与旧 tensor 最大序列平移差 `0.125691 m`。
- 尚未启动 full cache 或训练；当前 Viser 进程和 GPU 上其他进程未停止。

## 2026-09-07 22:37:40 +0800 — 核对修正后 DExplore RL 物体轨迹与旧 OICM cache provenance

- activity_id: `ACT-20260907-223740-OICM-CORRECTED-OBJECT-TRAJECTORY-DIAGNOSTIC`
- timestamp: `2026-09-07 22:37:40 +0800`
- modification_version: `V1.2.4`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求只读浏览已更新的 DExplore RL 数据并确认重新训练 Cm 的语义；本事件不修改数据、代码、配置、split、GT、checkpoint 或运行状态。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留用户已有 CmDecoderv2、ObjectInteractionCm 查看器和日志改动）
- scope: 修正后 `inspire_rl_object_dexplore`、旧 `inspire_rl`、现有 `object_interaction_cm_dexplore_rl_v1` cache/index 及转换脚本的只读 provenance 与数值核对。
- conclusion: `SUPPORTED`（确认旧 OICM cache 使用旧参考物体轨迹，未使用修正后的实际模拟物体轨迹；尚未执行新 cache 构建或重新训练）

**原因**

用户说明旧 DExplore RL 导出遗漏实际模拟物体轨迹，并要求在重新训练 Cm 前核对更新数据。由于物体轨迹直接决定 object flow GT、cache 几何和 checkpoint 解释，先进行只读 provenance 与数值检查，避免从旧 cache 恢复或覆盖旧证据。

**证据与发现**

- [修正后 RL 数据 manifest](../../../../../data/processed_data/inspire_rl_object_dexplore/manifest.json) 记录 660 条 DExplore-compatible 序列；`object_pose[198:205]` 与 `inspire_qpos[373:391]` 均来自确定性策略的实际模拟状态，660 条序列的物体位姿均相对 geometric reference 发生变化。
- [旧 OICM cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) 和 [旧 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json) 明确绑定 `data/processed_data/inspire_rl`，生成时间早于修正数据；旧转换脚本实际从 tensor `198:205` 构造 `obj_pose_world`。
- 对旧 index 中同时存在于修正根的 273 条 Inspire-RL 序列逐条核对：旧 cache 的平移与旧 `inspire_rl[:,198:201]` 最大误差为 `0`；旧/新每序列最大平移差的 median / p95 / max 为 `0.099516 / 1.378110 / 3.827840 m`。因此旧训练输入不是修正后的实际物体轨迹。
- 修正根共 660 条，其中与旧 parent index、GRAB geometry 和 DExplore geometry 同时相交 630 条；沿用当前 seed=42、parent split 和互斥 variant 分配算法时，预计 train/val/test 为 `509/58/63`，train MANO/Inspire=`254/255`、val=`28/30`、test=`63/0`。这与旧 `1004/126/125` split 数量不同，属于后续重建前必须确认的数据语义变化。
- 旧 cache、旧 OICM checkpoint 以及依赖该 checkpoint 的 CmDecoderv2/rollout/classifier 结果均保留，仅能解释旧错误数据链路；本诊断没有删除或覆盖这些可回滚证据。

**验证**

- 只读检查两个 RL 根的 tensor shape、序列集合、有限值及新 manifest；旧根 1335 条，修正后的 DExplore-compatible 根 660 条，tensor 均为 `[T,598]`。
- 只读运行现有 `_intersection` 与 `_assign_variants` 得到 630 条相交序列及上述预计 split/variant 计数；没有写出 assignment、cache 或 manifest。
- `ps` 检查未发现 OICM 训练或 cache builder 进程；未启动训练。现有 ObjectInteractionCm Viser 进程保持不变。

## 2026-09-07 20:35:48 +0800 — V1.2.4 将 Viser 查看器对齐近期训练索引与两类手 mesh

- activity_id: `ACT-20260907-203548-OBJECTINTERACTIONCM-TRAINING-VISER-PROVENANCE`
- timestamp: `2026-09-07 20:35:48 +0800`
- modification_version: `V1.2.4`
- type: `code`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户质疑查看器是否使用近期训练数据；Agent 说明旧实现默认读取旧 `cm_object_v2/grab` cache，并提出改为近期训练 `index.json`、按 index variant 加载 MANO/Inspire-RL mesh 的方案，用户回复“可以”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有 CmDecoderv2 改动及未跟踪文件）
- final_plan: [V1.2.3 Dexplore RL 混合训练计划](../plan/V1.2.3.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py`；只读消费近期训练 index、逐序列 geometry manifest/数组、GRAB canonical object mesh、MANO parent mesh 和 Inspire RL qpos/URDF，不写入或迁移数据。
- conclusion: `SUPPORTED`（工程静态检查、几何 provenance 检查和服务端 smoke；未进行浏览器端人工点击回归，不构成科研效果结论）

**文件与数据入口**

- [Viser 查看器](../../visualize_grab.py)：默认入口改为 [近期训练 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json)，轨迹严格按 index 的 split/source/variant 枚举；点云直接 mmap 每条 indexed sequence 的训练 geometry。
- [近期训练配置](../../configs/active/dexplore_rl_v1_2_3.yaml)：其 `data.index` 与查看器新默认值一致。
- [MANO 检查序列 manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/train/mano/s1_airplane_fly_1/geometry/manifest.json) 与 [Inspire-RL 检查序列 manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)：记录本次 smoke 的 split、source、variant、坐标系、parent cache 或 RL qpos 来源。
- [本活动记录](activity_log.md)：登记语义修正、审批、验证和回滚入口。

**原因**

旧查看器默认读取历史 `data/processed_data/cm_object_v2/grab` 双手点云，与近期混合训练使用的 indexed MANO/Inspire-RL 单变体数据不是同一输入合同。此次修正让可视化证据与实际训练 index、逐序列 manifest 和坐标系一致，并在界面中显式暴露 provenance。

**实现与保护边界**

- GUI 仍全部使用鼠标，保留轨迹前后切换、split/轨迹下拉、播放/暂停、逐帧跳转、1–5 cm 累计阈值着色、最近手物点距离、点大小/FPS 和物体/手 mesh 选择控件。
- 查看器显示每帧完整 4096 点物体 pool 与 1538 点右手；状态栏明确提示训练运行时从 4096 pool 确定性采样 1024 点。最近距离和阈值着色基于所显示的完整训练 cache 点云。
- 物体 mesh 使用 GRAB canonical mesh 和该 indexed sequence 的 `obj_pose_world`。MANO 手 mesh 从 manifest 的 `input_parent_cache` 读取，并复现 cache builder 的旧物体系到新 Dexplore 物体姿态变换；Inspire-RL 手 mesh 从 manifest 的 `rl_q.tensor` 读取 native qpos，并复用 cache builder 的 `InspireUrdfModel`、joint reorder 与 FK。
- mesh 依赖缺失时只在状态栏降级提示，点云查看不受影响。未修改 index、geometry 数组、cache/schema、split、GT、训练配置、checkpoint、output、公共 `src/base` 或 `src/task/CmDecoderv2/`。
- 回滚入口：恢复本条目前的 `visualize_grab.py` 旧 root-based loader；所有消费的数据和 mesh 资产均为只读，无数据产物需要回滚。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split train --sequence s1/airplane_fly_1`：通过；确认 `train/grab/mano`、279 帧、4096/1024 物体点合同、1538 右手点；物体 mesh 与点云 bbox 中心误差 `0.1184 mm`，MANO mesh 与点云误差 `0.8879 mm`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split val --sequence s1/banana_lift`：通过；确认 `val/inspire_f1/inspire_rl`、578 帧、4096/1024 物体点合同、1538 右手点；物体 mesh 与点云 bbox 中心误差 `0.0978 mm`，13 个 Inspire visual mesh（742,236 顶点）与 cache 手点云误差 `2.0421 mm`。
- `timeout 12s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split train --sequence s1/airplane_fly_1 --mesh-display both --host 127.0.0.1 --port 8127 --fps 1`：Viser HTTP/WebSocket 启动并完成 MANO 点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `timeout 15s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split val --sequence s1/banana_lift --mesh-display both --host 127.0.0.1 --port 8128 --fps 1`：Viser HTTP/WebSocket 启动并完成 Inspire-RL 点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm`：通过；未生成或修改数据/cache/运行产物。

## 2026-09-07 10:28:50 +0800 — V1.2.3 为 GRAB Viser 查看器增加物体/手 mesh 控件

- activity_id: `ACT-20260907-102850-OBJECTINTERACTIONCM-GRAB-VISER-MESH`
- timestamp: `2026-09-07 10:28:50 +0800`
- modification_version: `V1.2.3`
- type: `code`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求继续为现有查看器增加 mesh 控件，可选择加载物体或手 mesh；用户在确认仓库已有 mesh 来源后多次回复“继续”。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（只修改 ObjectInteractionCm 查看器及本日志，保留用户已有 CmDecoderv2 改动）
- scope: `src/task/ObjectInteractionCm/visualize_grab.py`；只读对齐现有 sampled-point cache、mesh/object-pose cache 与 GRAB canonical object mesh，不改变坐标系、cache/schema、GT、split、训练或评估实现。
- conclusion: `SUPPORTED`（工程 smoke 证据；不构成科研效果结论）

**文件**

- [GRAB Viser 查看器](../../visualize_grab.py)：新增 `Mesh 显示` 鼠标控件（关闭/仅物体/仅手/物体+手）、透明度控件、raw frame 对齐、缺失 mesh 降级提示，以及 `--mesh-root`、`--object-mesh-root`、`--mesh-display`、`--mesh-opacity` 参数。
- [本活动记录](activity_log.md)：登记 mesh 来源、坐标处理、验证和保护边界。

**原因**

现有点云 cache 不保存 mesh；仓库已有按序列保存的 [GRAB mesh/object-pose cache](../../../../../data/processed_data/cm_object_v2_mesh_object_pose_20260830/) 和 [GRAB canonical object meshes](../../../../../data/raw_data/GRAB/tools/object_meshes/contact_meshes/)。查看器按序列相对路径和 `raw_frame_id` 对齐两类 cache：手 mesh 直接读取世界系逐帧顶点，物体 canonical mesh 通过对应帧 `obj_pose_world` 放置到世界系。mesh cache 缺失时不阻止点云查看，只在状态栏报告不可用原因。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift`：通过；物体 banana mesh 为 48,370 顶点/96,736 面，左右手各 778 顶点/1,538 面，物体 mesh 与 sampled cloud 的首帧 bbox 中心误差为 `0.0977 mm`。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift --mesh-display both --host 127.0.0.1 --port 8127 --fps 1`：Viser HTTP/WebSocket 启动并完成物体+双手 mesh 初始渲染，timeout 主动退出（退出码 124）。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --sequence s1/doorknob_use_1`：无对应 mesh cache 时返回 `mesh.available=false` 和明确错误，点云数据仍正常加载。
- Viser handle 定向 smoke 通过 mesh `vertices`、`opacity`、`visible`、`wxyz`、`position` 更新；`git diff --check` 通过。未进行浏览器端人工点击回归，未修改任何数据/cache 或生成运行产物。

## 2026-09-06 20:29:23 +0800 — V1.2.3 新增 GRAB sampled point-cloud Viser 查看器

- activity_id: `ACT-20260906-202923-OBJECTINTERACTIONCM-GRAB-VISER-VIEWER`
- timestamp: `2026-09-06 20:29:23 +0800`
- modification_version: `V1.2.3`
- type: `code`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求脚本放入 ObjectInteractionCm、使用 Viser、全部通过鼠标 GUI 操作、采用累计距离阈值着色，并支持轨迹/帧播放控制。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留用户已有的 CmDecoderv2 修改和未跟踪文件，未触碰）
- scope: `src/task/ObjectInteractionCm/visualize_grab.py`；只读消费现有 `data/processed_data/cm_object_v2/grab` 点云 cache，不改变 cache、schema、GT、split、坐标语义或训练代码。
- conclusion: `SUPPORTED`（工程 smoke 证据；不构成科研效果结论）

**文件**

- [GRAB sampled point-cloud Viser 查看器](../../visualize_grab.py)：新增鼠标 GUI 轨迹切换、播放/暂停、逐帧前后跳转、左右手选择、1–5 cm 累计阈值高亮、点大小/FPS 控件和每帧左右手最近距离显示；提供 `--check-only` 只读检查入口。
- [本活动记录](activity_log.md)：登记实现范围、审批、验证与保护边界。

**原因**

为便于检查 GRAB 双手与物体的采样点云关系，需要一个不依赖键盘快捷键的浏览器界面。训练用 `_SequenceView` 会强制要求 object pose，而当前 GRAB 点云 cache 已有世界系物体/手点云但没有 pose 文件；查看器因此使用只读轻量 loader，仅读取点云、raw frame 和 metadata，不扩展或改写 cache schema。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py` 通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift` 通过：578 帧、物体 4096 点、左右手各 1538 点；首帧最近距离左 `1370.14 mm`、右 `1359.92 mm`。
- `timeout 5s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift --host 127.0.0.1 --port 8124 --fps 1` 通过启动 Viser HTTP/WebSocket 服务并完成初始渲染（timeout 主动退出，退出码 124）；未进行浏览器端人工交互回归。
- `git diff --check` 通过；未修改或生成数据/cache、checkpoint、训练输出和公共 `src/base` 文件。

## 2026-09-06 12:01:31 +0800 — V1.2.3 核对 best.pt 的模型选择指标

- activity_id: `ACT-20260906-120131-OBJECTINTERACTIONCM-BEST-METRIC-DIAGNOSTIC`
- timestamp: `2026-09-06 12:01:31 +0800`
- modification_version: `V1.2.3`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 ObjectInteractionCm 的 best checkpoint 是否按 loss 选择；只读检查配置、runner、BaseRunner 和冻结 checkpoint。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有改动，未修改代码、配置或 checkpoint）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（确认 checkpoint 选择规则；不构成模型效果结论）
- scope: `src/task/ObjectInteractionCm/`、`src/base/base_runner.py` 及冻结 OICM run 的 config/checkpoint 元数据

**原因**

需要确认当前送入 CmDecoderv2 的 OICM `best.pt` 是按总 loss 还是按 object-flow 验证指标保存，避免误解 checkpoint 的优化目标。

**验证**

- [OICM config](../../configs/active/dexplore_rl_v1_2_3.yaml) 与 [run config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json) 均显示 `metric_for_best: val/obj/flow_epe_mm`、`lower_is_better: true`。
- [ObjectInteractionCm runner](../../runner.py) 的 `evaluate_all()` 将 `val/grab/obj/flow_epe_mm` 与 `val/inspire_f1/obj/flow_epe_mm` 做等权算术平均，写入 `val/obj/flow_epe_mm`；不按样本/帧数加权。
- [BaseRunner best 保存逻辑](../../../../../src/base/base_runner.py) 在每次 validation 后按该指标取更小值保存 `best.pt`；`val/loss` 只记录，不作为当前 run 的 best 选择指标。
- 冻结 best.pt 记录的 `best_metric=9.422408395136284`，对应 `val/obj/flow_epe_mm`。

## 2026-09-06 11:55:07 +0800 — V1.2.3 按用户要求停止 OICM 续训并冻结 best.pt

- activity_id: ACT-20260906-115507-OBJECTINTERACTIONCM-STOP-BEST-FREEZE
- timestamp: 2026-09-06 11:55:07 +0800
- modification_version: V1.2.3
- type: operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求停止当前 OICM 续训，并使用 best.pt 启动 CmDecoderv2；仅停止该 run，不删除或覆盖已有产物
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 27316ef8e9552b7b335e53400453902d745b1ebc
- worktree_dirty: true（保留用户已有工作区改动）
- run_id: `object-interaction-cm-dexplore-rl-v1-2-3-20260905-234051`
- run_status: STOPPED
- scope: `src/task/ObjectInteractionCm/`；仅停止既有 OICM run 并冻结其 best checkpoint，不改变上游代码、数据 cache 或旧运行产物
- last_step: 185850
- last_epoch: 105
- best_metric: `val/obj/flow_epe_mm=9.422408395136284`（best.pt step 122130 / epoch 69）
- stop_reason: 用户要求切换到 CmDecoderv2 训练；向 torchrun 主进程 481539 发送 SIGTERM，并确认 OICM 训练进程及其子进程全部退出
- frozen_checkpoint_sha256: `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`

**原因**

用户要求结束当前 OICM 续训并将 `best.pt` 直接作为 CmDecoderv2 的冻结上游；继续让 OICM 运行会改变 decoder 的输入版本，破坏 checkpoint provenance。

**产物与证据**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)
- [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)
- [train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)

**验证**

- `src/task/ObjectInteractionCm/` — 本 Task 的既有配置、缓存构建工具、文档和实验记录均保留；本次仅追加停止记录。
- `ps` 复核 torchrun 主进程、3 个 rank 和 DataLoader 子进程均已退出；`sha256sum` 复核 best.pt 与 decoder 配置 gate 相同。
- latest.pt 保留在停止时的 step 185850 / epoch 105，可作为后续恢复入口；此停止条目不宣称 OICM 已完全收敛。

**验证与边界**

- 进程核对：OICM torchrun、3 个 worker 及 DataLoader 子进程均已退出；未触碰 GPU 4–7 上的其他任务。
- `sha256sum` 复核 best.pt 与配置 gate 一致；best.pt 未被续训覆盖（mtime 2026-09-06 09:52:45 +0800）。
- `nvidia-smi`：GPU 0/1/2 仅有残留显存约 2.5/5/6 MiB，利用率为 0%，可用于后续三卡 decoder run。
- 该条目记录停止操作和 checkpoint 冻结，不代表 OICM 或 decoder 的科研效果结论；旧 checkpoint、cache、数据和运行目录均保留，可从 latest.pt 恢复。

## 2026-09-06 09:38:15 +0800 — V1.2.3 从 latest checkpoint 续训启动

- activity_id: `ACT-20260906-093815-OICM-DEXPLORE-RL-FULL-RESUME-STARTED`
- timestamp: 2026-09-06 09:38:15 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1（沿用既有 final plan 和配置恢复正式训练；不改变科研变量）
- approval: user-approved
- approval_basis: 用户明确要求“先续训吧”。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 同一 V1.2.3 run、同一 index/config/seed/batch/stride，GPU 0/1/2 从 `latest.pt` step 115050 恢复至绝对 stop step 202300。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `RUNNING`
- resume_from: `outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt`（epoch 65 / step 115050）
- conclusion: INCONCLUSIVE（续训刚启动；终态前不作科研效果结论）

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [resume config snapshot](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config_resume_20260906_093911.json)、[resume metadata](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metadata_resume_20260906_093911.json)、[resume run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest_resume_20260906_093911.json)
- [resume checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)、[best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

原正式 run 在 step 115900 无终止标记地停止；latest checkpoint 含 model/optimizer/scheduler/scaler，用户要求先从该点继续完成既定 202300-step 预算。

**验证**

- `torch.load(latest.pt, map_location=cpu)` 通过，确认 step `115050`、epoch `65`、best metric `9.470077`，optimizer/scheduler/scaler 均存在。
- 启动前 `ps` 无旧训练进程；GPU 0/1/2 分别约 2.55/0.005/0.006 GiB，占用可用。
- 命令：`CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml --set train.resume=outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt --distributed`
- 恢复后 launcher/rank 正常运行并推进到 step `115600`；GPU 0/1/2 utilization 约 `86%/77%/82%`，恢复后的 `metrics.jsonl`、`train.log` 持续更新。

## 2026-09-06 09:25:13 +0800 — V1.2.3 收敛性诊断

- activity_id: `ACT-20260906-092513-OICM-DEXPLORE-RL-CONVERGENCE-DIAGNOSTIC`
- timestamp: 2026-09-06 09:25:13 +0800
- modification_version: V1.2.3
- type: diagnostic
- change_level: L0（只读分析既有 metrics/checkpoint，并记录证据；不启动续训）
- approval: user-approved
- approval_basis: 用户询问“目前收敛了吗”。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 分析 V1.2.3 的 epoch-level train/validation 曲线、最佳/最后 checkpoint、source-specific EPE 和停止状态。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `STOPPED`（诊断不改变既有终态）
- conclusion: INCONCLUSIVE（validation 已平台化，但训练未达到计划终点，不能宣称完全收敛或最终效果成立）

**文件**

- [metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)、[best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

需要区分“validation 是否达到平台”和“训练是否完成/优化是否仍在变化”，避免把中断在 57.3% steps 的 run 误报为收敛。

**验证**

- 使用 Python 读取 `metrics.jsonl` 的 65 个 validation epoch，统计最佳点、最近窗口均值/斜率和 source-specific 曲线；未改动 metrics 或 checkpoint。
- 最佳 epoch 46 / step 81420：equal-source object EPE `9.470077 mm`；最后完整 validation epoch 65 / step 115050：`9.574271 mm`。
- 最后 10 个 validation 的 equal-source均值 `9.601506 mm`、首末 `9.631856→9.574271 mm`；最后 20 个均值 `9.644924 mm`，在 `9.47–9.89 mm` 间震荡。
- train object EPE 从 epoch 1 的 `30.27025 mm` 降至 epoch 65 的 `18.32059 mm`，train loss 从 `0.03751` 降至 `0.02047`；说明优化曲线仍在下降，不能称完全收敛。
- 最后 20 个 validation 的 MANO object EPE 均值 `6.466792 mm`、RL-Inspire `12.823055 mm`，差距约 `6.36 mm`，source/domain 差异仍稳定存在。

## 2026-09-06 08:09:59 +0800 — V1.2.3 全量训练非正常停止终态核对

- activity_id: `ACT-20260906-080959-OICM-DEXPLORE-RL-FULL-STOPPED`
- timestamp: 2026-09-06 08:09:59 +0800
- modification_version: V1.2.3
- type: operation / diagnostic
- change_level: L0（只读核对并记录已发生的运行终态；未恢复或改变训练）
- approval: user-approved
- approval_basis: 用户询问“现在怎么样了”；终态记录继承 V1.2.3 final plan。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 核对 V1.2.3 正式 run 的进程、GPU、日志、metrics、checkpoint 和系统 OOM 线索；不自行续跑。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `STOPPED`
- last_step: 115900 / 202300
- last_epoch: 66（最后完整 validation/checkpoint 为 epoch 65）
- best_metric: `val/obj/flow_epe_mm=9.470077`，epoch 46 / step 81420
- best_checkpoint: `checkpoints/best.pt`
- latest_checkpoint: `checkpoints/latest.pt`，epoch 65 / step 115050
- exit_reason: 进程与三个 rank 均已消失，日志于 2026-09-06 03:16:50 +0800 无终止标记地停止；未发现 Python traceback、CUDA OOM 或对应时段 kernel/journal OOM，准确外部信号未知。
- conclusion: INCONCLUSIVE（完成约 57.3% 计划 steps，存在可恢复 checkpoint，但正式训练未完成）

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

旧 activity 仍标记 `RUNNING`，但用户查询时 GPU 已空闲且进程不存在；必须以实际日志和 checkpoint 更新唯一活动时间线，避免把中断 run 误报为完成。

**验证**

- `ps` 未找到 launcher/rank/DataLoader 进程；GPU 0/1/2 utilization 均为 0%，显存回到旧可视化进程占用约 2.55/0.005/0.006 GiB。
- `train.log`/`metrics.jsonl` 最后更新时间均为 03:16:50，最后记录 step 115900 / epoch 66；没有 `Training finished`、`Training failed`、`Traceback` 或 OOM 标记。
- 65 次完整 validation 中，最佳 equal-source object EPE 为 `9.470077 mm`（MANO `6.301521 mm`、RL-Inspire `12.638633 mm`）；最后完整 validation 为 `9.574271 mm`。
- 本地 `best.pt` 和 `latest.pt` 均可读取且包含 model/optimizer/scheduler/scaler；latest 为 step 115050 / epoch 65，可作为后续同配置恢复入口。
- 未恢复训练、未修改 config、数据、GT、split 或 checkpoint。

## 2026-09-05 23:54:31 +0800 — V1.2.3 全量训练人工状态检查

- activity_id: `ACT-20260905-235431-OICM-DEXPLORE-RL-FULL-PROGRESS`
- timestamp: 2026-09-05 23:54:31 +0800
- modification_version: V1.2.3
- type: operation / diagnostic
- change_level: L0（只读检查进程、日志、指标和 checkpoint，并更新运行记录）
- approval: user-approved
- approval_basis: 用户询问“现在还在训练吗”。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 只读核对 V1.2.3 正式 run 的进程、GPU、实时日志、validation 指标和 checkpoint 状态；不改变训练。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `RUNNING`
- last_step: 7100 / 202300
- last_epoch: 5（epoch 4 validation 已完成）
- best_metric: `val/obj/flow_epe_mm=10.283183`，epoch 4 / step 7080
- conclusion: INCONCLUSIVE（训练仍在早期运行；当前 validation 只作进度证据）

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

核对用户询问时训练是否真实存活，而不是只依据旧 activity 的 `RUNNING` 文本判断。

**验证**

- launcher PID 26636 和三个 rank PID 26747/26748/26749 均存活；rank 进程状态为 `Rsl`。
- GPU 0/1/2 utilization 为 89%/85%/88%，显存约 8.98/6.43/6.43 GiB。
- `metrics.jsonl` 与 `train.log` 在 23:54:21 继续更新；未发现 `Traceback`、OOM 或 training failed。
- 最近 validation：MANO object EPE `7.151705 mm`，RL-Inspire object EPE `13.414660 mm`，equal-source `10.283183 mm`；sample valid ratio `0.997809`。
- 当前吞吐约 `1026 samples/s`，runner ETA 约 `5.04 h`。

## 2026-09-05 23:36:08 +0800 — V1.2.3 Dexplore RL/MANO scale 校准启动

- activity_id: `ACT-20260905-233608-OICM-DEXPLORE-RL-SCALE-STARTED`
- timestamp: 2026-09-05 23:36:08 +0800
- modification_version: V1.2.3
- type: data / operation
- change_level: L1（只生成 train-only scale JSON，并修正 stride policy metadata；不改变 geometry、GT 或 split）
- approval: user-approved
- approval_basis: 用户确认按 V1.2.3 默认训练方案执行。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有 CmDecoder 与 ObjectInteractionCm 用户改动）
- scope: 新 index 的两 source 统一 30 Hz stride `[1..10]`；从 train split 重新校准 `s_geo/s_hand_flow/s_obj_flow`。
- run_id: `oicm-dexplore-rl-scale-v1_2_3-20260905-233608`
- run_status: `COMPLETED`
- conclusion: SUPPORTED（train-only scale 统计和 20 个 source/stride group 产出通过；不构成 Cm 效果结论）

**命令与 PENDING 产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.tools.data.calibrate_scales --index data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json --output data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json --max-sequences-per-source 32 --frames-per-sequence-stride 4 --radius-m 0.05 --knn-k 8 --seed 42 --grab-strides 1 2 3 4 5 6 7 8 9 10 --inspire-strides 1 2 3 4 5 6 7 8 9 10`
- [scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json) — `s_geo=0.03182924`、`s_hand_flow=0.09928792`、`s_obj_flow=0.07794207`，20 个 group。
- 验证：index `stride_policy` 与 scale sampling 均为两 source `[1..10]`；输出 JSON 可解析，输入仅为 train split。

## 2026-09-05 23:38:44 +0800 — V1.2.3 三卡 smoke 启动

- activity_id: `ACT-20260905-233844-OICM-DEXPLORE-RL-SMOKE-STARTED`
- timestamp: 2026-09-05 23:38:44 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1（Task-local 训练配置与运行；不修改模型科研语义或旧输出）
- approval: user-approved
- approval_basis: 用户确认按 V1.2.3 final plan 执行。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: Dexplore RL/MANO right-hand mixed Cm；三卡 GPU 0/1/2；smoke 每卡 batch 2、global batch 6、2 steps；from scratch。
- run_id: `oicm-dexplore-rl-smoke-20260905-233844`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（smoke 运行中；不构成科研效果结论）

**命令与 PENDING 产物**

- 命令：`CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3_smoke.yaml --distributed`
- [smoke output](../../../../../outputs/objectinteractioncm/) — PENDING，具体 timestamp run directory 待 runner 创建。

## 2026-09-05 23:39:47 +0800 — V1.2.3 smoke 完成

- activity_id: `ACT-20260905-233947-OICM-DEXPLORE-RL-SMOKE-COMPLETED`
- timestamp: 2026-09-05 23:39:47 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1
- approval: user-approved
- approval_basis: V1.2.3 final plan。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（三卡工程链路通过；仅为 smoke，不代表 Cm 科研效果）

**验证与产物**

- 三卡命令使用 `CUDA_VISIBLE_DEVICES=0,1,2`、world size 3、每卡 batch 2、global batch 6；完成 2 steps，无 NaN/Inf。
- `sample/valid_ratio=1`、`sample/sampling_miss_ratio=0`、slot effective count=16；loss、梯度和 metrics 均写出。
- [smoke output](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/) — 含配置、metadata、run manifest、metrics、train log。
- [smoke metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/metrics.jsonl)
- [smoke train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/train.log)
- [smoke checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/checkpoints/latest.pt)

## 2026-09-05 23:40:15 +0800 — V1.2.3 全量 Cm 训练启动

- activity_id: `ACT-20260905-234015-OICM-DEXPLORE-RL-FULL-STARTED`
- timestamp: 2026-09-05 23:40:15 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1（Task-local 正式训练）
- approval: user-approved
- approval_basis: smoke 已通过，用户确认按 V1.2.3 final plan 执行。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: Dexplore RL/MANO mixed right-hand Cm；from scratch；三卡 GPU 0/1/2；每卡 batch 32、global batch 96、202300 steps。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `RUNNING`
- conclusion: INCONCLUSIVE（正式训练进行中；终态前不作科研效果结论）

**命令与 PENDING 产物**

- 命令：`CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml --distributed`
- [full output](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/) — 当前运行目录。
- [full config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)
- [full run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)
- [full metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)
- [full train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- 当前证据：step 500/202300，约 1091 samples/s，ETA 约 4.9 h；GPU 0/1/2 均约 87% utilization，显存分别约 8.98/6.43/6.43 GiB。

**文件**

- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)
- [experiment log](experiment_log.md)
- [scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json)

**原因**

新 cache 的 MANO 与 RL-Inspire 都是 30 Hz；训练前需要把旧 Inspire 偶数 stride 口径替换为两 source 统一 `[1..10]`，并用 train-only 统计重新校准 scale，避免把旧数据跨度或统计量带入正式实验。

**验证**

- 配置加载、`py_compile`、index JSON 解析和 `git diff --check` 已通过。
- 两 step 三卡 smoke 已 `SUPPORTED`；正式 run 当前 `RUNNING`，终态前不作科研效果结论。

## 2026-09-05 23:38:59 +0800 — V1.2.3 smoke 首次启动失败（配置语法）

- activity_id: `ACT-20260905-233859-OICM-DEXPLORE-RL-SMOKE-FAILED`
- timestamp: 2026-09-05 23:38:59 +0800
- modification_version: V1.2.3
- type: operation
- change_level: L1
- approval: user-approved
- approval_basis: 继承 V1.2.3 final plan。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- run_id: `oicm-dexplore-rl-smoke-20260905-233844`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（YAML 未加引号的 `off` 被解析为布尔 `false`，runner 在 PerformanceMonitor 初始化阶段拒绝；未进入 forward/backward，无 checkpoint 或科研证据生成）
- 保护边界：三 rank 已正常拉起并退出；旧 output、cache、checkpoint 和 GPU 0 可视化进程未修改。
- 修复：smoke 配置改为 `performance.mode: "off"`，随后以同一三卡命令重跑。

## 2026-09-05 21:40:32 +0800 — Dexplore RL-Inspire 右手数据转换启动

- activity_id: `ACT-20260905-214032-OICM-DEXPLORE-RL-CONVERSION`
- timestamp: 2026-09-05 21:40:32 +0800
- modification_version: V1.2.2
- type: data_change / operation
- change_level: L2（sequence split、坐标系、GT geometry cache 与兼容 index）
- approval: user-approved
- approval_basis: 用户最新确认“按默认执行”；默认方案已冻结为右手、GRAB 母 split、train/val 近似 1:1 variant、test MANO、Dexplore RL source。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有 CmDecoder 用户改动）
- scope: 新增 Dexplore RL-Inspire→1538 点几何转换脚本和 V1.2.2 final plan；先执行单序列 pilot，旧 cache/index/split 不覆盖。
- run_id: `oicm-dexplore-rl-pilot-20260905-214032`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（pilot 产物、坐标与 candidate/contact 检查待生成；未据此作科研效果结论）

**文件与计划**

- [V1.2.2 执行计划](../plan/V1.2.2.md) — 冻结 split、坐标、surface sampling、pilot/full 验证和回滚边界。
- [build_dexplore_rl_cache.py](../../tools/data/build_dexplore_rl_cache.py) — 新增 MANO/RL-Inspire 统一 geometry cache、assignment、index 与 run manifest 生成器。
- [pilot output](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/) — PENDING，单序列 pilot 运行产物。

**原因与验证**

- 只使用 `inspire_rl` native q slice `[373:391]` 与 Dexplore right Inspire URDF；不读取 `inspire_geometric`。
- 目标 schema 复用 loader 已支持的 `geometry/manifest.json`，hand 固定 1538 点，object pool 固定 4096，30 Hz，future flow 由 loader 按帧产生。
- 已完成静态验证：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py` 通过。
- 下一步命令（pilot）：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode pilot --sequence s1/airplane_fly_1 --output data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot`。

## 2026-09-05 21:41:18 +0800 — pilot 首次运行失败并修复路径

- activity_id: `ACT-20260905-214118-OICM-DEXPLORE-RL-PILOT-FAILED`
- timestamp: 2026-09-05 21:41:18 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: 继承 V1.2.2 计划与本轮“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: pilot 运行后的 contact-check 路径修复；不改变数据语义、坐标、split 或旧 cache。
- run_id: `oicm-dexplore-rl-pilot-20260905-214032`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（转换阶段完成，末端 contact-check 将 sequence 根误传给 geometry 根；已定位并修正）

**验证与回滚**

- 失败命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode pilot --sequence s1/airplane_fly_1 --output data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot`。
- 证据：异常为 `FileNotFoundError`，目标多拼/少拼一层 `geometry`；未进入坐标或数值结论。
- 已用 `apply_patch` 将 pilot contact-check 输入修正为 `<sequence>/geometry`；失败生成的 37 MB pilot 目录将删除后重跑。

## 2026-09-05 21:42:49 +0800 — Dexplore RL-Inspire pilot 完成

- activity_id: `ACT-20260905-214249-OICM-DEXPLORE-RL-PILOT-COMPLETED`
- timestamp: 2026-09-05 21:42:49 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 计划与用户“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 单序列 `s1/airplane_fly_1` 的 Dexplore RL-Inspire 右手转换、loader probe 与 candidate/contact 几何 sanity check。
- run_id: `oicm-dexplore-rl-pilot-20260905-214249`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅表示 pilot 数据合同与转换 wiring 通过，不表示 Cm/解码科学效果成立）

**产物与验证**

- [pilot cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/) — 单序列 geometry、assignment、index、manifest。
- [pilot run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/run_manifest.json) — 记录输入、q slice、URDF hash、seed、计数与 validation。
- [pilot contact check](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/pilot_contact_check.json) — native contact active 212/279，实际 5 cm 几何 active 222/279，分歧 10/279（3.58%），因此 full 仍沿用已导出的 native contact 语义并保留该偏差证据。
- [pilot validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/validation_summary.json) — object `[T,4096,3]`、hand `[T,1538,3]`、normals/pose/finite 检查通过。
- loader probe：`_SequenceView` 识别 `kind=inspire`、`effective_fps=30`；固定 stride=2 的 sample 输出 object `[1024,3]`、hand `[1538,3]`、future hand flow `[1538,3]` 且 finite。
- coordinate probe：object-frame 点到 Dexplore airplane mesh 顶点最近距离 median 0.415 mm、mean 0.417 mm；说明当前 object pose 转换使用 `native quaternion.T` 与 object-frame contract 一致。

## 2026-09-05 21:43:20 +0800 — Dexplore RL-Inspire full conversion 启动

- activity_id: `ACT-20260905-214320-OICM-DEXPLORE-RL-FULL-STARTED`
- timestamp: 2026-09-05 21:43:20 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: pilot 已通过且用户确认“按默认执行”
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 对 1255 条 GRAB/Dexplore RL 交集序列执行 assignment、MANO/RL geometry 转换、兼容 index 与 run manifest；test MANO-only，train/val variant 近似 1:1。
- run_id: `oicm-dexplore-rl-full-20260905-214320`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING full conversion；运行中不作科研效果结论）

**命令与 PENDING 产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 22:12:25 +0800 — Dexplore RL-Inspire full conversion 完成

- activity_id: `ACT-20260905-221225-OICM-DEXPLORE-RL-FULL-COMPLETED`
- timestamp: 2026-09-05 22:12:25 +0800
- modification_version: V1.2.2
- type: operation / data_change
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 final plan、pilot 通过、用户“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有 CmDecoder 用户改动与未跟踪实验状态）
- scope: 完成三方 intersection 的 1255 条右手 MANO/RL-Inspire geometry cache、assignment、兼容 index 和 loader 终态 probe；旧 cache/index/split 未覆盖。
- run_id: `oicm-dexplore-rl-full-20260905-221207`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（数据合同、转换 wiring、split/loader 审计通过；不代表 Cm 训练效果或未来 CmDecoderV2 科研结论）

**最终产物**

- [full cache root](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — 1255 条 sequence geometry，约 49 GB，不纳入 Git。
- [full index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json) — 兼容 `ref2dex_object_interaction_cm_index_v1_1`，train/val/test=1004/126/125，source=501/64/125 MANO 与 503/62/0 RL-Inspire。
- [assignment](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/assignment.json) — seed=42、parent_seq_id/variant、三方 intersection、两条损坏 MANO 强制 RL 记录。
- [run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — `run_status=COMPLETED`、`conclusion=SUPPORTED`、输入 hash、URDF、计数、1255 条 validation。
- [validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/validation_summary.json) — 每条 sequence 的 shape/finite/pose 检查。

**终态验证**

- 三方 intersection：parent GRAB object cache、`dexplore_grab/sequences`、`inspire_rl` 均 1255 条，missing=0。
- 无泄露：全局 `parent_seq_id` 唯一；train/val 不跨 variant；test 全部 MANO；train/val variant 近似 1:1。
- schema：object pool `[T,4096,3]`、right hand `[T,1538,3]`、normals/pose/frame id/frame time 全部 finite；30 Hz；native q slice `[373:391]` 与 Dexplore native→URDF reorder 已写入 manifest。
- loader probe：三 split 均可由现有 `ObjectInteractionCmDataset` 读取；rows=304343/39538/36365；抽样 object `[1024,3]`、hand `[1538,3]`、future flow `[1538,3]` finite。
- 保护边界：未修改旧 `cm_object_v2_surface512_object_pose_20260830`、旧 `object_interaction_cm_v1_1/index.json`、共享 `src/base`、CmDecoder v2；未启动 Cm 训练。

**文件、原因与验证入口**

- [V1.2.2 执行计划](../plan/V1.2.2.md) — 固化本次数据 split、坐标、schema、pilot/full 顺序和回滚边界。
- [build_dexplore_rl_cache.py](../../tools/data/build_dexplore_rl_cache.py) — 实现三方 intersection、variant assignment、Dexplore q→URDF surface FK、MANO object-frame 重投影、cache/index/run manifest 与 resume。

**原因**

- 需要在同一 GRAB parent split 上混合 MANO 与 Dexplore RL-Inspire，同时禁止同一 `parent_seq_id` 跨 variant 泄露；右手-only 口径要求不伪造损坏的旧 MANO 文件。
- 保留 object_pose_t、4096 object pool、1538 hand surface、future hand flow 和 30 Hz，使后续 Cm 训练可以直接复用现有 loader 合同。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py` — PASS。
- 三方 intersection/assignment/index 计数审计 — PASS，missing=0、parent id 全局唯一、test MANO-only。
- `ObjectInteractionCmDataset` 三 split loader probe — PASS，rows `304343/39538/36365`，抽样 object/hand/future-flow finite。
- [validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/validation_summary.json) — 1255/1255 sequence shape、pose、finite 检查通过。

## 2026-09-05 22:15:44 +0800 — RL candidate mask 几何口径 refresh 启动

- activity_id: `ACT-20260905-221544-OICM-DEXPLORE-RL-CANDIDATE-REFRESH-STARTED`
- timestamp: 2026-09-05 22:15:44 +0800
- modification_version: V1.2.2
- type: data_change / operation
- change_level: L2（只更新新 cache 的 RL candidate mask，不改变点云、split、pose 或旧 cache）
- approval: user-approved
- approval_basis: pilot 已量化 native-contact 与几何 5 cm mask 的 3.58% 分歧；为公平性采用同一几何规则。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 对新 full/pilot cache 的 RL-Inspire entries，以 object-frame 生成 hand/object 近邻的 5 cm 规则重写 `obj_candidate_mask_5cm.npy`；MANO mask 不变。
- run_id: `oicm-dexplore-rl-candidate-refresh-20260905-221544`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING mask refresh 与 loader 复核）

**命令与产物**

- full：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --refresh-rl-candidates --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- pilot：PENDING，full 完成后对 pilot 同步 refresh。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING refresh 终态。

## 2026-09-05 22:42:42 +0800 — RL candidate mask 几何口径 refresh 完成

- activity_id: `ACT-20260905-224242-OICM-DEXPLORE-RL-CANDIDATE-REFRESH-COMPLETED`
- timestamp: 2026-09-05 22:42:42 +0800
- modification_version: V1.2.2
- type: data_change / operation
- change_level: L2
- approval: user-approved
- approval_basis: 用户已确认默认执行；pilot 分歧证据支持统一 5 cm geometry active 口径
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: full 565 条 RL-Inspire sequence 与 pilot 的 candidate mask 已按生成 hand/object geometry 的 object-frame 5 cm KD-tree 规则刷新；MANO mask、点云、pose、split 未改变。
- run_id: `oicm-dexplore-rl-full-20260905-224017`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（mask/loader 数据合同通过；不代表 Cm/decoder 科研效果）

**文件、原因与验证**

- [V1.2.2 执行计划](../plan/V1.2.2.md) — 本次数据处理边界与回滚入口。
- [build_dexplore_rl_cache.py](../../tools/data/build_dexplore_rl_cache.py) — 新增 `--refresh-rl-candidates` 几何 mask refresh。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — `run_status=COMPLETED`、`candidate_refresh.rl_sequences=565`、`SUPPORTED`。
- [pilot contact check](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/pilot_contact_check.json) — refresh 后 native/geometry active 均 222/279，分歧 0。

**原因**

- MANO 与 RL 必须使用同一 `active_only` 5 cm 几何规则；native Dexplore contact 与生成表面几何存在 3.58% pilot 分歧，不能混用两种过滤语义。
- 发现的 1 条旧 partial RL 目录 `s5/waterbottle_shake_1` 不属于最终 assignment，已移至 [recovery 目录](../../../../../data/processed_data/_recovery/object_interaction_cm_dexplore_rl_v1_stale_s5_waterbottle_shake_1_20260905/)，最终 geometry/index 均为 1255/1255，无额外 variant。

**验证**

- full index/geometry 审计：1255 entries、1255 geometry manifests、MANO 690、RL 565、extra ids=0。
- active-only loader probe：train/val/test rows=`169860/21911/20612`；抽样 object `[1024,3]`、hand `[1538,3]`、future flow `[1538,3]` finite。
- pilot/full manifest candidate semantics：`generated_rl_hand_to_object_surface_5cm`；刷新未改变 object/hand shape、pose、split 或旧 cache。

## 2026-09-05 22:10:13 +0800 — full manifest intersection 约束补记并刷新

- activity_id: `ACT-20260905-221013-OICM-DEXPLORE-RL-MANIFEST-REFRESH-STARTED`
- timestamp: 2026-09-05 22:10:13 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 默认方案要求同时核对 `dexplore_grab`、旧 object cache 与 `inspire_rl`；只补充已验证的输入约束，不改变已生成 geometry。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: converter 现在显式检查三方 intersection；用 `--resume` 只复用 geometry 并刷新 index/run manifest。
- run_id: `oicm-dexplore-rl-manifest-refresh-20260905-221013`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING metadata refresh）

**命令与产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING（geometry 复用）。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 21:57:41 +0800 — full resume 遇到 parent 右手 MANO 损坏项

- activity_id: `ACT-20260905-215741-OICM-DEXPLORE-RL-FULL-RESUME-FAILED`
- timestamp: 2026-09-05 21:57:41 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: 继承 V1.2.2 与用户“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: resume 在约 700 条序列处发现 `s7/headphones_lift` 的旧 right MANO mmap 损坏；只读扫描确认共有 2 条 train 序列损坏（另一个 `s2/mug_drink_2` 已分到 RL）。
- run_id: `oicm-dexplore-rl-full-resume-20260905-214854`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（转换器原 assignment 将损坏 right MANO 分给 MANO；不生成伪 MANO，改为强制 RL variant）

**证据与修正**

- right-only 文件扫描：1255 条中仅 `s2/mug_drink_2`、`s7/headphones_lift` 的 `right/hand_points_world.npy` 无法 mmap，且缺 `hand_normals_world.npy`/candidate；两条均在 train，test 无损坏项。
- 已通过 `apply_patch` 将 `mano_available` 写入 assignment：train/val 中不可用项强制 `inspire_rl`，test 若不可用则直接失败；不修改旧 cache。
- full partial 目录继续保留，下一次使用 `--resume` 会复用已完成 geometry 并只转换剩余项。

## 2026-09-05 21:58:08 +0800 — full conversion resume-2 启动

- activity_id: `ACT-20260905-215808-OICM-DEXPLORE-RL-FULL-RESUME2-STARTED`
- timestamp: 2026-09-05 21:58:08 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: parent right-hand 损坏项已证据化，按默认方案保留序列并强制其进入 RL variant
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 以新 assignment 重建并继续 full cache；已完成 geometry 复用，RL 分支不依赖损坏 MANO 文件。
- run_id: `oicm-dexplore-rl-full-resume2-20260905-215808`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING resume-2 终态）

**命令与产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 21:48:31 +0800 — full conversion 首次运行失败并切换安全 resume

- activity_id: `ACT-20260905-214831-OICM-DEXPLORE-RL-FULL-FAILED`
- timestamp: 2026-09-05 21:48:31 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 计划与 pilot 通过后的继续执行
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: full conversion 在已完成约 305 条序列后遇到旧 parent GRAB cache `s2/mug_drink_2/right/hand_points_world.npy` 的 mmap 文件损坏；修复脚本使 RL variant 不再读取无关 MANO hand，并加入不覆盖的 `--resume`。
- run_id: `oicm-dexplore-rl-full-20260905-214320`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（失败来自转换器对 RL 分支读取无关旧 hand 文件；不是新坐标/FK 证据）

**证据与保护**

- 失败异常：`ValueError: mmap length is greater than file size`，发生在旧 cache hand mmap；当时已生成约 13 GB、305 条 sequence geometry，未覆盖旧数据。
- 已通过 `apply_patch`：RL variant 仅读取 object pool/pose；MANO variant 仍强校验右手 hand geometry；新增 `--resume` 只复用存在完整 `geometry/manifest.json` 的序列。
- 原失败 pilot 临时目录仍位于 [recovery 目录](../../../../../data/processed_data/_recovery/object_interaction_cm_dexplore_rl_v1_pilot_failed_20260905_214118/)，可人工清理；本次 full partial 目录保留用于 resume。

## 2026-09-05 21:48:54 +0800 — full conversion 安全 resume 启动

- activity_id: `ACT-20260905-214854-OICM-DEXPLORE-RL-FULL-RESUME-STARTED`
- timestamp: 2026-09-05 21:48:54 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: 继承 V1.2.2 与 pilot 结果；resume 不改变 split/坐标/schema
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 在既有 full partial 目录上复用完整 sequence geometry，继续生成剩余序列并最终重建 assignment/index/run manifest。
- run_id: `oicm-dexplore-rl-full-resume-20260905-214854`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING resume 终态）

**命令与产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 20:30:09 +0800 — RL-Inspire native q 与 OICM 几何接口诊断

- activity_id: `ACT-20260905-203009-OICM-RL-NATIVE-Q-GEOMETRY-CONTRACT`
- timestamp: 2026-09-05 20:30:09 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读接口与 native tensor schema 核查；未改变代码、配置、数据、cache、split、GT 或 checkpoint）
- approval: user-requested
- approval_basis: 用户确认主实验使用 Dexplore RL-Inspire variant，CmDecoderv2 延后单独建立，当前仅先训练 Cm
- skills_used: research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留已有 CmDecoder 修改与未跟踪文件）
- scope: 核对 `inspire_rl` 的 native tensor 布局与 ObjectInteractionCmDataset 当前支持的 hand geometry 布局，并将用户确认的 sequence-level embodiment 划分语义记录为后续 plan 输入。
- conclusion: INCONCLUSIVE（实验定义已基本明确，但 RL q→固定 Inspire 表面点/flow 的数据桥接和无 paired target 的测试指标尚未实现/验证）

**文件与证据**

- [Dexplore RL export README](../../../../../data/processed_data/inspire_rl_before_full_coordinate_fix_20260905/EXPORT_INFO.txt) — 记录 RL checkpoint rollout、右手、30 Hz、1335 条序列、native `(T,598)`。
- [Dexplore export schema](../../../../../data/processed_data/inspire_rl/) — 当前 RL 文件为 `interaction_hand_inspire.pt`；native tensor 的 Inspire 18-DOF slice 为 `373:391`，不能直接作为 OICM hand point cloud。
- [ObjectInteractionCm dataset](../../dataset.py) — 当前 loader 要求 hand geometry、normals 和 5 cm candidate metadata，Inspire 路径要求 `geometry/manifest.json`、`hand_points_world.npy` 等文件。

**原因**

- 用户方案应解释为：原始 GRAB sequence 只选择一个 embodiment；MANO sequence 只进入 OICM source 混合训练，RL-Inspire sequence 进入 OICM，并作为未来 Inspire-q decoder 的训练数据；同一 parent sequence 不同时出现两种 variant。
- 最终 held-out MANO → Cm → CmDecoderv2 是无 paired Inspire target 的跨 embodiment 泛化测试，不是逐帧 MANO→Inspire q 监督测试。
- 因此 decoder 训练阶段可以只使用 RL-Inspire subset；若把 MANO subset 也送入 q decoder，必须另定义 target，不能默认为 Inspire supervision。

**验证**

- 读取 RL/几何 export 说明及样例 tensor，确认 `inspire_rl` 与 `inspire_geometric` 是独立产物，且 RL 版本来自 deterministic policy rollout。
- 读取当前 OICM dataset loader，确认 q native tensor 不能直接满足 1538 点手几何合同；后续需用固定 Inspire URDF/mesh/FK 将 RL q 变成 surface points、normals 和 future flow。
- 本次未生成转换 cache、未修改索引、未启动训练或评估。

**边界、建议与回滚**

- 本条仅记录诊断与用户澄清；删除本条活动记录即可回滚本次文档差异。
- 后续 plan 需要单独冻结 embodiment assignment、sequence/object split、source-balanced sampling、RL q→geometry 的 manifest，以及无 paired target 时的 simulation/object-interaction 评价指标。

## 2026-09-05 20:25:24 +0800 — Dexplore RL 与几何重定向数据口径复核

- activity_id: `ACT-20260905-202524-OICM-DEXPLORE-RL-DATASET-CLARIFICATION`
- timestamp: 2026-09-05 20:25:24 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读数据目录与导出说明复核；未改变代码、配置、数据、cache、split、GT 或 checkpoint）
- approval: user-requested
- approval_basis: 用户明确指定 Dexplore 的 RL 数据集，而不是几何重定向数据集，并补充 Cm/decoder 实验边界
- skills_used: research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留已有 CmDecoder 修改与未跟踪文件）
- scope: 只读核对本地 `dexplore_grab`、`inspire_rl`、`inspire_geometric`、RL/几何导出说明和当前 OICM/CmDecoder 接口；不启动生成、训练或评估运行。
- conclusion: INCONCLUSIVE（已确认 RL 数据入口和用户实验语义，但 Cm 混合训练与后续 CmDecoderv2 尚未实现/验证）

**文件与证据**

- [Dexplore GRAB motion/object](../../../../../data/processed_data/dexplore_grab/) — 1335 条 GRAB 处理序列，包含 MANO motion 与 object。
- [Dexplore Inspire RL](../../../../../data/processed_data/inspire_rl/) — 1335 个 `interaction_hand_inspire.pt`，每条 shape 为 `(T, 598)`，由 `/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth` 的 Isaac Gym deterministic policy rollout 生成，右手、18-DOF、30 Hz。
- [Dexplore Inspire geometric](../../../../../data/processed_data/inspire_geometric/) — 同样 1335 条，但属于几何 retarget，不是本次目标数据。
- [RL export info](../../../../../data/processed_data/inspire_rl_before_full_coordinate_fix_20260905/EXPORT_INFO.txt) — 明确记录 `Export: GRAB -> Inspire (RL checkpoint rollout)`、`Robot: Inspire right hand`、`Rate: 30 Hz`。
- [几何 export info](../../../../../data/processed_data/inspire_geometric_before_full_coordinate_fix_20260905/EXPORT_INFO.txt) — 明确记录几何重定向及 30 Hz 下采样，作为排除项。

**原因**

- 用户的目标不是拿几何 retarget 结果充当 Inspire target，而是把 RL policy rollout 作为 Inspire variant 的 hand trajectory source，与原始 GRAB MANO variant 一起训练 Cm；未来再由独立的 `CmDecoderv2` 输出 Inspire q。
- 为避免泄露，不能让同一原始 GRAB sequence 的 MANO 和 Inspire RL variant 跨 train/test 出现；更严格的做法是先按原始 sequence/object 划分，再为每个 split 选择 variant，保留 `parent_seq_id`。
- 测试阶段只需要 held-out MANO 提取 Cm；不要求生成对应 Inspire 数据是合理的，但这样只能先验证 Cm 的跨 embodiment 表征/下游可解码性，不能在当前阶段得到 Inspire q 的定量测试结果。

**验证**

- 读取四个数据根目录的文件布局和 `EXPORT_INFO.txt`，确认 `inspire_rl` 与 `inspire_geometric` 是两套独立产物，且 RL 版本确实来自 checkpoint rollout。
- 读取样例 RL/几何 tensor，二者均为 `(T, 598)` native tensor；RL 与几何不能仅按文件名区分，数据 manifest 必须显式写入 `source_type=rl`。
- 当前仅做只读检查，未生成新数据、未改索引、未启动训练；后续代码实现仍需用户确认并定稿对应 plan。

**边界、建议与回滚**

- 本条只记录数据口径复核；删除本条活动记录即可回滚本次文档差异。
- 后续 Cm 训练建议采用右手、30 Hz、同一 source frame/horizon，并把 MANO 与 RL-Inspire 作为 source variant；`CmDecoderv2` 单独建 Task，target 只定义为 Inspire q，不把几何版本混入主实验。

## 2026-09-05 19:59:22 +0800 — Dexplore/GRAB MANO-Inspire 混合训练方案诊断

- activity_id: `ACT-20260905-195922-OICM-DEXPLORE-PIPELINE-DIAGNOSTIC`
- timestamp: 2026-09-05 19:59:22 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读仓库、数据布局、接口与实验语义核查；未改变代码、配置、数据、cache、split、GT 或 checkpoint）
- approval: user-requested
- approval_basis: 用户要求检查使用 Dexplore 将 GRAB MANO 与 Inspire 重定向数据混合训练 Cm/解码器的可行性与疑问
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留已有 CmDecoder 修改与未跟踪文件）
- scope: 读取 ObjectInteractionCm/CmDecoder 当前数据合同、模型输入输出、索引与 split 约束；检查本地 `dexplore_grab` 处理数据的序列数量、字段和与当前 GRAB cache 的交集；核对 Dexplore 官方转换脚本和模型卡。未修改代码、配置、原始数据、cache、checkpoint 或运行目录。
- conclusion: INCONCLUSIVE（工程上可构造该实验，但源/目标手、未来 hand-flow 条件、split、时间尺度、伪 GT 与采样权重仍需先固定，当前没有新的训练/评估证据支持方案效果）

**文件与证据**

- [Dexplore processed root](../../../../../data/processed_data/dexplore_grab/) — 本地 1335 个序列目录，每个当前只有 `motion.npz` 与 `object.npz`；未发现 Inspire q 或 1538 点几何输出。
- [ObjectInteractionCm dataset](../../dataset.py)、[ObjectInteractionCm model](../../model.py) — 当前 OICM 将双 MANO 手或单 Inspire 手填充到最大 3076 点；forward 把 future `hand_flow` 作为输入条件，模型本身不消费 `stride/delta_time_s`。
- [CmDecoder adapter](../../../CmDecoder/object_interaction_cm_model.py) — 当前适配器把同一个 1538 点手及其 flow 同时送入 OICM 和解码目标，尚无 MANO source → Inspire target 的分离字段。
- [当前 OICM index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 现有混合索引仍是 GRAB 与 HRDexDB Inspire；Dexplore 数据尚未接入。
- [Dexplore conversion script](https://raw.githubusercontent.com/NVlabs/dexplore/main/data_processing/convert_grab.py) — 官方转换会对 GRAB 做 `::4` 采样、逐帧 retarget 到机器人 DOF，并写出 `interaction_hand_inspire.pt`；这不是本地 `dexplore_grab` 目录现有的完整输出。
- [Dexplore model card](https://github.com/NVlabs/dexplore/blob/main/MODEL_CARD.md) — 发布模型使用约 1269 条 GRAB 中的 658 条，未提供专门的 held-out GRAB 测试 split。

**原因**

- 本地 Dexplore 处理序列归一化名称后与当前 GRAB object cache 有 1255 条交集、80 条缺失；处理帧数约为 raw 120 Hz 的四分之一，说明当前对象轨迹大致为 30 Hz，但仍需由 manifest 确认 frame mapping。
- 同一原始序列生成 MANO 与 Inspire 两个 variant 时，split 必须在 variant 生成前按原始 sequence（必要时 object）划分；否则测试 MANO 与训练 Inspire 的同序列配对会泄漏。
- 如果目标是 MANO Cm → Inspire 解码器，训练样本必须显式区分 `source_hand_*` 与 `target_hand_*`；不能把 MANO 和 Inspire 都作为无标签 target 混在同一个 decoder 监督池里。当前 decoder guidance/plan 仍是 Inspire-only。
- Dexplore 的 Inspire q/几何是 retarget 产生的伪 GT，不是实测机器人轨迹；结果应命名为 retarget-label reconstruction/retarget consistency，并额外报告关节限位、平滑性和接触关系。
- 若用未来 `hand_flow` 提取 Cm，评估属于 teacher-forced/offline 条件重建；不能直接宣称在线或未来预测。在线声明需要只用观测到的 flow/闭环 rollout 另测。
- MANO 当前可双手，而 Dexplore Inspire 输出通常是单个机器人手；需固定右手/左手口径或明确双手拼接规则，否则 padding、接触和 embodiment 差异会成为 shortcut。

**验证**

- 只读统计确认本地 Dexplore 目录包含 1335 个序列，归一化名称后与当前 GRAB object cache 交集为 1255、缺失 80；样例 raw GRAB 为 120 Hz，Dexplore 处理帧数约为 raw 的四分之一。
- 读取当前 ObjectInteractionCm 的 dataset/model、CmDecoder adapter、索引及 Task 指导/plan，确认现有 decoder 仍是 Inspire-only，同一 1538 点手/flow 同时承担 OICM 输入和 decoder target，尚未支持 MANO source → Inspire target。
- 使用 `audit_diff.py --check-links` 做活动记录审计；首次检查提示本条缺少标准“原因/验证”段，已按仓库合同补齐后重新执行。

**边界、建议与回滚**

- 本次只产生诊断记录；代码、配置、数据、cache、checkpoint、split、GT 和既有运行均未改动，删除本条活动记录即可回滚本次文档差异。
- 后续若实现该方案，需要先定稿与指导同版本的 plan，至少冻结：source/target 字段、手侧、sequence/object-disjoint split、共同物理 horizon、Dexplore converter/URDF/mesh/点采样 manifest、variant 采样权重，以及 teacher-forced 与 online 两类评估边界。

## 2026-09-05 17:18:12 +0800 — GRAB 原始 120 Hz 与 HRDexDB 物理时间跨度对齐诊断

- activity_id: `ACT-20260905-171812-OICM-GRAB-120HZ-HR-TIMESCALE`
- timestamp: 2026-09-05 17:18:12 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读时间分辨率与运动幅度核查；未改变代码、配置、数据、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求比较 GRAB 原始 120 Hz 与 HRDexDB 的运动幅度，并判断时间采样是否造成表观差异
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留 CmDecoder 用户改动与未跟踪文件）
- scope: 读取 GRAB 原始 `.npz` 的物体轴角/平移、当前 30 Hz GRAB object-pose cache、旧 cache 中的 5 cm candidate 点索引，以及当前 Inspire-F1 HRDexDB geometry；在 val 序列上按物理时间跨度重算 object-pose frame 的物体表面位移。未修改代码、配置、原始数据、cache、checkpoint 或运行目录。
- conclusion: SUPPORTED（120 Hz 的单 raw frame 位移因时间间隔缩短而变小，但在相同物理时间跨度下 GRAB 仍比 Inspire-F1 HRDexDB 大约 8–11 倍；若只把 GRAB 改成 120 Hz 而保持整数 stride `1..10`，会把 GRAB horizon 错移到约 `8–83 ms`，进一步破坏与 HR `67–667 ms` 的时间语义匹配）

**文件与证据**

- [GRAB 原始样例](../../../../../data/raw_data/GRAB/grab/s1/airplane_fly_1.npz) — 原始字段声明 `framerate=120 Hz`，物体轨迹保留每个 raw frame 的 `global_orient` 与 `transl`。
- [GRAB cache meta](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s1/airplane_lift/shared/meta.json) — 当前 cache 为 `source_fps=120`、`ds_rate=4`，有效频率为 30 Hz，物体池 4096 点。
- [ObjectInteractionCm index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 当前训练的 GRAB stride 为 `1..10`、Inspire-F1 stride 为偶数 `2..20`，两者都由各自 cache 的 frame index 解释。
- [HRDexDB geometry manifest](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/episodes/78013e164ab75a34/geometry/manifest.json) — HR geometry 的 `delta_time_median_s≈30 ms`、最大约 `60.7 ms`，并提供 4096 点池与 5 cm object candidate mask。

**原因**

- 需要把“120 Hz 逐帧看起来位移更小”和“同一物理时间内实际运动更小”分开；如果不对齐 `Δt`，直接比较整数 stride 会把采样率差异误判成数据分布差异。

**验证**

- 统计口径：GRAB 使用当前 index 的 126 条 val 序列、原始 120 Hz 物体 pose；以旧 GRAB cache 的 candidate 点索引筛选当前有接触的 source frame，每行最多抽 256 个候选物体表面点。HR 使用当前 Inspire-F1 val 的 58 条序列、1024 个固定池点，并筛选 5 cm candidate 点。位移是在当前物体 pose 的坐标系中计算；这是分布诊断，不是模型 benchmark。
- GRAB 原始 120 Hz（candidate object points，均值 / 中位数 / p95，mm）：stride 1（8.3 ms）=`2.60/1.26/9.85`；stride 4（33.3 ms）=`9.99/4.89/38.97`；stride 8（66.7 ms）=`19.67/9.61/76.44`；stride 40（333.3 ms）=`87.44/46.59/340.20`。
- Inspire-F1 HRDexDB 30 Hz（同类 candidate object points，mm）：stride 1（约 33.3 ms）=`1.36/0.64/4.55`；stride 2（约 66.7 ms）=`2.37/1.11/8.13`；stride 10（约 333 ms）=`9.58/4.37/33.92`。
- 物理时间对齐后：约 66.7 ms 的 GRAB stride 8 / HR stride 2，其均值、中位数、p95 比约为 `8.3×/8.6×/9.4×`；约 333 ms 的 GRAB stride 40 / HR stride 10，比约为 `9.1×/10.6×/10.0×`。这与既有 30 Hz active-flow 诊断（GRAB 中位数约 `9.1 mm`、HR 约 `1.0 mm`）一致，说明简单下采样没有消除域差。
- 仅比较单整数 stride 会产生误导：GRAB raw stride 1 的间隔只有 8.3 ms；把它与 HR stride 2 的 66.7 ms 直接比较，会把较短 horizon 误认为较小运动。当前模型 forward 读取 object/hand 几何与 flow，但不读取 `stride` 或 `delta_time_s`，因此不同物理 horizon 会共享同一个预测函数。

**边界、建议与回滚**

- 本次只量化了物体表面位移；GRAB 120 Hz 的全手 MANO surface 未在全量上重建。既有同口径 66.7 ms 诊断仍显示 hand flow 中位数约 GRAB `13.5 mm`、Inspire-F1 `1.9 mm`（HR human MANO 约 `5.4 mm`），方向与 object 结论一致。
- 若保留原始 120 Hz，建议先按物理 horizon 重定义 stride：GRAB raw `4..40` 才覆盖当前 30 Hz `33..333 ms`，与 HR stride 2/10 对齐分别约用 raw stride 8/40；或者先统一重采样到 30 Hz。任何 stride/采样合同变更都应单独建 plan 并做受控 ablation，不能把本诊断当作效果因果证据。
- 仅新增本活动条目；删除本条即可回滚文档差异。代码、配置、数据、cache、checkpoint 和既有运行均未改动。

## 2026-09-05 17:49:32 +0800 — 按位移幅度匹配 GRAB 原始 120 Hz 与 HRDexDB stride 诊断

- activity_id: `ACT-20260905-174932-OICM-AMPLITUDE-STRIDE-MAP`
- timestamp: 2026-09-05 17:49:32 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读幅度分布与 stride 映射核查；未改变代码、配置、数据、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求在考虑运动幅度对齐时估计 GRAB 原始 120 Hz 与 HRDexDB stride 的对应关系
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留 CmDecoder 用户改动、未跟踪文件及本任务既有诊断记录）
- run_status: NOT_STARTED（只读统计；未启动正式训练/评估 run）
- conclusion: SUPPORTED（在当前 val 物体候选点口径下，中心位移幅度的经验映射约为 `GRAB raw stride ≈ ceil(HR stride / 3)`；这不是物理时间或速度语义的等价映射，映射用于训练尺度 ablation 的可行起点，收益因果仍为 INCONCLUSIVE）
- scope: 使用当前 ObjectInteractionCm index 的 GRAB val 126 条序列和 Inspire-F1 HRDexDB val 58 条序列；GRAB 读取原始 120 Hz object pose，HR 使用约 30 Hz geometry，按 5 cm candidate object points 统计多个 stride 的位移均值/中位数/p95，并为每个 HR stride 选择分布距离最近的 GRAB raw stride。未修改代码、配置、原始数据、cache、checkpoint 或运行目录。

**文件与证据**

- [ObjectInteractionCm index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 当前 HR stride 为偶数 `2..20`，GRAB cache stride 为 `1..10`。
- [GRAB 原始样例](../../../../../data/raw_data/GRAB/grab/s1/airplane_fly_1.npz) — 原始 `framerate=120 Hz`。
- [GRAB cache meta](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s1/airplane_lift/shared/meta.json) — 现有 GRAB cache 仍为 `source_fps=120`、`ds_rate=4`，即有效 30 Hz。
- [HRDexDB geometry manifest](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/episodes/78013e164ab75a34/geometry/manifest.json) — HR 几何帧间隔中位数约 30 ms。

**原因**

- 物理时间对齐回答“相同时间内移动了多少”；本条额外回答“若只让监督位移的统计幅度相近，应取哪个 raw stride”。两者目标不同，不能把幅度映射解释成动力学等价。

**验证**

- 统计口径：GRAB 每个 active source frame 最多抽取 128 个 candidate object surface points，HR 每条序列从 4096 点池中使用 5 cm candidate mask 后统计；数值为 val 分布诊断，不是 benchmark。
- GRAB raw120 的 candidate object displacement（均值/中位数/p95，mm）随 stride 近似为：`g1=2.59/1.26/9.84`、`g4=9.96/4.87/38.93`、`g8=19.62/9.56/76.37`、`g12=29.03/14.22/113.16`。
- HRDexDB Inspire-F1 约 30 Hz 的对应统计为：`h2=2.37/1.11/8.13`、`h10=9.58/4.37/33.92`、`h20=17.65/8.36/61.58`。
- 按均值/中位数/p95 的最近邻结果综合后，当前使用的 HR 偶数 stride 建议映射为：`h2→g1`、`h4→g2`、`h6→g2~3`、`h8→g3`、`h10→g4`、`h12→g4~5`、`h14→g5`、`h16→g5~6`、`h18→g6`、`h20→g7`；简化成单值即 `{1,2,2,3,4,4,5,6,6,7}`，可记作 `g≈ceil(h/3)`。
- 该映射的时间跨度明显更短：例如 HR `h2` 约 60–67 ms，而幅度匹配的 GRAB `g1` 只有 8.3 ms；HR `h10` 约 300–333 ms，而幅度匹配的 GRAB `g4` 只有 33.3 ms。相同物理时间则仍是 `h2↔g8`、`h10↔g40`，且 GRAB 位移约大 8–11 倍。
- 作为 hand-flow 边界检查，HR 当前 1538 点 MANO 在 `h2/h10/h20` 的中位数约为 `1.79/8.70/15.98 mm`；本次没有重建 GRAB raw120 全量 MANO surface，因此上述 stride 映射目前只对 object target 直接成立，不能宣称 hand 与 object 共用同一精确映射。
- 现有 GRAB 30 Hz cache（raw `g4/g8/g40`）的全手点位移中位数约为 `5.68/11.28/52.22 mm`，与 HR `h2/h10/h20` 的 `1.79/8.70/15.98 mm` 对照，线性插值给出的 hand 幅度映射更接近 `g≈0.5~0.65h`；这进一步说明 object 的 `ceil(h/3)` 不能未经验证地同时用于 hand。

**边界、建议与回滚**

- 若目标是消除 loss 中 target magnitude 的主导差异，可把 `g≈ceil(h/3)` 作为受控 ablation 的初始配对，并同时记录真实 `Δt`/速度；现有 GRAB 30 Hz cache 的 stride `1` 已相当于 raw `g4`，无法表示 `g1~3` 的短幅度 horizon。
- 若 object 与 hand loss 同时等权，建议先把这两种映射作为两个独立 ablation（或按 target RMS 做 loss reweight），不要假设存在一个同时精确对齐的 stride；hand 的 raw120 精确映射需另做 MANO 重建统计。
- 若目标是相同动作时间或可解释的未来预测，应继续使用物理时间映射（raw `g≈4h`，例如 `h2↔g8`、`h10↔g40`），并考虑显式输入 `delta_time_s` 或预测速度。当前 model forward 未读取 stride/delta-time，故任一映射都应作为明确的训练变量记录。
- 结论状态：幅度映射观察为 `SUPPORTED`；其能否改善最终效果尚未运行实验，科研结论为 `INCONCLUSIVE`。仅新增本活动条目；删除本条即可回滚文档差异，代码、配置、数据、cache、checkpoint 和既有运行均未改动。


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
