## 2026-08-24 — OakInk 以官方 root quaternion 重建手根系并转换 translation 语义

- scope: task:correspondence_ptv3_v2 / OakInk Stage 3
- anchor: branch `oyx` / 2026-08-24

**未指定点**

用户要求重新导出为“真正的手坐标系”，但未指定 OakInk wrist-position `hand_tsl` 与训练端 `smplx.MANO.transl` 的语义差异如何处理，也未要求覆盖旧产物。

**实际选择**

使用官方 `general_info` 的 root quaternion、`hand_tsl` 和 `cam_extr` 构造完整 hand-root pose，对手物几何共同执行逆 SE(3)。保留旧目录，新建独立 true-hand-root 目录。为使训练端 MANO forward 与官方 `hand_v` 一致，将 `mano_transl` 写为 `hand_tsl - shaped_wrist_template_offset`，而 `hand_root_pose` 的 translation 仍保存官方 wrist 世界位置。

**选择理由与影响**

仅减 wrist 会保留 camera rotation，不能满足跨数据集统一手根系；直接复制 `hand_tsl` 又会让 `smplx.MANO` 重建整体偏移约 96 mm。上述表示使存储几何、root pose 和 runtime MANO reconstruction 三者一致，并保留旧实验可复现性。

**可逆性 / 是否需要用户确认**

完全可逆；旧目录未覆盖。该选择实现用户已明确的 true-hand-root 目标，不改变训练协议，因此无需额外确认。

## 2026-08-21 — ARCTIC MANO 小子集采用 axis-angle45 5mm 标定并做三条件评估

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`

**未指定点**

用户只要求先导出部分带 MANO 的 ARCTIC 数据并评估，没有指定导出规模、hand noise 表示和条件矩阵。

**实际选择**

复用已有 min11 物体清单，每个物体取一个 `s01` 左手文件，共 11 文件 / 4175 帧；从 raw `.mano.npy` 重新生成 Stage 2/3 到 NAS。ARCTIC 使用 axis-angle45，不转换成 PCA；沿用 ARCTIC 9mm geometry calibration，并按 `5/9` 缩放到 5mm RMS。评估 `object_only`、`hand_only`、`hand_object` 三个条件，关闭 runtime resampling。

**其他合理选择**

1. 直接把 ARCTIC axis-angle45 转换为 GRAB PCA24 后加噪声。
2. 只做 hand-only，不做 joint condition。
3. 导出多个 subject 或全量 ARCTIC。

**选择理由**

axis-angle45 是 ARCTIC 原生 MANO 表示，避免引入 PCA basis/mean 的额外转换；min11 已覆盖全部 object 类别且控制评测成本；三条件能区分 hand-only 学习和 object/compound exposure。

**对结果的影响**

结果适合判断方向和暴露分布问题，不代表跨 subject 或全量 ARCTIC 结论；5mm RMS 与 GRAB PCA5mm 是几何尺度对齐，不是参数分布完全相同。

**可逆性**

完全可逆；NAS 输出使用独立版本目录，不覆盖旧 Stage 3。

**建议用户确认**

否

## 2026-08-20 — V1 object-macro 通过按 object 单独评估再取平均

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`

**未指定点**

V1 要求报告 object-macro，但现有 evaluator 只直接输出全局 micro，没有对象维度的汇总接口。

**实际选择**

先用确定性分层子集覆盖全部 object / subject / action，再为每个 object 单独建立软链子集，分别跑 pure GRAB 与 GRAB+ContactPose，最后对 11 个 object 的结果取算术平均作为 object-macro。

**其他合理选择**

1. 修改 evaluator 增加对象维度的汇总输出。
2. 只报告 micro，不补 object-macro。
3. 用 object 频率加权平均近似 macro。

**选择理由**

不改 evaluator 可以保持评测协议稳定，且 per-object 重跑得到的 macro 口径直接、可复查。算术平均符合 V1 的 object-macro 语义。

**对结果的影响**

object-macro 与 micro 会略有差异，但都来自同一套协议和同一批 object 子集，便于判断是否存在单一高频 object 偏置。

**可逆性**

完全可逆

**建议用户确认**
否

## 2026-08-20 — ARCTIC 时间受限评测采用 subject/object/action 确定性分层

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`

**未指定点**

用户要求在全量 ARCTIC 或覆盖全部物体类别的分层子集上评估，但未指定时间受限子集的每层采样规则。

**实际选择**

全量评估优先；只有 GPU 时间受限时才使用分层子集。分层按 `(subject, object)` 覆盖全部可用组合，每层优先各取字典序最小的 `grab` 和 `use` paired sequence，并保留实际存在的左右手文件。

**其他合理选择**

1. 每个物体类别固定相同帧数，不强制覆盖所有 subject。
2. 按全量 ARCTIC 的类别频率等比例抽样。
3. 完全随机抽取后用 seed 固定结果。

**选择理由**

当前旧子集因字典序截取而全部落在 `box_*`。显式覆盖 subject、object 和 action 能直接消除该类别偏置；确定性字典序规则易复现，且不依赖额外随机状态。

**对结果的影响**

分层结果更适合做类别 macro 比较，但其自然类别频率与全量 ARCTIC 不同，不能代替全量 frame-weighted micro 指标。正式结论仍优先使用全量评估。

**可逆性**

完全可逆

**建议用户确认**

否

## 2026-08-20 — 三域首轮混训采用独立 Dataset 与等比例 batch sampler

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`

**未指定点**

用户指定了三个域各占 1/3，但没有指定是把文件拼到一个目录后加权采样，还是保留三个 Dataset 并在 batch 级别控制比例；也没有指定首轮 batch size 和验证 loader 的组织方式。

**实际选择**

新增 `DomainConcatDataset` 与 `DomainBalancedSampler`：每个本地 batch 固定使用三个域相同数量的样本，DDP 时按 rank 切分每个域的采样流。首轮配置使用 batch size 48（每域 16），保留统一 object pose perturbation，关闭 MANO 重建/手部扰动、runtime object resampling 和 scale augmentation。每个域单独构造 clean/perturbed validation loader。

**其他合理选择**

1. 将三个目录软链接到一个目录，使用普通随机 sampler；
2. 使用带 replacement 的 PyTorch WeightedRandomSampler；
3. 训练只保留一个混合 validation loader，再由 batch 内 dataset id 统计域指标。

**选择理由**

batch 级等比例采样能直接保证每个 optimizer step 的域贡献，且不会被数据集 frame 数量支配；独立 validation loader 能暴露某一域退化，不需要修改 evaluator 的指标语义。首轮不引入自适应 loss/gradient balancing，避免把采样问题和优化调节问题混为一个实验。

**对结果的影响**

相比普通 concat sampler，小域会被重复循环，大域不会按原始数量占主导；因此训练预算应按 optimizer step 和每域曝光量记录。batch size 48 在单卡时 global batch 为 48，在双卡时为 96，后续与纯 GRAB 做严格比较时需显式记录这一点。

**可逆性**

完全可逆；删除 `data.domain_paths` 即回到原有单域 loader。

**建议用户确认**

否；这是对“1/3 domain sampling、不开 runtime、不加 scale”的直接工程实现。

## 2026-08-19 — 5 mm no-PCA runtime 训练以 5 mm 配置为底，仅关闭 hand perturb

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`

**未指定点**

新 run 应该在 5 mm 配置基础上保留哪些字段，尤其是 `exclusive_hand_object_perturb`、`runtime_resample_object` 和训练超参数是否都保持不变。

**实际选择**

以 `full_grab_50ep_geometry_5mm_exclusive_ddp2.yaml` 为底，只把 `meta.apply_hand_perturb` 设为 `false`，其余训练预算、数据路径、object perturb 与 runtime resampling 全部沿用 5 mm。

**其他合理选择**

1. 额外把 `hand_perturb_prob` 设为 `0.0`。
2. 直接复制旧 no-PCA 配置，再手动补回 5 mm 的 runtime sampling 语义。

**选择理由**

用户要求“配置和这版 5mm 一致”，因此最保守的做法是只关闭 PCA 扰动本身，避免把训练 recipe 一起改掉，保证和 5 mm 的对照关系清晰。

**对结果的影响**

如果结果变化，基本可以归因到“是否施加 hand perturb”这一项，而不是 batch、LR、epoch 或 runtime 采样预算。

**可逆性**

完全可逆

**建议用户确认**

否
## 2026-08-21 — H50/O50 首轮先使用 GPU 0 单卡启动

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`

**未指定点**

用户确认了 H50/O50 互斥比例，但未指定必须等待空闲 GPU、共享已有 GPU，还是先用单卡启动。

**实际选择**

在不终止其他任务的前提下，先用物理 GPU 0 单卡启动 H50/O50，覆盖 `train.distributed.enable=false`，保留 `train.max_steps=454000`。

**其他合理选择**

- 等待 GPU 3 空闲后按原 DDP2 运行；
- 与 GPU 3 上的其他任务共享显存启动 DDP2；
- 先只创建配置，不启动训练。

**选择理由**

GPU 1/2/3 当时均有活动进程；共享 GPU 3 可能引入显存和吞吐干扰，终止其他任务也不符合当前协作约束。GPU 0 剩余显存足够完成单卡启动，因此先取得 H50/O50 的可运行性和中期曲线。

**对结果的影响**

单卡 global batch 从原 DDP2 的 32 降为 16。虽然 optimizer step 上限仍为 454000，但每步样本数减半，不能把该 run 解释为严格 matched-sample budget；后续正式比较应在空闲两卡上复跑或按样本数重新对齐。

**可逆性**

完全可逆；配置不改变旧 checkpoint，可停止后在 DDP2 重新开始。

**建议用户确认**

否；待 GPU 资源允许时自动改用 DDP2 复跑即可。
## 2026-08-24 — HOCap 首版按序列/物体实例/手拆分为外部 Stage 3

- scope: task:correspondence_ptv3_v2
- anchor: working tree / 2026-08-24

**未指定点**

用户要求直接开始 HOCap 转换验证，但未指定一个 HOCap 序列中多个物体实例如何映射到当前“一样本一个物体、一只手”的 Stage 3 契约。

**实际选择**

每个 HOCap 序列的每个 `object_id` 和每只有效手单独写一个 Stage 3 NPZ；物体使用 `cleaned_mesh_10000.obj` 的固定 4096 表面采样，手使用 HOCap 的 MANO PCA45、`flat_hand_mean=false` 和标准 1538 面拓扑；接触距离由 hand-root 坐标中的几何最近邻派生。HOCap 只作为外部测试数据，不加入当前 mixed 训练。

**选择理由与影响**

不把同一序列的多个物体错误拼成一个 correspondence 目标，并保留对象 ID 维度以便后续按物体宏平均；该接触目标是几何派生量，不应解释为 HOCap 人工接触标注。

**可逆性 / 是否需要用户确认**

输出使用独立 NAS 版本目录，完全可逆；后续可以改为按交互物体筛选或保留完整多物体场景，不覆盖原始 HOCap 数据。
## 2026-08-25 — HRDexDB human 单 episode 先走 clean Stage 3 smoke

- scope: task:correspondence_ptv3_v2 / HRDexDB 数据适配
- anchor: working tree / 2026-08-25

**未指定点**

用户要求先使用一部分 HRDexDB 打通链路，但未指定 episode、输出位置和是否立即启用 MANO reconstruction。

**实际选择**

选择当前完整度较高的 `human/apple/0`，完整导出 257 帧到 NAS 独立目录；先使用 HRDexDB 已保存的 MANO OBJ 生成 1538 个 face-center 手点，物体网格固定采样 4096 点，暂不写入或启用 MANO reconstruction 字段。

以每帧 MANO 参数中的 `global_orient` 和 `joints[0]` 构造 `T_world_from_hand_root`；human 的 object pose 与 MANO OBJ 已处于同一 HRDexDB world frame，因此不套用机器人 pipeline 的 `C2R.npy`。

**选择理由与影响**

该选择可以先验证 Stage 3 schema、hand-root 变换、法向、接触距离和 `CorrStaticDatasetV2`，同时避免尚未确认 HRDexDB 旋转矩阵/`transl` 语义就把错误 MANO 参数带入训练。它只证明 clean geometry 链路，不证明 MANO forward/hand perturbation 已可用。

**可逆性 / 是否需要用户确认**

输出使用 NAS 独立版本目录，不覆盖现有数据和训练配置；后续可在同一 episode 上补做 MANO reconstruction 校验。
## 2026-08-25 — HRDexDB human MANO 参数验证后启用 reconstruction 版本

- scope: task:correspondence_ptv3_v2 / HRDexDB-human
- anchor: working tree / 2026-08-25

**未指定点**

上一版 smoke 尚未确认 HRDexDB `flat_hand_mean` 和矩阵参数能否与训练端 `smplx.MANO` 对齐，也未指定是否立即写入 MANO 字段。

**实际选择**

用标准 `MANO_RIGHT.pkl`、`flat_hand_mean=True`、`use_pca=False`，将 HRDexDB 的 `global_orient` 和 15 个局部 3×3 矩阵转换为 axis-angle45；保留原始 `transl`、`betas` 和标准 `v_template`，新增带 MANO 字段的独立 smoke 版本。

机器手暂不伪装成 MANO；其扰动设计保留为单独的 URDF/FK q-space 分支，使用固定 link/barycentric hand-point binding，并对 q 加受限噪声。

**选择理由与影响**

257 帧重建误差达到数值精度，说明 HRDexDB human 可以使用当前 reconstruction/hand perturbation 接口。机器人手的点语义和运动学不同，必须单独处理；但其 qpos、URDF、固定点绑定和扰动后的几何位移已经具备实现基础。

**可逆性 / 是否需要用户确认**

带 MANO 字段的输出仍使用独立 NAS 版本目录，未修改 mixed 配置。机器人 q-space runner 分支和 human 批量导出仍需在用户确认后继续。

## 2026-08-25 — 按 hand-joint q-space 扰动接入 Inspire DFTP

- scope: task:correspondence_ptv3_v2 / HRDexDB-robot
- anchor: working tree / 2026-08-25

**未指定点**

用户确认“按照建议的扰动形式”，但未要求把机械臂基座/臂关节也纳入扰动。

**实际选择**

仅对 Inspire DFTP 的六个 hand q joints `[6..11]` 施加独立高斯噪声（当前 smoke 标准差 `0.03 rad`），按 URDF lower/upper limits clip；arm/base qpos 固定，使用固定 link-local surface binding 执行 FK 并回到 clean hand-root。

**选择理由与影响**

任务目标是模拟手部观测几何误差，同时保持样本的 hand-root 坐标契约和物体相对关系稳定。扰动 arm/base 会改变 hand-root，本质上混入坐标系扰动，需要另设 root pose 和物体同步变换，不适合作为第一版 robot corruption。该路径与 MANO/PCA 分支隔离，GT 保持 clean robot surface。

**可逆性 / 是否需要用户确认**

配置默认关闭，且输出位于独立 NAS smoke 目录；可通过增加 arm/base q-space 分支扩展，不影响现有 MANO 或 mixed run。用户已确认继续实现，无额外确认项。

## 2026-08-25 — minimal allhands Stage3 适配范围与 pose 选择

- scope: task:correspondence_ptv3_v2 / HRDexDB minimal archive
- anchor: working tree / 2026-08-25

**未指定点**

用户确认按四手型、2088 个 C2R-valid episode 开始，但未指定 v1/v2 object pose 同时存在时的优先级，以及三种机器人是否共用同一扰动字段合同。

**实际选择**

适配器优先读取集中式 `object_6d_pose_v2`，缺失时回退 `v1`；human 输出 MANO axis-angle45 reconstruction，Inspire DFTP/F1/Allegro V5 共用 robot-FK Stage3 字段，但按各自 URDF 和 q 维度生成 hand-q index、限位和 link binding。缺失 `C2R.npy` 的 16 个 F1 episode 直接不进入全量清单。

**选择理由与影响**

v2 是当前 HRDexDB 处理链约定，v1 是完整性回退；三种机器人虽然关节数量不同，但都能表达为 arm 6 DoF + hand q-space，统一字段便于 loader/domain 复用，同时保留每种 URDF 的真实 q 维度。输出是独立 Stage3 版本，不覆盖原始 minimal archive 或现有 mixed 数据。

**可逆性 / 是否需要用户确认**

完全可逆；适配器可通过 manifest/errors 复核并重新运行。用户已确认该范围。

## 2026-08-25 — HRDexDB 先按 human/robot 分域验证

- scope: task:correspondence_ptv3_v2 / HRDexDB training integration
- anchor: working tree / 2026-08-25

**未指定点**

用户要求“尝试”HRDexDB，但未要求立即修改当前三域 mixed，也未指定 human 与 robot 是否同一训练配置。

**实际选择**

先建立两套只用于 smoke 的 domain 配置：human 单域启用 MANO reconstruction；robot 三手型等比例启用 URDF/FK q-space 扰动。仅运行一批真实 forward/backward，不启动长时间训练。

**选择理由与影响**

当前 reconstruction 开关和字段校验是全局 meta 级别，将两类样本混在同一 loader 会产生 MANO/robot 字段合同冲突。分域验证能先确认数据、扰动和模型接口闭环，同时不污染正在运行的 mixed baseline。

**可逆性 / 是否需要用户确认**

配置为新增文件，未改变现有训练入口；后续可在确认训练策略后分别 fine-tune，或再实现显式按样本类型 dispatch 的统一 loader。

## 2026-08-29 — Object-centered 手侧扰动与惰性坐标转换

- scope: task:correspondence_ptv3_v2 / GRAB + Inspire F1 object-centered training
- anchor: working tree / 2026-08-29

**未指定点**

用户指定“直接给手腕根加扰动”，未进一步规定增量的左右乘法坐标语义；同时两套源 Stage3 合计约 72 GB，完整复制会明显挤占 NAS 可用空间。

**实际选择**

将 10°/10 mm SE(3) 增量定义在局部 hand-root 坐标并右乘到 `T_object_from_hand_root`，实现绕腕根旋转的手侧等效扰动；object-centered 几何转换默认在 Dataset worker 首次读文件时惰性执行，另提供可选离线转换脚本。

**选择理由与影响**

右乘使旋转中心明确落在腕根，符合“给手腕根加扰动”而不是绕物体原点旋转；惰性转换保持旧 Stage3 不变并避免新增约 72 GB 副本，代价是每个 worker 首次加载文件增加一次 CPU 矩阵变换。

**可逆性 / 是否需要用户确认**

均可逆：关闭 `transform_to_object_frame` 或切换坐标系即可恢复旧路径；若用户希望增量改为 object-frame 左乘，只需调整单一采样函数。
## 2026-08-25 — 七域统一 mixed 训练并按样本分流 MANO/robot

- scope: task:correspondence_ptv3_v2 / seven-domain training
- anchor: working tree / 2026-08-25

**未指定点**

用户确认采用 A 版统一混合训练、随机初始化、开启扰动，并要求 HRDexDB 机器手约 10 mm 扰动；未指定 batch size 和机器人各手型共享的 q-space 标准差。

**实际选择**

使用七域等比例 `DomainBalancedSampler`，本地 batch 设为 56（每域 8 个样本）；启用 MANO 与 robot FK 的 mixed sample-wise dispatch、runtime object resampling 和 5 cm clean 交互帧过滤。机器人使用 DFTP/F1/Allegro = 16/10/5 的经验乘数，W&B online，从头训练。

**选择理由与影响**

七域等比例避免 HRDexDB 帧数优势压制其他数据集；batch 56 可被 7 个域整除且与现有单卡显存余量相容。不同机器人 link 几何和 q 限位不同，统一 q-space std 无法稳定达到 10 mm，因此按 FK smoke 的 domain 乘数近似目标，并在训练状态中明确其为约束而非精确保证。5 cm 过滤在 clean 距离上执行，避免把远离物体的帧当成交互监督；runtime 重采样则使用扰动后的手 proxy。

**可逆性 / 是否需要用户确认**

配置为新增文件，数据输出为独立 NAS 目录，不覆盖历史三域产物；可通过配置关闭任一扰动或恢复旧配置。核心研究目标和 GT 语义未改变。

## 2026-08-26 — 运行时物体采样与 4:4:2 扰动协议

- scope: task:correspondence_ptv3_v2 / mixed training
- anchor: working tree / 2026-08-26

**未指定点**

用户明确要求 runtime sampling 不采用近邻/全局配额或 hand-FPS proxy，并指定手扰动、物体扰动、无扰动按 4:4:2 采样。

**实际选择**

先对完整 4096 物体池施加当前样本的 SE(3) 扰动，再从扰动后的观测均匀随机抽取 512 点；训练增强固定为 hand-only 40%、object-only 40%、clean 20%，手与物体噪声不叠加。

**选择理由与影响**

runtime 采样只依赖受扰观测，不把 clean GT 距离或 MANO/机器人拓扑引入输入选择，避免采样策略泄漏监督先验；4:4:2 通过稳定 frame seed 决策，跨 worker、epoch 和 resume 可复现。

**可逆性 / 是否需要用户确认**

均可由配置关闭或调整概率；本次七域配置已明确写入该协议。

补充：复核真实 allhands NPZ 后发现原 DFTP/F1/Allegro 的 16/10/5 乘数在坐标-RMS 定义下仅产生约 5--7 mm；七域配置已改为 25/15/9，使目标更接近 10 mm，并通过 `robot_perturb_rms_m` 按机器人样本记录实际值。

## 2026-08-29 — 采用 object-frame sidecar 与域内序列局部性

- scope: task:correspondence_ptv3_v2 / data I/O
- anchor: working tree / 2026-08-29

**未指定点**

用户要求优化 data-wait，但未指定缓存格式、缓存位置和 worker 并发度。

**实际选择**

在 NAS 上为 GRAB 与 Inspire F1 各构建保持原相对路径的未压缩 object-frame NPZ sidecar，并将训练配置设为 `array_cache_required=true`、每 rank 2 workers、`prefetch_factor=1`。域均衡 sampler 启用域内按文件/序列局部性排序；域间 batch 配额和样本扰动协议不变。

**选择理由与影响**

sidecar 消除运行时 NPZ 解压和整序列坐标变换；按文件排序提升 worker 单文件缓存命中率。实测 8 workers 会进入 NFS D-state，2 workers + 局部性在当前 NAS 上将 data-wait 从约 80--85% 降至约 0.03%，总 ETA 从 260--320 小时降至约 50 小时。缓存实际占用约 92 GB，源数据只读。

**可逆性 / 是否需要用户确认**

该选择仅改变 IO 表示与 frame 顺序，不改变字段、GT、坐标语义、域比例或扰动分布；删除 `array_cache_path`、关闭 `sequence_locality_shuffle` 即可回退，历史训练输出不受影响。

## 2026-08-31 — 七域续训选择最新完整 checkpoint

- scope: task:correspondence_ptv3_v2 / training resume
- anchor: `mixed_seven_domain_mano_robot_10mm_5cm_20260826_061027` / 2026-08-31

**未指定点**

用户要求续训之前的七域混合，但历史目录中存在多次启动痕迹，未指定是否回退到更早 epoch。

**实际选择**

使用唯一包含完整滚动 checkpoint 的最新七域输出目录中的 `checkpoints/latest.pt`，其状态为 step 145000 / epoch 5；沿用原配置、optimizer、scheduler 和保存策略原地续训。

**选择理由与影响**

该 checkpoint 是已验证可加载且训练进度最高的七域状态，避免重复计算已完成的 145k steps；不会改变数据域、扰动协议或模型结构。

**可逆性 / 是否需要用户确认**

可随时停止并从同目录任一 `step_*.pt` 回退；若用户希望从更早 epoch 重跑，再显式指定对应 checkpoint 即可。

## 2026-09-01 — 扩展 object-centered 多域训练采用七域等权与 OakInk2 object-disjoint split

- scope: task:correspondence_ptv3_v2 / expanded object-centered training
- anchor: `mixed_grab_arctic_hrdexdb_oakink2_object_centered.yaml` / 2026-09-01

**未指定点**

用户确认加入全量 ARCTIC、HRDexDB 四手型和 OakInk2，并从头训练，但未指定多域采样权重、OakInk2 存储布局和验证划分的具体实现。

**实际选择**

使用七域等权 sampler；OakInk2 每个序列×物体×手侧单独导出，按 object ID 做确定性 train/val 划分；保持 object-centered、4096→512 runtime sampling、手根/手姿态/clean 4:4:2 互斥扰动和 50 epochs/5000-step 保存。

**选择理由与影响**

等权沿用已有七域训练的跨域宏平均口径，避免 ARCTIC/OakInk2 帧数主导训练。object-disjoint 验证能检验未见物体泛化。OakInk2 的 canonical object pool 对每个文件只存一次并在 Dataset 广播，节省重复存储而不改变模型输入。

**可逆性 / 是否需要用户确认**

配置、split manifest 和导出目录均独立，可改回按序列划分或调整域权重；上述方案已由用户确认后执行。
