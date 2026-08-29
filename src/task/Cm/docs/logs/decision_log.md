# Cm AI 自主决策记录

## 2026-08-28 — 混合手流重建采用独立几何 decoder 与 source 等权校准

- scope: task:Cm / GRAB + Inspire-F1 混合训练
- anchor: `src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml` / 2026-08-28

**未指定点**

用户已确认从 Inspire-F1 decoder-only `latest.pt` 初始化、手流 decoder 采用 object 类似的逐 slot 加和结构、独立 train-only RMS、`lambda_hand=1`、冻结 DenseToken、GRAB/Inspire 等概率、global batch 96 和 50 epoch；未另行指定新阶段学习率及是否继承旧 optimizer/scheduler 计数。

**实际选择**

- 仅载入 `latest.pt` 模型权重，optimizer/scheduler/epoch/step 全部重置；沿用当前阶段 lr=`3e-4`。
- 手 decoder 只接收原始手点/法线、Cm、相对 anchor 几何和 anchor 法线；不直接拼接 `z_hand`、contact 或 GT hand flow。
- 对 GRAB train 与 Inspire-F1 train 先各自按 stride 1--10 等权统计，再对 source 等权，得到 hand RMS=`0.06986298856554198 m`、scale=`14.313730639533834`。
- checkpoint 选择继续使用 object `val/mean_stride_epe_mm`；手流 EPE/relative EPE/zero-flow improvement 作为独立诊断。

**选择理由与影响**

fresh optimizer 避免把新 decoder 参数接入旧 Adam moments 和已推进的 cosine schedule；保持 lr 不变使新增变量集中在数据混合和手流辅助监督。source 等权 scale 与训练抽样分布一致，避免样本量更大的 source 主导归一化。保留 object checkpoint 标准可与已有 Cm 曲线连续比较。

**可逆性 / 是否需要用户确认**

源 checkpoint 与既有输出均不修改；手 decoder、loss 权重、scale 和 source mix 都是独立配置项，可关闭或替换。上述关键科研口径均已由用户确认。

## 2026-08-27 — Inspire-F1 采用冻结 DenseToken 的 decoder-only continuation

- scope: task:Cm / HRDexDB 微调
- anchor: `src/task/Cm/configs/active/hrdexdb_inspire_f1_decoder_only_resume.yaml` / 2026-08-27

**未指定点**

用户确认停止全量微调，使用当前最新权重冻结 DenseToken 后继续训练 decoder，并确认不使用 cache；未单独指定是否继承源 optimizer 状态及 step/scheduler 计数。

**实际选择**

- 从 Inspire-F1 全量微调的 epoch 8 / step `19160` checkpoint 载入模型；不再加载 GRAB initializer。
- `freeze_dense_encoder=true`，DenseToken 仍在线 no-grad 前向；`data.use_dense_cache=false`。
- decoder-only optimizer 只包含 `requires_grad=true` 的 decoder 参数；跳过源 DenseToken Adam moments/scaler，scheduler 与 `global_step=19160` 接续。
- continuation checkpoint 显式保留 DenseToken 权重，避免后续恢复时丢失已适配的 encoder。

**选择理由与影响**

这样保持了已完成的 Inspire-F1 DenseToken 适配，同时将后续更新集中到 Cm decoder，显著减少反向图与 optimizer 状态显存；继承 global step/scheduler 可避免重新开始 cosine 进度。在线 no-grad DenseToken 保留了现有随机几何采样下的输入语义，但吞吐仍受前向成本影响。

**可逆性 / 是否需要用户确认**

源全量微调 checkpoint 未覆盖，可随时恢复全量 DenseToken 训练；decoder-only continuation 作为独立输出目录运行。本选择基于用户已明确的“冻结 DenseToken、继续训练 decoder、不用 cache”要求，无额外确认。

## 2026-08-28 — 采用 slot-wise t-SNE 检查 GRAB/Inspire-F1 表征分布

- scope: task:Cm / latent visualization
- anchor: `src/task/Cm/research/tsne_slots.py` / 2026-08-28

**未指定点**

用户指定使用 decoder-only 续训 `latest.pt`、GRAB 与 Inspire-F1 两边 test、固定 stride=5、逐 slot 绘图，但未指定每边样本上限及 t-SNE 的稳定参数。

**实际选择**

- 使用当前 `latest.pt`，每边从 test transition 中确定性等量抽取 512 条；GRAB 按 object-v2 test sequence，Inspire 按 HRDexDB manifest 的 `inspire_f1` test episode。
- 对每个 `cm_tokens[:, slot, :]` 独立运行 t-SNE，16 个 slot 汇总为 4×4 图；保存高维 token、二维坐标和参数元数据。
- t-SNE 使用 `init=pca`、`learning_rate=auto`、`perplexity=30`、`n_iter=1000`、固定 seed；同时记录原始 C=64 空间的 slot silhouette，避免只依据二维图形下结论。

**选择理由与影响**

逐 slot 保留用户要求的局部表征视角，并避免把无固定语义的 slot flatten 后引入排列伪差异。等量抽样防止 GRAB 样本规模主导图形；512/边在当前显存和推理吞吐下可在分钟级完成。t-SNE 与 silhouette 仅用于域分布诊断，不改变训练或 checkpoint 选择。

**可逆性 / 是否需要用户确认**

完全可逆；脚本支持通过 `--max-samples`、`--perplexity`、`--n-iter` 和路径参数重跑，不修改任何训练数据或 checkpoint。

## 2026-08-26 — Inspire-F1 additive 微调解冻 DenseToken并重置预算

- scope: task:Cm / HRDexDB 微调
- anchor: working tree / 2026-08-26

**未指定点**

用户确认微调允许解冻 DenseToken，并选择后者预算；未指定 checkpoint 加载时是否继承源 run 的 optimizer/epoch 计数。

**实际选择**

使用当前 C=64 additive strict checkpoint 初始化模型权重，`freeze_dense_encoder=false`、`data.use_dense_cache=false`；global batch=64、50 epoch、202300 steps、lr=`3e-4`，optimizer/scheduler/step/epoch 全部从微调阶段重新开始。数据仅筛选 manifest 中 `inspire_f1/` 前缀的 446/67/63 split。

**选择理由与影响**

继承 GRAB checkpoint 的 optimizer 和 step 会把新数据训练错误地接到旧 cosine 进度，并且复用旧输出目录；独立初始化入口保留 checkpoint head 权重，同时提供可复现的全新微调预算。解冻 DenseToken 是用户明确允许的训练变量，故不再声称这是“仅换数据集”的严格实验。

**可逆性 / 是否需要用户确认**

完全可逆；基础 checkpoint 不修改，冻结 DenseToken 的 additive 配置仍可独立复现。

## 2026-08-26 — 新 C=32 hard-gate 隔离目标变量并与 C=16 additive 共用 GPU

- scope: task:Cm 训练
- anchor: working tree / 2026-08-26

**未指定点**

用户要求新 C=32 hard-gate 使用“现在相同配置”，并在显存允许时启动 C=16 additive；未指定 GPU 分配及是否允许两条新 run 共用算力。

**实际选择**

C=32 hard-gate 以 C=32 additive strict 为基准，只恢复 hard gate、5+5 warm-up 和原 count/confidence loss；C=16 additive 只改变 `cm_dim`。两条均使用 GPU 2/3、global batch=64 并行运行。

**选择理由与影响**

GPU 2/3 启动前空闲；第一条启动后每卡仅占约 1.8 GB，第二条加入后合计约 3.3 GB，显存余量充分。共用 GPU 避免占用其他用户或现有 Cm run 的卡，但使两条吞吐降至约 `102--104 samples/s`；训练数值预算不变。

**可逆性 / 是否需要用户确认**

两条 run 均为独立配置和输出，可单独停止或迁移；没有修改已有 checkpoint。

## 2026-08-25 — C=32 additive 严格容量对照仅改变 Cm 宽度

- scope: task:Cm 训练
- anchor: working tree / 2026-08-25

**未指定点**

用户要求“C=32 additive，一样配置”，未逐项重复指定运行卡号。

**实际选择**

以当前 C=64 additive strict-budget YAML 为基准，仅覆盖 `meta.cm_dim=32`；保持 global batch=64、50 epoch、202300 steps、loss 和数据条件不变。由于 GPU 1/2 与 4/6 已有 Cm 训练，使用 GPU 0/7 的空闲显存容量启动 DDP。

**选择理由与影响**

这样可以直接检验 Cm 宽度对 additive 多 slot 训练的影响；GPU 0/7 与其他任务共用，吞吐会下降，但不改变 optimizer budget 或模型语义。

**可逆性 / 是否需要用户确认**

完全可逆；新 run 独立输出，不影响 C=64 additive 和 hard-gate checkpoint。

## 2026-08-24 — candidate mixture pilot 采用显式 soft-min 而非 aggregate flow

- scope: task:Cm 训练目标
- anchor: working tree / 2026-08-24

**未指定点**

用户确认采用 candidate-level mixture loss，但未要求保留旧 aggregate flow loss，也未指定 pilot 的预算覆盖方式。

**实际选择**

首轮将 `loss_flow_weight=0`，仅优化 per-slot candidate loss 的 soft-min mixture；关闭 hard gate、count、confidence 和 overlap loss，使用 `tau=0.05`。由于配置继承器保留了基础 epoch 调度，另以显式 `max_steps=16184` 固定约 10 epoch 的单卡 global batch=32 预算。

**选择理由与影响**

若同时保留旧 aggregate loss，candidate mixture 的 slot responsibility 梯度会与“候选平均后再比较 GT”的目标混合，无法判断新目标是否有效；关闭 gate/count 可以先验证 candidate 是否具有可分工性。该选择只改变本 pilot 的优化目标，不改变数据、GT、模型输入和评估指标。

**可逆性 / 是否需要用户确认**

完全可逆；旧配置和 checkpoint 不受影响，用户已确认 C=64、GRAB 和该训练方向。

## 2026-08-23 — 数据工具实现归入 tools/data 并保留兼容模块

- scope: task:Cm 工具入口
- anchor: working tree / 2026-08-23

**未指定点**

用户确认可以分类整理 Cm 的独立数据脚本，但未指定哪些 import/CLI 必须迁移或是否允许破坏旧命令。

**实际选择**

将 sampling bank、split、DenseToken cache、flow calibration 和 object-v2 statistics 六类独立实现归入 `src/task/Cm/tools/data/`；`model/dataset/runner/train/eval` 等运行时模块保持原位。旧根级薄 wrapper 不保留，调用方统一使用新模块路径。

**选择理由与影响**

实现代码按数据维护职责集中，避免在根目录维护无实际逻辑的兼容层；没有改变 dataset schema、训练输入或实验语义。

**可逆性 / 是否需要用户确认**

完全可逆；目录范围已由用户确认，无需再次确认。

## 2026-08-23 — HRDexDB 恢复 Cm candidate 与 source×stride 评估合同

- scope: task:Cm / 共享 HRDexDB geometry layer
- anchor: working tree / 2026-08-23

**未指定点**

现有 HRDexDB smoke cache 只有全表面 512 点，没有 Cm 所需的 4096 点池和逐帧 5cm candidate；验证入口也只有单一固定 stride。

**实际选择**

保留 Cm 的既有 4096 点池、当前手 5cm candidate、运行时 512 点采样和 `obj_valid_mask` 合同；CmDecoder legacy task layer 继续保留独立 512 点字段。HRDexDB 的验证/测试按 source×stride（1/5/10）建立 loader，并由 Cm runner 汇总全局及 source-specific 指标。HRDexDB 使用真实 timestamp delta；flow scale 沿用用户确认的 base scale，不重新 calibration。

**选择理由与影响**

这样修复的是数据入口与评估实现，不改变用户确认的 Cm 监督语义或 loss；真实 timestamp 可保留 HRDexDB 丢帧信息，固定 scale 避免把不同 embodiment 的数值分布误当成需要重新统一的物理目标。

**可逆性 / 是否需要用户确认**

可通过独立 cache 字段、配置和 loader 回退；scale 选择已由用户确认，无需再次确认。

- scope: task:Cm
- last_updated: 2026-08-23
- related: [当前状态](status_log.md)、[架构记录](architecture_log.md)、[实验记录](experiment_log.md)

## 2026-08-23 — 四卡续训释放 GPU 4/7

- scope: task:Cm 运行资源
- anchor: branch `oyx` / 2026-08-23

**未指定点**

用户指定两条训练各缩为两张卡、保持 global batch 144，并授权自主分配卡号，但没有指定释放哪两张卡。

**实际选择**

C=256 保留 GPU 2/3，C=64 保留 GPU 1/6，释放各自原卡组末端的 GPU 4/7；world size 从 3 改为 2，per-device batch 从 48 改为 72。

**选择理由与影响**

保留四张既有训练卡可避免与 GPU 0/5 上的 Viewer、Decoder 和外部任务竞争；global batch 和 scheduler 总步数不变。单卡显存上升但仍有大幅余量，吞吐下降属于减少并行卡数的预期影响。

**可逆性 / 是否需要用户确认**

可通过停止并从最新 checkpoint 用原 3 GPU × 48 入口恢复；global batch 语义由用户明确确认，卡号分配在授权范围内。

## 2026-08-22 — DexYCB 评测保留全部序列并使用非评测占位 train stream

- scope: task:Cm 评测入口
- anchor: branch `oyx` / 2026-08-22

**未指定点**

Cm 共用 loader 强制要求非空 train split，但本次目标是将 subject-10 的全部 50 条 DexYCB 序列作为 held-out test；从 DexYCB 留出 train 会减少测试覆盖。

**实际选择**

test split 保留全部 50 条 DexYCB right stream；train split 放入一条有有效 candidate 的 GRAB stream，只供 Runner 初始化 Dataset metadata，评测过程从不迭代该 train loader。`split.json` 显式记录 `train_placeholder_only=true`。

**选择理由与影响**

不修改共享 loader，也不牺牲目标数据的测试覆盖；GRAB 占位不进入任何 test metric，因此不造成测试混样。代价是 split 不能被误解为可训练的 DexYCB 划分。

**可逆性 / 是否需要用户确认**

完全可逆；若共享 loader 后续允许 eval-only split，可删除占位 stream。本选择不改变 GT、指标或研究目标，无需额外确认。

## 2026-08-18 — V1.2 object cache 首版采用转换器

- branch: 当前工作分支
- post-commit: `HEAD`

**未指定点**

V1.2 要求保留 mmap、ragged candidate 和 B=4 sampling，但未规定是否重算几何。

**实际选择**

复用既有 GRAB/ARCTIC Stage4 object-only NPZ，新增 NPZ 到 mmap/ragged 的转换器。

**其他合理选择**

重写 Stage4，或继续直接使用 NPZ。

**选择理由**

避免重复 MANO、物体采样和近邻计算，满足既定 cache 合同且易验证、易回退。

**对结果的影响**

转换依赖既有 Stage4 输出，不改变模型输入、GT 或 loss。

**可逆性**

高；保留原始 NPZ，删除派生 cache 即可回退。

## 2026-08-19 — V1.2.1 采用共享 epoch、LRU cache 和确定性 split

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

V1.2.1 指导要求修复 worker epoch、缓存泄漏和训练划分，但没有限定具体同步原语、cache 上限和 split 生成策略。

**实际选择**

1. `_MmapSequenceDataset` 用 `mp.Value("q", 0, lock=True)` 共享 epoch；
2. `_cache` 改为 `OrderedDict` LRU，打开序列上限设为 16；
3. 固定 split 采用 sequence 级确定性打散后按比例切分，并写入 `splits.json`。

**其他合理选择**

1. 用线程锁/共享内存封装替代 `mp.Value`；
2. 用无界缓存配合手工清理；
3. 用随机在线切分或单一全局列表切分。

**选择理由**

`mp.Value`、`OrderedDict` 和显式 `splits.json` 都是最小改动、最易验证、最易回退的实现；它们能直接对应指导中的可复现性要求，也更方便后续排查数据泄漏和 worker 状态漂移。

**对结果的影响**

只影响训练稳定性、缓存占用和实验可复现性，不改变模型语义、GT 定义或评估口径。

**可逆性**

完全可逆。

**建议用户确认**

否


## 2026-08-19 — V1.2.1 采用 3 GPU DDP + batch 48 作为正式长训起点

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

用户要求“用多卡一起训练”，但没有指定是 2 GPU 还是 3 GPU，也没有指定是按吞吐还是按效率选卡数。

**实际选择**

正式长训使用 3 GPU DDP（`CUDA_VISIBLE_DEVICES=0,1,5`）并把 mixed/object-v2 full training 的 batch size 设为 48。

**其他合理选择**

2 GPU DDP，或者 batch size 24 / 32 / 64。

**选择理由**

train-only benchmark 显示 mixed 的纯训练吞吐在 3 GPU 下随 batch 增长到 48 时达到峰值（约 366.7 samples/s），64 已回落到约 357.1 samples/s；因此 48 是当前能拿到的最优吞吐点，同时保留了 DDP 的并行收益。

**对结果的影响**

只影响训练时间和每步全局 batch，不改变模型语义、GT 或评估口径；DDP 需保持 `find_unused_parameters=true`。

**可逆性**

完全可逆。

**建议用户确认**

否

## 2026-08-20 — V2 GRAB 数据脚本采用 raw asset 解析与严格 subject template

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

用户要求修正 V2 数据脚本，但没有指定当 `v_template` 缺失时是继续生成还是停止。

**实际选择**

按正式 Cm/V2 数据的语义要求，默认停止并报错；保留显式 `--allow-default-mano` 作为调试/旧流程兼容开关，并让 adapter 同时支持 `dataset/GRAB` 与 `dataset/GRAB/data` 两种 root。

**其他合理选择**

继续静默回退平均 MANO，或在脚本中固定一种 root 布局。

**选择理由**

平均 MANO 会改变手点、手根和当前 candidate mask，结果不能作为 V2 正式实验；严格失败能尽早暴露路径/资产问题。

**对结果的影响**

只改变数据生成的资产解析和错误处理，不改变模型、loss、GT 定义或评估口径；已有错误 cache 不会自动修复。

**可逆性**

完全可逆；可用兼容开关恢复旧回退行为。

**建议用户确认**

否

是（建议时机：启动正式长训前）

## 2026-08-19 — GRAB gate+cm64 候选保持 time condition 并按 50 epochs 比较

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

用户指定 GRAB-only、开启 slot gate、把 slot embedding 维数从 256 改为 64，但未重复指定 time condition、训练 epoch 和 batch。

**实际选择**

保持 `use_time_condition=true` 和其余数据/GT/loss/split/calibration 不变；正式预算沿用当前确认的 50 epochs。batch 48 仅作为吞吐 sweep 起点，正式 batch 要在空闲多卡上重新测定。

**其他合理选择**

关闭 time condition；沿用 10000-step pilot；直接复用 mixed 的 batch 48 而不重新 benchmark。

**选择理由**

保持 time condition 可把用户要求之外的模型变化降到最少；按 epoch 控制可让 GRAB-only 完整遍历次数明确；`cm_dim=64` 会改变显存和计算瓶颈，因此需要重新确定 batch 峰值。

**对结果的影响**

该候选同时改变数据范围、gate 和 slot capacity，是复合候选，不应被解释为单独的 gate 或 cm_dim 消融。50 epochs 保证预算口径清晰，但与 mixed 的样本总量不同。

**可逆性**

完全可逆；使用独立配置，不修改现有 mixed 与 GRAB-only baseline。

**建议用户确认**

否；当前选择遵循用户刚确认的 50-epoch 正式训练口径，启动前仍需确认可用 GPU 不与其他任务冲突。

## 2026-08-20 — gate+cm64 采用全开后渐进启用的 warm-up

- branch: `oyx`
- post-commit: `453806a`

**未指定点**

用户要求增加 gate warm-up，但未指定 warm-up 时长以及从全 slot 到 hard gate 的切换方式。

**实际选择**

前 5 个 epoch 强制 16 个 slot 全部参与，随后 5 个 epoch 将 threshold 从 0 线性升至 0.85、count loss 权重从 0 线性升至配置值；第 11 个 epoch 起恢复完整 gate 配置。

**其他合理选择**

只关闭 gate 若干 epoch 后直接切换；或仅降低 threshold 而不渐进 count loss。

**选择理由**

Hard-Concrete 初始 nonzero probability 约 0.83，而原 threshold 为 0.85，直接切换会再次触发单 slot fallback。全开阶段先让 slot/decoder 学到有区分度的表示，渐进阶段再引入稀疏压力，可避免切换瞬间重新坍缩。

**对结果的影响**

改变 gate 的优化日程，不改变输入、GT、数据划分、评估指标或最终 gate 超参数；该版本应与原 gate+cm64 作为不同训练策略比较。

**可逆性**

完全可逆；warm-up 通过独立配置开关控制，默认关闭。

**建议用户确认**

否（方案已由用户确认）。

## 2026-08-22 — 长程结果对比仅纳入至少 20 epoch 的 run

- scope: task:Cm 结果汇总
- anchor: 2026-08-22

**未指定点**

用户要求排除训练步数太少的实验，但未指定明确阈值；现有结果包含 80-step 吞吐测试、300-step smoke、10000-step/约 2--3 epoch pilot，以及 30--50 epoch 长训。

**实际选择**

主比较只纳入至少完成 20 个完整 epoch 的 run。10000-step pilot 保留在排除表和路径索引中，但不参与主指标排名。

**选择理由与影响**

Cm 各数据 root 的每 epoch step 数不同，直接使用固定 step 阈值会让 mixed 与 GRAB-only 的训练覆盖不可比；以完整 epoch 作为门槛更贴合数据遍历次数。20 epoch 能排除明确标注为 smoke/pilot 的结果，同时保留仍在进行但已有稳定验证趋势的 warm-up run。

**可逆性 / 是否需要用户确认**

完全可逆；只影响汇总表的纳入规则，不改变实验、checkpoint 或原始记录。用户已明确要求排除短训，无需再次确认。

## 2026-08-22 — 新版 mixed cache 采用 sequence 级相对软链接树

- scope: task:Cm
- anchor: 2026-08-22

**未指定点**

用户已指定使用软链接组合修复版 GRAB 与既有 ARCTIC，但没有指定链接粒度、根目录命名和配置命名。

**实际选择**

使用带日期的根目录 `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820`。保留真实的 dataset/subject/sequence 目录，只把每条 sequence 的 `shared/left/right` 目录链接到源 cache；新增同日期独立 YAML，完全继承旧 mixed 的 no-gate、time condition、stride 1--10、batch、epoch 和评估设置，仅替换路径与重新校准的 scale。

**选择理由与影响**

顶层 `grab/`、`arctic/` 目录链接不会被当前递归 glob 扫描，且 resolved split 路径会触发防越界校验；sequence 级链接树无需放宽安全条件，也不复制约 115G 数组。路径进入确定性采样 seed，因此新版根目录必须重新生成全部元数据，不能复制旧 mixed metadata。

**可逆性 / 是否需要用户确认**

完全可逆；删除约 24M 的链接树和独立配置即可回退，不影响两个实体 cache。用户已经确认软链接、复用 ARCTIC、重新生成元数据且暂不启动训练。

## 2026-08-23 — Cm 采用全量 HRDexDB 的 C=64 方案 A 微调框架

- scope: task:Cm
- anchor: `src/task/Cm/configs/active/hrdexdb_finetune_cm64.yaml`

**未指定点**

HRDexDB 内部不同机器人手型与 MANO 是否需要再做分层采样，以及 DenseToken 是否在 base 阶段解冻。

**实际选择**

纳入 HRDexDB 的 MANO/人类数据和所有可用机器人手型，统一视作一个 HRDexDB source bucket；内部按自然数据量采样，不增加额外 embodiment 配额。GRAB、ARCTIC、HRDexDB 三个外部 bucket 按近似均等概率抽样。先使用已完成的 base C=64 checkpoint，再仅在 HRDexDB 微调阶段解冻 DenseToken；训练和验证仍保持现有 Cm object-flow 语义与随机 stride `1..10`。

**选择理由与影响**

用户已明确要求全量 HRDexDB（不止 Inspire）并要求 MANO 数据也纳入，且不需要内部额外拆分。三 bucket 混合保持 base 数据的梯度存在，同时不会把某个机器人 embodiment 的定义误当成新的 Cm 语义。可训练 DenseToken 的 checkpoint 会包含 `dense_encoder.*`，冻结阶段旧 checkpoint 仍可加载。

**可逆性 / 是否需要用户确认**

数据源、概率和 `freeze_dense_encoder` 都由独立配置控制，可逆；全量 cache 尚未生成，因此当前只搭框架，不启动微调。

## 2026-08-24 — mixed 平台期停止并改用双卡 C=64 容量对照

- scope: task:Cm 运行资源与对照实验
- anchor: branch `oyx` / 2026-08-24

**未指定点**

用户要求把 GRAB C=64 对照切换为双卡并适当增大 batch，但未指定卡号和具体增幅。

**实际选择**

停止已在 epoch 45--48（C=256）和 epoch 35--40（C=64）验证平台期的 mixed 续训，保留最近完整 epoch 48/40 checkpoint；新 C=64 使用 GPU 1/2 DDP，per-device batch 32、global batch 64。

**选择理由与影响**

两条 mixed 曲线近四个 epoch 的 mean stride EPE 已基本持平，继续占用资源不能增加当前对照信息。GPU 1/2 当时空闲，global batch 64 相比原单卡 C=32 的 48 只增加约三分之一，属于小幅增大；其余数据、模型输入、gate warm-up 和优化设置继承 C=32 配置。

**可逆性 / 是否需要用户确认**

可从保留 checkpoint 恢复 mixed，或调整 GPU/batch 重新启动；停止与资源调整均在用户本轮授权范围内，无需额外确认。
## 2026-08-24 — 将 candidate mixture 改为 additive slot contributions

- scope: task:Cm
- anchor: `src/task/Cm/configs/active/object_v2_grab_additive_cm64_geometry_only_no_time.yaml`

**未指定点**

- 用户确认不再用 candidate-level soft-min 作为当前主路线，希望保留所有有效 slot 的 additive contribution，并通过轻量复杂度正则学习 effective slot 数。
- 未指定训练阶段是否立即做动态结构裁剪。

**实际选择**

- 保留 `K_max=16` 计算图，训练阶段不使用 hard gate；每个 slot 直接输出 3-D contribution，最终 `pred_flow=Σ_k contribution_k`。
- 使用 aggregate Smooth-L1 作为主监督；关闭 candidate mixture、count、confidence 和 overlap loss。
- 使用 group sparsity `loss_slot_group_sparsity_weight=1e-3`，按每个 slot 的 contribution vector norm 聚合；训练后再依据全局 contribution usage 做结构化 pruning。
- 直接输出 contribution 而不是显式 `a_k·Δf_k`，从参数化上避免乘法尺度不可辨识。

**选择理由与影响**

- 训练目标与部署输出保持一致；不会再出现 candidate soft-min 监督而 weighted-average 评估的语义错位。
- group sparsity 只学习 effective slot 数，暂不减少训练期计算；实际 K 的降低留到 checkpoint 后剪枝和短暂 fine-tune。

**可逆性 / 是否需要用户确认**

- 可逆：通过 config 恢复 legacy aggregate、hard-gate 或 candidate-mixture 路线；本次 additive 结构属于已由用户确认的研究方向，group sparsity 权重是轻量可调工程选择。

## 2026-08-25 — 将 additive pilot 对齐 C=64 hard-gate 的完整预算

- scope: task:Cm 训练对照
- anchor: `src/task/Cm/configs/active/object_v2_grab_additive_cm64_geometry_only_no_time_budget50.yaml`

**未指定点**

- 用户要求“按照相同预算再训练”，未单独指定 GPU；旧 hard-gate 使用 global batch=64、50 epoch、202300 steps。

**实际选择**

- 严格复用 global batch=64、50 epoch、202300 steps、AdamW/cosine 设置；使用空闲 GPU 4/6，per-device batch=32。
- 保持 additive contribution、正确 group-sparsity 归一化和 `1e-3` 权重不变，只改变预算与 global batch。

**选择理由与影响**

- 这样同 epoch 的数据遍历次数、optimizer 更新密度和 LR schedule 可与旧 C=64 对照直接比较；短预算单卡版本保留为探索性结果，不再作为严格结论。

**可逆性 / 是否需要用户确认**

- 可逆；旧短预算 output 保留，strict-budget run 可独立停止或继续，不覆盖任何旧 checkpoint。
