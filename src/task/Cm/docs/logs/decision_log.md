# Cm AI 自主决策记录

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
- anchor: `src/task/Cm/configs/hrdexdb_finetune_cm64.yaml`

**未指定点**

HRDexDB 内部不同机器人手型与 MANO 是否需要再做分层采样，以及 DenseToken 是否在 base 阶段解冻。

**实际选择**

纳入 HRDexDB 的 MANO/人类数据和所有可用机器人手型，统一视作一个 HRDexDB source bucket；内部按自然数据量采样，不增加额外 embodiment 配额。GRAB、ARCTIC、HRDexDB 三个外部 bucket 按近似均等概率抽样。先使用已完成的 base C=64 checkpoint，再仅在 HRDexDB 微调阶段解冻 DenseToken；训练和验证仍保持现有 Cm object-flow 语义与随机 stride `1..10`。

**选择理由与影响**

用户已明确要求全量 HRDexDB（不止 Inspire）并要求 MANO 数据也纳入，且不需要内部额外拆分。三 bucket 混合保持 base 数据的梯度存在，同时不会把某个机器人 embodiment 的定义误当成新的 Cm 语义。可训练 DenseToken 的 checkpoint 会包含 `dense_encoder.*`，冻结阶段旧 checkpoint 仍可加载。

**可逆性 / 是否需要用户确认**

数据源、概率和 `freeze_dense_encoder` 都由独立配置控制，可逆；全量 cache 尚未生成，因此当前只搭框架，不启动微调。
