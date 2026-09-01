# Cm 实验记录

- scope: task:Cm
- last_updated: 2026-08-29
- last_verified: 2026-08-29
- related: [当前状态](status_log.md)、[架构记录](architecture_log.md)、[接手记忆](repo_memory.md)、[V1.2.1 指导](../指导/V1.2.1.md)

## EXP-025 — 当前 best 的 source-specific stride t-SNE

### 日期

2026-09-01

### 假设与边界

当前训练中 Inspire-F1 的数据基础 stride 为 2；若直接按 1..10 统计会把数据 stride 与训练跨度混淆。使用统一 t-SNE 坐标，GRAB 统计 stride 1..10，Inspire-F1 统计偶数 stride 2..20，并按 dataset、stride、hand-flow RMS 和 object-flow RMS 着色。t-SNE 仅作探索性诊断。

### 输入、命令与产物

- checkpoint：`outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222/checkpoints/best.pt`（epoch 10 / step 42630，`coordinate_frame=object_pose_t`）；
- test cache：checkpoint config 自动解析的 GRAB object-pose cache 与 Inspire-F1 v4 cache；
- 正式命令：`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.Cm.research.tsne_slots.run --checkpoint outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222/checkpoints/best.pt --output-root output/research/cm_tsne_current_best_stride_grab1_10_inspire2_20_20260901 --device cuda --max-samples 512 --n-iter 1000 --batch-size 32`；
- 产物：`output/research/cm_tsne_current_best_stride_grab1_10_inspire2_20_20260901/` 下 pooled、slot、natural stride、hand-flow 渐变、source-specific fixed control、matched 图及 `tsne.npz`、`metadata.json`、`run_manifest.json`。

### 结果与结论

- GRAB/Inspire-F1 各 520 条；逐 stride 均为 52 条，范围分别为 `1..10` 与 `{2,4,...,20}`；metadata 已记录 source-specific stride 集合；
- pooled 原空间 dataset silhouette=`0.0916688`；2 mm hand-RMS bins matched 为 267 条/source，连续 `[4,6] mm` matched 为 20 条/source；
- 正式运行及 object-pose_t、cache 和 stride 标签 smoke 均成功。结论状态：`SUPPORTED`（诊断实现与产物有效，不将 t-SNE silhouette 解释为训练收敛指标）。

## EXP-024 — object_pose_t Cm checkpoint 的 GRAB/Inspire-F1 t-SNE

### 日期

2026-08-31

### 假设与边界

使用新的 `object_pose_t` 坐标合同 checkpoint，在匹配的 object-pose cache 上重跑 GRAB/Inspire-F1 test 的 stride 1--10 pooled-Cm t-SNE。该结果不能与此前 `hand_root_t` 图直接做数值比较；t-SNE 仍只用于表征分布诊断。

### 输入、命令与产物

- checkpoint：`outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/checkpoints/best.pt`（epoch 32 / step 136480，`coordinate_frame=object_pose_t`）；
- cache：`data/processed_data/cm_object_v2_surface512_object_pose_20260830` 与 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4`；脚本从 checkpoint config 自动解析这些路径、坐标系和 candidate/activity mask；
- 每 source 每 stride 52 条，共 520/source、1040 条；
- 产物：`output/research/cm_tsne_object_pose_20260831/` 下 pooled、slot、natural stride、hand-flow 渐变、fixed stride=5、matched 图及 `tsne.npz`、`metadata.json`、`run_manifest.json`。

### 结果与结论

- `object_pose_t` smoke 与正式运行均成功；pooled 原空间 dataset silhouette=`0.1202`；fixed-stride hand-flow bins matched 为 189 条/source，连续 `[4,6] mm` matched 为 20 条/source；
- 新坐标下自然 stride dataset 图仍显示 GRAB/Inspire 主体区域差异及局部重合。由于坐标合同、cache、checkpoint 均变化，不能把 `0.1202` 与旧 `hand_root_t` 的 `0.1137` 解释为表征变好或变坏；结论状态 `SUPPORTED`（运行和分布诊断完成），不代表坐标变更的因果效果。

## 当前研究状态

V1.2 object-only GRAB + ARCTIC 的 full-data Stage4、object-v2 cache、B=4 sampling bank 和 E1 统计已经完成。2026-08-20 确认旧 GRAB cache 因 raw asset root 解析错误使用平均 MANO template；修复版 GRAB 已重建。2026-08-22 已用修复版 GRAB 和既有 ARCTIC 建立新版 mixed 链接 cache并重算元数据；新版 mixed C=256/C=64 续训已在验证平台期停止，最近完整 checkpoint 分别为 epoch 48/40。旧 gate+cm64 warm-up 在 epoch 38 按用户要求停止。DexYCB subject-10/right 修复 cache 已重建，并用成熟 C=256 checkpoint 完成正式跨数据集评测。2026-08-23 新增 GRAB object-only、C=32、物体侧 geometry-only、无时间条件的 hard-gate 瓶颈实验；2026-08-24 启动仅将 `cm_dim` 改回 64 的双卡容量对照。

## EXP-023 — GRAB/Inspire-F1 pooled-Cm t-SNE 与 stride/幅度对照

### 日期

2026-08-30

### 假设与边界

在同一 Cm checkpoint 上合并 GRAB 与 Inspire-F1 test transition，stride 1--10 等量抽样；将 16 个 slot mean-pool 后统一 fit 一次 t-SNE，并通过 dataset、stride、hand-flow RMS 与 object-flow RMS 的同坐标着色，检查 domain 分离是否主要由 motion magnitude 造成。固定 stride=5 和 hand-flow RMS 4--6 mm 的 matched 子集作为辅助对照。t-SNE 仅作探索性可视化，不作为训练收敛或 checkpoint 选择指标。

### 输入、命令与产物

- checkpoint：`outputs/cm/cm_hrdexdb_inspire_f1_decoder_only_resume_20260827_011451/checkpoints/latest.pt`；
- 数据：GRAB test 与 HRDexDB Inspire-F1 test；每 source 每 stride 52 条，共 520/source、1040 条；
- 命令：`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.Cm.research.tsne_slots --checkpoint outputs/cm/cm_hrdexdb_inspire_f1_decoder_only_resume_20260827_011451/checkpoints/latest.pt --output-root output/research/cm_tsne_inspire_f1_latest_stride1_10 --device cuda --max-samples 512 --n-iter 1000 --batch-size 32`；
- 产物：`output/research/cm_tsne_inspire_f1_latest_stride1_10/{pooled_tsne_panels.png,slot_tsne.png,fixed_stride5_tsne_panels.png,matched_hand_magnitude_tsne.png,matched_hand_bins_stride5_tsne.png,tsne.npz,metadata.json}`。

### 结果与结论

- pooled 原空间 dataset silhouette=`0.0959`，逐 slot silhouette 均值=`0.0917`；t-SNE dataset panel 视觉上仍能看到 GRAB/Inspire 的明显区域差异，而 stride panel 各颜色充分混合；
- hand/object flow 着色显示高幅度点主要集中在少数区域，但不能仅凭此图断言 domain 分离完全由幅度解释；
- matched `[4,6] mm` 子集仅剩 16 条/source，样本过少，结论为 `INCONCLUSIVE`，后续应扩大匹配区间或按分位数匹配后再判断；
- 补充固定 stride=5 定量核对：GRAB/Inspire hand-flow RMS 均值=`54.42/7.28 mm`、中位数=`34.28/4.83 mm`，motion magnitude 确为强 confounder；仅用 hand RMS 的 5-fold AUC=`0.891`。进一步按 2 mm hand-RMS bins 等量匹配后保留 172 条/source，两边 hand RMS 均值=`11.50/11.40 mm`，magnitude-only AUC=`0.481`，但 pooled-Cm domain AUC 仍为 `0.995`、silhouette=`0.101`；object magnitude AUC=`0.553`。因此尺度差异贡献明显，但不足以解释 Cm 的 domain separation。该补充为 sample-level CV，尚未做 episode-grouped CV，结论保持 `INCONCLUSIVE` 到 `SUPPORTED` 之间，不能作最终因果归因；
- 额外保留 loader 的自然 stride 分布（两边各 512 条，不按 stride 等量约束）并单独 fit pooled t-SNE，输出 `all_stride_natural_tsne.png`；stride 计数为 GRAB=`[48,62,44,42,61,50,52,56,42,55]`、Inspire=`[44,55,61,64,45,44,54,44,43,58]`，原空间 silhouette=`0.1137`。图上两域存在明显主体区域，但在中间带有部分重合；
- 本次运行和旧 checkpoint import-path 兼容修复均成功，未修改模型、数据或训练进程。

## 历史证据索引

历史实验原文保留在 [`../../research/log.md`](../../research/log.md)，其中包含 V1.2 smoke、Scene Cache V1/V1.1、DenseToken parity 和 V1.1.2 两层 cache 路线的假设、结果及后续决策。本文档作为规范入口，维护当前状态和正式 EXP 记录。

## EXP-021 — GRAB/Inspire-F1 等权混合的手流重建辅助训练

### 日期

2026-08-28

### 假设与边界

在 Inspire-F1 已适配且冻结的 DenseToken/Cm 权重上，同时使用 GRAB 与 Inspire-F1 等权训练，并要求同一组 Cm 以独立几何 decoder 重建手流，检验手动作场监督能否促使跨数据源 Cm 学到更一致的动作表征。手 decoder 不直接读取 `z_hand`、contact 或 GT hand flow；本实验仍以 object flow 为主模型选择指标，不能仅凭手流 loss 下降宣称 Cm 跨域对齐。

### 实现与配置

- config: `src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml`；
- initializer: Inspire-F1 decoder-only `latest.pt`，epoch `30` / step `71850`，fresh optimizer/scheduler；
- GRAB/Inspire-F1 source 概率=`0.5/0.5`，train stride 从 `1..10` 均匀采样，val/test 固定 `1/5/10`；
- C=64、K=16 additive，DenseToken 冻结且 checkpoint 保留其已适配权重；
- global batch=`96`（GPU 0/1/2，per-device `32`），50 epoch，lr=`3e-4`；
- object/hand scaled Smooth-L1 权重均为 `1`；hand train-only RMS=`0.06986298856554198 m`、scale=`14.313730639533834`；
- checkpoint 仍按 object `val/mean_stride_epe_mm` 选择，另报 source×stride 的 hand EPE 与 zero-flow 对照。

### 启动前验证

- 两源 loader=`411801` virtual rows，概率精确为 `0.5/0.5`，val/test 各 6 个 source×stride loader；
- 源 checkpoint strict 加载时仅缺预期的新 hand decoder 参数；DenseToken 全部冻结；
- 真实混合 batch 32 单卡前向/反向通过，峰值 allocated/reserved=`1709/2300 MiB`；
- Cm slot/HRDexDB/ObjectV2/flow-scale 相关测试共 `21 passed`。

### 结论状态

正式 output: `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324`。用户于 2026-08-29 要求停止三卡训练，torchrun 已终止；最近完整 checkpoint 为 epoch 29 / step `124410`，未完成的 epoch 30 不计入正式结果。epoch 29 validation object/hand mean-stride EPE=`8.2792/2.5630 mm`；运行至停止前无 OOM/NaN。CmDecoder 在 GPU1 并行启动后，Cm step time 曾从约 `300--313 ms` 增至约 `650 ms`，global throughput从约 `306--335` 降至约 `146--148 samples/s`；该资源竞争诊断不改变 epoch 29 的验证指标。

`INCONCLUSIVE`（实现与启动 gate 已通过，等待首个完整 source×stride validation）。

### 当前结论

- V1.2 implementation gate：`SUPPORTED`（真实 smoke 级别）。
- V1.2.1 mixed data-only short training：`INCONCLUSIVE`（实现有效，但 300-step 级别未形成明确效果结论）。
- V1.2.1 single-dataset short training：`INCONCLUSIVE`（GRAB-only / ARCTIC-only 300-step 级别也未形成明确效果结论）。
- V1.2.1 throughput-guided full-data start：`SUPPORTED`（3 GPU + batch 48 为当前吞吐峰值点）。
- DexYCB subject-10 首次跨数据集评估：`INVALID_IMPLEMENTATION`（旧 cache 坐标错误 + epoch 1 checkpoint）；修复版 C=256 重测：`SUPPORTED`（平均 EPE `14.92 mm`，zero-flow 改善 `68.27%`）。
- subject-template 修复版 mixed C=256 正式长训：`INCONCLUSIVE`（训练进行中）。
- subject-template 修复版 mixed C=64 正式长训：`INCONCLUSIVE`（已启动，当前只有 step 100--200 启动证据）。
- GRAB C=32 geometry-only/no-time hard-gate：`INCONCLUSIVE`（实现 gate 已通过，正式训练刚启动）。
- GRAB C=64 geometry-only/no-time hard-gate：`INCONCLUSIVE`（双卡 global batch 64 的严格容量对照已启动，尚无 validation 结果）。
- V1.1.1 downstream cache parity：`SUPPORTED`；feature 逐元素差异归因于 spconv 非确定性，不作为科学反证。
- V1.1.2 全量 DenseToken bank：`INVALID_IMPLEMENTATION`/路线撤回，因资源成本过高而停止，不用于效果结论。

## EXP-013 — GRAB C=64 candidate-level mixture objective pilot

### 日期

2026-08-24

### 假设与边界

在 object-only、geometry-only/no-time 的 C=64 瓶颈设置中，直接使用 candidate-level soft-min mixture objective，使每个 object point 依据 `log(pi_k) - candidate_loss_k / tau` 形成隐式 responsibility；若该目标有效，candidate responsibility 不应像旧 count/confidence gate 一样快速坍缩到单一 slot，同时 aggregate flow EPE 应保持可训练。首轮不叠加 hard gate、slot-count、confidence 或 active-overlap 正则，以隔离 mixture objective 的作用。

### 实现与配置

- config: `src/task/Cm/configs/active/object_v2_grab_mixture_cm64_geometry_only_no_time.yaml`；
- data: `data/processed_data/cm_object_v2/grab`，GRAB object-only，固定 seed42 split，stride 1--10；
- `cm_dim=64`、`num_cm_tokens=16`、`use_object_context=false`、`use_time_condition=false`；
- `loss_flow_weight=0`、`loss_candidate_mixture_weight=1`、`candidate_mixture_temperature=0.05`，所有 gate/count/confidence/overlap 权重为 0；
- 单卡 GPU 7，global batch=32，`max_steps=80920`（按当前 loader 约 8092 steps/epoch，约 10 epoch），从头训练。

### 当前运行与首个 validation

- 正式 output: `outputs/cm/cm_object_v2_grab_mixture_cm64_geometry_only_no_time_20260824_190444`；
- 已完成 epoch 1 validation 并进入 epoch 2，无 OOM/NaN；训练 epoch 1 的 aggregate EPE=`51.33 mm`，zero-flow improvement=`-1.36%`；validation mean stride EPE=`51.50 mm`，stride 1/5/10=`10.74/49.88/93.89 mm`，暂时接近或略差于 zero-flow。
- decoder usage 的 `effective_branch_count=15.44`、global top1=`0.0935`；candidate responsibility entropy（stride 1）=`2.479`，global top1=`0.0846`。相对于 16-slot 均匀分布（熵 `ln(16)=2.773`、top1 `0.0625`），已有轻微集中，但不是单 slot collapse；stride 5/10 responsibility entropy 分别为 `2.184/1.958`，需要后续 epoch 观察。
- 与旧 C=64 hard-gate 对照的 epoch 1 validation（mean stride EPE=`31.36 mm`）相比，当前 mixture 首 epoch 性能明显更差；但 global batch、训练目标和 gate 状态不同，不能作最终容量结论。
- 当前日志中的 `mixture_effective_branch_count=inf` 来自诊断熵公式错误，不影响训练目标；已修正 runner，当前已启动进程仍使用修正前代码，后续新 run 将得到正确值。

### 结论状态

`INCONCLUSIVE`（运行中）。首个完整 validation 后补充 candidate responsibility entropy/effective branch count、aggregate/per-stride EPE 与 zero-flow improvement，再决定是否恢复稀疏 gate 或加入轻量 balance 正则。

## EXP-014 — GRAB C=64 additive slot contribution pilot

### 日期

2026-08-24

### 假设与边界

将每个 slot 从“完整 flow candidate”改为 3-D additive contribution，并直接监督 contribution 总和，能够避免 candidate soft-min 与 weighted-average 输出之间的目标错位；轻量 group sparsity 可以在不使用 hard gate 的情况下减少 effective slot 数。训练阶段保留 `K_max=16`，本实验不声称已经减少实际计算量。

### 实现与配置

- config: `src/task/Cm/configs/active/object_v2_grab_additive_cm64_geometry_only_no_time.yaml`；
- data: GRAB object-only、geometry-only/no-time，C=64、K_max=16、global batch=32；
- `pred_flow=Σ_k contribution_k`，主 loss 为 aggregate scaled Smooth-L1；
- candidate mixture、hard gate、count、confidence、overlap 均关闭；group sparsity weight=`1e-3`；
- max_steps=`80920`，约 10 epoch；训练后根据 contribution usage 再决定是否结构化 pruning。

### 当前运行

candidate mixture pilot 已按用户要求停止。首条 additive run `outputs/cm/cm_object_v2_grab_additive_cm64_geometry_only_no_time_20260824_211051` 因 group-sparsity 错误除以全 batch 有效点总数而按 `INVALID_IMPLEMENTATION` 停止。修正版通过 batch-size invariant 单元测试并在 `outputs/cm/cm_object_v2_grab_additive_cm64_geometry_only_no_time_20260824_230244` 从头训练；epoch 1--9 val mean stride EPE 从 `44.26` 持续下降至 `19.76 mm`（仅 epoch 5 短暂反弹至 `24.29 mm`），zero-flow improvement 从 `7.0%` 增至 `56.0%`。epoch 9 stride 1/5/10 EPE=`5.93/18.81/34.54 mm`；stride-1 contribution effective branch count=`14.11`、top1=`0.106`，没有单-slot collapse，但轻量正则只减少约 2 个 effective slots。当前正在 epoch 10。

### 结论状态

首 run：`INVALID_IMPLEMENTATION`（group-sparsity 归一化错误）；短预算修正版：`INCONCLUSIVE`（已停止，global batch=32/10 epoch）；strict-budget 修正版：`INCONCLUSIVE`（运行中）。当前证据支持 additive aggregate 可训练且避免单-slot collapse，但严格比较需等待 global batch=64/50 epoch 版本。

## EXP-015 — GRAB C=64 additive contribution strict-budget match

### 日期

2026-08-25

### 假设与边界

在保持 additive contribution 语义和正确 group-sparsity 归一化的前提下，将 global batch、epoch budget、总 optimizer steps 和 cosine schedule 对齐旧 C=64 hard-gate，能够区分 batch/schedule 混杂与模型结构本身的影响。

### 实现与配置

- config: `src/task/Cm/configs/active/object_v2_grab_additive_cm64_geometry_only_no_time_budget50.yaml`；
- C=64、K_max=16、GRAB object-only、geometry-only/no-time；
- global batch=64（GPU 4/6，per-device batch=32）、50 epoch、`max_steps=202300`；
- aggregate flow supervision、group sparsity=`1e-3`，candidate mixture/hard gate/count/confidence/overlap 关闭。

### 当前运行

- output: `outputs/cm/cm_object_v2_grab_additive_cm64_geometry_only_no_time_budget50_20260825_093713`；
- 已完成 epoch 8 validation，当前进入 epoch 9，无 OOM/NaN。epoch 8 mean stride EPE=`15.699 mm`，stride 1/5/10=`4.763/14.668/27.666 mm`；zero-flow improvement=`65.0%`。epoch 1--8 mean EPE 为 `20.032, 18.264, 17.398, 16.703, 16.550, 17.061, 15.990, 15.699 mm`。
- epoch 8 effective branch count=`13.59`、global top-1 usage=`13.1%`，仍未出现单-slot collapse；相较 epoch 2 的 effective branch=`13.54`，当前主要是性能继续改善而非 slot 数量进一步坍缩。

### 同 epoch 对比（当前可用验证）

| Epoch | C=32 hard-gate | C=64 hard-gate | C=64 additive strict |
| ---: | ---: | ---: | ---: |
| 1 | 42.995 mm | 31.357 mm | 20.032 mm |
| 2 | 42.960 mm | 19.006 mm | 18.264 mm |
| 8 | — | 15.603 mm | 15.699 mm |

该表只表示当前验证点；epoch 8 时 additive 与旧 C=64 hard-gate 已基本持平（差 `0.096 mm`），且 additive 保持约 13.6 个 effective branches，而 hard-gate 已约 1 个。C=64 additive 的优化目标不同，不能据此宣称最终优于 hard-gate。

### 结论状态

`INCONCLUSIVE`（运行中）。

## EXP-020 — Inspire-F1 C=64 additive decoder-only continuation

### 日期

2026-08-27

### 假设与边界

在 EXP-019 的 Inspire-F1 C=64 additive checkpoint 上冻结已适配的 DenseToken，只继续训练 Cm decoder，以降低反向传播开销；不启用 DenseToken cache，保持现有数据采样、global batch 和 flow/loss 语义。

### 实现与配置

- config: `src/task/Cm/configs/active/hrdexdb_inspire_f1_decoder_only_resume.yaml`；
- source checkpoint: EXP-019 `latest.pt`，epoch `8` / step `19160`；
- `freeze_dense_encoder=true`、`save_dense_encoder_in_checkpoint=true`、`data.use_dense_cache=false`；
- decoder-only optimizer：跳过源 DenseToken Adam 状态，仅注册可训练 decoder 参数；scheduler/global step 从 `19160` 接续；
- 保持 global batch=`64`、C=`64` additive、geometry-only/no-time、Inspire-F1 446/67/63 split。

### 当前运行

- output: `outputs/cm/cm_hrdexdb_inspire_f1_decoder_only_resume_20260827_011451`；
- 已确认只加载 Inspire-F1 source checkpoint，未重新加载 GRAB initializer；当前 step `19200`、epoch `9` 训练中；
- 当前已推进至 step `34500` / epoch `15`，最近完整 validation 为 epoch `14` / step `33530`；decoder-only 在线 DenseToken no-grad 吞吐约 `70 samples/s`，无 OOM/NaN。
- validation mean stride EPE：epoch 9/10/11/12/13/14 为 `3.707/3.873/3.753/3.814/3.756/3.656 mm`；当前最佳为 epoch 14 的 `3.656 mm`，相对全量 DenseToken 阶段 epoch 8 的 `4.205 mm` 改善约 `13.1%`。
- epoch 14 stride 1/5/10 EPE=`1.503/3.615/5.852 mm`；zero-flow improvement=`+2.54%`。effective branch count 约 `12.53`，未见 slot collapse。

### 结论状态

`INCONCLUSIVE`（运行中；目前有改善但验证曲线仍有约 `0.06--0.07 mm` 量级波动，不能宣称已收敛）。

## EXP-019 — Inspire-F1-only C=64 additive DenseToken fine-tune

### 日期

2026-08-26

### 假设与边界

在保持当前 C=64 additive、geometry-only、no-time 的 Cm 语义下，仅将适配数据切换为 HRDexDB Inspire-F1，并在微调阶段解冻 DenseToken。该实验不再使用默认的 GRAB/ARCTIC/HRDexDB 三源混合。

### 实现与配置

- config: `src/task/Cm/configs/active/hrdexdb_inspire_f1_finetune_cm64_additive.yaml`；
- base checkpoint: `outputs/cm/cm_object_v2_grab_additive_cm64_geometry_only_no_time_budget50_20260825_093713/checkpoints/best.pt`；
- Inspire-F1-only manifest prefix `inspire_f1/`：train/val/test=`446/67/63` episodes；
- C=64 additive、`use_object_context=false`、`use_time_condition=false`、`use_slot_gate=false`；
- `freeze_dense_encoder=false`、`data.use_dense_cache=false`，DenseToken 在线参与反向传播；
- global batch=64、50 epoch、fresh `max_steps=202300`、AdamW lr=`3e-4`、cosine、seed42；
- flow scale 沿用 additive base checkpoint：`object_flow_target_scale=10.73039338039742`。

### 当前运行

- output: `outputs/cm/cm_hrdexdb_inspire_f1_finetune_cm64_additive_20260826_091150`；
- loader smoke 已确认 446/67/63 episode split，checkpoint head 权重可加载；DenseToken 缺失的 frozen-stage 参数按设计在线初始化并解冻；
- 正式 DDP 在 GPU 1/3 持续运行，已完成 epoch 8（step `19160`），无 OOM/NaN；当前训练吞吐约 `42.7 samples/s`，日志 ETA 约 `76` 小时。最近一次 validation 为 epoch 8：mean stride EPE=`4.205 mm`，stride 1/5/10=`1.436/4.112/7.067 mm`；按各 stride 相对误差等权计算的 zero-flow improvement=`-4.0%`，其中 stride 1=`-33.7%`、stride 5=`+7.2%`、stride 10=`+14.4%`。由于 stride 1 的 GT 流幅度仅约 `1.07 mm`，该相对指标不能直接解读为所有 stride 都差于 zero-flow；按三 stride 的 mean EPE，模型为 `4.205 mm`，zero-flow 对应均值约 `4.586 mm`。epoch 8 train EPE=`3.984 mm`，effective branch=`11.41`，top-1 usage=`0.191`。结果仍属早期且有波动，尚不能下最终结论。

### 结论状态

`INCONCLUSIVE`（运行中）。

## EXP-016 — GRAB C=32 additive contribution strict-budget capacity match

### 日期

2026-08-25

### 假设与边界

将严格预算 C=64 additive 配置仅把 `cm_dim` 改为 32，检验降低 Cm 宽度后 additive aggregate supervision 是否仍能保持多 slot 并改善 C=32 hard-gate 的性能。

### 实现与配置

- config: `src/task/Cm/configs/active/object_v2_grab_additive_cm32_geometry_only_no_time_budget50.yaml`；
- C=32、K_max=16、GRAB object-only、geometry-only/no-time；
- global batch=64（GPU 0/7，per-device batch=32）、50 epoch、`max_steps=202300`；
- 其余严格继承 C=64 additive：aggregate flow supervision、group sparsity=`1e-3`，candidate mixture/hard gate/count/confidence/overlap 关闭。

### 当前运行

- output: `outputs/cm/cm_object_v2_grab_additive_cm32_geometry_only_no_time_budget50_20260825_144851`；
- 已完成 epoch 6 validation，当前进入 epoch 7，无 OOM/NaN。epoch 1--6 mean stride EPE=`24.719, 18.951, 17.445, 17.720, 16.834, 17.173 mm`；epoch 6 stride 1/5/10=`5.507/15.126/30.884 mm`、zero-flow improvement=`61.2%`。
- epoch 6 effective branch count=`13.76`、global top-1 usage=`10.7%`，未出现单-slot collapse；因 GPU 0/7 与其他任务共用，吞吐低于 C=64 additive。

### 同 epoch 初步对比

| Epoch | C=32 hard-gate | C=32 additive | C=64 additive |
| ---: | ---: | ---: | ---: |
| 1 | 42.995 mm | 24.719 mm | 20.032 mm |
| 2 | 42.960 mm | 18.951 mm | 18.264 mm |
| 3 | 35.952 mm | 17.445 mm | 18.513 mm |
| 4 | 30.581 mm | 17.720 mm | 16.703 mm |
| 5 | 26.204 mm | 16.834 mm | 16.550 mm |
| 6 | 25.096 mm | 17.173 mm | 17.061 mm |

C=32 additive 已稳定显著优于同宽度 hard-gate，且 epoch 2--6 与 C=64 additive 基本接近；当前仍不足以判断后期是否会因 C=32 容量形成更高平台。

### 结论状态

`INCONCLUSIVE`（运行中）。

## EXP-017 — GRAB C=32 hard-gate objective-only strict match

### 日期

2026-08-26

### 假设与边界

在与 C=32 additive strict 完全相同的数据、global batch、optimizer steps 和 cosine budget 下，仅恢复 legacy hard-gate 目标，隔离 additive contribution 相对 hard-gate 的效果。

### 实现与配置

- config: `src/task/Cm/configs/active/object_v2_grab_gate_cm32_geometry_only_no_time_budget50_bs64.yaml`；
- C=32、K_max=16、GRAB object-only、geometry-only/no-time；
- global batch=64（GPU 2/3，per-device batch=32）、50 epoch、`max_steps=202300`；
- `use_additive_slot_contributions=false`、`use_slot_gate=true`、5+5 epoch gate warm-up、count/confidence=`1e-3/0.1`；其余继承 C=32 additive strict。

### 当前运行

- output: `outputs/cm/cm_object_v2_grab_gate_cm32_geometry_only_no_time_budget50_bs64_20260826_004930`；
- 已完成 epoch 5 validation，当前进入 epoch 6，无 OOM/NaN。epoch 1--5 mean stride EPE=`35.774, 24.188, 21.527, 20.855, 18.497 mm`；epoch 5 stride 1/5/10=`6.081/16.889/32.520 mm`、zero-flow improvement=`57.5%`。
- epoch 5 validation 仍处于 full-open warm-up，hard active=`16`、effective branch=`15.28`，尚不能判断 gate 恢复后是否坍缩。与 C=16 additive 共用 GPU 2/3 后吞吐约 `103 samples/s`。

### 当前 objective-only 对照

| Epoch | C=32 hard-gate strict | C=32 additive strict |
| ---: | ---: | ---: |
| 1 | 35.774 mm | 24.719 mm |
| 2 | 24.188 mm | 18.951 mm |
| 3 | 21.527 mm | 17.445 mm |
| 4 | 20.855 mm | 17.720 mm |
| 5 | 18.497 mm | 16.834 mm |

additive 在前五个 epoch 均更优；epoch 5 优势为 `1.663 mm`（约 9.0%）。hard-gate 的关键检验点是 epoch 6--10 gate ramp，当前尚未覆盖。

### 结论状态

`INCONCLUSIVE`（运行中）。

## EXP-018 — GRAB C=16 additive contribution strict capacity match

### 日期

2026-08-26

### 假设与边界

在严格 additive 配置下仅将 `cm_dim` 从 32 降至 16，检验 additive 监督能否继续缓解低维 Cm 的性能退化并保持多 slot 使用。

### 实现与配置

- config: `src/task/Cm/configs/active/object_v2_grab_additive_cm16_geometry_only_no_time_budget50.yaml`；
- 除 `cm_dim=16` 外，数据、输入、global batch=64、50 epoch、`max_steps=202300`、aggregate supervision 和 group sparsity=`1e-3` 均与 C=32 additive strict 一致；
- GPU 2/3、DDP world size=2、per-device batch=32，与 C=32 hard-gate 并行共用显存/算力。

### 当前运行

- output: `outputs/cm/cm_object_v2_grab_additive_cm16_geometry_only_no_time_budget50_20260826_005305`；
- 已完成 epoch 5 validation，当前进入 epoch 6，无 OOM/NaN。epoch 1--5 mean stride EPE=`40.738, 31.244, 20.872, 18.500, 18.215 mm`；epoch 5 stride 1/5/10=`5.578/16.342/32.724 mm`、zero-flow improvement=`59.5%`。
- epoch 5 effective branch count=`8.88`、global top-1 usage=`27.7%`：相比 C=32 additive 同 epoch 的 `13.71/10.2%` 明显更集中，但没有单-slot collapse。两条 run 合计占 GPU 2/3 约 `3.3 GB/卡`，显存余量充足；当前吞吐约 `102 samples/s`。

### 当前容量对照

| Epoch | C=16 additive | C=32 additive | C=64 additive |
| ---: | ---: | ---: | ---: |
| 1 | 40.738 mm | 24.719 mm | 20.032 mm |
| 2 | 31.244 mm | 18.951 mm | 18.264 mm |
| 3 | 20.872 mm | 17.445 mm | 18.513 mm |
| 4 | 18.500 mm | 17.720 mm | 16.703 mm |
| 5 | 18.215 mm | 16.834 mm | 16.550 mm |

C=16 前两轮收敛明显更慢，epoch 5 已追至距 C=32 `1.381 mm`、距 C=64 `1.665 mm`；是否形成更高平台仍需后续 epoch。

### 结论状态

`INCONCLUSIVE`（运行中）。

## EXP-011 — GRAB C=32 geometry-only/no-time hard-gate 瓶颈实验

### 日期

2026-08-23

### 假设与边界

若 object-flow decoder 不直接接收 DenseToken `z_obj/object_context`，而只接收原始物体几何和 `C_m`，当前交互信息将更难绕过 `C_m` 瓶颈；同时去掉 `delta_time_s` 可以将本实验限定为 endpoint action representation，而不是显式动力学建模。本实验只用 GRAB object-only，不包含 environment/scene points。

### 实现与配置

- config: `src/task/Cm/configs/active/object_v2_grab_gate_cm32_geometry_only_no_time.yaml`；
- data: `data/processed_data/cm_object_v2/grab`，固定 seed42 split，stride 1--10；
- `cm_dim=32`、`num_cm_tokens=16`；
- `use_object_context=false`：object edge 输入为 raw point/normal、Cm token、object-to-anchor 和 anchor normal，不使用 `z_obj`；
- `use_time_condition=false`：数据兼容字段 `delta_time_s` 保留但模型忽略；
- Hard-Concrete gate：count/confidence/active-overlap 权重=`1e-3/0.1/0`，前 5 epoch 全开，随后 5 epoch 线性恢复 threshold/count loss；
- DenseToken 保持冻结，训练从头开始，不载入旧 Cm checkpoint。

### 实现 gate

- geometry-only decoder 对不同 `z_obj` 扰动输出不变；edge 输入维度为 `C+12=44`；
- 相关 Cm/CmDecoder tests 共 36 passed；
- 首次在线 W&B 启动因系统时间超过服务端证书有效期而在 step 0 前失败，不构成实验结果；正式 run 改用 offline W&B。

### 当前运行

- GPU 7 单卡，per-device/global batch=48，50 epoch，计划总步数 269750；
- output: `outputs/cm/cm_object_v2_grab_gate_cm32_geometry_only_no_time_20260823_235356`；
- launcher log: `output/exp/cm_v121/cm_object_v2_grab_gate_cm32_geometry_only_no_time_offline_gpu7.log`；
- 已完成 epoch 15 validation，随后在 epoch 16 训练阶段停止，最后完整验证为 mean stride EPE=`21.388 mm`，stride 1/5/10=`6.533/20.078/37.555 mm`，zero-flow improvement=`52.16%`；epoch 14 的 mean EPE=`21.121 mm`，已进入约 21 mm 平台。训练过程无 OOM/NaN。
- epoch 15 的 stride-1 effective branch count=`1.0002`、global top-1 usage=`99.99%`，确认 hard-gate 已坍缩为单 slot；该结果是中途停止证据，不代表完成 50 epoch。

### 结论状态

`INCONCLUSIVE`（中途停止；容量与坍缩诊断证据已形成，未完成完整预算）

### 与 C=64 hard-gate 的对齐观察

- 同 epoch 15：C=64 mean stride EPE=`15.743 mm`，C=32=`21.388 mm`；C=64 低约 `5.65 mm`（约 26%）。
- 同 optimizer step 约 `80920`：C=64 epoch 20 mean EPE=`14.187 mm`，仍优于 C=32 epoch 15 的 `21.388 mm`。
- 两者在 warm-up 后都坍缩到约 1 个 effective slot，因此 C=64 的优势主要体现为容量带来的性能，不是 slot 使用更分散；严格预算 additive C=64 仍需等待 validation。

当前只有实现与启动证据；必须等待固定 stride validation、zero-flow improvement、slot active count/fallback/effective branch 指标后再判断瓶颈假设。

## EXP-010 — DexYCB subject-10 修复版正式评估

### 日期

2026-08-22

### 假设与边界

修复 DexYCB object/MANO 坐标适配并用成熟的新版 mixed C=256 checkpoint 重测后，可以区分“旧实现错误”与“真实跨数据集泛化不足”。本实验仅覆盖 subject-10 的 50 条右手 capture，不代表全部 DexYCB 主体。

### 数据与实现 gate

- cache: `data/processed_data/stage4/data/dexycb`，50 sequences / 50 right streams / 2853 frames，0 skipped/failed；
- all-finite，50/50 序列在 `max_stride=10` 前均有 active current frame；active frame=`2291/2853=80.30%`，旧错误 cache 为 `287/3650=7.86%`；
- object 刚体回代跨帧漂移 mean/max=`3.48e-5/4.06e-5 mm`；法向单位长度误差 mean/max=`2.65e-8/1.79e-7`；
- 与官方 reference-camera label 对齐的最大首帧偏差为 rotation `1.744°`、translation `1.266 mm`；
- split: `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json`，test 保留全部 50 条 DexYCB。框架要求的非空 train split 仅含一条不参与评测的 GRAB 占位 stream，元数据已显式标注。

### 评测配置

- checkpoint: `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_20260822_125835/checkpoints/step_000036900_epoch_000010.pt`；C=256、no-gate + time condition，in-domain best=`13.604 mm`；
- config: `src/task/Cm/configs/active/eval_dexycb_subject10_c256_fixed_20260822.yaml`；
- stride: 1/5/10；batch 32；沿用 checkpoint 的 `object_flow_target_scale=13.645122770626802`；
- C=64 当前仍在早期训练，未纳入主结论。

### 结果

| Stride | GT flow norm | EPE | Relative EPE | Zero-flow 改善 | Pred/GT norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.96 mm | 4.54 mm | 0.415 | 58.54% | 0.869 |
| 5 | 52.35 mm | 14.76 mm | 0.282 | 71.80% | 0.896 |
| 10 | 99.70 mm | 25.46 mm | 0.255 | 74.47% | 0.873 |
| 三 stride 均值 | — | **14.92 mm** | **0.317** | **68.27%** | **0.879** |

旧错误 cache + 未成熟 C=64 checkpoint 的平均 EPE/relative EPE/zero-flow 改善为 `57.58 mm / 1.030 / -3.03%`；两次实验同时改变了数据实现和 checkpoint，因此该差值不能作为 C=64 对 C=256 的 capacity ablation，但足以说明旧“接近 zero-flow”表现不是可信泛化结论。

### 结论状态

`SUPPORTED`

### 解释

修复后模型在三个时间跨度都显著优于 zero-flow，预测流幅值约为 GT 的 87.9%，没有旧评估中的近零输出塌缩。对当前 subject-10/right 而言，泛化并非“太差到不可用”；仍存在约 25%--41% 的 relative EPE，且缺少跨主体方差，不能据此宣称 DexYCB 全数据集泛化已经解决。

### 证据

- 新评测：`outputs/cm/cm_eval_dexycb_subject10_c256_fixed_20260822/eval.log`
- 旧无效评测：`outputs/cm/cm_eval_dexycb_subject10/eval.log`
- cache 与 split：`data/processed_data/stage4/data/dexycb`、`data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json`

## EXP-009 — DexYCB subject-10 首次评估无效性诊断

### 日期

2026-08-22

### 假设

`outputs/cm/cm_eval_dexycb_subject10` 中接近 zero-flow 的结果可能反映 GRAB+ARCTIC 训练模型对 DexYCB 的真实泛化差距。EXP-010 已用修复 cache 证伪这一旧观察；本 EXP 仍保留为实现无效性记录。

### 实现审查与定量证据

- 评估加载 C=64 step 3690 / epoch 1 checkpoint，其 in-domain best metric 仍为 `23.09 mm`；
- 本地 `pose.npz` object quaternion 按 `xyzw` 还原时与 reference-camera 官方 3x4 label 旋转误差仅 `0.0013°`，旧 adapter 按 `wxyz` 解析时误差约 `149.8°`；
- 旧物体点按代码声称的 pose 回代到局部系，48 条 test sequence 的平均跨帧漂移为 `25.54 mm`，使用反向旋转只剩 `3.5e-5 mm`；
- sequence-level `pose_m` 已精确匹配 identity-extrinsic reference camera label，旧 adapter 又按 `serials[0]` 重复施加 extrinsic；
- 旧 adapter 将 `pose_m[3:48]` 的 PCA 系数直接当 axis-angle，且使用 `flat_hand_mean=true`；按官方 PCA basis + non-flat mean 解码后，21 关节与 reference-camera `joint_3d` 直接对齐平均误差为 `0.78 mm`（旧解码约 `12.69 mm`）；
- 旧 cache 50 条右手 sequence / 3650 帧中只有 287 帧（7.86%）存在 5 cm candidate，手物质心距离中位数约 `258.6 mm`。

修复后单条 `20201022_113530` 不落盘 smoke 自动裁掉无标注前缀，保留 raw frame 21--71；物体局部刚体漂移为 `3.6e-5 mm`，51 帧中 43 帧有 5 cm candidate。

### 结论状态

`INVALID_IMPLEMENTATION`

### 解释

旧评估同时受错误 GT/手物坐标和未收敛 checkpoint 影响，不能支持或反驳跨数据集泛化假设。实现错误不记为科研反证。

### 下一步

重建 DexYCB Stage4 cache 与 split，检查全量 right-hand candidate/rigidity 统计，并在 C=64 或 C=256 收敛 checkpoint 上重新评估。

### 证据

- 旧评估：`outputs/cm/cm_eval_dexycb_subject10/eval.log`
- 旧 cache：`data/processed_data/stage4/data/dexycb`
- 修复实现：`process/DexYCB/raw.py`、`process/DexYCB/stage4_cm.py`
- 回归测试：`tests/test_dexycb_raw.py`

## EXP-008 — subject-template 修复版 mixed C=64 长训

### 日期

2026-08-22

### 假设

在新版 mixed cache、训练预算和优化设置完全不变时，仅将 `cm_dim` 从 256 降到 64，可以直接比较 Cm capacity 对效果和吞吐的影响。

### 对照约束

resolved config 静态比较确认只有三处差异：实验名、`meta.cm_dim: 256→64` 和 W&B 标签；数据、split、calibration、no-gate、time condition、batch、学习率、scheduler、loss 与预算均一致。

### 训练配置

- config: `src/task/Cm/configs/active/object_v2_grab_arctic_subject_template_20260820_cm64.yaml`
- data: `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820`
- GPU: `CUDA_VISIBLE_DEVICES=1,6,7`
- world size: 3
- per-device / global batch: 48 / 144
- budget: 50 epochs / 184650 steps
- model: no-gate + time condition，`cm_dim=64`
- initialization: 从头训练

### 启动结果

step 100--200 无 cache/calibration/schema 报错，无 OOM/NaN；grad norm 约 `0.110--0.111`，吞吐约 `237 samples/s`，data wait 低于 0.1%。当前 C=256 与 C=64 共占用六张 GPU，首段吞吐不解释为单模型 capacity 的最终速度结论。

2026-08-23 19:42 CST 按用户要求释放两张 GPU：从最近完整 `latest.pt`（step 84870 / epoch 23 / best `13.024 mm`）恢复到 GPU 1/6，改为 2 GPU × per-device batch 72，global batch 仍为 144。首步确认 optimizer/scheduler 正确恢复，无 OOM/NaN；单卡显存约 `3.5--3.6 GB`、吞吐约 `243 samples/s`。step 84870 之后尚未落盘的约 1.5 epoch 不纳入续训。

### 结论状态

`INCONCLUSIVE`（训练进行中）

### 下一步

保持当前配置运行，使用与 EXP-007 相同的 validation 指标和训练覆盖比较 C=64/C=256。

### 证据

- output: `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_cm64_20260822_190435`
- log: `output/exp/cm_v121/cm_object_v2_grab_arctic_subject_template_cm64_3gpu_bs48_50ep_gpu167_20260822_190414.log`
- resume log: `output/exp/cm_v121/cm_object_v2_grab_arctic_subject_template_cm64_2gpu_bs72_resume_20260823.log`
- metrics: `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_cm64_20260822_190435/metrics.jsonl`
- W&B: `https://wandb.ai/hitsz-oyx/ref2dex/runs/cgtaodph`

## EXP-007 — subject-template 修复版 mixed 正式长训

### 日期

2026-08-22

### 假设

在 EXP-006 的修复版 mixed cache 上复用既有吞吐峰值配置（3 GPU DDP、per-device batch 48），可以不改变模型、loss 和预算地从头训练一版数据语义正确的 mixed baseline。

### Baseline

- 旧吞吐 sweep 峰值：3 GPU、per-device batch 48、global batch 144，`366.732 samples/s`；
- 旧 mixed 正式 run 使用受平均 MANO template 影响的 GRAB，不作为新版数据的有效结果；
- 新 cache 与配置见 EXP-006。

### 训练配置

- config: `src/task/Cm/configs/active/object_v2_grab_arctic_subject_template_20260820.yaml`
- data: `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820`
- GPU: `CUDA_VISIBLE_DEVICES=2,3,4`
- world size: 3
- per-device / global batch: 48 / 144
- budget: 50 epochs / 184650 steps
- model: no-gate + time condition，`cm_dim=256`
- initialization: 从头训练，不复用旧 cache checkpoint

### 启动结果

step 100--1000 已完成：无 cache/calibration/schema 报错，无 OOM/NaN；10 个 performance 记录点的吞吐范围为 `282.993--328.158 samples/s`、均值约 `302 samples/s`，data wait ratio 约 `0.07%--0.12%`。step 1000 grad norm 为 `1.1899`，训练按既有 `grad_clip_norm=1` 处理。当前未稳定复现旧 benchmark 的 `366.732 samples/s`；整机同时运行旧 mixed、warm-up、Viewer 和图形上下文，因此旧独占条件吞吐不能直接视为当前保证值。

2026-08-23 19:42 CST 按用户要求释放两张 GPU：从最近完整 `latest.pt`（step 118080 / epoch 32 / best `12.066 mm`）恢复到 GPU 2/3，改为 2 GPU × per-device batch 72，global batch 仍为 144。首步确认 optimizer/scheduler 正确恢复，无 OOM/NaN；单卡显存约 `7.6--7.9 GB`、吞吐约 `200 samples/s`。step 118080 之后尚未落盘的约 1 epoch 不纳入续训。

### 结论状态

`INCONCLUSIVE`（训练进行中）

### 下一步

观察首个完整 epoch 的 train/validation、实际平均吞吐和 checkpoint；在没有 OOM、NaN 或持续退化前保持当前配置，不做额外微调。

### 证据

- output: `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_20260822_125835`
- log: `output/exp/cm_v121/cm_object_v2_grab_arctic_subject_template_3gpu_bs48_50ep_gpu234_20260822_125824.log`
- resume log: `output/exp/cm_v121/cm_object_v2_grab_arctic_subject_template_2gpu_bs72_resume_20260823.log`
- metrics: `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_20260822_125835/metrics.jsonl`
- W&B: `https://wandb.ai/hitsz-oyx/ref2dex/runs/ih6ia14q`

## EXP-006 — subject-template 修复版 GRAB+ARCTIC mixed cache

### 日期

2026-08-22

### 假设

复用已验证的修复版 GRAB ObjectV2 与既有 ARCTIC ObjectV2，通过不复制实体数组的链接 cache 重新生成联合 split/statistics/calibration，可以恢复正确 GRAB hand geometry，同时保持现有 mixed Dataset、模型、loss 和评估合同不变。

### Baseline

- 旧 mixed root: `data/processed_data/cm_object_v2`
- 修复版 GRAB: `data/processed_data/cm_object_v2_subject_template_20260820/grab`
- 复用 ARCTIC: `data/processed_data/cm_object_v2/arctic`
- 旧 config: `src/task/Cm/configs/active/object_v2_grab_arctic.yaml`

### 本次修改

- 建立 `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820`；dataset/subject/sequence 是真实目录，每条 sequence 的 `shared/left/right` 是相对软链接；
- seed42 按 dataset 分层重新生成 sequence-disjoint split；
- 重新计算全量 E1 statistics 和只读 train split 的 stride 1--10 calibration；
- 新增独立 no-gate + time condition 训练配置，不启动训练。

### 结果

| 项目 | 结果 |
| --- | ---: |
| 总 sequences | 1636 |
| train / val / test | 1308 / 164 / 164 |
| GRAB samples | 327798 |
| ARCTIC samples | 334248 |
| train hand streams | 2616 |
| train pairs（10 strides 合计） | 5313210 |
| train-only flow RMS | 0.07328625889337191 m |
| object flow target scale | 13.645122770626802 |
| 软链接数 / 断链数 | 4908 / 0 |
| 链接 cache 自身占用 | 约 24M |

配置加载、split 全覆盖与互斥、数据集标签、scale/metadata 精确一致性均通过静态校验。没有启动训练，也没有产生新版模型指标。

### 解释

该结果只支持新版数据入口的实现与元数据闭环，不证明修复 hand template 会提高模型效果。旧 mixed calibration 为 RMS `0.0732755479335936 m`，新版为 `0.07328625889337191 m`；尺度变化很小不代表几何修复影响很小，因为 subject template 主要改变 hand geometry、candidate mask 和 DenseToken 输入，而不是物体 flow 本身。

### 结论状态

`SUPPORTED`（cache / metadata implementation gate）

### 下一步

获得用户确认后从头启动新版 mixed 训练；不得从旧数据 checkpoint 续训后将结果解释为严格的数据修复对照。

### 证据

- cache: `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820`
- split: `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/splits_seed42/splits.json`
- statistics: `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/object_v2_statistics.json`
- calibration: `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/metadata.json`
- config: `src/task/Cm/configs/active/object_v2_grab_arctic_subject_template_20260820.yaml`

## EXP-005 — GRAB gate+cm64 全 slot warm-up

### 日期

2026-08-20

### 对应指导

`docs/指导/V1.2.1.md`

### 假设

先让 16 个 slot 在无 hard gate 的条件下共同训练，再逐步引入 gate threshold 与稀疏 count loss，可以让低维 `cm_dim=64` 的模型先形成分工，降低原 gate 训练中单 slot fallback 导致的 branch collapse。

### Baseline

- commit: `453806a`
- config: `src/task/Cm/configs/active/object_v2_grab_gate_cm64.yaml`
- checkpoint: 原候选迁移训练的 step 16188 checkpoint 仅作诊断对照，不续训

### 本次修改

- 新配置使用 GRAB-only、time condition、gate、`cm_dim=64` 和相同 fixed split/calibration；
- 前 5 epoch 强制所有 16 个 slot 参与 decoder，关闭 count loss；
- 后 5 epoch 将 threshold 从 0 线性升到 0.85、count loss 权重从 0 线性升到 `1e-3`；
- 第 11 epoch 起使用完整 gate 配置；其余数据、GT、评估和 50 epoch budget 保持不变。

### 实现审查

Verdict: PASS

关键检查：
- 11 个 `tests/test_cm_slot_attention.py` 单测通过；
- warm-up 调度 smoke 输出 epoch 0/4 全开、epoch 5/7 ramp、epoch 9 完整 gate；
- 首个训练日志显示 `gate_force_all=1`、`hard_active_mean=16`、`effective_branch_count=16`、`global_top1_usage≈0.0625`；
- 无 OOM、NaN 或 DDP 崩溃。

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  -m src.task.Cm.src.train \
  --config src/task/Cm/configs/active/object_v2_grab_gate_cm64_warmup.yaml \
  --distributed
```

### 结果

该 run 最终按用户要求在 step 102524 / epoch 38 停止；最后一个完整 checkpoint 为 `step_000094430_epoch_000035.pt`，同时是停止时的 `best.pt` / `latest.pt`。epoch 38 的 train EPE 为 `11.362 mm`，zero-flow improvement 为 `0.7756`；gate 已重新收缩到 effective branch count `1.05`、global top-1 usage `0.9773`，说明 warm-up 没有在长程保持多 slot 分工。

| Metric | Step 100 |
| --- | ---: |
| train flow EPE | `46.376 mm` |
| hard active mean | `16.0` |
| effective branch count | `16.0` |
| global top-1 usage | `0.06251` |
| samples/s | `102.1` |

### 关键观察

warm-up 初始阶段实现了全 slot 参与，但进入完整 gate 后仍逐渐回到近单 slot 路由；它改善了早期启动，却没有解决长程 branch collapse。

### 解释

当前证据表明 warm-up 路径有效，但不足以长期保持多 slot 使用。由于该 run 使用受平均 MANO template 影响的旧 GRAB cache，不继续消耗资源，也不作为新版 mixed 的有效效果结论。

### 结论状态

INCONCLUSIVE

### 决策

按用户要求停止训练并保留 epoch 35 best/latest checkpoint；后续若重访 gate，需要重新设计 active-count 目标，而不是继续延长该旧数据 run。

### 下一步

不再续训该旧数据 run；新版 mixed 当前只比较 no-gate 的 C=256/C=64。

### 证据

- log: `output/exp/cm_v121/cm_v121_grab_gate_cm64_warmup_2gpu_bs48_50ep_gpu01_20260820_111403.log`
- metrics: `outputs/cm/cm_object_v2_grab_gate_cm64_warmup_20260820_111406/metrics.jsonl`
- W&B: `https://wandb.ai/hitsz-oyx/ref2dex/runs/8grohy8u`
- commit: `453806a`

## EXP-002 — V1.2.1 fixed split + no-gate/time mixed short training

### 日期

2026-08-19

### 对应指导

`docs/指导/V1.2.1.md`

### 假设

只改数据链路，并把模型保持在稳定的 no-gate + time condition 语义下，配合固定 sequence split 和 train-only flow calibration，可以先把 V1.2.1 的实现问题收敛到可重复的训练基线，再判断 mixed GRAB/ARCTIC 数据是否会带来早期优化信号。

### Baseline

- commit: `ccb76ca`
- config: `src/task/Cm/configs/active/object_v2_grab_arctic.yaml`
- checkpoint: 无；本次只做短训验证

### 本次修改

- 增加 sequence 固定 split 生成器；
- `_MmapSequenceDataset` 改为 `mp.Value` 共享 epoch；
- object-v2 cache 加入 LRU 打开数上限；
- train-only flow calibration 接入 object-v2 mixed root；
- mixed config 固定为 `use_time_condition=true`、`use_slot_gate=false`；
- runner 增加联合 root 的 object-v2 识别。

### 实现审查

Verdict: PASS

关键检查：
- train/val/test 按 sequence 互斥；
- dataloader 能稳定读取联合 `grab/` + `arctic/` root；
- 2-step smoke 与 3-seed 短训均无 NaN / 崩溃 / 数据错误；
- no-gate + time condition 与当前配置一致。

### 实验命令

```bash
python -m src.task.Cm.src.train \
  --config src/task/Cm/configs/active/object_v2_grab_arctic.yaml \
  --set train.max_steps=300 \
  --set train.seed=42
```

### 结果

| Seed | Final train/mean_stride_epe_mm | Final train/val_mean_stride_epe_mm | Final train/relative_epe | Final train/zero_flow_improvement |
| --- | ---: | ---: | ---: | ---: |
| 42 | 28.0734 mm | 28.0734 mm | 1.00372 | -0.003715 |
| 43 | 27.5051 mm | 27.5051 mm | 1.00405 | -0.0040464 |
| 44 | 28.0194 mm | 28.0194 mm | 1.00533 | -0.0053268 |

证据文件：
- `output/exp/cm_v121/cm_v121_mixed_seed42.stdout.log`
- `output/exp/cm_v121/cm_v121_mixed_seed43.stdout.log`
- `output/exp/cm_v121/cm_v121_mixed_seed44.stdout.log`
- `outputs/cm/cm_v121_mixed_seed42_20260819_094543/metrics.jsonl`
- `outputs/cm/cm_v121_mixed_seed43_20260819_094543/metrics.jsonl`
- `outputs/cm/cm_v121_mixed_seed44_20260819_094543/metrics.jsonl`

### 关键观察

- 数据链路与训练链路都能闭环，说明 V1.2.1 的实现修正是有效的；
- 三个 seed 的 300-step 结果都没有给出明显的优化信号，zero-flow 对比略差于当前预测；
- 这个 budget 更像实现 gate，而不是足够强的科学判定。

### 解释

fixed split 和 train-only calibration 解决的是可复现性与统计口径问题；它们让实验可比，但并不会自动提升指标。300 step 训练太短，且 cosine schedule 已明显衰减，当前结果不足以判断 mixed data-only 假设是否成立。

### 结论状态

INCONCLUSIVE

### 决策

不把当前 300-step checkpoint 作为候选最佳模型；保留配置与日志，继续做更有区分度的 GRAB-only、ARCTIC-only、mixed 对照。

### 下一步

按 V1.2.1 指导继续跑同预算的 GRAB-only / ARCTIC-only / mixed 对照，并在需要时提高训练步数再比较。

### 证据

- commit: `HEAD`
- config: `src/task/Cm/configs/active/object_v2_grab_arctic.yaml`
- train log: `output/exp/cm_v121/`
- metrics: `outputs/cm/cm_v121_mixed_seed*/metrics.jsonl`

## EXP-003 — V1.2.1 grab-only / arctic-only small-scale comparison

### 日期

2026-08-19

### 对应指导

`docs/指导/V1.2.1.md`

### 假设

如果 V1.2.1 的固定 split 与 train-only calibration 足够稳定，那么把 mixed 数据拆成 GRAB-only / ARCTIC-only 后，至少应在 300-step 小预算上看到比 zero-flow 更一致的优化趋势；否则说明这个预算只够验证链路，不足以验证数据假设。

### Baseline

- commit: `cb7ae88`
- config: `src/task/Cm/configs/active/object_v2_grab_only.yaml` / `src/task/Cm/configs/active/object_v2_arctic_only.yaml`
- checkpoint: 无；只做短训对照

### 本次修改

- 新增 GRAB-only 与 ARCTIC-only object-v2 配置；
- 修正 object-v2 flow calibration 的 sequence 计数口径；
- 复用各自独立的固定 split + train-only calibration；
- 维持 `use_time_condition=true`、`use_slot_gate=false` 不变。

### 实现审查

Verdict: PASS

关键检查：
- 单数据集 split 可读；
- loader 能稳定返回 train/val/test；
- calibration metadata 与 config scale 一致；
- 300-step run 无 NaN / 崩溃 / 数据错误。

### 实验命令

```bash
python -m src.task.Cm.src.train \
  --config src/task/Cm/configs/active/object_v2_grab_only.yaml \
  --set train.max_steps=300 \
  --set train.seed=42
```

### 结果

| Run | val/mean_stride_epe_mm | val/mean_stride_relative_epe | val/zero_flow_improvement | 备注 |
| --- | ---: | ---: | ---: | --- |
| GRAB seed42 | 51.3116 | 1.00292 | -0.00292 | 已完成 |
| GRAB seed43 | 51.3039 | 1.00245 | -0.00245 | 已完成 |
| ARCTIC seed42 | 23.5727 | 1.00296 | -0.00296 | 已完成 |

### 关键观察

- GRAB-only 和 ARCTIC-only 都能正常收敛到稳定的 300-step 轨迹，但都没有明显优于 zero-flow；
- GRAB 和 ARCTIC 的尺度差异仍然显著，说明 train-only calibration 是必要的，但仅靠校准不能让短预算立刻出现正向信号；
- 这批结果和 mixed 小预算一起看，仍然更像是“实现可用”而不是“科学假设已证实”。

### 解释

固定 split 与 train-only calibration 已经把可复现性问题收住了；剩下的瓶颈是预算太短，cosine lr 也已经衰减到零，模型还没进入能分辨数据差异的区间。

### 结论状态

INCONCLUSIVE

### 决策

不再继续用更多 300-step seed 去堆重复证据；如果后面要进一步判断数据假设，应把预算加长，而不是只加 seed。

### 下一步

先基于当前 completed runs 更新文档和提交，再考虑是否把预算提高到更能区分 mixed / single-dataset 的级别。

### 证据

- `output/exp/cm_v121/cm_v121_grab_seed42.stdout.log`
- `output/exp/cm_v121/cm_v121_grab_seed43.stdout.log`
- `output/exp/cm_v121/cm_v121_arctic_seed42.stdout.log`
- `outputs/cm/cm_v121_grab_seed42_20260819_103530/metrics.jsonl`
- `outputs/cm/cm_v121_grab_seed43_20260819_103530/metrics.jsonl`
- `outputs/cm/cm_v121_arctic_seed42_20260819_103530/metrics.jsonl`

## EXP-001 — V1.2 full-data Stage4/cache 与 E1 统计

### 日期

2026-08-18

### 对应指导

`docs/指导/V1.2.md`

### 假设

复用现有 GRAB/ARCTIC Stage4，转换为统一 object-v2 mmap/ragged cache 后，可以在不改变 Cm 模型和 loss 的前提下获得足够的跨数据集 transition，用于决定混合采样和 flow calibration。

### Baseline

- code: 当前 V1.2 implementation（commit `48e3b16`）
- config: `configs/active/object_v2_grab_arctic.yaml`
- checkpoint: 无；本 EXP 只做数据与 cache gate

### 本次修改

- 生成 GRAB/ARCTIC 全量 Stage4 object-only cache；
- 转换为 `cm_object_v2` mmap/ragged cache；
- 构建 B=4、512 点 sampling bank；
- 对两套 cache 分别及联合计算 E1 统计。

### 实验命令

```bash
python -m process.GRAB.stage4_cm ... --ds-rate 4 --num-obj-points 4096
python -m process.ARCTIC.stage4_cm ... --ds-rate 1 --num-obj-points 4096
python -m process.common.object_cache_v2 ...
python -m src.task.Cm.build_object_sampling_bank ... --bank-size 4 --num-points 512
python -m src.task.Cm.compute_object_v2_stats ... --min-stride 1 --max-stride 10
```

### 结果

| Dataset | sequences | frames | valid samples | flow RMS |
| --- | ---: | ---: | ---: | ---: |
| GRAB | 1335 source / 1028 newly written | 643090 | 328309 | 53.20 mm |
| ARCTIC | 301 | 436546 | 334248 | 23.55 mm |
| Combined | 1636 cache sequences | — | 662557 | 41.01 mm（按两侧统计合并） |

证据文件：`data/processed_data/cm_object_v2/object_v2_statistics.json`、`grab_statistics.json`、`arctic_statistics.json`。cache 约 `110 GB`，sampling bank 已生成。

### 关键观察

- GRAB 与 ARCTIC 有效 sample 数接近，`sqrt(N)` 初始采样比例约 `1:1`；
- flow RMS 相差约 `2.26×`（GRAB 更大），不能直接假设两个 dataset 的梯度分布一致；
- 首次联合统计命令暴露了 combined root 兼容问题，已修复 `CmObjectV2Dataset` 对嵌套 `grab/`、`arctic/` root 的读取。

### 解释

当前证据支持使用近似均衡的 dataset sampler 作为 E2 初始方案，但不支持 dataset-specific normalization 或最终 sampler 决策。当前统计是 E1 诊断统计，不等同于严格 train-only calibration metadata。

### 实现审查

Verdict: PASS（cache/E1 gate）；正式训练 calibration gate 尚未完成。

### 结论状态

`SUPPORTED`（仅支持 V1.2 数据链路可运行和 E1 统计，不代表模型效果假设已验证）。

### 决策

保留 full-data object-v2 cache；下一步先补 train-only calibration、固定 sequence split 和 GRAB-only/ARCTIC-only/mixed 短训配置。

### 下一步

实现 object-v2 的 train-only calibration 与可复现 split，然后执行三组 5k--10k step 短训。

## EXP-004 — V1.2.1 mixed 全量吞吐与正式 batch 选择

### 日期

2026-08-19

### 对应指导

`docs/指导/V1.2.1.md`

### 假设

在固定 3 GPU DDP 和相同数据/模型路径下，增大 per-device batch 会先提高有效样本吞吐，随后受到显存和单步计算开销限制而饱和或回落；选取吞吐峰值可以缩短 full-data 长训的墙钟时间，同时不改变 no-gate + time condition 的研究语义。

### Baseline

- commit: `3e32de8`
- config: `src/task/Cm/configs/active/object_v2_grab_arctic.yaml`
- checkpoint: 无；本 EXP 为训练吞吐与正式入口选择，不比较模型效果

### 本次修改

- 在 3 GPU（`CUDA_VISIBLE_DEVICES=0,1,5`）上对 mixed object-v2 做 per-device batch 8/16/24/32/48/64 sweep；
- 训练步数固定为 80，关闭验证影响，只统计 train-only steady-state `perf/samples_per_s`；
- 完成一个 batch 24 的 mixed full pilot，确认 10000-step 全量训练与 online 记录链路可闭环；
- 正式配置的 mixed、GRAB-only、ARCTIC-only 入口统一设为 `batch_size=48`、`val_batch_size=48`、`wandb.mode=online`，并保留 DDP 的 `find_unused_parameters=true`。

### 实现审查

Verdict: PASS

关键检查：
- 6 个 batch 候选均完成 80-step 训练，无 OOM、NaN 或 DDP 崩溃；
- 吞吐取自每个 run 的最后一个带 `perf/` 的训练日志，避免启动 warm-up 干扰；
- batch 24 full pilot 完成至 step 10000，产生 val/test 统计和 checkpoint/metrics；
- no-gate + time condition、GT、split、calibration 和评估口径未改变。

### 实验命令

吞吐 sweep 使用同一训练入口，仅覆盖 `train.max_steps=80` 和候选 `data.batch_size`；正式入口为：

```bash
CUDA_VISIBLE_DEVICES=0,1,5 \
  /home2/wyy/miniconda3/envs/graspenv/bin/torchrun \
  --standalone --nproc_per_node=3 \
  -m src.task.Cm.src.train \
  --config src/task/Cm/configs/active/object_v2_grab_arctic.yaml \
  --distributed
```

### 结果

| Per-device batch | Global batch (3 GPU) | step_ms | samples/s |
| ---: | ---: | ---: | ---: |
| 8  | 24  | 115.860 | 207.146 |
| 16 | 48  | 169.865 | 282.577 |
| 24 | 72  | 218.675 | 329.256 |
| 32 | 96  | 279.466 | 343.512 |
| 48 | 144 | 392.657 | **366.732** |
| 64 | 192 | 537.682 | 357.088 |

batch 24 mixed full pilot 的最终汇总为：`val/mean_stride_epe_mm=29.3987`、`val/zero_flow_improvement=0.1677`，证据位于 `outputs/cm/cm_v121_mixed_bs24_3gpu_full_20260819_131714/metrics.jsonl`。该 pilot 只用于验证 full-data 运行链路，不与 batch 选择混为效果结论。

吞吐证据位于 `output/exp/cm_v121_throughput/mixed_bs{8,16,24,32,48,64}_3gpu_trainonly.log`。

### 关键观察

- 吞吐从 batch 8 持续提升到 batch 48；batch 64 的单步时间继续增长，但样本吞吐回落约 2.6%；
- 数据等待占比约千分之一量级，当前瓶颈主要是模型计算/显存，不是 dataloader；
- batch 48 比 batch 32 约快 6.8%，比 batch 24 约快 11.4%，因此长训时间收益足以抵消更大的单步延迟。

### 解释

结果支持“在当前 3 GPU 资源和实现下存在 batch-size 吞吐峰值”的工程假设。batch 48 是观测区间内峰值，batch 64 已进入回落区；这只决定训练效率，不构成模型效果优越性的科学结论。full pilot 的正向 zero-flow improvement 说明链路可运行，但由于只有单个 pilot，不能替代 mixed/GRAB-only/ARCTIC-only 的正式效果比较。

### 结论状态

`SUPPORTED`

### 决策

正式长训从 3 GPU DDP + per-device batch 48（global batch 144）开始，mixed、GRAB-only、ARCTIC-only 使用相同入口约束。保留 batch 64 作为显存/吞吐上界参考，不作为默认设置。

### 下一步

启动 mixed 的正式 10000-step online wandb 长训；完成后在相同 batch/卡数下启动 GRAB-only 与 ARCTIC-only，并汇总固定 val/test 指标及 zero-flow/action intervention 对照。

### 证据

- `output/exp/cm_v121_throughput/mixed_bs8_3gpu_trainonly.log`
- `output/exp/cm_v121_throughput/mixed_bs16_3gpu_trainonly.log`
- `output/exp/cm_v121_throughput/mixed_bs24_3gpu_trainonly.log`
- `output/exp/cm_v121_throughput/mixed_bs32_3gpu_trainonly.log`
- `output/exp/cm_v121_throughput/mixed_bs48_3gpu_trainonly.log`
- `output/exp/cm_v121_throughput/mixed_bs64_3gpu_trainonly.log`
- `outputs/cm/cm_v121_mixed_bs24_3gpu_full_20260819_131714/metrics.jsonl`
- commit: `3e32de8`

## EXP-012 — GRAB C=64 geometry-only/no-time hard-gate 容量对照

### 日期

2026-08-24

### 假设与边界

在 EXP-011 的 GRAB object-only、原始物体几何、无时间条件和 gate warm-up 设定下，将 `cm_dim` 从 32 改为 64，检验 C=32 的退化是否主要来自 Cm 容量不足。输入、GT、DenseToken 冻结、gate loss 与 5+5 epoch warm-up 保持不变，不包含场景点；但实际运行同时改变了 global batch 和 DDP 方式，因此不是纯粹的单变量容量 ablation。

### 实现与运行

- config: `src/task/Cm/configs/active/object_v2_grab_gate_cm64_geometry_only_no_time.yaml`，继承 C=32 配置，覆盖 `meta.cm_dim=64` 与双卡运行所需的 batch；
- GPU: `CUDA_VISIBLE_DEVICES=1,2`，DDP world size=2；per-device batch=32，global batch=64，validation batch=32；
- 训练: 从头开始，50 epoch，W&B offline，预计 `202300` optimizer steps；相对 C=32 的 global batch=48、单卡 269750 steps，C=64 使用 global batch=64、DDP 双卡 202300 steps；
- output: `outputs/cm/cm_object_v2_grab_gate_cm64_geometry_only_no_time_20260824_155338`；
- launcher log: `output/exp/cm_v121/cm_object_v2_grab_gate_cm64_geometry_only_no_time_offline_gpu12_bs32.log`；
- 启动核验: 两个 rank 均完成初始化，`train_setup` 明确报告 `world_size=2/global_batch=64`，无 OOM/NaN；
- 单卡 C=64 的旧尝试仅到约 600 step、没有完整 epoch/checkpoint，已停止，不纳入结果。

### 结论状态

`INCONCLUSIVE`

当前结果显示 C=64 在同 epoch/同 optimizer-step 都优于 C=32，但由于 global batch、每 epoch step 数和 cosine 总步数均不同，该差异只能作为“容量+训练条件”的联合证据；不能称为完全公平的 C-only 结论。两者 warm-up 后均坍缩到约 1 个 effective slot。
