# InteractionTransfer 研究日志

## 实验：V0 最小 forward gate

**假设**
当前静态 hand-object relation 调制 hand action，可以产生严格 zero-preserving 的动态 interaction message，并由 object-only decoder 输出 object flow。

**改动**
新建独立 `InteractionTransfer` 包：edge builder、冻结静态编码器、relation encoder、无 bias action/message path、attention aggregator、object-only effect decoder 和 one-step dataset reader。

**结果**
合成张量 forward 通过 shape/finite-value 检查；零 hand flow 时 edge message 最大绝对值为 0。真实 cache 训练尚未启动，故不报告 EPE。

**决策**
保留 V0 实现，下一步再以真实 GRAB cache 与 direct forward baseline 比较。

## 实验：V0.1 action intervention

**假设**
固定当前 `(O,H)` 后，GT、反向和 shuffle action 应产生非零且不同的动态 message；zero action 必须严格为零。

**改动**
接入本地 vendored DenseToken/PTv3 checkpoint loader（默认使用 `src/task/Cm/densetoken_ckpt/best.pt`，不依赖其他 Task 运行时 import）；修正 object projection 为无 bias，新增 GT/zero/reverse/shuffle intervention 诊断。诊断脚本自动优先使用 CUDA；无 CUDA 时才回退 synthetic encoder。

**结果**
synthetic intervention：`message_norm(gt)=0.0109645`、`zero=0`、`reverse=0.0107316`、`shuffle=0.0099526`；四种 effect 均可计算，GT 相对 zero 的 effect 差异为 `1.0365e-4`。

**决策**
保留。zero-preserving 和 action sensitivity gate 通过；已在 `graspenv` 的 RTX 3090 上用真实 DenseToken/PTv3 完成 `(1,512,1538)` forward，输出 shape 为 `(1,512,3)`，zero-message 最大绝对值为 `0.0`。

## 实验：V0.2 数据与 DenseToken fidelity 修正

**假设**
只有在 DenseToken 输入特征、transition 筛选和点数与既有 one-step baseline 对齐后，InteractionTransfer 的 forward/intervention 结果才具备可比较性。

**改动**
补齐 opposite-cloud centroid projection 特征；将 pretrained edge feature 与 contact probability 接入 relation encoder；dataset 改为 512 object / 1538 hand，并筛选 right-active、left-inactive 的 one-step transition；修复 shuffle 为 hand-point 维度置乱；训练脚本自动选择 CUDA 并支持 `--device`。

**结果**
`graspenv` + CUDA 诊断通过：object flow shape `(1,512,3)`，valid edges `5868`，zero-message `0.0`。GT/reverse/shuffle message norm 分别为 `0.0891573/0.0904426/0.0893798`；相对 zero 的 effect delta 分别为 `2.8672e-4/2.8384e-4/2.7575e-4`。

**决策**
保留。数据控制变量和 DenseToken relation prior 已接入；真实 GRAB EPE 对比待提供 cache 路径后运行。

## 实验：真实 cache 路径核查与单样本 forward

**观察**
仓库已有真实 GRAB cache，并非缺失。PointWorldWAM overfit 配置使用 `data/processed_data/interaction_dynamics_v1/data/grab`；Cm/Stage4 使用 `data/processed_data/stage4/data/grab`。

**结果**
前者包含 20 个序列；InteractionTransfer 在 `s1/airplane_fly_1` 上按 V0.2 筛选得到有效 transition，读取为 `512 object / 1538 hand`。`graspenv` CUDA forward 输出 shape 为 `(1,512,3)`，全为有限值，有效 edge 数 `127`。

**决策**
保留。此前“待提供 cache 路径”的结论撤回；下一步可以直接进行真实 cache 的短训和 direct baseline EPE 对比。

## 实验：V0.3 DenseToken fidelity 与公平 transition 对齐

**假设**
补齐 DenseToken 的 opposite-cloud centroid projection，并复用 PointWorld one-step 的数据校验与运动排序后，InteractionTransfer 的输入特征和 transition 定义才完全一致；模型应继续满足 zero-preserving，并能在真实 cache 上下降训练 loss。

**改动**
在冻结 DenseToken 编码器中计算 opposite cloud centroid-direction·normal，替换原先恒为零的第三个几何特征。数据集加入 `source_fps=120`、`raw_frame_id` 间隔为 4 的校验，按 object motion 降序排序，并复用 PointWorld 的固定点索引。

**结果**
`graspenv` + RTX 3090 forward 通过：输出 `(1,512,3)`，有效 edge `5868`，`zero-message=0.0`。GT/reverse/shuffle message norm 为 `0.0890855/0.0903455/0.0892916`，对应 effect delta 为 `2.8514e-4/2.8261e-4/2.7422e-4`。真实 `s1/airplane_fly_1` cache 单序列 10 步 CUDA 短训 loss 从 `0.0561577` 降至 `0.0168040`，无 NaN/Inf。

**决策**
保留 V0.3 修改；实现 gate 和短训 gate 均通过。完整多序列训练及 direct baseline EPE 留待下一实验。

## 实验：V0.4 正式结构与数据入口固化

**假设**
将 V0 forward 的信息约束正式固化后，当前模型应保持 `ΔH=0 => C_m=0`，且 effect decoder 只从 object points、normals 和聚合后的 object-side field 预测 flow。

**改动**
InteractionTransfer dataset 内置确定性 `fixed_point_indices`，移除对 PointWorldWAM 的运行时依赖；V0.4 固定 one-step `gap=1`，不接受其他 gap。指导文档同步纳入版本管理。

**结果**
`graspenv` + RTX 3090 forward 通过：输出 `(1,512,3)`，有效 edge `5868`，`zero-message=0.0`。真实 `s1/airplane_fly_1` cache CUDA 短训 5 步 loss 从 `0.0485412` 降至 `0.0265517`，无 NaN/Inf。

**决策**
保留 V0.4 固化；正式结构 gate 和短训 gate 通过。完整多序列 EPE 与 direct baseline 对比仍属于后续实验。

## 实验：V0.4 direct baseline 与训练后 intervention

**假设**
同一 transition 上训练 direct `(O,H,ΔH)->ΔO` baseline 后，Cm bottleneck 应达到 `EPE_Cm <= 1.25 EPE_direct`，且 GT action 的 EPE 优于 zero/reverse/shuffle；同时 object-side field 应随 intervention 改变。

**改动**
新增自包含 `baselines/direct_forward.py` 和 `research/compare_v04.py`。两者使用同一 GRAB dataset、SmoothL1 effect loss 和 100 步 CUDA 训练；比较四种 action 的 EPE，并计算 `D(C_GT,C_variant)`。

**结果**
Cm loss `0.0416540 -> 0.0131193`，direct loss `0.0384623 -> 0.0057540`。Cm intervention EPE：GT `0.0099098`、zero `0.0404241`、reverse `0.0669083`、shuffle `0.0097842`；direct GT EPE `0.0507196`，因此 `EPE_Cm/EPE_direct=0.195`，通过 1.25 gate。field distance（GT 对比）为 zero `0.0273926`、reverse `0.0488763`、shuffle `0.0005350`。

**诊断**
Cm 已明显利用 action，GT/zero/reverse 分离且容量 gate 通过；shuffle 与 GT 的 EPE 和 field 距离几乎相同，尚不能声称完整通过 intervention 语义判据。该结果是单序列 overfit，不能替代多序列评估。

**决策**
保留 direct baseline 和比较脚本；V0.4 生死判据部分通过（容量与 zero/reverse），shuffle 判据不充分，后续需扩大数据和训练再判断。

## 实验：V0.5 relation prior 与 action-field 验证

**假设**
修正 DenseToken edge/contact head 的 checkpoint 映射并强制 frozen eval 后，relation prior 应确实来自预训练权重；在 sequence-level benchmark 中，cross-sample action 应比 GT 更差，DirectEdge 应提供公平的容量对照。

**改动**
将 loader 的模块名对齐为 `edge_shared_backbone` / `cross_edge_head`，并对 backbone、edge head、contact head 做严格 missing 检查；覆盖 `train()` 保持整个 DenseToken eval。新增 `diag_action_field.py`、自包含 `DirectEdge` 和 `benchmark_v05.py`，支持 cross-sample、point-shuffle、mean-flow、EPE/translation/rotation 与 object-field distance。

**结果**
真实 checkpoint 校验：edge head 与 checkpoint 的最大权重差 `0.0`，contact head 最大权重差 `0.0`，encoder training state 为 `False`。`s1/airplane_fly_1` 的 64 个 transition 中 action local ratio 均值 `0.05991`（范围 `0.02543–0.08810`），因此 point-shuffle 是弱干预；DirectEdge 单样本 CUDA forward 输出 `(1,512,3)` 通过。

**观察到的失败 / 现象**
V0.5 多序列 benchmark 脚本已实现 sequence split 和顺序释放模型显存，但在当前机器上同时进行 PTv3 多序列训练/evaluation 时进程被 OOM 终止，未获得可报告的 held-out EPE。此前 V0.4 单序列结果仍只能作为 overfit capacity evidence。

**决策**
保留两个 P0 修复、action-field 诊断和 DirectEdge/benchmark 实现；V0.5 的正式 multi-sequence causal gate 尚未完成，不能宣称通过。后续需降低点数/模型显存或分卡后重新运行 benchmark。

## 实验：V0.6 独立进程内存修复与 V0.5 sequence split benchmark

**假设**
Cm 与 DirectEdge 在独立 subprocess 中各自完成 PTv3 初始化、训练和 evaluation 后，native CPU allocator 不会跨模型滞留；在未见 sequence 上，Cm 应优于 zero/reverse/cross-sample，并接近 DirectEdge。

**改动**
将 `benchmark_v05.py` 改为不导入 torch 的轻量 launcher，新增 `train_eval_v05.py` 单模型 worker，两个 worker 通过 `subprocess.run` 顺序执行并写 JSON。worker 内一次初始化 PTv3，完成训练、GT/zero/reverse/cross-sample/point-shuffle/mean-flow evaluation，并输出 RSS/GPU memory。cross-sample 改为使用下一个真实 transition 的 hand flow。

**结果**
独立 worker 的 RSS 峰值约 `2.20 GiB`，GPU allocated 约 `376.5 MiB`，不再被 OOM 杀死。使用 `s1/airplane_fly_1` 训练 4 transitions、20 steps，在两个未见 sequence 各评估 1 transition：

- `flashlight_on_2`：Cm GT `47.19 mm`、zero `47.90 mm`、reverse `47.90 mm`、cross `47.19 mm`；DirectEdge GT `91.03 mm`；Cm/DirectEdge `0.518`。
- `bowl_drink_2`：Cm GT `30.66 mm`、zero `28.74 mm`、reverse `28.64 mm`、cross `30.66 mm`；DirectEdge GT `49.42 mm`；Cm/DirectEdge `0.620`。

**诊断**
容量 gate 在这两个 held-out transition 上通过，但 GT 没有稳定优于 zero/reverse，cross-sample 与 GT 几乎相同；V0.5 causal gate 未通过。此前 action local ratio 仅约 `0.0599`，说明该 cache 的 point-shuffle/cross-sample 干预仍然较弱或模型未学到对应关系。

**决策**
保留独立进程修复和 benchmark 链路；不宣称 V0.5 表示验证通过。下一步若继续，应先使用 V0.6 建议的 offline DenseToken static cache，并扩大训练/验证 transition 后重新评估。

## 实验：V0.6.1 正确 cross-sample 与多 transition intervention

**假设**
修正 val 单样本导致的 cross-sample=GT bug，并扩大到 64 个 train transitions、16 个 held-out transitions、1000 steps 后，Cm 的 GT EPE 应低于真正错配的 cross-sample/zero/reverse，并与 DirectEdge 做容量比较。

**改动**
`train_eval_v05.py` 强制 cross-sample 至少需要 2 个 validation transition，使用 roll derangement，额外报告 action distance。V0.5 worker 运行 64 train transitions、16 val transitions、1000 steps；Cm 与 DirectEdge 仍使用独立进程。

**结果**
Cm loss `0.06017 -> 0.000811`，DirectEdge loss `0.04140 -> 0.010756`。Cm held-out EPE：GT `5.44 mm`、cross-sample `22.13 mm`、zero `28.16 mm`、reverse `57.95 mm`、point-shuffle `4.89 mm`、mean-flow `4.89 mm`。DirectEdge GT `34.53 mm`，Cm/DirectEdge `0.158`。Cm 的 field distance：cross `0.02454`、zero `0.03552`、reverse `0.07027`、point-shuffle `0.00516`、mean-flow `0.00501`；cross action distance `0.01917 m`。

**诊断**
cross-sample 修复后不再等于 GT，且 GT 明显优于 cross/zero/reverse，V0.5 的主要 causal gate 在 16 个 held-out transitions 上通过。point-shuffle/mean-flow 仍略优于 GT，与 action local ratio 约 `0.0599` 的弱局部干预相符。

**决策**
保留 V0.6.1 修复和实验结果；Cm 的容量与主要 causal gate 通过，但不把弱 point-shuffle/mean-flow 判据宣称为通过。offline DenseToken static cache 仍是后续性能和规模化方向。

## 实验：V0.7 刚体 twist intervention 与公平收敛对比

**假设**
如果 Cm 使用 hand point 的局部空间运动结构，那么 rigid-twist 分解中的 GT 应优于 translation-only；DirectEdge 则应通过独立 best-validation 选择后再比较。

**改动**
新增最小二乘 rigid twist 拟合，评估 `translation_only` 与 `rotation_only`；worker 支持每隔 500 steps 在 held-out EPE 上保存可训练参数的 best state，避免复制冻结 PTv3。Cm 与 DirectEdge 均使用 64 train transitions、16 held-out transitions、1000 steps。

**结果**
Cm held-out EPE：GT `9.12 mm`、cross `24.40 mm`、zero `27.53 mm`、reverse `59.20 mm`、translation-only `8.57 mm`、rotation-only `27.53 mm`、point-shuffle `8.64 mm`、mean-flow `8.57 mm`。DirectEdge best-run GT `34.05 mm`，Cm/DirectEdge `0.268`。Cm loss `0.06017 -> 0.001316`，DirectEdge loss `0.04140 -> 0.007521`。

**诊断**
主要 action causal gate 继续通过：GT 明显优于 cross/zero/reverse。translation-only 和 mean-flow 略优于 GT，rotation-only 接近 zero，说明当前 GRAB one-step cache 中整体平移占主导，尚未证明 Cm 编码了强局部 hand-object interaction；这与 action local ratio 约 `0.0599` 一致。

**决策**
保留 twist intervention 与 best-validation 机制；不进入 inverse action optimization。下一步仍应优先做 offline DenseToken static cache，并寻找高旋转/高局部差异 transition 子集后再验证局部 interaction。

## 实验：V0.8 offline static cache 与全量 20 epoch 训练

**假设**
将 frozen DenseToken/PTv3 的静态特征离线缓存后，训练可扩展到全部序列和全量 transitions；全量数据训练的 EPE 应显著低于此前 64-transition 小样本。

**改动**
`model.py` 拆分 `encode_static`/`forward_core`（静态部分只依赖 `(O_t,H_t)`）；新增 `precompute_static.py`（分片并行、断点续算、保真度 sanity）与 `cached_dataset.py`（fp32 memmap 直读）；`train_full.py` 使用与 Cm full_grab_v1 完全一致的 1068/134/133 序列划分训练 20 epoch，val EPE 选 best。

**结果**
cache 覆盖 1335 序列、175,529 transitions（train 141,571 / val 16,794 / test 17,164），随机抽样保真度最大误差 `1.8e-6`。训练收敛于 epoch 16，val EPE `3.42 mm`（V0.7 小样本为 `9.12 mm`）。epoch 1 即达 `3.82 mm`，说明此前小样本主要受数据量限制。

**决策**
保留 static cache 链路和划分；V0.8 全量训练 gate 通过。训练后段 val 在 3.4–3.8 mm 波动无明显过拟合。

## 实验：V0.9 checkpoint 定量验证与中间过程可视化

**假设**
全量训练后的 best checkpoint 在 held-out test 上满足 `EPE_GT << EPE_zero/reverse/cross`；且 ΔH→M→C_obj→ΔO 链路的中间表示可在 Viser 中随 action 切换观察变化。

**改动**
新增 `eval_ckpt.py`（test split 上 GT/zero/reverse/cross 四种 action intervention 的 EPE，支持 best/last 对比诊断）与 `visualize_intermediate.py`（Viser viewer：object/hand 点云、hand flow arrows（stride 可调）、M norm/attention Top-K edges、C_obj norm heatmap、GT/Pred flow、action 实时切换与 sample EPE）。

**结果**
test 17,164 transitions，best.pt（epoch 16）：GT `3.28 mm`、zero `11.15 mm`、reverse `21.58 mm`、cross `4.35 mm`；last.pt（epoch 17）：GT `3.47 mm`，与 best 接近。Quantitative Gate 通过。cross 与 GT 差距相对较小（`4.35` vs `3.28`），与相邻 transition 的 hand flow 天然相似一致。Viewer 在 `http://localhost:8080` 正常运行，action 切换链路可视化可用。

**决策**
保留 eval/viewer 脚本；V0.9 Quantitative Gate 通过。PCA-RGB、difference heatmap、selected point panel 等 viewer 扩展与 inverse 方向留待后续。

## 实验：V0.10 inverse optimization

**假设**
冻结 forward model、以 hand action 为优化变量时，梯度可以把 action 引导到产生目标 object effect 的解：500 个 test transitions 上 optimized EPE 应显著低于 zero init，并接近或优于 GT action forward。

**改动**
新增 `inverse_optimize.py`（`optimize_hand_flow` backend：Rigid SE(3) 6 参数 twist（含 3 轮迭代精化的最小二乘 twist 拟合，单元验证误差 4e-6）与 Free Point Flow 两种参数化；Effect / C_obj 双 target；返回 initial/optimized/GT 三路完整状态与 history）与 `eval_inverse.py`（test split 均匀抽样 500 transitions，zero init、Adam lr 0.01、300 steps，batch=16）。`visualize_intermediate.py` 升级为 Forward+Inverse 一体化：View Mode 切换、Initial/Optimized/GT 三路并排（含各自 EPE label、C_obj difference heatmap、optimization history）、Optimize/Reset、point shape（circle+gradient shading）与 object/hand point size 滑块。

**结果**
500 test transitions（zero init，target=GT object flow，300 steps）：

- Zero action `11.00 mm`
- Optimized rigid SE(3) `1.87 mm`
- Optimized free flow `1.74 mm`
- GT action forward `3.27 mm`

单样本 history 典型收敛：`29.24 → 4.94(25 步) → 1.98 mm(50 步)`。rigid 优于 GT action forward，说明 learned forward map 中存在比 GT action 更精确产生目标 effect 的 action 解（forward map 非单射）；free 仅比 rigid 低 0.13 mm，可优化空间几乎全部位于 6 维 rigid 子空间内，与 V0.7 "平移占主导"的结论一致。

**决策**
保留 inverse backend / eval / viewer；V0.10 inverse gate 通过：`GT object effect → optimized rigid hand action → low object EPE` 在未见 test sequences 上成立。下一步可将 Rigid SE(3) backend 替换为 MANO / robot FK，或接入 C_obj target 做 cross-embodiment 方向验证。

## 重构：V0.10.1 主 Viewer（真实 mesh + 时间序轨迹播放）

**动机**（指导 V0.10.1）
旧 `visualize_intermediate.py` 把 Forward、Inverse、M/attention/C_obj、三列对比全塞进一个类，且 Inverse 三列画的是同一堆点 + flow 箭头，没有真正展示"优化后的手"和"预测后的物体"；transition 顺序来自 motion 排序而非时间序。正式主 Viewer 与调试工具应拆开。

**改动**
新建 `viewer/` 包取代旧 viewer 的主 Viewer 地位（旧文件保留为 legacy debug）：

- `mesh_provider.py`：复用 `process/GRAB/raw.py` 加载 MANO layer（按 subject v_template 注入）、canonical object mesh 与 raw object pose；`twist_transform` 把 6 维 twist 直接作用到 mesh 顶点。
- `provider.py`：`SequenceBundle` 按 cache 帧序（真实时间序）组织完整序列，`model_inputs(i)` 与 `GRABOneStepDataset.__getitem__` 逐字段一致；`valid_transition` 只决定"是否可运行模型"，不筛帧，序列外帧仍显示 mesh 并注明模型不可用。
- `scene_renderer.py`：`SceneBatch` 用 `server.atomic()` 批量重建场景消除闪烁；`auto_spacing` 按 object 直径 `max(0.15, 1.5d)` 自动排布三路 panel（Optimized 居中）。
- `trajectory_viewer.py`（正式主 Viewer）：Object→Mesh、Hand→MANO Mesh；Forward/Inverse 两模式；Current/Pred Next/GT Next 以半透明 mesh 状态可视化，默认不画 flow 箭头；Rigid inverse 的 (t,ω) 经 `twist_transform` 真正作用到 MANO mesh 得到"优化后的手"；Overlay/Side-by-side；Play/Stop/FPS 按时间序播放完整序列；极简 GUI，M/attention/point size 等收进折叠 Advanced。
- `interaction_debug.py`（调试 Viewer）：Edge View Off/Top20/Top50/Selected，Selected 模式点击 object point 只显示其 K=16 条 edge 并列出 ||M||/attention 表；C_obj 四种显示（Off/Magnitude/GT-Current diff/GT-Optimized diff）；Free Point Flow 只在此处提供。
- `smoke_test.py`：无头冒烟测试。

**结果**
smoke 5/5 通过：MANO face centers 与 cache 1538 点 parity `0.0004 mm`；raw object params 重建与 cache 点云运动一致性 `0.0004 mm`；`model_inputs` 与 dataset 逐字段 `0.00e+00`；GT object flow 的刚体 twist 提升到 mesh 后与下一帧对齐 `0.000 mm`；单帧 inverse（zero init, 300 steps）`11.33 → 0.90 mm`（GT action forward `1.71 mm`）。`trajectory_viewer`（:8080）与 `interaction_debug`（:8082）真机启动正常（HTTP 200）。

**决策**
`visualize_intermediate.py` 降级为 legacy debug；V1.0 双手 + 随机 timestep 改造时 viewer 需同步扩展（bimanual mesh、Δt 显示）。

## 重构：V1.0 Bimanual + Random Surface + Random Horizon（指导 V1.0）

**假设**
固定 1538 face-center 点、固定 gap=1、right-only 三大人造约束都在放大模型对 MANO 点布局 / 33 ms 尺度 / 单手数据分布的依赖；一次性切换到「双手 769+769 在线表面采样 + 随机 gap ∈ {1,2,4,8} + 9D action（含 Δt）+ split KNN 8+8」后，R→M→C_obj→ΔO 核心结构不动即可覆盖 Right-only / Left-only / Bimanual 三类 interaction。

**改动**
- `geometry_cache.py`：face-center = A·V（A 为常量 face-vertex 平均矩阵，rank 778），用伪逆从 stage4 现有 cache 精确重建每帧 MANO 顶点（残差 ~2e-7 m），存 `hand_vertices_world [T,778,3] + active_mask`，1335 序列全量构建。
- `dataset.py`：新增 `GRABRandomTransitionDataset`（random frame + random gap + Dirichlet barycentric 在线采样，t/t+g 复用同一组 face id + 权重；valid = (R∨L) 在 [i,i+g] 连续 active；train 随机 / val/test 按 index 播种）。
- `edge_builder.py`：`build_edges_bimanual` split KNN（K_L=8, K_R=8，右手索引偏移后 concat）；`model.py`/`modules.py`：action 8D→9D（+normalized Δt），无 side embedding，static 每次 forward 在线跑 frozen PTv3（放弃 V0.8 offline static cache）。
- `inverse_optimize.py`：rigid 6D→12D（ξ_L, ξ_R 各自 SE(3)，`fit_rigid_twist_pair`/`twist_pair_to_flow`）；`train_full.py`/`eval_ckpt.py`/`eval_inverse.py` 全部切换 V1.0 数据流（cross intervention 同时替换 Δt）。
- viewer：`provider.py`（geometry cache 双手 mesh + gap-aware valid transitions）、`trajectory_viewer.py`（Gap 下拉、双手 mesh、rigid 12D 分别提升回两只手 MANO mesh）、`smoke_test.py` 升级 V1.0。

**结果**
smoke 7/7 通过：geometry/cache face-center parity `0.0002 mm`；raw object params 与 cache 点云一致性 `0.0004 mm`；dataset deterministic 播种与手工重放逐字段 `0.00e+00`（item 0: gaps=(1,2,4,8) 随机取 8）；provider 采样手工重放 `0.00e+00`；rigid 12D twist 往返 `7.45e-07`；zero flow + Δt=0 时 edge message 严格为 0（bias-free action path 保持）；单帧 rigid 12D inverse loss 下降（random init trainable 上 62.66→62.46 mm，仅验证链路）；GT object flow 刚体提升 mesh 对齐 `0.000 mm`。全量 20 epoch 训练待 geometry cache 构建完成后启动，结果另记。

**决策**
保留 V1.0 数据流为核心路径；V0.x 的 `GRABOneStepDataset`/`interaction_debug.py`/`visualize_intermediate.py` 保留为 legacy。后续：全量训练 + `eval_ckpt`（GT/zero/reverse/cross gate）+ `eval_inverse`（rigid 12D vs free vs GT action）。
