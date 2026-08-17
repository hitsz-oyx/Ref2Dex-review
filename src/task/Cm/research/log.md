# Cm 研究日志

## 实验：Scene Cache V1 合成 parity

**假设**

将 object-only 输入扩展为 object + environment 的 local scene，并使用 mmap geometry、ragged candidate、sampling bank 和 DenseToken bank，不应改变旧 Cm 的输入/损失语义；静态 environment 点应产生零 flow。

**观察到的失败 / 现象**

工作区没有可用的 GRAB 原始 `.npz`，无法在本机完成真实 Stage4、DenseToken 全量预计算和正式训练。

**诊断**

需要先用合成数据验证 cache 各层和旧路径的行为一致性，再把真实数据实验留到提供 GRAB 原始数据的环境执行。

**改动**

增加 Scene Cache V1 构建器、通用 environment asset 读取、ragged candidate、sampling bank、DenseToken bank、fingerprint 校验、scene dataset/runner 路由和 object/environment 诊断指标；补充 Cm 任务文档与 parity 测试。

**结果**

`python3 -m pytest -q tests/test_cm_scene.py`：9 passed。覆盖 object-only regression、ragged candidate parity、sampling bank determinism/diversity、DenseToken FP16 cache parity、fingerprint guard、loss parity 和 scene flow calibration；environment candidate 被计入校准且零 flow 会降低 RMS。

**决策**

保留当前实现。真实 GRAB 训练结果不在本次环境中虚构，待原始数据可用后按 V1 三步 pipeline 执行。

**下一步**

在具备 GRAB 原始数据和对应 MANO/DenseToken 依赖的环境中构建全量 scene cache，完成 train-only flow calibration、dense bank parity 和正式训练。

## 实验：Scene Cache V1.1 variable-asset 与 sparse-dense

**假设**

每个资产固定采样 4096 点、不同 sequence 允许不同 asset 数量，左右手作为两个独立的 1538 点 sample，使用 uint32 sampling index 和 active-frame-only dense mmap，不应改变 Cm 的单手输入合同或 loss。

**观察到的失败 / 现象**

V1 schema 固定 root scene pool 大小且 sampling index 为 uint16；dense builder 会先在内存中创建整条 sequence 的 feature tensor。

**诊断**

这两点会阻碍多资产 sequence 和大 scene pool，并造成不必要的 inactive frame 计算/存储。

**改动**

升级 schema 为 `ref2dex_cm_scene_v1_1`，增加 `asset_offsets`/`scene_asset_id`，取消 sequence 间 pool-size 一致性校验；sampling bank 改为 uint32；DenseToken bank 使用 `open_memmap`、`active_frame_id` 和 `frame_to_dense`；新增 V1.1 配置和 variable-asset 测试。

**结果**

`python3 -m pytest -q tests/test_cm_scene.py`：10 passed。验证了不同 scene pool 共存、uint32 index、candidate/flow/loss parity、FP16 dense parity，以及 inactive frame 不产生 dense feature row。

**决策**

保留 V1.1 实现。真实 GRAB 资产数量和正式训练仍需在原始数据可用环境执行。

## 实验：真实 GRAB cup_lift V1.1 smoke

**假设**

V1.1 cache pipeline 可以在真实 GRAB、MANO 和冻结 DenseToken checkpoint 上运行，并将左右手作为独立 sample；没有 local candidate 的 side 不应阻塞整个 sequence。

**观察到的失败 / 现象**

`cup_lift` 开头 16 帧没有 candidate；完整 934 帧中 left stream 没有 candidate，right stream 有 764 个 active current-frame 起点。首次 dense 构建将空 left stream 当成错误。online PTv3/spconv 在同一 CUDA 输入重复前向也不稳定，抽样 z_scene/z_hand 最大差分别约 `1.68/1.94`，严格 online-vs-cache FP16 parity 未通过。

**诊断**

空 side 是 V1.1 合法的零样本 stream，应写入 `active_frame_id=[]` 和全 `-1` 映射。DenseToken 差异来自 spconv implicit-GEMM CUDA backbone；输入特征和 scene bank index 已逐元素一致，启用 PyTorch deterministic 仍不能消除该 kernel 非确定性。

**改动**

空 stream 生成零行 sparse dense mmap；Scene dataset 在 `data.root` 模式下将空 `train_path` 绑定到 root，避免 shared split helper 扫描仓库 cwd。

**结果**

真实 cache：`s1/cup_lift`，934 帧，scene pool `8192=[4096 manipulated + 4096 table]`；left dense rows `0`，right dense rows `764`。train-only calibration：`flow_target_rms_m=0.053138431`，`flow_target_scale=18.8187717`，`statistics_scene_environment_candidate_points=87692`。真实缓存读取 shape 为 scene `512`、hand `1538`、token dim `96`。

缓存特征路径 20-step CUDA 训练 smoke 成功：最终 epoch loss `0.604981`，整体 EPE `34.2918 mm`，object EPE `35.3342 mm`，environment EPE `0.19496 mm`，梯度有限且无 NaN/Inf。`tests/test_cm_scene.py`：10 passed。

完整 `tests` 在 7 个测试后被已有 `tests/test_base_runner_max_steps.py` 的 `BaseRunner.performance` 缺失错误阻断，与 Cm 改动无关。

**决策**

保留 sparse cache 和 cached-feature 训练路径；严格 online/cache parity 记录为当前 spconv kernel 的未通过 gate，不把该 smoke 解释为正式全量效果。正式全量 dense cache 前需要决定是否接受该非确定性，或更换确定性 backbone/kernel。

## 实验：V1.1.1 DenseToken 三层 parity benchmark

**假设**

严格逐元素 online/cache parity 不是必要条件；如果 cache 与 online 的 downstream Cm flow 差异不超过 online-online 的 kernel 波动，则可以接受 cache 并进入全量链路。

**观察到的失败 / 现象**

原始 smoke 输出目录已被清理，因此先按相同配置重跑 20-step cached-feature smoke，生成临时 Cm checkpoint。benchmark 固定 `cup_lift` 的 `32` 个 sample、`stride=1`、`bank=0`，并分别比较两次 online forward 与 cache；cache 构建时使用 batch size 64，benchmark 复核使用 batch size 32。

**诊断**

feature 层仍受 spconv kernel 和 batch shape 影响：batch 32 时 object/hand 的 online-online RMS 为 `0.0486/0.0361`，online-cache RMS 为 `0.3627/0.3386`。但 Cm downstream 输出变化很小：output flow RMS 为 online-online `4.61e-7`、online-cache `5.97e-6`；EPE 为 cached `10.464178 mm`，两次 online 为 `10.463841/10.463593 mm`，最大差约 `0.0006 mm`。

**改动**

新增 `research/dense_cache_v1_1_1/benchmark_dense_cache.py`，固定 dataset view 后报告 feature、Cm output 和 flow EPE 三层指标。同步清理 Stage4 V1.1 过时的固定 8192 文案、sampling bank 未使用变量，并补充 V1.1.1 文档索引。

**结果**

downstream online-cache 差异远小于训练 smoke 的毫米级误差和数据/模型变化，V1.1.1 的行为 parity gate 通过；feature 绝对值差异保留为 spconv 非确定性诊断，不再作为阻塞条件。

**决策**

保留 sparse DenseToken cache，允许进入全量 scene/sampling 链路；全量 dense cache 仍需根据 scene cache 后的 active-frame 总量和磁盘估算决定具体执行规模。

## 实验：V1.1.2 正式两层 cache 路线

**假设**

Scene geometry/candidate 与 B=4 sampling bank 能去除重复的原始数据、MANO、mesh 和 5cm 搜索开销；DenseToken feature bank 的数百 GB 存储、构建时间和随机 IO 成本高于收益，正式训练改为 online Frozen DenseToken 更合适。

**观察到的失败 / 现象**

全量扫描在 1335 条原始 sequence 中成功构建 1255 条，80 条因 environment 持续移动而不满足 `static_world` schema。现有 325857 个 active side-frame 的完整 DenseToken bank 预算为约 `517 GB`；运行约 20 分钟只完成 158 条 sequence 的部分 cache，root 从 71 GB 增长到约 150 GB。

**诊断**

Level 3 每个 active side-frame、B=4、token dim 96 需要约 `1.51 MiB`，且必须预先运行四次 Frozen PTv3。该成本只在大量重复 Cm ablation 时可能摊回；当前正式路线没有足够收益支持该预计算。

**改动**

停止全量 DenseToken 构建并删除 `cm_scene_v1_1_full/**/dense_bank`，回收约 79 GB；保留 71 GB 的 Level 1/2 cache、2510 个 sampling bank 和 calibration。正式 config 改为 `use_dense_cache: false`，训练时在线运行 Frozen DenseToken；feature bank builder 保留为显式诊断/消融工具。

**结果**

前两层 root 统计为 1255 sequences、2510 sides、765512 side-frames、325857 active side-frames。sampling bank 完成 2510/2510；train-only calibration 为 `flow_target_rms_m=0.0927736881`、`flow_target_scale=10.7789183`。删除后 `dense_bank=0`，sampling index 文件仍为 2510，root 占用 71 GB。

真实 `cup_lift` online Frozen DenseToken 3-step smoke 成功（`amp=False`）：epoch loss `0.249824`、EPE `14.5185 mm`、grad norm mean `0.540904`，无 NaN/Inf。该结果只验证删除 Level 3 后的训练链路可运行，不作为正式效果结论。

按无正式 split 的第一种方案，对 `cm_scene_v1_1_full` 显式设置 `split_json_path=null`，限制前 6 个 sample、batch size 2，在 `cuda:5` 上完成 3-step online smoke（`amp=False`）。epoch loss `1.33097e-05`、EPE `0.0812016 mm`、grad norm mean `0.0217312`，三步均为有限值并正常退出。前 6 个 sample 的极小 flow 不具备效果代表性，该实验仅确认全量前两层 root 的 loader、sampling bank 与 online Frozen DenseToken 训练路径可用。未生成新的正式 train/val/test split。

**决策**

保留两层 cache 路线，撤回 DenseToken 全量预计算。下一步使用 online Frozen DenseToken 做真实训练和 evaluation；80 条动态 environment sequence 暂不进入 static-world 数据集。
