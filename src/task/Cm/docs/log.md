# Cm 研究日志

## 当前研究状态

V1.2 已切换为 object-only GRAB + ARCTIC 数据合同，模型和 point-flow loss 保持不变，DenseToken 首阶段 frozen。object mmap、ragged candidate、B=4 sampling bank、统一 Dataset、sequence-disjoint loader、分数据集指标和 action intervention 指标已经实现。

真实 GRAB `s1/cup_lift` 与 ARCTIC `s05/laptop_use_01` 已通过 E0/E1 和 mixed Runner smoke；smoke flow RMS 分别约为 31.08mm 与 64.05mm。该样本不足以确定全量 sampler 或 normalization。下一步是构建 full-data object-v2 cache，完成 GRAB-only、ARCTIC-only、mixed 三组 5k--10k step 短训。

---

## 2026-08-18：V1.2 object-only implementation、E0/E1 与 mixed smoke

**假设**

现有 GRAB/ARCTIC Stage4 已提供相同 object-only schema，因此无需重复 MANO forward、object surface sampling 或 candidate cdist；数据变化可在不修改模型和 loss 的条件下单独验证。

**改动**

- 新增 Stage4 NPZ 到 object mmap/ragged cache 的转换器。
- 新增 `uint32 [T,4,512]` sampling bank。
- 新增统一 GRAB/ARCTIC Dataset、sequence-disjoint loader 和 E1 统计命令。
- `dataset_id` 只用于分数据集 EPE/GT norm，不输入模型。
- evaluation 增加 zero-flow、action shuffle 和 action reverse 指标。
- 修正 ARCTIC root 自动探测，实际 raw 位于 `data/raw_data/ARCTIC/arctic/data/arctic_data/data/raw_seqs`。

**结果**

GRAB `s1/cup_lift`：934 帧，右手 764 个 active current-frame 起点，左手无 active 起点；smoke 统计为 764 samples，flow RMS `0.0310844 m`。

ARCTIC `s05/laptop_use_01`：64 帧，左右 candidate 中位数为 570/500；smoke 统计为 53 samples，flow RMS `0.0640490 m`。ARCTIC/GRAB RMS 比约 2.06，超出指导中的 0.7--1.4 初始接近区间。

E0 shape/finite/delta-time gate 通过：object `[512,3]`、hand `[1538,3]`、30Hz 下 `delta_time_s=1/30`。GRAB-only 1-step CUDA Runner loss `0.0125128`、EPE `0.0715053 mm`；mixed 1-step loss `0.0125159`、EPE `0.150698 mm`，均无 NaN/Inf。相关测试 `30 passed`。

**结论**

保留当前 V1.2 实现，implementation 与真实 smoke gate 通过。smoke root 每个 dataset 只有一个 sequence，不能替代正式 sequence split、混合比例统计或三组短训；full-data 统计前不引入 dataset-specific normalization。

---

## Scene Cache V1 合成 parity

**假设**

将 object-only 输入扩展为 object + environment 的 local scene，并使用 mmap geometry、ragged candidate、sampling bank 和 DenseToken bank，不应改变旧 Cm 的输入/损失语义；静态 environment 点应产生零 flow。

**改动与结果**

增加 Scene Cache V1 构建器、通用 environment asset 读取、ragged candidate、sampling bank、DenseToken bank、fingerprint 校验、scene dataset/runner 路由和 object/environment 诊断指标。`tests/test_cm_scene.py` 为 9 passed，覆盖 object-only regression、candidate parity、sampling determinism/diversity、FP16 cache parity、fingerprint guard、loss parity 和 flow calibration。

**结论**

保留实现；当时未具备真实 GRAB 数据，不把合成 parity 解释为真实训练证据。

## Scene Cache V1.1 variable-asset 与 sparse-dense

**假设**

每个资产固定 4096 点、不同 sequence 允许不同 asset 数量，左右手独立为 1538 点 sample，使用 uint32 sampling index 和 active-frame-only dense mmap，不应改变单手输入合同或 loss。

**改动与结果**

schema 升级为 `ref2dex_cm_scene_v1_1`，增加 `asset_offsets`/`scene_asset_id`，取消 sequence 间 pool-size 一致性；sampling bank 改为 uint32；DenseToken bank 使用 `open_memmap`、`active_frame_id` 和 `frame_to_dense`。`tests/test_cm_scene.py` 为 10 passed，覆盖不同 pool、candidate/flow/loss parity、FP16 dense parity 和 inactive frame 行为。

**结论**

保留 V1.1 variable-asset 与 sparse cache 实现。

## 真实 GRAB cup_lift V1.1 smoke

**现象与诊断**

`cup_lift` 共 934 帧；left stream 无 candidate，right stream 有 764 个 active 起点。空 side 是合法零样本 stream，应写空 `active_frame_id` 和全 `-1` 映射。online PTv3/spconv 重复前向存在 kernel 非确定性，z_scene/z_hand 最大差约 `1.68/1.94`，严格逐元素 online/cache parity 未通过。

**结果**

scene pool 为 `8192=[4096 object + 4096 table]`。train-only calibration 为 `flow_target_rms_m=0.053138431`、scale `18.8187717`。cached-feature 20-step smoke 最终 loss `0.604981`、整体 EPE `34.2918 mm`、object EPE `35.3342 mm`、environment EPE `0.19496 mm`，梯度有限且无 NaN/Inf。

**结论**

保留 sparse cache；非确定性必须通过 downstream 行为 parity 判断，不能把严格 feature parity 失败误记为科学反证。

## V1.1.1 DenseToken 三层 parity benchmark

**假设**

如果 cache 与 online 的 downstream Cm flow 差异不超过 online-online kernel 波动，则无需要求 feature 逐元素一致。

**结果**

固定 32 samples、stride 1、bank 0。feature online-online RMS 为 object/hand `0.0486/0.0361`，online-cache 为 `0.3627/0.3386`；但 output flow RMS 仅为 `4.61e-7/5.97e-6`。EPE 为 cached `10.464178 mm`，两次 online 为 `10.463841/10.463593 mm`，最大差约 `0.0006 mm`。

**结论**

downstream parity gate 通过；feature 绝对差异保留为 spconv 非确定性诊断，不再阻塞全量链路。

## V1.1.2 正式两层 cache 路线

**现象与诊断**

1335 条 GRAB sequence 中成功构建 1255 条，80 条因 environment 持续移动不满足 `static_world`。325857 个 active side-frame 的完整 DenseToken bank 预算约 517GB；运行约 20 分钟仅完成 158 条，root 从 71GB 增至约 150GB，成本高于当前收益。

**改动与结果**

停止并删除部分 DenseToken bank，回收约 79GB；保留 71GB geometry/candidate/sampling cache、2510 个 sampling bank 和 calibration。正式配置改为 `use_dense_cache: false`，在线运行 Frozen DenseToken。前两层 root 为 1255 sequences、2510 sides、765512 side-frames、325857 active side-frames；train-only RMS `0.0927736881 m`、scale `10.7789183`。

真实 `cup_lift` online 3-step smoke：loss `0.249824`、EPE `14.5185 mm`、grad norm `0.540904`。全量 root 前 6 samples 的 3-step smoke：loss `1.33097e-05`、EPE `0.0812016 mm`、grad norm `0.0217312`。两者只证明 loader、sampling bank 与 online Frozen DenseToken 路径可运行。

**结论**

保留两层 cache，撤回全量 DenseToken 预计算。Scene Cache V1.1.2 作为历史对照保留；当前主路线转向 V1.2 object-only multi-dataset。
