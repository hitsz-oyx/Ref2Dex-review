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
