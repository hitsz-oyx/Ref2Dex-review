# ACT-20260922-CMV2-V114B-PORTABLE-BILATERAL

- timestamp：`2026-09-22T00:03:37+08:00`
- activity_id：`ACT-20260922-CMV2-V114B-PORTABLE-BILATERAL`
- work_version：`V1.14`
- base_commit：`0b04ee12774bccd9e50d1e9467412d260afbf8b5`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope：双手可移植数据入口、六组可配置 split 与 V1.14a DDP runner
- level：`L2`；后续本机依赖安装为 `L3`
- approval：用户于 2026-09-21/22 明确确认 V1.14b 范围并要求执行，并于 2026-09-22 明确允许自行安装缺失依赖

## 完成内容

- 便携 raw subset 在包内复核 SHA256 后从仓库根迁入忽略目录
  `data/raw_data/ObjectInteractionCmv2/oicmv2_mano_raw_subset_5x5/`；payload SHA 保持
  `6a6ebfff204fa7c1e1168bfedac988638dc5978df8344dc183d67b2ff6653fad`。
- 新增 OakInk2 raw→既有 `train_corr_static_v2` Stage3 contract 的 producer，输出左右手距离、静态物体
  点/法向和逐帧部件位姿；后续继续复用既有 single-object index、temporal selection、MANO/Inspire producer
  和 part adapter。
- GRAB/ARCTIC 与 OakInk2 Inspire retarget 支持 dex-retargeting checkout 或 site-packages 两种布局；本机左右
  retargeter 均成功初始化，表面池均为每侧 10135 点。
- 新增六组 registry、显式 active groups、归一化 group weights、动态 split builder 和 V1.14a DDP runner；
  本地配置固定 GRAB MANO/Inspire 两组，服务器模板覆盖三域六组。模型入口仍固定每手 2048 点。
- OakInk2 高分辨率 Inspire backfill 不再强制要求历史 `existing_root`，可从 portable raw 派生产物独立构建。
- 按 [V1.14b.1](../plan/V1.14b.1.md) 在专用 Conda 环境 `graspenv` 安装 `manotorch v0.0.2`，来源固定为
  `lixiny/manotorch@da235c9096a5560e2a09102c20f61abb4d0d2309`。安装前后 `pip freeze` 对比只新增该包，未升级
  PyTorch、NumPy 或其它依赖。
- OakInk2 materializer 增加仅改变读取窗口的 `--frame-offset`，便于 bounded smoke 直接选择已知交互帧；默认值
  为 `0`，正式全量物化语义不变。

## 可移植执行顺序

OakInk2 在本机或服务器上按以下入口顺序物化，所有输出写入 `data/processed_data/` 或外部 cache 根：

1. `materialize_oakink2_stage3_v114b`：raw annotation/mesh → 既有 Stage3 contract；
2. `build_oakink2_single_object_index`；
3. `export_oakink2_inspire_v1_4 select`：复用既有 official-30 Hz temporal selection；
4. `build_oakink2_mano_highres_v1_4` 与 `backfill_oakink2_inspire_v1_4`（portable 路径可省略
   `--existing-root`）；
5. `build_oakink2_part_adapter_v1_11h`；
6. `build_active_group_split_v114b`；
7. 经单独批准的配置将 `run_authorization` 改为 `approved` 后，使用 `train_part_se3_v114_ddp launch`。

ARCTIC/GRAB 继续复用 bilateral Stage4 与 `build_stage4_inspire_highres_v1_4 --domain grab|arctic`；新 split
builder 负责把 `arctic/inspire_f1` 纳入可选第六组。

## 验证

- `PYTHONPATH=. .../graspenv/bin/pytest -q src/task/ObjectInteractionCmv2/tests`：`78 passed`；新增定向测试单独为
  `6 passed`。
- Python 语法编译、`git diff --check` 和 `python3 tools/verify.py --changed`：通过。
- 本机 dex-retargeting runtime：left/right 初始化通过，surface sampling 均为 `(10135, 3)`。
- OakInk2 fixture 验证了 deterministic mesh sampling、object-frame 变换和左右 Stage3 NPZ consumer contract。
- runtime 身份：Python `3.8.20`、PyTorch `2.4.1+cu121`、`manotorch 0.0.2`、dex-retargeting `0.4.5`；
  左右 quaternion MANO layer 均能从本地 MANO assets 初始化。
- 真实 OakInk2 双手交互窗口 smoke：sequence offset `1`、frame offset `2100`、`8` 帧、CPU，run status
  `COMPLETED`；生成 5 个临时 Stage3 NPZ（left `2`、right `3`），共 40 object-side frames，`failures=0`。
  所有文件满足 `train_corr_static_v2`、`obj_points=(4096,3)`、`hand_points=(8,1538,3)` 且数值 finite。临时证据
  位于 `/tmp/cmv2-v114b1-oakink-bilateral.KiFZgI`，不属于正式 cache。

## 保护、限制与回滚

- 未改变 GT、坐标、单位、loss、stride、KNN32、2 cm 或 checkpoint 解释；未启动 full cache 或训练。
- OakInk2 官方 quaternion MANO 重建必须使用 `manotorch`；不能用未验证的 MANO 实现静默替代。其它服务器执行前
  仍需固定同一提交并先跑一条真实序列 smoke。
- 本机环境回滚：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m pip uninstall manotorch`；代码回滚不自动
  修改该外部 Conda 环境。
- 回滚代码到 base commit；数据只需把已校验的 raw bundle 移回原位置，不删除任何 payload 或旧 cache。
