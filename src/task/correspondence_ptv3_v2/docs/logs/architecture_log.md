# correspondence_ptv3_v2 architecture log

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-09-01
- last_verified: 2026-09-01
- related: [当前状态](status_log.md) / [仓库记忆](repo_memory.md) / [实验](experiment_log.md)

## 当前架构入口

详细 pipeline 保留在任务文档 [`../架构.md`](../架构.md)。本日志记录该架构最近的跨数据集约定：

1. Stage 3 输出提供 clean full object pool 和 clean hand geometry。
2. Dataset 可对 full pool 应用共享 object SE(3)，将整个 pool 交给 runner。
3. Runtime 模式下 runner 在受扰的 full pool 上均匀随机抽取 512 个 object points；不使用
   clean-GT 近邻、global 配额或 hand-FPS proxy。

可选的 `data.domain_paths[*].array_cache_path` 指向与域目录保持相同相对路径的
未压缩 NPZ sidecar。Dataset 在 schema 校验、交互帧过滤和 sample 读取时优先使用
sidecar；缺失时默认回退源 NPZ，设置 `array_cache_required=true` 则 fail-fast。sidecar
只改变 IO 表示，不改变字段、frame index、seed、GT 或扰动协议。

OakInk2 object-centered Stage3 对 frame-invariant 的 canonical object pool 采用紧凑
存储：`obj_points`/`obj_normals` 可保存为 `[4096,3]`，Dataset 读取时以 `raw_frame_id`
长度广播为 `[T,4096,3]` 视图；逐帧的 `hand_points`、距离、root pose 和 MANO 字段仍保持
`[T,...]`。这只改变存储布局，不改变 runner 输入和 4096→512 runtime sampling 合同。

## HRDexDB-human smoke 适配

`process/HRDexDB/stage3_export.py` 当前支持单 episode 的 clean geometry 验证。它从 HRDexDB human 的 778-vertex/1538-face MANO OBJ、每帧 object 4×4 pose 和 `v0_nonvideo/assets/mesh` 物体网格生成标准 Stage 3 v2.1：物体池 `[T,4096,3]`、MANO face-center 手点 `[T,1538,3]`、法向和 `hand_to_obj_min_dist`。HRDexDB human 的 MANO OBJ 与 object pose 已处于同一 world frame；smoke 以 MANO `global_orient` 和 `joints[0]` 构造当前帧 `hand_root`，不应用机器人专用 `C2R.npy`。

`--include-mano` 选项会把矩阵转换为 axis-angle45，并写入 `mano_global_orient`、`mano_transl`、`mano_pose`、`mano_betas`、标准 `mano_v_template` 及 reconstruction descriptors。对 `human/apple/0` 的 257 帧核验显示，`MANO_RIGHT + flat_hand_mean=True` 重建 OBJ 的 vertex RMSE 为 `6.33e-8 m`，最大误差 `2.78e-7 m`；因此该参数链路可以启用 hand perturbation。NAS 的 clean smoke 为 `hrdexdb_human_smoke_v1`，带 MANO 字段的验证版本为 `hrdexdb_human_smoke_v2_mano`，二者都不是训练配置中的 domain。

## HRDexDB-robot FK / q-space 路径

机器手不进入 MANO reconstruction。`process/HRDexDB/robot_stage3_export.py` 当前先支持 `inspire_dftp`，将原始 Inspire qpos 插值到 object-pose 帧，固定采样 4096 个物体点和 1538 个 link/barycentric 手点，并以 `base_link` 的世界位姿建立 `hand_root` 坐标系。Stage 3 同时保存 `robot_qpos`、`robot_hand_qpos_indices`、URDF q 限位、`robot_c2r`、每个点的 link-local 坐标/法向、link index、URDF 路径和 clean `hand_root_pose`。

`robot_recon.py` 在 dataset 取样时仅对 hand q joints `[6..11]` 加 `robot_hand_qpos_noise_std`（当前 smoke 为 `0.03 rad`），再按 URDF limits 截断并执行 FK；arm/base qpos 固定，因此输入 hand-root 坐标系不随扰动漂移。扰动后的手点是模型 input，存储 clean 手点仍是 GT；普通和 domain-balanced dataloader 都通过 `meta.use_robot_reconstruction` / `meta.apply_robot_perturb` 传递该分支。

当前已验证 `inspire_dftp/mug_holder/2` 64 帧 smoke：clean FK 与存储点 RMSE 约 `2.3e-8 m`，GPU PTv3 单 batch 前向/反向 finite。七域训练中机器人样本仍走该独立 FK 路径；dataset 在 mixed mode 下按样本检查 robot contract，q-space 噪声乘数按 domain dispatch（当前坐标-RMS 复核为 DFTP/F1/Allegro 分别为 25/15/9），实际值由 `robot_perturb_rms_m` 记录。

## HRDexDB minimal allhands 适配器

`process/HRDexDB/correspondence_minimal_adapter.py` 读取
`HRDexDB_correspondence_minimal_allhands_20260825.tar.gz` 解压后的
`v0_nonvideo`：集中式 `object_6d_pose_v2` 优先、缺失时回退 `v1`，物体网格从
`assets/mesh_v2/<object>` 解析。human 写入已验证的 MANO axis-angle45 字段；
Inspire DFTP、Inspire F1 和 Allegro V5 分别使用对应 URDF/FK，并保存 robot
q-space 扰动所需字段。输出仍遵守 `[T,4096,3]` object pool、`[T,1538,3]`
hand points 和 `hand_root` 坐标合同。

当前 smoke 四手型均通过；全量转换按 human=441、Inspire DFTP=618、Inspire
F1=576、Allegro V5=453 共 2088 个 C2R-valid episode 已完成，输出目录为
`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_v1`。

## ContactPose MANO / Stage3 边界

ContactPose 原始 `data/contactpose_data/full*_{use,handoff}/<object>/` 序列包含
`mano_fits_15.json`、`betas`、18 维 MANO pose（3 维 global axis-angle + 15 维 PCA）和
`mTc` 变换。当前 NAS 的 `use_stage3_v2` 是历史 schema `2.0.0`，只保存预计算的
hand geometry，因此不能开启 `use_mano_reconstruction`；这不是原始数据能力缺失。
`process/ContactPose/stage3_export.py` 已能将这些字段写入 v2.1，并把 MANO 输出与
`mTc` 变换保持一致；同时在导出时保留 `raw_frame_id`，过滤
`min(hand_to_obj_min_dist) > 0.05 m` 的非交互帧。全量 `use` 版本已写出 1477 个
单手 NPZ、621514 帧，`num_skipped=0`，每个文件含完整 MANO contract；loader
过滤后实际训练帧为 554608。

## OakInk true hand-root / MANO 路径

2026-08-24 重新导出的 OakInk 样本使用官方 `general_info` 中的 root quaternion、`hand_tsl` 和 `cam_extr` 构造 `T_camera_root = T_camera_world @ T_world_root`，手、物体和法向统一变换到 rotation-canonicalized `hand_root`。同一物理帧的四个 camera view 在该坐标系中应重合，而不是只做 wrist 平移中心化。

新产物同时保存 axis-angle45 的 `mano_pose`、`mano_global_orient`、`mano_betas`、`hand_root_pose` 和标准 MANO template。OakInk 的 `hand_tsl` 是腕关节世界位置，不能直接作为 `smplx.MANO.transl`；导出器必须减去 shape-dependent MANO wrist template offset，使训练端 MANO forward 后的 joint 0 回到官方 `hand_tsl`。因此新产物可启用严格 MANO reconstruction 和后续手姿态扰动。

旧 wrist-centered camera-frame 产物只允许作为历史无手扰动数据使用；其 stored-hand proxy 路径仍可复现旧实验，但不是当前严格 hand-root 数据合同。

## 七域等比例混训路径

`configs/mixed_seven_domain_mano_robot_10mm_5cm.yaml` 使用 `data.domain_paths` 为
GRAB、ContactPose、OakInk、HRDexDB human 和三个 robot domain 分别构造
`CorrStaticDatasetV2`，再由
`DomainConcatDataset` 合并索引。`DomainBalancedSampler` 在每个本地 batch
中固定抽取七个域相同数量的样本，并在 DDP 下按 rank 切分每个域的采样流；它
不会按数据集 frame 数量加权，也不会把较小域静默耗尽后改变比例，而是
按确定性 seed 循环采样。

七域配置显式设置 `use_mano_reconstruction=true`、`use_robot_reconstruction=true`、
`mixed_hand_reconstruction=true`、手/物体扰动和 `runtime_resample_object=true`。
runner 将 batch 按 `has_mano` / `has_robot` 样本切片：MANO 子 batch 在 GPU
重建后回写，robot 子 batch 保留 dataset FK 几何；两类样本共享默认 collate
所需的惰性占位字段。Dataset 在索引阶段先应用 5 cm 交互帧过滤，再由 runner
在受扰 full object pool 上均匀重采样 512 点。训练增强按稳定 frame seed 固定为
hand-only 40%、object-only 40%、clean 20%，手与物体扰动不同时出现。每个域拥有独立的
`val_clean/<domain>/` 与 `val_perturbed/<domain>/` loader；七域 checkpoint 指标为各域
`cross_edge_random_qfl` 的等权宏平均，同时保留各域明细。

## MANO 扰动评估不变量

`research/contactpose_checkpoint_compare/evaluate.py` 的 hand-only / hand+object 条件必须显式固定 `hand_perturb_prob=1.0`，不得继承 checkpoint 的 H80/H50 训练门控比例。评估条件表示所有样本采用同一 corruption stream；训练 exposure 只能作为被比较的模型属性，不能改变评估输入分布。

历史 v2.0 checkpoint 的 config 可能没有 `meta.mano_model_dir`；评估器在 hand 条件下使用当前 Task `Config.meta.mano_model_dir` 作为只读兼容默认值，不修改 checkpoint，也不改变模型权重或协议扰动。

## Object-centered GRAB + Inspire F1 协议（2026-08-29）

新配置 `mixed_grab_inspire_f1_object_centered_root_pose.yaml` 仅使用 GRAB 与
Inspire F1。训练时优先读取由 `build_uncompressed_npz_cache.py
--coordinate-frame object` 一次性生成的 object-frame sidecar，源树保持只读。
未启用 sidecar 时仍可通过 `transform_to_object_frame=true` 在每个 worker 首次读入
NPZ 时按 `inv(obj_root_pose_world)` 惰性转换。转换后的 `obj_points` 来自完整 4096
点池，训练 runner 再均匀随机抽取 512 点；法向同步旋转，MANO/robot 世界姿态字段
不变。

增强模式由稳定 frame seed 决定：40% 手腕根刚体扰动、40% MANO/robot 手姿态扰动、
20% clean。根扰动在局部 hand-root 坐标中采样 10° / 10 mm 的 SE(3)，右乘到
`T_object_from_hand_root`，因此旋转中心是手腕根；姿态扰动与根扰动不叠加，物体侧
扰动关闭。验证 loader 分为 `val_clean`、`val_perturbed`（姿态）和
`val_root_perturbed`（根刚体）三类。

协议 E 使用 ARCTIC axis-angle45 的 10 mm RMS hand perturb；object 扰动保持 rotation std 10° / translation std 10 mm。绝对退化定义为同条件 `perturbed random QFL - clean random QFL`，并与 perturbed QFL、`pseudo_recovery_brier` 并列报告。
