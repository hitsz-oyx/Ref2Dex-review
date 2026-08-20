# correspondence_ptv3_v2

这是当前使用的 PTv3 手物 correspondence 任务。模型输入为 hand-root 坐标系中的物体与手点云；训练时只扰动物体的输入位姿，仍以 clean 手物几何监督接触关系。

当前模型包含三条监督流：

1. Random128 cross-edge 主流：每个有效物点均匀、无放回抽取 128 个手点，使用连续 contact target 和 QFL。
2. Contact-aux cross-edge 辅助流：按 contact strength 分层采样，并加入 2 到 3 cm hard negative；与主流共享同一个 edge head。
3. Dense hand-contact heatmap：每个手点预测它到完整物体表面点池的接触概率，使用弱权重 BCE。

默认 contact target 为：

    y(d) = clamp(1 - d / 0.02, 0, 1)

默认 loss 为：

    L = 1.0 * random128_QFL
      + 0.05 * contact_aux_QFL
      + 0.005 * dense_hand_BCE

完整数据、模型、监督流、指标与扰动语义请看 [架构.md](架构.md)。

## 数据要求

输入是由 process/common/stage3_corr.py 生成的 Stage 3 v2 npz。当前默认要求 coordinate_frame 为 hand_root；训练会拒绝与配置不一致的数据。新 schema 只保留 v2 所需字段，不兼容旧 correspondence_ptv3 或旧 render 脚本；已有包含这些最小字段的旧 v2 数据仍可读取。

典型数据规格：

| 内容 | 数量 |
| --- | ---: |
| 完整 object point pool | 4096 |
| 每次运行时采样 object points | 512 |
| hand points | 1538 |
| random 主监督边 | 每物点 128 |
| auxiliary 边 | 每物点最多 80 |

每个 epoch 都从 5 cm object candidate pool 确定性重采样 512 个物点。验证固定采样 epoch 0，并同时产生 val_clean 与 val_perturbed；后者只对输入物体应用共享 SE(3)，GT 仍是 clean geometry。

## 训练

在 graspenv 环境中运行：

    PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.train \
      --config src/task/correspondence_ptv3_v2/configs/baseline.yaml \
      --data <stage3_root> \
      --output-dir outputs/train/<run_name>

baseline.yaml 当前默认使用 AdamW、lr 1e-4、cosine + 3% warmup、100 epoch、batch size 8、val batch size 24，并启用 object pose perturbation 和 dense hand heatmap 监督。

OakInk 的 clean Stage 3 没有 MANO pose/betas，可用存储的 1538 个 hand face-center 代理点保留 full-pool runtime object resampling，而不执行 MANO forward 或 PCA 扰动：

    PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.train \
      --config src/task/correspondence_ptv3_v2/configs/oakink_50ep_no_hand_perturb_runtime.yaml \
      --set train.output_dir=output/exp/<run_name>

该配置必须同时设置 `use_mano_reconstruction=false`、`apply_hand_perturb=false`、`runtime_resample_object=true` 和版本化的 `stored_hand_proxy_indices_path`。缺少 proxy 时 runner 会 fail-fast，不会静默回退到 MANO 或 face-index linspace。

实验参数通过重复传入 --set 覆盖，例如：

    --set train.epochs=50
    --set meta.loss_hand_contact_weight=0.005
    --set meta.obj_rot_std_deg=10.0

输出目录中的 config.json 是该 run 的最终有效配置，应优先于本文或 baseline.yaml。

### MANO 几何噪声标定

不要直接假设相同的 PCA/axis-angle 系数标准差会产生相同的手部变化。可从 Stage 3 v2.1 数据统计每个 MANO pose 维度对手部几何的实际影响：

    python tools/calibrate_mano_geometry_noise.py <stage3_root> \
      --mano-model-dir dataset/arctic/data/body_models/mano \
      --output tmp/mano_geometry_calibration/<dataset>_target_9mm.json \
      --max-files 64 \
      --max-samples 1024 \
      --target-rms-mm 9

标定固定 `global_orient=0`、`transl=0`，并在每次 forward 后减去各自的 MANO wrist joint，以 778 个 root-aligned 顶点的 RMS 位移（mm）作为度量。GRAB PCA 默认以统一系数 `std=0.5` 为参考；ARCTIC axis-angle 默认以 `std=0.05 rad` 为参考。输出包含逐维灵敏度、参考噪声的位移分布，以及命中 `--target-rms-mm` 中位 RMS 的 inverse-sensitivity 候选尺度；省略目标时匹配参考中位数。该命令只写统计 JSON，不会自动改训练配置。

训练与可视化通过 `dataset_id + side + MANO representation/dimensions/mean` 选择 profile；旧文件缺少 `dataset_id` 时会按 MANO 描述回退推断。三数据集本地兼容检查可使用：

    src/task/correspondence_ptv3_v2/configs/mixed_stage3_geometry_9mm_sample.yaml

该配置严格要求 GRAB PCA24、ARCTIC axis-angle45 和 ContactPose PCA15 的 9 mm profile 都能命中。工程 `tmp/mano_geometry_calibration/` 中 GRAB JSON 来自 1024 帧统计；文件名带 `_sample` 的 ARCTIC 和 ContactPose JSON 是用于兼容验证的初步样本统计，全量训练前应从更广的数据覆盖重新标定。只有启用 `meta.apply_hand_perturb=true` 时才要求样本携带 MANO 字段；关闭手部扰动时，旧 Stage 3 的 `schema_version` 只作为标识，不阻止加载。但所有混合数据仍必须使用同一个 `coordinate_frame`；若要启用 MANO 扰动，应使用 `process/ContactPose/stage3_export.py` 导出带 MANO 字段的数据。

## 评估

    PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.eval \
      --checkpoint outputs/train/<run_name>/checkpoints/best.pt \
      --device cuda

best checkpoint 默认按 val_clean/cross_edge_random_qfl 选择。这保证主流抽样分布可比，但在类别不均衡下不应只看它；建议同时检查：

- val_perturbed/pseudo_recovery_brier 和 pseudo_recovery_projection
- fake-contact 与 missed-contact 两类 recovery
- random128 的 nonzero/zero 分布、target-strength bin MAE
- dense hand-contact 的 nonzero MAE

Pseudo recovery 衡量模型相对“直接相信扰动物体几何”的伪 contact target 向 clean GT 修正了多少；它只在 val_perturbed 中有意义。

## 继续训练

普通 resume 直接指定 train.resume。若要从已有 checkpoint 开一个新的余弦微调阶段，使用 cosine_restart，并注意 max_steps 是绝对停止 step：

    --set train.resume=<checkpoint.pt> \
    --set train.scheduler=cosine_restart \
    --set train.lr=5e-5 \
    --set train.min_lr=5e-6 \
    --set train.finetune_steps=<new_phase_steps> \
    --set train.max_steps=<checkpoint_step + new_phase_steps> \
    --set train.warmup_steps=0 \
    --set train.warmup_ratio=0

## 可视化

    PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.visualize \
      --config src/task/correspondence_ptv3_v2/configs/baseline.yaml \
      --input <single_stage3_sequence.npz> \
      --device cuda

省略 `--checkpoint` 时为 GT-only 模式：CrossEdge 和 HandHeatmap 都显示数据集 GT，仍可使用 `P` 和平移/旋转控制查看手工 object 扰动。v2.1 数据还会启用独立的 MANO hand perturbation 控制：按样本描述自动选择 PCA 或 axis-angle 路径，`1.0x` 对应 YAML 中的噪声强度，variant 可在不改变帧和 object 采样的情况下切换稳定噪声。传入 `--checkpoint` 后才启用 Eval 着色；`--input` 省略时使用 YAML 中的 `data.val_path`，再回退到 `data.train_path`。

可增加 --check-only 进行无窗口的 checkpoint 和数据兼容性检查。

可视化有两个相互独立的主模式：

| 按键 | 模式或操作 |
| --- | --- |
| H | CrossEdge / HandHeatmap |
| G | 当前模式的 GT / Eval |
| , / . | 选择 CrossEdge 的 object point |
| P | 手工开关 object pose 扰动 |
| M | 开关 MANO hand 扰动（仅 GT-only v2.1 数据） |
| F | 扰动时 RefGT / CurrentPseudo |
| A / D、[ / ]、R | 帧、sampling epoch、相机控制 |

CrossEdge Eval 使用 shared cross-edge head 对选中物点和所有手点做 dense 推理。HandHeatmap Eval 只使用 pred_hand_contact_prob，不会混用 CrossEdge 颜色或概率。HandHeatmap 需要 checkpoint 在训练时启用 meta.loss_hand_contact_weight。
