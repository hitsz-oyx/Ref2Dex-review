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

实验参数通过重复传入 --set 覆盖，例如：

    --set train.epochs=50
    --set meta.loss_hand_contact_weight=0.005
    --set meta.obj_rot_std_deg=10.0

输出目录中的 config.json 是该 run 的最终有效配置，应优先于本文或 baseline.yaml。

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
      --checkpoint outputs/train/<run_name>/checkpoints/best.pt \
      --input <single_stage3_sequence.npz> \
      --device cuda

可增加 --check-only 进行无窗口的 checkpoint 和数据兼容性检查。

可视化有两个相互独立的主模式：

| 按键 | 模式或操作 |
| --- | --- |
| H | CrossEdge / HandHeatmap |
| G | 当前模式的 GT / Eval |
| , / . | 选择 CrossEdge 的 object point |
| P | 手工开关 object pose 扰动 |
| F | 扰动时 RefGT / CurrentPseudo |
| A / D、[ / ]、R | 帧、sampling epoch、相机控制 |

CrossEdge Eval 使用 shared cross-edge head 对选中物点和所有手点做 dense 推理。HandHeatmap Eval 只使用 pred_hand_contact_prob，不会混用 CrossEdge 颜色或概率。HandHeatmap 需要 checkpoint 在训练时启用 meta.loss_hand_contact_weight。
