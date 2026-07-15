# correspondence_ptv3_v2

`correspondence_ptv3_v2` 是一个独立任务，用于从 noisy hand-object geometry 中预测：

1. sampled object-hand cross-edge contact target
2. dense hand contact heatmap（B/C 实验）

固定设计：

- continuous 0-2 cm linear contact target
- uniform random 128 supervision edges per object
- 额外 contact-aux 监督：4 个 target-strength bin + 2-3 cm hard negatives
- random128 主损失使用 Quality Focal Loss
- scalar sigmoid outputs
- PTv3 unified point backbone
- C 实验额外使用 `HandGlobalEncoder(H)`：它只读取 runtime hand、hand normal、
  canonical hand point、finger/region id；所得 32-D context 只拼到 hand token，
  object token 对应通道固定为 0，随后仍只执行一次 joint PTv3

说明：

- object points 仍从 Stage 3 的 `obj_candidate_mask_5cm` 候选池中随机采样
- cross-edge supervision 才是对全部 hand points 的 uniform random 128
- 当前 v2 dataset 要求 Stage 3 有 `hand_to_obj_min_dist`、`hand_cano_points`、
  `hand_finger_id` 与 `hand_region_id`。请使用当前 `prepare_corr_static.py` 生成的
  hand-root 数据，例如 `processed_data/generated/stage3/*_hand_root`。

建议按以下三组配置做消融，选择 checkpoint 的指标始终是无偏的
`val_clean/cross_edge_random_qfl`：

| 配置 | 实验变量 |
| --- | --- |
| `baseline.yaml` | A：原始 11-D joint PTv3 + cross-edge loss |
| `hand_heatmap.yaml` | B：A + object-conditioned hand heatmap BCE |
| `hand_context.yaml` | C：B + hand-only global context early fusion（研究分支） |

当前数据 pilot 中 B 优于 A；C 的 hand heatmap 更准但 correspondence 更差。因此默认训练 B：

```bash
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.train \
  --config src/task/correspondence_ptv3_v2/configs/hand_heatmap.yaml \
  --data processed_data/generated/stage3/grab_subset100_initonly_4096_ds4_hand_root \
  --output-dir outputs/train/<run>
```

复现实验记录见 [实验记录.md](实验记录.md)。C 仍应在更多 split/seed 与改进的融合方式下继续验证，
不应直接作为当前默认模型。

评估：

```bash
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.eval \
  --checkpoint outputs/train/<run>/checkpoints/best.pt
```

可视化：

```bash
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.visualize \
  --checkpoint outputs/train/<run>/checkpoints/best.pt \
  --input <stage3_npz>
```

如需临时放大 GT 着色范围，可额外传：

```bash
  --vis-contact-radius 0.02
```

交互键位：

- `A / D` 或左右键：切 frame
- `[` / `]`：切 sampling epoch
- `G`：GT / Eval
- `,` / `.`：切 selected object point
- `R`：reset camera
