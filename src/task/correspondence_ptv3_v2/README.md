# correspondence_ptv3_v2

`correspondence_ptv3_v2` 是一个独立任务，用于从 noisy hand-object geometry 中预测：

1. object point contact target
2. sampled object-hand cross-edge contact target

固定设计：

- continuous 0-1 cm linear contact target
- uniform random 128 supervision edges per object
- Quality Focal Loss
- scalar sigmoid outputs
- PTv3 unified point backbone

说明：

- object points 仍从 Stage 3 的 `obj_candidate_mask_5cm` 候选池中随机采样
- cross-edge supervision 才是对全部 hand points 的 uniform random 128

训练：

```bash
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.train \
  --config src/task/correspondence_ptv3_v2/configs/baseline.yaml \
  --data <stage3_dataset> \
  --output-dir outputs/train/<run>
```

评估：

```bash
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.eval \
  --checkpoint outputs/train/<run>/checkpoints/latest.pt
```

可视化：

```bash
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.visualize \
  --checkpoint <checkpoint> \
  --input <stage3_npz>
```

交互键位：

- `A / D` 或左右键：切 frame
- `[` / `]`：切 sampling epoch
- `G`：GT / Eval
- `C`：heatmap / cross
- `,` / `.`：切 selected object point
- `R`：reset camera
