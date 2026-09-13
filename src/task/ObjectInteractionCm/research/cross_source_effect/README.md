# 冻结 decoder 的跨源 object-effect 诊断

本实验测试混合源训练的同一个冻结模型在 MANO / Inspire RL 上的重建误差与对照差距。
完整输入为物体几何、Cm tokens 和动态 anchors；不是 token-only 表征测试，也不是 unseen-hand 测试。
范围见 [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../../docs/plan/V1.3.md) 第 7 节。

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.ObjectInteractionCm.research.cross_source_effect.run --run-id <unique_run_id>
```

先添加 `--smoke` 运行双源小样本 wiring 验证；正式运行不加该参数。默认 batch=16、workers=2。
checkpoint、split 和物理分箱固定在脚本中，输出目录拒绝覆盖。无需新增依赖，不修改正式 cache。

## 指标与限制

- 保持原 val seed=42、epoch=0、stride=2 与 Dataset fixed-stride 行边界（每序列 T-2 行）。额外加载空交互帧只用于
  覆盖率审计；主误差使用 full-pool active 且 sampled-valid 的行，逐样本保留无效标记。
- EPE 是逐物体点三维误差范数的均值；RMSE 是三维误差平方范数均值的平方根，不是单坐标 RMSE。
  hand RMS 只对有效边去重的 hand 点计算。单位均为 mm。
- frame micro、sequence macro 分开；95% CI 以序列为 cluster、2000 次 bootstrap。
  匹配 CI 条件于已选择的子集，不计分箱/子集选择的不确定性。
- 匹配只控制 object name、hand RMS、GT object RMS 与接触比例的粗分箱，不控制所有交互因素。
  GT 只在离线匹配/指标中使用，从未传入模型；两源不是同交互的 paired 数据。
- hand-flow 对照是样本内部有效 hand 点的空间置换，保留向量集合和几何/KNN。
  对几乎一致的刚体平移可能没有破坏力，不能据此判断模型是否完全忽略运动。
- zero-token 保留原动态 anchors，因此 anchors 仍可带运动信息；这也是 OOD 推理消融，
  不是经过训练的 geometry-only baseline。
- val 曾用于 checkpoint 选择。不能将本实验当成独立测试集泛化、潜变量不变性或等价性证明。

输出包含 `run_manifest.json`、`config.json`、`metadata.json`、`metrics.jsonl`、`run.log`、
`effects_*.npz`、`matched_selection.json`、`effect_summary.json`、`effect_comparison.png`。
NPZ 的 `sample_index` 与 metrics 行对应，保存 object points/normals、GT、各预测、tokens、anchors
和 validity；metadata 锁定序列、数据文件 stat、manifest/index/scale/checkpoint/代码 SHA256。
运行终态以 [src/task/ObjectInteractionCm/docs/logs/activity_log.md](../../docs/logs/activity_log.md) 为准。

可用 `python -m src.task.ObjectInteractionCm.research.cross_source_effect.verify <output_run_dir>`
独立重算全部 NPZ 的 EPE/MSE、对照误差、掩码、匹配平衡与输入/代码摘要。
`sequence_macro.rmse_mm` 是先对各序列 MSE 等权平均再开方，而非各序列 RMSE 的算术均值。
快照使用 `evaluation_partition: val`，避免通用 manifest 将字符串 `split: val` 当成文件路径。

## 无学习机制诊断

`diagnose.py` 对相同归档有效样本比较接触 unique-hand 平均位移、手点刚体拟合和预测的刚体投影，
同时检查 GT 刚性、误差份额、slots 真实扩散和训练日志。它不改 Cm、不训练网络；
手刚体拟合假设接触手点与物体近似无滑移，不代表所有物理交互都满足该条件。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.ObjectInteractionCm.research.cross_source_effect.diagnose --run-id <unique_run_id>
```

`--smoke` 只处理两源各三个样本。结果 `diagnosis.json` 的 `full_minus_*` 为正表示 Cm 更差，
`projection_gain_mm` 为正表示刚体投影改善；置信区间按序列重采样，控制比较保持样本配对。
`gt_rigid_residual_mm` 是使用 GT 的数据一致性 oracle，不能作为可部署基线。
全部输出落到新的 `output/<run_id>/`，不改变旧运行的代码或产物摘要。
