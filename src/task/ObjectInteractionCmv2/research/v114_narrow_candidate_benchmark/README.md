# V1.14 窄交互 Candidate Benchmark

本实验用同一份随机 synthetic endpoint 输入比较 `interaction_dim={128,64,32}`，只检验工程延迟与显存，
不检验预测质量。主合同固定为 `B=1,K=8,N=1024,E=32`、FP32、TF32 关闭，并分别记录 static object
encode、local interaction、共享后的完整 candidate forward 和 encode+forward。

endpoint 构建以相同对象和 `H=4096` 手点单独计时；`endpoint + encode + forward` 是顺序执行时间之和，
不是隐藏 KNN 成本的“纯模型”数字。输出写入 `output/<run_id>/`，拒绝覆盖旧运行。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 \
  -m src.task.ObjectInteractionCmv2.research.v114_narrow_candidate_benchmark.run \
  --run-id <unique_run_id>
```

运行不加载 checkpoint、不训练模型、不读取或修改 cache。随机初始化仅用于结构等价的性能比较。
