# Cm 研究实验

这里存放 Cm 的 task-local 研究实验定义。每个独立问题必须使用一个目录，不能把脚本直接放在
`research/` 根目录：

```text
research/<experiment_id>/
├── run.py 或其他实验脚本
├── README.md
├── experiment.yaml
└── output/<run_id>/       # 生成产物，已加入忽略规则
```

实验输出只允许写入对应目录的 `output/<run_id>/`。训练或带 checkpoint 的正式运行仍使用根级
`outputs/<task>/<run_id>/`；不要再新建根级 `output/research/` 或 `results/`。
实验入口在未显式指定 `--output` 时自动创建带时间戳的 run 目录，并写入
`config.json`、`metadata.json`、结果文件和 `run_manifest.json`；显式输出路径只用于兼容旧调用。

当前实验：

- `tsne_slots/`：Cm 表征的 t-SNE 诊断；旧 `tsne_slots.py` 保留为兼容软链接；
- `dense_cache_v1_1_1/`：DenseToken cache parity benchmark；
- `hand_flow_decoder/`：手流尺度校准。

`log.md` 和 `results/` 是历史记录入口。新的活动摘要写入 `../docs/logs/activity_log.md`，新的实验结论写入 `../docs/logs/experiment_log.md`，
不要继续在这里增加结果汇总文件。
