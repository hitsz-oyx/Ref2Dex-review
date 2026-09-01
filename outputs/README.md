# Outputs 目录规范

`outputs/<task>/<run_id>/` 是正式训练、评估和带 checkpoint benchmark 的唯一运行目录。每个
运行目录至少包含：

- `run_manifest.json`：本次运行的代码、配置、数据、资产、合同和输出追踪；
- `config.json`、`metadata.json`：配置和运行快照；
- `metrics.jsonl`、日志和 `checkpoints/`：运行产物。

研究诊断脚本使用对应的
`src/task/<Task>/research/<experiment>/output/<run_id>/`，不要再新建根级 `output/research/`。

根级 `output/` 是历史兼容目录，仅供旧实验读取；新运行不得写入。数据/cache 在
`data/processed_data/`，预训练模型在 `assets/`，两者都不属于本目录。
