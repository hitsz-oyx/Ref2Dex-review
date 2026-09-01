# Cm t-SNE Slots

这是 Cm 的 task-local t-SNE 诊断实验包。

- 入口：`run.py`；
- 兼容入口：`python -m src.task.Cm.research.tsne_slots` 和上级目录的 `tsne_slots.py`；
- 默认产物：`output/<run_id>/`；
- 结果说明和科学结论：`../../docs/logs/experiment_log.md`。

显式传入 `--output-root` 时也必须指向本实验的输出目录，不要覆盖其他实验或训练 run。
