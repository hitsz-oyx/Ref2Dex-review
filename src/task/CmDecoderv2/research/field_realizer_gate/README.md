# V1.1.16 FieldRealizer gate

本实验按 [src/task/CmDecoderv2/docs/plan/v1.1.md](../../docs/plan/v1.1.md) 执行 F7 合同审计、
Realizer 训练和对照。`audit` 仅验证静态声明；函数存在不等于 GPU pipeline 下可以提取 pairwise contact。
2026-09-13 真实仿真 probe 已确认当前 `get_env_rigid_contacts` 在 GPU simulation 启动后不可用，
正式物理 Gate 暂不能用该接口记录接触；不能把返回的空数组解释为零接触。

```bash
python -m src.task.CmDecoderv2.research.field_realizer_gate.audit \
  --run-id field_gate_audit_<timestamp>
```

可在 B1 训练之外独立运行的入口：

- `evaluate_conditions --checkpoint <explicit_epoch.pt> --output <new_run_dir>`：固定权重的 B1/C0/Cs
  离线敏感性，使用全部 val windows；不是 contact50 rollout，也不用于重新挑选超参。
- `probe_gpu_contacts --source-manifest <contact50/manifest.json> --output <new_run_dir>`：64 env、
  相同 GPU physics 的四步仪器能力检查。需在干净 Python 中先 `import isaacgym`，然后用
  `runpy.run_module(..., run_name="__main__")` 启动，避免父包提前导入 torch。
- `audit_mano_h_source --output <new_run_dir>`：284 条 parent MANO-H 来源、模板和 raw-frame 对齐
  检查。可用性不等于 MANO mesh parity；D 尚需 joint0 wrist 和 subject-template 语义验证。

实验结果进入 [src/task/CmDecoderv2/docs/logs/experiment_log.md](../../docs/logs/experiment_log.md)，
运行状态进入 [src/task/CmDecoderv2/docs/logs/activity_log.md](../../docs/logs/activity_log.md)。
