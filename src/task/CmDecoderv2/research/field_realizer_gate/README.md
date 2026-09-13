# V1.1.16 FieldRealizer gate

本实验先执行合同审计，不训练、不修改正式 cache，也不运行物理 rollout。审计冻结 F7、paired split、
source provenance 和 contact evaluator 能力；通过后才允许生成 B1/D 训练 view。Isaac Gym contact API
可用不等于 task 已接入 pairwise logging；接入前不得报告 hand↔object contact 指标。

```bash
python -m src.task.CmDecoderv2.research.field_realizer_gate.audit \
  --run-id field_gate_audit_<timestamp>
```
