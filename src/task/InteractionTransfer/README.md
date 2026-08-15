# InteractionTransfer V0

该 Task 独立实现 `Current Relation + Hand Action -> Transferred Interaction -> Object Effect`。运行最小诊断：

```bash
python3 -m src.task.InteractionTransfer.research.forward_v0.diag_forward
python3 -m src.task.InteractionTransfer.research.intervention_v01
```

真实训练使用 `train.py --root <cache> --sequence <sequence>`，输入为 one-step GRAB cache。V0 仅使用 flow loss；与 direct forward baseline 的 EPE 比较和 V0.1 intervention 在最小 gate 通过后进行。
