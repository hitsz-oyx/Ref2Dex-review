# 冻结跨手 Cm 交换

按 [最终计划](../../docs/plan/v1.1.md) V1.1.9 执行。固定 V1.1.8 的 OICM、Temporal-D2 和 Inspire 当前状态，重建同 parent/raw timestamp 的原始 MANO 2048 点源流，仅替换完整 Cm（tokens、anchors、normals）。

五个条件：correct、identity、mano_sync、mano_shift、mano_transported。最后一项将 MANO 物体相对几何搬到目标实际物体轨迹，含目标运动信息，只是 oracle 对照。严格筛选门槛见 experiment.yaml 和 diagnostics.py；未来目标 q/wrist 仅计算误差，不参与筛选或模型输入。

```bash
CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_cm_swap.run --run-id <unique_run_id> --smoke
# smoke 通过后去掉 --smoke，运行完整 30 条 val。
```

单步 reference compatibility 不等于物理成功；val 曾用于 checkpoint 选择，不能称为独立 test。严格运动子集少于 100 窗口或 10 parent 不作总体迁移结论。保持原始 MANO 物体轨迹，不能将 oracle 的改善解释成独立跨手成功。

新产物位于忽略的 output/run_id，包含配置、来源/stat/hash、完整源 Cm、配对描述、预测、逐行指标、summary、图与工程核验。冻结参数与历史输入前后核验；正式数据和旧输出不改写。
