# V1.12 端点 KNN 与逐部件直接 SE(3) 实现

- timestamp：`2026-09-20T16:28:39+00:00`
- activity_id：`ACT-20260920-CMV2-V112-ENDPOINT-PART-SE3`
- Task / work_version：`ObjectInteractionCmv2` / `V1.12`
- base_commit：`3d5d73c53ed93029d917c66c09b0eb5c5110716e`；branch：`ai/ObjectInteractionCmv2/v1.12-endpoint-knn`；实现尚未提交，`git_commit` 待交接提交后补记。
- scope / impact：Task-local model、dataset、构造/checkpoint 合同、非运行配置、测试与状态文档；L2。
- approval：用户确认逐部件直接 SE(3)、固定起始物体 query、起止 top-32 并集后按最小端点距离重排且最终仍为 32、整体 1024 点按部件分层采样且每部件至少 64 点、固定 `log(rho+eps)` prior、无直接 interaction 的 part surface 输出严格为零、part ID 不 embedding、pose sample 内 part 等权、SO(3) geodesic、flow 普通逐点平均及随机初始化，并授权修改。
- run_status：`NOT_STARTED`；本 Activity 未启动训练、GPU smoke、cache 构建或数据写入。
- scientific conclusion：`INCONCLUSIVE`；当前证据只证明实现合同、数据 replay 与接线，不证明预测效果。

## 已完成

- `part_se3.py`：以起始物体点为固定 query，分别计算起始/末端手点 top-32，稳定并集后按 `min(d_start,d_end)` 与手点 ID 重排为最终 32；2 cm mask 在最终集合上生效。
- `part_se3_data.py`：GRAB、ARCTIC 与 OakInk2 均读取同一 4096 点身份池，按部件 quota 无放回抽取总计 1024 点；输出逐部件直接 SE(3) GT，并以 0.2 mm 上限 fail-fast replay 对应未来点。
- `part_se3.py` 模型：V1.3 风格 geometry/contact/surface tokens，显式 token assignment；以 `rho=part_mass/token_mass` 的 `log(rho+eps)` 作为固定 cross-attention prior，无直接 interaction 的 part surface 输出乘为严格零，再进行 part self-attention；共享线性 head 输出每部件 6D 运动，part ID 只用于 segment/routing。
- pose loss 为 sample 内 part 等权再 batch 等权：translation SmoothL1/2 cm、rotation SO(3) geodesic；flow 保持 SmoothL1/2 cm 的普通逐点平均。
- 独立 schema `object_interaction_cmv2_part_se3_v1_12` 与 checkpoint schema `object_interaction_cmv2_part_se3_checkpoint_v1`；配置显式 `initialization: random`、`run_authorization: not_approved`，旧 V1.11 checkpoint 会被拒绝。

## 验证

- Task-local pytest：`59 passed`；其中 V1.12 定向测试 `11 passed`，覆盖 endpoint union/rerank/tie、少于 32 有效手点 fail-fast、分层 quota、GRAB/ARCTIC/OakInk2 replay、strict-zero routing、rho 归一、point/part permutation、近零/近 π geodesic backward、随机初始化与 checkpoint 隔离。
- 五组各取一个真实 train transition 做只读检查：GRAB MANO/Inspire 为单部件 1024 点；ARCTIC MANO quota `[422, 602]`；OakInk2 MANO/Inspire quota 均为 `[517, 507]`。逐部件 replay 最大残差分别约 `0.000117 / 0.000240 / 0.000069 / 0.000123 / 0.000069 mm`，均低于 0.2 mm。
- 五组真实样本的 1024×全手点 endpoint KNN 均与独立全量 PyTorch 参考逐索引一致；输出形状均为 `[1,1024,32]`。
- V1.12 CPU 合成 forward/backward finite。没有运行单 GPU 真实 batch forward/backward 或 checkpoint 落盘 smoke，因为计划要求实现证据汇报后另行协商。

## 保护与回滚

未修改 `src/base`、既有 V1.11 模型/runner/config、原始数据、cache、adapter、checkpoint 或 outputs；未停止或改变正在运行的 V1.11.1 source snapshot。用户维护的 `指导/V1.12.md` 内容保留。

回滚可删除本 Activity 所列 V1.12 独立源码、配置、测试与状态增量，并把 Task work_version 指针恢复为 `V1.11.1`；无需迁移或删除任何数据和运行产物。
