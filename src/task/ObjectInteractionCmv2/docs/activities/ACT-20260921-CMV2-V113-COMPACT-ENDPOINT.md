# V1.13 紧凑 Endpoint Cache 实现

- timestamp：`2026-09-21T03:49:43+00:00`
- activity_id：`ACT-20260921-CMV2-V113-COMPACT-ENDPOINT`
- Task / work_version：`ObjectInteractionCmv2` / `V1.13`
- base_commit：`303bf296140a4774a108f0d111501fc6d8995e5c`；核心实现 commit：`a3ffe8602ad2ad8390fde08e93deba5f284ac769`；当前 pilot source commit：`c176d5eeb358d522acbb4129c05c01578b0e9667`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope / impact：Task-local compact cache schema、producer、reader、预选 edge 模型入口、DDP backend 接线、测试与版本指针；`L2`。
- approval：用户确认 V1.13 FINAL 计划后明确要求直接实现。
- run_status：两次 pilot 均已 `STOPPED`；未构建 full cache，未启动 benchmark、smoke 或正式训练。
- scientific conclusion：`N/A`；工程 parity 不证明预测效果。

## 完成内容

- 新增 `object_interaction_cmv2_compact_endpoint_v1`：每条 transition 保存确定性 1024 点、GT、最终 endpoint-32 边，以及按 transition 去重的 float32 手点/法线/flow；使用 `[1024,32]` 局部索引恢复边。
- cache 使用约 1 GiB shard、固定宽度 `index.bin`、view ranges 和带 source/shard hash 的 manifest；禁止每样本小文件。reader 只打开所需 shard，并以有界 LRU 管理文件句柄。
- 模型新增 compact edge 输入分支，跳过完整手流 endpoint KNN，但复用原 edge encoder 及全部模型参数；reference 与 compact backend 不能混用。
- producer 支持 `pilot` / `full`；full 必须引用 source identity 一致、`bad_count=0` 且序列化/源 working-set 比例不超过 `1.25` 的 pilot manifest。失败保留 `.partial` 与 `FAILED` run manifest，不自动删除。
- DDP runner 仅在已批准 V1.13 配置同时声明 `data_backend: compact` 与明确 cache root 时启用新路径；V1.12 reference 配置行为保持不变。仓库中未创建或伪造 V1.13 正式训练批准配置。

## 验证

- Python 语法检查通过；producer CLI `--help` 可用。
- ObjectInteractionCmv2 Task 全量测试：`65 passed`。覆盖 reference/compact edge ID、mask、distance、forward 与 loss 的 `1e-6` parity，shard/index 过滤、未验证 manifest 拒绝、backend 选择和正式配置审批门禁。
- 五组各一条真实 transition 只读核对通过；compact 展开后的 hand point/normal/flow 与 reference 按原 hand ID gather 完全一致。
- 五组单样本 `unique/source hand points`：GRAB/MANO `59/4096`、ARCTIC/MANO `173/4096`、OakInk2/MANO `688/4096`、GRAB/Inspire `50/20270`、OakInk2/Inspire `2098/20270`。这些仅是可行性样本，不替代计划要求的完整 pilot 容量统计。

## Pilot 生命周期

- 早期 run `cmv2_v113_compact_pilot_20260921T035137Z` 在最后已记录 100 条时受控停止；发现两个 edge index 可由 int32 收窄为 int16。约 69 MiB `.partial` 与 `STOPPED` manifest 保留，未删除或覆盖。
- int16 仍覆盖最大 20,269 的 hand-point ID；Task 测试重新通过。首条真实 record 从约 `355756` bytes 降为 `224684` bytes，减少约 36.8%。
- run_id：`cmv2_v113_compact_pilot_20260921T035408Z`；service：`ref2dex-cmv2-v113-compact-pilot-20260921T035408Z.service`；source commit：`c176d5eeb358d522acbb4129c05c01578b0e9667`。用户要求先使用已导出数据检验，该运行于 `2026-09-21T04:12:19+00:00` 收到 SIGTERM 后受控停止，终态 manifest 为 `STOPPED`，共保留 2510 records / 623726760 serialized bytes。
- partial 输出：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicmv2_endpoint_knn/object_interaction_cmv2_compact_endpoint_v1/cmv2_v113_compact_pilot_20260921T035408Z.partial`。逐记录反序列化、schema、必需 tensor、`[1024,32]` edge shape、lookup range、index offset 与 shard size 检查的 `bad_count=0`。
- 阶段统计为 serialized/source-sample-tensor ratio `0.929356`，unique/source hand-point ratio `0.124524`。覆盖仅包含 `grab/mano`、`arctic/mano`、`oakink2/mano`和 `grab/inspire_f1` 的 train stride 1/2/3；尚未覆盖 `oakink2/inspire_f1` 及任何 validation view。因此该结果只证明已覆盖记录的工程结构可用且阶段容量比低于 1.25，不构成完整 pilot 或 full-build gate 通过。

## 保护与回滚

未改变模型参数、1024 点采样、endpoint union-rerank、坐标/单位、GT、loss、split、训练预算或 checkpoint 参数解释；未修改 `src/base`，未写入、覆盖或删除原始数据、旧 cache、checkpoint 和 outputs。

回滚可把配置切回默认 `reference` backend；模型和 checkpoint 无需转换。代码级回滚入口为实现提交的父提交 `61a1c1b43eb9b1b2e8654956f016239103c544cf`。任何 `.partial`、pilot/full cache 或本地镜像的删除仍需单独授权。
