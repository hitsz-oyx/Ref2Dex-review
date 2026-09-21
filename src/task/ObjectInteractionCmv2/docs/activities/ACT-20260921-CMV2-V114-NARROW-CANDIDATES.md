# V1.14 窄交互与 Candidate 共享实现

- timestamp：`2026-09-21T21:50:15+08:00`
- activity_id：`ACT-20260921-CMV2-V114-NARROW-CANDIDATES`
- Task / work_version：`ObjectInteractionCmv2` / `V1.14`
- base_commit：`5265797`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope / impact：Task-local 模型、随机初始化配置、checkpoint 合同、测试、文档与版本指针；`L2`。
- approval：用户确认 V1.14 初稿后于 2026-09-21 明确许可继续实施。
- run_status：`N/A`；未启动 cache build、正式 benchmark、短程训练或正式训练。
- scientific conclusion：`N/A`；随机权重 smoke 只证明接口和数值接线，不证明预测效果或正式性能收益。

## 完成内容

- 新增独立架构 `v1_14_narrow_interaction_candidate_shared`；旧 V1.12 模型保持不变。
- endpoint edge MLP 与 Q/K/V 使用可配置 `interaction_dim`，主配置为 32D；point contact、surface-token
  interaction 和 part interaction 保持低维，仅在 16 个 token 与 part 层升回 128D。
- 新增 `encode_object()` 和 `forward_candidates()`；geometry point/part feature 每个 state 只编码一次，
  contact、token assignment、routing 与逐部件输出保留 candidate 轴。
- `forward(batch)` 是 `K=1` 包装；reference hand stream 与 V1.13 compact edge 均可进入同一新模型。
- 新 checkpoint schema 只接受 V1.14 随机初始化模型，显式拒绝 V1.12/V1.13 checkpoint。
- V1.14 配置保持 `run_authorization: not_approved`，默认使用当前可用的 reference backend；构造合同也接受
  显式提供真实 cache root 的 compact backend，但不构成训练授权。

## 验证

- V1.12、V1.13、V1.14 定向回归：`22 passed`；已覆盖 K=1 包装等价、批量 candidate 与逐候选等价、
  candidate 重排、双样本不同 part padding、compact/reference parity、finite backward 和 checkpoint 拒绝。
- 排除依赖本机未安装可选包 `dex_retargeting` 的旧 V1.4 producer 测试后，其余 Task 测试为 `68 passed`；
  未排除的全量收集在该旧测试 import 阶段失败，因此不宣称 Task 全量通过。
- GPU0（RTX 3090）随机初始化工程 smoke：`B=1,K=8,N=1024,E=32` 的 forward/backward finite；输出
  `delta_xi_part [1,8,3,6]`、`obj_flow_pred [1,8,1024,3]`、`contact [1,8,1024,32]`；峰值 allocated
  `242.38 MiB`。该 smoke 未采用正式计时协议，不是性能 benchmark。
- `git diff --check` 与统一 `python3 tools/verify.py --changed` 通过；仓库默认 `python` 指向 Python 2，
  因验证脚本包含非 ASCII 字符而不可用，本次明确使用 Python 3。

## 保护与回滚

未修改 V1.13 compact cache schema、数据、GT、loss、split、坐标、单位、旧 checkpoint 或运行产物；未启动
任何长任务。代码回滚为继续使用 V1.12/V1.13 model/config，V1.13 compact cache 无需迁移或删除。
