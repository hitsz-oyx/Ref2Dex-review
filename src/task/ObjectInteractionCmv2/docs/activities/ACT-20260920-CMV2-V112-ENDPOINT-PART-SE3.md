# V1.12 端点 KNN 与逐部件直接 SE(3) 实现

- timestamp：`2026-09-20T16:28:39+00:00`
- activity_id：`ACT-20260920-CMV2-V112-ENDPOINT-PART-SE3`
- Task / work_version：`ObjectInteractionCmv2` / `V1.12`
- base_commit：`3d5d73c53ed93029d917c66c09b0eb5c5110716e`；git_commit：`961e6d61db44c234618fb9fe02ef48183c64a291`；branch：`ai/ObjectInteractionCmv2/v1.12-endpoint-knn`。
- scope / impact：Task-local model、dataset、构造/checkpoint 合同、非运行配置、测试与状态文档；L2。
- approval：用户确认逐部件直接 SE(3)、固定起始物体 query、起止 top-32 并集后按最小端点距离重排且最终仍为 32、整体 1024 点按部件分层采样且每部件至少 64 点、固定 `log(rho+eps)` prior、无直接 interaction 的 part surface 输出严格为零、part ID 不 embedding、pose sample 内 part 等权、SO(3) geodesic、flow 普通逐点平均及随机初始化，并授权修改。
- run_status：单 GPU 与 GPU0+2 B64 双卡 smoke 均 `COMPLETED`；正式训练已获用户批准、尚未启动；未构建 cache。
- scientific conclusion：`INCONCLUSIVE`；当前证据只证明实现合同、数据 replay 与接线，不证明预测效果。

## 已完成

- `part_se3.py`：以起始物体点为固定 query，分别计算起始/末端手点 top-32，稳定并集后按 `min(d_start,d_end)` 与手点 ID 重排为最终 32；2 cm mask 在最终集合上生效。
- `part_se3_data.py`：GRAB、ARCTIC 与 OakInk2 均读取同一 4096 点身份池，按部件 quota 无放回抽取总计 1024 点；输出逐部件直接 SE(3) GT，并以 0.2 mm 上限 fail-fast replay 对应未来点。
- `part_se3.py` 模型：V1.3 风格 geometry/contact/surface tokens，显式 token assignment；以 `rho=part_mass/token_mass` 的 `log(rho+eps)` 作为固定 cross-attention prior，无直接 interaction 的 part surface 输出乘为严格零，再进行 part self-attention；共享线性 head 输出每部件 6D 运动，part ID 只用于 segment/routing。
- pose loss 为 sample 内 part 等权再 batch 等权：translation SmoothL1/2 cm、rotation SO(3) geodesic；flow 保持 SmoothL1/2 cm 的普通逐点平均。
- 独立 schema `object_interaction_cmv2_part_se3_v1_12` 与 checkpoint schema `object_interaction_cmv2_part_se3_checkpoint_v1`；配置显式 `initialization: random`、`run_authorization: not_approved`，旧 V1.11 checkpoint 会被拒绝。

## 验证

- Task-local pytest：`60 passed`；其中 V1.12 定向测试 `12 passed`，覆盖 endpoint union/rerank/tie、少于 32 有效手点 fail-fast、分层 quota、GRAB/ARCTIC/OakInk2 replay、strict-zero routing、rho 归一、point/part permutation、近零/近 π geodesic backward、随机初始化、checkpoint 隔离与正式配置资源合同。
- 五组各取一个真实 train transition 做只读检查：GRAB MANO/Inspire 为单部件 1024 点；ARCTIC MANO quota `[422, 602]`；OakInk2 MANO/Inspire quota 均为 `[517, 507]`。逐部件 replay 最大残差分别约 `0.000117 / 0.000240 / 0.000069 / 0.000123 / 0.000069 mm`，均低于 0.2 mm。
- 五组真实样本的 1024×全手点 endpoint KNN 均与独立全量 PyTorch 参考逐索引一致；输出形状均为 `[1,1024,32]`。
- V1.12 CPU 合成 forward/backward finite；单 GPU 真实 batch smoke 见下节。

## 单 GPU 真实 batch smoke

- timestamp：`2026-09-21T02:42:20+00:00` 至 `2026-09-21T02:42:22+00:00`
- run_id：`cmv2_v112_part_se3_single_gpu_smoke_20260921T000000Z`
- git_commit：`e47767bcf217b7dff1f159a27000f5f0f6a7014d`
- run_status：`COMPLETED`；last_step：`1`；best metric：`N/A`；scientific conclusion：`N/A`
- protocol：物理 GPU2，随机初始化；真实 `oakink2/inspire_f1` train、stride 1、batch 1；1024 个物体点、20270 个手点、2 个部件；执行 1 次 forward 与 backward，optimizer step 为 0，随后做 V1.12 checkpoint strict round-trip。
- finite loss：total `40.279236`、translation `19.780777`、rotation `0.931926`、flow `19.566534`。这些随机初始化单 batch 数值只作接线证据，不用于模型选择或科学结论。
- 工程证据：forward/backward 用时约 `0.322 s`；峰值 allocated/reserved 约 `0.109/0.133 GiB`；checkpoint round-trip 输出最大绝对差 `0.0`；逐部件 GT replay 最大残差约 `0.000069 mm`。
- 输出目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v112_part_se3_single_gpu_smoke_20260921T000000Z`；入口为同目录 `run_manifest.json`、`config.json`、`metrics.jsonl`、`train.log` 与 `smoke.pt`。
- 资源保护：启动前四卡均有既有进程；GPU2 有约 39 GiB 空闲且只承载 V1.11.1 本 Task 进程。本 smoke 约 2 秒完成，未停止、重启或修改 V1.11.1 服务、进程、输出与 checkpoint。

## V1.11.1 停止与 V1.12 双卡门禁

- 用户于 `2026-09-21` 明确要求停止 V1.11.1，并让 V1.12 沿用此前三域五组训练的 GPU0+2、B64×2、16 epochs、LR 0.001、seed 42、stride 1..3、组间比例、validation 与 checkpoint 节奏；用户明确要求本次启用不另建 plan。该批准记录于 Activity、正式配置与 run manifest。
- V1.11.1 run `cmv2_v111i_mixed_ddp_formal_20260920T114242Z` 于 `2026-09-21T02:53:58+00:00` 受控停止；run_status `STOPPED`，最后 step `18040`、epoch `2`、best metric `0.0552500159`。`latest.pt`、`best.pt`、metrics、日志和 source snapshot 均保留；未删除或覆盖旧产物。
- V1.12 DDP 实现 commit：`e60e4caf7f1d25c9e94d24753d6d92e4697cc994`；正式配置为 `configs/active/mixed_part_se3_v1_12_ddp.yaml`，独立 runner 为 `train_part_se3_ddp.py`。随机初始化且拒绝 V1.11 checkpoint。
- 双卡 smoke run_id：`cmv2_v112_part_se3_ddp_smoke_20260921T025835Z`；物理 GPU0+2、每卡64、global batch128、2 steps、五组采样和15组 validation；run_status `COMPLETED`，checkpoint round-trip 与两 rank 观察到的 checkpoint SHA256 一致。
- smoke 训练 step1/2 total loss 为 `40.697411 / 72.862259`；峰值 allocated/reserved 约 `5.735/6.145 GiB`。随机初始化的两步 loss 与 validation 数值只作 finite/wiring 证据，不作效果结论。
- smoke 输出目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v112_part_se3_ddp_smoke_20260921T025835Z`；入口为同目录 `run_manifest.json`、`config.json`、`metrics.jsonl`、`train.log`、`service.log`、`latest.pt` 和 `best.pt`。
- GPU0 上 lxp 用户的 GR00T server（PID `1759014`，约 8.7 GiB）全程保留；smoke 结束后 GPU2 显存归零，GPU0 仅保留该外部进程。

## 保护与回滚

未修改 `src/base`、既有 V1.11 模型/runner/config、原始数据、cache、adapter、checkpoint 或历史 outputs。V1.11.1 仅按用户明确授权受控停止，产物完整保留；GPU0 外部 GR00T 未触碰。用户维护的 `指导/V1.12.md` 内容保留。

回滚可删除本 Activity 所列 V1.12 独立源码、配置、测试与状态增量，并把 Task work_version 指针恢复为 `V1.11.1`；无需迁移或删除任何数据和运行产物。
