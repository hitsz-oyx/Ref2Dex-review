# ACT-20260921-CMV2-V114A-SHARED-START

- timestamp：2026-09-21T23:02:07+08:00
- activity_id：`ACT-20260921-CMV2-V114A-SHARED-START`
- work_version：`V1.14`
- base_commit：`aad77c1`
- implementation_commit：`74fcec4`
- benchmark_commit：`225c143`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope：V1.14a 每手 2048 点合同、candidate 共享 start KNN、compact-v2 pilot 身份与随机初始化接口验证
- approval：用户批准 V1.14a，并明确 2048 是每只手的点数；双手合计 4096

## 完成内容

- MANO 保持每侧 2048 点；Inspire 每侧按 variant、side 和 seed 42 固定无放回选择 2048 个稳定点 ID；
- current/future、点、法线和 flow 复用同一索引，V1.14a loader 输出固定 `[4096,3]`；
- `forward_candidates()` 的起点、法线和 valid mask 改为共享 `[B,H,...]`，仅 flow 保留 `[B,K,H,3]`；
- start top-32 每个 state 只搜索一次，candidate end top-32 与 union-rerank 语义不变；
- 新 checkpoint/config identity 拒绝旧 V1.14；compact-v2 pilot 拒绝 V1.13 record；
- benchmark harness 同时记录 duplicated-start 与 shared-start endpoint 延迟。

## 验证

- `python3 -m pytest -q ...test_v1_14_narrow_candidates.py ...test_v1_12_part_se3.py ...test_v1_4_three_domain.py`：26 passed；
- shared-start 与 duplicated-start oracle：edge ID 完全一致，距离 `atol=1e-6, rtol=1e-6`；
- V1.14a compact-v2 内存 round-trip 与 reference 输出一致；旧 V1.13 record 拒绝通过；
- `test_v1_4_5_highres_producers.py` 未收集：当前环境缺少 `dex_retargeting`；
- 五组真实 transition 审计未运行：配置中的 `/mnt/ugreen_nas/...` 在当前环境未挂载。

## Benchmark 运行终态

- run_id：`cmv2_v114a_shared_start_gpu3_20260921T2306`
- run_status：`COMPLETED`
- command：`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 -m src.task.ObjectInteractionCmv2.research.v114_narrow_candidate_benchmark.run --run-id cmv2_v114a_shared_start_gpu3_20260921T2306 --device cuda:0 --object-chunk 1024 --hand-chunk 1024`
- output：`research/v114_narrow_candidate_benchmark/output/cmv2_v114a_shared_start_gpu3_20260921T2306/`
- manifest：`run_manifest.json`；config：`config.json`；metrics：`metrics.jsonl`；log：`run.log`；summary：`summary.json`
- last step/epoch、best metric、checkpoint：不适用（随机权重 latency benchmark）
- duplicated-start endpoint median：`33.316 ms`；shared-start：`19.958 ms`；speedup：`1.669x`
- 相对历史 tuned endpoint `32.926 ms`：`1.650x`
- shared endpoint + 32D encode/candidate median：`25.851 ms`
- conclusion：`SUPPORTED`（全部四个自动工程门槛通过）

## 保护与结论

- 未覆盖或迁移任何原始 cache、V1.13 compact cache、checkpoint 或运行输出；
- 未启动 full cache build、短程训练或正式训练；
- 已证明接口/数值 parity 和固定 synthetic GPU 工程加速；尚无真实输入覆盖率或预测质量结论。

## 回滚

回滚入口为基线提交 `aad77c1`；数据无需回滚。
