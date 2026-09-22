# V1.18c OICmv2 V1.14a grasp integration

- timestamp: 2026-09-22T17:31:00+08:00
- activity_id: ACT-20260922-173100-CMRESIDUAL-V118C-V114A-INTEGRATION
- work_version: V1.18c
- git_commit: `592262f0b3111a49f36defe3ae59a690879803a4`（formal paired runs；后续 runner smoke 修复见分支 HEAD）
- base_commit: `a4cd6f9`
- branch: `ai/cmresidual/v118c-v114a-integration`
- scope: V1.14a frozen adapter、单手 2048 点、object-local input、shared K=8 candidate axis、1-env rollout
- approval: 用户确认训练结束后直接进入抓取任务并执行最小接线
- verification: 14 tests、py_compile、Hydra preflight、`python3 tools/verify.py --changed`、`git diff --check`
- rollback: revert V1.18c implementation commits；不删除 run/data/checkpoint

## Result

V1.14a `best.pt`（SHA `9d2940f9…9197297`）严格加载并完成 64-step/1-env/GPU6 PPO smoke；checkpoint、
buffer 64 samples 与 TensorBoard 均 finite。相同代码、seed、state/action/PPO 配置下，旧 V1.3 control 也完成。

| variant | step FPS | total FPS | play time |
| --- | ---: | ---: | ---: |
| V1.3 | 5.383 | 5.038 | 11.889 s |
| V1.14a | 7.243 | 6.640 | 8.836 s |

因此该最小 rollout 上 step FPS 为 `1.345x`、total FPS 为 `1.318x`。补充的 B=1/K=8 CUDA 单步诊断中，
V1.14a 使用 `object_chunk=1024, hand_chunk=2048` 的 median 为 `15.535 ms`，旧 V1.3 为 `57.321 ms`；
沿用旧 `object_chunk=32` 会使 V1.14a 回退到约 `194 ms`，故新配置单独固定大 chunk 与 microbatch 1。

前两个运行分别因 task registry 缺项、1-env minibatch 仍为 8192 而失败，第三次因 Hydra 继承的两个 legacy
model-config key 导致 strict identity mismatch；三项均修复并保留失败 manifest。MANO-only 1-epoch checkpoint
未覆盖 Inspire 域，因此这里只支持工程接线与单环境吞吐，不支持抓取效果、模型 utility 或多环境扩展结论。

正式 runs：

- `cmresidual_v118c_v114a_1env_smoke_gpu6_20260922_172814`
- `cmresidual_v118c_v13_1env_control_gpu6_20260922_172907`
