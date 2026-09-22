# ACT-20260922-CMV2-V114D-MANO-TRAIN

- timestamp：2026-09-22T15:07:24+08:00
- activity_id：`ACT-20260922-CMV2-V114D-MANO-TRAIN`
- work_version：`V1.14`
- git_commit：`22d6c42`
- base_commit：`d61eebc`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope：本地 GRAB/MANO lean cache pose 纠偏、单组 split、optimizer smoke 与 1-epoch 正式训练启动
- approval：用户明确要求先用 MANO 手开始训练，并批准 V1.14d.1 pose 纠偏
- rollback：停止 `cmv2-v114d-mano-train-r2.service`；各 run/output 独立保留，不覆盖 cache、split 或旧 checkpoint

## 输入与门禁

- MANO cache：1335 条、400616 帧、split `1068/134/133`、生成 failure 0。
- pose repair run：`cmv2_v114d1_mano_pose_repair_20260922T1255`，1335/1335、最大点 replay
  `2.7930434138429005e-07 m`、failure 0，门槛 `0.2 mm`。
- MANO-only active split：train/val/test `1068/134/133`；真实 batch16 optimizer smoke loss finite，最大 replay
  `2.604924418392329e-07 m`。
- 定向 pytest 12 项、语法检查、`python3 tools/verify.py --changed` 和 diff check 通过。

## 运行事件

- `cmv2_v114d_grab_mano_lean_e1_20260922T1500`：`FAILED`，step 0；systemd soft nofile 为 1024，构造全量
  memmap dataset 时触发 `EMFILE`，没有 optimizer step/checkpoint。
- `cmv2_v114d_grab_mano_lean_e1_20260922T1501`：`STOPPED`，step 240；用于确认 nofile=262144 后训练接线，
  随后发现旧 runner 未兑现配置的 periodic checkpoint，未作为正式证据继续。
- 正式 run：`cmv2_v114d_grab_mano_lean_e1_20260922T1504`，`RUNNING`；GPU6、random init、batch16、
  20012 planned steps、1 epoch、LR 0.001、seed42。step 200 `latest.pt` 已核对 schema/work_version/run_id/
  commit 均正确；记录时已超过 step 300，loss finite。
- service：`cmv2-v114d-mano-train-r2.service`；manifest、`metrics.jsonl`、`train.log`、`latest.pt` 位于
  `outputs/objectinteractioncmv2/cmv2_v114d_grab_mano_lean_e1_20260922T1504/`。

## 当前结论

run_status 与工程结论为 `RUNNING`；周期恢复点和真实训练接线通过。尚未完成 epoch/validation，科研结论保持
`INCONCLUSIVE`，不得用早期 training loss 判断效果。
