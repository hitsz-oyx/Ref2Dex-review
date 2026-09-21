# V1.12 端点交互与逐部件直接 SE(3) 五组训练

- work_version：`V1.12`
- hypothesis：固定起始物体 query 的起止端点 KNN、整体多部件采样、grounded surface-token routing 与共享逐部件直接 SE(3) head，能够在五组混合训练中稳定优化，并在固定五组×stride 1/2/3 validation 上形成可复核的收敛证据。
- protocol：GRAB/MANO、ARCTIC/MANO、OakInk2/MANO、GRAB/Inspire、OakInk2/Inspire；沿用 V1.11.1 split、组间 batch 配比与 stride 1..3；物理 GPU0+2、每卡64、global batch128、16 epochs、Adam LR 0.001、seed 42；随机初始化，不加载 V1.11 checkpoint；以五组加权的三 stride total loss 选择 best。
- implementation：`e60e4caf7f1d25c9e94d24753d6d92e4697cc994`；`train_part_se3_ddp.py`；`configs/active/mixed_part_se3_v1_12_ddp.yaml`。
- runs：单 GPU真实 batch smoke `cmv2_v112_part_se3_single_gpu_smoke_20260921T000000Z`（COMPLETED）；GPU0+2 B64 smoke `cmv2_v112_part_se3_ddp_smoke_20260921T025835Z`（COMPLETED）；正式 run `cmv2_v112_part_se3_ddp_formal_20260921T030019Z`（STOPPED，step 200 / epoch 1，未进入 validation）。
- evidence：[实现、停止与双卡门禁 Activity](../activities/ACT-20260920-CMV2-V112-ENDPOINT-PART-SE3.md)；各 run 的原始 manifest、metrics、日志与 checkpoint 位于其 NAS output 目录。
- conclusion：`INCONCLUSIVE`。当前 smoke 只证明双卡接线、finite forward/backward、15组 validation 覆盖与 checkpoint 一致性；正式 run 在首轮 validation 前停止，其前 200 step loss 不证明模型效果。
- limitations：直接逐部件 SE(3) 不施加真实关节约束；普通逐点 flow loss 可能由大部件主导；在线 endpoint KNN 仍读取完整高分辨率手点流；正式 run 的数据加载阶段约耗时 17 分钟；V1.12 loss/checkpoint 语义与 V1.11.1 不可直接横向比较。
- next：在独立 V1.13 分支固定加速问题的研究边界，诊断并验证 I/O 优化后，再由用户确认是否重启正式训练；不得把工程吞吐改善冒充科学效果。
