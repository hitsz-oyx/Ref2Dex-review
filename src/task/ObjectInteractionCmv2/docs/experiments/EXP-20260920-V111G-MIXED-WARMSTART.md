# 五组混合训练：权重初始化与启动门禁

- work_version：`V1.11.1`
- hypothesis：以既有二域best权重初始化的三域MANO/Inspire混合训练，可在分组验证上获得可复核的泛化表现。
- protocol：见 [V1.11g](../plan/V1.11g.md)，五组固定采样概率、两卡每卡64、16epochs、stride1..3，validation按domain×variant×stride分别报告。
- runs：`cmv2_v111g_split_20260920`（COMPLETED）；旧整对象 smoke（FAILED）；`cmv2_v111h_oakink2_parts_20260920T091403Z`（adapter COMPLETED）；GPU1+GPU3 smoke（COMPLETED）；`cmv2_v111h_mixed_ddp_formal_20260920T111159Z_nofile262k`（STOPPED at step160）；GPU0+GPU2 smoke（COMPLETED）；`cmv2_v111i_mixed_ddp_formal_20260920T114242Z` 当前 `RUNNING`。
- evidence：[Activity及NAS原始入口](../activities/ACT-20260920-CMV2-V111G-MIXED-DDP-GATE.md)。
- conclusion：部件 adapter 与 GPU1+GPU3 smoke 的工程门禁通过；formal 旧 run 已受控停止且未完成 epoch，科学结论仍为 `INCONCLUSIVE`。
- limitations：OakInk2含436/1849多部件片段，整对象GT不能统一按首部件位姿做刚体FK。当前adapter的适用性假设被真实数据否定；该结果不是模型效果结论。
- next：观察 `cmv2_v111i_mixed_ddp_formal_20260920T114242Z` 至 16 epochs 完成，再核对 checkpoint、metrics 与分组验证；运行中不改变资源或研究变量。
