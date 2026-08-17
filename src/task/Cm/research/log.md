# Cm 研究日志

## 实验：Scene Cache V1 合成 parity

**假设**

将 object-only 输入扩展为 object + environment 的 local scene，并使用 mmap geometry、ragged candidate、sampling bank 和 DenseToken bank，不应改变旧 Cm 的输入/损失语义；静态 environment 点应产生零 flow。

**观察到的失败 / 现象**

工作区没有可用的 GRAB 原始 `.npz`，无法在本机完成真实 Stage4、DenseToken 全量预计算和正式训练。

**诊断**

需要先用合成数据验证 cache 各层和旧路径的行为一致性，再把真实数据实验留到提供 GRAB 原始数据的环境执行。

**改动**

增加 Scene Cache V1 构建器、通用 environment asset 读取、ragged candidate、sampling bank、DenseToken bank、fingerprint 校验、scene dataset/runner 路由和 object/environment 诊断指标；补充 Cm 任务文档与 parity 测试。

**结果**

`python3 -m pytest -q tests/test_cm_scene.py`：9 passed。覆盖 object-only regression、ragged candidate parity、sampling bank determinism/diversity、DenseToken FP16 cache parity、fingerprint guard、loss parity 和 scene flow calibration；environment candidate 被计入校准且零 flow 会降低 RMS。

**决策**

保留当前实现。真实 GRAB 训练结果不在本次环境中虚构，待原始数据可用后按 V1 三步 pipeline 执行。

**下一步**

在具备 GRAB 原始数据和对应 MANO/DenseToken 依赖的环境中构建全量 scene cache，完成 train-only flow calibration、dense bank parity 和正式训练。
