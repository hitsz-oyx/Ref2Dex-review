# correspondence_ptv3

`correspondence_ptv3` 直接读取当前 Stage 3 点池格式，训练时按 epoch 采样
512 个物体点并生成增强后的输入 KNN。

实现上参考了 `/home/oyx/test_ws/PointTransformerV3` 的序列化思路，但为了适配当前环境，去掉了 `spconv / torch_scatter / flash_attn` 依赖，保留：

- object stem: `[xyz, normal] -> hidden`
- hand stem: `[xyz, normal, hand_cano] -> hidden`
- unified serialized patch self-attention backbone
- object contact head
- object-to-hand canonical correspondence head
- hand-to-object cross-edge contact head

可选项：

- `model.use_finger_region_head=true` 时启用 finger / region 分类头

训练：

```bash
python -m src.task.correspondence_ptv3.train \
  --data processed_data/generated/stage3/... \
  --output-dir outputs/train/correspondence_ptv3
```
