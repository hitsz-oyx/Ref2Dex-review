# correspondence_ptv3

`correspondence_ptv3` 使用与 `corresponse_v1` 相同的 Stage 3 数据格式和训练入口，但把编码主干替换成了 two-stem + unified serialized attention backbone。

实现上参考了 `/home/oyx/test_ws/PointTransformerV3` 的序列化思路，但为了适配当前环境，去掉了 `spconv / torch_scatter / flash_attn` 依赖，保留：

- object stem: `[xyz, normal] -> hidden`
- hand stem: `[xyz, normal, hand_cano] -> hidden`
- unified serialized patch self-attention backbone
- object contact head
- object-to-hand canonical correspondence head
- hand-to-object cross-edge contact head

可选项：

- `meta.use_finger_region_head=true` 时启用 finger / region 分类头

训练：

```bash
python -m src.task.correspondence_ptv3.train \
  --data outputs/train_corr_static/... \
  --output-dir outputs/train/correspondence_ptv3
```
