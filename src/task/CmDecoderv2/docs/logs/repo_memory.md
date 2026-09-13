# CmDecoderv2 可迁移事实

## 数据与手型合同

- decoder view schema 为 `ref2dex_cm_decoder_v2_dexplore_view_v1`，规范入口位于
  `data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/manifest.json`，训练索引为同目录 `index.json`。
- Dexplore RL tensor 的 Inspire native q 位于 `[:,373:391]`；Task-local URDF 入口为
  `src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf`，机器上由软链接提供，view manifest
  必须绑定源 tensor、URDF、geometry 与实现 SHA256。
- 正式 OICM checkpoint 必须等对应训练终态后固定 SHA256；中途 `best.pt` 只允许工程 smoke，不能作为
  正式 decoder 训练的无歧义输入。

## 运行边界

- `CmDecoderv2` 不修改或兼容导入旧 `src/task/CmDecoder/` 的 q/wrist/point-flow 路径。
- 训练入口为 `python -m src.task.CmDecoderv2.train`；MANO 定性入口为
  `python -m src.task.CmDecoderv2.visualize`，后者必须显式记录初始 Inspire state 且只执行 `h=1`。
