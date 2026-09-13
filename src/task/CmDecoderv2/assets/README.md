# CmDecoderv2 本机资产

训练和可视化使用 Dexplore RL 的右手 Inspire URDF。仓库内约定的本机入口为：

```text
src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf
```

`inspire_hand_new/` 应链接到 Dexplore 的同名资产目录；该目录及网格由根 `.gitignore` 排除，不能提交。
数据 view 的 manifest 会记录 URDF SHA256，防止换手型后静默复用旧 GT。
