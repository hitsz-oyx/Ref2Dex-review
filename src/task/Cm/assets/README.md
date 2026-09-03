# Cm runtime assets

这里存放 Cm Task 使用、由运行时读取但不是当前代码生成的外部资产。大型文件不提交到 Git，
该目录及其内容由根 `.gitignore` 排除。

目录规则：

- `src/task/Cm/assets/checkpoints/<model>/`：Cm 使用的预训练模型；
- `src/task/Cm/assets/body_models/`：Cm 专属 MANO、SMPL-X 等 body model；
- `src/task/Cm/assets/robot_models/`：Cm 专属机器人 URDF 和相关静态资产；
- 当前 `checkpoints/densetoken` 是指向旧 `src/task/Cm/densetoken_ckpt` 的过渡软链接。

仓库根 `assets` 目前只是指向本目录的兼容软链接；新配置和文档应使用
`src/task/Cm/assets/`，不要再把新资产放到根目录。

训练过程中生成的 checkpoint 不放这里，而放在 `outputs/<task>/<run_id>/checkpoints/`。cache 不放
这里，而放在 `data/processed_data/`。
