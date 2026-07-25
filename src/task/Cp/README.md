# Cp human closure

Run the stages in order (all use `src.base.BaseRunner`):

```bash
PYTHONPATH=. python -m process.stage5.prepare_cp --seq s1/cup_lift --output-root processed_data/generated/stage5 --device cuda
PYTHONPATH=. python -m src.task.Cp.train --config src/task/Cp/configs/baseline.yaml --set meta.stage=hand_decoder --output-dir outputs/train/cp_hand_decoder
PYTHONPATH=. python -m src.task.Cp.train --config src/task/Cp/configs/baseline.yaml --set meta.stage=cp_effect --output-dir outputs/train/cp_effect
PYTHONPATH=. python -m src.task.Cp.train --config src/task/Cp/configs/baseline.yaml --set meta.stage=closed_loop --set meta.hand_decoder_checkpoint=outputs/train/cp_hand_decoder/checkpoints/best.pt --set meta.cp_effect_checkpoint=outputs/train/cp_effect/checkpoints/best.pt --output-dir outputs/train/cp_closed_loop
DISPLAY=localhost:10.0 PYTHONPATH=. python -m src.task.Cp.visualize --checkpoint outputs/train/cp_closed_loop/checkpoints/best.pt --input processed_data/generated/stage5/s1/cup_lift_right.npz
```

Stage A's teacher encoder sees GT `hand_flow`; the hand decoder does not.
Stage C freezes the Stage-A hand decoder and Stage-B Cp encoder/effect decoder;
only the current-hand-conditioned `Cp -> Cm` generator is optimized.
