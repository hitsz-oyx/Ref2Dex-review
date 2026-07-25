# Cp human closure

Run the stages in order (all use `src.base.BaseRunner`):

```bash
PYTHONPATH=. python -m process.stage5.prepare_cp --grab-root dataset/GRAB/data --seq s1/cup_lift --output-root processed_data/generated/stage5 --device cuda
# Warm-start the supplied old Cm checkpoint and train only its new Cm->Fm decoder.
PYTHONPATH=. python -m src.task.Cm.train --config src/task/Cm/configs/hand_decoder_warm_start.yaml --output-dir outputs/train/cm_hand_decoder
PYTHONPATH=. python -m src.task.Cp.train --config src/task/Cp/configs/baseline.yaml --set meta.stage=cp_effect --output-dir outputs/train/cp_effect
PYTHONPATH=. python -m src.task.Cp.train --config src/task/Cp/configs/baseline.yaml --set meta.stage=closed_loop --set meta.cm_hand_checkpoint=outputs/train/cm_hand_decoder/checkpoints/best.pt --set meta.cp_effect_checkpoint=outputs/train/cp_effect/checkpoints/best.pt --output-dir outputs/train/cp_closed_loop
DISPLAY=localhost:10.0 PYTHONPATH=. python -m src.task.Cp.visualize --checkpoint outputs/train/cp_closed_loop/checkpoints/best.pt --input processed_data/generated/stage5/s1/cup_lift_right.npz
```

The legacy Cm checkpoint is a warm start, not `train.resume`: its old Cm
encoder/object-flow branch is frozen while its new hand decoder is trained.
Stage C freezes the complete pretrained Cm current-state encoder and Cm->Fm
decoder, plus the Stage-B Cp encoder/effect decoder. Only the independent
current-hand-conditioned `Cp -> Cm` generator is optimized.
