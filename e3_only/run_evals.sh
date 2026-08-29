#!/bin/bash
cd /home/cse-sdpl/Downloads/point_only_semseg
VENV_PYTHON=".venv/bin/python"
LOGDIR="e3_only/runs/eval_logs"
PREDIR="e3_only/runs/eval_predictions"

mkdir -p "$LOGDIR" "$PREDIR/E3_epoch_0030" "$PREDIR/E3_epoch_0050"

echo "[$(date)] Starting EP30 eval..."
$VENV_PYTHON -u -m e3_only.run_experiment --evaluate \
    --config e3_only/configs/e3_teacher_student.py \
    --checkpoint e3_only/runs/checkpoints/E3_epoch_0030.pt \
    --device cuda --num-workers 0 \
    --save-preds "$PREDIR/E3_epoch_0030" \
    --log "$LOGDIR/E3_epoch_0030_eval.log" \
    > "$LOGDIR/E3_epoch_0030_stdout.log" 2>&1
echo "[$(date)] EP30 done. Exit code: $?"

echo "[$(date)] Starting EP50 eval..."
$VENV_PYTHON -u -m e3_only.run_experiment --evaluate \
    --config e3_only/configs/e3_teacher_student.py \
    --checkpoint e3_only/runs/checkpoints/E3_epoch_0050.pt \
    --device cuda --num-workers 0 \
    --save-preds "$PREDIR/E3_epoch_0050" \
    --log "$LOGDIR/E3_epoch_0050_eval.log" \
    > "$LOGDIR/E3_epoch_0050_stdout.log" 2>&1
echo "[$(date)] EP50 done. Exit code: $?"

echo "[$(date)] All evaluations complete."
