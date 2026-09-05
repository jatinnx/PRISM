#!/usr/bin/env bash
# Queue: train PRISM with the shadow branch ON, then the same config with it OFF,
# evaluating each four ways.  Start once and leave alone.
#
# WHY THIS PAIR EXISTS.  Every PRISM run measured before 2026-09-04 was launched
# with --ablation no-shadow-improved, which sets w_shadow = w_shead = 0.  The
# shadow mechanism had therefore never executed, and there is no measurement
# anywhere in the project of whether it helps.  The reason was not a design
# choice: with w_shadow > 0 the run dies at the first shadowed step with
# OutOfMemoryError in attn.softmax, because two grad-carrying 1024x1024 encoder
# passes do not fit in 15.57 GiB.  PrismNet.checkpointed() now recomputes the
# shadowed pass's 12 ViT blocks in backward instead of storing them; a 24-image
# smoke run reached shadow=1.6485 shead=0.8976 at 1.7 s/step, so the branch runs.
#
# ROW 1 is the shadow row.  ROW 2 is its control: identical constants, identical
# 40-epoch cosine schedule, shadow losses zeroed.  Only w_shadow/w_shead differ,
# so the difference between the two rows is attributable to the shadow terms and
# to nothing else.  40 epochs rather than the default 60 because the previous
# 60-epoch run peaked at epoch 35 and never came back (0.5477 -> 0.5460 at 60);
# both rows share the shorter schedule, so they stay comparable with each other.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/cse-sdpl/Downloads/point_only_semseg/.venv/bin/python
PKG=/home/cse-sdpl/Downloads/point_only_semseg/PRISM/e3_only
QDIR=$PKG/runs/_queue_v8
QLOG=$QDIR/queue.log
LOCK=$QDIR/queue.lock
NEED_MIB=7000
EPOCHS=40

mkdir -p "$QDIR"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "queue already running as PID $(cat "$LOCK")" >&2; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
say() { echo "[$(date +%FT%T)] $*" >> "$QLOG"; }

say "queue v8 started (PID $$), $EPOCHS epochs per row"
for _ in $(seq 1 120); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge "$NEED_MIB" ] && break
  sleep 30
done
say "GPU free: ${free:-unknown} MiB"

# ------------------------------------------------------------------ evaluate --
# Absolute paths everywhere.  --save-dir/--log/--save-preds go through
# configs.prism.resolve(), which prefers PACKAGE_ROOT/<relative path> and so
# turns "e3_only/runs/x" into "e3_only/e3_only/runs/x"; --checkpoint bypasses
# resolve() and is cwd-relative.  Absolute paths are immune to both.
eval_four () {                        # $1 = run directory (absolute)
  local dir=$1 ck
  ck=$(ls -t "$dir"/*_best.pt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then say "ABORT eval: no *_best.pt in $dir"; return 1; fi
  say "best checkpoint: $ck"
  local name flags
  for spec in "plain|--save-preds $dir/preds_plain" \
              "presence_gate|--presence-gate 1.0 --save-preds $dir/preds_presence_gate" \
              "region_vote|--region-vote" \
              "gate_and_vote|--presence-gate 1.0 --region-vote --save-preds $dir/preds_gate_and_vote"; do
    name=${spec%%|*}; flags=${spec#*|}
    say "eval [$name] $flags"
    # shellcheck disable=SC2086
    $PY -u -m e3_only.evaluate_prism --checkpoint "$ck" --which teacher $flags \
        --log "$dir/eval_${name}.log" >> "$dir/eval_stdout.log" 2>&1
    if [ $? -eq 0 ]; then
      say "eval [$name] ok: $(grep -m1 '^mIoU' "$dir/eval_${name}.log") $(grep -m1 '^PA' "$dir/eval_${name}.log") $(grep -m1 'band 3px' "$dir/eval_${name}.log")"
    else
      say "eval [$name] FAILED -- tail:"; tail -12 "$dir/eval_stdout.log" >> "$QLOG"
    fi
  done
}

# ------------------------------------------------------------------- train ----
run_row () {                          # $1 = ablation, $2 = run dir name
  local abl=$1 dir=$PKG/runs/$2
  mkdir -p "$dir"
  say "=== ROW $abl -> $dir"
  $PY -u -m e3_only.train_prism --ablation "$abl" --epochs "$EPOCHS" \
      --batch-size 1 --save-dir "$dir" >> "$dir/train_stdout.log" 2>&1
  local rc=$?
  say "training [$abl] exited rc=$rc"
  if [ $rc -ne 0 ]; then
    say "ABORT row $abl: training failed.  tail of train_stdout.log:"
    tail -25 "$dir/train_stdout.log" >> "$QLOG"
    return $rc
  fi
  say "training [$abl] best: $(grep -o 'done\. best mIoU [0-9.]* at epoch [0-9]*' "$dir"/*_train.log | tail -1)"
  eval_four "$dir"
}

run_row improved          prism-v8-shadow-on
run_row no-shadow-improved prism-v8-shadow-off

say "queue finished.  summary:"
for d in "$PKG"/runs/prism-v8-shadow-on "$PKG"/runs/prism-v8-shadow-off; do
  for f in "$d"/eval_*.log; do
    [ -f "$f" ] || continue
    printf '  %-28s %-24s %s  %s\n' "$(basename "$d")" "$(basename "$f" .log)" \
      "$(grep -m1 '^mIoU' "$f")" "$(grep -m1 'band 3px' "$f" | cut -d' ' -f1-5)" >> "$QLOG"
  done
done
