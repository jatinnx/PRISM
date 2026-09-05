#!/usr/bin/env bash
# Queue v9 -- GPU-only, runs AFTER queue v8 (the shadow pair) has finished.
#
# WHY IT WAITS FOR v8.  v8 (run_queue_v8.sh) is still on the card: it trains the
# prism-v8-shadow-on row, evaluates its best four ways, then trains the
# prism-v8-shadow-off control and evaluates that.  All of that is already
# ordered correctly on the GPU and must not be raced.  v9 therefore does
# nothing until no run_queue_v8.sh / train_prism process exists any more, then
# executes its own rows one at a time on the GPU.  Nothing in v9 uses the CPU
# for measurement; every row below is an evaluate_prism GPU pass.
#
# WHAT v9 RUNS (decision-rule rows only -- zero training):
#   Stage 2 post-hoc rows on the prism-v8-shadow-on best checkpoint, the
#   matrix that decides the class-prior logit adjustment:
#     plain            reproduces the in-training mIoU (integrity gate)
#     gate1            presence-gate 1.0 -- the ONLY row that keeps
#                      --save-preds, per the disk budget (96% full, 18 GB)
#     vote / gate1_vote
#     adj rows         --logit-adjust in {-1, +0.5, +1, +2} with gate 1.0,
#                      + adj_p1 alone and adj_p1_vote
#   Needs, already measured: artifacts/class_priors.json
#   (python -m e3_only.tools.measure_class_priors) and artifacts/regions_val.npz
#   for the region-vote rows.
#
# FUTURE TRAINING ROWS: append them to the TRAIN_AFTER list at the bottom once
# approved (Stage 1a 4-term, Stage 1b aggregator, Stage 2 retrain without
# rare_class_factor).  Each entry is <ablation> <run-dir-name> and gets the same
# train -> best-eval treatment as v8's rows.
#
# Usage:
#   nohup bash e3_only/run_queue_v9.sh > /dev/null 2>&1 &
#   (PY overridable: PY=/path/to/python bash e3_only/run_queue_v9.sh)
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/home/cse-sdpl/Downloads/point_only_semseg/.venv/bin/python}
PKG=$(pwd)
QDIR=$PKG/e3_only/runs/_queue_v9
QLOG=$QDIR/queue.log
LOCK=$QDIR/queue.lock
NEED_MIB=7000
mkdir -p "$QDIR"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "queue v9 already running as PID $(cat "$LOCK")" >&2; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
say() { echo "[$(date +%FT%T)] $*" >> "$QLOG"; }

say "queue v9 started (PID $$), waiting for queue v8 (shadow pair) to finish"
while pgrep -f "run_queue_v8.sh|train_prism" > /dev/null; do sleep 60; done
say "no v8/training process left; GPU rows can start"
for _ in $(seq 1 240); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge "$NEED_MIB" ] && break
  sleep 30
done
say "GPU free: ${free:-unknown} MiB"

# ------------------------------------------------------------------ evaluate --
eval_one () {                         # $1 = label, rest = evaluate_prism flags
  local name=$1; shift
  say "eval [$name] $*"
  # shellcheck disable=SC2086
  $PY -u -m e3_only.evaluate_prism --checkpoint "$CK" --which teacher "$@" \
      --log "$OUT/eval_${name}.log" >> "$OUT/stdout.log" 2>&1
  if [ $? -eq 0 ]; then
    say "eval [$name] ok: $(grep -m1 '^mIoU' "$OUT/eval_${name}.log")  $(grep -m1 '^GHOST' "$OUT/eval_${name}.log")  $(grep -m1 '^worst 5' "$OUT/eval_${name}.log")"
  else
    say "eval [$name] FAILED -- tail:"; tail -12 "$OUT/stdout.log" >> "$QLOG"
  fi
}

# --- Stage 2 post-hoc rows on the shadow-on best --------------------------
CK=$(ls -t "$PKG"/e3_only/runs/prism-v8-shadow-on/*_best.pt 2>/dev/null | head -1)
OUT=$PKG/e3_only/runs/stage2-posthoc
if [ -z "$CK" ]; then
  say "ABORT: no *_best.pt in runs/prism-v8-shadow-on"
else
  mkdir -p "$OUT"
  say "=== Stage 2 rows on $CK"
  eval_one plain
  eval_one gate1          --presence-gate 1.0 --save-preds "$OUT/preds_gate1"
  eval_one vote           --region-vote
  eval_one gate1_vote     --presence-gate 1.0 --region-vote
  eval_one adj_neg1_gate1 --logit-adjust -1.0 --presence-gate 1.0
  eval_one adj_p05_gate1  --logit-adjust 0.5  --presence-gate 1.0
  eval_one adj_p1_gate1   --logit-adjust 1.0  --presence-gate 1.0
  eval_one adj_p1         --logit-adjust 1.0
  eval_one adj_p2_gate1   --logit-adjust 2.0  --presence-gate 1.0
  eval_one adj_p1_vote    --logit-adjust 1.0  --region-vote
  say "=== Stage 2 rows done -- winner: re-run it with --save-preds and read"
  say "    it with tools/diagnose_failures.py for the ghost/absorption split."
fi

# --- future training rows (append when approved) --------------------------
# Each entry: "ablation run-dir-name".  Example rows from v8-plan.md:
#   "no-shadow-improved prism-v9-control"   -- already covered by queue v8
#   Stage 1a (4 terms) / Stage 1b (aggregator k_temperature) / Stage 2
#   (retrain with rare_class_factor removed) -- decide after Stage-2 evals.
TRAIN_AFTER=()

if [ ${#TRAIN_AFTER[@]} -gt 0 ]; then
  say "=== training rows follow"
  for spec in "${TRAIN_AFTER[@]}"; do
    abl=${spec%% *}; dir=${spec#* }
    say "=== TRAIN $abl -> runs/$dir"
    $PY -u -m e3_only.train_prism --ablation "$abl" --epochs 40 --batch-size 1 \
        --save-dir "$PKG/e3_only/runs/$dir" >> "$PKG/e3_only/runs/$dir/train_stdout.log" 2>&1
    say "training [$abl] exited rc=$?"
  done
else
  say "no training rows in TRAIN_AFTER -- v9 is eval-only"
fi

say "queue v9 finished."
