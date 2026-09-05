#!/usr/bin/env bash
# Stage 2 (post-hoc) evaluation rows -- v8-plan.md Stage 2, decision-rule only.
#
# ZERO TRAINING. Every row evaluates the SAME best checkpoint and changes only
# what happens between the classifier logits and the argmax:
#   --logit-adjust tau   z_c <- z_c - tau*log pi_c   (class-prior term; tau>0 =
#                        balanced rule, tau<0 reverses -- both directions run)
#   --presence-gate 1.0  soft per-image inventory prior from the presence head
#   --region-vote        pool over the frozen SAM partition
# Rows are therefore directly comparable to each other AND to the checkpoint's
# own in-training eval (plain row, which reproduces it).
#
# Needs, once, before the first run:
#   python -m e3_only.tools.measure_class_priors      (writes artifacts/class_priors.json)
# and the region cache for --region-vote rows:
#   python -m e3_only.tools.build_region_cache --split val
#
# Each row is ~10-20 min on the card. All rows land in runs/stage2-posthoc/ as
# eval_<row>.log; the summary at the end greps the three numbers that decide the
# next step (mIoU for the headline, GHOST for the inventory claim, and the
# worst-class list to read against the Stage-2 scoreboard).
#
# Usage:
#   PY=/path/to/venv/bin/python bash run_stage2_eval.sh
#   (PY defaults to `python`; run from anywhere, paths are resolved here.)
set -u
cd "$(dirname "$0")/.." || exit 1           # -> PRISM/ (the e3_only package parent)
PY=${PY:-python}
PKG=$(pwd)
CK=$PKG/e3_only/runs/prism-v8-shadow-on/PRISM-improved_best.pt
OUT=$PKG/e3_only/runs/stage2-posthoc
mkdir -p "$OUT"

[ -f "$CK" ] || { echo "no best checkpoint at $CK" >&2; exit 1; }
[ -f "$PKG/e3_only/artifacts/class_priors.json" ] || {
  echo "run tools/measure_class_priors.py first (needed by --logit-adjust rows)" >&2; }

eval_row () {                               # $1 = row name, rest = flags
  local name=$1; shift
  echo "[$(date +%FT%T)] row $name: $*" | tee -a "$OUT/summary.log"
  # shellcheck disable=SC2086
  $PY -u -m e3_only.evaluate_prism --checkpoint "$CK" --which teacher "$@" \
      --log "$OUT/eval_${name}.log" >> "$OUT/stdout.log" 2>&1 \
    || { echo "  row $name FAILED:"; tail -8 "$OUT/stdout.log"; return; }
  grep -E '^(mIoU|PA|GHOST|worst 5)' "$OUT/eval_${name}.log" \
    | sed "s/^/  /" | tee -a "$OUT/summary.log"
}

eval_row plain
eval_row gate1          --presence-gate 1.0
eval_row vote           --region-vote
eval_row gate1_vote     --presence-gate 1.0 --region-vote
eval_row adj_neg1_gate1 --logit-adjust -1.0 --presence-gate 1.0
eval_row adj_p05_gate1  --logit-adjust 0.5  --presence-gate 1.0
eval_row adj_p1_gate1   --logit-adjust 1.0  --presence-gate 1.0
eval_row adj_p1         --logit-adjust 1.0
eval_row adj_p2_gate1   --logit-adjust 2.0  --presence-gate 1.0
eval_row adj_p1_vote    --logit-adjust 1.0  --region-vote

echo
echo "done. rows in $OUT -- next: re-run the winner with --save-preds and"
echo "read it with tools/diagnose_failures.py for the error-budget split."
