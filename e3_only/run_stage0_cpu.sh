#!/usr/bin/env bash
# Stage 0 (v8-plan.md) run entirely ON CPU, so that it needs no GPU and the
# prism-v8-shadow-on training run is never touched.
#
# WHY CPU IS A VALID PLACE TO MEASURE THIS.  evaluate_prism._posterior and both
# new measurement tools run under torch.no_grad() with NO autocast anywhere
# (grep: the only "amp" hits in the eval path are in unrelated docstrings), so
# eval is fp32 on either device.  CPU and CUDA fp32 differ only by summation
# order, ~1e-6 on a cosine, which cannot move an argmax except at an exact tie.
# The numbers below are therefore the same numbers the GPU would print -- they
# just take ~4 s/image instead of ~0.4 s.
#
# WHY IT IS SEQUENTIAL AND THREAD-CAPPED.  The trainer holds one full core
# (100% of PID 77901) plus two light dataloader workers.  The box has 36, so
# 16 threads for this script leaves the trainer untouched; running the three
# passes in parallel instead would triple memory and contend for cache for no
# wall-clock gain, since each pass is already multi-threaded.
#
# Usage:  nohup ./e3_only/run_stage0_cpu.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1

PY=/home/cse-sdpl/Downloads/point_only_semseg/.venv/bin/python
PKG=/home/cse-sdpl/Downloads/point_only_semseg/PRISM/e3_only
ART=$PKG/artifacts
V5=$PKG/e3_only/runs/prism-v5-corrected/PRISM-no-shadow-improved_best.pt
LOG=$ART/stage0_cpu.log

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p "$ART"
say() { echo "[$(date +%FT%T)] $*" | tee -a "$LOG"; }

if [ ! -f "$V5" ]; then say "ABORT: checkpoint of record missing: $V5"; exit 1; fi
say "Stage 0 on CPU started (PID $$).  checkpoint of record: $(basename "$V5")"
say "  $(stat -c %s "$V5") B  -- must be ~11.9 MB (rank-8 LoRA present; ~7.1 MB means the adapter was dropped)"

step () {                              # $1 = label, rest = command
  local label=$1; shift
  say "--- BEGIN $label"
  local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then
    say "--- END   $label  ok  ($((SECONDS - t0)) s)"
  else
    say "--- END   $label  FAILED rc=$?  ($((SECONDS - t0)) s)  -- see $LOG"
  fi
}

# 0e -- prototype geometry over the full val set.  The checkpoint-only columns
# are already known; what this adds is the REALISED excess (aggregate - max_k
# cos) at every pixel and at argmax-winning pixels, over 1319 images instead of
# 8, and the rank correlation against the per-class ghost rate.  That
# correlation is the number that decides whether Stage 1b is a lever.
step "0e proto_geometry (full val)" \
  $PY -u -m e3_only.tools.proto_geometry --checkpoint "$V5" --which teacher \
      --log "$ART/proto_geometry_v5corrected.txt"

# 0d -- inventory oracle.  Restricting the argmax to the classes the image
# actually contains is the ceiling of the whole L_inventory mechanism, in the
# units of the metric, exactly as oracle_partition.py's 0.9438 is the ceiling
# of the region machinery.  MEASUREMENT ONLY: it reads dense val masks.
step "0d oracle_inventory (full val)" \
  $PY -u -m e3_only.tools.oracle_inventory --checkpoint "$V5" --which teacher \
      --log "$ART/oracle_inventory_v5corrected.txt"

# 0b -- region vote, the highest-value zero-training measurement in the project.
# Two rows on one checkpoint: plain argmax (must reproduce 0.5477, which is the
# integrity check that caught the LoRA bug) and --region-vote, whose ceiling is
# the 0.9438 already on disk.  No --save-preds: predictions are ~500 MB and the
# volume is at 98%.
step "0b eval plain (reproduce 0.5477)" \
  $PY -u -m e3_only.evaluate_prism --checkpoint "$V5" --which teacher \
      --log "$PKG/e3_only/runs/prism-v5-corrected/eval_cpu_plain.log"
step "0b eval --region-vote" \
  $PY -u -m e3_only.evaluate_prism --checkpoint "$V5" --which teacher --region-vote \
      --log "$PKG/e3_only/runs/prism-v5-corrected/eval_cpu_region_vote.log"

say "Stage 0 (CPU) finished.  headline numbers:"
for f in "$PKG/e3_only/runs/prism-v5-corrected/eval_cpu_plain.log" \
         "$PKG/e3_only/runs/prism-v5-corrected/eval_cpu_region_vote.log"; do
  [ -f "$f" ] && say "  $(basename "$f" .log): $(grep -m1 '^mIoU' "$f")  $(grep -m1 '^PA' "$f")  $(grep -m1 'band 3px' "$f" | cut -d' ' -f1-5)"
done
