# Archived runs — and the 2026-09-04 delete manifest

**This directory used to promise that nothing in it had been deleted. That is no
longer true.** On 2026-09-04, Stage C Tier 1 of `v8-plan.md` removed 8.4 GB of
checkpoints and prediction PNGs from `PRISM/` (17 GB -> 8.8 GB) to clear a volume
that had reached 98% full.

The rule that was applied, and that anything deleted later should also satisfy:

> **every measured number stays on disk as text.** Only bytes that are either
> (a) reproducible by a documented command from a checkpoint that was kept, or
> (b) output of a model already recorded as invalid, were removed.

So all 54 `*_train.log`, `*_eval.log`, `*_metrics.jsonl` and `*.md` files under
`_archive/` survive, as do all 11 files in `../artifacts/`. Nothing that a number
in `METHOD.md` or `v8-plan.md` is quoted from was touched.

## What was deleted, and how to get it back

| deleted | size | its number, which is still on disk | regenerate with |
|---|---|---|---|
| `smoke/` (4 run dirs, `*.pt`) | 1.5 GB | mIoU 0.0865 @ep0 — `smoke/runs-smoke/PRISM-full_epoch_0001_eval.log` | don't; smoke tests have no evidential value |
| `prism-superseded/**/*.pt` (5 dominated runs) | 1.6 GB | 0.5388 / 0.5178 / 0.3523, and two never evaluated — each run's kept `*_eval.log` + `*_metrics.jsonl` | re-train; they are strictly dominated by `prism-v5-corrected` |
| `prism-superseded/prism-no-shadow-eval-outputs/preds_*` | 991 MB | 0.1807 and 0.1428 — `eval_ep20.log`, `eval_ep40_regionvote.log` | **never.** Output of the LoRA-less network; see `INVALID_PREDICTIONS.md` |
| `rejected-fixed-version/eval_predictions/` | 971 MB | E3 @ep50 = **0.5037 / PA 0.7000** — `rejected-fixed-version/eval_logs/E3_epoch_0050_eval.log`, the only record of that endpoint anywhere | `run_experiment --evaluate --checkpoint e3-baseline/checkpoints/E3_epoch_0050.pt --save-preds ...` |
| `e3-baseline/eval_predictions/` | 989 MB | E3 @ep30 = **0.5417 / PA 0.7239**, plus `../artifacts/diagnose_E3_ep30.txt` and `confusion_E3_ep30_rownorm.csv` | `METHOD.md` step 2a (checkpoint `E3_epoch_0030.pt` was kept for exactly this) |
| `../runs/prism-no-shadow/*.pt` | 388 MB | 0.5493 @ep20 in-training, **unreproducible** — see `INVALID_PREDICTIONS.md` | impossible, and that is the point: 38 of 132 tensors were saved |
| `../e3_only/runs/prism-v6-lora12/` | 916 MB | rank-12 LoRA, 0.4438 @ep5 vs rank-8's 0.4677 @ep4; quoted in `configs/prism.py:65-68` | re-train with `--lora-rank 12` if the ablation row is wanted |
| `../e3_only/runs/prism-v5-corrected/*.pt` (11 periodic) | 623 MB | one `*_eval.log` per epoch, all kept — the full 60-epoch curve is intact as text | re-train; `_best.pt` and `_epoch_0035.pt` (the 0.5477 epoch) were kept |
| `../e3_only/runs/prism-v5-corrected/PRISM_best_predictions/` | 495 MB | `PRISM_best_eval.log` + `../artifacts/diagnose_v5corrected_0.5477.txt` | `METHOD.md` step 2a, ~10 min from the kept `_best.pt` |
| orphans: `_diag_floor.py`, `data/{train_fixed,smoke}.json`, `run_evals.sh`, `run_queue_v7.sh`, every `__pycache__/`, the triple-nested `e3_only/e3_only/e3_only/` | ~1 MB | none | nothing referenced them; `grep -rn` over the live tree confirmed it before deleting |

## What is still here, and why

### `e3-baseline/checkpoints/` — 7.2 GB, ten checkpoints, deliberately kept

E3 saved the **full frozen ViT** in every checkpoint, not just the trainables,
which is why one epoch costs 732 MB. This is the single largest thing left in
`PRISM/` and it is Stage C **Tier 2**: it can only go once **Stage 0f** has
evaluated `E3_epoch_{0005,0010,0015,0035,0040,0045}.pt`, the six that have never
been evaluated. Eval logs exist for epochs 20, 25, 30 and 50 only, so deleting
them today would leave the E3 degradation curve permanently 4 points wide — and
that curve ("a loss made mostly of the network's own output degrades with
training") is the entire justification for PRISM existing.

After Stage 0f: keep `E3_epoch_0030.pt` (0.5417, the paper's comparison row) and
`E3_epoch_0050.pt` (0.5037, the degradation endpoint), delete the other eight,
recover 6.1 GB.

### Every `.log`, `.jsonl`, `.md` and `.txt` under this directory

~1.3 MB total, and the only irreplaceable bytes in the tree. Never delete these.

### The two kept PRISM runs are the same model twice

| kept | what it is |
|---|---|
| `../runs/prism-no-shadow` | logs only now. In-training **0.5493 @ep20**, but the checkpoints held 38 of 132 tensors and are gone; the directory survives as the *evidence for that bug* |
| `../e3_only/runs/prism-v5-corrected` | **the checkpoint of record: 0.5477 / PA 0.7345 at epoch 35**, offline eval reproduces it exactly |

They differ in **5 of 89 config fields** and none of the five is a
hyper-parameter (`epochs`, `experiment`, `save_dir`, `finch_init_batches`,
`gate_warmup` present-vs-absent). So they are one model trained twice, the
0.16pp between them is run-to-run noise, and the baseline PRISM figure to quote
is **0.548 +/- 0.002** — not two independent results.

## The `e3_only/e3_only/` nesting

Not a mistake in the tree, a mistake in the launch command. `configs.base.resolve`
checks `PACKAGE_ROOT` (= `PRISM/e3_only`) *first*, so a run launched from
`PRISM/e3_only` with `--save-dir e3_only/runs/x` lands in
`PRISM/e3_only/e3_only/runs/x`. A third level had accumulated the same way; it was
deleted on 2026-09-04. The checkpoint of record lives at the second level and is
staying there — the path is in `METHOD.md` and in the Stage 0 scripts. New runs
pass an absolute `--save-dir`.







# claude 

