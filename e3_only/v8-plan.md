# v8 plan — collapse PRISM to a 4-term model and fix the classifier

## Context

PRISM currently carries 14 loss terms and delivers **+0.60 pp mIoU** over the E3
baseline (0.5477 vs 0.5417). One class of seventeen (`chaparral`, +0.237 recall)
carries all of it; remove that class and PRISM is **−0.55 pp worse** than E3.
Meanwhile **0 of the 16 declared ablation rows in `METHOD.md` §8 have ever been
run**, so every term is an assertion. That is the publishability problem — not the
number 14.

Measurement says the complexity is aimed at the wrong target. Decomposing the
objective at convergence (final epoch of the 0.5477 run), four terms carry
**88.9%** of it and three carry exactly zero:

```
prop 43.42%  point 30.19%  hom 10.05%  self 5.22%   <- 88.9%
repel 4.42%  bnd 2.18%  potts 1.92%  anchor 1.49%  absent 0.96%  present 0.16%
area 0.00% (inert from epoch 0)   shadow 0.00%  shead 0.00% (never executed)
rim 0.00% (weight is literally 0.0)
```

And the error is a *classifier* error, not a shape error. The frozen SAM partition
supports **mIoU 0.9438** if each region is labelled with its majority class
(`tools/oracle_partition.py`); the model delivers 0.5477. Only **0.0210** of all
error is speckle a 5×5 filter would fix, against **0.2201** that is right-shape-
wrong-label and **0.3964** that is pixels assigned to a class the image does not
contain.

Two things the classifier audit changed in this plan. First, an earlier draft
blamed `embed_dim=64` for crowding 68 prototypes; that is **refuted** — the Welch
bound for 68 unit vectors in R^64 is 0.031, and `model/decoder_v2.py:75` projects
from a 64-channel trunk anyway, so raising `embed_dim` alone cannot add a single
dimension of room. Measured within-class prototype cosine is **+0.440** on the
checkpoint of record, 14× the geometric floor: the crowding is produced by the
objective and the aggregator, not by the dimension. Second, the aggregator itself
contains an unaccounted per-class logit term worth **1.386× the angular margin**
(next section) — a one-line lever nobody has touched.

**Goal of this plan:** a simple, defensible architecture — 4 loss terms — whose
complexity budget is spent on class discrimination instead of on shape priors, with
every remaining term backed by an ablation row that has actually been run.

### Verdict on the premises behind this plan

| premise | verdict |
|---|---|
| "E3's ~55% mIoU is impressive" | **Yes, and that is the problem.** 0.5417 from ~15 clicks/image is a strong point-only result — but it is 98.9% of what 14 PRISM terms deliver (0.5477), so the added machinery has no measured value yet. E3 also *degrades* with training (0.5417 @ep30 → 0.5037 @ep50); PRISM's real contribution so far is that it does not. |
| "3 of the 6 issues are major: shadows, boundaries, misclassification" | **Two-thirds right.** Misclassification is major and is really issues 2+5+6 fused (0.3964 + 0.2201 of all error). Shadows are moderate and untested (1.254× ratio). **Boundaries are not major** — 0.0210 of error, and the partition ceiling is 0.9438. Boundary work goes last. |
| "everything comes down to misclassification" | **Correct, and now mechanised.** It splits into (A) rare-class ghosting and (B) absorption by a spectral neighbour, both driven by per-class decision bias — one source of which is now identified in the aggregator itself. |
| "we need confidence-aware self-training" | **Half-built already.** Confidence *gating* exists and passes 81% of candidates; confidence *weighting* does not exist anywhere in `L_self`. Worth doing, but only per-class (Stage 3) — the scalar form entrenches both signatures. |

**Order of work, by measured mass ÷ cost:** disk cleanup (Stage C, first — see below,
the volume is 98% full) → free measurements (Stage 0) → 4 terms + aggregator (Stage 1)
→ classifier decision rule (Stage 2) → self-training (Stage 3) → shadow (Stage 4,
already in flight) → boundary (Stage 5).

"We don't want a crowd, we want a model" applies to two different crowds, and they
need separating: the **loss** carries 14 terms and collapses to 4 (the rest of this
plan), while the **repository** carries 17 GB of which ~3 GB is load-bearing —
and there the crowd is almost entirely dead binaries, not code. The code deletions
below total ~500 KB across seven orphan files; nothing that the model or the paper
depends on is removed.

### GPU schedule (decided): pause the queue, run Stage 0 first

Stage 0 is eval-only but still needs the card, and `prism-v8-shadow-on` (PID 77901)
holds 13.2 of 16.4 GiB. One correction to how this gets done: **`SIGSTOP` would not
free the VRAM** — a stopped process keeps its CUDA context and every allocation. The
mechanism is therefore *checkpoint, terminate, resume later*:

1. ~~Wait for the next periodic checkpoint.~~ **Already satisfied:**
   `PRISM-improved_epoch_0010.pt` was written at 14:45 (59,424,811 B), so the gate this
   schedule was waiting on is met and step 2 can run as soon as the plan is approved.
   Epoch time has drifted 824 s → 964 s, so waiting for epoch 15 instead would cost
   another ≈80 min for nothing.
2. `kill -TERM 77901` (the parent; 78039/78040 are its dataloader workers).
3. Run Stage 0a-0f (~3 h).
4. Resume with `--resume runs/prism-v8-shadow-on/PRISM-improved_epoch_0010.pt`.
   `train_prism.py:322-333` restores `student`, `optimizer`, `scheduler`, `gate` and
   `start_epoch`, so nothing is lost. **Use the 59 MB `epoch_XXXX.pt`, never the
   11.9 MB `_best.pt`** — the latter holds the teacher weights only and cannot resume
   an optimiser.
5. Then `prism-v8-shadow-off` as originally queued.

Net effect: the first new measurement arrives in ≈3 h instead of ≈11.5 h, and the
shadow pair still completes.

One thing that changes the urgency of the order: **the volume is 98% full — 9.6 GB
free of 433 GB.** `runs/prism-v8-shadow-on` needs ≈350 MB more (six more periodic
checkpoints at 59 MB; its ~496 MB `_best_predictions` dir is already on disk and is
overwritten in place rather than accumulated), the shadow-off row needs ≈1.1 GB, and
each Stage-1 row needs ≈600 MB. That fits in 9.6 GB, but only just, and a full disk
mid-epoch loses the run. Stage C below is pure `rm` on untracked, regenerable files,
costs no GPU, and frees ~8.5 GB — so it goes first and can run while the shadow row is
still training.

---

## Stage C — repository cleanup: 17 GB → ~3 GB, no GPU

> **Tier 1 EXECUTED 2026-09-04.** 8.4 GB / 52,577 files removed; `PRISM/` is
> 17 GB → **8.8 GB** and free space 9.4 GB → **17 GB** (98% → 96%). Manifest of
> what went and how to regenerate each item: `_archive/README.md`. The two
> documented dependencies below were fixed in the same pass (`METHOD.md` gained
> step 2a; `_archive/README.md` is now the manifest, not a no-delete promise).
> **Tier 2 is untouched** — the ten E3 checkpoints (7.2 GB) still gate on Stage 0f.

Permission granted by the user: *"u r allow to delete files from PRISM which is
unnecessory."* Everything below is inside `PRISM/`. Three properties hold for every
Tier-1 item, and each is checked rather than assumed:

- **untracked** — `git status --porcelain` in `PRISM/` lists 12 modified live files
  and 5 untracked paths; `_archive/` is one of the untracked ones, so `git ls-files`
  knows nothing about any of it and deleting it loses no history.
- **regenerable or superseded** — every deleted checkpoint has its per-epoch eval log
  kept beside it, and every deleted prediction directory has the numbers derived from
  it already extracted into `artifacts/`.
- **not read by any Stage-0 measurement** — the keep-list below is exactly the set of
  files Stage 0a–0e opens.

### Tier 1 — delete now, ~8.5 GB, zero information loss

| path | size | why it can go |
|---|---|---|
| `_archive/smoke/` | 1.5 GB | four smoke-test run dirs (`smoke_cpu`, `smoke_cpu_host`, `runs-smoke`, `nested-smoke`); no measurement ever came from them |
| `_archive/prism-superseded/**/*.pt` + `prism-no-shadow-eval-outputs/` | 2.6 GB | five strictly-dominated runs (0.5388 / 0.5178 / 0.3523 / 2 never evaluated). **Keep** every `*_train.log`, `*_metrics.jsonl`, `*_eval.log` and `INVALID_PREDICTIONS.md` — the scores live there |
| `_archive/rejected-fixed-version/eval_predictions/` | 972 MB | the version the user rejected outright. **Keep `eval_logs/E3_epoch_0050_eval.log`** — 8 KB, and the only record anywhere of E3's 0.5037 endpoint |
| `_archive/e3-baseline/eval_predictions/` | 989 MB | regenerable from the kept ep30/ep50 checkpoints; `artifacts/diagnose_E3_ep30.txt` + `confusion_E3_ep30_rownorm.csv` already hold everything read off them |
| `runs/prism-no-shadow/*.pt` | 388 MB | the LoRA-less run: 38 of 132 tensors saved, its 0.5493 unreproducible. **Keep `INVALID_CHECKPOINTS.md`, both eval logs, the jsonl and the train log** — that directory is the *evidence* for the bug |
| `e3_only/runs/prism-v6-lora12/` minus logs | 916 MB | rank-12 ablation, tracked behind rank 8 (0.4438 @ep5 vs 0.4677 @ep4); the comparison is already quoted in `configs/prism.py:65-68` |
| `e3_only/runs/prism-v5-corrected/*.pt` except `_best.pt` and `_epoch_0035.pt` | 650 MB | 11 redundant periodic checkpoints, each with its eval log kept |
| `e3_only/runs/prism-v5-corrected/PRISM_best_predictions/` | 495 MB | 6596 PNGs; regenerable in ~10 min from the kept `_best.pt`, and offline reproduction to 0.5477 is already verified |
| orphans (see list below) | ~500 KB | referenced by nothing in the live tree |

The orphan files, each confirmed unreferenced by `grep -rn` over `*.py`/`*.sh`/`*.md`
outside `_archive/`: `_diag_floor.py` (a scratch E3 diagnostic that loads
`runs/checkpoints/E3_epoch_0035.pt`, a path that no longer exists),
`data/train_fixed.json` (471 KB, the rejected version's manifest), `data/smoke.json`,
`run_evals.sh` and `run_queue_v7.sh` (superseded by `run_queue_v8.sh`), every
`__pycache__/`, the doubly-nested `e3_only/e3_only/e3_only/` stub left by the
`resolve()` path bug, and `../clude_MEMORY.txt.bak`.

### Tier 2 — 6.1 GB, gated on one measurement (new Stage 0f)

`_archive/e3-baseline/checkpoints/` holds **ten 766 MB checkpoints** — E3 saved the
full frozen ViT, not just the trainables, which is why the baseline costs 7.2 GB. Eval
logs exist for epochs 20, 25, 30 and 50 only; 5, 10, 15, 35, 40 and 45 have never been
evaluated, and deleting them would make the E3 degradation curve permanently
4 points wide.

So: **Stage 0f** evaluates those six checkpoints (~10 min each, ~1 h total) to complete
the curve that is the whole justification for PRISM existing — "a loss made mostly of
the network's own output degrades with training" is currently supported by
0.5417 @ep30 → 0.5037 @ep50 and nothing in between. Once the six numbers are on disk,
**keep only `E3_epoch_0030.pt` (0.5417, the comparison row) and `E3_epoch_0050.pt`
(0.5037, the degradation endpoint)** and delete the other eight: 6.1 GB.

If the card is wanted back sooner, this tier simply waits — the eight checkpoints cost
disk and nothing else, and Tier 1 has already bought the headroom.

### Tier 3 — never delete

The keep-list, which is also exactly what Stage 0 reads: `artifacts/` (all 9 files —
`regions_{train,val}.npz`, `prop_trust.json`, and the four diagnose/oracle outputs that
every number in this plan is quoted from), `data/{train,val}.json`,
`data/class_map.py`, `dlrsd/` and `data/val_masks_remapped/` (dataset and the
measurement-only masks), `sam_vit_b_01ec64.pth`, `METHOD.md`, `../clude_MEMORY.txt`,
`e3_only/runs/prism-v5-corrected/{PRISM-no-shadow-improved_best.pt,_epoch_0035.pt,
PRISM_best_eval.log,*_train.log,*_metrics.jsonl,*_eval.log}`, all of
`runs/prism-v8-shadow-on/`, `_archive/README.md`, and **every `.log`, `.jsonl`, `.md`
and `.txt` under `_archive/`** — the text is ~1.3 MB in total and is the only
irreplaceable thing in the tree.

Also kept, and worth saying explicitly because it looks like clutter and is not: the
**E3 baseline code path** — `train.py`, `evaluate.py`, `run_experiment.py`,
`configs/{base,e1_point_only,e2_prototypes,e3_teacher_student}.py`,
`core/{losses,prompts,prototypes,pseudo}.py`, `data/dataset.py`,
`model/{sam_wrapper,decoder}.py`. It is ~90 KB, it is unreachable from
`train_prism.py`, and it is the definition of the paper's baseline row, so it stays
until the paper is submitted. `configs/base.py` is load-bearing for a second reason:
`tools/{build_region_cache,validate_inventory,validate_regions}.py` import `resolve`
from it, and those three tools produced two of the three MEASURED constants in
`configs/prism.py`.

### Two documented dependencies this breaks, and the fix for each

1. `METHOD.md:1089` gives a reproduce command that reads
   `_archive/e3-baseline/eval_predictions/E3_epoch_0030`, which Tier 1 deletes. The
   recipe stays valid end-to-end but gains a first step: regenerate that directory with
   `evaluate.py --save-preds` from the kept `E3_epoch_0030.pt`. Edit the recipe, do not
   keep 989 MB to avoid editing three lines.
2. `_archive/README.md` opens with *"Nothing here was deleted -- every directory was
   moved, and moving it back restores the original layout."* That invariant is exactly
   what Stage C ends, so the README is rewritten as a manifest: what was deleted, what
   its measured score was, and the one command that regenerates it.

**Result:** `PRISM/` goes 17 GB → ≈3.1 GB after Tier 1+2 (`_archive` 14 GB → 1.8 GB,
the nested `e3_only/runs/` 2.1 GB → 72 MB, `runs/` 953 MB → 565 MB), free space goes
9.6 GB → ≈24 GB, and every number quoted anywhere in this plan or in `METHOD.md` is
still either on disk as text or regenerable from a kept checkpoint by a documented
command.

---

## The six issues, ranked by measured mass

Numbers from `artifacts/diagnose_v5corrected_0.5477.txt` (all 1319 val images).

| # | Issue | Measured | Verdict |
|---|---|---|---|
| 2 | Ghost class hallucination | GHOST 0.2263; **0.3964 of all error** | **MAJOR** |
| 5 | Correct shape, wrong label | **0.2201 of all error** | **MAJOR** |
| 6 | Spectral confusion | veg/soil 0.2866, bright surfaces 0.1588 of all confusion | **MAJOR** |
| 4 | Shadow mislabelling | err 0.3193 in shadow-like vs 0.2546 elsewhere = **1.254×** | moderate, **never tested** |
| 1 | Salt-and-pepper | rate 0.0160; 0.0210 of error | minor (**PRISM worse than E3's 0.0101**) |
| 3 | Dominant flooding | 0.0106 | effectively solved |

**Issues 2, 5 and 6 are one disease.** The user's instinct that "everything comes
down to misclassification" is correct and measured. One correction: **boundaries
are not a major issue** — 0.0210 of error is boundary/speckle-fixable and the shape
ceiling is 0.9438. Boundary work goes last, not first.

## The disease has two signatures, and they need different fixes

From the per-class table (`pred/GT` = predicted pixel count ÷ true pixel count):

**(A) Rare-class ghosting** — sprayed into images that do not contain the class:
`field` 2.11× at precision 0.163 (present in **8/1319** images), `mobile home`
1.66× (0.488), `ship` 1.37×, `cars` 1.33×, `court` 1.25×. Almost all are classes
in ≤ 5.5% of images.

**(B) Absorption by a larger spectral neighbour** — `bare soil` 0.70× → `grass`
0.185, `sand` 0.74× → `bare soil` 0.179, `dock` 0.64× → `ship` 0.350, `buildings`
0.84× → `pavement` 0.109. Each loses mass to a bigger, spectrally adjacent class.

Both are one thing: **the decision boundary is biased by class frequency, in both
directions.** `field` vs `grass` is an 80:1 imbalanced binary problem between two
spectrally near-identical classes. No shape prior can fix that; a class-balanced
decision rule and a per-image class restriction can.

### Signature (A) measured at image level — this is the decisive table

For each class: images whose GT contains it, images where it is predicted, and
images where it is predicted but **absent from GT**. Measured on the 0.5477
predictions, all 1319 val images:

| class | GT imgs | pred imgs | ghost imgs | image-level precision |
|---|---|---|---|---|
| `field` | **8** | 29 | 28 | **3%** |
| `tanks` | 70 | 202 | 134 | 34% |
| `court` | 73 | 175 | 103 | 41% |
| `sea` | 70 | 164 | 94 | 43% |
| `mobile home` | 72 | 154 | 84 | 45% |
| `airplane` | 70 | 137 | 67 | 51% |
| `bare soil` | 546 | 884 | 380 | 57% |
| `grass` | 669 | 1039 | 374 | 64% |

**Every rare class is detected in nearly every image that contains it and in one to
two hundred that do not.** `field` is the extreme: predicted in 29 images, and only
**one** of those 29 contains it. Total across all classes: **2398 ghost
(class, image) instances**, 1.82 per image — of which 1320 are large enough to clear
the area threshold the GHOST metric uses.

This says the misclassification problem is not primarily "the wrong class inside a
region". It is **"a class that is not in this scene at all"**, and the per-image
class inventory — which the point annotations give *exactly* at training time — is
the constraint that removes it.

Note that `class_weighting=True, rare_class_factor=4.0` upweights rare classes in
training — it *causes* (A) while trying to fix (B). A principled logit adjustment
replaces it.

### A mechanism for signature (A): the model already applies an *uncontrolled* per-class logit

The prototype aggregator, `core/protobank.py:145-148`, is

```python
def aggregate(self, cos):                     # (B,C,K,H,W) -> (B,C,H,W)
    t = max(self.k_temperature, 1e-3)         # t = 0.20
    return t * torch.logsumexp(cos / t, dim=2)
```

Write `m_c = max_k cos_ck`. Then `aggregate_c = m_c + t·log Σ_k exp((cos_ck−m_c)/t)`,
and the second term lies in **`[0, t·log K]`**: zero when the K prototypes disagree,
`t·log K` when they have collapsed onto one direction. At the shipped `t=0.20, K=4`
that ceiling is `0.20·ln 4 = 0.2773` cosine units. It is a **per-class additive
term in the logits**, so it moves the argmax between classes, and it is **not
removable by any uniform normalisation** — subtracting `t·log K` shifts every class
equally and changes nothing. (`_pool_presence` at `model/decoder_v2.py:125` does
subtract its `log N`, but there `N` is the varying pixel count, so that subtraction
is doing real work; here `K` is the same for all 17 classes.)

The size is the point. Divide by the point loss's margin, `margin=0.20`
(`core/objective.py:94`), and the learned scale cancels exactly:

> collapse bonus ÷ enforced margin = `t·log K / margin` = **`ln 4` = 1.386**

So **a class whose K prototypes have collapsed carries a standing advantage 39%
larger than the entire angular margin `L_point` fights to establish, at every pixel
of every image.** In absolute logits at the checkpoint's measured `scale 8.8`: 2.44
against a margin of 1.76.

Collapse is measured, not hypothesised. `PRISM_best_eval.log` reports
`intra_cos +0.440` — mean within-class prototype cosine — against
`repulsion_loss`'s own stated target of ≤ −0.10, and epoch 59 logs `repel=0.6136`,
i.e. the term meant to prevent collapse is nowhere near satisfied at `w_repel=0.05`
while the aggregator is paying 0.277 to maintain it. The live `prism-v8-shadow-on`
run confirms it independently and from a fresh initialisation: `repel` is
**0.7537 → 0.7545 → 0.7529 → 0.7496** across epochs 4–7 — flat, no progress — and
still **0.7365 at epoch 9**, with `intra_cos` **+0.652 at epoch 3 and +0.633 at
epoch 9**: six epochs moved the within-class prototype cosine by 0.019 while every
other term fell monotonically (`point` 1.91→1.15, `hom` 0.67→0.39).
And the classes that *start* collapsed are the
rare ones: `finch_init` fills unused slots with jittered near-duplicates when a class
has fewer clicked modes than K (`core/protobank.py:256-262`, `269-271`; jitter
0.01–0.02 of a unit vector, so an initialised duplicate pair sits at cosine ≈ 0.9999).
Its docstring says "which the repulsion term then spreads out" — two runs say it does
not.

Same root cause, secondary consequence: `cos` can reach `1 + t·log K = 1.277`, so
the quantity `margin_point_loss` subtracts a margin from is not a cosine, and
`margin=0.20` is not an angular margin in the ArcFace sense it is modelled on.

**This is the strongest single finding of the audit and it is nearly free to test.**
Three candidate fixes, in increasing order of departure from the current model:
`k_temperature` 0.20 → 0.05 (bonus falls to 35% of the margin), a hard `max_k`
(bonus exactly 0, sparse gradient — which is what a mixture model wants anyway), or
per-class `K_c` sized by the class's measured number of modes. **Correlation with
the ghost table is not asserted — it is a Stage 0 measurement** (below).

---

## Target architecture: 4 terms, down from 14

Keep the four that carry 88.9%, fold the survivors into them, delete the dead.

| term | what it merges | why it survives |
|---|---|---|
| `L_point` | `point` | the only human supervision; 30.2% of L |
| `L_region` | `prop` + `hom` + `self` | region consistency on the frozen partition; 58.7% of L |
| `L_inventory` | `absent` + `present` + `pres_head` | the fix for issue 2 — 0.3964 of all error |
| `L_shadow` | `shadow` + `shead` | the paper's novelty claim, and it has never once executed |

**Deleted outright:** `rim` (weight 0.0, and guarded by `if self.w.rim:` at
`core/objective.py:314` so it has never even been computed — dead code in a
published term list), `area` (0.0676 → 0.0001, never constrained anything).
**Deleted pending the ablation that proves it:** `potts` and `bnd` — 1.92% + 2.18%
of L, and PRISM's speckle rate is 0.0160 against E3's 0.0101, so the evidence
currently runs *against* the two terms whose job is smoothness.
**Coupled to the aggregator, not deleted on faith:** `anchor` (1.49%) and `repel`
(4.42%) are prototype regularisers, not supervision. `repel` **stays** — the
measured `intra_cos +0.440` and `repel=0.6136` say its constraint is unsatisfied,
not unnecessary, and the reason is that the aggregator pays for the collapse it
fights. Fix the aggregator first, then re-measure whether `repel` is still earning
its 0.05. `anchor` is at `0.1037` ⇒ cos(prototype, point-EMA anchor) ≈ 0.68, i.e.
the trained prototypes sit ~47° off their point-derived anchors; and the `ema` /
`ema_count` buffers are **not checkpointed** (`train_prism.py:96-99` filters
`state_dict()` by `requires_grad`), which is why every standalone eval log prints a
meaningless `proto live 0/68`.

Presented as **four supervision terms** with one ablation row each, which is what
§8's grouped rows already do (`no-inventory` zeroes three terms at once, `no-region`
zeroes three more). Stop advertising a count of 14.

---

## Staged execution — one issue at a time, each stage gated on a measurement

Every stage names the measurement that decides it and the condition that kills it.
No stage starts before the previous one's number exists.

### Stage 0 — measure what is already built but never measured (0 training)

Both inference mechanisms in `evaluate_prism.py` are confirmed **never validly
measured**: the string `presence_gate` appears in no log, txt or jsonl on disk, and
the only `--region-vote` number that exists (0.1428) came from the repudiated
LoRA-less load. Neither reads a dense mask — `pred` is finalised at
`evaluate_prism.py:202-208` and `gt` is not pulled from the batch until 214-216 — so
both are legitimate test-time mechanisms, not leaks.

**0a. One code fix must land first.** `evaluate_prism.py:338-350` classes
`decoder.presence.*` as *optional* and prints "inert unless --presence-gate > 0" —
then does not enforce that condition. Every checkpoint except `prism-v8-shadow-on`
predates the presence head, so `--presence-gate 1.0` on `prism-v5-corrected` would
leave the head **randomly initialised** and multiply the posterior by a random
prior. Make it `raise` when `optional and gate > 0`. Related quiet failure, same
file family: `dataset_prism.py:177-180` returns `n_sam = 0` when the key is absent,
which makes `region < n_sam` false everywhere and silently degrades `--region-vote`
to plain argmax. `artifacts/regions_val.npz` does carry `n_sam` (1319 ids, 2/75.2/301
min/mean/max), so it is latent — make it raise too.

**0b. Region vote on `e3_only/e3_only/runs/prism-v5-corrected/PRISM-no-shadow-improved_best.pt`.**
This is the highest-value zero-training measurement in the project: its ceiling is
*exactly* the 0.9438 already in `artifacts/oracle_partition_val.txt` (SAM-only,
`min_region=24` — the same eligibility rule `_region_vote` uses), and it is the only
mechanism that puts object-scale context at the decision step, which the classifier
itself cannot do (it is 1×1 at 256×256). Rows: plain / `--region-vote`.

**0c. Presence gate** — run on the v8 checkpoint once row 1 finishes, since that is
the only one with a trained head. Rows: plain / `--presence-gate 1.0` /
`--region-vote` / both, i.e. `run_queue_v8.sh:58-65`'s `eval_four`, which was
written and never fired (`runs/_queue_v8/queue.log` logged only the training launch).
Note the gate is a *soft* log-prior — `logits + strength·log(sigmoid(p)(1−floor)+floor)`
at `core/inventory.py:284-306`, worst case −3.0 logits at `floor=0.05` — so it cannot
delete a class, and a large effect would be surprising.

**0d. `tools/oracle_inventory.py`** — new, the exact analogue of
`tools/oracle_partition.py`: restrict the argmax to the classes the image actually
contains, per GT. That is the **ceiling of the whole inventory mechanism** in the
units of the metric, sizing issue 2 the way 0.9438 sized the shape prior.
Measurement-only and labelled as such — it reads dense masks, so like
`oracle_partition` it can never produce a reported number.

**0e. Prototype geometry, per class** — the measurement that decides whether the
aggregator finding is a lever or a curiosity. From the checkpoint alone plus one
val pass: per-class within-class prototype cosine, and the per-class *realised*
excess `aggregate − max_k cos` (the quantity bounded by 0.2773). Correlate against
the per-class ghost-image counts in the table above. If the over-predicted classes
are the collapsed ones, Stage 1b is the priority; if not, the aggregator is a
correctness fix worth one sentence and Stage 2 carries the mIoU alone.

**0f. Complete the E3 degradation curve** — evaluate `E3_epoch_{0005,0010,0015,0035,
0040,0045}.pt` with `evaluate.py`, the six of the ten archived checkpoints that have
never been evaluated. Two payoffs, and the second is why it is here rather than in a
"nice to have" list: it turns the plan's central baseline claim from a 4-point curve
into a 10-point one, and it is the **gate on deleting 6.1 GB** (Stage C Tier 2) — the
checkpoints cannot go while they are the only route to those numbers. ~1 h, and it is
the one Stage-0 item that may be deferred if the card is wanted back sooner.

**Already on disk, no work needed** (do not re-measure): E3's endpoints are recorded —
**0.5417 / PA 0.7239 at epoch 30** and **0.5037 / PA 0.7000 at epoch 50**
(the latter misfiled under `_archive/rejected-fixed-version/eval_logs/E3_epoch_0050_eval.log`,
corroborated by `_archive/README.md:24-25`), plus epochs 20 and 25 in
`_archive/e3-baseline/eval_logs/`. PRISM's own two seeds are 0.5493 @ ep20
(LoRA-less, unreproducible) and 0.5477 @ ep35 (reproduced offline) ⇒ quote
**0.548 ± 0.002**. Every in-training eval on disk is plain argmax, verified.

**Decides:** whether region vote and the inventory gate are worth pp for zero
training cost, what the inventory ceiling is, and whether the aggregator bias tracks
the ghost table. If the inventory ceiling is small, issue 2 is not the lever and
Stage 2 re-orders around signature (B).
**Cost:** ~2 h of eval for 0a–0e, +1 h for 0f, no training.

### Stage 1 — the 4-term model must match 0.5477, and the aggregator gets its row

Two training rows, both at the *baseline's* settings (batch 2, 60 epochs) so the
numbers are comparable to 0.5477. ≈8 h each at the measured 477 s/epoch.

- **1a — 4 terms, aggregator unchanged.** `configs/prism.py`: new `PrismConfig`
  defaults with the deleted weights at 0 and the four merged groups exposed as four
  weights. `core/objective.py`: the `add(name, value, weight)` helper already makes
  a term inert at `weight == 0.0`, so removal is a config change first and a code
  deletion second — in that order, so Stage 1 is reversible and 1a is a pure
  term-count claim.
- **1b — 4 terms + the aggregator fix** chosen by Stage 0e. This is the ablation row
  for the finding above, and it is the only row where the two are separable.

**Kill criterion:** if 1a scores materially below 0.5477, a removed term earned its
place. It goes back **with the ablation row that proves it**, which is a better paper
than removing it. Either outcome is publishable; that is the point of running it.

### Stage 2 — the classifier (this is where the mIoU is)

Targets `field` 0.1247, `sand` 0.3374, `bare soil` 0.3792, `mobile home` 0.4380
(read off `PRISM_best_eval.log`'s `per_class_IoU`, which sums to 0.5477).
Headroom: those four at 0.70 → **mIoU 0.6372**.

- **One deliberate per-class logit, replacing two accidental ones.** The model today
  applies a per-class additive logit from the aggregator (Stage 1b) *and* a per-class
  loss reweighting from `rare_class_factor=4.0`, neither derived from anything.
  Replace both with a single logit adjustment by class prior, applied at the decision
  rather than as an extra loss term. Targets signature (B) directly; adds no term to
  the objective. This is the item with the clearest mechanism and the clearest
  ablation row.
- **Per-image class restriction** at whatever strength Stage 0c/0d justify.
- **Not `embed_dim`.** The earlier hypothesis is dead: 68 unit vectors fit in R^64 at
  worst-pair cosine 0.031 (Welch), the model sits at +0.440, and
  `model/decoder_v2.py:75` is `nn.Conv2d(64, embed_dim, 1)` — the trunk is already
  64 channels, so widening `embed_dim` alone adds no rank at all. If capacity is ever
  the suspect, the literal `64` repeated at `model/decoder_v2.py:73-78` is what has
  to move, and 81.9% of the trainable decoder is at 64×64 (`coarse` 44.1% +
  `context` 37.8%) while every head together is 0.8% — so capacity is not where the
  parameters are missing.
- **Context at the decision step is the one architectural gap worth naming.** The
  classifier is 1×1 at 256×256, so nothing spatial happens where the class is
  chosen; the decoder's trained reach is ≈88–90 px (35% of the tile) via the dilated
  pair, and the only full-resolution evidence is 32 stem channels at ≈23 px. `dock`
  vs `ship` and `sand` vs `pavement` are object-vs-material calls that need more —
  and the mechanisms that supply it are the region vote (Stage 0b) and `L_region`,
  not a wider embedding.

**Decides:** whether the misclassification story has a fix or is a data limit.
`field` at 8 val images may simply be unlearnable and should then be *reported* as
such, not engineered around.

### Stage 3 — confidence-aware self-training, class-normalised

The user's proposal, checked and then narrowed to the only form that cannot backfire.
Confidence *gating* already exists (`gate_kappa`, `gate_floor`, `self_min_margin`,
`tau = mu − kappa·sigma` over an EMA of per-region confidences) and the run logs
`accept=0.8138` — it passes ~81% of already-filtered candidates, so it is nominally
confidence-aware and in practice barely filtering. Confidence *weighting* does not
exist: `conf` and `margin` are computed at `core/regions.py:292-293`, used only in
boolean comparisons at line 305, then collapsed to a hard label or −1 at line 307,
after which `hard_ce` averages uniformly. `core/shadow.py:224-247` already implements
the weighting pattern (`w = w * pc.max(1).values`) for `L_sh`, so the idiom is in the
codebase and unused here. (`core/pseudo.py` holds an unwired per-pixel / percentile /
per-class-threshold stack and is dead in the PRISM path — only `train.py:35` imports
it. Reuse the ideas, not the file.)

**Plain self-training entrenches both signatures:** a model that confidently calls
`bare soil` `grass` will pseudo-label it `grass` and train on that. The fix is
*per-class adaptive thresholds* — a higher bar for over-predicted classes, lower for
under-predicted — which replaces three scalar knobs with one class-aware rule. A
simplification, not an addition, and it composes with Stage 2's logit adjustment
because both are statements about the same per-class bias.

### Stage 4 — shadow

`prism-v8-shadow-on` is live (PID 77901, **epoch 10 of 40 done**, now 964 s/epoch), and
per the GPU schedule above its epoch-10 checkpoint already exists, so it is stopped for
Stage 0 and resumed from there; `prism-v8-shadow-off` follows at ≈4 h. It is the first
matched pair in the project's history and the first time `L_shadow` has executed at all
(`shadow` 0.424 → 0.272 → 0.234 over epochs 4–9, so it is genuinely training). Both rows
run at `--batch-size 1`, so **only treatment-minus-control is readable** — do not compare
either to 0.5477, which was batch 2. Target to beat: the **1.254** shadow error ratio
measured with the shadow terms off. Keep or drop `L_shadow` by that number.

Caveat already visible: at epoch 5 the shadow-on row is at mIoU 0.4289 with
`val_ghost` **0.3376** against the 0.5477 model's 0.2263 — early, and at batch 1, so
not yet evidence either way, but it is the number to watch.

### Stage 5 — boundary, last

0.0210 of all error. Only if Stages 1-3 land, and only to reclaim `potts`/`bnd` if
Stage 1 showed they were carrying something.

---

## Files to modify

| file | change |
|---|---|
| `e3_only/v8-plan.md` | **new** — this plan, as the user asked, committed in-repo |
| `e3_only/evaluate_prism.py` | **Stage 0a, first:** `raise` when the presence head is absent and `--presence-gate > 0` (the condition at `:338-350` is stated and not enforced); later, the class-balanced decision |
| `e3_only/dataset_prism.py` | **Stage 0a:** `raise` instead of `n_sam = 0` at `:177-180`, which silently turns `--region-vote` into plain argmax |
| `e3_only/tools/oracle_inventory.py` | **new, Stage 0d** — GT-inventory-restricted argmax; measurement-only, modelled line-for-line on `tools/oracle_partition.py` |
| `e3_only/tools/proto_geometry.py` | **new, Stage 0e** — per-class within-class prototype cosine and realised `aggregate − max` excess, from a checkpoint + one val pass |
| `e3_only/core/protobank.py` | **Stage 1b:** the aggregator (`k_temperature`, hard max, or per-class `K_c`) at `:145-148` |
| `e3_only/configs/prism.py` | 4-term defaults; drop `w_rim`/`w_area`; logit-adjustment field; rewrite `ablation()` to the new row set (`embed_dim` sweep **dropped** — refuted) |
| `e3_only/core/objective.py` | merge `prop`+`hom`+`self` → `L_region`, `absent`+`present`+`pres_head` → `L_inventory`; delete `rim`/`area`; keep the `add()` zero-weight short-circuit |
| `e3_only/core/regions.py` | **Stage 3:** per-class thresholds; `conf`/`margin` at `:292-293` currently computed and discarded |
| `e3_only/METHOD.md` | rewrite §4 as four mechanisms; §8 gains a **results** column; new subsection for the aggregator term; **Stage C:** the reproduce recipe at `:1089` gains a "regenerate the E3 predictions from `E3_epoch_0030.pt`" first step |
| `e3_only/run_queue_v8.sh` | append the Stage-0/1 rows behind the running pair; `eval_four` at `:58-65` is written and has never fired |
| `e3_only/_archive/README.md` | **Stage C:** rewrite from "nothing was deleted, every directory was moved" to a manifest — what was deleted, its measured score, and the command that regenerates it |
| deleted in Stage C | `_archive/smoke/`, `_archive/prism-superseded/**/*.pt`, `_archive/rejected-fixed-version/eval_predictions/`, `_archive/e3-baseline/eval_predictions/`, 8 of 10 `_archive/e3-baseline/checkpoints/*.pt` (after 0f), `runs/prism-no-shadow/*.pt`, `e3_only/runs/prism-v6-lora12/*.pt` + its predictions, `e3_only/runs/prism-v5-corrected/` periodic `*.pt` + `PRISM_best_predictions/`, and the seven orphans (`_diag_floor.py`, `data/train_fixed.json`, `data/smoke.json`, `run_evals.sh`, `run_queue_v7.sh`, `__pycache__/`, the `e3_only/e3_only/e3_only/` stub) |

Reuse rather than rebuild: `tools/oracle_partition.py` (the 0.9438 shape ceiling *is*
the region-vote ceiling — same SAM-only, `min_region=24` rule), `tools/diagnose_failures.py
--confusion-csv` (six-issue table, per-class recall/precision), `evaluate_prism.py::_region_vote`,
`core/inventory.py::apply_presence_gate`, `MultiPrototypeClassifier.report()` (already
prints `intra_cos`), `core/shadow.py:224-247` (the confidence-weighting idiom),
`PrismNet.checkpointed()`, `configs/prism.py::ARCH_FIELDS` + `config_from_checkpoint`.

## Verification

1. **Integrity guard first, every time.** A checkpoint must hold 132 trainable
   tensors including 96 LoRA (`sam.image_encoder.blocks.N.*.{A,B}`). A rank-8
   checkpoint is ~11.9 MB; ~7.1 MB means the adapter was dropped and the score is
   meaningless. `evaluate_prism.py:336-347` already raises on this — do not weaken it,
   and note it exempts `decoder.presence.*`, which is exactly the hole Stage 0a closes.
2. **Offline must reproduce in-training.** Re-evaluate the saved checkpoint and
   confirm it matches the training-time mIoU, as `prism-v5-corrected` does at 0.5477.
   This is the check that caught the LoRA bug.
3. **The aggregator change must be argmax-neutral where it should be.** Subtracting a
   uniform `t·log K` changes nothing; only a change that alters the *spread*
   dependence (lower `t`, hard max, per-class `K_c`) can move a prediction. Assert
   this on a fixed batch before spending 8 h of GPU on it.
4. **Per-stage:** `evaluate_prism.py` for mIoU/PA, then `tools/diagnose_failures.py
   --confusion-csv` on the saved predictions, compared against the Stage-0 row. A
   stage that raises mIoU while worsening its target issue has not worked.
5. **The four weak classes are the scoreboard**, not just mIoU: report `field`,
   `sand`, `bare soil`, `mobile home` IoU every stage, plus `intra_cos` per class once
   Stage 0e exists.
6. `python -m compileall -q e3_only` must return 0; dense masks
   (`dlrsd/train_1cmasks`, `data/val_masks_remapped`) stay measurement-only and are
   never read by training code — the two new oracle/geometry tools are measurement
   tools and must be labelled as such in their own docstrings.
7. **Stage C is verified by re-running Stage 0, not by inspection.** After the
   deletions: `git status --porcelain` in `PRISM/` must show no new deletion of a
   *tracked* file (everything removed is untracked); `du -sh PRISM/` must report
   ≈3.1 GB against 17 GB before; `df -h /` must show ≈24 GB free; and Stage 0a–0e must
   still run to the same numbers, which is the real test that the keep-list was right.
   Delete in tier order and never delete a checkpoint whose eval log is not already
   beside it — the log is the number, the checkpoint is only a way to recompute it.
