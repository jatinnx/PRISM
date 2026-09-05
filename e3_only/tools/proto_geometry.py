"""Per-class prototype geometry, and the size of the logit the aggregator adds.

MEASUREMENT ONLY. The geometry half reads nothing but the checkpoint; the
correlation half reads dense val masks to count ghost images, exactly as
tools/oracle_partition.py and tools/diagnose_failures.py do. Nothing here is
imported by training code and nothing here can be quoted as a result.

The thing being measured
------------------------
core/protobank.py's aggregator is

    aggregate_c = t * logsumexp_k(cos_ck / t)
                = max_k cos_ck  +  t * log SUM_k exp((cos_ck - max_k cos_ck)/t)
                  \\___________/     \\_________________________________________/
                   the cosine          an excess in [0, t*log K]

The excess is ZERO when a class's K prototypes disagree and t*log K when they have
collapsed onto one direction. It is a PER-CLASS ADDITIVE TERM IN THE LOGITS, so it
moves the argmax between classes, and no uniform normalisation removes it --
subtracting t*log K shifts all C classes equally. Divided by the point loss's
angular margin the learned scale cancels:

    t*log K / margin  =  0.20*ln 4 / 0.20  =  ln 4  =  1.386

so a collapsed class carries a standing advantage 39% larger than the whole margin
L_point works to establish. `intra_cos` in the training log is a scalar mean over
all 17 classes; the question this tool answers is whether the collapse is
CONCENTRATED on the classes that ghost.

What it prints
--------------
  from the checkpoint alone   per-class mean/max within-class prototype cosine,
                              nearest other class, and the excess each class's
                              geometry implies
  from one val pass           the REALISED excess per class, both over all pixels
                              and over the pixels where that class wins -- the
                              latter is where the excess actually changed a call
  correlation                 rank correlation of per-class collapse against
                              per-class ghost rate (images predicting the class
                              but not containing it / images predicting it)

Decision rule this feeds (v8-plan Stage 0e -> 1b): if the over-predicted classes
are the collapsed ones, the aggregator is a lever and Stage 1b runs it as an
ablation row. If not, it is a correctness fix worth one sentence and Stage 2's
class-prior logit adjustment carries the mIoU alone.

Usage:
    python -m e3_only.tools.proto_geometry --checkpoint <path> [--limit N] [--log path]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.configs.prism import PrismConfig, resolve                  # noqa: E402
from e3_only.data.class_map import CLASS_NAMES                          # noqa: E402
from e3_only.data.dataset_prism import PrismDataset, collate_prism      # noqa: E402
from e3_only.evaluate_prism import load_for_eval                        # noqa: E402

GHOST_AREA = 0.005          # same threshold evaluate_prism's GHOST metric uses


def rank(v):
    """Average ranks, so ties do not fake a correlation."""
    order = np.argsort(v)
    r = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--which", default="teacher")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    cfg = PrismConfig()
    model, cfg = load_for_eval(cfg, a.checkpoint, None, a.which, False, False, 0.0)
    device = next(model.parameters()).device
    clf = model.decoder.classifier
    C, K = clf.num_classes, clf.k
    t = max(clf.k_temperature, 1e-3)
    ceiling = t * math.log(K)

    # ---- from the checkpoint alone -------------------------------------- #
    w = F.normalize(clf.weight.detach(), dim=-1)                  # (C,K,D)
    g = torch.einsum("ckd,cjd->ckj", w, w)                        # within class
    eye = torch.eye(K, device=w.device, dtype=torch.bool)[None]
    intra_mean = g.masked_fill(eye, 0.0).sum((1, 2)) / max(1, K * (K - 1))
    intra_max = g.masked_fill(eye, -2.0).amax((1, 2)) if K > 1 else torch.zeros(C)
    flat = w.reshape(C * K, -1)
    inter = flat @ flat.T                                          # (CK,CK)
    same = (torch.arange(C * K, device=w.device) // K)
    inter = inter.masked_fill(same[:, None] == same[None, :], -2.0)
    inter_max = inter.amax(1).reshape(C, K).amax(1)                # nearest other class

    # the excess a class's geometry implies IF every pixel saw exactly these
    # cosines: t*log sum_k exp((cos_k - max)/t) with cos_k - max == the gram row
    # is not well defined without an embedding, so this is the two-prototype
    # bound using the mean within-class cosine -- an indication, not the realised
    # value, which the val pass measures.
    implied = t * torch.log1p((K - 1) * torch.exp((intra_mean - 1.0) / t))

    # ---- one val pass: the REALISED excess, and the ghost counts --------- #
    ds = PrismDataset(resolve(cfg.val_manifest), cfg.image_size, training=False,
                      region_npz=None, num_classes=C)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                    collate_fn=collate_prism)

    ex_sum = torch.zeros(C, dtype=torch.float64, device=device)
    ex_n = torch.zeros(C, dtype=torch.float64, device=device)
    exw_sum = torch.zeros(C, dtype=torch.float64, device=device)
    exw_n = torch.zeros(C, dtype=torch.float64, device=device)
    gt_img = np.zeros(C, np.int64)
    pred_img = np.zeros(C, np.int64)
    ghost_img = np.zeros(C, np.int64)
    n_done = 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if a.limit is not None and i >= a.limit:
                break
            image = batch["image_weak"].to(device, non_blocking=True)
            out = model(image)
            cos_k = clf.cosines(out["embed"])                       # (B,C,K,H,W)
            agg = clf.aggregate(cos_k)                              # (B,C,H,W)
            excess = (agg - cos_k.amax(2)).clamp_min(0.0)           # in [0, t*log K]
            pred = agg.argmax(1)                                    # (B,H,W)

            ex_sum += excess.sum((0, 2, 3)).double()
            ex_n += float(excess.shape[0] * excess.shape[2] * excess.shape[3])
            oh = F.one_hot(pred, C).permute(0, 3, 1, 2).to(excess.dtype)
            exw_sum += (excess * oh).sum((0, 2, 3)).double()
            exw_n += oh.sum((0, 2, 3)).double()

            n_done += 1
            p = pred[0].cpu().numpy()
            area = np.bincount(p.ravel(), minlength=C) / float(p.size)
            big = area >= GHOST_AREA
            pred_img += big
            if "mask" in batch:
                gt = batch["mask"][0].numpy().astype(np.int64)
                v = (gt >= 0) & (gt < C)
                if v.any():
                    has = np.zeros(C, bool)
                    has[np.unique(gt[v])] = True
                    gt_img += has
                    ghost_img += big & ~has

    ex = (ex_sum / ex_n.clamp_min(1)).cpu().numpy()
    exw = (exw_sum / exw_n.clamp_min(1)).cpu().numpy()
    ghost_rate = ghost_img / np.maximum(pred_img, 1)

    im = intra_mean.cpu().numpy()
    ix = intra_max.cpu().numpy()
    it = inter_max.cpu().numpy()
    imp = implied.cpu().numpy()

    L = [f"prototype geometry, {Path(a.checkpoint).name} which={a.which} "
         f"over {n_done} val images",
         f"  K={K}  t={t:.3f}  ceiling t*logK = {ceiling:.4f} cosine units "
         f"= {ceiling / 0.20:.3f}x the 0.20 point margin",
         f"  scale = {float(clf.scale):.2f}  =>  ceiling is {ceiling * float(clf.scale):.3f} "
         f"absolute logits against a margin of {0.20 * float(clf.scale):.3f}",
         f"  mean over classes: intra_cos {im.mean():+.4f}   realised excess "
         f"{ex.mean():.4f} ({ex.mean() / ceiling:.3f} of ceiling)",
         f"  K_eff = exp(excess/t) in [1, {K}] -- how many prototypes are "
         f"effectively active: mean {math.exp(ex.mean() / t):.2f}",
         "",
         "  NOTE on `implied`: it is the K-way bound built from the MEAN within-class",
         "  cosine, so it under-reads badly when ONE pair has collapsed and the rest",
         "  have not -- which is the finch_init duplicate signature. `max_pair` is the",
         "  column that predicts the realised excess; `implied` is kept only to show",
         "  that the mean is the wrong statistic here.",
         "",
         "  class         intra_cos  max_pair  near_other  implied  excess  K_eff  "
         "excess@win  gt_img  pred_img  ghost  ghost_rate"]
    for c, name in enumerate(CLASS_NAMES):
        L.append(f"  {name:<12s}  {im[c]:+8.4f}  {ix[c]:+8.4f}  {it[c]:+10.4f}  "
                 f"{imp[c]:7.4f}  {ex[c]:6.4f}  {math.exp(ex[c] / t):5.2f}  "
                 f"{exw[c]:10.4f}  "
                 f"{gt_img[c]:6d}  {pred_img[c]:8d}  {ghost_img[c]:5d}  "
                 f"{ghost_rate[c]:.4f}")

    pairs = [("intra_cos", im), ("realised excess", ex), ("excess@win", exw)]
    L += ["", "  rank correlation against per-class ghost_rate (n=17):"]
    for label, v in pairs:
        L.append(f"    {label:<18s} rho = {spearman(v, ghost_rate):+.4f}")
    L += [f"    {'gt_img (rarity)':<18s} rho = {spearman(gt_img, ghost_rate):+.4f}"
          f"   <- the null: rarity alone"]
    L += ["",
          "READ IT LIKE THIS: a strongly POSITIVE rho for excess@win, larger than the",
          "rarity null, means the aggregator's uncontrolled logit is concentrated on",
          "exactly the classes that ghost, and Stage 1b (k_temperature 0.20->0.05, a",
          "hard max_k, or per-class K_c) is a lever. If rarity alone explains the ghost",
          "rate and the excess does not, the aggregator is a correctness fix and the",
          "mIoU has to come from Stage 2's class-prior logit adjustment."]
    text = "\n".join(L)
    print(text)
    if a.log:
        p = Path(resolve(a.log))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
