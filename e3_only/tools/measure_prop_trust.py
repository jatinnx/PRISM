"""Estimate a per-class propagation trust WITHOUT reading a dense mask.

Why this tool exists
--------------------
``L_prop`` smooths its target by a single scalar ``prop_eps`` -- the measured rate
at which a point-propagated label is wrong. One scalar is the right model only if
the noise is class-independent, and it is not: ``validate_regions.py`` reports a
propagation purity that ranges from ~0.99 on large homogeneous classes down to
~0.49 on thin ones that share their SAM region with a neighbour. Training on a
label that is right half the time with the same confidence as a label that is
right 99% of the time is the mechanism behind the largest confusion pair in the
eval (dock <-> ship).

The obvious fix -- copy the 17 per-class purities out of ``validate_regions.py``
-- would import 17 numbers measured from dense masks into the training loop, and
the whole point of this project is that training reads none. So the per-class
*shape* is estimated here from points and the frozen partition only, and only the
global *scale* comes from the single ``prop_eps`` constant that is already
declared, measured and disclosed.

The label-free signal
---------------------
A propagated label is wrong exactly when its region straddles a semantic
boundary. Straddling is observable without a mask: if a region contains points of
two different classes, ``propagate_points_np`` already flags it as a conflict and
drops it. For class c let

    n_c = # regions containing at least one class-c point
    k_c = # of those that also contain a point of some other class
    q_c = k_c / n_c

``q_c`` is P(straddle detected by a second point | region holds a c point). It
under-estimates the true straddle rate, because a straddling region is only
flagged when another class's ~5 grid points happen to land inside it, and that
detection probability is well below one. But the detection probability is a
property of the point sampler, not of the class, so it acts as a common factor:
``q_c`` is a biased estimate of the rate and an unbiased *ranking* of it.

So the rate is reconstructed by fixing the coverage-weighted mean to the measured
scalar:

    eps_c = clip( prop_eps * q_c / sum_c (w_c q_c),  eps_lo, eps_hi )

with ``w_c`` the share of propagated pixels carrying class c -- also label-free.
The result is a 17-vector whose *scale* carries exactly the one bit of
mask-derived information the method already declares, and whose *shape* carries
none.

Validation
----------
Run with ``--validate`` to print the GT-measured per-class purity beside the
estimate and their rank correlation. That path reads dense train masks and is
measurement only: nothing it prints is written into the artifact.

Usage:
    python -m e3_only.tools.measure_prop_trust                       # writes the artifact
    python -m e3_only.tools.measure_prop_trust --validate            # + GT check
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.configs.prism import PrismConfig, resolve              # noqa: E402
from e3_only.core.regions import propagate_points_np                # noqa: E402
from e3_only.data.class_map import CLASS_NAMES, NUM_CLASSES         # noqa: E402

DENSE_TRAIN_MASKS = Path("dlrsd/train_1cmasks")


def scale_points(item, h: int, w: int) -> np.ndarray:
    """Points in the cached partition's frame, scaled exactly as the dataloader does."""
    pts = np.asarray(item.get("points", []), dtype=np.float32).reshape(-1, 3)
    if not len(pts):
        return pts
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] * (w / float(item.get("width", w)))
    out[:, 1] = pts[:, 1] * (h / float(item.get("height", h)))
    out[:, 2] = pts[:, 2]
    return out


def measure(items, regions_all, n_sam_all, rid, num_classes: int, limit=None):
    """-> per-class conflict rate q, propagated-pixel coverage w, filler share."""
    if limit:
        items = items[:limit]
    n_reg = np.zeros(num_classes, np.int64)          # regions holding a c point
    k_reg = np.zeros(num_classes, np.int64)          # ... that also hold another class
    cov_px = np.zeros(num_classes, np.int64)         # propagated pixels labelled c
    fill_px = np.zeros(num_classes, np.int64)        # ... inside a filler region
    d_sum = np.zeros(num_classes, np.float64)        # summed detection probability
    n_img = 0
    for it in items:
        key = str(it.get("id"))
        if key not in rid:
            continue
        region = regions_all[rid[key]].astype(np.int32)
        n_sam = int(n_sam_all[rid[key]])
        h, w = region.shape
        pts = scale_points(it, h, w)
        n_img += 1

        seen = defaultdict(set)
        for x, y, c in pts.tolist():
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            r = int(region[yi, xi])
            if r >= 0:
                seen[r].add(int(c) % num_classes)
        n_pt = np.bincount(pts[:, 2].astype(np.int64) % num_classes,
                           minlength=num_classes) if len(pts) else np.zeros(num_classes, np.int64)
        area = np.bincount(region[region >= 0].ravel(),
                           minlength=int(region.max()) + 1 if region.max() >= 0 else 1)
        for r, cs in seen.items():
            multi = len(cs) > 1
            a_frac = float(area[r]) / float(h * w)
            for c in cs:
                n_reg[c] += 1
                if multi:
                    k_reg[c] += 1
                # P(a straddle of this region is DETECTED) = P(>=1 foreign point
                # lands in it). Poisson with lambda = area share x foreign count,
                # i.e. foreign points treated as uniform over the frame. They are
                # not -- they are grid-spread inside their own class -- so this is
                # an approximation, and it is the only approximation in the chain.
                lam = a_frac * float(n_pt.sum() - n_pt[c])
                d_sum[c] += 1.0 - float(np.exp(-lam))

        prop, _conf = propagate_points_np(region, pts, num_classes)
        prop = prop.astype(np.int32)
        covered = prop >= 0
        filler = region >= n_sam
        for c in np.unique(prop[covered]):
            sel = covered & (prop == c)
            cov_px[int(c)] += int(sel.sum())
            fill_px[int(c)] += int((sel & filler).sum())
    return n_reg, k_reg, cov_px, fill_px, d_sum, n_img


def estimate(n_reg, k_reg, cov_px, d_sum, prop_eps: float, eps_lo: float,
             eps_hi: float, debias: bool = True, d_floor: float = 0.05):
    """Per-class eps from the conflict ranking, scaled to the measured mean.

    ``debias`` divides the observed conflict rate by the probability that a
    straddle of that class's regions would have been *seen*. Without it a thin
    class is doubly penalised: its regions are small, so they rarely catch a
    foreign point, so its straddles go unrecorded and it looks reliable. With it
    the estimate is P(straddle), not P(straddle observed).
    """
    q_obs = np.where(n_reg > 0, k_reg / np.maximum(1, n_reg), 0.0)
    d = np.where(n_reg > 0, d_sum / np.maximum(1, n_reg), 1.0)
    q = q_obs / np.clip(d, d_floor, 1.0) if debias else q_obs
    q = np.clip(q, 0.0, 1.0)
    w = cov_px / max(1, cov_px.sum())
    denom = float((w * q).sum())
    if denom <= 0:
        return np.full(len(q), prop_eps, np.float64), q, w
    eps = prop_eps * q / denom
    # a class with no observed conflict is not noise-free, it is unmeasured: give
    # it the floor rather than zero, so its labels stay confident but not exact.
    eps = np.clip(eps, eps_lo, eps_hi)
    return eps, q, w, q_obs, d


def gt_purity(items, regions_all, rid, num_classes: int, limit=None):
    """MEASUREMENT ONLY: per-class propagation purity against the dense masks."""
    import cv2
    if limit:
        items = items[:limit]
    hit = np.zeros(num_classes, np.int64)
    tot = np.zeros(num_classes, np.int64)
    for it in items:
        key = str(it.get("id"))
        if key not in rid:
            continue
        mp = DENSE_TRAIN_MASKS / (Path(it["image"]).stem + ".png")
        if not mp.exists():
            continue
        dense = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if dense is None:
            continue
        region = regions_all[rid[key]].astype(np.int32)
        h, w = region.shape
        if dense.shape != (h, w):
            dense = cv2.resize(dense, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = dense.astype(np.int32) - 1
        valid = (gt >= 0) & (gt < num_classes)
        prop, _ = propagate_points_np(region, scale_points(it, h, w), num_classes)
        prop = prop.astype(np.int32)
        covered = (prop >= 0) & valid
        for c in np.unique(prop[covered]):
            sel = covered & (prop == c)
            tot[int(c)] += int(sel.sum())
            hit[int(c)] += int((sel & (gt == c)).sum())
    return hit, tot


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency."""
    def rank(x):
        o = np.argsort(x)
        r = np.empty(len(x), np.float64)
        r[o] = np.arange(len(x), dtype=np.float64)
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    cfg = PrismConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=cfg.region_cache_train)
    ap.add_argument("--manifest", default=cfg.train_manifest)
    ap.add_argument("--out", default="artifacts/prop_trust.json")
    ap.add_argument("--prop-eps", type=float, default=cfg.prop_eps,
                    help="the measured scalar the per-class vector is scaled to")
    ap.add_argument("--eps-lo", type=float, default=0.01)
    ap.add_argument("--eps-hi", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-debias", action="store_true",
                    help="use the raw conflict rate without the detection-probability "
                         "correction (ablation for the estimator itself)")
    ap.add_argument("--validate", action="store_true",
                    help="also print GT-measured purity (measurement only, never written)")
    a = ap.parse_args()

    items = json.loads(Path(resolve(a.manifest)).read_text())
    z = np.load(resolve(a.regions), allow_pickle=True)
    rid = {str(k): i for i, k in enumerate(z["ids"])}
    regions_all, n_sam_all = z["regions"], z["n_sam"]

    n_reg, k_reg, cov_px, fill_px, d_sum, n_img = measure(
        items, regions_all, n_sam_all, rid, NUM_CLASSES, a.limit)
    eps, q, w, q_obs, det = estimate(n_reg, k_reg, cov_px, d_sum, a.prop_eps,
                                     a.eps_lo, a.eps_hi, not a.no_debias)

    print(f"images: {n_img}   propagated pixels: {cov_px.sum()}  "
          f"({cov_px.sum() / max(1, n_img):.0f} px/img)")
    print(f"scaled so the coverage-weighted mean eps equals prop_eps={a.prop_eps}\n")
    print(f"  {'class':<12s} {'q_obs':>6s} {'P(det)':>7s} {'q_c':>6s} "
          f"{'cov':>7s} {'filler':>7s} {'eps_c':>7s}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<12s} {q_obs[c]:6.3f} {det[c]:7.3f} {q[c]:6.3f} "
              f"{w[c]:7.4f} {fill_px[c] / max(1, cov_px[c]):7.3f} {eps[c]:7.3f}")
    print(f"\n  coverage-weighted mean eps: {float((w * eps).sum()):.4f}")

    if a.validate:
        hit, tot = gt_purity(items, regions_all, rid, NUM_CLASSES, a.limit)
        pur = np.where(tot > 0, hit / np.maximum(1, tot), np.nan)
        true_eps = 1.0 - pur
        ok = np.isfinite(true_eps)
        print("\n  VALIDATION (dense masks, measurement only -- not written):")
        print(f"  {'class':<12s} {'eps_c':>7s} {'1-purity':>9s} {'ratio':>7s}")
        for c in range(NUM_CLASSES):
            if not ok[c]:
                continue
            print(f"  {CLASS_NAMES[c]:<12s} {eps[c]:7.3f} {true_eps[c]:9.3f} "
                  f"{eps[c] / max(1e-6, true_eps[c]):7.2f}")
        print(f"\n  spearman(eps_c, 1-purity_c) = "
              f"{spearman(eps[ok], true_eps[ok]):+.3f}   over {int(ok.sum())} classes")
        print(f"  a positive correlation is the whole claim: the conflict rate ranks "
              f"propagation\n  reliability without ever reading a mask.")

    out = Path(resolve(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "point-region conflict frequency (label-free)",
        "manifest": str(a.manifest), "regions": str(a.regions),
        "images": int(n_img), "prop_eps_scale": float(a.prop_eps),
        "eps_lo": float(a.eps_lo), "eps_hi": float(a.eps_hi),
        "class_names": list(CLASS_NAMES),
        "debias": not a.no_debias,
        "conflict_rate_observed": [float(v) for v in q_obs],
        "detect_prob": [float(v) for v in det],
        "conflict_rate": [float(v) for v in q],
        "coverage": [float(v) for v in w],
        "filler_share": [float(fill_px[c] / max(1, cov_px[c])) for c in range(NUM_CLASSES)],
        "prop_eps_per_class": [float(v) for v in eps],
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
