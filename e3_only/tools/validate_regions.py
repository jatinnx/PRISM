"""Measure whether the region prior is trustworthy, before training on it.

Three numbers decide whether the design is sound. All are computed against the
dense training masks, which are used HERE FOR MEASUREMENT ONLY and are never
read by the training code:

  1. REGION HOMOGENEITY (the ceiling).  For each region, the share of its
     pixels belonging to its own majority GT class. The area-weighted mean is
     the best mIoU-relevant accuracy any region-constant labelling can reach.
     If this is low, region-constant regularisation is harmful and the design
     must be abandoned. If it is high, forcing homogeneity inside regions is
     nearly free and kills salt-and-pepper noise by construction.

  2. PROPAGATION COVERAGE (the density win).  Share of pixels that receive a
     label from a region containing points of exactly one class. This is how
     many supervised pixels ~15 clicks actually buy.

  3. PROPAGATION PURITY (the noise level).  Of those pixels, the share whose
     propagated label matches GT. 1 - purity is the label noise the loss must
     be made robust to, and it fixes the label-smoothing constant.

A nearest-point Voronoi propagation is reported alongside as the control: it
has 100% coverage, so if region propagation is not markedly more accurate the
regions add nothing.

Usage:
    python -m e3_only.tools.validate_regions [--regions artifacts/regions_train.npz]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.configs.base import Config, resolve                 # noqa: E402
from e3_only.core.regions import propagate_points_np             # noqa: E402
from e3_only.data.class_map import CLASS_NAMES, NUM_CLASSES      # noqa: E402

DENSE_TRAIN_MASKS = Path("dlrsd/train_1cmasks")


def voronoi_labels(points, shape):
    """Nearest-labelled-point class for every pixel (the control baseline)."""
    h, w = shape
    if not len(points):
        return np.full((h, w), -1, np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    best_d = np.full((h, w), np.inf, np.float32)
    out = np.full((h, w), -1, np.int16)
    for x, y, c in points:
        d = (xx - x) ** 2 + (yy - y) ** 2
        upd = d < best_d
        best_d[upd] = d[upd]
        out[upd] = int(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="artifacts/regions_train.npz")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-region", type=int, default=24,
                    help="regions smaller than this are ignored in the homogeneity "
                         "stats. Defaults to PrismConfig.min_region, so the measured "
                         "ceiling is the ceiling for the regions L_hom actually acts on")
    args = ap.parse_args()

    cfg = Config()
    items = json.loads(Path(resolve(args.manifest or cfg.train_manifest)).read_text())
    z = np.load(resolve(args.regions), allow_pickle=True)
    rid = {str(k): i for i, k in enumerate(z["ids"])}
    regions_all = z["regions"]
    n_sam_all = z["n_sam"]
    if args.limit:
        items = items[:args.limit]

    # 1. homogeneity
    hom_px_correct = 0
    hom_px_total = 0
    hom_sam_correct = 0
    hom_sam_total = 0
    hom_fill_correct = 0
    hom_fill_total = 0
    region_sizes = []
    per_region_purity = []

    # 2/3. propagation
    prop_cov_px = 0
    prop_hit_px = 0
    conflict_regions = 0
    single_regions = 0
    empty_regions = 0
    err_pair_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    prop_cov_per_class = defaultdict(int)
    prop_hit_per_class = defaultdict(int)

    # control
    vor_hit_px = 0
    vor_total_px = 0

    valid_px = 0                    # all GT-valid pixels, the honest denominator
    n = 0
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
        n_sam = int(n_sam_all[rid[key]])
        h, w = region.shape
        if dense.shape != (h, w):
            dense = cv2.resize(dense, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = dense.astype(np.int32) - 1
        valid = (gt >= 0) & (gt < NUM_CLASSES)
        n += 1
        valid_px += int(valid.sum())

        pts = np.asarray(it.get("points", []), dtype=np.float32).reshape(-1, 3)
        # scaled exactly the way data/dataset_prism.py scales them -- unrounded and
        # unclamped -- so a point that falls outside the frame is DROPPED here as
        # it is there, rather than being clamped onto the border and counted
        pts_scaled = np.empty((len(pts), 3), np.float32)
        if len(pts):
            pts_scaled[:, 0] = pts[:, 0] * (w / float(it.get("width", w)))
            pts_scaled[:, 1] = pts[:, 1] * (h / float(it.get("height", h)))
            pts_scaled[:, 2] = pts[:, 2]

        nreg = int(region.max()) + 1
        # per-region GT histogram, vectorised
        flat_r = region.ravel()
        flat_g = gt.ravel()
        m = valid.ravel() & (flat_r >= 0)      # -1 would silently index hist[-1]
        hist = np.zeros((max(nreg, 1), NUM_CLASSES), np.int64)
        np.add.at(hist, (flat_r[m], flat_g[m]), 1)
        sizes = hist.sum(1)
        majc = hist.max(1)

        big = sizes >= args.min_region
        hom_px_correct += int(majc[big].sum())
        hom_px_total += int(sizes[big].sum())
        is_sam = np.arange(len(sizes)) < n_sam
        hom_sam_correct += int(majc[big & is_sam].sum())
        hom_sam_total += int(sizes[big & is_sam].sum())
        hom_fill_correct += int(majc[big & ~is_sam].sum())
        hom_fill_total += int(sizes[big & ~is_sam].sum())
        region_sizes.extend(sizes[big].tolist())
        per_region_purity.extend((majc[big] / np.maximum(1, sizes[big])).tolist())

        # point -> region propagation, through the SAME function the dataloader
        # calls. Reimplementing it here would let the measured purity drift away
        # from the purity the loss actually sees, which is the one number this
        # tool exists to produce. (Conflict regions come back as -1, so they are
        # excluded from ``covered`` below exactly as they are from L_prop.)
        prop16, _conflict = propagate_points_np(region, pts_scaled, NUM_CLASSES)
        prop = prop16.astype(np.int32)
        classes_in_region = defaultdict(set)
        for x, y, c in pts_scaled.tolist():
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            r = int(region[yi, xi])
            if r >= 0:
                classes_in_region[r].add(int(c) % NUM_CLASSES)
        n_single = sum(1 for cs in classes_in_region.values() if len(cs) == 1)
        single_regions += n_single
        conflict_regions += len(classes_in_region) - n_single
        # only count real regions as unpointed: ids are dense in [0, nreg) by
        # construction in build_region_cache, but a resize can drop one
        present_ids = np.unique(region[region >= 0])
        empty_regions += len(present_ids) - len(classes_in_region)
        covered = (prop >= 0) & valid
        prop_cov_px += int(covered.sum())
        hits = covered & (prop == gt)
        prop_hit_px += int(hits.sum())
        for c in np.unique(prop[covered]):
            sel = covered & (prop == c)
            prop_cov_per_class[int(c)] += int(sel.sum())
            prop_hit_per_class[int(c)] += int((sel & (gt == c)).sum())
        bad = covered & (prop != gt)
        if bad.any():
            # vectorised: a python loop over every mismatching pixel is tens of
            # millions of iterations across the split
            gv = gt[bad].ravel().astype(np.int64)
            pv = prop[bad].ravel().astype(np.int64)
            np.add.at(err_pair_mat, (gv, pv), 1)

        vor = voronoi_labels(pts_scaled, (h, w))
        vsel = (vor >= 0) & valid
        vor_total_px += int(vsel.sum())
        vor_hit_px += int((vsel & (vor == gt)).sum())

    if not n:
        print("no images matched — build the region cache first")
        return

    print(f"images matched: {n}")
    print(f"\n[1] REGION HOMOGENEITY  (ceiling for region-constant labelling)")
    print(f"    area-weighted purity, all regions >= {args.min_region}px : "
          f"{hom_px_correct / max(1, hom_px_total):.4f}")
    print(f"      SAM regions    : {hom_sam_correct / max(1, hom_sam_total):.4f} "
          f"({hom_sam_total / max(1, valid_px):.3f} of pixels)")
    print(f"      filler regions : {hom_fill_correct / max(1, hom_fill_total):.4f} "
          f"({hom_fill_total / max(1, valid_px):.3f} of pixels)")
    print(f"      (regions < {args.min_region}px hold the remaining "
          f"{1 - hom_px_total / max(1, valid_px):.3f} of pixels and are excluded "
          f"from L_hom too)")
    pr = np.array(per_region_purity)
    print(f"    per-region purity: mean {pr.mean():.4f}  median {np.median(pr):.4f}  "
          f"share >=0.90: {(pr >= 0.90).mean():.4f}  share <0.60: {(pr < 0.60).mean():.4f}")
    print(f"    region size: mean {np.mean(region_sizes):.0f}px  "
          f"median {np.median(region_sizes):.0f}px  ({len(region_sizes) / n:.1f} usable regions/img)")

    print(f"\n[2] PROPAGATION COVERAGE")
    print(f"    pixels labelled by single-class point regions: "
          f"{prop_cov_px / max(1, valid_px):.4f} of all GT-valid pixels  "
          f"({prop_cov_px / n:.0f} px/img vs ~{sum(len(i.get('points', [])) for i in items) / n:.0f} clicked)")
    print(f"    => each click buys ~"
          f"{prop_cov_px / max(1, sum(len(i.get('points', [])) for i in items)):.0f} supervised pixels")
    print(f"    regions: {single_regions / n:.1f} single-class, {conflict_regions / n:.1f} conflicting, "
          f"{empty_regions / n:.1f} unpointed  per image")

    print(f"\n[3] PROPAGATION PURITY  (the label noise the loss must tolerate)")
    pur = prop_hit_px / max(1, prop_cov_px)
    vor = vor_hit_px / max(1, vor_total_px)
    print(f"    region propagation : {pur:.4f}   -> set prop_eps = {1 - pur:.3f}")
    print(f"    voronoi control    : {vor:.4f}  (coverage 1.0000)")
    print(f"    => regions are {pur - vor:+.4f} more accurate than "
          f"nearest-point at {prop_cov_px / max(1, valid_px):.2f} coverage")
    if pur <= vor:
        print("    !! propagation does NOT beat the Voronoi control. The partition is "
              "adding nothing;\n       the design should be reconsidered rather than tuned.")
    print(f"\n    per-class propagation purity:")
    for c in sorted(prop_cov_per_class, key=lambda k: -prop_cov_per_class[k]):
        print(f"      {CLASS_NAMES[c]:<12s} {prop_hit_per_class[c] / max(1, prop_cov_per_class[c]):.3f}  "
              f"({prop_cov_per_class[c] / max(1, n):.0f} px/img)")
    print(f"\n    top propagation errors (GT -> propagated):")
    prop_err_total = max(1, prop_cov_px - prop_hit_px)
    order = np.dstack(np.unravel_index(np.argsort(-err_pair_mat.ravel()),
                                       err_pair_mat.shape))[0]
    for a, b in order[:10]:
        k = int(err_pair_mat[a, b])
        if k == 0:
            break
        print(f"      {CLASS_NAMES[a]:<12s} -> {CLASS_NAMES[b]:<12s} "
              f"{k / prop_err_total:.3f} of propagation error")


if __name__ == "__main__":
    main()
