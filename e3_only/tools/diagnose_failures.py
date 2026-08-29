"""Quantify the six named failure modes on a directory of saved predictions.

Reads the `<id>.png` / `<id>_gt.png` / `<id>_real.png` triples written by
``evaluate.py --save-preds`` (E3) or ``evaluate_prism.py --save-preds`` (PRISM),
inverts the palette back to class indices, and reports one number per failure
mode so design decisions rest on measurement rather than on looking at a handful
of images.

Comparability with evaluate_prism.py
------------------------------------
This tool measures the E3 *baseline* and evaluate_prism.py measures the PRISM
*result*, so any number that appears in both must be defined identically or the
before/after table is meaningless. Three are:

  SPECKLE  identical by construction (share of pixels disagreeing with their own
           5x5 mode; ones-kernel filter2D == boxFilter(normalize=False), and both
           break ties toward the lower class index).
  GHOST    evaluate_prism counts (image, class) PAIRS predicted over >=0.5% of
           the image with zero GT pixels, over all such pairs. This file used to
           count every pred-only class regardless of area, which is a strictly
           larger and differently normalised number. Both are reported now; the
           one labelled "eval-matching" is the one to quote.
  FLOOD    evaluate_prism flags an image when the largest predicted class covers
           >=1.5x the largest GT class. This file used an absolute rule
           (pred >70% while GT max <50%). Both are reported; again the
           eval-matching row is the one to quote.

Usage:
    python -m e3_only.tools.diagnose_failures <pred_dir> [--limit N]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.data.class_map import CLASS_NAMES, PALETTE  # noqa: E402

NC = len(CLASS_NAMES)

# groups of spectrally similar land cover (paper's own examples)
SIMILAR_GROUPS = [
    ("vegetation/soil", [1, 4, 7, 8, 15]),      # bare soil, chaparral, field, grass, trees
    ("bright surfaces", [1, 10, 11, 6]),        # bare soil, pavement, sand, dock
    ("water", [12, 16]),                        # sea, water
    ("built", [2, 9, 5]),                       # buildings, mobile home, court
]


def _lut():
    """Exact RGB -> class index lookup for the 17 palette colours."""
    table = {}
    for i, rgb in enumerate(PALETTE):
        table[(int(rgb[0]), int(rgb[1]), int(rgb[2]))] = i
    return table


LUT = _lut()


def rgb_to_index(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) RGB -> (H,W) int8 class index; unknown colours map to -1."""
    key = (rgb[:, :, 0].astype(np.int32) << 16 |
           rgb[:, :, 1].astype(np.int32) << 8 |
           rgb[:, :, 2].astype(np.int32))
    out = np.full(key.shape, -1, dtype=np.int16)
    for (r, g, b), i in LUT.items():
        out[key == (r << 16 | g << 8 | b)] = i
    return out


def mode_filter(lab: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-pixel majority class over a k x k window (vectorised, 17 classes)."""
    votes = np.empty((NC,) + lab.shape, dtype=np.uint8)
    ones = np.ones((k, k), np.float32)
    for c in range(NC):
        votes[c] = cv2.filter2D((lab == c).astype(np.float32), -1, ones,
                                borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    return votes.argmax(0).astype(np.int16)


def shadow_proxy(bgr: np.ndarray) -> np.ndarray:
    """Cast-shadow proxy with NO shadow labels: a pixel is shadow-like when it
    is much darker than its wide-neighbourhood illumination estimate while its
    chromaticity stays comparable (cast shadow attenuates intensity, not hue).

    Returns a bool mask. Deliberately conservative — used only to compare error
    rates inside vs outside, so precision matters more than recall.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    illum = cv2.GaussianBlur(L, (0, 0), 15.0)
    ratio = L / np.maximum(illum, 1e-3)
    dark = L < np.percentile(L, 35)
    return (ratio < 0.80) & dark


def analyse(pred_dir: Path, limit=None):
    ids = sorted(p.stem for p in pred_dir.glob("*_gt.png"))
    ids = [i[:-3] for i in ids]                       # strip the "_gt"
    if limit:
        ids = ids[:limit]

    conf = np.zeros((NC, NC), dtype=np.int64)
    n = 0
    speckle_px = 0
    speckle_fixable = 0
    total_px = 0
    total_err = 0
    ghost_classes = 0
    ghost_err_px = 0
    ghost_num = ghost_den = 0          # evaluate_prism's definition (see module docstring)
    flooded = 0
    flood_eval = 0                     # evaluate_prism's definition
    shape_ok_label_wrong_regions = 0
    total_regions = 0
    shape_ok_label_wrong_px = 0
    err_in_shadow = 0
    px_in_shadow = 0
    err_out_shadow = 0
    px_out_shadow = 0
    missing_gt_classes = 0
    gt_classes_total = 0

    for image_id in ids:
        p_png = pred_dir / f"{image_id}.png"
        g_png = pred_dir / f"{image_id}_gt.png"
        r_png = pred_dir / f"{image_id}_real.png"
        if not (p_png.exists() and g_png.exists()):
            continue
        pred = rgb_to_index(cv2.cvtColor(cv2.imread(str(p_png)), cv2.COLOR_BGR2RGB))
        gt = rgb_to_index(cv2.cvtColor(cv2.imread(str(g_png)), cv2.COLOR_BGR2RGB))
        ok = (gt >= 0) & (pred >= 0)
        if not ok.any():
            continue
        n += 1
        err = ok & (pred != gt)
        total_px += int(ok.sum())
        total_err += int(err.sum())

        # 1. salt-and-pepper: pixel disagrees with its own 5x5 majority
        maj = mode_filter(pred, 5)
        isolated = ok & (pred != maj)
        speckle_px += int(isolated.sum())
        speckle_fixable += int((err & (maj == gt)).sum())

        # 2. ghost classes: predicted somewhere, absent from GT entirely
        gt_present = set(np.unique(gt[gt >= 0]).tolist())
        pr_present = set(np.unique(pred[pred >= 0]).tolist())
        ghosts = pr_present - gt_present
        ghost_classes += len(ghosts)
        if ghosts:
            ghost_err_px += int(np.isin(pred, list(ghosts)).sum())
        missing_gt_classes += len(gt_present - pr_present)
        gt_classes_total += len(gt_present)

        # the same quantity evaluate_prism reports as GHOST, so the before/after
        # table compares like with like: (image, class) pairs predicted over
        # >=0.5% of the image, and how many of those have zero GT pixels
        n_ok = int(ok.sum())
        pr_area = np.bincount(pred[ok].ravel(), minlength=NC) / float(n_ok)
        gt_area = np.bincount(gt[ok].ravel(), minlength=NC) / float(n_ok)
        for c in range(NC):
            if pr_area[c] < 0.005:
                continue
            ghost_den += 1
            if c not in gt_present:
                ghost_num += 1

        # 3. dominant-class flooding
        pr_frac = max((pred == c).mean() for c in pr_present)
        gt_frac = max((gt == c).mean() for c in gt_present)
        if pr_frac > 0.70 and gt_frac < 0.50:
            flooded += 1
        if pr_area.max() >= 1.5 * max(gt_area.max(), 1e-6):
            flood_eval += 1

        # 4. shadow-conditioned error rate
        if r_png.exists():
            sh = shadow_proxy(cv2.imread(str(r_png)))
            sh &= ok
            nsh = ok & ~sh
            px_in_shadow += int(sh.sum())
            err_in_shadow += int((err & sh).sum())
            px_out_shadow += int(nsh.sum())
            err_out_shadow += int((err & nsh).sum())

        # 5. right shape, wrong label: a GT region predicted homogeneously as a
        #    single WRONG class (SAM found the object, the classifier missed it)
        for c in gt_present:
            m = (gt == c).astype(np.uint8)
            ncomp, lab_cc = cv2.connectedComponents(m)
            for k in range(1, ncomp):
                region = lab_cc == k
                area = int(region.sum())
                if area < 200:                       # ignore slivers
                    continue
                total_regions += 1
                vals, cnts = np.unique(pred[region], return_counts=True)
                top = int(vals[cnts.argmax()])
                purity = float(cnts.max() / area)
                if top != c and purity > 0.80:
                    shape_ok_label_wrong_regions += 1
                    shape_ok_label_wrong_px += area

        # confusion matrix
        gv = gt[ok].ravel()
        pv = pred[ok].ravel()
        np.add.at(conf, (gv, pv), 1)

    # ---- report ----
    print(f"images analysed: {n}")
    print(f"overall pixel error rate: {total_err / max(1, total_px):.4f}")
    print()
    print("[1] salt-and-pepper")
    print(f"    pixels disagreeing with own 5x5 majority : {speckle_px / max(1, total_px):.4f}")
    print(f"    share of ALL errors a 5x5 majority filter would fix: "
          f"{speckle_fixable / max(1, total_err):.4f}")
    print()
    print("[2] ghost-class hallucination")
    print(f"    GHOST (eval-matching): {ghost_num}/{ghost_den} = "
          f"{ghost_num / max(1, ghost_den):.4f}   <- quote this one")
    print(f"    ghost classes per image (any area)        : {ghost_classes / max(1, n):.2f}")
    print(f"    share of all errors from ghost classes    : "
          f"{ghost_err_px / max(1, total_err):.4f}")
    print(f"    GT classes MISSED entirely per image      : {missing_gt_classes / max(1, n):.2f}"
          f"  ({missing_gt_classes}/{gt_classes_total} class-instances)")
    print()
    print("[3] dominant-class flooding")
    print(f"    FLOOD (eval-matching): {flood_eval}/{n} = {flood_eval / max(1, n):.4f}"
          f"   <- quote this one  (largest pred class >=1.5x largest GT class)")
    print(f"    images where one pred class >70% but GT max <50%: "
          f"{flooded}/{n} = {flooded / max(1, n):.4f}")
    print()
    print("[4] shadow mislabelling  (illumination-ratio proxy, no shadow labels)")
    r_in = err_in_shadow / max(1, px_in_shadow)
    r_out = err_out_shadow / max(1, px_out_shadow)
    print(f"    error rate inside shadow-like pixels : {r_in:.4f} "
          f"({px_in_shadow / max(1, total_px):.3f} of pixels)")
    print(f"    error rate elsewhere                 : {r_out:.4f}")
    print(f"    ratio (>1 means shadows are harder)  : {r_in / max(1e-9, r_out):.3f}")
    print()
    print("[5] correct shape, wrong label")
    print(f"    GT regions (>=200px) predicted >80% homogeneous but WRONG class: "
          f"{shape_ok_label_wrong_regions}/{total_regions} = "
          f"{shape_ok_label_wrong_regions / max(1, total_regions):.4f}")
    print(f"    share of all errors from those regions: "
          f"{shape_ok_label_wrong_px / max(1, total_err):.4f}")
    print()
    print("[6] spectral class confusion")
    offdiag = conf.copy()
    np.fill_diagonal(offdiag, 0)
    tot_off = offdiag.sum()
    for name, group in SIMILAR_GROUPS:
        m = 0
        for a in group:
            for b in group:
                if a != b:
                    m += offdiag[a, b]
        print(f"    within-group confusion '{name}': {m / max(1, tot_off):.4f} of all confusion")
    print("    top 12 confused (GT -> pred) pairs:")
    idx = np.dstack(np.unravel_index(np.argsort(-offdiag.ravel()), offdiag.shape))[0]
    for a, b in idx[:12]:
        print(f"      {CLASS_NAMES[a]:<12s} -> {CLASS_NAMES[b]:<12s} "
              f"{offdiag[a, b] / max(1, tot_off):.4f}  "
              f"({offdiag[a, b] / max(1, conf[a].sum()):.3f} of true {CLASS_NAMES[a]})")
    print()
    print("    per-class recall / precision:")
    for c in range(NC):
        tp = conf[c, c]
        rec = tp / max(1, conf[c].sum())
        prec = tp / max(1, conf[:, c].sum())
        print(f"      {CLASS_NAMES[c]:<12s} rec={rec:.3f} prec={prec:.3f} "
              f"gt_px={conf[c].sum():>9d} pred_px={conf[:, c].sum():>9d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    analyse(Path(a.pred_dir), a.limit)
