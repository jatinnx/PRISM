"""Measure the assumptions the new loss design rests on — before building it.

Two questions, answered on the 630 training images against their dense masks
(used HERE for measurement only; the training signal never sees them):

  Q1. INVENTORY ASSUMPTION.  Chakraborty's protocol annotates 3-5 pixels for
      every class present in an image. If that holds, the class support of the
      point set equals the class support of the dense mask — which means a
      class with NO points in an image is absent from that image, and can be
      suppressed at all 65 536 pixels. That converts ~15 point labels into a
      dense negative constraint. This script measures the violation rate, both
      per image and weighted by the pixel area at risk, so the loss can be
      designed around the real noise level rather than an assumption.

  Q2. CLASS CO-OCCURRENCE.  How many classes appear per image, i.e. how much
      of the 17-way decision the inventory constraint actually removes.

Usage:
    python -m e3_only.tools.validate_inventory
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PRISM.e3_only.configs.base import resolve                      # noqa: E402
from PRISM.e3_only.data.class_map import CLASS_NAMES, NUM_CLASSES    # noqa: E402

DLRSD = Path("/home/cse-sdpl/Downloads/point_only_semseg/dlrsd")
DENSE_TRAIN_MASKS = DLRSD / "train_1cmasks"


def main(manifest="/home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/data/train.json"):
    items = json.loads(Path(resolve(manifest)).read_text())
    print(f"manifest: {manifest}\nimages: {len(items)}")

    n = 0
    n_clean = 0
    missing_per_class = Counter()          # GT-present but no point
    missing_area_per_class = defaultdict(float)
    ghost_per_class = Counter()            # point-present but GT-absent (should be 0)
    n_gt_classes = []
    n_pt_classes = []
    total_px = 0
    at_risk_px = 0                         # pixels of GT classes with no point
    per_image_missing = Counter()

    for it in items:
        mp = DENSE_TRAIN_MASKS / (Path(it["image"]).stem + ".png")
        if not mp.exists():
            continue
        dense = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if dense is None:
            continue
        gt = dense.astype(np.int32) - 1                       # 1..17 -> 0..16
        h, w = gt.shape
        n += 1
        total_px += h * w

        gt_classes = set(int(c) for c in np.unique(gt) if 0 <= c < NUM_CLASSES)
        pt_classes = set(int(c) for _, _, c in it.get("points", []))
        n_gt_classes.append(len(gt_classes))
        n_pt_classes.append(len(pt_classes))

        missing = gt_classes - pt_classes
        ghosts = pt_classes - gt_classes
        per_image_missing[len(missing)] += 1
        if not missing and not ghosts:
            n_clean += 1
        for c in missing:
            a = int((gt == c).sum())
            missing_per_class[c] += 1
            missing_area_per_class[c] += a
            at_risk_px += a
        for c in ghosts:
            ghost_per_class[c] += 1

    print(f"\nimages with a dense mask available: {n}")
    if n == 0:
        raise SystemExit(
            f"no dense masks were found under {DENSE_TRAIN_MASKS}.\n"
            f"This tool measures the inventory assumption against the dense masks "
            f"(for measurement only -- training never reads them). Point it at the "
            f"right directory by editing DENSE_TRAIN_MASKS at the top of this file.\n"
            f"Until it has run, leave configs/prism.py's inventory_leak at its "
            f"conservative default rather than guessing a lower one.")
    print(f"classes per image  — GT: mean {np.mean(n_gt_classes):.2f} "
          f"(min {min(n_gt_classes)}, max {max(n_gt_classes)})"
          f"   points: mean {np.mean(n_pt_classes):.2f}")
    print(f"=> the inventory constraint removes on average "
          f"{NUM_CLASSES - np.mean(n_pt_classes):.1f} of {NUM_CLASSES} classes per image")

    print(f"\nQ1  images where point support == GT support exactly: "
          f"{n_clean}/{n} = {n_clean / max(1, n):.4f}")
    print("    distribution of #GT-classes-without-a-point per image:")
    for k in sorted(per_image_missing):
        print(f"      {k} missing: {per_image_missing[k]:>4d} images "
              f"({per_image_missing[k] / max(1, n):.3f})")
    print(f"\n    PIXEL RISK: share of all pixels belonging to a GT class that has "
          f"no point in its image: {at_risk_px / max(1, total_px):.4f}")
    print("    (this is the fraction of pixels an absent-class suppression term "
          "would push in the wrong direction)")

    print("\n    per class — how often the class is in the mask but has no point:")
    print(f"      {'class':<12s} {'#images':>8s} {'mean area when missing':>24s}")
    for c in sorted(missing_per_class, key=lambda k: -missing_area_per_class[k]):
        print(f"      {CLASS_NAMES[c]:<12s} {missing_per_class[c]:>8d} "
              f"{missing_area_per_class[c] / max(1, missing_per_class[c]):>24.0f} px")
    if ghost_per_class:
        print("\n    points for a class NOT in the dense mask (annotation noise / leak):")
        for c, k in ghost_per_class.most_common():
            print(f"      {CLASS_NAMES[c]:<12s} {k} images")
    else:
        print("\n    no class has points without being in the dense mask.")


if __name__ == "__main__":
    main(*sys.argv[1:])
