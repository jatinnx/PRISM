"""Build the point-only manifests for DLRSD from Chakraborty's ACTUAL points.

Source of supervision: ``dlrsd/point_1cmasks/*.png`` — the Chakraborty paper's
real point annotations (630 files; pixel value 1..17 = labelled class, 18 =
unlabelled). These are interior-only by construction (0% boundary points) and
spatially well spread. We do NOT generate points from scratch — we extract the
paper's own points, capped at exactly ``POINTS_PER_CLASS`` per class via
grid-based spatial selection, so the comparison with Chakraborty et al. uses
the same supervision locations.

**Ship fix (2026-08-17):** the published point annotations contain ZERO ship
points even though ships exist in the training images (845,666 dense-GT ship
pixels across 31 images). A class with no positive example is structurally
unlearnable, so when a class is absent from the paper's point mask but present
in the dense train GT, we supplement it with up to ``POINTS_PER_CLASS``
interior-only, grid-spread points sampled from the dense mask. This is the
only use of dense masks in the training split: generating point coordinates
(the point-only contract is untouched — training still consumes (x, y, class)
tuples and never a mask).

``train.json`` gets (x, y, class 0..16) tuples and NO ``mask`` key (the
dataset enforces the point-only contract). Dense masks are used only to build
the held-out ``val.json`` for evaluation, exactly as before.

Source data (sibling of this repo root):
  dlrsd/point_1cmasks/*.png                     (630 point-annotation maps)
  dlrsd/train_1cmasks/*.png                     (630 dense train masks — ship supplement only)
  dlrsd/train_images/*.png                      (630 training images)
  dlrsd/full_test_images/*.png                  (1319 held-out images)
  dlrsd/test_for_compare/full_test_1cmasks/*.png (1319 dense test masks)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .class_map import NUM_CLASSES, UNLABELLED

PACKAGE_ROOT = Path(__file__).resolve().parents[1]   # .../point_only_sam_rs_Es_5pt
PROJECT_ROOT = PACKAGE_ROOT.parent                   # repo root (holds dlrsd/)
DATA_DIR = PACKAGE_ROOT / "data"
POINTS_PER_CLASS = 5
GRID = 3            # GRID x GRID spatial cells for spread selection
SEED = 42

DLRSD = PROJECT_ROOT / "dlrsd"
POINT_MASKS = DLRSD / "point_1cmasks"
DENSE_TRAIN_MASKS = DLRSD / "train_1cmasks"
TRAIN_IMAGES = DLRSD / "train_images"
TEST_IMAGES = DLRSD / "full_test_images"
TEST_MASKS = DLRSD / "test_for_compare" / "full_test_1cmasks"


def _grid_pick(ys: np.ndarray, xs: np.ndarray, n: int, rng: np.random.Generator,
               grid: int = GRID) -> np.ndarray:
    """Pick `n` well-spread (y, x) points from the given labelled pixels.

    Divides the image into a GRID x GRID grid of cells, then greedily picks the
    cells with the most labelled pixels (one point per cell, chosen nearest the
    cell's own centroid) until `n` cells are covered. This guarantees spatial
    spread — no two points can come from the same 1/3-of-image cell — which is
    exactly the property Chakraborty's original sampling has. Falls back to
    filling from remaining pixels when fewer than `n` cells contain pixels.
    """
    h, w = 256, 256
    if len(ys) <= n:
        return np.stack([ys, xs], 1)
    cy = np.clip(ys // (h // grid), 0, grid - 1)
    cx = np.clip(xs // (w // grid), 0, grid - 1)
    cells = {}
    for i in range(len(ys)):
        cells.setdefault((int(cy[i]), int(cx[i])), []).append(i)
    # order cells by pixel count (most populated first)
    ordered = sorted(cells.items(), key=lambda kv: -len(kv[1]))
    chosen = []
    for (key, idxs) in ordered:
        if len(chosen) >= n:
            break
        sub_y, sub_x = ys[idxs], xs[idxs]
        cyc, cxc = sub_y.mean(), sub_x.mean()            # cell centroid
        d = (sub_y - cyc) ** 2 + (sub_x - cxc) ** 2
        chosen.append((sub_y[int(d.argmin())], sub_x[int(d.argmin())]))
    if len(chosen) < n:
        used = set(chosen)
        rest = [(int(y), int(x)) for y, x in zip(ys, xs) if (int(y), int(x)) not in used]
        rng.shuffle(rest)
        chosen.extend(rest[: n - len(chosen)])
    return np.asarray(chosen, dtype=np.int64)


def sample_points(point_mask: np.ndarray, dense_mask: np.ndarray | None,
                  per_class: int, rng: np.random.Generator) -> list:
    """Extract up to `per_class` (x, y, class 0..16) points per class.

    Primary source: the Chakraborty point-annotation map (value 18, unlabelled,
    is skipped). If a class has no points there but IS present in the dense
    train GT, supplement interior-only points from the dense mask (ship fix).
    """
    points = []
    n_supplemented = 0
    for c in range(1, NUM_CLASSES + 1):
        ys, xs = np.where(point_mask == c)
        supplemented = False
        if len(xs) == 0 and dense_mask is not None and (dense_mask == c).any():
            # class absent from the paper's points -> sample interior-only from GT
            region = (dense_mask == c).astype(np.uint8)
            eroded = cv2.erode(region, np.ones((3, 3), np.uint8))
            ys, xs = np.where(eroded > 0)
            if len(xs) == 0:                       # too thin to erode -> raw region
                ys, xs = np.where(region > 0)
            supplemented = True
        n = min(per_class, len(xs))
        if n == 0:
            continue
        yx = _grid_pick(ys, xs, n, rng)
        for y, x in yx:
            points.append([int(x), int(y), c - 1])
        if supplemented:
            n_supplemented += n
    return points, n_supplemented


def build_train():
    images = sorted(TRAIN_IMAGES.glob("*.png"))
    pmasks = sorted(POINT_MASKS.glob("*.png"))
    dmasks = sorted(DENSE_TRAIN_MASKS.glob("*.png"))
    assert len(images) == len(pmasks), (len(images), len(pmasks))
    rows = []
    n_total = 0
    n_ship = 0
    n_sup_total = 0
    for i, (img_p, pm_p) in enumerate(zip(images, pmasks)):
        assert img_p.stem == pm_p.stem, (img_p, pm_p)
        img = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
        pm = cv2.imread(str(pm_p), cv2.IMREAD_GRAYSCALE)
        dm = cv2.imread(str(dmasks[i]), cv2.IMREAD_GRAYSCALE) if dmasks else None
        rng = np.random.default_rng(SEED + i)
        points, n_sup = sample_points(pm, dm, POINTS_PER_CLASS, rng)
        n_ship += sum(1 for p in points if p[2] == 13)
        n_sup_total += n_sup
        n_total += len(points)
        rows.append({
            "id": img_p.stem,
            "image": str(img_p),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            # deliberately NO "mask" key — point-only contract
            "points": points,
        })
    out = DATA_DIR / "train.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"train.json: {len(rows)} items, {n_total} points "
          f"({POINTS_PER_CLASS}/class max; {n_ship} ship points; "
          f"{n_sup_total} supplemented from dense GT)")


def build_val():
    images = sorted(TEST_IMAGES.glob("*.png"))
    masks = sorted(TEST_MASKS.glob("*.png"))
    assert len(images) == len(masks), (len(images), len(masks))
    out_dir = DATA_DIR / "val_masks_remapped"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_p, mask_p in zip(images, masks):
        img = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        assert mask.min() >= 1 and mask.max() <= NUM_CLASSES, (mask_p, mask.min(), mask.max())
        remapped = (mask.astype(np.int32) - 1).astype(np.uint8)   # 1..17 -> 0..16 at write time
        dst = out_dir / mask_p.name
        cv2.imwrite(str(dst), remapped)
        rows.append({
            "id": img_p.stem,
            "image": str(img_p),
            "mask": str(dst),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
        })
    out = DATA_DIR / "val.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"val.json: {len(rows)} items; masks remapped 1..17 -> 0..16 -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true",
                        help="rebuild train.json only (val masks already exist)")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_train()
    if not args.train_only:
        build_val()


if __name__ == "__main__":
    main()
