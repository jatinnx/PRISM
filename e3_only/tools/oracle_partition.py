"""How good could the frozen partition possibly be? (the region-constant oracle)

Every region-based term in the objective -- L_prop (S4.2), L_hom (S4.7),
L_self (S4.10) -- and the inference-time region vote all push the prediction
towards being CONSTANT INSIDE EACH REGION. This tool measures the best mIoU any
such labelling can reach by giving each region its own majority GT class. It is
therefore an upper bound on the entire region machinery, and it is the number
that decides whether the partition or the classifier is the bottleneck.

Dense val masks are read HERE FOR MEASUREMENT ONLY (as in evaluate_prism.py);
nothing this tool computes is ever fed back into training.

Two oracles, because the code uses the partition in two different scopes:

  ALL-REGIONS   every pixel that has a region id takes that region's majority GT
                class. This is the ceiling on L_hom / L_self, which act on SAM
                masks and filler alike.

  SAM-ONLY      only pixels in an ELIGIBLE region take the majority class, where
                eligible means exactly what evaluate_prism._region_vote means by
                it: id < n_sam and size >= min_region. Every other pixel keeps
                its own GT. This is the ceiling on the inference-time region vote.

SAM-ONLY scores HIGHER than ALL-REGIONS by construction -- it forces fewer
pixels to a region-constant answer -- so the two numbers bracket the machinery
rather than competing.

Usage:
    python -m e3_only.tools.oracle_partition [--limit N] [--log path]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.configs.prism import PrismConfig, resolve              # noqa: E402
from e3_only.data.class_map import CLASS_NAMES, NUM_CLASSES         # noqa: E402
from e3_only.data.dataset_prism import PrismDataset, collate_prism  # noqa: E402


def miou(conf):
    """conf[gt, pred] -> (mIoU over present classes, PA, per-class IoU)."""
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    pa = tp.sum() / max(1.0, conf.sum())
    return np.nanmean(iou), pa, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-region", type=int, default=None,
                    help="default: PrismConfig.region_vote_min_size")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    cfg = PrismConfig()
    min_region = a.min_region if a.min_region is not None else cfg.region_vote_min_size
    cache = Path(resolve(cfg.region_cache_val))
    if not cache.exists():
        raise FileNotFoundError(
            f"need {cache}; build it with\n"
            f"  python -m e3_only.tools.build_region_cache --split val")

    ds = PrismDataset(resolve(cfg.val_manifest), cfg.image_size, training=False,
                      region_npz=str(cache), num_classes=NUM_CLASSES)
    n = len(ds) if a.limit is None else min(a.limit, len(ds))

    C = NUM_CLASSES
    conf_all = np.zeros((C, C), np.int64)
    conf_sam = np.zeros((C, C), np.int64)
    px_forced_all = px_forced_sam = px_valid = 0
    n_done = 0

    for i in range(n):
        s = ds[i]
        if "mask" not in s or s.get("region") is None:
            continue
        gt = np.asarray(s["mask"]).astype(np.int64)
        region = np.asarray(s["region"]).astype(np.int64)
        n_sam = int(s.get("n_sam", 0))
        valid = (gt >= 0) & (gt < C)
        if not valid.any():
            continue
        n_done += 1
        px_valid += int(valid.sum())

        nreg = int(region.max()) + 1
        if nreg <= 0:
            continue
        hist = np.zeros((nreg, C), np.int64)
        m = valid & (region >= 0)
        np.add.at(hist, (region[m], gt[m]), 1)
        sizes = hist.sum(1)
        major = hist.argmax(1)                       # region -> majority GT class

        # ALL-REGIONS: every pixel with a region takes its region's majority.
        pred_all = gt.copy()
        pred_all[m] = major[region[m]]
        px_forced_all += int(m.sum())

        # SAM-ONLY: eligible == id < n_sam and size >= min_region (evaluate_prism).
        elig_reg = (np.arange(nreg) < n_sam) & (sizes >= min_region)
        e = m & elig_reg[region.clip(0, max(nreg - 1, 0))]
        pred_sam = gt.copy()
        pred_sam[e] = major[region[e]]
        px_forced_sam += int(e.sum())

        np.add.at(conf_all, (gt[valid], pred_all[valid]), 1)
        np.add.at(conf_sam, (gt[valid], pred_sam[valid]), 1)

    mi_a, pa_a, iou_a = miou(conf_all)
    mi_s, pa_s, iou_s = miou(conf_sam)

    lines = [
        f"region-constant ORACLE over {n_done} val images, min_region={min_region}",
        f"  ALL-REGIONS  mIoU {mi_a:.4f}  PA {pa_a:.4f}   "
        f"({px_forced_all / max(1, px_valid):.4f} of valid px forced region-constant)",
        f"  SAM-ONLY     mIoU {mi_s:.4f}  PA {pa_s:.4f}   "
        f"({px_forced_sam / max(1, px_valid):.4f} of valid px forced region-constant)",
        "",
        "per-class IoU ceiling (all-regions / sam-only):",
    ]
    for c, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:<12s} {iou_a[c]:.4f}  {iou_s[c]:.4f}")
    lines += [
        "",
        "READ IT LIKE THIS: the gap between a trained model's mIoU and ALL-REGIONS is",
        "how much of the error the region machinery could still remove; the gap that",
        "remains BELOW ALL-REGIONS after the machinery is perfect is the partition's",
        "own fault. If a model sits far under ALL-REGIONS, the shape prior is not the",
        "bottleneck and the classifier is.",
    ]
    text = "\n".join(lines)
    print(text)
    if a.log:
        p = Path(resolve(a.log))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
