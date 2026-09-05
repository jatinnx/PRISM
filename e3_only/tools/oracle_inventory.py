"""How good could the per-image class inventory possibly be? (the inventory oracle)

MEASUREMENT ONLY. This tool reads dense val masks to build the oracle it reports,
exactly as tools/oracle_partition.py does, so nothing it prints can ever be quoted
as a result of the method -- only as a ceiling on one of the method's mechanisms.
No training code imports it.

Motivation, measured. 0.3964 of all pixel error on the 0.5477 checkpoint is
pixels assigned to a class the image does not contain at all
(artifacts/diagnose_v5corrected_0.5477.txt, "share of all errors from ghost
classes"), and at image level 2398 (class, image) ghost instances exist over 1319
val images. L_absent / L_present / L_pres_head and the inference-time
--presence-gate all attack that. This tool answers the question those mechanisms
are worth exactly as much as: if the per-image class set were known PERFECTLY and
the argmax restricted to it, what mIoU would this model score?

It is the inventory analogue of oracle_partition's 0.9438: that number sized the
shape prior, this one sizes the inventory constraint, both in the units of the
reported metric.

Four rows, because the two mechanisms compose:

  plain            argmax over all C classes                    (the reported number)
  inventory        argmax restricted to the GT class set        (issue-2 ceiling)
  region           plain + the SAM-only region vote            (shape ceiling, realised)
  inventory+region both                                        (joint ceiling)

Read the gap plain -> inventory as the whole budget available to every inventory
mechanism in the model. If it is small, issue 2 is not the lever and the ghost
pixels are being spent on classes the image DOES contain -- which would be
signature (B), not (A), and would re-order Stage 2.

Usage:
    python -m e3_only.tools.oracle_inventory --checkpoint <path> [--limit N] [--log path]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.configs.prism import PrismConfig, resolve                  # noqa: E402
from e3_only.data.class_map import CLASS_NAMES, NUM_CLASSES             # noqa: E402
from e3_only.data.dataset_prism import PrismDataset, collate_prism      # noqa: E402
from e3_only.evaluate_prism import (load_for_eval, _posterior,          # noqa: E402
                                    _region_vote)


def miou(conf):
    """conf[gt, pred] -> (mIoU over present classes, PA, per-class IoU)."""
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    pa = tp.sum() / max(1.0, conf.sum())
    return np.nanmean(iou), pa, iou


ROWS = ("plain", "inventory", "region", "inventory+region")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--which", default="teacher")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    cfg = PrismConfig()
    model, cfg = load_for_eval(cfg, a.checkpoint, None, a.which, a.tta, True, 0.0)
    device = next(model.parameters()).device
    C = cfg.num_classes

    cache = Path(resolve(cfg.region_cache_val))
    if not cache.exists():
        raise FileNotFoundError(
            f"need {cache}; build it with\n"
            f"  python -m e3_only.tools.build_region_cache --split val")

    ds = PrismDataset(resolve(cfg.val_manifest), cfg.image_size, training=False,
                      region_npz=str(cache), num_classes=C)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                    collate_fn=collate_prism)

    conf = {r: np.zeros((C, C), np.int64) for r in ROWS}
    # how much of the error the oracle actually removes, and where from
    n_done = 0
    px_valid = 0
    px_repainted = 0          # pixels the inventory restriction moved
    ghost_px = 0              # pixels plain argmax put on an absent class
    gt_classes = 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if a.limit is not None and i >= a.limit:
                break
            if "mask" not in batch:
                continue
            image = batch["image_weak"].to(device, non_blocking=True)
            prob = _posterior(model, image, a.tta, 0.0, cfg.presence_floor)
            gt = batch["mask"][0].numpy().astype(np.int64)
            valid = (gt >= 0) & (gt < C)
            if not valid.any():
                continue
            n_done += 1
            px_valid += int(valid.sum())

            present = np.unique(gt[valid])
            gt_classes += len(present)
            # The oracle: mass on an absent class cannot win the argmax. Done on
            # the posterior rather than the logits because argmax is invariant to
            # the monotone softmax, and _posterior is where the gate already lives.
            keep = torch.zeros(C, dtype=torch.bool, device=prob.device)
            keep[torch.as_tensor(present, device=prob.device)] = True
            prob_inv = prob * keep.view(1, C, 1, 1)

            region = batch["region"].to(device)
            n_sam = batch.get("n_sam")
            pred = {
                "plain": prob.argmax(1),
                "inventory": prob_inv.argmax(1),
                "region": _region_vote(prob, region, n_sam,
                                       cfg.region_vote_sam_only,
                                       cfg.region_vote_min_size),
                "inventory+region": _region_vote(prob_inv, region, n_sam,
                                                 cfg.region_vote_sam_only,
                                                 cfg.region_vote_min_size),
            }
            for r in ROWS:
                p = pred[r][0].cpu().numpy().astype(np.int64)
                np.add.at(conf[r], (gt[valid], p[valid]), 1)
                if r == "inventory":
                    p0 = pred["plain"][0].cpu().numpy().astype(np.int64)
                    px_repainted += int((valid & (p != p0)).sum())
                    ghost_px += int((valid & ~np.isin(p0, present)).sum())

    # Over a SUBSET, a class absent from every sampled image leaves the confusion
    # matrix empty for that class, nanmean drops it, and mIoU rises for a reason
    # that has nothing to do with the oracle. Over the full 1319 every class is
    # present, so the four rows are comparable; a limited run is a smoke test.
    warn = ([f"  WARNING --limit {a.limit}: rows are NOT comparable. A class absent "
             f"from the sampled images is dropped by nanmean, which moves mIoU on "
             f"its own. Quote the full-val run only."] if a.limit else [])
    lines = warn + [f"inventory ORACLE over {n_done} val images "
             f"(checkpoint {Path(a.checkpoint).name}, which={a.which}, tta={a.tta})",
             f"  mean |S(I)| = {gt_classes / max(1, n_done):.4f} GT classes per image "
             f"of {C}",
             f"  plain argmax puts {ghost_px / max(1, px_valid):.4f} of valid pixels "
             f"on a class the image does not contain",
             f"  the restriction moves {px_repainted / max(1, px_valid):.4f} of valid "
             f"pixels", ""] + warn
    ious = {}
    for r in ROWS:
        mi, pa, iou = miou(conf[r])
        ious[r] = iou
        lines.append(f"  {r:<18s} mIoU {mi:.4f}  PA {pa:.4f}")
    base, _, _ = miou(conf["plain"])
    inv, _, _ = miou(conf["inventory"])
    lines += ["",
              f"  INVENTORY HEADROOM: {inv - base:+.4f} mIoU "
              f"({(inv - base) * 100:+.2f} pp) from a perfect per-image class set",
              "",
              "per-class IoU (" + " / ".join(ROWS) + "):"]
    for c, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:<12s} " + "  ".join(f"{ious[r][c]:.4f}" for r in ROWS))
    lines += ["",
              "READ IT LIKE THIS: plain -> inventory is the ENTIRE budget of every",
              "inventory mechanism in the model (L_absent, L_present, L_pres_head,",
              "--presence-gate). A trained presence estimate can only ever claim a",
              "fraction of it. If the budget is small, the ghost pixels are landing on",
              "classes the image does contain -- signature (B), not (A) -- and Stage 2",
              "should lead with the class-prior logit adjustment instead."]
    text = "\n".join(lines)
    print(text)
    if a.log:
        p = Path(resolve(a.log))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
