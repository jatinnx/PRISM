"""Evaluation for PRISM.

Dense masks are used HERE ONLY -- this is the single place in the package where
they may appear, and nothing in this file feeds back into training.

Beyond the usual mIoU / PA / precision / recall, this reports four numbers that
correspond directly to the failure modes the method was designed against, so the
eval log is itself the evidence rather than something to be inspected by eye:

  SPECKLE      share of pixels whose label differs from its own 5x5 mode
               -> failure mode 1 (salt-and-pepper)
  GHOST        share of (image, class) pairs predicted over >=0.5% of the image
               with zero GT pixels anywhere in that image
               -> failure mode 2 (hallucinated classes)
  FLOOD        share of images whose largest predicted class covers >=1.5x the
               area of the largest GT class
               -> failure mode 3 (dominant class flooding)
  TRIMAP PA    pixel accuracy restricted to a band of width d around GT
               boundaries -> the boundary claim, which whole-image mIoU hides

Two optional inference-time switches, both label-free and both reported as their
own rows rather than folded into the headline number:

  --tta          average the posterior over the four flip/mirror variants
  --region-vote  pool the posterior over each frozen SAM region and take the
                 region argmax. This uses the same cached, class-agnostic
                 partition training used; it involves no labels and no learning,
                 and it is the cleanest possible test of the claim that the
                 partition carries the object geometry.
"""
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .configs.prism import (ARCH_FIELDS, PrismConfig, config_from_checkpoint,
                            resolve)
from .core.regions import RegionIndex
from .data.class_map import CLASS_NAMES
from .data.dataset_prism import PrismDataset, collate_prism
from .model.net import PrismNet


def _out(msg: str, log=None):
    print(msg, flush=True)
    if log is not None:
        log.write(msg + "\n")
        log.flush()


# --------------------------------------------------------------------------- #
#  failure-mode counters                                                      #
# --------------------------------------------------------------------------- #
def mode_filter(pred: np.ndarray, num_classes: int, k: int = 5) -> np.ndarray:
    """Per-pixel majority label in a k x k window, via C box filters."""
    best = None
    best_n = None
    for c in range(num_classes):
        n = cv2.boxFilter((pred == c).astype(np.float32), -1, (k, k),
                          normalize=False, borderType=cv2.BORDER_REPLICATE)
        if best is None:
            best, best_n = np.full_like(pred, c), n
        else:
            take = n > best_n
            best = np.where(take, c, best)
            best_n = np.where(take, n, best_n)
    return best


def boundary_band(gt: np.ndarray, width: int) -> np.ndarray:
    """Pixels within ``width`` of a GT label change."""
    g = gt.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(g, k) != cv2.erode(g, k))
    if width > 1:
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((width, width), np.uint8)) > 0
    return edge


# --------------------------------------------------------------------------- #
#  inference                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _posterior(model, image, tta: bool):
    """-> (B,C,H,W) probabilities."""
    p = model(image)["logits"].softmax(1)
    if not tta:
        return p
    for dims in ([3], [2], [2, 3]):
        q = model(torch.flip(image, dims=dims))["logits"].softmax(1)
        p = p + torch.flip(q, dims=dims)
    return p / 4.0


@torch.no_grad()
def _region_vote(prob: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Region-pooled argmax; pixels with no region keep their own argmax."""
    ridx = RegionIndex(region)
    m = ridx.mean(prob)                                  # (total, C)
    lab = m.argmax(1)
    voted = ridx.scatter_back_1d(lab.to(torch.float32)).long()
    own = prob.argmax(1)
    has_region = ridx.scatter_back_1d(
        (torch.arange(ridx.total, device=prob.device) != ridx.dump).to(torch.float32)) > 0.5
    return torch.where(has_region, voted, own)


# --------------------------------------------------------------------------- #
def run_eval(model, cfg: PrismConfig, val_manifest: Optional[str] = None,
             log=None, save_preds: Optional[str] = None, tta: bool = False,
             region_vote: bool = False, limit: Optional[int] = None):
    """Evaluate an already-built model.

    Split out from ``evaluate`` so the training loop can score its EMA weights
    in place, without constructing a second copy of the ViT on the same GPU.
    """
    device = next(model.parameters()).device
    C = cfg.num_classes
    manifest = resolve(val_manifest or cfg.val_manifest)

    region_npz = None
    if region_vote:
        cand = Path(resolve(cfg.region_cache_val))
        if not cand.exists():
            raise FileNotFoundError(
                f"--region-vote needs {cand}; build it with\n"
                f"  python -m e3_only.tools.build_region_cache --split val")
        region_npz = str(cand)

    ds = PrismDataset(manifest, cfg.image_size, training=False,
                      region_npz=region_npz, num_classes=C)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                    collate_fn=collate_prism)

    was_training = model.training
    model.eval()

    pred_dir = None
    if save_preds:
        from .core.colors import make_legend
        pred_dir = Path(resolve(save_preds))
        pred_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pred_dir / "legend.png"), make_legend()[:, :, ::-1])

    conf = np.zeros((C, C), dtype=np.int64)
    band_hits = {3: [0, 0], 5: [0, 0]}
    speckle_num = speckle_den = 0
    ghost_num = ghost_den = 0
    flood_num = n_masked = n_total = 0

    for i, batch in enumerate(dl):
        if limit is not None and i >= limit:
            break
        n_total += 1
        image = batch["image_weak"].to(device, non_blocking=True)
        prob = _posterior(model, image, tta)
        if region_vote:
            pred_t = _region_vote(prob, batch["region"].to(device))
        else:
            pred_t = prob.argmax(1)
        pred = pred_t[0].cpu().numpy().astype(np.uint8)

        m5 = mode_filter(pred, C, 5)
        speckle_num += int((pred != m5).sum())
        speckle_den += pred.size

        gt = None
        if "mask" in batch:
            gt = batch["mask"][0].numpy().astype(np.int64)
            n_masked += 1
            valid = (gt >= 0) & (gt < C)
            conf += np.bincount((gt[valid] * C + pred[valid]).ravel(),
                                minlength=C * C).reshape(C, C)

            gt_present = set(np.unique(gt[valid]).tolist())
            pa_pred = np.bincount(pred.ravel(), minlength=C) / float(pred.size)
            for c in range(C):
                if pa_pred[c] < 0.005:
                    continue
                ghost_den += 1
                if c not in gt_present:
                    ghost_num += 1
            gt_area = np.bincount(gt[valid].ravel(), minlength=C) / float(valid.sum())
            if pa_pred.max() >= 1.5 * max(gt_area.max(), 1e-6):
                flood_num += 1

            for d, acc in band_hits.items():
                b = boundary_band(gt, d) & valid
                if b.any():
                    acc[0] += int((pred[b] == gt[b]).sum())
                    acc[1] += int(b.sum())

        if pred_dir is not None:
            _save_prediction_images(batch, pred, gt, pred_dir, C)

    if was_training:
        model.train()

    # ---------------------------------------------------------------- #
    _out(f"evaluated {n_total} images ({n_masked} with masks)", log)
    _out(f"SPECKLE  {speckle_num / max(1, speckle_den):.4f}  "
         f"(share of pixels disagreeing with their own 5x5 mode)", log)
    if n_masked == 0:
        _out("no dense masks in manifest -- skipping mIoU/PA", log)
        return {}

    ious, precs, recs = [], [], []
    for c in range(C):
        tp = int(conf[c, c])
        fn = int(conf[c].sum()) - tp
        fp = int(conf[:, c].sum()) - tp
        ious.append(tp / (tp + fn + fp) if (tp + fn + fp) else float("nan"))
        precs.append(tp / (tp + fp) if (tp + fp) else float("nan"))
        recs.append(tp / (tp + fn) if (tp + fn) else float("nan"))
    miou = float(np.nanmean(ious))
    pa = float(conf.diagonal().sum()) / max(1, int(conf.sum()))

    _out(f"mIoU: {miou:.4f}", log)
    _out(f"PA:   {pa:.4f}", log)
    _out(f"mPrec:{float(np.nanmean(precs)):.4f}  mRecall:{float(np.nanmean(recs)):.4f}", log)
    for d, (h, t) in sorted(band_hits.items()):
        _out(f"TRIMAP PA (band {d}px): {h / max(1, t):.4f}   [{t} px]", log)
    _out(f"GHOST    {ghost_num}/{ghost_den} = {ghost_num / max(1, ghost_den):.4f}  "
         f"(predicted >=0.5% area, zero GT pixels)", log)
    _out(f"FLOOD    {flood_num}/{n_masked} = {flood_num / max(1, n_masked):.4f}  "
         f"(largest predicted class >=1.5x largest GT class)", log)
    _out("per_class_IoU: " + str([round(v, 4) for v in ious]), log)
    _out("per_class_names: " + ", ".join(CLASS_NAMES), log)
    worst = sorted(range(C), key=lambda c: ious[c])[:5]
    _out("worst 5: " + ", ".join(f"{CLASS_NAMES[c]} {ious[c]:.3f}" for c in worst), log)

    return {"mIoU": miou, "PA": pa, "mPrec": float(np.nanmean(precs)),
            "mRecall": float(np.nanmean(recs)), "per_class_IoU": ious,
            "speckle": speckle_num / max(1, speckle_den),
            "ghost": ghost_num / max(1, ghost_den),
            "flood": flood_num / max(1, n_masked),
            "trimap3": band_hits[3][0] / max(1, band_hits[3][1]),
            "trimap5": band_hits[5][0] / max(1, band_hits[5][1])}


# --------------------------------------------------------------------------- #
def evaluate(cfg: PrismConfig, checkpoint: str, val_manifest: Optional[str] = None,
             log=None, save_preds: Optional[str] = None, tta: bool = False,
             region_vote: bool = False, which: str = "teacher",
             limit: Optional[int] = None):
    device = torch.device(cfg.device)

    # The checkpoint carries the config it was trained with. Rebuild the
    # architecture from it BEFORE constructing the model, so an ablation row with
    # different tensor shapes -- single-prototype, a different embed_dim, a
    # raw-RGB stem -- evaluates correctly without the caller having to remember a
    # matching flag. (load_state_dict raises on a shape mismatch even with
    # strict=False, so getting this wrong is a crash rather than a wrong score;
    # the point here is not to crash.)
    ckpt = torch.load(checkpoint, map_location=device)
    stored = ckpt.get("config")
    if isinstance(stored, dict):
        changed = {k: (getattr(cfg, k, None), stored[k]) for k in ARCH_FIELDS
                   if k in stored and getattr(cfg, k, None) != stored[k]}
        cfg = config_from_checkpoint(stored, cfg)
        if changed:
            _out("architecture taken from the checkpoint: "
                 + ", ".join(f"{k} {a!r}->{b!r}" for k, (a, b) in changed.items()), log)
    C = cfg.num_classes

    model = PrismNet(resolve(cfg.sam_checkpoint), C, str(device),
                     cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout,
                     stem_channels=cfg.stem_channels, embed_dim=cfg.embed_dim,
                     prototypes_per_class=cfg.prototypes_per_class,
                     invariant_stem=cfg.invariant_stem,
                     invariant_window=cfg.invariant_window,
                     dilated_context=cfg.dilated_context,
                     k_temperature=cfg.k_temperature, scale_init=cfg.scale_init,
                     sam_normalize=cfg.sam_normalize).to(device)

    state = ckpt.get(which) or ckpt.get("teacher") or ckpt.get("student")
    if state is None:
        raise KeyError(f"checkpoint has no usable weights: {list(ckpt)}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        _out(f"WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}", log)
    lora_missing = [k for k in missing if "lora" in k.lower()]
    if lora_missing:
        _out(f"WARNING: {len(lora_missing)} LoRA keys missing from the checkpoint -- "
             f"the backbone adaptation is NOT loaded.", log)
    model.eval()
    _out(f"checkpoint {Path(checkpoint).name} epoch {ckpt.get('epoch', '?')} "
         f"weights='{which}' tta={tta} region_vote={region_vote}", log)
    _out(model.decoder.classifier.report(), log)

    return run_eval(model, cfg, val_manifest, log, save_preds, tta, region_vote, limit)


# --------------------------------------------------------------------------- #
def _sample_gt_points(gt: np.ndarray, num_classes: int, per_class: int = 5):
    """Interior, grid-spread points from a GT mask, for the figure only.

    Used solely to draw the "what a human would have clicked" panel for val
    images, which carry no annotated points. It never touches training. (The E3
    version of this function had its per-class loop body fall out of the loop, so
    it only ever drew points for the last class; this one does not.)
    """
    h, w = gt.shape
    pts = []
    for c in range(num_classes):
        mask = (gt == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        ys, xs = np.where(eroded > 0)
        if len(ys) == 0:
            ys, xs = np.where(mask > 0)
        cells = {}
        cy = np.clip(ys // max(1, h // 3), 0, 2)
        cx = np.clip(xs // max(1, w // 3), 0, 2)
        for i in range(len(ys)):
            cells.setdefault((int(cy[i]), int(cx[i])), []).append(i)
        chosen = []
        for _, idxs in sorted(cells.items(), key=lambda kv: -len(kv[1])):
            if len(chosen) >= per_class:
                break
            sy, sx = ys[idxs], xs[idxs]
            d = (sy - sy.mean()) ** 2 + (sx - sx.mean()) ** 2
            k = int(d.argmin())
            chosen.append((int(sx[k]), int(sy[k]), c))
        pts.extend(chosen)
    return pts


def _save_prediction_images(batch, pred, gt, pred_dir, num_classes):
    """Five panels per image, in a fixed order so a folder reads like a report:
    ``_real``, ``_points``, ``_gt``, ``_overlay``, and the bare predicted map."""
    from .core.colors import colorize, overlay
    from .core.prompts import draw_points
    iid = batch["image_id"][0]
    img = (batch["image_weak"][0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    cv2.imwrite(str(pred_dir / f"{iid}_real.png"), img[:, :, ::-1])
    pts = batch["points"][0].numpy() if len(batch["points"][0]) else None
    if pts is None and gt is not None:
        pts = _sample_gt_points(gt.astype(np.uint8), num_classes)
    if pts is not None and len(pts):
        cv2.imwrite(str(pred_dir / f"{iid}_points.png"), draw_points(img, pts)[:, :, ::-1])
    if gt is not None:
        cv2.imwrite(str(pred_dir / f"{iid}_gt.png"), colorize(gt.astype(np.uint8))[:, :, ::-1])
    cv2.imwrite(str(pred_dir / f"{iid}_overlay.png"), overlay(img, pred)[:, :, ::-1])
    cv2.imwrite(str(pred_dir / f"{iid}.png"), colorize(pred)[:, :, ::-1])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--which", default="teacher", choices=["teacher", "student"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--region-vote", action="store_true")
    ap.add_argument("--save-preds", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    # the architecture comes from the checkpoint itself (see ``evaluate``), so no
    # per-ablation flag is needed here
    cfg = PrismConfig()

    log = None
    if a.log:
        p = Path(resolve(a.log))
        p.parent.mkdir(parents=True, exist_ok=True)
        log = p.open("w")
    try:
        evaluate(cfg, a.checkpoint, a.val_manifest, log, a.save_preds,
                 a.tta, a.region_vote, a.which, a.limit)
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
