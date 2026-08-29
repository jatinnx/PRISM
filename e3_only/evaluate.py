"""Evaluation: mIoU, pixel accuracy (PA), mean precision (mPrec) / recall.

Dense masks are used HERE ONLY (the one place they may appear). If the val
manifest has no masks, the run is counted and reported gracefully.

This file produces the **E3 baseline** column of the comparison table, so its job
is to be boringly correct rather than clever. Two notes on that:

* Eval-time prototype refinement (``proto_use_refine_at_eval``) is not
  implemented — the ``PrototypeBank.refine`` this file used to call never existed,
  so the flag could only ever have raised ``AttributeError``. It is now reported
  and ignored instead of crashing. Every number this file has ever produced is
  therefore an un-refined number, which is exactly what the baseline should be
  (v1 measured refinement as inert at best and corrupting with stale prototypes at
  worst). The checkpoint still carries ``prototype_refresh_epoch`` so a stale bank
  is flagged.
* The confusion matrix is accumulated with the same ``np.bincount`` expression
  ``evaluate_prism.py`` uses, over the same validity test ``0 <= y < C``. mIoU, PA,
  mPrec and mRecall are then computed identically in both files, which is what
  makes "before" and "after" the same quantity.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .configs.base import Config, resolve
from .data.class_map import CLASS_NAMES
from .data.dataset import PointOnlyDataset, collate_points
from .model.sam_wrapper import PointOnlySAM


def _out(msg: str, log=None):
    print(msg, flush=True)
    if log is not None:
        log.write(msg + "\n")
        log.flush()


def evaluate(cfg: Config, checkpoint: str, val_manifest: str | None = None, log=None,
             save_preds: str | None = None):
    """Evaluate a checkpoint. With ``save_preds``, writes per-image coloured
    segmentation maps, overlays and GT to that directory."""
    device = torch.device(cfg.device)
    manifest = val_manifest or cfg.val_manifest
    ds = PointOnlyDataset(resolve(manifest), cfg.image_size, False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                    collate_fn=collate_points)
    pred_dir = None
    if save_preds:
        from .core.colors import make_legend
        pred_dir = Path(resolve(save_preds))
        pred_dir.mkdir(parents=True, exist_ok=True)
        (pred_dir / "legend.png").write_bytes(cv2.imencode(".png", make_legend())[1].tobytes())

    model = PointOnlySAM(resolve(cfg.sam_checkpoint), cfg.num_classes, str(device),
                         cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout,
                         cfg.background_class,
                         spatial_context=cfg.spatial_context).to(device)
    ckpt = torch.load(checkpoint, map_location=device, mmap=True)
    state = ckpt.get("teacher") or ckpt.get("student")
    if state is None:
        raise KeyError(f"checkpoint has neither 'teacher' nor 'student': {list(ckpt)}")
    model.load_state_dict(state, strict=False)
    model.eval()

    # ``PrototypeBank`` has no ``refine``; the branch that called it could only
    # ever have raised. Say so and carry on un-refined rather than crash, and
    # rather than silently implying a refinement step happened.
    if (ckpt.get("prototypes") is not None and cfg.use_prototypes
            and cfg.proto_use_refine_at_eval):
        _out("NOTE: proto_use_refine_at_eval is set, but eval-time prototype "
             "refinement is not implemented (PrototypeBank has no refine()). "
             "Reporting un-refined logits -- this is the baseline definition.", log)

    if ckpt.get("prototypes") is not None:
        ref_epoch = int(ckpt.get("prototype_refresh_epoch", 0))
        ckpt_epoch = int(ckpt.get("epoch", 0))
        if cfg.proto_refresh_every > 0 and ref_epoch < ckpt_epoch - cfg.proto_refresh_every:
            _out(f"WARNING: checkpoint prototypes last refreshed at epoch {ref_epoch} "
                 f"(checkpoint epoch {ckpt_epoch}) — older than one refresh window; "
                 f"treat prototype-derived signals with caution.", log)

    C = cfg.num_classes
    conf = np.zeros((C, C), dtype=np.int64)
    n_with_mask = 0
    n_total = 0
    with torch.no_grad():
        for batch in dl:
            n_total += 1
            image = batch["image_weak"].to(device)
            feat = model.encode(image)
            logits = model.semantic_logits(feat, image.shape[-2:])
            pred = logits.argmax(1)
            gt = None
            if "mask" in batch:
                gt = batch["mask"].to(device)
                n_with_mask += 1
                # same expression as evaluate_prism.py, so the two are comparable
                g = gt.reshape(-1).cpu().numpy().astype(np.int64)
                p = pred.reshape(-1).cpu().numpy().astype(np.int64)
                valid = (g >= 0) & (g < C)
                conf += np.bincount((g[valid] * C + p[valid]).ravel(),
                                    minlength=C * C).reshape(C, C)
            if pred_dir is not None:
                _save_prediction_images(batch, pred, gt, pred_dir)

    if pred_dir is not None:
        _out(f"saved predictions to {pred_dir} ({n_total} images)", log)
    _out(f"evaluated {n_total} images ({n_with_mask} with masks)", log)
    if n_with_mask == 0:
        _out("no dense masks in manifest — skipping mIoU/PA/mPrec", log)
        return

    per_class = {}
    ious, precs, recs = [], [], []
    for c in range(cfg.num_classes):
        tp = conf[c, c]
        fn = conf[c].sum() - tp
        fp = conf[:, c].sum() - tp
        denom = tp + fn + fp
        iou = float(tp / denom) if denom else float("nan")
        prec = float(tp / (tp + fp)) if (tp + fp) else float("nan")
        rec = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        ious.append(iou)
        precs.append(prec)
        recs.append(rec)
        per_class[c] = {"IoU": iou, "Prec": prec, "Recall": rec}
    pa = float(conf.diagonal().sum()) / max(1, int(conf.sum()))
    _out(f"mIoU: {float(np.nanmean(ious)):.4f}", log)
    _out(f"PA:   {pa:.4f}", log)
    _out(f"mPrec:{float(np.nanmean(precs)):.4f}  mRecall:{float(np.nanmean(recs)):.4f}", log)
    _out("per_class_IoU: " + str([round(v, 4) for v in ious]), log)
    _out("per_class_names: " + ", ".join(CLASS_NAMES), log)


def _save_prediction_images(batch, pred, gt, pred_dir):
    """Write all FIVE per-image outputs (consistent order, so a folder can be
    read like a report):

      1. ``<id>_real.png``    — the original input image
      2. ``<id>_points.png``  — input + class-coloured point clicks
      3. ``<id>_gt.png``      — dense ground truth (reference)
      4. ``<id>_overlay.png`` — prediction overlaid on the real image
      5. ``<id>.png``         — the model's (novel) predicted class map

    Points come from the manifest when present; for test images (no annotated
    points) they are sampled interior-only from the GT mask, 5/class, in the
    same style as Figure 1 of the report.
    """
    from .core.colors import colorize, overlay
    from .core.prompts import draw_points
    image_id = batch["image_id"][0]
    img = (batch["image_weak"][0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    p = pred[0].cpu().numpy().astype("uint8")

    pts = batch["points"][0].cpu().numpy() if len(batch["points"][0]) else None
    g = gt[0].cpu().numpy().astype("uint8") if gt is not None else None
    if pts is None and g is not None:
        pts = _sample_gt_points(g, per_class=5)

    # 1. real image
    cv2.imwrite(str(pred_dir / f"{image_id}_real.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # 2. point-labelled image
    if pts is not None:
        img_pts = draw_points(img, pts)
        cv2.imwrite(str(pred_dir / f"{image_id}_points.png"),
                    cv2.cvtColor(img_pts, cv2.COLOR_RGB2BGR))
    # 3. ground truth
    if g is not None:
        cv2.imwrite(str(pred_dir / f"{image_id}_gt.png"),
                    cv2.cvtColor(colorize(g), cv2.COLOR_RGB2BGR))
    # 4. overlay (prediction on the real image)
    cv2.imwrite(str(pred_dir / f"{image_id}_overlay.png"),
                cv2.cvtColor(overlay(img, p), cv2.COLOR_RGB2BGR))
    # 5. novel predicted class map
    cv2.imwrite(str(pred_dir / f"{image_id}.png"),
                cv2.cvtColor(colorize(p), cv2.COLOR_RGB2BGR))


def _sample_gt_points(gt, per_class=5):
    """Interior-only point sampling from a GT mask (val images have no annotated
    points): erode each class region, then grid-spread up to ``per_class`` points.
    Returns a list of (x, y, class) tuples.

    For the ``_points.png`` figure panel only -- it never touches training, and it
    is the reason this file may read a dense mask at all.

    The per-class body used to sit *outside* the ``for c`` loop, so every image got
    points for one class only (whichever happened to be last with a non-empty
    mask) -- and an image with no class at all raised ``NameError`` on an unbound
    ``ys``. Both are fixed here; ``evaluate_prism._sample_gt_points`` is the same
    routine for the PRISM side.
    """
    from .data.class_map import PALETTE
    h, w = gt.shape
    pts = []
    for c in range(len(PALETTE)):
        mask = (gt == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
        ys, xs = np.where(eroded > 0)
        if len(ys) == 0:
            ys, xs = np.where(mask > 0)

        # grid spread over a 3x3 cell layout (same idea as make_manifests): pick
        # the most populated cells first, one point per cell. max(1, .) because a
        # mask narrower than 3 px would otherwise divide by zero.
        rng = np.random.default_rng((int(gt.sum()) * 31 + c) % (2 ** 32 - 1))
        chosen = []
        cells = {}
        cy = np.clip(ys // max(1, h // 3), 0, 2)
        cx = np.clip(xs // max(1, w // 3), 0, 2)
        for i in range(len(ys)):
            cells.setdefault((int(cy[i]), int(cx[i])), []).append(i)
        for _, idxs in sorted(cells.items(), key=lambda kv: -len(kv[1])):
            if len(chosen) >= per_class:
                break
            sub_y, sub_x = ys[idxs], xs[idxs]
            d = (sub_y - sub_y.mean()) ** 2 + (sub_x - sub_x.mean()) ** 2
            k = int(d.argmin())
            chosen.append((int(sub_x[k]), int(sub_y[k]), c))
        if len(chosen) < per_class:                 # fill leftover from remaining
            used = set(chosen)
            rest = [(int(x), int(y), c) for y, x in zip(ys, xs)
                    if (int(x), int(y), c) not in used]
            rng.shuffle(rest)
            chosen.extend(rest[: per_class - len(chosen)])
        pts.extend(chosen)
    return pts
