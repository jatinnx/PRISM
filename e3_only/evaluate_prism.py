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

Inference-time switches, all label-free and each reported as its own row rather
than folded into the headline number:

  --tta          average the posterior over the four flip/mirror variants
  --region-vote  pool the posterior over each frozen SAM region and take the
                 region argmax. This uses the same cached, class-agnostic
                 partition training used; it involves no labels and no learning,
                 and it is the cleanest possible test of the claim that the
                 partition carries the object geometry.
  --presence-gate   soft per-image inventory prior from the presence head
                 (see inventory.apply_presence_gate)
  --logit-adjust    Stage 2 decision-time class-prior term, z_c - tau*log pi_c
                 (see inventory.apply_logit_adjust). tau>0 = the balanced-prior
                 rule, tau<0 reverses; priors measured by
                 tools/measure_class_priors.py
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
from .core import inventory as inv
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
def _posterior(model, image, tta: bool, gate: float = 0.0, floor: float = 0.05,
               logit_adjust: Optional[float] = None,
               log_prior: Optional[torch.Tensor] = None):
    """-> (B,C,H,W) probabilities.

    ``gate`` > 0 multiplies the posterior by the presence head's own estimate of
    the image's class set before the softmax. See inventory.apply_presence_gate:
    this is the only inventory constraint that can act at test time, because it
    is the only one that does not need the points.

    ``logit_adjust`` / ``log_prior`` apply the Stage 2 class-prior term
    z_c <- z_c - tau*log pi_c (inventory.apply_logit_adjust) after the gate and
    before the softmax; both are per-class log offsets, so they compose. tau=0 or
    no prior is the identity.
    """
    def _one(img, flip=None):
        out = model(img)
        z = out["logits"]
        if gate > 0 and "presence_logit" in out:
            z = inv.apply_presence_gate(z, out["presence_logit"], gate, floor)
        if logit_adjust is not None and log_prior is not None and logit_adjust != 0:
            z = inv.apply_logit_adjust(z, log_prior, logit_adjust)
        z = z.softmax(1)
        return z if flip is None else torch.flip(z, dims=flip)

    p = _one(image)
    if not tta:
        return p
    for dims in ([3], [2], [2, 3]):
        p = p + _one(torch.flip(image, dims=dims), flip=dims)
    return p / 4.0


@torch.no_grad()
def _region_vote(prob: torch.Tensor, region: torch.Tensor,
                 n_sam: Optional[torch.Tensor] = None,
                 sam_only: bool = True, min_size: int = 24) -> torch.Tensor:
    """Region-pooled argmax; pixels outside an eligible region keep their own argmax.

    Eligibility is not "has a region". A SAM mask is a class-agnostic object
    proposal and pooling over it is safe; a filler region is a connected component
    of whatever SAM did not cover, and pooling over one is only safe if it happens
    to be homogeneous. Measured on train at >=24px, with the point-conflict
    exclusion applied as training applies it: SAM masks are 0.971 pure, filler that
    survives the exclusion is 0.952 -- but at test time there are no points, so the
    exclusion cannot run, and the 449 filler regions it would have dropped sit at
    0.686 purity over 14.5M pixels, 417 of them larger than 8000px. Voting those
    repaints an entire blob with one class.

    So: pool over SAM ids only (``sam_only``), and only when the region is big
    enough for its mean to be worth more than a pixel's own posterior
    (``min_size``). Everything else keeps its argmax, which is the un-voted
    prediction -- the vote can then only ever help.
    """
    ridx = RegionIndex(region)
    m = ridx.mean(prob)                                  # (total, C)
    lab = m.argmax(1)
    voted = ridx.scatter_back_1d(lab.to(torch.float32)).long()
    own = prob.argmax(1)

    rid = torch.arange(ridx.total, device=prob.device)
    ok = rid != ridx.dump                                # not the "no region" slot
    if min_size > 1:
        ok = ok & (ridx.count.view(-1) >= min_size)
    eligible = ridx.scatter_back_1d(ok.to(torch.float32)) > 0.5
    if sam_only and n_sam is not None:
        ns = n_sam.to(region.device).view(-1, *([1] * (region.ndim - 1)))
        eligible = eligible & (region < ns) & (region >= 0)
    return torch.where(eligible, voted, own)


# --------------------------------------------------------------------------- #
def _load_class_priors(cfg: PrismConfig, name: str, num_classes: int):
    """Read one prior (C,) vector from artifacts/class_priors.json.

    Measured by tools/measure_class_priors.py from the click inventories only
    (no dense mask). Refuse to guess: a missing file or a bad entry would
    silently feed the decision a wrong prior, and a wrong prior is worse than
    none because it looks measured.
    """
    import json
    p = Path(resolve(cfg.class_priors_json))
    if not p.exists():
        raise FileNotFoundError(
            f"--logit-adjust needs per-class priors at {p}; measure them with\n"
            f"  python -m e3_only.tools.measure_class_priors")
    data = json.loads(p.read_text())
    if name not in data:
        raise KeyError(f"{p} holds {sorted(data)}, not '{name}'")
    v = np.asarray(data[name], dtype=np.float64)
    if v.shape != (num_classes,):
        raise ValueError(f"prior '{name}' in {p} is {v.shape}, expected "
                         f"({num_classes},)")
    if float(v.min()) <= 0.0:
        raise ValueError(f"prior '{name}' has a non-positive entry -- log would "
                         f"be -inf; re-measure with tools/measure_class_priors.py")
    return v


# --------------------------------------------------------------------------- #
def run_eval(model, cfg: PrismConfig, val_manifest: Optional[str] = None,
             log=None, save_preds: Optional[str] = None, tta: bool = False,
             region_vote: bool = False, limit: Optional[int] = None,
             presence_gate: Optional[float] = None,
             logit_adjust: Optional[float] = None,
             logit_prior: Optional[str] = None):
    """Evaluate an already-built model.

    Split out from ``evaluate`` so the training loop can score its EMA weights
    in place, without constructing a second copy of the ViT on the same GPU.
    """
    device = next(model.parameters()).device
    C = cfg.num_classes
    manifest = resolve(val_manifest or cfg.val_manifest)
    gate = cfg.presence_gate if presence_gate is None else presence_gate
    adj = cfg.logit_adjust if logit_adjust is None else logit_adjust
    prior_name = cfg.logit_prior if logit_prior is None else logit_prior
    log_prior = None
    if adj != 0.0:
        freqs = _load_class_priors(cfg, prior_name, C)
        log_prior = torch.log(torch.as_tensor(freqs, dtype=torch.float32,
                                              device=device))
        lo = float((adj * log_prior).min())
        hi = float((adj * log_prior).max())
        _out(f"logit adjust tau={adj:g} prior='{prior_name}' "
             f"(per-class offset range {lo:.3f}..{hi:.3f} logits)", log)
        _out("  tau>0 = balanced rule (raises under-represented classes); "
             "tau<0 penalises them -- which direction wins is the measurement", log)

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
        prob = _posterior(model, image, tta, gate, cfg.presence_floor,
                          adj if adj != 0.0 else None, log_prior)
        if region_vote:
            pred_t = _region_vote(prob, batch["region"].to(device),
                                  batch.get("n_sam"), cfg.region_vote_sam_only,
                                  cfg.region_vote_min_size)
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
def load_for_eval(cfg: PrismConfig, checkpoint: str, log=None,
                  which: str = "teacher", tta: bool = False,
                  region_vote: bool = False,
                  presence_gate: Optional[float] = None):
    """Build the architecture the checkpoint describes, load it, and guard it.

    Split out of ``evaluate`` so that every consumer of a checkpoint -- the
    evaluator, tools/proto_geometry.py, tools/oracle_inventory.py -- passes
    through the SAME completeness guard. A LoRA-less checkpoint scored 0.1807 and
    was read as 0.5493 once already; the guard must have exactly one home, and
    a measurement tool that skipped it would re-open that hole from the side.

    Returns ``(model, cfg)``. The returned cfg is the CHECKPOINT's architecture,
    which may differ from the caller's in any of ARCH_FIELDS.
    """
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

    # A missing TRAINED tensor means this eval measures a partly-random model.
    # It used to be a warning; a warning scrolled past and 6596 prediction PNGs
    # from a crippled model (0.1807 instead of its true 0.5493) were read as if
    # they were the method's output. Only the presence head may be absent: it
    # was added after some checkpoints were written and is inert at
    # presence_gate=0. Everything else is fatal.
    trainable = {n for n, prm in model.named_parameters() if prm.requires_grad}
    lost = sorted(k for k in missing if k in trainable)
    optional = [k for k in lost if k.startswith("decoder.presence.")]
    fatal = [k for k in lost if k not in optional]
    if fatal:
        raise RuntimeError(
            f"{len(fatal)}/{len(trainable)} trained tensors are missing from "
            f"{Path(checkpoint).name}, e.g. {fatal[:3]}. This checkpoint cannot "
            f"restore the model it claims to hold; evaluating it would report a "
            f"number and write predictions for a partly-random network. "
            f"Re-train, or evaluate a checkpoint written after the "
            f"trainable_state() completeness guard was added.")
    # "inert unless --presence-gate > 0" was printed and never enforced. A
    # missing presence head is a head of RANDOM weights, so a run with the gate on
    # multiplies the posterior by a random per-class prior and reports the result
    # as a measurement of the gate. Every checkpoint written before the head
    # existed hits this, which is most of them.
    eff_gate = cfg.presence_gate if presence_gate is None else presence_gate
    if optional and eff_gate > 0:
        raise RuntimeError(
            f"--presence-gate {eff_gate} was requested but the presence head is "
            f"absent from {Path(checkpoint).name} ({len(optional)} tensors "
            f"missing, e.g. {optional[:2]}). The head would be RANDOMLY "
            f"initialised, so the gate would multiply the posterior by a random "
            f"per-class prior and the score would measure noise. Evaluate the "
            f"gate only on a checkpoint trained with w_pres_head > 0.")
    if optional:
        _out(f"note: presence head absent from the checkpoint ({len(optional)} "
             f"tensors) -- pre-dates it; gate is refused, not silently inert", log)
    _out(f"loaded {len(trainable) - len(lost)}/{len(trainable)} trained tensors", log)
    model.eval()
    _out(f"checkpoint {Path(checkpoint).name} epoch {ckpt.get('epoch', '?')} "
         f"weights='{which}' tta={tta} region_vote={region_vote} "
         f"presence_gate={cfg.presence_gate if presence_gate is None else presence_gate}",
         log)
    _out(model.decoder.classifier.report(), log)
    return model, cfg


def evaluate(cfg: PrismConfig, checkpoint: str, val_manifest: Optional[str] = None,
             log=None, save_preds: Optional[str] = None, tta: bool = False,
             region_vote: bool = False, which: str = "teacher",
             limit: Optional[int] = None, presence_gate: Optional[float] = None,
             logit_adjust: Optional[float] = None,
             logit_prior: Optional[str] = None):
    model, cfg = load_for_eval(cfg, checkpoint, log, which, tta, region_vote,
                               presence_gate)
    return run_eval(model, cfg, val_manifest, log, save_preds, tta, region_vote, limit,
                    presence_gate, logit_adjust, logit_prior)


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
    ap.add_argument("--presence-gate", type=float, default=None,
                    help="soft inventory prior at inference; 0 disables, ~1.0 is the "
                         "measured-motivated default (see core/inventory.apply_presence_gate)")
    ap.add_argument("--logit-adjust", type=float, default=None,
                    help="Stage 2 class-prior term z_c - tau*log pi_c at the decision "
                         "(see core/inventory.apply_logit_adjust). tau>0 = balanced "
                         "rule, tau<0 reverses; needs tools/measure_class_priors.py "
                         "run once. Default 0 = off.")
    ap.add_argument("--logit-prior", default=None, choices=["presence", "point_share"],
                    help="which prior in class_priors.json to adjust by "
                         "(default: the config's, 'presence')")
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
                 a.tta, a.region_vote, a.which, a.limit, a.presence_gate,
                 a.logit_adjust, a.logit_prior)
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
