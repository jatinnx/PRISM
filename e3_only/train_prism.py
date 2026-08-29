"""PRISM training loop.

Design decisions that are not obvious from the code
---------------------------------------------------
EMA BY WEIGHT SWAP, NOT BY A SECOND MODEL. The teacher differs from the student
only in the ~5M trainable parameters; the 86M frozen ViT weights are identical.
Keeping a full second copy would cost ~350MB of GPU for nothing, so the shadow
holds only the trainable tensors and is swapped in for the teacher forward. The
arithmetic is identical to a duplicated model.

THE TEACHER FORWARD IS SKIPPED UNTIL IT IS USED. Nothing consumes teacher logits
before ``e_self``, so for the first 8 epochs there is no teacher pass at all. That
is roughly a third of the step time back, and it makes explicit that early
training contains no model-derived supervision whatsoever.

PROTOTYPES ARE SEEDED AT THE END OF EPOCH 0, NOT AT STEP 0. FINCH-clustering a
randomly initialised embedding would partition noise. Epoch 0 runs on the margin
point loss and the inventory, collecting annotated-pixel features as it goes;
those features are then clustered per class and become the prototypes. Only
human-clicked pixels are ever collected, so the 0% dense-label contract holds.

THE LOSS RUNS IN FP32 EVEN UNDER AMP. The forward is bf16 for speed, but the
objective contains logsumexp over 17 classes, per-image quantiles and region
means over up to 65 536 pixels; those accumulate error quickly in low precision,
and a loss that is quietly wrong by a few percent is worse than a slower one.

CHECKPOINTS STORE ONLY WHAT CHANGED. The frozen ViT is on disk already, so
saving it 8 times per run would cost ~5GB for no information. Only LoRA, stem,
and decoder tensors are written.
"""
import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .configs.prism import PrismConfig, ablation, resolve
from .core import shadow as sh
from .core.objective import ObjectiveWeights, PrismObjective
from .core.regions import AdaptiveRegionGate
from .data.class_map import CLASS_NAMES
from .data.dataset_prism import PairAugmentR, PrismDataset, collate_prism
from .model.net import PrismNet, point_embeddings


# --------------------------------------------------------------------------- #
#  EMA over the trainable parameters only                                     #
# --------------------------------------------------------------------------- #
class EmaShadow:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int):
        # a step-dependent decay stops the shadow from being dominated by its
        # random initialisation during the first few hundred steps
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        params = dict(model.named_parameters())
        for n, s in self.shadow.items():
            s.mul_(d).add_(params[n].detach().to(s.dtype), alpha=1.0 - d)

    @contextlib.contextmanager
    def swapped(self, model: torch.nn.Module):
        params = dict(model.named_parameters())
        backup = {}
        with torch.no_grad():
            for n, s in self.shadow.items():
                backup[n] = params[n].detach().clone()
                params[n].copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for n, b in backup.items():
                    params[n].copy_(b)


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def trainable_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """LoRA + stem + decoder tensors. Everything else is the released SAM."""
    keep = {}
    for k, v in model.state_dict().items():
        if "lora" in k.lower() or k.startswith("stem.") or k.startswith("decoder."):
            keep[k] = v.detach().cpu()
    return keep


def class_weights_from_points(manifest: str, num_classes: int,
                              cap: float = 4.0) -> torch.Tensor:
    """Inverse-square-root point frequency, mean-normalised, clamped to [1/cap, cap].

    The frequency is measured on the POINTS, not on pixels -- pixel frequencies
    would need dense masks. That is a feature rather than a compromise: the point
    sampler emits up to 5 clicks per present class regardless of that class's
    area, so point counts track how often a class *appears*, which is closer to
    what mIoU rewards than area is.

    Square root rather than plain inverse, and clamped: at 17 classes a raw
    inverse frequency reaches ratios above 30:1, and a weight that large turns
    the rare class into the dominant term and produces the opposite failure --
    ``field`` flooding the image instead of vanishing from it.
    """
    items = json.loads(Path(manifest).read_text())
    counts = np.zeros(num_classes, dtype=np.float64)
    for it in items:
        for _, _, c in it.get("points", []):
            ci = int(c)
            if 0 <= ci < num_classes:
                counts[ci] += 1
    seen = counts > 0
    if not seen.any():
        return torch.ones(num_classes)
    freq = counts / counts.sum()
    w = np.ones(num_classes, dtype=np.float64)
    w[seen] = np.sqrt(freq[seen].mean() / freq[seen])
    w = np.clip(w, 1.0 / cap, cap)
    w[seen] /= w[seen].mean()
    return torch.tensor(w, dtype=torch.float32)


def param_groups(model: torch.nn.Module, cfg: PrismConfig):
    lora, decay, no_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in n.lower():
            lora.append(p)
        elif p.ndim <= 1 or "log_scale" in n or n.endswith("classifier.weight"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": lora, "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0},
    ]


def lr_lambda(cfg: PrismConfig, steps_per_epoch: int):
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = max(1, cfg.lr_warmup_epochs * steps_per_epoch)

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        if not cfg.lr_use_cosine_decay:
            return 1.0
        t = (step - warm) / max(1, total - warm)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return f


def to_device(batch: Dict, device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif k == "points":
            out[k] = [p.to(device, non_blocking=True) for p in v]
        else:
            out[k] = v
    return out


def f32(out: Dict) -> Dict:
    """Cast the network's tensor outputs to fp32 for the loss."""
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
#  training                                                                   #
# --------------------------------------------------------------------------- #
def train(cfg: PrismConfig, resume: Optional[str] = None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(cfg.device)

    save_dir = Path(resolve(cfg.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{cfg.experiment}_train.log"
    metrics_path = save_dir / f"{cfg.experiment}_metrics.jsonl"
    log = log_path.open("a")

    def say(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say("=" * 78)
    say(f"{cfg.experiment}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(json.dumps({k: v for k, v in cfg.__dict__.items()}, indent=None, default=str))

    # ---- data ------------------------------------------------------- #
    region_train = Path(resolve(cfg.region_cache_train))
    if not region_train.exists():
        raise FileNotFoundError(
            f"the frozen region partition is missing: {region_train}\n"
            f"build it once (it takes a few minutes and never changes):\n"
            f"  python -m e3_only.tools.build_region_cache --split train\n"
            f"  python -m e3_only.tools.validate_regions      # sets prop_eps")
    aug = PairAugmentR(cfg.image_size, strong_brightness=cfg.strong_brightness,
                       strong_contrast=cfg.strong_contrast,
                       strong_noise_std=cfg.strong_noise_std,
                       multi_scale=cfg.multi_scale_crop,
                       crop_scale_range=(cfg.crop_scale_lo, cfg.crop_scale_hi))
    ds = PrismDataset(resolve(cfg.train_manifest), cfg.image_size, training=True,
                      region_npz=str(region_train), augment=aug,
                      num_classes=cfg.num_classes)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=collate_prism,
                    drop_last=False, pin_memory=(device.type == "cuda"),
                    persistent_workers=cfg.num_workers > 0)
    steps_per_epoch = len(dl)
    say(f"train {len(ds)} images, {steps_per_epoch} steps/epoch, batch {cfg.batch_size}")

    # ---- model ------------------------------------------------------ #
    model = PrismNet(resolve(cfg.sam_checkpoint), cfg.num_classes, str(device),
                     cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout,
                     stem_channels=cfg.stem_channels, embed_dim=cfg.embed_dim,
                     prototypes_per_class=cfg.prototypes_per_class,
                     invariant_stem=cfg.invariant_stem,
                     invariant_window=cfg.invariant_window,
                     dilated_context=cfg.dilated_context,
                     k_temperature=cfg.k_temperature, scale_init=cfg.scale_init,
                     sam_normalize=cfg.sam_normalize).to(device)
    model.decoder.classifier.ema_decay = cfg.proto_ema
    say(model.param_report())
    if not cfg.sam_normalize:
        say("NOTE: sam_normalize=False -- reproducing E3's un-normalised encoder input")

    ema = EmaShadow(model, cfg.ema_decay)
    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps_per_epoch))

    cw = class_weights_from_points(resolve(cfg.train_manifest), cfg.num_classes,
                                   cfg.rare_class_factor).to(device) \
        if cfg.class_weighting else None
    if cw is not None:
        say("class weights: " + ", ".join(f"{n}={float(v):.2f}"
                                          for n, v in zip(CLASS_NAMES, cw)))

    weights = ObjectiveWeights(
        point=cfg.w_point, prop=cfg.w_prop, absent=cfg.w_absent,
        present=cfg.w_present, area=cfg.w_area, hom=cfg.w_hom, potts=cfg.w_potts,
        bnd=cfg.w_bnd, shadow=cfg.w_shadow, shead=cfg.w_shead,
        self_train=cfg.w_self, anchor=cfg.w_anchor, repel=cfg.w_repel,
        rim=cfg.w_rim, e_hom=cfg.e_hom, e_shadow=cfg.e_shadow, e_self=cfg.e_self,
        ramp=cfg.ramp_epochs)
    gate = AdaptiveRegionGate(kappa=cfg.gate_kappa, floor=cfg.gate_floor)
    objective = PrismObjective(
        weights, cfg.num_classes, class_weight=cw, leak=cfg.inventory_leak,
        prop_eps=cfg.prop_eps, margin=cfg.margin, potts_sigma=cfg.potts_sigma,
        hom_temperature=cfg.hom_temperature, min_region=cfg.min_region,
        edge_quantile=cfg.edge_quantile, edge_radius=cfg.edge_radius, gate=gate,
        classifier=model.decoder.classifier, self_min_margin=cfg.self_min_margin,
        js_homogeneity=cfg.js_homogeneity, soft_self=cfg.soft_self,
        pres_const_k=cfg.pres_const_k)

    start_epoch = 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["student"], strict=False)
        for n, t in ck["ema"].items():
            if n in ema.shadow:
                ema.shadow[n].copy_(t.to(device))
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        gate.load_state_dict(ck.get("gate", {}))
        start_epoch = int(ck["epoch"])
        say(f"resumed from {resume} at epoch {start_epoch}")

    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if cfg.amp \
        else contextlib.nullcontext
    say(f"amp={'bf16' if cfg.amp else 'off'}  "
        f"curriculum: hom@{cfg.e_hom} shadow@{cfg.e_shadow} self@{cfg.e_self} "
        f"(ramp {cfg.ramp_epochs})")
    say(f"measured constants: inventory_leak={cfg.inventory_leak} "
        f"prop_eps={cfg.prop_eps}")

    gstep = start_epoch * steps_per_epoch
    seed_feats: List[torch.Tensor] = []
    seed_labels: List[torch.Tensor] = []
    best = {"mIoU": -1.0, "epoch": -1}

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        acc: Dict[str, float] = {}
        n_acc = 0

        for step, raw in enumerate(dl):
            batch = to_device(raw, device)
            image = batch["image_strong"]

            # -- teacher (only once anything reads it) ------------------
            teacher_out = None
            if epoch >= cfg.e_self:
                with torch.no_grad(), ema.swapped(model), amp_ctx():
                    teacher_out = f32(model(batch["image_weak"]))

            with amp_ctx():
                student = model(image)
            student = f32(student)
            student["classifier"] = model.decoder.classifier

            # -- shadowed twin -----------------------------------------
            shadow_out = shadow_mask = None
            if cfg.w_shadow > 0 and epoch >= cfg.e_shadow and \
                    (step % max(1, cfg.shadow_every) == 0):
                shadowed, shadow_mask = sh.synth_shadow(
                    image.detach(), cfg.shadow_prob, cfg.shadow_atten_lo,
                    cfg.shadow_atten_hi, cfg.shadow_blue_bias, cfg.shadow_penumbra)
                with amp_ctx():
                    shadow_out = model(shadowed)
                shadow_out = f32(shadow_out)

            loss, logd = objective(batch, student, epoch, teacher_out,
                                   shadow_out, shadow_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                                   cfg.grad_clip)
            opt.step()
            sched.step()
            gstep += 1
            ema.update(model, gstep)

            # the prototype anchor reads only human-clicked pixels
            model.decoder.classifier.update_ema(student["embed"].detach(),
                                                batch["points"], cfg.proto_patch)
            if epoch == 0 and cfg.finch_init and step < cfg.finch_init_batches:
                f, l = point_embeddings(student["embed"].detach(), batch["points"],
                                        cfg.proto_patch)
                if f is not None:
                    seed_feats.append(f.detach())
                    seed_labels.append(l)

            logd["gnorm"] = float(gnorm)
            for k, v in logd.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            n_acc += 1

            if (step + 1) % cfg.log_every == 0:
                say(f"  e{epoch:03d} s{step + 1:04d}/{steps_per_epoch} "
                    + " ".join(f"{k}={logd[k]:.3f}" for k in
                               ("total", "point", "prop", "absent", "present",
                                "area", "hom", "potts", "bnd", "shadow", "shead",
                                "self") if k in logd)
                    + f" tau={logd.get('tau', 0):.3f}"
                      f" acc={logd.get('accept', 0):.2f}"
                      f" lr={sched.get_last_lr()[1]:.2e}")

        # -- prototype seeding, once, at the end of epoch 0 ------------
        if epoch == 0 and cfg.finch_init and seed_feats:
            F_ = torch.cat(seed_feats)
            L_ = torch.cat(seed_labels)
            model.decoder.classifier.finch_init(F_, L_, cfg.finch_max_per_class)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in ema.shadow:
                        ema.shadow[n].copy_(p.detach())
            say(f"FINCH-seeded prototypes from {len(F_)} point features "
                f"({int(L_.bincount(minlength=cfg.num_classes).min())} min/class); "
                + model.decoder.classifier.report())
            seed_feats, seed_labels = [], []

        mean = {k: v / max(1, n_acc) for k, v in acc.items()}
        say(f"epoch {epoch:03d} done in {time.time() - t0:.0f}s  "
            + " ".join(f"{k}={mean[k]:.4f}" for k in sorted(mean))
            + "  " + model.decoder.classifier.report())
        rec = {"epoch": epoch, "time": time.time() - t0, **mean}

        # -- checkpoint and eval --------------------------------------
        is_last = epoch == cfg.epochs - 1
        if (epoch + 1) % cfg.save_every == 0 or is_last:
            with ema.swapped(model):
                teacher_state = trainable_state(model)
            ckpt = {"epoch": epoch + 1, "config": cfg.__dict__,
                    "student": trainable_state(model), "teacher": teacher_state,
                    "ema": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "gate": gate.state_dict()}
            path = save_dir / f"{cfg.experiment}_epoch_{epoch + 1:04d}.pt"
            torch.save(ckpt, path)
            say(f"saved {path.name}")

        if (epoch + 1) % cfg.eval_every == 0 or is_last:
            from .evaluate_prism import run_eval
            elog = (save_dir / f"{cfg.experiment}_epoch_{epoch + 1:04d}_eval.log").open("w")
            try:
                with ema.swapped(model):
                    m = run_eval(model, cfg, log=elog)
            finally:
                elog.close()
            rec.update({f"val_{k}": v for k, v in m.items()
                        if not isinstance(v, list)})
            say(f"  VAL e{epoch + 1}: mIoU={m.get('mIoU', 0):.4f} "
                f"PA={m.get('PA', 0):.4f} speckle={m.get('speckle', 0):.4f} "
                f"ghost={m.get('ghost', 0):.4f} flood={m.get('flood', 0):.4f} "
                f"trimap3={m.get('trimap3', 0):.4f}")
            miou = m.get("mIoU")
            if miou is not None and miou > best["mIoU"]:
                best = {"mIoU": miou, "epoch": epoch + 1}
                with ema.swapped(model):
                    torch.save({"epoch": epoch + 1, "config": cfg.__dict__,
                                "teacher": trainable_state(model),
                                "mIoU": miou},
                               save_dir / f"{cfg.experiment}_best.pt")
                say(f"  new best mIoU {best['mIoU']:.4f} at epoch {best['epoch']}")
            elif miou is not None and miou < best["mIoU"] - 0.02:
                say(f"  WARNING: mIoU is {best['mIoU'] - miou:.4f} below the "
                    f"epoch-{best['epoch']} best -- this is the degradation "
                    f"signature E3 showed; check the 'accept' and 'tau' columns.")

        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    say(f"done. best mIoU {best['mIoU']:.4f} at epoch {best['epoch']}")
    log.close()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="full",
                    help="full | no-inventory | no-region | no-self | soft-self | "
                         "no-shadow | no-invariant-stem | no-boundary | "
                         "single-prototype | no-margin | js-homogeneity | "
                         "const-k-present | e3-normalisation")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--leak", type=float, default=None,
                    help="measured inventory violation rate (validate_inventory.py)")
    ap.add_argument("--prop-eps", type=float, default=None,
                    help="measured propagation error rate (validate_regions.py)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    cfg = ablation(a.ablation)
    if a.epochs is not None:
        cfg.epochs = a.epochs
    if a.batch_size is not None:
        cfg.batch_size = a.batch_size
    if a.lr is not None:
        cfg.lr = a.lr
    if a.leak is not None:
        cfg.inventory_leak = a.leak
    if a.prop_eps is not None:
        cfg.prop_eps = a.prop_eps
    if a.no_amp:
        cfg.amp = False
    if a.save_dir:
        cfg.save_dir = a.save_dir
    cfg.__post_init__()
    train(cfg, a.resume)


if __name__ == "__main__":
    main()
