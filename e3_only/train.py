"""Training loop for the point-only framework (v2).

Invariant: the only human label information that touches the model during
training is the (x, y, class) point list. No dense cross-entropy, no mask key
in the training dataset — by construction.

E2 (prototypes) is built on LIVE prototypes: the bank is updated every step
from the model's own high-confidence predictions and fully refreshed from all
training points every ``proto_refresh_every`` epochs, so it tracks the
evolving encoder. The E2 supervision is cosine regularisation toward those
live centroids — no dead KL self-distillation, no unlabelled-region
distillation.

Diagnostics (every 10 steps, E2): point_CE and proto_reg components of the
loss, the number of bank pixels accepted this step, and per-class mean top-1
cosine similarity — you can SEE whether the mechanism is engaging.
"""
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .configs.base import Config, resolve
from .core.losses import (boundary_smoothness_loss, consistency_loss,
                          point_cross_entropy, pseudo_cross_entropy)
from .core.prompts import NegativePromptSampler
from .core.prototypes import PrototypeBank, MultiPrototypeBank
from .core.pseudo import make_pseudo
from .data.dataset import PairAugment, PointOnlyDataset, collate_points
from .model.sam_wrapper import PointOnlySAM
from .model.lora import clone_teacher, ema_update, trainable_parameters


def _log(msg, log=None):
    print(msg, flush=True)
    if log is not None:
        log.write(msg + "\n")
        log.flush()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(cfg: Config, device, log=None):
    """Median-frequency point weighting from the training manifest."""
    items = json.loads(Path(resolve(cfg.train_manifest)).read_text())
    counts = np.zeros(cfg.num_classes)
    for it in items:
        for _, _, c in it.get("points", []):
            if 0 <= int(c) < cfg.num_classes:
                counts[int(c)] += 1
    pos = counts > 0
    median = np.median(counts[pos]) if pos.any() else 1.0
    w = np.ones(cfg.num_classes)
    w[pos] = np.minimum(cfg.rare_class_factor, median / counts[pos])
    _log(f"class point weights: {[round(float(v), 2) for v in w]}", log)
    return torch.tensor(w, dtype=torch.float32, device=device)


def train(cfg: Config, log=None):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    augment = PairAugment(cfg.image_size, 0.05, 0.20, 0.20, 0.02,
                           multi_scale=cfg.multi_scale_crop,
                           crop_scale_range=(cfg.crop_scale_lo, cfg.crop_scale_hi))
    ds = PointOnlyDataset(resolve(cfg.train_manifest), cfg.image_size, True, augment)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=collate_points,
                    pin_memory=(device.type == "cuda"))

    student = PointOnlySAM(resolve(cfg.sam_checkpoint), cfg.num_classes, str(device),
                           cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout,
                           cfg.background_class,
                           spatial_context=cfg.spatial_context).to(device)
    teacher = clone_teacher(student).to(device) if cfg.use_teacher_student else None
    optimizer = torch.optim.AdamW(trainable_parameters(student), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    prompt_sampler = NegativePromptSampler(cfg.max_negative_points)

    # ---- Learning Rate Schedule ----
    # Why: batch_size=1 means very noisy gradients (only ~15 points per image).
    # Warmup prevents large initial jumps, cosine decay helps settle into a minimum.
    # Schedule:  warmup (epochs 0-1) -> peak lr -> cosine decay -> near zero at end
    def warmup_cosine_lr(epoch):
        """Returns a multiplier (0..1) for the base learning rate.

        Phase 1 - Linear warmup (epoch 0..warmup-1):
            lr increases linearly from 10% to 100% of base lr.
            Starting at 10% (not 0) ensures the model learns from epoch 1.
            This avoids large, destabilizing updates at the start.

        Phase 2 - Cosine decay (epoch warmup..epochs-1):
            lr follows a cosine curve from lr_peak down to ~0.
            This lets the model settle into a sharp minimum.
        """
        warmup = cfg.lr_warmup_epochs
        total = cfg.epochs
        if warmup <= 0 or not cfg.lr_use_cosine_decay:
            return 1.0  # no schedule: constant lr
        if epoch < warmup:
            # Phase 1: linear warmup from 0.1 to 1.0 (NOT from 0)
            return 0.1 + 0.9 * (epoch / warmup)
        # Phase 2: cosine decay from 1.0 to ~0
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)

    # E3: bank is used for prototypes (E2) and teacher pseudo-labels (E3)
    # Ported from PointOnlySAM-research01-fixed: use MultiPrototypeBank when configured
    need_bank = cfg.use_prototypes or cfg.use_teacher_student
    bank = None
    if need_bank:
        if getattr(cfg, 'use_multi_prototypes', False):
            # MultiPrototypeBank: K prototypes per class (handles multi-modal distributions)
            # Ported from PointOnlySAM-research01-fixed/pointonlysam/fixed_objectives.py
            bank = MultiPrototypeBank(
                cfg.num_classes, 256, str(device),
                prototypes_per_class=cfg.prototypes_per_class,
                ema=cfg.proto_ema,
                bank_capacity=cfg.bank_capacity,
                sim_threshold=cfg.proto_sim_threshold)
            _log(f"[{cfg.experiment}] using MultiPrototypeBank (K={cfg.prototypes_per_class} per class)", log)
        else:
            # Original PrototypeBank: single prototype per class
            bank = PrototypeBank(cfg.num_classes, 256, cfg.proto_ema, cfg.proto_feature_patch,
                                 cfg.image_size, str(device), cfg.bank_capacity,
                                 cfg.proto_sim_threshold)

    class_weights = None
    if cfg.class_weighting:
        class_weights = build_class_weights(cfg, device, log)

    save_dir = Path(resolve(cfg.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- prototype initialisation over the ENTIRE training set (epoch 0) ----
    if bank is not None:
        init_model = teacher if teacher is not None else student
        t0 = time.time()
        _log(f"[{cfg.experiment}] initialising prototypes from ALL training points...", log)
        bank.initialize_from_dataset(init_model, ds.items, cfg.image_size, str(device))
        bank.refresh_epoch = 0
        _log(f"[{cfg.experiment}] prototype init took {time.time() - t0:.1f}s", log)

    _log(f"[{cfg.experiment}] trainable params: "
         f"{sum(p.numel() for p in trainable_parameters(student))}", log)

    for epoch in range(cfg.epochs):
        student.train()
        if teacher is not None:
            teacher.eval()
        if cfg.pseudo_warmup_cosine:
            # Cosine warmup: smooth 0→1 over pseudo_warmup_epochs
            t = min(1.0, epoch / max(1, cfg.pseudo_warmup_epochs))
            l_pseudo_eff = cfg.l_pseudo * 0.5 * (1.0 - np.cos(np.pi * t))
        else:
            l_pseudo_eff = cfg.l_pseudo * min(1.0, epoch / max(1, cfg.pseudo_warmup_epochs))
        ep_loss = 0.0
        ep_point_ce = 0.0
        ep_proto_reg = 0.0
        ep_bank_px = 0
        n_steps = 0
        t0 = time.time()
        for step, batch in enumerate(dl):
            weak = batch["image_weak"].to(device, non_blocking=True)
            strong = batch["image_strong"].to(device, non_blocking=True)
            points = batch["points"][0].to(device)

            # ---- teacher pseudo-label generation (one encoder call, E3+) ----
            t_feat = t_sem = sam_w = fused = pseudo = conf_map = valid = None
            a = b = None
            if teacher is not None:
                with torch.no_grad():
                    t_feat = teacher.encode(weak)
                    t_sem = teacher.semantic_logits(t_feat, weak.shape[-2:])
                    if cfg.use_sam_prompt_masks:
                        # E3: use positive prompts only (no negative prompts)
                        sam_w = teacher.sam_class_logits(
                            t_feat, weak.shape[-2], points, prompt_sampler,
                            use_negative=False)
                    pseudo, conf_map, valid, fused, a, b = make_pseudo(
                        t_sem, sam_w, t_feat, bank, weak, points, cfg, epoch)

            # ---- student forward (one encoder call) ----
            s_feat = student.encode(strong)
            s_sem = student.semantic_logits(s_feat, strong.shape[-2:])

            loss = cfg.l_point * point_cross_entropy(s_sem, points, class_weights)
            point_ce = loss.item()

            proto_reg = torch.zeros((), device=device)
            n_bank_px = 0
            if bank is not None and cfg.use_prototypes and teacher is None:
                # E2: LIVE prototypes — update from the model's own confident
                # predictions every step, then pull features toward the
                # (now-fresh) centroids on those same confident pixels.
                n_bank_px = bank.update_from_predictions(
                    s_feat, s_sem, cfg.proto_self_conf_threshold,
                    cfg.proto_sim_threshold)
                proto_reg = cfg.l_proto * bank.cosine_reg_self(
                    s_feat, s_sem, cfg.proto_self_conf_threshold)
                loss = loss + proto_reg
            elif teacher is not None and valid is not None and valid.any():
                # E3: pseudo-label loss + consistency loss
                loss = loss + l_pseudo_eff * pseudo_cross_entropy(s_sem, pseudo, conf_map, valid)
                loss = loss + cfg.l_consistency * consistency_loss(s_sem, fused)
                # Note: E4+ proto_reg and E7 structural_loss are NOT used in E3

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(student), cfg.grad_clip)
            optimizer.step()
            if teacher is not None:
                ema_update(student, teacher, cfg.ema_decay)

            ep_loss += loss.item()
            ep_point_ce += point_ce
            ep_proto_reg += proto_reg.item()
            ep_bank_px += n_bank_px
            n_steps += 1
            if step % 10 == 0:
                a_s = "n/a" if a is None else f"{a:.3f}"
                b_s = "n/a" if b is None else f"{b:.3f}"
                proto_s = ""
                if bank is not None and cfg.use_prototypes and teacher is None:
                    st = bank.engagement_stats(s_feat, s_sem.argmax(1))
                    top1s = ",".join(f"{v:.2f}" for v in st["per_class_top1"][:6])
                    proto_s = (f" proto_px={n_bank_px} "
                               f"point_CE={point_ce:.4f} proto_reg={proto_reg.item():.4f} "
                               f"top1[:6]=[{top1s}]")
                _log(f"[{cfg.experiment}] ep {epoch + 1}/{cfg.epochs} step {step} "
                     f"loss {loss.item():.4f} (pseudo_w={l_pseudo_eff:.2f}) "
                     f"gate_px {int(valid.sum()) if valid is not None else 0} "
                     f"a={a_s} b={b_s}{proto_s}", log)

        # ---- Step the learning rate scheduler (once per epoch) ----
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()  # advance to next epoch's lr
        _log(f"[{cfg.experiment}] ep {epoch + 1}/{cfg.epochs} mean_loss "
             f"{ep_loss / max(1, n_steps):.4f} "
             f"(point_CE {ep_point_ce / max(1, n_steps):.4f} | "
             f"proto_reg {ep_proto_reg / max(1, n_steps):.4f} | "
             f"bank_px {ep_bank_px // max(1, n_steps)}) "
             f"lr={current_lr:.6f} "
             f"({time.time() - t0:.0f}s)", log)

        # ---- prototypes must track the evolving encoder: full refresh every
        # N epochs from ALL training points with the CURRENT weights. ----
        if (bank is not None and cfg.proto_refresh_every > 0
                and (epoch + 1) % cfg.proto_refresh_every == 0):
            refresh_model = teacher if teacher is not None else student
            _log(f"[{cfg.experiment}] refreshing prototypes from ALL training points "
                 f"(epoch {epoch + 1})...", log)
            bank.refresh(refresh_model, ds.items, cfg.image_size, str(device),
                         epoch + 1)

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = {
                "experiment": cfg.experiment,
                "epoch": epoch + 1,
                "config": asdict(cfg),
                "student": student.state_dict(),
                "teacher": teacher.state_dict() if teacher is not None else None,
            }
            if bank is not None:
                ckpt["prototypes"] = bank.prototypes.detach().cpu()
                ckpt["prototypes_initialized"] = bank.initialized.detach().cpu()
                ckpt["prototype_refresh_epoch"] = bank.refresh_epoch
            path = save_dir / f"{cfg.experiment}_epoch_{epoch + 1:04d}.pt"
            torch.save(ckpt, path)
            _log(f"[{cfg.experiment}] saved {path}", log)
