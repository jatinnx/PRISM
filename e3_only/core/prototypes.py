"""Class prototypes derived from point annotations — with CORRECT lifetime.

The old codebase's fatal flaw: prototypes were initialised from epoch-0 encoder
features and then frozen for the entire run. As LoRA shifted the feature space,
the prototypes went stale — the distillation targets drifted away from reality
and eval-time refinement with stale prototypes corrupted predictions.

This bank fixes the lifecycle with three cooperating mechanisms:

* ``initialize_from_dataset`` — epoch-0 init from ALL training points.
* ``update_from_predictions``   — EVERY training step, feed the model's own
  high-confidence predictions (softmax max > threshold) into the per-class
  FIFO bank and EMA-update the prototypes. No teacher needed. This is the
  fundamental staleness fix: prototypes track the encoder continuously.
* ``refresh``                   — full re-initialisation from all training
  points with the CURRENT encoder every N epochs (e.g. 10), keeping prototypes
  anchored to real point features rather than drifting on self-predictions.

``refine_consistency`` is GONE. It was a KL self-distillation toward
prototype-refined predictions that (a) barely fired and (b) with stale
prototypes taught the decoder wrong labels. The correct use of live prototypes
is cosine regularisation (``cosine_reg_self``), which aligns the student's
features with the class centroids on confident pixels — no distillation of
unlabelled regions, no way to teach phantom classes.
"""
import time

import torch
import torch.nn.functional as F
from collections import deque


# ======================================================================== #
# MultiPrototypeBank — ported from PointOnlySAM-research01-fixed           #
# ======================================================================== #
# This class maintains K prototypes per class to handle multi-modal visual
# distributions (e.g., chaparral can be green shrub OR dry brown).
# Instead of one averaged prototype that represents neither mode well,
# each pixel is assigned to its NEAREST sub-prototype.
#
# Original location: PointOnlySAM-research01-fixed/pointonlysam/fixed_objectives.py
# Adapted for E3: added FIFO bank, per-pixel cosine reg, and live updates.
# ======================================================================== #

class MultiPrototypeBank:
    """K prototypes per class, updated from human point features.

    Each incoming labeled point is assigned to the nearest prototype of its
class. This preserves multiple modes within a semantic class instead of
collapsing all appearances to one vector.

    Original: PointOnlySAM-research01-fixed/pointonlysam/fixed_objectives.py
    Adapted for E3: added FIFO bank, per-pixel cosine reg, and live updates.
    """

    def __init__(self, num_classes: int, feat_dim: int, device: str = "cuda",
                 prototypes_per_class: int = 4, ema: float = 0.90,
                 temperature: float = 0.15, bank_capacity: int = 512,
                 sim_threshold: float = 0.50):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.k = prototypes_per_class
        self.ema = ema
        self.temperature = temperature
        self.sim_threshold = sim_threshold
        self.device = device
        # K prototypes per class: (num_classes, K, feat_dim)
        self.prototypes = torch.zeros(num_classes, self.k, feat_dim, device=device)
        self.seen = torch.zeros(num_classes, self.k, dtype=torch.bool, device=device)
        # Backing field for compatibility property
        self._initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)
        # Per-class-slot FIFO bank for live updates
        self.bank = {c: {s: deque(maxlen=bank_capacity) for s in range(self.k)}
                     for c in range(num_classes)}

    @property
    def initialized(self):
        """Compatibility property: True if any sub-prototype has been seen for that class.
        Used by make_pseudo() which expects bank.initialized."""
        self._initialized = self.seen.any(dim=1)
        return self._initialized

    @torch.no_grad()
    def initialize_from_dataset(self, model, items, image_size: int, device: str):
        """Epoch-0 init: assign each point feature to nearest sub-prototype.

        Original: PointOnlySAM-research01-fixed MultiPrototypeBank.update()
        Adapted: batch processing over all training images.
        """
        from PIL import Image
        import numpy as np

        sums = torch.zeros(self.num_classes, self.k, self.feat_dim, device=device)
        counts = torch.zeros(self.num_classes, self.k, device=device)
        n_imgs = 0

        for item in items:
            image = Image.open(item["image"]).convert("RGB")
            image = image.resize((image_size, image_size), Image.BILINEAR)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image = image.to(device)

            points = item.get("points", [])
            if not points:
                continue

            feat = model.encode(image)  # (1, 256, 64, 64)
            b, d, h, w = feat.shape

            for x, y, c in points:
                c = int(c)
                if c < 0 or c >= self.num_classes:
                    continue
                # Map point coordinates to feature map coordinates
                fx = int(float(x) * w / image_size)
                fy = int(float(y) * h / image_size)
                fx = min(w - 1, max(0, fx))
                fy = min(h - 1, max(0, fy))
                feat_vec = F.normalize(feat[0, :, fy, fx], dim=0)

                # Find nearest sub-prototype or next empty slot
                if not self.seen[c].any():
                    slot = 0
                elif not self.seen[c].all():
                    slot = int((~self.seen[c]).nonzero(as_tuple=False)[0].item())
                else:
                    sims = torch.mv(F.normalize(self.prototypes[c], dim=1), feat_vec)
                    slot = int(sims.argmax().item())

                sums[c, slot] += feat_vec
                counts[c, slot] += 1
                self.seen[c, slot] = True

            n_imgs += 1

        # Average and normalize
        for c in range(self.num_classes):
            for s in range(self.k):
                if counts[c, s] > 0:
                    self.prototypes[c, s] = F.normalize(sums[c, s] / counts[c, s], dim=0)

        # Seed FIFO banks with individual features (not just mean)
        # This gives the bank real variance from day 1
        # Original: PointOnlySAM-research01-fixed seeds with mean only
        # Improved: seeds with ALL individual point features
        for item in items:
            image = Image.open(item["image"]).convert("RGB")
            image = image.resize((image_size, image_size), Image.BILINEAR)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image = image.to(device)
            points = item.get("points", [])
            if not points:
                continue
            feat = model.encode(image)
            b, d, h, w = feat.shape
            for x, y, c in points:
                c = int(c)
                if c < 0 or c >= self.num_classes or not self.seen[c].any():
                    continue
                fx = min(w - 1, max(0, int(float(x) * w / image_size)))
                fy = min(h - 1, max(0, int(float(y) * h / image_size)))
                feat_vec = F.normalize(feat[0, :, fy, fx], dim=0)
                # Assign to nearest sub-prototype and add to its bank
                sims = torch.mv(F.normalize(self.prototypes[c], dim=1), feat_vec)
                slot = int(sims.argmax().item())
                self.bank[c][slot].append(feat_vec)

        n_classes = int(self.seen.any(dim=1).sum())
        n_slots = int(self.seen.sum())
        print(f"[multi-prototypes] initialized {n_classes} classes, {n_slots} sub-prototypes "
              f"from {sum(sum(len(self.bank[c][s]) for s in range(self.k)) for c in range(self.num_classes))} bank features")

    @torch.no_grad()
    def refresh(self, model, items, image_size: int, device: str, epoch: int):
        """Full re-initialisation with CURRENT encoder weights."""
        t0 = time.time()
        # Reset
        self.prototypes.zero_()
        self.seen.zero_()
        for c in range(self.num_classes):
            for s in range(self.k):
                self.bank[c][s].clear()
        self.initialize_from_dataset(model, items, image_size, device)
        print(f"[multi-prototypes] refresh at epoch {epoch} took {time.time() - t0:.1f}s")

    @torch.no_grad()
    def update_from_predictions(self, feat: torch.Tensor, semantic_logits: torch.Tensor,
                                conf_threshold: float = 0.85, sim_threshold: float = 0.50,
                                max_pixels_per_class: int = 256) -> int:
        """Live update: feed model's high-confidence predictions into bank.

        Each pixel is assigned to its NEAREST sub-prototype of its predicted class.
        This preserves multi-modal distributions.
        """
        b, d, h, w = feat.shape
        probs = torch.softmax(semantic_logits, dim=1)
        if probs.shape[-2:] != (h, w):
            probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
        conf_map = probs.max(1).values
        labels = probs.argmax(1)
        valid = conf_map >= conf_threshold
        if not valid.any():
            return 0

        feat_flat = F.normalize(feat[0].permute(1, 2, 0).reshape(-1, d), dim=1)
        labels_flat = labels[0].reshape(-1)
        conf_flat = conf_map[0].reshape(-1)
        valid_flat = valid[0].reshape(-1)
        proto = F.normalize(self.prototypes, dim=2)  # (C, K, D)
        n_accepted = 0

        for cls in range(self.num_classes):
            if not self.seen[cls].any():
                continue
            idx = torch.where((labels_flat == cls) & valid_flat &
                              (conf_flat >= conf_threshold))[0]
            if idx.numel() == 0:
                continue
            z = feat_flat[idx]
            # Filter by similarity to nearest sub-prototype
            sims_all = z @ proto[cls].t()  # (N, K)
            best_sim, best_slot = sims_all.max(dim=1)  # (N,)
            mask = best_sim >= sim_threshold
            z = z[mask]
            best_slot = best_slot[mask]
            if z.numel() == 0:
                continue
            if z.shape[0] > max_pixels_per_class:
                perm = torch.randperm(z.shape[0], device=z.device)[:max_pixels_per_class]
                z = z[perm]
                best_slot = best_slot[perm]
            # Add each pixel to its nearest sub-prototype's bank
            for vec, slot_idx in zip(z.tolist(), best_slot.tolist()):
                self.bank[cls][slot_idx].append(torch.tensor(vec, device=self.device))
            # EMA update each sub-prototype from its bank
            for s in range(self.k):
                if self.bank[cls][s]:
                    bank_mean = F.normalize(torch.stack(list(self.bank[cls][s])).mean(0), dim=0)
                    self.prototypes[cls, s] = F.normalize(
                        self.ema * self.prototypes[cls, s] + (1 - self.ema) * bank_mean, dim=0)
            n_accepted += z.shape[0]
        return n_accepted

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of every pixel to nearest sub-prototype per class.

        Returns (1, C, H, W) scores — max over K sub-prototypes per class.
        """
        b, d, h, w = feat.shape
        z = F.normalize(feat, dim=1)  # (B, D, H, W)
        p = F.normalize(self.prototypes, dim=2)  # (C, K, D)
        # (B, D, H, W) x (C, K, D) -> (B, C, K, H, W) -> max over K -> (B, C, H, W)
        sim = torch.einsum("bdhw,ckd->bckhw", z, p)  # (B, C, K, H, W)
        sim = sim.masked_fill(~self.seen[None, :, :, None, None], -1e4)
        scores = sim.max(dim=2).values / self.temperature  # (B, C, H, W)
        return scores

    def prototype_confidence(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-pixel confidence: cosine sim to nearest sub-prototype of predicted class."""
        b, d, h, w = feat.shape
        z = F.normalize(feat, dim=1)
        p = F.normalize(self.prototypes, dim=2)  # (C, K, D)
        sim = torch.einsum("bdhw,ckd->bckhw", z, p)  # (B, C, K, H, W)
        sim = sim.masked_fill(~self.seen[None, :, :, None, None], -1e4)
        best_sim = sim.max(dim=2).values  # (B, C, H, W) — best sub-prototype per class
        lab = labels
        if lab.shape[-2:] != (h, w):
            lab = F.interpolate(lab.float().unsqueeze(1), size=(h, w), mode="nearest").long()[:, 0]
        lab = lab.clamp(0, self.num_classes - 1)
        conf = best_sim.gather(1, lab.unsqueeze(1))[:, 0]  # (B, H, W)
        conf = (conf + 1.0) / 2.0  # map [-1, 1] to [0, 1]
        return conf

    def cosine_reg_self(self, feat: torch.Tensor, semantic_logits: torch.Tensor,
                        threshold: float = 0.85) -> torch.Tensor:
        """Per-pixel cosine regularization — pull each confident pixel toward
        its NEAREST sub-prototype (not a class mean).

        Original (per-class mean): one gradient path per class
        Improved (per-pixel): N gradient paths per class — smoother loss surface.

        Ported from PointOnlySAM-research01-fixed, adapted for multi-prototype.
        """
        b, d, h, w = feat.shape
        probs = torch.softmax(semantic_logits, dim=1)
        if probs.shape[-2:] != (h, w):
            probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
        conf_map = probs.max(1).values
        labels = probs.argmax(1)
        valid = conf_map >= threshold
        z = F.normalize(feat[0].permute(1, 2, 0).reshape(-1, d), dim=1)  # (N, D)
        y = labels[0].reshape(-1)  # (N,)
        c = conf_map[0].reshape(-1)  # (N,)
        v = valid[0].reshape(-1)  # (N,)
        p = F.normalize(self.prototypes, dim=2)  # (C, K, D)
        all_losses = []

        for cls in range(self.num_classes):
            if not self.seen[cls].any():
                continue
            idx = torch.where((y == cls) & v & (c >= threshold))[0]
            if idx.numel() == 0:
                continue
            z_cls = z[idx]  # (N_cls, D)
            sims = z_cls @ p[cls].t()  # (N_cls, K) — sim to each sub-prototype
            best_sim, _ = sims.max(dim=1)  # (N_cls,) — sim to nearest sub-prototype
            # Per-pixel loss: each pixel independently pulled toward its nearest
            # This produces N gradient paths instead of 1 averaged path
            all_losses.append((1.0 - best_sim).mean())

        if not all_losses:
            return feat.sum() * 0.0
        return torch.stack(all_losses).mean()

    def point_margin_loss(self, features: torch.Tensor, points: list[torch.Tensor],
                           image_size: int, margin: float = 0.15) -> torch.Tensor:
        """Point margin loss — ported from PointOnlySAM-research01-fixed.

        For each annotated point, maximize gap between:
          - similarity to own class's best sub-prototype (positive)
          - similarity to nearest competing class's sub-prototype (negative)

        Original: PointOnlySAM-research01-fixed/pointonlysam/fixed_objectives.py
        """
        terms = []
        _, _, h, w = features.shape
        p = F.normalize(self.prototypes, dim=2)  # (C, K, D)
        for bi, sample in enumerate(points):
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample, device=features.device)
            for x, y, c in sample.tolist():
                c = int(c)
                if c < 0 or c >= self.num_classes or not self.seen[c].any():
                    continue
                xx = min(w - 1, max(0, int(x * w / image_size)))
                yy = min(h - 1, max(0, int(y * h / image_size)))
                feat = F.normalize(features[bi, :, yy, xx], dim=0)
                sim = torch.einsum("ckd,d->ck", p, feat).masked_fill(~self.seen, -1e4).max(1).values
                pos = sim[c]
                neg = sim.masked_fill(torch.arange(self.num_classes, device=sim.device) == c, -1e4).max()
                terms.append(F.relu(margin - (pos - neg)))
        return torch.stack(terms).mean() if terms else features.sum() * 0.0

    @torch.no_grad()
    def engagement_stats(self, feat: torch.Tensor, labels=None):
        """Diagnostics: mean top-1 cosine per class."""
        if not self.seen.any():
            return {"per_class_top1": [0.0] * self.num_classes, "pixels_fraction": 0.0}
        sims = self.logits(feat)  # (1, C, Hf, Wf)
        top1 = sims.max(1).values
        per_class = []
        if labels is not None:
            if labels.shape[-2:] != top1.shape[-2:]:
                lab = F.interpolate(labels.float().unsqueeze(1), size=top1.shape[-2:],
                                    mode="nearest").long()[:, 0]
            else:
                lab = labels
            for c in range(self.num_classes):
                m = (lab == c)
                per_class.append(float(top1[m].mean()) if m.any() else float("nan"))
        else:
            per_class = [float(top1.mean())] * self.num_classes
        return {"per_class_top1": per_class, "pixels_fraction": float(top1.mean())}

    def state_dict(self) -> dict:
        return {
            "prototypes": self.prototypes.cpu(),
            "seen": self.seen.cpu(),
            "k": self.k,
        }

    def load_state_dict(self, state: dict) -> None:
        self.prototypes.copy_(state["prototypes"].to(self.prototypes.device))
        self.seen.copy_(state["seen"].to(self.seen.device))


class PrototypeBank:
    def __init__(self, num_classes: int, feat_dim: int, ema: float = 0.95,
                 patch: int = 3, image_size: int = 256, device: str = "cuda",
                 bank_capacity: int = 512, sim_threshold: float = 0.30):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.ema = ema
        self.patch = patch
        self.image_size = image_size
        self.device = device
        self.sim_threshold = sim_threshold
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.refresh_epoch = 0                 # epoch prototypes were last (re)initialised
        # per-class FIFO memory bank of L2-normalized features
        self.bank = {c: deque(maxlen=bank_capacity) for c in range(num_classes)}

    # ------------------------------------------------------------------ #
    #  point feature extraction                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _point_features(self, feat: torch.Tensor, points: torch.Tensor):
        """Patch-pooled, L2-normalized features at each labelled point."""
        b, d, h, w = feat.shape
        k = max(1, self.patch // 2)
        pooled = F.avg_pool2d(feat, kernel_size=self.patch, stride=1, padding=k)
        rows = []
        for x, y, c in points.detach().cpu().tolist():
            if int(c) < 0 or int(c) >= self.num_classes:
                continue
            fx = float(x) * (w / float(self.image_size))
            fy = float(y) * (h / float(self.image_size))
            gx = (fx / max(1.0, (w - 1))) * 2.0 - 1.0
            gy = (fy / max(1.0, (h - 1))) * 2.0 - 1.0
            grid = torch.tensor([[[[gx, gy]]]], device=feat.device, dtype=feat.dtype)
            z = F.grid_sample(pooled, grid, mode="bilinear", align_corners=True)[0, :, 0, 0]
            rows.append((int(c), z))
        return rows

    # ------------------------------------------------------------------ #
    #  whole-training-set (re)initialisation                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def initialize_from_dataset(self, model, items, image_size: int, device: str):
        """Aggregate point features over ALL training images -> one robust
        prototype per class. Used at epoch 0 and by ``refresh``."""
        sums = torch.zeros(self.num_classes, self.feat_dim, device=device)
        counts = torch.zeros(self.num_classes, device=device)
        n_imgs = 0
        for item in items:
            image = _load_image(item["image"], image_size, device)
            points = torch.tensor(_scale_points(item.get("points", []), item, image_size),
                                  dtype=torch.float32, device=device)
            if len(points) == 0:
                continue
            feat = model.encode(image)          # one encoder call per image
            for cls, z in self._point_features(feat, points):
                sums[cls] += z
                counts[cls] += 1
            n_imgs += 1
        for c in range(self.num_classes):
            if counts[c] > 0:
                self.prototypes[c] = F.normalize(sums[c] / counts[c], dim=0)
                self.initialized[c] = True
        # seed the FIFO bank with the init features so EMA starts from real data
        for c in range(self.num_classes):
            self.bank[c].clear()
            if counts[c] > 0:
                self.bank[c].append(F.normalize(sums[c] / counts[c], dim=0))
        print(f"[prototypes] initialised {int(counts.gt(0).sum())}/{self.num_classes} "
              f"classes from {int(counts.sum())} labelled points over {n_imgs} images")

    @torch.no_grad()
    def refresh(self, model, items, image_size: int, device: str, epoch: int):
        """Full re-initialisation from ALL training points with the CURRENT
        encoder. Call every ``proto_refresh_every`` epochs so prototypes stay
        anchored to real point features."""
        t0 = time.time()
        self.initialize_from_dataset(model, items, image_size, device)
        self.refresh_epoch = epoch
        print(f"[prototypes] refresh at epoch {epoch} took {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    #  cosine machinery                                                  #
    # ------------------------------------------------------------------ #
    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of every pixel feature to every class prototype."""
        p = F.normalize(self.prototypes, dim=1)
        z = F.normalize(feat, dim=1)
        return torch.einsum("bdhw,cd->bchw", z, p)

    def prototype_confidence(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-pixel prototype evidence: cosine sim of each pixel to the
        prototype of its (predicted) class, mapped to [0, 1]. Uninitialised-class
        pixels get 0.0 (no evidence), not 0.5."""
        b, d, h, w = feat.shape
        z = F.normalize(feat, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        sims = torch.einsum("bdhw,cd->bchw", z, p)
        lab = labels
        if lab.shape[-2:] != (h, w):
            lab = F.interpolate(lab.float().unsqueeze(1), size=(h, w),
                                mode="nearest").long()[:, 0]
        lab = lab.clamp(0, self.num_classes - 1)
        conf = sims.gather(1, lab.unsqueeze(1))[:, 0]
        conf = (conf + 1.0) / 2.0
        uninit = ~self.initialized[lab]
        conf = conf * uninit.logical_not().float()
        return conf

    # ------------------------------------------------------------------ #
    #  LIVE self-supervised updates (the staleness fix)                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_from_predictions(self, feat: torch.Tensor, semantic_logits: torch.Tensor,
                                conf_threshold: float = 0.85, sim_threshold: float = 0.30,
                                max_pixels_per_class: int = 256) -> int:
        """Every training step: enqueue the model's OWN high-confidence
        predictions (softmax max > threshold) into the FIFO bank and EMA-update
        the prototypes. No teacher, no pseudo-label machinery. Returns the
        number of pixels accepted (for diagnostics)."""
        b, d, h, w = feat.shape
        probs = torch.softmax(semantic_logits, dim=1)
        # semantic_logits are at IMAGE resolution (256x256) while feat is at
        # FEATURE resolution (64x64). Align probs to the feature map so the
        # per-pixel indices match feat_flat — otherwise the first confident
        # pixel crashes with an out-of-bounds gather (device-side assert).
        if probs.shape[-2:] != (h, w):
            probs = F.interpolate(probs, size=(h, w), mode="bilinear",
                                  align_corners=False)
        conf_map = probs.max(1).values
        labels = probs.argmax(1)
        valid = conf_map >= conf_threshold
        if not valid.any():
            return 0
        feat_flat = F.normalize(feat[0].permute(1, 2, 0).reshape(-1, d), dim=1)
        labels_flat = labels[0].reshape(-1)
        conf_flat = conf_map[0].reshape(-1)
        valid_flat = valid[0].reshape(-1)
        proto = F.normalize(self.prototypes, dim=1)
        n_accepted = 0
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((labels_flat == cls) & valid_flat &
                              (conf_flat >= conf_threshold))[0]
            if idx.numel() == 0:
                continue
            z = feat_flat[idx]
            sims = (z @ proto[cls]).clamp(-1, 1)
            z = z[sims >= sim_threshold]
            if z.numel() == 0:
                continue
            if z.shape[0] > max_pixels_per_class:
                z = z[torch.randperm(z.shape[0], device=z.device)[:max_pixels_per_class]]
            for vec in z.tolist():
                self.bank[cls].append(torch.tensor(vec, device=self.device))
            if self.bank[cls]:
                bank_mean = F.normalize(torch.stack(list(self.bank[cls])).mean(0), dim=0)
                self.prototypes[cls] = F.normalize(
                    self.ema * self.prototypes[cls] + (1 - self.ema) * bank_mean, dim=0)
                n_accepted += z.shape[0]
        return n_accepted

    @torch.no_grad()
    def update_from_pixels(self, feat: torch.Tensor, labels: torch.Tensor,
                           conf_map: torch.Tensor, valid: torch.Tensor,
                           pixel_threshold: float = 0.80,
                           sim_threshold: float = 0.30,
                           max_pixels_per_class: int = 256) -> int:
        """E4: enqueue gated TEACHER pseudo-label pixels into the FIFO bank.
        Same structure as update_from_predictions but uses teacher features
        and teacher pseudo-labels (not self-predictions).
        Returns the number of pixels accepted."""
        b, d, h, w = feat.shape
        if labels.shape[-2:] != (h, w):
            labels = F.interpolate(labels.float().unsqueeze(1), size=(h, w),
                                   mode="nearest").long()[:, 0]
            conf_map = F.interpolate(conf_map.unsqueeze(1), size=(h, w),
                                     mode="bilinear", align_corners=False)[:, 0]
            valid = F.interpolate(valid.float().unsqueeze(1), size=(h, w),
                                  mode="nearest").bool()[:, 0]
        feat_flat = F.normalize(feat[0].permute(1, 2, 0).reshape(-1, d), dim=1)
        labels_flat = labels[0].reshape(-1)
        conf_flat = conf_map[0].reshape(-1)
        valid_flat = valid[0].reshape(-1)
        proto = F.normalize(self.prototypes, dim=1)
        n_accepted = 0
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((labels_flat == cls) & valid_flat &
                              (conf_flat >= pixel_threshold))[0]
            if idx.numel() == 0:
                continue
            z = feat_flat[idx]
            sims = (z @ proto[cls]).clamp(-1, 1)
            z = z[sims >= sim_threshold]
            if z.numel() == 0:
                continue
            if z.shape[0] > max_pixels_per_class:
                z = z[torch.randperm(z.shape[0], device=z.device)[:max_pixels_per_class]]
            for vec in z.tolist():
                self.bank[cls].append(torch.tensor(vec, device=self.device))
            if self.bank[cls]:
                bank_mean = F.normalize(torch.stack(list(self.bank[cls])).mean(0), dim=0)
                self.prototypes[cls] = F.normalize(
                    self.ema * self.prototypes[cls] + (1 - self.ema) * bank_mean, dim=0)
                n_accepted += z.shape[0]
        return n_accepted

    # ------------------------------------------------------------------ #
    #  cosine regularisation (the E2 supervision, replaces distillation) #
    # ------------------------------------------------------------------ #
    def cosine_reg_self(self, feat: torch.Tensor, semantic_logits: torch.Tensor,
                        threshold: float = 0.85) -> torch.Tensor:
        """Align the mean student feature of confident self-predicted pixels
        with the (detached) class prototype — a pull toward live centroids at
        locations the model itself is sure about. No teacher, no unlabelled
        distillation. Uses softmax-max confidence (not a gate)."""
        b, d, h, w = feat.shape
        probs = torch.softmax(semantic_logits, dim=1)
        # same resolution alignment as update_from_predictions (see above)
        if probs.shape[-2:] != (h, w):
            probs = F.interpolate(probs, size=(h, w), mode="bilinear",
                                  align_corners=False)
        conf_map = probs.max(1).values
        labels = probs.argmax(1)
        valid = conf_map >= threshold
        z = feat[0].permute(1, 2, 0).reshape(-1, d)
        y = labels[0].reshape(-1)
        c = conf_map[0].reshape(-1)
        v = valid[0].reshape(-1)
        losses = []
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((y == cls) & v & (c >= threshold))[0]
            if idx.numel() < 2:
                continue
            zp = F.normalize(z[idx].mean(0), dim=0)
            ref = F.normalize(self.prototypes[cls].detach(), dim=0)
            losses.append(1.0 - torch.sum(zp * ref))
        if not losses:
            return feat.sum() * 0.0
        return torch.stack(losses).mean()

    def cosine_reg(self, feat: torch.Tensor, labels: torch.Tensor,
                    conf_map: torch.Tensor, valid: torch.Tensor,
                    threshold: float) -> torch.Tensor:
        """E4: align the mean student feature of gated TEACHER pseudo pixels
        with the (detached) class prototype. Same idea as cosine_reg_self but
        uses the teacher's gated pseudo-labels instead of self-predictions."""
        b, d, h, w = feat.shape
        if labels.shape[-2:] != (h, w):
            labels = F.interpolate(labels.float().unsqueeze(1), size=(h, w),
                                   mode="nearest").long()[:, 0]
            conf_map = F.interpolate(conf_map.unsqueeze(1), size=(h, w),
                                     mode="bilinear", align_corners=False)[:, 0]
            valid = F.interpolate(valid.float().unsqueeze(1), size=(h, w),
                                  mode="nearest").bool()[:, 0]
        z = feat[0].permute(1, 2, 0).reshape(-1, d)
        y = labels[0].reshape(-1)
        c = conf_map[0].reshape(-1)
        v = valid[0].reshape(-1)
        losses = []
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((y == cls) & v & (c >= threshold))[0]
            if idx.numel() < 2:
                continue
            zp = F.normalize(z[idx].mean(0), dim=0)
            ref = F.normalize(self.prototypes[cls].detach(), dim=0)
            losses.append(1.0 - torch.sum(zp * ref))
        if not losses:
            return feat.sum() * 0.0
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------ #
    #  engagement diagnostics (no more flying blind)                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def engagement_stats(self, feat: torch.Tensor, labels=None):
        """How much is the prototype mechanism touching? Mean top-1 cosine per
        class and the overall mean top-1 cosine. Logged every 10 steps."""
        if not self.initialized.any():
            return {"per_class_top1": [0.0] * self.num_classes, "pixels_fraction": 0.0}
        sims = self.logits(feat)                          # (1, C, Hf, Wf)
        top1 = sims.max(1).values
        per_class = []
        if labels is not None:
            # labels are at image resolution; sims at feature resolution
            if labels.shape[-2:] != top1.shape[-2:]:
                lab = F.interpolate(labels.float().unsqueeze(1), size=top1.shape[-2:],
                                    mode="nearest").long()[:, 0]
            else:
                lab = labels
            for c in range(self.num_classes):
                m = (lab == c)
                per_class.append(float(top1[m].mean()) if m.any() else float("nan"))
        else:
            per_class = [float(top1.mean())] * self.num_classes
        return {"per_class_top1": per_class, "pixels_fraction": float(top1.mean())}


def _load_image(path: str, image_size: int, device: str) -> torch.Tensor:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def _scale_points(points, item, image_size: int):
    import numpy as np
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3).copy()
    if len(pts):
        pts[:, 0] *= image_size / item.get("width", image_size)
        pts[:, 1] *= image_size / item.get("height", image_size)
    return pts
