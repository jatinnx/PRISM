"""Region geometry: turn a frozen, class-agnostic partition into supervision.

The partition R = {R_1..R_M} comes from tools/build_region_cache.py and is
computed once from the *pretrained* SAM, before training starts. It never
changes. That immutability is the point: every other target in a self-training
loop is a function of the network's own recent output, so errors are free to
compound (measured: E3 loses 3.8 mIoU between epoch 30 and epoch 50, on every
class). A constraint that is fixed before the first gradient step cannot
compound with anything.

The partition says *which pixels share a label*, never *what that label is*, so
it is compatible with 0% dense supervision by construction.

Three uses, in increasing order of trust required:

  1. PROPAGATION (uses only points + geometry, no network).
     A region containing points of exactly one class is that class, everywhere.
     Regions with conflicting points are discarded rather than guessed.
     This is what makes ~15 clicks per image into thousands of labelled pixels.

  2. HOMOGENEITY (uses only geometry + the current output's *shape*).
     Inside a region the posterior should be constant. Implemented as
     distillation onto the region's own sharpened mean, which is the version
     that has no uniform-distribution degeneracy. This is the direct answer to
     failure mode 1 (salt-and-pepper) -- an isolated wrong pixel is, by
     definition, a pixel that disagrees with its region.

  3. REGION-LEVEL SELF-TRAINING (uses the teacher).
     Confidence is pooled over a whole region before it is thresholded, so a
     target is accepted or rejected as a coherent object rather than pixel by
     pixel. An accepted target is automatically free of speckle, and the
     threshold adapts to the running distribution instead of a hand-set ramp.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  numpy side: point -> region propagation (runs in the dataloader)           #
# --------------------------------------------------------------------------- #
def propagate_points_np(region: np.ndarray, points: np.ndarray,
                        num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """-> (labels (H,W) int16 with -1 for unlabelled, conflict mask (H,W) bool).

    A region inherits a class only when every point inside it agrees. Regions
    holding points of two classes are marked as conflicts and excluded from
    *every* downstream term, including homogeneity -- a region that provably
    straddles a semantic boundary is evidence the partition erred there, and
    forcing it constant would import that error.
    """
    h, w = region.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    conflict = np.zeros((h, w), dtype=bool)
    if not len(points):
        return labels, conflict

    seen: Dict[int, set] = {}
    for x, y, c in points:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        r = int(region[yi, xi])
        if r < 0:
            continue
        seen.setdefault(r, set()).add(int(c) % num_classes)

    for r, cs in seen.items():
        sel = region == r
        if len(cs) == 1:
            labels[sel] = next(iter(cs))
        else:
            conflict[sel] = True
    return labels, conflict


# --------------------------------------------------------------------------- #
#  torch side: batched scatter over regions                                   #
# --------------------------------------------------------------------------- #
class RegionIndex:
    """Flattens per-image region ids into one global index so every region
    statistic is a single index_add_ instead of a Python loop over regions."""

    def __init__(self, region: torch.Tensor):
        """region: (B, H, W) long, ids in [0, M_b) per image, -1 allowed."""
        b, h, w = region.shape
        self.shape = (b, h, w)
        n_per = (region.reshape(b, -1).max(dim=1).values + 1).clamp_min(1).long()
        offs = torch.cat([torch.zeros(1, dtype=torch.long, device=region.device),
                          n_per.cumsum(0)[:-1]])
        self.total = int(n_per.sum().item()) + 1              # +1 = bucket for id -1
        self.dump = self.total - 1
        idx = region.long() + offs[:, None, None]
        self.idx = torch.where(region >= 0, idx, torch.full_like(idx, self.dump))
        self.flat = self.idx.reshape(-1)
        ones = torch.ones_like(self.flat, dtype=torch.float32)
        self.count = torch.zeros(self.total, device=region.device).index_add_(0, self.flat, ones)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (total, C) region means."""
        b, c, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        s = torch.zeros(self.total, c, device=x.device, dtype=x.dtype)
        s.index_add_(0, self.flat, flat)
        return s / self.count.clamp_min(1.0).to(x.dtype)[:, None]

    def scatter_back(self, v: torch.Tensor) -> torch.Tensor:
        """(total, C) -> (B, C, H, W)."""
        b, h, w = self.shape
        return v[self.flat].reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

    def scatter_back_1d(self, v: torch.Tensor) -> torch.Tensor:
        """(total,) -> (B, H, W)."""
        b, h, w = self.shape
        return v[self.flat].reshape(b, h, w)

    def any_true(self, m: torch.Tensor) -> torch.Tensor:
        """(B, H, W) bool -> (total,) bool: does the region contain a True?"""
        s = torch.zeros(self.total, device=m.device)
        s.index_add_(0, self.flat, m.reshape(-1).float())
        return s > 0


# --------------------------------------------------------------------------- #
#  L_hom : within-region homogeneity by self-distillation                     #
# --------------------------------------------------------------------------- #
def region_homogeneity_loss(logits: torch.Tensor, ridx: RegionIndex,
                            present: Optional[torch.Tensor] = None,
                            temperature: float = 0.5, min_size: int = 24,
                            exclude: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-entropy from every pixel to its own region's sharpened mean.

    target_R  propto ( mean_{j in R} p_j restricted to S ) ** (1/T),  stop-grad
    L         = mean_j CE( p_j, target_{R(j)} )

    Minimiser: p_j equals target_R for all j in R, i.e. the posterior is
    constant on each region *and* sharper than its own average. The sharpening
    is what removes the degenerate solution: the raw
    "minimise within-region divergence" objective is also minimised by making
    every pixel uniform, which would be catastrophic here; raising the mean to
    1/T with T < 1 makes the uniform point a maximum instead.

    Because the target is a detached function of the region mean, this term
    supplies *shape* information only -- it can move probability mass around
    inside a region but never decides which class the region is. That decision
    is left to the point terms and the inventory.
    """
    logp = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        p = logp.exp()
        if present is not None:
            p = p * present[:, :, None, None].float()
            p = p / p.sum(1, keepdim=True).clamp_min(1e-6)
        m = ridx.mean(p)                                   # (total, C)
        m = m.clamp_min(1e-6) ** (1.0 / max(temperature, 1e-3))
        m = m / m.sum(1, keepdim=True).clamp_min(1e-6)
        target = ridx.scatter_back(m)

    valid = ridx.scatter_back_1d((ridx.count >= min_size).to(logits.dtype)) > 0.5
    if exclude is not None:
        valid = valid & ~exclude
    if not valid.any():
        return logits.sum() * 0.0
    ce = -(target * logp).sum(1)
    return ce[valid].mean()


def region_js_divergence(logits: torch.Tensor, ridx: RegionIndex,
                         present: Optional[torch.Tensor] = None,
                         min_size: int = 24,
                         exclude: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The unsharpened information-theoretic form, kept for the ablation table.

    D_R = H(mean_{j in R} p_j) - mean_{j in R} H(p_j) >= 0, with equality iff
    every pixel in R carries the same posterior.

    This is the term ``region_homogeneity_loss`` is the practical surrogate for,
    and the ablation row exists to show why the surrogate is necessary: D_R is
    minimised by ANY within-region-constant assignment, including the uniform
    one, so its optimum set is a manifold containing the degenerate solution.
    Sharpening turns the uniform point into a strict repeller instead.

    It takes ``present`` and ``exclude`` for the same reason the sharpened version
    does -- so the ablation moves exactly one variable (the sharpening) rather
    than also silently removing the inventory projection and the conflict mask.
    """
    p = logits.softmax(1)
    if present is not None:
        p = p * present[:, :, None, None].to(p.dtype)
        p = p / p.sum(1, keepdim=True).clamp_min(1e-6)
    mean = ridx.mean(p)
    ent_mean = -(mean * mean.clamp_min(1e-6).log()).sum(1)             # (total,)
    ent_pix = -(p * p.clamp_min(1e-6).log()).sum(1)                    # (B,H,W)
    mean_ent = ridx.mean(ent_pix[:, None])[:, 0]
    d = (ent_mean - mean_ent).clamp_min(0.0)
    w = ridx.count * (ridx.count >= min_size).to(d.dtype)
    if exclude is not None:
        # a region containing a conflict is dropped entirely, matching the
        # sharpened version's per-pixel ``valid`` mask at region granularity
        w = w * (~ridx.any_true(exclude)).to(w.dtype)
    return (d * w).sum() / w.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
#  region-level self-training targets                                         #
# --------------------------------------------------------------------------- #
class AdaptiveRegionGate:
    """tau = mu - kappa * sigma over the running distribution of region
    confidences, rather than a hand-tuned absolute ramp.

    E3 gated per pixel against a fixed 0.70 -> 0.80 schedule. Two things are
    wrong with that. It is absolute, so as the teacher's calibration drifts the
    same number means different things; and it is per pixel, so an accepted
    target can be a single confident pixel in the middle of an unconfident
    object -- speckle, promoted to supervision.

    Pooling to regions first and thresholding relative to the current
    distribution fixes both: a fixed *fraction* of the region population is
    rejected at every point in training, and whatever is accepted is a whole
    object. Statistics are tracked by EMA because batch size is 1 and a single
    image's regions are far too few to estimate mu and sigma from.
    """

    def __init__(self, kappa: float = 0.50, momentum: float = 0.99,
                 floor: float = 0.50, warmup: int = 50):
        self.kappa = kappa
        self.momentum = momentum
        self.floor = floor
        self.warmup = warmup
        self.mu: Optional[float] = None
        self.var: Optional[float] = None
        self.seen = 0

    def update(self, conf: torch.Tensor) -> float:
        if not len(conf):
            return self.value()
        m = float(conf.mean())
        v = float(conf.var(unbiased=False)) if len(conf) > 1 else 0.0
        if self.mu is None:
            self.mu, self.var = m, v
        else:
            b = self.momentum
            self.mu = b * self.mu + (1 - b) * m
            self.var = b * self.var + (1 - b) * v
        self.seen += 1
        return self.value()

    def value(self) -> float:
        if self.mu is None or self.seen < self.warmup:
            return 1.01                       # accept nothing until calibrated
        return max(self.floor, self.mu - self.kappa * (self.var ** 0.5))

    def state_dict(self):
        return {"mu": self.mu, "var": self.var, "seen": self.seen}

    def load_state_dict(self, s):
        self.mu, self.var, self.seen = s.get("mu"), s.get("var"), s.get("seen", 0)


@torch.no_grad()
def region_self_training_targets(teacher_logits: torch.Tensor, ridx: RegionIndex,
                                 present: torch.Tensor, gate: AdaptiveRegionGate,
                                 supervised: Optional[torch.Tensor] = None,
                                 exclude: Optional[torch.Tensor] = None,
                                 min_size: int = 24, min_margin: float = 0.10,
                                 update_gate: bool = True):
    """-> (labels (B,H,W) long with -1, tau, accepted_fraction).

    Pipeline, in this order, because each step removes errors the next would
    otherwise amplify:
      1. project the teacher onto the inventory (absent classes get zero mass);
      2. average the projected posterior over each region;
      3. drop regions that are tiny, that conflict, or that a point already
         supervises -- a human label always beats a teacher label;
      4. keep what clears the adaptive threshold and a top-2 margin;
      5. emit HARD labels.

    Hard labels are deliberate. E3's target was a soft blend whose peak was
    capped near 0.74 by a temperature-free prototype vote, and a KL onto that
    blend put a ceiling on how sharp the student could ever become. Region
    argmax has no such ceiling.
    """
    p = teacher_logits.softmax(1)
    p = p * present[:, :, None, None].float()
    p = p / p.sum(1, keepdim=True).clamp_min(1e-6)

    m = ridx.mean(p)                                            # (total, C)
    top2 = m.topk(2, dim=1)
    conf = top2.values[:, 0]
    margin = top2.values[:, 0] - top2.values[:, 1]
    lab = top2.indices[:, 0]

    ok = (ridx.count >= min_size)
    ok[ridx.dump] = False
    if supervised is not None:
        ok = ok & ~ridx.any_true(supervised)
    if exclude is not None:
        ok = ok & ~ridx.any_true(exclude)

    cand = conf[ok]
    tau = gate.update(cand) if update_gate else gate.value()
    keep = ok & (conf >= tau) & (margin >= min_margin)

    labels = torch.where(keep, lab, torch.full_like(lab, -1))
    out = ridx.scatter_back_1d(labels.to(torch.float32)).long()
    frac = float(keep.float().sum() / ok.float().sum().clamp_min(1.0)) if ok.any() else 0.0
    return out, tau, frac
