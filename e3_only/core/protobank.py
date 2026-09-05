"""A multi-prototype cosine classifier that IS the classifier.

What was wrong before
---------------------
E3 kept prototypes in a bank beside the network and mixed their opinion into the
teacher target with weight 0.30. Two defects, both measured:

  * ``PrototypeBank.logits`` returned a raw cosine in [-1, 1] with no
    temperature. Softmaxed over 17 classes that is nearly uniform -- the largest
    achievable peak is about 1/17 * e^2 / (…) -- so 30% of the teacher target was
    noise and the fused peak was capped near 0.74. A KL onto a target that cannot
    exceed 0.74 puts a hard ceiling on the student's sharpness. That ceiling was
    the loss floor.
  * the update path was unreachable: ``train.py:205`` guards it with
    ``teacher is None``, and E3 always has a teacher, so ``proto_reg`` logged
    0.0000 and ``bank_px`` logged 0 for all 50 epochs. Tuning ``proto_ema`` and
    ``proto_sim_threshold`` changed nothing because nothing read them.

The fix is structural rather than numerical: make the prototypes the classifier's
own weights. Then there is no fusion weight to tune, no separate softmax to be
flat, and no code path that can be silently skipped -- if prototypes are broken
the network cannot predict at all.

Why K per class
---------------
Failure mode 6 is wholesale confusion between spectrally similar classes
(chaparral <-> bare soil, field <-> grass, sand <-> pavement). A single mean per
class is a unimodal model; "field" in DLRSD covers ploughed earth and green crop,
whose means sit on opposite sides of "grass". No amount of tuning fixes a
unimodal model of a bimodal class. K prototypes with a soft-max over K is a
mixture, and the decision surface it induces is a union of cones rather than one
cone -- which is what the measured field IoU of 0.0888 needs.

Aggregation over K uses log-sum-exp, not max. A hard max sends gradient to one
prototype only, so unlucky prototypes never receive an update and die; LSE at a
small temperature approximates the max while keeping every prototype alive.

The point-derived anchor keeps 0% dense intact
----------------------------------------------
Prototypes are additionally pulled towards an EMA of *point* features -- the ~15
annotated pixels per image, nothing else. No dense mask and no pseudo-label ever
enters the bank, so the anchor is a human-derived quantity and cannot drift with
the network.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  FINCH first partition (self-contained, no external dependency)             #
# --------------------------------------------------------------------------- #
def finch_first_partition(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free clustering: the first partition of FINCH.

    Link every sample to its single nearest neighbour, declare i and j adjacent
    when j is i's neighbour, i is j's, or they share one, then take connected
    components. No K, no distance threshold, no iterations -- which is why it is
    the right tool for "how many modes does this class have", a question we have
    no labels to answer.

    x: (N, D) L2-normalised. Returns (N,) cluster ids.
    """
    n = x.shape[0]
    if n <= 1:
        return torch.zeros(n, dtype=torch.long, device=x.device)
    sim = x @ x.t()
    sim.fill_diagonal_(-2.0)
    kappa = sim.argmax(1)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    k = kappa.tolist()
    for i in range(n):
        union(i, k[i])
    # samples sharing a first neighbour belong together
    from collections import defaultdict
    shared = defaultdict(list)
    for i in range(n):
        shared[k[i]].append(i)
    for _, group in shared.items():
        for j in group[1:]:
            union(group[0], j)

    roots = {}
    out = torch.empty(n, dtype=torch.long, device=x.device)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out


# --------------------------------------------------------------------------- #
#  the classifier                                                             #
# --------------------------------------------------------------------------- #
class MultiPrototypeClassifier(nn.Module):
    """(B, D, H, W) embedding -> (B, C, H, W) logits, via C x K prototypes."""

    def __init__(self, dim: int, num_classes: int, prototypes_per_class: int = 4,
                 scale_init: float = 12.0, k_temperature: float = 0.20,
                 ema_decay: float = 0.95):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.k = prototypes_per_class
        self.k_temperature = k_temperature
        self.ema_decay = ema_decay

        w = torch.randn(num_classes, prototypes_per_class, dim)
        self.weight = nn.Parameter(F.normalize(w, dim=-1))
        # learnable logit scale: a cosine in [-1,1] cannot produce a confident
        # softmax on its own, which is exactly the bug this replaces. Kept in
        # log space so it stays positive, and clamped so it cannot run away.
        self.log_scale = nn.Parameter(torch.tensor(float(scale_init)).log())
        self.register_buffer("ema", F.normalize(w.clone(), dim=-1))
        self.register_buffer("ema_count", torch.zeros(num_classes, prototypes_per_class))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(4.0, 40.0)

    def cosines(self, embed: torch.Tensor) -> torch.Tensor:
        """(B, C, K, H, W) cosine between each pixel embedding and each prototype."""
        e = F.normalize(embed, dim=1)
        w = F.normalize(self.weight, dim=-1).reshape(self.num_classes * self.k, self.dim, 1, 1)
        c = F.conv2d(e, w)
        b, _, h, wd = c.shape
        return c.reshape(b, self.num_classes, self.k, h, wd)

    def aggregate(self, cos: torch.Tensor) -> torch.Tensor:
        """(B,C,K,H,W) -> (B,C,H,W). Smooth max over the K prototypes."""
        t = max(self.k_temperature, 1e-3)
        return t * torch.logsumexp(cos / t, dim=2)

    def forward(self, embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (logits, aggregated cosine). The cosine is returned so the point
        loss can apply an angular margin before scaling."""
        cos = self.aggregate(self.cosines(embed))
        return self.scale * cos, cos

    # ------------------------------------------------------------------ #
    #  point-derived anchor (0% dense: only annotated pixels enter)       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_ema(self, embed: torch.Tensor, points: List[torch.Tensor],
                   patch: int = 3):
        """EMA of annotated-pixel features into the nearest prototype of the
        point's own class. Hard assignment within the class, so the K slots
        specialise into the class's modes.

        A ``patch``-wide average is used instead of the single pixel because a
        click can land a pixel or two off the object; the surrounding 3x3 is a
        cheap variance reduction that does not change what is being measured.
        """
        e = F.normalize(embed, dim=1)
        b, d, h, w = e.shape
        r = patch // 2
        for bi, p in enumerate(points):
            if p is None or not len(p):
                continue
            for x, y, c in p.tolist():
                xi, yi, ci = int(round(x)), int(round(y)), int(c) % self.num_classes
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                x0, x1 = max(0, xi - r), min(w, xi + r + 1)
                y0, y1 = max(0, yi - r), min(h, yi + r + 1)
                f = F.normalize(e[bi, :, y0:y1, x0:x1].mean(dim=(1, 2)), dim=0)
                sims = F.normalize(self.ema[ci], dim=-1) @ f
                # unused slots claim the point first, so all K get populated
                empty = (self.ema_count[ci] == 0).nonzero()
                k = int(empty[0]) if len(empty) else int(sims.argmax())
                m = self.ema_decay if self.ema_count[ci, k] > 0 else 0.0
                self.ema[ci, k] = F.normalize(m * self.ema[ci, k] + (1 - m) * f, dim=0)
                self.ema_count[ci, k] += 1

    def anchor_loss(self) -> torch.Tensor:
        """Pull the learnable prototypes towards the point-feature EMA.

        This is the paper's prototype refinement, made consequential: because the
        prototypes are the classifier weights, moving them moves the decision
        boundary. In E3 the analogous term was both inert and, when it did fire,
        only able to nudge a 30% side vote.

        Uses 1 - cos_sim squared to penalise large deviations more strongly:
        prototypes that drift far from their EMA anchor get a quadratic penalty
        rather than linear, preventing them from decaying too quickly.
        """
        live = self.ema_count > 0
        if not live.any():
            return self.weight.sum() * 0.0
        w = F.normalize(self.weight, dim=-1)[live]
        t = F.normalize(self.ema, dim=-1)[live]
        cos_sim = (w * t).sum(-1)
        return ((1.0 - cos_sim) ** 2).mean()

    def repulsion_loss(self, margin: float = 0.10) -> torch.Tensor:
        """Keep the K prototypes of a class apart with a negative margin.

        Without this, gradient descent is free to collapse all K onto one point,
        which silently reduces the mixture back to the unimodal model that failure
        mode 6 is caused by. Penalises only *within-class* similarity; pushing
        different classes apart is the classifier's own job.

        The margin pushes prototypes to have cosine similarity <= -margin,
        ensuring they occupy distinct regions of the feature space rather than
        merely avoiding overlap (cos_sim > 0).
        """
        if self.k < 2:
            return self.weight.sum() * 0.0
        w = F.normalize(self.weight, dim=-1)
        g = torch.einsum("ckd,cjd->ckj", w, w)
        eye = torch.eye(self.k, device=w.device, dtype=torch.bool)[None]
        off = g.masked_select(~eye)
        # push similarity below -margin: penalise (cos_sim + margin).clamp_min(0)
        return (off + margin).clamp_min(0.0).mean()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def finch_init(self, feats: torch.Tensor, labels: torch.Tensor,
                   max_per_class: int = 3000):
        """Initialise the K prototypes per class from clustered point features.

        For each class, FINCH partitions that class's collected point features
        and the K largest clusters' means become the prototypes. Classes with
        fewer modes than K get their remaining slots filled with jittered copies,
        which the repulsion term then spreads out.

        ``max_per_class`` bounds the N x N similarity matrix FINCH builds. A
        common class can contribute several thousand clicks over an epoch, and
        the cost is quadratic; a uniform subsample of the class's own features
        estimates the same mode structure at a fraction of the memory.
        """
        for c in range(self.num_classes):
            sel = labels == c
            if sel.sum() == 0:
                continue
            x = F.normalize(feats[sel], dim=-1)
            if x.shape[0] > max_per_class:
                keep = torch.randperm(x.shape[0], device=x.device)[:max_per_class]
                x = x[keep]
            if x.shape[0] <= self.k:
                for k in range(self.k):
                    self.weight[c, k] = F.normalize(
                        x[k % x.shape[0]] + 0.01 * torch.randn_like(x[0]), dim=0)
                    self.ema[c, k] = self.weight[c, k]
                    self.ema_count[c, k] = 1
                continue
            cid = finch_first_partition(x)
            sizes = torch.bincount(cid)
            order = torch.argsort(sizes, descending=True)
            for k in range(self.k):
                if k < len(order):
                    mu = F.normalize(x[cid == order[k]].mean(0), dim=0)
                else:
                    mu = F.normalize(x[cid == order[k % len(order)]].mean(0)
                                     + 0.02 * torch.randn(self.dim, device=x.device), dim=0)
                self.weight[c, k] = mu
                self.ema[c, k] = mu
                self.ema_count[c, k] = 1

    def report(self) -> str:
        live = int((self.ema_count > 0).sum())
        w = F.normalize(self.weight.detach(), dim=-1)
        g = torch.einsum("ckd,cjd->ckj", w, w)
        eye = torch.eye(self.k, device=w.device, dtype=torch.bool)[None]
        intra = float(g.masked_select(~eye).mean()) if self.k > 1 else 0.0
        return (f"proto live {live}/{self.num_classes * self.k} "
                f"intra_cos {intra:+.3f} scale {float(self.scale.detach()):.1f}")
