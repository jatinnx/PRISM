"""Set-level constraints derived from the point annotations alone.

The idea this module exists to exploit
--------------------------------------
``make_manifests.py:sample_points`` loops over every class index and emits up to
five grid-spread points for *every class that appears* in the annotator's point
mask. So the point file carries two signals, not one:

  * WHICH class ~15 individual pixels are  (the signal E3 already used), and
  * WHICH classes the image contains at all -- its *inventory* S.

The second is far denser than the first. There are 17 classes and (measured on
this manifest) only a handful occur per image, so the complement of the
inventory is a **hard negative constraint that applies to all 65 536 pixels**.
That is what turns 0% dense supervision into a dense training signal, and it is
the only mechanism here that can suppress a class the network has never been
told anything positive about -- failure modes 2 (ghost classes) and 6 (spectral
confusion) both reduce to "the network chose a class that is not in this image",
and both become unreachable once the inventory is enforced.

Three terms, each with a stated minimiser
-----------------------------------------
  L_abs   partial-label likelihood. Minimised exactly when every pixel puts all
          its mass inside S. Robust form bounds the gradient so the annotation
          noise measured by tools/validate_inventory.py cannot dominate.
  L_pres  multiple-instance coverage. Minimised when every class in S owns at
          least k confident pixels. Prevents a present class being erased
          (measured symptom: field IoU 0.0888).
  L_area  one-sided area floor. Minimised when each class covers at least the
          area its own points' regions already occupy. Prevents one class
          flooding the image at the expense of the others (failure mode 3),
          using a floor that is itself derived only from points and geometry.

None of the three reads a dense mask, a pseudo-label, or the network's own past
predictions, so none of them can participate in confirmation bias.
"""
from typing import List, Optional

import math

import torch
import torch.nn.functional as F

_NEG_INF = -1e4


# --------------------------------------------------------------------------- #
#  inventory extraction                                                       #
# --------------------------------------------------------------------------- #
def inventory_from_points(points: List[torch.Tensor], num_classes: int,
                          device=None) -> torch.Tensor:
    """(B, C) bool: which classes the annotator clicked in each image.

    ``points[b]`` is an (N_b, 3) tensor of (x, y, class). An image with no
    points gets an all-True row, which makes every constraint in this module a
    no-op for it rather than a source of wrong gradient.
    """
    device = device or (points[0].device if len(points) else "cpu")
    out = torch.zeros(len(points), num_classes, dtype=torch.bool, device=device)
    for b, p in enumerate(points):
        if p is None or not len(p):
            out[b] = True
            continue
        cls = p[:, 2].long().clamp_(0, num_classes - 1)
        out[b, cls] = True
    return out


def mask_absent_logits(logits: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Set logits of absent classes to -inf. Used for *targets*, never for the
    term the gradient flows through -- a hard mask has no gradient to give, so
    L_abs below is what actually teaches the network the constraint."""
    m = present[:, :, None, None].expand_as(logits)
    return logits.masked_fill(~m, _NEG_INF)


# --------------------------------------------------------------------------- #
#  L_abs : partial-label (candidate-set) likelihood                           #
# --------------------------------------------------------------------------- #
def absent_class_loss(logits: torch.Tensor, present: torch.Tensor,
                      leak: float = 0.05,
                      pixel_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """-log( (1-eta) * sum_{c in S} p_c  +  eta ), averaged over pixels.

    This is the maximum-likelihood objective for a pixel whose label is known to
    lie in the candidate set S but is otherwise unknown: the likelihood of the
    observation is the total mass assigned to S. It is the dense counterpart of
    the pointwise cross-entropy, and it needs no pseudo-label.

    Minimiser: sum_{c in S} p_c = 1 at every pixel. Nothing about *which* class
    inside S wins is constrained, so the term cannot introduce a class bias.

    The ``leak`` eta is not smoothing for its own sake. Measured on this
    manifest, a class occasionally appears in an image without receiving a
    point; for those pixels the correct answer lies outside S and the plain
    objective would push it to zero with unbounded force. Mixing in a uniform
    leak caps the loss at -log(eta) and the gradient at (1-eta)/eta, so a
    violating pixel contributes a bounded, survivable amount. Set eta from the
    violation rate that tools/validate_inventory.py reports.
    """
    logp = F.log_softmax(logits, dim=1)
    neg = torch.where(present[:, :, None, None].expand_as(logp),
                      logp, torch.full_like(logp, _NEG_INF))
    lse_in = torch.logsumexp(neg, dim=1)                        # log sum_{c in S} p_c
    if leak > 0:
        a = lse_in + math.log(1.0 - leak)
        loss = -torch.logaddexp(a, torch.full_like(a, math.log(leak)))
    else:
        loss = -lse_in
    if pixel_weight is not None:
        w = pixel_weight
        return (loss * w).sum() / w.sum().clamp_min(1.0)
    return loss.mean()


# --------------------------------------------------------------------------- #
#  L_pres : multiple-instance coverage                                        #
# --------------------------------------------------------------------------- #
def present_coverage_loss(logits: torch.Tensor, present: torch.Tensor,
                          topk_frac: float = 0.005, min_k: int = 64,
                          area_floor: torch.Tensor | None = None) -> torch.Tensor:
    """Every class in the inventory must own at least k confident pixels.

    The image is a bag and each present class is a positive bag label; the bag
    score is the mean of the k largest per-pixel probabilities, which is the
    standard smooth relaxation of max-pooling (a hard max would hand the whole
    gradient to a single pixel and is unstable at batch size 1).

    Minimiser: for each c in S there exist >= k pixels with p_c -> 1. Combined
    with L_abs -- which forbids mass outside S -- this makes "predict one class
    everywhere" infeasible whenever the image has two or more classes.

    Choosing k, and why a constant k is wrong
    -----------------------------------------
    k has exactly one job: make the gradient land on a neighbourhood instead of a
    single argmax pixel. It is NOT an area estimate -- that is ``L_area``'s job,
    and ``L_area`` does it from a measured lower bound. A constant
    ``k = max(64, 0.005 * 256^2) = 327`` conflates the two, and the conflation is
    not symmetric: it asks a class that truly covers 40 px to produce 327 pixels
    of confident mass, so the only way to satisfy the term is to over-claim by
    almost an order of magnitude. That pressure falls hardest on the rare classes
    (ship, court, tanks, mobile home) whose IoU the method is supposed to rescue,
    and it manifests as a rare class bleeding into its surroundings -- failure
    mode 5, correct shape with the wrong label, on the *neighbour's* pixels.

    So when ``area_floor`` (the (B,C) *fractional* floor from
    ``area_floor_from_propagation``) is supplied, k is set per (image, class) to
    that measured floor, clipped into ``[min_k, k_max]``:

        k_bc = clip(floor_bc * H * W,  min_k,  max(min_k, topk_frac * H * W))

    A class whose own regions already cover 3000 px is asked for 327 (unchanged,
    the cap binds); a class whose regions cover 40 px is asked for ``min_k``, the
    stability floor, and nothing more. k never exceeds the old constant, so this
    only ever *relaxes* the term -- the anti-flooding role is untouched, because
    flooding is blocked by L_abs and L_area, not by this ceiling.

    A class with no propagated region at all (its point fell in a conflict
    region) gets ``floor = 0`` -> ``k = min_k``, which is the right default: the
    inventory still asserts the class exists, and nothing has been measured about
    how big it is.

    ``area_floor`` carries no gradient (it is read off integer labels), so the
    per-class selection mask below is a constant and the term stays a clean
    top-k mean over ``p``.
    """
    b, c, h, w = logits.shape
    p = logits.softmax(1).reshape(b, c, h * w)
    k_max = min(max(min_k, int(topk_frac * h * w)), h * w)
    vals = p.topk(k_max, dim=2).values                           # (B, C, k_max)
    if area_floor is None:
        bag = vals.mean(2)
    else:
        # .float() and an int64 arange keep the comparison exact. Done in vals.dtype
        # instead, a bf16 autocast would round k_max = 327 to 328 -- harmless here,
        # but a silent dtype-dependent change in a loss is not worth carrying.
        k_bc = (area_floor.detach().float() * float(h * w)).clamp(min=float(min_k),
                                                                 max=float(k_max))
        idx = torch.arange(k_max, device=p.device).view(1, 1, -1)
        sel = (idx < k_bc[:, :, None]).to(vals.dtype)            # (B, C, k_max)
        bag = (vals * sel).sum(2) / sel.sum(2).clamp_min(1.0)
    loss = -torch.log(bag.clamp_min(1e-6))
    m = present.float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
#  L_area : one-sided area floor from the point regions                       #
# --------------------------------------------------------------------------- #
def area_floor_from_propagation(prop: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, C) area floors read off the point-propagated partial labels.

    ``prop`` is (B, H, W) with class ids where a single-class region was labelled
    by a point and -1 elsewhere. The area a class already occupies through its
    own regions is a lower bound on its true area that used no dense mask and no
    network output -- exactly the kind of quantity that is safe to constrain
    against.
    """
    b, h, w = prop.shape
    out = torch.zeros(b, num_classes, device=prop.device)
    valid = prop >= 0
    if valid.any():
        flat = prop.reshape(b, -1)
        for c in range(num_classes):
            out[:, c] = (flat == c).float().sum(1)
    return out / float(h * w)


def area_floor_loss(logits: torch.Tensor, floors: torch.Tensor,
                    safety: float = 0.60) -> torch.Tensor:
    """relu(safety * floor_c - mean_x p_c(x)), averaged over classes with a floor.

    One-sided on purpose: a class is never punished for growing past its floor,
    only for shrinking below the evidence. ``safety`` < 1 absorbs the case where
    a propagated region overshoots the true object.

    Minimiser: mean_x p_c(x) >= safety * floor_c for every class with points.
    """
    p = logits.softmax(1).mean(dim=(2, 3))                       # (B, C)
    gap = F.relu(safety * floors - p)
    m = (floors > 0).float()
    return (gap * m).sum() / m.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
#  restricted information maximisation (ablation row)                         #
# --------------------------------------------------------------------------- #
def restricted_information_loss(logits: torch.Tensor, present: torch.Tensor,
                                w_pixel: float = 1.0, w_marginal: float = 1.0
                                ) -> torch.Tensor:
    """-I(class ; pixel) restricted to the inventory: sharpen each pixel while
    keeping the image-level class marginal spread over S.

    I = H(mean_x p) - mean_x H(p). Minimising -I lowers per-pixel entropy
    (crisp decisions) and raises marginal entropy (all present classes used).
    The marginal half is a weaker, prior-free alternative to L_area -- it pushes
    towards *equal* class areas, which DLRSD does not satisfy, so it is kept at
    a low weight and reported as its own ablation row rather than used by
    default.
    """
    p = logits.softmax(1)
    m = present[:, :, None, None].float()
    p = (p * m)
    p = p / p.sum(1, keepdim=True).clamp_min(1e-6)
    ent_pixel = -(p * p.clamp_min(1e-6).log()).sum(1).mean()
    bar = p.mean(dim=(2, 3))
    ent_marg = -(bar * bar.clamp_min(1e-6).log()).sum(1).mean()
    return w_pixel * ent_pixel - w_marginal * ent_marg
