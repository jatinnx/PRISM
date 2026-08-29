"""Structural regularisers read off the image and the frozen partition.

Both terms here are *one-sided*. They forbid a label change where nothing in the
image supports one; they never demand a label change where an edge exists.
That asymmetry matters. A two-sided boundary loss rewards drawing contours, and
under 0% dense supervision the cheapest way to earn that reward is to
over-segment -- which is the failure the paper would be criticised for. Forbidding
unsupported contours has no such degenerate optimum: the do-nothing solution
(one label everywhere) is already excluded by the inventory terms.

E3 shipped ``boundary_smoothness_loss`` and a weight ``l_smooth = 0.2`` for it,
but the function was imported and never called, so no boundary term has ever
actually run in this project.

The shadow coupling
-------------------
A cast shadow produces a strong photometric edge with no semantic edge behind
it. Any purely appearance-driven smoothness term therefore *permits* the label
to change at the shadow rim, which is precisely failure mode 4. So the candidate
boundary set can be given a ``suppress`` map -- the shadow head's estimate of
which edges are illumination-only -- and those pixels are removed from the set of
places a contour is allowed. The shadow branch and the boundary branch are not
two independent additions; the first is what makes the second correct.
"""
from typing import Optional

import torch
import torch.nn.functional as F


def _shifts(x: torch.Tensor):
    """The four unique neighbour offsets; the other four follow by symmetry.
    Each entry is (shifted_tensor, valid_mask_slice) as an aligned pair."""
    return [
        (x[:, :, :, 1:], x[:, :, :, :-1]),        # horizontal
        (x[:, :, 1:, :], x[:, :, :-1, :]),        # vertical
        (x[:, :, 1:, 1:], x[:, :, :-1, :-1]),     # main diagonal
        (x[:, :, 1:, :-1], x[:, :, :-1, 1:]),     # anti-diagonal
    ]


def _disagree(pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
    """P(labels differ) under independent draws = 1 - sum_c p_a(c) p_b(c)."""
    return 1.0 - (pa * pb).sum(1)


# --------------------------------------------------------------------------- #
#  L_potts : edge-aware pairwise smoothness                                   #
# --------------------------------------------------------------------------- #
def edge_aware_potts_loss(logits: torch.Tensor, image: torch.Tensor,
                          sigma: float = 0.08) -> torch.Tensor:
    """sum_{i~j} w_ij * P(y_i != y_j) / sum w_ij,  w_ij = exp(-||I_i-I_j||^2 / 2 sigma^2).

    The differentiable Potts / dense-CRF pairwise energy used as a training loss
    rather than as inference-time post-processing. Photometrically similar
    neighbours are pushed to the same label; the weight decays as the colour
    difference grows, so real edges are cheap to cross.

    Minimiser: the labelling is piecewise constant on regions of near-constant
    colour. It is scale-free in the sense that it needs no notion of how many
    classes exist, which is what lets it operate on all 65 536 pixels of a
    0%-dense image.
    """
    p = logits.softmax(1)
    num = logits.new_zeros(())
    den = logits.new_zeros(())
    for (ia, ib), (pa, pb) in zip(_shifts(image), _shifts(p)):
        d2 = ((ia - ib) ** 2).sum(1)
        w = torch.exp(-d2 / (2.0 * sigma * sigma))
        num = num + (w * _disagree(pa, pb)).sum()
        den = den + w.sum()
    return num / den.clamp_min(1.0)


# --------------------------------------------------------------------------- #
#  candidate boundaries                                                       #
# --------------------------------------------------------------------------- #
def region_boundary(region: torch.Tensor) -> torch.Tensor:
    """(B,H,W) long partition -> (B,H,W) bool mask of its own boundaries.

    These are boundaries the frozen SAM partition asserts. They include contours
    with weak photometric contrast -- a building roof against similarly bright
    pavement -- which is exactly what a gradient-based edge map misses and what
    the appearance-only Potts term therefore cannot protect.
    """
    r = region[:, None].float()
    out = torch.zeros_like(region, dtype=torch.bool)
    for (a, b) in _shifts(r):
        diff = (a != b)[:, 0]
        pad_h = region.shape[1] - diff.shape[1]
        pad_w = region.shape[2] - diff.shape[2]
        # both padding directions, so a disagreeing pair marks BOTH of its members
        # rather than an arbitrary one of them. Exact for the two axis-aligned
        # shifts; for the two diagonals the marked pixel is the one diagonally
        # adjacent to both members, i.e. off by a single step. That is harmless
        # because every consumer dilates: ``candidate_boundary`` widens this map by
        # ``radius`` before use, and ``boundary_precision_loss`` compares against
        # an already-dilated ``allowed`` band. It does mean ``edge_radius >= 1`` is
        # a correctness requirement of the diagonal terms, not merely a tolerance.
        d = F.pad(diff.float(), (0, pad_w, 0, pad_h))
        out = out | (d > 0.5)
        d = F.pad(diff.float(), (pad_w, 0, pad_h, 0))
        out = out | (d > 0.5)
    return out


def image_edges(image: torch.Tensor, quantile: float = 0.80) -> torch.Tensor:
    """(B,H,W) bool: pixels whose gradient magnitude is in the top quantile.

    Thresholded per image by quantile rather than by an absolute value, so a
    hazy low-contrast scene and a crisp one both yield a comparable amount of
    candidate boundary.
    """
    grey = image.mean(1, keepdim=True)
    kx = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None] / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(F.pad(grey, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(grey, (1, 1, 1, 1), mode="replicate"), ky)
    mag = (gx * gx + gy * gy).sqrt()[:, 0]
    b = mag.shape[0]
    thr = torch.quantile(mag.reshape(b, -1), quantile, dim=1)[:, None, None]
    return mag >= thr


def dilate(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask[:, None].float(), k, stride=1, padding=radius)[:, 0] > 0.5


def candidate_boundary(image: torch.Tensor, region: Optional[torch.Tensor] = None,
                       quantile: float = 0.80, radius: int = 2,
                       suppress: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Where a semantic contour is *allowed* to be.

    Union of strong photometric edges and frozen-partition boundaries, dilated
    to tolerate a pixel or two of localisation error, minus any pixel the
    ``suppress`` map flags as an illumination-only edge.
    """
    cand = image_edges(image, quantile)
    if region is not None:
        cand = cand | region_boundary(region)
    cand = dilate(cand, radius)
    if suppress is not None:
        cand = cand & ~dilate(suppress, radius)
    return cand


# --------------------------------------------------------------------------- #
#  L_bnd : one-sided boundary precision                                       #
# --------------------------------------------------------------------------- #
def boundary_precision_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Penalise predicted label discontinuity only where no contour is allowed.

    g_j  = max over neighbours of P(y_j != y_neighbour)     (soft contour map)
    L    = mean_j  g_j * [j not in allowed]

    Minimiser: the predicted label map's contour set is contained in the allowed
    set. There is no term rewarding contours, so the loss cannot be gamed by
    inventing them; and because ``allowed`` already contains every real edge the
    image or the partition knows about, satisfying it costs a correct
    segmentation nothing.

    ``g`` is built with the same both-directions padding as ``region_boundary``,
    so it inherits the same one-step offset on the diagonal shifts. Pass an
    ``allowed`` map that has been dilated by at least one pixel (which
    ``candidate_boundary`` does by default) or a contour sitting exactly on the
    edge of the band will be charged through its diagonal neighbour.

    If ``allowed`` happens to cover the whole frame, ``forbidden.sum()`` is zero
    and the clamp makes the loss exactly 0 rather than NaN -- the correct reading,
    since a fully-permitted image forbids nothing.
    """
    p = logits.softmax(1)
    b, _, h, w = logits.shape
    g = logits.new_zeros(b, h, w)
    for (pa, pb) in _shifts(p):
        d = _disagree(pa, pb)
        pad_h = h - d.shape[1]
        pad_w = w - d.shape[2]
        g = torch.maximum(g, F.pad(d, (0, pad_w, 0, pad_h)))
        g = torch.maximum(g, F.pad(d, (pad_w, 0, pad_h, 0)))
    forbidden = (~allowed).float()
    return (g * forbidden).sum() / forbidden.sum().clamp_min(1.0)
