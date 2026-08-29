"""Shadows: a physical synthesis, a provable invariance, and a learned residual.

Failure mode 4 -- an object's cast shadow labelled as a different class from the
surface it falls on -- is not a labelling problem, it is a representation
problem. A cast shadow changes a pixel's appearance without changing its
material, so any classifier that reads appearance directly must be told, somehow,
that the change is irrelevant. With 0% dense labels and no shadow annotations,
that has to come from the image formation model rather than from data.

The dichromatic model
---------------------
For a Lambertian surface under sun plus sky,

    I_c = rho_c * ( V * L_c^dir * cos(theta)  +  L_c^amb )

with V in {0,1} the binary visibility of the sun. In the umbra V = 0, so

    I'_c / I_c = L_c^amb / (L_c^dir cos(theta) + L_c^amb)  =:  alpha_c

which is a *per-channel multiplicative attenuation independent of the surface
albedo rho*. Two consequences drive everything in this file:

  (a) shadowing is (locally) a per-channel gain, so it can be synthesised
      faithfully without any shadow annotation -- ``synth_shadow`` below;
  (b) any feature invariant to a per-channel gain is invariant to shadow.
      Skylight is blue-rich, so alpha_B > alpha_G > alpha_R: shadows are darker
      *and* bluer, which is why plain intensity normalisation is not enough and
      the invariant has to be built per channel.

The provable part
-----------------
Write l_c = log I_c. A shadow acts on log-space by TRANSLATION: l -> l + log alpha.
Let W be a window on which alpha is constant, mu_W the local mean over W, and

    c1 = l_R - l_G,   c2 = l_B - l_G,   l = sum_c w_c l_c   (w = (.299,.587,.114))

THEOREM. Define
    phi(I) = ( c1 - mu_W c1,  c2 - mu_W c2,  l - mu_W l,  sigma_W l,  |grad l| ).
If I -> alpha (*) I with alpha > 0 constant on W, then phi(I') = phi(I) exactly.

PROOF. c1' = c1 + log(alpha_R/alpha_G), a constant on W, so mu_W shifts by the
same constant and the difference is unchanged; likewise c2. l' = l + k with
k = sum_c w_c log alpha_c constant on W, so l' - mu_W l' = l - mu_W l; a standard
deviation is shift-invariant; and a spatial derivative of a constant is zero. []

COMPLETENESS. The gain group is 3-dimensional and acts on (l_R, l_G, l_B) by
translation, so the local invariants can be at most 3-dimensional. The map
(l_R, l_G, l_B) -> (c1, c2, l) is linear with determinant -(w_R+w_G+w_B) = -1,
hence invertible, and mean-removal commutes with it. The first three channels of
phi are therefore a COMPLETE (maximal) invariant: nothing is discarded except the
three local illumination coordinates themselves. Channels 3 and 4 are functions
of those invariants, included explicitly because a shallow stem should not have
to rediscover them.

Note this is why channel 2 uses the geometric-mean luminance sum_c w_c log I_c
rather than log(sum_c w_c I_c). The arithmetic version picks up
log(sum_c w_c alpha_c I_c), which does not separate into image-plus-constant
unless alpha is achromatic -- and since skylight is blue-rich, alpha in a real
shadow never is.

None of this has to be learned, and no dataset has to contain it.

The honest caveat, and why the learned term is still needed
----------------------------------------------------------
alpha is only locally constant in the umbra interior. Across the penumbra it
varies, so the invariance degrades within roughly half a window width of the
shadow rim -- and the rim is exactly where the boundary terms are deciding
whether a contour is allowed. ``shadow_equivariance_loss`` covers that residual
by requiring the prediction on a synthetically shadowed image to match the
prediction on the clean one, and the shadow head supplies the map that
``structure.candidate_boundary`` uses to stop treating a shadow rim as a licence
to change label.
"""
import math
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

_EPS = 1e-4


# --------------------------------------------------------------------------- #
#  local statistics                                                           #
# --------------------------------------------------------------------------- #
def _local_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def _local_std(x: torch.Tensor, k: int) -> torch.Tensor:
    m = _local_mean(x, k)
    v = _local_mean(x * x, k) - m * m
    return v.clamp_min(0.0).sqrt()


# --------------------------------------------------------------------------- #
#  illumination-invariant channels                                            #
# --------------------------------------------------------------------------- #
def invariant_channels(image: torch.Tensor, window: int = 15) -> torch.Tensor:
    """(B,3,H,W) in [0,1] -> (B,8,H,W).

    Channels 0..4 are EXACTLY invariant to a locally-constant per-channel gain,
    i.e. to shadow. See the theorem in the module docstring.

      0,1  local-mean-removed log chromaticity  log(R/G), log(B/G)
      2    local-mean-removed log-average luminance
      3    local standard deviation of log-average luminance   (texture contrast)
      4    Sobel gradient magnitude of log-average luminance   (edge structure)
      5..7 the raw RGB

    Note channel 2 uses the LOG-AVERAGE (geometric mean) luminance
    ``l = sum_c w_c log I_c``, not ``log(sum_c w_c I_c)``. The distinction is the
    whole theorem. Under I_c -> alpha_c I_c the geometric version picks up
    ``sum_c w_c log alpha_c``, a constant, which local-mean removal cancels
    exactly; the arithmetic version picks up ``log(sum_c w_c alpha_c I_c)``,
    which does not separate from the image unless alpha is achromatic. Since
    skylight is blue-rich, alpha is *never* achromatic in a real shadow -- the
    arithmetic form is invariant only to first order in the blue bias, and the
    blue bias is exactly the part of the signal that matters here.

    Channels 5..7 are deliberately NOT invariant. Colour genuinely carries land
    cover -- sea is blue, sand is bright -- so it is kept rather than discarded;
    the shadow head and the equivariance loss are what teach the network when the
    raw view is unreliable.

    ``window`` must exceed the texture scale and stay below the shadow scale;
    15 px at 256x256 is about 6% of the field of view, which for DLRSD sits
    between roof texture and building shadow.
    """
    x = image.clamp_min(_EPS)
    logr, logg, logb = x[:, 0:1].log(), x[:, 1:2].log(), x[:, 2:3].log()
    logy = 0.299 * logr + 0.587 * logg + 0.114 * logb      # log-average luminance

    c1 = logr - logg
    c2 = logb - logg
    inv = [c1 - _local_mean(c1, window),
           c2 - _local_mean(c2, window),
           logy - _local_mean(logy, window),
           _local_std(logy, window)]

    kx = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None] / 8.0
    ky = kx.transpose(-1, -2)
    pad = F.pad(logy, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(pad, kx), F.conv2d(pad, ky)
    inv.append((gx * gx + gy * gy).clamp_min(0.0).sqrt())

    return torch.cat(inv + [image], dim=1)


# --------------------------------------------------------------------------- #
#  synthesis                                                                  #
# --------------------------------------------------------------------------- #
def _soft_field(b: int, h: int, w: int, device, coarse: int = 8,
                blur: float = 0.0) -> torch.Tensor:
    """Blobby low-frequency random field in [0,1], upsampled from ``coarse``."""
    z = torch.rand(b, 1, coarse, coarse, device=device)
    f = F.interpolate(z, size=(h, w), mode="bicubic", align_corners=False)
    return f.clamp(0.0, 1.0)


def _oriented_bar(b: int, h: int, w: int, device) -> torch.Tensor:
    """An elongated half-plane-ish blob: cast shadows of buildings and aircraft
    are directional, so an isotropic noise field alone is not representative."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device), indexing="ij")
    out = []
    for _ in range(b):
        ang = random.uniform(0, math.pi)
        cx, cy = random.uniform(-.5, .5), random.uniform(-.5, .5)
        u = (xx - cx) * math.cos(ang) + (yy - cy) * math.sin(ang)
        v = -(xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang)
        hw = random.uniform(0.10, 0.40)
        hl = random.uniform(0.30, 0.90)
        d = torch.maximum(u.abs() / hw, v.abs() / hl)
        out.append((1.0 - d).clamp(0.0, 1.0))
    return torch.stack(out)[:, None]


def synth_shadow(image: torch.Tensor, prob: float = 0.65,
                 atten_lo: float = 0.30, atten_hi: float = 0.68,
                 blue_bias_hi: float = 0.28, penumbra: float = 2.5,
                 coarse: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (shadowed image, soft shadow mask in [0,1]).

    Applies I -> I * (1 - m * (1 - alpha_c)) with alpha sampled to respect the
    dichromatic model: a base attenuation in ``[atten_lo, atten_hi]`` and a blue
    bias so alpha_B > alpha_G > alpha_R. The mask is the max of a low-frequency
    blob field and a directional bar, Gaussian-blurred to give a penumbra.

    Free supervision: the returned mask is an exact label for the shadow head,
    and the pair (image, shadowed image) is an exact positive pair for the
    equivariance loss -- both without a single annotation.
    """
    b, _, h, w = image.shape
    dev = image.device
    field = torch.maximum(_soft_field(b, h, w, dev, coarse=coarse),
                          _oriented_bar(b, h, w, dev))
    m = (field - 0.55).clamp_min(0.0) / 0.45
    if penumbra > 0:
        k = int(2 * round(penumbra) + 1)
        g = torch.arange(k, device=dev, dtype=image.dtype) - k // 2
        g = torch.exp(-(g ** 2) / (2 * penumbra ** 2))
        g = g / g.sum()
        m = F.conv2d(F.pad(m, (k // 2, k // 2, 0, 0), mode="replicate"), g[None, None, None, :])
        m = F.conv2d(F.pad(m, (0, 0, k // 2, k // 2), mode="replicate"), g[None, None, :, None])
    m = m.clamp(0.0, 1.0)

    on = (torch.rand(b, 1, 1, 1, device=dev) < prob).to(image.dtype)
    m = m * on

    base = torch.empty(b, 1, 1, 1, device=dev).uniform_(atten_lo, atten_hi)
    bias = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.05, blue_bias_hi)
    alpha = torch.cat([base * (1.0 - 0.5 * bias), base, base * (1.0 + bias)], dim=1)
    alpha = alpha.clamp(0.05, 0.98)

    shadowed = (image * (1.0 - m * (1.0 - alpha))).clamp(0.0, 1.0)
    return shadowed, m[:, 0]


# --------------------------------------------------------------------------- #
#  losses                                                                     #
# --------------------------------------------------------------------------- #
def shadow_equivariance_loss(clean_logits: torch.Tensor, shadow_logits: torch.Tensor,
                             mask: torch.Tensor, conf_weight: bool = True,
                             mask_thresh: float = 0.25) -> torch.Tensor:
    """The prediction must not change when only the illumination changes.

    L = mean over shadowed pixels of  CE( p_shadow , sg[p_clean] )

    The clean prediction is the teacher and the shadowed one the student, never
    the reverse: the clean view is the one whose statistics the rest of the loss
    is fitted on, so it is the one that carries information. Pixels are weighted
    by the clean prediction's own confidence, because early in training the
    clean prediction is not worth matching and an unweighted term would spend
    its gradient copying noise into the shadow branch.

    Minimiser: the classifier's decision is a function of material, not of
    illumination -- exactly the property failure mode 4 says is missing.
    """
    with torch.no_grad():
        pc = clean_logits.softmax(1)
        w = (mask > mask_thresh).float()
        if conf_weight:
            w = w * pc.max(1).values
    ce = -(pc * F.log_softmax(shadow_logits, dim=1)).sum(1)
    return (ce * w).sum() / w.sum().clamp_min(1.0)


def shadow_head_loss(shadow_logit: torch.Tensor, mask: torch.Tensor,
                     neg_weight: float = 0.3,
                     restrict: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Asymmetrically weighted BCE of the shadow head against the synthesis mask.

    Two purposes. It forces the shared trunk to represent illumination
    explicitly rather than entangling it with class evidence, and it produces
    the map ``structure.candidate_boundary`` needs in order to stop treating a
    shadow rim as a place a semantic contour may appear.

    The asymmetry is not a tuning choice, it is what the supervision actually
    supports. A mask=1 pixel is *certainly* shadowed -- we darkened it ourselves.
    A mask=0 pixel is only probably unshadowed: the scene may contain a real,
    unannotated shadow there, and we want the deployed head to fire on those.
    Penalising every mask=0 activation at full strength would train the head to
    suppress exactly the real shadows it exists to find, so negatives carry
    weight ``neg_weight``. DLRSD's real-shadow prevalence is on the order of
    10%, so a down-weighted negative is a mildly noisy label rather than a
    systematic one.

    ``restrict`` limits the loss to a pixel subset, which is how the clean view
    is supervised: inside the synthesised blob the clean image is known to be
    unshadowed *relative to* its shadowed twin, and that contrast -- not
    darkness in the absolute -- is what makes the head discriminative instead of
    collapsing onto "dark implies shadow".
    """
    z = shadow_logit.squeeze(1)
    t = mask.clamp(0.0, 1.0).to(z.dtype)
    bce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    w = t + neg_weight * (1.0 - t)
    if restrict is not None:
        w = w * restrict.to(w.dtype)
    return (bce * w).sum() / w.sum().clamp_min(1e-6)


def shadow_edge_mask(shadow_prob: torch.Tensor, thresh: float = 0.45,
                     band: int = 2) -> torch.Tensor:
    """(B,H,W) bool: the rim of the predicted shadow -- an illumination-only edge.

    Fed to ``candidate_boundary(suppress=...)`` so that a photometric edge caused
    purely by shadow is removed from the set of places the label map is allowed
    to change.
    """
    m = (shadow_prob > thresh).float()[:, None]
    dil = F.max_pool2d(m, 2 * band + 1, stride=1, padding=band)
    ero = -F.max_pool2d(-m, 2 * band + 1, stride=1, padding=band)
    return ((dil - ero)[:, 0] > 0.5)
