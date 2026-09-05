"""Progressive decoder with a cosine-prototype head and a shadow head.

Differences from the E3 decoder, each tied to a measured failure
---------------------------------------------------------------
1. PROGRESSIVE UPSAMPLING WITH FULL-RES SKIPS (64 -> 128 -> 256), instead of one
   4x bilinear interpolation at the end. Boundaries become representable at the
   resolution they occur at. Addresses failure mode 1 (salt-and-pepper is partly
   an artefact of interpolating a coarse decision) and every boundary metric.

2. A COSINE CLASSIFIER whose weights are the prototypes. In E3 the final layer
   was a plain 1x1 convolution and prototypes were a separate, near-uniform side
   vote. Here the prototypes decide the prediction, so prototype quality and
   prediction quality are the same quantity. Addresses failure mode 6.

3. A SHADOW HEAD sharing the trunk. Its output is not an end in itself: it is
   consumed by ``structure.candidate_boundary`` to stop shadow rims from
   licensing a label change, and its auxiliary loss forces the trunk to
   represent illumination separately from class. Addresses failure mode 4.

4. L2-NORMALISED EMBEDDINGS returned alongside the logits, so the point loss can
   apply an angular margin and the prototype bank can be anchored on annotated
   pixels.

5. AN IMAGE-LEVEL PRESENCE HEAD, pooled with a soft-max (log-sum-exp) over space.
   The point inventory S(I) is exact -- validate_inventory finds a point for every
   GT class in 630/630 train images -- but at test time there are no points, so the
   dense head is free to paint a class the image does not contain, and measurably
   does: 10.52% of val pixels carry a class outside their own image's inventory.
   This branch learns to predict S(I) itself, from a separate 1x1 stack on the same
   trunk, so its answer is not a summary of the segmentation and can disagree with
   it. LSE rather than average pooling: average pooling over 65k pixels cannot see a
   class that occupies 40 of them, and the classes that get hallucinated over
   (field, sand, mobile home) are exactly the small ones.
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.protobank import MultiPrototypeClassifier


def _block(cin: int, cout: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(groups, cout), cout),
        nn.GELU(),
    )


class ProgressiveDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, stem_channels: int = 32,
                 embed_dim: int = 64, prototypes_per_class: int = 4,
                 mid: int = 192, k_temperature: float = 0.20,
                 scale_init: float = 12.0, dilated_context: bool = True,
                 presence_temperature: float = 0.5):
        super().__init__()
        self.coarse = nn.Sequential(_block(in_channels, mid, 32), _block(mid, mid, 32))
        # a dilated pair widens the receptive field on the 64x64 grid, which is
        # what separates spectrally similar classes by context rather than colour
        # (a green patch inside a runway is grass; the same green in a block of
        # fields is field). Cheap at 64x64, unaffordable at 256x256.
        self.context = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(32, mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(32, mid), nn.GELU(),
        ) if dilated_context else None

        self.up1 = _block(mid + stem_channels, 96, 16)
        self.up2 = _block(96 + stem_channels, 64, 16)
        self.refine = _block(64, 64, 16)
        self.embed = nn.Conv2d(64, embed_dim, 1)
        self.shadow = nn.Conv2d(64, 1, 1)
        self.presence = nn.Sequential(
            nn.Conv2d(64, 64, 1), nn.GELU(), nn.Conv2d(64, num_classes, 1))
        self.presence_temperature = presence_temperature
        self.classifier = MultiPrototypeClassifier(
            embed_dim, num_classes, prototypes_per_class,
            scale_init=scale_init, k_temperature=k_temperature)
        self.embed_dim = embed_dim

    def forward(self, feat: torch.Tensor, stem_full: torch.Tensor,
                stem_half: torch.Tensor, out_size) -> Dict[str, torch.Tensor]:
        x = self.coarse(feat)
        if self.context is not None:
            x = x + self.context(x)

        h2, w2 = stem_half.shape[-2:]
        x = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, stem_half], dim=1))

        h1, w1 = stem_full.shape[-2:]
        x = F.interpolate(x, size=(h1, w1), mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, stem_full], dim=1))
        x = self.refine(x)

        if (h1, w1) != tuple(out_size):
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

        e = F.normalize(self.embed(x), dim=1)
        logits, cos = self.classifier(e)
        return {"logits": logits, "cos": cos, "embed": e,
                "shadow_logit": self.shadow(x), "scale": self.classifier.scale,
                "presence_logit": self._pool_presence(x),
                "classifier": self.classifier}

    def _pool_presence(self, x: torch.Tensor) -> torch.Tensor:
        """Log-sum-exp pool the presence map to one logit per class.

        LSE_T(z) = T log mean exp(z / T) interpolates between the spatial mean
        (T large) and the spatial max (T -> 0). The mean is the wrong pool here:
        a class covering 40 of 65536 pixels contributes 0.06% of it, so a head
        trained on the mean learns to predict only the large classes -- and the
        hallucinated classes are the small ones. The max is the right limit but
        has a single-pixel gradient. T = 0.5 keeps the max's sensitivity and the
        mean's gradient spread.
        """
        z = self.presence(x)
        b, c = z.shape[:2]
        t = max(self.presence_temperature, 1e-3)
        f = z.flatten(2).float() / t
        return (torch.logsumexp(f, dim=2) - math.log(f.shape[2])).mul(t).to(z.dtype)
