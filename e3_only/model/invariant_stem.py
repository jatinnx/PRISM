"""Full-resolution illumination-invariant stem.

Why a second input path exists
------------------------------
SAM's image encoder outputs a 64x64 grid. E3's decoder ran three convolutions on
that grid and then bilinearly upsampled 4x to 256x256, so every predicted
boundary was, geometrically, a smooth interpolation of a 64x64 decision. No loss
term can sharpen a boundary the architecture cannot represent: at 4x
interpolation the narrowest possible transition is about four pixels wide, and
DLRSD objects like cars and dock edges are a handful of pixels across.

The stem supplies what the 64x64 grid destroyed: full-resolution detail. And it
supplies it in the *illumination-invariant* form derived in core.shadow, so the
high-frequency skip that sharpens boundaries does not simultaneously re-import
shadow edges as class evidence -- which is what a raw-RGB skip connection would
do, and would make failure mode 4 worse rather than better.

Both halves matter and they are separable in the ablation: the stem can be run on
raw RGB (``invariant=False``) to show that the resolution alone is not what helps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.shadow import invariant_channels


def _block(cin: int, cout: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(groups, cout), cout),
        nn.GELU(),
    )


class InvariantStem(nn.Module):
    """(B,3,H,W) image -> features at H and H/2.

    Deliberately shallow. Its job is to carry precise local geometry and local
    photometric invariants up to the decoder, not to do semantics -- the ViT
    already does semantics, and a deep stem here would just relearn it worse from
    a 256x256 receptive field.
    """

    def __init__(self, out_channels: int = 32, window: int = 15,
                 invariant: bool = True):
        super().__init__()
        self.invariant = invariant
        self.window = window
        cin = 8 if invariant else 3
        self.enc = nn.Sequential(_block(cin, out_channels), _block(out_channels, out_channels))
        self.down = _block(out_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, image: torch.Tensor):
        x = invariant_channels(image, self.window) if self.invariant else image
        f_full = self.enc(x)
        f_half = self.down(F.avg_pool2d(f_full, 2))
        return f_full, f_half
