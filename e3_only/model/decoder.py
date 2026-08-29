"""Lightweight semantic decoder head on top of SAM image features."""
import torch.nn as nn
import torch.nn.functional as F


class SemanticDecoder(nn.Module):
    """(1, 256, H, W) SAM features -> (1, C, out_size, out_size) logits.

    When ``spatial_context=True`` (E7+), two extra 3x3 conv layers give a
    9x9 effective receptive field for disambiguating spectrally similar
    classes (grass/bare soil, court/water, trees/bare soil).
    """

    def __init__(self, in_channels: int, num_classes: int, spatial_context: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        ]
        if spatial_context:
            # E7: two extra 3x3 layers -> 9x9 effective receptive field
            layers += [
                nn.Conv2d(128, 128, 3, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            ]
        layers.append(nn.Conv2d(128, num_classes, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat, out_size):
        x = self.net(feat)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
