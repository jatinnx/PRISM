"""Negative prompt sampling for per-class SAM mask decoding."""
import random

import numpy as np


def draw_points(image_rgb, points, radius=5):
    """Draw class-coloured point clicks (white ring + filled dot) on an RGB
    image (0..255). ``points``: iterable of (x, y, class). Used by the
    evaluator for the ``<id>_points.png`` output.
    """
    from PIL import Image, ImageDraw
    from ..data.class_map import PALETTE
    img = image_rgb.copy()
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for (x, y, c) in points:
        col = tuple(int(v) for v in PALETTE[int(c)])
        d.ellipse([x - radius, y - radius, x + radius, y + radius],
                  outline=(255, 255, 255), width=2)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    return np.asarray(pil)


class NegativePromptSampler:
    """For a target class, returns (positive points, negative points).

    Negatives are points of competing classes, chosen as the ones *nearest* to
    the target class's own points first (plan.txt Step 13: "select
    nearby/conflicting points first") — most informative for visually similar
    remote-sensing classes."""

    def __init__(self, max_negative_points: int = 8):
        self.max_negative_points = max_negative_points

    def sample(self, points, target_class: int, use_negative: bool = True):
        pts = points.detach().cpu().tolist() if hasattr(points, "detach") else points.tolist()
        pos = [(float(x), float(y)) for x, y, c in pts if int(c) == int(target_class)]
        if not use_negative or not pos:
            return pos, []
        others = [(float(x), float(y), int(c)) for x, y, c in pts
                  if int(c) != int(target_class)]
        if not others:
            return pos, []
        # distance from each negative to the nearest positive point
        ranked = sorted(others, key=lambda o: min((o[0] - px) ** 2 + (o[1] - py) ** 2
                                                  for px, py in pos))
        neg = [(x, y) for x, y, _ in ranked[:self.max_negative_points]]
        return pos, neg
