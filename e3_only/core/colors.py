"""Colour rendering helpers — the ONLY palette is data/class_map.py::PALETTE.

``colorize`` and ``make_legend`` both read from that single source, so a
prediction PNG and its legend can never disagree about a class colour.
"""
import numpy as np

from ..data.class_map import CLASS_NAMES, PALETTE


def colorize(labels, palette=PALETTE):
    """labels: (H, W) uint8/long class ids (0..16) -> (H, W, 3) uint8 RGB."""
    return palette[labels.astype(np.int64) % len(palette)]


def overlay(image_rgb, labels, alpha=0.5, palette=PALETTE):
    """Semi-transparent colour overlay of labels on an RGB image (0..255)."""
    img = image_rgb.astype(np.float32)
    col = colorize(labels, palette).astype(np.float32)
    return (img * (1 - alpha) + col * alpha).astype(np.uint8)


def make_legend(swatch=40, pad=8, names=None):
    """Vertical legend image (colour swatch + class name), built from the SAME
    PALETTE that colorize() uses — consistency by construction."""
    import cv2
    names = names or CLASS_NAMES
    rows = []
    for i, name in enumerate(names):
        sw = np.full((swatch, swatch, 3), PALETTE[i], dtype=np.uint8)
        label = np.full((swatch, 140, 3), 255, dtype=np.uint8)
        cv2.putText(label, name, (6, swatch - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack([sw, np.full((swatch, pad, 3), 255, dtype=np.uint8), label]))
    return np.vstack(rows)
