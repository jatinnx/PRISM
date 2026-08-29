"""Loss terms and quality scores for the point-only framework.

No dense cross-entropy exists here — the only human labels that touch the
model are the (x, y, class) point tuples.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  supervision losses                                                          #
# --------------------------------------------------------------------------- #
def point_cross_entropy(logits, points, class_weights=None):
    """CE at the labelled pixels only (the base supervision, always on)."""
    if len(points) == 0:
        return logits.sum() * 0.0
    h, w = logits.shape[-2:]
    ys = points[:, 1].long().clamp(0, h - 1)
    xs = points[:, 0].long().clamp(0, w - 1)
    cls = points[:, 2].long()
    sampled = logits[0, :, ys, xs].t()
    return F.cross_entropy(sampled, cls, weight=class_weights)


def pseudo_cross_entropy(student_logits, pseudo_labels, conf_map, valid):
    """Per-pixel CE on gated pseudo pixels, weighted by confidence."""
    if not valid.any():
        return student_logits.sum() * 0.0
    ce = F.cross_entropy(student_logits, pseudo_labels, reduction="none")
    conf = conf_map.detach().clamp(0.0, 1.0)
    return (ce * valid.float() * conf)[valid].mean()


def consistency_loss(student_logits, teacher_probs):
    """Per-pixel KL(student || teacher). (reduction='batchmean' would divide
    only by batch size, making the loss ~H*W too large and drowning other terms.)"""
    kl = F.kl_div(F.log_softmax(student_logits, 1), teacher_probs.detach(),
                  reduction="none")
    return kl.sum(1).mean()


def boundary_smoothness_loss(image, probs, sigma=0.10):
    """E7: similar-looking neighbours should predict similarly; strong image
    edges suppress smoothing. A dense structural prior with NO dense mask."""
    dx = probs[:, :, :, 1:] - probs[:, :, :, :-1]
    dy = probs[:, :, 1:, :] - probs[:, :, :-1, :]
    ix = image[:, :, :, 1:] - image[:, :, :, :-1]
    iy = image[:, :, 1:, :] - image[:, :, :-1, :]
    wx = torch.exp(-((ix * ix).mean(1, keepdim=True)) / (sigma ** 2)).detach()
    wy = torch.exp(-((iy * iy).mean(1, keepdim=True)) / (sigma ** 2)).detach()
    return (wx * dx.abs()).mean() + (wy * dy.abs()).mean()


# --------------------------------------------------------------------------- #
#  quality scores                                                              #
# --------------------------------------------------------------------------- #
def point_agreement(mask, points):
    """Fraction of human points the prediction matches. Returns None (not 1.0)
    when there are no points — callers must skip the image in that case."""
    if len(points) == 0:
        return None
    h, w = mask.shape[-2:]
    ok = 0
    for x, y, c in points.detach().cpu().tolist():
        xx = min(max(int(round(x)), 0), w - 1)
        yy = min(max(int(round(y)), 0), h - 1)
        ok += int(int(mask[0, yy, xx]) == int(c))
    return ok / len(points)


def structural_boundary_score(mask, image):
    """Boundary quality: Canny alignment of predicted boundaries with image
    edges, plus a compactness term. In [0, 1]."""
    m = mask[0].detach().cpu().numpy().astype(np.uint8)
    rgb = image[0].detach().cpu().permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    pred_boundary = np.zeros_like(m, dtype=np.uint8)
    pred_boundary[1:] |= (m[1:] != m[:-1]).astype(np.uint8)
    pred_boundary[:, 1:] |= (m[:, 1:] != m[:, :-1]).astype(np.uint8)
    pred_boundary = cv2.dilate(pred_boundary, np.ones((3, 3), np.uint8))
    edge = (cv2.Canny(gray, 50, 150) > 0).astype(np.uint8)
    denom = max(int(pred_boundary.sum()), 1)
    align = float((pred_boundary * edge).sum()) / denom
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate((m > 0).astype(np.uint8), kernel)
    ero = cv2.erode((m > 0).astype(np.uint8), kernel)
    fragment = float((dil ^ ero).mean())
    compact = 1.0 / (1.0 + 3.0 * fragment)
    return float(np.clip(0.5 * align + 0.5 * compact, 0.0, 1.0))
