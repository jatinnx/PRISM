"""Pseudo-label generation with a per-pixel confidence gate.

Fusion of up to three label sources, each a distribution over classes:
  * semantic decoder softmax            (always)
  * SAM prompt-mask hard labels         (per-pixel argmax over binary logits —
    replaces the broken sigmoid->softmax normalisation; pixels no class claims
    get a zero vote)
  * prototype cosine distribution       (softmax over cosine sims)

The gate is **per-pixel**: every pixel is judged independently and an image
contributes whatever pixels pass. With E6 the confidence becomes the
three-component score C(x) = lam_a*A + lam_b*B + lam_p*P(x) (point agreement,
boundary quality, prototype cosine).
"""
import torch
import torch.nn.functional as F

from .losses import point_agreement, structural_boundary_score
from . import prompts  # noqa: F401  (imported for interface documentation)


def sam_hard_labels(sam_logits: torch.Tensor, threshold: float = 0.5,
                    background_class: int | None = None):
    """Per-pixel argmax over per-class binary SAM logits -> hard one-hot labels.

    A pixel is "claimed" only if at least one class's sigmoid exceeds
    `threshold`; unclaimed pixels contribute a zero vector (so SAM's vote never
    injects a random class where it has no evidence). Returns the one-hot map,
    the claim mask, and the claim strength (max sigmoid per pixel)."""
    p = torch.sigmoid(sam_logits)                    # (1, C, H, W)
    labels = p.argmax(1)                             # (1, H, W)
    p_max = p.max(1).values                          # (1, H, W)
    valid = p_max >= threshold
    onehot = F.one_hot(labels, sam_logits.shape[1]).permute(0, 3, 1, 2).float()
    onehot = onehot * valid.unsqueeze(1).float()
    claim_strength = p_max * valid.float()
    return onehot, valid, claim_strength


def _ramp(lo, hi, epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return hi
    t = min(1.0, epoch / ramp_epochs)
    return lo + t * (hi - lo)


def make_pseudo(semantic_logits, sam_logits, features, bank, image, points,
                cfg, epoch: int):
    """Produce pseudo labels, a per-pixel confidence map, a per-pixel validity
    mask and the fused probability distribution (teacher target).

    Returns (labels, conf_map, valid, fused_probs, a, b):
      labels      (1, H, W) argmax of the fused distribution
      conf_map    (1, H, W) per-pixel confidence in [0, 1]
      valid       (1, H, W) bool, per-pixel gate
      fused_probs (1, C, H, W) detached teacher target for consistency
      a, b        image-level point agreement / boundary score (for logging)
    """
    sem_probs = torch.softmax(semantic_logits, dim=1)
    fused = cfg.fusion_w_sem * sem_probs

    sam_claim = torch.zeros_like(sem_probs[:, 0])
    if sam_logits is not None:
        sam_hard, sam_valid, sam_claim = sam_hard_labels(
            sam_logits, cfg.sam_mask_threshold, cfg.background_class)
        fused = fused + cfg.fusion_w_sam * sam_hard

    if bank is not None and bank.initialized.any():
        proto_probs = torch.softmax(bank.logits(features), dim=1)
        proto_probs = F.interpolate(proto_probs, size=semantic_logits.shape[-2:],
                                    mode="bilinear", align_corners=False)
        fused = fused + cfg.fusion_w_proto * proto_probs

    fused = fused / fused.sum(1, keepdim=True).clamp_min(1e-6)
    conf, labels = fused.max(1, keepdim=True)
    labels = labels[:, 0]
    conf = conf[:, 0]

    a = point_agreement(labels, points)               # None if no points
    b = structural_boundary_score(labels, image)

    # E3: simple per-pixel gate (no E6 confidence fusion)
    # E6 confidence fusion is NOT used in E3
    if getattr(cfg, 'use_confidence_fusion', False):
        # E6 only: three-component weighted score
        pass  # Not used in E3
    else:
        # E3-E5: per-pixel gate = max agreement of any evidence source with the
        # fused label. Averaged fused probabilities stay dilute early on (the
        # gate would never open), so strong SAM claims and prototype matches let
        # pixels through until the decoder's own softmax concentrates and
        # dominates the max.
        sem_max = sem_probs.max(1).values
        if bank is not None and bank.initialized.any():
            proto_conf = bank.prototype_confidence(features, labels)
            proto_conf = F.interpolate(proto_conf.unsqueeze(1),
                                       size=labels.shape[-2:], mode="bilinear",
                                       align_corners=False)[:, 0]
        else:
            proto_conf = torch.zeros_like(conf)
        conf = torch.maximum(torch.maximum(sem_max, sam_claim), proto_conf)

        if getattr(cfg, 'adaptive_gate', False):
            # E4+: percentile-based adaptive gate.  Compute the q-th percentile
            # of the actual per-pixel confidence across this image and gate out
            # the least-confident (1-q) fraction.  This guarantees a fixed
            # fraction of pixels is always gated through, regardless of the
            # absolute softmax scale (which may plateau at ~0.6 — well below
            # a fixed 0.70 threshold, leaving the gate permanently open).
            flat_conf = conf.flatten()
            n = flat_conf.numel()
            k = max(1, int(n * cfg.adaptive_gate_percentile))
            sorted_conf, _ = flat_conf.sort()
            gate_tau = sorted_conf[min(k, n - 1)].item()
            # Clamp to [min, max] so the gate never becomes trivial
            gate_tau = max(cfg.adaptive_gate_min, min(cfg.adaptive_gate_max, gate_tau))
        else:
            gate_tau = _ramp(cfg.pseudo_conf_min, cfg.pseudo_conf_max,
                             epoch, cfg.tau_ramp_epochs)

    # --- E5: class-aware gate ---
    # Apply per-class threshold multipliers and teacher-prototype agreement.
    use_cag = getattr(cfg, 'use_class_aware_gate', False)
    if use_cag:
        gate_multi = getattr(cfg, 'class_pseudo_gate_multi', {})
        agree_gate = getattr(cfg, 'class_agreement_gate', {})
        # Flatten labels for per-class operations
        labels_flat = labels.reshape(-1)
        conf_flat = conf.reshape(-1)
        valid_flat = (conf >= gate_tau).reshape(-1)
        num_cls = labels.shape[1] if labels.dim() == 3 else cfg.num_classes

        # Apply per-class threshold multipliers
        if gate_multi:
            for cls_id, multi in gate_multi.items():
                cls_mask = (labels_flat == cls_id)
                cls_tau = gate_tau * multi
                valid_flat = valid_flat & (~cls_mask | (conf_flat >= cls_tau))

        # Apply teacher-prototype agreement gate for ambiguous classes
        if agree_gate and bank is not None and bank.initialized.any():
            # Get prototype predictions: argmax of cosine similarity per pixel
            proto_sims = bank.logits(features)  # (1, C, Hf, Wf)
            proto_labels = proto_sims.argmax(1)[0].reshape(-1)  # (Hf*Wf,)
            # Align proto_labels to image resolution if needed
            if proto_labels.shape[0] != labels_flat.shape[0]:
                proto_labels_img = F.interpolate(
                    proto_sims.argmax(1).float().unsqueeze(1),
                    size=labels.shape[-2:], mode='nearest'
                )[:, 0].long().reshape(-1)
            else:
                proto_labels_img = proto_labels
            for cls_id in agree_gate:
                cls_mask = (labels_flat == cls_id)
                disagree = cls_mask & (labels_flat != proto_labels_img)
                valid_flat = valid_flat & ~disagree

        valid = valid_flat.reshape(labels.shape)
    else:
        valid = conf >= gate_tau

    if a is None:
        valid = torch.zeros_like(valid)               # no point evidence -> skip image
    return labels, conf, valid, fused, a, b
