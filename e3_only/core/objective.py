"""The PRISM training objective.

    L = L_point + w_prop L_prop                      (human labels: ~15 clicks, then their regions)
      + w_abs  L_abs  + w_pres L_pres + w_area L_area (the point inventory, as a dense set constraint)
      + w_phead L_phead                              (the point inventory, as a prediction target)
      + w_hom  L_hom  + w_bnd  L_bnd  + w_potts L_potts (geometry: frozen partition + image evidence)
      + w_sh   L_sh   + w_shead L_shead              (illumination: physical shadow model)
      + w_self L_self                                (self-training, projected onto all of the above)
      + w_anchor L_anchor + w_repel L_repel          (prototype geometry)

Why this decomposition and not another
--------------------------------------
The measured problem with E3 is not that its loss was too weak, it is that its
loss was almost entirely a function of the network's own output. Of the three
terms that carried weight -- ``point_cross_entropy`` (~15 pixels),
``pseudo_cross_entropy`` and ``consistency_loss`` (both derived from the EMA
teacher) -- two were self-referential and one was 0.02% of the pixels. A loop like
that has no mechanism for noticing its own mistakes, and the eval confirms the
prediction: mIoU 54.17 at epoch 30 falls to 50.37 at epoch 50, with every one of
the 17 classes losing ground. Training longer made it worse.

So the terms are grouped by *where their information comes from*, and the group
that comes from the network is both the smallest and the last to switch on:

  human-derived   L_point L_prop L_abs L_pres L_area L_phead   annotations only
  image-derived   L_potts L_bnd L_sh L_shead           image formation only
  frozen-geometry L_hom L_bnd                          partition fixed before step 1
  model-derived   L_self                               and only through the filter of the rest

L_self cannot reinforce an error that violates the inventory (the target is
projected onto S first), cannot reinforce a speckle (targets are whole regions),
and cannot outvote a human label (regions containing points are excluded from it).
That is the structural reason the degradation should stop, as opposed to a hope
that a better threshold will help.

Every term below states its minimiser, because "the loss goes down" is not
evidence in a weakly-supervised setting -- the question is always what the loss is
minimised *by*, and whether that thing is the segmentation we want.

Curriculum
----------
Terms are enabled in order of how much they trust the model: annotations and image
evidence from step one, region homogeneity once the output has a shape worth
making consistent, shadow equivariance once the clean prediction is worth copying,
and region self-training last, after the adaptive gate has seen enough regions to
estimate its own statistics.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import inventory as inv
from . import regions as reg
from . import shadow as sh
from . import structure as st


# --------------------------------------------------------------------------- #
#  supervision at annotated and propagated pixels                             #
# --------------------------------------------------------------------------- #
def margin_point_loss(cos: torch.Tensor, scale: torch.Tensor,
                      points: List[torch.Tensor], class_weight: Optional[torch.Tensor],
                      margin: float = 0.20) -> torch.Tensor:
    """Additive angular margin cross-entropy at the annotated pixels.

    L = CE( s * (cos - m * onehot(y)) , y )

    With ~15 labels per image, plain cross-entropy is satisfied as soon as the
    right class is marginally ahead, which leaves the decision boundary sitting
    directly on top of the training points -- and DLRSD's similar-looking classes
    then swap wholesale on unseen pixels (failure mode 6). Requiring the target
    cosine to lead by ``m`` forces an angular corridor around each prototype
    instead, which is the standard remedy when labels are few and classes are
    fine-grained.

    Minimiser: cos(f, mu_y) >= cos(f, mu_c) + m for every wrong class c, at every
    annotated pixel.
    """
    b, c, h, w = cos.shape
    picked, target = [], []
    for bi, p in enumerate(points):
        if p is None or not len(p):
            continue
        xs = p[:, 0].round().long().clamp_(0, w - 1)
        ys = p[:, 1].round().long().clamp_(0, h - 1)
        picked.append(cos[bi, :, ys, xs].t())                    # (N, C)
        target.append(p[:, 2].long().clamp_(0, c - 1))
    if not picked:
        return cos.sum() * 0.0
    z = torch.cat(picked, 0)
    y = torch.cat(target, 0)
    z = z - margin * F.one_hot(y, c).to(z.dtype)
    return F.cross_entropy(scale * z, y, weight=class_weight)


def smoothed_region_ce(logits: torch.Tensor, labels: torch.Tensor,
                       present: torch.Tensor, eps=0.10,
                       class_weight: Optional[torch.Tensor] = None,
                       ignore: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-entropy on propagated labels, smoothed *inside the inventory*.

    Propagation is not exact: a region that straddles a boundary hands its
    minority pixels the wrong class. tools/validate_regions.py measures that rate
    directly, and ``eps`` should be set to it. Ordinary label smoothing would
    spread the residual mass over all 17 classes including ones the image does not
    contain, undoing L_abs; here it is spread only over S, so the two terms agree.

    ``eps`` may be a scalar or a (C,) tensor indexed by the propagated label. The
    per-class form exists because the noise is not uniform across classes: dock
    and ship propagate through thin waterfront regions that straddle each other,
    while field and sea occupy regions a SAM mask captures whole. Measured, the
    spread is an order of magnitude (tools/validate_regions.py: per-class
    1 - purity from 0.000 to 0.507). Using the mean for every class asks the
    network to distrust its most reliable labels and trust its least.

    Where the 17 numbers come from matters for the point-only claim: NOT from
    validate_regions, which reads dense masks. tools/measure_prop_trust.py
    estimates each class's relative risk from how often its point regions also
    contain a foreign point -- annotations only -- and only the global scale comes
    from the already-declared measured scalar. Rank agreement with the dense-mask
    truth is Spearman +0.801 over the 17 classes.

    Minimiser: p_y = 1 - eps_y + eps_y/|S| at every propagated pixel, i.e.
    confident but not saturated -- which is the correct target for a label known to
    be right about (1 - eps_y) of the time.
    """
    b, c, h, w = logits.shape
    valid = labels >= 0
    if ignore is not None:
        valid = valid & ~ignore
    if not valid.any():
        return logits.sum() * 0.0

    logp = F.log_softmax(logits, dim=1)
    pm = present[:, :, None, None].to(logp.dtype)
    n_present = pm.sum(1, keepdim=True).clamp_min(1.0)
    hard = F.one_hot(labels.clamp_min(0), c).permute(0, 3, 1, 2).to(logp.dtype)
    if torch.is_tensor(eps) and eps.ndim == 1:
        e = eps.to(logp.device, logp.dtype)[labels.clamp_min(0)][:, None]  # (B,1,H,W)
    else:
        e = float(eps)
    target = (1.0 - e) * hard + e * pm / n_present
    ce = -(target * logp).sum(1)

    if class_weight is not None:
        cw = class_weight[labels.clamp_min(0)]
        ce = ce * cw
        return ce[valid].sum() / cw[valid].sum().clamp_min(1e-6)
    return ce[valid].mean()


def hard_ce(logits: torch.Tensor, labels: torch.Tensor,
            class_weight: Optional[torch.Tensor] = None,
            ignore: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Plain hard-label cross-entropy on accepted self-training regions.

    Hard, not soft, on purpose. E3's target was a three-way soft blend whose peak
    was capped near 0.74 by a temperature-free prototype vote, and a KL onto a
    target that cannot exceed 0.74 caps the student's own confidence at the same
    place -- a blur floor built into the loss. A region argmax has no ceiling.
    """
    valid = labels >= 0
    if ignore is not None:
        valid = valid & ~ignore
    if not valid.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, labels.clamp_min(0), weight=class_weight,
                         reduction="none")
    return ce[valid].mean()


# --------------------------------------------------------------------------- #
#  weights and schedule                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ObjectiveWeights:
    point: float = 1.0
    prop: float = 0.60        # strong prop supervision for sparse 5-point labels
    absent: float = 0.50
    present: float = 0.20
    area: float = 0.30
    hom: float = 0.40        # region consistency prevents fragmentation
    potts: float = 0.20      # pairwise smoothness reduces speckle
    bnd: float = 0.15
    shadow: float = 0.40
    shead: float = 0.20
    self_train: float = 0.60        # V2 proven default
    pres_head: float = 0.30   # image-level presence BCE against the exact inventory
    anchor: float = 0.10     # prototypes track encoder fine at default
    repel: float = 0.05      # light repel; heavy repel hurts classification
    rim: float = 0.0          # restricted information maximisation (ablation row)

    # when each term switches on, in epochs
    e_hom: int = 1
    e_shadow: int = 3
    e_self: int = 8
    ramp: int = 3             # linear fade-in length for the gated terms


def _gate(epoch: int, start: int, ramp: int) -> float:
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (epoch - start + 1) / float(ramp))


# --------------------------------------------------------------------------- #
#  the objective                                                              #
# --------------------------------------------------------------------------- #
class PrismObjective:
    def __init__(self, w: ObjectiveWeights, num_classes: int,
                 class_weight: Optional[torch.Tensor] = None,
                 leak: float = 0.05, prop_eps: float = 0.10,
                 margin: float = 0.20, potts_sigma: float = 0.08,
                 hom_temperature: float = 0.5, min_region: int = 24,
                 edge_quantile: float = 0.80, edge_radius: int = 2,
                 gate: Optional[reg.AdaptiveRegionGate] = None,
                 classifier=None, shead_neg: float = 0.30,
                 self_min_margin: float = 0.10, js_homogeneity: bool = False,
                 soft_self: bool = False, pres_const_k: bool = False,
                 prop_eps_per_class: Optional[torch.Tensor] = None,
                 pres_head_pos_weight: float = 4.1146):
        self.w = w
        self.C = num_classes
        self.classifier = classifier
        self.shead_neg = shead_neg
        self.self_min_margin = self_min_margin
        self.js_homogeneity = js_homogeneity
        self.soft_self = soft_self
        self.pres_const_k = pres_const_k
        self.class_weight = class_weight
        self.leak = leak
        self.prop_eps = prop_eps
        self.prop_eps_per_class = prop_eps_per_class
        self.pres_head_pos_weight = pres_head_pos_weight
        self.margin = margin
        self.potts_sigma = potts_sigma
        self.hom_temperature = hom_temperature
        self.min_region = min_region
        self.edge_quantile = edge_quantile
        self.edge_radius = edge_radius
        self.gate = gate or reg.AdaptiveRegionGate()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _soft_region_ce(logits: torch.Tensor, teacher_logits: torch.Tensor,
                        ridx: reg.RegionIndex, present: torch.Tensor,
                        accepted: torch.Tensor) -> torch.Tensor:
        """Ablation row: soft region-mean target instead of a region argmax.

        This is the E3-style target restricted to regions, and it exists to make
        the "hard labels have no confidence ceiling" claim measurable rather than
        asserted. A region mean over a 17-way posterior rarely peaks above ~0.8,
        so a KL onto it caps the student at ~0.8 no matter how long it trains --
        which is the mechanism behind E3's blurred maps.
        """
        with torch.no_grad():
            p = teacher_logits.softmax(1)
            p = p * present[:, :, None, None].to(p.dtype)
            p = p / p.sum(1, keepdim=True).clamp_min(1e-6)
            target = ridx.scatter_back(ridx.mean(p))
        if not accepted.any():
            return logits.sum() * 0.0
        ce = -(target * F.log_softmax(logits, dim=1)).sum(1)
        return ce[accepted].mean()

    # ------------------------------------------------------------------ #
    def __call__(self, batch: Dict, student: Dict, epoch: int,
                 teacher: Optional[Dict] = None,
                 shadow_out: Optional[Dict] = None,
                 shadow_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = student["logits"]
        cos = student["cos"]
        scale = student["scale"]
        image = batch["image_weak"]                # affinities from the un-jittered view
        points = batch["points"]
        region = batch["region"]                   # (B,H,W) long, frozen partition
        prop = batch["prop"]                       # (B,H,W) long, -1 where unpropagated
        conflict = batch["conflict"]               # (B,H,W) bool, regions with disagreeing points

        present = inv.inventory_from_points(points, self.C, device=logits.device)
        ridx = reg.RegionIndex(region)
        log: Dict[str, float] = {}
        total = logits.sum() * 0.0

        def add(name: str, value: torch.Tensor, weight: float):
            nonlocal total
            log[name] = float(value.detach())
            if weight != 0.0:
                total = total + weight * value

        # -- human-derived ------------------------------------------------
        add("point", margin_point_loss(cos, scale, points, self.class_weight, self.margin),
            self.w.point)
        eps_prop = self.prop_eps if self.prop_eps_per_class is None \
            else self.prop_eps_per_class
        add("prop", smoothed_region_ce(logits, prop, present, eps_prop,
                                       self.class_weight, ignore=conflict),
            self.w.prop)
        add("absent", inv.absent_class_loss(logits, present, self.leak), self.w.absent)
        # floors first: L_pres sizes its top-k per class from the SAME measured
        # lower bound L_area constrains against, so a rare class is never asked to
        # produce more confident pixels than the point regions say it owns.
        floors = inv.area_floor_from_propagation(prop, self.C)
        add("present", inv.present_coverage_loss(
                logits, present,
                area_floor=None if self.pres_const_k else floors),
            self.w.present)
        add("area", inv.area_floor_loss(logits, floors), self.w.area)
        if self.w.rim:
            add("rim", inv.restricted_information_loss(logits, present), self.w.rim)
        # the inventory as a prediction target, not only as a constraint: this is the
        # one inventory term that survives to inference, where there are no points
        if self.w.pres_head and "presence_logit" in student:
            add("phead", inv.presence_head_loss(student["presence_logit"], present,
                                                self.pres_head_pos_weight),
                self.w.pres_head)
        else:
            log["phead"] = 0.0

        # -- illumination -------------------------------------------------
        g_sh = _gate(epoch, self.w.e_shadow, self.w.ramp)
        if shadow_out is not None and shadow_mask is not None:
            inside = shadow_mask > 0.25
            add("shead",
                sh.shadow_head_loss(shadow_out["shadow_logit"], shadow_mask, self.shead_neg)
                + 0.5 * sh.shadow_head_loss(student["shadow_logit"],
                                            torch.zeros_like(shadow_mask),
                                            neg_weight=1.0, restrict=inside),
                self.w.shead)
            add("shadow", sh.shadow_equivariance_loss(logits, shadow_out["logits"], shadow_mask),
                self.w.shadow * g_sh)
        else:
            log["shead"] = 0.0
            log["shadow"] = 0.0

        # -- image and frozen geometry -----------------------------------
        # the rim mask is only trustworthy once the head has been trained, so
        # boundary suppression waits for the same epoch the shadow terms do
        suppress = sh.shadow_edge_mask(student["shadow_prob"].detach()) \
            if (self.w.shead > 0 and epoch >= self.w.e_shadow) else None
        allowed = st.candidate_boundary(image, region, self.edge_quantile,
                                        self.edge_radius, suppress)
        add("potts", st.edge_aware_potts_loss(logits, image, self.potts_sigma), self.w.potts)
        add("bnd", st.boundary_precision_loss(logits, allowed), self.w.bnd)

        g_hom = _gate(epoch, self.w.e_hom, self.w.ramp)
        if self.js_homogeneity:
            add("hom", reg.region_js_divergence(logits, ridx, present,
                                                self.min_region, exclude=conflict),
                self.w.hom * g_hom)
        else:
            add("hom", reg.region_homogeneity_loss(logits, ridx, present, self.hom_temperature,
                                                   self.min_region, exclude=conflict),
                self.w.hom * g_hom)

        # -- model-derived, last and filtered ----------------------------
        g_self = _gate(epoch, self.w.e_self, self.w.ramp)
        if teacher is not None and g_self > 0:
            labels, tau, frac = reg.region_self_training_targets(
                teacher["logits"], ridx, present, self.gate,
                supervised=(prop >= 0), exclude=conflict,
                min_size=self.min_region, min_margin=self.self_min_margin)
            if self.soft_self:
                add("self", self._soft_region_ce(logits, teacher["logits"], ridx,
                                                 present, labels >= 0),
                    self.w.self_train * g_self)
            else:
                add("self", hard_ce(logits, labels, self.class_weight),
                    self.w.self_train * g_self)
            log["tau"] = tau
            log["accept"] = frac
        else:
            log["self"] = 0.0
            log["tau"] = float(self.gate.value())
            log["accept"] = 0.0

        # -- prototype geometry ------------------------------------------
        clf = self.classifier if self.classifier is not None else student.get("classifier")
        if clf is not None:
            add("anchor", clf.anchor_loss(), self.w.anchor)
            add("repel", clf.repulsion_loss(margin=0.10), self.w.repel)
        else:
            log["anchor"] = 0.0
            log["repel"] = 0.0

        log["total"] = float(total.detach())
        log["prop_px"] = float((prop >= 0).float().mean())
        log["n_present"] = float(present.float().sum(1).mean())
        return total, log
