"""PRISM configuration.

A fresh dataclass rather than an extension of ``Config``: E3's config carries a
dozen fields that describe machinery PRISM deletes (fusion weights, SAM prompt
masks, the FIFO bank, the confidence gate ramp), and ``dataclasses.replace``
raises on unknown fields anyway. Keeping them separate means an E3 run and a
PRISM run cannot silently share a stale hyper-parameter.

Three constants are marked MEASURED. They are not tuning knobs -- each is a
property of the supervision itself, and each has a tool that measures it on this
dataset:

    inventory_leak         <- tools/validate_inventory.py, the "PIXEL RISK" line
    prop_eps               <- tools/validate_regions.py, 1 - propagation purity
    pres_head_pos_weight   <- tools/validate_inventory.py, mean |S(I)|

All three now carry the numbers those tools reported on DLRSD, replacing the
conservative placeholders the v2-v5 runs used. Over-estimating an error rate
weakens a constraint; under-estimating it teaches the network something false --
so the placeholders were chosen high, and the measurement moved both of the first
two down. Changing one of these is a claim about the annotations, and the claim
has to be re-measured rather than swept.

The per-class SPENDING of prop_eps (per_class_prop_eps) is a separate thing: the
scale is the measured scalar, the distribution over the 17 classes comes from
tools/measure_prop_trust.py using clicks only. Dense masks are used to validate
its rank ordering and never to produce it.
"""
import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    pkg = PACKAGE_ROOT / path
    if pkg.exists():
        return str(pkg)
    if os.path.exists(path):
        return path
    return str(pkg)


@dataclass
class PrismConfig:
    # --- identity ---
    experiment: str = "PRISM"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    # --- data ---
    train_manifest: str = "data/train.json"
    val_manifest: str = "data/val.json"
    region_cache_train: str = "artifacts/regions_train.npz"
    region_cache_val: str = "artifacts/regions_val.npz"
    image_size: int = 256
    num_classes: int = 17

    # --- model ---
    sam_checkpoint: str = "../sam_vit_b_01ec64.pth"
    lora_rank: int = 8                  # MEASURED best. rank 12 / alpha 24 was tried
    lora_alpha: float = 16.0            # (prism-v6-lora12) and tracked BEHIND rank 8:
                                        # val mIoU 0.4438 at epoch 5 against 0.4677 at
                                        # epoch 4 for rank 8. 2x rank either way.
    lora_dropout: float = 0.0
    sam_normalize: bool = True          # False reproduces E3's un-normalised encoder input
    stem_channels: int = 32
    invariant_stem: bool = True         # False = raw-RGB stem (resolution-only ablation)
    invariant_window: int = 15
    dilated_context: bool = True
    embed_dim: int = 64
    prototypes_per_class: int = 4
    k_temperature: float = 0.20
    scale_init: float = 12.0

    # --- optimisation ---
    epochs: int = 60
    batch_size: int = 2
    lr: float = 2e-4                    # decoder/stem/prototypes
    lr_backbone: float = 5e-5           # LoRA: a quarter, it adapts a pretrained prior
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    lr_warmup_epochs: int = 2
    lr_use_cosine_decay: bool = True
    ema_decay: float = 0.999
    amp: bool = True                    # bf16 forward; the loss always runs in fp32

    # --- augmentation ---
    multi_scale_crop: bool = False          # V2 best at False; True adds noise for 630 images
    crop_scale_lo: float = 0.5
    crop_scale_hi: float = 1.0
    strong_brightness: float = 0.25
    strong_contrast: float = 0.25
    strong_noise_std: float = 0.03

    # --- MEASURED supervision-error rates ---
    # Both now carry the numbers their tools actually reported on DLRSD, not the
    # conservative placeholders. Neither is a tuning knob; changing one is a claim
    # about the annotations, and the claim has to be re-measured.
    inventory_leak: float = 0.0         # MEASURED: tools/validate_inventory.py reports
                                        # PIXEL RISK 0.0000 -- 630/630 images have a
                                        # point for every class they contain, so the
                                        # absent-class constraint is exact and the
                                        # 0.05 placeholder was giving away a hard one.
    prop_eps: float = 0.040             # MEASURED: tools/validate_regions.py reports
                                        # propagation purity 0.960.
    prop_trust_json: str = "artifacts/prop_trust.json"
    per_class_prop_eps: bool = True      # spend the measured 0.040 unevenly across the
                                        # 17 classes, in proportion to a LABEL-FREE
                                        # estimate of each class's propagation risk
                                        # (tools/measure_prop_trust.py). False = one
                                        # scalar for every class, the ablation row.

    # --- loss weights (see core/objective.ObjectiveWeights) ---
    w_point: float = 1.0
    w_prop: float = 0.60                # strong prop supervision compensates for sparse 5-point labels
    w_absent: float = 0.50
    w_present: float = 0.20
    w_area: float = 0.30
    w_hom: float = 0.40                # region consistency prevents prediction fragmentation
    w_potts: float = 0.20              # pairwise smoothness reduces speckle noise
    w_bnd: float = 0.30                # doubled: trimap3 PA 0.5953 vs PA 0.7345 says
                                       # the residual error is concentrated at edges
    w_shadow: float = 0.40
    w_shead: float = 0.20
    w_self: float = 0.60               # strong self-training for spatial diversity
    w_anchor: float = 0.10              # prototypes track encoder fine at default
    w_repel: float = 0.05              # prototypes separate naturally; heavy repel hurts
    w_rim: float = 0.0
    w_pres_head: float = 0.30          # image-level multi-label BCE against S(I)

    # --- curriculum (epoch at which each group switches on) ---
    e_hom: int = 1
    e_shadow: int = 3
    e_self: int = 8
    ramp_epochs: int = 3

    # --- term hyper-parameters ---
    margin: float = 0.20                # angular margin prevents class confusion at points
    potts_sigma: float = 0.08           # colour bandwidth of the pairwise kernel
    hom_temperature: float = 0.5        # sharpening of the region-mean target
    min_region: int = 24                # regions smaller than this are ignored
    edge_quantile: float = 0.80         # per-image gradient quantile for candidate edges
    edge_radius: int = 1                # dilation of the candidate boundary band. 1 is
                                        # the floor the diagonal shifts in
                                        # structure.candidate_boundary require; 2 licensed
                                        # a 5px-wide band, which is wider than the trimap
                                        # the boundary metric scores.
    gate_kappa: float = 1.0             # tau = mu - kappa * sigma
    gate_floor: float = 0.50            # moderate floor: enough regions pass for training signal
    gate_warmup: int = 0                 # steps before gate activates; 0 = auto (1 epoch)
    self_min_margin: float = 0.10      # permissive: more pseudo-labels = more training signal

    # --- ablation switches that change a term's FORM rather than its weight ---
    # explicit fields rather than inferred from the experiment name: a run whose
    # behaviour depends on a substring of its own label is a run that changes
    # behaviour when it is renamed.
    js_homogeneity: bool = False        # L_hom as JS divergence instead of sharpened distillation
    soft_self: bool = False             # L_self onto a soft region mean instead of a region argmax
    pres_const_k: bool = False          # L_pres witness set fixed at 0.5% of pixels for every
                                        # class, instead of sized by the measured area floor

    # --- shadow branch ---
    shadow_every: int = 1               # run the shadowed pass every k-th step (cost knob)
    shadow_prob: float = 0.65
    shadow_atten_lo: float = 0.30
    shadow_atten_hi: float = 0.68
    shadow_blue_bias: float = 0.28
    shadow_penumbra: float = 2.5

    # --- prototype seeding ---
    finch_init: bool = True             # FINCH-cluster point embeddings at epoch 0
    finch_init_batches: int = 0         # 0 = auto (use steps_per_epoch; capped at epoch 0)
    finch_max_per_class: int = 3000     # subsample cap: FINCH builds an N x N similarity
    proto_ema: float = 0.95
    proto_patch: int = 3

    # --- class balance ---
    class_weighting: bool = True
    rare_class_factor: float = 4.0     # moderate upweighting; too high overweights rare classes

    # --- presence (image-level inventory head) ---
    pres_head_pos_weight: float = 4.1146  # MEASURED: mean |S(I)| = 3.3238 over 630 train
                                          # images, so (C - m)/m balances the 17-way BCE.
    presence_gate: float = 0.0            # inference-time soft gate strength; 0 = off.
    presence_floor: float = 0.05          # delta in the gate; never a hard -inf, so a
                                          # wrong presence prediction costs logit mass
                                          # rather than deleting a class outright.

    # --- inference-time class-prior logit adjustment (Stage 2, v8-plan) ---
    # Decision-time term z_c <- z_c - tau * log pi_c replacing the two accidental
    # per-class logit biases (rare_class_factor in the loss; the aggregator's LSE
    # collapse bonus). pi_c measured label-free from the click inventories by
    # tools/measure_class_priors.py; tau>0 = the balanced-prior rule (raises
    # under-represented classes, the absorption fix), tau<0 reverses (penalises
    # the spray). Sign and strength are Stage 0e/0c measurements, so the CLI
    # accepts both and the ablation row is tau=0.
    logit_adjust: float = 0.0
    logit_prior: str = "presence"         # key into class_priors_json
    class_priors_json: str = "artifacts/class_priors.json"

    # --- inference-time region vote ---
    region_vote_sam_only: bool = True      # pool only over SAM masks. MEASURED: with the
                                           # point-conflict exclusion unavailable at test
                                           # time, the 449 conflicted filler regions sit
                                           # at homogeneity purity 0.686 and would pull a
                                           # whole 8000+px blob to one class.
    region_vote_min_size: int = 24

    # --- io ---
    save_dir: str = "runs/prism"
    save_every: int = 5
    eval_every: int = 5
    log_every: int = 25
    save_preds: bool = True              # save prediction images for the best mIoU checkpoint

    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device == "cpu":
            self.amp = False


CONFIG = PrismConfig()


# --------------------------------------------------------------------------- #
#  ablation ladder for the paper                                              #
#                                                                             #
#  Each row removes exactly one mechanism from the full model, so the table    #
#  reads as a set of independent claims rather than a search over settings.    #
# --------------------------------------------------------------------------- #
def ablation(name: str) -> PrismConfig:
    c = PrismConfig(experiment=f"PRISM-{name}")
    if name == "full":
        return c
    if name == "no-inventory":            # claim: the point set is a dense constraint
        c.w_absent = c.w_present = c.w_area = 0.0
    elif name == "no-region":             # claim: a frozen partition beats a learned one
        c.w_hom = 0.0
        c.w_prop = 0.0
        c.w_self = 0.0
    elif name == "no-self":               # claim: self-training helps *once filtered*
        c.w_self = 0.0
    elif name == "soft-self":             # claim: hard region labels beat soft blends
        c.soft_self = True
    elif name == "no-shadow":             # claim: the shadow losses earn their weight
        # the losses only: ``no-invariant-stem`` is the separate row for the
        # representation, so each row moves exactly one mechanism. Running both
        # together removes the shadow story entirely, if that row is ever wanted.
        c.w_shadow = c.w_shead = 0.0
    elif name == "no-invariant-stem":     # claim: invariance, not resolution
        c.invariant_stem = False
    elif name == "no-boundary":           # claim: one-sided boundary precision helps
        c.w_bnd = 0.0
        c.w_potts = 0.0
    elif name == "single-prototype":      # claim: multi-modal classes need multi-prototypes
        c.prototypes_per_class = 1
    elif name == "no-margin":             # claim: the angular margin fixes class confusion
        c.margin = 0.0
    elif name == "js-homogeneity":        # claim: sharpened distillation beats JS
        c.js_homogeneity = True
    elif name == "const-k-present":       # claim: the MIL witness set must be sized by
        # evidence, not by a constant. With a constant k every present class must
        # claim 0.5% of the image, so a 40px class can only satisfy L_pres by
        # over-claiming ~8x. Expect this row to cost rare-class IoU and boundary
        # precision while leaving mIoU on the common classes roughly intact.
        c.pres_const_k = True
    elif name == "e3-normalisation":      # claim: the SAM input bug mattered
        c.sam_normalize = False
    elif name == "no-presence-head":      # claim: an independent image-level presence
        # estimate is worth its 0.3 weight. MEASURED motivation: 10.52% of val pixels are
        # predicted as a class the image's own point inventory excludes.
        c.w_pres_head = 0.0
    elif name == "scalar-prop-eps":        # claim: propagation trust is class-dependent
        c.per_class_prop_eps = False
    elif name == "wide-boundary":          # claim: the tighter band is what tightens edges
        c.w_bnd = 0.15
        c.edge_radius = 2
    elif name == "improved":             # optimised defaults for best mIoU
        c.gate_warmup = 0      # auto: 1 epoch of warmup
    elif name == "no-shadow-improved":   # no-shadow + gate warmup + multi-scale
        c.w_shadow = c.w_shead = 0.0
        c.gate_warmup = 0      # auto: 1 epoch of warmup
    else:
        raise KeyError(name)
    return c


# --------------------------------------------------------------------------- #
#  fields that change the shape of the weight tensors                         #
#                                                                             #
#  ``evaluate_prism`` rebuilds these from the checkpoint's stored config       #
#  before constructing the model. Without that, loading a ``single-prototype`` #
#  or ``no-invariant-stem`` checkpoint into a default-shaped model silently    #
#  drops the mismatched tensors (load_state_dict is called with strict=False   #
#  because the frozen ViT is intentionally absent) and reports a plausible     #
#  but meaningless score -- which would corrupt the ablation table rather than #
#  crash.                                                                     #
# --------------------------------------------------------------------------- #
ARCH_FIELDS = (
    "num_classes", "image_size", "lora_rank", "lora_alpha", "lora_dropout",
    "sam_normalize", "stem_channels", "invariant_stem", "invariant_window",
    "dilated_context", "embed_dim", "prototypes_per_class", "k_temperature",
    "scale_init",
)


def config_from_checkpoint(stored: dict, base: "PrismConfig" = None) -> "PrismConfig":
    """Copy the architecture fields out of a checkpoint's stored config."""
    c = base or PrismConfig()
    for k in ARCH_FIELDS:
        if k in stored:
            setattr(c, k, stored[k])
    return c
