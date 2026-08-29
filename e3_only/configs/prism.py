"""PRISM configuration.

A fresh dataclass rather than an extension of ``Config``: E3's config carries a
dozen fields that describe machinery PRISM deletes (fusion weights, SAM prompt
masks, the FIFO bank, the confidence gate ramp), and ``dataclasses.replace``
raises on unknown fields anyway. Keeping them separate means an E3 run and a
PRISM run cannot silently share a stale hyper-parameter.

Two constants are marked MEASURED. They are not tuning knobs -- each is an
estimate of an error rate in the supervision itself, and each has a tool that
measures it on this dataset:

    inventory_leak   <- tools/validate_inventory.py, the "PIXEL RISK" line
    prop_eps         <- tools/validate_regions.py, 1 - propagation purity

Set them from those numbers before the headline run. The defaults here are
conservative placeholders, chosen high rather than low: over-estimating the
error rate weakens a constraint, under-estimating it teaches the network
something false.
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
    lora_rank: int = 8
    lora_alpha: float = 16.0
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
    epochs: int = 40
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
    multi_scale_crop: bool = False
    crop_scale_lo: float = 0.5
    crop_scale_hi: float = 1.0
    strong_brightness: float = 0.25
    strong_contrast: float = 0.25
    strong_noise_std: float = 0.03

    # --- MEASURED supervision-error rates ---
    inventory_leak: float = 0.05        # P(a pixel's class has no point in its image)
    prop_eps: float = 0.10              # P(a propagated label is wrong)

    # --- loss weights (see core/objective.ObjectiveWeights) ---
    w_point: float = 1.0
    w_prop: float = 0.60
    w_absent: float = 0.50
    w_present: float = 0.20
    w_area: float = 0.30
    w_hom: float = 0.40
    w_potts: float = 0.20
    w_bnd: float = 0.15
    w_shadow: float = 0.40
    w_shead: float = 0.20
    w_self: float = 0.60
    w_anchor: float = 0.10
    w_repel: float = 0.05
    w_rim: float = 0.0

    # --- curriculum (epoch at which each group switches on) ---
    e_hom: int = 1
    e_shadow: int = 3
    e_self: int = 8
    ramp_epochs: int = 3

    # --- term hyper-parameters ---
    margin: float = 0.20                # CosFace additive angular margin at points
    potts_sigma: float = 0.08           # colour bandwidth of the pairwise kernel
    hom_temperature: float = 0.5        # sharpening of the region-mean target
    min_region: int = 24                # regions smaller than this are ignored
    edge_quantile: float = 0.80         # per-image gradient quantile for candidate edges
    edge_radius: int = 2                # dilation of the candidate boundary band
    gate_kappa: float = 1.0             # tau = mu - kappa * sigma
    gate_floor: float = 0.50
    self_min_margin: float = 0.10

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
    finch_init_batches: int = 100000    # batches of epoch 0 to collect from (default: all)
    finch_max_per_class: int = 3000     # subsample cap: FINCH builds an N x N similarity
    proto_ema: float = 0.95
    proto_patch: int = 3

    # --- class balance ---
    class_weighting: bool = True
    rare_class_factor: float = 4.0

    # --- io ---
    save_dir: str = "runs/prism"
    save_every: int = 5
    eval_every: int = 5
    log_every: int = 25

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
