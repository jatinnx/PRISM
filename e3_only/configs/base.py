"""Base experiment configuration for E3 (teacher-student + prototypes).

E3 = E2 (prototypes) + EMA teacher + SAM prompt masks + gated pseudo-labels.

Central invariant: **no dense masks anywhere in the training split**. The
training manifest has no ``mask`` key (enforced by the dataset) and this
codebase contains no dense cross-entropy loss at all. Training points come
from the Chakraborty paper's own point annotations.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> str:
    """Resolve a path so the caller's cwd never matters.

    Absolute paths pass through. A relative path is anchored to the PACKAGE ROOT
    first: only when nothing exists there is a cwd-relative path of the same name
    accepted. Fresh (not-yet-created) relative paths therefore land in the package
    root, which is what lets ``build_region_cache`` write the very string the
    config later reads, no matter which directory either was launched from.
    """
    if os.path.isabs(path):
        return path
    pkg_cand = PACKAGE_ROOT / path
    if pkg_cand.exists():
        return str(pkg_cand)
    if os.path.exists(path):
        return path
    return str(pkg_cand)


@dataclass
class Config:
    # --- identity ---
    experiment: str = "E0"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 2

    # --- data ---
    train_manifest: str = "data/train.json"      # points only, NO mask key
    val_manifest: str = "data/val.json"          # masks present (remapped 0..16 at write time)
    image_size: int = 256
    num_classes: int = 17                        # DLRSD classes 0..16 after remap
    background_class: int | None = None          # DLRSD has no background class

    # --- model ---
    sam_checkpoint: str = "../sam_vit_b_01ec64.pth"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    spatial_context: bool = False

    # --- training ---
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    save_dir: str = "runs/checkpoints"
    save_every: int = 5

    # --- data augmentation ---
    multi_scale_crop: bool = False
    crop_scale_lo: float = 0.5
    crop_scale_hi: float = 1.0

    # --- learning rate schedule ---
    lr_warmup_epochs: int = 2
    lr_use_cosine_decay: bool = True

    # --- experiment flags ---
    use_prototypes: bool = False         # E2+: live prototype bank + cosine reg
    use_teacher_student: bool = False    # E3: EMA teacher, pseudo labels, consistency
    use_sam_prompt_masks: bool = False   # E3: per-class SAM binary masks into fusion

    # --- losses ---
    l_point: float = 1.0        # always on (the base supervision)
    l_pseudo: float = 1.0
    l_consistency: float = 1.0
    l_proto: float = 0.5
    l_smooth: float = 0.2
    pseudo_warmup_epochs: int = 5   # l_pseudo ramps 0 -> l_pseudo over these epochs (E3+)
    pseudo_warmup_cosine: bool = False  # use cosine (True) or linear (False) schedule for warmup

    # --- pseudo-label gate (E3) ---
    pseudo_conf_min: float = 0.70
    pseudo_conf_max: float = 0.80
    tau_ramp_epochs: int = 10

    # --- prototypes (v2: LIVE bank) ---
    proto_ema: float = 0.95
    proto_feature_patch: int = 3
    proto_pixel_confidence: float = 0.80  # min fused conf for TEACHER bank pixels
    proto_sim_threshold: float = 0.30     # min cosine sim to prototype for bank acceptance
    bank_capacity: int = 512              # per-class FIFO capacity
    # NEW in v2:
    proto_refresh_every: int = 10         # full re-init from ALL train points every N epochs
    proto_self_conf_threshold: float = 0.85  # min softmax-max conf for SELF bank updates (E2)
    proto_use_refine_at_eval: bool = False   # apply bank refine() at eval time (default OFF — it
                                             # was inert at best, corrupting at worst in v1)

    # --- fusion weights (E3) ---
    fusion_w_sem: float = 0.45
    fusion_w_sam: float = 0.25
    fusion_w_proto: float = 0.30

    # --- prompts ---
    max_negative_points: int = 8
    sam_mask_threshold: float = 0.5    # sigmoid(SAM logit) must exceed this for a pixel vote

    # --- misc ---
    class_weighting: bool = True
    rare_class_factor: float = 4.0

    def __post_init__(self):
        if self.device == "cuda" and not __import__("torch").cuda.is_available():
            self.device = "cpu"


def ramp(lo: float, hi: float, epoch: int, ramp_epochs: int) -> float:
    """Linear ramp from lo to hi over ramp_epochs (clamped)."""
    if ramp_epochs <= 0:
        return hi
    t = min(1.0, epoch / ramp_epochs)
    return lo + t * (hi - lo)
