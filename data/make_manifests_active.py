"""Generate an active-learning training manifest for E6.

Instead of fixed 5 points per class (random interior), this script:
1. Loads the SAM encoder and runs a forward pass on each training image
2. For each class, ranks candidate interior pixels by **prediction entropy**
   (high entropy = model is uncertain = most informative to label)
3. Applies farthest-point sampling in FEATURE SPACE to ensure diversity
4. Produces `train_active.json` with the same schema as `train.json`

The result: each image gets up to POINTS_PER_CLASS well-chosen points that
maximize the information gained per annotation. This is a one-time offline
step — the manifest is consumed by the same training pipeline.

Usage (from repo root):
    .venv/bin/python -m point_only_sam_rs_Es_5pt.data.make_manifests_active
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .class_map import NUM_CLASSES, UNLABELLED

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PACKAGE_ROOT / "data"

DLRSD = PROJECT_ROOT / "dlrsd"
POINT_MASKS = DLRSD / "point_1cmasks"
DENSE_TRAIN_MASKS = DLRSD / "train_1cmasks"
TRAIN_IMAGES = DLRSD / "train_images"
SAM_CHECKPOINT = PROJECT_ROOT / "sam_vit_b_01ec64.pth"

POINTS_PER_CLASS = 5
SEED = 42
IMAGE_SIZE = 256
SAM_INPUT_SIZE = 1024  # SAM encoder expects 1024x1024 input
ENTROPY_FRACTION = 0.6   # fraction of points chosen by entropy ranking
DIVERSITY_FRACTION = 0.4  # fraction chosen by farthest-point diversity


def _load_sam_encoder(sam_ckpt: str, device: str):
    """Load SAM ViT-B image encoder (no decoder needed)."""
    import sys
    # Need to import from the model package
    sys.path.insert(0, str(PROJECT_ROOT))
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    sam.eval()
    return sam.image_encoder.to(device)


@torch.no_grad()
def _extract_features(encoder, image_np: np.ndarray, image_size: int, device: str):
    """Run SAM encoder on a single image. Returns (C, Hf, Wf) feature tensor.

    SAM expects 1024x1024 input; we resize the 256x256 image up, encode,
    and the output feature map is 64x64 (1024/16).  We keep features at
    SAM resolution for accurate entropy/diversity computation.
    """
    # Resize to SAM input resolution
    img_large = cv2.resize(image_np, (SAM_INPUT_SIZE, SAM_INPUT_SIZE),
                           interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img_large).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    feat = encoder(img)  # (1, 256, 64, 64)
    return feat[0].cpu()  # (256, 64, 64)


def _compute_entropy(feature_np: np.ndarray, class_prototypes: np.ndarray,
                     candidate_mask: np.ndarray) -> np.ndarray:
    """Compute per-pixel prediction entropy for candidate pixels.

    Args:
        feature_np: (D, H, W) L2-normalized features
        class_prototypes: (C, D) L2-normalized class prototypes
        candidate_mask: (H, W) bool mask of candidate pixels

    Returns:
        entropy: (H, W) float, high = more uncertain
    """
    d, h, w = feature_np.shape
    # Cosine similarity -> softmax -> entropy
    sims = np.einsum("dhw,cd->chw", feature_np, class_prototypes)  # (C, H, W)
    # Temperature-scaled softmax
    sims = sims / 0.07  # temperature
    sims = sims - sims.max(axis=0, keepdims=True)  # stability
    exp_sims = np.exp(sims)
    probs = exp_sims / (exp_sims.sum(axis=0, keepdims=True) + 1e-10)
    # Entropy
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=0)  # (H, W)
    entropy[~candidate_mask] = -1.0  # exclude non-candidates
    return entropy


def _farthest_point_sampling(features: np.ndarray, n_points: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Farthest-point sampling in feature space.

    Args:
        features: (N, D) L2-normalized features of candidate pixels
        n_points: number of points to select
        rng: numpy random generator

    Returns:
        selected_indices: (n_points,) indices into features
    """
    n = len(features)
    if n <= n_points:
        return np.arange(n)

    selected = [int(rng.integers(0, n))]
    min_dist = np.full(n, np.inf)

    for _ in range(n_points - 1):
        last = features[selected[-1]]
        dists = 1.0 - features @ last  # cosine distance
        min_dist = np.minimum(min_dist, dists)
        min_dist[selected] = -1.0  # exclude already selected
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

    return np.array(selected)


def _build_class_prototypes_from_points(mask: np.ndarray, features: np.ndarray,
                                         num_classes: int) -> np.ndarray:
    """Build class prototypes from the existing point annotations.

    Args:
        mask: (H, W) Chakraborty point annotation map (1..17 = class)
        features: (D, H, W) encoder features (L2-normalized)

    Returns:
        prototypes: (C, D) per-class mean feature
    """
    d, h, w = features.shape
    prototypes = np.zeros((num_classes, d), dtype=np.float32)
    for c in range(1, num_classes + 1):
        ys, xs = np.where(mask == c)
        if len(ys) == 0:
            continue
        # Sample features at point locations (nearest grid cell)
        feats = []
        for y, x in zip(ys, xs):
            feats.append(features[:, y, x])
        prototypes[c - 1] = np.mean(feats, axis=0)
    # L2 normalize
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-10
    prototypes = prototypes / norms
    return prototypes


def select_active_points(features: np.ndarray, class_mask: np.ndarray,
                          dense_mask: np.ndarray, point_mask: np.ndarray,
                          class_prototypes: np.ndarray, per_class: int,
                          rng: np.random.Generator) -> list:
    """Select active points for one image using entropy + diversity.

    Args:
        features: (D, Hf, Wf) encoder features
        class_mask: (H, W) point annotation map (1..17)
        dense_mask: (H, W) dense GT mask (1..17), or None
        point_mask: (H, W) Chakraborty point annotations (1..17)
        class_prototypes: (C, D) per-class prototypes from point annotations
        per_class: target points per class
        rng: random generator

    Returns:
        points: list of [x, y, class_id_0indexed]
    """
    d, hf, wf = features.shape  # typically (256, 64, 64) for SAM ViT-B
    h, w = class_mask.shape      # (256, 256) image resolution
    feat_norm = features / (np.linalg.norm(features, axis=0, keepdims=True) + 1e-10)

    # SAM features: 1024 input -> 64x64 feature map (stride 16)
    # Points are at 256x256 image resolution
    # Feature coord = point_coord * (feat_size / SAM_INPUT_SIZE) = point_coord * 64/1024 = point_coord / 16
    # But we're using 256x256 images, so: feature_coord = point_coord * (64 / 256) = point_coord / 4
    feat_scale_x = wf / w  # 64/256 = 0.25
    feat_scale_y = hf / h  # 64/256 = 0.25

    points = []
    for c in range(1, NUM_CLASSES + 1):
        # Candidate pixels: class c present in dense_mask
        source = dense_mask if dense_mask is not None else class_mask
        candidates_ys, candidates_xs = np.where(source == c)

        if len(candidates_ys) == 0:
            continue

        n = min(per_class, len(candidates_ys))

        # Map candidate pixel coords to feature map coords
        feat_ys = np.clip((candidates_ys * feat_scale_y).astype(int), 0, hf - 1)
        feat_xs = np.clip((candidates_xs * feat_scale_x).astype(int), 0, wf - 1)

        # Extract features at candidate locations
        cand_features = feat_norm[:, feat_ys, feat_xs].T  # (N, D)
        cand_features = cand_features / (np.linalg.norm(cand_features, axis=1, keepdims=True) + 1e-10)

        # Compute entropy at each candidate
        # Use prototype-based softmax for entropy
        sims = cand_features @ class_prototypes.T  # (N, C)
        sims = sims / 0.07
        sims = sims - sims.max(axis=1, keepdims=True)
        exp_sims = np.exp(sims)
        probs = exp_sims / (exp_sims.sum(axis=1, keepdims=True) + 1e-10)
        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)

        # Split into entropy-based and diversity-based selections
        n_entropy = max(1, int(n * ENTROPY_FRACTION))
        n_diversity = n - n_entropy

        selected_local = []

        # 1. Entropy-based: pick highest entropy points
        if n_entropy > 0 and len(entropy) > 0:
            top_idx = np.argsort(-entropy)[:n_entropy]
            selected_local.extend(top_idx.tolist())

        # 2. Diversity-based: farthest-point sampling from remaining
        remaining = np.setdiff1d(np.arange(len(cand_features)), np.array(selected_local))
        if n_diversity > 0 and len(remaining) > 0:
            remaining_feats = cand_features[remaining]
            div_idx = _farthest_point_sampling(remaining_feats, min(n_diversity, len(remaining)), rng)
            selected_local.extend(remaining[div_idx].tolist())

        # Fill any remaining slots
        used = set(selected_local)
        for idx in range(len(cand_features)):
            if len(selected_local) >= n:
                break
            if idx not in used:
                selected_local.append(idx)
                used.add(idx)

        selected_local = selected_local[:n]

        for idx in selected_local:
            y = int(candidates_ys[idx])
            x = int(candidates_xs[idx])
            points.append([x, y, c - 1])

    return points


def build_train_active(sam_checkpoint: str = None, device: str = "cuda",
                        points_per_class: int = POINTS_PER_CLASS):
    """Build the active-learning training manifest."""
    if sam_checkpoint is None:
        sam_checkpoint = str(SAM_CHECKPOINT)

    print(f"Loading SAM encoder from {sam_checkpoint}...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder = _load_sam_encoder(sam_checkpoint, str(device))
    encoder.eval()

    images = sorted(TRAIN_IMAGES.glob("*.png"))
    pmasks = sorted(POINT_MASKS.glob("*.png"))
    dmasks = sorted(DENSE_TRAIN_MASKS.glob("*.png"))
    assert len(images) == len(pmasks), (len(images), len(pmasks))

    # First pass: extract features from all images
    print("Extracting features from all training images...")
    all_features = []
    all_point_masks = []
    t0 = time.time()
    for i, (img_p, pm_p) in enumerate(zip(images, pmasks)):
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        feat = _extract_features(encoder, img, IMAGE_SIZE, str(device))
        all_features.append(feat.numpy())
        pm = cv2.imread(str(pm_p), cv2.IMREAD_GRAYSCALE)
        all_point_masks.append(pm)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(images)}] features extracted ({time.time() - t0:.1f}s)")
    print(f"Feature extraction complete: {len(all_features)} images in {time.time() - t0:.1f}s")

    # Build global class prototypes from Chakraborty's point annotations
    # Point masks are at 256x256, features at 64x64 (SAM stride 16)
    d = all_features[0].shape[0]  # 256 channels
    hf, wf = all_features[0].shape[1], all_features[0].shape[2]  # 64x64
    prototypes = np.zeros((NUM_CLASSES, d), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.int32)
    for c in range(1, NUM_CLASSES + 1):
        for i in range(len(images)):
            mask_i = all_point_masks[i]
            feat_i = all_features[i]
            ys, xs = np.where(mask_i == c)
            if len(ys) > 0:
                # Map point coords (256x256) to feature coords (64x64)
                feat_ys = np.clip((ys * hf / IMAGE_SIZE).astype(int), 0, hf - 1)
                feat_xs = np.clip((xs * wf / IMAGE_SIZE).astype(int), 0, wf - 1)
                feats_c = feat_i[:, feat_ys, feat_xs].T  # (N, D)
                prototypes[c - 1] += feats_c.sum(axis=0)
                counts[c - 1] += len(ys)
    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            prototypes[c] /= counts[c]
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-10
    prototypes = prototypes / norms
    print(f"Class prototypes built: {counts.tolist()} points used")

    # Second pass: select active points
    print("Selecting active points...")
    rows = []
    n_total = 0
    t0 = time.time()
    for i, (img_p, pm_p) in enumerate(zip(images, pmasks)):
        assert img_p.stem == pm_p.stem, (img_p, pm_p)
        feat = all_features[i]
        pm = cv2.imread(str(pm_p), cv2.IMREAD_GRAYSCALE)
        dm = cv2.imread(str(dmasks[i]), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)

        rng = np.random.default_rng(SEED + i)
        points = select_active_points(
            feat, pm, dm, pm, prototypes, points_per_class, rng
        )
        n_total += len(points)
        rows.append({
            "id": img_p.stem,
            "image": str(img_p),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "points": points,
        })
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(images)}] {n_total} points, {elapsed:.1f}s")

    elapsed = time.time() - t0
    out = DATA_DIR / "train_active.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\ntrain_active.json: {len(rows)} items, {n_total} points "
          f"({points_per_class}/class max, active selection) in {elapsed:.1f}s")
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate active-learning training manifest")
    parser.add_argument("--checkpoint", default=None,
                        help="SAM checkpoint (default: ../sam_vit_b_01ec64.pth)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-class", type=int, default=POINTS_PER_CLASS)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_train_active(args.checkpoint, args.device, args.points_per_class)


if __name__ == "__main__":
    main()
