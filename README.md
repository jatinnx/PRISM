# PRISM — Point-supervised Remote-sensing Image Segmentation Model

PRISM is a point-supervised semantic segmentation model for remote sensing imagery. It uses a frozen SAM (Segment Anything Model) image encoder with LoRA adapters, invariant stems, and a multi-component curriculum loss to train from sparse point annotations only (5 points per image).

## Architecture

```
SAM ViT-B encoder (frozen) + LoRA
        ↓
Invariant Stem (15-channel: RGB × 5 augmented views)
        ↓
Dilated Context Block
        ↓
Prototype-based Decoder (4 prototypes × 17 classes)
        ↓
CosFace Angular Margin Classification
```

**Trainable parameters:** ~2.95M (out of 96.7M total)

## Repository Structure

```
PRISM/
├── e3_only/                    # Main Python package
│   ├── configs/
│   │   └── prism.py            # PRISM config (all hyperparams)
│   ├── core/
│   │   ├── losses.py           # CosFace, Potts, homogeneity losses
│   │   ├── objectives.py       # Multi-component loss orchestrator
│   │   ├── regions.py          # SAM partition + region propagation
│   │   ├── prototypes.py       # FINCH-seeded prototype bank
│   │   ├── shadow.py           # Shadow equivariance augmentation
│   │   ├── structure.py        # Boundary / Potts / homogeneity
│   │   ├── inventory.py        # Inventory set computation
│   │   └── prompts.py          # Prompt construction
│   ├── model/
│   │   ├── net.py              # PrismNet (SAM + stem + decoder)
│   │   ├── decoder.py          # Prototype decoder
│   │   └── lora.py             # LoRA adapters for SAM
│   ├── data/
│   │   ├── dataset_prism.py    # PRISM dataset + augmentations
│   │   └── dataset.py          # Base dataset
│   ├── tools/
│   │   ├── build_region_cache.py   # Build SAM region partitions
│   │   ├── validate_inventory.py   # Measure inventory leak
│   │   └── validate_regions.py     # Measure propagation purity
│   ├── train_prism.py          # Training entry point
│   ├── evaluate_prism.py       # Evaluation entry point
│   └── artifacts/              # Region caches (regenerate if missing)
├── data/                       # Manifests + val masks
│   ├── train.json              # Training manifest (points, no masks)
│   ├── val.json                # Validation manifest (masks for eval)
│   └── val_masks_remapped/     # Remapped validation masks (0..16)
├── dlrsd/                      # Dataset (see setup below)
│   ├── train_images/           # 630 training images
│   ├── full_test_images/       # 1319 validation images
│   └── train_1cmasks/          # Dense GT masks (for validation tools)
├── sam_vit_b_01ec64.pth        # SAM ViT-B checkpoint (download below)
└── README.md
```

## Environment Setup

### Option 1: Local Machine (24GB+ GPU required)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segment-anything numpy opencv-python pillow tqdm scikit-learn
```

### Option 2: Lightning AI (free 24GB L4 GPU)

```bash
# After creating a Studio with L4 GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segment-anything numpy opencv-python pillow tqdm scikit-learn
```

### Option 3: Google Colab / Kaggle

```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install segment-anything numpy opencv-python pillow tqdm scikit-learn
```

## Data Setup

### 1. Download SAM Checkpoint

Download `sam_vit_b_01ec64.pth` (375 MB) and place it in the repo root:

```bash
# Option A: from Meta's official release
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Option B: from HuggingFace
huggingface-cli download facebook/sam-vit-base sam_vit_b_01ec64.pth --local-dir .
```

### 2. Dataset (DLRSD)

Upload or copy the `dlrsd/` folder into the repo root:

```
PRISM/
└── dlrsd/
    ├── train_images/           # 630 training images (.png)
    ├── full_test_images/       # 1319 validation images (.png)
    └── train_1cmasks/          # Dense GT masks for validation tools
```

### 3. Region Caches

Pre-built caches are in `e3_only/artifacts/`. If missing, regenerate:

```bash
python -m e3_only.tools.build_region_cache --split train
python -m e3_only.tools.build_region_cache --split val
```

## Training

### Full PRISM Training (40 epochs, ~4 hours on L4)

```bash
cd PRISM

python -m e3_only.train_prism \
    --ablation full \
    --save-dir runs/prism
```

### With Custom Hyperparameters

```bash
python -m e3_only.train_prism \
    --ablation full \
    --leak 0.0000 \
    --prop-eps 0.040 \
    --save-dir runs/prism
```

### Key Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ablation` | `full` | `full`, `no-shadow`, `no-self`, `no-hom`, `point-only` |
| `--leak` | `0.0` | Inventory leak rate η (measure with `validate_inventory.py`) |
| `--prop-eps` | `0.04` | Propagation error rate (measure with `validate_regions.py`) |
| `--save-dir` | `runs/prism` | Output directory for checkpoints and logs |

### GPU Requirements

- **Minimum:** 20 GB VRAM (full training with shadow + self-training)
- **Recommended:** 24 GB (RTX 3090, RTX 4090, A5000, L4)
- **Note:** Batch size is set to 1 to fit in 16GB, but epochs 3+ need ~20GB due to dual SAM forward passes (student + shadow). Use `--ablation no-shadow` for 16GB GPUs.

## Evaluation

```bash
python -m e3_only.evaluate_prism \
    --checkpoint runs/prism/PRISM-full_best.pt \
    --save-dir runs/prism/eval
```

## Loss Function

PRISM uses a 13-component curriculum loss:

| Component | Weight | Activates | Purpose |
|-----------|--------|-----------|---------|
| `L_point` | 1.0 | Epoch 0 | CE at point locations with CosFace margin |
| `L_prop` | 0.6 | Epoch 0 | Label-smoothed CE from SAM region propagation |
| `L_abs` | 0.5 | Epoch 0 | Pushes probability into inventory set |
| `L_pres` | 0.2 | Epoch 0 | Ensures present classes own confident pixels |
| `L_area` | 0.3 | Epoch 0 | Prevents classes from shrinking below area floor |
| `L_hom` | 0.4 | Epoch 1 | Forces constant posteriors within regions |
| `L_potts` | 0.2 | Epoch 0 | Photometrically similar neighbors → same label |
| `L_bnd` | 0.15 | Epoch 0 | Penalizes unsupported contours |
| `L_shadow` | 0.4 | Epoch 3 | Shadow equivariance via synthetic augmentation |
| `L_shead` | 0.2 | Epoch 3 | Shadow detection head |
| `L_self` | 0.6 | Epoch 8 | Self-training on teacher-predicted regions |
| `L_anchor` | 0.1 | Epoch 0 | Keeps features near class prototypes |
| `L_repel` | 0.05 | Epoch 0 | Pushes different-class prototypes apart |

## Validation Tools

```bash
# Measure inventory leak rate (reports η for --leak flag)
python -m e3_only.tools.validate_inventory

# Measure propagation purity (reports 1-ε for --prop-eps flag)
python -m e3_only.tools.validate_regions
```

## License

For research use only. Contact the repository owner for commercial licensing.
