"""Build a class-agnostic region partition for every training image, ONCE.

Why this exists
---------------
The measured failure of E3 is confirmation bias: mIoU 54.17 at epoch 30 decays
to 50.37 at epoch 50, with every one of the 17 classes losing ground. A
self-training loop whose only targets come from its own EMA copy has no way to
notice its own errors, so it sharpens them instead.

The fix is to add constraints that do not depend on the network. A *frozen*,
*class-agnostic* partition of each image is one such constraint: it says which
pixels must share a label without saying what that label is. Because it is
computed once from the pretrained SAM and never updated, it cannot drift with
the student, so it acts as a fixed point of reference the loop cannot corrupt.

This is also what makes 0% dense labels workable. A point that lands in a region
labels the whole region, so ~15 clicks per image become thousands of supervised
pixels — without a single dense mask being read.

What it does NOT use
--------------------
Only ``item["image"]``. No masks, no class labels, no GT of any kind. The
partition is class-agnostic by construction.

Method (self-contained grid-prompt AMG)
---------------------------------------
1. Run the *pretrained* (LoRA-free) SAM image encoder once per image.
2. Prompt a G x G grid of single points; take all 3 multimask outputs each.
3. Keep masks by predicted IoU and stability score; drop tiny/huge ones.
4. Greedy IoU-NMS to remove duplicates.
5. Paint accepted masks in descending area order, so the *smallest* mask
   containing a pixel wins -> a partition, finest-scale-first.
6. Any pixel no mask claimed becomes a filler region: a connected component of
   the residual. Filler regions are still spatially coherent, so they remain
   valid for the homogeneity term; they are flagged so their reliability can be
   measured separately.

SAM's low-res mask output is 256x256 and our images are 256x256, so masks come
back at native resolution with no resampling.

Usage
-----
    python -m e3_only.tools.build_region_cache --split train
    python -m e3_only.tools.build_region_cache --split val    # optional
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.configs.base import Config, resolve  # noqa: E402

_SAM_INPUT = 1024


# --------------------------------------------------------------------------- #
#  scoring helpers (mirrors segment-anything's own definitions)               #
# --------------------------------------------------------------------------- #
def stability_score(logits: torch.Tensor, offset: float = 1.0) -> torch.Tensor:
    """IoU between the mask thresholded at +offset and at -offset.

    A mask whose area barely changes when the threshold moves is a mask with a
    crisp boundary. (N, H, W) logits -> (N,) score.
    """
    hi = (logits > offset).flatten(1).sum(1, dtype=torch.float32)
    lo = (logits > -offset).flatten(1).sum(1, dtype=torch.float32)
    return hi / torch.clamp(lo, min=1.0)


def iou_nms(masks: torch.Tensor, scores: torch.Tensor, thresh: float = 0.70):
    """Greedy IoU non-maximum suppression over boolean masks. Returns indices."""
    order = torch.argsort(scores, descending=True)
    flat = masks.flatten(1).float()                       # (N, HW)
    areas = flat.sum(1)
    keep = []
    alive = torch.ones(len(order), dtype=torch.bool, device=masks.device)
    for pos in range(len(order)):
        if not alive[pos]:
            continue
        i = order[pos]
        keep.append(int(i))
        if pos + 1 >= len(order):
            break
        rest_pos = torch.arange(pos + 1, len(order), device=masks.device)
        rest_pos = rest_pos[alive[pos + 1:]]
        if not len(rest_pos):
            break
        j = order[rest_pos]
        inter = flat[j] @ flat[i]
        union = areas[j] + areas[i] - inter
        drop = (inter / torch.clamp(union, min=1.0)) > thresh
        alive[rest_pos[drop]] = False
    return keep


# --------------------------------------------------------------------------- #
#  partition builder                                                          #
# --------------------------------------------------------------------------- #
class RegionPartitioner:
    def __init__(self, sam_checkpoint: str, device: str = "cuda",
                 grid: int = 24, chunk: int = 96,
                 iou_thresh: float = 0.84, stab_thresh: float = 0.90,
                 nms_thresh: float = 0.70, min_area_frac: float = 0.0004,
                 max_area_frac: float = 0.92):
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device).eval()
        for p in self.sam.parameters():
            p.requires_grad = False
        self.device = device
        self.grid = grid
        self.chunk = chunk
        self.iou_thresh = iou_thresh
        self.stab_thresh = stab_thresh
        self.nms_thresh = nms_thresh
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac

    def _grid_points(self, size: int) -> torch.Tensor:
        """(G*G, 2) point prompts in the SAM 1024 frame, offset off the border."""
        g = self.grid
        step = size / g
        c = (torch.arange(g, dtype=torch.float32) + 0.5) * step
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        return pts * (_SAM_INPUT / size)

    @torch.no_grad()
    def masks_for(self, image_u8: np.ndarray) -> torch.Tensor:
        """(N, H, W) bool masks, deduplicated and score-filtered."""
        h, w = image_u8.shape[:2]
        x = torch.from_numpy(image_u8).permute(2, 0, 1).float()[None] / 255.0
        x = x.to(self.device)
        x = F.interpolate(x, size=(_SAM_INPUT, _SAM_INPUT), mode="bilinear", align_corners=False)
        # the pretrained SAM expects its own normalisation
        x = (x * 255.0 - self.sam.pixel_mean) / self.sam.pixel_std
        emb = self.sam.image_encoder(x)                                   # (1,256,64,64)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()

        pts = self._grid_points(max(h, w)).to(self.device)
        keep_masks, keep_scores = [], []
        for s in range(0, len(pts), self.chunk):
            pc = pts[s:s + self.chunk, None, :]                           # (n,1,2)
            pl = torch.ones(pc.shape[0], 1, dtype=torch.int64, device=self.device)
            sparse, dense = self.sam.prompt_encoder(points=(pc, pl), boxes=None, masks=None)
            logits, iou_pred = self.sam.mask_decoder(
                image_embeddings=emb, image_pe=dense_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=True,
            )                                                             # (n,3,256,256)
            n, k = logits.shape[:2]
            lg = logits.reshape(n * k, *logits.shape[-2:])
            iq = iou_pred.reshape(n * k)
            st = stability_score(lg)
            ok = (iq >= self.iou_thresh) & (st >= self.stab_thresh)
            if ok.any():
                m = lg[ok] > 0.0
                if m.shape[-2:] != (h, w):
                    m = F.interpolate(m[:, None].float(), size=(h, w),
                                      mode="nearest")[:, 0] > 0.5
                keep_masks.append(m)
                keep_scores.append(iq[ok] * st[ok])
            del logits, iou_pred, lg
        if not keep_masks:
            return torch.zeros(0, h, w, dtype=torch.bool, device=self.device)

        masks = torch.cat(keep_masks, 0)
        scores = torch.cat(keep_scores, 0)
        area = masks.flatten(1).sum(1).float() / float(h * w)
        band = (area >= self.min_area_frac) & (area <= self.max_area_frac)
        masks, scores = masks[band], scores[band]
        if not len(masks):
            return masks
        idx = iou_nms(masks, scores, self.nms_thresh)
        return masks[idx]

    @torch.no_grad()
    def partition(self, image_u8: np.ndarray):
        """-> (region int16 (H,W), n_sam_regions int).

        Region ids [0, n_sam) come from SAM masks; ids >= n_sam are filler
        connected components of the residual.
        """
        h, w = image_u8.shape[:2]
        masks = self.masks_for(image_u8)
        region = np.full((h, w), -1, dtype=np.int32)
        if len(masks):
            m_np = masks.cpu().numpy()
            areas = m_np.reshape(len(m_np), -1).sum(1)
            order = np.argsort(-areas)                     # large first, small paints over
            for new_id, i in enumerate(order):
                region[m_np[i]] = new_id
        n_sam = int(region.max()) + 1 if region.max() >= 0 else 0

        # filler: connected components of whatever no mask claimed
        holes = (region < 0).astype(np.uint8)
        if holes.any():
            ncc, cc = cv2.connectedComponents(holes, connectivity=8)
            for k in range(1, ncc):
                region[cc == k] = n_sam + (k - 1)
        return region.astype(np.int16), n_sam


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    manifest = args.manifest or (cfg.train_manifest if args.split == "train" else cfg.val_manifest)
    out = Path(resolve(args.out or f"artifacts/regions_{args.split}.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)

    items = json.loads(Path(resolve(manifest)).read_text())
    if args.limit:
        items = items[:args.limit]
    print(f"manifest {manifest}\nimages   {len(items)}\nout      {out}")

    part = RegionPartitioner(resolve(cfg.sam_checkpoint), device=args.device, grid=args.grid)

    ids, maps, n_sams = [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        img = cv2.imread(str(it["image"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(it["image"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (args.image_size, args.image_size):
            img = cv2.resize(img, (args.image_size, args.image_size),
                             interpolation=cv2.INTER_LINEAR)
        region, n_sam = part.partition(img)
        ids.append(it.get("id", str(i)))
        maps.append(region)
        n_sams.append(n_sam)
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            el = time.time() - t0
            print(f"  {i + 1}/{len(items)}  {el:.0f}s  "
                  f"({el / (i + 1):.2f}s/img, eta {el / (i + 1) * (len(items) - i - 1) / 60:.1f}min)  "
                  f"regions/img {np.mean([m.max() + 1 for m in maps[-25:]]):.1f}", flush=True)

    np.savez_compressed(out, ids=np.array(ids), regions=np.stack(maps),
                        n_sam=np.array(n_sams, dtype=np.int32))
    total = np.stack(maps)
    print(f"\nsaved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"regions per image : mean {np.mean([m.max() + 1 for m in maps]):.1f}  "
          f"min {min(m.max() + 1 for m in maps)}  max {max(m.max() + 1 for m in maps)}")
    print(f"SAM-covered pixels: {float((total < np.array(n_sams)[:, None, None]).mean()):.4f} "
          f"(rest are filler components)")


if __name__ == "__main__":
    main()
