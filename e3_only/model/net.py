"""PRISM: Point-inventory, Region-consistency, Illumination-invariant Semantic Mapping.

Frozen SAM ViT-B + LoRA, an illumination-invariant full-resolution stem, and a
progressive decoder ending in a multi-prototype cosine classifier.

Two things this deliberately drops from E3
------------------------------------------
* ``sam_class_logits``. E3 called SAM's mask decoder once per class, 17 times per
  step, to build a one-hot mask vote that then entered the teacher target with
  weight 0.25. It dominated the step cost (~1.24 s/step, ~835 s/epoch) and the
  geometry it produced is the same geometry the cached partition supplies for
  free. Removing it is both the main speed-up and a removal of a noisy target
  component.
* Feeding [0,1] pixels straight into SAM's image encoder. SAM was trained on
  ``(x*255 - pixel_mean) / pixel_std`` with mean ~123 and std ~58, so an input in
  [0,1] presents the encoder with roughly 1/60th of the contrast it expects. E3
  did exactly that -- ``encode()`` interpolates and calls ``image_encoder``
  directly with no normalisation -- and LoRA has been spending capacity papering
  over it ever since. ``sam_normalize=True`` fixes it; the flag exists so the old
  behaviour stays reproducible for the comparison row.
"""
import contextlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from segment_anything import sam_model_registry

from .decoder_v2 import ProgressiveDecoder
from .invariant_stem import InvariantStem
from .lora import inject_lora

_SAM_INPUT = 1024


class PrismNet(nn.Module):
    def __init__(self, checkpoint: str, num_classes: int, device: str,
                 lora_rank: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0,
                 stem_channels: int = 32, embed_dim: int = 64,
                 prototypes_per_class: int = 4, invariant_stem: bool = True,
                 invariant_window: int = 15, dilated_context: bool = True,
                 k_temperature: float = 0.20, scale_init: float = 12.0,
                 sam_normalize: bool = True):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.sam.to(device)
        inject_lora(self.sam.image_encoder, lora_rank, lora_alpha, lora_dropout)
        # the prompt encoder and mask decoder are unused now that the region
        # partition is cached; keep them frozen and out of the optimiser
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = False

        self.stem = InvariantStem(stem_channels, invariant_window, invariant_stem).to(device)
        self.decoder = ProgressiveDecoder(
            256, num_classes, stem_channels=stem_channels, embed_dim=embed_dim,
            prototypes_per_class=prototypes_per_class, k_temperature=k_temperature,
            scale_init=scale_init, dilated_context=dilated_context).to(device)

        self.num_classes = num_classes
        self.device = device
        self.sam_normalize = sam_normalize
        # Set by ``checkpointed()``. Off by default, so a run without the shadow
        # branch is bit-identical to every run measured before this existed.
        self.grad_checkpoint = False

    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def checkpointed(self):
        """Recompute the 12 ViT blocks in backward instead of storing them.

        WHY THIS EXISTS. Images are upsampled 256 -> 1024 before the encoder, so
        one grad-carrying pass holds 4096 tokens of activation through 12 blocks
        (~11.8 GiB of the 15.57 GiB card at batch 1). The shadow equivariance
        term needs a SECOND grad-carrying pass over a shadowed copy of the same
        image, and two such graphs do not fit: with ``w_shadow > 0`` the run dies
        at the first shadowed step with ``OutOfMemoryError`` in
        ``attn.softmax``. THAT -- not a design preference -- is why every run
        measured so far used ``--ablation no-shadow-improved``, and why the
        shadow mechanism had never actually been executed.

        Trading recompute for memory makes it fit. Scope it to the shadowed pass
        so the main pass keeps its speed:

            with model.checkpointed():
                shadow_out = model(shadowed)
        """
        was, self.grad_checkpoint = self.grad_checkpoint, True
        try:
            yield
        finally:
            self.grad_checkpoint = was

    def _encode_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """``ImageEncoderViT.forward`` with each block wrapped in a checkpoint.

        Reimplemented rather than wrapped in modules so that no parameter is
        renamed: a checkpoint written with this on loads into a model with it
        off, and vice versa.
        """
        enc = self.sam.image_encoder
        x = enc.patch_embed(x)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed
        for blk in enc.blocks:
            x = _ckpt.checkpoint(blk, x, use_reentrant=False)
        return enc.neck(x.permute(0, 3, 1, 2))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [0,1] -> (B,256,64,64)."""
        x = F.interpolate(image, size=(_SAM_INPUT, _SAM_INPUT), mode="bilinear",
                          align_corners=False)
        if self.sam_normalize:
            x = (x * 255.0 - self.sam.pixel_mean) / self.sam.pixel_std
        if self.grad_checkpoint and torch.is_grad_enabled():
            return self._encode_blocks(x)
        return self.sam.image_encoder(x)

    def forward(self, image: torch.Tensor, out_size=None) -> Dict[str, torch.Tensor]:
        feat = self.encode(image)
        stem_full, stem_half = self.stem(image)
        out = self.decoder(feat, stem_full, stem_half, out_size or image.shape[-2:])
        out["shadow_prob"] = out["shadow_logit"].sigmoid()[:, 0]
        return out

    def forward_shadow_branch(self, image: torch.Tensor, out_size=None) -> Dict[str, torch.Tensor]:
        """Second forward pass on a synthetically shadowed copy.

        Kept as its own method to make the cost explicit: it is a full extra
        encoder pass, so it roughly doubles the step time when enabled. That is
        the price of the equivariance term and it is why the config ramps it in
        rather than running it from step one.
        """
        return self.forward(image, out_size)

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_report(self) -> str:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and n.startswith("sam.") and
                   any(seg in n for seg in (".A", ".B")))
        stem = sum(p.numel() for p in self.stem.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return (f"params total {tot / 1e6:.1f}M  trainable {tr / 1e6:.2f}M "
                f"(lora {lora / 1e6:.2f}M, stem {stem / 1e6:.2f}M, decoder {dec / 1e6:.2f}M)")


@torch.no_grad()
def point_embeddings(embed: torch.Tensor, points: List[torch.Tensor], patch: int = 3):
    """Collect (N,D) annotated-pixel features and their (N,) classes.

    Used once at initialisation to seed the prototypes by FINCH clustering. Reads
    only pixels a human clicked.
    """
    feats, labels = [], []
    b, d, h, w = embed.shape
    r = patch // 2
    for bi, p in enumerate(points):
        if p is None or not len(p):
            continue
        for x, y, c in p.tolist():
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            x0, x1 = max(0, xi - r), min(w, xi + r + 1)
            y0, y1 = max(0, yi - r), min(h, yi + r + 1)
            feats.append(F.normalize(embed[bi, :, y0:y1, x0:x1].mean(dim=(1, 2)), dim=0))
            labels.append(int(c))
    if not feats:
        return None, None
    return torch.stack(feats), torch.tensor(labels, device=embed.device)
