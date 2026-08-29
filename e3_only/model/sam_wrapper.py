"""PointOnlySAM: SAM ViT-B (LoRA-adapted) + semantic decoder.

The critical correctness piece: the image encoder is run **once per step** and
the resulting embedding is shared by
  1. the semantic decoder, and
  2. the per-class SAM prompt mask decoder.

We call ``MaskDecoder.forward`` directly with the cached embedding (verified
against segment-anything 1.0: ``MaskDecoder.forward(image_embeddings,
image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
multimask_output)``). This avoids ``SamPredictor.set_image`` entirely, which
would re-run the encoder on every call. No hidden re-encoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from .decoder import SemanticDecoder
from .lora import inject_lora

_SAM_INPUT = 1024  # SAM's fixed positional-embedding input size


class PointOnlySAM(nn.Module):
    def __init__(self, checkpoint: str, num_classes: int, device: str,
                 lora_rank: int = 12, lora_alpha: float = 32.0, lora_dropout: float = 0.0,
                 background_class: int | None = None,
                 spatial_context: bool = False):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.sam.to(device)
        inject_lora(self.sam.image_encoder, lora_rank, lora_alpha, lora_dropout)
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = False
        self.decoder = SemanticDecoder(256, num_classes, spatial_context=spatial_context).to(device)
        self.num_classes = num_classes
        self.background_class = background_class
        self.device = device

    # ------------------------------------------------------------------ #
    #  encoder                                                           #
    # ------------------------------------------------------------------ #
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """One image-encoder call. Returns (1, 256, 64, 64) features.

        SAM's image encoder requires 1024x1024 input; any other resolution is
        resized first (values already in [0, 1])."""
        x = F.interpolate(image, size=(_SAM_INPUT, _SAM_INPUT), mode="bilinear",
                          align_corners=False)
        return self.sam.image_encoder(x)

    def semantic_logits(self, feat: torch.Tensor, out_size) -> torch.Tensor:
        """Semantic decoder on a *pre-computed* embedding (no re-encoding)."""
        return self.decoder(feat, out_size)

    # ------------------------------------------------------------------ #
    #  SAM prompt masks on a cached embedding                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sam_class_logits(self, embedding: torch.Tensor, image_size: int, points: torch.Tensor,
                         negative_sampler, use_negative: bool = True) -> torch.Tensor:
        """Per-class binary SAM masks, decoded from the given embedding.

        Points are in the original image frame; they are scaled into SAM's
        1024 input frame before prompting. Returns (1, C, image_size, image_size)
        logits; classes with no positive points stay at -10."""
        result = torch.full((1, self.num_classes, image_size, image_size), -10.0,
                            device=embedding.device)
        if self.background_class is not None:
            result[:, self.background_class] = 0.0
        scale = _SAM_INPUT / float(image_size)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()
        for cls in range(self.num_classes):
            if self.background_class is not None and cls == self.background_class:
                continue
            pos, neg = negative_sampler.sample(points, cls, use_negative=use_negative)
            if len(pos) == 0:
                continue
            coords = pos + neg
            labels = [1] * len(pos) + [0] * len(neg)
            pc = torch.tensor([coords], dtype=torch.float32, device=embedding.device) * scale
            pl = torch.tensor([labels], dtype=torch.int64, device=embedding.device)
            sparse, dense = self.sam.prompt_encoder(points=(pc, pl), boxes=None, masks=None)
            masks, _ = self.sam.mask_decoder(
                image_embeddings=embedding,
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )  # (1, 1, 256, 256)
            m = F.interpolate(masks, size=(image_size, image_size), mode="bilinear",
                              align_corners=False)
            result[:, cls] = m[:, 0]
        return result
