"""LoRA adaptation of the frozen SAM image encoder + EMA teacher utilities."""
from copy import deepcopy

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        delta = torch.matmul(self.dropout(x), self.A.t())
        delta = torch.matmul(delta, self.B.t()) * self.scale
        return self.base(x) + delta


def _replace_linear(module: nn.Module, name: str, rank: int, alpha: float, dropout: float):
    child = getattr(module, name)
    if isinstance(child, nn.Linear) and child.in_features > 128:
        setattr(module, name, LoRALinear(child, rank, alpha, dropout))


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
    """Freeze the whole model, then wrap the attention MLP linears with LoRA."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        for name in ("qkv", "proj", "lin1", "lin2"):
            if hasattr(module, name):
                _replace_linear(module, name, rank, alpha, dropout)
    return model


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def clone_teacher(student: nn.Module):
    teacher = deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def ema_update(student: nn.Module, teacher: nn.Module, decay: float):
    with torch.no_grad():
        s = dict(student.named_parameters())
        for name, tp in teacher.named_parameters():
            tp.mul_(decay).add_(s[name], alpha=1.0 - decay)
        sb = dict(student.named_buffers())
        for name, tb in teacher.named_buffers():
            if name in sb:
                tb.copy_(sb[name])
