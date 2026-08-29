"""E3 — E2 + EMA teacher, SAM prompt masks, gated pseudo-labels."""
from dataclasses import replace

from PRISM.e3_only.configs.e2_prototypes import CONFIG as E2

CONFIG = replace(E2, experiment="E3",
                 use_teacher_student=True, use_sam_prompt_masks=True)
