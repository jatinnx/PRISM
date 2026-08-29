"""E2 — E1 + point-derived prototypes (prototype bank + cosine reg)."""
from dataclasses import replace

from PRISM.e3_only.configs.e1_point_only import CONFIG as E1

CONFIG = replace(E1, experiment="E2",
                 use_prototypes=True)
