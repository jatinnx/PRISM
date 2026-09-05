"""Measure the per-class priors the Stage 2 logit adjustment is defined against.

LABEL-FREE, like measure_prop_trust.py: reads only the point annotations in
data/train.json, never a dense mask. The prior pi_c is the share of training
images whose click inventory contains class c, i.e. the frequency of the
(validated) per-image class set S(I) the whole inventory mechanism is built on.
tools/validate_inventory.py has already verified S(I) equals the image's true
class set in 630 of 630 images, so this is a measurement of the annotations, not
of the model.

Why the logit adjustment needs it (v8-plan.md, Stage 2). The model today carries
two per-class logit terms nobody designed -- rare_class_factor=4.0 in the loss
and the aggregator's LSE collapse bonus (t*log K = 0.277 per class) -- and both
correlate with class frequency, so the decision boundary is biased by it in both
directions: rare classes are sprayed into images that do not contain them
(signature A) while mid-frequency classes are absorbed by a larger spectral
neighbour (signature B). Stage 2 replaces both accidental terms with one
deliberate decision-time adjustment, z_c <- z_c - tau * log pi_c. This file is
the pi_c.

The file also writes the point-share prior (share of all clicked points a class
owns), the alternative a sweep might want; the presence prior is the default
because the accidental rare-class boost was created through the *image-level*
click inventory, not through pixel counts.

Self-check printed at the end: mean |S(I)| = sum_c pi_c must reproduce the
MEASURED constant 3.3238 that configs/prism.py quotes (mean |S(I)| over the same
630 images, from tools/validate_inventory.py). If it does not, the manifest or
the class-channel convention changed and both constants are suspect together.

Usage:
    python -m e3_only.tools.measure_class_priors [--manifest data/train.json]
    writes artifacts/class_priors.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e3_only.configs.prism import PrismConfig, resolve           # noqa: E402
from e3_only.data.class_map import CLASS_NAMES, NUM_CLASSES      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="train manifest with point annotations (default: the "
                         "config's data/train.json)")
    ap.add_argument("--out", default=None,
                    help="output json path (default: the config's "
                         "artifacts/class_priors.json)")
    a = ap.parse_args()

    cfg = PrismConfig()
    manifest = resolve(a.manifest or cfg.train_manifest)
    out = Path(resolve(a.out or cfg.class_priors_json))
    items = json.loads(Path(manifest).read_text())

    presence = Counter()          # images whose click inventory contains class c
    point_mass = Counter()        # clicked points of class c
    n_images = 0
    for it in items:
        pts = it.get("points") or []
        if not pts:
            continue
        n_images += 1
        classes = {int(p[2]) for p in pts}
        if any(c < 0 or c >= NUM_CLASSES for c in classes):
            raise ValueError(f"{it.get('id')}: class index outside 0..{NUM_CLASSES - 1}")
        for c in classes:
            presence[c] += 1
        for p in pts:
            point_mass[int(p[2])] += 1

    if n_images == 0:
        raise SystemExit(f"no images with points in {manifest}")

    n_pts = sum(point_mass.values())
    priors = {
        "presence": [presence[c] / float(n_images) for c in range(NUM_CLASSES)],
        "point_share": [point_mass[c] / float(n_pts) for c in range(NUM_CLASSES)],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors, indent=1) + "\n")

    mean_s = sum(priors["presence"])
    rows = []
    for c, name in enumerate(CLASS_NAMES):
        rows.append(f"  {name:<12s} presence={priors['presence'][c]:.4f}  "
                    f"-log pi={-np.log(priors['presence'][c]):.3f}  "
                    f"point_share={priors['point_share'][c]:.4f}")
    text = "\n".join([
        f"class priors over {n_images} train images ({manifest})",
        f"  mean |S(I)| = {mean_s:.4f}   (validate_inventory MEASURED 3.3238)",
        f"  pos weight  (C-m)/m = {(NUM_CLASSES - mean_s) / mean_s:.4f}   "
        f"(config MEASURED 4.1146)",
        *rows,
        f"wrote {out}",
    ])
    print(text)
    if not np.isclose(mean_s, 3.3238, atol=1e-3):
        print("WARNING: mean |S| does not reproduce the documented 3.3238 -- "
              "re-check the manifest / channel convention before quoting either.")


if __name__ == "__main__":
    main()
