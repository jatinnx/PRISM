"""Numerically verify the invariance theorem in core.shadow.

The claim being checked -- that channels 0..4 of ``invariant_channels`` are
EXACTLY unchanged by a per-channel gain that is constant on the invariance
window -- is a property of the transform, so it can be verified to machine
precision rather than argued about. Run this before trusting the shadow branch;
it takes a couple of seconds and needs no data, no checkpoint and no GPU.

Three cases, in increasing difficulty:

  GLOBAL      alpha constant over the whole image. The theorem's hypothesis holds
              everywhere, so the residual must be at float32 round-off.
  ACHROMATIC  alpha_R = alpha_G = alpha_B. Included because the *arithmetic*
              luminance the first version of this code used is invariant in this
              case only -- it is the case that hides the bug.
  LOCAL       alpha constant on a large blob, varying across its rim. The
              theorem does not apply within half a window of the rim, so the
              interior must be exact and the rim must not be. That the rim
              residual is large is not a failure; it is the honest statement of
              the method's limit, and the reason the learned equivariance term
              exists at all.

Also reported: the same measurement for a RAW-RGB stem input, which is what an
ordinary high-resolution skip connection would carry. The contrast between the
two residuals is the quantitative case for the invariant stem.
"""
import argparse

import torch

from ..core.shadow import invariant_channels, synth_shadow

INV = 5          # channels 0..4 are the invariant block
NAMES = ["log(R/G)-mu", "log(B/G)-mu", "logY-mu", "std(logY)", "|grad logY|"]


def _report(title: str, a: torch.Tensor, b: torch.Tensor, restrict=None):
    d = (a - b).abs()
    scale = a.abs().amax(dim=(0, 2, 3)).clamp_min(1e-6)
    if restrict is not None:
        m = restrict[:, None].expand_as(d)
        per = torch.stack([(d[:, c:c + 1][m[:, c:c + 1]]).amax() if m[:, c].any()
                           else torch.zeros((), dtype=d.dtype)
                           for c in range(d.shape[1])])
    else:
        per = d.amax(dim=(0, 2, 3))
    print(f"\n{title}")
    for c in range(min(INV, d.shape[1])):
        rel = float(per[c] / scale[c])
        flag = "OK  " if rel < 1e-4 else ("~   " if rel < 1e-2 else "FAIL")
        print(f"  {flag} ch{c} {NAMES[c]:<16} max|delta| {float(per[c]):.3e}  "
              f"rel {rel:.2e}")
    inv_rel = float((per[:INV] / scale[:INV]).max())
    print(f"  worst relative deviation over the invariant block: {inv_rel:.3e}")
    return inv_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    W = a.window
    # a textured image with structure at several scales, so the local statistics
    # are non-trivial and a spurious invariance cannot pass by accident
    img = torch.rand(a.batch, 3, a.size, a.size)
    for k in (4, 16, 64):
        img = img + torch.nn.functional.interpolate(
            torch.rand(a.batch, 3, k, k), size=(a.size, a.size), mode="bicubic",
            align_corners=False)
    img = (img / img.amax(dim=(1, 2, 3), keepdim=True)).clamp(0.02, 1.0)

    phi = invariant_channels(img, W)
    print(f"image {tuple(img.shape)}  window {W}  dtype {img.dtype}")

    # ---- 1. global chromatic gain ---------------------------------- #
    alpha = torch.tensor([0.42, 0.38, 0.55])[None, :, None, None]   # blue-rich umbra
    g = _report("GLOBAL chromatic gain  alpha = (0.42, 0.38, 0.55)",
                phi, invariant_channels((img * alpha).clamp(1e-4, 1.0), W))

    # ---- 2. global achromatic gain --------------------------------- #
    ac = _report("ACHROMATIC gain  alpha = 0.40 on every channel",
                 phi, invariant_channels((img * 0.40).clamp(1e-4, 1.0), W))

    # ---- 3. local gain: interior exact, rim not --------------------- #
    shadowed, mask = synth_shadow(img, prob=1.0, penumbra=2.5)
    psi = invariant_channels(shadowed, W)
    m = mask > 0.99
    # erode by the window radius: the theorem's hypothesis holds only where the
    # whole window sits inside the constant-alpha region
    r = W // 2 + 3
    interior = -torch.nn.functional.max_pool2d(-m.float()[:, None], 2 * r + 1,
                                               stride=1, padding=r)[:, 0] > 0.5
    rim = (mask > 0.05) & (mask < 0.95)
    print(f"\nLOCAL synthesised shadow: umbra {float(m.float().mean()):.1%} of pixels, "
          f"interior after eroding {r}px {float(interior.float().mean()):.1%}, "
          f"rim {float(rim.float().mean()):.1%}")
    li = _report("  ...umbra INTERIOR (theorem applies)", phi, psi, interior) \
        if interior.any() else 0.0
    lr = _report("  ...penumbra RIM (theorem does NOT apply -- large is correct)",
                 phi, psi, rim) if rim.any() else 0.0

    # ---- 4. the raw-RGB comparison --------------------------------- #
    raw = (img - (img * alpha).clamp(1e-4, 1.0)).abs().amax()
    print(f"\nRAW RGB under the same global gain: max|delta| {float(raw):.3e}  "
          f"(this is what a plain high-resolution skip would carry into the decoder)")

    print("\n" + "=" * 70)
    ok = g < 1e-4 and ac < 1e-4 and li < 1e-3
    print("VERDICT:", "invariance holds as proved" if ok else
          "INVARIANCE VIOLATED -- do not trust the shadow branch")
    if li and lr:
        print(f"rim/interior residual ratio {lr / max(li, 1e-12):.1f}x  "
              f"-- the penumbra is exactly the residual the learned "
              f"equivariance term is there to cover")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
