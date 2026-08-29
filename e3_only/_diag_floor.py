"""Diagnostic: measure the sharpness of every fusion component and the
resulting hard floor on pseudo_CE + consistency. Read-only."""
import torch, torch.nn.functional as F

ck = torch.load("e3_only/runs/checkpoints/E3_epoch_0035.pt", map_location="cpu",
                weights_only=False)
P = ck["prototypes"]                       # (17, 256)
init = ck["prototypes_initialized"]
print("prototypes:", tuple(P.shape), "initialized:", int(init.sum()), "/", len(init))

Pn = F.normalize(P, dim=1)
S = Pn @ Pn.t()                            # inter-prototype cosine matrix
off = S[~torch.eye(len(S), dtype=torch.bool)]
print(f"inter-prototype cosine: mean {off.mean():.3f}  min {off.min():.3f}  max {off.max():.3f}")

# Sharpest POSSIBLE proto_probs: a pixel feature identical to prototype c.
# logit row = S[c]; max is 1.0, the rest are the inter-prototype cosines.
pp = torch.softmax(S, dim=1)               # (17, 17)
best = pp.gather(1, torch.arange(len(S)).unsqueeze(1))[:, 0]
print(f"softmax(cosine) peak, BEST case: mean {best.mean():.4f}  max {best.max():.4f}  "
      f"(uniform = {1/len(S):.4f})")

# ---- fused ceiling: sem one-hot, SAM agrees or abstains, proto near-uniform ----
w_sem, w_sam, w_pr = 0.45, 0.25, 0.30
u = float(best.mean())                     # proto peak (optimistic)
o = (1.0 - u) / (len(S) - 1)               # proto off-peak
for name, sam in (("SAM claims pixel", 1.0), ("SAM abstains", 0.0)):
    top = w_sem * 1.0 + w_sam * sam + w_pr * u
    oth = w_pr * o
    Z = top + (len(S) - 1) * oth
    f_top = top / Z
    print(f"  {name:18s}: fused peak ceiling = {f_top:.4f}  ->  "
          f"pseudo_CE floor = -log(peak) = {-torch.log(torch.tensor(f_top)):.4f}")

# ---- joint floor of  conf*CE(student, argmax fused) + KL(fused || student) ----
# optimum is p* proportional to (conf*onehot + fused); evaluate it exactly.
C = len(S)
for name, sam in (("SAM claims pixel", 1.0), ("SAM abstains", 0.0)):
    top = w_sem + w_sam * sam + w_pr * u
    oth = w_pr * o
    f = torch.full((C,), oth); f[0] = top; f = f / f.sum()
    for conf in (0.8, 0.9, 1.0):
        coef = f.clone(); coef[0] += conf
        p = coef / coef.sum()
        ce = conf * (-torch.log(p[0]))
        kl = (f * (f.log() - p.log())).sum()
        print(f"  {name:18s} conf={conf:.1f}:  pseudo_CE {ce:.4f} + consistency {kl:.4f} "
              f"= {ce + kl:.4f}   (student peak can only reach {p[0]:.3f})")
