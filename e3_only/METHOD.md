# PRISM — Point-inventory, Region-consistency, Illumination-invariant Semantic Mapping

Weakly-supervised semantic segmentation of DLRSD (17 classes, 256×256, 630 train /
1319 val) from **point annotations only**. No dense mask is read at training time,
anywhere, by anything. The val masks are read in exactly one file
(`evaluate_prism.py`) and in the two measurement tools, and nothing they compute
feeds back into training.

---

## 1. What was actually wrong

The predecessor (E3) reaches mIoU 0.5417 at epoch 30 and **0.5037 at epoch 50**.
Every one of the 17 classes loses ground; the worst are chaparral (−16.2 pp),
sea (−9.2 pp) and sand (−6.0 pp). Pixel accuracy falls 0.7239 → 0.7000. Training
longer made the model worse, monotonically, on every class.

That is the signature of a self-referential objective, and E3's loss is one.
Weighting the three terms that carried gradient:

| term | weight | where its information comes from | pixels it touches |
|---|---|---|---|
| `point_cross_entropy` | 1.0 | human annotation | ~15 of 65 536 (0.02%) |
| `pseudo_cross_entropy` | 1.0 | the network's own EMA copy | all |
| `consistency_loss` | 1.0 | the network's own EMA copy | all |
| `proto_reg` | 0.5 | — **inert**, logged 0.0000 for all 50 epochs | 0 |
| `boundary_smoothness` | 0.2 | — **never called** | 0 |

**67% of E3's gradient came from the network's own output**, and the 33% that
came from a human touched 0.02% of the pixels. There is no mechanism in that loop
for detecting its own error, so errors compound; the eval confirms it.

Three further defects, each measured rather than suspected:

1. **A blur ceiling built into the loss.** `PrototypeBank.logits` returned a raw
   cosine in [−1, 1] with no temperature; softmaxed over 17 classes that is
   nearly flat, so the fused teacher target peaked at ≈0.74. Cross-entropy onto a
   target with max 0.74 is minimised at *p = 0.74* (Gibbs' inequality, §4.9), so
   the student could not become sharper than its target no matter how long it
   trained.
2. **The prototype path was unreachable.** `train.py:205` guards it with
   `teacher is None`; E3 always has a teacher. `proto_reg` logged 0.0000 and
   `bank_px` logged 0 for all 50 epochs. Every prototype hyper-parameter was
   dead.
3. **SAM was fed the wrong input range.** `sam_wrapper.encode()` interpolates to
   1024² and calls `image_encoder` directly on values in [0, 1]. SAM was trained
   on `(x*255 − pixel_mean) / pixel_std` with mean ≈123, std ≈58 — the encoder was
   receiving roughly 1/60 of the contrast it expects, and LoRA has been spending
   its capacity compensating.

And one thing that was available and unused: `make_manifests.py:sample_points`
emits up to 5 grid-spread clicks **for every class present in the image**. The
clicks therefore determine, for free, the *set* of classes the image contains —
and E3 used them only as ~15 labelled pixels.

---

## 2. The observation the method is built on

Let \(S(I) \subseteq \{1..C\}\) be the set of classes that received at least one
click in image \(I\). Under the annotation protocol,

> **(A1)** every class with pixels in \(I\) has at least one click in \(I\).

Then for **every** pixel \(j\), \(y_j \in S(I)\). Measured on DLRSD,
\(|S|\) is small: the manifest carries clicks for a handful of classes per image
out of 17.

Counting the supervision two ways, per image:

- as labelled pixels: \(15 \log_2 17 \approx 61\) bits;
- as a candidate-set restriction at every pixel:
  \(65\,536 \cdot \log_2(17/|S|)\) bits — about **137 kbit** at \(|S| = 4\).

Roughly a **2000× ratio**. These bits are not independent, so this is an upper
bound rather than an effective sample size; but it is the right order of
magnitude for what E3 discarded, and it is the reason the largest single change
in this method is not a new architecture, it is reading the annotation properly.

---

## 3. The objective

$$
\begin{aligned}
\mathcal{L} =\;& \mathcal{L}_{\text{point}} + w_{\text{prop}}\mathcal{L}_{\text{prop}}
&&\text{(human labels: the clicks, then their regions)}\\
+\;& w_{\text{abs}}\mathcal{L}_{\text{abs}} + w_{\text{pres}}\mathcal{L}_{\text{pres}} + w_{\text{area}}\mathcal{L}_{\text{area}}
&&\text{(the point \emph{inventory}, as a dense set constraint)}\\
+\;& w_{\text{hom}}\mathcal{L}_{\text{hom}} + w_{\text{bnd}}\mathcal{L}_{\text{bnd}} + w_{\text{potts}}\mathcal{L}_{\text{potts}}
&&\text{(geometry: frozen partition + image evidence)}\\
+\;& w_{\text{sh}}\mathcal{L}_{\text{sh}} + w_{\text{shead}}\mathcal{L}_{\text{shead}}
&&\text{(illumination: the dichromatic model)}\\
+\;& w_{\text{self}}\mathcal{L}_{\text{self}}
&&\text{(self-training, projected onto all of the above)}\\
+\;& w_{\text{anc}}\mathcal{L}_{\text{anc}} + w_{\text{rep}}\mathcal{L}_{\text{rep}}
&&\text{(prototype geometry)}
\end{aligned}
$$

Terms are grouped **by where their information comes from**, and the group that
comes from the network is both the smallest and the last to switch on:

| information source | terms | weight | share |
|---|---|---|---|
| human annotation | point, prop, abs, pres, area, anchor | 2.70 | **57%** |
| image formation | potts, bnd, shadow, shead | 0.95 | 20% |
| model output, geometry-constrained | hom, self | 1.00 | 21% |
| regulariser | repel | 0.05 | 1% |

Compare E3: human 33% (on 0.02% of pixels), model 67%.

---

## 4. Term by term, with the minimiser

Every claim below is stated as *what the loss is minimised by*, because in a
weakly-supervised setting "the loss went down" is not evidence — the question is
always whether the minimiser is the segmentation we want.

Notation: \(p_c(j)\) is the predicted posterior at pixel \(j\); \(S\) is the
inventory; \(\mathcal{R} = \{R_1..R_M\}\) is the frozen partition; \(f_j\) is the
L2-normalised embedding; \(\mu_{c,k}\) is the \(k\)-th prototype of class \(c\).

### 4.1 `L_point` — additive angular margin at the clicks

$$\mathcal{L}_{\text{point}} = \mathrm{CE}\big(s\,(\cos_c(f_j) - m\,\mathbb{1}[c=y_j])\;,\;y_j\big),\quad j \in \text{clicks}$$

**Minimiser.** \(\cos(f_j, \mu_{y_j}) \ge \cos(f_j, \mu_c) + m\) for every wrong
class \(c\), at every annotated pixel.

**Proposition (generalisation radius).** If the margin condition holds at an
annotated pixel with embedding \(f\), then every embedding \(f'\) with
\(\lVert f' - f\rVert \le m/2\) receives the same label.

*Proof.* \(\langle f', \mu_y - \mu_c\rangle = \langle f, \mu_y - \mu_c\rangle +
\langle f' - f, \mu_y - \mu_c\rangle \ge m - \lVert f'-f\rVert\,\lVert \mu_y-\mu_c\rVert
\ge m - (m/2)(2) = 0.\) ∎

With ~15 labels per image this is the point of the margin: plain cross-entropy is
satisfied the moment the right class is marginally ahead, which leaves the
decision boundary sitting *on top of* the training points and lets DLRSD's
similar-looking classes swap wholesale on unseen pixels. Requiring a corridor of
width \(m\) buys a certified radius of \(m/2\) around every click.
→ **failure mode 6** (spectral confusion).

### 4.2 `L_prop` — point → region propagation

A region containing clicks of exactly one class is that class, everywhere.
Regions holding clicks of two classes are marked *conflict* and excluded from
**every** downstream term — a region that provably straddles a semantic boundary
is evidence the partition erred there, and forcing it constant would import the
error. This is what turns ~15 clicks into thousands of labelled pixels **using no
network output at all**.

Cross-entropy, label-smoothed **inside the inventory**:
\(t = (1-\epsilon)\,\text{onehot}(y) + \epsilon\,\mathbb{1}_S/|S|\).

**Minimiser.** \(p_y = 1-\epsilon+\epsilon/|S|\) — confident but not saturated,
which is the correct target for a label known to be right \((1-\epsilon)\) of the
time. \(\epsilon\) is **measured**, not tuned (§6). Ordinary label smoothing
would spread the residual over all 17 classes including absent ones, undoing
§4.3; restricting it to \(S\) makes the two terms agree.

### 4.3 `L_abs` — the inventory as a dense negative constraint

$$\mathcal{L}_{\text{abs}} = -\frac{1}{HW}\sum_j \log\Big(\eta + (1-\eta)\!\!\sum_{c\in S}p_c(j)\Big)$$

**Minimiser (η = 0).** \(\sum_{c \in S} p_c(j) = 1\) at every pixel, i.e.
\(\mathrm{supp}\,p_j \subseteq S\). This is the negative log-likelihood of the
*observed* information under partial-label (candidate-set) maximum likelihood: the
annotation reveals a superset of the label, and this is exactly its likelihood.
It removes \(C - |S|\) degrees of freedom at **every one of the 65 536 pixels**.

**Proposition (robustness to violations of A1).** With leak \(\eta>0\), write
\(u = \sum_{c\in S}p_c\). Then \(\mathcal{L} = -\log(\eta+(1-\eta)u)\) satisfies
\(\mathcal{L} \le -\log\eta\) and
\(\left|\partial\mathcal{L}/\partial u\right| = \frac{1-\eta}{\eta+(1-\eta)u} \le \frac{1-\eta}{\eta}\).

*Consequence.* An image in which A1 fails — a class present but never clicked —
contributes **bounded** loss and **bounded** gradient instead of an unbounded
penalty for being right. Set \(\eta\) to the measured violation rate (§6): the
term is then a correctly-specified likelihood for an annotator who misses classes
at rate \(\eta\), not an approximation of one. ∎

→ **failure mode 2** (ghost classes) is *structurally* prevented: a class absent
from \(S\) is pushed to zero over the whole image.

### 4.4 `L_pres` — multiple-instance coverage, with a witness set sized by evidence

$$\mathcal{L}_{\text{pres}} = -\frac{1}{|S|}\sum_{c\in S}\log\Big(\text{mean of the top-}k_c\ p_c(\cdot)\Big)$$

**Minimiser.** For each \(c \in S\) there exist at least \(k_c\) pixels with
\(p_c \to 1\). This is the standard MIL guarantee with the image as the bag and a
witness set of size \(k_c\).

Nothing in E3 required a class known to be present to be predicted *anywhere*,
which is why the rarest point class collapsed: **field IoU 0.0888**.

**Why \(k_c\) is per-class and not a constant.** \(k\) has exactly one job: to
spread the gradient over a neighbourhood instead of the single argmax pixel
(\(k=1\) is max-pooling and is unstable at batch size 1). It is *not* an area
estimate — that is §4.5's job, and §4.5 does it from a bound the ground truth
provably satisfies. A constant \(k = 0.5\%\ \text{of } HW = 327\) conflates the
two, and the conflation is one-sided: a class that truly covers 40 px is required
to produce 327 px of confident mass, so **the only way to satisfy the term is to
over-claim by an order of magnitude**. That pressure falls hardest on precisely
the rare classes the method exists to rescue (ship, court, tanks, mobile home),
and it surfaces as a rare class bleeding across its border — failure mode 5,
correct shape and wrong label, charged to the *neighbour's* pixels.

So the witness set is sized from the same measured floor §4.5 constrains against:

$$k_c \;=\; \mathrm{clip}\big(|P_c|,\ k_{\min},\ k_{\max}\big),\qquad k_{\min}=64,\quad k_{\max}=0.5\%\ \text{of } HW$$

with \(|P_c|\) the propagated pixel count of §4.5. A class whose own regions
already cover 3000 px is asked for \(k_{\max}\) — unchanged, the cap binds. A class
whose regions cover 40 px is asked for \(k_{\min}\), the stability floor, and
nothing more. Since \(k_c \le k_{\max}\) always, this only ever *relaxes* the term
relative to the constant-\(k\) form; the anti-flooding role is untouched, because
flooding is blocked by §4.3 (no mass outside \(S\)) and §4.5 (every present class
holds a floor), never by this ceiling. A class with no propagated region at all —
its click landed in a conflict region — gets \(|P_c| = 0 \Rightarrow k_c = k_{\min}\),
which is the right default: the inventory still asserts the class exists, and
nothing has been measured about how large it is.

\(|P_c|\) is read off integer labels and carries no gradient, so \(k_c\) is a
constant of the step and the term remains a clean top-\(k\) mean over \(p\).

→ the dual of **failure mode 2**; the per-class \(k_c\) is what keeps it from
paying for that dual in **failure mode 5**.

### 4.5 `L_area` — a model-free one-sided area floor

Let \(P_c\) be the pixels propagation assigns to \(c\) (points + geometry only,
no network), and \(1-\varepsilon\) its purity. Then the true area fraction
satisfies

$$A_c \;\ge\; (1-\varepsilon)\,\frac{|P_c|}{HW}.$$

So requiring \(\frac{1}{HW}\sum_j p_c(j) \ge \gamma\,|P_c|/HW\) with
\(\gamma \le 1-\varepsilon\) is a constraint **the ground truth never violates**.
Implemented as a hinge, \(\gamma = 0.60\), leaving slack for \(\varepsilon\) up to
0.40.

**One-sided on purpose.** Propagation covers only part of each class, so
\(|P_c|\) is a lower bound and there is no valid upper bound to impose. A
two-sided version would cap a class at its propagated area and suppress correct
predictions.
→ **failure mode 3** (flooding) from below, by giving every present class a
guaranteed floor that a flooding class must leave room for.

### 4.6 `L_hom` — region homogeneity by sharpened self-distillation

$$t_R \;\propto\; \Big(\underbrace{\textstyle\frac{1}{|R|}\sum_{j\in R} \Pi_S\,p_j}_{\text{region mean, projected onto }S}\Big)^{1/\tau},\ \ \text{stop-grad};\qquad
\mathcal{L}_{\text{hom}} = \frac{1}{HW}\sum_j \mathrm{CE}\big(p_j,\,t_{R(j)}\big)$$

**Minimiser.** \(p_j = t_R\) for all \(j \in R\): the posterior is constant on
each region *and* sharper than its own average.

**Proposition (no uniform degeneracy).** Consider the coupled iteration
\(p_j \leftarrow t_R(p)\). Its fixed points are the distributions \(q\) with
\(q \propto q^{1/\tau}\), i.e. \(q\) uniform on its support. Linearising at the
fully uniform point \(q = \frac{1}{C}(1+\delta)\), \(\sum\delta = 0\):
\(t \propto (1+\delta)^{1/\tau} \approx \frac{1}{C}(1 + \delta/\tau)\), so the map
is \(\delta \mapsto \delta/\tau\) with multiplier \(1/\tau > 1\) for \(\tau<1\).
**The uniform distribution is a strict repeller; the one-hot distributions are
the attractors.** ∎

This is the exact reason the term is written as sharpened distillation rather
than as within-region divergence minimisation. For the pure information-theoretic
form
\(D_R = H\!\left(\bar p_R\right) - \overline{H(p_j)} \ge 0\), **every** constant
assignment \(p_j \equiv q\) is a global minimum with value 0 — a whole manifold
of degenerate optima including the uniform one, and gradient descent has no
reason to prefer the corner. \(D_R\) is retained only as an ablation row
(`region_js_divergence`, `--ablation js-homogeneity`).

Two further properties, both load-bearing:

- the target is a **detached** function of the region mean, so this term supplies
  *shape* information only — it can move probability mass around inside a region
  but never decides which class the region is. That decision stays with §4.1–4.5.
- the projection \(\Pi_S\) means the attractors are restricted to classes in
  \(S\): homogeneity can never converge onto an absent class.

→ **failure mode 1** (salt-and-pepper): an isolated wrong pixel is, by
definition, a pixel that disagrees with its own region.

### 4.7 `L_potts`, `L_bnd` — image evidence, one-sided

$$\mathcal{L}_{\text{potts}} = \sum_j \sum_{k \in N(j)} w_{jk}\Big(1 - \sum_c p_c(j)p_c(k)\Big),\qquad
w_{jk} = \exp\!\Big(-\tfrac{\lVert I_j - I_k\rVert^2}{2\sigma^2}\Big)$$

\(1 - \sum_c p_c(j)p_c(k)\) is the probability that two independent draws from the
two posteriors disagree, so this is the standard differentiable Potts / dense-CRF
pairwise relaxation. **Minimiser:** label changes are confined to places where
\(w\) is small, i.e. where the image actually changes colour.

$$\mathcal{L}_{\text{bnd}} = \frac{1}{HW}\sum_j g_j\,\mathbb{1}[j \notin \mathcal{A}],
\qquad g_j = \max_{k\in N(j)}\Big(1-\sum_c p_c(j)p_c(k)\Big)$$

with the **candidate boundary set** \(\mathcal{A}\) = (per-image 80th-percentile
Sobel edges ∪ frozen-partition boundaries) ∖ (predicted shadow rims), dilated by
2 px.

**One-sided on purpose.** It penalises a contour *outside* \(\mathcal{A}\) and
never *requires* one inside it. A two-sided version would demand a label change
at every image edge, which is wrong — a ploughed field is full of high-gradient
texture edges with a single label.

The shadow-rim subtraction is the coupling that matters: an illumination-only
edge is **removed from the set of places a semantic contour is allowed**, so a
shadow rim no longer licenses a label change.
→ **failure mode 4** (shadow mislabelling), and the boundary metrics.

### 4.8 `L_sh`, `L_shead` — the dichromatic shadow model

For a Lambertian surface under sun plus sky,
\(I_c = \rho_c\,(V L^{\text{dir}}_c \cos\theta + L^{\text{amb}}_c)\) with
\(V \in \{0,1\}\) the sun's visibility. In the umbra,

$$\frac{I'_c}{I_c} \;=\; \frac{L^{\text{amb}}_c}{L^{\text{dir}}_c\cos\theta + L^{\text{amb}}_c} \;=:\; \alpha_c,$$

a per-channel multiplicative attenuation **independent of the albedo \(\rho\)**.
Skylight is blue-rich, so \(\alpha_B > \alpha_G > \alpha_R\): shadows are darker
*and* bluer, which is why intensity normalisation alone cannot fix them.

Two consequences:

**(a) Shadows can be synthesised exactly, with no annotation.**
\(I \mapsto I\,(1 - m\,(1-\alpha))\) with \(\alpha\) sampled to respect the blue
bias and \(m\) the max of a low-frequency blob field and a directional bar
(cast shadows of buildings and aircraft are directional), Gaussian-blurred into a
penumbra. The mask \(m\) is then an **exact** label for the shadow head and
\((I, I')\) an **exact** positive pair for equivariance.

**(b) Any feature invariant to a per-channel gain is invariant to shadow.**
Write \(l_c = \log I_c\); a shadow acts on log-space by *translation*. Let \(W\)
be a window on which \(\alpha\) is constant, \(\mu_W\) the local mean, and
\(c_1 = l_R-l_G\), \(c_2 = l_B-l_G\), \(l = \sum_c w_c l_c\) with
\(w=(.299,.587,.114)\).

> **Theorem.** \(\phi(I) = \big(c_1-\mu_W c_1,\ c_2-\mu_W c_2,\ l-\mu_W l,\
> \sigma_W l,\ \lVert\nabla l\rVert\big)\) satisfies \(\phi(\alpha\odot I) = \phi(I)\)
> **exactly**, for any \(\alpha>0\) constant on \(W\).
>
> *Proof.* \(c_1' = c_1 + \log(\alpha_R/\alpha_G)\), a constant on \(W\), so
> \(\mu_W\) shifts by the same constant and the difference is unchanged; likewise
> \(c_2\). \(l' = l + \sum_c w_c\log\alpha_c = l + k\) with \(k\) constant on
> \(W\), so \(l'-\mu_W l' = l-\mu_W l\); a standard deviation is shift-invariant;
> a spatial derivative of a constant is zero. ∎
>
> **Completeness.** The gain group is 3-dimensional and acts on
> \((l_R,l_G,l_B)\) by translation, so local invariants are at most
> 3-dimensional. \((l_R,l_G,l_B)\mapsto(c_1,c_2,l)\) is linear with determinant
> \(-(w_R+w_G+w_B) = -1\), hence invertible, and mean-removal commutes with it.
> The first three channels of \(\phi\) are therefore a **complete (maximal)
> invariant**: nothing is discarded except the three local illumination
> coordinates themselves. ∎

The geometric-mean luminance \(\sum_c w_c \log I_c\) is essential and is *not*
\(\log \sum_c w_c I_c\). The arithmetic version picks up
\(\log(\sum_c w_c \alpha_c I_c)\), which does not separate into image-plus-constant
unless \(\alpha\) is achromatic — and in a real shadow \(\alpha\) never is. (This
was a bug in the first draft of this file; `tools/verify_invariance.py` is the
test that catches it, because the *achromatic* case passes either way.)

**The honest caveat, and why a learned term is still needed.** \(\alpha\) is only
constant in the umbra *interior*. Across the penumbra it varies, so the invariance
degrades within about half a window of the rim — and the rim is exactly where
§4.7 is deciding whether a contour is allowed. That residual is covered by

$$\mathcal{L}_{\text{sh}} = \frac{\sum_j w_j\,\mathrm{CE}(p^{\text{shadow}}_j,\ \mathrm{sg}[p^{\text{clean}}_j])}{\sum_j w_j},\qquad w_j = \mathbb{1}[m_j>0.25]\cdot\max_c p^{\text{clean}}_c(j)$$

**Minimiser.** The classifier's decision is a function of *material*, not of
illumination — precisely the property failure mode 4 says is missing. The clean
view is the teacher and never the reverse, since it is the view whose statistics
the rest of the loss is fitted on; the confidence weight stops the term from
copying early-training noise into the shadow branch.

\(\mathcal{L}_{\text{shead}}\) is an asymmetrically weighted BCE. The asymmetry is
what the supervision supports, not a tuning choice: a mask=1 pixel is *certainly*
shadowed (we darkened it), a mask=0 pixel is only *probably* unshadowed, and the
deployed head must be free to fire on real shadows. Negatives therefore carry
weight 0.3. The clean view is supervised toward 0 **only inside the synthesised
blob**, because it is the *contrast* between the pair — not darkness in the
absolute — that makes the head discriminative instead of collapsing onto
"dark implies shadow".

### 4.9 `L_self` — model-derived, last, and filtered

Pipeline, in this order, because each step removes errors the next would amplify:

1. project the teacher posterior onto \(S\) (absent classes get zero mass);
2. average it over each frozen region;
3. drop regions that are tiny, that conflict, or **that a click already
   supervises** — a human label always beats a teacher label;
4. keep what clears an adaptive threshold \(\tau = \mu - \kappa\sigma\) over the
   EMA distribution of region confidences, plus a top-2 margin;
5. emit **hard** labels.

**Proposition (bounded target family).** The set of targets \(\mathcal{L}_{\text{self}}\)
can ever present is contained in

$$\mathcal{T} = \{\text{maps constant on each } R \in \mathcal{R},\ \text{valued in } S,\ \text{agreeing with propagation}\},$$

a **finite set that does not grow with the number of training steps**, because
\(\mathcal{R}\) is computed from the *pretrained* SAM before step 1 and never
changes. E3's target family was the whole simplex at every pixel and moved with
the network. This is the structural reason the degradation should stop — not a
hope that a better threshold will help.

The four filters map one-to-one onto failure modes: (1) → mode 2 (cannot
hallucinate an absent class); (2) → mode 1 (cannot produce isolated pixels, and
its geometry cannot drift); (3) → mode 5 (cannot overwrite a human label);
(4)+(5) → mode 3 (a *fixed fraction* of regions is rejected at every point in
training, so the accept rate cannot silently drift to 100%).

**Why hard labels.** Cross-entropy \(\mathrm{CE}(p,t) = -\sum_c t_c \log p_c\) is
minimised over the simplex at \(p=t\) (Gibbs). So a target with
\(\max_c t_c = 0.74\) — E3's, capped by a temperature-free prototype vote —
**caps the student at 0.74 as well**. A region argmax has no such ceiling. The
soft variant is kept as the `--ablation soft-self` row so this is measured, not
asserted.

### 4.10 `L_anc`, `L_rep` — prototype geometry

Prototypes **are** the classifier weights (§5), so moving them moves the decision
boundary; in E3 the analogous term was both inert and, when it fired, able only
to nudge a 30% side vote.

\(\mathcal{L}_{\text{anc}} = 1 - \cos(\mu_{c,k},\ \bar{f}^{\text{EMA}}_{c,k})\)
pulls prototypes toward an EMA of **annotated-pixel** features. Only human-clicked
pixels ever enter; no dense mask and no pseudo-label does, so the anchor is a
human-derived quantity that cannot drift with the network. **This is the paper's
prototype refinement, made consequential.**

\(\mathcal{L}_{\text{rep}}\) penalises within-class prototype similarity.
Without it, gradient descent may collapse all \(K\) onto one point, silently
reducing the mixture back to the unimodal model that mode 6 is caused by.

---

## 5. Architecture

Frozen SAM ViT-B + LoRA (r=8, α=16) → an illumination-invariant full-resolution
stem → a progressive decoder → a multi-prototype cosine classifier + a shadow
head.

**Progressive upsampling with full-resolution skips (64 → 128 → 256).** E3 ran
three convolutions on SAM's 64×64 grid and then bilinearly upsampled 4×, so every
predicted boundary was geometrically a smooth interpolation of a 64×64 decision:
the narrowest representable transition is ≈4 px, and DLRSD cars and dock edges are
a handful of pixels across. **No loss term can sharpen a boundary the
architecture cannot represent.**

**The stem carries what the 64×64 grid destroyed** — and carries it in the
invariant form of §4.8, so the high-frequency skip that sharpens boundaries does
not simultaneously re-import shadow edges as class evidence, which is exactly what
a raw-RGB skip would do. Both halves are separable in the ablation
(`--ablation no-invariant-stem` runs the same stem on raw RGB).

**Dilated context at 64×64** (dilations 2 then 4, residual). What separates
spectrally similar classes is context, not colour: a green patch inside a runway
is grass, the same green in a block of fields is field. Cheap at 64×64,
unaffordable at 256×256.

**Multi-prototype cosine classifier.**

> **Proposition (why \(K>1\) is necessary, not merely helpful).** A single-prototype
> cosine classifier predicts \(\arg\max_c \langle f, \mu_c\rangle\); its decision
> regions are convex cones. Suppose class \(a\) is bimodal with unit modes
> \(m_1, m_2\) and class \(b\) occupies the direction
> \(\widehat{m_1+m_2}\). Then the best single \(\mu_a\) is \(\widehat{m_1+m_2}\),
> which coincides with \(\mu_b\): **no choice of \(\mu_a\) separates them.** With
> \(K \ge 2\) and max-aggregation the region for \(a\) is a *union* of cones,
> which contains \(m_1\) and \(m_2\) while excluding the midpoint. ∎

DLRSD exhibits exactly this configuration — `field` covers ploughed earth and
green crop, with `grass` between them — and `field` IoU was 0.0888.

Aggregation over \(K\) is \(\mathrm{LSE}_\theta(z) = \theta\log\sum_k e^{z_k/\theta}\),
not a hard max, because \(\max \le \mathrm{LSE}_\theta \le \max + \theta\log K\):
at \(\theta=0.2, K=4\) it approximates the max within 0.28 while giving every
prototype non-zero gradient weight \(\mathrm{softmax}(z/\theta)_k > 0\). A hard
max sends gradient to one prototype only, so unlucky prototypes never update and
die.

The logit scale is learnable in log space and clamped to [4, 40] — a cosine in
[−1, 1] cannot produce a confident 17-way softmax on its own, which was precisely
E3's prototype bug.

Prototypes are seeded at the **end of epoch 0** by FINCH (parameter-free: nearest-
neighbour graph + connected components; no \(K\), no threshold, no iterations)
over collected annotated-pixel features. Clustering a randomly initialised
embedding at step 0 would partition noise; by the end of epoch 0 the margin point
loss and the inventory have shaped the embedding, and *only human-clicked pixels
are ever collected*.

**Dropped from E3:** `sam_class_logits`, which called SAM's mask decoder once per
class — 17 times per step — to build a mask vote that entered the teacher with
weight 0.25. It dominated the step cost (~1.24 s/step, ~835 s/epoch) and the
geometry it produced is the geometry the cached partition supplies for free.
**And the SAM normalisation bug is fixed** (`sam_normalize=True`), with the flag
kept so the old behaviour is reproducible as a comparison row.

---

## 6. Two constants are measured, not tuned

| constant | what it is | measured by |
|---|---|---|
| \(\eta\) (`inventory_leak`) | \(P(\text{a pixel's class has no click in its image})\) | `tools/validate_inventory.py`, the *PIXEL RISK* line |
| \(\epsilon\) (`prop_eps`) | \(P(\text{a propagated label is wrong})\) | `tools/validate_regions.py`, \(1 -\) propagation purity |

Neither is a knob. Each is an estimate of an error rate **in the supervision
itself**, and §4.3 and §4.2 are correctly specified only when they match it.
Erring high is safe (a weaker constraint); erring low teaches the network
something false.

`tools/validate_regions.py` additionally reports the two numbers that decide
whether the frozen partition is worth building at all:

- **region homogeneity** — area-weighted majority-class purity per region. This is
  the *ceiling* on any region-constant labelling, so it upper-bounds what §4.6
  and §4.9 can achieve.
- **propagation coverage and purity**, against a nearest-point Voronoi control at
  100% coverage. If propagation cannot beat Voronoi on purity, the partition adds
  nothing and the method should be abandoned rather than tuned.

`tools/verify_invariance.py` checks the §4.8 theorem to float32 round-off, and
reports the umbra-interior residual (must be ~0) against the penumbra-rim
residual (must be large — that is the honest limit, and the reason
\(\mathcal{L}_{\text{sh}}\) exists).

---

## 7. Curriculum

Terms switch on in order of how much they trust the model:

| epoch | added | why here |
|---|---|---|
| 0 | point, prop, abs, pres, area, potts, bnd, anchor, repel | annotations and image formation only — no model-derived signal exists yet |
| 1 | hom | the output now has a shape worth making consistent |
| 3 | shadow, shead | the clean prediction is now worth copying |
| 8 | self | the adaptive gate has seen enough regions to estimate μ and σ |

Each gated term ramps linearly over 3 epochs. Prototypes are FINCH-seeded at the
end of epoch 0, and the EMA shadow is re-synced to the seeded weights so the
teacher does not average across the discontinuity.

---

## 8. Ablation ladder

Each row moves exactly one mechanism, so the table reads as a set of independent
claims rather than a hyper-parameter search. The third column names what
`configs/prism.py::ablation` actually changes, because "one mechanism" is a claim
about the code and should be checkable against it rather than taken on trust.

| `--ablation` | claim under test | what it changes |
|---|---|---|
| `full` | — | — |
| `no-inventory` | the click *set* is a dense constraint, not 15 labels | `w_absent = w_present = w_area = 0` |
| `no-region` | a partition frozen before training beats a learned one | `w_hom = w_prop = w_self = 0` |
| `no-self` | self-training helps *once filtered* | `w_self = 0` |
| `soft-self` | hard region labels have no confidence ceiling (§4.9) | `soft_self = True` |
| `no-shadow` | the shadow **model**, not extra capacity | `w_shadow = w_shead = 0` |
| `no-invariant-stem` | *invariance*, not resolution (same stem, raw RGB) | `invariant_stem = False` |
| `no-boundary` | boundary and smoothness terms earn their weight | `w_bnd = w_potts = 0` |
| `single-prototype` | multi-modal classes need multi-prototypes (§5) | `prototypes_per_class = 1` |
| `no-margin` | the angular margin fixes spectral confusion (§4.1) | `margin = 0` |
| `js-homogeneity` | sharpened distillation beats JS minimisation (§4.6) | `js_homogeneity = True` |
| `const-k-present` | the MIL witness set must be sized by evidence (§4.4) | `pres_const_k = True` |
| `e3-normalisation` | the SAM input-range bug mattered | `sam_normalize = False` |

Four rows deserve a note, because each was **de-confounded** after a first pass
made it test two things at once:
- **`no-shadow` is losses-only.** It zeroes \(\mathcal{L}_{\text{sh}}\) and
  \(\mathcal{L}_{\text{shead}}\) and leaves the invariant stem and the shadow head
  in place, so the row isolates the *supervision* the physical model provides
  rather than also deleting parameters. The representation is the separate
  `no-invariant-stem` row. Running both together removes the shadow story
  entirely, if that combined row is ever wanted.
- **`js-homogeneity` isolates the sharpening alone.** `region_js_divergence` takes
  the same `present` projection and the same `conflict` exclusion the sharpened
  version takes (`core/regions.py:171`, called at `core/objective.py:311`), so the
  arms differ in the \(1/T\) exponent and nothing else. Without that, the row
  would silently also remove the inventory projection and the conflict mask, and
  a worse score would prove nothing about sharpening.
- **`no-region` and `no-boundary` are compound by necessity, and are labelled as
  such.** \(\mathcal{L}_{\text{prop}}\), \(\mathcal{L}_{\text{hom}}\) and
  \(\mathcal{L}_{\text{self}}\) are all *defined on* the partition; there is no
  version of the model that keeps propagation but drops the partition, so the row
  tests the partition as a whole. `no-self` is the single-term row inside it.
  Likewise \(\mathcal{L}_{\text{bnd}}\) is one-sided — it penalises boundaries
  outside the candidate set but never rewards boundaries inside it — so it is only
  meaningful next to a term that wants smoothness; `w_potts` goes with it.
- **`const-k-present` changes a term's *form*, not its weight.** \(\mathcal{L}_{\text{pres}}\)
  stays at full weight and keeps the same MIL structure; only the witness-set size
  changes, from \(k_c = \mathrm{clip}(|P_c|, 64, 327)\) back to \(k \equiv 327\) for
  every class. So the row isolates the sizing rule and nothing else — it is not a
  weaker inventory (that is `no-inventory`) and not a different pooling operator.
  Read it on the **rare-class IoU columns and boundary precision**, not on mIoU:
  the classes it should hurt are the ones contributing least to the mean.

Inference-time rows, both label-free and reported separately rather than folded
into the headline number:

| switch | what it tests |
|---|---|
| `--tta` | flip/mirror posterior averaging |
| `--region-vote` | pooling the posterior over each frozen region and taking the region argmax — the cleanest possible test of "the partition carries the object geometry" |

---

## 9. Metrics: the eval log is the evidence

Alongside mIoU / PA / mPrec / mRecall, `evaluate_prism.py` reports four numbers
that map onto the failure taxonomy directly, so no visual inspection is needed to
tell whether a mode was fixed:

| metric | definition | failure mode |
|---|---|---|
| **SPECKLE** | share of pixels whose label differs from its own 5×5 mode | 1 salt-and-pepper |
| **GHOST** | share of (image, class) pairs predicted over ≥0.5% of the image with zero GT pixels | 2 hallucination |
| **FLOOD** | share of images whose largest predicted class covers ≥1.5× the largest GT class | 3 flooding |
| **TRIMAP PA** (3 px, 5 px) | pixel accuracy restricted to a band around GT boundaries | boundary quality, which whole-image mIoU hides |

Modes 4 (shadow) and 5 (right shape, wrong label) are covered by
`tools/diagnose_failures.py`, which adds a shadow-proxy analysis and a full
confusion matrix with per-class precision/recall.

### 9.1 Reading the before/after table

The "before" column comes from `tools/diagnose_failures.py` run on saved E3
predictions (§10 step 2) and the "after" column from `evaluate_prism.py` (§10
step 4). Two different programs, so a number that appears in both must be
*defined* identically or the comparison is decoration. SPECKLE is identical by
construction — a ones-kernel `filter2D` equals `boxFilter(normalize=False)`, and
both break argmax ties toward the lower class index. GHOST and FLOOD were not:
`diagnose_failures` originally counted every pred-only class regardless of area
(a strictly larger, differently normalised quantity) and flagged flooding by an
absolute rule (pred > 70% while GT max < 50%). It now reports **both**
definitions, and the rows carrying the eval-matching one are marked
`<- quote this one` (the numerals below are **format only** — the E3 baseline has
not been diagnosed yet, and nothing in this file should be read as a measured
value until §10 step 2 has actually run):

```
[2] ghost-class hallucination
    GHOST (eval-matching): <num>/<den> = 0.xxxx   <- quote this one
[3] dominant-class flooding
    FLOOD (eval-matching): <num>/<n> = 0.xxxx     <- quote this one
```

Take the baseline GHOST and FLOOD from those two lines only. The secondary rows
are still useful as diagnostics — "ghost classes per image at any area" and "GT
classes missed entirely" say *how* the hallucination is distributed — but they do
not belong in a column headed by an `evaluate_prism` number.

**Baseline to beat (E3, epoch 30):** mIoU 0.5417, PA 0.7239, mPrec 0.6904,
mRecall 0.7015 — and, critically, **no degradation to epoch 50**. E3's epoch-50
mIoU of 0.5037 is the number the structural argument in §4.9 predicts should not
recur.

---

## 10. Reproduction

```bash
cd /home/cse-sdpl/Downloads/point_only_semseg

# 0. verify the invariance theorem numerically (no data needed, ~2 s)
python -m e3_only.tools.verify_invariance

# 1. measure the two loss constants
python -m e3_only.tools.validate_inventory              # -> eta  (PIXEL RISK)
python -m e3_only.tools.build_region_cache --split train
python -m e3_only.tools.build_region_cache --split val
python -m e3_only.tools.validate_regions                # -> eps  (1 - purity)

# 2. diagnose the baseline, for the before/after table
python -m e3_only.tools.diagnose_failures \
    e3_only/runs/eval_predictions/E3_epoch_0030 --limit 300

# 3. train (substitute the measured constants)
python -m e3_only.train_prism --ablation full --leak <eta> --prop-eps <eps>

# 4. evaluate, with the inference-time rows
python -m e3_only.evaluate_prism --checkpoint e3_only/runs/prism/PRISM_best.pt \
    --save-preds e3_only/runs/prism/preds --log e3_only/runs/prism/eval_full.log
python -m e3_only.evaluate_prism --checkpoint e3_only/runs/prism/PRISM_best.pt \
    --tta --region-vote --log e3_only/runs/prism/eval_tta_regionvote.log
```

Step 1 is not optional. Training refuses to start without the region cache, and
the two constants are the difference between a correctly specified likelihood and
a plausible-looking guess.







How to Run on Lightning AI

// bash
#### 1. Upload the entire PRISM/ folder to Lightning AI
#### 2. SSH into your Lightning Studio, then:
 
cd PRISM
pip install torch torchvision segment-anything numpy opencv-python pillow tqdm scikit-learn
 
#### 3. Build region caches (if not already in artifacts/)
python -m e3_only.tools.build_region_cache --split train
python -m e3_only.tools.build_region_cache --split val
 
#### 4. Train
python -m e3_only.train_prism --ablation full --save-dir runs/prism