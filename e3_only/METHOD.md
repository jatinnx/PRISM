# PRISM — Point-inventory, Region-consistency, Illumination-invariant Semantic Mapping

Weakly-supervised semantic segmentation of DLRSD (17 classes, 256×256, 630 train /
1319 val) from **point annotations only**. No dense mask is read at training time,
anywhere, by anything. The val masks are read in exactly one file
(`evaluate_prism.py`); the dense *train* masks are read only by the measurement
tools under `tools/` (`validate_inventory`, `validate_regions`, `measure_prop_trust`,
`oracle_partition`, `diagnose_failures`), and nothing any of them computes feeds
back into training.

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
   target with max 0.74 is minimised at *p = 0.74* (Gibbs' inequality, §4.10), so
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
+\;& w_{\text{phead}}\mathcal{L}_{\text{phead}}
&&\text{(the point \emph{inventory}, as a prediction target)}\\
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
| human annotation | point, prop, abs, pres, area, **phead**, anchor | 3.00 | **58%** |
| image formation | potts, **bnd**, shadow, shead | 1.10 | 21% |
| model output, geometry-constrained | hom, self | 1.00 | 19% |
| regulariser | repel | 0.05 | 1% |

Compare E3: human 33% (on 0.02% of pixels), model 67%.

Two of these weights moved after the first full run was measured, and both moves
are the subject of a measurement rather than a search:

* \(w_{	ext{bnd}}\) 0.15 → 0.30, with the candidate band narrowed from radius 2
  to radius 1. The reason is a gap between two numbers in the same eval log:
  overall PA 0.7345 against **trimap-3px PA 0.5953**. Fourteen points of the
  remaining error is inside three pixels of a boundary, which is not something
  the class-level terms can reach.
* \(w_{	ext{phead}}\) 0 → 0.30, a new term (§4.6). The reason is that **10.52%
  of val pixels are predicted as a class the image's own point inventory
  excludes** — field 83.4%, mobile home 39.0%, sand 32.9%. Every existing
  inventory term needs the points and therefore evaporates at test time; this one
  learns the inventory instead of consuming it.

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

**Minimiser.** \(p_y = 1-\epsilon_y+\epsilon_y/|S|\) — confident but not
saturated, which is the correct target for a label known to be right
\((1-\epsilon_y)\) of the time. \(\epsilon\) is **measured**, not tuned (§6).
Ordinary label smoothing would spread the residual over all 17 classes including
absent ones, undoing §4.3; restricting it to \(S\) makes the two terms agree.

**\(\epsilon\) is per class, and the 17 numbers cost no dense labels.**
Propagation noise is not uniform: measured against the masks,
\(1-\text{purity}\) runs from 0.000 (field, sea) to **0.507 (dock)**. One scalar
asks the network to distrust its most reliable labels and to trust its least. But
reading those 17 numbers into the loss would put dense masks inside training, so
they are *estimated from the annotations alone*
(`tools/measure_prop_trust.py`). A region that straddles a boundary tends to
contain clicks of more than one class, so per class \(c\) the conflict frequency

$$q_c^{\text{obs}} = \frac{\#\{\text{regions with a }c\text{ click and a foreign click}\}}{\#\{\text{regions with a }c\text{ click}\}}$$

is an observable proxy for its straddle rate. It is a **biased** proxy, and the
bias has a closed form: a straddle is only *visible* when a foreign click happens
to land in the region, whose probability grows with region area, so thin-region
classes look artificially safe. Modelling the foreign clicks as Poisson over the
region gives the detection probability
\(d_r = 1-\exp(-\tfrac{|r|}{HW}F_c)\) with \(F_c\) the foreign-click count, and
dividing it out, \(q_c = q_c^{\text{obs}}/\max(d_c, 0.05)\), leaves a debiased
risk. The already-measured scalar then sets the scale, so the coverage-weighted
mean is unchanged and only its *distribution* across classes is new:

$$\epsilon_c = \text{clip}\Big(\epsilon \cdot \frac{q_c}{\sum_{c'} w_{c'} q_{c'}},\ 0.01,\ 0.50\Big),\qquad w_c = \text{propagated-pixel share of }c$$

**Validation.** Rank agreement with the dense-mask truth
\(1-\text{purity}_c\) is **Spearman \(\rho = +0.801\)** over the 17 classes
(\(+0.500\) without the Poisson debiasing — the correction is doing real work).
It independently picks out dock as the single least reliable class
(\(q_c = 1.000\), and dock is the argmax of the mask-measured error too) with
ship 0.997 and chaparral 0.981 next, and correctly leaves field at the floor
(\(q_c=0\), mask purity 1.000). The coverage-weighted mean comes to 0.0411
against the 0.040 it was calibrated to. Since dock↔ship is the confusion pair
worth the most recoverable mIoU on this dataset, that is the pair the estimator
finds without being told about it.

The 17 values are written to `artifacts/prop_trust.json` and *loaded*; if the file
is missing the run falls back to the scalar and says so in its log, rather than
training on a vector nobody measured.

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

### 4.6 `L_phead` — the inventory as a prediction target

Every term in §4.3–4.5 *consumes* the inventory \(S(I)\). None of them can run at
test time, because a test image has no clicks. So nothing at all constrains the
class set at inference, and the measurement says the model takes the opening:
**10.52% of val pixels carry a class the image's own inventory excludes** — field
83.4% of its predicted pixels, mobile home 39.0%, sand 32.9%. The whole-image
consequence is `GHOST 0.2263`: 1320 of 5834 image×class slots are predicted at
≥0.5% area with *zero* ground-truth pixels.

This term closes the gap by learning \(S(I)\) instead of consuming it. A separate
1×1 stack on the decoder trunk produces a per-class map, pooled over space to one
logit per class, trained as 17 independent binary problems:

$$\mathcal{L}_{\text{phead}} = \frac{1}{C}\sum_c \text{BCE}\big(z_c,\ \mathbb{1}[c \in S(I)]\big),\qquad z_c = T\log\frac{1}{HW}\sum_j e^{u_c(j)/T}$$

**The target is not a pseudo-label.** `tools/validate_inventory.py` reports that
\(S(I)\) equals the image's true class set in **630 of 630** training images, so
this is exact supervision at one bit per class per image, already implied by the
clicks. It is the cheapest dense-free signal in the method and it was going unused.

**Log-sum-exp, not average, pooling.** \(T\log\text{mean}\exp(\cdot/T)\)
interpolates between the spatial mean (\(T\) large) and the spatial max
(\(T\to0\)); \(T = 0.5\). The mean is the wrong pool: a class covering 40 of
65 536 pixels contributes 0.06% of it, so a mean-pooled head learns to predict
only large classes — and the hallucinated classes are precisely the small ones.
The max is the right limit but back-propagates through one pixel.

**Class balance.** Mean \(|S(I)| = 3.3238\) over the 630 train images (min 1,
max 8), a positive rate of 0.1955, so unweighted BCE is 82%-accurate by answering
"absent" to everything. The positive class is weighted \((C-\bar m)/\bar m =
4.1146\) — measured, not chosen.

**Minimiser.** \(\sigma(z_c) = 1\) for every \(c \in S(I)\) and \(0\) for every
\(c \notin S(I)\).

**Why a separate branch and not a pooled read of the segmentation logits.** A
pooled read is a *summary* of the dense prediction, so it agrees with it by
construction and can never contradict it. The point of this head is to be a second
opinion, which is what makes it usable as a gate.

**The inference-time gate.** At test time the head's answer is applied as a prior
in log-space, i.e. multiplicatively on the posterior:

$$\text{logit}_c \leftarrow \text{logit}_c + \lambda \log\big(\sigma(z_c)(1-\delta) + \delta\big),\qquad \delta = 0.05$$

Deliberately **not** a hard mask. The head is a prediction, not an annotation, and
\(-\infty\) on a mistake deletes a class from an image that contains it — trading a
hallucination for a guaranteed miss. With \(\delta = 0.05\) the worst a wrong
"absent" can do is subtract \(\log 0.05 \approx -3.0\), which a confident dense
prediction still overcomes. \(\lambda = 0\) is the identity, so the gate is one
number away from off and the ablation row runs the same code path
(`--presence-gate`, default off; the headline training-loop eval is un-gated).

### 4.7 `L_hom` — region homogeneity by sharpened self-distillation

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

### 4.8 `L_potts`, `L_bnd` — image evidence, one-sided

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

### 4.9 `L_sh`, `L_shead` — the dichromatic shadow model

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
§4.8 is deciding whether a contour is allowed. That residual is covered by

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

**What the pair of terms costs, and the failure it caused.** Both terms need a
*second* forward pass, on the shadowed view, with gradients. The encoder runs at
1024² (SAM's trained input size, §5), so one grad-carrying pass holds 4096 tokens
through 12 blocks — about 11.8 GiB of activations on the 15.57 GiB card these
numbers come from. Two of them do not fit: with `w_shadow > 0` training died at the
first shadowed step with `torch.OutOfMemoryError` inside `attn.softmax`, 360 MiB
free. **That, and not a design preference, is why every run recorded in this file
before 2026-09-04 used `--ablation no-shadow-improved`, i.e. with these two terms
multiplied by zero.** Any earlier reading of the ablation table that treated the
absence of a shadow-on row as a result was reading a hardware limit.

The fix is `model/net.py::PrismNet.checkpointed()`, a context manager that routes
`encode()` through a re-implementation of `ImageEncoderViT.forward` with each of
the 12 blocks wrapped in `torch.utils.checkpoint` (`use_reentrant=False`), applied
to the shadowed pass **only**. It is a re-implementation rather than a module
wrapper specifically so that **no parameter is renamed**: a checkpoint written with
gradient checkpointing on loads into a model with it off, and vice versa.

Measured cost, on the live 40-epoch run, from the four epoch timings either side
of the switch-on: **363 / 355 / 360 s** per 630-step epoch for epochs 0–2 (0.57 s/step,
no shadowed pass) and **813 s** for epoch 3, the first epoch with
\(e_{\text{shadow}}=3\) active (1.29 s/step). The shadow row therefore costs
**2.26× the wall-clock** of its control -- 40 epochs is ≈8.7 h against ≈4.0 h. That
is the price of the row, and it is what the `no-shadow` ablation is now genuinely
measuring against instead of against nothing.

### 4.10 `L_self` — model-derived, last, and filtered

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

### 4.11 `L_anc`, `L_rep` — prototype geometry

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
stem → a progressive decoder → a multi-prototype cosine classifier, plus two
auxiliary heads on the same trunk: a per-pixel shadow head (§4.9) and an
image-level presence head (§4.6). 96.69M parameters total, **2.9582M trainable**
— LoRA 1.1796M, stem 0.0209M, decoder-and-heads 1.7577M — so the ViT is frozen and
the method's claims are about the losses and the 3M, not about capacity.

r=8/α=16 is measured, not inherited: r=12/α=24 was run (`prism-v6-lora12`) and
tracked *behind* at matched epochs, so doubling the adapter rank is not where the
remaining error lives.

**Progressive upsampling with full-resolution skips (64 → 128 → 256).** E3 ran
three convolutions on SAM's 64×64 grid and then bilinearly upsampled 4×, so every
predicted boundary was geometrically a smooth interpolation of a 64×64 decision:
the narrowest representable transition is ≈4 px, and DLRSD cars and dock edges are
a handful of pixels across. **No loss term can sharpen a boundary the
architecture cannot represent.**

That argument is about what the decoder can *represent*, and it stands. It is
**not** a claim that representation is what limits the current number: §6.1
measures the region-constant ceiling at 0.9438 mIoU against a delivered 0.5477, and
the largest confusions on the intact model are `bare soil`→`grass` (2.08M px),
`bare soil`→`pavement` (1.23M), `grass`→`trees` (1.18M) — spectrally adjacent pairs
at *region* scale, not thin-structure failures. Resolution buys the boundary; it
does not buy the class name.

**The stem carries what the 64×64 grid destroyed** — and carries it in the
invariant form of §4.9, so the high-frequency skip that sharpens boundaries does
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

**The two auxiliary heads are 1×1 stacks on the shared decoder trunk**, not
separate networks: the shadow head predicts a per-pixel shadow probability at full
resolution (used both as a loss target and to *suppress* shadow edges from the
§4.8 candidate boundary set), and the presence head predicts the 17-way class
inventory for the whole image, LSE-pooled over space. Together they are **5,330 of
the 2,958,227 trainable parameters** (presence 5,265; shadow 65 — a single 1×1
convolution), and they are the only two outputs that survive to inference as
something other than the segmentation itself. That is what makes `--presence-gate`
possible at test time, where no clicks exist: 0.18% of the trainable parameters
carry the inventory constraint past the end of training.

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

## 6. The loss constants are measured, not tuned

| constant | what it is | measured by | measured value |
|---|---|---|---|
| \(\eta\) (`inventory_leak`) | \(P(\text{a pixel's class has no click in its image})\) | `tools/validate_inventory.py`, the *PIXEL RISK* line | **0.0000** |
| \(\epsilon\) (`prop_eps`) | \(P(\text{a propagated label is wrong})\) | `tools/validate_regions.py`, \(1 -\) propagation purity | **0.040** |
| \(\bar m\) (`pres_head_pos_weight`) | mean \(|S(I)|\), for the §4.6 BCE balance | `tools/validate_inventory.py` | **3.3238** \(\Rightarrow\) 4.1146 |

Both of the first two carried conservative placeholders (0.05 and 0.10) through
the v2–v5 runs, and both placeholders were wrong in the direction that costs
accuracy. \(\eta = 0.05\) reserved 5% of the probability mass of every pixel for
classes the image provably does not contain: PIXEL RISK is **0.0000**, because
630/630 images have a click for every class they contain, so \(\mathcal{L}_
{\text{abs}}\) is an *exact* constraint and the placeholder was giving away a hard
one. \(\epsilon = 0.10\) told the network to distrust 10% of its propagated
labels when only 4.0% of them are wrong.

\(\epsilon\) is additionally **spent unevenly across the classes** (§4.2). The
*scale* is the measured 0.040 above; the *distribution* over the 17 classes is
estimated label-free from click-conflict frequency by
`tools/measure_prop_trust.py`, and only its rank ordering is validated against the
dense masks (Spearman \(\rho = +0.801\)). No dense mask enters the training path:
the vector is read from `artifacts/prop_trust.json`, and a missing or malformed
file falls back to the scalar with a logged warning rather than silently
substituting a guess.

None of the three is a knob. \(\eta\) and \(\epsilon\) are estimates of an error
rate **in the supervision itself**, and §4.3 and §4.2 are correctly specified only
when they match it; \(\bar m\) is a property of the annotation budget that fixes
the §4.6 class balance. Erring high on \(\eta\) or \(\epsilon\) is safe (a weaker
constraint); erring low teaches the network something false. Changing any of them
is a claim about the *annotations*, and the claim has to be re-measured with the
tool named beside it — which is why they are set from the tools' output rather than
swept.

`tools/validate_regions.py` additionally reports the two numbers that decide
whether the frozen partition is worth building at all:

- **region homogeneity** — area-weighted majority-class purity per region. This is
  the *ceiling* on any region-constant labelling, so it upper-bounds what §4.7
  and §4.10 can achieve.
- **propagation coverage and purity**, against a nearest-point Voronoi control at
  100% coverage. If propagation cannot beat Voronoi on purity, the partition adds
  nothing and the method should be abandoned rather than tuned.

### 6.1 That ceiling, measured in the units of the metric — and what it rules out

Purity is a per-pixel share; the reported metric is mIoU. `tools/oracle_partition.py`
closes that gap by giving each region its own **majority GT class** and scoring the
result with the same confusion matrix `evaluate_prism.py` uses. It is the exact
upper bound of \(\mathcal{L}_{\text{prop}}\), \(\mathcal{L}_{\text{hom}}\),
\(\mathcal{L}_{\text{self}}\) and the inference-time region vote *combined*:
none of them can do anything but push the prediction towards region-constant.
Over all 1319 val images (`artifacts/oracle_partition_val.txt`):

| oracle | scope | mIoU | PA | share of px forced region-constant |
|---|---|---|---|---|
| **all-regions** | every pixel takes its region's majority (ceiling on §4.7/§4.10) | **0.7412** | 0.8651 | 1.0000 |
| **SAM-only** | only `id < n_sam` and `size ≥ 24` (ceiling on the region vote) | **0.9438** | 0.9765 | 0.5756 |

SAM-only is *higher* by construction — it forces fewer pixels to a region-constant
answer — so the two bracket the machinery instead of competing. **The consequence
is the single most important measured fact in this file:** the partition supports
0.9438 and the trained model delivers 0.5477, so roughly 40 points of the error
are in **what the classifier calls a region, not in where the region is**. Any
further work on shapes, boundaries, or the vote is bounded by a ceiling the model
is nowhere near; §5's boundary-representation argument is why the ceiling *can* be
approached, not evidence that it is what limits the current number.

The per-class column says where region-constant labelling itself is genuinely
expensive: all-regions `dock` 0.4637, `chaparral` 0.4919, `trees` 0.5996 are the
only three under 0.62 (thin and interleaved structures a connected component
cannot resolve), while under SAM-only every one of the 17 sits at ≥ 0.8389.

`tools/verify_invariance.py` checks the §4.9 theorem to float32 round-off, and
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
| `soft-self` | hard region labels have no confidence ceiling (§4.10) | `soft_self = True` |
| `no-shadow` | the shadow **model**, not extra capacity | `w_shadow = w_shead = 0` |
| `no-invariant-stem` | *invariance*, not resolution (same stem, raw RGB) | `invariant_stem = False` |
| `no-boundary` | boundary and smoothness terms earn their weight | `w_bnd = w_potts = 0` |
| `single-prototype` | multi-modal classes need multi-prototypes (§5) | `prototypes_per_class = 1` |
| `no-margin` | the angular margin fixes spectral confusion (§4.1) | `margin = 0` |
| `js-homogeneity` | sharpened distillation beats JS minimisation (§4.7) | `js_homogeneity = True` |
| `const-k-present` | the MIL witness set must be sized by evidence (§4.4) | `pres_const_k = True` |
| `e3-normalisation` | the SAM input-range bug mattered | `sam_normalize = False` |
| `no-presence-head` | an independent image-level presence estimate earns its weight (§4.6) | `w_pres_head = 0` |
| `scalar-prop-eps` | propagation trust is class-**dependent** (§4.2) | `per_class_prop_eps = False` |
| `wide-boundary` | the *tighter* band is what tightens edges, not the weight | `w_bnd = 0.15`, `edge_radius = 2` |
| `improved` | — the shipping configuration (gate warm-up on) | `gate_warmup = 0` (= auto, 1 epoch) |
| `no-shadow-improved` | `improved` minus the shadow model — **the control for `improved`** | `w_shadow = w_shead = 0`, `gate_warmup = 0` |

**Status of the `no-shadow` claim.** Until 2026-09-04 the shadow terms had never
executed once: `w_shadow > 0` exhausted the card (§4.9), so every recorded run —
including every number in §9 — was `no-shadow-improved`. The row therefore had a
control and no treatment. With gradient checkpointing in place the pair
`improved` / `no-shadow-improved` is being run at 40 epochs each as the treatment
and its matched control; until both finish, **the shadow model is an untested
claim and must be written as one.**

One property of that pair has to be stated with it. Both rows run at
`--batch-size 1` (`run_queue_v8.sh:81`), because the shadowed pass needs the
headroom even with checkpointing. Treatment and control therefore match each
other exactly, and the *difference* between them is the claim the row makes --
but neither is directly comparable to the 0.5477 in §9, which was trained at
batch 2. A shadow-on number below 0.5477 does not by itself refute the shadow
losses, and one above it is not by itself worth two decimal places: the only
sound reading of the pair is treatment minus control, both at batch 1.

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

The three rows added with the boundary/presence revision are each a *negative
control on a measurement*, which is the only reason they exist — every one of them
reverts a constant to the value the measurement says is wrong, so the row tests
whether the measurement was worth taking:
- **`no-presence-head`** removes the only inventory term that survives to
  inference. Read it on `GHOST` and on the three classes that hallucinate most
  (field, mobile home, sand), not on mIoU: it is a precision mechanism.
- **`scalar-prop-eps`** keeps the measured scale \(\epsilon = 0.040\) and spends it
  *uniformly*. The full model spends it in proportion to a label-free per-class
  risk estimate whose ranking agrees with the dense-mask truth at
  \(\rho = +0.801\). The classes at the extremes of that ranking are where the row
  should show: dock and ship (estimated riskiest, \(q_c\) 1.000 and 0.997) against
  field (at the 0.01 floor, mask purity 1.000).
- **`wide-boundary`** is the de-confounder for the \(\mathcal{L}_{\text{bnd}}\)
  change, which moved *two* things at once: the weight 0.15 → 0.30 and the band
  radius 2 → 1. This row restores both, so the pair full / `wide-boundary`
  attributes the trimap gain to the band-tightening rather than to the weight.
  Radius 1 is the floor `structure.candidate_boundary` admits (the diagonal shifts
  need it); radius 2 licensed a 5px-wide band, wider than the 3px trimap the metric
  scores, which is a supervision signal that cannot see its own target.

Inference-time rows, both label-free and reported separately rather than folded
into the headline number:

| switch | what it tests |
|---|---|
| `--tta` | flip/mirror posterior averaging |
| `--region-vote` | pooling the posterior over each frozen region and taking the region argmax — the cleanest possible test of "the partition carries the object geometry" |
| `--presence-gate 1.0` | the §4.6 soft prior applied at inference; \(\lambda = 0\) (default) is bit-exact identity |

`--region-vote` pools over **SAM masks only** (`region_vote_sam_only`, default
`True`), and that restriction is measured rather than stylistic. In training, a
region whose clicks disagree is excluded from every region term; at test time there
are no clicks, so the exclusion cannot run — and the 449 conflicted filler regions
sit at homogeneity purity **0.686** over 14.5M pixels. Voting inside one of those
drags an 8000+px blob to a single wrong class. Restricted to SAM masks the vote is
a clean win (measured on the CPU smoke run: `SPECKLE 0.0320 → 0.0124`); unrestricted
it is a coin flip on the largest regions in the image. Regions below
`region_vote_min_size = 24` px keep their own per-pixel argmax.

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
`<- quote this one`. Step 2 has now run on `_archive/e3-baseline/eval_predictions/E3_epoch_0030`, so
these are measured, and `artifacts/diagnose_E3_ep30.txt` is the file they came
from:

```
[2] ghost-class hallucination
    GHOST (eval-matching): 1411/5984 = 0.2358   <- quote this one
    ghost classes per image (any area)        : 1.72
    share of all errors from ghost classes    : 0.3791
[3] dominant-class flooding
    FLOOD (eval-matching): 19/1319 = 0.0144    <- quote this one
```

The PRISM column of the same two lines, from `diagnose_v5corrected_0.5477.txt`,
is GHOST 1320/5834 = 0.2263 and FLOOD 14/1319 = 0.0106. Both improve, and both
improve by little: 0.95pp of ghosting and 0.38pp of flooding. A ghost class still
appears on 22.63% of (image, class) opportunities, and 39.64% of all misclassified
pixels still belong to a class the image does not contain — a *higher* share than
E3's 37.91%, because PRISM's total error is smaller while its ghost error is
almost the same size. §9.2 is where that observation is followed to its end.

Take the baseline GHOST and FLOOD from those two lines only. The secondary rows
are still useful as diagnostics — "ghost classes per image at any area" and "GT
classes missed entirely" say *how* the hallucination is distributed — but they do
not belong in a column headed by an `evaluate_prism` number.

**Baseline to beat (E3, epoch 30):** mIoU 0.5417, PA 0.7239, mPrec 0.6904,
mRecall 0.7015 — and, critically, **no degradation to epoch 50**. E3's epoch-50
mIoU of 0.5037 is the number the structural argument in §4.10 predicts should not
recur.

### 9.2 What replacing the objective actually bought (measured, and it is small)

Reported honestly, because a 14-term objective invites the question:

| comparison | value |
|---|---|
| PRISM (14 terms, 0.5477) − E3 (3 terms, 0.5417) | **+0.60 pp mIoU** |
| classes that **regressed** | **6 of 17** |
| the same delta with `chaparral` excluded | **−0.55 pp** (PRISM is *worse*) |
| Spearman ρ(E3 per-class IoU, PRISM per-class IoU) | **+0.809** |
| worst-5 classes | **the same five in both models** |

A rank correlation that high, with an identical worst-5, is the signature of two
models failing on the same pixels for the same reason — a reason that **survived
replacing the entire objective**.

And the confusion matrix of the intact 0.5477 model over all 1319 val images says
what that reason is. Reproduced by
`tools/diagnose_failures.py --confusion-csv`; the numbers below are read off
`artifacts/confusion_v5corrected_rownorm.csv` and
`artifacts/diagnose_v5corrected_0.5477.txt`, where each row is normalised by that
class's true pixel count. Four classes recall under 55%:

| class | recall | dominant destination (share of that class's true px) |
|---|---|---|
| `field` | **0.345** | `grass` 0.479, `pavement` 0.149 |
| `sand` | **0.438** | `bare soil` 0.179, `pavement` 0.156 |
| `bare soil` | **0.467** | `grass` 0.185, `pavement` 0.109 |
| `dock` | **0.543** | `ship` 0.350, `water` 0.040 |

Three more sit in a 0.67–0.70 band (`buildings` 0.668, `airplane` 0.694, `water`
0.699) and the remaining ten recall 0.77–0.93 — so this is not a uniform weakness,
it is a short list of specific naming failures. Two of them are *precision*
failures instead, which a recall column hides: `mobile home` recalls 0.810 at
precision 0.488, and `buildings`→`mobile home` alone is **34.2% of every pixel
predicted `mobile home`** (0.99M of 2.88M).

Largest confusions by absolute pixel mass: `bare soil`→`grass` 2.08M,
`bare soil`→`pavement` 1.23M, `grass`→`trees` 1.18M, `buildings`→`pavement` 1.04M,
`bare soil`→`trees` 1.02M, `buildings`→`mobile home` 0.99M. Every pair is
spectrally adjacent and, at the level of a single pixel, semantically arbitrary — a
green pixel genuinely *is* both grass and field, and which one it is depends on
context the 64×64 dilated stack has to supply. The same file puts **0.2201 of all
error** in GT regions ≥200px that were predicted >80% homogeneously **but with the
wrong class** — right shape, wrong name — against **0.0210** of all error that a
5×5 majority filter would fix. Read with §6.1 (a 0.9438 shape ceiling against
0.5477 delivered), the three point at one place: what the encoder-plus-classifier
can separate between spectrally adjacent classes.

**The same classes, confused into the same classes, in both models.** Recall and
the single largest off-diagonal destination per class, from
`artifacts/confusion_E3_ep30_rownorm.csv` and
`artifacts/confusion_v5corrected_rownorm.csv` — two independent runs of the same
tool on two saved prediction sets:

| class | E3 recall | PRISM recall | Δ | E3 → dominant | PRISM → dominant | |
|---|---|---|---|---|---|---|
| `airplane` | 0.803 | 0.694 | -0.109 | `pavement` 0.100 | `cars` 0.140 | — |
| `bare soil` | 0.418 | 0.467 | +0.048 | `grass` 0.199 | `grass` 0.185 | **same** |
| `buildings` | 0.686 | 0.668 | -0.018 | `pavement` 0.141 | `pavement` 0.109 | **same** |
| `cars` | 0.847 | 0.875 | +0.029 | `pavement` 0.114 | `pavement` 0.095 | **same** |
| `chaparral` | 0.595 | 0.832 | +0.237 | `bare soil` 0.323 | `bare soil` 0.103 | **same** |
| `court` | 0.912 | 0.905 | -0.007 | `buildings` 0.029 | `grass` 0.044 | — |
| `dock` | 0.660 | 0.543 | -0.117 | `ship` 0.222 | `ship` 0.349 | **same** |
| `field` | 0.338 | 0.345 | +0.007 | `grass` 0.450 | `grass` 0.479 | **same** |
| `grass` | 0.746 | 0.775 | +0.029 | `trees` 0.094 | `trees` 0.082 | **same** |
| `mobile home` | 0.747 | 0.810 | +0.063 | `buildings` 0.072 | `trees` 0.048 | — |
| `pavement` | 0.878 | 0.855 | -0.023 | `buildings` 0.033 | `grass` 0.040 | — |
| `sand` | 0.477 | 0.438 | -0.039 | `pavement` 0.201 | `bare soil` 0.179 | — |
| `sea` | 0.668 | 0.797 | +0.129 | `sand` 0.146 | `field` 0.092 | — |
| `ship` | 0.842 | 0.926 | +0.084 | `dock` 0.099 | `buildings` 0.027 | — |
| `tanks` | 0.771 | 0.847 | +0.077 | `buildings` 0.120 | `buildings` 0.042 | **same** |
| `trees` | 0.785 | 0.772 | -0.013 | `grass` 0.093 | `grass` 0.105 | **same** |
| `water` | 0.752 | 0.699 | -0.053 | `grass` 0.091 | `grass` 0.128 | **same** |

**10 of 17 classes send their largest error to the identical wrong class**, and
the four weakest classes (`field`, `bare soil`, `sand`, `dock`) are the four weakest
in both. `chaparral` is the one large gain (+0.237 recall, `bare soil` 0.323 → 0.103)
— which is precisely why removing `chaparral` from the average turns the +0.60 pp
into −0.55 pp: **one class carries the entire headline improvement.** Eight classes
lost recall, `dock` worst at −0.117 with `dock`→`ship` rising 0.222 → 0.349.

Those failure *rates* are also nearly identical where the mechanism is supposed to
have changed:

| diagnostic (same tool, same definitions) | E3 ep30 | PRISM 0.5477 |
|---|---|---|
| overall pixel error | 0.2761 | **0.2655** |
| GHOST (eval-matching) | 0.2358 | **0.2263** |
| FLOOD (eval-matching) | 0.0144 | **0.0106** |
| speckle (px ≠ own 5×5 majority) | **0.0101** | 0.0160 |
| error a 5×5 majority filter would fix | **0.0133** | 0.0210 |
| error inside shadow-like px / elsewhere | 0.3443 / 0.2622 = **1.313** | 0.3193 / 0.2546 = **1.254** |
| right shape, wrong label (share of all error) | 0.2082 | 0.2201 |
| within-group confusion, vegetation/soil | 0.3189 | 0.2866 |

PRISM is ahead on the two hallucination modes and behind on speckle — \(\mathcal{L}_{\text{potts}}\)
and \(\mathcal{L}_{\text{hom}}\) did not deliver the smoothness they were added for, and
that is a result to report rather than to bury. And the shadow ratio moved 1.313 →
1.254 **in runs where the shadow terms were switched off entirely** (§4.9), so that
1.254 is the number the first shadow-on run has to beat, not evidence the shadow
model works.
The terms in §4 are individually justified and each has a minimiser; that is a
different claim from "the objective is where the remaining error is", and this
table is the evidence against the second claim. It belongs in the paper, not in a
footnote.

---

## 10. Reproduction

All commands run from the directory that **contains** the `e3_only` package, so
that `python -m e3_only.…` resolves:

```bash
cd /home/cse-sdpl/Downloads/point_only_semseg/PRISM

# 0. verify the invariance theorem numerically (no data needed, ~2 s)
python -m e3_only.tools.verify_invariance

# 1. measure the loss constants
python -m e3_only.tools.validate_inventory              # -> eta = 0.0000, mean|S| = 3.3238
python -m e3_only.tools.build_region_cache --split train
python -m e3_only.tools.build_region_cache --split val
python -m e3_only.tools.validate_regions                # -> eps = 0.040 (1 - purity)

# 1b. distribute eps over the 17 classes, LABEL-FREE (clicks only)
#     writes artifacts/prop_trust.json, which train_prism reads; the validation
#     against the dense masks is printed but never fed back into the vector
python -m e3_only.tools.measure_prop_trust

# 1c. the region-constant CEILING in mIoU units (S6.1). Needs the val cache from
#     step 1. Writes artifacts/oracle_partition_val.txt, which is the upper bound
#     every region term and the region vote are measured against.
python -m e3_only.tools.oracle_partition \
    --log artifacts/oracle_partition_val.txt     # -> 0.7412 all-regions / 0.9438 SAM-only

# 2a. regenerate the two prediction directories step 2 reads. Both were deleted
#     in the 2026-09-04 cleanup (Stage C Tier 1) because they are 1.5 GB of PNGs
#     that these two commands reproduce exactly from checkpoints that were kept.
#     The diagnose outputs they produced are already in artifacts/ and were NOT
#     deleted, so step 2 only has to be re-run if you change diagnose_failures.py.
python -m e3_only.run_experiment --config e3_only/configs/e3_teacher_student.py \
    --evaluate --checkpoint e3_only/_archive/e3-baseline/checkpoints/E3_epoch_0030.pt \
    --save-preds e3_only/_archive/e3-baseline/eval_predictions/E3_epoch_0030
python -m e3_only.evaluate_prism --which teacher \
    --checkpoint e3_only/e3_only/runs/prism-v5-corrected/PRISM-no-shadow-improved_best.pt \
    --save-preds e3_only/e3_only/runs/prism-v5-corrected/PRISM_best_predictions

# 2. diagnose the baseline, for the before/after table. Both invocations are the
#    ones that produced artifacts/diagnose_E3_ep30.txt and
#    artifacts/diagnose_v5corrected_0.5477.txt -- no --limit, all 1319 images,
#    because the before/after table is not allowed to be a subsample.
python -m e3_only.tools.diagnose_failures \
    e3_only/_archive/e3-baseline/eval_predictions/E3_epoch_0030 \
    --confusion-csv artifacts/confusion_E3_ep30_rownorm.csv
python -m e3_only.tools.diagnose_failures \
    e3_only/e3_only/runs/prism-v5-corrected/PRISM_best_predictions \
    --confusion-csv artifacts/confusion_v5corrected_rownorm.csv

# 3. train. The measured constants are the defaults in configs/prism.py
#    (eta = 0.0, eps = 0.040); --leak / --prop-eps override them for the
#    sensitivity rows only.
python -m e3_only.train_prism --ablation full --save-dir runs/prism

# 4. evaluate. Four rows, all label-free, reported separately: the headline
#    number is the ungated per-pixel argmax, and each switch is one mechanism.
#    MIND THE PATH ASYMMETRY: --save-dir/--log/--save-preds go through
#    configs.prism.resolve() and are therefore PACKAGE-relative, while
#    --checkpoint is handed straight to torch.load and is CWD-relative.
R=runs/prism                       # resolve()d  -> PRISM/e3_only/runs/prism
CK=e3_only/$R/PRISM-full_best.pt   # cwd-relative -> same directory, spelled out
python -m e3_only.evaluate_prism --checkpoint $CK --save-preds $R/preds_plain \
    --log $R/eval_plain.log                                       # headline
python -m e3_only.evaluate_prism --checkpoint $CK --presence-gate 1.0 \
    --save-preds $R/preds_presence_gate --log $R/eval_presence_gate.log
python -m e3_only.evaluate_prism --checkpoint $CK --region-vote \
    --log $R/eval_region_vote.log
python -m e3_only.evaluate_prism --checkpoint $CK --presence-gate 1.0 \
    --region-vote --save-preds $R/preds_gate_and_vote \
    --log $R/eval_gate_and_vote.log
```

Step 1 is not optional. Training refuses to start without the region cache, and
the constants are the difference between a correctly specified likelihood and a
plausible-looking guess. Step 1b is optional in the weaker sense that its absence
is *detected*: `per_class_prop_eps = True` with an unreadable `prop_trust.json`
logs a warning and falls back to the measured scalar, rather than proceeding with
a fabricated vector.

Step 4's four rows are chained automatically off step 3 by `run_queue_v8.sh`
(`run_queue_v7.sh` before it), which waits for the GPU, trains, and — only on exit
code 0 — runs all four evals against the newest `*_best.pt` in the run directory.
That ordering matters for a claim: an eval row that silently scored a stale
checkpoint would be indistinguishable from a real result.

**Two integrity guards, both added because the failure they catch had already
happened.** `runs/prism-no-shadow/*.pt` was written by an earlier
`trainable_state()` that returned `state_dict()` filtered by the trainable-name
set — and silently dropped every trainable tensor `state_dict()` did not expose.
Those files hold **38 of 132 tensors, missing all 96
`sam.image_encoder.blocks.*.{attn.qkv,attn.proj,mlp.lin1,mlp.lin2}.{A,B}` LoRA
matrices**. The same epoch-20 weights scored **mIoU 0.5493 in-training** (live
model) and **0.1807 loaded back from disk**; predictions saved from those files are
the output of the 0.18 network and are stamped `DO_NOT_USE`. So:

* `train_prism.trainable_state()` diffs the trainable-name set against the dict it
  is about to return and **raises** naming the lost tensors. A run that cannot save
  itself now dies at the save, not at the paper.
* `evaluate_prism.evaluate()` intersects `load_state_dict`'s `missing` with
  `requires_grad` names and **raises** — the old behaviour was a printed warning.
  `decoder.presence.*` is the one whitelisted absence (it post-dates older
  checkpoints and is inert at `--presence-gate 0`). Every eval now prints
  `loaded N/132 trained tensors`, so the count is on the record beside the score.

A checkpoint predating these guards must be checked before it is believed:

```bash
python - <<'EOF'
import torch, sys
sd = torch.load(sys.argv[1] if len(sys.argv)>1 else "CKPT", map_location="cpu",
                weights_only=False)["teacher"]
lora = [k for k in sd if k.startswith("sam.image_encoder")]
print(len(sd), "tensors,", len(lora), "LoRA (expect 132 and 96 at r=8)")
EOF
```







How to Run on Lightning AI

// bash
#### 1. Upload the entire PRISM/ folder to Lightning AI
#### 2. SSH into your Lightning Studio, then:
 
cd PRISM          # the directory containing e3_only/, not e3_only/ itself
pip install torch torchvision segment-anything numpy opencv-python pillow tqdm scikit-learn
 
#### 3. Build region caches (if not already in artifacts/)
python -m e3_only.tools.build_region_cache --split train
python -m e3_only.tools.build_region_cache --split val
python -m e3_only.tools.measure_prop_trust
 
#### 4. Train
python -m e3_only.train_prism --ablation full --save-dir runs/prism