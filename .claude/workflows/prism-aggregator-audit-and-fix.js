export const meta = {
  name: 'prism-aggregator-audit-and-fix',
  description: 'Audit the PRISM prototype aggregator for defects, then design the fix for misclassification',
  phases: [
    { title: 'Audit', detail: '7 independent lenses over the aggregator and everything that consumes it' },
    { title: 'Verify', detail: 'code-reality + magnitude verifiers per finding, adversarially prompted' },
    { title: 'Design', detail: '5 independent fix proposals, each from a different angle' },
    { title: 'Judge', detail: '3 judges score every proposal against the measured error budget' },
    { title: 'Synthesize', detail: 'one ranked recommendation with a gating measurement per step' },
  ],
}

const R = '/home/cse-sdpl/Downloads/point_only_semseg/PRISM/e3_only'

const CTX = `
You are auditing PRISM, a point-supervised (0% dense masks) semantic segmentation
model for remote sensing (DLRSD, 17 classes). Repo root: ${R}

HARD RULES — violating these breaks a live experiment:
 - READ ONLY. Do not edit, create or delete any file. Do not run git commands that write.
 - Do NOT run training, evaluation, or ANY python that touches the GPU. A 40-epoch run
   (PID 77901, prism-v8-shadow-on) owns the card, and a CPU measurement pass
   (run_stage0_cpu.sh) owns 16 cores. The disk is at 96% — write nothing large.
 - Cheap read-only shell (cat/sed/grep/find/head/python3 -c for pure arithmetic on
   numbers you paste inline) is fine. Nothing that imports torch.

THE PROBLEM, as measured over all 1319 val images (artifacts/diagnose_v5corrected_0.5477.txt):
 - delivered mIoU 0.5477 / PA 0.7345 (checkpoint of record:
   ${R}/e3_only/runs/prism-v5-corrected/PRISM-no-shadow-improved_best.pt, epoch 35)
 - the frozen SAM region partition supports mIoU 0.9438 if each region takes its
   majority GT class (artifacts/oracle_partition_val.txt). So SHAPE is ~solved and
   ~39pp of headroom is pure class NAMING.
 - GHOST (class predicted in an image with zero GT pixels of it) = 0.3964 of all error.
   Image level: 2398 ghost (class,image) pairs, 1.82/image. 'field' is predicted in
   29 images and present in 1. tanks 34% / court 41% / sea 43% image precision.
 - correct-shape-wrong-label = 0.2201 of all error. Boundary+speckle = only 0.0210.

THE AGGREGATOR UNDER QUESTION — core/protobank.py, MultiPrototypeClassifier:
   def aggregate(self, cos):            # cos is (B,C,K,H,W)
       t = max(self.k_temperature, 1e-3)
       return t * torch.logsumexp(cos / t, dim=2)
   def forward(self, embed):
       cos = self.aggregate(self.cosines(embed))
       return self.scale * cos, cos     # scale = log_scale.exp().clamp(4,40)
K=4 prototypes/class, k_temperature=0.20, measured scale=8.82.

THE FRESH MEASUREMENT of that aggregator (artifacts/proto_geometry_v5corrected.txt,
finished 2026-09-04, full val set) — READ THIS FILE, it is the ground truth here:
 - ceiling of the excess term t*log(K) = 0.2773 cosine units = 2.445 logits at scale 8.82
 - mean realised excess 0.1984 = 0.715 of ceiling; K_eff = exp(excess/t) mean 2.70
 - max_pair (highest within-class prototype-pair cosine) is >= +0.96 for 15 of 17
   classes, several at +0.99 => near-duplicate prototypes almost everywhere
 - near_other (cosine to the nearest OTHER class's prototype) is +0.75..+0.84 for many
   classes (grass +0.8405, trees +0.8405, mobile home +0.8163, sea +0.7987)
 - Spearman rho of per-class ghost_rate against: realised excess +0.4436,
   excess@win +0.4044, intra_cos +0.1201, and the rarity-only null gt_img -0.2406

FILES THAT MATTER:
 core/protobank.py     the aggregator, finch_init, update_ema, anchor_loss, repulsion
 core/objective.py     how the 14 loss terms are composed, and which see cos vs logits
 core/inventory.py     absent_class_loss, present_coverage_loss, area, presence head/gate
 core/structure.py     region/boundary terms
 model/decoder_v2.py   what emits the embedding the classifier consumes, and at what res
 model/net.py          SAM wrapper, LoRA, shadow head
 configs/prism.py      every hyperparameter incl. class_weighting, rare_class_factor=4.0
 evaluate_prism.py     the eval path: _posterior, --region-vote, --presence-gate
 train_prism.py        the training loop
 METHOD.md             the method write-up with propositions
 v8-plan.md            the current plan; 'the six issues' and 'two signatures' sections
 artifacts/            every measurement quoted above
`

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'short kebab-case slug' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['blocking', 'major', 'moderate', 'minor', 'non-issue'] },
          mechanism: { type: 'string', description: 'the precise causal chain, in terms of the code' },
          evidence: { type: 'string', description: 'file:line plus the number on disk that supports it' },
          magnitude: { type: 'string', description: 'quantified effect, in cosine units / logits / pp of mIoU, or "unquantified" and say why' },
          fix_sketch: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['id', 'title', 'severity', 'mechanism', 'evidence', 'magnitude', 'fix_sketch', 'confidence'],
      },
    },
    aggregator_verdict: { type: 'string', description: 'from THIS lens only: is the aggregator good enough? one paragraph.' },
  },
  required: ['findings', 'aggregator_verdict'],
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    reasoning: { type: 'string' },
    corrected_magnitude: { type: 'string', description: 'if the claim is real but the number is wrong, the corrected one; else ""' },
  },
  required: ['refuted', 'reasoning', 'corrected_magnitude'],
}

const PROPOSAL = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    angle: { type: 'string' },
    mechanism: { type: 'string', description: 'what changes, mathematically, and why that removes error' },
    which_error_it_attacks: { type: 'string', description: 'name the measured slice and its share' },
    expected_gain_pp: { type: 'string', description: 'a range with the reasoning that bounds it; say "unbounded/unknown" rather than invent' },
    changes: { type: 'array', items: { type: 'string', description: 'file:symbol — what edit' } },
    training_cost: { type: 'string' },
    gating_measurement: { type: 'string', description: 'the cheap measurement that decides whether to do this at all' },
    risks: { type: 'array', items: { type: 'string' } },
    kills_or_subsumes: { type: 'string', description: 'existing terms/flags this makes redundant' },
  },
  required: ['name', 'angle', 'mechanism', 'which_error_it_attacks', 'expected_gain_pp', 'changes', 'training_cost', 'gating_measurement', 'risks', 'kills_or_subsumes'],
}

const SCORES = {
  type: 'object',
  properties: {
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          soundness: { type: 'number', description: '0-10, is the mechanism actually implied by the measurements' },
          expected_value: { type: 'number', description: '0-10, pp of mIoU per unit of effort' },
          risk: { type: 'number', description: '0-10, 10 = safest' },
          total: { type: 'number' },
          why: { type: 'string' },
          fatal_objection: { type: 'string', description: 'the strongest argument against it, or "" if none' },
        },
        required: ['name', 'soundness', 'expected_value', 'risk', 'total', 'why', 'fatal_objection'],
      },
    },
    ordering: { type: 'array', items: { type: 'string' }, description: 'proposal names, best first' },
  },
  required: ['scores', 'ordering'],
}

const LENSES = [
  {
    key: 'logsumexp-math',
    prompt: `LENS: the mathematics of aggregate() itself.
Derive exactly what t*logsumexp(cos/t, dim=K) computes. Decompose it into
max_k cos_k plus an excess term and state the exact bound on that excess. Then answer:
is the excess a legitimate part of a classifier score, or an uncontrolled per-class bias?
Compare its magnitude to (a) the inter-class prototype separation implied by the
near_other column, (b) the 0.20 angular point margin, (c) the per-class logit differences
that actually decide an argmax at scale 8.82. Check the gradient of aggregate w.r.t. each
prototype and what it does to a collapsed pair vs a separated pair. Check the t -> 0 and
t -> inf limits and whether t=0.20 is anywhere near either. Check for numerical issues.
Is the aggregator scale-consistent across classes with different K_eff?`,
  },
  {
    key: 'prototype-collapse',
    prompt: `LENS: why max_pair is +0.96..+0.99 for 15 of 17 classes, i.e. why the K=4
prototypes are near-duplicates. Read finch_init / finch_first_partition, update_ema
(note: hard nearest-slot assignment, and the "unused slots claim the point first" rule),
anchor_loss, and any within-class repulsion term (L_rep) — find where it is defined, what
its weight is in configs/prism.py, and whether it is actually called in core/objective.py.
Trace how a prototype slot can end up a duplicate and never recover. Does update_ema's
argmax assignment create a rich-get-richer dynamic? What happens to a class whose points
all land in one mode? Does ema_count ever reset? Quantify: with K_eff=2.70 of 4, what is
the multi-prototype bank actually buying, and what is it costing.`,
  },
  {
    key: 'inter-class-margin',
    prompt: `LENS: inter-class separation, which may be the real defect. near_other in
artifacts/proto_geometry_v5corrected.txt is the cosine to the nearest OTHER class's
prototype: grass +0.8405, trees +0.8405, mobile home +0.8163, chaparral +0.8163,
sea +0.7987, airplane +0.7987. Work out what a cosine gap of (1 - near_other) means in
logits at scale 8.82, and compare it to the aggregator's excess spread across classes
(column 'excess', 0.1541..0.2336). Then: is there ANY term in the objective that pushes
prototypes of DIFFERENT classes apart? Search core/objective.py, core/protobank.py and
configs/prism.py for it. If there is none, say so plainly and quantify what that costs.
Cross-check against artifacts/confusion_v5corrected_rownorm.csv: do the class pairs with
the highest near_other match the pairs that actually get confused?`,
  },
  {
    key: 'objective-consumers',
    prompt: `LENS: who consumes the aggregator's two outputs and does the excess leak into
the losses. forward() returns (scale*cos, cos). Trace EVERY consumer of each in
core/objective.py, core/inventory.py, core/structure.py, model/decoder_v2.py and
train_prism.py. Specifically: does the point/margin loss apply its angular margin to the
AGGREGATED cosine (which already contains the excess) or to the per-prototype cosine? If
the former, the margin is being applied to a quantity inflated by up to 0.2773 — work out
whether that makes the margin effectively class-dependent, and by how much. Same question
for absent_class_loss and present_coverage_loss: do they see logits that carry the excess?
Check whether any term is computed under autocast and whether that matters here.`,
  },
  {
    key: 'class-prior-and-imbalance',
    prompt: `LENS: the class-frequency machinery and whether it fights the aggregator.
configs/prism.py has class_weighting=True and rare_class_factor=4.0. Find exactly where
they are applied, to which loss, and with what normalisation. Then reason about direction:
v8-plan.md's 'two signatures' section argues these upweight rare classes and thereby CAUSE
signature (A) ghosting while trying to fix signature (B) absorption. Verify that from the
code and the numbers (per-class ghost_rate and pred_img vs gt_img in
proto_geometry_v5corrected.txt; the per-class IoU list in
e3_only/runs/prism-v5-corrected/PRISM_best_eval.log). Is there any principled
logit-adjustment / balanced-softmax anywhere, at train or test time? Is the decision rule
Bayes-consistent for the val prior? Quantify the bias a 4.0x rare-class factor puts on
the decision boundary in logits.`,
  },
  {
    key: 'eval-time-path',
    prompt: `LENS: what happens at test time, since the user's question is 'on testing'.
Read evaluate_prism.py end to end: _posterior, the plain argmax path, --region-vote, and
--presence-gate (core/inventory.py:~284-306). Answer: (1) does anything at eval time
compensate for the aggregator's per-class excess, or is the raw biased argmax what gets
scored; (2) is the region vote able to fix a naming error, or does it only propagate the
same biased argmax within a region — read _region_vote carefully and say what it votes on;
(3) the presence gate is described as a soft log-prior, logits + strength*log(sigmoid(p)*(1-floor)+floor),
worst case -3.0 logits at floor=0.05 — compare -3.0 logits to the 2.445-logit excess and to
the inter-class gaps, and say whether the gate is even strong enough to delete a ghost;
(4) any eval/train mismatch (normalisation, resolution, teacher vs student, TTA).
Also: dataset_prism.py returning n_sam=0 silently degrades region vote — is that live?`,
  },
  {
    key: 'resolution-and-context',
    prompt: `LENS: what information the classifier is even given. The classifier is a 1x1
convolution (F.conv2d with 1x1 prototypes) over an embedding. Establish from
model/decoder_v2.py and model/net.py: the embedding dimension, the spatial resolution at
which the classifier decides, how it is upsampled to the 256x256 output, and what
receptive field each decided pixel actually has. Then reason: a per-pixel 1x1 cosine
classifier has NO object-scale context, so 'field vs grass' and 'buildings vs mobile home'
must be decided from local spectral appearance alone. Is that sufficient in principle for
the 17 DLRSD classes? Which of the confused pairs in
artifacts/confusion_v5corrected_rownorm.csv are separable ONLY with context? Quantify how
much of the 0.2201 wrong-label error is plausibly context-limited rather than
aggregator-limited — this decides whether fixing the aggregator can possibly be enough.`,
  },
]

const VERIFY_LENSES = [
  {
    key: 'code-reality',
    ask: `Your job is to REFUTE the finding on the grounds that the code does not do what it
claims. Open the exact files and lines cited. Check the claim against the actual control flow:
is the term actually called? behind a flag that is off? overridden downstream? already handled
elsewhere? Is the cited symbol even the one in the live path (train_prism.py / evaluate_prism.py),
or is it dead E3-baseline code? Default to refuted=true if you cannot confirm the mechanism
from the source. Only refuted=false if you traced it and it holds.`,
  },
  {
    key: 'magnitude',
    ask: `Your job is to REFUTE the finding on the grounds that its size is wrong or negligible.
Check every number against the measurement files in ${R}/artifacts/ and the eval logs. Redo the
arithmetic. Ask: even if the mechanism is real, does it move an argmax? Is the claimed effect
smaller than the inter-class gaps it would have to overcome, or already dominated by another
effect? Is a correlation being read as causation (n=17, rho +0.44 is moderate)? Default to
refuted=true if the magnitude is unquantified hand-waving or if the arithmetic does not check out.
Only refuted=false if the size claim survives your own recomputation.`,
  },
]

const ANGLES = [
  {
    key: 'inventory-first',
    ask: `ANGLE: attack ghosting (0.3964 of all error) with a per-image class inventory
constraint. The point annotations give the exact set of classes present in each training
image for free, and the presence head is meant to predict it at test time. Design the
strongest version of this: how the inventory is supervised, how it is applied at
inference (hard restriction vs soft prior — note the current gate's worst case is only
-3.0 logits), what happens to recall when the presence predictor is wrong, and how to
calibrate it. Be explicit about what is measurement-only (reading dense masks) vs
deployable. Note Stage 0d (oracle_inventory) is running right now and will report the
ceiling of exactly this mechanism.`,
  },
  {
    key: 'decision-rule-first',
    ask: `ANGLE: attack both signatures with a principled class-prior correction to the
DECISION RULE — logit adjustment / balanced softmax / LA loss — replacing the ad-hoc
class_weighting=True, rare_class_factor=4.0. Derive it properly: what prior do you
subtract, estimated from what (the point annotations only — no dense masks), at train
time or test time or both, and why that is Bayes-consistent. Quantify the logit shift for
the extreme classes ('field': 8 GT images of 1319) and check it against the measured
inter-class gaps. Say what it does to the absorption direction (bare soil -> grass) at the
same time as the ghosting direction.`,
  },
  {
    key: 'classifier-geometry-first',
    ask: `ANGLE: fix the prototype bank itself. max_pair >= +0.96 for 15 of 17 classes
(near-duplicate prototypes), K_eff 2.70 of 4, near_other up to +0.8405 (classes barely
separated from each other), realised excess 0.1984 of a 0.2773 ceiling. Design the fix:
some combination of a bias-free aggregation (hard max / per-class-normalised / learned bias
with a zero-sum constraint), k_temperature, per-class K_c, a within-class repulsion that
actually fires, and — most importantly — an INTER-class separation term, which appears to
be entirely absent. Say which single change you would make first and what measurement
proves it worked. Quantify expected effect in logits, not adjectives.`,
  },
  {
    key: 'context-first',
    ask: `ANGLE: the classifier is 1x1 per-pixel and therefore has no object-scale context,
which may make 'field vs grass' and 'buildings vs mobile home' undecidable no matter how
good the prototypes are. Design the fix that adds context to the DECISION: region-level
classification over the frozen SAM partition (whose oracle ceiling is 0.9438), a
region-pooled embedding, image-level context injected into the per-pixel logit, or a small
context head. Be concrete about where in model/decoder_v2.py and evaluate_prism.py it
goes, and about the anti-confirmation-bias property that the SAM partition is
class-agnostic, computed once, and frozen. Say honestly how much of the 0.2201 wrong-label
error this can reach and how much it cannot.`,
  },
  {
    key: 'minimal-correctness-first',
    ask: `ANGLE: the skeptical, cheapest path. Assume the big architectural ideas are
unnecessary and the mIoU is being lost to a small number of outright defects plus an
uncalibrated decision rule. Identify the smallest set of changes — ideally ones that need
NO retraining, or one short run — that you can defend as strictly correctness fixes, and
order them by pp-per-hour. Include test-time-only options (they cost nothing and can be
measured on the existing 0.5477 checkpoint today: the CPU eval path is already wired).
Explicitly argue against the more elaborate proposals where you think they are unjustified
by the measurements. Your job is to be the one who is right when the fancy ideas fail.`,
  },
]

// ---------------------------------------------------------------------------
// Audit -> Verify, pipelined: a finding starts verification as soon as its own
// lens returns, without waiting for the other six lenses.
// ---------------------------------------------------------------------------
log(`auditing the aggregator through ${LENSES.length} independent lenses`)

const audited = await pipeline(
  LENSES,
  (L) => agent(
    `${CTX}\n\n${L.prompt}\n\nReturn at most 3 findings, the most consequential first. A finding may` +
    ` be "this is fine and here is the proof" — severity "non-issue" — and that is a valuable answer;` +
    ` do not manufacture defects. Quantify everything you can in cosine units, logits, or pp of mIoU.` +
    ` Cite file:line. Never invent a number: if it is not on disk, say so.`,
    { label: `audit:${L.key}`, phase: 'Audit', schema: FINDINGS },
  ),
  (res, L) => {
    const fs = ((res && res.findings) || []).slice(0, 3)
    if (!fs.length) return []
    return parallel(fs.map((f) => () =>
      parallel(VERIFY_LENSES.map((V) => () =>
        agent(
          `${CTX}\n\nA finding from an audit of this codebase is below. ${V.ask}\n\n` +
          `FINDING\n  id: ${f.id}\n  title: ${f.title}\n  severity: ${f.severity}\n` +
          `  mechanism: ${f.mechanism}\n  evidence: ${f.evidence}\n  magnitude: ${f.magnitude}\n` +
          `  claimed confidence: ${f.confidence}`,
          { label: `verify:${V.key}:${f.id}`, phase: 'Verify', schema: VERDICT },
        ),
      )).then((vs) => {
        const good = vs.filter(Boolean)
        const kept = good.filter((v) => !v.refuted)
        return {
          ...f,
          lens: L.key,
          votes_kept: kept.length,
          votes_total: good.length,
          survived: good.length > 0 && kept.length === good.length,
          partially_survived: kept.length > 0,
          verifier_notes: good.map((v) => `[${v.refuted ? 'REFUTED' : 'upheld'}] ${v.reasoning}` +
            (v.corrected_magnitude ? ` | corrected magnitude: ${v.corrected_magnitude}` : '')),
        }
      }),
    ))
  },
)

const allFindings = audited.filter(Boolean).flat().filter(Boolean)
const confirmed = allFindings.filter((f) => f.survived)
const contested = allFindings.filter((f) => !f.survived && f.partially_survived)
const verdicts = audited.filter(Boolean).length

log(`${allFindings.length} findings audited: ${confirmed.length} survived both verifiers, ` +
    `${contested.length} contested, ${allFindings.length - confirmed.length - contested.length} refuted`)

const digest = (list) => list.map((f) =>
  `- [${f.severity}/${f.lens}] ${f.title}\n    mechanism: ${f.mechanism}\n    evidence: ${f.evidence}\n` +
  `    magnitude: ${f.magnitude}\n    fix sketch: ${f.fix_sketch}\n    verifiers: ${f.verifier_notes.join(' || ')}`,
).join('\n')

const EVIDENCE = `
CONFIRMED FINDINGS (survived a code-reality verifier AND a magnitude verifier, both
prompted to refute):
${digest(confirmed) || '  (none)'}

CONTESTED FINDINGS (one verifier upheld, one refuted — treat as unproven, and say so if
you build on them):
${digest(contested) || '  (none)'}
`

// ---------------------------------------------------------------------------
// Design: needs every confirmed finding at once, so a barrier is correct here.
// ---------------------------------------------------------------------------
phase('Design')
log(`designing ${ANGLES.length} independent fixes against the confirmed findings`)

const proposals = (await parallel(ANGLES.map((A) => () =>
  agent(
    `${CTX}\n\n${EVIDENCE}\n\n${A.ask}\n\n` +
    `Produce ONE proposal, the strongest version of your assigned angle. You are advocating` +
    ` for it, but you must be honest about what it cannot reach — a proposal with a clear-eyed` +
    ` limit is stronger than one that claims everything. Ground every quantity in a number that` +
    ` is on disk. Where you need a number that does not exist, name the cheap measurement that` +
    ` would produce it. Remember the deployment constraint: 0% dense masks at training time;` +
    ` anything that reads a dense mask is measurement-only and must be labelled as such.`,
    { label: `design:${A.key}`, phase: 'Design', schema: PROPOSAL },
  ),
))).filter(Boolean)

const PROPOSAL_TEXT = proposals.map((p, i) =>
  `### ${i + 1}. ${p.name}  (angle: ${p.angle})\n` +
  `attacks: ${p.which_error_it_attacks}\nmechanism: ${p.mechanism}\n` +
  `expected gain: ${p.expected_gain_pp}\ntraining cost: ${p.training_cost}\n` +
  `changes: ${p.changes.join('; ')}\ngating measurement: ${p.gating_measurement}\n` +
  `risks: ${p.risks.join('; ')}\nsubsumes: ${p.kills_or_subsumes}`,
).join('\n\n')

// ---------------------------------------------------------------------------
// Judge: each judge must see all proposals to rank them, so barrier again.
// ---------------------------------------------------------------------------
phase('Judge')

const JUDGE_STANCES = [
  `You are the reviewer who cares only about whether the MECHANISM IS IMPLIED BY THE
MEASUREMENTS. Punish any proposal whose causal story is not forced by a number on disk.
Remember rho=+0.4436 at n=17 is moderate evidence, not proof.`,
  `You are the reviewer who cares only about pp OF mIoU PER HOUR OF GPU. The card is busy
for ~30h with prism-v8-shadow-on. Reward test-time-only changes measurable today on the
existing 0.5477 checkpoint. Punish anything needing many retrains for an unproven gain.`,
  `You are the reviewer who cares only about whether this SURVIVES PEER REVIEW as a
contribution. The stated novelty is 0% dense labels producing dense semantic maps, with
shadows and boundaries as the named contribution — but boundaries are 0.0210 of error and
the shadow branch has never run. Judge whether each proposal strengthens or dilutes the
paper's claim, and whether its evaluation could be attacked as leaking dense supervision.`,
]

const panel = (await parallel(JUDGE_STANCES.map((stance, i) => () =>
  agent(
    `${CTX}\n\n${EVIDENCE}\n\nFIVE COMPETING PROPOSALS:\n\n${PROPOSAL_TEXT}\n\n${stance}\n\n` +
    `Score every proposal 0-10 on soundness, expected_value and risk (10 = safest), give` +
    ` total = soundness + expected_value + risk, and state the single strongest objection to` +
    ` each. Then order them best-first. You may rank a proposal last and say it should not be` +
    ` done at all.`,
    { label: `judge:${['mechanism', 'cost', 'reviewer'][i]}`, phase: 'Judge', schema: SCORES },
  ),
))).filter(Boolean)

const tally = {}
for (const p of proposals) tally[p.name] = { total: 0, votes: 0, objections: [] }
for (const j of panel) {
  for (const s of j.scores || []) {
    if (!tally[s.name]) tally[s.name] = { total: 0, votes: 0, objections: [] }
    tally[s.name].total += s.total || 0
    tally[s.name].votes += 1
    if (s.fatal_objection) tally[s.name].objections.push(s.fatal_objection)
  }
}
const ranked = Object.entries(tally)
  .map(([name, t]) => ({ name, mean: t.votes ? t.total / t.votes : 0, objections: t.objections }))
  .sort((a, b) => b.mean - a.mean)

log(`panel ranking: ${ranked.map((r) => `${r.name} ${r.mean.toFixed(1)}`).join(' > ')}`)

// ---------------------------------------------------------------------------
phase('Synthesize')

const final = await agent(
  `${CTX}\n\n${EVIDENCE}\n\nFIVE PROPOSALS:\n\n${PROPOSAL_TEXT}\n\n` +
  `PANEL RANKING (mean of 3 judges, each with a different stance — mechanism-purist,` +
  ` cost-per-pp, and peer-reviewer):\n` +
  ranked.map((r) => `- ${r.name}: ${r.mean.toFixed(1)}/30\n    objections raised: ${r.objections.join(' || ') || 'none'}`).join('\n') +
  `\n\nFULL JUDGE REASONING:\n` +
  panel.map((j, i) => `judge ${i + 1} ordering: ${(j.ordering || []).join(' > ')}\n` +
    (j.scores || []).map((s) => `  ${s.name}: ${s.total} — ${s.why}`).join('\n')).join('\n\n') +
  `\n\nWrite the final answer for the researcher, who asked two things: (1) how do we solve` +
  ` the misclassification problem, and (2) is the current aggregator good enough — does it` +
  ` have issues.\n\nStructure it as:\n` +
  `A. THE AGGREGATOR VERDICT. Answer (2) directly and without hedging: is t*logsumexp(cos/t)` +
  ` over K prototypes an acceptable classifier score here, yes or no, and what exactly is wrong` +
  ` with it. Lead with the defect that has the largest quantified effect. Separate "provably a` +
  ` defect" from "suspicious but unproven".\n` +
  `B. WHAT THE MISCLASSIFICATION ACTUALLY IS. Reconcile the error budget with the audit: which` +
  ` share is aggregator bias, which is missing inter-class separation, which is the class prior,` +
  ` which is missing context, which is out of reach of the classifier entirely. Be explicit where` +
  ` the shares are estimates rather than measurements.\n` +
  `C. THE ORDERED PLAN. Steps, each with the change, the gating measurement that decides whether` +
  ` to proceed, the expected effect, and the cost. Put anything measurable today on the existing` +
  ` checkpoint first, since the GPU is busy for ~30h. Say which existing terms/flags each step` +
  ` makes redundant — the model has 14 loss terms and simplification is a goal.\n` +
  `D. WHAT WOULD FALSIFY THIS. The measurement that would show the whole diagnosis is wrong.\n\n` +
  `Be direct and quantitative. Do not pad. Do not invent numbers — every figure must trace to a` +
  ` file named above, and where a proposal's expected gain is a guess, mark it a guess. If the` +
  ` panel is split on something important, say so rather than papering over it.`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'max' },
)

return {
  aggregator_lens_verdicts: audited.length,
  findings_total: allFindings.length,
  findings_confirmed: confirmed.map((f) => `[${f.severity}] ${f.title} (${f.lens}) — ${f.magnitude}`),
  findings_contested: contested.map((f) => `[${f.severity}] ${f.title} (${f.lens})`),
  panel_ranking: ranked.map((r) => `${r.name}: ${r.mean.toFixed(1)}/30`),
  recommendation: final,
}
