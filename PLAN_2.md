# thinair — Phase 2: The Measurement Space

*This file is self-contained grounding. Link it, as-is, to an LLM and it carries
everything needed to (a) look at raw data the right way, (b) produce a concrete
measurement strategy for creating value from that data, and (c) turn such strategies
into thinair programs. Part 1 is the theory, Part 2 is what you do when handed data,
Part 3 is the framework the strategies run on, Part 4 is for the implementer.*

---

# Part 1 — The Pillars

## What this is

A large language model is not an oracle. It is a **measurement instrument**: it projects
objects onto axes you name. Once you accept this, arbitrary objects — texts, events,
people, collections, programs — acquire enough structure to import classical
mathematics: ordering, statistics, algorithms, even calculus. The price is honest
bookkeeping: every reading carries uncertainty, that uncertainty propagates through
everything computed from the readings, and confidence must be earned, never assumed.

Handed a pile of raw data, do not ask *"what could I generate from this?"* Ask:
*"what measurements can I take, by how many genuinely independent paths, what laws can
check them, and what ground can calibrate them?"*

**Definitions:**

- A **thing** is whatever you can point at: a text, an event, a person, a collection,
  a sequence, a measurement instrument.
- An **axis** is a named question with a **declared scale type** — nominal (categories),
  ordinal (order only), interval (differences meaningful), ratio (zero meaningful).
  Scale type determines the licensed math: medians and Spearman on ordinal readings;
  means, Pearson, and differences only on interval or better. An axis with no declared
  scale type is not yet an axis.
- A **reading** is an opinion `(value, p)`. `p` is the instrument's probability that the
  reading would be sustained — by the frozen ground where one exists, by validation
  otherwise. `p` is itself a reading: initially untrusted, calibratable over time.

## Pillar I — Measurement, not knowledge

Anything — an object, a collection, a sequence, an instrument itself — can be measured
along any nameable axis. Every reading is an opinion `(value, p)`, never a fact. No set
of readings exhausts the thing: there is always another axis.

- **Decomposition is naming axes.** "Extract the concepts" means: choose projections.
  The thing stays whole; you hold some of its shadows.
- **Collections are things.** A sort order is a measurement of a collection on an
  ordering axis; a faithful summary is a measurement on a synthesis axis, with its own
  `(value, p)`. (Quantities are still computed exactly — see Pillar II; qualitative
  synthesis is a measurement.)
- **Instruments are things.** A model, prompt, or judge is measurable: reliability
  (same reading twice, under reshuffled context?), character (more lawful comparing
  than scoring?), calibration (when it said 0.8, how often did the frozen record later
  agree?), and the dissimilarity between two instruments. The space measures itself
  with its own machinery.
- **Always another axis ⇒ always a forking path.** Measuring many axes and reporting
  the coherent ones is a selection artifact. Pre-register the axes a strategy will
  measure; label late additions exploratory — they earn less.

## Pillar II — Uncertainty enters only at the readings

All *epistemic* uncertainty enters at measurement: never ask the instrument to
adjudicate what code can compute. Downstream, computation is exact and adds no
uncertainty of its own. But exact is not clean — **uncertainty propagates, and the most
useful operations amplify it**:

- **Selection amplifies.** argmax, top-k, thresholds, sort near-ties: the winner of a
  noisy contest is disproportionately the entry with the largest positive error
  (winner's curse). Best-of-n at large n selects the candidate that best exploits the
  judge's biases. Every selected winner must survive a judge the selection was not
  optimized against.
- **Differencing amplifies.** Reading noise σ² becomes 2σ² in first differences, 6σ²
  in second differences. "Is the change changing" is the least reliable move here.
- **Repetition shrinks variance, never bias.** Readings from one model share
  systematic error. With intraclass correlation ρ, N readings are worth about
  n_eff = N/(1+(N−1)ρ) independent ones — bounded by 1/ρ forever. Confidence is bought
  with mechanism dissimilarity, never repetition count.
- Every derived quantity — rank, cluster, correlation — carries a distribution, not a
  value; a strategy must say how reading uncertainty reaches the deliverable.

**Cost model.** The measurement call is the unit of cost, with a **batch dimension**:
one call can compare two things, or order fifty short things listwise in one context —
often cheaper *and* more lawful than hundreds of isolated pairwise judgments. Realistic
cost is calls × tokens; plan in those units. Classical lower bounds transfer; classical
algorithms mostly need their noise-aware variants, because the oracle errs: with
resampleable errors, sorting costs Θ(n log(n/δ)); with **persistent** errors (same pair,
same wrong answer every time) the exact order is unrecoverable at any cost — the honest
target is approximate recovery. Trap: **caching and freezing convert independent errors
into persistent ones**, forfeiting repeat-and-vote. Sometimes the right trade — make it
deliberately.

## Pillar III — Confidence is unbuilt coherence

**Agreement is evidence exactly in proportion to how likely disagreement was.** The
only internal source of confidence is coherence that nothing enforced:

- **Laws holding.** Transitivity of comparisons, symmetry of similarity, associativity
  of folds, round-trips returning home, invariance under irrelevant rephrasing and
  reordering, monotonicity. No law is assumed — each is a validator, and its *degree of
  violation is a reading*, not an error.
- **Independent paths agreeing.** Sort by comparison and, separately, by absolute
  scores; answer from two disjoint halves of the evidence. Agreement earns confidence;
  divergence *localizes* the genuine uncertainty — a finding, not a failure.

Everything else about confidence derives from the one law above:

- Coherence you built in earns nothing — enforcement (schemas, dedup, sealed inputs)
  made disagreement impossible. Enforce anyway where it removes error sources; just
  know that what you enforce becomes *unobservable*, not absent.
- A constant-output instrument passes every law and is worthless — its readings could
  never have disagreed across objects. Readings must vary with the things measured.
- Dissimilarity is mechanism AND exposure: two prompts on one model share mechanism,
  and four methods reading the same biased or poisoned bytes share exposure — either
  way, disagreement was never likely, so agreement is cheap.
- A round-trip certifies only what could have failed in it: *stability*, not truth. A
  stable misunderstanding round-trips beautifully. Internal coherence sets a ceiling;
  only the Ground anchors it.

## Pillar IV — Readings persist; structure becomes instrument

A reading stands until the observed state changes **or a dissimilar path disagrees** —
persistence is a default, not a vow. Structure built from past readings — orders,
extracted axes, clusters, indexes, calibration histories — is an **instrument for the
next question**: a verified order places a new item in a handful of comparisons; a case
base turns forecasting into lookup plus adjustment. Never re-measure the world when
stored structure localizes the question.

**Freezing is stronger**: a deliberate act (code result, human assignment) that ends
consultation. A frozen wrong reading is a persistent error the machinery can no longer
see. Freeze consciously, freeze on the ground where possible, never on eloquence.

## The Ground

There is no truth in this space — only opinions. But some opinions are **frozen**: the
result of executed code, a resolved observable outcome, an explicit human assignment.
**Calibration — concordance between an instrument's past readings and the frozen
record — is the only place the space touches reality.** Coherence earns confidence;
concordance with the frozen ledger earns calibration; nothing else earns anything.
Where a domain offers no frozen outcomes (taste, counterfactuals, the unresolved
future), say so: everything there is coherence-only, capped by the calibration the
instrument earned in domains that do freeze.

## The moves

- **Prefer comparisons.** The instrument judges relative better than absolute (a
  meta-reading — confirm per instrument). Absolute scores are the cheap
  second-mechanism path; **listwise in one call** is a legitimate third when many short
  things fit one context.
- **Declare scale types; use only licensed statistics.** Ordinal → medians, Spearman.
  Interval or better → means, Pearson, differences. This habit alone removes a class of
  confident nonsense.
- **Read parts, compute derivations.** When the measurand derives from printed
  components — a sum, a rate, a difference — measure the components, each one
  groundable in the text, and derive in exact code. A grounding validator attached to
  the derived answer does worse than block it: it pulls the instrument toward whatever
  number *is* printed, and the result passes every check and is wrong.
- **Block before pairwise work.** Cheap projections partition; expensive pairwise work
  happens within blocks. Cautions: blocking errors are unrecoverable (validate block
  assignment on a sample), and a global ranking needs the comparison graph connected
  across blocks.
- **Trusted zones.** Coherence is local; a verified order carries a confidence
  *profile*, and tight regions let future queries inherit trust almost free.
- **Spend where information is bought, not merely where trust is thin.** The confidence
  profile is one input; the loss function — what a wrong answer costs where — is
  chosen, human-owned. Distinguish reducible from irreducible uncertainty before
  spending; an irreducibly noisy axis absorbs unlimited budget and returns nothing.
- **Shared axes make unlike things comparable.** Axes from corpus A, measured on corpus
  B: two point clouds in one space, compared with exact arithmetic. Validate by
  reversing the projection; check the finding survives a different defensible axis set.
- **Calculus on sequences — as validator, not theorem.** Diff adjacent things, diff the
  diffs, fold cumulatively; check that folded diffs recover the endpoint. On additive
  numbers that check passes by construction (validates nothing); on narrative diffs it
  is empirical — which is exactly what makes it a check. Mind 2σ²/6σ².
- **Values are chosen, never measured.** No hidden-weight aggregates. Per-axis
  measurements, exact aggregation, exposed human-owned weights; report which
  conclusions are robust across plausible weightings and which are weight-sensitive.
- **Creation is sampling; selection is measurement.** Generate N, then select — but
  selection amplifies (Pillar II): the winner must survive a judge it was not optimized
  against.

## The grading rule

**An operation carries confidence to the extent that it admits substantive checkable
laws and genuinely independent re-derivations — independent in mechanism and exposure.**
Law-free, single-path work can still be valuable (unique events, hypothesis
generation) but is *reported as a claim with explicit low probability*, never as a
finding. The scale runs from vibes (no law, one path, no ground) to calibrated findings
(laws held, dissimilar paths agreed, instrument calibrated against frozen outcomes).
Every deliverable states where it sits.

## Out of scope — say so, loudly

**Every internal path shares exposure to the data-generating process: whatever went
wrong upstream of the data is invisible to coherence by construction.** The pillars are
a theory of measurement and validation, not of where the data came from. Four named
consequences:

- **Causal questions.** Dissimilar paths reading the same non-randomized data share its
  confounding exactly. Causal claims need an assignment model, randomization, or stated
  causal assumptions; correlations delivered here are descriptive.
- **Problem formulation.** Choosing the question, measurand, and axes is prior and
  human-led — no reading detects that "incident" was redefined in March. Revisit the
  measurand as a standing practice.
- **Adversarial data.** An adversary builds coherence *into the data itself*; a
  competent fraud passes consistency checks, and injected instructions are read
  identically by every path. Unexpected smoothness is itself suspicious; only
  exposure-diverse paths (external records, out-of-band checks) help.
- **The absent population.** No internal path sees who is not in the data — silent
  churners, unfiled incidents. Name the absent population in every deliverable; only
  external paths reach it.

## Mathematical anchors

- **Representational theory of measurement** (Krantz–Luce–Suppes–Tversky): when numeric
  readings are licensed at all; scale types; meaningful statistics per type. The
  foundation.
- **Noisy sorting / rank aggregation**: approximate recovery under persistent noise;
  **Bradley–Terry** turns comparisons into interval scores with confidence intervals,
  makes comparison- and score-paths commensurable, and its lack-of-fit is a principled
  transitivity test (identifiable only if the comparison graph is connected).
- **Psychometrics**: Campbell–Fiske multitrait–multimethod analysis (the method factor
  = purchased agreement, quantified); generalizability theory (variance decomposition
  across instrument/prompt/occasion/order; answers "how many raters"); item response
  theory (per-reading standard errors; differential item functioning as the proper
  rephrasing-invariance test).
- **Design effect** n_eff = N/(1+(N−1)ρ): how correlated readings stop adding
  information. (The jury theorem assumes independence and *reverses* under shared bias.)
- **Metamorphic testing**: law-checking without an oracle; necessary, never sufficient;
  relation diversity beats count. **Property testing**: laws spot-checked in sublinear
  queries (guarantee is global, not top-k). **Multiple testing**: the price of "always
  another axis". **Winner's curse / Goodhart** for selection; **expected information
  gain** for budgets; order effects are classical context-dependence — randomize and
  counterbalance.

---

# Part 2 — The experiment protocol

When handed sample data and asked to create value from it, produce a **measurement
strategy** — a document, not code — with exactly these sections:

1. **Inventory.** What things, collections, sequences exist. What observable state is
   there. What frozen ground exists or could exist (outcomes, code-checkable facts,
   human labels) — and *what population is absent from the data*.
2. **Question and axes.** The stakeholder question in one sentence. The pre-registered
   axes with declared scale types. What in the question is a value/weight (chosen, to
   be surfaced to the human) versus a measurement.
3. **Measurement design.** Per axis: comparative, absolute, or listwise; batch shape;
   estimated cost in calls and tokens. What the classical zero-call pre-pass does first
   (dedup, counts, existing metadata — exhaust the free structure before spending).
4. **Validation design.** Per operation: which laws check it and what a violation will
   mean; which dissimilar paths re-derive it (dissimilar in mechanism AND exposure);
   the discrimination check; the instrument-reliability sub-pass (re-measure a sample
   under reshuffled context; the resulting agreement caps all downstream confidence).
   A law can check only a reading that exists: cells that never resolve are coverage
   gaps, not clean negatives — count them, and separate a refusal (a judgement) from a
   dead cell (the absence of one) by their recorded reasons.
5. **Confidence plan.** How per-finding confidence is composed from law readings and
   path agreement; the n_eff honesty note wherever paths share mechanism; what will
   remain a claim rather than a finding; where calibration against frozen ground is
   possible.
6. **Deliverable shape.** A matrix or profile, not a fused score; robust-vs-
   weight-sensitive split under exposed weightings; named missing axes and absent
   populations; the grading-rule position of each conclusion.
7. **Budget.** Two tiers: findings-producing work and validation-only work, priced
   separately, so the buyer of confidence knows what confidence costs.
8. **Persistence.** What structure survives for future queries (index, taxonomy,
   calibration record), what gets frozen and why, and what events invalidate what.

## Worked micro-example (condensed)

*Data: 800 support tickets + 2000 forum posts, 6 months. Question: what should
engineering fix first?*

- **Pre-pass** (0 calls): dedup, weekly volumes, tag frequencies.
- **Base measurement** (~350 calls, batched): every doc read on 6 pre-registered axes —
  canonical failure phrase (nominal), product area (nominal, *independent of the
  provided tag*), impact type (ordinal), consequence class (nominal), reproducibility
  (nominal), defect class (nominal). **Reliability sub-pass** (~35 calls): re-measure
  10% reshuffled; agreement per axis caps all downstream confidence.
- **Cluster into issues**: embed the failure phrases, cluster classically; second path
  partitions by discrete label agreement; arbitrate only disagreements (~50 calls);
  round-trip each cluster (statement re-accepts members, rejects near-non-members,
  ~40 calls).
- **Cross-corpus**: project forum posts onto ticket-derived issue axes AND tickets onto
  forum-derived axes; the residual (forum-loud, ticket-quiet) is a finding only if both
  directions flag it.
- **Severity**: comparison-sort top 25 issues (~116 calls) + independent absolute
  scores (25 calls); Spearman between the two orders = earned confidence, divergences
  reported as located uncertainty; transitivity spot-check on sampled triples.
- **Instrument finding** (0 extra calls): agreement between provided product-area tags
  and the independent re-read — the org's routing instrument, measured.
- **Deliverable**: issue × axis matrix + three named weightings + robustness split
  ("these 3 are top-5 under every plausible weighting") + stated absences (engineering
  cost axis; silent churners) + two-tier budget (~555 finding calls, ~330 validation
  calls).

That is the shape a strategy should take: every call priced, every claim graded, every
absence named.

---

# Part 3 — thinair, the framework underneath

## The framework in brief

thinair is a Python framework in which the measurement space is executable. The whole
of it:

- **Three nouns, no fourth.** A **Belief** is a function `b(e, a) → (v, p) | None` —
  `None` means "no opinion here", and abstention is the only scoping mechanism. An
  **Opinion** is a belief's recorded evaluation at a **cell** (an
  `(entity, attribute)` address): `(belief, entity, attr, value, p, t, frozen, meta)`.
  **frozen** is a flag on an opinion that ends consultation for its cell — set only by
  code or the human (assignment, executed code, an explicit `freeze`), never by a
  model.
- **A Thing owns no values.** Its attributes are cells; values live only in opinions;
  reading an attribute is selection among them. The framework surface is dunders —
  `__beliefs__` (the ordered panel of beliefs consulted on every cell) and
  `__ledger__` (append-only memory, the only memory) — plus operators: `+x` extracts
  the value, `~x` the probability, `@` is the exit ramp to plain Python (schema
  coercion, confidence gate, recast).
- **Beliefs are pure functions of a sealed snapshot** and consult each other by
  calling entries of `e.__beliefs__` directly (memoized, cycle-safe). Model, human,
  code check, validator — all are Beliefs, each with a **durable id** encoding its
  configuration, so calibration and dissimilarity are computable from the ledger
  alone, after the fact.
- **Resolution runs in rounds.** The routed head proposes; the panel judges; a
  `necessary` belief below the veto line kills the candidate and its objection feeds
  the next round; exhausted budgets escalate to the next proposer and end in
  `Unresolvable`. Results change only through observable state change.
- **Method calls are episodes.** Calling an undeclared method runs an imagined
  episode: the model proposes actions against a sealed snapshot, the resulting
  changeset commits atomically as ordinary opinions, and the return value faces the
  panel like any other candidate. Chained calls operate on fresh child things.
- **Verbs live in the module, not on objects**: `contract` (declares an attribute's
  shape and attaches its validators), `model`, `human`, `snapshot`, `source`,
  `freeze`, `freeze_call`, and `fn` — functions as cells: a call is a read of a
  `(call_id, "result")` cell; pure code freezes it (memoization), model-served calls
  stay opinions.

## The mapping

Every element of a strategy is a thinair construct:

| Strategy element | thinair construct |
|---|---|
| Measurement of thing `e` on axis `a` | Belief call `b(e, a) → (v, p)` |
| Reading | Opinion in the ledger |
| Axis with scale type | Contracted attribute cell (`contract(float, ...)`, `@`-coercion) |
| Law / validator | Deterministic Belief subclass (form, grounding, consistency, reference, executable families) |
| Dissimilar paths on one question | Panel of beliefs in `__beliefs__` on one cell |
| Instrument reliability / calibration | Meta-measurement over the ledger, keyed by durable belief ids |
| The Ground | **Frozen opinions** (code results, human assignments) |
| Persistent structure / index | Ledger + frozen opinions on derived cells |
| Budget tiers | Rounds / escalation, runtime-owned policy |
| Exposed weights | Human belief / frozen assignments — never a model output |

Layer 2 — scoring beliefs by dissimilarity-weighted agreement — remains deferred. Its
design evidence comes from these experiments: the ledger of every run records which
beliefs agreed, on what, with what mechanism/exposure overlap, and how their readings
fared against frozen outcomes. Method-factor and design-effect machinery over that
record *is* Layer 2. The path to building it is part of this plan: experiments
accumulate ledgers; once the record spans the moves — ordering, shared axes, persistent
structure — it is distilled into an implementation brief for Layer 2, derived from what
the ledgers show rather than from theory. That brief, not this file, is the build
document.

# Part 4 — For the implementer

When a strategy graduates from document to code, build it on the thinair package
(SPEC.md in the repository is the as-built contract). Rules of the build:

1. **No new framework surface.** A strategy is beliefs, validators, panels, policies,
   and module verbs — if implementing one seems to need a new Thing method, the
   strategy is misread. Re-map it via the table above. Instruments the package does
   not ship — an embedding model, a classical clusterer — enter as code beliefs; they
   need no framework change.
2. **Every experiment leaves a complete ledger.** Durable belief ids (model + prompt +
   version) on every opinion, so calibration and dissimilarity are computable after the
   fact. An experiment whose ledger cannot answer "which paths agreed and how
   independent were they?" was wasted.
3. **Pre-registration is write-once.** Hash the strategy and its inputs before the
   first call; a resumed or repaired run never re-stamps. Drift between the stamp and
   the code that actually ran is disclosed, not overwritten.
4. **Freezing discipline.** Frozen = code results and human acts only. In experiments,
   freeze outcomes (ground) eagerly and conclusions never.
5. **Validation is budgeted, not bolted on.** The two-tier budget from Part 2 §7 is a
   runtime concern: findings calls and validation calls are separately countable in the
   ledger.
6. **Reliability before findings.** The instrument sub-pass (Part 2 §4) runs first;
   its reading is recorded and caps reported confidence downstream. If the instrument
   can't read an axis reliably, the axis is redesigned or dropped before money is
   spent on it.
7. **Fold, then archive.** An experiment is finished only when its *general* lessons
   are folded into the grounding documents (this file; SPEC.md when the contract
   moved) and its record — pre-registered strategy, code, ledger — is committed
   whole. The record is evidence, not documentation: a finding severed from its
   ledger degrades to a claim, and no one should ever need an experiment's report
   to use the framework.

The settlement half of every strategy — veto-aware reading of the record,
scale-licensed agreement, reliability/drift/discrimination, concordance with the
frozen ground, calibration, Bradley–Terry, the two-tier budget — ships as
`thinair/evaluate.py` (SPEC.md §12) and is pure classical math over the ledger:
never re-implement it inside an experiment, and never ask the model to compute
what it already computes. The model is a column factory; `evaluate` is the
grader that makes trying many columns cheap.

The immediate use of this file needs no code at all: link it to an LLM, provide sample
data, and ask for a measurement strategy per Part 2. The strategies are the experiment;
their quality is the test of this document.
