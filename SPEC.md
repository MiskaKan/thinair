# SPEC.md — `thinair`

The contract: what the code guarantees, stated so it can be checked.

---

## 1. Vocabulary

Three nouns, and there is no fourth.

| noun | what it is | where it lives |
| --- | --- | --- |
| **Belief** | a function `b(e, a) -> (v, p)` | `thinair/beliefs.py`, and user code |
| **Opinion** | a Belief's recorded evaluation at a cell | `thinair/ledger.py` |
| **frozen** | a flag on an Opinion that ends consultation for its cell | `Opinion.frozen` |

A **cell** is an `(entity, attr)` address.  A **panel** is `thing.__beliefs__`,
an ordered list of Beliefs.  There is no truth here, only opinions.

## 2. Invariants

These hold in every module, and each has a test that fails loudly.

| # | invariant | enforced by |
| --- | --- | --- |
| 1 | One record shape: every opinion is `(belief, entity, attr, value, p, t, frozen, meta)` | `Ledger.add` raises `TypeError` on anything else |
| 2 | The ledger is dumb, append-only memory; no deletion API, no evaluative math | `test_invariants.py` |
| 3 | Beliefs are pure functions of a sealed snapshot | `Snapshot.__setattr__` raises; determinism table-tested per validator |
| 4 | Freezing is a code-only capability | no `frozen=True` in any model-reachable module; episode returns carrying `frozen` are rejected |
| 5 | Vetoes are control flow; everything else is measurement | no policy returns a `p` no belief stated |
| 6 | Belief identity is durable | id derives from the full configuration; `register` refuses a clash |
| 7 | No network in the core | import-graph assertion over `ledger`, `thing`, `policy`, `debug`, `validators`, `evaluate` |

## 3. The Belief contract

```python
class Belief:
    necessary: bool = False     # veto right
    veto_line: float = 0.5      # the line below which a necessary belief vetoes
    proposes: bool = False      # declared: routing must know before spending a call
    id: str                     # durable; names a fixed configuration
    def __call__(self, e, attr) -> tuple | None: ...
```

* Returning `None` means **no opinion here**, and that is also the only
  scoping mechanism.  A validator handed a candidate that is not its kind of
  thing abstains; it does not veto.
* Returning `(value, p)` or `(value, p, meta)` states a belief.  `p ∈ [0, 1]`.
* The runtime **never checks `isinstance`**.  Any callable with `id`,
  `necessary` and this signature participates.
* A belief must be deterministic: same snapshot → same answer.  Graded is
  fine; stochastic is not.

## 4. The snapshot

`e` is a sealed value, never the live Thing.

| name | meaning |
| --- | --- |
| `getattr(e, "<domain name>")` | that cell's resolving `Opinion`, or `None` |
| `e.__beliefs__` | the panel, memoized; `[0]` is always the routed proposer |
| `e.__ledger__` | this snapshot's visible slice of memory |
| `e.__entity__` `__value__` `__p__` `__purpose__` `__provenance__` | identity and carried value |
| `e.__contracts__` `__methods__` `__arguments__` `__objections__` | what a proposer is told |
| `e.__episode__` `__call_arguments__` | present only for episodes and call-cells |
| `e.__attrs__()` `e.__opinion__(attr)` | the resolved cells |

Two guarantees, both invisible at the interface: panel entries are memoized
(one evaluation per belief per cell per round), and re-entry yields `None`
(cycles resolve instead of recursing).

## 5. The read pipeline

`thing.attr`:

0. **Frozen short-circuit.**  Latest frozen opinion wins; no consultation.
0b. **Replay.**  Under a policy with `replays_from_record` (the default,
   `Proposed`), if the head can fingerprint its context without spending
   (`ModelBelief.exposure`) and the cell's *latest* round-1 reading carries
   the same `exposure`, the recorded negotiation is this read: its final
   candidate — provided it survived its judges, all of whom must sit on this
   panel, and clears any active `require` — resolves at zero calls and zero
   writes.  Any mismatch or doubt falls through to live rounds.  This is
   how the same program against the same record replays to the first cell
   whose world moved, and runs live from exactly there.
1. **Open.**  Build `e₀` from frozen state and standing resolutions, *minus
   the cell being derived*.  No generative member → `Unresolvable`.
2. **Round.**  Every belief is called once with the round's snapshot.  Every
   non-`None` answer is recorded, never frozen.  Generative members below the
   routed head are the ladder, not the panel, and are not consulted.
3. **Veto → next round.**  A `necessary` belief whose p *for the current
   candidate* is below its veto line opens the next round, whose snapshot
   carries the prior rounds' opinions and objections.  Budget: 3 rounds, then
   escalate the route; 2 more per escalated route; then `Unresolvable`.
4. **Resolve.**  The active `ResolutionPolicy` selects.  It never blends.
   `Consulted` makes every generative member answer and records the spread;
   the resolution is still the head's vetted candidate, its own p.
5. Return a child `Cell` bound to the resolving opinion.

## 6. Freezing

Exactly three paths:

| path | author | p |
| --- | --- | --- |
| `thing.attr = x` (and constructor kwargs) | the resident `HumanBelief` | 1.0 |
| `@fn` bodies, real method writes | `code:<qualname>@<source-hash>` | 1.0 |
| `freeze(thing.attr)` | unchanged | unchanged |

Latest frozen wins; every predecessor stays in the ledger.  Freezing bypasses
the veto gate deliberately: validators gate proposals, not authority.
Assignment is idempotent: re-stating the latest frozen value from the same
author records nothing, so a replayed program leaves no diary of identical
re-statements.

## 7. Operators

| form | meaning |
| --- | --- |
| `+t` | the value of the resolving opinion |
| `~t` | its probability |
| `t @ <schema>` | coerce; conforming is free, impossible returns a `None`-carrier keeping the diagnostic p |
| `t @ 0.8` | confidence gate; pure |
| `t @ ThingSubclass` | recast: entity and frozen state carry over, caches invalidate |
| `blob @ ThingSubclass` | revive (`ThingMeta.__rmatmul__`) |

`@` never raises on epistemic failure.  Comparisons and numeric/container
protocols delegate to the carried value; none of them trigger inference.

## 8. Contracts

`contract(template, extracted_from=, range=, enum=, length=, format=,
checksum=, sums_to=, unique=, elaborates=, necessary=, beliefs=)` appends
scoped beliefs to `__beliefs__` and nothing else.  A bare annotation
(`source_text: str`) is a contract too.  The `Schema` a contract builds is the
same object that constrains the engine's structured output and performs the
post-hoc check.

## 9. Episodes

Calling an undefined name runs an episode: a pure function from a state
snapshot to `(changeset, return_value, p)`.

* Grammar: `get <attr>`, `call <real_method>(args)`, `return {changes, value, p}`.
  There is no freeze action, and a return carrying `frozen` is rejected.
* Budgets: 8 actions, 3 corrections, depth 1.
* The return value faces the discriminative panel like any other candidate.
* Every write must target a declared, unfrozen attribute and pass the full
  per-attribute pipeline.  Commit is atomic: all of it or none of it.
* Repetition keys on `(entity, state hash, call expression)`, where the
  state hash covers the host *and every Thing-valued argument* — an argument
  whose state changed invalidates the memo exactly like an interleaved
  assignment on the host.  `returns=` is stripped from the identity — it
  governs acceptance, not identity.

Disambiguation: contract-declared name called → re-derivation; real method →
real code; any other called name → episode.  Plain access never runs one.

## 10. Functions as cells

`net_total(100, 0.24)` is the cell `("fn:net_total(100, 0.24)", "result")`.
Call ids are implementation-free, so code and model opinions about one call
are comparable forever.

| `@fn` shape | behaviour |
| --- | --- |
| body, `pure=True` | frozen opinion authored `code:…@<hash>`; memoization is the frozen short-circuit; editing the body re-freezes |
| no body | the model serves the cell through the ordinary pipeline; never frozen |
| both | competing implementations over one cell; `fallback=True` uses the model when code raises |
| `pure=False` | a sensor: a fresh cell per call, tagged `observed`, never cached |

`freeze_call(...)` installs a fixture, which *is* a frozen opinion.  A mock is
one in a scoped ledger.

## 11. Persistence and observability

`thing.__getstate__()` → `{"thinair": 1, class, entity, opinions, resolved}`;
`blob @ Class` revives; `pickle` rides the same path.  Reviving into a
different class is a recast and drops cached resolutions.  Ledgers save and
load whole.

`with Thing.debug():` prints one bordered box per operation: route, each
proposal, each verdict with its reason, the resolution, episode actions, and
commit or rollback.  `source(thing)` / `thing.__source__` renders frozen
attributes plain and believed ones annotated `# p=0.93 ← model/extract-v3`.

**The store.**  The default ledger is durable: `.thinair/opinions.db`,
created lazily on the first append.  `THINAIR_STORE=off` keeps the old
in-memory default; `THINAIR_STORE=<path>` relocates the file; explicit
`__ledger__=` and `use_ledger(...)` are untouched.  `Ledger` itself is the
backend interface — a backend overrides the kernel (`add`, `opinions`,
`cells`, `beliefs`, `next_t`, `__len__`, `__iter__`) and inherits the rest;
`store.SqliteLedger` is the shipped one.  On relaunch the tape continues
(`t` resumes from `MAX(t)`), and everything frozen short-circuits from the
record at zero calls.  The store is operational truth; the committed
evidence of an experiment remains a JSON extract (§13 rule 7), from which
the database is rebuildable — storage slices, `evaluate` judges.

The store also keeps a `belief` table: id, kind, `necessary`, `veto_line`,
`proposes`, description — written once per speaking belief, read back via
`belief_row(id)`.  Descriptions, never bodies: the database stores what the
system said and exactly who said it (invariant 6 makes the id the full
configuration); mechanism lives in code.  A fresh process can therefore
settle any record without reconstructing the strategy's classes.

The `thinair` command inspects any store or `ledger.json` archive
git-style — `log`, `show`, `status`, `branch` (entities are branches),
`blame` — a pure derivation over §12's `history`, spending nothing.

## 12. Settlement: `thinair.evaluate`

Layer 2's first slice, as built: what the record *earned*, computed from the
ledger and nothing else.  Everything below the driver is classical math with
zero model calls — the import-graph assertion of invariant 7 covers
`evaluate.py`.

| group | names |
| --- | --- |
| driver | `evaluate(thing, order=)` — resolve every declared cell; `order` is context (the twin reshuffle); unresolvable cells profile as `gap`, never raise |
| record | `reading` (veto-aware; an unregistered judge raises rather than reading as a low score), `verdicts`, `coverage` (resolved / vetoed / absent) |
| agreement | `agree` / `adjacent` (scale-typed), `kappa` (chance-corrected: agreement is evidence in proportion to how likely disagreement was) |
| statistics | `ranks`, `median`, `spearman` (paired), `mannwhitney` (unpaired), `wilson`, `n_eff` |
| orders | `bradley_terry` — strengths with SEs, graph connectivity, consistency (the principled transitivity reading) |
| instrument | `reliability` (+ bespoke `compare`), `drift`, `discrimination`, `grounded` → `concordance`, `calibration`, `separation`, `tiers` |
| record structure | `graph` (typed edges: authored / ref / host / child; exposure groups), `lineage` (upstream: what a value rests on), `invalidated` (downstream: what a change calls into question) |
| commits | `history` — the record as authored, atomic state transitions: the tree is the state hash; assignments, episode changesets (parent tree re-derived and checked against the recorded pointer) and settlements commit; deliberation lives inside its commit; corroborations are notes; replay commits nothing |
| data | `LICENSED` (scale type → licensed statistics), `GRADES` (vibes → claim → finding → calibrated) |

Ground has two homes, one gatherer: `grounded(ledger)` reads outcomes frozen
*after* measurement on the same cell — calibration accrues from the ledger
alone; `ground=fn` maps measured entities to parallel ground entities that
stay invisible to the instrument.  Model opinions carry an `exposure`
fingerprint in `meta` (a hash of the rendered context), so dissimilarity in
mechanism AND exposure is computable from the ledger after the fact.  Where
a Thing or Cell crosses into the record — assignment, episode arguments,
`fn` arguments — `meta.refs` keeps the durable addresses (`entity`,
`entity#attr`) that value-reduction would otherwise destroy; the record's
dependency graph is thereby a pure derivation, never a stored structure.

**Second opinions.**  The module verb `corroborate(thing, attrs=, beliefs=)`
(it spends calls, so it lives with the runtime, not in `evaluate`) consults
beliefs the read never asked — by default the generative ladder below the
head; explicitly, any belief, on the panel or not — once per resolved cell
against the standing snapshot, recording each opinion **into the same cell**
tagged `corroboration`.  The resolution is untouched, nothing freezes, and
the verb is idempotent by exposure: a belief that already answered a cell
under the same context is not asked again.  What it records is exactly what
this section's functions settle.

## 13. Building a strategy

The measurement theory and the strategy protocol live in
`thinair/GROUNDING.md` — a self-contained file made to be linked to an LLM
together with raw data.  When a strategy graduates from document to code,
these are the rules of the build — not what the code guarantees, but what
the strategist owes the record:

1. **No new framework surface.**  A strategy is beliefs, validators, panels,
   policies, and module verbs — if implementing one seems to need a new
   Thing method, the strategy is misread; re-map it via the table in
   GROUNDING.md Part 3.  Instruments the package does not ship — an
   embedding model, a classical clusterer — enter as ordinary code beliefs.
2. **Every experiment leaves a complete ledger.**  Durable belief ids on
   every opinion, so calibration and dissimilarity are computable after the
   fact.  An experiment whose ledger cannot answer "which paths agreed and
   how independent were they?" was wasted.
3. **Pre-registration is write-once.**  Hash the strategy and its inputs
   before the first call; a resumed or repaired run never re-stamps.  Drift
   between the stamp and the code that ran is disclosed, not overwritten.
4. **Freezing discipline.**  Frozen = code results and human acts only.  In
   experiments, freeze outcomes (ground) eagerly and conclusions never.
5. **Validation is budgeted, not bolted on.**  The two-tier budget
   (GROUNDING.md Part 2 §7) is a runtime concern: findings calls and
   validation calls are separately countable in the ledger
   (`evaluate.tiers`).
6. **Reliability before findings.**  The instrument sub-pass (GROUNDING.md
   Part 2 §4) runs first; its reading caps reported confidence downstream.
   An axis the instrument cannot read reliably is redesigned or dropped
   before money is spent on it.
7. **Fold, then archive.**  An experiment is finished only when its
   *general* lessons are folded into GROUNDING.md (or this contract, when
   the contract moved) and its record — pre-registered strategy, code,
   ledger — is committed whole.  The record is evidence, not documentation:
   a finding severed from its ledger degrades to a claim, and no one should
   ever need an experiment's report to use the framework.

The settlement half of every strategy ships as §12 and is classical math
over the ledger: never re-implement it inside an experiment, and never ask
the model to compute what code computes.  The model is a column factory;
`evaluate` is the grader that makes trying many columns cheap.

## 14. Deferred, and permanently excluded

**Deferred (Layer 2, second slice):** credibility, similarity between
beliefs, dissimilarity-weighted pooling, credibility-driven routing,
`Pooled` / `MostCredible`.  A `scoreboard` needs no surface: it is the §12
instrument measurements grouped by durable belief id.  Also: nested imagined
calls, handles for oversized episode values, async consultation,
non-OpenAI-compatible transports.

**Permanently excluded:** model-served actuators; any freeze path reachable by
a model; any narrative-memory runtime.  State plus the ledger is the whole
story.
