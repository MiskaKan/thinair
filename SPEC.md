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
    frozen: bool = False        # what this belief settles is pinned (CodeBelief: True)
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
   the cell being derived*.  No generative member → consult
   `Thing.__default__` (a class attribute naming the fallback proposer, MRO
   inheritance); still none → `Unresolvable`.
2. **Round.**  Every belief is called once with the round's snapshot.  Every
   non-`None` answer is recorded, never frozen.  Generative members below the
   routed head are the ladder, not the panel, and are not consulted.
3. **Veto → next round.**  A `necessary` belief whose p *for the current
   candidate* is below its veto line opens the next round, whose snapshot
   carries the prior rounds' opinions and objections.  Budget: 3 rounds, then
   escalate the route; 2 more per escalated route; then `Unresolvable`.
4. **Resolve.**  The active `ResolutionPolicy` selects.  It never blends.
   `Consulted` makes every generative member answer and records the spread;
   the resolution is still the head's vetted candidate, its own p.  A head
   declared `frozen=True` pins what it settles — at settlement, never at
   statement, so a vetoed candidate can pin nothing (§6).
5. Return a child `Cell` bound to the resolving opinion.

## 6. Freezing

Freezing is a property of the belief, never of a model's eloquence.  The
paths:

| path | author | p |
| --- | --- | --- |
| `thing.attr = x` (and constructor kwargs) | the resident `HumanBelief` | 1.0 |
| `@fn` bodies, real method writes | `code:<qualname>@<source-hash>` | 1.0 |
| a panel member with `frozen=True` settling (§5 step 4) | that belief | its own |

There is no freeze *verb*: a value code already knows is assigned, and a
belief entitled to pin declares `frozen=True`.  Latest frozen wins; every predecessor stays in the ledger.  Freezing bypasses
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
| `t += belief` / `t -= belief` | change the panel in place; recorded as a panel declaration (a `[belief]` commit when it moves) |
| `t + belief` / `t - belief` | a fresh Thing on the same entity and ledger with the changed panel |

`@` never raises on epistemic failure.  Comparisons and numeric/container
protocols delegate to the carried value; on a *deferred* cell (an
undeclared attribute never read before) delegation resolves it first —
`THINAIR_OFFLINE=1` (§11) turns any such surprise consultation into an
immediate error instead of a spend.  A Thing assigned to an attribute
reduces to its **entity id** (a reference), with `meta.refs` carrying the
same address.

## 8. Declarations

`Thing(template, extracted_from=, range=, enum=, length=, format=,
checksum=, sums_to=, unique=, elaborates=, necessary=, beliefs=, doc=)` in
a class body declares an attribute: it appends scoped beliefs to
`__beliefs__` and nothing else.  A bare annotation (`source_text: str`) is
a declaration too.  `beliefs=` entries are first-class declarations, not
second-class attachments: each is auto-scoped to the attribute, keeps its
veto terms, and its `describe()` joins the prompt description — the named
options are shorthand over the same validator library
(`enum=[...]` ≡ `beliefs=[EnumBelief([...])]`), and the two spellings must not
diverge.  (A Thing constructed with a positional shape is a
*declaration*: no entity, no ledger, no record — the metaclass consumes it
at class creation.)  The `SchemaBelief` a declaration builds is the
same object that constrains the engine's structured output and performs the
post-hoc check.  Docstrings are prompt material: the class docstring is the
snapshot's `purpose`, `doc=` joins the declaration's description, and a real
method's first docstring line rides along with its signature.

Three more options never reach the prompt.  `p=` (a floor, or `(lo, hi)`
bounds) and `deviation=` (a max spread) declare **expectations** — Layer 2
statements about the cell, stamped into the record at settlement and judged
there (`history` derives per-cell consensus; the CLI colors violations).  A
belief by itself is never wrong, so expectations mark, they never gate — and
they are deliberately invisible to the answering belief, whose p must stay
honest rather than inflate to a declared bar.  `eager=True` resolves the
attribute at construction instead of first read (replay and the frozen
short-circuit make this free on a warm record).

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

**Environment.**  One table, because every knob is discoverable nowhere
else: `THINAIR_MODEL` / `THINAIR_BASE_URL` / `THINAIR_API_KEY` /
`THINAIR_MAX_TOKENS` name the endpoint; `THINAIR_TIMEOUT` shortens the
900 s transport leash (reasoning models think for minutes — cutting them
off mid-thought retries the same long thought, slower);
`THINAIR_OFFLINE=1` makes any would-be network call raise *before* the
socket, so an accidental consultation is a stack trace, not a spend; a
belief resolving to model `default` against the fallback base URL raises
the same way — misconfiguration fails loudly, never by posting to
localhost; `THINAIR_PROGRESS=1` narrates each model call on stderr as it
starts and lands; `THINAIR_STORE` as above and `THINAIR_PAGER` below;
`THINAIR_MODELS_PATH` adds model folders; `NO_COLOR` wins over color.

The store also keeps a `belief` table: id, kind, `necessary`, `veto_line`,
`proposes`, description — written once per speaking belief, read back via
`belief_row(id)`.  Descriptions, never bodies: the database stores what the
system said and exactly who said it (invariant 6 makes the id the full
configuration); mechanism lives in code.  A fresh process can therefore
settle any record without reconstructing the strategy's classes.

The `thinair` command inspects any store or `ledger.json` archive
git-style — `log`, `show [commit]`, `status`, `branch [-d name]` (entities
are branches), `blame`, `diff A...B` (two trees, cell by cell),
`belief add <file.py> | rm <name> | list` (custom instruments, below) — a
pure derivation over §12's `history`, spending nothing.  (`source` and
`beliefs` were folded into `show`, which renders the annotated tree and
the panel.)  A *commit*
argument resolves the way git resolves names: a hash prefix, a branch
(entity) name meaning that ref's tip, or `HEAD` (the newest commit
overall; the default).  The log carries no per-commit branch column —
membership is ancestry, so refs appear only as tip decorations, on by
default as in git (`--no-decorate` for plain hashes).  `--graph` draws
one colored swim lane per live chain, following parent hashes — chains
that fork from one commit genuinely converge, the extra lanes bending
home with `/` on a connector row above their shared parent (`|/`); no
merges exist in the record, so `\` never appears.  On a terminal, `log`,
`show`, `diff`, `blame` and `branch` page through `less -FRX` exactly as
git does (`THINAIR_PAGER`, then `PAGER`, override; empty or `cat`
disables), and color survives the pager.  `--oneline` rows cap at the
screen's width, cut with an ellipsis (ANSI-aware); a piped reader gets
whole lines, never a cut.  A believed cell
renders everywhere as its **trust signature** — `(p 0.93 ±0.04)`, the
resolving belief's honest p with the agreeing voices' deviation, always
numeric (`±0.00` when unmeasured).  The text is painted
on the overlap gradient, and the gradient counts **every belief that
spoke** — the resolving reading or fiat, the negotiation's panel
verdicts, its *other* candidate stances (recorded on the settle commit
as `readings` — escalation is a second voice), and corroborations.
There is exactly one kind of voice: models, validators, humans and
code all count the same — a validator dissents through its p, which
the `±` spread carries.  With two or more voices the score folds each
dissenting voice's `evaluate.similarity`
to the held value (trigram Jaccard for strings, relative closeness for
numbers, token Jaccard for containers — classical, model-free) — green
for agreement, mid-ramp for dissent, distance as color rather
than text.  An **unopposed** cell — nothing else spoke at all —
renders its `p` dim: the value channel is unearned.
The `±` wears its own color regardless,
from the *min-max range* of the recorded ps rather than their deviation
(one voice far from the rest barely moves a deviation; the range refuses
to average it away) — it colors
whenever two ps exist and dims only when the cell is alone —
the printed number stays the deviation.  A violated
declared expectation (§8) is always the theme's red.  The *parens* carry the
other channel: coverage — of the mechanisms this client could consult
*on this cell*, the fraction that have spoken; the pool is per
attribute (every model, the mechanisms whose scoped wrappers name the
attribute, and rebuildable mechanisms no wrapper claims — priority's
enum never counts against customer; durable criteria only, so warm and
cold processes agree).  Green parens mean the cell's panel is complete;
red parens mean almost nobody was asked.  `--ai-readable` states both
channels in text (`agree=` — or `agree=unopposed` for a cell where
nothing else spoke — `asked=`, `expect-violated`) for readers without a palette.  The signature is the standard rendering wherever a
believed cell appears: the log, `show`'s context lines, `blame`, the
matrix footer.  The readings panel shades each reading by its own
overlap with the held value — the record's disagreements are visible
reading by reading.  `show` renders the *whole
tree* as of the commit with what moved highlighted and the rest as dim
context, then a belief × attribute matrix over the tree (each cell that
belief's latest p, shaded by its value's overlap with the held one via
`evaluate.similarity`, in the terminal's own palette — the *bright*
green for agreement (the branch hue without the bold), the gradient's
one and only green, theme
red for none, faded red/yellow between; the *resolving* reading —
the value the program actually served, its owning commit's author —
renders underlined (emphasis by line, never a second shade of the
color); one row per *mechanism*,
scoped wrappers pooling into their inner belief's row since the column
already names the attribute; an empty cell says why it is empty — `?`
could be asked and never was, which is exactly what `evaluate` fills
in, `x` this client has no way to call it *on this cell* — askability
follows the cell's panel (the same attachment rule as the parens'
coverage: models anywhere, wrapper-claimed mechanisms only on their
named attributes, unclaimed rebuildable mechanisms anywhere), and
`evaluate` consults by the same rule, so a `?` is always a consultation
`evaluate` would actually perform; in a *frozen* column recorded
readings still show, but empty cells read `frozen` rather than inviting
anyone, judged as of the commit shown — closing with a `(held)` footer
row stating each cell's standing consensus, `p 0.95 ±0.02`, which
updates live while `evaluate` fills the table.  The footer is the one
place a frozen cell speaks in probability: the fiat is `p 1.00`, and its
deviation measures how close the declared fact sits to what other
beliefs have read — `±0.00` until someone has), then
the readings panel: every proposer, per changed cell — its latest
reading, or `-` where it never spoke — so what `evaluate` records is
visible there from then on.  Matrix and readings *unwind to the
commit*: only opinions on record by the end of its negotiation appear —
a later override never haunts an earlier commit — plus corroborations
later targeted at exactly this commit, which are measurements of this
state and `evaluate`'s cache.  Matrix and readings pool opinions across the commit's refs:
one commit is one content, so it gets one panel; branches are pointers,
and per-ref detail stays reachable as notes and in the ledger itself.  Commit identity is git's: sha1(parent |
tree | author | kind | message | changes) — **the entity is not in the
hash; entities are refs.**  Anonymous runs with byte-identical histories
collapse into one chain carrying every ref, and different histories fork
exactly where they diverge; a commit's Date is the first time its state
was reached.  The tree itself stays the bare state hash episodes point at.
Output colors only a terminal; `NO_COLOR` wins.  `thinair ground`
dumps the shipped GROUNDING.md as-is — no meta explaining the file to
its reader — minus the strategy-design-only stretches (the mathematical
anchors, Part 2 — the experiment protocol — and the Layer 2 outlook), so
the pillars, the moves, and the framework land whole in an agent
harness's inline tool output; `--full` restores the cuts.  Both close with two generated
appendices — the built-in
belief roster, and a client manual for agents (a runnable program
skeleton — the deliverable is a running program, and the manual demands
nothing of the project's structure: no file is required reading and
none has to exist — the command set, `--ai-readable`, the verify loop:
debug through the record and finish only when `agree=` is high with the
matrix asked full, how to add beliefs, what the marks mean) — store-free
and pipe-pure, so an agentic session's first command can be its own
grounding *and* its own tool manual.  Two commands write.  `evaluate [commit] [belief|*]`
rebuilds the commit's tree as a sealed snapshot, consults reconstructible
beliefs, and records the answers as corroborations, idempotent per
(commit, belief, cell) — a collapsed commit's refs share one evaluation,
since the content is the same; archives are refused, because evaluation
writes.  Frozen columns are skipped by default — a cell pinned by fiat
is not a question — and the skip is announced; `--include-frozen`
(bare, or naming one attribute) turns the question back on.  Reconstructible means models (their ids parse back into
configurations — invariant 6) *and* built-in validators: the belief
table stores every described belief's constructor configuration (a
`config` column; scoped wrappers also describe their inner mechanism),
so `evaluate '*'` rebuilds each validator whose config survived JSON and
has it re-judge the held tree — the candidate it judges is the record's
own value.  A config that did not survive JSON is skipped entirely: a
judge that cannot be rebuilt exactly must not be rebuilt at all.
Custom instruments extend the same door: `thinair belief add <file.py>`
copies a belief module into `<store dir>/beliefs/`, and `evaluate`
imports every registered file before rebuilding — classes resolve by
kind exactly like the built-ins, and module-level instances register
under their durable ids and are used live.  `belief rm` and
`belief list` manage the folder; it travels with the store.  On a
terminal it is the matrix answering itself: the table sits below the log
and fills in one cell at a time (`…` marks the consultation in flight,
`?` becomes a colored p as each reading lands); piped output stays plain
log lines with the finished table at the end.  `branch -d`
is the package's one deletion: it drops the ref's opinions from the
*operational* store (episode sub-entities included), which invariant 2
tolerates exactly because the store is derived structure, rebuildable from
archives — commits shared with surviving refs are re-derived from those
refs' opinions, untouched.

## 12. Settlement: `thinair.evaluate`

Layer 2's first slice, as built: what the record *earned*, computed from the
ledger and nothing else.  Everything below the driver is classical math with
zero model calls — the import-graph assertion of invariant 7 covers
`evaluate.py`.

| group | names |
| --- | --- |
| driver | `evaluate(thing, order=)` — resolve every declared cell; `order` is context (the twin reshuffle); unresolvable cells profile as `gap`, never raise |
| record | `reading` (veto-aware; an unregistered judge raises rather than reading as a low score; frozen fiat deliberately returns `None` — a fiat never inflates an instrument's record), `settled` (the standing value a program would be served: latest frozen, else the last reading), `verdicts`, `coverage` (resolved / vetoed / absent) |
| agreement | `agree` / `adjacent` (scale-typed), `kappa` (chance-corrected: agreement is evidence in proportion to how likely disagreement was), `similarity` (graded 0..1 overlap, licensed by type — how far apart two readings landed where `values_equal` only answers whether they landed together) |
| statistics | `ranks`, `median`, `spearman` (paired), `mannwhitney` (unpaired), `wilson`, `n_eff` |
| orders | `bradley_terry` — strengths with SEs, graph connectivity, consistency (the principled transitivity reading) |
| instrument | `reliability` (+ bespoke `compare`), `drift`, `discrimination`, `grounded` → `concordance`, `calibration`, `separation`, `tiers` |
| record structure | `graph` (typed edges: authored / ref / host / child; exposure groups), `lineage` (upstream: what a value rests on), `invalidated` (downstream: what a change calls into question) |
| commits | `history` — the record as authored, atomic transitions: the tree is the state hash; assignments, episode changesets (parent tree re-derived and checked against the recorded pointer), settlements and *panel changes* commit (kind `belief`: `Thing.__init__` records the declared panel fingerprint idempotently as `__panel__`; the first declaration per entity is a silent baseline, a changed one commits `+ added - removed` with the tree untouched — the instrument is part of what the history is a history of); deliberation lives inside its commit; corroborations are notes; replay commits nothing.  Each commit carries per-cell `consensus` — agreeing voices (`n`), their p-deviation *and* min-max `range`, `dissent` count with its mean `similarity` — every belief that spoke counts the same, models and validators alike; the negotiation's other candidate beliefs ride the commit as `readings` (each instrument's last stance) — and any declared `expect` (§8) — a belief is never wrong alone; the spread is the signal |
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
non-OpenAI-compatible transports.  And remotes (`fetch` / `push` / `pull`
between stores): honest merging must reconcile two tapes whose `t` clocks
never met — interleaving them re-times every negotiation and re-parents
every derived commit — so it waits until it can be done without lying about
order.

**Permanently excluded:** model-served actuators; any freeze path reachable by
a model; any narrative-memory runtime.  State plus the ledger is the whole
story.
