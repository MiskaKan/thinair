# Measurement strategy — 12 wire items, one newsroom question

*Written to the protocol in `PLAN_2.md` Part 2. Eight sections, every call
priced, every claim graded, every absence named. The code that executes it is
`strategy.py` + `run.py`; the analysis is `analyze.py`; the result is
`REPORT.md`.*

Instrument under test: **DeepSeek V4** (`deepseek-v4-flash`, `deepseek-v4-pro`)
over an OpenAI-compatible endpoint, temperature 0.2, no seed.

---

## 1. Inventory

**Things.** 12 news items (`corpus.py`), 57–83 words each. One collection: the
corpus itself. One sequence: none — the items are near-simultaneous, so no
calculus-on-sequences move is available here and none is attempted.

**Observable state.** Per item: the wire text, and nothing else. No metadata,
no source reputation, no traffic, no editor's tag. This is a deliberately thin
inventory — everything measured must come out of 70 words.

**Frozen ground that exists.** The corpus is synthetic and authored, so four
axes have a designed-in human assignment, which is ground in the sense
PLAN_2 means (an explicit human act, not a model output): `country`,
`people_affected`, `certainty`, `horizon`. Two further grounds are *computable*
rather than assigned: the arithmetic each item asserts about itself
(`calculate`, no model in the loop) and the weekday of each ISO dateline
(`CalendarFact`). Two axes have **no ground at all**: `severity` and
`follow_up`. Everything said about those two is coherence-only, capped by the
calibration the instrument earns on the four that do freeze.

**The absent population — the largest caveat in this document.** The corpus is
synthetic. Its absent population is *everything*: real wires, real ambiguity,
real adversarial copy, real duplicates that are not near-verbatim, and every
story nobody filed. Worse than absent data, there is **shared authorship**: the
items were written by a language model and are read by a language model, so
some part of any agreement between corpus and reader is one process recognising
its own output, not evidence. Two consequences are load-bearing:

1. **No conclusion in the deliverable is about the news.** The deliverable is a
   demonstration of shape. Every finding that survives is a finding about *the
   instrument*, which is legitimate here because instruments are things and can
   be measured (Pillar I).
2. Concordance figures on the four grounded axes are an **upper bound** on what
   the same instrument would achieve on real copy, not an estimate of it.

## 2. Question and axes

**Stakeholder question (one sentence).** *Of these twelve incoming wires, which
should a small newsroom staff for follow-up this week?*

**Pre-registered axes.** Registered here, before any call is made; anything
added later is labelled exploratory and earns less (Pillar I, forking paths).

| # | axis | scale type | licensed statistics | ground |
|---|---|---|---|---|
| 1 | `country` — ISO 3166-1 alpha-2 of the dateline | nominal | mode, agreement rate | assigned |
| 2 | `people_affected` — see measurand below | ratio | mean, ratios, Pearson | assigned |
| 3 | `certainty` — 1 rumour / 2 attributed / 3 official / 4 record | ordinal | median, Spearman | assigned (rubric decidable from the text) |
| 4 | `horizon` — 1 days / 2 weeks / 3 months / 4 never | ordinal | median, Spearman | assigned (each item names its resolution event) |
| 5 | `ratio_claim` — the item's own arithmetic, transcribed as `{expression, result}` | nominal (a transcription) | agreement rate | computable |
| 6 | `severity` — 1..5, harm already done | ordinal | median, Spearman | **none** |
| 7 | `follow_up` — 1..5, warrants newsroom follow-up this week | ordinal | median, Spearman | **none** |

**Measurand for `people_affected`,** stated so it is decidable rather than
tasteful: *the number of people the item says are directly affected; where two
sources conflict, the official figure; where only components are given, their
sum; where no people are affected, 0.* Three items make this bite — `n03` and
`n06` give only components, `n09` gives two conflicting counts, `n10` has no
people at all.

**Values, not measurements.** `follow_up` is not a measurement: it is a
*value*, and the weights that turn severity, certainty, horizon and scale into
"worth staffing" are chosen by the newsroom, not read off the wire. It is
measured here anyway, as an axis, precisely so it can be compared against an
exact aggregation of the other axes under **exposed weights** (§6). No hidden
weight aggregate is reported.

## 3. Measurement design

**Zero-call pre-pass, first.** Exhaust free structure before spending anything:
token and number inventories, ISO dateline extraction, weekday check via
`CalendarFact`, the dateline country name looked up in the vendored ISO table,
and near-duplicate detection by **three signals declared together** so that no
threshold is tuned after seeing the answer — 5-shingle Jaccard ≥ 0.5 (word
order), unigram Jaccard ≥ 0.5 (vocabulary), and an exact match on
`(dateline date, multiset of numbers)` (numbers alone). Three signals rather
than one because they differ in mechanism, and a paraphrased duplicate defeats
some of them. Expected to find the `n11`/`n12` pair and the planted wrong
weekday without a single call.

**One call per (item, axis).** thinair's grain is the cell, and here that grain
is a virtue rather than a cost: reading `severity` in its own call means the
reading cannot be conditioned on the `certainty` the model itself just stated.
Batching all seven axes into one call would be ~7× cheaper and would
manufacture exactly the internal coherence that Pillar III says earns nothing.
We pay for the separation.

**Not separated, and admitted:** within an item, later axes *do* see earlier
resolved axes in the sealed snapshot, because that is what thinair shows a
proposer. So cross-axis coherence within one item is partly built. It is
therefore never used as evidence; it is only ever used as the thing the
reliability sub-pass perturbs (§4).

**Listwise where the batch dimension is real.** The ordering path reads all
twelve items in one context and returns a permutation — one call, and more
lawful than 66 pairwise judgments would be. It is a call-cell
(`rank_stories(ids)`), so it lands in the ledger like everything else.

| pass | shape | model | calls |
|---|---|---|---|
| pre-pass | classical | — | 0 |
| base measurement | 12 items × 7 axes, per-cell | flash | 84 |
| ordering path A | absolute `follow_up`, cold entity, one item per call | pro | 12 |
| ordering path B | listwise permutation, whole corpus in one context | pro | 1 |
| **findings tier** | | | **97** |
| reliability sub-pass | 4 items × 7 axes, reversed axis order, twin entity | flash | 28 |
| transitivity spot-check | 4 triples × 3 pairwise comparisons | flash | 12 |
| order-effect probe | the `(a, c)` pair of each triple, asked reversed | flash | 4 |
| **validation tier** | | | **44** |

141 calls, plus whatever extra rounds vetoes open — a vetoed candidate costs
another call, so the true total is a measurement, not a prediction, and the run
counts it. At ~70-word items the prompts run ~600–1,200 tokens.

**Three orders, not two.** Path A (pro, absolute, cold) and path B (pro,
listwise) differ *only* in batch shape, which is the comparison worth making.
The base pass also yields a third order — flash's `follow_up`, read with its
sibling axes already resolved in the snapshot. That third order differs from
path A in **both** model and context, so it is reported as a confounded
comparison and never used to claim independence.

## 4. Validation design

**Laws, and what a violation will mean.**

| law | checks | a violation means |
|---|---|---|
| `TokenSubset` on numbers | every number in a reading appears in the item | a fabricated figure — the classic hallucination tell |
| `Range` / `Enum` | ordinal axes stay inside their declared scale | the instrument is not reading the rubric |
| `IsoCountry` | `country` names a country in the pinned table | either a bad reading **or** a gap in the table — the table is itself an instrument, and `n02`, `n08`, `n11`, `n12` sit outside its 35-country coverage on purpose |
| `Calculator` | the transcribed `ratio_claim` actually evaluates | the *item's* arithmetic is wrong (a finding about the data) or the transcription is (a finding about the instrument) — the two are separated by hand in §6 |
| `CalendarFact` | the dateline weekday holds | a defect in the wire |
| transitivity | over 4 sampled triples of pairwise `follow_up` comparisons | the comparison instrument does not induce an order, and any ranking built from it is not recoverable |
| order invariance | the `(a, c)` pair of each triple asked in both directions | the comparison instrument is reading position, not content — a classical context effect, and fatal to any ranking |
| set-preservation | the listwise permutation is the input set, no drops, no inventions | the batch dimension is not safe at n=12 |

`TokenSubset` is attached **necessary** on `people_affected` — and this is
expected to misfire on `n03` and `n06`, whose correct answers (16, 385) are
sums that appear nowhere in the text. That is not a bug to be worked around: it
is the pre-registered prediction that a grounding validator and an arithmetic
measurand are in genuine tension, and the run will show which wins.

**Independent paths.**

- `follow_up`: absolute scores (`pro`, per item) vs a listwise permutation
  (`pro`, one context) vs pairwise comparisons (`flash`, sampled). Spearman
  between the first two; the third only spot-checks the law.
- `people_affected`: model reading vs the frozen assignment (calibration, not a
  path) and vs `TokenSubset`'s deterministic number extraction (a genuinely
  dissimilar mechanism, same exposure).
- **n_eff honesty note, up front:** flash and pro are the same family on the
  same server reading the same bytes. Mechanism overlap is high and exposure
  overlap is total. Their agreement is worth **far less than two independent
  readings** — the honest reading of two such paths is n_eff barely above 1.
  No number in the deliverable will be justified by "two models agreed".

**Discrimination check.** For every axis, does the reading vary across items?
A constant-output instrument passes every law and is worthless (Pillar III). An
axis whose readings collapse onto one value is reported as failed, not as
unanimous.

**Instrument reliability sub-pass, run before the findings are believed.** Four
items (`n01`, `n06`, `n09`, `n10` — spanning the grounded and ungrounded, the
easy and the deliberately hard) are re-measured on all seven axes on a twin
entity with the axis order reversed, so each reading sees a different set of
already-resolved siblings. Per-axis agreement between the two passes **caps all
downstream confidence on that axis**.

Pre-registered rule, with teeth: **an axis whose sub-pass agreement is below
0.5 is dropped from the deliverable's weighted aggregation.** Its readings are
still collected and reported, but nothing is built on them. Four items is a
thin base for that estimate and the report says so; the rule is declared here
so it cannot be negotiated after the numbers arrive.

## 5. Confidence plan

Per-axis confidence is composed, in this order, and the *lowest* binds:

1. **Reliability ceiling** — the sub-pass agreement rate for that axis. Nothing
   downstream may claim more.
2. **Discrimination** — an axis that does not vary is capped at zero regardless
   of everything else.
3. **Law readings** — the fraction of that axis's readings that survived its
   validators, and for `people_affected` and `country` the exact concordance
   with the frozen assignment.
4. **Path agreement**, for `follow_up` only, discounted by the n_eff note.

The instrument's own stated `p` is treated as **a reading, not a probability**:
it is recorded, and then checked against outcomes on the four grounded axes.
That check — stated `p` versus observed hit rate — is the only calibration this
experiment can produce, and it is produced on 48 readings, so it will be
reported with its width and not as a number.

**What will remain a claim, never a finding:** anything about `severity` or
`follow_up`. No ground exists for them, both are coherence-only, and the corpus
is synthetic. They are reported at the "law-free / single-family" end of the
grading rule.

## 6. Deliverable shape

A **matrix, not a score**: 12 items × 7 axes, each cell carrying `(value, p)`
and its validator verdicts, plus a per-axis confidence row from §5.

The newsroom question is answered with an exact aggregation over the measured
axes under **three exposed weightings**, all human-owned and printed in full:

- *harm-first* — severity 0.5, people 0.3, certainty 0.1, horizon 0.1
- *verifiability-first* — certainty 0.4, horizon 0.3, severity 0.2, people 0.1
- *scale-first* — people 0.5, severity 0.3, certainty 0.1, horizon 0.1

with a **robustness split**: which items are top-4 under every weighting
(robust), and which move (weight-sensitive). The model's own `follow_up` axis
is shown *beside* the weighted aggregations, never fused with them.

Also reported, because they are the honest outputs of a small run: the
**instrument findings** (reliability per axis, discrimination per axis,
concordance per grounded axis, stated-p calibration), the **data findings**
(which items' own arithmetic fails, which dateline is wrong, which pair is a
duplicate), and the **named absences**: no engineering-cost axis, no audience
axis, no source-reputation axis, no real-world corpus, no independent
mechanism.

Every conclusion carries its grading-rule position, in one of three words:
**calibrated** (law held, dissimilar paths agreed, concordance with frozen
ground), **coherent** (laws held, paths agreed, no ground), **claim** (single
path, no law, no ground).

## 7. Budget

| tier | what it buys | calls | share |
|---|---|---|---|
| findings | the 12 × 7 matrix, the two ordering paths | 97 | 69% |
| validation | reliability sub-pass, transitivity and order probes | 44 | 31% |

Priced separately on purpose: the buyer of confidence should see that nearly a
third of the spend buys no findings at all. The ledger tags every opinion with its
belief id, so the split is recomputable from the record rather than trusted
from this table.

## 8. Persistence

**Survives the run** (`ledger.json`): every opinion ever rendered, keyed by
durable belief id — model name, temperature, prompt template version and model
folder version. That is what makes "which paths agreed, and how independent
were they?" answerable after the fact, which is the standing requirement of
PLAN_2 Part 4 rule 2, and what a future Layer 2 would be trained on.

**Frozen deliberately:** the corpus text (constructor assignment, `human:desk`)
and the designed-in ground, frozen on a separate `Ground` entity per item so
that a calibration query is a ledger read rather than a re-measurement.
Conclusions are frozen **never** (Part 4 rule 3).

**Invalidation:** a new prompt template version, a new model folder version, a
different model name or a different temperature all mint a new belief id, so
old readings do not silently mix with new ones — they sit side by side and
become comparable instead. Editing an item's text changes the entity's frozen
state and invalidates every reading of that item, which is the only
invalidation the framework performs automatically.
