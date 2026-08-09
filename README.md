# thinair

Python objects whose attributes are beliefs, not values.

Reading `invoice.total` consults a panel — a model, your code, validators,
you — and hands back a value together with an honest probability. Everything
anyone ever said is kept, with a name attached.

```python
from thinair import Thing, contract, model, human
from thinair.validators import TokenSubset

class Invoice(Thing):
    """An invoice document to be understood."""
    __beliefs__ = [model(), human("jane"), TokenSubset("source_text")]
    source_text: str
    total = contract(float, extracted_from="source_text", range=(0, 1e6))

inv = Invoice(source_text=open("invoice.txt").read())

+inv.total     # 1249.5 — the value
~inv.total     # 0.93   — how sure the answering belief is, its own honest p
```

**Code the certain, believe the rest.**

## The surface

What you write yourself is certain, and the model can never touch it:

```python
inv.total = 1249.50      # your assignment: probability 1.0, final
```

What you left blank is believed. The first belief in the panel answers;
the validators judge its candidate and can veto it (`TokenSubset` above
refuses any number that isn't actually in the text) — but they only ever
gate, never inflate: the probability you get is the answering belief's own.
Corroboration doesn't turn a 0.6 into a 0.9; it turns it into a 0.6 that
nothing objected to. Even attributes nobody declared work this way:

```python
inv.due_date             # never declared — imagined on first read,
                         # validated like everything else, priced like
                         # everything else
```

Three operators cover the rest — `+`, `~`, and `@` in three costumes:

| form | meaning |
|---|---|
| `+thing` | the value |
| `~thing` | the probability |
| `thing @ {"total": float}` | coerce to a schema |
| `thing @ 0.9` | confidence gate — below the bar collapses to a falsy carrier |
| `blob @ Invoice` | revive a saved one |

Low confidence fails *visibly*: a gated value that didn't clear the bar is
falsy and keeps its probability, so failures explain themselves instead of
flowing onward.

```python
guess = inv.total @ 0.9
if guess:                # gate whole branches on how sure the answer is
    pay(+guess)
```

## Methods nobody wrote

Calling an undefined method runs an *episode*: the model works against a
sealed snapshot of the object, proposes changes and a return value, and the
same validators judge the result before anything lands. Writes commit
atomically or not at all — and a model can never mark anything certain.

```python
summary = inv.summarize()
+summary, ~summary       # a value and a probability, like every read
```

## There is no truth here — only opinions

The model is one belief among several. So is your code, so is every
validator, so are you (`human("jane")`). The framework records who said
what and never referees: when *differently built* beliefs agree, that is
your best evidence you're onto something. Disagreement is signal too — it
tells you exactly where to look.

Every opinion lands in a durable ledger (`.thinair/opinions.db`, on by
default — `THINAIR_STORE=off` to opt out). Relaunch your program and
everything certain is served from the record; nothing you or your code
established is ever asked twice.

## Install

```bash
pip install thinair
```

Point it at any OpenAI-compatible endpoint — local or hosted:

```bash
export THINAIR_MODEL=...       # model name
export THINAIR_BASE_URL=...    # e.g. http://127.0.0.1:8000/v1
export THINAIR_API_KEY=...     # if the endpoint wants one
```

Python ≥ 3.11, zero runtime dependencies.

## Going deeper

The quiet payoff: once model readings arrive as honest `(value, p)` pairs,
data no parser can read — text, events, judgments — becomes *measurable*,
and a new kind of data analysis opens up.
[`thinair/GROUNDING.md`](thinair/GROUNDING.md) is that theory, written to be
handed to an LLM together with your raw data ("propose a measurement
strategy for this"); it ships inside the package. `thinair.evaluate` then
grades what the readings earned — reliability, concordance, calibration —
in pure classical math.

- [`SPEC.md`](SPEC.md) — the contract: every guarantee, stated so it can be checked.
- [`thinair/GROUNDING.md`](thinair/GROUNDING.md) — the measurement theory, LLM-linkable.
- [`experiments/`](experiments/) — a real, disclosed run: strategy, ledger, findings.

MIT licensed.
