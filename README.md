# thinair

Python objects whose attributes are beliefs, not values.

```python
from thinair import Thing, model, human
from thinair.validators import TokenSubset

class Invoice(Thing):
    """An invoice document to be understood."""
    __beliefs__ = [model("deepseek-v4-flash"), human("jane"),
                   TokenSubset("source_text")]
    source_text: str
    total = Thing(float, extracted_from="source_text", range=(0, 1e6))

inv = Invoice(source_text=open("invoice.txt").read())

+inv.total     # 1249.5 — the value
~inv.total     # 0.93   — how sure the answer is
```

You declared `total` but never computed it. Reading it asks a model,
checks the answer against the source text, and hands back a value with
an honest probability.

**Code the certain, believe the rest.**

## Write what you know

Anything you set yourself is certain, and nothing can change it:

```python
inv.total = 1249.50        # yours: probability 1.0, final
```

Anything you leave blank is believed — even attributes you never
declared:

```python
inv.due_date               # works anyway: proposed on first read,
                           # validated like everything else
```

## Every answer carries its price

Three operators cover the whole surface:

| form | meaning |
|---|---|
| `+thing` | the value |
| `~thing` | the probability |
| `thing @ 0.9` | confidence gate — below the bar turns falsy |
| `thing @ {"total": float}` | coerce to a schema |
| `blob @ Invoice` | revive a saved one |

Low confidence fails *visibly* instead of flowing onward:

```python
guess = inv.total @ 0.9
if guess:                  # only runs when the answer clears the bar
    pay(+guess)
```

## Validators keep answers honest

A declaration attaches checks that can veto a bad answer — a number that
isn't in the source text, a value out of range, a string outside an
enum. Checks reject; they never inflate. The probability you get is
always the answering belief's own.

```python
priority = Thing(str, enum=["low", "normal", "high", "urgent"])
amount   = Thing(float, extracted_from="source_text", range=(0, 10_000))
```

You can also declare *expectations* — a probability bar, a maximum
disagreement — which never block a read but mark the record wherever
they are missed:

```python
total = Thing(float, p=0.9, deviation=0.1)
```

## Methods nobody wrote

Calling an undefined method runs the model against a sealed snapshot of
the object. Proposed changes are validated and land atomically — or not
at all. A model can never mark anything certain.

```python
summary = inv.summarize()
+summary, ~summary         # a value and a probability, like every read
```

## Everything is remembered

Every opinion — who said it, what it saw, what the validators thought —
lands in a durable store (`.thinair/opinions.db`, automatic; set
`THINAIR_STORE=off` to opt out). Run the same program again and settled
answers come back from the record at zero cost. Nothing you or your
code established is ever asked twice.

## Inspect it like git

The record maps onto git so cleanly the CLI is a deliberate copy:
commits are whatever changed the object, every entity is a branch, and
changing the belief panel is itself a commit.

```console
$ thinair log --oneline
40ea1fe90c0f (HEAD -> ticket-4417) [freeze] refund_amount = 89.9 (frozen)
918571d773f7 [settle] sentiment ⇒ "frustrated" (p 0.66 ±0.00)
b68dab468888 [assign] priority = "urgent" (frozen)
01c676a6af53 [settle] customer ⇒ "Anna Virtanen" (p 0.91 ±0.04)

$ thinair show HEAD        # the whole object + a belief × attribute matrix
$ thinair blame ticket-4417
$ thinair diff 01c676...40ea1fe
$ thinair branch
```

Every believed value wears its **trust signature**: `(p 0.91 ±0.04)` —
the probability, and how far apart the beliefs that checked it landed.
Color says the rest at a glance: the number is green when everything on
record agrees and slides toward red when readings disagree; the parens
are green when every belief that *could* be asked has been.

## Ask for second opinions

`show` displays a matrix of every belief against every attribute, with
`?` on each question nobody asked yet. One command asks them all:

```console
$ thinair evaluate HEAD    # consults models and validators against the
                           # record, fills the matrix, remembers forever
```

Evaluation is idempotent — re-running costs nothing — and agreement
between *independent* beliefs is the strongest evidence this system
offers.

## Bring your own beliefs

A belief is just a class with a `judge` method. Register one and the
CLI can rebuild and consult it like the built-ins:

```console
$ thinair belief add checks.py     # lives in .thinair/beliefs/
$ thinair belief list
```

## Built for agents

One command gives a coding agent everything: the measurement theory,
the list of built-in checks, and a manual for this CLI.

```console
$ thinair ground                   # pipe it into the agent's context
$ thinair --ai-readable log        # the colors, stated as text
```

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

## Learn more

- [`SPEC.md`](SPEC.md) — the contract: every guarantee, stated so it can be checked.
- [`thinair/GROUNDING.md`](thinair/GROUNDING.md) — the measurement theory, written to be handed to an LLM.
- [`experiments/`](experiments/) — a real, disclosed run: strategy, ledger, findings.

MIT licensed.
