# experiments

Measurement strategies from `thinair/GROUNDING.md`, executed on the
`thinair` package under the build rules of SPEC.md §13.

Each experiment is a directory with the same five parts, in the order the
grounding puts them:

| file | what it is |
|---|---|
| `corpus.py` | the data, and whatever ground it carries |
| `STRATEGY.md` | the Part 2 document: inventory, axes, measurement design, validation design, confidence plan, deliverable shape, budget, persistence — written **before** the first call |
| `strategy.py` | that document as a thinair program: contracts, panels, and any belief the package does not ship |
| `prepass.py` / `run.py` | the classical zero-call pass, then the phases, leaving `ledger.json` |
| `analyze.py` / `report.py` | the ledger read back, and `REPORT.md` rendered from it |

Two rules make the shape work, both from SPEC.md §13:

* **No new framework surface.** An experiment is beliefs, validators, panels,
  policies and module verbs. An instrument thinair does not ship — a
  clusterer, an embedder, "is this list a permutation of that set" — enters as
  an ordinary code belief and needs no framework change.
* **Every experiment leaves a complete ledger.** Durable belief ids on every
  opinion, so "which paths agreed, and how independent were they?" is
  answerable after the fact rather than during. `analyze.py` reads the ledger
  and never the runner's own log, which is how that stays true.

Runs are resumable: `run.py` writes the ledger after every item and skips any
item already in it.

## news

Twelve synthetic wire items, seven pre-registered axes, one newsroom question.
Four axes carry designed-in ground, so concordance is computable; two carry
none, and everything said about those is coherence-only.

```
THINAIR_BASE_URL=http://127.0.0.1:8000/v1 python experiments/news/run.py
python experiments/news/report.py
```

Offline, the machinery is covered by `tests/test_news_experiment.py` — the
corpus, the pre-pass, the two custom beliefs, the statistics, and one whole
item measured end to end against a scripted engine. The *findings* need an
endpoint; the machinery does not.
