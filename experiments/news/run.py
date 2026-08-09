"""Execute STRATEGY.md against a live endpoint, leaving a complete ledger.

    THINAIR_BASE_URL=http://127.0.0.1:8000/v1 python experiments/news/run.py

Resumable: the ledger is written after every item, and an item whose readings
are already in it is skipped.  Phases run in the order PLAN_2 Part 4 rule 5
demands --- the reliability sub-pass before the findings it caps.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from thinair import Ledger, model, use_ledger                    # noqa: E402
from thinair.engine.openai_compat import OpenAICompatEngine      # noqa: E402
from thinair.policy import Unresolvable                          # noqa: E402

import corpus                                                     # noqa: E402
import prepass                                                    # noqa: E402
import strategy                                                   # noqa: E402

LEDGER_PATH = os.path.join(HERE, "ledger.json")
PREPASS_PATH = os.path.join(HERE, "prepass.json")
RUNLOG_PATH = os.path.join(HERE, "runlog.json")

FLASH = "deepseek-v4-flash"
PRO = "deepseek-v4-pro"

#: Pre-registered, before the run: the reliability sample and the triples whose
#: pairwise comparisons are checked for transitivity.
RELIABILITY_SAMPLE = ("n01", "n06", "n09", "n10")
TRIPLES = (("n01", "n04", "n11"), ("n03", "n06", "n09"),
           ("n05", "n07", "n10"), ("n02", "n08", "n12"))
#: The (a, c) pair of each triple is also asked reversed: an order-effect probe.
ORDER_PROBE = tuple((c, a) for a, _b, c in TRIPLES)

#: An axis whose sub-pass agreement falls below this is dropped from the
#: deliverable's weighted aggregation.  Declared before the numbers exist.
RELIABILITY_FLOOR = 0.5


class CountingEngine(OpenAICompatEngine):
    """The transport, counting.  The two-tier budget must be a measurement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0

    def complete(self, *args, **kwargs):
        text, meta = super().complete(*args, **kwargs)
        self.calls += 1
        self.prompt_tokens += meta.get("prompt_tokens") or 0
        self.completion_tokens += meta.get("completion_tokens") or 0
        self.seconds += meta.get("latency") or 0.0
        return text, meta

    def snapshot(self):
        return dict(calls=self.calls, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    seconds=round(self.seconds, 1))


def _delta(before, after):
    return {k: round(after[k] - before[k], 1) for k in before}


def read(thing, attr, log):
    """One reading.  An unresolvable cell is an outcome, not a crash."""
    started = time.time()
    try:
        cell = getattr(thing, attr)
        value, p = +cell, ~cell
        outcome = "resolved"
    except Unresolvable as exc:
        log.append(dict(entity=thing.__entity__, attr=attr,
                        outcome="unresolvable", why=str(exc)[:400]))
        print(f"    {attr:16s} UNRESOLVABLE  {str(exc)[:90]}", flush=True)
        return None, 0.0
    except Exception as exc:                  # transport, parse, anything
        # A failed cell is an outcome too, and the ledger keeps whatever
        # rounds did happen.  Losing the other 140 readings to it is not.
        log.append(dict(entity=thing.__entity__, attr=attr, outcome="error",
                        why=f"{type(exc).__name__}: {exc}"[:400]))
        print(f"    {attr:16s} ERROR  {type(exc).__name__}: "
              f"{str(exc)[:80]}", flush=True)
        return None, 0.0
    log.append(dict(entity=thing.__entity__, attr=attr, outcome=outcome,
                    value=value, p=p, seconds=round(time.time() - started, 1)))
    print(f"    {attr:16s} {str(value)[:56]:56s} p={p:.2f}"
          f"  {time.time() - started:5.1f}s", flush=True)
    return value, p


def already(ledger, entity, attrs):
    """True when every one of these cells already carries an opinion."""
    return all(ledger.opinions(entity=entity, attr=a) for a in attrs)


def measure_item(Story, item, entity, axes, ledger, log):
    if already(ledger, entity, axes):
        print(f"  {entity}: already in the ledger, skipping", flush=True)
        return
    print(f"  {entity}", flush=True)
    story = Story(__entity__=entity, text=item["text"])
    for axis in axes:
        read(story, axis, log)
    ledger.dump(LEDGER_PATH)


def main():
    os.environ.setdefault("THINAIR_BASE_URL", "http://127.0.0.1:8000/v1")
    base_url = os.environ["THINAIR_BASE_URL"]

    # A vetoed cell re-asks with the objections appended, and a long objection
    # prompt can send this server into a very long reasoning trace.  One
    # generous attempt beats three patient ones: a hung call should cost the
    # cell, not the run.
    timeout = float(os.environ.get("THINAIR_TIMEOUT", "420"))
    flash_engine = CountingEngine(base_url=base_url, model=FLASH,
                                  definition=None, timeout=timeout)
    pro_engine = CountingEngine(base_url=base_url, model=PRO,
                                definition=None, timeout=timeout)
    flash_engine.transport_retries = 1
    pro_engine.transport_retries = 1
    flash = model(FLASH, engine=flash_engine)
    pro = model(PRO, engine=pro_engine)
    # The transport needs the folder's def for quirks and request shaping; the
    # belief resolved it already, so hand the same object to the engine.
    flash_engine.definition = flash.definition
    pro_engine.definition = pro.definition

    ledger = Ledger.load(LEDGER_PATH) if os.path.exists(LEDGER_PATH) else Ledger()
    print(f"ledger: {len(ledger)} opinions at start", flush=True)

    Story = strategy.story_class(flash)
    StoryPro = strategy.story_class(pro, name="StoryPro")
    Ground = strategy.ground_class()
    Corpus = strategy.corpus_class([i["id"] for i in corpus.ITEMS], pro)
    Pair = strategy.pair_class(flash)

    log, tiers = [], {}
    started = time.time()
    # The pre-registration stamp: what the strategy, the corpus and the code
    # said *before* the first call.  Editing any of them after the fact
    # changes this hash and the change is visible.
    stamp = {name: _digest(os.path.join(HERE, name))
             for name in ("STRATEGY.md", "corpus.py", "strategy.py", "run.py")}
    print("pre-registration: " +
          ", ".join(f"{k} {v}" for k, v in stamp.items()), flush=True)

    with use_ledger(ledger):
        # -- phase 0: the free structure ----------------------------------
        pre = prepass.run(corpus.ITEMS)
        with open(PREPASS_PATH, "w") as fh:
            json.dump(pre, fh, indent=2)
        print(f"pre-pass: 0 calls, duplicates {[r['pair'] for r in pre['near_duplicates']]}, "
              f"dateline faults {pre['dateline_faults']}", flush=True)

        # -- the ground, frozen -------------------------------------------
        for item in corpus.ITEMS:
            Ground(__entity__=f"{item['id']}!ground", **item["ground"])
        ledger.dump(LEDGER_PATH)
        print(f"ground: {len(corpus.ITEMS)} entities frozen, 0 calls", flush=True)

        # -- phase 1: the reliability sample, first pass -------------------
        mark = flash_engine.snapshot()
        print("\nphase 1 -- reliability sample, declared axis order", flush=True)
        for name in RELIABILITY_SAMPLE:
            measure_item(Story, corpus.BY_ID[name], name, strategy.AXES, ledger, log)

        # -- phase 2: the same items, reversed axis order, twin entities ---
        print("\nphase 2 -- reliability sub-pass, reversed axis order", flush=True)
        for name in RELIABILITY_SAMPLE:
            measure_item(Story, corpus.BY_ID[name], f"{name}~twin",
                         tuple(reversed(strategy.AXES)), ledger, log)
        tiers["reliability"] = _delta(mark, flash_engine.snapshot())

        # -- phase 3: the rest of the base measurement ---------------------
        mark = flash_engine.snapshot()
        print("\nphase 3 -- base measurement, remaining items", flush=True)
        for item in corpus.ITEMS:
            if item["id"] in RELIABILITY_SAMPLE:
                continue
            measure_item(Story, item, item["id"], strategy.AXES, ledger, log)
        tiers["base"] = _delta(mark, flash_engine.snapshot())

        # -- phase 4: ordering path A, absolute, cold, pro -----------------
        mark = pro_engine.snapshot()
        print("\nphase 4 -- ordering path A: absolute follow_up, pro, cold", flush=True)
        for item in corpus.ITEMS:
            measure_item(StoryPro, item, f"{item['id']}#pro", ("follow_up",),
                         ledger, log)

        # -- phase 5: ordering path B, listwise, one call ------------------
        print("\nphase 5 -- ordering path B: listwise permutation, one call", flush=True)
        briefs = "\n\n".join(f"[{i['id']}] {i['text']}" for i in corpus.ITEMS)
        if not already(ledger, "corpus", ("order",)):
            whole = Corpus(__entity__="corpus", briefs=briefs)
            read(whole, "order", log)
            ledger.dump(LEDGER_PATH)
        else:
            print("  corpus: already in the ledger, skipping", flush=True)
        tiers["ordering"] = _delta(mark, pro_engine.snapshot())

        # -- phase 6: pairwise comparisons, the transitivity law -----------
        mark = flash_engine.snapshot()
        print("\nphase 6 -- pairwise comparisons", flush=True)
        pairs = []
        for a, b, c in TRIPLES:
            pairs += [(a, b), (b, c), (a, c)]
        pairs += list(ORDER_PROBE)
        for left, right in pairs:
            entity = f"pair:{left}>{right}"
            if already(ledger, entity, ("winner",)):
                print(f"  {entity}: already in the ledger, skipping", flush=True)
                continue
            print(f"  {entity}", flush=True)
            pair = Pair(__entity__=entity,
                        left_id=left, left=corpus.BY_ID[left]["text"],
                        right_id=right, right=corpus.BY_ID[right]["text"])
            read(pair, "winner", log)
            ledger.dump(LEDGER_PATH)
        tiers["transitivity"] = _delta(mark, flash_engine.snapshot())

    ledger.dump(LEDGER_PATH)
    budget = dict(preregistration=stamp, tiers=tiers,
                  findings=_sum(tiers, ("base", "ordering")),
                  validation=_sum(tiers, ("reliability", "transitivity")),
                  flash=flash_engine.snapshot(), pro=pro_engine.snapshot(),
                  wall_seconds=round(time.time() - started, 1))
    with open(RUNLOG_PATH, "w") as fh:
        json.dump(dict(budget=budget, readings=log), fh, indent=2)
    print(f"\nledger: {len(ledger)} opinions -> {LEDGER_PATH}")
    print(json.dumps(budget, indent=2))


def _digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def _sum(tiers, keys):
    out = {}
    for key in keys:
        for field, value in tiers.get(key, {}).items():
            out[field] = round(out.get(field, 0) + value, 1)
    return out


if __name__ == "__main__":
    main()
