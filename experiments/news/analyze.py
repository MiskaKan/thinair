"""Everything the run earns, computed from the ledger and nothing else.

SPEC.md §13 rule 2: an experiment whose ledger cannot answer "which paths
agreed and how independent were they?" was wasted.  So this module reads
``ledger.json`` and ``corpus.py`` and never the runner's own log --- if a number
here cannot be recovered from the record, it does not get reported.

The generic settlement machinery --- veto-aware reading, scale-licensed
agreement, reliability, drift, discrimination, concordance, calibration,
budget tiers --- ships as ``thinair.evaluate``; what remains here is the
experiment: its axes, its weightings, its entity conventions, and the
compositions (three orders on one question, pre-registered transitivity
triples) that only this strategy asked for.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from thinair import lookup, model                                # noqa: E402
from thinair.evaluate import (                                   # noqa: E402
    agree, calibration, concordance, discrimination, drift, grounded, median,
    ranks, reading, reliability, separation, spearman, tiers, verdicts,
    wilson,
)
from thinair.ledger import Ledger, values_equal                  # noqa: E402

import corpus                                                    # noqa: E402
import strategy                                                  # noqa: E402

FLASH_ID = model("deepseek-v4-flash").id
PRO_ID = model("deepseek-v4-pro").id


def _register_beliefs():
    """Build the experiment's classes so every validator is in the registry.

    Veto detection asks the registry whether the belief that scored a
    candidate low was ``necessary``; ``thinair.evaluate`` raises on an
    unregistered judge rather than silently reading vetoes as low scores.
    """
    proposer = model("deepseek-v4-flash")
    strategy.story_class(proposer)
    strategy.corpus_class([i["id"] for i in corpus.ITEMS], proposer)
    strategy.pair_class(proposer)


_register_beliefs()

RELIABILITY_SAMPLE = ("n01", "n06", "n09", "n10")
TRIPLES = (("n01", "n04", "n11"), ("n03", "n06", "n09"),
           ("n05", "n07", "n10"), ("n02", "n08", "n12"))
RELIABILITY_FLOOR = 0.5

#: Exposed, human-owned, and printed in full.  The sign is a value choice, not
#: a measurement: `+` means "more of this axis argues for following up".
WEIGHTINGS = {
    "harm-first": {"severity": +0.5, "people_affected": +0.3,
                   "certainty": +0.1, "horizon": -0.1},
    "verifiability-first": {"certainty": +0.4, "horizon": -0.3,
                            "severity": +0.2, "people_affected": +0.1},
    "scale-first": {"people_affected": +0.5, "severity": +0.3,
                    "certainty": +0.1, "horizon": -0.1},
}
#: STRATEGY.md declared the weights but not the *direction* of `certainty`:
#: chase the well-sourced story, or chase the unconfirmed one?  Rather than
#: pick one after the fact, every weighting is run both ways and the
#: robustness split absorbs the ambiguity.
CERTAINTY_SIGNS = (+1, -1)

IDS = [item["id"] for item in corpus.ITEMS]


# --------------------------------------------------------------------------
# the experiment's own equivalences and conventions
# --------------------------------------------------------------------------

def compare(axis, a, b):
    """The strategy's agreement: scale-typed, except that a ratio_claim
    reading is a dict compared by its ``result`` alone --- two transcriptions
    of the same asserted arithmetic agree even when the expressions differ."""
    if a is None or b is None:
        return None
    if axis == "ratio_claim":
        a = a.get("result") if isinstance(a, dict) else a
        b = b.get("result") if isinstance(b, dict) else b
        return values_equal(a, b)
    return agree(a, b, strategy.SCALE[axis])


def flash_reading(ledger, entity, attr):
    return reading(ledger, entity, attr, belief=FLASH_ID)


# --------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------

def matrix(ledger):
    """12 × 7 of readings, straight from the record."""
    return {item_id: {axis: flash_reading(ledger, item_id, axis)
                      for axis in strategy.AXES}
            for item_id in IDS}


def run_reliability(ledger):
    """Two passes over four items, axis order reversed.  This caps everything."""
    return reliability(
        ledger, [(i, f"{i}~twin") for i in RELIABILITY_SAMPLE],
        strategy.AXES, strategy.SCALE, belief=FLASH_ID, compare=compare)


def run_discrimination(m):
    """A constant instrument passes every law and is worthless (Pillar III)."""
    rows = {}
    for axis in strategy.AXES:
        values = []
        for item_id in IDS:
            got = m[item_id].get(axis)
            if got is None:
                continue
            value = got["value"]
            values.append(value.get("result") if isinstance(value, dict)
                          else value)
        rows[axis] = discrimination(values)
    return rows


def run_concordance(ledger):
    """The only place this space touches reality: readings vs the frozen record."""
    return concordance(
        grounded(ledger, axes=strategy.GROUNDED, entities=IDS,
                 ground=lambda entity: f"{entity}!ground", belief=FLASH_ID),
        strategy.SCALE)


def laws(ledger, m):
    """Every deterministic verdict in the record, gathered per axis."""
    from collections import defaultdict
    rows = defaultdict(lambda: defaultdict(lambda: dict(n=0, mean_p=0.0, below=[])))
    for item_id in IDS:
        for axis in strategy.AXES:
            seen = verdicts(ledger, item_id, axis, exclude=(FLASH_ID, PRO_ID))
            for belief, judgments in seen.items():
                value, p, reason = judgments[-1]
                cell = rows[axis][belief]
                cell["n"] += 1
                cell["mean_p"] += p
                judge = lookup(belief)
                line = getattr(judge, "veto_line", 0.5) if judge else 0.5
                if p < line:
                    cell["below"].append((item_id, round(p, 2), reason))
    for axis in rows:
        for belief in rows[axis]:
            cell = rows[axis][belief]
            cell["mean_p"] = round(cell["mean_p"] / cell["n"], 3)
    return {a: dict(v) for a, v in rows.items()}


def orders(ledger, m):
    """Three orders on one question, and what their agreement is worth."""
    flash_scores = [m[i]["follow_up"]["value"] if m[i].get("follow_up") else None
                    for i in IDS]
    pro_cell = [reading(ledger, f"{i}#pro", "follow_up", belief=PRO_ID)
                for i in IDS]
    pro_scores = [c["value"] if c else None for c in pro_cell]
    listwise = reading(ledger, "corpus", "order", belief=PRO_ID)

    listwise_scores = None
    if listwise and isinstance(listwise["value"], list):
        position = {name: k for k, name in enumerate(listwise["value"])}
        # highest first -> invert so that larger means "more follow-up", which
        # is the direction the two score paths use.
        listwise_scores = [len(IDS) - position[i] if i in position else None
                           for i in IDS]

    out = {
        "flash_absolute_warm": flash_scores,
        "pro_absolute_cold": pro_scores,
        "pro_listwise": listwise["value"] if listwise else None,
        "listwise_vetoed": bool(listwise and listwise["vetoed"]),
    }
    out["spearman"] = {
        "pro_absolute vs pro_listwise": spearman(pro_scores, listwise_scores),
        "flash_absolute vs pro_absolute": spearman(flash_scores, pro_scores),
        "flash_absolute vs pro_listwise": spearman(flash_scores, listwise_scores),
    }
    # The listwise path can satisfy the permutation law and still have done
    # nothing: handing back the order it was given is a valid permutation.
    # Correlating the output against the *input* order is the discrimination
    # check for an ordering, and it has to be run before the agreement above
    # is allowed to mean anything.
    presented = [len(IDS) - k for k in range(len(IDS))]
    out["listwise_vs_input_order"] = spearman(presented, listwise_scores)
    out["listwise_is_identity"] = (
        listwise["value"] == IDS if listwise and
        isinstance(listwise["value"], list) else None)
    out["listwise_displacement"] = (
        [(name, IDS.index(name) - k) for k, name in enumerate(listwise["value"])
         if name in IDS and IDS.index(name) != k]
        if listwise and isinstance(listwise["value"], list) else None)
    return out


def comparisons(ledger):
    """Transitivity over the pre-registered triples, and the order-effect probe."""
    def winner(a, b):
        got = flash_reading(ledger, f"pair:{a}>{b}", "winner")
        return None if got is None else got["value"]

    triples, cycles = [], 0
    for a, b, c in TRIPLES:
        ab, bc, ac = winner(a, b), winner(b, c), winner(a, c)
        verdict = "incomplete"
        if None not in (ab, bc, ac):
            # a > b and b > c should imply a > c
            if ab == a and bc == b:
                verdict = "transitive" if ac == a else "cyclic"
            elif ab == b and bc == c:
                verdict = "transitive" if ac == c else "cyclic"
            else:
                verdict = "no chain"          # the pair verdicts imply nothing
            cycles += verdict == "cyclic"
        triples.append(dict(triple=(a, b, c), ab=ab, bc=bc, ac=ac,
                            verdict=verdict))

    probe = []
    for a, _b, c in TRIPLES:
        forward, backward = winner(a, c), winner(c, a)
        probe.append(dict(pair=(a, c), forward=forward, backward=backward,
                          stable=(forward is not None and forward == backward)))
    stable = [p for p in probe if p["stable"]]
    return dict(triples=triples, cycles=cycles, order_probe=probe,
                order_stability=(round(len(stable) / len(probe), 3)
                                 if probe else None))


def run_tiers(ledger):
    """The two-tier budget, recomputed from the record rather than trusted."""
    return tiers(
        ledger,
        validation=lambda o: (o.entity.endswith("~twin")
                              or o.entity.startswith("pair:")),
        proposers=(FLASH_ID, PRO_ID))


def deliverable(m, reliable):
    """Exact aggregation over ranks, under exposed weights, both directions."""
    axes = ("severity", "people_affected", "certainty", "horizon")
    usable = [a for a in axes if reliable.get(a)]

    def usable_value(item_id, axis):
        got = m[item_id].get(axis)
        if got is None or got["vetoed"]:
            return None                       # a vetoed candidate is not a reading
        return got["value"] if isinstance(got["value"], (int, float)) else None

    rankable = [i for i in IDS
                if all(usable_value(i, a) is not None for a in usable)]
    excluded = [i for i in IDS if i not in rankable]

    columns = {}
    for axis in usable:
        values = [float(usable_value(i, axis)) for i in rankable]
        # ranks, because three of these four are ordinal and a weighted sum of
        # ordinal *values* is not a licensed statistic.
        columns[axis] = [len(rankable) + 1 - r for r in ranks(values)]

    tables = {}
    for name, weights in WEIGHTINGS.items():
        for sign in CERTAINTY_SIGNS:
            label = f"{name} (certainty {'+' if sign > 0 else '-'})"
            scores = []
            for k, item_id in enumerate(rankable):
                total = 0.0
                for axis, weight in weights.items():
                    if axis not in columns:
                        continue
                    w = weight * (sign if axis == "certainty" else 1)
                    total += w * columns[axis][k]
                scores.append((round(total, 3), item_id))
            scores.sort(reverse=True)
            tables[label] = scores
    top4 = [set(i for _s, i in table[:4]) for table in tables.values() if table]
    robust = set.intersection(*top4) if top4 else set()
    movable = (set.union(*top4) - robust) if top4 else set()
    return dict(axes_used=usable, axes_dropped=[a for a in axes if a not in usable],
                rankable=rankable, excluded=excluded,
                columns=columns, tables=tables,
                robust_top4=sorted(robust), weight_sensitive=sorted(movable))


def run(path=None):
    ledger = Ledger.load(path or os.path.join(HERE, "ledger.json"))
    m = matrix(ledger)
    rel = run_reliability(ledger)
    reliable = {axis: (row["exact"] is not None and row["exact"] >= RELIABILITY_FLOOR)
                for axis, row in rel.items()}
    conc, per_reading = run_concordance(ledger)
    return dict(
        opinions=len(ledger),
        tiers=run_tiers(ledger),
        matrix=m,
        reliability=rel,
        order_effect=drift(rel, strategy.SCALE),
        reliable=reliable,
        discrimination=run_discrimination(m),
        concordance=conc,
        stated_p=calibration(per_reading),
        p_separation=separation(per_reading),
        laws=laws(ledger, m),
        orders=orders(ledger, m),
        comparisons=comparisons(ledger),
        deliverable=deliverable(m, reliable),
    )


if __name__ == "__main__":
    result = run(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps(result, indent=2, default=str))
