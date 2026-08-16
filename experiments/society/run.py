"""Run the society experiment (STRATEGY.md), leaving a complete ledger.

Live, against an endpoint (one model plays every mind):

    THINAIR_BASE_URL=http://127.0.0.1:8000/v1 THINAIR_MODEL=<name> \\
        python experiments/society/run.py

Offline rehearsal, scripted minds through the identical machinery:

    python experiments/society/run.py --scripted

Either way: build the cast on one shared in-memory ledger, record the
pre-registration stamp, inject exactly one frozen fact (anna's car is
broken), and let the sweep run to quiescence.  The ledger is archived whole
(§13 rule 2); inspect it with ``thinair log --graph experiments/society/ledger.json``.
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

import cast                                                       # noqa: E402
import minds                                                      # noqa: E402
from sweep import Sweep                                           # noqa: E402

LEDGER_PATH = os.path.join(HERE, "ledger.json")
RUNLOG_PATH = os.path.join(HERE, "runlog.json")

#: Pre-registered: the cast, their ids, the one injected fact, the budgets.
BUDGET = 312.0
RATES = {"mech-dave": 480.0, "mech-tom": 260.0}
DIRECTORY = {"car repair": "forum"}
MAX_SWEEPS = 15


def beliefs(scripted: bool) -> dict:
    if scripted:
        return {name: model(f"scripted-{name}",
                            engine=minds.MindEngine(minds.MINDS[name]))
                for name in minds.MINDS}
    shared = model()                    # THINAIR_MODEL / THINAIR_BASE_URL
    return {name: shared for name in minds.MINDS}


def build(ledger, minded: dict):
    """The cast, on one shared ledger.  Order is the sweep order, and it is
    pre-registered: the expensive mechanic before the cheap one, so the
    budget veto demonstrably fires before the acceptable quote arrives."""
    with use_ledger(ledger):
        Internet = cast.internet_class(minded["internet"])
        internet = Internet(__entity__="internet", directory=dict(DIRECTORY))
        Forum = cast.forum_class(minded["forum"])
        forum = Forum(__entity__="forum")
        Customer = cast.customer_class(minded["anna"])
        anna = Customer(__entity__="anna", budget=BUDGET, internet=internet)
        Mechanic = cast.mechanic_class(minded["mech-dave"])
        dave = Mechanic(__entity__="mech-dave", rate=RATES["mech-dave"],
                        forum=forum)
        MechanicToo = cast.mechanic_class(minded["mech-tom"], name="Mechanic")
        tom = MechanicToo(__entity__="mech-tom", rate=RATES["mech-tom"],
                          forum=forum)
    return [anna, internet, forum, dave, tom]


def _digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def main():
    scripted = "--scripted" in sys.argv
    stamp = {name: _digest(os.path.join(HERE, name))
             for name in ("STRATEGY.md", "cast.py", "sweep.py", "minds.py",
                          "run.py")}
    print("pre-registration: " +
          ", ".join(f"{k} {v}" for k, v in stamp.items()), flush=True)

    ledger = Ledger()
    minded = beliefs(scripted)
    society = build(ledger, minded)
    started = time.time()

    sweeper = Sweep(society)
    with use_ledger(ledger):
        society[0].car_broken = True          # THE injected fact: code, frozen
        sweeps = sweeper.run(max_sweeps=MAX_SWEEPS)

    ledger.dump(LEDGER_PATH)

    def standing(entity, attr):
        # report from the record, never by derivation: an unresolved cell is
        # an outcome, not a question to spend on
        writes = [o for o in ledger.opinions(entity=entity, attr=attr)
                  if (o.meta or {}).get("from_changeset") or o.frozen]
        return writes[-1].value if writes else None

    outcome = {
        "mode": "scripted" if scripted else "live",
        "preregistration": stamp,
        "sweeps_to_quiescence": sweeps,
        "opinions": len(ledger),
        "accepted_quote": standing("anna", "accepted_quote"),
        "mechanic": standing("anna", "mechanic"),
        "wall_seconds": round(time.time() - started, 1),
    }
    with open(RUNLOG_PATH, "w") as fh:
        json.dump(dict(outcome=outcome, turns=sweeper.log), fh, indent=2)
    print(f"ledger: {len(ledger)} opinions -> {LEDGER_PATH}")
    print(json.dumps(outcome, indent=2))
    for entry in sweeper.log:
        print(f"  sweep {entry.get('sweep')}: "
              + ", ".join(f"{k}={v}" for k, v in entry.items()
                          if k != "sweep"))


if __name__ == "__main__":
    main()
