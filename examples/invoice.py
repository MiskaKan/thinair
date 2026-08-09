"""The acceptance demo.  Every claim here is true.

Run it against a local OpenAI-compatible endpoint::

    THINAIR_BASE_URL=http://localhost:8000/v1 THINAIR_MODEL=qwen3-35b \\
        python examples/invoice.py

or run it offline, against a scripted engine, which is what the test suite
does::

    python examples/invoice.py --offline
"""

import os
import sys

from thinair import (
    Thing,
    contract,
    fn,
    freeze,
    freeze_call,
    human,
    model,
    snapshot,
)
from thinair.validators import TokenSubsetBelief

HERE = os.path.dirname(os.path.abspath(__file__))


def build(engine=None, escalation=None):
    """The demo class.  ``engine=`` is how the offline run scripts it."""

    class Invoice(Thing):
        """An invoice document to be understood."""

        __beliefs__ = [
            model("small-fast", engine=engine),               # default proposer
            model("large-think", think=True, engine=escalation),  # escalation tier
            human("jane"),                                    # speaks through assignment
            TokenSubsetBelief("source_text"),
        ]
        source_text: str
        total = contract(float, extracted_from="source_text", range=(0, 1e6))
        vendor = contract(str, extracted_from="source_text")
        line_items = contract([{"desc": str, "amount": float}],
                              extracted_from="source_text", sums_to="total")

    return Invoice


def net_total_fn(engine=None):
    @fn(returns=float, pure=True,
        models=[model("small-fast", engine=engine)] if engine else None)
    def net_total(gross: float, vat: float) -> float:
        """Total with VAT removed."""
        ...                            # bodiless: the model serves it, contract-checked

    return net_total


def demo(Invoice, net_total, out=sys.stdout):
    def show(*args):
        print(*args, file=out)

    with open(os.path.join(HERE, "invoice.txt")) as fh:
        inv = Invoice(source_text=fh.read())          # kwarg -> frozen, human:jane

    show(+inv.total, ~inv.total)                      # the resolving opinion

    e = snapshot(inv)                                 # sealed and inert: reads never spend
    show({b.id: b(e, "total")                         # consult the panel by hand:
          for b in inv.__beliefs__})                  #   beliefs are just pure functions

    total = +(inv.total @ float @ 0.9)                # the exit ramp: a float or None
    show("exit ramp:", total)

    inv.vendor = "ACME Oy"                            # human speaks: frozen, p = 1.0
    freeze(inv.total)                                 # pin the model's answer, keeping p

    show(inv.__ledger__.opinions(attr="total"))       # every voice, every author
    show(inv.__source__)

    show(+net_total(1249.50, 0.24), ~net_total(1249.50, 0.24))
    freeze_call(net_total, (1249.50, 0.24), value=1007.66)   # a fixture IS a frozen opinion
    show(+net_total(1249.50, 0.24), ~net_total(1249.50, 0.24))

    return inv, net_total


def offline():
    """The same demo with scripted engines -- no network, fully deterministic."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
    from fakes import FakeEngine

    from thinair import Ledger
    from thinair.thing import _ledgers

    items = [{"desc": "Widget, large", "amount": 999.00},
             {"desc": "Shipping", "amount": 250.50}]
    engine = FakeEngine(lambda messages: _script(messages, items))
    _ledgers.append(Ledger())
    try:
        return demo(build(engine, FakeEngine(lambda m: _script(m, items))),
                    net_total_fn(FakeEngine([{"value": 1007.66, "p": 0.71}])))
    finally:
        _ledgers.pop()


def _script(messages, items):
    """A scripted model: answers by which cell it was asked about."""
    asked = messages[-1]["content"]
    if '"total"' in asked:
        return {"value": 1249.50, "p": 0.93}
    if '"vendor"' in asked:
        return {"value": "ACME Oy", "p": 0.88}
    if '"line_items"' in asked:
        return {"value": items, "p": 0.81}
    if '"result"' in asked:
        return {"value": 1007.66, "p": 0.71}
    return {"value": None, "p": 0.0}


if __name__ == "__main__":
    if "--offline" in sys.argv:
        offline()
    else:
        demo(build(), net_total_fn())
