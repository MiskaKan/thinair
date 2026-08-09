"""Replay, idempotent assignment, corroborate, Consulted, the belief table.

The stateless-replay contract: run the same program against the same record
and every read binds to the ledger at zero model calls, until the first cell
whose rendered context no longer matches its exposure stamp -- execution
goes live from exactly there.  Corroboration thickens the record for
settlement without touching any resolution.
"""

from __future__ import annotations

import pytest

from fakes import FakeEngine
from thinair import Consulted, Thing, contract, corroborate, human, model
from thinair.beliefs import Discriminative
from thinair.ledger import Ledger, Opinion
from thinair.policy import Unresolvable
from thinair.store import SqliteLedger

SOURCE = "Widget 999.00\nShipping 250.50\nTotal 1249.50 EUR"


def invoice_class(engine, extra=()):
    class Invoice(Thing):
        """An invoice document to be understood."""
        __beliefs__ = [model("small-fast", engine=engine), human("jane"),
                       *extra]
        source_text: str
        total = contract(float, extracted_from="source_text")
    return Invoice


# --------------------------------------------------------------------------
# idempotent assignment
# --------------------------------------------------------------------------

def test_restating_the_same_assignment_records_nothing():
    ledger = Ledger()
    inv = invoice_class(FakeEngine())(
        __entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    before = len(ledger)
    inv.source_text = SOURCE                 # the replayed program re-states
    assert len(ledger) == before
    inv.source_text = "Total 8.00 EUR"       # a real change still records
    assert len(ledger) == before + 1


# --------------------------------------------------------------------------
# replay: the recorded negotiation serves the re-run
# --------------------------------------------------------------------------

def test_the_same_program_against_the_same_record_spends_nothing():
    ledger = Ledger()
    first_engine = FakeEngine([{"value": 1249.5, "p": 0.93}])
    Invoice = invoice_class(first_engine)
    run1 = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    assert +run1.total == 1249.5
    assert first_engine.call_count == 1

    # "launch the app again": fresh process state, same record
    second_engine = FakeEngine([{"value": 777.0, "p": 0.5}])   # never asked
    Rerun = invoice_class(second_engine)
    run2 = Rerun(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    got = run2.total
    assert (+got, ~got) == (1249.5, 0.93)
    assert second_engine.call_count == 0


def test_execution_goes_live_from_the_cell_that_diverged():
    ledger = Ledger()
    Invoice = invoice_class(FakeEngine([{"value": 1249.5, "p": 0.93}]))
    +Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE).total

    diverged = FakeEngine([{"value": 8.0, "p": 0.9}])
    Rerun = invoice_class(diverged)
    run2 = Rerun(__entity__="inv-1", __ledger__=ledger,
                 source_text="Total 8.00 EUR")     # the world moved
    assert +run2.total == 8.0
    assert diverged.call_count == 1                # live from the divergence


def test_a_vetoed_record_is_never_replayed():
    class Never(Discriminative):
        necessary = True

        def judge(self, value, e, attr):
            return 0.0, "never good enough"

    ledger = Ledger()
    engine = FakeEngine([{"value": 1249.5, "p": 0.93}])
    Invoice = invoice_class(engine, extra=[Never(id="law:never")])
    inv = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    with pytest.raises(Unresolvable):
        +inv.total
    spent = engine.call_count

    rerun = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    with pytest.raises(Unresolvable):
        +rerun.total
    assert engine.call_count > spent               # asked live, not served


def test_replay_is_off_under_a_non_replaying_policy():
    ledger = Ledger()
    Invoice = invoice_class(FakeEngine([{"value": 1249.5, "p": 0.93}]))
    +Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE).total

    fresh = FakeEngine([{"value": 1249.5, "p": 0.8}])
    Rerun = invoice_class(fresh)
    run2 = Rerun(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    with Rerun.policy(Consulted()):
        +run2.total
    assert fresh.call_count == 1                   # being asked fresh is the point


# --------------------------------------------------------------------------
# Consulted: the stream mode, as a policy
# --------------------------------------------------------------------------

def test_consulted_records_the_spread_and_resolves_the_head():
    head_engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    other_engine = FakeEngine([{"value": 1200.0, "p": 0.6}])

    class Invoice(Thing):
        """An invoice document to be understood."""
        __beliefs__ = [model("small-fast", engine=head_engine),
                       model("qwen3-35b", engine=other_engine),
                       human("jane")]
        source_text: str
        total = contract(float, extracted_from="source_text")

    ledger = Ledger()
    inv = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    with Invoice.policy(Consulted()):
        got = inv.total
    assert (+got, ~got) == (1249.5, 0.9)           # the head's, its own p
    assert other_engine.call_count == 1            # but the spread was taken
    others = [o for o in ledger.opinions(entity="inv-1", attr="total")
              if o.belief.startswith("model:qwen3-35b")]
    assert [o.value for o in others] == [1200.0]   # divergence is data, not
                                                   # an exception


# --------------------------------------------------------------------------
# corroborate: second opinions into the same cells
# --------------------------------------------------------------------------

def corroborable():
    head_engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    other_engine = FakeEngine([{"value": 1249.5, "p": 0.7}])

    class Invoice(Thing):
        """An invoice document to be understood."""
        __beliefs__ = [model("small-fast", engine=head_engine),
                       model("qwen3-35b", engine=other_engine),
                       human("jane")]
        source_text: str
        total = contract(float, extracted_from="source_text")

    ledger = Ledger()
    inv = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    return inv, ledger, head_engine, other_engine


def test_corroborate_records_the_ladder_and_touches_nothing():
    inv, ledger, head_engine, other_engine = corroborable()
    assert +inv.total == 1249.5
    assert (head_engine.call_count, other_engine.call_count) == (1, 0)

    got = corroborate(inv, attrs=["total"])
    assert other_engine.call_count == 1
    recorded = next(iter(got["total"].values()))
    assert recorded.meta["corroboration"] and recorded.value == 1249.5
    assert +inv.total == 1249.5                    # resolution untouched
    assert head_engine.call_count == 1             # and not re-derived


def test_corroborate_is_idempotent_by_exposure():
    inv, _ledger, _head, other_engine = corroborable()
    +inv.total
    corroborate(inv, attrs=["total"])
    corroborate(inv, attrs=["total"])              # asked and answered
    assert other_engine.call_count == 1


def test_corroborate_takes_off_panel_beliefs_and_validates_attrs():
    inv, ledger, _head, _other = corroborable()
    +inv.total
    outsider_engine = FakeEngine([{"value": 1249.5, "p": 0.6}])
    outsider = model("deepseek-v4-flash", engine=outsider_engine)
    got = corroborate(inv, attrs=["total"], beliefs=[outsider])
    assert outsider_engine.call_count == 1
    assert outsider.id in got["total"]
    with pytest.raises(ValueError):
        corroborate(inv, attrs=["grand_total"])


def test_corroborate_skips_cells_that_never_resolved():
    inv, _ledger, _head, other_engine = corroborable()
    # nothing read yet: total has no opinion, source_text is frozen ground
    got = corroborate(inv, attrs=["total"])
    assert got == {} and other_engine.call_count == 0


# --------------------------------------------------------------------------
# the belief table: descriptions in the record
# --------------------------------------------------------------------------

def test_the_store_describes_every_speaking_belief(tmp_path):
    class Fussy(Discriminative):
        necessary = True
        veto_line = 0.7

        def judge(self, value, e, attr):
            return 1.0

        def describe(self):
            return "always content"

    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    Invoice = invoice_class(engine, extra=[Fussy(id="law:fussy")])
    inv = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    +inv.total

    row = ledger.belief_row("law:fussy")
    assert row["necessary"] and row["veto_line"] == 0.7
    assert row["kind"] == "Fussy" and row["description"] == "always content"
    assert ledger.belief_row(model("small-fast", engine=engine).id)["proposes"]


def test_reading_falls_back_to_stored_descriptions(tmp_path):
    """A fresh process settles the record without the strategy's classes."""
    from thinair.evaluate import reading

    ledger = SqliteLedger(tmp_path / "o.db")
    ledger.add(Opinion(belief="model:ghost", entity="e1", attr="size",
                       value=4, p=0.9, meta={"model": "ghost"}))
    ledger.add(Opinion(belief="law:ghost-judge", entity="e1", attr="size",
                       value=4, p=0.1, meta={"judged": "law:ghost-judge",
                                             "reason": "too big"}))
    # the row a previous process would have written; this one never
    # constructed the class
    ledger._db(create=True).execute(
        "INSERT INTO belief (id, kind, necessary, veto_line, proposes, description) "
        "VALUES ('law:ghost-judge', 'Ghost', 1, 0.5, 0, 'a judge')")

    got = reading(ledger, "e1", "size", belief="model:ghost")
    assert got["vetoed"] and got["vetoes"] == [("law:ghost-judge", "too big")]
