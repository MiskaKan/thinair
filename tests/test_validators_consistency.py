"""Family C — internal consistency: several cells at once.

These judge *relations*, so they have two things to demonstrate that no other
family has: they emit opinions about **virtual attributes** in the one record
shape (invariant 1), and at a related real attribute they pull the candidate
and can veto it.
"""

from __future__ import annotations

import datetime

import pytest

from thinair.beliefs import Judgment
from thinair.ledger import Ledger, Opinion
from thinair.validators import (
    Conservation,
    FunctionalDependency,
    ItemsSumTo,
    MutuallyExclusive,
    Recompute,
    Relation,
    SumsTo,
    TemporalOrder,
)

from fakes import FakeSnapshot

ITEMS = [{"desc": "Widget", "amount": 999.0}, {"desc": "Shipping", "amount": 250.5}]


def resolved(**cells):
    """A snapshot with cells already resolved and nothing proposing."""
    return FakeSnapshot(beliefs=[lambda e, a: None], **cells)


def at_virtual(belief, **cells):
    """The relation's verdict about itself, judged from resolved state."""
    return belief(resolved(**cells), belief.virtual)


# --------------------------------------------------------------------------
# virtual attributes
# --------------------------------------------------------------------------

def test_a_relation_names_a_virtual_attribute():
    belief = SumsTo(["net", "vat"], "total")
    assert belief.virtual == "…" + belief.id
    assert belief.virtual.startswith("…sumsTo[")


def test_a_virtual_verdict_is_a_boolean_about_the_relation():
    belief = SumsTo(["net", "vat"], "total")
    good = at_virtual(belief, net=1000.0, vat=249.5, total=1249.5)
    bad = at_virtual(belief, net=1000.0, vat=249.5, total=9999.0)
    assert (+good, ~good) == (True, 1.0)
    assert (+bad, ~bad) == (False, 0.0)


def test_a_virtual_opinion_has_the_same_record_shape_as_any_other():
    """Invariant 1: no second record type, virtual or not."""
    ledger = Ledger()
    belief = SumsTo(["net", "vat"], "total")
    got = at_virtual(belief, net=1.0, vat=2.0, total=3.0)
    ledger.add(Opinion(belief=belief.id, entity="inv-1", attr=belief.virtual,
                       value=+got, p=~got, meta=dict(got.meta)))
    (recorded,) = ledger.opinions(entity="inv-1")
    assert recorded.attr == belief.virtual and recorded.value is True
    assert recorded.meta["relates"] == ["net", "vat", "total"]


def test_a_relation_is_undecided_while_a_cell_is_missing():
    assert at_virtual(SumsTo(["net", "vat"], "total"), net=1.0, total=3.0) is None


# --------------------------------------------------------------------------
# a relation can veto a proposal at a related real attribute
# --------------------------------------------------------------------------

def test_a_relation_judges_the_candidate_at_a_related_attribute():
    belief = SumsTo(["net", "vat"], "total")
    e = FakeSnapshot(beliefs=[lambda x, a: (1249.5, 0.93)],
                     net=999.0, vat=250.5)
    got = belief(e, "total")
    assert (+got, ~got) == (1249.5, 1.0)          # the candidate rides through
    assert got.meta["relates"] == ["net", "vat", "total"]


def test_a_relation_vetoes_a_candidate_that_breaks_it():
    belief = SumsTo(["net", "vat"], "total")
    e = FakeSnapshot(beliefs=[lambda x, a: (9999.0, 0.99)], net=999.0, vat=250.5)
    got = belief(e, "total")
    assert (+got, ~got) == (9999.0, 0.0)          # high confidence, still vetoed
    assert belief.necessary is True


def test_a_relation_substitutes_the_candidate_for_the_standing_resolution():
    """The candidate under judgment replaces the cell's current value."""
    belief = SumsTo(["net", "vat"], "total")
    e = FakeSnapshot(beliefs=[lambda x, a: (1249.5, 0.93)],
                     net=999.0, vat=250.5, total=9999.0)
    assert ~belief(e, "total") == 1.0


def test_a_relation_is_silent_at_an_unrelated_attribute():
    belief = SumsTo(["net", "vat"], "total")
    e = FakeSnapshot(beliefs=[lambda x, a: ("ACME Oy", 0.9)], net=1.0, vat=2.0)
    assert belief(e, "vendor") is None


def test_a_relation_is_silent_when_nothing_is_proposed():
    belief = SumsTo(["net", "vat"], "total")
    assert belief(resolved(net=1.0, vat=2.0), "total") is None


# --------------------------------------------------------------------------
# the relations themselves
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cells,expected", [
    (dict(net=999.0, vat=250.5, total=1249.5), True),
    (dict(net=999.0, vat=250.5, total=1249.500000001), True),      # tolerance
    (dict(net=999.0, vat=250.5, total=1249.6), False),
    (dict(net=0, vat=0, total=0), True),
    (dict(net="lots", vat=250.5, total=1249.5), None),             # undecidable
    (dict(net=True, vat=250.5, total=1249.5), None),               # bool is not a number
])
def test_sums_to(cells, expected):
    got = at_virtual(SumsTo(["net", "vat"], "total"), **cells)
    assert (None if got is None else +got) is expected


@pytest.mark.parametrize("cells,expected", [
    (dict(line_items=ITEMS, total=1249.5), True),
    (dict(line_items=ITEMS, total=1349.5), False),
    (dict(line_items=[], total=0.0), True),
    (dict(line_items=[{"desc": "x"}], total=0.0), False),          # field missing
    (dict(line_items="not a list", total=0.0), None),
])
def test_items_sum_to(cells, expected):
    got = at_virtual(ItemsSumTo("line_items", "amount", "total"), **cells)
    assert (None if got is None else +got) is expected


def test_items_sum_to_vetoes_a_bad_total_against_good_items():
    belief = ItemsSumTo("line_items", "amount", "total")
    e = FakeSnapshot(beliefs=[lambda x, a: (9999.0, 0.95)], line_items=ITEMS)
    assert ~belief(e, "total") == 0.0


def test_items_sum_to_vetoes_bad_items_against_a_frozen_total():
    """The relation works in both directions -- it is about cells, not roles."""
    belief = ItemsSumTo("line_items", "amount", "total")
    invented = [{"desc": "Widget", "amount": 500.0}]
    e = FakeSnapshot(beliefs=[lambda x, a: (invented, 0.9)], total=1249.5)
    assert ~belief(e, "line_items") == 0.0


def test_recompute():
    belief = Recompute("net", lambda total, vat: total / (1 + vat), ["total", "vat"])
    assert +at_virtual(belief, total=1249.5, vat=0.24, net=1007.66) is False
    assert +at_virtual(belief,
                       total=1249.5, vat=0.24, net=1249.5 / 1.24) is True


def test_recompute_survives_a_raising_function():
    belief = Recompute("net", lambda a, b: a / b, ["total", "vat"])
    got = at_virtual(belief, total=1249.5, vat=0, net=1.0)
    assert (+got, ~got) == (False, 0.0)
    assert "ZeroDivisionError" in got.meta["reason"]


def test_recompute_tolerance_is_configurable():
    belief = Recompute("net", lambda t, v: t / (1 + v), ["total", "vat"], tol=0.01)
    assert +at_virtual(belief, total=1249.5, vat=0.24, net=1007.66) is True


@pytest.mark.parametrize("a,b,expected", [
    ("2024-01-01", "2024-02-01", True),
    ("2024-02-01", "2024-01-01", False),
    ("2024-01-01", "2024-01-01", True),                       # not *after*
    ("2024-01-01T10:00:00", "2024-01-01T11:00:00", True),
    ("2024-01-01T10:00:00Z", "2024-01-01T11:00:00+00:00", True),
    (datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), True),
    (1, 2, True),
    ("whenever", "2024-01-01", False),                        # not orderable
])
def test_temporal_order(a, b, expected):
    got = at_virtual(TemporalOrder("issued", "due"), issued=a, due=b)
    assert +got is expected


@pytest.mark.parametrize("cells,expected", [
    (dict(paid=100.0, refunded=20.0, net_change=80.0), True),
    (dict(paid=100.0, refunded=20.0, net_change=100.0), False),
    (dict(paid=100.0, refunded=20.0), None),
])
def test_conservation(cells, expected):
    got = at_virtual(Conservation("paid", "refunded", "net_change"), **cells)
    assert (None if got is None else +got) is expected


@pytest.mark.parametrize("records,expected", [
    ([{"sku": "A", "price": 1.0}, {"sku": "A", "price": 1.0}], True),
    ([{"sku": "A", "price": 1.0}, {"sku": "A", "price": 2.0}], False),
    ([{"sku": "A", "price": 1.0}, {"sku": "B", "price": 2.0}], True),
    ([], True),
    ("not records", None),
])
def test_functional_dependency(records, expected):
    got = at_virtual(FunctionalDependency("sku", "price", over="line_items"),
                     line_items=records)
    assert (None if got is None else +got) is expected


@pytest.mark.parametrize("cells,expected", [
    (dict(paid=True, overdue=False), True),
    (dict(paid=True, overdue=True), False),
    (dict(paid=False, overdue=False), True),
    (dict(paid=True), None),                                  # only one is known
])
def test_mutually_exclusive(cells, expected):
    got = at_virtual(MutuallyExclusive(["paid", "overdue"]), **cells)
    assert (None if got is None else +got) is expected


# --------------------------------------------------------------------------
# family behavior
# --------------------------------------------------------------------------

FAMILY_C = [
    SumsTo(["net", "vat"], "total"),
    ItemsSumTo("line_items", "amount", "total"),
    Recompute("net", lambda t, v: t / (1 + v), ["total", "vat"]),
    TemporalOrder("issued", "due"),
    Conservation("paid", "refunded", "net_change"),
    FunctionalDependency("sku", "price", over="line_items"),
    MutuallyExclusive(["paid", "overdue"]),
]


@pytest.mark.parametrize("belief", FAMILY_C, ids=lambda b: b.id)
def test_family_c_is_necessary_by_default(belief):
    assert belief.necessary is True


@pytest.mark.parametrize("belief", FAMILY_C, ids=lambda b: b.id)
def test_every_relation_declares_the_cells_it_relates(belief):
    assert isinstance(belief.cells(), tuple)
    assert all(isinstance(c, str) for c in belief.cells())


@pytest.mark.parametrize("belief", FAMILY_C, ids=lambda b: b.id)
def test_every_relation_is_undecided_on_an_empty_entity(belief):
    assert belief(resolved(), belief.virtual) is None


def test_a_user_can_write_a_relation_in_five_lines():
    """Relations are user-level code, like every other belief."""
    class NotOverdrawn(Relation):
        def cells(self):
            return ("balance",)

        def check(self, values):
            balance = values.get("balance")
            if balance is None:
                return None
            return 1.0 if balance >= 0 else (0.0, f"balance is {balance}")

    belief = NotOverdrawn()
    assert +at_virtual(belief, balance=10) is True
    assert +at_virtual(belief, balance=-1) is False
    assert isinstance(belief(FakeSnapshot(beliefs=[lambda x, a: (5, 0.8)]),
                             "balance"), Judgment)
