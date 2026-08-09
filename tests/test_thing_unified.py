"""The unified Thing surface.

One class carries the whole of it: ``Thing(shape, ...)`` in a class body is
a declaration, a Thing assigned to an attribute is a reference, the panel
changes through operators, freezing is a property of the belief, and
``Thing.__default__`` is the fallback proposer.  There are no module verbs
for any of these moves -- the surface is the whole spelling.
"""

from __future__ import annotations

import pytest

from fakes import ScriptedBelief
from thinair import Thing, human, model
from thinair.ledger import Ledger
from thinair.policy import Unresolvable


# --------------------------------------------------------------------------
# declarations: Thing(shape, ...) in a class body
# --------------------------------------------------------------------------

def test_a_thing_with_a_shape_declares_the_attribute():
    class Ticket(Thing):
        __beliefs__ = [ScriptedBelief({"priority": ("high", 0.9)},
                                      "scripted:declares")]
        source_text: str
        priority = Thing(str, enum=["low", "normal", "high"])

    assert "priority" in Ticket.__contracts__
    spec = Ticket.__contracts__["priority"]
    assert spec.template is str and spec.enum == ["low", "normal", "high"]
    # the declaration attached its validators to the one attachment point
    assert any(getattr(b, "id", "").startswith("enumBelief[")
               for b in Ticket.__beliefs__)

    t = Ticket(__ledger__=Ledger())
    assert +t.priority == "high" and ~t.priority == 0.9


def test_declaring_touches_no_ledger_and_mints_no_entity():
    from thinair.ledger import use_ledger

    ledger = Ledger()
    with use_ledger(ledger):
        declared = Thing(str, enum=["a", "b"])
    assert len(ledger) == 0
    assert declared.__declaration__.enum == ["a", "b"]


def test_a_declaration_carries_its_full_spec():
    spec = Thing(float, extracted_from="src", range=(0, 10)).__declaration__
    assert (spec.template, spec.extracted_from, spec.range) \
        == (float, "src", (0, 10))


# --------------------------------------------------------------------------
# the panel, through operators
# --------------------------------------------------------------------------

def test_iadd_and_isub_change_the_panel_on_the_record():
    class Card(Thing):
        __beliefs__ = [ScriptedBelief(("x", 0.5), "scripted:first")]

    ledger = Ledger()
    card = Card(__ledger__=ledger, __entity__="card-1")
    second = ScriptedBelief(("y", 0.6), "scripted:second")

    card += second
    assert second in card.__beliefs__
    panels = ledger.opinions(entity="card-1", attr="__panel__")
    assert len(panels) == 2                       # baseline, then the change
    assert "scripted:second" in panels[-1].value

    card -= second
    assert second not in card.__beliefs__
    panels = ledger.opinions(entity="card-1", attr="__panel__")
    assert "scripted:second" not in panels[-1].value


def test_removing_an_absent_belief_is_loud():
    card = Thing(__ledger__=Ledger())
    with pytest.raises(ValueError):
        card -= ScriptedBelief(("x", 0.5), "scripted:absent")


def test_plus_and_minus_hand_back_a_fresh_thing_on_the_same_entity():
    class Card(Thing):
        __beliefs__ = [ScriptedBelief(("x", 0.5), "scripted:plus-base")]

    ledger = Ledger()
    card = Card(__ledger__=ledger, __entity__="card-2")
    extra = ScriptedBelief(("y", 0.6), "scripted:plus-extra")

    grown = card + extra
    assert grown is not card
    assert grown.__entity__ == card.__entity__
    assert extra in grown.__beliefs__ and extra not in card.__beliefs__

    shrunk = grown - extra
    assert extra not in shrunk.__beliefs__


def test_the_operators_reject_non_beliefs():
    card = Thing(__ledger__=Ledger())
    with pytest.raises(TypeError):
        card + 3
    with pytest.raises(TypeError):
        card += "not a belief"


# --------------------------------------------------------------------------
# freezing is a property of the belief
# --------------------------------------------------------------------------

class PinningBelief(ScriptedBelief):
    frozen = True


def test_a_frozen_belief_pins_what_it_settles():
    class Doc(Thing):
        __beliefs__ = [PinningBelief({"grade": ("A", 0.7)}, "scripted:pins")]

    ledger = Ledger()
    doc = Doc(__ledger__=ledger, __entity__="doc-1")
    assert +doc.grade == "A"
    pinned = ledger.latest_frozen("doc-1", "grade")
    assert pinned is not None
    assert pinned.p == 0.7                        # an honest 0.7 stays 0.7
    assert pinned.belief == "scripted:pins"


def test_an_unfrozen_belief_still_records_plain_opinions():
    class Doc(Thing):
        __beliefs__ = [ScriptedBelief({"grade": ("B", 0.7)},
                                      "scripted:no-pin")]

    ledger = Ledger()
    doc = Doc(__ledger__=ledger, __entity__="doc-2")
    assert +doc.grade == "B"
    assert ledger.latest_frozen("doc-2", "grade") is None


def test_code_beliefs_freeze_by_default():
    from thinair.beliefs import CodeBelief

    assert CodeBelief.frozen is True


# --------------------------------------------------------------------------
# the fallback proposer
# --------------------------------------------------------------------------

def test_default_belief_serves_a_panel_without_a_proposer():
    class Bare(Thing):
        __beliefs__ = [human("jane")]
        __default__ = ScriptedBelief(("served", 0.8), "scripted:default")

    bare = Bare(__ledger__=Ledger())
    assert +bare.answer == "served" and ~bare.answer == 0.8


def test_without_a_default_a_bare_panel_is_unresolvable():
    class Bare(Thing):
        __beliefs__ = [human("jane")]

    bare = Bare(__ledger__=Ledger())
    with pytest.raises(Unresolvable):
        +bare.answer


# --------------------------------------------------------------------------
# writing a known value is assignment -- the 026 trap has no verb to spring
# --------------------------------------------------------------------------

def test_assignment_never_consults_the_panel():
    counting = ScriptedBelief({"total": (999.0, 0.9)}, "scripted:counting")

    class Invoice(Thing):
        __beliefs__ = [counting]
        total = Thing(float)

    ledger = Ledger()
    inv = Invoice(__ledger__=ledger, __entity__="inv-9")
    inv.total = 1249.5
    assert counting.calls == []                   # nothing resolved, nothing spent
    pinned = ledger.latest_frozen("inv-9", "total")
    assert pinned.value == 1249.5 and pinned.belief.startswith("human:")
    assert +inv.total == 1249.5                   # served frozen, still zero calls
    assert counting.calls == []


# --------------------------------------------------------------------------
# nesting: a Thing crossing the boundary is a reference
# --------------------------------------------------------------------------

def test_assigning_a_thing_records_its_entity_as_a_reference():
    ledger = Ledger()
    part = Thing(__ledger__=ledger, __entity__="part-7")
    whole = Thing(__ledger__=ledger, __entity__="whole-1")
    whole.engine_block = part
    recorded = ledger.latest("whole-1", "engine_block")
    assert recorded.value == "part-7"
    assert "part-7" in (recorded.meta or {}).get("refs", [])


# --------------------------------------------------------------------------
# beliefs=[...] is the declarative spelling, fully equal to the kwargs
# --------------------------------------------------------------------------

def test_a_beliefs_list_is_first_class_declaration():
    """Import the validator, put it in the list, done: auto-scoped to the
    attribute, veto rights honored, and described to the model exactly as
    the named options would be -- `enum=` / `range=` are shorthand for
    this, never more."""
    from thinair.validators import EnumBelief, RangeBelief

    class Ticket(Thing):
        __beliefs__ = [ScriptedBelief({"amount": (20_000.0, 0.9),
                                       "priority": ("low", 0.9)},
                                      "scripted:lists")]
        amount = Thing(float, beliefs=[RangeBelief(0, 10_000)])
        priority = Thing(str, beliefs=[EnumBelief(["low", "high"])])

    # the law reaches the prompt: the model is told what will be enforced
    assert "10000" in Ticket.__contracts__["amount"].describe()
    assert "one of 'low', 'high'" in Ticket.__contracts__["priority"].describe()

    t = Ticket(__ledger__=Ledger())
    assert +t.priority == "low"                  # RangeBelief is scoped off priority
    with pytest.raises(Unresolvable):
        +t.amount                                # 20_000 vetoed, never served
