"""The object surface: interception, resolution, freezing, operators.

Every guarantee appears here as a named test.  Nothing in this
file touches a network: the generative belief is always a ``ModelBelief``
wrapping a scripted ``FakeEngine``, which is also how engine calls become
countable.
"""

from __future__ import annotations

import pytest

from thinair.beliefs import Belief, model, human
from thinair.ledger import Ledger, Opinion
from thinair.policy import (
    Disagreement,
    LowConfidence,
    Proposed,
    Threshold,
    Unanimous,
    Unresolvable,
)
from thinair.thing import Cell, Snapshot, Thing, contract, freeze, snapshot
from thinair.validators import Range, Schema, TokenSubset

from fakes import FakeEngine

SOURCE = "INVOICE\nACME Oy\nWidget 999.00\nShipping 250.50\nTotal 1249.50 EUR"


def invoice_class(engine, *, escalation=None, extra=()):
    """A fresh Invoice class per test -- belief lists are class state."""
    panel = [model("small-fast", engine=engine)]
    if escalation is not None:
        panel.append(model("large-think", think=True, engine=escalation))
    panel += [human("jane"), *extra]

    class Invoice(Thing):
        """An invoice document to be understood."""

        __beliefs__ = panel
        source_text: str
        total = contract(float, extracted_from="source_text", range=(0, 1e6))
        vendor = contract(str, extracted_from="source_text")

    return Invoice


def make(script, **kwargs):
    """An invoice, its engine, and its ledger."""
    engine = FakeEngine(script)
    ledger = Ledger()
    cls = invoice_class(engine, **kwargs)
    return cls(source_text=SOURCE, __ledger__=ledger), engine, ledger


# --------------------------------------------------------------------------
# the demo half of examples/invoice.py
# --------------------------------------------------------------------------

def test_a_read_returns_a_value_and_an_honest_probability():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    total = inv.total
    assert (+total, ~total) == (1249.5, 0.93)
    assert engine.call_count == 1


def test_tilde_equals_extracting_from_the_resolving_opinion():
    """Spelled as the identity it names."""
    inv, _, ledger = make([{"value": 1249.5, "p": 0.93}])
    total = inv.total
    resolving = total.__opinion__
    assert (+total, ~total) == (+resolving, ~resolving)
    assert resolving in ledger.opinions(entity=inv.__entity__, attr="total")


def test_a_thing_never_owns_a_value():
    """A cell is an address; values live only in opinions."""
    inv, _, _ = make([{"value": 1249.5, "p": 0.93}])
    total = inv.total
    assert isinstance(total, Cell)
    assert total.__cell__ == (inv.__entity__, "total")
    assert "total" not in vars(inv)


def test_reads_are_cached_because_the_recorded_opinions_are_the_cache():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    assert +inv.total == +inv.total == +inv.total
    assert engine.call_count == 1


def test_the_call_form_forces_re_derivation():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}, {"value": 1249.5, "p": 0.8}])
    assert ~inv.total == 0.93
    assert ~inv.total() == 0.8
    assert engine.call_count == 2


# --------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------

def test_assignment_freezes_at_p_one_authored_by_the_human():
    inv, _, ledger = make([{"value": 1249.5, "p": 0.93}])
    inv.vendor = "ACME Oy"
    vendor = inv.vendor
    assert (+vendor, ~vendor) == ("ACME Oy", 1.0)
    assert vendor.__opinion__.belief == "human:jane"
    assert vendor.__opinion__.frozen is True


def test_constructor_kwargs_are_assignments():
    inv, _, _ = make([{"value": 1, "p": 1}])
    assert (+inv.source_text, ~inv.source_text) == (SOURCE, 1.0)
    assert inv.source_text.__opinion__.frozen is True


def test_freeze_pins_a_model_opinion_keeping_author_and_p():
    inv, _, _ = make([{"value": 1249.5, "p": 0.93}])
    pinned = freeze(inv.total)
    assert (+pinned, ~pinned) == (1249.5, 0.93)     # an honest 0.93, still
    assert pinned.__opinion__.belief.startswith("model:small-fast")
    assert pinned.__opinion__.frozen is True


def test_freeze_accepts_both_spellings():
    inv, _, _ = make([{"value": 1249.5, "p": 0.93}])
    assert ~freeze(inv, "total") == 0.93


def test_a_frozen_cell_makes_zero_engine_calls():
    """Frozen short-circuit: no consultation, no model call, ever."""
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    freeze(inv.total)
    before = engine.call_count
    for _ in range(5):
        assert +inv.total == 1249.5
    assert engine.call_count == before


def test_latest_frozen_wins_and_every_predecessor_survives():
    inv, _, ledger = make([{"value": 1249.5, "p": 0.93}])
    inv.total = 1.0
    inv.total = 2.0
    assert +inv.total == 2.0
    frozen = ledger.opinions(entity=inv.__entity__, attr="total", frozen=True)
    assert [o.value for o in frozen] == [1.0, 2.0]


def test_freezing_bypasses_the_veto_gate_deliberately():
    """Authority outranks contracts; validators gate only proposals."""
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    inv.total = -5                                   # against Range(0, 1e6)
    assert +inv.total == -5
    assert engine.call_count == 0


def test_there_is_no_deletion():
    inv, _, _ = make([{"value": 1, "p": 1}])
    inv.vendor = "ACME Oy"
    with pytest.raises(AttributeError, match="append-only"):
        del inv.vendor


# --------------------------------------------------------------------------
# the veto gate (invariant 5)
# --------------------------------------------------------------------------

def test_a_high_confidence_proposal_failing_a_necessary_check_is_re_proposed():
    """Invariant 5 as a test: never resolved, always re-proposed."""
    inv, engine, ledger = make([
        {"value": 9_999_999.0, "p": 0.99},           # outside Range(0, 1e6)
        {"value": 1249.5, "p": 0.7},
    ])
    total = inv.total
    assert (+total, ~total) == (1249.5, 0.7)
    assert engine.call_count == 2
    rejected = [o for o in ledger.opinions(attr="total") if o.value == 9_999_999.0]
    assert rejected                                  # recorded, never resolved


def test_the_objection_reaches_the_next_round_verbatim():
    """Feedback is not a mechanism; it is rounds accumulating."""
    inv, engine, _ = make([
        {"value": 9_999_999.0, "p": 0.99},
        {"value": 1249.5, "p": 0.7},
    ])
    +inv.total
    second_prompt = "\n".join(m["content"] for m in engine.calls[1]["messages"])
    assert "9999999" in second_prompt.replace(".0", "")
    assert "above the declared maximum" in second_prompt


def test_the_budget_runs_out_and_says_what_was_tried():
    inv, engine, _ = make([{"value": 9_999_999.0, "p": 0.99}])
    with pytest.raises(Unresolvable) as caught:
        +inv.total
    assert engine.call_count == 3                    # ROUNDS_PER_ROUTE
    assert len(caught.value.attempts) == 3
    assert "above the declared maximum" in str(caught.value)


def test_appending_a_validator_changes_no_resolved_value_or_p():
    """Invariant 5 as a test: verdicts are measurement."""
    plain, engine_a, _ = make([{"value": 1249.5, "p": 0.62}])
    loaded, engine_b, ledger = make([{"value": 1249.5, "p": 0.62}],
                                    extra=[TokenSubset("source_text"),
                                           Schema(float), Range(0, 1e6)])
    assert (+plain.total, ~plain.total) == (+loaded.total, ~loaded.total)
    assert engine_a.call_count == engine_b.call_count == 1
    # ...and the extra voices are all in the ledger, all agreeing, all ignored
    assert len(ledger.opinions(attr="total")) > 4


def test_permuting_discriminative_members_changes_no_opinion():
    """List position carries no visibility semantics."""
    checks = [TokenSubset("source_text"), Schema(float), Range(0, 1e6)]

    def run(order):
        inv, engine, ledger = make([{"value": 1249.5, "p": 0.71}], extra=order)
        total = inv.total
        return (+total, ~total), sorted(
            (o.belief, o.value, o.p) for o in ledger.opinions(attr="total"))

    forward = run(checks)
    backward = run(list(reversed(checks)))
    assert forward == backward


# --------------------------------------------------------------------------
# escalation — proven again in test_routing.py
# --------------------------------------------------------------------------

def test_escalation_fires_when_the_veto_gate_exhausts_its_budget():
    fast = FakeEngine([{"value": 9_999_999.0, "p": 0.99}])
    strong = FakeEngine([{"value": 1249.5, "p": 0.88}])
    ledger = Ledger()
    inv = invoice_class(fast, escalation=strong)(source_text=SOURCE,
                                                 __ledger__=ledger)
    total = inv.total
    assert (+total, ~total) == (1249.5, 0.88)
    assert fast.call_count == 3 and strong.call_count == 1
    assert total.__opinion__.belief.startswith("model:large-think")


def test_a_thing_with_no_generative_belief_serves_frozen_cells_and_nothing_else():
    class Ledgered(Thing):
        __beliefs__ = [human("jane")]
        note = contract(str)

    thing = Ledgered(note="written by hand", __ledger__=Ledger())
    assert +thing.note == "written by hand"

    other = Ledgered(__ledger__=Ledger())
    with pytest.raises(Unresolvable, match="no generative belief"):
        +other.note


# --------------------------------------------------------------------------
# snapshots are inert
# --------------------------------------------------------------------------

def test_reading_any_cell_of_a_sealed_snapshot_makes_zero_engine_calls():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    e = snapshot(inv)
    for attr in ("total", "vendor", "source_text", "anything_at_all"):
        getattr(e, attr)
    assert engine.call_count == 0


def test_an_unresolved_cell_of_a_snapshot_reads_as_absent():
    inv, _, _ = make([{"value": 1249.5, "p": 0.93}])
    e = snapshot(inv)
    assert e.total is None
    assert isinstance(e.source_text, Opinion)


def test_a_snapshot_is_sealed():
    inv, _, _ = make([{"value": 1, "p": 1}])
    e = snapshot(inv)
    with pytest.raises(AttributeError, match="sealed"):
        e.total = 1
    with pytest.raises(AttributeError):
        del e.total


def test_a_re_derivations_e0_is_byte_identical_to_the_first_ones():
    """Statelessness: a fresh derivation's inputs match the first one's."""
    seen = []

    class Recorder(Belief):
        proposes = True

        def __call__(self, e, attr):
            seen.append(e.__fingerprint__())
            return (1249.5, 0.93)

    class Invoice(Thing):
        __beliefs__ = [Recorder(id="recorder:1"), human("jane")]
        source_text: str
        total = contract(float)

    inv = Invoice(source_text=SOURCE, __ledger__=Ledger())
    +inv.total
    +inv.total()                                     # forced re-derivation
    assert len(seen) == 2 and seen[0] == seen[1]


def test_surveying_the_panel_by_hand_records_nothing():
    """Inspection never mutates memory."""
    inv, _, ledger = make([{"value": 1249.5, "p": 0.93}])
    before = len(ledger)
    e = snapshot(inv)
    survey = {b.id: b(e, "total") for b in e.__beliefs__}
    assert len(survey) == len(inv.__beliefs__)
    assert len(ledger) == before


def test_a_manual_survey_costs_one_evaluation_per_belief():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    e = snapshot(inv)
    for _ in range(3):
        {b.id: b(e, "total") for b in e.__beliefs__}
    assert engine.call_count == 1


# --------------------------------------------------------------------------
# consultation is pull-based
# --------------------------------------------------------------------------

def test_the_proposer_is_evaluated_once_per_round_however_many_validators_pull():
    checks = [TokenSubset("source_text"), Schema(float), Range(0, 1e6),
              Schema(float), Range(0, 2e6)]
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}], extra=checks)
    +inv.total
    assert engine.call_count == 1


def test_a_consultation_cycle_resolves_to_none_rather_than_erroring():
    class Ouroboros(Belief):
        necessary = False

        def __call__(self, e, attr):
            return e.__beliefs__[1](e, attr)          # asks the belief that asks it

    class Mirror(Belief):
        def __call__(self, e, attr):
            got = e.__beliefs__[1](e, attr)
            return got

    inv, engine, ledger = make([{"value": 1249.5, "p": 0.93}],
                               extra=[Ouroboros(id="cycle:a")])
    assert +inv.total == 1249.5
    assert not [o for o in ledger.opinions(belief="cycle:a")]


def test_an_external_opinion_appears_in_queries_and_changes_nothing_else():
    """External opinions need no verb: they are ledger.add."""
    inv, engine, ledger = make([{"value": 1249.5, "p": 0.93}])
    ledger.add(Opinion(belief="human:bob", entity=inv.__entity__, attr="total",
                       value=99.0, p=0.4))
    total = inv.total
    assert (+total, ~total) == (1249.5, 0.93)
    assert 99.0 in [o.value for o in ledger.opinions(attr="total")]
    assert engine.call_count == 1


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def test_extractors_never_trigger_inference():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    total = inv.total
    calls = engine.call_count
    for _ in range(5):
        +total, ~total
    assert engine.call_count == calls


def test_comparisons_delegate_to_the_carried_value():
    inv, _, _ = make([{"value": 1249.5, "p": 0.93}])
    total = inv.total
    assert total == 1249.5 and total > 1000 and total < 2000
    assert total != 99.0


def test_numeric_and_container_protocols_delegate():
    inv, _, _ = make([{"value": 1249.5, "p": 0.93},
                      {"value": ["a", "b"], "p": 0.8}])
    total = inv.total
    assert float(total) == 1249.5
    assert total + 0.5 == 1250.0 and 0.5 + total == 1250.0
    assert round(total) == 1250 and abs(-total) == 1249.5
    assert f"{total:.2f}" == "1249.50"

    items = inv.line_items
    assert len(items) == 2 and "a" in items and items[0] == "a"
    assert list(items) == ["a", "b"]


def test_a_frozen_human_value_extracts_at_p_one():
    inv, _, _ = make([{"value": 1, "p": 1}])
    inv.vendor = "ACME Oy"
    assert ~inv.vendor == 1.0


def loose(script):
    """An Invoice whose ``note`` is declared but unconstrained."""
    engine = FakeEngine(script)

    class Note(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        note = contract()

    return Note(__ledger__=Ledger()), engine


COERCE_ROWS = [
    # (scripted value, schema, expected +, expected ~)
    (1249.5, float, 1249.5, 0.93),                   # already conforms: zero work
    (1249, float, 1249.0, 0.93),                     # a cheap conversion
    (1249.5, str, "1249.5", 0.93),                   # another cheap conversion
    ("ACME Oy", str, "ACME Oy", 0.93),
]


@pytest.mark.parametrize("value,schema,plus,tilde", COERCE_ROWS)
def test_the_coercion_fast_path(value, schema, plus, tilde):
    note, engine = loose([{"value": value, "p": 0.93}])
    coerced = note.note @ schema
    assert (+coerced, ~coerced) == (plus, tilde)
    assert engine.call_count == 1                    # conforming costs nothing


def test_coercion_that_cannot_conform_returns_a_none_carrying_diagnostic():
    """``@`` never raises on epistemic failure."""
    note, engine = loose([{"value": "not a number at all", "p": 0.4}])
    coerced = note.note @ float
    assert +coerced is None
    assert ~coerced > 0.0                            # the diagnostic p survives


def test_coercion_is_cached_per_cell_and_schema():
    note, engine = loose([{"value": 1249, "p": 0.93}])
    first = note.note @ float
    calls = engine.call_count
    second = note.note @ float
    assert +second == +first and engine.call_count == calls


@pytest.mark.parametrize("p,gate,expected", [
    (0.93, 0.9, 1249.5),
    (0.93, 0.93, 1249.5),                            # inclusive
    (0.93, 0.99, None),
])
def test_the_confidence_gate(p, gate, expected):
    inv, _, _ = make([{"value": 1249.5, "p": p}])
    assert +(inv.total @ gate) is expected or +(inv.total @ gate) == expected


def test_the_gate_is_pure_and_keeps_the_p_for_diagnosis():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.4}])
    gated = inv.total @ 0.9
    assert +gated is None and ~gated == 0.4
    assert engine.call_count == 1                    # gating derives nothing


def test_the_composable_exit_idiom():
    """``total = +(inv.total @ float @ 0.9)`` -- a float or None."""
    inv, _, _ = make([{"value": 1249.5, "p": 0.93}])
    assert +(inv.total @ float @ 0.9) == 1249.5

    weak, _, _ = make([{"value": 1249.5, "p": 0.5}])
    assert +(weak.total @ float @ 0.9) is None


def test_recast_swaps_the_belief_list_and_invalidates_caches():
    inv, engine, ledger = make([{"value": 1249.5, "p": 0.93}])
    +inv.total

    other_engine = FakeEngine([{"value": 1111.0, "p": 0.5}])

    class Receipt(Thing):
        """A receipt, which is an invoice seen differently."""

        __beliefs__ = [model("other", engine=other_engine), human("jane")]
        total = contract(float, range=(0, 10))

    recast = inv @ Receipt
    assert isinstance(recast, Receipt)
    assert recast.__entity__ == inv.__entity__       # entity carries over
    assert +recast.source_text == SOURCE             # frozen state carries over
    assert recast.__resolved__ == {}                 # cached resolutions invalidate
    assert any(b.id.startswith("model:other") for b in recast.__beliefs__)


def test_recast_lets_the_new_classes_contracts_govern():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])

    tight = FakeEngine([{"value": 5.0, "p": 0.6}])

    class Small(Thing):
        __beliefs__ = [model("tight", engine=tight), human("jane")]
        total = contract(float, range=(0, 10))

    recast = inv @ Small
    assert +recast.total == 5.0                      # re-derived under Small


# --------------------------------------------------------------------------
# the four reserved words
# --------------------------------------------------------------------------

def test_require_raises_below_the_bar():
    inv, _, _ = make([{"value": 1249.5, "p": 0.4}])
    with Thing.require(0.9):
        with pytest.raises(LowConfidence):
            +inv.total


def test_require_escalates_before_it_raises():
    """A resolved p below an active require walks the ladder."""
    fast = FakeEngine([{"value": 1249.5, "p": 0.4}])
    strong = FakeEngine([{"value": 1249.5, "p": 0.95}])
    inv = invoice_class(fast, escalation=strong)(source_text=SOURCE,
                                                 __ledger__=Ledger())
    with Thing.require(0.9):
        total = inv.total
    assert ~total == 0.95 and strong.call_count == 1


def test_policy_selects_who_resolves():
    inv, _, _ = make([{"value": 1249.5, "p": 0.4}])
    with Thing.policy(Threshold(0.9)):
        total = inv.total
    assert +total is None and ~total == 0.4


def test_unanimous_consults_every_generative_member():
    agree_a = FakeEngine([{"value": 1249.5, "p": 0.9}])
    agree_b = FakeEngine([{"value": 1249.5, "p": 0.7}])
    inv = invoice_class(agree_a, escalation=agree_b)(source_text=SOURCE,
                                                     __ledger__=Ledger())
    with Thing.policy(Unanimous()):
        total = inv.total
    assert (+total, ~total) == (1249.5, 0.7)
    assert agree_a.call_count == 1 and agree_b.call_count == 1


def test_unanimous_raises_when_they_disagree():
    a = FakeEngine([{"value": 1249.5, "p": 0.9}])
    b = FakeEngine([{"value": 99.0, "p": 0.9}])
    inv = invoice_class(a, escalation=b)(source_text=SOURCE, __ledger__=Ledger())
    with Thing.policy(Unanimous()):
        with pytest.raises(Disagreement):
            +inv.total


def test_the_policy_stack_unwinds():
    assert isinstance(_active(), Proposed)
    with Thing.policy(Threshold(0.9)):
        assert isinstance(_active(), Threshold)
    assert isinstance(_active(), Proposed)


def _active():
    from thinair.thing import active_policy

    return active_policy()


def test_defaults_injects_a_ledger_for_everything_built_inside_it():
    scoped = Ledger()
    engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    cls = invoice_class(engine)
    with Thing.defaults(ledger=scoped):
        inv = cls(source_text=SOURCE)
        +inv.total
    assert len(scoped) > 0
    assert inv.__ledger__ is scoped


def test_defaults_models_prepends_generative_beliefs():
    engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    other = FakeEngine([{"value": 999.0, "p": 0.99}])
    cls = invoice_class(engine)
    preferred = model("preferred", engine=other)
    with cls.defaults(models=[preferred], ledger=Ledger()):
        assert cls.__beliefs__[0] is preferred
        inv = cls(source_text=SOURCE)
        assert +inv.total == 999.0                    # the prepended model wins
        assert engine.call_count == 0
    assert cls.__beliefs__[0] is not preferred        # restored on exit


# --------------------------------------------------------------------------
# the dunder surface
# --------------------------------------------------------------------------

def test_instance_method_count_is_zero():
    """The verbs live in the module; the class has none."""
    public = [name for name in vars(Thing)
              if not name.startswith("_") and callable(vars(Thing)[name])]
    assert public == []


def test_the_reserved_words_are_exactly_four():
    reserved = {name for name, value in vars(Thing).items()
                if not name.startswith("_") and isinstance(value, classmethod)}
    assert reserved == {"require", "policy", "debug", "defaults"}


def test_source_renders_frozen_plain_and_believed_annotated():
    inv, _, _ = make([{"value": 1249.5, "p": 0.93}])
    +inv.total
    inv.vendor = "ACME Oy"
    text = inv.__source__
    assert "vendor = 'ACME Oy'" in text
    assert "total = 1249.50  # p=0.93 ← model/extract-v3" in text


def test_source_costs_nothing():
    inv, engine, _ = make([{"value": 1249.5, "p": 0.93}])
    inv.__source__
    assert engine.call_count == 0


def test_every_plain_attribute_name_belongs_to_the_users_domain():
    """The premise of interception.

    Four names are reserved -- ``require``, ``policy``, ``debug``,
    ``defaults`` -- and the plan says they grow never.  Every other plain
    name, however framework-flavoured it sounds, is the user's.
    """
    inv, _, _ = make([{"value": "whatever", "p": 0.5}])
    for name in ("value", "ledger", "beliefs", "source", "contracts", "freeze",
                 "entity", "p", "opinion", "attrs", "state", "snapshot",
                 "id", "meta", "frozen", "belief", "round", "route"):
        assert isinstance(getattr(inv, name), Cell), name


def test_a_domain_attribute_may_shadow_nothing_the_framework_needs():
    """A Thing with hostile attribute names still resolves."""
    engine = FakeEngine([{"value": "fine", "p": 0.9}])

    class Awkward(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        value = contract(str)
        ledger = contract(str)
        beliefs = contract(str)

    thing = Awkward(ledger="the accounting kind", __ledger__=Ledger())
    assert +thing.ledger == "the accounting kind"
    assert +thing.value == "fine"
    assert isinstance(thing.__ledger__, Ledger)


def test_private_names_are_not_intercepted():
    inv, _, _ = make([{"value": 1, "p": 1}])
    with pytest.raises(AttributeError):
        inv._not_a_domain_name
    with pytest.raises(AttributeError):
        inv.__not_a_domain_name__
