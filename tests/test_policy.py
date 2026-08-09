"""Resolution policies and the route ladder (SPEC.md §5)."""

from __future__ import annotations

import pytest

from thinair.beliefs import Belief
from thinair.ledger import Opinion
from thinair.policy import (
    ESCALATED_ROUNDS,
    ROUNDS_PER_ROUTE,
    Attempt,
    Disagreement,
    LowConfidence,
    Proposed,
    Route,
    Threshold,
    Unanimous,
    Unresolvable,
)


class Generative(Belief):
    proposes = True


def opinion(belief, value, p, attr="total"):
    return Opinion(belief=belief, entity="inv-1", attr=attr, value=value, p=p, t=1.0)


CELL = ("inv-1", "total")


# --------------------------------------------------------------------------
# Proposed — the default
# --------------------------------------------------------------------------

def test_proposed_returns_the_routed_proposers_own_p():
    """Corroboration is measurement, never a bonus (invariant 5)."""
    opinions = [
        opinion("model:a", 1249.5, 0.6),
        opinion("schemaBelief[float]", 1249.5, 1.0),
        opinion("rangeBelief[0,100000]", 1249.5, 1.0),
    ]
    assert Proposed().resolve(opinions, CELL, proposer="model:a") == (1249.5, 0.6)


def test_proposed_takes_the_latest_opinion_of_the_proposer():
    opinions = [opinion("model:a", 1.0, 0.4), opinion("model:a", 2.0, 0.8)]
    assert Proposed().resolve(opinions, CELL, proposer="model:a") == (2.0, 0.8)


def test_proposed_has_nothing_to_say_about_an_empty_cell():
    assert Proposed().resolve([], CELL, proposer="model:a") is None


def test_proposed_consults_only_the_routed_head():
    assert Proposed().consults_all_proposers is False


# --------------------------------------------------------------------------
# Unanimous
# --------------------------------------------------------------------------

def test_unanimous_accepts_agreement_at_the_weakest_p():
    """Unanimity is worth its least confident voice, not their average."""
    opinions = [opinion("model:a", 1249.5, 0.9), opinion("model:b", 1249.5, 0.7)]
    assert Unanimous().resolve(opinions, CELL) == (1249.5, 0.7)


def test_unanimous_uses_the_normalizing_comparator():
    opinions = [opinion("model:a", "ACME  Oy", 0.9), opinion("model:b", "acme oy", 0.8)]
    assert Unanimous().resolve(opinions, CELL)[1] == 0.8


def test_unanimous_raises_on_disagreement():
    opinions = [opinion("model:a", 1249.5, 0.9), opinion("model:b", 99.0, 0.9)]
    with pytest.raises(Disagreement) as caught:
        Unanimous().resolve(opinions, CELL)
    assert "model:a" in str(caught.value) and "model:b" in str(caught.value)


def test_unanimous_enforces_its_floor():
    opinions = [opinion("model:a", 1249.5, 0.4), opinion("model:b", 1249.5, 0.9)]
    with pytest.raises(LowConfidence):
        Unanimous(min_p=0.5).resolve(opinions, CELL)


def test_unanimous_makes_the_runtime_ask_everyone():
    assert Unanimous().consults_all_proposers is True


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------

def test_threshold_returns_none_below_the_bar_and_keeps_the_p():
    opinions = [opinion("model:a", 1249.5, 0.4)]
    assert Threshold(0.9).resolve(opinions, CELL, proposer="model:a") == (None, 0.4)


def test_threshold_passes_the_value_through_above_the_bar():
    opinions = [opinion("model:a", 1249.5, 0.95)]
    assert Threshold(0.9).resolve(opinions, CELL, proposer="model:a") == (1249.5, 0.95)


# --------------------------------------------------------------------------
# no policy performs evaluative math (invariant 5)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("policy", [Proposed(), Unanimous(), Threshold(0.0)])
def test_a_policy_only_ever_returns_a_p_that_some_belief_stated(policy):
    opinions = [opinion("model:a", 1249.5, 0.6), opinion("model:b", 1249.5, 0.8)]
    value, p = policy.resolve(opinions, CELL, proposer="model:a")
    assert p in {o.p for o in opinions}
    assert value in {o.value for o in opinions}


@pytest.mark.parametrize("policy", [Proposed(), Unanimous(), Threshold(0.0)])
def test_agreement_never_raises_the_probability(policy):
    """Ten validators agreeing with a 0.6 leave it a 0.6."""
    opinions = [opinion("model:a", 1249.5, 0.6)]
    opinions += [opinion(f"check:{i}", 1249.5, 1.0) for i in range(10)]
    assert policy.resolve(opinions, CELL, proposer="model:a")[1] == 0.6


# --------------------------------------------------------------------------
# Route — list-order routing with escalation
# --------------------------------------------------------------------------

def test_a_route_is_the_generative_members_in_list_order():
    a, b = Generative(id="model:a"), Generative(id="model:b")
    checker = Belief(id="check:x")
    route = Route([a, checker, b])
    assert route.members == [a, b]
    assert route.head is a and route.name == "model:a"


def test_a_panel_without_a_generative_member_is_empty():
    assert Route([Belief(id="check:y")]).empty is True


def test_the_routed_head_leads_the_panel():
    """Routing is expressed purely as list order."""
    a, b = Generative(id="model:a"), Generative(id="model:b")
    checker = Belief(id="check:z")
    route = Route([a, checker, b])
    route.escalate()
    assert [x.id for x in route.panel([a, checker, b])] == \
        ["model:b", "model:a", "check:z"]


def test_the_first_route_gets_three_rounds_and_the_next_two():
    a, b = Generative(id="model:a"), Generative(id="model:b")
    route = Route([a, b])
    assert route.budget() == ROUNDS_PER_ROUTE
    for i in range(ROUNDS_PER_ROUTE):
        assert not route.exhausted
        route.spend(Attempt(i, route.name))
    assert route.exhausted
    assert route.escalate()
    assert route.budget() == ESCALATED_ROUNDS and route.rounds_spent == 0


def test_a_route_cannot_escalate_past_the_last_generative_member():
    route = Route([Generative(id="model:only")])
    assert route.can_escalate is False and route.escalate() is False


def test_the_budget_is_runtime_owned():
    """A model can never grant itself another attempt."""
    route = Route([Generative(id="model:a")])
    for i in range(ROUNDS_PER_ROUTE):
        route.spend(Attempt(i, route.name))
    assert route.exhausted
    route.spend(Attempt(99, route.name))
    assert route.exhausted                       # spending more does not help


def test_a_route_keeps_every_attempt_for_the_diagnosis():
    route = Route([Generative(id="model:a")])
    route.spend(Attempt(1, "model:a", value=1.0, p=0.9,
                        vetoes=[("rangeBelief[0,10]", 0.0, "1.0 is above 10")]))
    assert len(route.attempts) == 1
    assert route.attempts[0].vetoed


# --------------------------------------------------------------------------
# Attempt
# --------------------------------------------------------------------------

def test_an_attempt_renders_objections_the_way_prompts_quote_them():
    attempt = Attempt(1, "model:a", value=1.0, p=0.9,
                      vetoes=[("rangeBelief[0,10]", 0.0, "1.0 is above 10")])
    (objection,) = attempt.objections()
    assert objection == {"value": 1.0, "belief": "rangeBelief[0,10]", "p": 0.0,
                         "reason": "1.0 is above 10"}


def test_an_unvetoed_attempt_says_so():
    attempt = Attempt(1, "model:a", value=1.0, p=0.9)
    assert not attempt.vetoed and "unvetoed" in str(attempt)


# --------------------------------------------------------------------------
# the exceptions carry what a reader needs
# --------------------------------------------------------------------------

def test_unresolvable_carries_the_full_attempt_history():
    attempts = [
        Attempt(1, "model:a", value=99.0, p=0.9,
                vetoes=[("rangeBelief[0,10]", 0.0, "99.0 is above 10")]),
        Attempt(2, "model:a", value=50.0, p=0.8,
                vetoes=[("rangeBelief[0,10]", 0.0, "50.0 is above 10")]),
    ]
    error = Unresolvable(CELL, attempts, "the veto budget was exhausted")
    text = str(error)
    assert "99.0" in text and "50.0" in text and "above 10" in text
    assert len(error.attempts) == 2


def test_low_confidence_names_the_bar_it_missed():
    assert "0.90" in str(LowConfidence(CELL, 0.6, 0.9))
