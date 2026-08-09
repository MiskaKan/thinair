"""Imagined method calls: every guarantee, one per test."""

from __future__ import annotations

import pytest

from thinair import Thing, contract, freeze, human
from thinair.beliefs import model
from thinair.episode import ACTION_BUDGET, CORRECTIONS, call_expression
from thinair.ledger import Ledger
from thinair.policy import Unresolvable
from thinair.validators import NonEchoBelief

from fakes import FakeEngine

SOURCE = "Widget 999.00\nShipping 250.50\nTotal 1249.50 EUR"


def ret(value, changes=None, p=0.8):
    return {"action": "return", "changes": changes or {}, "value": value, "p": p}


def invoice(script, *, extra=(), source=SOURCE):
    engine = FakeEngine(script)

    class Invoice(Thing):
        """An invoice document to be understood."""

        __beliefs__ = [model("small-fast", engine=engine), human("jane"), *extra]
        source_text: str
        total = contract(float, extracted_from="source_text", range=(0, 1e6))
        vat = contract(float, range=(0, 1))
        status = contract(str, enum=["paid", "pending", "overdue"])
        note = contract(str)

        def recheck(self) -> bool:
            """A real method: real code, and its writes freeze."""
            return True

    ledger = Ledger()
    return Invoice(source_text=source, __ledger__=ledger), engine, ledger


# --------------------------------------------------------------------------
# the seam: what a called name means
# --------------------------------------------------------------------------

def test_a_real_method_is_real_code():
    inv, engine, _ = invoice([ret("never reached")])
    assert inv.recheck() is True
    assert engine.call_count == 0


def test_a_contract_declared_name_called_is_a_re_derivation():
    inv, engine, _ = invoice([{"value": 1249.5, "p": 0.9},
                              {"value": 1249.5, "p": 0.6}])
    assert ~inv.total == 0.9
    assert ~inv.total() == 0.6
    assert all("action" not in c["messages"][0]["content"] for c in engine.calls)


def test_any_other_called_name_is_an_episode():
    inv, engine, _ = invoice([ret("a summary")])
    result = inv.summarize()
    assert +result == "a summary" and ~result == 0.8


def test_plain_access_never_runs_an_episode():
    inv, engine, _ = invoice([ret("never reached")])
    inv.summarize                                    # accessed, not called
    assert engine.call_count == 0


# --------------------------------------------------------------------------
# changesets are proposed, never applied
# --------------------------------------------------------------------------

def test_a_scripted_episode_commits_a_multi_cell_changeset_atomically():
    inv, engine, ledger = invoice([
        ret("Marked paid at the full amount.",
            {"status": "paid", "vat": 0.24, "note": "settled"})])
    result = inv.mark_paid()
    assert +result == "Marked paid at the full amount."
    assert (+inv.status, +inv.vat, +inv.note) == ("paid", 0.24, "settled")


def test_one_necessary_check_failure_leaves_zero_cells_applied():
    inv, engine, ledger = invoice([
        ret("done", {"status": "paid", "vat": 99.0}),  # vat is out of RangeBelief(0, 1)
    ])
    with pytest.raises(Unresolvable) as caught:
        inv.mark_paid()
    assert "nothing was applied" in str(caught.value)
    assert ("inv" , "status") not in ledger.cells()
    assert inv.__root__.__resolved__ == {}


def test_a_refused_write_is_fed_back_and_the_next_attempt_lands():
    inv, engine, _ = invoice([
        ret("done", {"status": "paid", "vat": 99.0}),
        ret("done", {"status": "paid", "vat": 0.24}),
    ])
    result = inv.mark_paid()
    assert +result == "done" and +inv.vat == 0.24
    assert engine.call_count == 2
    second = "\n".join(m["content"] for m in engine.calls[1]["messages"])
    assert "rejected" in second and "vat" in second


def test_a_write_to_an_undeclared_attribute_is_refused_with_the_reason():
    inv, engine, _ = invoice([ret("done", {"invented": 1})] * 9)
    with pytest.raises(Unresolvable) as caught:
        inv.mark_paid()
    assert "not a declared attribute" in str(caught.value)


def test_a_write_to_a_frozen_cell_is_refused():
    inv, engine, _ = invoice([ret("done", {"status": "paid"})] * 9)
    inv.status = "pending"                           # the human speaks; frozen
    with pytest.raises(Unresolvable) as caught:
        inv.mark_paid()
    assert "frozen" in str(caught.value)
    assert +inv.status == "pending"


def test_a_changeset_carrying_frozen_is_rejected():
    """Invariant 4 as a test: a model can never pin its own answer."""
    inv, engine, ledger = invoice([
        {"action": "return", "changes": {"status": "paid"}, "value": "done",
         "p": 0.9, "frozen": True},
        ret("done", {"status": "paid"}),
    ])
    result = inv.mark_paid()
    assert +result == "done"
    assert engine.call_count == 2
    rebuke = "\n".join(m["content"] for m in engine.calls[1]["messages"])
    assert "frozen" in rebuke
    assert not any(o.frozen and o.attr == "status" for o in ledger)


def test_the_committed_writes_carry_their_provenance():
    inv, engine, ledger = invoice([ret("done", {"status": "paid"})])
    result = inv.mark_paid(1, style="terse")
    assert result.__opinion__.meta["call"] == "mark_paid(1, style='terse')"
    assert result.__opinion__.meta["changes"] == {"status": "paid"}


# --------------------------------------------------------------------------
# the action grammar
# --------------------------------------------------------------------------

def test_the_get_action_reads_through_the_pipeline():
    inv, engine, _ = invoice([
        {"action": "get", "attr": "total"},
        ret("The total is 1249.50."),
    ])
    inv.total = 1249.5                               # frozen, so `get` costs nothing
    result = inv.explain()
    assert +result == "The total is 1249.50."
    observation = engine.calls[-1]["messages"][-1]["content"]
    assert "1249.5" in observation


def test_the_get_action_refuses_a_name_that_is_not_there():
    inv, engine, _ = invoice([
        {"action": "get", "attr": ""},
        ret("gave up gracefully"),
    ])
    assert +inv.explain() == "gave up gracefully"
    assert "refused" in engine.calls[-1]["messages"][-1]["content"]


def test_the_call_action_runs_real_code():
    inv, engine, _ = invoice([
        {"action": "call", "method": "recheck", "args": []},
        ret("rechecked"),
    ])
    assert +inv.verify() == "rechecked"
    assert "true" in engine.calls[-1]["messages"][-1]["content"].lower()


def test_the_call_action_refuses_an_imagined_method():
    """Depth 1: an imagined method may not call another."""
    inv, engine, _ = invoice([
        {"action": "call", "method": "imagine_something"},
        ret("understood"),
    ])
    assert +inv.explain() == "understood"
    observation = engine.calls[-1]["messages"][-1]["content"]
    assert "imagined" in observation


def test_an_unknown_action_is_corrected_not_obeyed():
    inv, engine, _ = invoice([
        {"action": "freeze", "attr": "total"},       # there is no freeze action
        ret("fine"),
    ])
    assert +inv.explain() == "fine"
    assert "Unknown action" in engine.calls[-1]["messages"][-1]["content"]


def test_the_action_budget_is_finite():
    inv, engine, _ = invoice([{"action": "get", "attr": "source_text"}] * 40)
    with pytest.raises(Unresolvable):
        inv.explain()
    assert engine.call_count <= ACTION_BUDGET * (CORRECTIONS + 1) + 4


def test_an_oversized_value_is_truncated_with_a_marker():
    huge = "x" * 9000
    inv, engine, _ = invoice([
        {"action": "get", "attr": "source_text"},
        ret("read it"),
    ], source=huge)
    inv.explain()
    observation = engine.calls[-1]["messages"][-1]["content"]
    assert "truncated" in observation and len(observation) < 4000


# --------------------------------------------------------------------------
# statelessness and repetition
# --------------------------------------------------------------------------

def test_the_same_snapshot_and_engine_give_a_byte_identical_result():
    def once():
        inv, engine, _ = invoice([ret("identical", {"status": "paid"})])
        result = inv.mark_paid()
        return +result, ~result, result.__opinion__.meta["changes"]

    assert once() == once()


def test_repeating_a_call_on_unchanged_state_consults_the_engine_once():
    inv, engine, _ = invoice([ret("a summary")])
    first = inv.summarize()
    second = inv.summarize()
    assert engine.call_count == 1
    assert (+first, ~first) == (+second, ~second)
    assert first.__entity__ == second.__entity__


def test_an_interleaved_assignment_triggers_a_fresh_episode():
    inv, engine, _ = invoice([ret("first"), ret("second")])
    assert +inv.summarize() == "first"
    inv.note = "something changed"
    assert +inv.summarize() == "second"
    assert engine.call_count == 2


def test_a_committed_changeset_triggers_a_fresh_episode():
    inv, engine, _ = invoice([ret("first", {"status": "paid"}), ret("second")])
    assert +inv.mark_paid() == "first"
    assert +inv.mark_paid() == "second"
    assert engine.call_count == 2


def test_a_changed_thing_argument_triggers_a_fresh_episode():
    """The argument's snapshot is part of what the episode saw, so its state
    is part of the repetition key -- a changed argument must invalidate the
    memo exactly like an interleaved assignment on the host does."""
    inv, engine, ledger = invoice([ret("prioritize it"), ret("drop everything")])

    class Doc(Thing):
        __beliefs__ = [human("desk")]
        title: str

    doc = Doc(__entity__="doc-1", __ledger__=ledger, title="Q3 report")
    assert +inv.operate(doc) == "prioritize it"
    doc.title = "URGENT: building on fire"
    assert +inv.operate(doc) == "drop everything"
    assert engine.call_count == 2


def test_an_unchanged_thing_argument_still_hits_the_memo():
    inv, engine, ledger = invoice([ret("prioritize it")])

    class Doc(Thing):
        __beliefs__ = [human("desk")]
        title: str

    doc = Doc(__entity__="doc-1", __ledger__=ledger, title="Q3 report")
    assert +inv.operate(doc) == "prioritize it"
    assert +inv.operate(doc) == "prioritize it"
    assert engine.call_count == 1


def test_different_arguments_are_different_cells():
    inv, engine, _ = invoice([ret("terse"), ret("full")])
    assert +inv.summarize(style="terse") == "terse"
    assert +inv.summarize(style="full") == "full"
    assert engine.call_count == 2


def test_the_call_expression_normalizes_its_arguments():
    assert call_expression("f", (1,), {}) == call_expression("f", (1.0,), {})
    assert call_expression("f", (), {"b": 1, "a": 2}) == "f(a=2, b=1)"


def test_returns_is_stripped_from_the_call_cell_id():
    """``returns=`` governs acceptance, not identity."""
    inv, engine, _ = invoice([ret("a summary")])
    typed = inv.summarize(returns=str)
    plain = inv.summarize()
    assert typed.__entity__ == plain.__entity__
    assert engine.call_count == 1


# --------------------------------------------------------------------------
# typed ad-hoc calls
# --------------------------------------------------------------------------

def test_a_returns_typed_call_vetoes_a_wrong_shape_and_accepts_the_correction():
    inv, engine, _ = invoice([ret("not a number"), ret(1007.66)])
    result = inv.net_total(returns=float)
    assert +result == 1007.66
    assert engine.call_count == 2
    retry = "\n".join(m["content"] for m in engine.calls[1]["messages"])
    assert "rejected" in retry


def test_a_returns_typed_call_gives_up_when_the_shape_never_arrives():
    inv, engine, _ = invoice([ret("not a number")] * 20)
    with pytest.raises(Unresolvable, match="declared shape"):
        inv.net_total(returns=float)


def test_the_declared_return_shape_reaches_the_prompt():
    inv, engine, _ = invoice([ret(1007.66)])
    inv.net_total(returns=float)
    prompt = "\n".join(m["content"] for m in engine.calls[0]["messages"])
    assert "float" in prompt


# --------------------------------------------------------------------------
# recursion on the result, not the input
# --------------------------------------------------------------------------

def test_x_f_f_f_runs_three_episodes_on_three_entities():
    inv, engine, _ = invoice([ret("first pass"), ret("second pass"),
                              ret("third pass")])
    a = inv.explain_in_depth()
    b = a.explain_in_depth()
    c = b.explain_in_depth()
    assert (+a, +b, +c) == ("first pass", "second pass", "third pass")
    assert len({a.__entity__, b.__entity__, c.__entity__}) == 3
    assert engine.call_count == 3


def test_a_childs_snapshot_is_its_value_plus_one_provenance_line():
    inv, engine, _ = invoice([ret("first pass"), ret("second pass")])
    child = inv.explain_in_depth()
    child.explain_in_depth()
    prompt = "\n".join(m["content"] for m in engine.calls[1]["messages"])
    assert "first pass" in prompt
    assert "provenance" in prompt
    assert prompt.count("provenance") == 1           # never an ancestor transcript


def test_a_scripted_echo_is_vetoed_by_non_echo_and_re_proposed():
    carried = "The invoice totals 1249.50 euros and is addressed to ACME Oy."
    elaborated = ("Two line items make up the sum: a large widget at 999.00 "
                  "and shipping at 250.50, before any Finnish VAT is removed.")
    inv, engine, _ = invoice([ret(carried), ret(carried), ret(elaborated)],
                             extra=[NonEchoBelief()])
    child = inv.explain_in_depth()
    assert +child == carried                          # nothing to echo yet

    deeper = child.explain_in_depth()
    assert +deeper == elaborated
    assert engine.call_count == 3


# --------------------------------------------------------------------------
# cross-entity calls
# --------------------------------------------------------------------------

def test_a_thing_valued_argument_renders_its_own_snapshot():
    inv, engine, _ = invoice([ret("they differ by 100.00")])
    other, other_engine, _ = invoice([ret("never reached")], source="Total 1149.50 EUR")
    other.vendor_note = "the second invoice"

    result = inv.compare(other)
    assert +result == "they differ by 100.00"
    prompt = "\n".join(m["content"] for m in engine.calls[0]["messages"])
    assert "argument #0" in prompt
    assert "the second invoice" in prompt


# --------------------------------------------------------------------------
# freezing, still code-only (invariant 4)
# --------------------------------------------------------------------------

def test_nothing_an_episode_commits_is_frozen():
    inv, engine, ledger = invoice([ret("done", {"status": "paid", "vat": 0.24})])
    inv.mark_paid()
    unfrozen = [o for o in ledger if o.attr in ("status", "vat")]
    assert unfrozen and not any(o.frozen for o in unfrozen)


def test_an_episode_result_can_be_pinned_by_code():
    inv, engine, _ = invoice([ret("a summary")])
    result = inv.summarize()
    pinned = freeze(result)
    assert (+pinned, ~pinned) == ("a summary", 0.8)
    assert pinned.__opinion__.frozen is True


def test_a_thing_without_a_generative_belief_cannot_imagine():
    class Plain(Thing):
        __beliefs__ = [human("jane")]
        note = contract(str)

    thing = Plain(note="hello", __ledger__=Ledger())
    with pytest.raises(Unresolvable, match="imagine"):
        thing.summarize()
