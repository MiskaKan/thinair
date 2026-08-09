"""The one contract (SPEC.md §3): what every belief is, and what none of them may be."""

from __future__ import annotations

import pytest

from thinair import beliefs as B
from thinair.beliefs import (
    Belief,
    Discriminative,
    Judgment,
    MemoBelief,
    Scoped,
    as_judgment,
    code_identity,
    generative_members,
    human,
    lookup,
    opinion_at,
    reason_of,
    registry,
    value_at,
)
from thinair.ledger import Ledger, Opinion

from fakes import FakeSnapshot, ScriptedBelief, head


# --------------------------------------------------------------------------
# the legal minimum
# --------------------------------------------------------------------------

def test_bare_belief_is_silent():
    """A belief with nothing to say returns None -- the legal minimum."""
    assert Belief(id="bare:1")(head(1), "anything") is None


def test_the_runtime_never_checks_isinstance():
    """Membership is structural: a plain callable participates."""
    scripted = ScriptedBelief({"total": (1249.5, 0.93)}, "duck:1")
    assert not isinstance(scripted, Belief)
    assert scripted(head(None), "total") == (1249.5, 0.93)
    assert generative_members([scripted]) == [scripted]


def test_a_belief_is_a_pure_function_of_the_snapshot():
    """Same snapshot in, same (v, p) out -- always (invariant 3)."""
    from thinair.validators import FuzzyBelief

    e = head("ACME Oy Ltd", source_text="Bill from ACME Oy")
    first = FuzzyBelief("source_text")(e, "vendor")
    for _ in range(5):
        assert FuzzyBelief("source_text")(e, "vendor") == first


# --------------------------------------------------------------------------
# Judgment: the answer shape
# --------------------------------------------------------------------------

def test_judgment_is_a_pair_that_carries_its_reason():
    j = Judgment(1249.5, 0.93, {"reason": "found verbatim"})
    assert j == (1249.5, 0.93)
    assert (+j, ~j) == (1249.5, 0.93)
    v, p = j
    assert (v, p) == (1249.5, 0.93)
    assert reason_of(j) == "found verbatim"


@pytest.mark.parametrize("got,expected", [
    (None, None),
    ((1, 0.5), (1, 0.5)),
    ([1, 0.5], (1, 0.5)),
    ((1, 0.5, {"reason": "r"}), (1, 0.5)),
    (Judgment(1, 0.5), (1, 0.5)),
])
def test_as_judgment_normalizes_every_legal_answer(got, expected):
    result = as_judgment(got)
    assert (result is None and expected is None) or tuple(result) == expected


def test_as_judgment_normalizes_an_opinion():
    o = Opinion(belief="b", entity="e", attr="a", value=7, p=0.4, t=1.0)
    assert tuple(as_judgment(o)) == (7, 0.4)


@pytest.mark.parametrize("bad", ["nope", 3, (1, 2, 3, 4), {"value": 1}])
def test_as_judgment_rejects_anything_else(bad):
    with pytest.raises(TypeError):
        as_judgment(bad)


def test_with_reason_does_not_mutate_the_original():
    j = Judgment(1, 0.5)
    annotated = j.with_reason("because")
    assert reason_of(j) is None and reason_of(annotated) == "because"


# --------------------------------------------------------------------------
# identity is durable (invariant 6)
# --------------------------------------------------------------------------

def test_id_derives_from_the_configuration():
    from thinair.validators import RangeBelief, SchemaBelief

    assert RangeBelief(0, 100).id == "rangeBelief[0,100]"
    assert SchemaBelief(float).id == "schemaBelief[float]"
    assert SchemaBelief([{"desc": str, "amount": float}]).id == \
        "schemaBelief[[{amount:float,desc:str}]]"


def test_changing_the_configuration_mints_a_new_id():
    from thinair.validators import RangeBelief

    assert RangeBelief(0, 100).id != RangeBelief(0, 200).id


def test_same_configuration_is_the_same_belief():
    from thinair.validators import RangeBelief

    assert RangeBelief(0, 100).id == RangeBelief(0, 100).id
    assert lookup("rangeBelief[0,100]").id == "rangeBelief[0,100]"


def test_the_registry_does_not_intern_away_an_attachment_site_override():
    """Same check, same id, different veto right -- all three at once."""
    from thinair.validators import VerbatimBelief

    strict = VerbatimBelief("src")
    lenient = VerbatimBelief("src", necessary=False)
    assert strict.id == lenient.id
    assert strict.necessary is True and lenient.necessary is False


def test_a_long_configuration_hashes_rather_than_sprawls():
    from thinair.validators import EnumBelief

    belief = EnumBelief([f"member-{i:03d}" for i in range(40)])
    assert belief.id.startswith("enumBelief[#") and len(belief.id) < 40


def test_one_id_cannot_name_two_kinds_of_belief():
    class Impostor(Belief):
        pass

    Belief(id="collision:1")
    with pytest.raises(ValueError, match="invariant 6"):
        Impostor(id="collision:1")


def test_the_registry_maps_ids_to_instances():
    belief = Belief(id="registered:1")
    assert lookup("registered:1") is belief
    assert registry()["registered:1"] is belief
    registry()["registered:1"] = "tampered"          # a copy, not the registry
    assert lookup("registered:1") is belief


def test_code_identity_tracks_the_source_not_the_name():
    def net(g, v):
        return g / (1 + v)

    first = code_identity(net)

    def net(g, v):                                    # noqa: F811 - deliberate
        return g / (1.0 + v)

    assert first != code_identity(net)
    assert first.startswith("code:") and "@" in first


# --------------------------------------------------------------------------
# options at the attachment site
# --------------------------------------------------------------------------

def test_necessary_is_overridable_at_construction():
    from thinair.validators import VerbatimBelief

    assert VerbatimBelief("source_text").necessary is True
    assert VerbatimBelief("source_text", necessary=False).necessary is False


def test_options_do_not_leak_into_the_id():
    """A veto right is a policy, not a configuration: same check, same id."""
    from thinair.validators import VerbatimBelief

    assert VerbatimBelief("source_text").id == VerbatimBelief("source_text", necessary=False).id


def test_veto_line_must_be_a_probability():
    with pytest.raises(ValueError):
        Belief(id="bad-line:1", veto_line=1.5)


# --------------------------------------------------------------------------
# reading a sealed snapshot is inert
# --------------------------------------------------------------------------

def test_snapshot_reads_yield_opinions_and_absent_cells_read_as_none():
    e = FakeSnapshot(total=(1249.5, 0.93))
    assert isinstance(opinion_at(e, "total"), Opinion)
    assert value_at(e, "total") == 1249.5
    assert opinion_at(e, "vendor") is None
    assert value_at(e, "vendor", "missing") == "missing"


def test_a_snapshot_is_sealed():
    e = FakeSnapshot(total=(1249.5, 0.93))
    with pytest.raises(AttributeError):
        e.total = 1


# --------------------------------------------------------------------------
# the snapshot's two invisible guarantees
# --------------------------------------------------------------------------

def test_the_proposer_is_evaluated_once_per_round_however_many_pull_it():
    """Memoization: pulling costs nothing after the first pull."""
    scripted = ScriptedBelief({"total": (1249.5, 0.93)})
    memo, active = {}, set()
    entry = MemoBelief(scripted, memo, active)
    e = FakeSnapshot(beliefs=[entry])

    results = [e.__beliefs__[0](e, "total") for _ in range(4)]

    assert len(scripted.calls) == 1
    assert all(tuple(r) == (1249.5, 0.93) for r in results)


def test_memoization_is_per_cell():
    scripted = ScriptedBelief({"total": (1, 0.9), "vendor": ("ACME", 0.8)})
    entry = MemoBelief(scripted, {}, set())
    e = FakeSnapshot(beliefs=[entry])
    assert tuple(entry(e, "total")) == (1, 0.9)
    assert tuple(entry(e, "vendor")) == ("ACME", 0.8)
    assert len(scripted.calls) == 2


def test_a_consultation_cycle_resolves_to_none_rather_than_recursing():
    """Re-entry yields None -- the value that already means 'no opinion'."""
    seen = []

    def ouroboros(e, attr):
        seen.append(attr)
        return e.__beliefs__[0](e, attr)          # asks itself

    memo, active = {}, set()
    entry = MemoBelief(ouroboros, memo, active)
    e = FakeSnapshot(beliefs=[entry])

    assert entry(e, "total") is None
    assert len(seen) == 1


def test_a_two_belief_cycle_also_resolves():
    memo, active = {}, set()

    def first(e, attr):
        return e.__beliefs__[1](e, attr)

    def second(e, attr):
        got = e.__beliefs__[0](e, attr)
        return (0, 0.0) if got is None else got

    panel = [MemoBelief(first, memo, active), MemoBelief(second, memo, active)]
    e = FakeSnapshot(beliefs=panel)
    assert tuple(panel[0](e, "total")) == (0, 0.0)


# --------------------------------------------------------------------------
# generative vs discriminative is observable, not a stored kind
# --------------------------------------------------------------------------

def test_a_discriminative_belief_is_silent_in_an_empty_cell():
    """The behavioral kinds differ only in *when they can speak*."""
    from thinair.validators import VerbatimBelief

    empty = FakeSnapshot(beliefs=[lambda e, a: None], source_text="anything")
    assert VerbatimBelief("source_text")(empty, "vendor") is None


def test_a_generative_belief_speaks_into_an_empty_cell():
    scripted = ScriptedBelief({"total": (1249.5, 0.93)})
    empty = FakeSnapshot(beliefs=[scripted])
    assert tuple(as_judgment(scripted(empty, "total"))) == (1249.5, 0.93)


def test_generativity_is_observable_by_asking_an_empty_cell():
    """Generativity, spelled as the observation it names."""
    from thinair.validators import RangeBelief

    def can_speak_into_an_empty_cell(belief):
        empty = FakeSnapshot(beliefs=[lambda e, a: None])
        return belief(empty, "total") is not None

    assert can_speak_into_an_empty_cell(ScriptedBelief({"total": (1, 0.9)}))
    assert not can_speak_into_an_empty_cell(RangeBelief(0, 10))


# --------------------------------------------------------------------------
# discriminative plumbing
# --------------------------------------------------------------------------

class _Even(Discriminative):
    """A minimal user-written validator, exactly as SPEC.md §3 shows one."""

    necessary = True

    def judge(self, value, e, attr):
        if not isinstance(value, int):
            return None
        return (1.0, "even") if value % 2 == 0 else (0.0, "odd")


def test_a_discriminative_belief_returns_the_candidate_it_judged():
    e = head(4, 0.7)
    got = _Even()(e, "n")
    assert tuple(got) == (4, 1.0)
    assert reason_of(got) == "even"


def test_a_discriminative_belief_judges_the_proposal_not_its_own_value():
    """The candidate rides through; only p is the belief's own."""
    got = _Even()(head(5, 0.99), "n")
    assert tuple(got) == (5, 0.0)


def test_out_of_scope_is_none_not_a_zero():
    assert _Even()(head("text"), "n") is None


def test_scoped_returns_none_off_its_attribute():
    inner = _Even()
    scoped = Scoped("total", inner)
    e = head(4)
    assert tuple(scoped(e, "total")) == (4, 1.0)
    assert scoped(e, "vendor") is None
    assert scoped.id == f"{inner.id}@total"
    assert scoped.necessary is inner.necessary


# --------------------------------------------------------------------------
# the human is a belief
# --------------------------------------------------------------------------

def test_the_default_human_is_silent_on_consultation():
    assert human("jane")(head(1), "total") is None
    assert human("jane").id == "human:jane"


def test_an_interactive_human_is_asked(monkeypatch):
    answers = iter(["1249.50", "0.8"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    got = human("bob", interactive=True)(head(None), "total")
    assert tuple(got) == (1249.5, 0.8)


def test_an_interactive_human_may_decline(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert human("carol", interactive=True)(head(None), "total") is None


def test_a_prompted_human_corroborates_but_does_not_freeze(monkeypatch):
    """Only assignment and freeze freeze (invariant 4)."""
    answers = iter(["yes", ""])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    got = human("dave", interactive=True)(head(None), "note")
    assert not getattr(got, "frozen", False)
    assert ~got == 1.0                       # an empty p answer means certain
    assert "frozen" not in (getattr(got, "meta", None) or {})


def test_two_people_do_not_share_an_id():
    assert human("jane").id != human("bob").id


# --------------------------------------------------------------------------
# direct calls record nothing
# --------------------------------------------------------------------------

def test_consulting_the_panel_by_hand_writes_no_opinions():
    """Inspection never mutates memory."""
    from thinair.validators import RangeBelief, TokenSubsetBelief

    ledger = Ledger()
    panel = [ScriptedBelief({"total": (1249.5, 0.93)}),
             TokenSubsetBelief("source_text"), RangeBelief(0, 1e6)]
    e = FakeSnapshot(beliefs=panel, ledger=ledger,
                     source_text="Total 1249.50 EUR")

    survey = {getattr(b, "id", repr(b)): b(e, "total") for b in panel}

    assert len(survey) == 3
    assert len(ledger) == 0


def test_a_scripted_panel_writes_correct_opinions_into_a_ledger():
    """A hand-run round, recorded."""
    from thinair.validators import RangeBelief, TokenSubsetBelief

    ledger = Ledger()
    proposer = ScriptedBelief({"total": (1249.5, 0.93)}, "scripted:proposer")
    panel = [proposer, TokenSubsetBelief("source_text"), RangeBelief(0, 1e6)]
    memo, active = {}, set()
    entries = [MemoBelief(b, memo, active) for b in panel]
    e = FakeSnapshot(entity="inv-1", beliefs=entries, ledger=ledger,
                     source_text="Total 1249.50 EUR")

    for entry in entries:
        got = entry(e, "total")
        if got is None:
            continue
        ledger.add(Opinion(belief=entry.id, entity="inv-1", attr="total",
                           value=+got, p=~got, meta=dict(got.meta)))

    recorded = ledger.opinions(entity="inv-1", attr="total")
    assert [o.belief for o in recorded] == \
        ["scripted:proposer", "tokenSubsetBelief[source_text,kinds=[numbers]]", "rangeBelief[0,1000000]"]
    assert {o.value for o in recorded} == {1249.5}      # one candidate, three voices
    assert [o.p for o in recorded] == [0.93, 1.0, 1.0]
    assert not any(o.frozen for o in recorded)          # invariant 4
    assert len(proposer.calls) == 1                     # memoized across the panel


# --------------------------------------------------------------------------
# configuration cascade -- no network involved
# --------------------------------------------------------------------------

def test_config_falls_back_through_mro_then_process_then_env(monkeypatch):
    class Base:
        pass

    class Child(Base):
        pass

    monkeypatch.delenv("THINAIR_MODEL", raising=False)
    assert B.config("model", Child) is None

    monkeypatch.setenv("THINAIR_MODEL", "from-env")
    assert B.config("model", Child) == "from-env"

    with B.config_scope(model="process-wide"):
        assert B.config("model", Child) == "process-wide"
        with B.config_scope(Base, model="from-base"):
            assert B.config("model", Child) == "from-base"
        assert B.config("model", Child) == "process-wide"

    assert B.config("model", Child) == "from-env"


def test_constructing_a_model_belief_touches_no_network():
    """Invariant 7 at the constructor: configuration is not a call."""
    from thinair.beliefs import model

    belief = model("qwen3-35b", temperature=0.2)
    assert belief.proposes is True
    assert belief.id == "model:qwen3-35b@T0.2/extract-v3/qwen3_35b/v1"
    assert belief.short == "model/extract-v3"


def test_model_identity_includes_every_knob(monkeypatch):
    from thinair.beliefs import model

    ids = {
        model("qwen3-35b").id,
        model("qwen3-35b", temperature=0.7).id,
        model("qwen3-35b", think=True).id,
        model("gpt-oss-20b").id,
    }
    assert len(ids) == 4
