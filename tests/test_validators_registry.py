"""What holds of *every* validator, including ones not yet written.

A contributed validator is proven by being listed in ``FAMILIES``; these
tests then cover it with no new test code.  That is the same bargain the
model folders make in ``tests/test_models.py``.
"""

from __future__ import annotations

import inspect

import pytest

from thinair import validators as V
from thinair.beliefs import Belief, Discriminative
from thinair.ledger import Ledger, Opinion
from thinair.validators.consistency import Relation

from fakes import FakeSnapshot

#: one constructed instance per class, with the arguments each one needs.
SAMPLES = {
    V.Schema: (float,),
    V.Format: ("email",),
    V.Checksum: ("luhn",),
    V.Range: (0, 100),
    V.Enum: (["a", "b"],),
    V.Length: (1, 10),
    V.Unique: (),
    V.Sorted: (),
    V.Parses: ("json",),
    V.Verbatim: ("source_text",),
    V.Normalized: ("source_text",),
    V.Fuzzy: ("source_text",),
    V.TokenSubset: ("source_text",),
    V.QuoteIntegrity: ("source_text",),
    V.SpanValid: ("source_text",),
    V.FrozenConsistent: (),
    V.NonEcho: (),
    V.SumsTo: (["a", "b"], "t"),
    V.ItemsSumTo: ("items", "amount", "t"),
    V.Recompute: ("t", sum, ["a"]),
    V.TemporalOrder: ("a", "b"),
    V.Conservation: ("i", "o", "d"),
    V.FunctionalDependency: ("k", "v"),
    V.MutuallyExclusive: (["a", "b"],),
    V.IsoCountry: (),
    V.IsoCurrency: (),
    V.Timezone: (),
    V.CalendarFact: (),
    V.Executes: (),
    V.PassesTests: ([bool],),
    V.RoundTrip: ("json",),
    V.Calculator: (),
    V.RegexBehavior: (["a"],),
}

ALL = V.validators()
INSTANCES = [cls(*SAMPLES[cls]) for cls in ALL]
IDS = [b.id for b in INSTANCES]


# --------------------------------------------------------------------------
# the registry is complete and consistent
# --------------------------------------------------------------------------

def test_every_exported_validator_is_in_a_family():
    exported = {
        name for name in V.__all__
        if isinstance(getattr(V, name), type) and issubclass(getattr(V, name), Belief)
        and getattr(V, name) is not Relation
    }
    listed = {cls.__name__ for cls in ALL}
    assert exported == listed


def test_every_family_module_is_fully_exported():
    """A validator that exists but is not registered is a validator nobody finds."""
    import thinair.validators.consistency as c
    import thinair.validators.executable as x
    import thinair.validators.form as f
    import thinair.validators.grounding as g
    import thinair.validators.reference as r

    for module in (f, g, c, r, x):
        defined = {
            name for name, obj in vars(module).items()
            if isinstance(obj, type) and issubclass(obj, Belief)
            and obj.__module__ == module.__name__
            and not name.startswith("_") and obj is not Relation
        }
        assert defined <= {cls.__name__ for cls in ALL}, module.__name__


def test_every_class_appears_in_exactly_one_family():
    assert len(ALL) == len(set(ALL))
    for cls in ALL:
        assert V.family_of(cls) is not None


def test_the_sample_table_covers_the_registry():
    assert set(SAMPLES) == set(ALL)


def test_family_letters_have_names():
    assert set(V.FAMILY_NAMES) == set(V.FAMILIES)


# --------------------------------------------------------------------------
# what every validator must do
# --------------------------------------------------------------------------

@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_is_a_belief(belief):
    assert isinstance(belief, Belief)
    assert isinstance(belief.id, str) and belief.id
    assert isinstance(belief.necessary, bool)
    assert 0.0 <= belief.veto_line <= 1.0
    assert len(inspect.signature(type(belief).__call__).parameters) == 3


@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_is_discriminative(belief):
    """None of them can speak into an empty cell -- that is what a check is."""
    assert isinstance(belief, (Discriminative, Relation))
    assert belief.proposes is False


@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_is_silent_when_nothing_is_proposed(belief):
    """No candidate, no opinion -- the resolution flow depends on this."""
    empty = FakeSnapshot(beliefs=[lambda e, a: None])
    attr = getattr(belief, "virtual", "some_attribute")
    assert belief(empty, "some_attribute") is None
    if attr != "some_attribute":                      # relations judge from state
        assert belief(empty, attr) is None


@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_returns_the_candidate_it_judged(belief, monkeypatch):
    """A discriminative belief judges; it does not substitute."""
    monkeypatch.setenv(V.ALLOW_EXEC_ENV, "1")
    candidate = "2024-02-29"
    e = FakeSnapshot(beliefs=[lambda x, a: (candidate, 0.9)],
                     source_text=candidate, value=candidate)
    got = belief(e, "some_attribute")
    if got is not None:
        assert got[0] == candidate


@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_answers_within_zero_and_one(belief, monkeypatch):
    monkeypatch.setenv(V.ALLOW_EXEC_ENV, "1")
    for candidate in ("2024-02-29", 42, [1, 2], {"a": 1}, "", None, True):
        e = FakeSnapshot(beliefs=[lambda x, a, c=candidate: (c, 0.9)],
                         source_text="2024-02-29 EUR FI", value="carried text")
        got = belief(e, "some_attribute")
        assert got is None or 0.0 <= got[1] <= 1.0


@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_is_deterministic(belief, monkeypatch):
    """Invariant 3: same snapshot, same (v, p), always."""
    monkeypatch.setenv(V.ALLOW_EXEC_ENV, "1")
    e = FakeSnapshot(beliefs=[lambda x, a: ("2024-02-29", 0.9)],
                     source_text="2024-02-29", value="carried")
    first = belief(e, "some_attribute")
    for _ in range(3):
        again = belief(e, "some_attribute")
        assert (first is None and again is None) or tuple(first) == tuple(again)


@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_can_be_demoted_to_a_measurement(belief):
    """The attachment site always gets the last word on veto rights."""
    cls = type(belief)
    demoted = cls(*SAMPLES[cls], necessary=False)
    assert demoted.necessary is False
    assert demoted.id == belief.id


@pytest.mark.parametrize("belief", INSTANCES, ids=IDS)
def test_every_validator_writes_the_one_record_shape(belief, monkeypatch):
    """Invariant 1: virtual attributes and real ones land the same way."""
    monkeypatch.setenv(V.ALLOW_EXEC_ENV, "1")
    ledger = Ledger()
    e = FakeSnapshot(entity="inv-1", ledger=ledger,
                     beliefs=[lambda x, a: ("2024-02-29", 0.9)],
                     source_text="2024-02-29", value="carried")
    attr = getattr(belief, "virtual", "some_attribute")
    got = belief(e, attr) or belief(e, "some_attribute")
    if got is None:
        pytest.skip(f"{belief.id} has no opinion about this candidate")
    ledger.add(Opinion(belief=belief.id, entity="inv-1", attr=attr,
                       value=+got, p=~got, meta=dict(got.meta)))
    (recorded,) = ledger.opinions(entity="inv-1")
    assert isinstance(recorded, Opinion) and not recorded.frozen


# --------------------------------------------------------------------------
# the library's independence (invariant 7)
# --------------------------------------------------------------------------

def _imports_of(path):
    """Every module name a file imports, as the import graph sees them."""
    import ast

    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def test_the_validator_library_contains_zero_model_calls():
    """The library earns its independence by containing no model calls.

    Asserted against the import graph rather than the text, so a docstring may
    say the word "engine" and a module still may not reach one.
    """
    import pathlib

    import thinair.validators as package

    folder = pathlib.Path(package.__file__).parent
    for path in sorted(folder.glob("*.py")):
        imports = _imports_of(path)
        assert not any("engine" in name for name in imports), path.name
        assert not any("models" in name for name in imports), path.name
        assert not any(name.split(".")[0] == "urllib" for name in imports), path.name
        assert "ModelBelief" not in path.read_text(encoding="utf-8"), path.name


def test_no_validator_can_freeze_anything():
    """Invariant 4: freezing is a code-only capability."""
    import pathlib

    import thinair.validators as package

    folder = pathlib.Path(package.__file__).parent
    for path in sorted(folder.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "frozen=True" not in source, path.name


def test_validators_never_write_to_the_ledger_themselves():
    """Only the resolution flow records; a consultation is pure."""
    ledger = Ledger()
    e = FakeSnapshot(entity="inv-1", ledger=ledger,
                     beliefs=[lambda x, a: ("2024-02-29", 0.9)],
                     source_text="2024-02-29", value="carried")
    for belief in INSTANCES:
        if isinstance(belief, V.Executes):
            continue                                  # would need the opt-in
        belief(e, getattr(belief, "virtual", "some_attribute"))
        belief(e, "some_attribute")
    assert len(ledger) == 0
