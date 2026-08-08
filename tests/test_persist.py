"""Persistence and observability, plus the property tests."""

from __future__ import annotations

import io
import json
import pickle

import pytest

from thinair import Thing, contract, freeze, human
from thinair.beliefs import model
from thinair.ledger import Ledger, Opinion, values_equal
from thinair.persist import VERSION, PersistError
from thinair.thing import _ledgers

from fakes import FakeEngine

SOURCE = "Widget 999.00\nShipping 250.50\nTotal 1249.50 EUR"


def invoice(script=None, ledger=None):
    engine = FakeEngine(script or [{"value": 1249.5, "p": 0.93}])

    class Invoice(Thing):
        """An invoice document to be understood."""

        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        source_text: str
        total = contract(float, extracted_from="source_text", range=(0, 1e6))
        vendor = contract(str, extracted_from="source_text")

    ledger = Ledger() if ledger is None else ledger
    return Invoice, Invoice(source_text=SOURCE, __ledger__=ledger), engine, ledger


@pytest.fixture
def scoped():
    """A ledger the revived Thing will land in."""
    ledger = Ledger()
    _ledgers.append(ledger)
    try:
        yield ledger
    finally:
        _ledgers.remove(ledger)


# --------------------------------------------------------------------------
# the blob
# --------------------------------------------------------------------------

def test_getstate_is_a_versioned_json_blob():
    _, inv, _, _ = invoice()
    +inv.total
    blob = inv.__getstate__()
    assert blob["thinair"] == VERSION
    assert blob["class"] == "Invoice"
    assert blob["entity"] == inv.__entity__
    json.dumps(blob)                                 # genuinely JSON-able


def test_the_blob_is_the_entitys_slice_of_the_ledger():
    _, inv, _, ledger = invoice()
    +inv.total
    ledger.add(Opinion(belief="human:bob", entity="somebody-else",
                       attr="total", value=1.0, p=0.5))
    blob = inv.__getstate__()
    assert all(o["entity"].startswith(inv.__entity__) for o in blob["opinions"])
    assert len(blob["opinions"]) < len(ledger)


def test_save_restore_continue(scoped):
    cls, inv, engine, _ = invoice()
    +inv.total
    inv.vendor = "ACME Oy"
    blob = inv.__getstate__()

    revived = blob @ cls
    assert (+revived.total, ~revived.total) == (1249.5, 0.93)
    assert (+revived.vendor, ~revived.vendor) == ("ACME Oy", 1.0)
    assert engine.call_count == 1                    # restoring consults nobody
    assert len(scoped) == len(blob["opinions"])


def test_a_resolution_is_restored_with_its_author_not_the_last_voice(scoped):
    """A validator's 1.0 means "I see nothing wrong", never "this is certain"."""
    cls, inv, _, _ = invoice()
    +inv.total
    revived = inv.__getstate__() @ cls
    assert ~revived.total == 0.93
    assert revived.total.__opinion__.belief.startswith("model:")


def test_frozen_state_survives_a_round_trip(scoped):
    cls, inv, _, _ = invoice()
    freeze(inv.total)
    revived = inv.__getstate__() @ cls
    assert revived.total.__opinion__.frozen is True
    assert ~revived.total == 0.93                    # pinning keeps the p


def test_restoring_twice_does_not_duplicate_opinions(scoped):
    cls, inv, _, _ = invoice()
    +inv.total
    blob = inv.__getstate__()
    blob @ cls
    blob @ cls
    assert len(scoped) == len(blob["opinions"])


def test_pickle_rides_the_same_mechanism(scoped):
    cls, inv, _, _ = invoice()
    +inv.total

    import __main__

    setattr(__main__, "PickledInvoice", cls)
    cls.__qualname__ = "PickledInvoice"
    cls.__module__ = "__main__"
    try:
        revived = pickle.loads(pickle.dumps(inv))
    finally:
        delattr(__main__, "PickledInvoice")
    assert revived.__entity__ == inv.__entity__
    assert +revived.total == 1249.5


def test_a_blob_may_be_revived_into_another_class(scoped):
    """Reviving into a different class is a recast and a load in one."""
    cls, inv, _, _ = invoice()
    +inv.total
    blob = inv.__getstate__()

    strict_engine = FakeEngine([{"value": 5.0, "p": 0.5}])

    class Receipt(Thing):
        """A receipt."""

        __beliefs__ = [model("strict", engine=strict_engine), human("jane")]
        source_text: str
        total = contract(float, range=(0, 10))

    receipt = blob @ Receipt
    assert isinstance(receipt, Receipt)
    assert +receipt.source_text == SOURCE            # frozen state carries over
    assert +receipt.total == 5.0                     # the new class governs


@pytest.mark.parametrize("blob", [
    {}, {"thinair": 99}, "not a blob", 42,
    {"thinair": VERSION, "entity": "e"},             # no class, no opinions
])
def test_a_blob_this_version_cannot_read_is_refused(blob):
    cls, _, _, _ = invoice()
    with pytest.raises(PersistError):
        blob @ cls


# --------------------------------------------------------------------------
# ledgers save and load whole
# --------------------------------------------------------------------------

def test_a_ledger_round_trips_through_a_file(tmp_path):
    _, inv, _, ledger = invoice()
    +inv.total
    inv.vendor = "ACME Oy"
    path = tmp_path / "ledger.json"
    ledger.dump(path)

    restored = Ledger.load(path)
    assert len(restored) == len(ledger)
    for before, after in zip(ledger, restored):
        assert (before.belief, before.entity, before.attr, before.p,
                before.frozen) == (after.belief, after.entity, after.attr,
                                   after.p, after.frozen)
        assert values_equal(before.value, after.value)


def test_losing_the_ledger_forgets_everything_ever_said():
    """The ledger is the memory -- there is no other copy."""
    cls, inv, engine, _ = invoice([{"value": 1249.5, "p": 0.93},
                                   {"value": 1249.5, "p": 0.4}])
    +inv.total
    blob = inv.__getstate__()
    blob["opinions"] = []

    fresh = Ledger()
    _ledgers.append(fresh)
    try:
        revived = blob @ cls
        assert ~revived.total == 0.4                 # re-derived from nothing
    finally:
        _ledgers.remove(fresh)


# --------------------------------------------------------------------------
# the property tests
# --------------------------------------------------------------------------

def test_persist_restore_is_the_identity_on_public_state(scoped):
    cls, inv, _, _ = invoice()
    +inv.total
    inv.vendor = "ACME Oy"
    freeze(inv.total)

    before = {attr: (+getattr(inv, attr), ~getattr(inv, attr))
              for attr in ("total", "vendor", "source_text")}
    revived = inv.__getstate__() @ cls
    after = {attr: (+getattr(revived, attr), ~getattr(revived, attr))
             for attr in ("total", "vendor", "source_text")}
    assert before == after


@pytest.mark.parametrize("a,b", [
    (1.0, 1.0), ("ACME  Oy", "acme oy"), ([1, 2], (1, 2)),
    ({"a": 1}, {"a": 1.0}), (None, None), (True, True),
])
def test_values_equal_is_symmetric_and_reflexive(a, b):
    assert values_equal(a, b) == values_equal(b, a)
    assert values_equal(a, a) and values_equal(b, b)


def test_after_any_freeze_reads_of_that_cell_make_zero_engine_calls():
    _, inv, engine, _ = invoice()
    freeze(inv.total)
    calls = engine.call_count
    for _ in range(10):
        +inv.total
        ~inv.total
    assert engine.call_count == calls


# --------------------------------------------------------------------------
# debug
# --------------------------------------------------------------------------

def test_the_debug_trace_shows_routes_verdicts_and_reasons():
    stream = io.StringIO()
    _, inv, _, _ = invoice([{"value": 9e9, "p": 0.99}, {"value": 1249.5, "p": 0.93}])
    with Thing.debug(stream):
        +inv.total
    raw = stream.getvalue()
    # boxes wrap long lines, so compare against the unwrapped text
    text = " ".join(raw.replace("│", " ").split())
    assert "route:" in text
    assert "9000000000.0" in text
    assert "above the declared maximum" in text      # the verdict, with its reason
    assert "VETOED" in text and "resolved: 1249.5" in text
    assert raw.startswith("┌─ read ") and raw.rstrip().endswith("─")


def test_the_debug_trace_shows_episode_actions_and_the_commit():
    stream = io.StringIO()
    engine = FakeEngine([
        {"action": "get", "attr": "source_text"},
        {"action": "return", "changes": {"vendor": "ACME Oy"}, "value": "done",
         "p": 0.8},
    ])

    class Invoice(Thing):
        """An invoice."""

        __beliefs__ = [model("m", engine=engine), human("jane")]
        source_text: str
        vendor = contract(str, extracted_from="source_text")

    inv = Invoice(source_text=SOURCE + "\nACME Oy", __ledger__=Ledger())
    with Thing.debug(stream):
        inv.identify()
    text = stream.getvalue()
    assert "episode" in text and "action:" in text
    assert "changeset" in text and "COMMITTED" in text


def test_the_debug_trace_shows_a_rollback():
    stream = io.StringIO()
    engine = FakeEngine([
        {"action": "return", "changes": {"nowhere": 1}, "value": "done", "p": 0.8},
    ])

    class Invoice(Thing):
        """An invoice."""

        __beliefs__ = [model("m", engine=engine), human("jane")]
        source_text: str

    inv = Invoice(source_text=SOURCE, __ledger__=Ledger())
    with Thing.debug(stream):
        with pytest.raises(Exception):
            inv.identify()
    assert "ROLLED BACK" in stream.getvalue()


def test_debug_is_silent_by_default(capsys):
    _, inv, _, _ = invoice()
    +inv.total
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_the_environment_can_turn_debugging_on(monkeypatch, capsys):
    monkeypatch.setenv("THINAIR_DEBUG", "1")
    _, inv, _, _ = invoice()
    +inv.total
    assert "route:" in capsys.readouterr().err


def test_source_is_the_rendering_behind_the_dunder():
    from thinair import source

    _, inv, _, _ = invoice()
    +inv.total
    inv.vendor = "ACME Oy"
    assert source(inv) == inv.__source__
    assert "class Invoice:" in inv.__source__
    assert '"""An invoice document to be understood."""' in inv.__source__


def test_source_shows_what_is_not_yet_determined():
    _, inv, engine, _ = invoice()
    text = inv.__source__
    assert "total = ...  # not yet determined" in text
    assert engine.call_count == 0
