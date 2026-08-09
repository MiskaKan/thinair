"""thinair.store: the default ledger made durable.

The store's one promise is that it is the same Ledger with a different
home -- so most tests here are parity tests against the in-memory base, and
the acceptance test is the news record round-tripping bit-for-bit.  The
suite runs with ``THINAIR_STORE=off`` (conftest); every store here is
explicit and lives in ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from fakes import FakeEngine
from thinair import Thing, contract, human, model, store
from thinair.ledger import Ledger, Opinion
from thinair.store import DEFAULT_PATH, SqliteLedger

HERE = pathlib.Path(__file__).parent
NEWS_LEDGER = HERE.parent / "experiments" / "news" / "ledger.json"


def sample_opinions():
    return [
        Opinion(belief="model:m", entity="e1", attr="size", value=4, p=0.9,
                meta={"model": "m", "exposure": "abc123def456"}),
        Opinion(belief="law:x", entity="e1", attr="size", value=4, p=0.2,
                meta={"judged": "law:x", "reason": "too big"}),
        Opinion(belief="human:jane", entity="e1", attr="kind", value="a",
                p=1.0, frozen=True, meta={"assigned": True}),
        Opinion(belief="model:m", entity="e2", attr="size",
                value={"nested": [1, 2.5, None]}, p=0.5, meta={"model": "m"}),
    ]


def both(tmp_path):
    plain, durable = Ledger(), SqliteLedger(tmp_path / "o.db")
    for o in sample_opinions():
        plain.add(o)
        durable.add(o)
    return plain, durable


# --------------------------------------------------------------------------
# parity: the same Ledger with a different home
# --------------------------------------------------------------------------

def test_the_whole_read_api_agrees_with_the_in_memory_ledger(tmp_path):
    plain, durable = both(tmp_path)
    assert len(durable) == len(plain)
    assert durable.to_json() == plain.to_json()          # values, meta, t, order
    assert durable.cells() == plain.cells()
    assert durable.beliefs() == plain.beliefs()
    assert durable.opinions(entity="e1", attr="size") == \
        plain.opinions(entity="e1", attr="size")
    assert durable.opinions(belief="model:m") == plain.opinions(belief="model:m")
    assert durable.opinions(frozen=True) == plain.opinions(frozen=True)
    assert durable.latest("e1", "size").p == 0.2
    assert durable.latest_frozen("e1", "kind").value == "a"


def test_the_store_accepts_exactly_one_record_type(tmp_path):
    durable = SqliteLedger(tmp_path / "o.db")
    with pytest.raises(TypeError):
        durable.add({"belief": "b"})


def test_stamping_matches_the_base_ledger(tmp_path):
    durable = SqliteLedger(tmp_path / "o.db")
    first = durable.add(Opinion(belief="b", entity="e", attr="a", value=1, p=0.5))
    second = durable.add(Opinion(belief="b", entity="e", attr="a", value=2, p=0.5))
    assert (first.t, second.t) == (1.0, 2.0)
    stamped = durable.add(Opinion(belief="b", entity="e", attr="a", value=3,
                                  p=0.5, t=99.0))
    assert stamped.t == 99.0                             # a given t is kept
    assert durable.next_t() == 100.0


# --------------------------------------------------------------------------
# relaunch: the tape survives the process
# --------------------------------------------------------------------------

def test_a_reopened_store_continues_the_same_tape(tmp_path):
    path = tmp_path / "o.db"
    first = SqliteLedger(path)
    for o in sample_opinions():
        first.add(o)
    first.close()

    relaunched = SqliteLedger(path)
    assert len(relaunched) == 4
    added = relaunched.add(Opinion(belief="b", entity="e9", attr="a",
                                   value=1, p=0.5))
    assert added.t == 5.0                                # continues, not restarts


def test_frozen_state_survives_relaunch_at_zero_calls(tmp_path):
    """The single-source-of-truth promise, scoped honestly: everything
    frozen -- assignments, code, fixtures -- short-circuits from the record
    on relaunch; believed cells re-derive unless the runner skips them."""
    path = tmp_path / "o.db"
    engine = FakeEngine([{"value": 12.5, "p": 0.9}])

    class Item(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        total = contract(float)

    run1 = Item(__entity__="item-1", __ledger__=SqliteLedger(path))
    run1.kind = "invoice"                                # frozen assignment
    assert +run1.total == 12.5
    assert engine.call_count == 1

    # "launch the app again": fresh process state, same store, same entity
    run2 = Item(__entity__="item-1", __ledger__=SqliteLedger(path))
    assert +run2.kind == "invoice"                       # from the record
    assert engine.call_count == 1                        # and it cost nothing


# --------------------------------------------------------------------------
# lazily on disk, and only on write
# --------------------------------------------------------------------------

def test_reading_a_store_that_does_not_exist_creates_nothing(tmp_path):
    durable = SqliteLedger(tmp_path / "deep" / "o.db")
    assert durable.opinions() == [] and len(durable) == 0
    assert durable.cells() == [] and durable.beliefs() == []
    assert durable.latest("e", "a") is None
    assert not (tmp_path / "deep").exists()


def test_the_first_append_creates_the_folder(tmp_path):
    durable = SqliteLedger(tmp_path / ".thinair" / "opinions.db")
    durable.add(Opinion(belief="b", entity="e", attr="a", value=1, p=0.5))
    assert (tmp_path / ".thinair" / "opinions.db").exists()


# --------------------------------------------------------------------------
# the default, and the off switch
# --------------------------------------------------------------------------

def test_install_respects_the_off_switch_and_the_path(tmp_path, monkeypatch):
    try:
        monkeypatch.setenv("THINAIR_STORE", "off")
        assert type(store.install()) is Ledger            # plain, in-memory

        monkeypatch.setenv("THINAIR_STORE", str(tmp_path / "here.db"))
        installed = store.install()
        assert isinstance(installed, SqliteLedger)
        assert installed.path == str(tmp_path / "here.db")

        monkeypatch.delenv("THINAIR_STORE")
        assert store.install(tmp_path / "there.db").path == \
            str(tmp_path / "there.db")
        assert DEFAULT_PATH == os.path.join(".thinair", "opinions.db")
    finally:
        store.disable()                                   # leave the suite hermetic


# --------------------------------------------------------------------------
# acceptance: the archived record is the store's equal
# --------------------------------------------------------------------------

def test_the_news_record_round_trips_through_the_store(tmp_path):
    """Archive -> store -> export is identity: the database is derived
    structure, rebuildable from the committed evidence at any time."""
    src = Ledger.load(NEWS_LEDGER)
    durable = SqliteLedger(tmp_path / "o.db")
    durable.extend(list(src))
    assert durable.to_json() == src.to_json()

    from thinair.evaluate import tiers
    flash = "model:deepseek-v4-flash"
    ids = [b for b in src.beliefs() if b.startswith("model:")]
    validation = lambda o: o.entity.endswith("~twin") or o.entity.startswith("pair:")  # noqa: E731
    assert tiers(durable, validation, proposers=ids) == \
        tiers(src, validation, proposers=ids)
    assert any(b.startswith(flash) for b in ids)
