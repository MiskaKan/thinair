"""M1 -- the memory.  One record type, append-only, latest-frozen-wins, dump/load."""

import json

import pytest

from thinair.ledger import (
    Ledger, Opinion, default_ledger, normal_form, set_default_ledger,
    use_ledger, values_equal,
)


def op(**kw):
    base = dict(belief="b1", entity="e1", attr="total", value=1.0, p=0.5)
    base.update(kw)
    return Opinion(**base)


# -- invariant 1: one record shape ----------------------------------------

def test_ledger_accepts_exactly_one_record_type():
    led = Ledger()
    with pytest.raises(TypeError):
        led.add({"belief": "b1", "entity": "e1", "attr": "a", "value": 1, "p": 1.0})
    with pytest.raises(TypeError):
        led.add(("b1", "e1", "a", 1, 1.0, 1, False, {}))
    assert len(led) == 0


def test_every_kind_of_belief_lands_in_the_same_shape():
    led = Ledger()
    for belief in ("model:small-fast@v1", "verbatimBelief[source_text]", "human:jane",
                   "code:net_total@abc123", "fixture:test_x"):
        led.add(op(belief=belief))
    assert {type(o) for o in led} == {Opinion}
    assert len({tuple(o.to_json()) for o in led}) == 1  # identical field sets


def test_opinion_validates_its_fields():
    with pytest.raises(ValueError):
        op(p=1.5)
    with pytest.raises(ValueError):
        op(p=-0.1)
    with pytest.raises(TypeError):
        op(p=True)
    with pytest.raises(ValueError):
        op(belief="")
    with pytest.raises(ValueError):
        op(entity="")
    with pytest.raises(ValueError):
        op(attr="")
    with pytest.raises(TypeError):
        op(frozen=1)
    with pytest.raises(TypeError):
        op(meta=[])


def test_opinion_is_immutable_and_hashable():
    o = op()
    with pytest.raises(Exception):
        o.value = 2
    assert {o, o} == {o}


# -- the extractors -------------------------------------------------------

def test_extractors_on_opinion():
    o = op(value=1249.5, p=0.93)
    assert +o == 1249.5
    assert ~o == 0.93


# -- append-only, monotonic t ---------------------------------------------

def test_no_deletion_api():
    led = Ledger()
    for name in ("delete", "remove", "pop", "clear", "update", "purge"):
        assert not hasattr(led, name), f"ledger must not expose {name}()"


def test_t_is_a_monotonic_sequence_number_not_wallclock():
    led = Ledger()
    stamped = [led.add(op(value=i)) for i in range(5)]
    assert [o.t for o in stamped] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_explicit_t_advances_the_counter():
    led = Ledger()
    led.add(op(t=100.0))
    assert led.add(op()).t == 101.0


def test_opinions_come_back_in_t_order():
    led = Ledger()
    led.add(op(belief="b2", t=9.0))
    led.add(op(belief="b1", t=3.0))
    assert [o.belief for o in led.opinions(entity="e1", attr="total")] == ["b1", "b2"]


# -- querying -------------------------------------------------------------

def test_query_filters():
    led = Ledger()
    led.add(op(belief="b1", attr="total"))
    led.add(op(belief="b2", attr="total", frozen=True))
    led.add(op(belief="b1", attr="vendor", entity="e2"))
    assert len(led.opinions()) == 3
    assert len(led.opinions(attr="total")) == 2
    assert len(led.opinions(belief="b1")) == 2
    assert len(led.opinions(entity="e2")) == 1
    assert len(led.opinions(frozen=True)) == 1
    assert led.opinions(entity="e1", attr="total", belief="b2")[0].frozen


def test_latest_frozen_wins():
    led = Ledger()
    led.add(op(belief="model", value=1.0, p=0.9))
    led.add(op(belief="human:jane", value=2.0, p=1.0, frozen=True))
    led.add(op(belief="model", value=3.0, p=0.8))
    led.add(op(belief="code:f@aa", value=4.0, p=1.0, frozen=True))
    led.add(op(belief="model", value=5.0, p=0.7))
    assert +led.latest_frozen("e1", "total") == 4.0
    assert +led.latest("e1", "total") == 5.0
    assert +led.latest("e1", "total", belief="model") == 5.0
    assert led.latest_frozen("e1", "nope") is None


def test_recurring_opinions_from_one_belief_keep_the_latest_current():
    led = Ledger()
    led.add(op(belief="model", value=1.0))
    led.add(op(belief="model", value=2.0))
    assert len(led.opinions(belief="model")) == 2
    assert +led.latest("e1", "total", belief="model") == 2.0


# -- persistence ----------------------------------------------------------

def test_dump_load_round_trips(tmp_path):
    led = Ledger()
    led.add(op(value={"a": [1, 2, {"b": "c"}]}, meta={"reason": "why not"}))
    led.add(op(belief="human:jane", value="ACME Oy", p=1.0, frozen=True))
    path = tmp_path / "l.json"
    led.dump(path)
    back = Ledger.load(path)
    assert [o.to_json() for o in back] == [o.to_json() for o in led]
    assert back.next_t() == led.next_t()


def test_load_rejects_foreign_blobs(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"thinair": 1, "kind": "thing"}))
    with pytest.raises(ValueError):
        Ledger.load(path)
    path.write_text(json.dumps({"thinair": 99, "kind": "ledger", "opinions": []}))
    with pytest.raises(ValueError):
        Ledger.load(path)


# -- values_equal ---------------------------------------------------------

EQUAL = [
    (1, 1.0), (1.0, 1.0 + 1e-12), ("Hello  World", "hello world"),
    (" a\tb\n", "a b"), ([1, 2], (1, 2)), ({"a": 1}, {"A": 1.0}),
    ({"a": [1, {"b": "X"}]}, {"a": [1.0, {"b": "x"}]}),
    (None, None), (True, True), ({1, 2}, {2, 1}),
]
UNEQUAL = [
    (1, 2), (True, 1), (1, True), (False, 0), ("a", "b"), ([1], [1, 2]),
    ({"a": 1}, {"a": 2}), ({"a": 1}, {"b": 1}), (None, 0), (None, ""),
    (1.0, 1.0 + 1e-6), ({1, 2}, {1, 3}),
]


@pytest.mark.parametrize("a,b", EQUAL)
def test_values_equal_positives(a, b):
    assert values_equal(a, b) and values_equal(b, a)


@pytest.mark.parametrize("a,b", UNEQUAL)
def test_values_equal_negatives(a, b):
    assert not values_equal(a, b) and not values_equal(b, a)


@pytest.mark.parametrize("v", [1, 1.0, "a", None, True, [1, "a"], {"a": 1}, {1, 2}])
def test_values_equal_is_reflexive_and_normal_form_agrees(v):
    assert values_equal(v, v)
    assert normal_form(v) == normal_form(v)
    assert hash(normal_form(v)) == hash(normal_form(v))


def test_normal_form_is_hashable_for_nested_values():
    assert hash(normal_form({"a": [1, {"b": (2, 3)}]}))


# -- the default ledger ---------------------------------------------------

def test_default_ledger_is_swappable_and_scoped():
    original = default_ledger()
    mine = Ledger()
    with use_ledger(mine):
        assert default_ledger() is mine
    assert default_ledger() is original
    with pytest.raises(TypeError):
        set_default_ledger(object())
