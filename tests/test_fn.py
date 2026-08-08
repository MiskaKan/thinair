"""Functions as cells: every guarantee, one per test."""

from __future__ import annotations

import pytest

from thinair import Thing, contract, human
from thinair.beliefs import model
from thinair.fn import Fn, call_id, fn, freeze_call
from thinair.ledger import Ledger
from thinair.policy import Unresolvable

from fakes import FakeEngine


@pytest.fixture
def ledger():
    return Ledger()


# --------------------------------------------------------------------------
# call identity
# --------------------------------------------------------------------------

def test_a_call_names_a_cell():
    assert call_id("net_total", (100, 0.24)) == "fn:net_total(100, 0.24)"


def test_call_ids_normalize_their_arguments():
    assert call_id("f", (1,)) == call_id("f", (1.0,))
    assert call_id("f", (), {"b": 1, "a": 2}) == call_id("f", (), {"a": 2, "b": 1})


def test_a_call_id_is_implementation_free():
    """No version, no belief: code and model land on the same cell."""
    identity = call_id("net_total", (100, 0.24))
    assert "code:" not in identity and "model:" not in identity
    assert "@" not in identity


def test_a_non_json_able_argument_is_a_type_error():
    with pytest.raises(TypeError, match="JSON-able"):
        call_id("f", (object(),))


def test_a_thing_argument_contributes_its_entity_and_state():
    class Doc(Thing):
        __beliefs__ = [human("jane")]
        text = contract(str)

    doc = Doc(text="hello", __ledger__=Ledger())
    identity = call_id("summarize", (doc,))
    assert doc.__entity__ in identity

    doc.text = "goodbye"                             # the state changed
    assert call_id("summarize", (doc,)) != identity


# --------------------------------------------------------------------------
# body + pure=True: code is a belief
# --------------------------------------------------------------------------

def coded(ledger):
    @fn(returns=float, pure=True, ledger=ledger)
    def net_total(gross: float, vat: float) -> float:
        """Total with VAT removed."""
        return gross / (1 + vat)

    return net_total


def test_a_coded_pure_call_freezes_its_result(ledger):
    net_total = coded(ledger)
    result = net_total(1249.50, 0.24)
    assert +result == pytest.approx(1007.66, abs=0.01)
    assert ~result == 1.0
    assert result.__opinion__.frozen is True
    assert result.__opinion__.belief == f"code:{net_total.qualname}@" \
        + result.__opinion__.belief.split("@")[-1]
    assert result.__opinion__.belief.startswith("code:")


def test_memoization_is_the_frozen_short_circuit(ledger):
    calls = []

    @fn(returns=float, pure=True, ledger=ledger)
    def counted(x: int) -> int:
        """Counts how often it really ran."""
        calls.append(x)
        return x * 2

    assert +counted(21) == 42
    assert +counted(21) == 42
    assert +counted(21) == 42
    assert calls == [21]


def test_a_coded_pure_call_reaches_no_engine(ledger):
    net_total = coded(ledger)
    +net_total(100, 0.24)
    assert not [b for b in net_total.beliefs() if getattr(b, "proposes", False)]


def test_editing_the_body_re_freezes_the_cell(ledger):
    @fn(returns=float, pure=True, ledger=ledger)
    def rate(x: float) -> float:
        """A rate."""
        return x * 2

    first = rate(10)
    assert +first == 20

    @fn(returns=float, pure=True, ledger=ledger)
    def rate(x: float) -> float:                     # noqa: F811 - the edit
        """A rate."""
        return x * 3

    second = rate(10)
    assert +second == 30                             # latest frozen wins
    frozen = ledger.opinions(entity=call_id(rate.qualname, (10,)), frozen=True)
    assert [o.value for o in frozen] == [20, 30]     # every predecessor survives
    assert len({o.belief for o in frozen}) == 2      # a different source hash


# --------------------------------------------------------------------------
# no body: an imagined function
# --------------------------------------------------------------------------

def imagined(ledger, script, **options):
    engine = FakeEngine(script)

    @fn(returns=float, models=[model("m", engine=engine)], ledger=ledger,
        **options)
    def net_total(gross: float, vat: float) -> float:
        """Total with VAT removed."""
        ...

    return net_total, engine


def test_a_bodiless_fn_resolves_through_the_engine(ledger):
    net_total, engine = imagined(ledger, [{"value": 1007.66, "p": 0.7}])
    result = net_total(1249.50, 0.24)
    assert (+result, ~result) == (1007.66, 0.7)
    assert engine.call_count == 1


def test_a_bodiless_result_is_never_frozen(ledger):
    """Invariant 4: pin it with ``freeze`` if you choose."""
    net_total, _ = imagined(ledger, [{"value": 1007.66, "p": 0.7}])
    assert net_total(1249.50, 0.24).__opinion__.frozen is False


def test_a_bodiless_fn_is_contract_checked(ledger):
    net_total, engine = imagined(ledger, [{"value": "not a number", "p": 0.9},
                                          {"value": 1007.66, "p": 0.6}])
    result = net_total(1249.50, 0.24)
    assert +result == 1007.66
    assert engine.call_count == 2


def test_a_bodiless_fn_that_never_conforms_is_unresolvable(ledger):
    net_total, _ = imagined(ledger, [{"value": "not a number", "p": 0.9}])
    with pytest.raises(Unresolvable):
        net_total(1249.50, 0.24)


def test_the_call_site_cannot_tell_the_difference(ledger):
    net_total, _ = imagined(ledger, [{"value": 1007.66, "p": 0.7}])
    other = coded(Ledger())
    for candidate in (net_total, other):
        result = candidate(1249.50, 0.24)
        assert +result == pytest.approx(1007.66, abs=0.01)
        assert 0.0 <= ~result <= 1.0


def test_the_docstring_and_signature_reach_the_prompt(ledger):
    net_total, engine = imagined(ledger, [{"value": 1007.66, "p": 0.7}])
    net_total(1249.50, 0.24)
    prompt = "\n".join(m["content"] for m in engine.calls[0]["messages"])
    assert "Total with VAT removed" in prompt
    assert "1249.5" in prompt and "0.24" in prompt


# --------------------------------------------------------------------------
# both: competing implementations over the same cells
# --------------------------------------------------------------------------

def test_code_and_model_opinions_about_one_cell_coexist_queryably(ledger):
    engine = FakeEngine([{"value": 1000.0, "p": 0.6}])

    @fn(returns=float, models=[model("m", engine=engine)], ledger=ledger)
    def net_total(gross: float, vat: float) -> float:
        """Total with VAT removed."""
        ...

    +net_total(1249.50, 0.24)

    @fn(returns=float, pure=True, ledger=ledger)
    def net_total(gross: float, vat: float) -> float:  # noqa: F811 - the body arrives
        """Total with VAT removed."""
        return gross / (1 + vat)

    +net_total(1249.50, 0.24)

    cell = call_id(net_total.qualname, (1249.50, 0.24))
    voices = {o.belief.split("@")[0].split("[")[0]
              for o in ledger.opinions(entity=cell, attr="result")}
    assert any(v.startswith("model:") for v in voices)
    assert any(v.startswith("code:") for v in voices)


def test_migration_runs_in_both_directions(ledger):
    """Start bodiless, write the body later -- no API change."""
    engine = FakeEngine([{"value": 1000.0, "p": 0.6}])
    bodiless, _ = imagined(ledger, [{"value": 1000.0, "p": 0.6}])
    assert ~bodiless(100, 0.1) == 0.6

    with_body = coded(ledger)
    assert ~with_body(100, 0.1) == 1.0


def test_code_may_fall_back_to_the_model_when_it_raises(ledger):
    engine = FakeEngine([{"value": 0.0, "p": 0.3}])

    @fn(returns=float, models=[model("m", engine=engine)], ledger=ledger,
        fallback=True)
    def divide(a: float, b: float) -> float:
        """a over b."""
        return a / b

    assert +divide(1.0, 2.0) == 0.5 and engine.call_count == 0
    assert +divide(1.0, 0.0) == 0.0 and engine.call_count == 1


def test_without_fallback_code_raising_is_just_code_raising(ledger):
    @fn(returns=float, pure=True, ledger=ledger)
    def divide(a: float, b: float) -> float:
        """a over b."""
        return a / b

    with pytest.raises(ZeroDivisionError):
        divide(1.0, 0.0)


# --------------------------------------------------------------------------
# purity is declared, never assumed
# --------------------------------------------------------------------------

def test_an_impure_fn_mints_a_fresh_cell_every_time(ledger):
    seen = iter([1.10, 1.11, 1.12])

    @fn(returns=float, pure=False, ledger=ledger)
    def fetch_rate(pair: str) -> float:
        """A sensor reading."""
        return next(seen)

    readings = [+fetch_rate("EURUSD") for _ in range(3)]
    assert readings == [1.10, 1.11, 1.12]            # never served from cache
    cells = {c[0] for c in ledger.cells()}
    assert len(cells) == 3


def test_a_sensor_reading_is_tagged_observed(ledger):
    @fn(returns=float, pure=False, ledger=ledger)
    def now() -> float:
        """A sensor reading."""
        return 1.0

    assert now().__opinion__.meta["observed"] is True


def test_a_pure_reading_is_not_tagged_observed(ledger):
    net_total = coded(ledger)
    assert "observed" not in net_total(100, 0.24).__opinion__.meta


# --------------------------------------------------------------------------
# fixtures and mocks are frozen opinions
# --------------------------------------------------------------------------

def test_a_fixture_is_a_frozen_opinion_that_the_next_call_serves(ledger):
    net_total, engine = imagined(ledger, [{"value": 999.0, "p": 0.1}])
    freeze_call(net_total, (1249.50, 0.24), value=1007.66)
    result = net_total(1249.50, 0.24)
    assert (+result, ~result) == (1007.66, 1.0)
    assert engine.call_count == 0


def test_a_fixture_overrides_even_a_body(ledger):
    net_total = coded(ledger)
    freeze_call(net_total, (100, 0.0), value=42.0)
    assert +net_total(100, 0.0) == 42.0


def test_a_fixture_is_authored_and_visible(ledger):
    net_total, _ = imagined(ledger, [{"value": 1.0, "p": 0.1}])
    opinion = freeze_call(net_total, (1249.50, 0.24), value=1007.66)
    assert opinion.belief == f"fixture:{net_total.qualname}"
    assert opinion.frozen is True


def test_a_mock_is_gone_when_its_ledger_is(ledger):
    """No mocking framework -- the ledger is the mocking framework."""
    scoped = Ledger()
    net_total = coded(ledger)
    freeze_call(net_total, (100, 0.0), value=42.0, ledger=scoped)
    assert +net_total(100, 0.0) == 100.0             # the real ledger is untouched
    assert len(scoped.opinions(frozen=True)) == 1


def test_freeze_call_works_on_an_undecorated_function(ledger):
    def not_decorated(a, b):
        return a + b

    opinion = freeze_call(not_decorated, (1, 2), value=99, ledger=ledger)
    assert opinion.entity == call_id(not_decorated.__qualname__, (1, 2))


# --------------------------------------------------------------------------
# episodes gain nothing new
# --------------------------------------------------------------------------

def test_an_episode_can_call_a_pure_fn_through_the_same_cell(ledger):
    net_total = coded(ledger)
    engine = FakeEngine([
        {"action": "call", "method": "net", "args": [1249.50, 0.24]},
        {"action": "return", "changes": {}, "value": "worked it out", "p": 0.8},
    ])

    class Invoice(Thing):
        """An invoice."""

        __beliefs__ = [model("m", engine=engine), human("jane")]
        total = contract(float)

        def net(self, gross, vat):
            """A real method that consults a pure @fn."""
            return +net_total(gross, vat)

    inv = Invoice(total=1249.50, __ledger__=ledger)
    assert +inv.explain() == "worked it out"
    observation = engine.calls[-1]["messages"][-1]["content"]
    assert "1007" in observation


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------

def test_fn_preserves_the_wrapped_functions_identity(ledger):
    net_total = coded(ledger)
    assert isinstance(net_total, Fn)
    assert net_total.__name__ == "net_total"
    assert net_total.__doc__.startswith("Total with VAT")


def test_fn_can_be_used_bare(ledger):
    @fn
    def double(x: int) -> int:
        """Twice x."""
        return x * 2

    assert +double(21) == 42


def test_defaults_are_part_of_the_cell(ledger):
    @fn(pure=True, ledger=ledger)
    def greet(name: str, greeting: str = "hello") -> str:
        """A greeting."""
        return f"{greeting}, {name}"

    assert greet("world").__entity__ == greet("world", "hello").__entity__


def test_an_episode_may_call_a_pure_fn_by_name(ledger):
    """'Tool calling' is not a protocol -- it is resolving a cell."""
    the_fn = coded(ledger)
    engine = FakeEngine([
        {"action": "call", "method": "net_total", "args": [1249.50, 0.24]},
        {"action": "return", "changes": {}, "value": "worked it out", "p": 0.8},
    ])

    class Invoice(Thing):
        """An invoice."""

        __beliefs__ = [model("m", engine=engine), human("jane")]
        total = contract(float)
        net_total = the_fn                           # offered to the episode

    inv = Invoice(total=1249.50, __ledger__=ledger)
    assert +inv.explain() == "worked it out"
    assert "1007" in engine.calls[-1]["messages"][-1]["content"]
    # the call landed on the ordinary cell, with code's frozen opinion on it
    pinned = ledger.latest_frozen(call_id(the_fn.qualname, (1249.50, 0.24)),
                                  "result")
    assert pinned is not None and pinned.belief.startswith("code:")
