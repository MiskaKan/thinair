"""Routing, escalation, and the engine layer.

The router is list order and nothing else, and the whole of it has to work
against scripted engines: not one test here opens a socket.
"""

from __future__ import annotations

import json

import pytest

from thinair.beliefs import config_scope, human, model
from thinair.engine import (
    MAX_PARSE_RETRIES,
    Engine,
    EngineError,
    ParseFailure,
    complete_json,
    extract_json,
)
from thinair.engine.openai_compat import OpenAICompatEngine, engine_for
from thinair.ledger import Ledger
from thinair.models import ModelDef, resolve
from thinair.policy import Unresolvable
from thinair.thing import Thing

from fakes import FakeEngine, FlakyFakeEngine

SOURCE = "Widget 999.00\nShipping 250.50\nTotal 1249.50 EUR"


def invoice(*models_, ledger=None):
    # `ledger or Ledger()` would be a bug: an empty Ledger is falsy.
    ledger = Ledger() if ledger is None else ledger

    class Invoice(Thing):
        """An invoice document to be understood."""

        __beliefs__ = [*models_, human("jane")]
        source_text: str
        total = Thing(float, extracted_from="source_text", range=(0, 1e6))

    return Invoice(source_text=SOURCE, __ledger__=ledger)


# --------------------------------------------------------------------------
# the router is list order
# --------------------------------------------------------------------------

def test_the_first_generative_belief_proposes():
    first = FakeEngine([{"value": 1249.5, "p": 0.9}])
    second = FakeEngine([{"value": 999.0, "p": 0.99}])
    inv = invoice(model("a", engine=first), model("b", engine=second))
    assert +inv.total == 1249.5
    assert second.call_count == 0                    # the ladder waits its turn


def test_escalation_fires_on_veto_exhaustion_and_walks_to_the_next_member():
    a = FakeEngine([{"value": 9e9, "p": 0.99}])       # always out of range
    b = FakeEngine([{"value": 9e9, "p": 0.99}])
    c = FakeEngine([{"value": 1249.5, "p": 0.6}])
    inv = invoice(model("a", engine=a), model("b", engine=b), model("c", engine=c))
    total = inv.total
    assert (+total, ~total) == (1249.5, 0.6)
    assert (a.call_count, b.call_count, c.call_count) == (3, 2, 1)
    assert total.__opinion__.belief.startswith("model:c")


def test_the_ladder_ends_in_unresolvable_carrying_every_attempt():
    a = FakeEngine([{"value": 9e9, "p": 0.99}])
    b = FakeEngine([{"value": 8e9, "p": 0.99}])
    inv = invoice(model("a", engine=a), model("b", engine=b))
    with pytest.raises(Unresolvable) as caught:
        +inv.total
    assert (a.call_count, b.call_count) == (3, 2)     # 3 rounds then 2 more
    assert len(caught.value.attempts) == 5
    assert {attempt.route for attempt in caught.value.attempts} == \
        {model("a", engine=a).id, model("b", engine=b).id}


def test_every_round_is_recorded_permanently():
    a = FakeEngine([{"value": 9e9, "p": 0.99}])
    b = FakeEngine([{"value": 1249.5, "p": 0.6}])
    ledger = Ledger()
    inv = invoice(model("a", engine=a), model("b", engine=b), ledger=ledger)
    +inv.total
    proposals = [o for o in ledger.opinions(attr="total")
                 if o.belief.startswith("model:")]
    assert [o.value for o in proposals] == [9e9, 9e9, 9e9, 1249.5]
    assert not any(o.frozen for o in proposals)       # invariant 4


def test_no_belief_ever_reads_past_negotiations_from_the_ledger():
    """Beliefs see only the snapshot they are handed."""
    seen = []
    a = FakeEngine([{"value": 9e9, "p": 0.99}, {"value": 1249.5, "p": 0.7}])
    ledger = Ledger()

    class Watcher:
        id = "watcher:1"
        necessary = False
        proposes = False

        def __call__(self, e, attr):
            seen.append(len(list(e.__ledger__)))
            return None

    class Invoice(Thing):
        __beliefs__ = [model("a", engine=a), human("jane"), Watcher()]
        source_text: str
        total = Thing(float, extracted_from="source_text", range=(0, 1e6))

    +Invoice(source_text=SOURCE, __ledger__=ledger).total
    # the snapshot's slice grows by this derivation's own rounds, and by
    # nothing else -- it is never the whole ledger
    assert seen == sorted(seen)
    assert max(seen) < len(list(ledger))


def test_the_routed_head_leads_the_snapshot_panel():
    """Beliefs that read ``[0]`` stay correct without knowing routing exists."""
    heads = []
    a = FakeEngine([{"value": 9e9, "p": 0.99}])
    b = FakeEngine([{"value": 1249.5, "p": 0.6}])

    class Peeker:
        id = "peeker:1"
        necessary = False
        proposes = False

        def __call__(self, e, attr):
            heads.append(e.__beliefs__[0].id)
            return None

    class Invoice(Thing):
        __beliefs__ = [model("a", engine=a), model("b", engine=b),
                       human("jane"), Peeker()]
        source_text: str
        total = Thing(float, extracted_from="source_text", range=(0, 1e6))

    +Invoice(source_text=SOURCE, __ledger__=Ledger()).total
    assert heads[0].startswith("model:a") and heads[-1].startswith("model:b")


# --------------------------------------------------------------------------
# re-proposal prompts quote the failing belief's reason verbatim
# --------------------------------------------------------------------------

def test_a_re_proposal_quotes_the_objection_verbatim():
    engine = FakeEngine([{"value": 9e9, "p": 0.99}, {"value": 1249.5, "p": 0.7}])
    +invoice(model("a", engine=engine)).total
    retry = "\n".join(m["content"] for m in engine.calls[1]["messages"])
    assert "above the declared maximum" in retry
    assert "Attempt 1" in retry


def test_the_first_prompt_carries_no_objections():
    engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    +invoice(model("a", engine=engine)).total
    first = "\n".join(m["content"] for m in engine.calls[0]["messages"])
    assert "rejected" not in first and "Attempt" not in first


def test_the_prompt_carries_the_contract_and_the_source():
    engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    +invoice(model("a", engine=engine)).total
    text = "\n".join(m["content"] for m in engine.calls[0]["messages"])
    assert "source_text" in text and "1249.50" in text
    assert "float" in text


# --------------------------------------------------------------------------
# the engine protocol
# --------------------------------------------------------------------------

def test_a_fake_engine_satisfies_the_protocol():
    assert isinstance(FakeEngine(), Engine)


@pytest.mark.parametrize("text,expected", [
    ('{"value": 1, "p": 0.5}', {"value": 1, "p": 0.5}),
    ('```json\n{"value": 1, "p": 0.5}\n```', {"value": 1, "p": 0.5}),
    ('Sure! {"value": 1, "p": 0.5} Hope that helps.', {"value": 1, "p": 0.5}),
    ('[1, 2, 3]', [1, 2, 3]),
    ('  \n{"a": {"b": 1}}\n  ', {"a": {"b": 1}}),
])
def test_extract_json_finds_the_payload(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "{unclosed", "I refuse."])
def test_extract_json_raises_on_prose(text):
    with pytest.raises(ParseFailure):
        extract_json(text)


def test_complete_json_retries_with_parse_failure_feedback():
    """Three retries with parse-failure feedback."""
    engine = FlakyFakeEngine([{"value": 1249.5, "p": 0.9}], bad=2)
    payload, meta = complete_json(engine, [{"role": "user", "content": "go"}])
    assert payload == {"value": 1249.5, "p": 0.9}
    assert engine.call_count == 3
    assert meta["parse_attempts"] == 3


def test_complete_json_gives_up_after_the_ladder():
    engine = FlakyFakeEngine(bad=99)
    with pytest.raises(ParseFailure):
        complete_json(engine, [{"role": "user", "content": "go"}])
    assert engine.call_count == MAX_PARSE_RETRIES


def test_a_parse_failure_is_not_a_route_escalation():
    """Two different ladders: parse retry is the engine's, escalation the route's."""
    engine = FlakyFakeEngine([{"value": 1249.5, "p": 0.9}], bad=1)
    total = invoice(model("a", engine=engine)).total
    assert +total == 1249.5
    assert engine.call_count == 2                    # one retry, still route one


# --------------------------------------------------------------------------
# the transport, exercised without a socket
# --------------------------------------------------------------------------

class Recorder:
    """Captures the payload ``OpenAICompatEngine`` would have posted."""

    def __init__(self, reply=None):
        self.payloads = []
        self.reply = reply or {"choices": [{"message": {"content": '{"value": 1, "p": 0.5}'}}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def __call__(self, url, payload):
        self.payloads.append(payload)
        return self.reply


def transport(definition, recorder, **kwargs):
    engine = OpenAICompatEngine(model="probe", definition=definition,
                                base_url="http://localhost:1/v1", **kwargs)
    engine._post = recorder
    return engine


def test_the_transport_sends_a_json_schema_when_the_folder_says_so():
    definition = ModelDef(match=("probe",), structured_output="json_schema",
                          version="1", name="probe")
    recorder = Recorder()
    transport(definition, recorder).complete(
        [{"role": "user", "content": "hi"}],
        schema={"type": "object", "properties": {}})
    (payload,) = recorder.payloads
    assert payload["response_format"]["type"] == "json_schema"


def test_the_transport_falls_back_to_json_mode():
    definition = ModelDef(match=("probe",), structured_output="json_mode",
                          version="1", name="probe")
    recorder = Recorder()
    transport(definition, recorder).complete([{"role": "user", "content": "hi"}],
                                             schema={"type": "object"})
    (payload,) = recorder.payloads
    assert payload["response_format"] == {"type": "json_object"}


def test_a_prompted_folder_sends_no_response_format():
    definition = ModelDef(match=("probe",), structured_output="prompted",
                          version="1", name="probe")
    recorder = Recorder()
    transport(definition, recorder).complete([{"role": "user", "content": "hi"}],
                                             schema={"type": "object"})
    (payload,) = recorder.payloads
    assert "response_format" not in payload


def test_the_no_system_role_quirk_is_handled_by_the_transport():
    definition = ModelDef(match=("probe",), quirks=("no_system_role",),
                          version="1", name="probe")
    recorder = Recorder()
    transport(definition, recorder).complete([
        {"role": "system", "content": "you are one belief"},
        {"role": "user", "content": "what is total?"},
    ])
    (payload,) = recorder.payloads
    assert all(m["role"] != "system" for m in payload["messages"])
    assert "one belief" in payload["messages"][0]["content"]


def test_the_single_user_turn_quirk_collapses_the_conversation():
    definition = ModelDef(match=("probe",), quirks=("single_user_turn",),
                          version="1", name="probe")
    recorder = Recorder()
    transport(definition, recorder).complete([
        {"role": "system", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "three"},
    ])
    (payload,) = recorder.payloads
    assert len(payload["messages"]) == 1


def test_the_think_payload_rides_along():
    definition = ModelDef(match=("probe",),
                          think=dict(on={"enable_thinking": True},
                                     off={"enable_thinking": False}),
                          version="1", name="probe")
    recorder = Recorder()
    transport(definition, recorder).complete([{"role": "user", "content": "hi"}],
                                             think=True)
    (payload,) = recorder.payloads
    assert payload["enable_thinking"] is True


def test_request_extra_rides_along():
    definition = ModelDef(match=("probe",), request_extra={"top_k": 20},
                          version="1", name="probe")
    recorder = Recorder()
    transport(definition, recorder).complete([{"role": "user", "content": "hi"}])
    (payload,) = recorder.payloads
    assert payload["top_k"] == 20


def test_the_meta_records_what_layer_2_will_need():
    """Model id, temperature, token counts and latency, from day one."""
    definition = ModelDef(match=("probe",), version="1", name="probe")
    recorder = Recorder()
    _, meta = transport(definition, recorder).complete(
        [{"role": "user", "content": "hi"}], temperature=0.7)
    assert meta["model"] == "probe" and meta["temperature"] == 0.7
    assert meta["prompt_tokens"] == 10 and meta["completion_tokens"] == 5
    assert meta["latency"] >= 0 and meta["transport"] == "openai_compat"


def test_engine_for_dispatches_on_the_transport_field():
    definition = ModelDef(match=("probe",), version="1", name="probe")
    assert isinstance(engine_for("probe", definition), OpenAICompatEngine)

    exotic = ModelDef(match=("probe2",), version="1", name="probe2",
                      transport="carrier-pigeon")
    with pytest.raises(EngineError, match="carrier-pigeon"):
        engine_for("probe2", exotic)


# --------------------------------------------------------------------------
# configuration cascade
# --------------------------------------------------------------------------

def test_the_environment_supplies_defaults(monkeypatch):
    monkeypatch.setenv("THINAIR_MODEL", "qwen3-35b")
    assert model().model_name == "qwen3-35b"


def test_a_class_default_beats_the_environment(monkeypatch):
    monkeypatch.setenv("THINAIR_MODEL", "from-env")
    with config_scope(model="from-scope"):
        assert model().model_name == "from-scope"


def test_an_injected_engine_beats_configuration():
    engine = FakeEngine([{"value": 1249.5, "p": 0.9}])
    belief = model("a", engine=engine)
    assert belief.engine() is engine


def test_constructing_a_model_belief_opens_no_socket(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("socket!"))
    belief = model("qwen3-35b", think=True, temperature=0.9)
    assert belief.definition is resolve("qwen3-35b")


# --------------------------------------------------------------------------
# the guards before the socket
# --------------------------------------------------------------------------

def test_offline_guard_raises_before_any_socket(monkeypatch):
    """THINAIR_OFFLINE=1: an accidental consultation is a one-line stack
    trace, never a spend and never a hang."""
    monkeypatch.setenv("THINAIR_OFFLINE", "1")
    engine = OpenAICompatEngine(base_url="http://example.invalid/v1",
                                model="some-model")
    with pytest.raises(EngineError, match="THINAIR_OFFLINE"):
        engine._post("http://example.invalid/v1/chat/completions", {})


def test_an_unconfigured_default_model_fails_loudly(monkeypatch):
    """model 'default' against the fallback base URL is a misconfiguration,
    not a request: it raises with instructions instead of retrying three
    times against localhost."""
    monkeypatch.delenv("THINAIR_BASE_URL", raising=False)
    monkeypatch.delenv("THINAIR_OFFLINE", raising=False)
    engine = OpenAICompatEngine(model="default")
    assert not engine.configured
    with pytest.raises(EngineError, match="THINAIR_MODEL"):
        engine._post(engine.base_url + "/chat/completions", {})


def test_an_explicit_base_url_counts_as_configured(monkeypatch):
    monkeypatch.delenv("THINAIR_BASE_URL", raising=False)
    engine = OpenAICompatEngine(base_url="http://myserver:1234/v1",
                                model="default")
    assert engine.configured        # a single-model server may ignore the name


def test_the_generic_fallback_never_sends_response_format():
    """An unknown server given response_format may mangle or ignore it --
    the fallback def stays prompted; folders that know their server opt in."""
    from thinair.models import GENERIC

    assert resolve("some-model-nobody-has-heard-of") is GENERIC
    assert GENERIC.structured_output == "prompted"
    engine = OpenAICompatEngine(base_url="http://x/v1", model="mystery",
                                definition=GENERIC)
    assert engine._response_format({"type": "object"}) is None
