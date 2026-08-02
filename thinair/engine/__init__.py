"""Engines: everything model-specific, in one subclassable place.

`Engine` is the whole of it — the prompts, the sampling table, the reply
envelopes, the parsing and forgiveness, the two-tier ladder mechanics,
and the transport (an OpenAI-compatible /chat/completions endpoint). What
is left in the base class is generic; what a particular model family
wants differently — its default model name, the knob that turns its
thinking off, its recommended sampling — lives in a subpackage beside
this one and registers itself here.

Adding a family is three things:

    thinair/engine/<family>/__init__.py   a folder
    class <Family>Engine(Engine): ...     a subclass
    register(<Family>Engine)              a registry line at the bottom

`register` teaches `resolve()` to pick that class when the model name
matches (`model_names`, or an overridden `handles()`); an unrecognized
name falls back to the generic base `Engine`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

from .._shared import (
    _DEFAULT_API_KEY,
    _DEFAULT_BASE_URL,
    _DEFAULT_MAX_TOKENS,
    _debug_box,
    _debugging,
    _meta_line,
    _op_debug,
    _op_stack,
    _render_schema,
    _snip,
    _told_event_line,
)


class _LoopWatch:
    """Online verbatim-repetition detector: the streaming twin of
    `_prune_repetition`, same thresholds. Feed it text as it arrives; it
    trips the moment any long-enough line has been seen three times —
    nonsense is stopped when it starts, not discovered 32k tokens later."""

    def __init__(self):
        self.counts = {}
        self.buffer = ""
        self.tripped = False

    def feed(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            key = line.strip()
            if len(key) < 10:
                continue  # short structural lines ("},", "],") repeat honestly
            self.counts[key] = self.counts.get(key, 0) + 1
            if self.counts[key] >= 3:
                self.tripped = True
        return self.tripped


class _StreamPrinter:
    """Live token rendering: the thinking tier's stream, printed into the
    operation's open debug box as it arrives, channel by channel."""

    def __init__(self, st=None):
        self.pad = "  " * len(_op_stack.get())
        self.step = f"step {st['round']} · " if st and st.get("plan") else ""
        self.section = None
        self.midline = False

    def breakline(self):
        if self.midline:
            sys.stderr.write("\n")
            self.midline = False

    def note(self, text):
        """A `├─` divider naming an event (a cut, a carry, a grant)."""
        self.breakline()
        sys.stderr.write(f"{self.pad}├─ {text} " + "─" * max(1, 56 - len(text)) + "\n")
        sys.stderr.flush()
        self.section = None

    def write(self, section, text):
        if section != self.section:
            label = section if self.section == "thoughts" else (
                f"<<< {self.step}{section} (streaming)"
            )
            self.note(label)
            self.section = section
        if not text:
            return
        if not self.midline:
            sys.stderr.write(f"{self.pad}│ ")
            self.midline = True
        sys.stderr.write(text.replace("\n", f"\n{self.pad}│ "))
        sys.stderr.flush()

    def finish(self, footer, close=True):
        self.breakline()
        if close:
            sys.stderr.write(
                f"{self.pad}└─ {footer}\n" if footer else self.pad + "└" + "─" * 59 + "\n"
            )
        elif footer:
            sys.stderr.write(f"{self.pad}├─ {footer}\n")
        sys.stderr.flush()
        self.section = None


class Engine:
    """Everything model-specific, in one subclassable place: the prompts,
    the sampling table, the reply envelopes, the parsing and forgiveness,
    the two-tier ladder mechanics, and the transport (an OpenAI-compatible
    /chat/completions endpoint by default). `Thing` holds the object
    model — story, strata, plan runtime, persistence — and asks its
    engine for the rest, so adapting thinair to another model means
    subclassing this and overriding prompt methods or `temperatures`,
    never touching `Thing`. Reached as `Thing.Engine`.

    The base class itself is family-neutral: it speaks plain
    OpenAI-compatible JSON and adds no sampling of its own. A family
    subclass names its `default_model`, the `model_names` that select it,
    and the payload knobs of the two tiers (`direct_payload`,
    `thinking_payload`) — see `thinair.engine.qwen`."""

    # -- the family's identity: what this engine drives, and by default --
    default_model = None   # the family's own model name, if it has one
    model_names = ()       # substrings that select it, case-insensitively

    # per-operation sampling; judgments and plans run cool, reads run
    # warm so an open prior actually gets sampled
    temperatures = {
        "read": 0.8,
        "judge": 0.3,
        "collapse": 0.3,
        "plan": 0.3,
        "condense": 0.3,
    }
    narrates = True  # complete() draws its own debug boxes

    def __init__(
        self,
        base_url=_DEFAULT_BASE_URL,
        api_key=_DEFAULT_API_KEY,
        model=None,
        max_tokens=_DEFAULT_MAX_TOKENS,
        request_extra=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = _DEFAULT_MODEL if model is None else model
        self.max_tokens = int(max_tokens)
        self.request_extra = dict(request_extra or {})
        self.last_meta = {}
        # optimistic capabilities; each is dropped if the server rejects it
        self._json_mode = True      # response_format json_object
        self._schema_mode = True    # response_format json_schema (direct tier)
        self._think_toggle = True   # the direct tier is real (direct_payload)
        self._stream_mode = True    # SSE streaming (thinking tier)

    @classmethod
    def handles(cls, model):
        """True when this family should drive `model`. The default is a
        case-insensitive substring match against `model_names`; a family
        that needs a real predicate overrides this."""
        name = str(model or "").lower()
        return any(token in name for token in cls.model_names)

    @property
    def tiers(self):
        """True when the thinking tier is genuinely distinct from the
        direct tier — the ladder only escalates where that is real."""
        return self._think_toggle

    # -- the family's knobs: how each tier is sampled ----------------------

    def direct_payload(self, payload):
        """Family knobs for the direct tier (thinking off): the request
        body is handed over for mutation, in place, right after the
        common fields. The base adds nothing — a plain server simply
        answers single-shot."""

    def thinking_payload(self, payload):
        """Family knobs for the thinking tier, applied the same way."""

    # -- the model's voice: prompts, envelopes, parsing --------------------

    def read_prompt(self, story, name):
        """-> (messages, value_schema) for inferring one attribute."""
        hint = ""
        value_schema = None
        if re.match(r"(is|can|has|should|does|was|will)_", name):
            hint = " The attribute name suggests a boolean."
            value_schema = bool
        return [
            {
                "role": "system",
                "content": (
                    "Infer the value of one attribute of a Python object from "
                    "the object's data below. Reply with exactly one JSON "
                    'object: {"value": <json>, "confidence": <0..1>}. '
                    "Confidence is the probability the value is correct. Use "
                    "natural JSON types (numbers as numbers, booleans as "
                    "booleans). A random outcome (a roll, a draw): pick one "
                    "result, confidence is its probability. Distinguish "
                    "unknown from absent: when the attribute surely has a "
                    "value that is merely unstated (a color, a serial number "
                    "nobody mentioned), ALWAYS commit to one concrete, "
                    "plausible value with honestly low confidence — never "
                    "null, never a refusal. null means the object truly HAS "
                    "no such value (the trailer of a car that tows none), "
                    "held with real confidence. Never placeholder strings "
                    'like "unknown". Never contradict the object\'s data. '
                    "Example replies:\n"
                    '{"value": "silver", "confidence": 0.3}\n'
                    '{"value": null, "confidence": 0.9}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT:\n{story}\n\n"
                    f"What is the value of the attribute `{name}`?{hint}"
                ),
            },
        ], value_schema

    def compare_prompt(self, left, right, symbol):
        """Messages for judging an ordering comparison between stories."""
        return [
            {
                "role": "system",
                "content": (
                    "Decide the comparison below between two objects. Reply "
                    "with exactly one JSON object: "
                    '{"value": true|false, "confidence": <0..1>}. Compare what '
                    "the objects ARE (size, cost, capability, whatever their "
                    "data makes relevant), not how they are spelled, unless "
                    "both are plain text. If several readings disagree, pick "
                    "the most likely and lower your confidence. Never "
                    "contradict the objects' data. Example reply:\n"
                    '{"value": false, "confidence": 0.85}'
                ),
            },
            {
                "role": "user",
                "content": f"LEFT:\n{left}\n\nRIGHT:\n{right}\n\nIs LEFT {symbol} RIGHT?",
            },
        ]

    def collapse_prompt(self, story, schema):
        """Messages for resolving the single value a story stands for."""
        demand = (
            f" The value MUST be a JSON {_render_schema(schema)}."
            if schema is not None
            else ""
        )
        return [
            {
                "role": "system",
                "content": (
                    "Give the single JSON value the object below most "
                    "naturally represents. Reply with exactly one JSON "
                    'object: {"value": <json>, "confidence": <0..1>}. A '
                    'described value represents itself ("Cat" gives "Cat"). '
                    "Use natural JSON types. Never contradict the object's "
                    'data. Example reply:\n{"value": "Cat", "confidence": 0.97}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT:\n{story}\n\n"
                    f"What single value does this object represent?{demand}"
                ),
            },
        ]

    def plan_prompt(self, story, call_repr, name, stack, returns_schema):
        """Messages that open an imagined method call's action loop."""
        return [
            {
                "role": "system",
                "content": (
                    "You produce the outcome of one imagined method call on a "
                    "Python object. The method has no written code; you act it "
                    "out step by step, using the object's data below.\n"
                    "Reply with exactly one JSON object per turn, one of:\n"
                    '{"action": "get", "name": "<attr>"}\n'
                    '{"action": "set", "name": "<attr>", "value": <json>, "confidence": <0..1>}\n'
                    '{"action": "delete", "name": "<attr>"}\n'
                    '{"action": "call", "name": "<method>", "args": [<json>...]}\n'
                    '{"action": "define", "name": "<name>", "meaning": "<text>"}\n'
                    '{"action": "return", "value": <json>, "confidence": <0..1>}\n'
                    "Facts: `call` runs the object's real, written methods. "
                    "Never write code. Written state is certain and "
                    "read-only; record changes to it under new names. "
                    "Imagined state and open slots are yours to set, change, "
                    "or delete. `return` is the only way "
                    "to finish, and its value must be written out whole, "
                    'never abbreviated (a stored handle string like "$1" '
                    "counts as whole). When the job names no output format, "
                    "return data as data, exactly as produced; never invent "
                    "formatting or summaries. Never draft a long value in "
                    "your thinking: write it once, directly in the reply.\n"
                    "Example turns for the job `describe_load()` on an object "
                    "with a real method `items`:\n"
                    '{"action": "call", "name": "items", "args": []}\n'
                    '-> result: ["anvil", "piano"]\n'
                    '{"action": "return", "value": "an anvil and a piano", "confidence": 0.9}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT:\n{story}\n\n"
                    f"Your job: produce the result of {call_repr} and make "
                    f"any changes to the object it implies. `{name}` itself "
                    "has no code and cannot be called."
                    + (
                        "\nCall stack: "
                        + " -> ".join(f"{f}()" for f in stack)
                        + "; never call anything already on it."
                        if len(stack) > 1
                        else ""
                    )
                    + (
                        "\nThe final return value MUST match this schema exactly: "
                        + _render_schema(returns_schema)
                        if returns_schema is not None
                        else ""
                    )
                ),
            },
        ]

    def condense_prompt(self, origin, events):
        """Messages for retelling the oldest events as one passage. Calls
        are rendered with their roles spelled out, so the retelling
        cannot mistake who said or did what."""

        def spoken(event):
            if event.get("event") == "call":
                args = ", ".join(repr(a) for a in event.get("args") or [])
                result = json.dumps(
                    event.get("result"), default=repr, ensure_ascii=False
                )
                return (
                    f"the outside world called {event.get('name')}"
                    f"({_snip(args, 300)}) and the object itself produced: "
                    + _snip(result, 400)
                )
            return _told_event_line(event)

        passage = "\n".join(spoken(e) for e in events)
        return [
            {
                "role": "system",
                "content": (
                    "Retell the oldest part of an object's story as one "
                    "short passage (2-5 sentences), so later readers can "
                    "drop the raw events. Keep every concrete fact, name, "
                    "number, decision, and standing commitment; keep who "
                    "did or said each thing — never merge different "
                    "actors. Drop process noise and repetition. "
                    "Never invent, never editorialize. "
                    "Reply with exactly one JSON object: "
                    '{"value": "<the retelling>", "confidence": <0..1>}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT: {origin}\n\nOLDEST EVENTS, in order:\n"
                    f"{passage}\n\nRetell these as one passage."
                ),
            },
        ]

    def query_envelope(self, value_schema=None):
        return _query_envelope(value_schema)

    def action_envelope(self, returns_schema=None, stack=()):
        return _action_envelope(returns_schema, stack)

    def receipt(self, handle, text):
        return _handle_receipt(handle, text)

    def extract(self, text):
        return _extract_json(text)

    def shape(self, data):
        return _answer_shape(data)

    def complete(self, messages, temperature=0.7, think=None, schema=None):
        """Inference climbs a two-tier ladder.

        The direct tier (`think=False`, on servers whose chat template can
        turn thinking off) answers single-shot with thinking disabled and,
        where the server supports it, decoding constrained to `schema` —
        cheap, fast, and immune to runaway thought. Constrained decoding
        is never combined with thinking: it would squeeze the reasoning
        channel shut and the thoughts would leak into the answer as prose.

        Everything else is the thinking tier: one streamed completion,
        supervised as it arrives. There are no scheduled windows — the
        stream is cut only on evidence, the moment either channel starts
        repeating itself verbatim. A cut carries the pruned reasoning
        forward (a reply rehearsed inside the loop is harvested as the
        answer); repeated cuts or a spent budget turn into a demand for
        the answer; an answer cut mid-way by the budget gets a completion
        grant sized from the draft."""
        debug = _debugging.get()
        st = _op_debug.get()
        # a plan's box is closed by its final `▸` step line, not by us
        conclude = not (st and st.get("plan"))
        if think is False and self._think_toggle:
            content, reasoning, meta = self._request(
                messages, temperature, self.max_tokens, think, schema
            )
            self.last_meta = meta
            if debug:
                sections = ([("thoughts", reasoning)] if reasoning else []) + [
                    ("answer", content)
                ]
                prefix = "<<< " + (
                    f"step {st['round']} · " if st and st.get("plan") else ""
                )
                sections[0] = (prefix + sections[0][0], sections[0][1])
                _debug_box(None, sections, _meta_line(meta), close=conclude)
            return content
        # -- thinking tier: stream, supervise, intervene on evidence ------
        thoughts = []
        allowance = self.max_tokens  # total budget across interventions
        interventions = 0
        attempts = 0
        demands_left = 2
        content = ""
        stalled_last = False
        printer = _StreamPrinter(st) if debug else None
        while True:
            attempts += 1
            demanding = allowance <= 0 or interventions >= 2 or attempts > 6
            if demanding:
                if demands_left <= 0:
                    # it never stopped thinking; the parse layer sees the
                    # length finish and raises the clear budget error
                    # instead of retrying
                    if printer:
                        printer.finish(_meta_line(self.last_meta), close=conclude)
                    return content
                demands_left -= 1
            budget = allowance if allowance > 0 else 2048
            content, reasoning, meta, cut = self._stream(
                self._with_thoughts(
                    messages, thoughts, final=demanding, stalled=stalled_last
                ),
                temperature,
                budget,
                printer,
            )
            if cut:
                interventions += 1
            meta["interventions"] = interventions
            self.last_meta = meta
            allowance -= int(meta.get("completion_tokens") or budget)
            # a length finish means the REQUEST hit its token cap, not that
            # the answer is incomplete: reasoning plus a whole answer can
            # land exactly on the cap, and asking again only confuses the
            # model about an answer it already delivered
            concluded = (
                bool(content)
                and not cut
                and (meta.get("finish_reason") != "length" or _is_complete_json(content))
            )
            # an interrupted answer is only an answer if JSON is underway
            # and the interruption was the budget, not a loop; prose in
            # the answer channel is thinking that escaped the reasoning
            # channel and must not be granted the full budget
            answer_underway = (
                bool(content)
                and not concluded
                and not cut
                and content.lstrip()[:1] in ("{", "[")
            )
            carried_reasoning = stalled_reasoning = None
            carried_draft = stalled_draft = None
            if not concluded:
                if reasoning:
                    carried_reasoning, stalled_reasoning = _prune_repetition(reasoning)
                if content and not answer_underway:
                    carried_draft, stalled_draft = _prune_repetition(content)
            stalled_last = bool(cut or stalled_reasoning or stalled_draft)
            harvested = None
            if not concluded and not content and (cut or stalled_reasoning):
                harvested = _harvest_reply(reasoning)
            if printer:
                if concluded:
                    printer.finish(_meta_line(meta), close=conclude)
                elif harvested:
                    printer.breakline()
                    _debug_box(
                        None,
                        [("answer, harvested from the looping thoughts", harvested)],
                        _meta_line(meta),
                        close=conclude,
                    )
                elif cut:
                    printer.note("loop detected — cut, pruned, carrying on")
                elif answer_underway:
                    printer.note("answer cut by the budget; granting completion")
                else:
                    printer.note(_meta_line(meta) or "carrying on")
            if concluded:
                return content  # chose to answer: the stream ran its course
            if harvested:
                return harvested  # the reply was rehearsed in the thinking
            if reasoning:
                if cut or stalled_reasoning:
                    carried_reasoning = (
                        (carried_reasoning + "\n" if carried_reasoning else "")
                        + "(the reasoning began repeating verbatim and was "
                        "cut; do not resume the loop)"
                    )
                thoughts.append(carried_reasoning)
            if answer_underway:
                thoughts.append(content)
                break  # the answer is underway: a grant to finish it
            if content:
                label = (
                    "(a draft, interrupted; do not continue this draft"
                    + (
                        ", and it began repeating verbatim, cut where the "
                        "loop started"
                        if cut or stalled_draft
                        else ""
                    )
                    + ")\n"
                )
                thoughts.append(
                    label + carried_draft
                    if carried_draft
                    else "(a draft here only repeated itself verbatim and "
                    "was discarded; take the next concrete step instead)"
                )
        # a grant to finish the answer that already began — sized from the
        # cut draft, never the whole budget in one uncapped shot; if even
        # this is not enough, the length finish becomes the clear budget
        # error instead of a runaway
        grant = min(
            self.max_tokens,
            2 * int(meta.get("completion_tokens") or 1024) + 2048,
        )
        content, reasoning, meta, cut = self._stream(
            self._with_thoughts(messages, thoughts, final=True),
            temperature,
            grant,
            printer,
        )
        meta["interventions"] = interventions
        self.last_meta = meta
        if printer:
            printer.finish(_meta_line(meta), close=conclude)
        return content

    def _stream(self, messages, temperature, max_tokens, printer=None):
        """One streaming completion, watched as it arrives: each channel
        feeds a `_LoopWatch`, and the connection is cut the moment either
        starts repeating itself verbatim. -> (content, reasoning, meta,
        cut). Falls back to a plain request on servers without SSE."""
        if not self._stream_mode:
            content, reasoning, meta = self._request(messages, temperature, max_tokens)
            return content, reasoning, meta, False
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.thinking_payload(payload)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.request_extra)
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        reasoning_parts, content_parts = [], []
        watches = {"thoughts": _LoopWatch(), "answer": _LoopWatch()}
        finish = None
        usage = {}
        cut = False
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    if choices[0].get("finish_reason"):
                        finish = choices[0]["finish_reason"]
                    delta = choices[0].get("delta") or {}
                    for channel, key, parts in (
                        ("thoughts", "reasoning_content", reasoning_parts),
                        ("answer", "content", content_parts),
                    ):
                        piece = delta.get(key) or ""
                        if not piece:
                            continue
                        parts.append(piece)
                        if printer:
                            printer.write(channel, piece)
                        if watches[channel].feed(piece):
                            cut = True
                    if cut:
                        break  # closing the connection stops the runaway
        except urllib.error.HTTPError as error:
            if error.code in (400, 404, 422):
                blame = ""
                with contextlib.suppress(Exception):
                    blame = error.read().decode("utf-8", "replace")
                if "stream" in blame:
                    self._stream_mode = False
                elif self._json_mode:
                    self._json_mode = False
                else:
                    self._stream_mode = False
                return self._stream(messages, temperature, max_tokens, printer)
            raise
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        meta = dict(usage)
        # a cut aborts before the usage chunk arrives; estimate the spend
        # so the allowance still ticks down
        meta.setdefault(
            "completion_tokens", max(1, (len(reasoning) + len(content)) // 3)
        )
        meta["finish_reason"] = "cut" if cut else finish
        if not self._json_mode:
            meta["json_mode"] = False
        return content, reasoning, meta, cut

    def _with_thoughts(self, messages, thoughts, final, stalled=False):
        if not thoughts and not final:
            return messages
        nudge = (
            "Reasoning budget spent. Emit the required JSON reply now, "
            "complete and from the beginning, nothing else."
            if final
            else "Your reasoning was interrupted; what matters is preserved "
            "above. Answer now with the required JSON reply if you can, "
            "otherwise keep reasoning — the next step, not a repetition."
        )
        if stalled and not final:
            nudge += (
                " Note: your last thoughts looped without progress and the "
                "loop was cut. Do not resume it; reply now, or state in one "
                "line what is genuinely missing."
            )
        st = _op_debug.get()
        if st and st.get("plan"):
            nudge += (
                " The reply is one small action object. A value already "
                "drafted in your thoughts is finished: put it in the reply "
                "now instead of rehearsing it again."
            )
        extended = list(messages)
        if thoughts:
            extended.append(
                {
                    "role": "assistant",
                    "content": "(reasoning so far, interrupted)\n"
                    + "\n".join(thoughts),
                }
            )
        extended.append({"role": "user", "content": nudge})
        return extended

    def _request(self, messages, temperature, max_tokens, think=None, schema=None):
        direct = think is False and self._think_toggle
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if direct:
            self.direct_payload(payload)
        else:
            self.thinking_payload(payload)
        if schema is not None and direct and self._schema_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema},
            }
        elif self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.request_extra)
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (400, 404, 422):
                blame = ""
                with contextlib.suppress(Exception):
                    blame = error.read().decode("utf-8", "replace")
                # drop the capability the server most plausibly rejected,
                # remember, and retry; every branch clears a flag that
                # gates it, so this terminates
                if "chat_template" in blame and self._think_toggle:
                    self._think_toggle = False
                elif ("json_schema" in blame or "schema" in blame) and self._schema_mode:
                    self._schema_mode = False
                elif direct and schema is not None and self._schema_mode:
                    self._schema_mode = False
                elif direct and self._think_toggle:
                    self._think_toggle = False
                elif self._json_mode:
                    self._json_mode = False  # server has no JSON mode
                else:
                    raise
                return self._request(messages, temperature, max_tokens, think, schema)
            raise
        choice = data["choices"][0]
        meta = dict(data.get("usage") or {})
        meta["finish_reason"] = choice.get("finish_reason")
        if direct:
            meta["direct"] = True
        if not self._json_mode:
            meta["json_mode"] = False
        message = choice["message"]
        return message.get("content") or "", message.get("reasoning_content") or "", meta


class _CallableEngine(Engine):
    """A bare callable as provider: inherits the engine's voice (prompts,
    envelopes, parsing); only the completion itself is the caller's."""

    tiers = False
    narrates = False

    def __init__(self, fn):
        self.fn = fn
        self.last_meta = {}

    def complete(self, messages, temperature=0.7, think=None, schema=None):
        try:
            return self.fn(messages, temperature=temperature)
        except TypeError:
            return self.fn(messages)


class _ProviderEngine(Engine):
    """An object with `complete(messages) -> text` as provider — the
    scene-10 protocol — wrapped so it too speaks with the engine's voice."""

    tiers = False
    narrates = False

    def __init__(self, provider):
        self.provider = provider

    @property
    def last_meta(self):
        return getattr(self.provider, "last_meta", None) or {}

    def complete(self, messages, temperature=0.7, think=None, schema=None):
        try:
            return self.provider.complete(messages, temperature=temperature)
        except TypeError:
            return self.provider.complete(messages)


# ---------------------------------------------------------------------------
# The registry: model name -> engine family
# ---------------------------------------------------------------------------

_families = []


def register(engine_cls):
    """Add an engine family to the registry. The most recently registered
    family that `handles()` a model name wins, so a late registration can
    claim a name an earlier one also matches."""
    _families.insert(0, engine_cls)
    return engine_cls


def engine_for(model):
    """The engine class that drives `model` — the generic base `Engine`
    when no family claims the name."""
    for family in _families:
        if family.handles(model):
            return family
    return Engine


def resolve(spec, cfg):
    """A `model=` spec (or nothing) as a live engine.

    `None` and a plain model name pick their family from the registry; a
    URL keeps the configured model name and only moves the endpoint; an
    `Engine` instance, an object with `.complete`, and a bare callable are
    taken as given."""
    max_tokens = cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)
    extra = cfg.get("request_extra")
    if spec is None:
        return engine_for(cfg["model"])(
            cfg["base_url"], cfg["api_key"], cfg["model"], max_tokens, extra
        )
    if isinstance(spec, str):
        if spec.startswith("file://"):
            raise NotImplementedError(
                "embedded models are future work; point `model` at a server URL"
            )
        if spec.startswith("http://") or spec.startswith("https://"):
            return engine_for(cfg["model"])(
                spec, cfg["api_key"], cfg["model"], max_tokens, extra
            )
        return engine_for(spec)(
            cfg["base_url"], cfg["api_key"], spec, max_tokens, extra
        )
    if isinstance(spec, Engine):
        return spec
    if hasattr(spec, "complete"):
        return _ProviderEngine(spec)
    if callable(spec):
        return _CallableEngine(spec)
    raise TypeError(f"cannot use {spec!r} as an inference engine")


# ---------------------------------------------------------------------------
# Parsing, forgiveness, and the reply contracts — the engine's own share
# ---------------------------------------------------------------------------

def _json_objects(text):
    """All balanced {...} substrings in the text, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        end = None
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            start = text.find("{", start + 1)
            continue
        out.append(text[start : end + 1])
        start = text.find("{", end + 1)
    return out


def _harvest_reply(reasoning):
    """A model stuck rehearsing its intended reply inside its thinking HAS
    answered, just in the wrong channel: when exactly one action object
    is repeated verbatim three or more times, that object is the reply."""
    counts = {}
    for raw in _json_objects(reasoning):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and ("action" in data or "value" in data):
            key = json.dumps(data, sort_keys=True, ensure_ascii=False)
            counts[key] = counts.get(key, 0) + 1
    rehearsed = [key for key, n in counts.items() if n >= 3]
    if len(rehearsed) == 1:
        return rehearsed[0]
    return None


def _is_complete_json(text):
    """True when the text is one whole, parseable JSON value."""
    try:
        json.loads(text.strip())
        return True
    except json.JSONDecodeError:
        return False


def _prune_repetition(text):
    """Cut a draft at the point it starts repeating itself verbatim.
    A degenerate loop fed back into the next block's prompt only deepens
    the loop; keep the useful prefix, drop the echo. -> (kept, stalled)"""
    lines = text.splitlines()
    counts = {}
    for i, line in enumerate(lines):
        key = line.strip()
        if len(key) < 10:
            continue  # short structural lines ("},", "],") repeat honestly
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= 3:
            kept = lines[:i]
            # the tail of the prefix is the seed of the loop; drop it too
            while kept and (
                not kept[-1].strip() or counts.get(kept[-1].strip(), 0) >= 2
            ):
                kept.pop()
            return "\n".join(kept).rstrip(), True
    return text, False


def _extract_json(text):
    """The JSON the model answered with. A reply that IS clean JSON (any
    type, bare arrays included) is taken whole; otherwise the text is
    scanned for balanced objects and the LAST plausible one wins
    (reasoning models explain first, answer last)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    candidates = []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        end = None
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            start = text.find("{", start + 1)
            continue
        try:
            candidates.append(json.loads(text[start : end + 1]))
            start = text.find("{", end + 1)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    for candidate in reversed(candidates):
        if isinstance(candidate, dict) and ("action" in candidate or "value" in candidate):
            return candidate
    if candidates:
        return candidates[-1]
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


def _answer_shape(data):
    """-> (value, clamped confidence). A reply that is not a
    {"value": ...} envelope IS the value: a model that answers with the
    bare data is forgiven, at even confidence."""
    if isinstance(data, dict) and "value" in data:
        value, confidence = data.get("value"), data.get("confidence", 0.5)
    else:
        value, confidence = data, 0.5
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    return value, max(0.01, min(confidence, 0.99))


def _json_schema_of(template):
    """A `returns=` template as a JSON Schema, for servers that can
    constrain decoding to it. `object` and `None` mean any value."""
    if template is None or template is object:
        return {}
    if isinstance(template, type):
        name = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }.get(template)
        return {"type": name} if name else {}
    if isinstance(template, dict):
        return {
            "type": "object",
            "properties": {k: _json_schema_of(v) for k, v in template.items()},
            "required": list(template),
        }
    if isinstance(template, list):
        out = {"type": "array"}
        if template:
            out["items"] = _json_schema_of(template[0])
        return out
    return {"const": template}


def _query_envelope(value_schema=None):
    """The reply contract of a single-shot question."""
    return {
        "type": "object",
        "properties": {
            "value": _json_schema_of(value_schema),
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["value", "confidence"],
    }


def _action_envelope(returns_schema=None, forbidden=()):
    """The plan protocol's reply contract: exactly one of the six actions.
    The `const` on `action` keeps the branches disjoint; a `returns=`
    template constrains the return branch's value to it. `forbidden` is
    accepted but unused: excluding stack names via `not`/`enum` was tried
    and measurably damaged task comprehension on servers that inject the
    schema as prompt text — the runtime refusal remains the cycle guard."""
    name = {"type": "string"}
    confidence = {"type": "number", "minimum": 0, "maximum": 1}
    return {"oneOf": [
        {"type": "object", "required": ["action", "name"],
         "properties": {"action": {"const": "get"}, "name": name}},
        {"type": "object", "required": ["action", "name", "value"],
         "properties": {"action": {"const": "set"}, "name": name,
                        "value": {}, "confidence": confidence}},
        {"type": "object", "required": ["action", "name"],
         "properties": {"action": {"const": "delete"}, "name": name}},
        {"type": "object", "required": ["action", "name"],
         "properties": {"action": {"const": "call"}, "name": name,
                        "args": {"type": "array"}}},
        {"type": "object", "required": ["action", "name", "meaning"],
         "properties": {"action": {"const": "define"}, "name": name,
                        "meaning": {"type": "string"}}},
        {"type": "object", "required": ["action", "value"],
         "properties": {"action": {"const": "return"},
                        "value": _json_schema_of(returns_schema),
                        "confidence": confidence}},
    ]}


def _handle_receipt(handle, text):
    """The feedback for a value too large to march through the model: a
    shape summary, a preview, and — taught exactly when it matters — how
    to pass the value on without retyping a byte of it."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            shape = f"a list of {len(data)} items"
        elif isinstance(data, dict):
            shape = "an object with keys " + ", ".join(str(k) for k in list(data)[:8])
        else:
            shape = f"a {type(data).__name__}"
    except ValueError:
        shape = "text"
    return (
        f"{shape}, {len(text)} chars — stored whole as {handle}. "
        f'{{"action": "return", "value": "{handle}"}} delivers all of it, '
        "unchanged and complete: for passing the data on, that one step is "
        f"the whole job, done. Reading {handle} back (get) is only for "
        "transforming or inspecting the items — never for returning them. "
        "Never retype the data; the preview below is incomplete.\n"
        "Preview (incomplete): "
        + text[:300]
        + ("…" if len(text) > 300 else "")
    )


# ---------------------------------------------------------------------------
# The families. One line each, and one of them named as the default —
# the family whose model name a Thing talks to when nobody names one.
# ---------------------------------------------------------------------------

from .deepseek import DeepSeekEngine  # noqa: E402  (families register below)
from .qwen import QwenEngine  # noqa: E402         (the base they subclass)

register(QwenEngine)
register(DeepSeekEngine)

_DEFAULT_FAMILY = QwenEngine
_DEFAULT_MODEL = os.environ.get("THINAIR_MODEL", _DEFAULT_FAMILY.default_model)
