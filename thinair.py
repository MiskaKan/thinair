"""thinair.py — probabilistic objects with one axiom.

An object is a story. Every interaction is a continuation of that story;
the continuation is appended. Written code and recorded state are
authoritative (p = 1.0, bare values); inference fills only the silence,
always returning child Things that carry a value and a confidence < 1.0.

Spec: SPEC.md. The entire public surface is the single class `Thing`.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
import os
import re
import sys
import textwrap
import traceback
import urllib.error
import urllib.request

__all__ = ["Thing"]

_DEFAULT_BASE_URL = os.environ.get("THINAIR_BASE_URL", "http://127.0.0.1:8000/v1")
_DEFAULT_API_KEY = os.environ.get("THINAIR_API_KEY", "1234")
_DEFAULT_MODEL = os.environ.get("THINAIR_MODEL", "Qwen3.6-35B-A3B-oQ8-mtp")
_DEFAULT_MAX_TOKENS = int(os.environ.get("THINAIR_MAX_TOKENS", "32768"))
# a plan-step result larger than this (rendered chars) is held by the
# runtime under a handle ("$1") instead of marching through the model
_HANDLE_LIMIT = int(os.environ.get("THINAIR_HANDLE_LIMIT", "4000"))

_UNSET = object()
_required = contextvars.ContextVar("thing_required_confidence", default=None)
_debugging = contextvars.ContextVar(
    "thing_debug", default=os.environ.get("THINAIR_DEBUG", "") not in ("", "0")
)
_purpose = contextvars.ContextVar("thing_purpose", default="inference")
_op_stack = contextvars.ContextVar("thing_op_stack", default=())
_op_debug = contextvars.ContextVar("thing_op_debug", default=None)


class LowConfidence(Exception):
    """A resolution came back below the active `Thing.require` threshold."""


class ContinuationLimit(Exception):
    """An imagined plan exceeded its step or depth budget."""


def _plain(value):
    """Strip any _Pending/Thing wrapper down to the native value."""
    if isinstance(value, _Pending):
        value = value._resolve()
    if isinstance(value, Thing):
        return value.__dict__.get("_thing_value", repr(value))
    return value


def _confidence_of(value):
    if isinstance(value, Thing):
        return value.__dict__.get("confidence", 1.0)
    return getattr(value, "confidence", 1.0)


def _own_doc(cls):
    """The docstring the programmer wrote: never Thing's, never framework noise."""
    for klass in cls.__mro__:
        if klass is Thing:
            return None
        doc = klass.__dict__.get("__doc__")
        if doc:
            return inspect.cleandoc(doc)
    return None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

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


class _HTTPBackend:
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url,
        api_key,
        model,
        max_tokens=_DEFAULT_MAX_TOKENS,
        request_extra=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = int(max_tokens)
        self.request_extra = dict(request_extra or {})
        self.last_meta = {}
        # optimistic capabilities; each is dropped if the server rejects it
        self._json_mode = True      # response_format json_object
        self._schema_mode = True    # response_format json_schema (direct tier)
        self._think_toggle = True   # chat_template_kwargs turns thinking off
        self._stream_mode = True    # SSE streaming (thinking tier)

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
            "top_p": 0.95,
            "presence_penalty": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
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
            # Qwen-recommended sampling for non-thinking mode
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["top_p"] = 0.8
        else:
            # thinking mode: wider nucleus, and a presence penalty so a
            # quantized model is less tempted to circle its own thoughts
            payload["top_p"] = 0.95
            payload["presence_penalty"] = 1.0
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


class _CallableBackend:
    def __init__(self, fn):
        self.fn = fn

    def complete(self, messages, temperature=0.7):
        return self.fn(messages)


def _resolve_backend(spec, cfg):
    max_tokens = cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)
    extra = cfg.get("request_extra")
    if spec is None:
        return _HTTPBackend(
            cfg["base_url"], cfg["api_key"], cfg["model"], max_tokens, extra
        )
    if isinstance(spec, str):
        if spec.startswith("file://"):
            raise NotImplementedError(
                "embedded models are future work; point `model` at a server URL"
            )
        if spec.startswith("http://") or spec.startswith("https://"):
            return _HTTPBackend(
                spec, cfg["api_key"], cfg["model"], max_tokens, extra
            )
        return _HTTPBackend(
            cfg["base_url"], cfg["api_key"], spec, max_tokens, extra
        )
    if hasattr(spec, "complete"):
        return spec
    if callable(spec):
        return _CallableBackend(spec)
    raise TypeError(f"cannot use {spec!r} as an inference backend")


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


def _render_schema(schema):
    if isinstance(schema, type):
        return schema.__name__
    if isinstance(schema, dict):
        return "{" + ", ".join(f'"{k}": {_render_schema(v)}' for k, v in schema.items()) + "}"
    if isinstance(schema, list):
        return "[" + (_render_schema(schema[0]) + ", ..." if schema else "...") + "]"
    return json.dumps(schema)


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


def _action_envelope(returns_schema=None):
    """The plan protocol's reply contract: exactly one of the six actions.
    The `const` on `action` keeps the branches disjoint; a `returns=`
    template constrains the return branch's value to it."""
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


def _matches(value, schema):
    """Check a JSON value against a template of types/dicts/lists. -> (ok, why)"""
    if schema is None:
        return True, ""
    if isinstance(schema, type):
        if schema is float and isinstance(value, int) and not isinstance(value, bool):
            return True, ""
        if schema is not bool and isinstance(value, bool):
            return False, f"expected {schema.__name__}, got bool"
        if isinstance(value, schema):
            return True, ""
        return False, f"expected {schema.__name__}, got {type(value).__name__} ({value!r})"
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            return False, f"expected an object, got {type(value).__name__}"
        for key, sub in schema.items():
            if key not in value:
                return False, f"missing key {key!r}"
            ok, why = _matches(value[key], sub)
            if not ok:
                return False, f"key {key!r}: {why}"
        return True, ""
    if isinstance(schema, list):
        if not isinstance(value, list):
            return False, f"expected a list, got {type(value).__name__}"
        for i, item in enumerate(value if schema else []):
            ok, why = _matches(item, schema[0])
            if not ok:
                return False, f"item {i}: {why}"
        return True, ""
    return (True, "") if value == schema else (False, f"expected literal {schema!r}")


@contextlib.contextmanager
def _op_scope(plan=False):
    """One inference operation's debug scope: its box opens on the first
    request and everything until the scope ends belongs to it."""
    token = _op_debug.set({"open": False, "shown": 0, "plan": plan, "round": 0})
    try:
        yield
    finally:
        _op_debug.reset(token)


@contextlib.contextmanager
def _require(threshold):
    token = _required.set(float(threshold))
    try:
        yield
    finally:
        _required.reset(token)


@contextlib.contextmanager
def _debug(enabled):
    token = _debugging.set(bool(enabled))
    try:
        yield
    finally:
        _debugging.reset(token)


def _render_messages(messages):
    out = []
    for message in messages:
        out.append(f"[{message['role']}]")
        for line in str(message.get("content", "")).splitlines() or [""]:
            out.append(f"  {line}")
    return "\n".join(out)


def _meta_line(meta):
    bits = []
    if meta.get("prompt_tokens") is not None or meta.get("completion_tokens") is not None:
        bits.append(
            f"tokens {meta.get('prompt_tokens', '?')} in → "
            f"{meta.get('completion_tokens', '?')} out"
        )
    finish = meta.get("finish_reason")
    if finish == "length":
        bits.append("hit the token budget")
    elif finish == "cut":
        bits.append("cut (loop detected)")
    elif finish:
        bits.append(finish)
    if meta.get("interventions"):
        bits.append(f"{meta['interventions']} intervention(s)")
    if meta.get("direct"):
        bits.append("direct (no thinking)")
    if meta.get("json_mode") is False:
        bits.append("freeform (server JSON mode off)")
    return " · ".join(bits)


def _snip(text, limit=100):
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


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


def _substitute_handles(value, results):
    """Exact-match handle strings anywhere in a JSON structure become the
    stored data; everything else passes through untouched."""
    if isinstance(value, str) and value in results:
        stored = results[value]
        try:
            return json.loads(stored)
        except ValueError:
            return stored
    if isinstance(value, list):
        return [_substitute_handles(item, results) for item in value]
    if isinstance(value, dict):
        return {k: _substitute_handles(v, results) for k, v in value.items()}
    return value


def _told_event_line(event):
    """One journal event rendered for the telling. A huge call result is
    elided down to a preview — the record keeps it whole. A trailing
    delete says what it means: the question re-opened, not answered."""
    if event.get("event") == "condense":
        return "(the older story, retold): " + str(event.get("summary", ""))
    if event.get("event") == "delete":
        return (
            json.dumps(event, default=repr, ensure_ascii=False)
            + " — the recorded value was retired; the question is open "
            "again, to be answered afresh (deletion is not absence)"
        )
    text = json.dumps(event, default=repr, ensure_ascii=False)
    if len(text) <= 600:
        return text
    slim = dict(event)
    if "result" in slim:
        rendered = json.dumps(slim["result"], default=repr, ensure_ascii=False)
        if len(rendered) > 300:
            slim["result"] = rendered[:300] + f"… ({len(rendered)} chars, elided)"
    text = json.dumps(slim, default=repr, ensure_ascii=False)
    return text if len(text) <= 900 else text[:900] + "…"


def _describe_step(step):
    """A plan step as one readable action, e.g. `call headlines("fi")`."""
    action = step.get("action")
    target = step.get("name", "")
    if action == "call":
        args = ", ".join(repr(a) for a in (step.get("args") or []))
        return f"call {target}({_snip(args, 40)})"
    if action == "set":
        value = json.dumps(step.get("value"), default=repr, ensure_ascii=False)
        return f"set {target} = {_snip(value, 40)}"
    if action in ("get", "delete", "define"):
        return f"{action} {target}"
    return str(action)


def _debug_action(index, text, final=False):
    """One `▸` line per plan step: the action chosen and what came back.
    Steps are children of their operation, so they indent one level
    deeper; the final step also closes the operation's box."""
    pad = "  " * len(_op_stack.get())
    print(f"{pad}  ▸ step {index} · {text}", file=sys.stderr)
    if final:
        print(pad + "└" + "─" * 59, file=sys.stderr)


def _call_site():
    """The user's own frames (thinair's filtered out), outermost first,
    indented down to the exact line that triggered this inference."""
    here = os.path.realpath(__file__)
    frames = [
        f
        for f in traceback.extract_stack()
        if os.path.realpath(f.filename) != here
    ][-5:]
    lines = []
    for i, frame in enumerate(frames):
        loc = f"{os.path.basename(frame.filename)}:{frame.lineno} {frame.name}"
        if frame.line:
            loc += f"  {frame.line.strip()}"
        lines.append("  " * i + loc)
    return "\n".join(lines)


def _debug_box(title, sections, footer="", opener="┌", close=True, tag=True):
    """One titled box. `opener="├"` continues the operation above it and
    `close=False` leaves the bottom open, so a whole operation reads as
    a single tall structure closed by its final block (or, for plans,
    by the final step's `▸` line). `tag=False` renders a bare segment
    title without the `thinair ·` prefix; `title=None` emits labeled
    sections only. Boxes born inside another operation say so in the
    title and shift right with depth."""
    chain = _op_stack.get()
    pad = "  " * len(chain)
    lines = []
    if title is not None:
        if tag:
            if chain:
                title = f"{title} · in {' › '.join(chain)}"
            title = f"thinair · {title}"
        lines.append(f"{opener}─ {title} " + "─" * max(1, 56 - len(title)))
    for label, body in sections:
        if label:
            lines.append(f"├─ {label} " + "─" * max(1, 56 - len(label)))
        for line in str(body).splitlines() or [""]:
            lines.append(f"│ {line}")
    if close:
        lines.append(f"└─ {footer}" if footer else "└" + "─" * 59)
    elif footer:
        lines.append(f"├─ {footer}")
    print("\n".join(pad + line for line in lines), file=sys.stderr)


def _debug_open_op(purpose, messages, st):
    """Open an operation's box (or append the newly sent messages to one
    already open): where it was called from first, then what goes to the
    model, marked `>>>`. Model output arrives later, marked `<<<`."""
    st["round"] += 1
    if not st["open"]:
        _debug_box(
            purpose,
            [
                ("called from", _call_site()),
                (">>> prompt", _render_messages(messages)),
            ],
            close=False,
        )
        st["open"] = True
    else:
        # a plan is one growing conversation; only the newly appended
        # messages are sent anew (the model's own echo is skipped: it
        # already appeared as `<<<`)
        fresh = [
            m for m in messages[st["shown"]:] if m.get("role") != "assistant"
        ]
        if fresh:
            _debug_box(
                None,
                [(f">>> step {st['round']} · appended", _render_messages(fresh))],
                close=False,
            )
    st["shown"] = len(messages)


def _check_required(confidence):
    threshold = _required.get()
    if threshold is not None and confidence < threshold:
        raise LowConfidence(f"confidence {confidence:.2f} < required {threshold:.2f}")


# ---------------------------------------------------------------------------
# _Pending — an unresolved name. Value-use collapses it; call-use runs a plan.
# ---------------------------------------------------------------------------

class _Pending:
    __slots__ = ("_owner", "_name", "_value")

    def __init__(self, owner, name):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_value", _UNSET)

    def _resolve(self):
        if self._value is _UNSET:
            object.__setattr__(self, "_value", self._owner._thing_read(self._name))
        return self._value

    def __call__(self, *args, **kwargs):
        return self._owner._thing_call(self._name, args, kwargs)

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __repr__(self):
        return repr(self._resolve())

    def __str__(self):
        return str(self._resolve())

    def __format__(self, spec):
        return format(self._resolve(), spec)

    def __bool__(self):
        return bool(self._resolve())

    def __eq__(self, other):
        return self._resolve() == other

    def __ne__(self, other):
        return self._resolve() != other

    def __lt__(self, other):
        return self._resolve() < other

    def __le__(self, other):
        return self._resolve() <= other

    def __gt__(self, other):
        return self._resolve() > other

    def __ge__(self, other):
        return self._resolve() >= other

    def __hash__(self):
        return hash(self._resolve())

    def __matmul__(self, operand):
        return self._resolve() @ operand

    def __pos__(self):
        return +self._resolve()

    def __invert__(self):
        return ~self._resolve()

    def __add__(self, other):
        return self._resolve() + other

    def __radd__(self, other):
        return other + self._resolve()

    def __sub__(self, other):
        return self._resolve() - other

    def __rsub__(self, other):
        return other - self._resolve()

    def __mul__(self, other):
        return self._resolve() * other

    def __rmul__(self, other):
        return other * self._resolve()

    def __truediv__(self, other):
        return self._resolve() / other

    def __int__(self):
        return int(self._resolve())

    def __float__(self):
        return float(self._resolve())

    def __index__(self):
        return int(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __iter__(self):
        return iter(self._resolve())

    def __contains__(self, item):
        return item in self._resolve()

    def __getitem__(self, key):
        return self._resolve()[key]


# ---------------------------------------------------------------------------
# Thing
# ---------------------------------------------------------------------------

class _ThingMeta(type):
    """Lets a frozen document be cast back to life: `blob @ Car`."""

    def __rmatmul__(cls, blob):
        if isinstance(blob, dict):
            obj = cls.__new__(cls)
            obj._thing_restore(blob)
            return obj
        return NotImplemented


class Thing(metaclass=_ThingMeta):
    """A probabilistic object. See module docstring and SPEC.md."""

    LowConfidence = LowConfidence
    ContinuationLimit = ContinuationLimit

    _thing_step_budget = 16
    _thing_depth_budget = 4
    _thing_correction_budget = 6
    _thing_telling_budget = 15  # told events beyond this get retold
    _thing_defaults: dict = {}

    def __init__(self, *parts, stateful=True, model=None, confidence=None, **state):
        self._thing_stateful = bool(stateful)
        self._thing_model_spec = model
        if confidence is not None:
            carried = parts[0] if parts else None
            self._thing_parts = [
                "a value the programmer asserted with explicit doubt",
                "value: " + json.dumps(carried, default=repr, ensure_ascii=False),
            ]
            object.__setattr__(self, "_thing_value", carried)
            object.__setattr__(
                self, "confidence", max(0.01, min(float(confidence), 0.99))
            )
        else:
            self._thing_parts = [p if isinstance(p, str) else repr(p) for p in parts]
        for name, value in state.items():
            setattr(self, name, value)

    # -- configuration ------------------------------------------------------

    @classmethod
    def defaults(
        cls, model=None, base_url=None, api_key=None, max_tokens=None,
        request_extra=None,
    ):
        """Set class-wide backend defaults; subclasses inherit via the MRO.
        `request_extra` is merged verbatim into every HTTP payload — the
        escape hatch for server-specific knobs (thinking budgets, template
        kwargs, reasoning effort, sampling overrides)."""
        cfg = dict(cls.__dict__.get("_thing_defaults", {}))
        if model is not None:
            cfg["model"] = model
        if base_url is not None:
            cfg["base_url"] = base_url
        if api_key is not None:
            cfg["api_key"] = api_key
        if max_tokens is not None:
            cfg["max_tokens"] = int(max_tokens)
        if request_extra is not None:
            cfg["request_extra"] = dict(request_extra)
        cls._thing_defaults = cfg

    @classmethod
    def require(cls, threshold):
        """Context manager: any resolution below `threshold` raises LowConfidence."""
        return _require(threshold)

    @classmethod
    def debug(cls, enabled=True):
        """Context manager: dump every prompt and raw completion to stderr
        while the block runs. THINAIR_DEBUG=1 turns it on globally."""
        return _debug(enabled)

    # -- internals ----------------------------------------------------------

    def _thing_ensure(self):
        """Lazy internals, so subclasses may skip super().__init__() entirely."""
        d = object.__getattribute__(self, "__dict__")
        d.setdefault("_thing_parts", [])
        d.setdefault("_thing_journal", [])
        d.setdefault("_thing_stateful", True)
        d.setdefault("_thing_model_spec", None)
        return d

    def _thing_config(self):
        cfg = {
            "base_url": _DEFAULT_BASE_URL,
            "api_key": _DEFAULT_API_KEY,
            "model": _DEFAULT_MODEL,
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        for klass in reversed(type(self).__mro__):
            cfg.update(getattr(klass, "_thing_defaults", None) or {})
        return cfg

    def _thing_backend(self):
        self._thing_ensure()
        backend = self.__dict__.get("_thing_backend_obj")
        if backend is None:
            backend = _resolve_backend(self._thing_model_spec, self._thing_config())
            object.__setattr__(self, "_thing_backend_obj", backend)
        return backend

    def _thing_log(self, event):
        self._thing_ensure()
        self._thing_journal.append(event)

    def _thing_surface(self):
        """The written stratum of subclasses: docs, class attrs, real methods."""
        doc = _own_doc(type(self))
        attrs, methods = {}, {}
        for klass in type(self).__mro__:
            if klass is Thing:
                break
            for name, value in vars(klass).items():
                if name.startswith("_"):
                    continue
                if callable(value):
                    methods.setdefault(name, inspect.getdoc(value) or "")
                else:
                    attrs.setdefault(name, value)
        return doc, attrs, methods

    def _thing_telling(self):
        """The journal as it deserves to be told: only events that still
        carry meaning. The record (`_thing_journal`) keeps everything;
        the telling folds away what the present state already says —
        superseded observations, overwritten sets, stale deletions,
        replaced definitions. Calls stay: they are the narrative. Once a
        `condense` event exists, the covered span renders as its
        retelling — except `define`s, which are standing contracts and
        outlive any summary."""
        events = self._thing_journal
        summary = None
        start = 0
        for i, event in enumerate(events):
            if event.get("event") == "condense":
                summary = event
                start = max(start, int(event.get("through", 0)))
        last_write = {}
        last_define = {}
        last_collapse = {}
        for i, event in enumerate(events):
            kind = event.get("event")
            if kind == "define":
                last_define[event.get("name")] = i
            if i < start:
                continue
            if kind in ("set", "observe", "delete"):
                last_write[event.get("name")] = i
            elif kind == "collapse":
                last_collapse[event.get("type", "")] = i
        told = []
        if summary is not None:
            told.append(summary)
        for name, i in sorted(last_define.items(), key=lambda kv: kv[1]):
            if i < start:
                told.append(events[i])  # contracts outlive the retelling
        for i in range(start, len(events)):
            event = events[i]
            kind = event.get("event")
            if kind in ("set", "observe", "condense"):
                continue  # state tells the first two; the summary, the third
            if kind == "delete":
                # only a still-standing deletion (nothing written since)
                # is a re-opened question worth telling
                if last_write.get(event.get("name")) == i:
                    told.append(event)
            elif kind == "define":
                if last_define.get(event.get("name")) == i:
                    told.append(event)
            elif kind == "collapse":
                if last_collapse.get(event.get("type", "")) == i:
                    told.append(event)
            else:
                told.append(event)
        return told

    def _thing_condense_if_needed(self):
        """When the telling outgrows its budget, one inference call
        retells the oldest events as a single passage, appended to the
        journal as a `condense` event. Never destructive — the record
        keeps every event beneath the retelling — and skipped inside
        plans, so a step never pauses to reminisce."""
        if not self.__dict__.get("_thing_stateful", True):
            return
        if _op_stack.get() or self.__dict__.get("_thing_condensing"):
            return
        told = self._thing_telling()
        if len(told) <= self._thing_telling_budget:
            return
        keep = max(3, self._thing_telling_budget // 2)
        old = [e for e in told[:-keep] if e.get("event") != "define"]
        if len(old) < 4:
            return  # nothing substantial to retell
        journal = self._thing_journal
        anchor = told[-keep]
        through = next((i for i, e in enumerate(journal) if e is anchor), None)
        if through is None:
            return

        def spoken(event):
            # calls are rendered with their roles spelled out, so the
            # retelling cannot mistake who said or did what
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

        passage = "\n".join(spoken(e) for e in old)
        origin = " | ".join(self._thing_parts) or type(self).__name__
        messages = [
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
        object.__setattr__(self, "_thing_condensing", True)
        try:
            with _op_scope():
                summary, confidence = self._thing_query(
                    messages,
                    temperature=0.3,
                    purpose=f"{type(self).__name__} · condense",
                    value_schema=str,
                    check=False,
                )
        finally:
            object.__setattr__(self, "_thing_condensing", False)
        if not isinstance(summary, str) or not summary.strip():
            return  # a failed retelling changes nothing; the record stands
        self._thing_log(
            {
                "event": "condense",
                "through": through,
                "summary": summary.strip(),
                "confidence": confidence,
            }
        )

    def _thing_story(self):
        self._thing_ensure()
        doc, attrs, methods = self._thing_surface()
        lines = []
        if type(self) is not Thing:
            lines.append(f"Class: {type(self).__name__}" + (f" — {doc}" if doc else ""))
        if methods:
            listing = "; ".join(
                f"{name}(...)" + (f" — {docstr}" if docstr else "")
                for name, docstr in methods.items()
            )
            lines.append("Real methods (these execute actual code): " + listing)
        if self._thing_parts:
            lines.append("Described as: " + " | ".join(self._thing_parts))
        carried = self.__dict__.get("_thing_value", _UNSET)
        if carried is not _UNSET:
            lines.append(
                f"Value of this object (p {self.__dict__.get('confidence', 0.5)}): "
                + json.dumps(carried, default=repr, ensure_ascii=False)
            )
        # the three strata, told apart — the model may only revise what
        # carries confidence, and here it can see which is which
        written = dict(attrs)
        imagined, slots = [], []
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if carried is not _UNSET and name == "confidence":
                continue
            if isinstance(value, _Pending):
                continue
            if (
                isinstance(carried, dict)
                and name in carried
                and carried[name] == value
            ):
                continue  # a mirror of the carried value: already told once
            if isinstance(value, Thing):
                if "_thing_value" in value.__dict__:
                    imagined.append(
                        (
                            name,
                            value.__dict__["_thing_value"],
                            value.__dict__.get("confidence", 0.5),
                        )
                    )
                else:
                    value._thing_ensure()
                    slots.append(
                        (name, " | ".join(value._thing_parts) or "open")
                    )
            else:
                written[name] = value
        if written:
            lines.append(
                "Written state (certain, untouchable): "
                + json.dumps(written, default=repr, ensure_ascii=False)
            )
        if imagined:
            lines.append(
                "Imagined state (revisable, confidence shown): "
                + "; ".join(
                    f"{name} = {json.dumps(value, default=repr, ensure_ascii=False)}"
                    f" (p {confidence:.2f})"
                    for name, value, confidence in imagined
                )
            )
        if slots:
            lines.append(
                "Open slots (yours to manage): "
                + "; ".join(f"{name} — {described}" for name, described in slots)
            )
        told = self._thing_telling()
        if told:
            lines.append("History (events that still matter, oldest first):")
            for i, event in enumerate(told, 1):
                lines.append(f"  {i}. {_told_event_line(event)}")
        return "\n".join(lines) or "An unspecified thing."

    def _thing_complete_json(
        self, messages, temperature, purpose="inference", think=None, schema=None
    ):
        backend = self._thing_backend()
        last_error = None
        st = _op_debug.get()
        for _ in range(3):
            if _debugging.get() and st is not None:
                _debug_open_op(purpose, messages, st)
            token = _purpose.set(purpose)
            try:
                try:
                    text = backend.complete(
                        messages, temperature=temperature, think=think, schema=schema
                    )
                except TypeError:
                    try:
                        text = backend.complete(messages, temperature=temperature)
                    except TypeError:
                        text = backend.complete(messages)
            finally:
                _purpose.reset(token)
            meta = getattr(backend, "last_meta", None) or {}
            if _debugging.get() and not isinstance(backend, _HTTPBackend):
                # HTTP backends narrate themselves block by block
                label = "<<< " + (
                    f"step {st['round']} · " if st and st.get("plan") else ""
                ) + "reply"
                _debug_box(
                    None,
                    [(label, text)],
                    _meta_line(meta),
                    close=not (st and st.get("plan")),
                )
            try:
                return _extract_json(text)
            except ValueError as error:
                last_error = error
                if meta.get("finish_reason") == "length":
                    # the budget died mid-thought; retrying would only burn
                    # the same tokens again
                    raise RuntimeError(
                        f"completion hit max_tokens while working on "
                        f"{purpose} (thinking counts against the same "
                        f"budget) — raise THINAIR_MAX_TOKENS / "
                        f"Thing.defaults(max_tokens=...), or shorten the story"
                    )
                if think is False:
                    # the direct tier answered with something unparseable;
                    # the question has earned the thinking tier
                    think = True
                    if _debugging.get():
                        _debug_box(
                            None,
                            [("↑ escalating", "no parseable JSON from the "
                              "direct tier; retrying with thinking")],
                            close=False,
                        )
                    continue
                messages = messages + [
                    {"role": "assistant", "content": text[:1000]},
                    {"role": "user", "content": "Reply with exactly one JSON object and nothing else."},
                ]
        raise RuntimeError(f"backend produced no parseable JSON: {last_error}")

    def _thing_query(self, messages, temperature, purpose, value_schema=None, check=True):
        """One single-shot question -> (value, confidence), climbing the
        two-tier ladder: the direct tier answers first; a reply below an
        active `require` threshold — or a refusal-shaped null, a
        non-answer rather than true absence — earns one thinking retry
        before LowConfidence is raised."""
        envelope = _query_envelope(value_schema)
        value, confidence = _answer_shape(
            self._thing_complete_json(
                messages, temperature, purpose, think=False, schema=envelope
            )
        )
        threshold = _required.get()
        refusal = value is None and confidence < 0.2
        # a null sits exactly on the unknown-vs-absent boundary — the
        # riskiest shape there is; before one may satisfy an explicit
        # require gate, the thinking tier gets a second opinion
        arbitrate = value is None and threshold is not None
        if (
            refusal
            or arbitrate
            or (threshold is not None and confidence < threshold)
        ) and getattr(self._thing_backend(), "_think_toggle", False):
            if _debugging.get():
                if refusal:
                    reason = f"a null at p {confidence:.2f} is a non-answer"
                elif arbitrate and confidence >= threshold:
                    reason = "a null at the require gate; second opinion"
                else:
                    reason = f"confidence {confidence:.2f} < required {threshold:.2f}"
                _debug_box(
                    None,
                    [("↑ escalating", reason + "; retrying with thinking")],
                    close=False,
                )
            value, confidence = _answer_shape(
                self._thing_complete_json(
                    messages, temperature, purpose + " · rethink",
                    think=True, schema=envelope,
                )
            )
        if check:
            _check_required(confidence)
        return value, confidence

    # -- the two continuations: read and call -------------------------------

    def _thing_read(self, name):
        self._thing_ensure()
        self._thing_condense_if_needed()
        hint = ""
        value_schema = None
        if re.match(r"(is|can|has|should|does|was|will)_", name):
            hint = " The attribute name suggests a boolean."
            value_schema = bool
        messages = [
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
                    f"OBJECT:\n{self._thing_story()}\n\n"
                    f"What is the value of the attribute `{name}`?{hint}"
                ),
            },
        ]
        with _op_scope():
            value, confidence = self._thing_query(
                messages,
                temperature=0.8,
                purpose=f"{type(self).__name__}.{name} · read",
                value_schema=value_schema,
            )
        child = self._thing_child(f"the attribute `{name}`", value, confidence)
        if self._thing_stateful:
            object.__setattr__(self, name, child)
            object.__setattr__(child, "_thing_owner", (self, name))
            self._thing_log(
                {
                    "event": "observe",
                    "name": name,
                    "value": _plain(child),
                    "confidence": confidence,
                }
            )
        return child

    def _thing_compare(self, symbol, other):
        """An ordering comparison is a judgment made in Thing space."""
        self._thing_ensure()
        self._thing_condense_if_needed()
        if isinstance(other, _Pending):
            other = other._resolve()
        left = self._thing_story()
        right = (
            other._thing_story()
            if isinstance(other, Thing)
            else json.dumps(other, default=repr, ensure_ascii=False)
        )
        messages = [
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
        with _op_scope():
            value, confidence = self._thing_query(
                messages, temperature=0.3, purpose=f"judge `{symbol}`",
                value_schema=bool,
            )
        return self._thing_child(
            f"the judgment `{symbol}` against ({_snip(right, 120)})",
            bool(value),
            confidence,
        )

    def _thing_call(self, name, args, kwargs, stack=()):
        self._thing_ensure()
        if not stack:
            self._thing_condense_if_needed()
        stack = tuple(stack) + (name,)
        if len(stack) > self._thing_depth_budget:
            raise ContinuationLimit(
                "imagined call depth exceeded: " + " -> ".join(stack)
            )
        kwargs = dict(kwargs)
        schema = kwargs.pop("returns", None)
        call_repr = (
            f"{name}("
            + ", ".join(
                [repr(_plain(a)) for a in args]
                + [f"{k}={_plain(v)!r}" for k, v in kwargs.items()]
            )
            + ")"
        )
        _, _, methods = self._thing_surface()
        messages = [
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
                    f"OBJECT:\n{self._thing_story()}\n\n"
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
                        + _render_schema(schema)
                        if schema is not None
                        else ""
                    )
                ),
            },
        ]
        floor = 1.0
        purpose = f"{type(self).__name__}.{_snip(call_repr, 60)} · imagined"
        with _op_scope(plan=True):
            return self._thing_plan(name, args, messages, schema, floor, stack, purpose)

    def _thing_plan(self, name, args, messages, schema, floor, stack, purpose):
        envelope = _action_envelope(schema)
        think = False  # steps start on the direct tier; stumbling escalates
        refusals = 0
        results = {}  # handles for values too large to retype: "$1" -> raw
        revealed = set()  # handles the model has read in full via get
        index = 0  # real actions executed, capped by the step budget
        turn = 0  # model replies, for the debug narration
        corrections = 0  # rejected returns draw on their own allowance

        def _rejected():
            nonlocal corrections
            corrections += 1
            if corrections > self._thing_correction_budget:
                raise ContinuationLimit(
                    f"imagined plan for `{name}` spent its correction budget"
                )

        while True:
            if index >= self._thing_step_budget:
                raise ContinuationLimit(
                    f"imagined plan for `{name}` exceeded its step budget"
                )
            turn += 1
            step = self._thing_complete_json(
                messages, temperature=0.3, purpose=purpose,
                think=think, schema=envelope,
            )
            if not isinstance(step, dict):
                # a bare JSON value in place of an action envelope is the
                # model returning its result directly; forgive it
                step = {"action": "return", "value": step}
            action = step.get("action")
            if action == "return":
                value = _substitute_handles(step.get("value"), results)
                # a return that tracks a stored value's prefix past what the
                # preview showed — without ever reading it — is fabrication:
                # the model is inventing data it has not seen
                rendered = json.dumps(value, default=repr, ensure_ascii=False)
                forged = None
                if len(rendered) > 350:
                    for handle, stored in results.items():
                        if (
                            handle not in revealed
                            and rendered != stored
                            and rendered[:250] == stored[:250]
                        ):
                            forged = handle
                            break
                if forged:
                    _rejected()
                    if _debugging.get():
                        _debug_action(
                            turn, f"return rejected: retyped {forged} from its preview"
                        )
                    messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"return rejected — this retypes {forged} from "
                                "its preview and invents the rest; the full "
                                "value was never shown to you. Return the "
                                f'exact string "{forged}" as the value to pass '
                                f"the stored data unchanged, or get {forged} "
                                "first if you truly need to transform it."
                            ),
                        }
                    )
                    continue
                if schema is not None:
                    ok, why = _matches(value, schema)
                    if not ok:
                        _rejected()
                        think = True  # the direct tier fumbled the shape
                        if _debugging.get():
                            _debug_action(
                                turn, f"{action} rejected: {_snip(why)}"
                            )
                        messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"return rejected — {why}. Emit a corrected "
                                    "return matching the schema: "
                                    + _render_schema(schema)
                                ),
                            }
                        )
                        continue
                confidence = max(0.01, min(float(step.get("confidence", 0.5)), floor, 0.99))
                threshold = _required.get()
                if threshold is not None and confidence < threshold and not think:
                    # below the required bar: one thinking retry before
                    # LowConfidence is allowed to fly
                    _rejected()
                    think = True
                    if _debugging.get():
                        _debug_action(
                            turn,
                            f"return (p {confidence:.2f}) below required "
                            f"{threshold:.2f}; rethinking",
                        )
                    messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"return rejected — its confidence {confidence:.2f} "
                                f"is below the required {threshold:.2f}. Reconsider "
                                "with care: gather more facts with get/call if that "
                                "helps, then return again, honestly."
                            ),
                        }
                    )
                    continue
                _check_required(confidence)
                if _debugging.get():
                    _debug_action(
                        turn,
                        f"{action} (p {confidence:.2f}) = "
                        + _snip(json.dumps(value, default=repr, ensure_ascii=False)),
                        final=True,
                    )
                if self._thing_stateful:
                    self._thing_log(
                        {
                            "event": "call",
                            "name": name,
                            "args": [_plain(a) for a in args],
                            "result": value,
                            "confidence": confidence,
                        }
                    )
                return self._thing_result(name, args, value, confidence)
            index += 1  # a real action consumes a step; corrections do not
            token = _op_stack.set(_op_stack.get() + (f"{name}()",))
            try:
                feedback, floor = self._thing_step(step, floor, stack, results)
            except ContinuationLimit:
                raise
            except Exception as error:
                feedback = f"error: {type(error).__name__}: {error}"
            finally:
                _op_stack.reset(token)
            if action == "get" and step.get("name") in results:
                revealed.add(step.get("name"))  # read in full: retyping is fair
            if (
                action in ("get", "call")
                and len(feedback) > _HANDLE_LIMIT
                and not str(step.get("name", "")).startswith("$")
            ):
                # too large to march through the model: hold it, hand back
                # a receipt that teaches the handle right when it matters
                handle = f"${len(results) + 1}"
                results[handle] = feedback
                feedback = _handle_receipt(handle, feedback)
                if len(results) == 1:
                    # the constrained decoder would refuse the handle string
                    # where the schema demands a list or object; from here
                    # on the runtime check alone guards the return's shape
                    envelope = _action_envelope(None)
            if feedback.startswith("refused"):
                refusals += 1
                if refusals >= 2 and not think:
                    think = True  # the plan is stumbling; let the steps think
                    if _debugging.get():
                        _debug_action(turn, "two refusals; rethinking")
            if _debugging.get():
                _debug_action(
                    turn, f"{_describe_step(step)} → {_snip(feedback)}"
                )
            messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": f"result of your {_describe_step(step)}: {feedback}",
                }
            )

    def _thing_result(self, call, args, value, confidence):
        """Wrap a call's return value as a child Thing, so results chain."""
        arg_repr = ", ".join(repr(_plain(a)) for a in args)
        return self._thing_child(
            f"the result of {call}({arg_repr})", value, confidence
        )

    def _thing_child(self, hop, value, confidence):
        """A Thing born from inference: carries its value and `.confidence`.
        Always a plain Thing, never the parent's class — a result is a new
        object, not the object it came from. Recast with `@ Class` when the
        class genuinely applies. `hop` is the one step that made this
        child; lineage renders shallow — own hop, parent's hop, root —
        with the middle of a long chain elided, never recited."""
        self._thing_ensure()
        parent_hop = self.__dict__.get("_thing_hop")
        if parent_hop is None:
            root = _snip(
                " | ".join(self._thing_parts) or type(self).__name__, 200
            )
            described = f"{hop} of ({root})"
        else:
            root = self.__dict__.get("_thing_root")
            described = f"{hop} of ({parent_hop} ← … ← {root})"
        child = Thing.__new__(Thing)
        d = child._thing_ensure()
        d["_thing_parts"] = [described]
        d["_thing_hop"] = hop
        d["_thing_root"] = root
        d["_thing_stateful"] = self._thing_stateful
        d["_thing_model_spec"] = self._thing_model_spec
        backend = self.__dict__.get("_thing_backend_obj")
        if backend is not None:
            object.__setattr__(child, "_thing_backend_obj", backend)
        object.__setattr__(child, "_thing_value", value)
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and key.isidentifier() and not key.startswith("_"):
                    object.__setattr__(child, key, item)
        object.__setattr__(child, "confidence", confidence)
        return child

    def _thing_step(self, step, floor, stack, results=None):
        """One plan step -> (feedback, floor)."""
        results = results if results is not None else {}
        action = step.get("action")
        target = step.get("name", "")
        if action == "get" and target in results:
            return results[target], floor  # the stored value, in full
        if not isinstance(target, str) or target.startswith("_"):
            return f"refused: invalid name {target!r}", floor
        if action == "call" and target in stack:
            return (
                f"refused: `{target}` is already executing (call stack: "
                f"{' -> '.join(stack)}); produce the result yourself and "
                "finish with a return",
                floor,
            )
        if action in ("set", "delete"):
            if target in stack:
                # a method is not state: recording an answer under the
                # method's own name would shadow it and break later calls
                return (
                    f"refused: `{target}` is the method being executed, not "
                    "state; record the fact under a state name (e.g. "
                    "`engine_status`), or just return the result",
                    floor,
                )
            current = self.__dict__.get(target, _UNSET)
            if current is _UNSET:
                current = getattr(type(self), target, _UNSET)
            if current is not _UNSET and not isinstance(current, (Thing, _Pending)):
                return (
                    f"refused: `{target}` is deterministic state the programmer "
                    "wrote; record the change under a new name instead",
                    floor,
                )
        if action == "get":
            if target == "value":
                # the story presents the carried value; asking for it must
                # be a free lookup, never a fresh imagined attribute
                carried = self.__dict__.get("_thing_value", _UNSET)
                if carried is not _UNSET:
                    floor = min(floor, self.__dict__.get("confidence", 1.0))
                    return (
                        json.dumps(carried, default=repr, ensure_ascii=False),
                        floor,
                    )
            value = getattr(self, target)
            if isinstance(value, _Pending):
                value = value._resolve()
            floor = min(floor, _confidence_of(value))
            plain = _plain(value)
            return json.dumps(plain, default=repr, ensure_ascii=False), floor
        if action == "set":
            confidence = step.get("confidence")
            confidence = max(
                0.01, min(float(confidence) if confidence is not None else floor, 0.99)
            )
            child = self._thing_child(
                f"the attribute `{target}` as set by an imagined plan",
                _substitute_handles(step.get("value"), results),
                confidence,
            )
            setattr(self, target, child)
            object.__setattr__(child, "_thing_owner", (self, target))
            return "ok", min(floor, confidence)
        if action == "delete":
            delattr(self, target)
            return "ok", floor
        if action == "call":
            call_args = _substitute_handles(step.get("args") or [], results)
            attr = getattr(self, target)
            if isinstance(attr, _Pending):
                value = self._thing_call(target, tuple(call_args), {}, stack)
            elif callable(attr):
                value = attr(*call_args)
            else:
                return f"error: `{target}` is not callable", floor
            floor = min(floor, _confidence_of(value))
            plain = _plain(value)
            return json.dumps(plain, default=repr, ensure_ascii=False), floor
        if action == "define":
            self.define(target, str(step.get("meaning", "")))
            return "ok", floor
        return (
            f"refused: unknown action {action!r}; to return data, reply "
            '{"action": "return", "value": <the data>}',
            floor,
        )

    # -- the attribute protocol seams ---------------------------------------

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        value = object.__getattribute__(self, "__dict__").get("_thing_value", _UNSET)
        if value is not _UNSET:
            # Code before inference: the carried value's real attributes and
            # methods win; only names the value lacks fall to imagination.
            try:
                return getattr(value, name)
            except AttributeError:
                pass
        return _Pending(self, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)
        if self._thing_ensure().get("_thing_stateful", True):
            event = {"event": "set", "name": name, "value": _plain(value)}
            confidence = _confidence_of(value)
            if confidence < 1.0:
                event["confidence"] = confidence
            self._thing_log(event)

    def __delattr__(self, name):
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        try:
            object.__delattr__(self, name)
        except AttributeError:
            pass
        if self._thing_ensure().get("_thing_stateful", True):
            self._thing_log({"event": "delete", "name": name})

    # -- public surface -----------------------------------------------------

    def define(self, name, meaning):
        """Record what `name` means — as story text, never as code."""
        self._thing_log({"event": "define", "name": name, "meaning": meaning})

    @property
    def __story__(self):
        self._thing_ensure()
        return list(self._thing_journal)

    @property
    def __source__(self):
        """The object rendered as Python-like source, as it looks right now.

        Written code appears verbatim; written state, imagined state, and
        `define`d vocabulary appear as annotated assignments and stubs.
        A view for humans, not round-trippable source.
        """
        self._thing_ensure()
        cls = type(self)
        pad = "    "
        carried = self.__dict__.get("_thing_value", _UNSET)

        def emit(block):
            sections.append("\n".join(pad + line if line.strip() else "" for line in block.split("\n")))

        sections = []

        doc = _own_doc(cls)
        story_bits = ([doc] if doc else []) + [p.strip() for p in self._thing_parts]
        if story_bits:
            text = "\n\n".join(story_bits)
            if "\n" in text:
                emit('"""\n' + text + '\n"""')
            else:
                emit(f'"""{text}"""')

        if carried is not _UNSET:
            confidence = self.__dict__.get("confidence", 0.5)
            emit(f"# carries value: {carried!r} (p = {confidence:.2f})")

        _, attrs, _ = self._thing_surface()
        if attrs:
            emit("\n".join(f"{name} = {value!r}" for name, value in attrs.items()))

        state_lines = []
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if carried is not _UNSET and name == "confidence":
                continue
            if isinstance(value, _Pending):
                continue
            if isinstance(value, Thing) and "_thing_value" in value.__dict__:
                inner = value.__dict__["_thing_value"]
                confidence = value.__dict__.get("confidence", 0.5)
                state_lines.append(f"{name} = {inner!r}  # imagined (p = {confidence:.2f})")
            else:
                state_lines.append(f"{name} = {_plain(value)!r}  # written (p = 1.0)")
        if state_lines:
            emit("\n".join(state_lines))

        seen = set()
        for klass in cls.__mro__:
            if klass is Thing:
                break
            for name, value in vars(klass).items():
                if name.startswith("_") or name in seen or not callable(value):
                    continue
                seen.add(name)
                try:
                    emit(textwrap.dedent(inspect.getsource(value)).rstrip())
                except (OSError, TypeError):
                    docstr = inspect.getdoc(value)
                    stub = f"def {name}(self, *args, **kwargs):"
                    if docstr:
                        stub += f'\n    """{docstr}"""'
                    emit(stub + "\n    ...  # written, but source unavailable")

        defined = {}
        for event in self._thing_journal:
            if event.get("event") == "define":
                defined[str(event.get("name"))] = str(event.get("meaning", ""))
        for name, meaning in defined.items():
            emit(
                f"def {name}(self):\n"
                f'    """{meaning}"""\n'
                f"    ...  # imagined: no written body, a plan is inferred at call time"
            )

        bases = ", ".join(base.__name__ for base in cls.__bases__)
        header = f"class {cls.__name__}({bases}):"
        if not sections:
            return header + "\n" + pad + "pass"
        return header + "\n" + "\n\n".join(sections)

    def __getstate__(self):
        """The object as a JSON-able document: description, state, story,
        flags. `blob @ Car` casts it back to life; `pickle` uses the same
        mechanism."""
        self._thing_ensure()
        carried = self.__dict__.get("_thing_value", _UNSET)
        state = {}
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if carried is not _UNSET and name == "confidence":
                continue
            if isinstance(value, _Pending):
                continue
            if isinstance(value, Thing) and "_thing_value" in value.__dict__:
                state[name] = {
                    "__imagined__": value.__dict__["_thing_value"],
                    "confidence": value.__dict__.get("confidence", 0.5),
                }
            else:
                try:
                    json.dumps(value)
                    state[name] = value
                except TypeError:
                    state[name] = repr(value)
        model = self._thing_model_spec
        blob = {
            "class": type(self).__name__,
            "description": list(self._thing_parts),
            "state": state,
            "story": list(self._thing_journal),
            "stateful": self._thing_stateful,
            "model": model if isinstance(model, str) else None,
        }
        if carried is not _UNSET:
            blob["value"] = carried
            blob["confidence"] = self.__dict__.get("confidence", 0.5)
        return blob

    def _thing_restore(self, blob):
        d = self._thing_ensure()
        d["_thing_parts"] = list(blob.get("description", []))
        d["_thing_journal"] = list(blob.get("story", []))
        d["_thing_stateful"] = bool(blob.get("stateful", True))
        d["_thing_model_spec"] = blob.get("model")
        for name, value in blob.get("state", {}).items():
            if isinstance(value, dict) and ("__imagined__" in value or "__approx__" in value):
                inner = value.get("__imagined__", value.get("__approx__"))
                object.__setattr__(
                    self,
                    name,
                    self._thing_child(
                        f"the restored attribute `{name}`",
                        inner,
                        float(value.get("confidence", 0.5)),
                    ),
                )
            else:
                object.__setattr__(self, name, value)
        if "value" in blob:
            object.__setattr__(self, "_thing_value", blob["value"])
            object.__setattr__(self, "confidence", float(blob.get("confidence", 0.5)))

    def __setstate__(self, blob):
        self._thing_restore(blob)

    def __repr__(self):
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is not _UNSET:
            return repr(value)
        self._thing_ensure()
        described = " | ".join(self._thing_parts)
        label = f" {described!r}" if described else ""
        return f"<{type(self).__name__}{label}>"

    # A Thing born from inference carries its value; these delegate to it so
    # imagined values still behave like values (printing, arithmetic,
    # comparison, iteration) wherever Python lets them.

    def __call__(self, *args, **kwargs):
        """Accessing an attribute reads the recorded fact; CALLING it
        re-derives: a Thing that lives as an attribute of an owner
        delegates the call back as a fresh imagined method call, so a
        recorded `can_drive` never shadows `can_drive()`."""
        owner = self.__dict__.get("_thing_owner")
        if owner is not None:
            obj, name = owner
            return obj._thing_call(name, args, kwargs)
        raise TypeError(f"'{type(self).__name__}' object is not callable")

    def __bool__(self):
        value = self.__dict__.get("_thing_value", _UNSET)
        return True if value is _UNSET else bool(value)

    def __str__(self):
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is _UNSET:
            return repr(self)
        return value if isinstance(value, str) else str(value)

    def __format__(self, spec):
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is _UNSET:
            return format(repr(self), spec)
        return format(value, spec)

    def __eq__(self, other):
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is _UNSET:
            return NotImplemented
        return value == _plain(other)

    def __hash__(self):
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is _UNSET:
            return object.__hash__(self)
        return hash(value)

    def _thing_value_or_raise(self):
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is _UNSET:
            raise TypeError(f"{type(self).__name__} object carries no value")
        return value

    def __iter__(self):
        return iter(self._thing_value_or_raise())

    def __len__(self):
        return len(self._thing_value_or_raise())

    def __getitem__(self, key):
        return self._thing_value_or_raise()[key]

    def __contains__(self, item):
        return _plain(item) in self._thing_value_or_raise()

    def __int__(self):
        return int(self._thing_value_or_raise())

    def __float__(self):
        return float(self._thing_value_or_raise())

    def __index__(self):
        return int(self._thing_value_or_raise())

    def __pos__(self):
        """`+t` — the value, no questions asked: the carried value if the
        Thing has collapsed, else the programmer's own words (the story
        text). Never runs inference; collapsing happens through typing
        (`t @ int`). A Thing that failed an `@` requirement yields None."""
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is not _UNSET:
            return value
        self._thing_ensure()
        return " | ".join(self._thing_parts)

    def __invert__(self):
        """`~t` — the probability, a bare float. Your own words are certain:
        an uncollapsed story Thing is 1.0; anything inference produced is
        always less. Never runs inference."""
        if "_thing_value" in self.__dict__:
            return self.__dict__.get("confidence", 0.5)
        return 1.0

    def __matmul__(self, operand):
        """`t @ int` / `t @ {"title": str, "year": int}` — approximate AS a
        type or JSON schema template; always returns a Thing — one whose
        value conforms, or Thing(None, 0.01) if the imagination cannot
        conform. `t @ SomeThingSubclass` re-classes the story as that
        subclass, free. `t @ SomeClass` imagines constructor arguments and
        lets the real constructor build the instance. `t @ 0.8` —
        confidence gate; the Thing itself when confidence >= 0.8, else a
        Thing whose value is gone but whose probability survives for
        diagnosis; never runs inference."""
        if isinstance(operand, type) and issubclass(operand, Thing):
            clone = operand.__new__(operand)
            clone._thing_restore(self.__getstate__())
            d = clone._thing_ensure()
            d["_thing_model_spec"] = self.__dict__.get("_thing_model_spec")
            backend = self.__dict__.get("_thing_backend_obj")
            if backend is not None:
                object.__setattr__(clone, "_thing_backend_obj", backend)
            return clone
        if isinstance(operand, type) and not issubclass(
            operand, (str, int, float, bool, list, dict)
        ):
            return self._thing_construct(operand)
        if isinstance(operand, (type, dict, list)):
            value = self.__dict__.get("_thing_value", _UNSET)
            if value is not _UNSET and _matches(value, operand)[0]:
                return self
            value, confidence = self._thing_collapse(operand)
            if value is _UNSET:
                return self._thing_failure(
                    f"could not be approximated as {_render_schema(operand)}",
                    confidence,
                )
            return self._thing_child(
                f"the approximation as {_render_schema(operand)}",
                value,
                confidence,
            )
        threshold = float(operand)
        value = self.__dict__.get("_thing_value", _UNSET)
        if value is _UNSET:
            return self  # a story Thing is the programmer's words: certain
        confidence = self.__dict__.get("confidence", 0.5)
        if confidence >= threshold or value is None:
            return self
        return self._thing_failure(
            f"fell below the confidence >= {threshold} requirement", confidence
        )

    def _thing_failure(self, why, confidence):
        """A Thing whose value is gone but whose probability survives."""
        return self._thing_child(f"a value that {why}", None, confidence)

    def _thing_construct(self, cls):
        """Approximate as an arbitrary class: the imagination supplies the
        constructor arguments, the real constructor builds the instance."""
        try:
            signature = inspect.signature(cls)
        except (TypeError, ValueError):
            return self._thing_failure(
                f"could not be approximated as {cls.__name__}", 0.01
            )
        schema = {}
        for name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            annotation = param.annotation
            schema[name] = (
                annotation
                if isinstance(annotation, type)
                and issubclass(annotation, (str, int, float, bool, list, dict))
                else object
            )
        kwargs, confidence = self._thing_collapse(schema or dict)
        if kwargs is _UNSET:
            return self._thing_failure(
                f"could not be approximated as {cls.__name__}", confidence
            )
        try:
            instance = cls(**kwargs)
        except Exception as error:
            return self._thing_failure(
                f"could not be constructed as {cls.__name__} "
                f"({type(error).__name__})",
                0.01,
            )
        return self._thing_child(
            f"the approximation as {cls.__name__}", instance, confidence
        )

    def _thing_collapse(self, schema=None):
        """Resolve the single (optionally typed) value a Thing stands for."""
        self._thing_ensure()
        self._thing_condense_if_needed()
        key = _render_schema(schema) if schema is not None else ""
        cache = self.__dict__.get("_thing_collapsed") or {}
        if key in cache:
            return cache[key]
        demand = (
            f" The value MUST be a JSON {_render_schema(schema)}."
            if schema is not None
            else ""
        )
        messages = [
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
                    f"OBJECT:\n{self._thing_story()}\n\n"
                    f"What single value does this object represent?{demand}"
                ),
            },
        ]
        with _op_scope():
            value, confidence = self._thing_query(
                messages,
                temperature=0.3,
                purpose=f"{type(self).__name__} @ {key} · collapse"
                if key
                else f"{type(self).__name__} · collapse",
                value_schema=schema,
                check=False,  # a mismatch must stay a diagnosis, not a raise
            )
        if schema is not None and not _matches(value, schema)[0]:
            # single shot, no retries: the mismatch itself is the answer,
            # and the model's confidence survives as the diagnosis
            return _UNSET, confidence
        _check_required(confidence)
        if self._thing_stateful:
            cache = dict(cache)
            cache[key] = (value, confidence)
            object.__setattr__(self, "_thing_collapsed", cache)
            event = {"event": "collapse", "value": value, "confidence": confidence}
            if schema is not None:
                event["type"] = _render_schema(schema)
            self._thing_log(event)
        return value, confidence

    def __lt__(self, other):
        return self._thing_compare("<", other)

    def __le__(self, other):
        return self._thing_compare("<=", other)

    def __gt__(self, other):
        return self._thing_compare(">", other)

    def __ge__(self, other):
        return self._thing_compare(">=", other)

    def __add__(self, other):
        return self._thing_value_or_raise() + _plain(other)

    def __radd__(self, other):
        return _plain(other) + self._thing_value_or_raise()

    def __sub__(self, other):
        return self._thing_value_or_raise() - _plain(other)

    def __rsub__(self, other):
        return _plain(other) - self._thing_value_or_raise()

    def __mul__(self, other):
        return self._thing_value_or_raise() * _plain(other)

    def __rmul__(self, other):
        return _plain(other) * self._thing_value_or_raise()

    def __truediv__(self, other):
        return self._thing_value_or_raise() / _plain(other)

    def __rtruediv__(self, other):
        return _plain(other) / self._thing_value_or_raise()
