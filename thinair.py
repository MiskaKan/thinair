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
import urllib.error
import urllib.request

__all__ = ["Thing"]

_DEFAULT_BASE_URL = os.environ.get("THINAIR_BASE_URL", "http://127.0.0.1:8000/v1")
_DEFAULT_API_KEY = os.environ.get("THINAIR_API_KEY", "1234")
_DEFAULT_MODEL = os.environ.get("THINAIR_MODEL", "Qwen3.6-35B-A3B-oQ6-mtp")
_DEFAULT_MAX_TOKENS = int(os.environ.get("THINAIR_MAX_TOKENS", "32768"))
_DEFAULT_THINK_CHUNK = int(os.environ.get("THINAIR_THINK_CHUNK", "2048"))

_UNSET = object()
_required = contextvars.ContextVar("thing_required_confidence", default=None)
_debugging = contextvars.ContextVar(
    "thing_debug", default=os.environ.get("THINAIR_DEBUG", "") not in ("", "0")
)
_purpose = contextvars.ContextVar("thing_purpose", default="inference")


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

class _HTTPBackend:
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url,
        api_key,
        model,
        max_tokens=_DEFAULT_MAX_TOKENS,
        request_extra=None,
        think_chunk=_DEFAULT_THINK_CHUNK,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = int(max_tokens)
        self.request_extra = dict(request_extra or {})
        self.think_chunk = int(think_chunk)
        self.last_meta = {}
        self._json_mode = True  # optimistic; dropped if the server rejects it

    def complete(self, messages, temperature=0.7):
        """Thinking runs in blocks of `think_chunk` tokens. Concluding with
        an answer inside a block ends the call; running out of a block
        means "still thinking" — the reasoning is carried forward and the
        model is nudged toward answering. When the thinking allowance
        (`max_tokens` in total) is spent, one final answer block gets the
        full `max_tokens` budget so the answer itself is never squeezed.
        `think_chunk=0` restores single-shot completions."""
        if self.think_chunk <= 0:
            content, _, self.last_meta = self._request(
                messages, temperature, self.max_tokens
            )
            return content
        thoughts = []
        blocks = max(1, self.max_tokens // self.think_chunk)
        for block in range(blocks):
            content, reasoning, meta = self._request(
                self._with_thoughts(messages, thoughts, final=False),
                temperature,
                self.think_chunk,
            )
            meta["thinking_blocks"] = block + 1
            self.last_meta = meta
            concluded = bool(content) and meta.get("finish_reason") != "length"
            if _debugging.get() and (reasoning or not concluded):
                _debug_block(
                    _purpose.get(),
                    f"thinking block {block + 1}",
                    reasoning,
                    "" if concluded else content,
                    meta,
                )
            if concluded:
                return content  # concluded within the block: chose to answer
            if reasoning:
                thoughts.append(reasoning)
            if content:
                thoughts.append(content)  # a cut answer is a thought too
        # allowance spent: the answer block, full budget, no more thinking
        content, reasoning, meta = self._request(
            self._with_thoughts(messages, thoughts, final=True),
            temperature,
            self.max_tokens,
        )
        meta["thinking_blocks"] = blocks
        self.last_meta = meta
        if _debugging.get() and reasoning:
            _debug_block(_purpose.get(), "answer block", reasoning, "", meta)
        return content

    def _with_thoughts(self, messages, thoughts, final):
        if not thoughts and not final:
            return messages
        nudge = (
            "Answer now with exactly one JSON object and nothing else."
            if final
            else "Continue reasoning only if truly necessary; otherwise "
            "answer now with exactly one JSON object."
        )
        extended = list(messages)
        if thoughts:
            extended.append(
                {"role": "assistant", "content": "(thinking so far)\n" + "\n".join(thoughts)}
            )
        extended.append({"role": "user", "content": nudge})
        return extended

    def _request(self, messages, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            if self._json_mode and error.code in (400, 404, 422):
                self._json_mode = False  # server has no JSON mode; remember
                return self._request(messages, temperature, max_tokens)
            raise
        choice = data["choices"][0]
        meta = dict(data.get("usage") or {})
        meta["finish_reason"] = choice.get("finish_reason")
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
    chunk = cfg.get("think_chunk", _DEFAULT_THINK_CHUNK)
    if spec is None:
        return _HTTPBackend(
            cfg["base_url"], cfg["api_key"], cfg["model"], max_tokens, extra, chunk
        )
    if isinstance(spec, str):
        if spec.startswith("file://"):
            raise NotImplementedError(
                "embedded models are future work; point `model` at a server URL"
            )
        if spec.startswith("http://") or spec.startswith("https://"):
            return _HTTPBackend(
                spec, cfg["api_key"], cfg["model"], max_tokens, extra, chunk
            )
        return _HTTPBackend(
            cfg["base_url"], cfg["api_key"], spec, max_tokens, extra, chunk
        )
    if hasattr(spec, "complete"):
        return spec
    if callable(spec):
        return _CallableBackend(spec)
    raise TypeError(f"cannot use {spec!r} as an inference backend")


def _extract_json(text):
    """All balanced JSON objects in the text; the answer is the LAST one
    that looks like one (reasoning models explain first, answer last)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
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


def _render_schema(schema):
    if isinstance(schema, type):
        return schema.__name__
    if isinstance(schema, dict):
        return "{" + ", ".join(f'"{k}": {_render_schema(v)}' for k, v in schema.items()) + "}"
    if isinstance(schema, list):
        return "[" + (_render_schema(schema[0]) + ", ..." if schema else "...") + "]"
    return json.dumps(schema)


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


def _debug_block(purpose, label, reasoning, partial, meta):
    lines = [f"┌─ thinair · {purpose} — {label} " + "─" * max(1, 44 - len(purpose) - len(str(label)))]
    for line in str(reasoning or "(no reasoning returned)").splitlines() or [""]:
        lines.append(f"│ {line}")
    if partial:
        lines.append("├─ partial answer, cut by the block " + "─" * 23)
        for line in str(partial).splitlines() or [""]:
            lines.append(f"│ {line}")
    summary = ", ".join(f"{k}={v}" for k, v in (meta or {}).items() if v is not None)
    if summary:
        lines.append(f"├─ {summary}")
    lines.append("└" + "─" * 59)
    print("\n".join(lines), file=sys.stderr)


def _debug_dump(purpose, messages, text, meta=None):
    lines = [f"┌─ thinair · {purpose} " + "─" * max(1, 56 - len(purpose))]
    for message in messages:
        lines.append(f"│ [{message['role']}]")
        for line in str(message.get("content", "")).splitlines() or [""]:
            lines.append(f"│   {line}")
    lines.append("├─ raw completion " + "─" * 42)
    for line in str(text).splitlines() or [""]:
        lines.append(f"│ {line}")
    if meta:
        summary = ", ".join(f"{k}={v}" for k, v in meta.items() if v is not None)
        if summary:
            lines.append(f"├─ {summary}")
    lines.append("└" + "─" * 59)
    print("\n".join(lines), file=sys.stderr)


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
    _thing_defaults: dict = {}

    def __init__(self, *parts, stateful=True, model=None, confidence=None, **state):
        self._thing_stateful = bool(stateful)
        self._thing_model_spec = model
        if confidence is not None:
            carried = parts[0] if parts else None
            self._thing_parts = [
                "a value the programmer asserted with explicit doubt",
                "value: " + json.dumps(carried, default=repr),
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
        request_extra=None, think_chunk=None,
    ):
        """Set class-wide backend defaults; subclasses inherit via the MRO.
        `request_extra` is merged verbatim into every HTTP payload — the
        escape hatch for server-specific knobs (thinking budgets, template
        kwargs, reasoning effort)."""
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
        if think_chunk is not None:
            cfg["think_chunk"] = int(think_chunk)
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
            "think_chunk": _DEFAULT_THINK_CHUNK,
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

    def _thing_story(self):
        self._thing_ensure()
        doc, attrs, methods = self._thing_surface()
        lines = []
        if type(self) is not Thing:
            lines.append(f"Class: {type(self).__name__}" + (f" — {doc}" if doc else ""))
        if attrs:
            lines.append("Class attributes (certain): " + json.dumps(attrs, default=repr))
        if methods:
            listing = "; ".join(
                f"{name}(...)" + (f" — {docstr}" if docstr else "")
                for name, docstr in methods.items()
            )
            lines.append("Real methods (these execute actual code): " + listing)
        if self._thing_parts:
            lines.append("Described as: " + " | ".join(self._thing_parts))
        state = {
            k: _plain(v) for k, v in self.__dict__.items() if not k.startswith("_")
        }
        if state:
            lines.append("Current state (authoritative): " + json.dumps(state, default=repr))
        if self._thing_journal:
            lines.append("History, oldest first:")
            for i, event in enumerate(self._thing_journal, 1):
                lines.append(f"  {i}. {json.dumps(event, default=repr)}")
        return "\n".join(lines) or "An unspecified thing."

    def _thing_complete_json(self, messages, temperature, purpose="inference"):
        backend = self._thing_backend()
        last_error = None
        for _ in range(3):
            token = _purpose.set(purpose)
            try:
                text = backend.complete(messages, temperature=temperature)
            except TypeError:
                text = backend.complete(messages)
            finally:
                _purpose.reset(token)
            meta = getattr(backend, "last_meta", None) or {}
            if _debugging.get():
                _debug_dump(purpose, messages, text, meta)
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
                messages = messages + [
                    {"role": "assistant", "content": text[:1000]},
                    {"role": "user", "content": "Reply with exactly one JSON object and nothing else."},
                ]
        raise RuntimeError(f"backend produced no parseable JSON: {last_error}")

    # -- the two continuations: read and call -------------------------------

    def _thing_read(self, name):
        self._thing_ensure()
        hint = ""
        if re.match(r"(is|can|has|should|does|was|will)_", name):
            hint = " The attribute name suggests a boolean."
        messages = [
            {
                "role": "system",
                "content": (
                    "You resolve attributes of a probabilistic object by inference "
                    "over its story. Reply with exactly one JSON object, no prose: "
                    '{"value": <json value>, "confidence": <0..1>}. Confidence has '
                    "exactly one meaning: the probability that the value is "
                    "correct. Use natural JSON types: numbers as numbers (a year, "
                    "a count, a die face are integers, never strings), booleans as "
                    "booleans. If the story treats the attribute as a random "
                    "outcome (a roll, a draw, a spin), sample one concrete outcome "
                    "and set confidence to that outcome's probability. If it is a "
                    "fixed but unstated fact, treat it as a draw from your prior: "
                    "answer your single best concrete guess of the natural type, "
                    "with confidence matching how constrained it is (a specific "
                    "year from a stated decade might be 0.1; a wide-open guess "
                    "0.02 — low confidence is honest, never evasive). If the "
                    "attribute genuinely has no value because nothing of the kind "
                    "exists, answer null with your confidence in that absence. "
                    'Never answer placeholder strings such as "unknown", "n/a", '
                    'or "unspecified". '
                    "Never contradict the story, its state, or its history."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT STORY:\n{self._thing_story()}\n\n"
                    f"Resolve the attribute `{name}`.{hint}"
                ),
            },
        ]
        data = self._thing_complete_json(
            messages, temperature=0.8, purpose=f"read `{name}`"
        )
        confidence = max(0.01, min(float(data.get("confidence", 0.5)), 0.99))
        _check_required(confidence)
        origin = " | ".join(self._thing_parts) or type(self).__name__
        child = self._thing_child(
            f"the attribute `{name}` of ({origin})", data.get("value"), confidence
        )
        if self._thing_stateful:
            object.__setattr__(self, name, child)
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
        if isinstance(other, _Pending):
            other = other._resolve()
        left = self._thing_story()
        right = (
            other._thing_story()
            if isinstance(other, Thing)
            else json.dumps(other, default=repr)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You judge order comparisons between probabilistic objects. "
                    "Reply with exactly one JSON object, no prose: "
                    '{"value": true or false, "confidence": <0..1>}. Compare '
                    "what the objects ARE — magnitude, size, cost, capability, "
                    "whatever their stories make relevant — never how they are "
                    "spelled, unless both are clearly plain text. If several "
                    "natural senses disagree, choose the most likely one and "
                    "lower your confidence to reflect the ambiguity; confidence "
                    "is the probability your judgment is correct. Never "
                    "contradict recorded state."
                ),
            },
            {
                "role": "user",
                "content": f"LEFT:\n{left}\n\nRIGHT:\n{right}\n\nIs LEFT {symbol} RIGHT?",
            },
        ]
        data = self._thing_complete_json(
            messages, temperature=0.3, purpose=f"judge `{symbol}`"
        )
        confidence = max(0.01, min(float(data.get("confidence", 0.5)), 0.99))
        _check_required(confidence)
        origin = " | ".join(self._thing_parts) or type(self).__name__
        return self._thing_child(
            f"the judgment: ({origin}) {symbol} ({right[:120]})",
            bool(data.get("value")),
            confidence,
        )

    def _thing_call(self, name, args, kwargs, depth=0):
        self._thing_ensure()
        if depth >= self._thing_depth_budget:
            raise ContinuationLimit(f"imagined call depth exceeded at `{name}`")
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
                    "You are the imagination runtime of a probabilistic Python "
                    "object. A method that has no written code was called; you "
                    "imagine its behavior by acting on the object step by step.\n"
                    "Each turn, reply with exactly one JSON object, no prose — "
                    "one of:\n"
                    '{"action": "get", "name": "<attr>"}\n'
                    '{"action": "set", "name": "<attr>", "value": <json>, "confidence": <0..1>}\n'
                    '{"action": "delete", "name": "<attr>"}\n'
                    '{"action": "call", "name": "<method>", "args": [<json>...]}\n'
                    '{"action": "define", "name": "<name>", "meaning": "<text>"}\n'
                    '{"action": "return", "value": <json>, "confidence": <0..1>}\n'
                    "Rules: prefer real methods when one fits — they execute actual "
                    "code. Never write or generate code. Keep plans short; mutate "
                    "state with `set` when the action changes the object — the "
                    "value is recorded with your confidence in it (omit confidence "
                    "to use the plan's current floor). You may create new "
                    "attributes freely and overwrite or delete any value that "
                    "carries confidence; values the programmer wrote bare are "
                    "certain and out of reach — record changes to those under "
                    "new names. Never call the method you are currently "
                    "executing; do its work yourself. Always "
                    'finish with "return".'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT STORY:\n{self._thing_story()}\n\n"
                    f"Execute the call: {call_repr}"
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
        for _ in range(self._thing_step_budget):
            step = self._thing_complete_json(
                messages, temperature=0.3, purpose=f"imagine `{name}(...)`"
            )
            action = step.get("action")
            if action == "return":
                value = step.get("value")
                if schema is not None:
                    ok, why = _matches(value, schema)
                    if not ok:
                        messages.append({"role": "assistant", "content": json.dumps(step)})
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
                _check_required(confidence)
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
            try:
                feedback, floor = self._thing_step(step, floor, depth, name)
            except ContinuationLimit:
                raise
            except Exception as error:
                feedback = f"error: {type(error).__name__}: {error}"
            messages.append({"role": "assistant", "content": json.dumps(step)})
            messages.append({"role": "user", "content": f"result: {feedback}"})
        raise ContinuationLimit(f"imagined plan for `{name}` exceeded its step budget")

    def _thing_result(self, call, args, value, confidence):
        """Wrap a call's return value as a child Thing, so results chain."""
        origin = " | ".join(self._thing_parts) or type(self).__name__
        arg_repr = ", ".join(repr(_plain(a)) for a in args)
        return self._thing_child(
            f"the result of {call}({arg_repr}) on ({origin})", value, confidence
        )

    def _thing_child(self, described, value, confidence):
        """A Thing born from inference: carries its value and `.confidence`."""
        child = type(self).__new__(type(self))
        d = child._thing_ensure()
        d["_thing_parts"] = [described, "value: " + json.dumps(value, default=repr)]
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

    def _thing_step(self, step, floor, depth, executing):
        action = step.get("action")
        target = step.get("name", "")
        if not isinstance(target, str) or target.startswith("_"):
            return f"refused: invalid name {target!r}", floor
        if action == "call" and target == executing:
            return (
                f"refused: `{target}` is the method you are executing; do its "
                "work yourself and finish with a return",
                floor,
            )
        if action in ("set", "delete"):
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
            value = getattr(self, target)
            if isinstance(value, _Pending):
                value = value._resolve()
            floor = min(floor, _confidence_of(value))
            return json.dumps(_plain(value), default=repr), floor
        if action == "set":
            confidence = step.get("confidence")
            confidence = max(
                0.01, min(float(confidence) if confidence is not None else floor, 0.99)
            )
            origin = " | ".join(self._thing_parts) or type(self).__name__
            child = self._thing_child(
                f"the attribute `{target}` as set by an imagined plan on ({origin})",
                step.get("value"),
                confidence,
            )
            setattr(self, target, child)
            return "ok", min(floor, confidence)
        if action == "delete":
            delattr(self, target)
            return "ok", floor
        if action == "call":
            call_args = step.get("args") or []
            attr = getattr(self, target)
            if isinstance(attr, _Pending):
                value = self._thing_call(target, tuple(call_args), {}, depth + 1)
            elif callable(attr):
                value = attr(*call_args)
            else:
                return f"error: `{target}` is not callable", floor
            floor = min(floor, _confidence_of(value))
            return json.dumps(_plain(value), default=repr), floor
        if action == "define":
            self.define(target, str(step.get("meaning", "")))
            return "ok", floor
        return f"refused: unknown action {action!r}", floor

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
        value_part = (
            "value: " + json.dumps(carried, default=repr) if carried is not _UNSET else None
        )
        story_bits = ([doc] if doc else []) + [
            p.strip() for p in self._thing_parts if p != value_part
        ]
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
                        f"the attribute `{name}` of a restored object",
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
            origin = " | ".join(self._thing_parts) or type(self).__name__
            return self._thing_child(
                f"({origin}) approximated as {_render_schema(operand)}",
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
        origin = " | ".join(self._thing_parts) or type(self).__name__
        return self._thing_child(f"({origin}) {why}", None, confidence)

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
        origin = " | ".join(self._thing_parts) or type(self).__name__
        return self._thing_child(
            f"({origin}) approximated as {cls.__name__}", instance, confidence
        )

    def _thing_collapse(self, schema=None):
        """Resolve the single (optionally typed) value a Thing stands for."""
        self._thing_ensure()
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
                    "You collapse a probabilistic object to the single JSON "
                    "value it most naturally stands for. Reply with exactly one "
                    'JSON object, no prose: {"value": <json value>, '
                    '"confidence": <0..1>}. A described value stands for '
                    'itself: an object described as "Cat" collapses to "Cat"; '
                    "an object with one obvious quantity collapses to that "
                    "number. Use natural JSON types and never contradict the "
                    "story or its state."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT STORY:\n{self._thing_story()}\n\n"
                    f"Collapse this object to its value.{demand}"
                ),
            },
        ]
        data = self._thing_complete_json(
            messages,
            temperature=0.3,
            purpose=f"collapse @ {key}" if key else "collapse",
        )
        value = data.get("value")
        confidence = max(0.01, min(float(data.get("confidence", 0.5)), 0.99))
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
