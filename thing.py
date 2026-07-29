"""thing.py — probabilistic objects with one axiom.

An object is a story. Every interaction is a continuation of that story;
the continuation is appended. Written code and recorded state are
authoritative (p = 1.0, bare values); inference fills only the silence,
always returning wrapped values with confidence < 1.0.

Spec: SPEC.md. The entire public surface is the single class `Thing`.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
import os
import re
import urllib.request

__all__ = ["Thing"]

_DEFAULT_BASE_URL = os.environ.get("THING_BASE_URL", "http://127.0.0.1:8000/v1")
_DEFAULT_API_KEY = os.environ.get("THING_API_KEY", "1234")
_DEFAULT_MODEL = os.environ.get("THING_MODEL", "Qwen3.6-35B-A3B-oQ6-mtp")

_UNSET = object()
_required = contextvars.ContextVar("thing_required_confidence", default=None)


class LowConfidence(Exception):
    """A resolution came back below the active `Thing.require` threshold."""


class ContinuationLimit(Exception):
    """An imagined plan exceeded its step or depth budget."""


# ---------------------------------------------------------------------------
# Approx — the p < 1 wrapper. Duck-types as its value by subclassing it.
# ---------------------------------------------------------------------------

class Approx:
    """Marker base for values produced by inference. Carries `.confidence`."""

    confidence: float


class _ApproxBool(int, Approx):
    """bool cannot be subclassed; an int that prints like a bool."""

    def __new__(cls, value):
        return super().__new__(cls, 1 if value else 0)

    def __repr__(self):
        return "True" if self else "False"

    __str__ = __repr__


class _ApproxNone(Approx):
    def __bool__(self):
        return False

    def __eq__(self, other):
        return other is None or isinstance(other, _ApproxNone)

    def __hash__(self):
        return hash(None)

    def __repr__(self):
        return "None"


class _ApproxOpaque(Approx):
    """Fallback for values whose type cannot be subclassed."""

    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return repr(self.value)


_approx_classes: dict[type, type] = {}


def _make_approx(value, confidence):
    confidence = max(0.01, min(float(confidence), 0.99))
    if value is None:
        out = _ApproxNone()
    elif isinstance(value, bool):
        out = _ApproxBool(value)
    else:
        base = type(value)
        cls = _approx_classes.get(base)
        if cls is None:
            try:
                cls = type("Approx", (base, Approx), {"__module__": __name__})
            except TypeError:
                cls = _ApproxOpaque
            _approx_classes[base] = cls
        out = cls(value) if cls is not _ApproxOpaque else _ApproxOpaque(value)
    out.confidence = confidence
    return out


def _plain(value):
    """Strip any Approx/_Pending wrapper down to the native value."""
    if isinstance(value, _Pending):
        value = value._resolve()
    if isinstance(value, _ApproxNone):
        return None
    if isinstance(value, _ApproxBool):
        return bool(int(value))
    if isinstance(value, _ApproxOpaque):
        return value.value
    if isinstance(value, Approx):
        return type(value).__mro__[1](value)
    return value


def _confidence_of(value):
    return getattr(value, "confidence", 1.0)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _HTTPBackend:
    """Any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url, api_key, model):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, messages, temperature=0.7):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
        return data["choices"][0]["message"].get("content") or ""


class _CallableBackend:
    def __init__(self, fn):
        self.fn = fn

    def complete(self, messages, temperature=0.7):
        return self.fn(messages)


def _resolve_backend(spec, cfg):
    if spec is None:
        return _HTTPBackend(cfg["base_url"], cfg["api_key"], cfg["model"])
    if isinstance(spec, str):
        if spec.startswith("file://"):
            raise NotImplementedError(
                "embedded models are future work; point `model` at a server URL"
            )
        if spec.startswith("http://") or spec.startswith("https://"):
            return _HTTPBackend(spec, cfg["api_key"], cfg["model"])
        return _HTTPBackend(cfg["base_url"], cfg["api_key"], spec)
    if hasattr(spec, "complete"):
        return spec
    if callable(spec):
        return _CallableBackend(spec)
    raise TypeError(f"cannot use {spec!r} as an inference backend")


def _extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
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
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


@contextlib.contextmanager
def _require(threshold):
    token = _required.set(float(threshold))
    try:
        yield
    finally:
        _required.reset(token)


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

class Thing:
    """A probabilistic object. See module docstring and SPEC.md."""

    Approx = Approx
    LowConfidence = LowConfidence
    ContinuationLimit = ContinuationLimit

    _thing_step_budget = 16
    _thing_depth_budget = 4
    _thing_defaults: dict = {}

    def __init__(self, *parts, stateful=True, model=None, **state):
        self._thing_parts = [p if isinstance(p, str) else repr(p) for p in parts]
        self._thing_stateful = bool(stateful)
        self._thing_model_spec = model
        for name, value in state.items():
            setattr(self, name, value)

    # -- configuration ------------------------------------------------------

    @classmethod
    def defaults(cls, model=None, base_url=None, api_key=None):
        """Set class-wide backend defaults; subclasses inherit via the MRO."""
        cfg = dict(cls.__dict__.get("_thing_defaults", {}))
        if model is not None:
            cfg["model"] = model
        if base_url is not None:
            cfg["base_url"] = base_url
        if api_key is not None:
            cfg["api_key"] = api_key
        cls._thing_defaults = cfg

    @classmethod
    def require(cls, threshold):
        """Context manager: any resolution below `threshold` raises LowConfidence."""
        return _require(threshold)

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
        doc = inspect.getdoc(type(self)) if type(self) is not Thing else None
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

    def _thing_complete_json(self, messages, temperature):
        backend = self._thing_backend()
        last_error = None
        for _ in range(3):
            try:
                text = backend.complete(messages, temperature=temperature)
            except TypeError:
                text = backend.complete(messages)
            try:
                return _extract_json(text)
            except ValueError as error:
                last_error = error
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
                    '{"value": <json value>, "confidence": <0..1>}. Use natural JSON '
                    "types: numbers as numbers (a year, a count, a die face are "
                    "integers, never strings), booleans as booleans. If the story "
                    "treats the attribute as a random outcome (a roll, a draw, a "
                    "spin), sample one concrete outcome and set confidence to that "
                    "outcome's probability. If it is a fixed but unstated fact, "
                    "give your best single concrete guess with confidence matching "
                    "how constrained it is (e.g. a specific year from a stated "
                    'decade). Answer the string "unknown" only as a last resort '
                    "when even a guess would be meaningless — and then confidence "
                    "is how sure you are that it is unknowable: being certain it "
                    "is unknown means HIGH confidence (e.g. 0.9), never low. "
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
        data = self._thing_complete_json(messages, temperature=0.8)
        value = _make_approx(data.get("value"), data.get("confidence", 0.5))
        _check_required(value.confidence)
        if self._thing_stateful:
            object.__setattr__(self, name, value)
            self._thing_log(
                {
                    "event": "observe",
                    "name": name,
                    "value": _plain(value),
                    "confidence": value.confidence,
                }
            )
        return value

    def _thing_call(self, name, args, kwargs, depth=0):
        self._thing_ensure()
        if depth >= self._thing_depth_budget:
            raise ContinuationLimit(f"imagined call depth exceeded at `{name}`")
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
                    '{"action": "set", "name": "<attr>", "value": <json>}\n'
                    '{"action": "delete", "name": "<attr>"}\n'
                    '{"action": "call", "name": "<method>", "args": [<json>...]}\n'
                    '{"action": "define", "name": "<name>", "meaning": "<text>"}\n'
                    '{"action": "return", "value": <json>, "confidence": <0..1>}\n'
                    "Rules: prefer real methods when one fits — they execute actual "
                    "code. Never write or generate code. Keep plans short; mutate "
                    "state with `set` when the action changes the object. Always "
                    'finish with "return".'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"OBJECT STORY:\n{self._thing_story()}\n\n"
                    f"Execute the call: {call_repr}"
                ),
            },
        ]
        floor = 1.0
        for _ in range(self._thing_step_budget):
            step = self._thing_complete_json(messages, temperature=0.3)
            action = step.get("action")
            if action == "return":
                confidence = min(float(step.get("confidence", 0.5)), floor)
                result = _make_approx(step.get("value"), confidence)
                _check_required(result.confidence)
                if self._thing_stateful:
                    self._thing_log(
                        {
                            "event": "call",
                            "name": name,
                            "args": [_plain(a) for a in args],
                            "result": _plain(result),
                            "confidence": result.confidence,
                        }
                    )
                return result
            try:
                feedback, floor = self._thing_step(step, floor, depth)
            except ContinuationLimit:
                raise
            except Exception as error:
                feedback = f"error: {type(error).__name__}: {error}"
            messages.append({"role": "assistant", "content": json.dumps(step)})
            messages.append({"role": "user", "content": f"result: {feedback}"})
        raise ContinuationLimit(f"imagined plan for `{name}` exceeded its step budget")

    def _thing_step(self, step, floor, depth):
        action = step.get("action")
        target = step.get("name", "")
        if not isinstance(target, str) or target.startswith("_"):
            return f"refused: invalid name {target!r}", floor
        if action == "get":
            value = getattr(self, target)
            if isinstance(value, _Pending):
                value = value._resolve()
            floor = min(floor, _confidence_of(value))
            return json.dumps(_plain(value), default=repr), floor
        if action == "set":
            setattr(self, target, step.get("value"))
            return "ok", floor
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
        return _Pending(self, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)
        if self._thing_ensure().get("_thing_stateful", True):
            self._thing_log({"event": "set", "name": name, "value": _plain(value)})

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

    def freeze(self):
        """The object as a JSON-able document: description, state, story, flags."""
        self._thing_ensure()
        state = {}
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(value, (Approx, _Pending)):
                if isinstance(value, _Pending):
                    continue
                state[name] = {
                    "__approx__": _plain(value),
                    "confidence": value.confidence,
                }
            else:
                try:
                    json.dumps(value)
                    state[name] = value
                except TypeError:
                    state[name] = repr(value)
        model = self._thing_model_spec
        return {
            "class": type(self).__name__,
            "description": list(self._thing_parts),
            "state": state,
            "story": list(self._thing_journal),
            "stateful": self._thing_stateful,
            "model": model if isinstance(model, str) else None,
        }

    @classmethod
    def thaw(cls, blob):
        """Restore a frozen object. Thawing into a subclass reattaches its code."""
        obj = cls.__new__(cls)
        obj._thing_restore(blob)
        return obj

    def _thing_restore(self, blob):
        d = self._thing_ensure()
        d["_thing_parts"] = list(blob.get("description", []))
        d["_thing_journal"] = list(blob.get("story", []))
        d["_thing_stateful"] = bool(blob.get("stateful", True))
        d["_thing_model_spec"] = blob.get("model")
        for name, value in blob.get("state", {}).items():
            if isinstance(value, dict) and "__approx__" in value:
                object.__setattr__(
                    self, name, _make_approx(value["__approx__"], value["confidence"])
                )
            else:
                object.__setattr__(self, name, value)

    def __getstate__(self):
        return self.freeze()

    def __setstate__(self, blob):
        self._thing_restore(blob)

    def __repr__(self):
        self._thing_ensure()
        described = " | ".join(self._thing_parts)
        label = f" {described!r}" if described else ""
        return f"<{type(self).__name__}{label}>"
