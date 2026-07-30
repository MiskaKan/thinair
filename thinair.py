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
_DEFAULT_MODEL = os.environ.get("THINAIR_MODEL", "Qwen3.6-35B-A3B-oQ6-mtp")
_DEFAULT_MAX_TOKENS = int(os.environ.get("THINAIR_MAX_TOKENS", "32768"))
_DEFAULT_THINK_CHUNK = int(os.environ.get("THINAIR_THINK_CHUNK", "512"))

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
        """Thinking runs in blocks, starting at `think_chunk` tokens and
        doubling each checkpoint (long work needs room, and every
        checkpoint is a round trip). Concluding with an answer inside a
        block ends the call; running out of a block means "still thinking"
        — the reasoning is carried forward (with verbatim repetition
        pruned, so a stalled draft cannot re-seed its own loop) and the
        model is nudged toward answering. When the thinking allowance
        (`max_tokens` in total) is spent, one final answer block gets the
        full `max_tokens` budget so the answer itself is never squeezed.
        `think_chunk=0` restores single-shot completions."""
        debug = _debugging.get()
        st = _op_debug.get()
        # a plan's box is closed by its final `▸` step line, not by us
        conclude = not (st and st.get("plan"))
        if self.think_chunk <= 0:
            content, reasoning, meta = self._request(
                messages, temperature, self.max_tokens
            )
            self.last_meta = meta
            if debug:
                _debug_box(
                    None,
                    ([("<<< thoughts", reasoning)] if reasoning else [])
                    + [("<<< answer", content)],
                    _meta_line(meta),
                    close=conclude,
                )
            return content
        thoughts = []
        blocks_used = 0
        allowance = self.max_tokens  # total thinking budget across blocks
        cap = self.think_chunk
        demands_left = 2
        content = ""
        while True:
            # phase 1: thinking blocks with soft checkpoints, each up to
            # twice the size of the last; phase 2: allowance spent, the
            # answer is demanded — but still capped, so a model that only
            # keeps thinking can never capture the full budget
            demanding = allowance <= 0
            if demanding:
                if demands_left <= 0:
                    # it never stopped thinking; the parse layer sees the
                    # length finish and raises the clear budget error
                    # instead of retrying
                    return content
                demands_left -= 1
                budget = self.think_chunk
            else:
                budget = min(cap, max(self.think_chunk, allowance))
                cap = min(cap * 2, self.max_tokens)
            blocks_used += 1
            content, reasoning, meta = self._request(
                self._with_thoughts(messages, thoughts, final=demanding),
                temperature,
                budget,
            )
            meta["thinking_blocks"] = blocks_used
            self.last_meta = meta
            allowance -= int(meta.get("completion_tokens") or budget)
            # a length finish means the REQUEST hit its token cap, not that
            # the answer is incomplete: reasoning plus a whole answer can
            # land exactly on the cap, and asking again only confuses the
            # model about an answer it already delivered
            concluded = bool(content) and (
                meta.get("finish_reason") != "length" or _is_complete_json(content)
            )
            # a cut answer is only an answer if JSON is underway; prose in
            # the answer channel is thinking that escaped the reasoning
            # channel (server JSON mode off) and must not be granted the
            # full budget
            answer_underway = (
                bool(content)
                and not concluded
                and content.lstrip()[:1] in ("{", "[")
            )
            carried_reasoning = stalled_reasoning = None
            carried_draft = stalled_draft = None
            if not concluded:
                if reasoning:
                    carried_reasoning, stalled_reasoning = _prune_repetition(reasoning)
                if content and not answer_underway:
                    carried_draft, stalled_draft = _prune_repetition(content)
                if stalled_reasoning or stalled_draft:
                    # a looping model earns no bigger block: back to the
                    # smallest checkpoint, where the nudge can interrupt
                    cap = self.think_chunk
            if debug:
                k = "<<< " + (
                    f"step {st['round']} · " if st and st.get("plan") else ""
                ) + f"block {blocks_used}"
                pruned = " (degenerated into repetition; pruned before carrying)"
                sections = []
                if reasoning:
                    label = "thoughts"
                    if demanding:
                        label += " (answer demanded)"
                    if stalled_reasoning:
                        label += pruned
                    sections.append((label, reasoning))
                if concluded:
                    sections.append(("answer", content))
                elif answer_underway:
                    sections.append(
                        (
                            "answer, cut mid-way (completing at full "
                            "budget next)",
                            content,
                        )
                    )
                elif content:
                    label = "draft (arrived in the answer channel)"
                    if stalled_draft:
                        label += pruned
                    sections.append((label, content))
                if not sections:
                    sections.append(("thoughts", "(nothing returned)"))
                # the block identity appears once; further sections are
                # the same response's other channel
                sections[0] = (f"{k} · {sections[0][0]}", sections[0][1])
                _debug_box(
                    None,
                    sections,
                    _meta_line(meta),
                    # the final block closes the operation's tall box: an
                    # answer, or the last demanded attempt (unless a cut
                    # answer means the full-budget block still follows)
                    close=conclude
                    and (
                        concluded
                        or (demanding and demands_left <= 0 and not answer_underway)
                    ),
                )
            if concluded:
                return content  # concluded within the block: chose to answer
            if reasoning:
                if stalled_reasoning:
                    carried_reasoning = (
                        (carried_reasoning + "\n" if carried_reasoning else "")
                        + "(the reasoning began repeating verbatim and was "
                        "cut; do not resume the loop)"
                    )
                thoughts.append(carried_reasoning)
            if answer_underway:
                thoughts.append(content)
                break  # the answer is underway: full budget to finish it
            if content:
                label = (
                    "(a draft, cut at a checkpoint; do not continue this draft"
                    + (
                        ", and it began repeating verbatim, cut where the "
                        "loop started"
                        if stalled_draft
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
        # phase 3: a grant to finish the answer that already began — sized
        # from the cut draft, never the whole budget in one uncapped shot;
        # if even this is not enough, the length finish becomes the clear
        # budget error instead of a runaway
        grant = min(
            self.max_tokens,
            2 * int(meta.get("completion_tokens") or budget) + self.think_chunk,
        )
        content, reasoning, meta = self._request(
            self._with_thoughts(messages, thoughts, final=True),
            temperature,
            grant,
        )
        meta["thinking_blocks"] = blocks_used
        self.last_meta = meta
        if debug:
            prefix = "<<< " + (
                f"step {st['round']} · " if st and st.get("plan") else ""
            )
            fb_sections = ([("thoughts", reasoning)] if reasoning else []) + [
                ("answer, completed at full budget", content)
            ]
            fb_sections[0] = (f"{prefix}{fb_sections[0][0]}", fb_sections[0][1])
            _debug_box(None, fb_sections, _meta_line(meta), close=conclude)
        return content

    def _with_thoughts(self, messages, thoughts, final):
        if not thoughts and not final:
            return messages
        nudge = (
            "Reasoning budget spent. Emit the required JSON reply now, "
            "complete and from the beginning, nothing else."
            if final
            else "Paused at a scheduled checkpoint; nothing was lost, and "
            "your thoughts so far are above. Answer now with the required "
            "JSON reply if you can, otherwise keep reasoning."
        )
        st = _op_debug.get()
        if st and st.get("plan"):
            nudge += (
                " The reply is one small action object; never output data "
                "you already have, return_result returns it as-is. A value "
                "already drafted in your thoughts is finished: put it in the "
                "reply now instead of rehearsing it again."
            )
        extended = list(messages)
        if thoughts:
            extended.append(
                {
                    "role": "assistant",
                    "content": "(reasoning so far, paused at a scheduled checkpoint)\n"
                    + "\n".join(thoughts),
                }
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
    if finish:
        bits.append("paused at checkpoint" if finish == "length" else finish)
    if meta.get("json_mode") is False:
        bits.append("freeform (server JSON mode off)")
    return " · ".join(bits)


def _snip(text, limit=100):
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
            lines.append("Class attributes (certain): " + json.dumps(attrs, default=repr, ensure_ascii=False))
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
                "Value of this object (authoritative, confidence "
                f"{self.__dict__.get('confidence', 0.5)}): "
                + json.dumps(carried, default=repr, ensure_ascii=False)
            )
        state = {
            k: _plain(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and not (carried is not _UNSET and k == "confidence")
        }
        if state:
            lines.append("Current state (authoritative): " + json.dumps(state, default=repr, ensure_ascii=False))
        if self._thing_journal:
            lines.append("History, oldest first:")
            for i, event in enumerate(self._thing_journal, 1):
                lines.append(f"  {i}. {json.dumps(event, default=repr, ensure_ascii=False)}")
        return "\n".join(lines) or "An unspecified thing."

    def _thing_complete_json(self, messages, temperature, purpose="inference"):
        backend = self._thing_backend()
        last_error = None
        st = _op_debug.get()
        for _ in range(3):
            if _debugging.get() and st is not None:
                _debug_open_op(purpose, messages, st)
            token = _purpose.set(purpose)
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
                    "Infer the value of one attribute of a Python object from "
                    "the object's data below. Reply with exactly one JSON "
                    'object: {"value": <json>, "confidence": <0..1>}. '
                    "Confidence is the probability the value is correct. Use "
                    "natural JSON types (numbers as numbers, booleans as "
                    "booleans). A random outcome (a roll, a draw): pick one "
                    "result, confidence is its probability. An unstated fact: "
                    "your single best concrete guess, honestly low confidence "
                    "when wide open. null only if no such value can exist; "
                    'never placeholder strings like "unknown". Never '
                    "contradict the object's data. Example reply:\n"
                    '{"value": 8, "confidence": 0.98}'
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
            value, confidence = _answer_shape(
                self._thing_complete_json(
                    messages,
                    temperature=0.8,
                    purpose=f"{type(self).__name__}.{name} · read",
                )
            )
        _check_required(confidence)
        origin = " | ".join(self._thing_parts) or type(self).__name__
        child = self._thing_child(
            f"the attribute `{name}` of ({origin})", value, confidence
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
            value, confidence = _answer_shape(
                self._thing_complete_json(
                    messages, temperature=0.3, purpose=f"judge `{symbol}`"
                )
            )
        _check_required(confidence)
        origin = " | ".join(self._thing_parts) or type(self).__name__
        return self._thing_child(
            f"the judgment: ({origin}) {symbol} ({right[:120]})",
            bool(value),
            confidence,
        )

    def _thing_call(self, name, args, kwargs, stack=()):
        self._thing_ensure()
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
                    '{"action": "return_result", "confidence": <0..1>}\n'
                    "Facts: `call` runs the object's real, written methods. "
                    "Never write code. Bare written values are read-only; "
                    "record changes to them under new names. Values carrying "
                    "confidence are yours to change. `return_result` finishes "
                    "with the latest step's result exactly as it is (before "
                    "any step: the object's own value); use it instead of "
                    "retyping data you already have. When the job names no "
                    "output format, return data as data, exactly as produced; "
                    "never invent formatting or summaries. Never draft a long "
                    "value in your thinking: write it once, directly in the "
                    "reply. Every call ends with `return` or `return_result`.\n"
                    "Example turns for the job `describe_load()` on an object "
                    "with a real method `items`:\n"
                    '{"action": "call", "name": "items", "args": []}\n'
                    '-> result: ["anvil", "piano"]\n'
                    '{"action": "return", "value": "an anvil and a piano", "confidence": 0.9}\n'
                    "When the latest result already IS the answer, end with "
                    "return_result instead of retyping it:\n"
                    '{"action": "return_result", "confidence": 0.95}'
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
                        "\nThe object's value shown above is one step away: "
                        '{"action": "get", "name": "value"} reads it, and '
                        "return_result returns it as-is."
                        if self.__dict__.get("_thing_value", _UNSET) is not _UNSET
                        else ""
                    )
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
        # a value-carrying Thing starts with its own value as the latest
        # result, so `return_result` works before any step has run
        last_result = self.__dict__.get("_thing_value", _UNSET)
        purpose = f"{type(self).__name__}.{_snip(call_repr, 60)} · imagined"
        with _op_scope(plan=True):
            return self._thing_plan(
                name, args, messages, schema, floor, last_result, stack, purpose
            )

    def _thing_plan(
        self, name, args, messages, schema, floor, last_result, stack, purpose
    ):
        for index in range(1, self._thing_step_budget + 1):
            step = self._thing_complete_json(
                messages, temperature=0.3, purpose=purpose
            )
            if not isinstance(step, dict):
                # a bare JSON value in place of an action envelope is the
                # model returning its result directly; forgive it
                step = {"action": "return", "value": step}
            action = step.get("action")
            if action == "return_result" and last_result is _UNSET:
                if _debugging.get():
                    _debug_action(
                        index, "return_result refused: nothing produced yet"
                    )
                messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
                messages.append(
                    {
                        "role": "user",
                        "content": "result: refused: no earlier step has produced a result to return",
                    }
                )
                continue
            if action in ("return", "return_result"):
                value = last_result if action == "return_result" else step.get("value")
                if schema is not None:
                    ok, why = _matches(value, schema)
                    if not ok:
                        if _debugging.get():
                            _debug_action(
                                index, f"{action} rejected: {_snip(why)}"
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
                _check_required(confidence)
                if _debugging.get():
                    _debug_action(
                        index,
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
            token = _op_stack.set(_op_stack.get() + (f"{name}()",))
            try:
                feedback, floor, produced = self._thing_step(step, floor, stack)
            except ContinuationLimit:
                raise
            except Exception as error:
                feedback, produced = f"error: {type(error).__name__}: {error}", _UNSET
            finally:
                _op_stack.reset(token)
            if _debugging.get():
                _debug_action(
                    index, f"{_describe_step(step)} → {_snip(feedback)}"
                )
            if produced is not _UNSET:
                last_result = produced
            messages.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": f"result of your {_describe_step(step)}: {feedback}",
                }
            )
        raise ContinuationLimit(f"imagined plan for `{name}` exceeded its step budget")

    def _thing_result(self, call, args, value, confidence):
        """Wrap a call's return value as a child Thing, so results chain."""
        origin = " | ".join(self._thing_parts) or type(self).__name__
        arg_repr = ", ".join(repr(_plain(a)) for a in args)
        return self._thing_child(
            f"the result of {call}({arg_repr}) on ({origin})", value, confidence
        )

    def _thing_child(self, described, value, confidence):
        """A Thing born from inference: carries its value and `.confidence`.
        Always a plain Thing, never the parent's class — a result is a new
        object, not the object it came from. Recast with `@ Class` when the
        class genuinely applies."""
        child = Thing.__new__(Thing)
        d = child._thing_ensure()
        d["_thing_parts"] = [described]
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

    def _thing_step(self, step, floor, stack):
        """One plan step -> (feedback, floor, produced result or _UNSET)."""
        action = step.get("action")
        target = step.get("name", "")
        if not isinstance(target, str) or target.startswith("_"):
            return f"refused: invalid name {target!r}", floor, _UNSET
        if action == "call" and target in stack:
            return (
                f"refused: `{target}` is already executing (call stack: "
                f"{' -> '.join(stack)}); produce the result yourself and "
                "finish with a return",
                floor,
                _UNSET,
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
                    _UNSET,
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
                        carried,
                    )
            value = getattr(self, target)
            if isinstance(value, _Pending):
                value = value._resolve()
            floor = min(floor, _confidence_of(value))
            plain = _plain(value)
            return json.dumps(plain, default=repr, ensure_ascii=False), floor, plain
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
            return "ok", min(floor, confidence), _UNSET
        if action == "delete":
            delattr(self, target)
            return "ok", floor, _UNSET
        if action == "call":
            call_args = step.get("args") or []
            attr = getattr(self, target)
            if isinstance(attr, _Pending):
                value = self._thing_call(target, tuple(call_args), {}, stack)
            elif callable(attr):
                value = attr(*call_args)
            else:
                return f"error: `{target}` is not callable", floor, _UNSET
            floor = min(floor, _confidence_of(value))
            plain = _plain(value)
            return json.dumps(plain, default=repr, ensure_ascii=False), floor, plain
        if action == "define":
            self.define(target, str(step.get("meaning", "")))
            return "ok", floor, _UNSET
        return (
            f"refused: unknown action {action!r}; to return data, reply "
            '{"action": "return", "value": <the data>}',
            floor,
            _UNSET,
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
            value, confidence = _answer_shape(
                self._thing_complete_json(
                    messages,
                    temperature=0.3,
                    purpose=f"{type(self).__name__} @ {key} · collapse"
                    if key
                    else f"{type(self).__name__} · collapse",
                )
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
