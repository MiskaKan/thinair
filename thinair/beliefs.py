"""Beliefs: one contract for everything (SPEC.md §3).

A **Belief** is a function ``b(e, a) -> (v, p)``.  The model is one, every
validator is one, the human is one, code is one.  ``None`` means "no opinion
here", which subsumes any separate scoping mechanism.

Subclassing is ergonomics, not a gate: the runtime never checks ``isinstance``.
Any callable with an ``id``, a ``necessary`` attribute and this signature
participates.

``engine`` and ``models`` are imported *only* by :class:`ModelBelief`, lazily,
so that no module-level import edge from the rest of the package reaches the
network layer (invariant 7).
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
import textwrap
from typing import Any, Callable, Iterable

from .ledger import Opinion

__all__ = [
    "Belief", "Judgment", "Discriminative", "Scoped", "HumanBelief", "human",
    "CodeBelief", "code_identity", "ModelBelief", "model", "MemoBelief",
    "registry", "register", "lookup", "value_at", "opinion_at", "reason_of",
    "set_config", "config", "config_scope", "generative_members",
]


# --------------------------------------------------------------------------
# what a belief hands back
# --------------------------------------------------------------------------

class Judgment(tuple):
    """A ``(value, p)`` pair that can carry a free-form ``meta`` (e.g. a reason).

    Beliefs may return a plain ``(v, p)`` tuple -- the runtime normalizes.  The
    contract is a 2-tuple; the optional third slot is provenance, and the
    ledger record shape is unchanged (invariant 1).
    """

    def __new__(cls, value: Any, p: float, meta: dict | None = None) -> "Judgment":
        self = super().__new__(cls, (value, float(p)))
        self.meta = dict(meta or {})
        return self

    @property
    def value(self) -> Any:
        return self[0]

    @property
    def p(self) -> float:
        return self[1]

    def __pos__(self) -> Any:
        return self[0]

    def __invert__(self) -> float:
        return self[1]

    def with_reason(self, reason: str) -> "Judgment":
        return Judgment(self[0], self[1], dict(self.meta, reason=reason))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        extra = f" {self.meta}" if self.meta else ""
        return f"({self[0]!r}, {self[1]:.3g}){extra}"


def as_judgment(got: Any) -> Judgment | None:
    """Normalize whatever a belief returned into a :class:`Judgment` or ``None``."""
    if got is None:
        return None
    if isinstance(got, Judgment):
        return got
    if isinstance(got, Opinion):
        return Judgment(got.value, got.p, got.meta)
    # a list is as good as a tuple here: persistence round-trips one into the
    # other, and a belief that came back through JSON is still a belief.
    if isinstance(got, (tuple, list)) and len(got) == 2:
        return Judgment(got[0], got[1])
    if isinstance(got, (tuple, list)) and len(got) == 3:
        return Judgment(got[0], got[1], got[2])
    raise TypeError(
        f"a belief must return None or (value, p[, meta]); got {got!r}")


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, "Belief"] = {}


def register(belief: "Belief") -> "Belief":
    """Assert id uniqueness: one id names one fixed configuration (invariant 6).

    The registry keeps the first instance seen under each id as that
    configuration's exemplar; it does **not** intern.  Two ``Verbatim("src")``
    objects are the same belief and share an id, yet one of them may have been
    attached ``necessary=False`` -- a veto right is a policy of the
    attachment site, not part of what the check *is*, so it must not fork the
    identity and must not be flattened onto the other instance either.
    """
    existing = _REGISTRY.get(belief.id)
    if existing is not None and type(existing) is not type(belief):
        raise ValueError(
            f"belief id {belief.id!r} already names a {type(existing).__name__}; "
            f"a belief id must name a fixed configuration (invariant 6)")
    if existing is not None:
        return existing
    _REGISTRY[belief.id] = belief
    return belief


def registry() -> dict[str, "Belief"]:
    return dict(_REGISTRY)


def lookup(belief_id: str) -> "Belief | None":
    return _REGISTRY.get(belief_id)


# --------------------------------------------------------------------------
# id derivation
# --------------------------------------------------------------------------

def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        return repr(round(value, 12)).rstrip("0").rstrip(".") or "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_render(v)}" for k, v in sorted(
            value.items(), key=lambda kv: str(kv[0]))) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_render(v) for v in value)) + "}"
    if callable(value):
        return getattr(value, "__qualname__", None) or getattr(value, "__name__", repr(value))
    return repr(value)


def _derive_id(belief: "Belief", args: tuple, kwargs: dict) -> str:
    name = type(belief).__name__
    name = name[0].lower() + name[1:] if name else name
    parts = [_render(a) for a in args]
    parts += [f"{k}={_render(v)}" for k, v in sorted(kwargs.items())]
    inner = ",".join(parts)
    if len(inner) > 100:
        digest = hashlib.sha1(inner.encode("utf-8")).hexdigest()[:12]
        inner = f"#{digest}"
    return f"{name}[{inner}]" if inner else name


# --------------------------------------------------------------------------
# the base class
# --------------------------------------------------------------------------

_OPTIONS = ("necessary", "veto_line", "id")


class Belief:
    """Base class: subclass and override ``__call__``.  Defaults do the rest."""

    #: veto right: a failing judgment kills the candidate (invariant 5)
    necessary: bool = False
    #: the line below which a ``necessary`` belief vetoes
    veto_line: float = 0.5
    #: can this belief speak into an *empty* cell?  Routing must know
    #: before it spends a call, so it is declared; a test holds the flag
    #: to what an empty-cell probe observes.
    proposes: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        options = {k: kwargs.pop(k) for k in list(kwargs) if k in _OPTIONS}
        self._args = tuple(args)
        self._kwargs = dict(kwargs)
        if "necessary" in options:
            self.necessary = bool(options["necessary"])
        if "veto_line" in options:
            line = float(options["veto_line"])
            if not 0.0 <= line <= 1.0:
                raise ValueError("veto_line must be in [0.0, 1.0]")
            self.veto_line = line
        self.id = options.get("id") or _derive_id(self, self._args, self._kwargs)
        register(self)

    def __call__(self, e: Any, attr: str) -> tuple | None:
        return None  # a belief with nothing to say -- the legal minimum

    # -- cosmetics ---------------------------------------------------------
    @property
    def short(self) -> str:
        """A human-sized label for ``__source__`` and debug boxes."""
        return self.id

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mark = "!" if self.necessary else ""
        return f"<{self.id}{mark}>"


# --------------------------------------------------------------------------
# reading a sealed snapshot
# --------------------------------------------------------------------------

def opinion_at(e: Any, attr: str) -> Opinion | None:
    """The snapshot's resolving opinion at ``attr``, or ``None`` if absent.

    Inert: reading a sealed snapshot never consults, never spends.
    """
    try:
        got = getattr(e, attr)
    except AttributeError:
        return None
    return got if isinstance(got, Opinion) else None


def value_at(e: Any, attr: str, default: Any = None) -> Any:
    opinion = opinion_at(e, attr)
    return default if opinion is None else opinion.value


def reason_of(got: Any) -> str | None:
    meta = getattr(got, "meta", None)
    return (meta or {}).get("reason") if isinstance(meta, dict) else None


# --------------------------------------------------------------------------
# memoization + cycle safety: the snapshot's two invisible guarantees
# --------------------------------------------------------------------------

class MemoBelief:
    """A panel entry as seen from inside a round.

    Calling the same belief at the same cell evaluates once and serves the memo
    thereafter -- semantically undetectable, since beliefs are pure functions of
    ``(e, a)``; only the cost differs.  A belief called while already on the
    call stack answers ``None``, so cycles resolve instead of recursing.
    """

    __slots__ = ("belief", "id", "necessary", "veto_line", "proposes",
                 "_memo", "_active")

    def __init__(self, belief: Any, memo: dict, active: set) -> None:
        self.belief = belief
        # membership is structural, so an id-less callable is legal; it
        # just memoizes under its own identity instead of a durable name.
        self.id = getattr(belief, "id", None) or f"anon:{id(belief):x}"
        self.necessary = bool(getattr(belief, "necessary", False))
        self.veto_line = float(getattr(belief, "veto_line", 0.5))
        self.proposes = bool(getattr(belief, "proposes", False))
        self._memo = memo
        self._active = active

    def __call__(self, e: Any, attr: str) -> Judgment | None:
        key = (self.id, attr)
        if key in self._memo:
            return self._memo[key]
        if key in self._active:
            return None                      # re-entry yields None
        self._active.add(key)
        try:
            got = as_judgment(self.belief(e, attr))
        finally:
            self._active.discard(key)
        self._memo[key] = got
        return got

    @property
    def short(self) -> str:
        return getattr(self.belief, "short", self.id)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<memo {self.id}>"


def generative_members(beliefs: Iterable[Any]) -> list[Any]:
    """The generative members of a panel, in preference order."""
    return [b for b in beliefs if getattr(b, "proposes", False)]


# --------------------------------------------------------------------------
# discriminative beliefs: obtain the candidate by calling the proposer
# --------------------------------------------------------------------------

class Discriminative(Belief):
    """Judges the proposer's candidate.  Subclasses implement :meth:`judge`.

    Pulling the candidate is a plain call on ``e.__beliefs__[0]`` -- there is no
    channel and no ordering the runtime imposes.
    """

    #: when set, this belief answers only at this attribute
    scope: str | None = None

    def scoped_to(self, attr: str) -> bool:
        return self.scope is None or attr == self.scope

    def candidate(self, e: Any, attr: str) -> Judgment | None:
        panel = getattr(e, "__beliefs__", ())
        if not panel:
            return None
        return as_judgment(panel[0](e, attr))    # ask the proposer -- a plain call

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        """Return ``None`` (out of scope), a probability, or ``(p, reason)``."""
        raise NotImplementedError

    def __call__(self, e: Any, attr: str) -> tuple | None:
        if not self.scoped_to(attr):
            return None
        got = self.candidate(e, attr)
        if got is None:
            return None                          # nothing proposed -- no opinion
        verdict = self.judge(got[0], e, attr)
        if verdict is None:
            return None
        if isinstance(verdict, tuple):
            p, reason = float(verdict[0]), verdict[1]
        else:
            p, reason = float(verdict), None
        meta = {"judged": self.id}
        if reason:
            meta["reason"] = reason
        return Judgment(got[0], p, meta)


class Scoped(Belief):
    """Restrict a belief to one attribute -- what contract sugar appends."""

    def __init__(self, attr: str, inner: Any, **options: Any) -> None:
        options.setdefault("necessary", getattr(inner, "necessary", False))
        options.setdefault("veto_line", getattr(inner, "veto_line", 0.5))
        options.setdefault("id", f"{inner.id}@{attr}")
        self.attr = attr
        self.inner = inner
        super().__init__(attr, inner.id, **options)

    @property
    def proposes(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.inner, "proposes", False))

    def __call__(self, e: Any, attr: str) -> tuple | None:
        if attr != self.attr:
            return None
        return self.inner(e, attr)

    @property
    def short(self) -> str:
        return getattr(self.inner, "short", self.inner.id)


# --------------------------------------------------------------------------
# the human is a belief
# --------------------------------------------------------------------------

class HumanBelief(Belief):
    """``human("jane")`` -> id ``human:jane``.

    On pipeline consultation the default human is silent; assignment is a human
    consultation with a predefined answer.  ``interactive=True``
    genuinely *asks* -- and blocks, which is why it is opt-in and must never sit
    in a default list where a background read would become a modal dialog.
    """

    def __init__(self, name: str, interactive: bool = False,
                 stream: Any = None, **options: Any) -> None:
        self.name = name
        self.interactive = bool(interactive)
        self.stream = stream
        options.setdefault("id", f"human:{name}")
        super().__init__(name, **options)

    @property
    def short(self) -> str:
        return f"human:{self.name}"

    def __call__(self, e: Any, attr: str) -> tuple | None:
        if not self.interactive:
            return None
        return self.ask(e, attr)

    # -- prompted input corroborates; only assignment and freeze freeze -----
    def ask(self, e: Any, attr: str) -> tuple | None:
        out = self.stream or sys.stdout
        entity = getattr(e, "__entity__", "?")
        try:
            out.write(f"\n[thinair] {self.name}, what is {entity}.{attr}? "
                      f"(empty = no opinion)\n> ")
            out.flush()
            raw = (input() or "").strip()
            if not raw:
                return None
            out.write("[thinair] how sure, 0..1? (empty = 1.0)\n> ")
            out.flush()
            raw_p = (input() or "").strip()
        except (EOFError, OSError):
            return None
        value = _coerce_input(raw)
        try:
            p = float(raw_p) if raw_p else 1.0
        except ValueError:
            p = 1.0
        p = min(1.0, max(0.0, p))
        return Judgment(value, p, {"prompted": True})


def _coerce_input(raw: str) -> Any:
    import json as _json
    try:
        return _json.loads(raw)
    except Exception:
        return raw


def human(name: str, interactive: bool = False, **options: Any) -> HumanBelief:
    return HumanBelief(name, interactive=interactive, **options)


# --------------------------------------------------------------------------
# code is a belief
# --------------------------------------------------------------------------

def code_identity(func: Callable) -> str:
    """``code:<qualname>@<source-hash>`` -- edit the body, mint a new belief."""
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", "?"))
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):  # pragma: no cover - builtins, REPL
        source = repr(func)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"code:{qualname}@{digest}"


class CodeBelief(Belief):
    """A body of code as a generative belief.  Its results freeze."""

    proposes = True

    def __init__(self, func: Callable, **options: Any) -> None:
        self.func = func
        options.setdefault("id", code_identity(func))
        super().__init__(getattr(func, "__qualname__", "?"), **options)

    @property
    def short(self) -> str:
        return self.id

    def __call__(self, e: Any, attr: str) -> tuple | None:
        arguments = getattr(e, "__call_arguments__", None)
        if arguments is None:
            return None
        args, kwargs = arguments
        value = self.func(*args, **kwargs)
        return Judgment(value, 1.0, {"code": self.id})


# --------------------------------------------------------------------------
# configuration cascade -- per class, MRO inheritance, env fallbacks
# --------------------------------------------------------------------------

_CONFIG: dict[Any, dict] = {}

_ENV_FALLBACKS = {
    "model": "THINAIR_MODEL",
    "base_url": "THINAIR_BASE_URL",
    "api_key": "THINAIR_API_KEY",
    "max_tokens": "THINAIR_MAX_TOKENS",
}


def set_config(owner: Any, **values: Any) -> dict:
    """Set config for one owner (a class, or ``None`` for process-wide)."""
    slot = _CONFIG.setdefault(owner, {})
    previous = dict(slot)
    slot.update({k: v for k, v in values.items() if v is not None})
    return previous


def restore_config(owner: Any, previous: dict) -> None:
    _CONFIG[owner] = dict(previous)


def config(key: str, owner: Any = None, default: Any = None) -> Any:
    """Resolve ``key``: owner's MRO, then process-wide, then the environment."""
    if owner is not None:
        for klass in getattr(owner, "__mro__", (owner,)):
            slot = _CONFIG.get(klass)
            if slot and key in slot:
                return slot[key]
    slot = _CONFIG.get(None)
    if slot and key in slot:
        return slot[key]
    env = _ENV_FALLBACKS.get(key)
    if env and os.environ.get(env):
        raw = os.environ[env]
        return int(raw) if key == "max_tokens" else raw
    return default


class config_scope:
    """Context manager form of :func:`set_config`; also effective on call."""

    def __init__(self, owner: Any = None, **values: Any) -> None:
        self.owner = owner
        self._previous = set_config(owner, **values)

    def __enter__(self) -> "config_scope":
        return self

    def __exit__(self, *exc: Any) -> None:
        restore_config(self.owner, self._previous)


def _exposure(messages: Any) -> str:
    """A fingerprint of the rendered context a model reading was shown.

    Dissimilarity between two readings is mechanism AND exposure; the belief
    id carries the mechanism, and without this stamp the exposure half is
    unrecoverable from the ledger after the fact.  Same rendered context ->
    same fingerprint, so "did these two agreeing readings see the same
    snapshot?" is answerable forever, warm or cold.
    """
    blob = _json_dumps(messages)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# the generative belief -- the only door to the network
# --------------------------------------------------------------------------

class ModelBelief(Belief):
    """An LLM as one ordinary list member, with a durable id (invariant 6).

    Identity is the full configuration tuple: model id, prompt template version,
    temperature, and the ``ModelDef`` version.  Change any component -> new
    belief.  Nothing here is special beyond position in ``__beliefs__``.
    """

    proposes = True

    def __init__(self, name: str | None = None, *, think: bool = False,
                 temperature: float | None = None, engine: Any = None,
                 max_tokens: int | None = None, template: str | None = None,
                 owner: Any = None, models_path: Any = None,
                 **options: Any) -> None:
        from .models import resolve                     # lazy: invariant 7
        from .engine import prompts

        self.model_name = (name or config("model")
                           or os.environ.get("THINAIR_MODEL") or "default")
        self.models_path = models_path or config("models_path", owner)
        self.definition = resolve(self.model_name, self.models_path)
        self.think = bool(think)
        self.temperature = float(
            temperature if temperature is not None
            else self.definition.defaults.get("temperature", 0.2))
        self.template = template or prompts.ATTRIBUTE_TEMPLATE
        self.episode_template = prompts.EPISODE_TEMPLATE
        self.max_tokens = max_tokens
        self.owner = owner
        self._engine = engine
        thought = "/think" if self.think else ""
        options.setdefault(
            "id",
            f"model:{self.model_name}@T{self.temperature:g}/"
            f"{self.template}/{self.definition.identity()}{thought}")
        super().__init__(self.model_name, **options)

    @property
    def short(self) -> str:
        return f"model/{self.template}"

    # -- engine acquisition ------------------------------------------------
    def engine(self, owner: Any = None) -> Any:
        if self._engine is not None:
            return self._engine
        injected = config("engine", owner or self.owner)
        if injected is not None:
            return injected
        from .engine.openai_compat import engine_for   # lazy: invariant 7
        return engine_for(
            self.model_name, self.definition,
            base_url=config("base_url", owner or self.owner),
            api_key=config("api_key", owner or self.owner),
            max_tokens=self.max_tokens or config("max_tokens", owner or self.owner))

    # -- the one question this belief answers ------------------------------
    def __call__(self, e: Any, attr: str) -> tuple | None:
        episode = getattr(e, "__episode__", None)
        if episode is not None and attr == episode.attr:
            return self.run_episode(e, episode)
        return self.propose(e, attr)

    def propose(self, e: Any, attr: str) -> tuple | None:
        from .engine import complete_json, ParseFailure
        from .engine import prompts

        contract = (getattr(e, "__contracts__", {}) or {}).get(attr)
        objections = list(getattr(e, "__objections__", ()) or ())
        messages = prompts.attribute_messages(
            e, attr, contract=contract, objections=objections,
            dialect=self.definition.prompt_dialect)
        schema = prompts.response_schema(getattr(contract, "schema", None))
        exposure = _exposure(messages)
        try:
            payload, meta = complete_json(
                self.engine(getattr(e, "__owner__", None)), messages, schema=schema,
                temperature=self.temperature, max_tokens=self.max_tokens,
                think=self.think)
        except ParseFailure as exc:
            return Judgment(None, 0.0, {"reason": f"unparseable completion: {exc}",
                                        "model": self.model_name,
                                        "exposure": exposure})
        if not isinstance(payload, dict) or "value" not in payload:
            return Judgment(None, 0.0, {"reason": "completion carried no 'value'",
                                        "model": self.model_name,
                                        "exposure": exposure})
        p = payload.get("p", 0.5)
        try:
            p = min(1.0, max(0.0, float(p)))
        except (TypeError, ValueError):
            p = 0.5
        provenance = dict(meta)
        provenance.update({"model": self.model_name, "think": self.think,
                           "template": self.template, "exposure": exposure})
        if payload.get("why"):
            provenance["why"] = payload["why"]
        return Judgment(payload["value"], p, provenance)

    # -- episodes are this belief's private implementation -----------
    def run_episode(self, e: Any, episode: Any) -> tuple | None:
        from .engine import complete_json, ParseFailure
        from .engine import prompts

        convo = prompts.episode_messages(
            e, episode.expression, returns=episode.returns,
            action_budget=episode.action_budget,
            dialect=self.definition.prompt_dialect)
        schema = prompts.episode_schema(episode.returns)
        engine = self.engine(getattr(e, "__owner__", None))
        exposure = _exposure(convo)          # the state the episode set out from
        actions_left = episode.action_budget
        transcript: list[dict] = []
        while True:
            if actions_left <= 0:
                return Judgment(None, 0.0, {
                    "reason": "action budget exhausted before returning",
                    "no_return": True, "actions": transcript})
            try:
                step, meta = complete_json(
                    engine, convo, schema=schema, temperature=self.temperature,
                    max_tokens=self.max_tokens, think=self.think)
            except ParseFailure as exc:
                return Judgment(None, 0.0, {"reason": f"unparseable action: {exc}",
                                            "no_return": True,
                                            "actions": transcript})
            actions_left -= 1
            if not isinstance(step, dict):
                convo.append({"role": "user", "content":
                              "Each reply must be one JSON action object."})
                continue
            action = step.get("action")
            transcript.append(step)
            convo.append({"role": "assistant", "content": _json_dumps(step)})
            _trace_action(e, step)
            if action == "get":
                kind, detail = episode.get(step.get("attr"))
                convo.append(prompts.episode_observation(kind, detail, actions_left))
                continue
            if action == "call":
                kind, detail = episode.call(step.get("method"),
                                            step.get("args") or [],
                                            step.get("kwargs") or {})
                convo.append(prompts.episode_observation(kind, detail, actions_left))
                continue
            if action == "return":
                if "frozen" in step or "frozen" in (step.get("changes") or {}):
                    # invariant 4: the grammar has no freeze action, and a
                    # changeset can never carry `frozen`.
                    convo.append({"role": "user", "content":
                                  "Rejected: nothing you return may carry 'frozen'. "
                                  "You state beliefs; only code freezes."})
                    continue
                p = step.get("p", 0.5)
                try:
                    p = min(1.0, max(0.0, float(p)))
                except (TypeError, ValueError):
                    p = 0.5
                changes = step.get("changes") or {}
                if not isinstance(changes, dict):
                    convo.append({"role": "user", "content":
                                  "'changes' must be an object of attribute writes."})
                    continue
                provenance = dict(meta)
                provenance.update({"model": self.model_name, "changes": changes,
                                   "actions": transcript,
                                   "template": self.episode_template,
                                   "exposure": exposure})
                return Judgment(step.get("value"), p, provenance)
            convo.append({"role": "user", "content":
                          f"Unknown action {action!r}. Use get, call or return."})


def _trace_action(e: Any, step: Any) -> None:
    from .debug import trace

    trace("action", (getattr(e, "__entity__", "?"), "result"), (step, None))


def _json_dumps(value: Any) -> str:
    import json as _json
    try:
        return _json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(value)


def model(name: str | None = None, **options: Any) -> ModelBelief:
    """Construct a generative belief.  Zero network at construction time."""
    return ModelBelief(name, **options)
