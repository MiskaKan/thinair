"""The object surface: interception, resolution, episodes (SPEC.md §4-§9).

One public class with **zero instance methods**.  On a Thing every plain
attribute name belongs to the user's domain -- that is the premise of
interception -- so the framework lives entirely in dunders and in four
classmethod context managers (``require``, ``policy``, ``debug``,
``defaults``).

**The Thing has all of it.**  A Thing with a positional template is a
*declaration* -- ``priority = Thing(str, enum=[...])`` in a class body
declares the attribute; a Thing assigned to an attribute is a *reference*
(the record keeps its entity id); the panel changes through the operators
(``t += belief``, ``t -= belief`` in place; ``t + belief``,
``t - belief`` for a fresh handle on the same entity); and pinning is a
property of the *belief* (``frozen=True`` -- code's results freeze,
assignment freezes), never a verb.  ``Thing.__default__`` names the
generative belief a panel falls back to when it declares none.  The
module keeps two verbs that are not moves the surface makes:
``snapshot`` (the sealed, inert view a belief is shown) and
``corroborate`` (spend calls on the beliefs a read never asked).

This module imports ``beliefs``, ``ledger`` and ``policy``.  It must never
import ``engine`` or ``models`` (invariant 7) -- the model reaches the
runtime only as an ordinary member of ``__beliefs__``.
"""

from __future__ import annotations

import contextlib
import inspect
import uuid
import weakref
from typing import Any

from .beliefs import (Belief, HumanBelief, MemoBelief, Scoped,
                      generative_members, human)
from .beliefs import config as _config
from .beliefs import config_scope as _config_scope
from .ledger import Opinion, default_ledger, values_equal
from .policy import (
    DEFAULT_POLICY,
    Attempt,
    Disagreement,
    LowConfidence,
    ResolutionPolicy,
    Route,
    Unresolvable,
)

__all__ = ["Thing", "Cell", "Snapshot", "ThingMeta", "corroborate",
           "snapshot", "boundary_snapshot", "LowConfidence", "Unresolvable",
           "Disagreement", "resident_human"]

#: how the framework spells itself on a Thing.  Anything matching this is the
#: framework's; everything else is the user's domain.
def _is_framework_name(name: str) -> bool:
    return name.startswith("_")


# --------------------------------------------------------------------------
# ambient state: the four context managers
# --------------------------------------------------------------------------

_policies: list[ResolutionPolicy] = []
_floors: list[float] = []
_ledgers: list[Any] = []


def active_policy() -> ResolutionPolicy:
    return _policies[-1] if _policies else DEFAULT_POLICY


def active_floor() -> float | None:
    return _floors[-1] if _floors else None


# --------------------------------------------------------------------------
# contracts -- sugar that appends scoped beliefs to __beliefs__
# --------------------------------------------------------------------------

class Contract:
    """A declared attribute: a template plus the checks it implies.

    Contracts are **sugar and nothing else**.  Everything a contract does, it
    does by appending scoped beliefs to the one attachment point, and
    the ``SchemaBelief`` it builds is the same object that constrains the engine's
    structured output and performs the post-hoc check -- so the two cannot
    drift.
    """

    def __init__(self, template=Any, *, extracted_from=None, range=None,
                 enum=None, length=None, format=None, checksum=None,
                 sums_to=None, unique=False, elaborates=False,
                 necessary=True, describe=None, beliefs=(), doc=None,
                 p=None, deviation=None, eager=False, public=False):
        self.template = template
        #: whether this cell crosses an entity boundary (SPEC.md §4).
        #: Default private, fail closed: only a declared ``public=True``
        #: attribute is part of what another entity perceives.
        self.public = bool(public)
        self.extracted_from = extracted_from
        self.range = range
        self.enum = enum
        self.length = length
        self.format = format
        self.checksum = checksum
        self.sums_to = sums_to
        self.unique = unique
        self.elaborates = elaborates
        self.necessary = necessary
        self.doc = doc
        if p is None:
            self.p_bounds = None
        elif isinstance(p, (int, float)):
            self.p_bounds = (float(p), 1.0)
        else:
            self.p_bounds = (float(p[0]), float(p[1]))
        if self.p_bounds is not None and not (
                0.0 <= self.p_bounds[0] <= self.p_bounds[1] <= 1.0):
            raise ValueError("contract p bounds must satisfy 0 <= lo <= hi <= 1")
        self.deviation = float(deviation) if deviation is not None else None
        self.eager = bool(eager)
        self._describe = describe
        self._extra = list(beliefs)
        self.attr = None
        self.schema = None

    @property
    def expectations(self) -> dict | None:
        """The declared Layer-2 expectations: ``p`` bounds and max deviation.

        Deliberately *not* part of :meth:`describe`: a belief told "you must
        be at least 0.6 sure" is a belief under pressure to inflate, and an
        inflated p is worthless.  Expectations are stamped into the record at
        settlement and judged there -- they mark, they never gate.
        """
        out = {}
        if self.p_bounds is not None:
            out["p"] = list(self.p_bounds)
        if self.deviation is not None:
            out["deviation"] = self.deviation
        return out or None

    # -- what the prompt builder shows the model --------------------------
    def describe(self) -> str:
        if self._describe:
            return self._describe
        from .validators import render_template

        parts = [render_template(self.template)]
        if self.extracted_from:
            parts.append(f"extracted from {self.extracted_from}")
        if self.range:
            parts.append(f"between {self.range[0]} and {self.range[1]}")
        if self.enum:
            parts.append("one of " + ", ".join(repr(m) for m in self.enum))
        if self.format:
            parts.append(f"a valid {self.format}")
        if self.sums_to:
            parts.append(f"summing to {self.sums_to}")
        # beliefs handed in as a list are first-class declarations: a law the
        # panel will enforce is a law the model is told about, exactly as the
        # named options are -- the two spellings must not diverge
        for extra in self._extra:
            told = getattr(extra, "describe", None)
            if told is not None:
                parts.append(told())
        if self.doc:
            parts.append(self.doc)
        return "; ".join(parts)

    # -- the beliefs this contract attaches -------------------------------
    def bind(self, attr: str) -> "Contract":
        self.attr = attr
        return self

    def beliefs(self, attr: str) -> list:
        """The scoped beliefs this contract appends for ``attr``.

        Each is wrapped in :class:`~thinair.beliefs.Scoped` so it returns
        ``None`` off its own attribute -- the contract narrows *when* a belief
        speaks, never what it is.
        """
        from . import validators as V

        out = []
        if self.template is not Any:
            self.schema = V.SchemaBelief(self.template, necessary=self.necessary)
            out.append(self.schema)
        if self.extracted_from:
            out.append(self._grounding(V))
        if self.range is not None:
            out.append(V.RangeBelief(self.range[0], self.range[1],
                               necessary=self.necessary))
        if self.enum is not None:
            out.append(V.EnumBelief(self.enum, necessary=self.necessary))
        if self.length is not None:
            out.append(V.LengthBelief(self.length[0], self.length[1],
                                necessary=self.necessary))
        if self.format is not None:
            out.append(V.FormatBelief(self.format, necessary=self.necessary))
        if self.checksum is not None:
            out.append(V.ChecksumBelief(self.checksum, necessary=self.necessary))
        if self.unique:
            out.append(V.UniqueBelief(necessary=self.necessary))
        if self.elaborates:
            out.append(V.NonEchoBelief(necessary=True))
        out.extend(self._extra)
        scoped = [Scoped(attr, b) for b in out]
        if self.sums_to:
            # A relation is not scoped to one attribute: it relates several,
            # and must be able to speak at any of them.
            scoped.append(self._sums_to(V, attr))
        return scoped

    def _grounding(self, V):
        """The grounding check that fits the declared shape.

        A verbatim check is right for a string and wrong for everything
        else -- ``1249.5`` is never a substring of "Total 1 249,50 EUR", so
        a literal reading would veto every correct numeric extraction.
        Numbers and containers get ``TokenSubsetBelief`` instead, ``necessary``
        because a missing number is the classic hallucination tell.
        """
        if self.template is str:
            return V.VerbatimBelief(self.extracted_from, necessary=self.necessary)
        return V.TokenSubsetBelief(self.extracted_from, necessary=self.necessary)

    def _sums_to(self, V, attr):
        field = _single_numeric_field(self.template)
        if field is None:
            return V.SumsToBelief([attr], self.sums_to, necessary=self.necessary)
        return V.ItemsSumToBelief(attr, field, self.sums_to, necessary=self.necessary)

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        return f"<contract {self.attr or '?'}: {self.describe()}>"


def _single_numeric_field(template) -> str | None:
    """The one numeric field of a ``[{...}]`` template, if there is exactly one."""
    if isinstance(template, list) and len(template) == 1:
        template = template[0]
    if not isinstance(template, dict):
        return None
    numeric = [k for k, v in template.items() if v in (int, float)]
    return numeric[0] if len(numeric) == 1 else None


# --------------------------------------------------------------------------
# the metaclass
# --------------------------------------------------------------------------

class ThingMeta(type):
    """Collects contracts, assembles ``__beliefs__``, and owns ``blob @ Class``."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        contracts = {}
        for base in bases:
            contracts.update(getattr(base, "__contracts__", {}) or {})

        declared = {}
        for key, value in list(namespace.items()):
            # ``Thing(str, ...)`` in a class body is a declaration; it and a
            # bare ``contract(...)`` are consumed identically.
            declaration = getattr(value, "__declaration__", None)
            if declaration is not None:
                value = declaration
            if isinstance(value, Contract):
                declared[key] = value.bind(key)
                del namespace[key]                   # let __getattr__ see the name
        # `source_text: str` declares an attribute too.  A bare
        # annotation carries a shape and nothing else, which is exactly the
        # contract it means.
        for key, annotation in (namespace.get("__annotations__") or {}).items():
            if key.startswith("_") or key in declared or key in contracts:
                continue
            template = annotation if isinstance(annotation, type) else Any
            declared[key] = Contract(template).bind(key)
        contracts.update(declared)

        namespace["__contracts__"] = contracts
        namespace.setdefault("__annotated__", tuple(namespace.get("__annotations__", ())))
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # __beliefs__ is the one attachment point: the declared list, then the
        # beliefs the contracts imply, then the resident human if the user did
        # not name one (the human is in every default panel).
        own = namespace.get("__beliefs__")
        if own is None:
            panel = list(getattr(cls, "__beliefs__", ()) or [])
            attaching = declared
        else:
            # an explicit list replaces the inherited panel, so every contract
            # still in force has to re-attach its checks -- otherwise a
            # subclass would silently inherit a declaration without its
            # guarantees.
            panel = list(own)
            attaching = contracts
        for attr, spec in attaching.items():
            panel.extend(spec.beliefs(attr))
        if not any(isinstance(b, HumanBelief) for b in panel):
            panel.append(human(_config("human") or "you"))
        cls.__beliefs__ = panel
        return cls

    # -- revive: blob @ MyClass --------------------------------------
    def __rmatmul__(cls, blob):
        from .persist import revive

        return revive(blob, cls)

    def __repr__(cls) -> str:                        # pragma: no cover - cosmetic
        return f"<Thing {cls.__name__}>"


def resident_human(thing) -> HumanBelief:
    """The human whose voice an assignment carries."""
    for belief in thing.__beliefs__:
        if isinstance(belief, HumanBelief):
            return belief
    return human("you")


def _declare_panel(thing) -> None:
    """Record the declared panel, idempotently.

    The declared panel is part of the record: adding or removing a belief
    changes what the strategy *is*, and the history should say so.  The
    first declaration is the baseline, a changed one derives as a
    ``[belief]`` commit.  Members without durable ids are structural
    guests and stay out of the fingerprint.
    """
    ledger = thing.__ledger__
    panel = [belief_id for b in getattr(thing, "__beliefs__", ())
             if (belief_id := getattr(b, "id", None))]
    if not panel:
        return
    declared = ledger.latest(thing.__entity__, "__panel__")
    if declared is None or not values_equal(declared.value, panel):
        ledger.add(Opinion(
            belief="panel:declared", entity=thing.__entity__,
            attr="__panel__", value=list(panel), p=1.0, frozen=True,
            meta={"panel": True}))


def _is_belief(candidate) -> bool:
    """What the panel operators accept: a Belief, or any callable carrying a
    durable ``id`` -- the same structural bar membership itself has."""
    if isinstance(candidate, Belief):
        return True
    return callable(candidate) and isinstance(
        getattr(candidate, "id", None), str)


def _forget(thing) -> None:
    """Standing resolutions predate a panel change; drop them."""
    resolved = thing.__root__.__resolved__
    for key in [k for k in resolved if k[0] == thing.__entity__]:
        resolved.pop(key, None)
    thing.__root__.__coerced__.clear()


def _with_panel(thing, panel):
    """A fresh Thing on the same entity and ledger, carrying ``panel``."""
    return type(thing)(__entity__=thing.__entity__,
                       __ledger__=thing.__ledger__, __beliefs__=panel)


# --------------------------------------------------------------------------
# the sealed snapshot
# --------------------------------------------------------------------------

class Snapshot:
    """An immutable value exposing the ordinary public interface.

    Attribute reads are **inert**: they never consult, never spend, never
    mutate, and an unresolved cell reads as ``None``.  Everything
    framework-side is a dunder so no domain attribute is ever shadowed.
    """

    __slots__ = ("__entity__", "__beliefs__", "__ledger__", "__value__", "__p__",
                 "__contracts__", "__methods__", "__arguments__", "__objections__",
                 "__owner__", "__episode__", "__call_arguments__", "__class_name__",
                 "__provenance__", "__purpose__", "__round__", "__peers__",
                 "__boundary__", "_cells")

    def __init__(self, *, entity, beliefs=(), ledger=None, cells=None, value=None,
                 p=0.0, contracts=None, methods=(), arguments=None, objections=(),
                 owner=None, episode=None, call_arguments=None, class_name="Thing",
                 provenance=(), purpose="", round=0, peers=None, boundary=False):
        set = object.__setattr__
        set(self, "__entity__", entity)
        set(self, "__beliefs__", list(beliefs))
        set(self, "__ledger__", ledger)
        set(self, "__value__", value)
        set(self, "__p__", float(p))
        set(self, "__contracts__", dict(contracts or {}))
        set(self, "__methods__", tuple(methods))
        set(self, "__arguments__", dict(arguments or {}))
        set(self, "__objections__", tuple(objections))
        set(self, "__owner__", owner)
        set(self, "__episode__", episode)
        set(self, "__call_arguments__", call_arguments)
        set(self, "__class_name__", class_name)
        set(self, "__provenance__", tuple(provenance))
        set(self, "__purpose__", purpose or "")
        set(self, "__round__", round)
        set(self, "__peers__", dict(peers or {}))
        set(self, "__boundary__", bool(boundary))
        set(self, "_cells", dict(cells or {}))

    # -- the ordinary public interface ------------------------------------
    def __getattr__(self, name):
        cells = object.__getattribute__(self, "_cells")
        if name in cells:
            return cells[name]
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return None                          # an unresolved cell reads as absent

    def __setattr__(self, name, value):
        raise AttributeError(
            f"a snapshot is sealed; {name!r} cannot be set (invariant 3)")

    def __delattr__(self, name):
        raise AttributeError("a snapshot is sealed (invariant 3)")

    def __attrs__(self):
        return dict(object.__getattribute__(self, "_cells"))

    def __opinion__(self, attr):
        return object.__getattribute__(self, "_cells").get(attr)

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Snapshot) and self.__fingerprint__() == other.__fingerprint__())

    def __hash__(self):
        return hash(self.__fingerprint__())

    def __fingerprint__(self):
        """Everything a belief can see, as a comparable value.

        Used by the statelessness property: a re-derivation's ``e₀`` is
        byte-identical to the first derivation's.  The panel is excluded: it is callables, and
        which memo wrapper holds them is not something a belief can observe.
        """
        cells = object.__getattribute__(self, "_cells")
        return (
            self.__entity__, self.__class_name__, self.__purpose__,
            repr(self.__value__), self.__p__, self.__provenance__,
            tuple(sorted((a, repr(o.value), o.p, o.frozen, o.belief)
                         for a, o in cells.items())),
            tuple(sorted(self.__contracts__)), self.__methods__,
            tuple(sorted((k, v.__fingerprint__()) for k, v in self.__arguments__.items())),
            tuple(repr(o) for o in self.__objections__),
            repr(self.__call_arguments__),
            tuple(sorted((k, v.__fingerprint__()) for k, v in self.__peers__.items())),
            self.__boundary__,
        )

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        return f"<snapshot {self.__entity__} r{self.__round__}>"


def snapshot(thing, attr=None) -> Snapshot:
    """A sealed, inert value of ``thing`` as the panel would see it.

    Its panel entries are memoized, exactly as a round's are: surveying by
    hand costs one evaluation per belief per cell, no matter how many
    validators pull the proposer.  Direct calls still record nothing --
    inspection never mutates memory.
    """
    memo, active = {}, set()
    entries = [MemoBelief(b, memo, active) for b in thing.__beliefs__]
    return _build_snapshot(thing, attr, beliefs=entries)


# --------------------------------------------------------------------------
# the entity boundary: perception between minds (SPEC.md §4)
# --------------------------------------------------------------------------

#: live root Things by entity id, weakly held.  A reference is a capability:
#: dereferencing an entity id at render time means finding its live handle
#: and reading its *current* standing state -- never a remembered copy.  The
#: registry keeps no state itself; the state is always read fresh.
_LIVE: "weakref.WeakValueDictionary[str, Thing]" = weakref.WeakValueDictionary()


def public_attrs(cls) -> set:
    """The attributes of ``cls`` declared ``public=True`` -- nothing else
    crosses an entity boundary.  Default private, fail closed."""
    return {name for name, declared in
            (getattr(cls, "__contracts__", {}) or {}).items()
            if getattr(declared, "public", False)}


def boundary_snapshot(thing) -> Snapshot:
    """The view of ``thing`` that crosses an entity boundary (SPEC.md §4).

    Perception between minds is deliberately poorer than self-knowledge:
    identity, purpose (the class docstring) and **public** cells only -- no
    panel, no ledger slice, no contracts, no methods, no private cells.  An
    attribute crosses only if declared ``public=True``; an assigned but
    undeclared cell is private like everything else.  Building this view
    never derives: an unresolved public cell is simply absent.
    """
    if isinstance(thing, Cell):
        # A cell crossing carries its already-resolved value only if its
        # attribute is public on the parent; never __resolve__ -- perception
        # must not trigger derivation.
        opinion = thing.__opinion__
        cells = {}
        if opinion is not None and \
                thing.__attr__ in public_attrs(type(thing.__parent__)):
            cells = {thing.__attr__: opinion}
        return Snapshot(entity=thing.__entity__, cells=cells,
                        class_name=type(thing.__parent__).__name__,
                        purpose="", boundary=True)
    allowed = public_attrs(type(thing))
    cells = {attr: opinion for attr, opinion in
             _standing(thing.__root__, thing.__entity__).items()
             if attr in allowed}
    return Snapshot(entity=thing.__entity__, cells=cells,
                    class_name=type(thing).__name__,
                    purpose=(type(thing).__doc__ or "").strip(),
                    boundary=True)


def _peer_views(thing, cells) -> dict:
    """Held references, dereferenced fresh at snapshot time (SPEC.md §4).

    An entity id in a standing cell's ``meta.refs`` renders as that entity's
    *boundary* view, looked up live -- never remembered.  An id with no live
    handle on this same ledger fails closed to the bare id: identity without
    state.
    """
    ledger = thing.__ledger__
    out: dict[str, Snapshot] = {}
    for opinion in cells.values():
        for ref in (opinion.meta or {}).get("refs", ()):
            base = str(ref).split("#", 1)[0]
            if base == thing.__entity__ or base in out:
                continue
            live = _LIVE.get(base)
            if live is not None and live.__ledger__ is ledger:
                out[base] = boundary_snapshot(live)
            else:
                out[base] = Snapshot(entity=base, class_name="", boundary=True)
    return out


def record_refs(value, ledger) -> list:
    """Entity ids carried by a *plain* value crossing into the record.

    The inverse of the Thing→id reduction (:func:`references`): a model that
    was shown an entity id can only hand it back as a string, so a changeset
    write whose value is -- or contains, walking containers -- a string equal
    to a known entity's id (or ``entity#attr`` address) earns the same
    ``meta["refs"]`` stamp an assignment of the live Thing would carry.
    Exact string match only: a mention inside prose is prose, not a
    capability.
    """
    known = {cell[0] for cell in ledger.cells()}
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            if v in known or v.split("#", 1)[0] in known:
                out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple, set, frozenset)):
            for x in v:
                walk(x)

    walk(value)
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------
# Thing
# --------------------------------------------------------------------------

class Thing(metaclass=ThingMeta):
    """A probabilistic object: attributes are beliefs, reads are consultations."""

    __beliefs__: list = []
    #: the generative belief a read falls back to when the panel declares
    #: none -- set once (``Thing.__default__ = model("...")`` process-wide,
    #: or on a subclass) instead of repeating the model in every panel.
    __default__ = None

    def __init__(self, *template, **kwargs):
        set = object.__setattr__
        if template:
            # A Thing with a positional shape is a *declaration*: it exists
            # to sit in a class body and declare an attribute --
            # ``priority = Thing(str, enum=[...])``.  No entity, no ledger,
            # no record: the metaclass consumes it at class creation.
            if len(template) > 1:
                raise TypeError("a declaration takes one shape: "
                                "Thing(str, ...), Thing([{...}], ...)")
            set(self, "__declaration__", Contract(template[0], **kwargs))
            set(self, "__entity__", "(declaration)")
            set(self, "__root__", self)
            set(self, "__resolved__", {})
            set(self, "__coerced__", {})
            return
        entity = kwargs.pop("__entity__", None)
        set(self, "__entity__", entity or
            f"{type(self).__name__.lower()}-{uuid.uuid4().hex[:12]}")
        # An empty Ledger is falsy, so this cannot be an `or` chain: injecting
        # a fresh ledger for a test is exactly the case that would break.
        ledger = kwargs.pop("__ledger__", None)
        if ledger is None:
            ledger = _ledgers[-1] if _ledgers else default_ledger()
        set(self, "__ledger__", ledger)
        set(self, "__root__", self)
        set(self, "__resolved__", {})
        set(self, "__coerced__", {})
        beliefs = kwargs.pop("__beliefs__", None)
        if beliefs is not None:
            set(self, "__beliefs__", list(beliefs))
        _LIVE[self.__entity__] = self
        _declare_panel(self)
        # constructor kwargs are assignments, and assignment is the human
        # speaking
        for name, value in kwargs.items():
            setattr(self, name, value)
        # eager contracts resolve now, while the constructor still owns the
        # stack -- deterministic panels serve from the record at zero calls
        for name, declared in type(self).__contracts__.items():
            if getattr(declared, "eager", False):
                getattr(self, name)

    # -- reading: the read pipeline (SPEC.md §5) -------------------------------------
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if _is_framework_name(name):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}")
        contracts = type(self).__contracts__
        if name in contracts:
            # a declared attribute resolves on access, so that + and ~ stay
            # pure extraction
            return _read(self, name, contracts[name])
        return Cell(self, name, deferred=True)

    # -- writing: assignment is a frozen human opinion --------------
    def __setattr__(self, name, value):
        if name.startswith("__") and name.endswith("__"):
            object.__setattr__(self, name, value)
            return
        if _is_framework_name(name):
            object.__setattr__(self, name, value)
            return
        author = resident_human(self)
        plain = _plain(value)
        latest = self.__ledger__.latest(self.__entity__, name, frozen=True)
        if latest is not None and latest.belief == author.id \
                and values_equal(latest.value, plain):
            # Idempotent re-statement: replaying the same program against the
            # same record is the intended usage, and a diary of identical
            # assignments informs nobody.  Nothing changed, nothing recorded,
            # caches stand.
            return
        meta = {"assigned": True}
        refs = references(value)
        if refs:
            meta["refs"] = refs
        self.__ledger__.add(Opinion(
            belief=author.id, entity=self.__entity__, attr=name,
            value=plain, p=1.0, frozen=True,
            meta=meta))
        self.__root__.__resolved__.pop((self.__entity__, name), None)
        self.__root__.__coerced__.clear()

    def __delattr__(self, name):
        raise AttributeError(
            "there is no deletion: the ledger is append-only (invariant 2). "
            "Record a new opinion instead.")

    # -- the dunder surface -----------------------------------------
    @property
    def __source__(self) -> str:
        from .debug import source

        return source(self)

    def __getstate__(self):
        from .persist import dump_thing

        return dump_thing(self)

    def __setstate__(self, blob):
        from .persist import load_into

        load_into(self, blob)

    def __reduce__(self):
        from .persist import __reduce_thing__

        return __reduce_thing__(self)

    # -- the four reserved words ------------------------------------
    @classmethod
    @contextlib.contextmanager
    def require(cls, floor: float):
        """Raise :class:`LowConfidence` for anything resolving below ``floor``."""
        _floors.append(float(floor))
        try:
            yield cls
        finally:
            _floors.pop()

    @classmethod
    @contextlib.contextmanager
    def policy(cls, policy: ResolutionPolicy):
        """Select the active :class:`~thinair.policy.ResolutionPolicy`."""
        _policies.append(policy)
        try:
            yield cls
        finally:
            _policies.pop()

    @classmethod
    @contextlib.contextmanager
    def debug(cls, stream=None):
        """Print one bordered box per operation to stderr."""
        from .debug import tracing

        with tracing(stream):
            yield cls

    @classmethod
    @contextlib.contextmanager
    def defaults(cls, *, ledger=None, models=None, **config):
        """Per-class configuration with MRO inheritance.

        ``models=[...]`` is sugar that prepends generative beliefs to this
        class's panel; there is no separate model-configuration mechanism.
        """
        owner = None if cls is Thing else cls
        previous_panel = cls.__dict__.get("__beliefs__")
        if models:
            cls.__beliefs__ = list(models) + list(cls.__beliefs__)
        if ledger is not None:
            _ledgers.append(ledger)
        try:
            with _config_scope(owner, **config):
                yield cls
        finally:
            if ledger is not None:
                _ledgers.pop()
            if models:
                if previous_panel is None:
                    del cls.__beliefs__
                else:
                    cls.__beliefs__ = previous_panel

    # -- operators ---------------------------------------------------
    def __matmul__(self, other):
        return _exit_ramp(self, other)

    # -- the panel, through operators ---------------------------------
    # ``t += belief`` / ``t -= belief`` change this Thing's panel in place;
    # ``t + belief`` / ``t - belief`` hand back a fresh Thing on the same
    # entity and ledger with the changed panel.  Either way the change is a
    # panel declaration: the strategy moved, and the record says so.
    def __iadd__(self, belief):
        if not _is_belief(belief):
            return NotImplemented
        object.__setattr__(self, "__beliefs__",
                           list(self.__beliefs__) + [belief])
        _declare_panel(self)
        _forget(self)
        return self

    def __isub__(self, belief):
        if not _is_belief(belief):
            return NotImplemented
        panel = list(self.__beliefs__)
        for i, member in enumerate(panel):
            if member is belief or getattr(member, "id", None) == belief.id:
                del panel[i]
                break
        else:
            raise ValueError(f"{belief.id!r} is not on this panel")
        object.__setattr__(self, "__beliefs__", panel)
        _declare_panel(self)
        _forget(self)
        return self

    def __add__(self, belief):
        if not _is_belief(belief):
            return NotImplemented
        return _with_panel(self, list(self.__beliefs__) + [belief])

    def __sub__(self, belief):
        if not _is_belief(belief):
            return NotImplemented
        panel = list(self.__beliefs__)
        for i, member in enumerate(panel):
            if member is belief or getattr(member, "id", None) == belief.id:
                del panel[i]
                break
        else:
            raise ValueError(f"{belief.id!r} is not on this panel")
        return _with_panel(self, panel)

    #: a Thing crossing the record's boundary reduces to its entity id --
    #: ``self.part = Thing(...)`` writes a *reference*, and ``meta["refs"]``
    #: carries the same address (see :func:`references`).
    def __thinair_unwrap__(self):
        return self.__entity__

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.__entity__}>"


# --------------------------------------------------------------------------
# a child Thing bound to a resolving opinion
# --------------------------------------------------------------------------

_MISSING = object()


class Cell(Thing):
    """What a read returns: a real child Thing, bound to a resolving opinion.

    A Thing never owns a value.  A cell *is* an ``(entity, attr)``
    address; the value lives in the opinion this child references, and ``+``
    / ``~`` extract from it.  There is no lazy proxy here -- a deferred cell
    is one whose *derivation* has not been asked for yet, which is a
    different thing from a value that pretends to exist.
    """

    def __init__(self, parent, attr, *, opinion=None, deferred=False,
                 contract=None, entity=None):
        root = parent.__root__
        set = object.__setattr__
        set(self, "__parent__", parent)
        set(self, "__attr__", attr)
        set(self, "__cell__", (parent.__entity__, attr))
        set(self, "__entity__", entity or f"{parent.__entity__}#{attr}")
        set(self, "__ledger__", parent.__ledger__)
        set(self, "__root__", root)
        set(self, "__resolved__", {})
        set(self, "__coerced__", {})
        set(self, "__beliefs__", list(parent.__beliefs__))
        set(self, "__contract__", contract)
        set(self, "__opinion__", opinion)
        set(self, "__deferred__", bool(deferred) and opinion is None)

    # -- resolution on demand ----------------------------------------------
    def __resolve__(self):
        if self.__opinion__ is None and self.__deferred__:
            parent, attr = self.__parent__, self.__attr__
            resolved = _read(parent, attr, self.__contract__)
            object.__setattr__(self, "__opinion__", resolved.__opinion__)
            object.__setattr__(self, "__deferred__", False)
        return self.__opinion__

    @property
    def __value__(self):
        opinion = self.__resolve__()
        return None if opinion is None else opinion.value

    @property
    def __p__(self):
        opinion = self.__resolve__()
        return 0.0 if opinion is None else opinion.p

    # -- extractors --------------------------------------------------
    def __pos__(self):
        return self.__value__

    def __invert__(self):
        return self.__p__

    #: how ``ledger._unwrap`` reduces a Thing to a plain value without an
    #: import edge between the two modules.
    def __thinair_unwrap__(self):
        return self.__value__

    # -- re-derivation: the call form --------------------------------
    def __call__(self, *args, **kwargs):
        parent, attr = self.__parent__, self.__attr__
        contracts = type(parent).__contracts__
        if attr in contracts and not args and not kwargs:
            return _read(parent, attr, contracts[attr], force=True)
        from .rounds import current_scope             # round-time (§15):

        scope = current_scope(parent.__ledger__)
        if scope is not None and not scope._evaluating:
            # inside ``with ledger.rounds()`` a call declares a pending
            # turn -- sealed input, evaluated at ``ledger.round()`` (or on
            # forcing).  The scope's own evaluations pass through.
            return scope.declare(parent, attr, args, kwargs)
        from .episode import run                      # imagined method call

        return run(parent, attr, args, kwargs)

    # -- delegation to the carried value ------------------------------------
    def __eq__(self, other):
        return values_equal(self.__value__, _plain(other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.__value__ < _plain(other)

    def __le__(self, other):
        return self.__value__ <= _plain(other)

    def __gt__(self, other):
        return self.__value__ > _plain(other)

    def __ge__(self, other):
        return self.__value__ >= _plain(other)

    def __hash__(self):
        value = self.__value__
        try:
            return hash(value)
        except TypeError:
            return hash(repr(value))

    def __bool__(self):
        return bool(self.__value__)

    def __len__(self):
        return len(self.__value__)

    def __iter__(self):
        return iter(self.__value__)

    def __contains__(self, item):
        return _plain(item) in self.__value__

    def __getitem__(self, key):
        return self.__value__[_plain(key)]

    def __int__(self):
        return int(self.__value__)

    def __float__(self):
        return float(self.__value__)

    def __index__(self):
        return int(self.__value__)

    def __str__(self):
        return str(self.__value__)

    def __format__(self, spec):
        return format(self.__value__, spec)

    def __add__(self, other):
        return self.__value__ + _plain(other)

    def __radd__(self, other):
        return _plain(other) + self.__value__

    def __sub__(self, other):
        return self.__value__ - _plain(other)

    def __rsub__(self, other):
        return _plain(other) - self.__value__

    def __mul__(self, other):
        return self.__value__ * _plain(other)

    def __rmul__(self, other):
        return _plain(other) * self.__value__

    def __truediv__(self, other):
        return self.__value__ / _plain(other)

    def __rtruediv__(self, other):
        return _plain(other) / self.__value__

    def __neg__(self):
        return -self.__value__

    def __abs__(self):
        return abs(self.__value__)

    def __round__(self, n=None):
        return round(self.__value__, n) if n is not None else round(self.__value__)

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        if self.__deferred__:
            return f"<{self.__entity__} (not yet derived)>"
        opinion = self.__opinion__
        if opinion is None:
            return f"<{self.__entity__} = None>"
        mark = " frozen" if opinion.frozen else ""
        return (f"<{self.__entity__} = {opinion.value!r} "
                f"p={opinion.p:.2f} per {opinion.belief}{mark}>")


def _plain(value):
    """A Thing reduces to its carried value; everything else is itself."""
    unwrap = getattr(type(value), "__thinair_unwrap__", None)
    return unwrap(value) if unwrap is not None else value


def references(value) -> list[str]:
    """Durable references carried by a value crossing into the record.

    ``_plain`` and call rendering reduce Things and Cells to their values --
    the right identity for *comparison* -- but the reduction destroys
    provenance: the record keeps ``"Q3"`` with no trace that it came from
    ``doc-1#title``.  This walk collects what the reduction discards, so the
    boundary can stamp it into ``meta["refs"]``: a Thing contributes its
    entity, a Cell its ``entity#attr`` address, containers are walked.  A
    plain object contributes nothing -- no identity, no reference; wrap it
    as a Thing to give it one.
    """
    out: list[str] = []

    def walk(v):
        if isinstance(v, Cell):                     # before Thing: Cell is one
            entity, attr = v.__cell__
            out.append(f"{entity}#{attr}")
        elif isinstance(v, Thing):
            out.append(v.__entity__)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple, set, frozenset)):
            for x in v:
                walk(x)

    walk(value)
    return list(dict.fromkeys(out))                 # dedupe, order kept


# --------------------------------------------------------------------------
# the read pipeline
# --------------------------------------------------------------------------

def _standing(owner, entity, deriving=None):
    """Frozen state and standing resolutions for one entity, as opinions.

    This -- and *not* a raw ledger slice -- is what ``e₀`` is built from, so
    that a re-derivation's inputs are byte-identical to the first one's.
    Prior negotiations exist in the ledger; they are memory,
    not input.
    """
    ledger = owner.__ledger__
    cells = {}
    for cell in ledger.cells(entity):
        if cell[1] == deriving:
            continue          # a cell under derivation is, by definition, open
        if cell[1].startswith("__"):
            continue          # framework records (the panel fingerprint) are
                              # never state a belief is shown
        pinned = ledger.latest_frozen(*cell)
        if pinned is not None:
            cells[cell[1]] = pinned
    for (ent, attr), opinion in owner.__root__.__resolved__.items():
        if ent == entity and attr not in cells and attr != deriving:
            cells[attr] = opinion
    return cells


def _methods_of(thing) -> tuple:
    out = []
    for name, member in inspect.getmembers(type(thing), inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            line = f"def {name}{inspect.signature(member)}"
        except (TypeError, ValueError):              # pragma: no cover - exotic
            line = f"def {name}(...)"
        doc = inspect.getdoc(member)
        if doc:
            line += f"  # {doc.splitlines()[0]}"
        out.append(line)
    return tuple(out)


def _visible_ledger(owner, entity, cells, extra=()):
    """The snapshot's visible slice of memory."""
    from .ledger import Ledger

    view = Ledger()
    for opinion in list(cells.values()) + list(extra):
        view.add(opinion)
    return view


def _build_snapshot(thing, attr=None, *, beliefs=(), objections=(), round=0,
                    extra=(), episode=None, arguments=None, call_arguments=None):
    call_arguments = (call_arguments if call_arguments is not None
                      else getattr(thing, "__call_arguments__", None))
    owner = thing.__root__
    entity = thing.__entity__
    if isinstance(thing, Cell):
        cells = {}
        value = thing.__opinion__.value if thing.__opinion__ is not None else None
        p = thing.__opinion__.p if thing.__opinion__ is not None else 0.0
        provenance = _provenance(thing)
        peers = {}
    else:
        cells = _standing(owner, entity, deriving=attr)
        value, p, provenance = None, 0.0, ()
        peers = _peer_views(thing, cells)
    return Snapshot(
        entity=entity,
        beliefs=list(beliefs),
        ledger=_visible_ledger(owner, entity, cells, extra),
        cells=cells,
        value=value,
        p=p,
        contracts=type(thing).__contracts__,
        methods=_methods_of(thing),
        arguments=arguments,
        objections=objections,
        owner=type(owner),
        episode=episode,
        call_arguments=call_arguments,
        class_name=getattr(thing, "__class_name__", None) or type(thing).__name__,
        provenance=provenance,
        purpose=(getattr(thing, "__purpose__", None)
                 or (type(thing).__doc__ or "").strip()),
        round=round,
        peers=peers,
    )


def _provenance(cell) -> tuple:
    """One line, never an ancestor transcript."""
    opinion = cell.__opinion__
    if opinion is None:
        return ()
    parent = cell.__parent__
    return (f"derived as {type(parent).__name__}.{cell.__attr__} "
            f"at p={opinion.p:.2f} per {opinion.belief}",)


def _read(thing, attr, contract=None, *, force=False):
    """``thing.attr`` -- the read pipeline (SPEC.md §5), start to finish."""
    from .debug import trace

    owner = thing.__root__
    entity = thing.__entity__
    cell = (entity, attr)
    ledger = thing.__ledger__

    # 0. frozen short-circuit: no consultation, no model call, ever
    pinned = ledger.latest_frozen(entity, attr)
    if pinned is not None and not force:
        return Cell(thing, attr, opinion=pinned, contract=contract)

    if not force:
        standing = owner.__resolved__.get(cell)
        if standing is not None:
            return Cell(thing, attr, opinion=standing, contract=contract)

    panel = list(thing.__beliefs__)
    if contract is not None and contract.attr is None:
        # An *ephemeral* contract -- the `returns=` slot of an episode, or the
        # schema on the right of `@`.  A declared contract already attached
        # its beliefs at class creation; this one has to attach them
        # for the length of this derivation, or the schema would constrain
        # nothing.
        panel = panel + contract.bind(attr).beliefs(attr)
    route = Route(panel)
    if route.empty:
        # the declared fallback: a panel that names no generative member
        # consults ``__default__`` -- one place to change the model for a
        # class, or (on Thing itself) for the whole process.
        fallback = getattr(type(thing), "__default__", None)
        if fallback is not None:
            panel = panel + [fallback]
            route = Route(panel)
    if route.empty:
        raise Unresolvable(
            cell, reason="no generative belief in __beliefs__ and no "
                         "__default__; a deterministic-only Thing serves "
                         "frozen cells and nothing else")

    floor = active_floor()
    policy = active_policy()
    # Declared expectations travel with every generative record for the cell,
    # so the settlement layer can judge the reading it actually kept.
    expect = contract.expectations if contract is not None else None
    objections: list[dict] = []
    recorded: list[Opinion] = []
    round_number = 0

    while True:
        round_number += 1
        # 1./2. one sealed snapshot, one round, every belief called once
        memo, active = {}, set()
        entries = [MemoBelief(b, memo, active) for b in route.panel(panel)]
        e = _build_snapshot(thing, attr, beliefs=entries, objections=objections,
                            round=round_number, extra=recorded)

        # 0b. replay: if the world this read would show the head is
        # byte-identical to what the record's last negotiation showed it,
        # that negotiation *is* this read (SPEC.md §5).
        if round_number == 1 and not force:
            served = _replay(thing, attr, contract, route, policy, entries,
                             e, ledger, entity, floor)
            if served is not None:
                return served

        opinions = []
        for entry in entries:
            # Generative members below the routed head are the *ladder*
            #, not this round's panel: consulting them would spend a
            # model call on a route nobody has escalated to.  A policy that
            # judges proposers against each other asks for them explicitly.
            if entry.proposes and entry.id != route.name \
                    and not policy.consults_all_proposers:
                continue
            got = entry(e, attr)
            if got is None:
                continue
            stated = dict(getattr(got, "meta", None) or {}, round=round_number)
            if expect and entry.proposes:
                stated["expect"] = expect
            opinions.append(ledger.add(Opinion(
                belief=entry.id, entity=entity, attr=attr,
                value=_plain(+got), p=~got, frozen=False, meta=stated)))
        recorded.extend(opinions)

        proposal = memo.get((route.name, attr))
        vetoes = _vetoes(entries, memo, attr, proposal)
        attempt = Attempt(round=round_number, route=route.name,
                          value=None if proposal is None else +proposal,
                          p=0.0 if proposal is None else ~proposal,
                          vetoes=vetoes, opinions=opinions)
        route.spend(attempt)
        trace("round", cell, attempt)

        if proposal is not None and not vetoes:
            chosen = policy.resolve(
                [o for o in opinions if o.belief in _proposer_ids(policy, entries, route)],
                cell, proposer=route.name)
            if chosen is not None:
                value, p = chosen
                if floor is None or p >= floor or not route.can_escalate:
                    if floor is not None and p < floor:
                        raise LowConfidence(cell, p, floor)
                    return _settle(thing, attr, contract, value, p, route,
                                   opinions, ledger, entity)
                # (b) resolved p below an active require -> escalate
                route.escalate()
                objections.append({
                    "value": value, "belief": route.name, "p": p,
                    "reason": f"resolved at p={p:.2f}, below the required {floor:.2f}"})
                continue

        objections.extend(attempt.objections())
        if route.exhausted and not route.escalate():
            raise Unresolvable(cell, route.attempts,
                               "the veto budget was exhausted")


def _replay(thing, attr, contract, route, policy, entries, e, ledger, entity,
            floor):
    """Serve the recorded negotiation when the world has not moved.

    A reading stands until the observed state changes (Pillar IV); the
    exposure stamp makes "unchanged" checkable per cell instead of assumed.
    The gate is deliberately conservative -- any doubt falls through to a
    live consultation, which is never wrong, only costlier:

    * only under a policy whose resolution is the head's vetted final
      candidate (``replays_from_record``);
    * only when the head can fingerprint its context without spending;
    * only when the record's *latest* round-1 reading carries the same
      exposure -- an older match means the world moved and moved back,
      which is a fresh question;
    * only when the recorded final candidate survived its judges, every one
      of which must be on this panel, and clears any active ``require``.

    Serving records nothing: replay is free, and the ledger stays a record
    of consultations rather than of cache hits.
    """
    if not getattr(policy, "replays_from_record", False):
        return None
    fingerprint = getattr(route.head, "exposure", None)
    if fingerprint is None:
        return None
    stated = [o for o in ledger.opinions(entity=entity, attr=attr,
                                         belief=route.name)
              if not (o.meta or {}).get("corroboration")]
    if not stated:
        return None
    openers = [o for o in stated if (o.meta or {}).get("round") == 1]
    if not openers or openers[-1].meta.get("exposure") != fingerprint(e, attr):
        return None
    final = stated[-1]
    if floor is not None and final.p < floor:
        return None
    by_id = {entry.id: entry for entry in entries}
    for other in ledger.opinions(entity=entity, attr=attr):
        if other.belief == route.name or other.frozen:
            continue
        if (other.meta or {}).get("corroboration"):
            continue
        if other.t < openers[-1].t:
            continue                       # an earlier negotiation's verdict
        if not values_equal(other.value, final.value):
            continue
        judge = by_id.get(other.belief)
        if judge is None:
            return None                    # a judge this panel no longer has
        if judge.necessary and other.p < judge.veto_line:
            return None                    # the record ended vetoed: ask live
    thing.__root__.__resolved__[(entity, attr)] = final
    from .debug import trace

    trace("replayed", (entity, attr), (final.value, final.p, final.belief))
    return Cell(thing, attr, opinion=final, contract=contract)


def _proposer_ids(policy, entries, route):
    """Whose opinions the policy is entitled to select among."""
    if policy.consults_all_proposers:
        return {entry.id for entry in entries if entry.proposes}
    return {route.name}


def _vetoes(entries, memo, attr, proposal):
    """Necessary beliefs whose p for *the current candidate* is below the line.

    A belief that judged something else is not vetoing this candidate --
    which matters because a relation may substitute values from other cells.
    """
    out = []
    for entry in entries:
        if not entry.necessary:
            continue
        got = memo.get((entry.id, attr))
        if got is None:
            continue
        if proposal is not None and not values_equal(+got, +proposal):
            continue
        if ~got < entry.veto_line:
            reason = (getattr(got, "meta", None) or {}).get("reason")
            out.append((entry.id, ~got, reason))
    return out


def _settle(thing, attr, contract, value, p, route, opinions, ledger, entity):
    """Record the standing resolution and hand back the child."""
    resolving = None
    for opinion in reversed(opinions):
        if opinion.belief == route.name and values_equal(opinion.value, value) \
                and opinion.p == p:
            resolving = opinion
            break
    if resolving is None:
        # The policy selected something the routed head did not state verbatim
        # -- Threshold's None, Unanimous's weakest p.  That is a refusal or a
        # choice, not a blend (invariant 5), and it is recorded under the
        # policy's own name so the ledger never attributes it to a model.
        policy = active_policy()
        resolving = ledger.add(Opinion(
            belief=f"policy:{policy.name}", entity=entity, attr=attr,
            value=value, p=p, frozen=False,
            meta={"selected_from": [o.belief for o in opinions]}))
    elif getattr(route.head, "frozen", False):
        # Freezing is a property of the belief, not of the Thing: a member
        # declared ``frozen=True`` (code, notably) pins what it settles --
        # at settlement, never at statement, so a vetoed candidate cannot
        # pin anything.
        resolving = ledger.add(Opinion(
            belief=resolving.belief, entity=entity, attr=attr,
            value=value, p=p, frozen=True,
            meta=dict(resolving.meta or {}, pinned=True)))
    thing.__root__.__resolved__[(entity, attr)] = resolving
    from .debug import trace

    trace("resolved", (entity, attr), (value, p, resolving.belief))
    return Cell(thing, attr, opinion=resolving, contract=contract)


class _Fixed(Belief):
    """A proposer with a predetermined answer.

    An imagined mutation is just another candidate origin: the write
    enters the same panel the model's own proposal would, and the validator
    library polices it for free.
    """

    proposes = True

    def __init__(self, value, p, meta=None, **options):
        options.setdefault("id", "changeset:proposed")
        super().__init__(**options)
        self.value = value
        self.p = float(p)
        self.meta = dict(meta or {})

    def __call__(self, e, attr):
        from .beliefs import Judgment

        return Judgment(self.value, self.p, dict(self.meta))


def review(thing, attr, value, p=1.0, *, meta=None, entity=None, record=True,
           **snapshot_options):
    """Put a candidate to the discriminative panel.  Returns ``(opinion, vetoes)``.

    This is the *same* round a read performs with the proposer
    replaced by a value someone already has -- which is exactly what a
    changeset write is and what an episode's return value is.
    The opinion is recorded but never resolved: the caller decides.
    """
    head = _Fixed(value, p, meta)
    panel = [head] + [b for b in thing.__beliefs__ if not getattr(b, "proposes", False)]
    memo, active = {}, set()
    entries = [MemoBelief(b, memo, active) for b in panel]
    e = _build_snapshot(thing, attr, beliefs=entries, round=1, **snapshot_options)

    ledger = thing.__ledger__
    entity = entity or thing.__entity__
    proposal = None
    for entry in entries:
        got = entry(e, attr)
        if got is None:
            continue
        if not record:
            if entry.id == head.id:
                proposal = Opinion(belief=entry.id, entity=entity, attr=attr,
                                   value=_plain(+got), p=~got)
            continue
        recorded = ledger.add(Opinion(
            belief=entry.id, entity=entity, attr=attr,
            value=_plain(+got), p=~got, frozen=False,
            meta=dict(getattr(got, "meta", None) or {}, proposed=True)))
        if entry.id == head.id:
            proposal = recorded
    vetoes = _vetoes(entries, memo, attr, memo.get((head.id, attr)))
    return (None, vetoes) if vetoes else (proposal, [])


def propose(thing, attr, value, p=1.0, meta=None):
    """A changeset write: declared, unfrozen, and through the panel."""
    contracts = type(thing).__contracts__
    if attr not in contracts:
        return None, [("thinair:undeclared", 0.0,
                       f"{attr!r} is not a declared attribute of "
                       f"{type(thing).__name__}")]
    pinned = thing.__ledger__.latest_frozen(thing.__entity__, attr)
    if pinned is not None:
        return None, [("thinair:frozen", 0.0,
                       f"{attr!r} is frozen at {pinned.value!r} "
                       f"(per {pinned.belief}) and cannot be overwritten")]
    # A changeset write whose value carries a known entity id is a reference
    # like any other: stamp the address the value-reduction cannot carry.
    meta = dict(meta or {})
    refs = record_refs(value, thing.__ledger__)
    if refs:
        meta.setdefault("refs", refs)
    return review(thing, attr, value, p, meta=meta)


def commit(thing, opinions):
    """Land a validated changeset: all of it, or none of it."""
    for opinion in opinions:
        thing.__root__.__resolved__[(opinion.entity, opinion.attr)] = opinion
    thing.__root__.__coerced__.clear()


def state_hash(thing, *, boundary: bool = False) -> str:
    """The state an episode is a pure function of.

    Frozen attributes *and* standing resolutions, because a committed
    changeset and an interleaved assignment must both invalidate a repeated
    call -- and a committed changeset lands as ordinary, unfrozen
    opinions.

    ``boundary=True`` hashes the entity *as another mind perceives it* --
    exactly the material :func:`boundary_snapshot` lets across (§4): public
    standing cells, plus the carried value where a Cell crossing carries
    one.  A pure function's repetition key must cover its input and nothing
    finer: keying an episode on a Thing argument's full state re-runs
    byte-identical prompts whenever the argument moved privately -- movement
    the episode could never perceive.  The same alignment governs a society
    runtime's wake check: what a peer cannot perceive must not wake it.
    """
    import hashlib

    from .ledger import normal_form

    if boundary:
        view = boundary_snapshot(thing)
        material = repr((view.__entity__, sorted(
            (attr, normal_form(opinion.value), opinion.p, opinion.frozen)
            for attr, opinion in view.__attrs__().items())))
        return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    cells = _standing(thing.__root__, thing.__entity__)
    material = repr(sorted(
        (attr, normal_form(opinion.value), opinion.p, opinion.frozen)
        for attr, opinion in cells.items()))
    if isinstance(thing, Cell):
        material += repr(normal_form(thing.__value__))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# second opinions: corroborate
# --------------------------------------------------------------------------

def corroborate(thing, attrs=None, beliefs=None):
    """Ask the beliefs that were *not* asked, and record what they say.

    Layer 1 answered each cell with one belief's vetted opinion; this verb
    deliberately spends more to thicken the record for settlement.  For every
    resolved cell (``attrs`` narrows; default: every declared attribute with
    an opinion on record), each chosen belief -- default: the generative
    members below the routed head; pass ``beliefs=`` for others, on the panel
    or not -- is consulted once against the standing snapshot, and its
    opinion is recorded **into the same cell**, tagged ``corroboration``.

    Three rules, absolute: the resolution is untouched (no re-derivation, no
    blending -- ``thinair.evaluate`` grades the agreement afterwards);
    nothing freezes; and the verb is idempotent -- a belief that already
    answered this cell under this exposure is not asked again.  Frozen cells
    are corroborated too: a reading against a human assignment is
    concordance-in-waiting.
    """
    contracts = type(thing).__contracts__
    names = list(attrs) if attrs is not None else list(contracts)
    unknown = [n for n in names if n not in contracts]
    if unknown:
        raise ValueError(f"not declared on {type(thing).__name__}: {unknown}")
    entity, ledger = thing.__entity__, thing.__ledger__
    if beliefs is None:
        others = generative_members(thing.__beliefs__)[1:]     # the ladder
    else:
        others = list(beliefs)
    out: dict[str, dict] = {}
    for attr in names:
        if ledger.latest(entity, attr) is None:
            continue                       # never resolved: a coverage gap,
                                           # not a corroboration target
        for belief in others:
            entry = MemoBelief(belief, {}, set())
            e = _build_snapshot(thing, attr, beliefs=[entry])
            fingerprint = getattr(belief, "exposure", None)
            stamp = fingerprint(e, attr) if fingerprint else None
            prior = ledger.opinions(entity=entity, attr=attr, belief=entry.id)
            if prior and (stamp is None or any(
                    (o.meta or {}).get("exposure") == stamp for o in prior)):
                continue                   # asked and answered: idempotent
            got = entry(e, attr)
            if got is None:
                continue                   # no opinion here is a legal answer
            opinion = ledger.add(Opinion(
                belief=entry.id, entity=entity, attr=attr,
                value=_plain(+got), p=~got, frozen=False,
                meta=dict(getattr(got, "meta", None) or {},
                          corroboration=True)))
            out.setdefault(attr, {})[entry.id] = opinion
    return out


# --------------------------------------------------------------------------
# the exit ramp: @
# --------------------------------------------------------------------------

def _exit_ramp(thing, other):
    if isinstance(other, type) and issubclass(other, Thing):
        return _recast(thing, other)
    if isinstance(other, (int, float)) and not isinstance(other, bool):
        return _gate(thing, float(other))
    return _coerce(thing, other)


def _gate(cell, floor):
    """``t @ 0.8`` -- pure.  At or above the bar, ``t``; below, a None-carrier."""
    if not isinstance(cell, Cell):
        raise TypeError("the confidence gate applies to an attribute, not a Thing")
    p = cell.__p__
    if p >= floor:
        return cell
    return _diagnostic(cell, p, f"p={p:.2f} is below the gate at {floor:.2f}")


def _coerce(cell, schema):
    """``t @ float`` -- re-resolve under an ephemeral contract.

    The one operator allowed to derive.  An already-conforming value returns
    ``t`` with zero work; a candidate that cannot conform returns a
    ``None``-carrying Thing whose ``~`` keeps the diagnostic p.
    """
    from .validators import SchemaBelief

    if not isinstance(cell, Cell):
        raise TypeError("coercion applies to an attribute, not a Thing")
    key = (cell.__cell__, repr(schema))
    cached = cell.__root__.__coerced__.get(key)
    if cached is not None:
        return cached

    check = SchemaBelief(schema)
    opinion = cell.__resolve__()
    if opinion is not None and check.judge(opinion.value, None, cell.__attr__) == 1.0:
        cell.__root__.__coerced__[key] = cell
        return cell

    converted = _convert(opinion.value if opinion else None, schema)
    if converted is not _MISSING:
        pinned = cell.__ledger__.add(Opinion(
            belief=opinion.belief if opinion else "coercion",
            entity=cell.__cell__[0], attr=cell.__attr__, value=converted,
            p=opinion.p if opinion else 0.0, frozen=False,
            meta={"coerced_to": repr(schema)}))
        out = Cell(cell.__parent__, cell.__attr__, opinion=pinned,
                   contract=cell.__contract__)
        cell.__root__.__coerced__[key] = out
        return out

    ephemeral = Contract(schema)
    try:
        out = _read(cell.__parent__, cell.__attr__, ephemeral, force=True)
    except (Unresolvable, LowConfidence, Disagreement) as exc:
        out = _diagnostic(cell, cell.__p__, str(exc).splitlines()[0])
    cell.__root__.__coerced__[key] = out
    return out


def _convert(value, schema):
    """The cheap conversions that are not really inference at all."""
    if value is None:
        return _MISSING
    if schema is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if schema is str and isinstance(value, (int, float)):
        return str(value)
    if schema is int and isinstance(value, float) and value.is_integer():
        return int(value)
    return _MISSING


def _diagnostic(cell, p, reason):
    """A ``None``-carrying child that keeps the diagnostic p.

    ``@`` never raises on epistemic failure; exceptions are for programming
    errors.  The p survives so the caller can tell "nobody knows" from
    "somebody knows, not well enough".
    """
    opinion = Opinion(belief="thinair:gate", entity=cell.__cell__[0],
                      attr=cell.__attr__, value=None, p=p, frozen=False,
                      meta={"reason": reason})
    return Cell(cell.__parent__, cell.__attr__, opinion=opinion,
                contract=cell.__contract__)


def _recast(thing, cls):
    """``t @ ThingSubclass`` -- entity, opinions and frozen state carry over."""
    root = thing.__root__ if not isinstance(thing, Cell) else thing
    recast = cls.__new__(cls)
    set = object.__setattr__
    set(recast, "__entity__", thing.__entity__)
    set(recast, "__ledger__", thing.__ledger__)
    set(recast, "__root__", recast)
    set(recast, "__resolved__", {})          # cached resolutions invalidate
    set(recast, "__coerced__", {})
    _LIVE[recast.__entity__] = recast
    return recast
