"""The object surface: interception, resolution, episodes (SPEC.md §4-§9).

One public class with **zero instance methods**.  On a Thing every plain
attribute name belongs to the user's domain -- that is the premise of
interception -- so the framework lives entirely in dunders and in four
classmethod context managers (``require``, ``policy``, ``debug``,
``defaults``).  The verbs are module-level: ``freeze``, ``contract``,
``snapshot``.

This module imports ``beliefs``, ``ledger`` and ``policy``.  It must never
import ``engine`` or ``models`` (invariant 7) -- the model reaches the
runtime only as an ordinary member of ``__beliefs__``.
"""

from __future__ import annotations

import contextlib
import inspect
import uuid
from typing import Any

from .beliefs import Belief, HumanBelief, MemoBelief, Scoped, human
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

__all__ = ["Thing", "Cell", "Snapshot", "ThingMeta", "Contract", "contract",
           "freeze", "snapshot", "LowConfidence", "Unresolvable", "Disagreement",
           "resident_human"]

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
    the ``Schema`` it builds is the same object that constrains the engine's
    structured output and performs the post-hoc check -- so the two cannot
    drift.
    """

    def __init__(self, template=Any, *, extracted_from=None, range=None,
                 enum=None, length=None, format=None, checksum=None,
                 sums_to=None, unique=False, elaborates=False,
                 necessary=True, describe=None, beliefs=(), doc=None):
        self.template = template
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
        self._describe = describe
        self._extra = list(beliefs)
        self.attr = None
        self.schema = None

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
            self.schema = V.Schema(self.template, necessary=self.necessary)
            out.append(self.schema)
        if self.extracted_from:
            out.append(self._grounding(V))
        if self.range is not None:
            out.append(V.Range(self.range[0], self.range[1],
                               necessary=self.necessary))
        if self.enum is not None:
            out.append(V.Enum(self.enum, necessary=self.necessary))
        if self.length is not None:
            out.append(V.Length(self.length[0], self.length[1],
                                necessary=self.necessary))
        if self.format is not None:
            out.append(V.Format(self.format, necessary=self.necessary))
        if self.checksum is not None:
            out.append(V.Checksum(self.checksum, necessary=self.necessary))
        if self.unique:
            out.append(V.Unique(necessary=self.necessary))
        if self.elaborates:
            out.append(V.NonEcho(necessary=True))
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
        Numbers and containers get ``TokenSubset`` instead, ``necessary``
        because a missing number is the classic hallucination tell.
        """
        if self.template is str:
            return V.Verbatim(self.extracted_from, necessary=self.necessary)
        return V.TokenSubset(self.extracted_from, necessary=self.necessary)

    def _sums_to(self, V, attr):
        field = _single_numeric_field(self.template)
        if field is None:
            return V.SumsTo([attr], self.sums_to, necessary=self.necessary)
        return V.ItemsSumTo(attr, field, self.sums_to, necessary=self.necessary)

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


def contract(template=Any, **options) -> Contract:
    """Declare an attribute: its shape, and the checks that shape implies.

    ``total = contract(float, extracted_from="source_text", range=(0, 1e6))``
    """
    return Contract(template, **options)


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
                 "__provenance__", "__purpose__", "__round__", "_cells")

    def __init__(self, *, entity, beliefs=(), ledger=None, cells=None, value=None,
                 p=0.0, contracts=None, methods=(), arguments=None, objections=(),
                 owner=None, episode=None, call_arguments=None, class_name="Thing",
                 provenance=(), purpose="", round=0):
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
# Thing
# --------------------------------------------------------------------------

class Thing(metaclass=ThingMeta):
    """A probabilistic object: attributes are beliefs, reads are consultations."""

    __beliefs__: list = []

    def __init__(self, **kwargs):
        set = object.__setattr__
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
        # constructor kwargs are assignments, and assignment is the human
        # speaking
        for name, value in kwargs.items():
            setattr(self, name, value)

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
        self.__ledger__.add(Opinion(
            belief=author.id, entity=self.__entity__, attr=name,
            value=_plain(value), p=1.0, frozen=True,
            meta={"assigned": True}))
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
            out.append(f"def {name}{inspect.signature(member)}")
        except (TypeError, ValueError):              # pragma: no cover - exotic
            out.append(f"def {name}(...)")
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
    else:
        cells = _standing(owner, entity, deriving=attr)
        value, p, provenance = None, 0.0, ()
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
        raise Unresolvable(
            cell, reason="no generative belief in __beliefs__; a "
                         "deterministic-only Thing serves frozen cells and "
                         "nothing else")

    floor = active_floor()
    policy = active_policy()
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
            opinions.append(ledger.add(Opinion(
                belief=entry.id, entity=entity, attr=attr,
                value=_plain(+got), p=~got, frozen=False,
                meta=dict(getattr(got, "meta", None) or {}, round=round_number))))
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
    return review(thing, attr, value, p, meta=meta)


def commit(thing, opinions):
    """Land a validated changeset: all of it, or none of it."""
    for opinion in opinions:
        thing.__root__.__resolved__[(opinion.entity, opinion.attr)] = opinion
    thing.__root__.__coerced__.clear()


def state_hash(thing) -> str:
    """The state an episode is a pure function of.

    Frozen attributes *and* standing resolutions, because a committed
    changeset and an interleaved assignment must both invalidate a repeated
    call -- and a committed changeset lands as ordinary, unfrozen
    opinions.
    """
    import hashlib

    from .ledger import normal_form

    cells = _standing(thing.__root__, thing.__entity__)
    material = repr(sorted(
        (attr, normal_form(opinion.value), opinion.p, opinion.frozen)
        for attr, opinion in cells.items()))
    if isinstance(thing, Cell):
        material += repr(normal_form(thing.__value__))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------

def freeze(target, attr=None, *, value=_MISSING, belief=None, p=None):
    """Pin a cell: re-record its resolving opinion with ``frozen=True``.

    Same author, same p -- pinning a model's 0.93 keeps it an honest 0.93.
    ``freeze(inv.total)`` and ``freeze(inv, "total")`` are the same verb.
    """
    if attr is not None:
        cell = getattr(target, attr)
    else:
        cell = target
    if not isinstance(cell, Cell):
        raise TypeError("freeze() wants a Thing's attribute, e.g. freeze(inv.total)")
    opinion = cell.__resolve__()
    if opinion is None and value is _MISSING:
        raise Unresolvable(cell.__cell__, reason="nothing to freeze")
    entity, name = cell.__cell__
    pinned = cell.__ledger__.add(Opinion(
        belief=belief or (opinion.belief if opinion else resident_human(cell).id),
        entity=entity, attr=name,
        value=_plain(value) if value is not _MISSING else opinion.value,
        p=float(p) if p is not None else (opinion.p if opinion else 1.0),
        frozen=True,
        meta=dict((opinion.meta if opinion else {}), pinned=True)))
    cell.__root__.__resolved__.pop((entity, name), None)
    from .debug import trace

    trace("frozen", (entity, name), pinned)
    return Cell(cell.__parent__, name, opinion=pinned, contract=cell.__contract__)


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
    from .validators import Schema

    if not isinstance(cell, Cell):
        raise TypeError("coercion applies to an attribute, not a Thing")
    key = (cell.__cell__, repr(schema))
    cached = cell.__root__.__coerced__.get(key)
    if cached is not None:
        return cached

    check = Schema(schema)
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
    return recast
