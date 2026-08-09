"""The ledger: dumb, append-only memory (invariant 2).

An :class:`Opinion` is a Belief evaluated at a cell.  The ledger memoizes
Beliefs into Opinions and does nothing else -- zero evaluative logic, no
deletion, no ranking.  All Layer 2 math is pure functions over this module's
data -- the built slice in ``evaluate`` (SPEC.md §12), the deferred rest
(SPEC.md §14) -- and needs no migration.

This module imports nothing else from the package.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator

__all__ = [
    "arguments_id",
    "Opinion",
    "Ledger",
    "values_equal",
    "normal_form",
    "default_ledger",
    "set_default_ledger",
    "use_ledger",
    "NUMERIC_TOL",
]

NUMERIC_TOL = 1e-9


# --------------------------------------------------------------------------
# normalization: one comparator for the whole framework
# --------------------------------------------------------------------------

def _unwrap(v: Any) -> Any:
    """Reduce framework objects to plain values without importing them."""
    hook = getattr(type(v), "__thinair_unwrap__", None)
    if hook is not None:
        return _unwrap(hook(v))
    return v


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _norm_str(s: str) -> str:
    return " ".join(s.split()).casefold()


def normal_form(v: Any) -> Any:
    """A hashable canonical form of ``v``.

    Used for call identity and for dict-key comparison.  Agrees with
    :func:`values_equal` except on floats, where ``values_equal`` applies a
    tolerance that no hashable form can express.
    """
    v = _unwrap(v)
    if v is None:
        return ("none",)
    if isinstance(v, bool):
        return ("bool", v)
    if _is_number(v):
        return ("num", float(v))
    if isinstance(v, str):
        return ("str", _norm_str(v))
    if isinstance(v, bytes):
        return ("bytes", v)
    if isinstance(v, dict):
        return ("dict", tuple(sorted(
            ((normal_form(k), normal_form(x)) for k, x in v.items()),
            key=repr,
        )))
    if isinstance(v, (set, frozenset)):
        return ("set", tuple(sorted((normal_form(x) for x in v), key=repr)))
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(normal_form(x) for x in v))
    return ("obj", repr(v))


def arguments_id(args: "tuple", kwargs: "dict") -> str:
    """Call arguments as a stable, readable string.

    The one place argument rendering is defined, because two things depend on
    it agreeing with itself: ``fn`` call identity and episode
    repetition semantics.  Sorted kwargs, :func:`normal_form` values,
    so ``f(1, x=2)`` and ``f(1.0, x=2)`` name the same cell.
    """
    def render(value):
        form = normal_form(value)
        kind = form[0]
        if kind == "num":
            number = form[1]
            return repr(int(number)) if float(number).is_integer() else repr(number)
        if kind in ("str", "bool", "none", "bytes"):
            return repr(_unwrap(value)) if kind != "none" else "None"
        return repr(form[1:]) if len(form) > 1 else kind

    parts = [render(a) for a in args]
    parts += [f"{k}={render(v)}" for k, v in sorted(kwargs.items())]
    return ", ".join(parts)


def values_equal(a: Any, b: Any) -> bool:
    """The one normalizing comparator.

    Numeric tolerance ``1e-9``; casefold + whitespace-collapse for strings;
    recursive for containers; ``bool`` never equals a number (the bool-is-not-int
    guard holds here too).  Lists and tuples are comparable, because a
    JSON round-trip turns tuples into lists and persistence must be identity
    on public state.
    """
    a, b = _unwrap(a), _unwrap(b)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return abs(float(a) - float(b)) <= NUMERIC_TOL
    if isinstance(a, str) and isinstance(b, str):
        return _norm_str(a) == _norm_str(b)
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) != len(b):
            return False
        keyed_b = {normal_form(k): v for k, v in b.items()}
        for k, av in a.items():
            nk = normal_form(k)
            if nk not in keyed_b:
                return False
            if not values_equal(av, keyed_b[nk]):
                return False
        return True
    if isinstance(a, (set, frozenset)) and isinstance(b, (set, frozenset)):
        return normal_form(a) == normal_form(b)
    seq = (list, tuple)
    if isinstance(a, seq) and isinstance(b, seq):
        if len(a) != len(b):
            return False
        return all(values_equal(x, y) for x, y in zip(a, b))
    if type(a) is not type(b) and not (isinstance(a, type(b)) or isinstance(b, type(a))):
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


# --------------------------------------------------------------------------
# the one record shape (invariant 1)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Opinion:
    """Belief ``belief`` holds that ``entity``'s ``attr`` is ``value`` with ``p``.

    The single record shape for every belief in the framework -- generative,
    deterministic, human, code (invariant 1).  ``t`` is a per-ledger monotonic
    sequence number, *not* wall-clock: ordering is what matters and wall-clock
    in core data breaks replay.  ``t == 0.0`` means "unstamped"; the
    ledger stamps it on :meth:`Ledger.add`.
    """

    belief: str
    entity: str
    attr: str
    value: Any = None
    p: float = 0.0
    t: float = 0.0
    frozen: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = self.p
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise TypeError(f"p must be a real number in [0,1], got {p!r}")
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"p must be in [0.0, 1.0], got {p!r}")
        if not isinstance(self.belief, str) or not self.belief:
            raise ValueError("belief must be a non-empty durable id (invariant 6)")
        if not isinstance(self.entity, str) or not self.entity:
            raise ValueError("entity must be a non-empty id")
        if not isinstance(self.attr, str) or not self.attr:
            raise ValueError("attr must be a non-empty name")
        if not isinstance(self.frozen, bool):
            raise TypeError("frozen must be a bool")
        if not isinstance(self.meta, dict):
            raise TypeError("meta must be a dict")

    # -- the extractors: defined once, here -------------------------
    def __pos__(self) -> Any:
        return self.value

    def __invert__(self) -> float:
        return float(self.p)

    @property
    def cell(self) -> tuple[str, str]:
        return (self.entity, self.attr)

    def to_json(self) -> dict:
        return {
            "belief": self.belief,
            "entity": self.entity,
            "attr": self.attr,
            "value": self.value,
            "p": float(self.p),
            "t": float(self.t),
            "frozen": bool(self.frozen),
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Opinion":
        return cls(
            belief=d["belief"],
            entity=d["entity"],
            attr=d["attr"],
            value=d.get("value"),
            p=d.get("p", 0.0),
            t=d.get("t", 0.0),
            frozen=d.get("frozen", False),
            meta=d.get("meta") or {},
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mark = " frozen" if self.frozen else ""
        return (f"Opinion({self.belief} :: {self.entity}.{self.attr} = "
                f"{self.value!r} p={self.p:.3g} t={self.t:g}{mark})")


# a frozen dataclass with eq=True gets a generated __hash__ that chokes on the
# ``meta`` dict; identity of a record is its authorship + position in the tape.
Opinion.__hash__ = lambda self: hash((self.belief, self.entity, self.attr, self.t))


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

class Ledger:
    """Append-only memory.  No deletion API, no evaluative logic.

    This class is also the storage interface: a backend is a Ledger with a
    different home.  The kernel a backend overrides is ``add``, ``opinions``,
    ``cells``, ``beliefs``, ``next_t``, ``__len__`` and ``__iter__``;
    everything else -- ``latest``, ``latest_frozen``, ``slice_for``,
    ``extend``, ``to_json``, ``dump`` -- derives from the kernel and comes
    for free.  ``thinair.store.SqliteLedger`` is the shipped example.
    """

    VERSION = 1

    def __init__(self, opinions: Iterable[Opinion] | None = None) -> None:
        self._all: list[Opinion] = []
        self._by_cell: dict[tuple[str, str], list[Opinion]] = {}
        self._by_belief: dict[str, list[Opinion]] = {}
        self._t: float = 0.0
        for o in opinions or ():
            self.add(o)

    # -- writing ----------------------------------------------------------
    def next_t(self) -> float:
        self._t += 1.0
        return self._t

    def add(self, opinion: Opinion) -> Opinion:
        """Append one opinion.  Returns the stamped record actually stored."""
        if not isinstance(opinion, Opinion):
            raise TypeError(
                "the ledger accepts exactly one record type: Opinion (invariant 1)"
            )
        if not opinion.t:
            opinion = replace(opinion, t=self.next_t())
        else:
            self._t = max(self._t, float(opinion.t))
        self._all.append(opinion)
        self._by_cell.setdefault(opinion.cell, []).append(opinion)
        self._by_belief.setdefault(opinion.belief, []).append(opinion)
        return opinion

    def extend(self, opinions: Iterable[Opinion]) -> list[Opinion]:
        return [self.add(o) for o in opinions]

    # -- reading ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._all)

    def __iter__(self) -> Iterator[Opinion]:
        return iter(tuple(self._all))

    def opinions(self, entity: str | None = None, attr: str | None = None,
                 belief: str | None = None, frozen: bool | None = None,
                 ) -> list[Opinion]:
        """Every matching opinion, oldest first by ``t``."""
        if entity is not None and attr is not None:
            pool: Iterable[Opinion] = self._by_cell.get((entity, attr), ())
        elif belief is not None:
            pool = self._by_belief.get(belief, ())
        else:
            pool = self._all
        out = []
        for o in pool:
            if entity is not None and o.entity != entity:
                continue
            if attr is not None and o.attr != attr:
                continue
            if belief is not None and o.belief != belief:
                continue
            if frozen is not None and o.frozen is not frozen:
                continue
            out.append(o)
        out.sort(key=lambda o: o.t)
        return out

    def latest(self, entity: str, attr: str, belief: str | None = None,
               frozen: bool | None = None) -> Opinion | None:
        """The latest opinion at a cell, optionally per belief / frozen-ness."""
        got = self.opinions(entity=entity, attr=attr, belief=belief, frozen=frozen)
        return got[-1] if got else None

    def latest_frozen(self, entity: str, attr: str) -> Opinion | None:
        """Latest frozen wins."""
        return self.latest(entity, attr, frozen=True)

    def cells(self, entity: str | None = None) -> list[tuple[str, str]]:
        return [c for c in self._by_cell if entity is None or c[0] == entity]

    def beliefs(self) -> list[str]:
        return list(self._by_belief)

    def slice_for(self, entity: str) -> list[Opinion]:
        return self.opinions(entity=entity)

    # -- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        return {"thinair": self.VERSION,
                "kind": "ledger",
                "opinions": [o.to_json() for o in self]}

    @classmethod
    def from_json(cls, blob: dict) -> "Ledger":
        if blob.get("kind") != "ledger":
            raise ValueError("not a thinair ledger blob")
        version = blob.get("thinair")
        if version != cls.VERSION:
            raise ValueError(f"unsupported ledger version {version!r}")
        return cls(Opinion.from_json(d) for d in blob.get("opinions", ()))

    def dump(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=1, sort_keys=False)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Ledger":
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Ledger {len(self._all)} opinions, {len(self._by_belief)} beliefs>"


# --------------------------------------------------------------------------
# the process-wide default ledger
# --------------------------------------------------------------------------

_default = Ledger()


def default_ledger() -> Ledger:
    return _default


def set_default_ledger(ledger: Ledger) -> Ledger:
    global _default
    if not isinstance(ledger, Ledger):
        raise TypeError("expected a Ledger")
    previous, _default = _default, ledger
    return previous


@contextlib.contextmanager
def use_ledger(ledger: Ledger) -> Iterator[Ledger]:
    """Scoped default ledger -- tests always inject their own."""
    previous = set_default_ledger(ledger)
    try:
        yield ledger
    finally:
        set_default_ledger(previous)
