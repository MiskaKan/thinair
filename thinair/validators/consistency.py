"""Family C -- internal consistency: several cells at once.

These judge *relations*, so they emit opinions about **virtual attributes** --
same record shape, one ledger (invariant 1).  A relation also answers at each
real attribute it relates, pulling that attribute's candidate from the proposer,
which is how a relation gets to veto a proposal rather than merely note it.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..beliefs import Belief, Judgment, as_judgment, value_at
from ..ledger import normal_form, values_equal

__all__ = ["RelationBelief", "SumsToBelief", "ItemsSumToBelief", "RecomputeBelief", "TemporalOrderBelief",
           "ConservationBelief", "FunctionalDependencyBelief", "MutuallyExclusiveBelief"]


class RelationBelief(Belief):
    """A judgment about several cells.

    Answers at its own virtual attribute (``…name(args)``) using resolved state,
    and at each related real attribute using that attribute's live candidate.
    """

    necessary = True

    def cells(self) -> tuple[str, ...]:
        raise NotImplementedError

    def check(self, values: dict) -> Any:
        """Return ``None`` (undecidable), a probability, or ``(p, reason)``."""
        raise NotImplementedError

    @property
    def virtual(self) -> str:
        return "…" + self.id

    def _gather(self, e: Any) -> dict:
        return {attr: value_at(e, attr) for attr in self.cells()}

    def __call__(self, e: Any, attr: str) -> tuple | None:
        related = self.cells()
        if attr == self.virtual:
            values = self._gather(e)
            verdict = self.check(values)
            return self._as(verdict, self._holds(verdict))
        if attr not in related:
            return None
        panel = getattr(e, "__beliefs__", ())
        if not panel:
            return None
        got = as_judgment(panel[0](e, attr))     # ask the proposer -- a plain call
        if got is None:
            return None
        values = self._gather(e)
        values[attr] = got[0]
        return self._as(self.check(values), got[0])

    @staticmethod
    def _holds(verdict: Any) -> Any:
        if verdict is None:
            return None
        p = verdict[0] if isinstance(verdict, tuple) else verdict
        return bool(float(p) >= 0.5)

    def _as(self, verdict: Any, value: Any) -> tuple | None:
        if verdict is None:
            return None
        if isinstance(verdict, tuple):
            p, reason = float(verdict[0]), verdict[1]
        else:
            p, reason = float(verdict), None
        meta = {"judged": self.id, "relates": list(self.cells())}
        if reason:
            meta["reason"] = reason
        return Judgment(value, p, meta)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class SumsToBelief(RelationBelief):
    """The parts add up to the total."""

    def __init__(self, parts_attrs: Iterable[str], total_attr: str,
                 tol: float = 1e-6, **options: Any) -> None:
        self.parts_attrs = tuple(parts_attrs)
        self.total_attr = total_attr
        self.tol = float(tol)
        super().__init__(self.parts_attrs, total_attr, tol=self.tol, **options)

    def cells(self) -> tuple[str, ...]:
        return self.parts_attrs + (self.total_attr,)

    def check(self, values: dict) -> Any:
        parts = [_number(values.get(a)) for a in self.parts_attrs]
        total = _number(values.get(self.total_attr))
        if total is None or any(p is None for p in parts):
            return None
        got = sum(parts)
        if abs(got - total) <= self.tol:
            return 1.0
        return (0.0, f"{' + '.join(self.parts_attrs)} = {got!r} but "
                     f"{self.total_attr} = {total!r}")


class ItemsSumToBelief(RelationBelief):
    """A list attribute's ``field`` values add up to another attribute."""

    def __init__(self, items_attr: str, field: str, total_attr: str,
                 tol: float = 1e-6, **options: Any) -> None:
        self.items_attr = items_attr
        self.field = field
        self.total_attr = total_attr
        self.tol = float(tol)
        super().__init__(items_attr, field, total_attr, tol=self.tol, **options)

    def cells(self) -> tuple[str, ...]:
        return (self.items_attr, self.total_attr)

    def check(self, values: dict) -> Any:
        items, total = values.get(self.items_attr), _number(values.get(self.total_attr))
        if items is None or total is None or not isinstance(items, (list, tuple)):
            return None
        parts = []
        for item in items:
            if not isinstance(item, dict) or self.field not in item:
                return (0.0, f"an item of {self.items_attr} has no "
                             f"{self.field!r}: {item!r}")
            number = _number(item[self.field])
            if number is None:
                return (0.0, f"{self.field!r} is not a number in {item!r}")
            parts.append(number)
        got = sum(parts)
        if abs(got - total) <= self.tol:
            return 1.0
        return (0.0, f"the {len(parts)} {self.items_attr} {self.field}s sum to "
                     f"{got!r}, but {self.total_attr} is {total!r}")


class RecomputeBelief(RelationBelief):
    """RecomputeBelief ``target`` from ``inputs`` with a pure function."""

    def __init__(self, target: str, fn: Callable, inputs: Iterable[str],
                 tol: float = 1e-6, **options: Any) -> None:
        self.target = target
        self.fn = fn
        self.inputs = tuple(inputs)
        self.tol = float(tol)
        super().__init__(target, fn, self.inputs, tol=self.tol, **options)

    def cells(self) -> tuple[str, ...]:
        return self.inputs + (self.target,)

    def check(self, values: dict) -> Any:
        arguments = [values.get(a) for a in self.inputs]
        if any(a is None for a in arguments) or values.get(self.target) is None:
            return None
        try:
            expected = self.fn(*arguments)
        except Exception as exc:
            return (0.0, f"recomputation raised {type(exc).__name__}: {exc}")
        got = values[self.target]
        both_numeric = _number(expected) is not None and _number(got) is not None
        if both_numeric:
            if abs(float(expected) - float(got)) <= self.tol:
                return 1.0
        elif values_equal(expected, got):
            return 1.0
        return (0.0, f"{self.target} is {got!r}, but recomputing from "
                     f"{', '.join(self.inputs)} gives {expected!r}")


class TemporalOrderBelief(RelationBelief):
    """``before`` is not after ``after``."""

    def __init__(self, before: str, after: str, **options: Any) -> None:
        self.before, self.after = before, after
        super().__init__(before, after, **options)

    def cells(self) -> tuple[str, ...]:
        return (self.before, self.after)

    def check(self, values: dict) -> Any:
        a, b = values.get(self.before), values.get(self.after)
        if a is None or b is None:
            return None
        ka, kb = _as_instant(a), _as_instant(b)
        if ka is None or kb is None:
            return (0.0, f"cannot order {a!r} and {b!r} as instants")
        if ka <= kb:
            return 1.0
        return (0.0, f"{self.before} ({a!r}) is after {self.after} ({b!r})")


def _as_instant(value: Any):
    import datetime
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):020.6f}"
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        for parse in (datetime.datetime.fromisoformat, datetime.date.fromisoformat):
            try:
                return parse(text).isoformat()
            except ValueError:
                continue
    return None


class ConservationBelief(RelationBelief):
    """``inflow - outflow == delta``: nothing appears from nowhere."""

    def __init__(self, inflow: str, outflow: str, delta: str,
                 tol: float = 1e-6, **options: Any) -> None:
        self.inflow, self.outflow, self.delta = inflow, outflow, delta
        self.tol = float(tol)
        super().__init__(inflow, outflow, delta, tol=self.tol, **options)

    def cells(self) -> tuple[str, ...]:
        return (self.inflow, self.outflow, self.delta)

    def check(self, values: dict) -> Any:
        numbers = {k: _number(values.get(k)) for k in self.cells()}
        if any(v is None for v in numbers.values()):
            return None
        got = numbers[self.inflow] - numbers[self.outflow]
        if abs(got - numbers[self.delta]) <= self.tol:
            return 1.0
        return (0.0, f"{self.inflow} - {self.outflow} = {got!r} but "
                     f"{self.delta} = {numbers[self.delta]!r}")


class FunctionalDependencyBelief(RelationBelief):
    """Within a list of records, ``key`` determines ``value``."""

    def __init__(self, key: str, value: str, over: str | None = None,
                 **options: Any) -> None:
        self.key, self.value_field, self.over = key, value, over
        super().__init__(key, value, over, **options)

    def cells(self) -> tuple[str, ...]:
        return (self.over,) if self.over else ()

    def _records(self, values: dict, e: Any = None) -> Any:
        if self.over:
            return values.get(self.over)
        return None

    def check(self, values: dict) -> Any:
        records = self._records(values)
        if not isinstance(records, (list, tuple)):
            return None
        seen: dict = {}
        for record in records:
            if not isinstance(record, dict) or self.key not in record:
                return None
            key = normal_form(record[self.key])
            got = record.get(self.value_field)
            if key in seen and not values_equal(seen[key], got):
                return (0.0, f"{self.key}={record[self.key]!r} maps to both "
                             f"{seen[key]!r} and {got!r}")
            seen[key] = got
        return 1.0


class MutuallyExclusiveBelief(RelationBelief):
    """At most one of these flags is true."""

    def __init__(self, flags: Iterable[str], **options: Any) -> None:
        self.flags = tuple(flags)
        super().__init__(self.flags, **options)

    def cells(self) -> tuple[str, ...]:
        return self.flags

    def check(self, values: dict) -> Any:
        known = {k: values.get(k) for k in self.flags if values.get(k) is not None}
        if len(known) < 2:
            return None
        on = [k for k, v in known.items() if bool(v)]
        if len(on) <= 1:
            return 1.0
        return (0.0, f"mutually exclusive flags are all set: {', '.join(on)}")

