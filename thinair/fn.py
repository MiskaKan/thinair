"""Functions as cells (SPEC.md §10).

**A function call is an attribute read of a call-shaped entity.**
``net_total(100, 0.24)`` names the cell
``(entity="fn:net_total(100, 0.24)", attr="result")``.  Layer 1 needs nothing
new for this -- entities are ids -- so everything the framework does for
attributes applies to calls verbatim:

===========================  ==========================================
consulting beliefs           calling implementations
frozen opinions              memoization
frozen call-cells            mocks and fixtures
===========================  ==========================================

There is no mocking framework here because the ledger already is one.
"""

from __future__ import annotations

import functools
import inspect
import itertools
from typing import Any

from .beliefs import CodeBelief, human, model
from .ledger import Opinion, arguments_id, default_ledger
from .thing import Contract, Thing

__all__ = ["fn", "freeze_call", "call_id", "FunctionCell", "Fn"]

_impure_sequence = itertools.count(1)


def call_id(qualname: str, args=(), kwargs=None) -> str:
    """``fn:net_total(100, 0.24)`` -- implementation-free by design.

    No version, no belief id: code results and model opinions about the same
    call land on the same cell and stay comparable forever.  Arguments must
    be JSON-able; a Thing contributes its entity id and a hash of its frozen
    state instead of its value.
    """
    args = tuple(_argument(a) for a in args)
    kwargs = {k: _argument(v) for k, v in (kwargs or {}).items()}
    return f"fn:{qualname}({arguments_id(args, kwargs)})"


def _argument(value):
    from .thing import state_hash

    if isinstance(value, Thing):
        return f"<{value.__entity__}@{state_hash(value)}>"
    _jsonable(value)
    return value


def _jsonable(value, path="argument"):
    import json

    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{path} {value!r} is not JSON-able, so it cannot name a cell: "
            f"{exc}") from None


# --------------------------------------------------------------------------
# the call-shaped entity
# --------------------------------------------------------------------------

class FunctionCell(Thing):
    """The entity a call names.  A Thing, because a cell is a cell."""

    def __init__(self, spec, entity, arguments):
        set = object.__setattr__
        # What the panel is being asked about is the *function*, not this
        # wrapper, so the snapshot speaks in the function's own terms.
        set(self, "__purpose__", _purpose(spec))
        set(self, "__class_name__", "function call")
        set(self, "__entity__", entity)
        set(self, "__ledger__", spec.ledger())
        set(self, "__root__", self)
        set(self, "__resolved__", {})
        set(self, "__coerced__", {})
        set(self, "__beliefs__", spec.beliefs())
        set(self, "__call_arguments__", arguments)
        set(self, "__spec__", spec)


class _Signature(Contract):
    """The ``returns=`` declaration, as the contract it is."""


# --------------------------------------------------------------------------
# @fn
# --------------------------------------------------------------------------

class Fn:
    """A decorated function: code, an imagined function, or both.

    * **body + ``pure=True``** -- the code is a deterministic generative
      belief ``code:<qualname>@<source-hash>``; its result is a frozen
      opinion on the call-cell.  Memoization *is* the frozen short-circuit;
      cache invalidation *is* re-freezing.
    * **no body** -- an imagined function: signature, docstring and
      ``returns`` drive the standard pipeline.  The result is a child Thing
      ``(v, p)``, never frozen.
    * **both** -- competing implementations over the same cells, side by side
      in the ledger for anyone (or Layer 2) to compare.
    """

    def __init__(self, func, *, returns=Any, pure=True, models=None,
                 ledger=None, beliefs=(), fallback=False):
        functools.update_wrapper(self, func)
        self.func = func
        self.qualname = getattr(func, "__qualname__", func.__name__)
        self.signature = inspect.signature(func)
        self.returns = returns
        self.pure = bool(pure)
        self.fallback = bool(fallback)
        self.bodiless = _is_bodiless(func)
        self.contract = Contract(returns, doc=(func.__doc__ or "").strip() or None)
        self.code = None if self.bodiless else CodeBelief(func)
        self._models = list(models) if models is not None else None
        self._ledger = ledger
        self._extra = list(beliefs)

    # -- configuration -----------------------------------------------------
    def ledger(self):
        from .thing import _ledgers

        if self._ledger is not None:
            return self._ledger
        return _ledgers[-1] if _ledgers else default_ledger()

    def beliefs(self):
        """The panel a call-cell consults.

        A bodiless ``@fn`` needs a proposer, so one is supplied unless the
        user named their own; a coded one does not, which is why a pure
        function with a body reaches no engine at all.
        """
        panel = list(self._models) if self._models is not None else []
        if self.bodiless and not panel:
            panel = [model()]
        panel += self._extra
        panel += self.contract.beliefs("result")
        panel.append(human("code"))
        return panel

    # -- the call ----------------------------------------------------------
    def cell(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        identity = call_id(self.qualname, bound.args, bound.kwargs)
        if not self.pure:
            # A sensor reading is never the same reading twice.
            identity = f"{identity}#{next(_impure_sequence)}"
        return identity, bound

    def _serves(self, pinned) -> bool:
        """Whether a frozen opinion still answers for this implementation.

        Editing the body changes the source hash, so the cell's frozen
        opinion is now some *other* code belief's -- so the next
        call re-freezes rather than serving a result this code never
        produced.  Anything not authored by code -- a fixture, a human, a
        pinned model answer -- outranks the body, deliberately.
        """
        if self.code is None or not pinned.belief.startswith("code:"):
            return True
        return pinned.belief == self.code.id

    def __call__(self, *args, **kwargs):
        entity, bound = self.cell(args, kwargs)
        site = FunctionCell(self, entity, dict(bound.arguments))
        ledger = site.__ledger__

        pinned = ledger.latest_frozen(entity, "result")
        if pinned is not None and self._serves(pinned):
            # memoization is the frozen short-circuit
            return _cell(site, pinned)

        if self.code is not None:
            try:
                value = self.func(*bound.args, **bound.kwargs)
            except Exception:
                if not (self.fallback and self._models is not False):
                    raise
            else:
                return _cell(site, ledger.add(Opinion(
                    belief=self.code.id, entity=entity, attr="result",
                    value=value, p=1.0, frozen=True,
                    meta=_meta(self, bound))))

        # bodiless (or code raised and a fallback was asked for): the model
        # serves the cell through the ordinary pipeline, contract-checked.
        from .thing import _read

        return _read(site, "result", self.contract, force=True)

    def __get__(self, instance, owner=None):          # methods keep working
        if instance is None:
            return self
        return functools.partial(self, instance)

    def __repr__(self) -> str:                        # pragma: no cover
        kind = "imagined" if self.bodiless else ("pure" if self.pure else "sensor")
        return f"<fn {self.qualname} ({kind})>"


def _purpose(spec) -> str:
    lines = [f"def {spec.qualname}{spec.signature}"]
    doc = (spec.func.__doc__ or "").strip()
    if doc:
        lines.append(doc)
    return "\n".join(lines)


def _meta(spec, bound):
    from .thing import references

    meta = {"call": spec.qualname, "args": list(bound.args)}
    refs = references((bound.args, bound.kwargs))
    if refs:
        meta["refs"] = refs
    if not spec.pure:
        meta["observed"] = True                      # a sensor reading
    return meta


def _cell(site, opinion):
    from .thing import Cell

    return Cell(site, "result", opinion=opinion, entity=site.__entity__)


def _is_bodiless(func) -> bool:
    """``...`` or a docstring and nothing else -- an imagined function."""
    import ast
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        # No source -- a REPL, an exec, a stripped .pyc.  Falling back to
        # "assume it has a body" would silently turn an imagined function
        # into a coded one that freezes None at p=1.0, so read the bytecode
        # instead: a bodiless function returns the constant None and does
        # nothing else.
        return _returns_none_immediately(func)
    body = tree.body[0].body
    statements = [
        node for node in body
        if not (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str))       # a docstring
    ]
    if not statements:
        return True
    return all(
        isinstance(node, ast.Pass)
        or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis)
        for node in statements)


def _returns_none_immediately(func) -> bool:
    import dis

    ops = [(i.opname, i.argval) for i in dis.get_instructions(func)
           if i.opname not in ("RESUME", "NOP", "CACHE")]
    if len(ops) == 1 and ops[0] == ("RETURN_CONST", None):
        return True                                  # 3.12+
    return [op for op, _ in ops] == ["LOAD_CONST", "RETURN_VALUE"] \
        and ops[0][1] is None                        # 3.11


def fn(_func=None, *, returns=Any, pure=True, models=None, ledger=None,
       beliefs=(), fallback=False):
    """Turn a function into a cell.

        @fn(returns=float, pure=True)
        def net_total(gross: float, vat: float) -> float:
            '''Total with VAT removed.'''
            return gross / (1 + vat)

    ``pure=False`` declares a **sensor**: each invocation mints a fresh cell
    and its result is tagged ``observed``, never served from cache.
    Purity is declared, never assumed.
    """
    def decorate(func):
        return Fn(func, returns=returns, pure=pure, models=models,
                  ledger=ledger, beliefs=beliefs, fallback=fallback)

    return decorate(_func) if _func is not None else decorate


# --------------------------------------------------------------------------
# fixtures and mocks are frozen opinions
# --------------------------------------------------------------------------

def freeze_call(func, args=(), kwargs=None, *, value, p=1.0, belief=None,
                ledger=None):
    """Install a fixture: ``freeze_call(net_total, (100, 0.24), value=80.645)``.

    A fixture *is* a frozen opinion on a call-cell; the next call serves it.
    A mock is the same thing installed in a scoped test ledger, gone when the
    ledger is discarded.
    """
    spec = func if isinstance(func, Fn) else None
    qualname = spec.qualname if spec else getattr(func, "__qualname__", str(func))
    if spec is not None:
        entity, _ = spec.cell(tuple(args), dict(kwargs or {}))
        target = ledger if ledger is not None else spec.ledger()
    else:
        entity = call_id(qualname, tuple(args), dict(kwargs or {}))
        target = ledger if ledger is not None else default_ledger()
    return target.add(Opinion(
        belief=belief or f"fixture:{qualname}", entity=entity, attr="result",
        value=value, p=float(p), frozen=True, meta={"fixture": True}))
