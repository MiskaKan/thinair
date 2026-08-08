"""Observability: per-operation trace boxes, and ``source(thing)`` (SPEC.md §11).

Nothing here changes what the runtime does; it only says what the runtime
did.  ``source`` is the rendering behind ``Thing.__source__``: frozen
attributes plain, believed attributes annotated with their probability and
their author, because "who said this and how sure were they" is the question
this framework exists to keep answerable.
"""

from __future__ import annotations

import contextlib
import os
import sys

__all__ = ["source", "tracing", "trace", "enabled"]

_stack: list = []
_WIDTH = 76


def enabled() -> bool:
    return bool(_stack) or os.environ.get("THINAIR_DEBUG", "").strip() in {"1", "true", "yes"}


def _stream():
    return _stack[-1] if _stack and _stack[-1] is not None else sys.stderr


@contextlib.contextmanager
def tracing(stream=None):
    """``with Thing.debug():`` -- one bordered box per operation."""
    _stack.append(stream)
    try:
        yield
    finally:
        _stack.pop()


def _box(title: str, lines) -> str:
    import textwrap

    head = f"┌─ {title} " + "─" * max(0, _WIDTH - len(title) - 4)
    body = []
    for line in lines:
        # Wrapped, not clipped: the reason a belief gave is the most useful
        # thing in the box, and it is always the part that runs long.
        wrapped = textwrap.wrap(str(line), width=_WIDTH - 2,
                                subsequent_indent="    ") or [""]
        body.extend(f"│ {piece}" for piece in wrapped)
    return "\n".join([head] + body + ["└" + "─" * (_WIDTH - 1)])


def trace(kind: str, cell, payload) -> None:
    """Emit one box.  Cheap and silent unless debugging is on.

    One box per operation, showing the route chosen, each proposal, each
    belief's verdict *with its reason*, and the resolution.  Episodes
    additionally show every action, the per-cell changeset validation, and
    the commit or rollback -- so a transcript is never something you have to
    reconstruct after the fact.
    """
    if not enabled():
        return
    entity, attr = cell
    lines = list(_LINES.get(kind, _fallback)(payload))
    print(_box(f"{_TITLES.get(kind, kind)} {entity}.{attr}", lines),
          file=_stream())


def _fallback(payload):
    return [str(payload)]


def _round(attempt):
    yield f"route: {attempt.route}"
    yield f"round: {attempt.round}"
    if attempt.value is None:
        yield "no candidate was proposed"
    else:
        yield f"proposed: {attempt.value!r}  (p={attempt.p:.2f})"
    vetoing = {belief for belief, _, _ in attempt.vetoes}
    for opinion in attempt.opinions:
        mark = "veto" if opinion.belief in vetoing else "    "
        reason = (opinion.meta or {}).get("reason")
        line = f" {mark} {_short(opinion.belief)}: p={opinion.p:.2f}"
        yield line + (f" -- {reason}" if reason else "")
    yield ("VETOED, opening the next round" if attempt.vetoes
           else f"resolved: {attempt.value!r} at p={attempt.p:.2f}")


def _resolved(payload):
    value, p, belief = payload
    yield f"resolved: {value!r}"
    yield f"       p: {p:.2f}"
    yield f"      by: {belief}"


def _frozen(opinion):
    yield f"frozen: {opinion.value!r} at p={opinion.p:.2f}"
    yield f"    by: {opinion.belief}"


def _action(payload):
    step, observation = payload
    yield f"action: {step}"
    if observation is not None:
        yield f"   saw: {_short(str(observation), 400)}"


def _changeset(payload):
    validated, refusals = payload
    for opinion in validated:
        yield f"  ok  {opinion.attr} = {opinion.value!r}"
    for refusal in refusals:
        yield f"  no  {refusal['reason']}"
    yield ("COMMITTED" if not refusals else "ROLLED BACK -- nothing was applied")


_LINES = {
    "round": _round,
    "resolved": _resolved,
    "frozen": _frozen,
    "action": _action,
    "changeset": _changeset,
}

_TITLES = {
    "round": "read",
    "resolved": "resolve",
    "frozen": "freeze",
    "action": "episode",
    "changeset": "changeset",
}


def _short(text, limit=48):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# source(thing) -- the rendering behind __source__
# --------------------------------------------------------------------------

def source(thing) -> str:
    """``total = 1249.50  # p=0.93 ← model/extract-v3``.

    Only what the ledger already holds: this never triggers a derivation, so
    printing an object cannot cost a model call.
    """
    from .beliefs import lookup

    entity = thing.__entity__
    ledger = thing.__ledger__
    lines = [f"class {type(thing).__name__}:"]
    doc = (type(thing).__doc__ or "").strip()
    if doc:
        lines.append(f'    """{doc}"""')

    resolved = dict(getattr(thing.__root__, "__resolved__", {}))
    seen = set()
    rendered = []
    for cell in ledger.cells(entity):
        attr = cell[1]
        if attr in seen:
            continue
        seen.add(attr)
        opinion = ledger.latest_frozen(entity, attr) or resolved.get(cell) \
            or ledger.latest(entity, attr)
        if opinion is None:
            continue
        rendered.append((attr, opinion))

    for attr, opinion in sorted(rendered):
        value = _literal(opinion.value)
        if opinion.frozen:
            lines.append(f"    {attr} = {value}")
        else:
            belief = lookup(opinion.belief)
            short = getattr(belief, "short", None) or opinion.belief
            lines.append(f"    {attr} = {value}  # p={opinion.p:.2f} ← {short}")

    undetermined = [a for a in type(thing).__contracts__ if a not in seen]
    for attr in sorted(undetermined):
        lines.append(f"    {attr} = ...  # not yet determined")
    if len(lines) == 1:
        lines.append("    pass")
    return "\n".join(lines)


#: a source listing is for reading; a 50KB document pasted into one is not.
MAX_LITERAL = 90


def _literal(value) -> str:
    if isinstance(value, float):
        text = f"{value:.2f}"
        return text if float(text) == value else repr(value)
    text = repr(value)
    if len(text) > MAX_LITERAL:
        return text[:MAX_LITERAL] + f"...  # {len(text)} chars"
    return text
