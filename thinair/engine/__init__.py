"""The engine layer: the *only* place a network call can happen (invariant 7).

Nothing outside ``ModelBelief`` may import this package.  ``Engine`` is a
structural protocol -- any object with a ``complete`` of this shape works, which
is how ``tests/fakes.FakeEngine`` replaces the network wholesale.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

__all__ = ["Engine", "EngineError", "ParseFailure", "complete_json",
           "extract_json", "MAX_PARSE_RETRIES"]

MAX_PARSE_RETRIES = 3


class EngineError(RuntimeError):
    """The transport could not produce a completion."""


class ParseFailure(ValueError):
    """A completion arrived but did not carry the requested JSON object."""


@runtime_checkable
class Engine(Protocol):
    def complete(self, messages: list[dict], schema: dict | None = None,
                 temperature: float = 0.2, max_tokens: int | None = None,
                 **extra: Any) -> tuple[str, dict]:
        """Return ``(text, meta)``.  ``meta`` is free-form provenance."""
        ...  # pragma: no cover - protocol


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Best-effort JSON object out of a completion; raises :class:`ParseFailure`."""
    if text is None:
        raise ParseFailure("empty completion")
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(m.group(1).strip() for m in _FENCE.finditer(text))
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ParseFailure(f"no JSON object in completion: {text[:200]!r}")


def complete_json(engine: Engine, messages: list[dict], schema: dict | None = None,
                  temperature: float = 0.2, max_tokens: int | None = None,
                  retries: int = MAX_PARSE_RETRIES, **extra: Any) -> tuple[Any, dict]:
    """Ask for JSON; on a parse failure re-ask with the failure quoted back.

    This is the engine-side escalation ladder -- distinct from the *route*
    escalation, which walks ``__beliefs__`` and lives in
    ``policy.Route`` so that ``thing.py`` never imports this package.
    """
    convo = list(messages)
    last: Exception | None = None
    meta: dict = {}
    for attempt in range(1, max(1, retries) + 1):
        text, meta = engine.complete(convo, schema=schema, temperature=temperature,
                                     max_tokens=max_tokens, **extra)
        try:
            payload = extract_json(text)
        except ParseFailure as exc:
            last = exc
            convo = convo + [
                {"role": "assistant", "content": text or ""},
                {"role": "user", "content":
                    f"That was not parseable JSON ({exc}). Reply with the JSON "
                    f"object only -- no prose, no code fence."},
            ]
            continue
        meta = dict(meta, parse_attempts=attempt)
        return payload, meta
    raise ParseFailure(str(last) if last else "unparseable completion")
