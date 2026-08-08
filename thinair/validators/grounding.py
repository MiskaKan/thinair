"""Family B -- grounding: the candidate against the entity's own state.

These are the anti-hallucination family: a number that is not in the input, a
quote that does not appear verbatim, a span whose offsets do not line up.  They
fail in ways unrelated to family A, which is what makes cross-family agreement
worth something.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

from ..beliefs import Discriminative, value_at
from ..ledger import values_equal

__all__ = ["Verbatim", "Normalized", "Fuzzy", "TokenSubset", "QuoteIntegrity",
           "SpanValid", "FrozenConsistent", "NonEcho", "normalize_text",
           "numbers_in", "entities_in"]

_PUNCT = dict.fromkeys(
    i for i in range(0x110000)
    if unicodedata.category(chr(i)).startswith("P")
    or unicodedata.category(chr(i)).startswith("S"))


def normalize_text(text: str) -> str:
    """Casefold + NFKC + strip punctuation + collapse whitespace."""
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(folded.translate(_PUNCT).split())


# Two shapes, tried in order: space-grouped thousands ("1 249,50", including
# the non-breaking and narrow spaces invoices actually use), then an ordinary
# run of digits with optional grouping and decimals.  Spaces are admitted only
# between groups of exactly three digits -- a looser class swallows
# "999.00 250.50" whole and reads it as one unparseable number.
_NUMBER = re.compile(
    r"[-+]?\d{1,3}(?:[\u0020\u00a0\u202f]\d{3})+(?:[.,]\d+)?"
    r"|[-+]?\d+(?:[.,]\d{3})*(?:[.,]\d+)?"
)


def numbers_in(text: str) -> list[float]:
    """Every number in ``text``, tolerant of thousands separators."""
    out = []
    for raw in _NUMBER.findall(str(text)):
        body = "".join(c for c in raw if c not in " \u00a0\u202f")
        if "," in body and "." in body:
            body = (body.replace(",", "") if body.rfind(".") > body.rfind(",")
                    else body.replace(".", "").replace(",", "."))
        elif body.count(",") == 1 and len(body.split(",")[-1]) in (1, 2):
            body = body.replace(",", ".")
        else:
            body = body.replace(",", "")
        body = body.rstrip(".")
        try:
            out.append(float(body))
        except ValueError:
            continue
    return out


_ENTITY = re.compile(r"\b[A-Z][\w&'\-]*(?:\s+[A-Z][\w&'\-]*)*\b")


def entities_in(text: str) -> list[str]:
    return [m.group(0) for m in _ENTITY.finditer(str(text))]


def _flatten(value: Any) -> str:
    """Every scalar in a candidate, as one searchable string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_flatten(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return f"{value!r} {int(value)}"
    return str(value)


class _Grounded(Discriminative):
    """Shared plumbing: a source attribute read inertly off the snapshot."""

    def __init__(self, source_attr: str, *rest: Any, **options: Any) -> None:
        self.source_attr = source_attr
        super().__init__(source_attr, *rest, **options)

    def source(self, e: Any) -> str | None:
        got = value_at(e, self.source_attr)
        return None if got is None else _flatten(got)

    def describe(self) -> str:
        return f"grounded in {self.source_attr}"


class Verbatim(_Grounded):
    """``v in thing.<source_attr>``.  p ∈ {0, 1}."""

    necessary = True

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        text = self.source(e)
        if text is None:
            return None                      # no source -- nothing to say
        needle = _flatten(value).strip()
        if not needle:
            return (0.0, "empty candidate cannot be grounded")
        if needle in text:
            return 1.0
        return (0.0, f"{needle[:80]!r} does not appear verbatim in "
                     f"{self.source_attr}")


class Normalized(_Grounded):
    """Found after casefold + whitespace/punctuation normalization."""

    necessary = True

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        text = self.source(e)
        if text is None:
            return None
        needle = normalize_text(_flatten(value))
        if not needle:
            return (0.0, "empty candidate cannot be grounded")
        if needle in normalize_text(text):
            return 1.0
        return (0.0, f"{needle[:80]!r} is absent from a normalized "
                     f"{self.source_attr}")


class Fuzzy(_Grounded):
    """Graded p from the best-matching source span.  Never ``necessary``."""

    necessary = False

    def __init__(self, source_attr: str, floor: float = 0.6, **options: Any) -> None:
        options.pop("necessary", None)       # deterministic, but never a veto
        self.floor = float(floor)
        super().__init__(source_attr, floor=floor, **options)

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        text = self.source(e)
        if text is None:
            return None
        needle = _flatten(value)
        if not needle:
            return (0.0, "empty candidate")
        matcher = difflib.SequenceMatcher(None, needle.casefold(), text.casefold())
        blocks = matcher.get_matching_blocks()
        best = max((b.size for b in blocks), default=0)
        ratio = best / max(1, len(needle))
        if ratio >= self.floor:
            return round(min(1.0, ratio), 4)
        return (round(ratio, 4),
                f"best source overlap {ratio:.2f} is below floor {self.floor}")


class TokenSubset(_Grounded):
    """Every number (and optionally entity) in ``v`` appears in the source.

    ``necessary`` for numbers by default: a number absent from the input is the
    classic hallucination tell.
    """

    necessary = True

    def __init__(self, source_attr: str, kinds=("numbers",), **options: Any) -> None:
        self.kinds = tuple(kinds)
        super().__init__(source_attr, kinds=self.kinds, **options)

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        text = self.source(e)
        if text is None:
            return None
        rendered = _flatten(value)
        checks, missing = 0, []
        if "numbers" in self.kinds:
            source_numbers = numbers_in(text)
            for number in numbers_in(rendered):
                checks += 1
                if not any(values_equal(number, s) for s in source_numbers):
                    missing.append(number)
        if "entities" in self.kinds:
            source_normal = normalize_text(text)
            for entity in entities_in(rendered):
                checks += 1
                if normalize_text(entity) not in source_normal:
                    missing.append(entity)
        if checks == 0:
            return None                      # nothing of this kind to check
        if not missing:
            return 1.0
        return (round(1.0 - len(missing) / checks, 4),
                f"absent from {self.source_attr}: "
                + ", ".join(repr(m) for m in missing[:5]))


_QUOTED = re.compile(r'"([^"]{3,})"|“([^”]{3,})”')


class QuoteIntegrity(_Grounded):
    """Quoted spans match the source verbatim even when the rest paraphrases."""

    necessary = True

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        text = self.source(e)
        if text is None:
            return None
        rendered = _flatten(value)
        quotes = [a or b for a, b in _QUOTED.findall(rendered)]
        if not quotes:
            return None                      # nothing quoted -- no opinion
        bad = [q for q in quotes if q not in text]
        if not bad:
            return 1.0
        return (round(1.0 - len(bad) / len(quotes), 4),
                f"quoted but not in {self.source_attr}: {bad[0][:80]!r}")


class SpanValid(_Grounded):
    """Candidate ``{start, end, text}`` satisfies ``source[start:end] == text``."""

    necessary = True

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        source = value_at(e, self.source_attr)
        if not isinstance(source, str):
            return None
        spans = value if isinstance(value, list) else [value]
        if not all(isinstance(s, dict) and {"start", "end", "text"} <= set(s)
                   for s in spans):
            return None                      # not span-shaped -- no opinion
        for span in spans:
            try:
                start, end = int(span["start"]), int(span["end"])
            except (TypeError, ValueError):
                return (0.0, f"non-integer offsets in {span!r}")
            if not 0 <= start <= end <= len(source):
                return (0.0, f"offsets [{start}:{end}] fall outside "
                             f"{self.source_attr} of length {len(source)}")
            if source[start:end] != span["text"]:
                return (0.0, f"{self.source_attr}[{start}:{end}] is "
                             f"{source[start:end]!r}, not {span['text']!r}")
        return 1.0


class FrozenConsistent(Discriminative):
    """``v`` does not contradict the cell's latest frozen opinion."""

    necessary = True

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        ledger = getattr(e, "__ledger__", None)
        entity = getattr(e, "__entity__", None)
        if ledger is None or entity is None:
            return None
        pinned = ledger.latest_frozen(entity, attr)
        if pinned is None:
            return None                      # nothing pinned -- no opinion
        if values_equal(value, pinned.value):
            return 1.0
        return (0.0, f"contradicts the frozen {pinned.value!r} "
                     f"(per {pinned.belief})")


class NonEcho(Discriminative):
    """The candidate is not a near-copy of the entity's own carried value.

    Default for elaboration-shaped calls, attached ``necessary=True`` there: a
    model that answers a deepening request by echoing its input is vetoed and
    re-proposed with the reason -- progress is enforced, not hoped for.
    """

    necessary = True

    def __init__(self, max_overlap: float = 0.8, **options: Any) -> None:
        self.max_overlap = float(max_overlap)
        super().__init__(max_overlap=self.max_overlap, **options)

    def describe(self) -> str:
        return f"not a near-copy of the input (overlap ≤ {self.max_overlap})"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        carried = getattr(e, "__value__", None)
        if carried is None:
            return None
        before = normalize_text(_flatten(carried)).split()
        after = normalize_text(_flatten(value)).split()
        if not before or not after:
            return None
        overlap = _token_overlap(before, after)
        if overlap <= self.max_overlap:
            return round(1.0 - overlap, 4)
        return (round(max(0.0, 1.0 - overlap), 4),
                f"{overlap:.0%} of the input is echoed back; this call must add "
                f"something (overlap must be ≤ {self.max_overlap:.0%})")


def _token_overlap(before: list[str], after: list[str]) -> float:
    """Fraction of the *new* text that is merely the old text repeated."""
    from collections import Counter
    old, new = Counter(before), Counter(after)
    shared = sum((old & new).values())
    return shared / max(1, len(after))
