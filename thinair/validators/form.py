"""Family A -- form: the candidate alone.  Default ``necessary=True``.

Deterministic means *reproducible*, not binary (invariant 3): a graded p is
welcome, a stochastic one is not.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..beliefs import Discriminative
from ..ledger import values_equal

__all__ = ["render_template",
           "Schema", "Format", "Checksum", "Range", "Enum", "Length", "Unique",
           "Sorted", "Parses", "match_template", "template_to_json_schema",
           "luhn_ok", "isbn10_ok", "isbn13_ok", "ean13_ok", "iban_ok"]


# --------------------------------------------------------------------------
# templates are plain Python
# --------------------------------------------------------------------------

def match_template(value: Any, template: Any, path: str = "value") -> str | None:
    """``None`` if ``value`` matches ``template``, else a reason.

    Templates: types (``int``, ``str``), dicts of templates, single-element
    lists, tuples as alternatives, and literals.  ``int`` is acceptable where
    ``float`` is asked for; ``bool`` is never an ``int`` (the bool-is-not-int
    guard).
    """
    if template is Any:
        return None
    if template is None:
        return None if value is None else f"{path}: expected None, got {value!r}"
    if isinstance(template, type):
        if template is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{path}: expected float, got {type(value).__name__}"
            return None
        if template is int:
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{path}: expected int, got {type(value).__name__}"
            return None
        if template is bool:
            return (None if isinstance(value, bool)
                    else f"{path}: expected bool, got {type(value).__name__}")
        if not isinstance(value, template) or (
                isinstance(value, bool) and template is not bool):
            return f"{path}: expected {template.__name__}, got {type(value).__name__}"
        return None
    if isinstance(template, dict):
        if not isinstance(value, dict):
            return f"{path}: expected an object, got {type(value).__name__}"
        missing = [k for k in template if k not in value]
        if missing:
            return f"{path}: missing key(s) {missing}"
        extra = [k for k in value if k not in template]
        if extra:
            return f"{path}: unexpected key(s) {extra}"
        for key, sub in template.items():
            reason = match_template(value[key], sub, f"{path}.{key}")
            if reason:
                return reason
        return None
    if isinstance(template, list):
        if not isinstance(value, (list, tuple)):
            return f"{path}: expected a list, got {type(value).__name__}"
        if not template:
            return None
        if len(template) == 1:
            for i, item in enumerate(value):
                reason = match_template(item, template[0], f"{path}[{i}]")
                if reason:
                    return reason
            return None
        if len(value) != len(template):
            return f"{path}: expected {len(template)} items, got {len(value)}"
        for i, (item, sub) in enumerate(zip(value, template)):
            reason = match_template(item, sub, f"{path}[{i}]")
            if reason:
                return reason
        return None
    if isinstance(template, tuple):
        reasons = [match_template(value, sub, path) for sub in template]
        if any(r is None for r in reasons):
            return None
        return f"{path}: matched none of {len(template)} alternatives"
    if values_equal(value, template):
        return None
    return f"{path}: expected {template!r}, got {value!r}"


def render_template(template: Any) -> str:
    """Render a template the way a reader would write it.

    Lives here rather than in ``engine/prompts.py`` because the template
    language is this module's, and ``validators`` must not import ``engine``
    (invariant 7).  The prompt builder imports it from here instead.
    """
    if isinstance(template, type):
        return template.__name__
    if isinstance(template, dict):
        inner = ", ".join(f"{k!r}: {render_template(v)}" for k, v in template.items())
        return "{" + inner + "}"
    if isinstance(template, (list, tuple)):
        inner = ", ".join(render_template(v) for v in template)
        return "[" + inner + "]"
    return repr(template)


def template_to_json_schema(template: Any) -> dict:
    """The same template as a JSON Schema, for structured-output constraints."""
    if template is None:
        return {"type": "null"}
    if template is Any:
        return {}
    if isinstance(template, type):
        return {bool: {"type": "boolean"}, int: {"type": "integer"},
                float: {"type": "number"}, str: {"type": "string"},
                list: {"type": "array"}, dict: {"type": "object"},
                }.get(template, {})
    if isinstance(template, dict):
        return {"type": "object",
                "properties": {k: template_to_json_schema(v)
                               for k, v in template.items()},
                "required": list(template),
                "additionalProperties": False}
    if isinstance(template, list):
        if len(template) == 1:
            return {"type": "array", "items": template_to_json_schema(template[0])}
        return {"type": "array",
                "prefixItems": [template_to_json_schema(t) for t in template]}
    if isinstance(template, tuple):
        return {"anyOf": [template_to_json_schema(t) for t in template]}
    return {"const": template}


class Schema(Discriminative):
    """Value matches a type/shape template.

    Doubles as the structured-output contract handed to engines -- the *same*
    object drives the constraint and the post-hoc check, so they cannot drift
   .
    """

    necessary = True

    def __init__(self, template: Any, **options: Any) -> None:
        self.template = template
        super().__init__(template, **options)

    def json_schema(self) -> dict:
        return template_to_json_schema(self.template)

    def describe(self) -> str:
        return render_template(self.template)

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        reason = match_template(value, self.template)
        return 1.0 if reason is None else (0.0, reason)


# --------------------------------------------------------------------------
# formats
# --------------------------------------------------------------------------

_EMAIL = re.compile(r"^[^@\s,;]+@[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?"
                    r"(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)+$")
_URL = re.compile(r"^(https?|ftp)://[^\s/?#]+[^\s]*$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
                     r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_E164 = re.compile(r"^\+[1-9]\d{6,14}$")


def _iso_date(text: str) -> bool:
    import datetime
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text or ""):
        return False
    try:
        datetime.date.fromisoformat(text)
        return True
    except ValueError:
        return False


def _iso_datetime(text: str) -> bool:
    import datetime
    try:
        datetime.datetime.fromisoformat((text or "").replace("Z", "+00:00"))
        return "T" in (text or "") or " " in (text or "")
    except ValueError:
        return False


class Format(Discriminative):
    """Regex/parser validation of a scalar's shape."""

    necessary = True
    KINDS = ("email", "url", "iso_date", "iso_datetime", "uuid", "semver",
             "phone_e164")

    def __init__(self, kind: str, **options: Any) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown format {kind!r}; known: {self.KINDS}")
        self.kind = kind
        super().__init__(kind, **options)

    def describe(self) -> str:
        return f"a {self.kind}"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        if not isinstance(value, str):
            return None                      # not a string -- out of scope
        ok = {
            "email": lambda s: bool(_EMAIL.match(s)),
            "url": lambda s: bool(_URL.match(s)),
            "uuid": lambda s: bool(_UUID.match(s)),
            "semver": lambda s: bool(_SEMVER.match(s)),
            "phone_e164": lambda s: bool(_E164.match(s)),
            "iso_date": _iso_date,
            "iso_datetime": _iso_datetime,
        }[self.kind](value)
        return 1.0 if ok else (0.0, f"{value!r} is not a valid {self.kind}")


# --------------------------------------------------------------------------
# checksums
# --------------------------------------------------------------------------

def _digits(text: str) -> str:
    return "".join(c for c in str(text) if c.isdigit())


def luhn_ok(text: str) -> bool:
    digits = _digits(text)
    if len(digits) < 2:
        return False
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def isbn10_ok(text: str) -> bool:
    body = [c for c in str(text).upper() if c.isalnum()]
    if len(body) != 10:
        return False
    total = 0
    for i, ch in enumerate(body):
        if ch == "X" and i == 9:
            n = 10
        elif ch.isdigit():
            n = int(ch)
        else:
            return False
        total += (10 - i) * n
    return total % 11 == 0


def isbn13_ok(text: str) -> bool:
    digits = _digits(text)
    if len(digits) != 13:
        return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
    return total % 10 == 0


def ean13_ok(text: str) -> bool:
    return isbn13_ok(text)


def iban_ok(text: str) -> bool:
    body = "".join(str(text).split()).upper()
    if not (15 <= len(body) <= 34) or not body[:2].isalpha() or not body[2:4].isdigit():
        return False
    rearranged = body[4:] + body[:4]
    total = 0
    for ch in rearranged:
        if ch.isdigit():
            total = (total * 10 + int(ch)) % 97
        elif ch.isalpha():
            total = (total * 100 + (ord(ch) - 55)) % 97
        else:
            return False
    return total == 1


class Checksum(Discriminative):
    """Self-checking identifiers: a wrong digit is a hallucination tell."""

    necessary = True
    KINDS = {"luhn": luhn_ok, "isbn10": isbn10_ok, "isbn13": isbn13_ok,
             "iban": iban_ok, "ean13": ean13_ok}
    #: how many significant characters each scheme has.  A candidate outside
    #: the range is not a failing identifier, it is not an identifier at all,
    #: so the belief abstains rather than vetoing.
    WIDTHS = {"luhn": (12, 19), "isbn10": (10, 10), "isbn13": (13, 13),
              "ean13": (13, 13), "iban": (15, 34)}

    def __init__(self, kind: str, **options: Any) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown checksum {kind!r}; known: {sorted(self.KINDS)}")
        self.kind = kind
        super().__init__(kind, **options)

    def describe(self) -> str:
        return f"a {self.kind}-valid identifier"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None                      # not an identifier -- out of scope
        text = str(value)
        significant = "".join(c for c in text if c.isalnum())
        lo, hi = self.WIDTHS[self.kind]
        if not significant or not lo <= len(significant) <= hi:
            return None                      # not this scheme's shape at all
        if self.kind != "iban" and not significant.isdigit() \
                and not (self.kind == "isbn10" and significant[:-1].isdigit()):
            return None
        ok = self.KINDS[self.kind](text)
        return 1.0 if ok else (0.0, f"{value!r} fails the {self.kind} check digit")


# --------------------------------------------------------------------------
# plain bounds
# --------------------------------------------------------------------------

class Range(Discriminative):
    necessary = True

    def __init__(self, lo: float | None = None, hi: float | None = None,
                 **options: Any) -> None:
        if lo is None and hi is None:
            raise ValueError("Range needs at least one bound")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Range({lo}, {hi}) is empty")
        self.lo, self.hi = lo, hi
        super().__init__(lo, hi, **options)

    def describe(self) -> str:
        return f"between {self.lo} and {self.hi}"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        # A non-number is out of scope, not wrong: an unscoped Range sits in
        # __beliefs__ and is consulted for *every* attribute, so it
        # must stay silent about the ones it was never meant to bound.  None
        # is the scoping mechanism; Schema is what polices type.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if self.lo is not None and value < self.lo:
            return (0.0, f"{value!r} is below the declared minimum {self.lo}")
        if self.hi is not None and value > self.hi:
            return (0.0, f"{value!r} is above the declared maximum {self.hi}")
        return 1.0


class Enum(Discriminative):
    necessary = True

    def __init__(self, members, **options: Any) -> None:
        self.members = list(members)
        super().__init__(self.members, **options)

    def describe(self) -> str:
        return "one of " + ", ".join(repr(m) for m in self.members)

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        if any(values_equal(value, m) for m in self.members):
            return 1.0
        return (0.0, f"{value!r} is not one of {self.members!r}")


class Length(Discriminative):
    necessary = True

    def __init__(self, lo: int | None = None, hi: int | None = None,
                 **options: Any) -> None:
        if lo is None and hi is None:
            raise ValueError("Length needs at least one bound")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Length({lo}, {hi}) is empty")
        self.lo, self.hi = lo, hi
        super().__init__(lo, hi, **options)

    def describe(self) -> str:
        return f"length between {self.lo} and {self.hi}"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        try:
            n = len(value)
        except TypeError:
            return None                      # nothing to measure -- out of scope
        if self.lo is not None and n < self.lo:
            return (0.0, f"length {n} is below {self.lo}")
        if self.hi is not None and n > self.hi:
            return (0.0, f"length {n} is above {self.hi}")
        return 1.0


class Unique(Discriminative):
    necessary = True

    def describe(self) -> str:
        return "without duplicates"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        from ..ledger import normal_form
        if not isinstance(value, (list, tuple)):
            return None
        seen, dupes = set(), []
        for item in value:
            key = normal_form(item)
            if key in seen:
                dupes.append(item)
            seen.add(key)
        return 1.0 if not dupes else (0.0, f"duplicate item(s): {dupes!r}")


class Sorted(Discriminative):
    necessary = True

    def __init__(self, key: str | None = None, reverse: bool = False,
                 **options: Any) -> None:
        self.key, self.reverse = key, reverse
        super().__init__(key, reverse=reverse, **options)

    def describe(self) -> str:
        return "in " + ("descending" if self.reverse else "ascending") + " order"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        if not isinstance(value, (list, tuple)):
            return None
        try:
            items = [v[self.key] if self.key is not None else v for v in value]
            pairs = list(zip(items, items[1:]))
        except (KeyError, TypeError) as exc:
            return (0.0, f"cannot order these items: {exc}")
        try:
            bad = [(a, b) for a, b in pairs
                   if (a < b if self.reverse else b < a)]
        except TypeError as exc:
            return (0.0, f"items are not mutually comparable: {exc}")
        return 1.0 if not bad else (0.0, f"out of order at {bad[0]!r}")


class Parses(Discriminative):
    """JSON / Python / regex compiles without executing."""

    necessary = True
    KINDS = ("json", "python", "regex")

    def __init__(self, kind: str, **options: Any) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown parse kind {kind!r}; known: {self.KINDS}")
        self.kind = kind
        super().__init__(kind, **options)

    def describe(self) -> str:
        return f"parseable {self.kind}"

    def judge(self, value: Any, e: Any, attr: str) -> Any:
        if not isinstance(value, str):
            return None                      # no source text -- out of scope
        try:
            if self.kind == "json":
                json.loads(value)
            elif self.kind == "python":
                ast.parse(value)          # parses; never executes
            else:
                re.compile(value)
            return 1.0
        except Exception as exc:
            return (0.0, f"does not parse as {self.kind}: {exc}")
