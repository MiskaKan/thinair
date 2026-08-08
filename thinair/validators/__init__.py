"""The validator library: deterministic discriminative Beliefs.

"Validator" is the colloquial name; there is no ``Validator`` class and no
second contract.  Every name re-exported here is an ordinary
:class:`~thinair.beliefs.Belief` — a pure function of a sealed snapshot,
consulted by being called.  The library is organized by *what evidence each
check consumes*, because families that fail in unrelated ways are what make
cross-family agreement worth anything:

======  ==================================  =================
family  evidence                            default ``necessary``
======  ==================================  =================
A form  the candidate alone                 True
B grou  candidate + the entity's own state  mostly True
C cons  several cells at once               True
D refe  a pinned vendored table             False
E exec  candidate + a runtime               True (``Executes`` opt-in)
======  ==================================  =================

Every default is overridable at the attachment site —
``Verbatim("source_text", necessary=False)`` demotes a veto to a
measurement, and nothing else about the belief changes.

The whole library contains **zero model calls** (invariant 7); that is
precisely what its independence is made of.
"""

from __future__ import annotations

from .consistency import (
    Conservation,
    FunctionalDependency,
    ItemsSumTo,
    MutuallyExclusive,
    Recompute,
    Relation,
    SumsTo,
    TemporalOrder,
)
from .executable import (
    ALLOW_EXEC_ENV,
    Calculator,
    Executes,
    ExecutionRefused,
    PassesTests,
    RegexBehavior,
    RoundTrip,
    calculate,
)
from .form import (
    Checksum,
    Enum,
    Format,
    Length,
    Parses,
    Range,
    Schema,
    Sorted,
    Unique,
    match_template,
    render_template,
    template_to_json_schema,
)
from .grounding import (
    FrozenConsistent,
    Fuzzy,
    NonEcho,
    Normalized,
    QuoteIntegrity,
    SpanValid,
    TokenSubset,
    Verbatim,
)
from .reference import CalendarFact, IsoCountry, IsoCurrency, Timezone

#: family letter -> the classes it contains, in canonical order.
#: This is the registry: a plain mapping anyone can read, iterate, or
#: table-test over.  Adding a validator means adding it here, and the
#: conformance test in ``tests/test_validators_registry.py`` then covers it
#: with no new test code.
FAMILIES = {
    "A": (Schema, Format, Checksum, Range, Enum, Length, Unique, Sorted, Parses),
    "B": (
        Verbatim,
        Normalized,
        Fuzzy,
        TokenSubset,
        QuoteIntegrity,
        SpanValid,
        FrozenConsistent,
        NonEcho,
    ),
    "C": (
        SumsTo,
        ItemsSumTo,
        Recompute,
        TemporalOrder,
        Conservation,
        FunctionalDependency,
        MutuallyExclusive,
    ),
    "D": (IsoCountry, IsoCurrency, Timezone, CalendarFact),
    "E": (Executes, PassesTests, RoundTrip, Calculator, RegexBehavior),
}

FAMILY_NAMES = {
    "A": "form",
    "B": "grounding",
    "C": "consistency",
    "D": "reference",
    "E": "executable",
}

#: validators that must never appear in a default belief list, and why.
#: ``Executes`` runs model-generated code; an interactive human blocks.  The
#: read pipeline consults this when it assembles defaults.
NEVER_DEFAULT = {
    Executes: "runs model-generated code; opt in explicitly",
}


def validators():
    """Every validator class, flattened, in family order."""
    return [cls for letter in sorted(FAMILIES) for cls in FAMILIES[letter]]


def family_of(cls):
    """The family letter a validator class belongs to, or ``None``."""
    for letter, members in FAMILIES.items():
        if cls in members:
            return letter
    return None


__all__ = [
    # family A — form
    "Schema", "Format", "Checksum", "Range", "Enum", "Length", "Unique",
    "Sorted", "Parses", "match_template", "template_to_json_schema",
    "render_template",
    # family B — grounding
    "Verbatim", "Normalized", "Fuzzy", "TokenSubset", "QuoteIntegrity",
    "SpanValid", "FrozenConsistent", "NonEcho",
    # family C — internal consistency
    "Relation", "SumsTo", "ItemsSumTo", "Recompute", "TemporalOrder",
    "Conservation", "FunctionalDependency", "MutuallyExclusive",
    # family D — pinned reference tables
    "IsoCountry", "IsoCurrency", "Timezone", "CalendarFact",
    # family E — executable
    "Executes", "PassesTests", "RoundTrip", "Calculator", "RegexBehavior",
    "ExecutionRefused", "calculate", "ALLOW_EXEC_ENV",
    # the registry itself
    "FAMILIES", "FAMILY_NAMES", "NEVER_DEFAULT", "validators", "family_of",
]
