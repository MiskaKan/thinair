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
E exec  candidate + a runtime               True (``ExecutesBelief`` opt-in)
======  ==================================  =================

Every default is overridable at the attachment site —
``VerbatimBelief("source_text", necessary=False)`` demotes a veto to a
measurement, and nothing else about the belief changes.

The whole library contains **zero model calls** (invariant 7); that is
precisely what its independence is made of.
"""

from __future__ import annotations

from .consistency import (
    ConservationBelief,
    FunctionalDependencyBelief,
    ItemsSumToBelief,
    MutuallyExclusiveBelief,
    RecomputeBelief,
    RelationBelief,
    SumsToBelief,
    TemporalOrderBelief,
)
from .executable import (
    ALLOW_EXEC_ENV,
    CalculatorBelief,
    ExecutesBelief,
    ExecutionRefused,
    PassesTestsBelief,
    RegexBehaviorBelief,
    RoundTripBelief,
    calculate,
)
from .form import (
    ChecksumBelief,
    EnumBelief,
    FormatBelief,
    LengthBelief,
    ParsesBelief,
    RangeBelief,
    SchemaBelief,
    SortedBelief,
    UniqueBelief,
    match_template,
    render_template,
    template_to_json_schema,
)
from .grounding import (
    FrozenConsistentBelief,
    FuzzyBelief,
    NonEchoBelief,
    NormalizedBelief,
    QuoteIntegrityBelief,
    SpanValidBelief,
    TokenSubsetBelief,
    VerbatimBelief,
)
from .reference import CalendarFactBelief, IsoCountryBelief, IsoCurrencyBelief, TimezoneBelief

#: family letter -> the classes it contains, in canonical order.
#: This is the registry: a plain mapping anyone can read, iterate, or
#: table-test over.  Adding a validator means adding it here, and the
#: conformance test in ``tests/test_validators_registry.py`` then covers it
#: with no new test code.
FAMILIES = {
    "A": (SchemaBelief, FormatBelief, ChecksumBelief, RangeBelief, EnumBelief, LengthBelief, UniqueBelief, SortedBelief, ParsesBelief),
    "B": (
        VerbatimBelief,
        NormalizedBelief,
        FuzzyBelief,
        TokenSubsetBelief,
        QuoteIntegrityBelief,
        SpanValidBelief,
        FrozenConsistentBelief,
        NonEchoBelief,
    ),
    "C": (
        SumsToBelief,
        ItemsSumToBelief,
        RecomputeBelief,
        TemporalOrderBelief,
        ConservationBelief,
        FunctionalDependencyBelief,
        MutuallyExclusiveBelief,
    ),
    "D": (IsoCountryBelief, IsoCurrencyBelief, TimezoneBelief, CalendarFactBelief),
    "E": (ExecutesBelief, PassesTestsBelief, RoundTripBelief, CalculatorBelief, RegexBehaviorBelief),
}

FAMILY_NAMES = {
    "A": "form",
    "B": "grounding",
    "C": "consistency",
    "D": "reference",
    "E": "executable",
}

#: validators that must never appear in a default belief list, and why.
#: ``ExecutesBelief`` runs model-generated code; an interactive human blocks.  The
#: read pipeline consults this when it assembles defaults.
NEVER_DEFAULT = {
    ExecutesBelief: "runs model-generated code; opt in explicitly",
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
    "SchemaBelief", "FormatBelief", "ChecksumBelief", "RangeBelief", "EnumBelief", "LengthBelief", "UniqueBelief",
    "SortedBelief", "ParsesBelief", "match_template", "template_to_json_schema",
    "render_template",
    # family B — grounding
    "VerbatimBelief", "NormalizedBelief", "FuzzyBelief", "TokenSubsetBelief", "QuoteIntegrityBelief",
    "SpanValidBelief", "FrozenConsistentBelief", "NonEchoBelief",
    # family C — internal consistency
    "RelationBelief", "SumsToBelief", "ItemsSumToBelief", "RecomputeBelief", "TemporalOrderBelief",
    "ConservationBelief", "FunctionalDependencyBelief", "MutuallyExclusiveBelief",
    # family D — pinned reference tables
    "IsoCountryBelief", "IsoCurrencyBelief", "TimezoneBelief", "CalendarFactBelief",
    # family E — executable
    "ExecutesBelief", "PassesTestsBelief", "RoundTripBelief", "CalculatorBelief", "RegexBehaviorBelief",
    "ExecutionRefused", "calculate", "ALLOW_EXEC_ENV",
    # the registry itself
    "FAMILIES", "FAMILY_NAMES", "NEVER_DEFAULT", "validators", "family_of",
]

