"""thinair -- Python objects whose attributes are beliefs, not values.

Reading ``invoice.total`` consults an ordered panel of belief functions and
returns a value together with an honest probability.  Every opinion ever
rendered is kept in a ledger.

**There is no truth in this framework -- only opinions.**  The model
hallucinates freely; at any point you can ask how differently-constructed
beliefs see a value, and when *dissimilar* beliefs agree, that is your best
bet that you are onto something.  It is your responsibility to ensure your
beliefs are not copies of one another -- the framework records who said what;
it does not referee.

Three nouns and no more.  A **Belief** is a function ``b(e, a) -> (v, p)``:
the model is one, every validator is one, the human is one.  An **Opinion**
is a Belief's recorded evaluation at a cell.  An opinion may be **frozen**,
which ends consultation for that cell -- and freezing is a code-only
capability: assignment, code execution, and the explicit ``freeze`` verb.

    from thinair import Thing, contract, model, human, freeze
    from thinair.validators import TokenSubset

    class Invoice(Thing):
        '''An invoice document to be understood.'''
        __beliefs__ = [model("small-fast"), human("jane"),
                       TokenSubset("source_text")]
        source_text: str
        total = contract(float, extracted_from="source_text", range=(0, 1e6))

    inv = Invoice(source_text=open("invoice.txt").read())
    print(+inv.total, ~inv.total)
"""

from __future__ import annotations

from .beliefs import (
    Belief,
    Discriminative,
    Judgment,
    ModelBelief,
    HumanBelief,
    Scoped,
    human,
    lookup,
    model,
    registry,
)
from .debug import source
from .fn import call_id, fn, freeze_call
from .ledger import (
    Ledger,
    Opinion,
    default_ledger,
    set_default_ledger,
    use_ledger,
    values_equal,
)
from .policy import (
    Disagreement,
    LowConfidence,
    Proposed,
    ResolutionPolicy,
    Threshold,
    Unanimous,
    Unresolvable,
)
from .thing import Cell, Snapshot, Thing, contract, freeze, snapshot
from .validators import (
    CalendarFact,
    Calculator,
    Checksum,
    Conservation,
    Enum,
    Executes,
    Format,
    FrozenConsistent,
    FunctionalDependency,
    Fuzzy,
    IsoCountry,
    IsoCurrency,
    ItemsSumTo,
    Length,
    MutuallyExclusive,
    NonEcho,
    Normalized,
    Parses,
    PassesTests,
    QuoteIntegrity,
    Range,
    Recompute,
    RegexBehavior,
    Relation,
    RoundTrip,
    Schema,
    Sorted,
    SpanValid,
    SumsTo,
    TemporalOrder,
    Timezone,
    TokenSubset,
    Unique,
    Verbatim,
)

__version__ = "0.1.0"

__all__ = [
    # the object surface
    "Thing", "Cell", "Snapshot", "contract", "snapshot", "freeze", "source",
    # the one contract
    "Belief", "Discriminative", "Judgment", "Opinion", "Ledger",
    "ModelBelief", "HumanBelief", "Scoped",
    "model", "human", "lookup", "registry",
    "default_ledger", "set_default_ledger", "use_ledger", "values_equal",
    # functions as cells
    "fn", "freeze_call", "call_id",
    # policies and what they raise
    "ResolutionPolicy", "Proposed", "Unanimous", "Threshold",
    "LowConfidence", "Unresolvable", "Disagreement",
    # the validator library, re-exported for convenience
    "Schema", "Format", "Checksum", "Range", "Enum", "Length", "Unique",
    "Sorted", "Parses",
    "Verbatim", "Normalized", "Fuzzy", "TokenSubset", "QuoteIntegrity",
    "SpanValid", "FrozenConsistent", "NonEcho",
    "Relation", "SumsTo", "ItemsSumTo", "Recompute", "TemporalOrder",
    "Conservation", "FunctionalDependency", "MutuallyExclusive",
    "IsoCountry", "IsoCurrency", "Timezone", "CalendarFact",
    "Executes", "PassesTests", "RoundTrip", "Calculator", "RegexBehavior",
    "__version__",
]
