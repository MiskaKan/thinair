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
which ends consultation for that cell -- and freezing is a property of the
*belief*, never of a model's eloquence: assignment freezes, and a belief
declared ``frozen=True`` (code, notably) pins what it settles.

    from thinair import Thing, model, human
    from thinair.validators import TokenSubsetBelief

    class Invoice(Thing):
        '''An invoice document to be understood.'''
        __beliefs__ = [model("small-fast"), human("jane"),
                       TokenSubsetBelief("source_text")]
        source_text: str
        total = Thing(float, extracted_from="source_text", range=(0, 1e6))

    inv = Invoice(source_text=open("invoice.txt").read())
    print(+inv.total, ~inv.total)
    inv += model("qwen3-35b")        # a dissimilar second voice, on record

A ``Thing(shape, ...)`` in a class body *declares* an attribute and attaches
the validators the shape implies; ``Thing.__default__ = model("...")`` names
the generative belief any panel without one falls back to.

**What this package is for.**  A large language model is not an oracle; it
is a **measurement instrument**: it projects things onto axes you name.
thinair makes that operational.  Any nameable question becomes a contracted
attribute; the model's reading arrives as an opinion ``(value, p)``, never a
fact; validators are laws whose degree of violation is itself a reading;
dissimilar beliefs on one panel are independent paths; and the ledger
records everything, so confidence is *earned* -- by coherence that nothing
enforced, and by concordance with the frozen ground -- never assumed.

This buys a new kind of data analysis.  Text, events, judgments -- data no
parser can read -- become measurable: sort things that have no numeric key
("which should we staff first?"), compare pairwise and settle the
comparisons into interval scores, project two unlike corpora onto shared
axes and correlate the clouds, measure a collection as one thing, measure
the instrument itself.  **The model is a column factory**: it is consulted
only where no deterministic reader exists, and everything downstream of a
reading -- sorting, statistics, aggregation -- stays classical, exact and
free.  If a regex can read the field, a regex is the belief.

The loop, when handed raw data and asked to create value from it:

1. Read ``thinair/GROUNDING.md`` -- it ships inside this package and is
   written to be linked, as-is, to an LLM together with sample data.  Part 1
   is the measurement theory (the four pillars), Part 2 the strategy
   protocol to answer with, Part 3 the mapping onto this framework.
2. Run the strategy as an ordinary thinair program: Things, contracts,
   panels -- it needs no framework surface beyond what you see above.
3. Settle the record with ``thinair.evaluate``: reliability, discrimination,
   concordance with frozen outcomes, calibration of the stated ``p`` -- what
   did the readings *earn*.  Only licensed statistics; settlement is
   classical math and spends no model calls.
4. Let the grades pick the next strategy.  Every opinion lands in the
   durable store (``.thinair/``) by default, so structure accumulates:
   readings persist, and yesterday's record is today's instrument.

``SPEC.md`` in the repository is the as-built contract.
"""

from __future__ import annotations

from . import evaluate
from . import store
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
    Consulted,
    Disagreement,
    LowConfidence,
    Proposed,
    ResolutionPolicy,
    Threshold,
    Unanimous,
    Unresolvable,
)
from .thing import Cell, Snapshot, Thing, corroborate, snapshot
from .validators import (
    CalculatorBelief,
    CalendarFactBelief,
    ChecksumBelief,
    ConservationBelief,
    EnumBelief,
    ExecutesBelief,
    FormatBelief,
    FrozenConsistentBelief,
    FunctionalDependencyBelief,
    FuzzyBelief,
    IsoCountryBelief,
    IsoCurrencyBelief,
    ItemsSumToBelief,
    LengthBelief,
    MutuallyExclusiveBelief,
    NonEchoBelief,
    NormalizedBelief,
    ParsesBelief,
    PassesTestsBelief,
    QuoteIntegrityBelief,
    RangeBelief,
    RecomputeBelief,
    RegexBehaviorBelief,
    RelationBelief,
    RoundTripBelief,
    SchemaBelief,
    SortedBelief,
    SpanValidBelief,
    SumsToBelief,
    TemporalOrderBelief,
    TimezoneBelief,
    TokenSubsetBelief,
    UniqueBelief,
    VerbatimBelief,
)

__version__ = "2.3.1"

__all__ = [
    # the object surface
    "Thing", "Cell", "Snapshot", "snapshot", "source", "corroborate",
    # settlement over the record, and where the record lives
    "evaluate", "store",
    # the one contract
    "Belief", "Discriminative", "Judgment", "Opinion", "Ledger",
    "ModelBelief", "HumanBelief", "Scoped",
    "model", "human", "lookup", "registry",
    "default_ledger", "set_default_ledger", "use_ledger", "values_equal",
    # functions as cells
    "fn", "freeze_call", "call_id",
    # policies and what they raise
    "ResolutionPolicy", "Proposed", "Consulted", "Unanimous", "Threshold",
    "LowConfidence", "Unresolvable", "Disagreement",
    # the validator library, re-exported for convenience
    "SchemaBelief", "FormatBelief", "ChecksumBelief", "RangeBelief", "EnumBelief", "LengthBelief", "UniqueBelief",
    "SortedBelief", "ParsesBelief",
    "VerbatimBelief", "NormalizedBelief", "FuzzyBelief", "TokenSubsetBelief", "QuoteIntegrityBelief",
    "SpanValidBelief", "FrozenConsistentBelief", "NonEchoBelief",
    "RelationBelief", "SumsToBelief", "ItemsSumToBelief", "RecomputeBelief", "TemporalOrderBelief",
    "ConservationBelief", "FunctionalDependencyBelief", "MutuallyExclusiveBelief",
    "IsoCountryBelief", "IsoCurrencyBelief", "TimezoneBelief", "CalendarFactBelief",
    "ExecutesBelief", "PassesTestsBelief", "RoundTripBelief", "CalculatorBelief", "RegexBehaviorBelief",
    "__version__",
]

# Tracking is on unless switched off (THINAIR_STORE=off): the default ledger
# becomes .thinair/opinions.db, created lazily on the first append.  Explicit
# ledgers and use_ledger() scopes are untouched.
store.install()
