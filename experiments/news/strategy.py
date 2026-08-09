"""The strategy of STRATEGY.md, as a thinair program.

Nothing here adds framework surface (PLAN_2 Part 4 rule 1).  It is beliefs,
contracts, panels and module verbs; the two instruments thinair does not ship
--- "is this list a permutation of that set" and "is this value one of the
entity's own attributes" --- arrive as ordinary code beliefs.
"""

from __future__ import annotations

# DISCLOSURE (2026-08-09): mechanically renamed validator classes to their
# thinair 2.2 *Belief names (Calculator -> CalculatorBelief, IsoCountry ->
# IsoCountryBelief).  No behavioral change.  The pre-registered original --
# the bytes the runlog.json hash stamps -- is preserved in git history.
from thinair import Thing, human, model
from thinair.beliefs import Discriminative, value_at
from thinair.ledger import values_equal
from thinair.validators import CalculatorBelief, IsoCountryBelief

from corpus import CERTAINTY, HORIZON, MEASURAND

# --------------------------------------------------------------------------
# the two instruments the package does not ship
# --------------------------------------------------------------------------

class PermutationOf(Discriminative):
    """The candidate is exactly ``members``: no drops, no inventions, no repeats.

    The law that makes a one-call listwise ordering safe.  Its degree of
    violation is a reading: three of twelve ids missing is p = 0.75, not an
    exception.
    """

    necessary = True

    def __init__(self, members, **options):
        self.members = tuple(members)
        super().__init__(self.members, **options)

    def describe(self) -> str:
        return f"a permutation of the {len(self.members)} given ids"

    def judge(self, value, e, attr):
        if not isinstance(value, (list, tuple)):
            return None                       # not list-shaped: out of scope
        got, want = list(value), set(self.members)
        missing = [m for m in self.members if m not in got]
        invented = [g for g in got if g not in want]
        repeated = [g for g in set(got) if got.count(g) > 1]
        wrong = len(missing) + len(invented) + len(repeated)
        if not wrong:
            return 1.0
        p = max(0.0, 1.0 - wrong / max(1, len(self.members)))
        return p, (f"missing {missing or '-'}, invented {invented or '-'}, "
                   f"repeated {repeated or '-'}")


class AmongAttributes(Discriminative):
    """The candidate equals one of the entity's own attributes.

    A comparison must name one of the two things compared.  Reads the
    alternatives off the sealed snapshot, so it never needs them at class
    definition time.
    """

    necessary = True

    def __init__(self, *attrs, **options):
        self.attrs = tuple(attrs)
        super().__init__(self.attrs, **options)

    def describe(self) -> str:
        return "one of " + " or ".join(self.attrs)

    def judge(self, value, e, attr):
        options = [value_at(e, a) for a in self.attrs]
        options = [o for o in options if o is not None]
        if not options:
            return None                       # nothing to compare against
        if any(values_equal(value, o) for o in options):
            return 1.0
        return 0.0, f"{value!r} is neither {' nor '.join(map(repr, options))}"


# --------------------------------------------------------------------------
# the pre-registered axes
# --------------------------------------------------------------------------

#: Read in this order.  The reliability sub-pass reads them reversed, which is
#: the context reshuffle: each reading then sees a different set of resolved
#: siblings in the snapshot.
AXES = ("country", "people_affected", "certainty", "horizon",
        "ratio_claim", "severity", "follow_up")

#: Axes carrying a designed-in human assignment in corpus.py.
GROUNDED = ("country", "people_affected", "certainty", "horizon")

#: Scale type per axis; only licensed statistics are computed from each.
SCALE = {"country": "nominal", "people_affected": "ratio", "certainty": "ordinal",
         "horizon": "ordinal", "ratio_claim": "nominal", "severity": "ordinal",
         "follow_up": "ordinal"}

_RUBRIC = lambda table: "; ".join(f"{k} = {v}" for k, v in table.items())

STORY_PURPOSE = (
    "One incoming wire item, to be measured on pre-registered axes. Read only "
    "what this item says. Do not import outside knowledge about the places or "
    "organisations named, and do not infer beyond the text."
)


def story_class(proposer, name="Story"):
    """The measured Thing.  ``proposer`` is the instrument under test."""

    namespace = dict(
        __doc__=STORY_PURPOSE,
        __beliefs__=[proposer, human("desk")],
        __annotations__={"text": str},
        country=Thing(
            str, length=(2, 2), beliefs=[IsoCountryBelief(necessary=False)],
            doc="the ISO 3166-1 alpha-2 code of the country in the dateline"),
        people_affected=Thing(
            int, extracted_from="text", range=(0, 10_000_000),
            doc=MEASURAND),
        certainty=Thing(
            int, enum=[1, 2, 3, 4],
            doc="how the central claim is sourced: " + _RUBRIC(CERTAINTY)),
        horizon=Thing(
            int, enum=[1, 2, 3, 4],
            doc="when the central claim becomes checkable against a public "
                "record: " + _RUBRIC(HORIZON)),
        ratio_claim=Thing(
            {"expression": str, "result": float},
            beliefs=[CalculatorBelief(necessary=False)],
            doc="the item's own arithmetic about itself, transcribed and not "
                "corrected: expression is a plain arithmetic expression built "
                "only from numbers the item states, result is the figure the "
                "item asserts for it. If the item asserts no arithmetic, "
                'return {"expression": "0", "result": 0}'),
        severity=Thing(
            int, enum=[1, 2, 3, 4, 5],
            doc="harm already done, 1 negligible to 5 catastrophic"),
        follow_up=Thing(
            int, enum=[1, 2, 3, 4, 5],
            doc="how strongly a small newsroom should staff this for "
                "follow-up this week, 1 not at all to 5 first priority"),
    )
    return Thing.__class__(name, (Thing,), namespace)


def ground_class():
    """The frozen ground: one entity per item, authored by the corpus, never read."""

    return Thing.__class__("Ground", (Thing,), dict(
        __doc__="The designed-in ground for one corpus item: a human "
                "assignment, frozen on arrival and never consulted.",
        __beliefs__=[human("corpus-author")],
        __annotations__={"country": str, "people_affected": int,
                         "certainty": int, "horizon": int},
    ))


def corpus_class(item_ids, proposer):
    """The collection as a thing, measured on an ordering axis in one call."""

    return Thing.__class__("Corpus", (Thing,), dict(
        __doc__="The whole corpus of wire items as one thing. Its briefs "
                "attribute holds every item in full; measure the collection, "
                "not any single item.",
        __beliefs__=[proposer, human("desk")],
        __annotations__={"briefs": str},
        order=Thing(
            [str], beliefs=[PermutationOf(item_ids)],
            doc=f"all {len(item_ids)} item ids, ordered by how strongly a "
                "small newsroom should staff each for follow-up this week, "
                "highest first; every id exactly once"),
    ))


def pair_class(proposer):
    """Two items, compared.  The comparative path, and the transitivity law."""

    return Thing.__class__("Pair", (Thing,), dict(
        __doc__="Two wire items, to be compared. Answer with an id, not a "
                "score.",
        __beliefs__=[proposer, human("desk")],
        __annotations__={"left_id": str, "left": str,
                         "right_id": str, "right": str},
        winner=Thing(
            str, beliefs=[AmongAttributes("left_id", "right_id")],
            doc="the id of the one item a small newsroom should staff for "
                "follow-up first; exactly one of left_id and right_id"),
    ))
