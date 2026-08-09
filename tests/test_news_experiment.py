"""The PLAN_2 news experiment: corpus, pre-pass, custom beliefs, analysis.

Offline and deterministic.  The experiment's *findings* need a live endpoint;
its machinery must not.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = os.path.join(ROOT, "experiments", "news")
if EXPERIMENT not in sys.path:
    sys.path.insert(0, EXPERIMENT)

import analyze                                                    # noqa: E402
import corpus                                                     # noqa: E402
import prepass                                                    # noqa: E402
import strategy                                                   # noqa: E402
from fakes import FakeEngine, FakeSnapshot                        # noqa: E402
from thinair import Ledger, model, use_ledger                     # noqa: E402
from thinair.validators.grounding import numbers_in               # noqa: E402


# --------------------------------------------------------------------------
# the corpus is what it says it is
# --------------------------------------------------------------------------

def test_corpus_shape():
    assert len(corpus.ITEMS) == 12
    assert len({i["id"] for i in corpus.ITEMS}) == 12
    for item in corpus.ITEMS:
        assert set(item["ground"]) == {"country", "people_affected",
                                       "certainty", "horizon"}
        assert 40 < len(item["text"].split()) < 120


@pytest.mark.parametrize("item_id", ["n03", "n06"])
def test_sum_only_items_state_no_total(item_id):
    """The planted tension: the correct answer is absent from the text."""
    item = corpus.BY_ID[item_id]
    assert item["ground"]["people_affected"] not in numbers_in(item["text"])


@pytest.mark.parametrize("item_id", [i["id"] for i in corpus.ITEMS
                                     if i["id"] not in ("n03", "n06")])
def test_stated_items_state_their_total(item_id):
    item = corpus.BY_ID[item_id]
    truth = item["ground"]["people_affected"]
    assert truth == 0 or truth in numbers_in(item["text"])


def test_conflicting_counts_resolve_to_the_official_one():
    text = corpus.BY_ID["n09"]["text"]
    assert 250000.0 in numbers_in(text) and 89000.0 in numbers_in(text)
    assert corpus.BY_ID["n09"]["ground"]["people_affected"] == 89000


# --------------------------------------------------------------------------
# the zero-call pre-pass
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pre():
    return prepass.run(corpus.ITEMS)


def test_prepass_spends_nothing(pre):
    assert pre["calls"] == 0


def test_prepass_finds_exactly_the_planted_duplicate(pre):
    assert [row["pair"] for row in pre["near_duplicates"]] == [("n11", "n12")]


def test_shingling_alone_would_have_missed_it(pre):
    """Two of three signals fire; the standard one does not.  A real result."""
    row = pre["near_duplicates"][0]
    assert row["fired"] == ["unigram", "fingerprint"]
    assert row["shingle5"] < prepass.DUPLICATE_AT


def test_prepass_finds_exactly_the_planted_weekday_fault(pre):
    assert [name for name, _why in pre["dateline_faults"]] == ["n05"]
    assert "Monday" in pre["dateline_faults"][0][1]


def test_reference_table_coverage_is_itself_measured(pre):
    outside = {name for name, record in pre["items"].items()
               if record["dateline"]["table_p"] < 1.0}
    assert outside == {"n02", "n08", "n11", "n12"}


# --------------------------------------------------------------------------
# the two beliefs the package does not ship
# --------------------------------------------------------------------------

def test_permutation_of_accepts_a_permutation():
    belief = strategy.PermutationOf(["a", "b", "c"])
    assert belief.judge(["c", "a", "b"], None, None) == 1.0


def test_permutation_of_grades_a_violation_and_says_what_is_wrong():
    belief = strategy.PermutationOf(["a", "b", "c"])
    p, reason = belief.judge(["a", "b", "b"], None, None)
    assert 0.0 <= p < 1.0
    assert "missing ['c']" in reason and "repeated ['b']" in reason


def test_permutation_of_abstains_off_its_shape():
    assert strategy.PermutationOf(["a"]).judge("not a list", None, None) is None


def test_among_attributes_reads_the_alternatives_off_the_snapshot():
    belief = strategy.AmongAttributes("left_id", "right_id")
    e = FakeSnapshot(left_id="n01", right_id="n02")
    assert belief.judge("n01", e, "winner") == 1.0
    p, reason = belief.judge("n99", e, "winner")
    assert p == 0.0 and "n99" in reason


def test_among_attributes_abstains_with_nothing_to_compare():
    belief = strategy.AmongAttributes("left_id", "right_id")
    assert belief.judge("n01", FakeSnapshot(), "winner") is None


@pytest.mark.parametrize("belief,value,e", [
    (strategy.PermutationOf(["a", "b"]), ["b", "a"], None),
    (strategy.AmongAttributes("left_id"), "n01", FakeSnapshot(left_id="n01")),
])
def test_custom_beliefs_are_deterministic(belief, value, e):
    first = belief.judge(value, e, "x")
    assert all(belief.judge(value, e, "x") == first for _ in range(5))


# --------------------------------------------------------------------------
# only licensed statistics
# --------------------------------------------------------------------------

def test_ranks_average_ties():
    assert analyze.ranks([5, 5, 1]) == [1.5, 1.5, 3.0]


def test_spearman_is_one_on_a_monotone_pair():
    rho, n = analyze.spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert rho == 1.0 and n == 4


def test_spearman_is_minus_one_reversed():
    rho, _ = analyze.spearman([1, 2, 3, 4], [40, 30, 20, 10])
    assert rho == -1.0


def test_spearman_declines_on_too_few_pairs():
    rho, n = analyze.spearman([1, None, 3], [1, 2, None])
    assert rho is None and n == 1


def test_wilson_brackets_the_estimate():
    point, lo, hi = analyze.wilson(9, 10)
    assert lo < point < hi and 0.0 <= lo and hi <= 1.0


def test_ratio_agreement_tolerates_a_percent_and_nothing_more():
    assert analyze.compare("people_affected", 1000, 1005) is True
    assert analyze.compare("people_affected", 1000, 1200) is False


def test_ordinal_agreement_is_exact_and_adjacency_is_separate():
    from thinair.evaluate import adjacent
    assert analyze.compare("severity", 3, 4) is False
    assert adjacent(3, 4, "ordinal") is True
    assert adjacent("FI", "SE", "nominal") is None


def test_ratio_claim_readings_compare_by_result_alone():
    """Two transcriptions of the same asserted arithmetic agree even when
    the expressions differ -- the experiment's own equivalence, kept local."""
    assert analyze.compare("ratio_claim",
                           {"expression": "3400/8500", "result": 0.4},
                           {"expression": "0.4", "result": 0.4}) is True


# --------------------------------------------------------------------------
# end to end, scripted: the machinery runs with zero network
# --------------------------------------------------------------------------

SCRIPT = {
    "country": {"value": "FI", "p": 0.9},
    "people_affected": {"value": 3400, "p": 0.95},
    "certainty": {"value": 3, "p": 0.8},
    "horizon": {"value": 1, "p": 0.7},
    "ratio_claim": {"value": {"expression": "3400/8500", "result": 0.4}, "p": 0.9},
    "severity": {"value": 2, "p": 0.6},
    "follow_up": {"value": 3, "p": 0.5},
}


def _scripted_engine():
    def reply(messages):
        asked = messages[-1]["content"]
        for axis, answer in SCRIPT.items():
            if f'"{axis}"?' in asked:
                return answer
        return {"value": None, "p": 0.0}
    return FakeEngine(reply)


def test_a_whole_item_measures_offline_and_lands_in_the_ledger():
    engine = _scripted_engine()
    belief = model("deepseek-v4-flash", engine=engine)
    ledger = Ledger()
    with use_ledger(ledger):
        Story = strategy.story_class(belief)
        story = Story(__entity__="n01", text=corpus.BY_ID["n01"]["text"])
        for axis in strategy.AXES:
            getattr(story, axis)

    assert engine.call_count == len(strategy.AXES)
    for axis in strategy.AXES:
        got = analyze.reading(ledger, "n01", axis, belief=belief.id)
        assert got is not None and not got["vetoed"]
        assert got["value"] == SCRIPT[axis]["value"]


def test_the_ledger_alone_answers_who_said_what():
    """PLAN_2 Part 4 rule 2, as an assertion."""
    engine = _scripted_engine()
    belief = model("deepseek-v4-flash", engine=engine)
    ledger = Ledger()
    with use_ledger(ledger):
        Story = strategy.story_class(belief)
        story = Story(__entity__="n01", text=corpus.BY_ID["n01"]["text"])
        story.country

    authors = {o.belief for o in ledger.opinions(entity="n01", attr="country")}
    assert belief.id in authors
    assert any(a.startswith("isoCountry") for a in authors)
    assert any("deepseek-v4-flash" in a and "T0.2" in a and "extract-v3" in a
               for a in authors)


def test_a_fabricated_number_is_vetoed_offline():
    """TokenSubset is necessary on people_affected; 999 is in no item."""
    engine = FakeEngine([{"value": 999, "p": 0.99}])
    belief = model("deepseek-v4-flash", engine=engine)
    ledger = Ledger()
    with use_ledger(ledger):
        Story = strategy.story_class(belief)
        story = Story(__entity__="n01", text=corpus.BY_ID["n01"]["text"])
        with pytest.raises(Exception):
            story.people_affected
    got = analyze.reading(ledger, "n01", "people_affected", belief=belief.id)
    assert got["vetoed"] and got["rounds"] > 1
