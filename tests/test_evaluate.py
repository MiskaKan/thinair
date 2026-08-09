"""thinair.evaluate: settlement over the record.

Every export is exercised here -- an export no test touches is speculation
and comes out.  The ledgers are built by hand: settlement is pure math over
the record, so the tests need no engine, except the two that pin the
exposure stamp (which is Layer 1's side of the bargain).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import thinair
from fakes import FakeEngine, FakeSnapshot, ScriptedBelief
from thinair import Thing, human, model
from thinair.beliefs import Belief
from thinair.evaluate import (
    GRADES, LICENSED, agree, adjacent, bradley_terry, calibration,
    concordance, coverage, discrimination, drift, evaluate, grounded, kappa,
    mannwhitney, median, n_eff, ranks, reading, reliability, separation,
    spearman, tiers, verdicts, wilson,
)
from thinair.ledger import Ledger, Opinion

MODEL_META = {"model": "fake"}


def opinion(entity, attr, value, p=0.9, belief="model:fake", frozen=False,
            meta=MODEL_META):
    return Opinion(belief=belief, entity=entity, attr=attr, value=value,
                   p=p, frozen=frozen, meta=dict(meta))


class Law(Belief):
    """A registered judge: veto rights live on the constructed belief."""

    necessary = True


LAW = Law(id="law:evaluate-test")


def verdict(entity, attr, value, p, belief=LAW.id, reason=None):
    meta = {"judged": belief}
    if reason:
        meta["reason"] = reason
    return Opinion(belief=belief, entity=entity, attr=attr, value=value,
                   p=p, meta=meta)


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------

def _thing(script, **kwargs):
    class Item(Thing):
        __beliefs__ = [ScriptedBelief(script, belief_id="scripted:eval"),
                       human("tester")]
        colour = Thing(str)
        size = Thing(int)
    return Item(__ledger__=Ledger(), **kwargs)


def test_evaluate_resolves_every_declared_cell():
    item = _thing({"colour": ("red", 0.9), "size": (3, 0.8)})
    profile = evaluate(item)
    assert profile["colour"]["status"] == "resolved"
    assert profile["colour"]["value"] == "red"
    assert profile["size"]["p"] == 0.8
    assert set(profile) == {"colour", "size"}


def test_evaluate_marks_frozen_cells_and_spends_nothing_on_them():
    item = _thing({"size": (3, 0.8)}, colour="blue")
    profile = evaluate(item)
    assert profile["colour"]["status"] == "frozen"
    assert profile["colour"]["p"] == 1.0
    assert item.__beliefs__[0].calls.count("colour") == 0


def test_evaluate_records_gaps_instead_of_raising():
    """A cell that cannot resolve is a coverage gap, not a crash."""
    class Bare(Thing):
        __beliefs__ = [human("tester")]
        name = Thing(str)
    profile = evaluate(Bare(__ledger__=Ledger()))
    assert profile["name"]["status"] == "gap"
    assert "Unresolvable" in profile["name"]["reason"]


def test_evaluate_order_is_context():
    """``order=`` drives reading order -- the reliability-twin reshuffle."""
    item = _thing({"colour": ("red", 0.9), "size": (3, 0.8)})
    evaluate(item, order=["size", "colour"])
    assert item.__beliefs__[0].calls[0] == "size"
    with pytest.raises(ValueError):
        evaluate(item, order=["colour", "weight"])


# --------------------------------------------------------------------------
# reading the record
# --------------------------------------------------------------------------

def test_reading_returns_the_last_candidate_with_veto_status():
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 3, p=0.4))
    ledger.add(opinion("e1", "size", 4, p=0.9))
    ledger.add(verdict("e1", "size", 4, p=0.1, reason="too big"))
    got = reading(ledger, "e1", "size", belief="model:fake")
    assert got["value"] == 4 and got["rounds"] == 2
    assert got["vetoed"] and got["vetoes"] == [(LAW.id, "too big")]


def test_reading_ignores_verdicts_about_other_candidates():
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 3, p=0.4))
    ledger.add(verdict("e1", "size", 3, p=0.1))       # about the old candidate
    ledger.add(opinion("e1", "size", 4, p=0.9))
    assert not reading(ledger, "e1", "size", belief="model:fake")["vetoed"]


def test_reading_defaults_to_the_cells_last_instrument():
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4, p=0.9))
    ledger.add(Opinion(belief="human:x", entity="e1", attr="size", value=7,
                       p=1.0, frozen=True, meta={"assigned": True}))
    got = reading(ledger, "e1", "size")
    assert got["value"] == 4                 # frozen authority is not a reading


def test_an_unregistered_judge_is_loud_not_a_low_score():
    """The bug this module exists to make unrepeatable."""
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4, p=0.9))
    ledger.add(Opinion(belief="law:ghost", entity="e1", attr="size", value=4,
                       p=0.1, meta={"judged": "law:ghost"}))
    with pytest.raises(LookupError):
        reading(ledger, "e1", "size", belief="model:fake")
    got = reading(ledger, "e1", "size", belief="model:fake", strict=False)
    assert not got["vetoed"]


def test_verdicts_groups_by_judge_and_excludes_the_instruments():
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4))
    ledger.add(verdict("e1", "size", 4, p=0.95))
    got = verdicts(ledger, "e1", "size", exclude=("model:fake",))
    assert list(got) == [LAW.id]
    assert got[LAW.id][0][:2] == (4, 0.95)


def test_coverage_separates_resolved_vetoed_and_absent():
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4, p=0.9))
    ledger.add(opinion("e2", "size", 9, p=0.8))
    ledger.add(verdict("e2", "size", 9, p=0.1, reason="refused"))
    got = coverage(ledger, ["e1", "e2", "e3"], ["size"], belief="model:fake")
    assert got["size"]["resolved"] == 1
    assert got["size"]["vetoed"] == [("e2", ["refused"])]
    assert got["size"]["absent"] == ["e3"]


# --------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------

def test_agree_is_scale_typed():
    assert agree("FI", "fi", "nominal")                  # one comparator
    assert agree(3, 3.0, "ordinal") and not agree(3, 4, "ordinal")
    assert agree(1000, 1005, "ratio") and not agree(1000, 1200, "ratio")
    assert not agree(1000, 1005, "ratio", tol=0.001)
    assert agree(None, 3, "ordinal") is None             # a gap, not a miss


def test_adjacent_is_the_ordinal_near_miss():
    assert adjacent(3, 4, "ordinal") and not adjacent(3, 5, "ordinal")
    assert adjacent("FI", "SE", "nominal") is None


def test_kappa_discounts_the_agreement_the_marginals_hand_out():
    assert kappa([("x", "x"), ("y", "y"), ("z", "z")])["kappa"] == 1.0
    # 80% raw agreement on a 90%-modal axis is worth little
    skewed = [("a", "a")] * 8 + [("a", "b"), ("b", "a")]
    got = kappa(skewed)
    assert got["po"] == 0.8 and got["kappa"] < 0.2
    enforced = kappa([("a", "a")] * 5)
    assert enforced["kappa"] is None and "enforced" in enforced["reason"]
    assert kappa([])["n"] == 0


# --------------------------------------------------------------------------
# licensed statistics
# --------------------------------------------------------------------------

def test_ranks_median_spearman_wilson():
    assert ranks([10, 20, 20, 5]) == [3.0, 1.5, 1.5, 4.0]
    assert median([3, 1, 2]) == 2 and median([None]) is None
    assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == (1.0, 4)
    assert spearman([1, 2], [2, 1]) == (None, 2)         # two points always agree
    low, lo, hi = wilson(9, 10)
    assert low == 0.9 and lo < 0.9 < hi                  # an interval, out loud


def test_mannwhitney_compares_unpaired_clouds():
    apart = mannwhitney([5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 1, 2, 3, 4])
    assert apart["r"] == 1.0 and apart["p"] < 0.01
    same = mannwhitney([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])
    assert abs(same["r"]) < 0.1 and same["p"] > 0.5
    tied = mannwhitney([3, 3], [3, 3])
    assert tied["z"] is None and "tied" in tied["reason"]
    assert mannwhitney([], [1])["reason"] == "a group is empty"


def test_n_eff_is_the_repetition_honesty_note():
    assert n_eff(100, 1.0) == 1.0                        # copies add nothing
    assert n_eff(10, 0.0) == 10.0
    assert 1 < n_eff(100, 0.5) < 2.0                     # bounded by 1/rho
    with pytest.raises(ValueError):
        n_eff(10, 1.5)


# --------------------------------------------------------------------------
# orders from comparisons
# --------------------------------------------------------------------------

def test_bradley_terry_recovers_a_transitive_order():
    wins = [("a", "b")] * 3 + [("b", "c")] * 3 + [("a", "c")] * 2
    got = bradley_terry(wins)
    assert got["ranking"] == ["a", "b", "c"]
    assert got["connected"] and got["consistency"] == 1.0
    assert got["scores"]["a"] > got["scores"]["b"] > got["scores"]["c"]
    assert all(got["se"][i] > 0 for i in "abc")


def test_bradley_terry_reads_intransitivity_instead_of_hiding_it():
    wins = [("a", "b")] * 3 + [("b", "c")] * 3 + [("a", "c")] * 2
    got = bradley_terry(wins + [("c", "a")])
    assert got["consistency"] < 1.0
    assert ("c", "a", 1) in got["inconsistent"]


def test_bradley_terry_flags_a_disconnected_graph():
    """Strengths across components mean nothing -- blocking without bridges."""
    got = bradley_terry([("a", "b"), ("c", "d")])
    assert not got["connected"]
    assert sorted(map(sorted, got["components"])) == [["a", "b"], ["c", "d"]]


def test_bradley_terry_edges():
    assert bradley_terry([])["n"] == 0
    with pytest.raises(ValueError):
        bradley_terry([("a", "a")])


# --------------------------------------------------------------------------
# the instrument, measured
# --------------------------------------------------------------------------

SCALES = {"size": "ordinal", "kind": "nominal"}


def _reliability_ledger():
    ledger = Ledger()
    for base, twin, a, b in [("e1", "e1~t", 3, 3), ("e2", "e2~t", 3, 4),
                             ("e3", "e3~t", 2, 4)]:
        ledger.add(opinion(base, "size", a))
        ledger.add(opinion(twin, "size", b))
    return ledger


def test_reliability_rates_exact_and_within_one():
    rows = reliability(_reliability_ledger(),
                       [("e1", "e1~t"), ("e2", "e2~t"), ("e3", "e3~t")],
                       ["size"], SCALES, belief="model:fake")
    row = rows["size"]
    assert row["n"] == 3
    assert row["exact"] == round(1 / 3, 3)
    assert row["within_one"] == round(2 / 3, 3)


def test_reliability_reports_stuck_pairs_separately():
    ledger = _reliability_ledger()
    ledger.add(verdict("e3", "size", 2, p=0.1))
    ledger.add(verdict("e3~t", "size", 4, p=0.1))
    rows = reliability(ledger, [("e3", "e3~t")], ["size"], SCALES,
                       belief="model:fake")
    assert rows["size"]["n"] == 0
    assert rows["size"]["unresolved_both"] == ["e3"]


def test_reliability_accepts_a_bespoke_equivalence():
    rows = reliability(_reliability_ledger(),
                       [("e2", "e2~t")], ["size"], SCALES, belief="model:fake",
                       compare=lambda axis, a, b: abs(a - b) <= 1)
    assert rows["size"]["exact"] == 1.0


def test_drift_reads_a_direction_that_averaging_cannot_shrink():
    rows = reliability(_reliability_ledger(),
                       [("e1", "e1~t"), ("e2", "e2~t"), ("e3", "e3~t")],
                       ["size"], SCALES, belief="model:fake")
    got = drift(rows, SCALES)["size"]
    assert got["deltas"] == [0, 1, 2]
    assert got["up"] == 2 and got["down"] == 0 and got["same_sign"]
    assert drift(rows, {"size": "nominal"}) == {}        # ordinal-only law


def test_discrimination_flags_the_constant_instrument():
    varied = discrimination(["a", "b", "c", "a"])
    assert varied["distinct"] == 3 and varied["modal_share"] == 0.5
    constant = discrimination(["a"] * 9 + [None])
    assert constant["n"] == 9                            # a gap is not a value
    assert constant["distinct"] == 1 and constant["modal_share"] == 1.0


# --------------------------------------------------------------------------
# the ground
# --------------------------------------------------------------------------

def _grounded_ledger():
    """Measure first; the outcome freezes on the same cell later."""
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4, p=1.0))
    ledger.add(Opinion(belief="human:x", entity="e1", attr="size", value=4,
                       p=1.0, frozen=True, meta={"assigned": True}))
    ledger.add(opinion("e2", "size", 5, p=0.2))
    ledger.add(Opinion(belief="human:x", entity="e2", attr="size", value=3,
                       p=1.0, frozen=True, meta={"assigned": True}))
    return ledger


def test_grounded_finds_same_cell_ground_with_no_map():
    """Calibration accrues from the ledger alone (Pillar IV, the Ground)."""
    cells = list(grounded(_grounded_ledger(), belief="model:fake"))
    assert [(e, a, t) for e, a, _r, t in cells] == \
        [("e1", "size", 4), ("e2", "size", 3)]


def test_grounded_takes_a_parallel_entity_map():
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4))
    ledger.add(Opinion(belief="human:author", entity="e1!g", attr="size",
                       value=4, p=1.0, frozen=True, meta={"assigned": True}))
    cells = list(grounded(ledger, axes=["size"], entities=["e1"],
                          ground=lambda e: f"{e}!g", belief="model:fake"))
    assert cells[0][3] == 4


def test_concordance_calibration_and_separation():
    rows, per_reading = concordance(
        grounded(_grounded_ledger(), belief="model:fake"), SCALES)
    assert rows["size"]["n"] == 2 and rows["size"]["hits"] == 1
    assert rows["size"]["misses"][0][:3] == ("e2", 5, 3)
    assert rows["size"]["rate"][0] == 0.5

    curve = calibration(per_reading)
    assert curve[1.0 if 1.0 in curve else 0.9]["hits"] == 1

    split = separation(per_reading)["size"]
    assert split["mean_p_hit"] == 1.0 and split["mean_p_miss"] == 0.2
    assert split["separation"] == 0.8                    # p was doing work


def test_tiers_prices_findings_and_validation_separately():
    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4))
    ledger.add(opinion("e1~t", "size", 4))
    ledger.add(Opinion(belief="human:x", entity="e1", attr="kind", value="a",
                       p=1.0, frozen=True, meta={"assigned": True}))
    got = tiers(ledger, validation=lambda o: o.entity.endswith("~t"))
    assert got == dict(findings=1, validation=1, total=2, validation_share=0.5)
    named = tiers(ledger, validation=lambda o: False,
                  proposers=["model:fake"])
    assert named["findings"] == 2                        # explicit instrument set


# --------------------------------------------------------------------------
# grounding as data
# --------------------------------------------------------------------------

def test_licensed_and_grades_are_data():
    assert set(LICENSED) == {"nominal", "ordinal", "interval", "ratio"}
    assert "spearman" in LICENSED["ordinal"]
    assert "pearson" not in LICENSED["ordinal"]          # the whole point
    assert [g for g, _ in GRADES] == \
        ["vibes", "claim", "finding", "calibrated"]


# --------------------------------------------------------------------------
# the exposure stamp (Layer 1's side of the bargain)
# --------------------------------------------------------------------------

def _snapshot(text):
    return FakeSnapshot(entity="fixed-entity", source_text=(text, 1.0))


def test_model_opinions_carry_an_exposure_fingerprint():
    belief = model("small-fast", engine=FakeEngine([{"value": 1, "p": 0.9}]))
    got = belief(_snapshot("alpha"), "total")
    stamp = got.meta["exposure"]
    assert len(stamp) == 12 and int(stamp, 16) >= 0


def test_exposure_separates_contexts_and_nothing_else():
    """Same rendered context -> same stamp; different context -> different.
    Without this, 'did these two agreeing readings see the same snapshot?'
    is unanswerable from the ledger, and Layer 2's exposure half is lost."""
    belief = model("small-fast", engine=FakeEngine([{"value": 1, "p": 0.9}]))
    warm = belief(_snapshot("alpha"), "total").meta["exposure"]
    again = belief(_snapshot("alpha"), "total").meta["exposure"]
    cold = belief(_snapshot("beta"), "total").meta["exposure"]
    assert warm == again
    assert warm != cold


# --------------------------------------------------------------------------
# the module practices what it teaches
# --------------------------------------------------------------------------

def test_evaluate_contains_zero_model_calls():
    """The column-factory rule as an import-graph test: settlement is
    classical math over the ledger; only the driver spends calls, and only
    through beliefs the strategist declared."""
    path = pathlib.Path(thinair.__file__).parent / "evaluate.py"
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    assert not any("engine" in i or "models" in i for i in imports), imports
    for network in ("urllib", "http", "socket", "requests", "ssl"):
        assert not any(i.split(".")[0] == network for i in imports)
    assert "complete_json" not in path.read_text()


# --------------------------------------------------------------------------
# settled: the standing value, fiat included
# --------------------------------------------------------------------------

def test_settled_serves_the_fiat_that_reading_refuses():
    """reading() grades instruments, so a fiat returns None there --
    settled() is the record read back the way a program would be served."""
    from thinair.evaluate import settled

    ledger = Ledger()
    ledger.add(opinion("e1", "size", 4, p=0.9, meta={"model": "m", "round": 1}))
    ledger.add(Opinion(belief="human:you", entity="e1", attr="size", value=5,
                       p=1.0, frozen=True, meta={"assigned": True}))
    got = settled(ledger, "e1", "size")
    assert got["value"] == 5 and got["frozen"] and got["belief"] == "human:you"
    assert reading(ledger, "e1", "size")["value"] == 4   # the instrument, apart

    # a cell only ever assigned: reading None, settled serves -- the silent
    # zero from the field report cannot recur through settled
    ledger.add(Opinion(belief="human:you", entity="e1", attr="name",
                       value="x", p=1.0, frozen=True, meta={"assigned": True}))
    assert reading(ledger, "e1", "name") is None
    assert settled(ledger, "e1", "name")["value"] == "x"
    assert settled(ledger, "e1", "ghost") is None


# --------------------------------------------------------------------------
# consensus: readers apart from judges, and the min-max range
# --------------------------------------------------------------------------

def test_consensus_counts_readers_apart_from_judges():
    """A judge verdicts the candidate it is handed -- its concord is
    purchased.  Only independent readings move `readers`, and the p spread
    keeps its min-max `range` beside the deviation."""
    from thinair.evaluate import history

    ledger = Ledger()
    ledger.add(Opinion(belief="model:a", entity="e9", attr="size", value=4,
                       p=0.9, meta={"model": "a", "round": 1}))
    ledger.add(Opinion(belief="schemaBelief[int]", entity="e9", attr="size",
                       value=4, p=1.0, meta={"judged": "schemaBelief[int]"}))
    view = history(ledger, entity="e9")[-1]["consensus"]["size"]
    assert view["readers"] == 1                    # the judge never counts

    ledger.add(Opinion(belief="model:b", entity="e9", attr="size", value=4,
                       p=0.5, meta={"model": "b", "corroboration": True}))
    view = history(ledger, entity="e9")[-1]["consensus"]["size"]
    assert view["readers"] == 2
    assert view["readers_dissent"] == 0
    assert view["range"] == pytest.approx(0.5)     # 1.0 (judge) down to 0.5
    assert view["dev"] < view["range"]             # the range refuses to average

    ledger.add(Opinion(belief="model:c", entity="e9", attr="size", value=7,
                       p=0.8, meta={"model": "c", "corroboration": True}))
    view = history(ledger, entity="e9")[-1]["consensus"]["size"]
    assert view["readers"] == 3 and view["readers_dissent"] == 1
