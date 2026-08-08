"""Family B — grounding: the candidate against the entity's own state."""

from __future__ import annotations

import pytest

from thinair.ledger import Ledger, Opinion
from thinair.validators import (
    FrozenConsistent,
    Fuzzy,
    NonEcho,
    Normalized,
    QuoteIntegrity,
    SpanValid,
    TokenSubset,
    Verbatim,
)
from thinair.validators.grounding import entities_in, normalize_text, numbers_in

from fakes import FakeSnapshot, head

SOURCE = (
    "INVOICE 2024-118\n"
    "ACME Oy — Helsinki\n"
    'Widget, large   999.00\nShipping         250.50\n'
    "Total 1 249,50 EUR\n"
    'The buyer said "delivery is final".'
)


def judge(belief, value, attr="x", **snapshot):
    got = belief(head(value, source_text=SOURCE, **snapshot), attr)
    return None if got is None else ~got


# --------------------------------------------------------------------------
# the text helpers, which everything else is built on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Total 1 249,50 EUR", [1249.5]),
    ("12.5 and 1,234", [12.5, 1234.0]),
    ("no numbers here", []),
    ("999.00 250.50", [999.0, 250.5]),
    ("-40 degrees", [-40.0]),          # the sign is part of the number
])
def test_numbers_in_is_thousands_separator_tolerant(text, expected):
    assert numbers_in(text) == expected


def test_entities_in_finds_capitalized_runs():
    assert "ACME Oy" in entities_in(SOURCE)


@pytest.mark.parametrize("a,b", [
    ("ACME  Oy", "acme oy"),
    ("Café", "Café"),                      # NFKC normalization
    ("total: 1,249.50!", "total 1,249.50"),
])
def test_normalize_text_collapses_the_differences_that_are_not_differences(a, b):
    assert normalize_text(a) == normalize_text(b)


# --------------------------------------------------------------------------
# Verbatim / Normalized / Fuzzy — the same question at three strictnesses
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("ACME Oy", 1.0),
    ("acme oy", 0.0),                            # verbatim means verbatim
    ("ACME Ab", 0.0),
    ("", 0.0),
])
def test_verbatim_is_binary(value, expected):
    assert judge(Verbatim("source_text"), value) == expected


@pytest.mark.parametrize("value,expected", [
    ("ACME Oy", 1.0),
    ("acme  oy", 1.0),
    ("ACME, Oy!", 1.0),
    ("ACME Ab", 0.0),
])
def test_normalized_forgives_case_and_punctuation(value, expected):
    assert judge(Normalized("source_text"), value) == expected


def test_fuzzy_is_graded_and_deterministic():
    strong = judge(Fuzzy("source_text"), "ACME Oy")
    weak = judge(Fuzzy("source_text"), "ACME Oyj Corporation International")
    assert strong == 1.0
    assert 0.0 < weak < 1.0
    assert judge(Fuzzy("source_text"), "ACME Oyj Corporation International") == weak


def test_fuzzy_is_never_necessary_even_if_asked():
    """Never ``necessary`` -- the constructor enforces it rather than trusting."""
    assert Fuzzy("source_text").necessary is False
    assert Fuzzy("source_text", necessary=True).necessary is False


def test_grounding_says_nothing_without_a_source():
    empty = head("ACME Oy")
    assert Verbatim("source_text")(empty, "vendor") is None
    assert Fuzzy("source_text")(empty, "vendor") is None


def test_grounding_says_nothing_without_a_candidate():
    e = FakeSnapshot(beliefs=[lambda e, a: None], source_text=SOURCE)
    assert Verbatim("source_text")(e, "vendor") is None


# --------------------------------------------------------------------------
# TokenSubset — the classic hallucination tell
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (1249.5, 1.0),
    (999.0, 1.0),
    (1349.5, 0.0),                               # a number that is not there
    ("Total is 1249.50", 1.0),
    ([{"desc": "Widget, large", "amount": 999.0}], 1.0),
    ([{"desc": "Widget", "amount": 111.0}], 0.0),
    ("no numbers at all", None),                 # nothing to check
])
def test_token_subset_over_numbers(value, expected):
    assert judge(TokenSubset("source_text"), value) == expected


def test_token_subset_is_graded_by_how_much_is_invented():
    p = judge(TokenSubset("source_text"), "999.00 plus 250.50 plus 7777.00")
    assert p == pytest.approx(2 / 3, abs=1e-4)   # rounded for a readable ledger


def test_token_subset_can_also_police_entities():
    belief = TokenSubset("source_text", kinds=("numbers", "entities"))
    assert judge(belief, "ACME Oy billed 999.00") == 1.0
    assert judge(belief, "Globex Inc billed 999.00") < 1.0


def test_token_subset_default_is_numbers_only():
    assert TokenSubset("source_text").kinds == ("numbers",)
    assert judge(TokenSubset("source_text"), "Globex Inc billed 999.00") == 1.0


# --------------------------------------------------------------------------
# QuoteIntegrity / SpanValid
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ('The buyer said "delivery is final".', 1.0),
    ('The buyer said "delivery is optional".', 0.0),
    ("A paraphrase with nothing quoted.", None),
])
def test_quote_integrity(value, expected):
    assert judge(QuoteIntegrity("source_text"), value) == expected


def test_quote_integrity_lets_the_unquoted_part_paraphrase_freely():
    value = 'In short, entirely reworded prose — "delivery is final" — indeed.'
    assert judge(QuoteIntegrity("source_text"), value) == 1.0


def test_span_valid():
    source = "ACME Oy — Helsinki"
    e = head({"start": 0, "end": 7, "text": "ACME Oy"}, source_text=source)
    assert ~SpanValid("source_text")(e, "vendor") == 1.0

    wrong = head({"start": 0, "end": 4, "text": "ACME Oy"}, source_text=source)
    assert ~SpanValid("source_text")(wrong, "vendor") == 0.0

    outside = head({"start": 0, "end": 999, "text": "ACME Oy"}, source_text=source)
    assert ~SpanValid("source_text")(outside, "vendor") == 0.0


def test_span_valid_says_nothing_about_a_value_that_is_not_span_shaped():
    assert judge(SpanValid("source_text"), "ACME Oy") is None


# --------------------------------------------------------------------------
# FrozenConsistent — reads the ledger through the snapshot
# --------------------------------------------------------------------------

def test_frozen_consistent_compares_against_the_pinned_opinion():
    ledger = Ledger()
    ledger.add(Opinion(belief="human:jane", entity="inv-1", attr="total",
                       value=1249.5, p=1.0, frozen=True))
    agrees = FakeSnapshot(entity="inv-1", ledger=ledger,
                          beliefs=[lambda e, a: (1249.5, 0.7)])
    differs = FakeSnapshot(entity="inv-1", ledger=ledger,
                           beliefs=[lambda e, a: (99.0, 0.99)])

    assert ~FrozenConsistent()(agrees, "total") == 1.0
    assert ~FrozenConsistent()(differs, "total") == 0.0


def test_frozen_consistent_is_silent_when_nothing_is_pinned():
    e = FakeSnapshot(entity="inv-1", ledger=Ledger(),
                     beliefs=[lambda e, a: (1249.5, 0.7)])
    assert FrozenConsistent()(e, "total") is None


def test_frozen_consistent_uses_the_normalizing_comparator():
    ledger = Ledger()
    ledger.add(Opinion(belief="human:jane", entity="inv-1", attr="vendor",
                       value="ACME  Oy", p=1.0, frozen=True))
    e = FakeSnapshot(entity="inv-1", ledger=ledger,
                     beliefs=[lambda e, a: ("acme oy", 0.7)])
    assert ~FrozenConsistent()(e, "vendor") == 1.0


# --------------------------------------------------------------------------
# NonEcho — progress is enforced, not hoped for
# --------------------------------------------------------------------------

CARRIED = "The invoice totals 1249.50 euros and is addressed to ACME Oy."


def test_non_echo_vetoes_a_verbatim_echo():
    e = FakeSnapshot(value=CARRIED, beliefs=[lambda a, b: (CARRIED, 0.99)])
    got = NonEcho()(e, "explanation")
    assert ~got < 0.5                              # below the default veto line
    assert "echoed" in got.meta["reason"]


def test_non_echo_passes_genuine_elaboration():
    elaborated = (
        "Two line items make up the sum: a large widget at 999.00 and "
        "shipping at 250.50. Finnish VAT of 24% is already included, which "
        "is why the net figure a bookkeeper would post differs."
    )
    e = FakeSnapshot(value=CARRIED, beliefs=[lambda a, b: (elaborated, 0.9)])
    assert ~NonEcho()(e, "explanation") > 0.5


def test_non_echo_is_graded_not_binary():
    half = CARRIED + " Additionally, entirely new prose appears here now."
    e = FakeSnapshot(value=CARRIED, beliefs=[lambda a, b: (half, 0.9)])
    p = ~NonEcho()(e, "explanation")
    assert 0.0 < p < 1.0


def test_non_echo_says_nothing_when_the_entity_carries_no_value():
    assert judge(NonEcho(), "anything") is None


def test_non_echo_threshold_is_configurable():
    e = FakeSnapshot(value=CARRIED, beliefs=[lambda a, b: (CARRIED, 0.99)])
    assert NonEcho(max_overlap=1.0).max_overlap == 1.0
    assert NonEcho(max_overlap=1.0).id != NonEcho().id


# --------------------------------------------------------------------------
# family behavior
# --------------------------------------------------------------------------

FAMILY_B = [Verbatim("source_text"), Normalized("source_text"),
            TokenSubset("source_text"), QuoteIntegrity("source_text"),
            SpanValid("source_text"), FrozenConsistent(), NonEcho()]


@pytest.mark.parametrize("belief", FAMILY_B, ids=lambda b: b.id)
def test_family_b_is_necessary_by_default(belief):
    assert belief.necessary is True


@pytest.mark.parametrize("belief", FAMILY_B + [Fuzzy("source_text")],
                         ids=lambda b: b.id)
def test_family_b_is_deterministic(belief):
    e = FakeSnapshot(beliefs=[lambda x, a: ("ACME Oy 999.00", 0.9)],
                     source_text=SOURCE, value=CARRIED)
    first = belief(e, "x")
    assert all(belief(e, "x") == first for _ in range(3))
