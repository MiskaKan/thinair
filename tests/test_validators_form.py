"""Family A — form: the candidate alone.

Table-driven: positive, negative, and the canonical edge rows —
bool-vs-int, int-acceptable-as-float, Luhn near-misses.
"""

from __future__ import annotations

import pytest

from thinair.validators import (
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
    template_to_json_schema,
)
from thinair.validators.form import ean13_ok, iban_ok, isbn10_ok, isbn13_ok, luhn_ok

from fakes import FakeSnapshot, head


def p_of(belief, value, attr="x"):
    """The probability a belief assigns a candidate, or None if out of scope."""
    got = belief(head(value), attr)
    return None if got is None else ~got


# --------------------------------------------------------------------------
# match_template — the shared shape language
# --------------------------------------------------------------------------

MATCHES = [
    (1249.5, float),
    (1249, float),                                   # int acceptable as float
    ("ACME", str),
    (None, None),
    (7, int),
    ({"desc": "x", "amount": 1.0}, {"desc": str, "amount": float}),
    ([1.0, 2.0], [float]),                           # single element = homogeneous
    ([], [float]),
    ([1, "a"], [int, str]),                          # fixed length
    ("either", (str, int)),                          # tuple = alternatives
    (3, (str, int)),
    ("pending", "pending"),                          # a literal
    ({"a": {"b": [1]}}, {"a": {"b": [int]}}),
    (object(), None if False else __import__("typing").Any),
]

MISMATCHES = [
    ("1249.5", float),
    (True, int),                                     # bool is not an int
    (1, bool),
    (1.5, int),
    ({"desc": "x"}, {"desc": str, "amount": float}),         # missing key
    ({"desc": "x", "amount": 1.0, "vat": 0.24},
     {"desc": str, "amount": float}),                        # unexpected key
    ([1.0, "a"], [float]),
    ([1], [int, str]),                                       # wrong length
    (1.5, (str, int)),
    ("shipped", "pending"),
    (None, str),
]


@pytest.mark.parametrize("value,template", MATCHES)
def test_match_template_accepts(value, template):
    assert match_template(value, template) is None


@pytest.mark.parametrize("value,template", MISMATCHES)
def test_match_template_rejects_with_a_reason(value, template):
    reason = match_template(value, template)
    assert isinstance(reason, str) and reason


def test_a_mismatch_reason_names_the_path():
    reason = match_template({"a": {"b": ["oops"]}}, {"a": {"b": [int]}})
    assert "a" in reason and "b" in reason


# --------------------------------------------------------------------------
# SchemaBelief — doubles as the structured-output contract
# --------------------------------------------------------------------------

def test_schema_judges_p_in_zero_one():
    schema = SchemaBelief(float)
    assert p_of(schema, 1249.5) == 1.0
    assert p_of(schema, "1249.5") == 0.0


def test_schema_is_necessary_by_default():
    assert SchemaBelief(float).necessary is True


def test_the_same_schema_object_drives_the_engine_and_the_check():
    """One object, so the constraint and the check cannot drift."""
    schema = SchemaBelief([{"desc": str, "amount": float}])
    emitted = schema.json_schema()
    assert emitted["type"] == "array"
    assert emitted["items"]["properties"]["amount"]["type"] == "number"
    assert set(emitted["items"]["required"]) == {"desc", "amount"}
    assert p_of(schema, [{"desc": "widget", "amount": 4.0}]) == 1.0
    assert p_of(schema, [{"desc": "widget"}]) == 0.0


@pytest.mark.parametrize("template,kind", [
    (float, "number"), (int, "integer"), (str, "string"), (bool, "boolean"),
])
def test_template_to_json_schema_maps_scalars(template, kind):
    assert template_to_json_schema(template)["type"] == kind


# --------------------------------------------------------------------------
# FormatBelief
# --------------------------------------------------------------------------

FORMAT_ROWS = [
    ("email", "jane@example.com", True),
    ("email", "jane@@example.com", False),
    ("email", "jane at example.com", False),
    ("url", "https://example.com/x?y=1", True),
    ("url", "example.com", False),
    ("iso_date", "2024-02-29", True),
    ("iso_date", "2024-02-30", False),
    ("iso_date", "29.02.2024", False),
    ("iso_datetime", "2024-02-29T12:00:00", True),
    ("iso_datetime", "2024-02-29T25:00:00", False),
    ("uuid", "123e4567-e89b-12d3-a456-426614174000", True),
    ("uuid", "123e4567-e89b-12d3-a456", False),
    ("semver", "1.2.3", True),
    ("semver", "1.2.3-rc.1+build.5", True),
    ("semver", "1.2", False),
    ("phone_e164", "+358401234567", True),
    ("phone_e164", "040 123 4567", False),
]


@pytest.mark.parametrize("kind,value,ok", FORMAT_ROWS)
def test_format(kind, value, ok):
    assert p_of(FormatBelief(kind), value) == (1.0 if ok else 0.0)


def test_format_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        FormatBelief("astrological-sign")


def test_format_has_nothing_to_say_about_a_non_string():
    """None is the scoping mechanism: an unscoped FormatBelief sits in
    ``__beliefs__`` and is consulted for every attribute, so it must stay
    silent about the ones it was never meant to check.  Type is SchemaBelief's job.
    """
    assert p_of(FormatBelief("email"), 42) is None


# --------------------------------------------------------------------------
# ChecksumBelief — including the classic near-misses
# --------------------------------------------------------------------------

CHECKSUM_ROWS = [
    ("luhn", "4539578763621486", True),
    ("luhn", "4539578763621487", False),            # last digit off by one
    ("luhn", "4539 5787 6362 1486", True),          # spacing is not the check
    ("isbn10", "0306406152", True),
    ("isbn10", "0306406153", False),
    ("isbn10", "080442957X", True),                 # X check digit
    ("isbn13", "9780306406157", True),
    ("isbn13", "9780306406158", False),
    ("ean13", "4006381333931", True),
    ("ean13", "4006381333932", False),
    ("iban", "GB82WEST12345698765432", True),
    ("iban", "GB82WEST12345698765433", False),
    ("iban", "FI2112345600000785", True),
]

#: candidates that are not identifiers of the scheme's shape at all -- these
#: draw silence, not a veto, so an unscoped ChecksumBelief cannot kill a vendor name.
CHECKSUM_OUT_OF_SCOPE = [
    ("luhn", 42), ("luhn", "hello"), ("isbn10", "123"),
    ("ean13", "4006381333931999"), ("iban", None), ("luhn", True),
]


@pytest.mark.parametrize("kind,value,ok", CHECKSUM_ROWS)
def test_checksum(kind, value, ok):
    assert p_of(ChecksumBelief(kind), value) == (1.0 if ok else 0.0)


@pytest.mark.parametrize("kind,value", CHECKSUM_OUT_OF_SCOPE)
def test_checksum_abstains_on_things_that_are_not_its_kind_of_identifier(kind, value):
    assert p_of(ChecksumBelief(kind), value) is None


@pytest.mark.parametrize("fn,good,bad", [
    (luhn_ok, "4539578763621486", "4539578763621487"),
    (isbn10_ok, "0306406152", "0306406153"),
    (isbn13_ok, "9780306406157", "9780306406158"),
    (ean13_ok, "4006381333931", "4006381333932"),
    (iban_ok, "GB82WEST12345698765432", "GB82WEST12345698765433"),
])
def test_the_checksum_helpers_stand_alone(fn, good, bad):
    assert fn(good) and not fn(bad)


def test_a_transposition_is_caught_where_the_algorithm_can_catch_it():
    assert luhn_ok("4539578763621486") and not luhn_ok("4535978763621486")


# --------------------------------------------------------------------------
# RangeBelief, EnumBelief, LengthBelief, UniqueBelief, SortedBelief
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lo,hi,value,expected", [
    (0, 1e6, 1249.5, 1.0),
    (0, 1e6, -5, 0.0),
    (0, 1e6, 1e7, 0.0),
    (0, 1e6, 0, 1.0),                                 # inclusive
    (0, 1e6, 1e6, 1.0),
    (0, None, 5, 1.0),
    (None, 10, 5, 1.0),
    (0, 10, "five", None),                            # out of scope, not wrong
    (0, 10, True, None),                              # a bool is not a number
])
def test_range(lo, hi, value, expected):
    assert p_of(RangeBelief(lo, hi), value) == expected


def test_range_needs_a_bound():
    with pytest.raises(ValueError):
        RangeBelief()


@pytest.mark.parametrize("value,expected", [
    ("paid", 1.0), ("PAID", 1.0), ("overdue", 0.0),
])
def test_enum_uses_the_normalizing_comparator(value, expected):
    assert p_of(EnumBelief(["paid", "pending"]), value) == expected


@pytest.mark.parametrize("lo,hi,value,expected", [
    (1, 10, "ACME", 1.0),
    (1, 3, "ACME", 0.0),
    (2, None, [1, 2, 3], 1.0),
    (None, 2, [1, 2, 3], 0.0),
    (1, 3, 42, None),                                 # nothing to measure -- silence
])
def test_length(lo, hi, value, expected):
    assert p_of(LengthBelief(lo, hi), value) == expected


@pytest.mark.parametrize("value,expected", [
    ([1, 2, 3], 1.0),
    ([1, 2, 2], 0.0),
    ([{"a": 1}, {"a": 1}], 0.0),                      # unhashable, still compared
    ([], 1.0),
    ("text", None),
])
def test_unique(value, expected):
    assert p_of(UniqueBelief(), value) == expected


@pytest.mark.parametrize("kwargs,value,expected", [
    ({}, [1, 2, 3], 1.0),
    ({}, [3, 2, 1], 0.0),
    ({"reverse": True}, [3, 2, 1], 1.0),
    ({"key": "amount"}, [{"amount": 1.0}, {"amount": 2.0}], 1.0),
    ({"key": "amount"}, [{"amount": 2.0}, {"amount": 1.0}], 0.0),
    ({}, [1], 1.0),
])
def test_sorted(kwargs, value, expected):
    assert p_of(SortedBelief(**kwargs), value) == expected


# --------------------------------------------------------------------------
# ParsesBelief — compiles without executing
# --------------------------------------------------------------------------

PARSE_ROWS = [
    ("json", '{"a": 1}', 1.0),
    ("json", "{'a': 1}", 0.0),
    ("python", "def f():\n    return 1\n", 1.0),
    ("python", "def f(:\n", 0.0),
    ("regex", r"^\d+$", 1.0),
    ("regex", r"^(\d+$", 0.0),
]


@pytest.mark.parametrize("kind,value,expected", PARSE_ROWS)
def test_parses(kind, value, expected):
    assert p_of(ParsesBelief(kind), value) == expected


def test_parses_python_does_not_execute_it():
    """Compiling is not running -- the file this would write must not exist."""
    import os
    import tempfile

    target = os.path.join(tempfile.gettempdir(), "thinair-parses-must-not-run")
    if os.path.exists(target):                        # pragma: no cover
        os.remove(target)
    code = f"open({target!r}, 'w').write('ran')"
    assert p_of(ParsesBelief("python"), code) == 1.0
    assert not os.path.exists(target)


# --------------------------------------------------------------------------
# family behavior shared by every member
# --------------------------------------------------------------------------

FAMILY_A = [SchemaBelief(float), FormatBelief("email"), ChecksumBelief("luhn"), RangeBelief(0, 10),
            EnumBelief(["a"]), LengthBelief(1, 2), UniqueBelief(), SortedBelief(), ParsesBelief("json")]


@pytest.mark.parametrize("belief", FAMILY_A, ids=lambda b: b.id)
def test_family_a_is_necessary_by_default(belief):
    assert belief.necessary is True


@pytest.mark.parametrize("belief", FAMILY_A, ids=lambda b: b.id)
def test_family_a_says_nothing_when_no_candidate_was_proposed(belief):
    assert belief(FakeSnapshot(beliefs=[lambda e, a: None]), "x") is None


@pytest.mark.parametrize("belief", FAMILY_A, ids=lambda b: b.id)
def test_family_a_reads_only_the_candidate(belief):
    """Family A consumes the candidate alone: no source attribute is touched."""
    class Trap:
        def __getattr__(self, name):                  # pragma: no cover
            raise AssertionError(f"family A read {name!r} off the entity")

    e = FakeSnapshot(beliefs=[lambda e, a: ("candidate", 0.5)])
    object.__setattr__(e, "_cells", Trap())
    try:
        belief(e, "x")
    except AssertionError:
        raise
    except Exception:
        pass                                          # any other failure is fine
