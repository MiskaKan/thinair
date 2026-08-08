"""Family E — executable checks: the strongest and the priciest.

``Executes`` is the one belief in the library that runs code it did not
write, so most of what is asserted here is about its refusal to do so
uninvited.
"""

from __future__ import annotations

import pathlib

import pytest

from thinair.validators import (
    ALLOW_EXEC_ENV,
    Calculator,
    Executes,
    ExecutionRefused,
    PassesTests,
    RegexBehavior,
    RoundTrip,
    calculate,
)

from fakes import FakeSnapshot, head


def p_of(belief, value):
    got = belief(head(value), "x")
    return None if got is None else ~got


# --------------------------------------------------------------------------
# calculate — an ast walk, never eval
# --------------------------------------------------------------------------

@pytest.mark.parametrize("expression,expected", [
    ("2 + 2", 4),
    ("12 * 3", 36),
    ("1249.50 / 1.24", pytest.approx(1007.6612903)),
    ("-5 + 1", -4),
    ("2 ** 10", 1024),
    ("7 // 2", 3),
    ("7 % 2", 1),
    ("(1 + 2) * 3", 9),
    ("1 < 2", True),
    ("1 < 2 < 3", True),
    ("2 == 2.0", True),
    ("3 != 3", False),
])
def test_calculate_evaluates_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo pwned')",
    "open('/tmp/x', 'w')",
    "x + 1",                                        # a name
    "'a' + 'b'",                                    # not arithmetic
    "[1, 2][0]",
    "lambda: 1",
    "print(1)",
    "().__class__",
])
def test_calculate_refuses_everything_that_is_not_arithmetic(expression):
    with pytest.raises((ValueError, SyntaxError)):
        calculate(expression)


def test_calculate_has_an_exponent_ceiling():
    """``2 ** 10**9`` is a denial of service written in four characters."""
    with pytest.raises(ValueError, match="ceiling"):
        calculate("2 ** 999999999")


def test_calculate_reports_division_by_zero_as_such():
    with pytest.raises(ZeroDivisionError):
        calculate("1 / 0")


def test_the_module_contains_no_eval():
    import thinair.validators.executable as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    assert "eval(" not in source.replace("_eval", "").replace(".eval", "")


# --------------------------------------------------------------------------
# Calculator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("12 * 3 = 36", 1.0),
    ("12 * 3 = 37", 0.0),
    ("12 * 3 == 36", 1.0),
    ("2 + 2", 1.0),                                 # evaluates, so it holds
    ("1 < 2", 1.0),
    ("2 < 1", 0.0),
    ("1 / 0 = 1", 0.0),
    ({"expression": "999.00 + 250.50", "result": 1249.5}, 1.0),
    ({"expression": "999.00 + 250.50", "result": 1349.5}, 0.0),
    ({"expression": "999.00 + 250.50"}, 1.0),
    ("the vendor is ACME Oy", None),                # no arithmetic asserted
    (1249.5, None),
    ({"result": 4}, None),
])
def test_calculator(value, expected):
    assert p_of(Calculator(), value) == expected


def test_calculator_judges_at_the_precision_the_claim_was_stated_to():
    """"1007.66" asserts two decimals, and is checked to two."""
    assert p_of(Calculator(), "1249.50 / 1.24 = 1007.66") == 1.0
    assert p_of(Calculator(), "1249.50 / 1.24 = 1007.67") == 0.0
    assert p_of(Calculator(tol=0.0), "1249.50 / 1.24 = 1007.66") == 0.0
    assert p_of(Calculator(tol=0.01), "1249.50 / 1.24 = 1007.66") == 1.0


def test_calculator_explains_the_arithmetic_it_did():
    got = Calculator()(head("12 * 3 = 37"), "x")
    assert "36" in got.meta["reason"] and "37" in got.meta["reason"]


# --------------------------------------------------------------------------
# RoundTrip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,value,expected", [
    ("date", "2024-02-29", 1.0),
    ("date", "2024-02-30", 0.0),
    ("date", "29.02.2024", 0.0),
    ("datetime", "2024-02-29T12:00:00", 1.0),
    ("datetime", "2024-02-29T25:00:00", 0.0),
    ("number", "1249.50", 0.0),                     # a string is not a number
    ("number", 1249.5, 1.0),
    ("number", "1,249.50", 0.0),                    # looks numeric, is not
    ("json", {"a": [1, 2]}, 1.0),
    ("json", [1, 2, 3], 1.0),
    ("python", "def f():\n    return 1\n", 1.0),
    ("python", "def f(:", 0.0),
])
def test_round_trip(fmt, value, expected):
    assert p_of(RoundTrip(fmt), value) == expected


def test_round_trip_rejects_an_unknown_format():
    with pytest.raises(ValueError):
        RoundTrip("papyrus")


def test_round_trip_catches_what_a_shape_check_cannot():
    """``"2024-02-30"`` is a well-formed string and an impossible date."""
    from thinair.validators import Schema

    assert p_of(Schema(str), "2024-02-30") == 1.0
    assert p_of(RoundTrip("date"), "2024-02-30") == 0.0


# --------------------------------------------------------------------------
# RegexBehavior
# --------------------------------------------------------------------------

def test_regex_behavior_is_graded_by_examples():
    belief = RegexBehavior(positives=["a@b.com", "x@y.fi"], negatives=["nope", "a@b"])
    assert p_of(belief, r"^\S+@\S+\.\S+$") == 1.0
    assert p_of(belief, r"^\S+@\S+$") == 0.75                # matches "a@b" too
    assert p_of(belief, r"^nothing$") == 0.5                 # misses both positives


def test_regex_behavior_flunks_a_pattern_that_does_not_compile():
    belief = RegexBehavior(positives=["a"])
    got = belief(head(r"^(a$"), "x")
    assert ~got == 0.0 and "compile" in got.meta["reason"]


def test_regex_behavior_needs_examples():
    with pytest.raises(ValueError):
        RegexBehavior()


def test_regex_behavior_says_nothing_about_a_non_string():
    assert p_of(RegexBehavior(positives=["a"]), 42) is None


# --------------------------------------------------------------------------
# PassesTests — each test callable is one check
# --------------------------------------------------------------------------

def positive(v):
    return v > 0


def under_ten(v):
    assert v < 10, f"{v} is not under ten"


def explodes(v):
    raise RuntimeError("the test itself is broken")


def test_passes_tests_is_graded():
    assert p_of(PassesTests([positive, under_ten]), 5) == 1.0
    assert p_of(PassesTests([positive, under_ten]), 42) == 0.5
    assert p_of(PassesTests([positive, under_ten]), -1) == 0.5


def test_a_failing_assertion_carries_its_message():
    got = PassesTests([under_ten])(head(42), "x")
    assert "not under ten" in got.meta["reason"]


def test_a_test_that_explodes_is_a_failed_test():
    got = PassesTests([explodes])(head(1), "x")
    assert ~got == 0.0 and "RuntimeError" in got.meta["reason"]


def test_passes_tests_identity_follows_the_test_source():
    """Editing a test mints a new belief rather than silently redefining one."""
    def threshold(v):
        return v > 0

    first = PassesTests([threshold]).id

    def threshold(v):                                # noqa: F811 - deliberate
        return v > 1

    assert PassesTests([threshold]).id != first


def test_passes_tests_needs_callables():
    with pytest.raises(TypeError):
        PassesTests(["not callable"])
    with pytest.raises(ValueError):
        PassesTests([])


# --------------------------------------------------------------------------
# Executes — opt-in, and nothing else
# --------------------------------------------------------------------------

def test_executes_refuses_without_an_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(ALLOW_EXEC_ENV, raising=False)
    with pytest.raises(ExecutionRefused, match=ALLOW_EXEC_ENV):
        Executes()(head("print(1)"), "code")


def test_executes_accepts_a_per_belief_opt_in(monkeypatch):
    monkeypatch.delenv(ALLOW_EXEC_ENV, raising=False)
    assert p_of(Executes(allow_exec=True), "x = 1 + 1") == 1.0


def test_executes_accepts_a_process_wide_opt_in(monkeypatch):
    monkeypatch.setenv(ALLOW_EXEC_ENV, "1")
    assert p_of(Executes(), "x = 1 + 1") == 1.0
    monkeypatch.setenv(ALLOW_EXEC_ENV, "0")
    with pytest.raises(ExecutionRefused):
        Executes()(head("x = 1"), "code")


def test_the_opt_in_is_not_part_of_the_belief_identity():
    """Same check, same id: willingness to run it is this process's business."""
    assert Executes(allow_exec=True).id == Executes().id


def test_executes_reports_the_exception_a_candidate_raised(monkeypatch):
    monkeypatch.setenv(ALLOW_EXEC_ENV, "1")
    got = Executes()(head('raise ValueError("boom")'), "code")
    assert ~got == 0.0 and "boom" in got.meta["reason"]


def test_executes_times_out_rather_than_hanging(monkeypatch):
    monkeypatch.setenv(ALLOW_EXEC_ENV, "1")
    got = Executes(timeout=0.5)(head("while True:\n    pass\n"), "code")
    assert ~got == 0.0 and "0.5s" in got.meta["reason"]


def test_executes_blocks_the_network(monkeypatch):
    monkeypatch.setenv(ALLOW_EXEC_ENV, "1")
    code = "import urllib.request\nurllib.request.urlopen('http://example.com')\n"
    got = Executes()(head(code), "code")
    assert ~got == 0.0 and "network disabled" in got.meta["reason"]


def test_executes_runs_in_a_scratch_directory(monkeypatch, tmp_path):
    """A candidate that writes a file must not write it into the project."""
    monkeypatch.setenv(ALLOW_EXEC_ENV, "1")
    monkeypatch.chdir(tmp_path)
    assert p_of(Executes(), "open('artifact.txt', 'w').write('hi')") == 1.0
    assert not (tmp_path / "artifact.txt").exists()


def test_executes_needs_a_positive_timeout():
    with pytest.raises(ValueError):
        Executes(timeout=0)


def test_executes_says_nothing_about_a_non_string(monkeypatch):
    monkeypatch.setenv(ALLOW_EXEC_ENV, "1")
    assert p_of(Executes(), 42) is None
    assert p_of(Executes(), "   ") is None


def test_executes_is_never_in_a_default_list():
    """Never auto-registered; the registry says so out loud."""
    from thinair.validators import NEVER_DEFAULT

    assert Executes in NEVER_DEFAULT


# --------------------------------------------------------------------------
# family behavior
# --------------------------------------------------------------------------

FAMILY_E = [Executes(), PassesTests([positive]), RoundTrip("json"),
            Calculator(), RegexBehavior(positives=["a"])]


@pytest.mark.parametrize("belief", FAMILY_E, ids=lambda b: b.id)
def test_family_e_is_necessary_by_default(belief):
    assert belief.necessary is True


@pytest.mark.parametrize("belief", FAMILY_E, ids=lambda b: b.id)
def test_family_e_is_silent_without_a_candidate(belief):
    assert belief(FakeSnapshot(beliefs=[lambda e, a: None]), "x") is None
