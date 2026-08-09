"""Family E — executable checks: the strongest and the priciest.

These beliefs judge a candidate by *running something*.  Two of them —
``CalculatorBelief`` and ``RoundTripBelief`` — run only their own code and are safe
anywhere.  ``PassesTestsBelief`` runs callables the user supplied, so its risk is
the user's own.  ``ExecutesBelief`` runs **model-generated code** and is therefore
gated behind an explicit opt-in; it is never in a default belief list.

Determinism, as everywhere, means reproducible rather than binary: the same
snapshot must yield the same ``(v, p)``.  A test suite that depends on wall
clock or network is the user's bug, not the framework's — but ``ExecutesBelief``
does what it can, running in a scratch directory with the network stubbed
out of the child interpreter.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile

from ..beliefs import Discriminative, code_identity
from ..ledger import values_equal

__all__ = [
    "ExecutesBelief",
    "PassesTestsBelief",
    "RoundTripBelief",
    "CalculatorBelief",
    "RegexBehaviorBelief",
    "ExecutionRefused",
    "calculate",
    "ALLOW_EXEC_ENV",
]

ALLOW_EXEC_ENV = "THINAIR_ALLOW_EXEC"


class ExecutionRefused(RuntimeError):
    """Raised when ``ExecutesBelief`` is consulted without an explicit opt-in.

    A programming error, not an epistemic one — hence an exception rather
    than a low ``p``.  The user asked for model-generated code to be run and
    has not said so out loud.
    """


# --------------------------------------------------------------------------
# CalculatorBelief — an ast walk, never eval
# --------------------------------------------------------------------------

_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_CMP = {
    ast.Eq: values_equal,
    ast.NotEq: lambda a, b: not values_equal(a, b),
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}
#: exponent ceiling — ``2 ** 10**9`` is a denial of service written in four
#: characters, and no honest arithmetic assertion needs it.
MAX_POW = 1024


class _Arith(ast.NodeVisitor):
    """Evaluate arithmetic and nothing else."""

    def generic_visit(self, node):
        raise ValueError(f"not arithmetic: {type(node).__name__}")

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"not a number: {node.value!r}")
        return node.value

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):
            return -self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +self.visit(node.operand)
        raise ValueError(f"not arithmetic: {type(node.op).__name__}")

    def visit_BinOp(self, node):
        op = _BIN.get(type(node.op))
        if op is None:
            raise ValueError(f"not arithmetic: {type(node.op).__name__}")
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW:
            raise ValueError(f"exponent {right} exceeds the {MAX_POW} ceiling")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")
        return op(left, right)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, right_node in zip(node.ops, node.comparators):
            fn = _CMP.get(type(op))
            if fn is None:
                raise ValueError(f"not a comparison: {type(op).__name__}")
            right = self.visit(right_node)
            if not fn(left, right):
                return False
            left = right
        return True


def calculate(expression):
    """Evaluate an arithmetic expression by walking its ast.

    Returns the number, or the boolean of a comparison chain.  Raises
    ``ValueError`` for anything that is not pure arithmetic — names, calls,
    attribute access, string concatenation.  There is no ``eval`` here and
    there never will be.
    """
    tree = ast.parse(str(expression).strip(), mode="eval")
    return _Arith().visit(tree)


class CalculatorBelief(Discriminative):
    """Re-evaluate arithmetic the candidate asserts.

    Judges three shapes: a bare expression (``"12 * 3"`` — must evaluate), an
    assertion (``"12 * 3 = 37"`` or ``"12 * 3 == 37"`` — must hold), and a
    mapping ``{"expression": ..., "result": ...}``.  Anything with no
    arithmetic in it draws ``None`` — out of scope, not wrong.
    """

    necessary = True

    def __init__(self, tol=None, **options):
        # tol=None means "judge at the precision the claim was stated to":
        # "1007.66" asserts two decimals and is checked to two, while
        # "12 * 3 = 37" asserts an integer and is checked exactly.  Pass a
        # float to fix the tolerance instead.
        super().__init__(*(() if tol is None else (tol,)), **options)
        self.tol = None if tol is None else float(tol)

    def _agrees(self, got, claimed):
        if isinstance(got, bool) or isinstance(claimed, bool):
            return values_equal(got, claimed), claimed
        if not isinstance(got, (int, float)) or not isinstance(claimed, (int, float)):
            return values_equal(got, claimed), claimed
        if self.tol is not None:
            return abs(got - claimed) <= self.tol, claimed
        places = self._places(claimed)
        return values_equal(round(got, places), round(claimed, places)), claimed

    @staticmethod
    def _places(number):
        text = repr(float(number))
        if "e" in text or "E" in text:
            return 12
        _, _, frac = text.partition(".")
        return len(frac.rstrip("0"))

    @staticmethod
    def _split(text):
        text = text.strip()
        if "==" in text:
            lhs, _, rhs = text.partition("==")
            return lhs, rhs
        # a single '=' that is not part of a comparison operator
        for i, ch in enumerate(text):
            if ch == "=" and text[i - 1 : i] not in ("<", ">", "!", "=") and text[i + 1 : i + 2] != "=":
                return text[:i], text[i + 1 :]
        return text, None

    def judge(self, value, e, attr):
        if isinstance(value, dict):
            expr = value.get("expression")
            claimed = value.get("result")
            if expr is None:
                return None
            try:
                got = calculate(expr)
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
                return 0.0, f"{expr!r} does not evaluate: {exc}"
            if claimed is None:
                return 1.0, f"{expr} = {got}"
            agrees, claimed = self._agrees(got, claimed)
            if agrees:
                return 1.0, f"{expr} = {got}"
            return 0.0, f"{expr} = {got}, not {claimed}"

        if not isinstance(value, str):
            return None
        lhs, rhs = self._split(value)
        try:
            got = calculate(lhs)
        except (ValueError, SyntaxError):
            return None  # no arithmetic to check — out of scope
        except (ZeroDivisionError, OverflowError) as exc:
            return 0.0, f"{lhs.strip()} does not evaluate: {exc}"
        if rhs is None:
            if got is False:
                return 0.0, f"{value.strip()} is false"
            return 1.0, f"{lhs.strip()} evaluates to {got}"
        try:
            claimed = calculate(rhs)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return 0.0, f"{rhs.strip()} does not evaluate: {exc}"
        agrees, claimed = self._agrees(got, claimed)
        if agrees:
            return 1.0, f"{lhs.strip()} = {got}"
        return 0.0, f"{lhs.strip()} = {got}, not {claimed}"


# --------------------------------------------------------------------------
# RoundTripBelief
# --------------------------------------------------------------------------

def _rt_json(v):
    return json.loads(json.dumps(v))


def _rt_date(v):
    return _dt.date.fromisoformat(str(v)).isoformat()


def _rt_datetime(v):
    return _dt.datetime.fromisoformat(str(v)).isoformat()


def _rt_number(v):
    text = str(v).strip()
    return float(text) if ("." in text or "e" in text.lower()) else int(text)


def _rt_python(v):
    return ast.unparse(ast.parse(str(v)))


_ROUNDTRIPS = {
    "json": (_rt_json, lambda v: v),
    "date": (_rt_date, lambda v: str(v).strip()),
    "datetime": (_rt_datetime, lambda v: str(v).strip()),
    "number": (_rt_number, lambda v: v),
    "python": (_rt_python, lambda v: ast.unparse(ast.parse(str(v)))),
}


class RoundTripBelief(Discriminative):
    """``parse(format(v)) == v``: the candidate survives a serialization cycle.

    A value that changes when written out and read back is malformed in a way
    a shape check alone will not catch — ``"2024-2-30"`` parses as text and
    fails as a date; ``"1,249.50"`` looks numeric and is not.
    """

    necessary = True

    def __init__(self, fmt="json", **options):
        if fmt not in _ROUNDTRIPS:
            raise ValueError(
                f"unknown round-trip format {fmt!r}; known: {', '.join(sorted(_ROUNDTRIPS))}"
            )
        super().__init__(fmt, **options)
        self.fmt = fmt

    def judge(self, value, e, attr):
        if value is None:
            return None
        parse, canonical = _ROUNDTRIPS[self.fmt]
        try:
            got = parse(value)
            want = canonical(value)
        except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
            return 0.0, f"does not round-trip as {self.fmt}: {exc}"
        if values_equal(got, want):
            return 1.0, f"round-trips as {self.fmt}"
        return 0.0, f"round-trips as {self.fmt} to {got!r}, not {want!r}"


# --------------------------------------------------------------------------
# RegexBehaviorBelief
# --------------------------------------------------------------------------

class RegexBehaviorBelief(Discriminative):
    """The candidate *is* a regex, judged by what it matches.

    Graded: p is the fraction of the positive/negative examples the pattern
    gets right, so a nearly-correct pattern reads as nearly correct instead
    of simply failing.  A pattern that does not compile is a flat 0.
    """

    necessary = True

    def __init__(self, positives=(), negatives=(), **options):
        positives, negatives = tuple(positives), tuple(negatives)
        if not positives and not negatives:
            raise ValueError("RegexBehaviorBelief needs at least one example")
        super().__init__(positives, negatives, **options)
        self.positives = positives
        self.negatives = negatives

    def judge(self, value, e, attr):
        import re

        if not isinstance(value, str):
            return None
        try:
            rx = re.compile(value)
        except re.error as exc:
            return 0.0, f"does not compile: {exc}"
        wrong = []
        for sample in self.positives:
            if not rx.search(sample):
                wrong.append(f"misses {sample!r}")
        for sample in self.negatives:
            if rx.search(sample):
                wrong.append(f"matches {sample!r}")
        total = len(self.positives) + len(self.negatives)
        p = (total - len(wrong)) / total
        if wrong:
            return p, "; ".join(wrong[:3])
        return p, f"correct on all {total} examples"


# --------------------------------------------------------------------------
# PassesTestsBelief
# --------------------------------------------------------------------------

class PassesTestsBelief(Discriminative):
    """Each test callable is one check; the suite is graded.

    A test is any callable taking the candidate.  Returning ``False`` or
    raising ``AssertionError`` fails it; anything else passes.  A single
    test callable makes a perfectly good Belief on its own — this
    class exists because a *suite* is the common case.
    """

    necessary = True

    def __init__(self, test_fns, **options):
        tests = tuple(test_fns)
        if not tests:
            raise ValueError("PassesTestsBelief needs at least one test")
        for t in tests:
            if not callable(t):
                raise TypeError(f"test {t!r} is not callable")
        # identity hashes each test's source, so editing a test mints a new
        # belief rather than silently changing what the old one meant
        # (invariant 6).  Names stay human-readable for the reasons.
        super().__init__(tuple(code_identity(t) for t in tests), **options)
        self.tests = tests
        self.names = tuple(getattr(t, "__name__", repr(t)) for t in tests)

    def judge(self, value, e, attr):
        failures = []
        for test, name in zip(self.tests, self.names):
            try:
                ok = test(value)
            except AssertionError as exc:
                failures.append(f"{name}: {exc or 'assertion failed'}")
                continue
            except Exception as exc:  # a test that explodes is a failed test
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            if ok is False:
                failures.append(f"{name}: returned False")
        p = (len(self.tests) - len(failures)) / len(self.tests)
        if failures:
            return p, "; ".join(failures[:3])
        return p, f"passes all {len(self.tests)} tests"


# --------------------------------------------------------------------------
# ExecutesBelief — opt-in, subprocess, scratch dir, no network
# --------------------------------------------------------------------------

#: prelude injected ahead of the candidate: severs the obvious network paths
#: inside the child interpreter.  This is a guard rail, not a sandbox — a
#: determined piece of code gets out.  The real control is the opt-in.
_NO_NETWORK = """\
import socket as _s
def _denied(*a, **k):
    raise OSError("network disabled by thinair ExecutesBelief")
class _NoSocket(_s.socket):
    def __init__(self, *a, **k):
        raise OSError("network disabled by thinair ExecutesBelief")
_s.socket = _NoSocket
_s.create_connection = _denied
_s.socketpair = _denied
_s.getaddrinfo = _denied
del _s, _denied
"""


def _exec_allowed(explicit):
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(ALLOW_EXEC_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class ExecutesBelief(Discriminative):
    """The candidate is code that runs to completion without raising.

    **This runs model-generated code.**  It refuses to do so unless the user
    said as much, either per-belief (``ExecutesBelief(allow_exec=True)``) or per
    process (``THINAIR_ALLOW_EXEC=1``).  Consulting a belief that has neither
    raises ``ExecutionRefused`` — a programming error deserves an exception,
    not a quiet zero.

    The child runs in a fresh temporary directory with the parent environment
    stripped to a minimum and sockets disabled inside the interpreter.  Never
    put this in a default belief list.
    """

    necessary = True

    def __init__(self, timeout=5.0, *, allow_exec=None, language="python", **options):
        if language != "python":
            raise ValueError("only python is executable in v1")
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        # allow_exec is a runtime permission, not part of the belief's
        # identity: the same check with the same timeout is the same belief
        # whether or not this process is willing to run it (invariant 6).
        super().__init__(timeout, **options)
        self.timeout = timeout
        self.language = language
        self.allow_exec = allow_exec

    @property
    def allowed(self):
        return _exec_allowed(self.allow_exec)

    def judge(self, value, e, attr):
        if not isinstance(value, str) or not value.strip():
            return None
        if not self.allowed:
            raise ExecutionRefused(
                f"{self.id} would run model-generated code. Pass "
                f"ExecutesBelief(allow_exec=True) or set {ALLOW_EXEC_ENV}=1 to permit it."
            )
        with tempfile.TemporaryDirectory(prefix="thinair-exec-") as cwd:
            script = os.path.join(cwd, "candidate.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(_NO_NETWORK)
                fh.write("\n")
                fh.write(value)
            env = {
                "PATH": "",
                "HOME": cwd,
                "TMPDIR": cwd,
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", script],
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return 0.0, f"did not finish within {self.timeout:g}s"
            except OSError as exc:
                return 0.0, f"could not be run: {exc}"
        if proc.returncode == 0:
            return 1.0, "runs without raising"
        tail = (proc.stderr or "").strip().splitlines()
        last = tail[-1] if tail else f"exit status {proc.returncode}"
        return 0.0, f"raised: {last}"

