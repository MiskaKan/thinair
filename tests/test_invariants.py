"""The seven invariants of SPEC.md §2, each as a test that fails loudly.

The per-module suites check these where they apply; this file checks them
where they are structural -- properties of the package as a whole, which no
single module's tests would notice breaking.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import thinair
from thinair.beliefs import Belief, human, model
from thinair.ledger import Ledger, Opinion
from thinair.thing import Thing, contract, freeze, snapshot

from fakes import FakeEngine

PACKAGE = pathlib.Path(thinair.__file__).parent


def modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path.relative_to(PACKAGE).as_posix(), path


def imports_of(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            names.add(prefix)
            names.update(f"{prefix}.{a.name}" for a in node.names)
    return names


# --------------------------------------------------------------------------
# 1. one record shape
# --------------------------------------------------------------------------

def test_the_ledger_accepts_exactly_one_record_type():
    ledger = Ledger()
    for impostor in [{"belief": "b", "value": 1}, ("b", "e", "a", 1, 0.5), "text", 42]:
        with pytest.raises(TypeError):
            ledger.add(impostor)


def test_every_kind_of_belief_lands_the_same_shape():
    engine = FakeEngine([{"value": 1249.5, "p": 0.93}])

    class Invoice(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        source_text: str
        total = contract(float, extracted_from="source_text")

    ledger = Ledger()
    inv = Invoice(source_text="Total 1249.50 EUR", __ledger__=ledger)
    +inv.total
    ledger.add(Opinion(belief="human:bob", entity=inv.__entity__,
                       attr="total", value=1249.5, p=0.8))

    fields = {tuple(sorted(vars(o))) for o in ledger}
    assert len(fields) == 1
    assert {o.belief.split("[")[0].split("@")[0] for o in ledger} >= \
        {"human:jane", "human:bob"}


# --------------------------------------------------------------------------
# 2. the ledger is dumb, append-only memory
# --------------------------------------------------------------------------

def test_the_ledger_has_no_deletion_api():
    forbidden = {"delete", "remove", "pop", "clear", "purge", "update", "__delitem__"}
    assert forbidden.isdisjoint(dir(Ledger))


def test_the_ledger_implements_no_evaluative_math():
    """Layer 2 is deferred *in its entirety*: nothing in v1 implements it."""
    source = (PACKAGE / "ledger.py").read_text(encoding="utf-8")
    for word in ("credibility", "scoreboard", "evaluate(", "pooled", "similarity"):
        assert word not in source.lower()


def test_there_is_no_layer2_module():
    assert not (PACKAGE / "layer2.py").exists()


# --------------------------------------------------------------------------
# 3. beliefs are pure functions of a sealed snapshot
# --------------------------------------------------------------------------

def test_a_snapshot_cannot_be_mutated():
    engine = FakeEngine([{"value": 1, "p": 1}])

    class Simple(Thing):
        __beliefs__ = [model("m", engine=engine), human("jane")]
        note = contract(str)

    e = snapshot(Simple(note="hello", __ledger__=Ledger()))
    with pytest.raises(AttributeError):
        e.note = "goodbye"
    with pytest.raises(AttributeError):
        e.__entity__ = "somebody-else"


# --------------------------------------------------------------------------
# 4. freezing is a code-only capability
# --------------------------------------------------------------------------

def test_nothing_a_model_can_reach_emits_frozen_true():
    """The model's own modules must not contain the capability at all."""
    for name in ("engine/__init__.py", "engine/prompts.py", "engine/openai_compat.py",
                 "validators/form.py", "validators/grounding.py",
                 "validators/consistency.py", "validators/reference.py",
                 "validators/executable.py"):
        source = (PACKAGE / name).read_text(encoding="utf-8")
        assert "frozen=True" not in source, name


def test_the_episode_grammar_has_no_freeze_action():
    source = (PACKAGE / "engine" / "prompts.py").read_text(encoding="utf-8")
    grammar = source[source.index("_EPISODE_SYSTEM"):]
    assert "freeze" not in grammar.split("_EPISODE_SYSTEM")[1].split('"""')[1].lower() \
        or "cannot freeze" in grammar.lower()


def test_only_three_things_freeze():
    """Assignment, code execution, and the explicit verb."""
    engine = FakeEngine([{"value": 1249.5, "p": 0.93}])

    class Invoice(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        source_text: str
        total = contract(float, extracted_from="source_text")

    ledger = Ledger()
    inv = Invoice(source_text="Total 1249.50 EUR", __ledger__=ledger)
    +inv.total
    assert not any(o.frozen for o in ledger.opinions(attr="total"))

    freeze(inv.total)                                  # the explicit verb
    inv.vendor = "ACME Oy"                             # assignment
    assert [o.attr for o in ledger.opinions(frozen=True)] == \
        ["source_text", "total", "vendor"]


# --------------------------------------------------------------------------
# 5. vetoes are control flow; everything else is measurement
# --------------------------------------------------------------------------

def test_no_policy_invents_a_probability():
    from thinair.policy import Proposed, Threshold, Unanimous

    opinions = [Opinion(belief=f"b{i}", entity="e", attr="a", value=1.0,
                        p=[0.6, 1.0, 1.0][i], t=i + 1) for i in range(3)]
    for policy in (Proposed(), Unanimous(), Threshold(0.0)):
        value, p = policy.resolve(opinions, ("e", "a"), proposer="b0")
        assert p in {o.p for o in opinions}


# --------------------------------------------------------------------------
# 6. belief identity is durable
# --------------------------------------------------------------------------

def test_changing_any_component_mints_a_new_belief_id():
    ids = {
        model("qwen3-35b").id,
        model("qwen3-35b", temperature=0.7).id,
        model("qwen3-35b", think=True).id,
        model("gpt-oss-20b").id,
    }
    assert len(ids) == 4


def test_a_model_def_version_hashes_into_the_belief_id():
    """Retuning a folder mints fresh ids."""
    from thinair.models import resolve

    definition = resolve("qwen3-35b")
    assert definition.identity() in model("qwen3-35b").id


# --------------------------------------------------------------------------
# 7. no network in the core
# --------------------------------------------------------------------------

CORE = ("ledger.py", "thing.py", "policy.py", "debug.py",
        "validators/__init__.py", "validators/form.py", "validators/grounding.py",
        "validators/consistency.py", "validators/reference.py",
        "validators/executable.py")


@pytest.mark.parametrize("name", CORE)
def test_the_core_imports_nothing_from_engine_or_models(name):
    """Invariant 7 by import-graph assertion."""
    imports = imports_of(PACKAGE / name)
    assert not any("engine" in i for i in imports), (name, imports)
    assert not any(i.endswith("models") or ".models." in i for i in imports), name


@pytest.mark.parametrize("name", CORE)
def test_the_core_opens_no_sockets(name):
    imports = imports_of(PACKAGE / name)
    for network in ("urllib", "http", "socket", "requests", "ssl", "ftplib"):
        assert not any(i.split(".")[0] == network for i in imports), (name, network)


def test_only_model_belief_reaches_the_transport():
    """The LLM lives inside ModelBelief and nowhere else."""
    reaching = {
        name for name, path in modules()
        if any("engine" in i for i in imports_of(path))
        and not name.startswith("engine/") and not name.startswith("models/")
    }
    assert reaching <= {"beliefs.py"}


def test_a_deterministic_thing_needs_no_engine(monkeypatch):
    """Delete every ModelBelief and thinair still works."""
    import socket

    def forbidden(*a, **k):                            # pragma: no cover
        raise AssertionError("the core opened a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    class Ledgered(Thing):
        __beliefs__ = [human("jane")]
        total = contract(float, range=(0, 100))

    thing = Ledgered(total=42.0, __ledger__=Ledger())
    assert (+thing.total, ~thing.total) == (42.0, 1.0)
    assert +freeze(thing.total) == 42.0


# --------------------------------------------------------------------------
# the dependency rule
# --------------------------------------------------------------------------

def test_the_ledger_imports_nothing_from_the_rest_of_the_package():
    imports = imports_of(PACKAGE / "ledger.py")
    assert not any(i.startswith(".") or i.startswith("thinair") for i in imports)


def test_validators_import_only_ledger_and_beliefs_types():
    for name, path in modules():
        if not name.startswith("validators/"):
            continue
        internal = {i for i in imports_of(path) if i.startswith(".")}
        roots = {i.lstrip(".").split(".")[0] for i in internal if i.lstrip(".")}
        assert roots <= {"beliefs", "ledger", "form", "grounding", "consistency",
                         "reference", "executable"}, (name, roots)


def test_there_are_no_third_party_runtime_dependencies():
    """Stdlib only."""
    import sys

    stdlib = set(sys.stdlib_module_names)
    for name, path in modules():
        for imported in imports_of(path):
            root = imported.split(".")[0]
            if not root or imported.startswith("."):
                continue
            assert root in stdlib or root == "thinair", (name, imported)
