"""The model folder conformance suite.

Every registered ``ModelDef`` is table-checked here: globs compile and do not
overlap, defaults are in range, think payloads are disjoint, quirks are
known.  **A contributed folder is proven with zero new test code** -- that is
the bargain the registry makes with contributors, and this file is the bargain.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib

import pytest

from thinair import models as M
from thinair.models import (
    GENERIC,
    KNOWN_QUIRKS,
    STRUCTURED_OUTPUT_MODES,
    ModelDef,
    register,
    resolve,
)

REGISTERED = sorted(M.registry().items()) if hasattr(M, "registry") else \
    sorted(M._REGISTRY.items())
NAMES = [name for name, _ in REGISTERED]
DEFS = [definition for _, definition in REGISTERED]
ALL = DEFS + [GENERIC]


def test_there_are_folders_to_check():
    assert len(DEFS) >= 2, "the registry wants at least two real folders plus _template"


def test_the_template_folder_is_not_registered():
    """``_template`` is a copy-me starting point, not a model."""
    assert "_template" not in NAMES
    assert (pathlib.Path(M.__file__).parent / "_template" / "model.py").exists()


# --------------------------------------------------------------------------
# the table-check, one parametrization per registered def
# --------------------------------------------------------------------------

@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_globs_compile(definition):
    for pattern in definition.match:
        assert isinstance(pattern, str) and pattern
        fnmatch.translate(pattern)                    # raises on a bad glob


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_defaults_are_in_range(definition):
    temperature = definition.defaults.get("temperature")
    if temperature is not None:
        assert 0.0 <= temperature <= 2.0
    max_tokens = definition.defaults.get("max_tokens")
    if max_tokens is not None:
        assert isinstance(max_tokens, int) and max_tokens > 0


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_think_payloads_are_disjoint(definition):
    on = definition.think.get("on", {}) if definition.think else {}
    off = definition.think.get("off", {}) if definition.think else {}
    assert on.items() != off.items() or not on
    for key in set(on) & set(off):
        assert on[key] != off[key], key


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_quirks_are_known(definition):
    assert set(definition.quirks) <= KNOWN_QUIRKS


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_structured_output_is_a_known_mode(definition):
    assert definition.structured_output in STRUCTURED_OUTPUT_MODES


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_a_version_is_declared(definition):
    """``ModelDef.version`` hashes into every belief id (invariant 6)."""
    assert definition.version and isinstance(definition.version, str)
    assert definition.identity().endswith(f"/v{definition.version}")


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_the_prompt_dialect_exists(definition):
    from thinair.engine.prompts import DIALECTS

    assert definition.prompt_dialect in DIALECTS


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_a_def_is_data_not_behaviour(definition):
    """Folders are data for the one transport."""
    for value in vars(definition).values():
        assert not callable(value), value


@pytest.mark.parametrize("definition", ALL, ids=lambda d: d.name)
def test_a_def_is_immutable(definition):
    with pytest.raises(Exception):
        definition.version = "99"


# --------------------------------------------------------------------------
# claims do not overlap
# --------------------------------------------------------------------------

def test_no_two_folders_claim_the_same_name():
    seen = {}
    for name, definition in REGISTERED:
        for pattern in definition.match:
            assert pattern not in seen, f"{name} and {seen.get(pattern)} both claim {pattern}"
            seen[pattern] = name


def test_no_glob_swallows_another_folders_literal_claim():
    for name, definition in REGISTERED:
        literals = [p for p in definition.match if "*" not in p and "?" not in p]
        for other_name, other in REGISTERED:
            if other_name == name:
                continue
            for glob in (p for p in other.match if "*" in p or "?" in p):
                for literal in literals:
                    assert not fnmatch.fnmatch(literal, glob), \
                        f"{other_name}'s {glob!r} swallows {name}'s {literal!r}"


def test_registering_an_overlapping_builtin_raises():
    clash = ModelDef(match=tuple(DEFS[0].match), version="1", name="clash")
    with pytest.raises(ValueError):
        register(clash, "clash")


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,definition", REGISTERED, ids=NAMES)
def test_every_folder_resolves_its_own_claims(name, definition):
    for pattern in definition.match:
        probe = pattern.replace("*", "x").replace("?", "x")
        assert resolve(probe) is definition, (name, pattern, probe)


def test_an_unknown_model_falls_back_to_the_generic_def():
    """Unknown models work, and a folder only ever improves one."""
    assert resolve("some-model-nobody-has-heard-of") is GENERIC
    assert GENERIC.structured_output in STRUCTURED_OUTPUT_MODES


def test_the_generic_def_is_not_in_the_registry():
    assert GENERIC not in DEFS


def test_the_most_specific_claim_wins():
    literal = ModelDef(match=("probe-model-7b",), version="1", name="specific")
    glob = ModelDef(match=("probe-model-*",), version="1", name="broad")
    assert literal.specificity("probe-model-7b") > glob.specificity("probe-model-7b")


def test_a_longer_glob_beats_a_shorter_one():
    long = ModelDef(match=("probe-model-7b-*",), version="1", name="long")
    short = ModelDef(match=("probe-*",), version="1", name="short")
    target = "probe-model-7b-instruct"
    assert long.specificity(target) > short.specificity(target)


def test_a_def_that_claims_nothing_scores_nothing():
    definition = ModelDef(match=("probe-x",), version="1", name="x")
    assert definition.claims("something-else") is False


# --------------------------------------------------------------------------
# user-side folders
# --------------------------------------------------------------------------

def test_a_user_folder_is_discovered_and_may_override_a_builtin(tmp_path, monkeypatch):
    folder = tmp_path / "my_model"
    folder.mkdir()
    (folder / "__init__.py").write_text("")
    (folder / "model.py").write_text(
        "from thinair.models import ModelDef\n"
        "MODEL = ModelDef(match=('my-model', 'qwen3-35b'), version='7',\n"
        "                 defaults=dict(temperature=0.9))\n")

    monkeypatch.setenv("THINAIR_MODELS_PATH", str(tmp_path))
    mine = resolve("my-model", models_path=[str(tmp_path)])
    assert mine.version == "7"
    # an external def may override a built-in claim
    assert resolve("qwen3-35b", models_path=[str(tmp_path)]).version == "7"


def test_a_user_folder_changes_the_belief_id(tmp_path):
    from thinair.beliefs import model

    folder = tmp_path / "retuned"
    folder.mkdir()
    (folder / "__init__.py").write_text("")
    (folder / "model.py").write_text(
        "from thinair.models import ModelDef\n"
        "MODEL = ModelDef(match=('retuned-7b',), version='3')\n")

    belief = model("retuned-7b", models_path=[str(tmp_path)])
    assert "/v3" in belief.id


# --------------------------------------------------------------------------
# validation at construction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(match=()),                                        # claims nothing
    dict(match=("x",), structured_output="telepathy"),
    dict(match=("x",), quirks=("no-such-quirk",)),
    dict(match=("x",), defaults=dict(temperature=5.0)),
    dict(match=("x",), defaults=dict(max_tokens=0)),
    dict(match=("x",), version=""),
    dict(match=("x",), think=dict(on={"a": 1}, off={"a": 1})),
])
def test_a_malformed_def_raises_at_import(kwargs):
    kwargs.setdefault("version", "1")
    with pytest.raises((ValueError, TypeError)):
        ModelDef(**kwargs)


# --------------------------------------------------------------------------
# live smoke, behind an env-var guard
# --------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("THINAIR_LIVE"),
                    reason="set THINAIR_LIVE=1 (and THINAIR_BASE_URL/MODEL) to smoke a real endpoint")
def test_live_smoke():
    from thinair import Thing, contract, human
    from thinair.beliefs import model
    from thinair.ledger import Ledger

    name = os.environ.get("THINAIR_MODEL", "gpt-oss-20b")

    class Note(Thing):
        """A short note to be understood."""

        __beliefs__ = [model(name), human("smoke")]
        source_text: str
        total = contract(float, extracted_from="source_text")

    note = Note(source_text="The bill came to 1249.50 euros.", __ledger__=Ledger())
    assert +note.total == pytest.approx(1249.5, abs=0.01)
    assert 0.0 <= ~note.total <= 1.0
