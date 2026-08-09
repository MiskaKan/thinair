"""The acceptance demo: *every claim in it must be true*.

The demo is executable documentation, so this file runs it and then checks
the claims its comments make -- one test per comment, in the order they
appear in ``examples/invoice.py``.
"""

from __future__ import annotations

import io

import pytest

from thinair import Ledger, Opinion, Thing, snapshot
from thinair.thing import Cell

from examples.invoice import build, demo, net_total_fn, offline


@pytest.fixture(scope="module")
def ran():
    """The demo, run once against scripted engines."""
    out = io.StringIO()
    import examples.invoice as demo_module

    original = demo_module.demo
    captured = {}

    def capturing(Invoice, net_total, out=out):
        captured["result"] = original(Invoice, net_total, out=out)
        return captured["result"]

    demo_module.demo = capturing
    try:
        offline()
    finally:
        demo_module.demo = original
    inv, net_total = captured["result"]
    return inv, net_total, out.getvalue()


def test_the_demo_runs(ran):
    inv, net_total, output = ran
    assert output


def test_a_kwarg_freezes_authored_by_the_human(ran):
    """``inv = Invoice(source_text=...)`` -- kwarg -> frozen, human:jane."""
    inv, _, _ = ran
    opinion = inv.source_text.__opinion__
    assert opinion.frozen is True and opinion.belief == "human:jane"


def test_plus_and_tilde_are_the_resolving_opinion(ran):
    """``print(+inv.total, ~inv.total)  # 1249.5 0.93``."""
    inv, _, output = ran
    assert output.splitlines()[0] == "1249.5 0.93"
    assert (+inv.total, ~inv.total) == (1249.5, 0.93)


def test_the_snapshot_is_sealed_and_inert(ran):
    """``e = snapshot(inv)  # a sealed, inert value -- reads never spend``."""
    inv, _, _ = ran
    e = snapshot(inv)
    with pytest.raises(AttributeError):
        e.total = 1
    assert isinstance(e.total, Opinion) or e.total is None


def test_the_panel_is_consultable_by_hand(ran):
    """``{b.id: b(e, "total") for b in inv.__beliefs__}``."""
    inv, _, output = ran
    e = snapshot(inv)
    survey = {b.id: b(e, "total") for b in inv.__beliefs__}
    assert len(survey) == len(inv.__beliefs__)
    assert all(getattr(b, "id", None) for b in inv.__beliefs__)
    assert "model:small-fast" in output                  # the survey was printed


def test_the_exit_ramp_yields_a_float_or_none(ran):
    """``total = +(inv.total @ float @ 0.9)``."""
    inv, _, _ = ran
    total = +(inv.total @ float @ 0.9)
    assert isinstance(total, float) and total == 1249.5
    assert +(inv.total @ float @ 0.999) is None


def test_assignment_makes_the_human_the_resolver(ran):
    """``inv.vendor = "ACME Oy"  # human speaks: frozen, p = 1.0``."""
    inv, _, _ = ran
    assert (+inv.vendor, ~inv.vendor) == ("ACME Oy", 1.0)
    assert inv.vendor.__opinion__.belief == "human:jane"
    assert inv.vendor.__opinion__.frozen is True


def test_freeze_pins_the_models_answer_keeping_p(ran):
    """``freeze(inv.total)  # pin the model's answer -- keeps p = 0.93``."""
    inv, _, _ = ran
    opinion = inv.total.__opinion__
    assert opinion.frozen is True and opinion.p == 0.93
    assert opinion.belief.startswith("model:small-fast")


def test_the_ledger_holds_every_voice_and_every_author(ran):
    """``inv.__ledger__.opinions(attr="total")``."""
    inv, _, _ = ran
    voices = inv.__ledger__.opinions(attr="total")
    authors = {o.belief for o in voices}
    assert any(a.startswith("model:") for a in authors)
    assert any(a.startswith("tokenSubset") for a in authors)
    assert any(a.startswith("range") for a in authors)
    assert sum(1 for o in voices if o.frozen) == 1       # only the pin


def test_source_renders_frozen_plain_and_believed_annotated(ran):
    inv, _, _ = ran
    text = inv.__source__
    assert 'class Invoice:' in text
    assert '"""An invoice document to be understood."""' in text
    assert "vendor = 'ACME Oy'" in text                  # frozen: plain


def test_a_bodiless_fn_is_served_by_the_model_contract_checked(ran):
    """``print(+net_total(1249.50, 0.24), ~net_total(1249.50, 0.24))``."""
    _, net_total, output = ran
    assert net_total.bodiless is True
    assert "1007.66 0.71" in output


def test_a_fixture_is_a_frozen_opinion(ran):
    """``freeze_call(net_total, (1249.50, 0.24), value=1007.66)``.

    The demo prints the same call twice: once served by the model at 0.71,
    once served by the fixture at 1.0.  The fixture lives in the demo's own
    scoped ledger and is gone with it, which is the whole point of a scoped fixture -- so the
    evidence is the two printed lines, not a call made afterwards.
    """
    _, net_total, output = ran
    assert output.rstrip().splitlines()[-2:] == ["1007.66 0.71", "1007.66 1.0"]


def test_the_demo_class_matches_the_plan():
    """The panel and the contracts are the demo's, verbatim."""
    Invoice = build()
    declared = Invoice.__contracts__
    assert set(declared) >= {"source_text", "total", "vendor", "line_items"}
    assert declared["line_items"].sums_to == "total"
    assert declared["total"].range == (0, 1e6)
    assert declared["total"].extracted_from == "source_text"

    ids = [b.id for b in Invoice.__beliefs__]
    assert ids[0].startswith("model:small-fast")
    assert ids[1].startswith("model:large-think")
    assert "/think" in ids[1]
    assert "human:jane" in ids
    assert any(i == "tokenSubsetBelief[source_text,kinds=[numbers]]" for i in ids)


def test_the_demo_reads_a_real_invoice_file():
    import os

    import examples.invoice as demo_module

    path = os.path.join(demo_module.HERE, "invoice.txt")
    with open(path) as fh:
        text = fh.read()
    assert "1 249,50" in text                            # the hard case, on purpose
    assert "999.00" in text and "250.50" in text
