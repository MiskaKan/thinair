"""References in the record, and the graph derived from them.

``_plain`` and call rendering reduce Things and Cells to values -- right for
comparison, destructive for provenance.  The ``refs`` stamp keeps what the
reduction discards; ``evaluate.graph`` turns the record into nodes and typed
edges; ``lineage`` walks up, ``invalidated`` walks down.  The end-to-end
tests go through real Things and a scripted engine, so the stamps and the
derivation are proven together.
"""

from __future__ import annotations

from fakes import FakeEngine
from thinair import Thing, human, model
from thinair.evaluate import graph, invalidated, lineage
from thinair.ledger import Ledger, Opinion
from thinair.thing import references


class Doc(Thing):
    __beliefs__ = [human("desk")]
    title: str


def doc(ledger, entity="doc-1", title="Q3 report"):
    return Doc(__entity__=entity, __ledger__=ledger, title=title)


# --------------------------------------------------------------------------
# what a value carries
# --------------------------------------------------------------------------

def test_references_collects_things_cells_and_containers():
    ledger = Ledger()
    d = doc(ledger)
    assert references(d) == ["doc-1"]
    assert references(d.title) == ["doc-1#title"]        # a Cell is an address
    assert references([d, {"k": d.title}, "plain"]) == ["doc-1", "doc-1#title"]
    assert references(d.title) + references(d.title) != []  # deterministic
    assert references(object()) == []                    # no identity, no ref
    assert references([d, d]) == ["doc-1"]               # deduped


# --------------------------------------------------------------------------
# the three crossing points stamp
# --------------------------------------------------------------------------

def test_assignment_stamps_refs():
    ledger = Ledger()
    d = doc(ledger)

    class Employee(Thing):
        __beliefs__ = [human("desk")]
        name: str

    emp = Employee(__entity__="emp-1", __ledger__=ledger, name="Ada")
    emp.assigned = d
    emp.summary = d.title                                # a Cell: value kept, ref too
    assert ledger.latest("emp-1", "assigned").meta["refs"] == ["doc-1"]
    kept = ledger.latest("emp-1", "summary")
    assert kept.value == "Q3 report" and kept.meta["refs"] == ["doc-1#title"]
    assert "refs" not in ledger.latest("emp-1", "name").meta   # lean when plain


def test_an_episode_stamps_the_arguments_it_saw():
    ledger = Ledger()
    engine = FakeEngine([{"action": "return", "value": "ok", "p": 0.9}])

    class Employee(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("desk")]
        name: str

    emp = Employee(__entity__="emp-1", __ledger__=ledger, name="Ada")
    d = doc(ledger)
    result = emp.operate(d, hint=d.title)
    episode_cell = result.__cell__[0]
    recorded = ledger.latest(episode_cell, "result")
    assert recorded.meta["refs"] == ["doc-1", "doc-1#title"]


def test_a_pure_fn_stamps_the_arguments_it_was_handed():
    from thinair import fn
    ledger = Ledger()
    d = doc(ledger)

    @fn(ledger=ledger)
    def word_count(document) -> int:
        return len((+document.title).split())

    got = word_count(d)
    assert +got == 2
    recorded = ledger.latest(got.__cell__[0], "result")
    assert recorded.meta["refs"] == ["doc-1"]


# --------------------------------------------------------------------------
# the graph, derived
# --------------------------------------------------------------------------

def episode_ledger():
    """emp-1 operates on doc-1; the record carries every edge kind."""
    ledger = Ledger()
    engine = FakeEngine([{"action": "return", "value": "prioritize", "p": 0.9}])

    class Employee(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("desk")]
        name: str

    emp = Employee(__entity__="emp-1", __ledger__=ledger, name="Ada")
    d = doc(ledger)
    result = emp.operate(d)
    return ledger, result.__cell__[0]                    # the episode entity


def by_kind(g, kind):
    return [(e["src"], e["dst"]) for e in g["edges"] if e["kind"] == kind]


def test_graph_derives_every_edge_kind():
    ledger, episode_entity = episode_ledger()
    g = graph(ledger)
    assert "emp-1" in g["entities"] and "doc-1" in g["entities"]
    assert (f"{episode_entity}#result", "doc-1") in by_kind(g, "ref")
    assert (episode_entity, "emp-1") in by_kind(g, "host")   # from provenance, exactly
    authored = by_kind(g, "authored")
    assert any(src.startswith("model:") and dst == f"{episode_entity}#result"
               for src, dst in authored)


def test_graph_child_edges_and_exposure_groups():
    ledger = Ledger()
    ledger.add(Opinion(belief="b", entity="n01", attr="country", value="FI",
                       p=0.9, meta={"model": "m", "exposure": "aaa111bbb222"}))
    ledger.add(Opinion(belief="b", entity="n01#country", attr="checked",
                       value=True, p=0.9, meta={"exposure": "aaa111bbb222"}))
    g = graph(ledger)
    assert ("n01#country", "n01") in by_kind(g, "child")
    assert len(g["exposures"]["aaa111bbb222"]) == 2      # shared context, grouped


def test_lineage_walks_up_and_invalidated_walks_down():
    ledger, episode_entity = episode_ledger()
    g = graph(ledger)
    result_cell = f"{episode_entity}#result"

    up = lineage(g, result_cell)
    assert "doc-1" in up                                  # what the call was handed
    assert "emp-1" in up                                  # who it ran on

    down = invalidated(g, "doc-1")
    assert result_cell in down                            # staleness, as a query
    assert "emp-1#name" not in down                       # unrelated cells stay put

    assert result_cell in invalidated(g, "emp-1")         # host state moved
    assert invalidated(g, "doc-1#title") == []            # nothing referenced the cell
