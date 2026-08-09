"""history(): the record as commits; the thinair CLI: git on the record.

The derivation's strongest check is internal: an episode records the parent
tree it ran against at run time, and history() re-derives the same hash from
the tape alone.
"""

from __future__ import annotations

import pytest

from fakes import FakeEngine
from thinair import Thing, contract, corroborate, human, model
from thinair.cli import main
from thinair.evaluate import history
from thinair.ledger import Ledger
from thinair.store import SqliteLedger

SOURCE = "Widget 999.00\nShipping 250.50\nTotal 1249.50 EUR"


def scripted_run(ledger):
    """Assignment, settlement, episode, idempotent restatement, a note."""
    engine = FakeEngine([
        {"value": 1249.5, "p": 0.93},                       # total settles
        {"action": "return", "value": "flagged",            # an episode
         "changes": {"note": "overdue"}, "p": 0.8},
    ])
    second = FakeEngine([{"value": 1249.5, "p": 0.7}])

    class Invoice(Thing):
        """An invoice document to be understood."""
        __beliefs__ = [model("small-fast", engine=engine),
                       model("qwen3-35b", engine=second), human("jane")]
        source_text: str
        total = contract(float, extracted_from="source_text")
        note = contract(str)

    class Memo(Thing):
        __beliefs__ = [human("jane")]
        text: str

    inv = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    +inv.total
    # a second chain interleaves the tape, so the graph has lanes to draw
    Memo(__entity__="memo-1", __ledger__=ledger, text="pay this one first")
    inv.flag()                                              # episode commit
    inv.source_text = SOURCE                                # idempotent: no commit
    corroborate(inv, attrs=["total"])                       # a note, not a commit
    return inv


def test_history_derives_the_three_commit_kinds():
    ledger = Ledger()
    scripted_run(ledger)
    commits = history(ledger, entity="inv-1")
    kinds = [c["kind"] for c in commits]
    assert kinds == ["assign", "settle", "episode"]

    assign, settle, episode = commits
    assert assign["author"].startswith("human:jane")
    assert assign["changes"]["source_text"][2] is True      # frozen
    assert settle["changes"]["total"][0] == 1249.5
    assert settle["author"].startswith("model:small-fast")
    assert episode["message"].startswith("flag(")
    assert episode["changes"]["note"][0] == "overdue"


def test_an_episode_commit_re_derives_its_recorded_parent():
    """The tape alone reproduces the tree hash the run computed live."""
    ledger = Ledger()
    scripted_run(ledger)
    episode = history(ledger, entity="inv-1")[-1]
    assert episode["parent_matches"], (episode["parent"],
                                       episode["recorded_parent"])


def test_hashes_chain_like_git():
    ledger = Ledger()
    scripted_run(ledger)
    commits = history(ledger, entity="inv-1")
    for earlier, later in zip(commits, commits[1:]):
        assert later["parent"] == earlier["hash"]
    assert len({c["hash"] for c in commits}) == len(commits)


def test_corroborations_are_notes_and_replays_commit_nothing():
    ledger = Ledger()
    scripted_run(ledger)
    commits = history(ledger, entity="inv-1")
    settle = commits[1]
    assert [n[1] for n in settle.get("notes", [])] == ["total"]

    # the same program again: replay + idempotent assignment = no new commits
    scripted_run(ledger)
    assert len(history(ledger, entity="inv-1")) == len(commits)


def test_an_identical_re_settlement_is_not_a_commit():
    """Context changed, the belief re-read, the value held: the tree stood
    still, so no commit -- the reading is on record, the log stays quiet."""
    from thinair.ledger import Opinion

    ledger = Ledger()
    for exposure in ("aaaa11112222", "bbbb33334444"):    # two negotiations
        ledger.add(Opinion(belief="model:m", entity="e1", attr="size",
                           value=4, p=0.9,
                           meta={"model": "m", "round": 1,
                                 "exposure": exposure}))
    commits = history(ledger, entity="e1")
    assert len(commits) == 1                             # one transition


def test_a_pinned_cell_shows_as_freeze_not_code():
    """Engine metas carry a 'call' counter; only fn results are code."""
    from thinair.ledger import Opinion

    ledger = Ledger()
    ledger.add(Opinion(belief="model:m", entity="e1", attr="size", value=4,
                       p=0.9, frozen=True,
                       meta={"model": "m", "call": 7, "pinned": True}))
    assert history(ledger, entity="e1")[0]["kind"] == "freeze"


def test_freeze_is_idempotent_like_assignment():
    from test_second_opinions import corroborable
    from thinair import freeze

    inv, ledger, engine, _other = corroborable()
    +inv.total
    freeze(inv.total)
    pinned = len(ledger)
    frozen_again = freeze(inv.total)                     # the replayed program
    assert len(ledger) == pinned
    assert +frozen_again == 1249.5
    assert engine.call_count == 1


def test_a_vetoed_negotiation_is_not_a_commit():
    from thinair.beliefs import Discriminative

    class Never(Discriminative):
        necessary = True

        def judge(self, value, e, attr):
            return 0.0, "never"

    class Doomed(Thing):
        """A thing whose every candidate is refused."""
        __beliefs__ = [model("small-fast",
                             engine=FakeEngine([{"value": 1, "p": 0.9}])),
                       human("jane"), Never(id="law:never-cli")]
        size = contract(int)

    ledger = Ledger()
    thing = Doomed(__entity__="doomed-1", __ledger__=ledger)
    with pytest.raises(Exception):
        +thing.size
    assert history(ledger, entity="doomed-1") == []


# --------------------------------------------------------------------------
# the CLI, end to end
# --------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    path = tmp_path / "opinions.db"
    scripted_run(SqliteLedger(path))
    return str(path)


def test_log_and_oneline(store, capsys):
    assert main(["--store", store, "log"]) == 0
    out = capsys.readouterr().out
    assert "commit " in out and "Author: human:jane" in out
    assert "flag(" in out and "total ⇒ 1249.5" in out

    main(["--store", store, "log", "--oneline", "-n", "1"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "[episode]" in lines[0]      # newest first


def test_show_status_branch_blame(store, capsys):
    main(["--store", store, "log", "--oneline"])
    a_hash = capsys.readouterr().out.split()[0]

    main(["--store", store, "show", a_hash])
    shown = capsys.readouterr().out
    assert f"commit {a_hash}" in shown and "diff --thinair" in shown

    main(["--store", store, "status"])
    status = capsys.readouterr().out
    assert "commits" in status and "HEAD is at" in status

    main(["--store", store, "branch"])
    assert "inv-1" in capsys.readouterr().out

    main(["--store", store, "blame", "inv-1"])
    blame = capsys.readouterr().out
    assert "source_text" in blame and "(frozen)" in blame
    assert "total" in blame and "p=0.93" in blame


def test_cli_reads_a_json_archive(tmp_path, capsys):
    ledger = Ledger()
    scripted_run(ledger)
    path = tmp_path / "ledger.json"
    ledger.dump(path)
    main(["--store", str(path), "log", "--oneline"])
    assert "inv-1" in capsys.readouterr().out


def test_cli_refuses_a_missing_store(tmp_path):
    with pytest.raises(SystemExit):
        main(["--store", str(tmp_path / "nope.db"), "status"])


def test_help_needs_no_store(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)                        # nowhere near a store
    assert main(["help"]) == 0
    assert "log" in capsys.readouterr().out
    assert main(["help", "log"]) == 0
    assert "--decorate" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["help", "rebase"])
    assert not (tmp_path / ".thinair").exists()


def test_a_dog(store, capsys):
    """log --all --decorate --oneline --graph: the go-to, made home."""
    main(["--store", store, "log", "--all", "--decorate", "--oneline",
          "--graph"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert all("*" in line for line in lines)          # every commit a star
    assert sum("HEAD -> " in line for line in lines) == 1
    assert any("(memo-1)" in line for line in lines)   # every tip decorated
    assert any(line.startswith("| *") or line.startswith("* |")
               for line in lines)                      # parallel lanes drawn


def test_graph_lanes_thread_the_full_format(store, capsys):
    main(["--store", store, "log", "--graph"])
    out = capsys.readouterr().out
    assert "* commit" in out
    assert "| Author:" in out or "Author:" in out


# --------------------------------------------------------------------------
# evaluate and beliefs: consultation from the terminal
# --------------------------------------------------------------------------

def test_beliefs_lists_who_spoke_at_a_commit(store, capsys):
    main(["--store", store, "log", "--oneline"])
    head = capsys.readouterr().out.split()[0]
    main(["--store", store, "beliefs", head])
    out = capsys.readouterr().out
    assert "inv-1" in out and "model:small-fast" in out
    assert "reconstructible" in out and "human:jane" in out

    main(["--store", store, "beliefs"])
    assert "beliefs on record:" in capsys.readouterr().out


def test_evaluate_consults_against_head_and_records(store, capsys):
    from thinair.beliefs import restore_config, set_config

    engine = FakeEngine([{"value": 1249.5, "p": 0.66}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate", "small-fast"])
        out = capsys.readouterr().out
        assert "agrees" in out                     # total re-read the same
        assert "DIFFERS" in out                    # note read as 1249.5
        spent = engine.call_count
        assert spent >= 3                          # one consult per cell

        main(["--store", store, "evaluate", "small-fast"])
        again = capsys.readouterr().out
        assert "asked and answered" in again       # idempotent by exposure
        assert engine.call_count == spent
    finally:
        restore_config(None, previous)

    ledger = SqliteLedger(store)
    noted = [o for o in ledger.opinions(entity="inv-1", attr="total")
             if (o.meta or {}).get("corroboration")]
    assert any((o.meta or {}).get("at") for o in noted)


def test_evaluate_star_uses_every_reconstructible_belief(store, capsys):
    from thinair.beliefs import restore_config, set_config

    engine = FakeEngine([{"value": 1249.5, "p": 0.5}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate"])       # belief defaults to *
        out = capsys.readouterr().out
        assert "small-fast" in out and "qwen3-35b" in out
    finally:
        restore_config(None, previous)


def test_identical_histories_collapse_into_refs(tmp_path):
    """Entities are refs, not commit identity: byte-identical histories are
    ONE chain carrying every ref, exactly git's content addressing."""
    ledger = SqliteLedger(tmp_path / "o.db")

    class Card(Thing):
        __beliefs__ = [human("jane")]
        text: str

    Card(__entity__="card-1", __ledger__=ledger, text="same words")
    Card(__entity__="card-2", __ledger__=ledger, text="same words")
    commits = history(ledger)
    assert len(commits) == 1                             # one commit
    assert commits[0]["entities"] == ["card-1", "card-2"]
    assert sorted(commits[0]["heads"]) == ["card-1", "card-2"]


def test_diverging_histories_fork_where_they_diverge(tmp_path):
    ledger = SqliteLedger(tmp_path / "o.db")

    class Card(Thing):
        __beliefs__ = [human("jane")]
        text: str

    one = Card(__entity__="card-1", __ledger__=ledger, text="same words")
    two = Card(__entity__="card-2", __ledger__=ledger, text="same words")
    two.note = "only mine"                               # card-2 diverges
    commits = history(ledger)
    assert len(commits) == 2
    root, fork = commits
    assert root["entities"] == ["card-1", "card-2"]      # shared ancestry
    assert fork["entities"] == ["card-2"]
    assert fork["parent"] == root["hash"]                # a true fork
    assert root["heads"] == ["card-1"]                   # card-1's tip is root
    assert fork["heads"] == ["card-2"]


def test_diff_between_two_commits(store, capsys):
    main(["--store", store, "log", "--oneline", "inv-1"])
    lines = capsys.readouterr().out.strip().splitlines()
    newest, oldest = lines[0].split()[0], lines[-1].split()[0]

    main(["--store", store, "diff", f"{oldest}...{newest}"])
    out = capsys.readouterr().out
    assert "diff --thinair" in out
    assert "+ total = 1249.5   (p=0.93)" in out          # gained since root
    assert '+ note = "overdue"' in out                   # the episode's write
    assert "- " not in out.replace("--thinair", "")      # nothing was lost

    main(["--store", store, "diff", oldest])             # A against branch tip
    assert "+ total" in capsys.readouterr().out


def test_source_renders_the_tree_annotated(store, capsys):
    main(["--store", store, "source"])                   # HEAD by default
    out = capsys.readouterr().out
    assert 'source_text = "Widget 999.00' in out
    assert "total = 1249.5   # p=0.93 ← model:small-fast" in out
    assert 'note = "overdue"' in out                     # the episode's write


def test_show_lists_every_proposer_with_dash_for_the_silent(store, capsys):
    """The readings panel: who read this cell, and who never did."""
    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]
    main(["--store", store, "show", settle])
    out = capsys.readouterr().out
    assert "readings:" in out
    assert "model:small-fast" in out and "resolving" in out
    assert "model:qwen3-35b" in out and "corroboration" in out
    assert "human:jane" in out
    silent = [l for l in out.splitlines() if l.strip().endswith(" -")
              or l.strip().endswith("-")]
    assert any("human:jane" in l for l in silent)      # jane never read total


def test_evaluate_is_cached_into_the_readings_panel(store, capsys):
    """thinair evaluate at a commit -> the reading is on record from then
    on: visible in show, and never asked again under the same exposure."""
    from thinair.beliefs import restore_config, set_config

    engine = FakeEngine([{"value": 89.9, "p": 0.6}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate", "deepseek-v4-flash"])
        capsys.readouterr()
        spent = engine.call_count

        main(["--store", store, "log", "--oneline", "inv-1"])
        settle = [l for l in capsys.readouterr().out.splitlines()
                  if "[settle]" in l][0].split()[0]
        main(["--store", store, "show", settle])
        out = capsys.readouterr().out
        assert "model:deepseek-v4-flash" in out        # the cache, visible
        assert "89.9" in out and "corroboration" in out

        main(["--store", store, "evaluate", "deepseek-v4-flash"])
        assert "asked and answered" in capsys.readouterr().out
        assert engine.call_count == spent              # cached from there on
    finally:
        restore_config(None, previous)


def test_ground_prints_the_grounding_pipe_pure(tmp_path, monkeypatch, capsys):
    """No store needed, none created, nothing on stdout but the file."""
    monkeypatch.chdir(tmp_path)
    assert main(["ground"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# thinair — The Measurement Space")
    assert "Pillar I" in out and "Part 2" in out
    assert not (tmp_path / ".thinair").exists()


def test_evaluate_refuses_a_read_only_archive(tmp_path):
    ledger = Ledger()
    scripted_run(ledger)
    path = tmp_path / "ledger.json"
    ledger.dump(path)
    with pytest.raises(SystemExit):
        main(["--store", str(path), "evaluate"])


# --------------------------------------------------------------------------
# revisions resolve like git's: HEAD, branch names, hash prefixes
# --------------------------------------------------------------------------

def test_show_resolves_head_and_branch_names(store, capsys):
    main(["--store", store, "show"])                     # HEAD is the default
    assert "flag(" in capsys.readouterr().out            # newest: the episode

    main(["--store", store, "show", "HEAD"])
    assert "flag(" in capsys.readouterr().out

    main(["--store", store, "show", "memo-1"])           # a branch -> its tip
    assert "pay this one first" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["--store", store, "show", "no-such-rev"])


def test_evaluate_takes_commit_then_belief(store, capsys):
    from thinair.beliefs import restore_config, set_config

    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]

    engine = FakeEngine([{"value": 1249.5, "p": 0.66}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate", settle, "small-fast"])
        out = capsys.readouterr().out
        assert f"@ {settle}" in out                      # that commit, not HEAD
        assert "small-fast" in out and "qwen3-35b" not in out
    finally:
        restore_config(None, previous)


def test_evaluate_shares_the_reading_across_a_commits_refs(tmp_path, capsys):
    """A collapsed commit IS the same content: its refs share one evaluation
    instead of spending one per branch name."""
    from thinair.beliefs import restore_config, set_config

    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": "big", "p": 0.8}])

    class Card(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        text: str
        size = contract(str)

    for name in ("card-a", "card-b", "card-c"):
        +Card(__entity__=name, __ledger__=ledger, text="same words").size
    assert len(history(ledger)) == 2                     # one shared chain

    second = FakeEngine([{"value": "big", "p": 0.7}])
    previous = set_config(None, engine=second)
    try:
        main(["--store", str(tmp_path / "o.db"), "evaluate"])
        spent = second.call_count
        assert spent == 2                                # size + text: once,
                                                         # not once per ref
        main(["--store", str(tmp_path / "o.db"), "evaluate"])
        assert "asked and answered" in capsys.readouterr().out
        assert second.call_count == spent
    finally:
        restore_config(None, previous)


# --------------------------------------------------------------------------
# branches delete like git's: the ref goes, shared commits survive
# --------------------------------------------------------------------------

def test_branch_delete_removes_the_ref(store, capsys):
    main(["--store", store, "branch", "-d", "memo-1"])
    assert "Deleted branch memo-1" in capsys.readouterr().out

    main(["--store", store, "log", "--all", "--oneline"])
    out = capsys.readouterr().out
    assert "memo-1" not in out and "inv-1" in out        # the rest stands

    with pytest.raises(SystemExit):
        main(["--store", store, "branch", "-d", "memo-1"])   # already gone


def test_branch_delete_refuses_archives(tmp_path):
    ledger = Ledger()
    scripted_run(ledger)
    path = tmp_path / "ledger.json"
    ledger.dump(path)
    with pytest.raises(SystemExit):
        main(["--store", str(path), "branch", "-d", "memo-1"])


# --------------------------------------------------------------------------
# expectations and deviation: declared on the contract, judged in the record
# --------------------------------------------------------------------------

def test_expectations_mark_the_log_and_never_gate(tmp_path, monkeypatch,
                                                  capsys):
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 42.0, "p": 0.4}])

    class Reading(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        value = contract(float, p=0.9)

    r = Reading(__entity__="r-1", __ledger__=ledger)
    assert +r.value == 42.0                              # marks, never gates

    commit = history(ledger, entity="r-1")[-1]
    assert commit["expect"]["value"]["p"] == [0.9, 1.0]

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    out = capsys.readouterr().out
    assert "p 0.4" in out
    assert "\x1b[31m" in out                             # below the bar: red


def test_deviation_shows_beside_the_stated_p(store, monkeypatch, capsys):
    """inv-1's total: resolved at 0.93, corroborated at 0.7 -- the spread
    is the interesting part, so the log carries it by default."""
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0]
    assert "±0." in settle

    commit = [c for c in history(SqliteLedger(store), entity="inv-1")
              if c["kind"] == "settle"][0]
    assert commit["consensus"]["total"]["n"] >= 2
    assert commit["consensus"]["total"]["dev"] > 0


def test_dissent_paints_the_stated_p_yellow(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 10.0, "p": 0.9}])
    other = FakeEngine([{"value": 99.0, "p": 0.8}])      # holds another value

    class Box(Thing):
        __beliefs__ = [model("small-fast", engine=engine),
                       model("qwen3-35b", engine=other), human("jane")]
        size = contract(float)

    b = Box(__entity__="box-1", __ledger__=ledger)
    +b.size
    corroborate(b, attrs=["size"])                       # the dissenting note

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    line = [l for l in capsys.readouterr().out.splitlines()
            if "[settle]" in l][0]
    assert "\x1b[33m" in line                            # dissent: yellow

    commit = history(ledger, entity="box-1")[-1]
    assert commit["consensus"]["size"]["dissent"] == 1


def test_max_deviation_is_declarable_and_violations_paint_red(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 5.0, "p": 0.95}])
    other = FakeEngine([{"value": 5.0, "p": 0.5}])       # agrees, weakly

    class Box(Thing):
        __beliefs__ = [model("small-fast", engine=engine),
                       model("qwen3-35b", engine=other), human("jane")]
        size = contract(float, deviation=0.1)

    b = Box(__entity__="box-2", __ledger__=ledger)
    +b.size
    corroborate(b, attrs=["size"])                       # spread: 0.95 vs 0.5

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    line = [l for l in capsys.readouterr().out.splitlines()
            if "[settle]" in l][0]
    assert "\x1b[31m" in line                            # over the declared max


def test_eager_contracts_resolve_at_construction():
    engine = FakeEngine([{"value": 7.0, "p": 0.9}])

    class Gauge(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        level = contract(float, eager=True)

    g = Gauge(__entity__="g-1", __ledger__=Ledger())
    assert engine.call_count == 1                        # spent in __init__
    assert +g.level == 7.0
    assert engine.call_count == 1                        # standing, not re-asked


# --------------------------------------------------------------------------
# the matrix, the roster, the docstrings
# --------------------------------------------------------------------------

def test_show_paints_the_belief_by_attribute_matrix(store, monkeypatch,
                                                    capsys):
    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", store, "show", settle])
    out = capsys.readouterr().out
    assert "matrix:" in out
    assert "model:small-fast" in out and "model:qwen3-35b" in out
    assert "\x1b[32m" in out                             # agreement in green


def test_ground_appends_the_builtin_roster(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["ground"])
    out = capsys.readouterr().out
    assert "built-in beliefs" in out
    assert "`TokenSubset`" in out and "`Range`" in out
    assert not (tmp_path / ".thinair").exists()


def test_method_docstrings_reach_the_snapshot():
    from thinair.thing import snapshot

    class Doc(Thing):
        __beliefs__ = [human("jane")]

        def approve(self):
            """Approve the invoice for payment."""

    d = Doc(__entity__="doc-1", __ledger__=Ledger())
    e = snapshot(d)
    assert any("# Approve the invoice for payment" in line
               for line in e.__methods__)


# --------------------------------------------------------------------------
# similarity: how far apart readings landed, not just whether they agree
# --------------------------------------------------------------------------

def test_similarity_is_licensed_by_type():
    from thinair.evaluate import similarity

    assert similarity("frustrated", "frustrated") == 1.0
    assert 0.0 < similarity("frustrated", "frustration") < 1.0
    assert similarity("frustrated", "negative") < 0.3    # barely overlap
    assert similarity(89.9, 89.9) == 1.0
    assert similarity(10.0, 99.0) == pytest.approx(1 - 89 / 99)
    assert similarity(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert similarity("high", 4) == 0.0                  # unlike types
    assert similarity(True, 1) == 0.0                    # bool is not 1


def test_dissent_shows_the_value_overlap_in_the_log(tmp_path, monkeypatch,
                                                    capsys):
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 10.0, "p": 0.9}])
    other = FakeEngine([{"value": 99.0, "p": 0.8}])

    class Box(Thing):
        __beliefs__ = [model("small-fast", engine=engine),
                       model("qwen3-35b", engine=other), human("jane")]
        size = contract(float)

    b = Box(__entity__="box-3", __ledger__=ledger)
    +b.size
    corroborate(b, attrs=["size"])

    commit = history(ledger, entity="box-3")[-1]
    assert commit["consensus"]["size"]["similarity"] == \
        pytest.approx(1 - 89 / 99)

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    line = [l for l in capsys.readouterr().out.splitlines()
            if "[settle]" in l][0]
    assert "~0.1" in line                                # the second metric

    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    out = capsys.readouterr().out
    assert ", ~0.1" in out                               # on the note too


def test_show_renders_the_whole_tree_with_changes_highlighted(store, capsys):
    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]
    main(["--store", store, "show", settle])
    out = capsys.readouterr().out
    assert "+ total = 1249.5" in out                     # the change, marked
    assert '  source_text = "Widget 999.00' in out       # context, unmarked
    assert "+ source_text" not in out


def test_show_pools_readings_across_a_commits_refs(tmp_path, capsys):
    """One commit, one panel: refs are pointers, so the readings and the
    matrix pool their opinions instead of repeating per branch name."""
    from thinair.beliefs import restore_config, set_config

    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": "big", "p": 0.8}])

    class Card(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        text: str
        size = contract(str)

    for name in ("card-x", "card-y"):
        +Card(__entity__=name, __ledger__=ledger, text="same words").size

    second = FakeEngine([{"value": "big", "p": 0.7}])
    previous = set_config(None, engine=second)
    try:
        main(["--store", str(tmp_path / "o.db"), "evaluate"])
        capsys.readouterr()
    finally:
        restore_config(None, previous)

    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    out = capsys.readouterr().out
    assert "[card-x]" not in out and "[card-y]" not in out
    corroborating = [l for l in out.splitlines() if "corroboration" in l]
    assert len(corroborating) == 1                       # pooled, not per ref


def test_log_has_no_branch_column_and_decorates_by_default(store, capsys):
    """Membership is ancestry: no per-commit entity label; refs appear only
    as tip decorations, which are on by default like git's."""
    main(["--store", store, "log", "--oneline"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert all(line.split()[1].startswith(("[", "("))    # kind or decoration:
               for line in lines)                        # never an entity name
    assert any("(HEAD -> " in line for line in lines)    # decorated, unasked

    main(["--store", store, "log", "--oneline", "--no-decorate"])
    out = capsys.readouterr().out
    assert "HEAD -> " not in out and "(inv-1)" not in out


def test_matrix_separates_out_of_scope_from_never_asked(store, capsys):
    """An empty matrix cell says why: '-' the belief is scoped elsewhere
    and cannot speak; '?' it could be asked and never was."""
    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]
    main(["--store", store, "show", settle])
    out = capsys.readouterr().out
    scoped = [l for l in out.splitlines() if "@total" in l][0]
    assert "-" in scoped and "?" not in scoped           # total-only belief
    jane = [l for l in out.splitlines()
            if l.strip().startswith("human:jane")][0]
    assert "?" in jane                                   # askable, unasked
