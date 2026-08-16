"""history(): the record as commits; the thinair CLI: git on the record.

The derivation's strongest check is internal: an episode records the parent
tree it ran against at run time, and history() re-derives the same hash from
the tape alone.
"""

from __future__ import annotations

import pytest

from fakes import FakeEngine
from thinair import Thing, corroborate, human, model
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
        total = Thing(float, extracted_from="source_text")
        note = Thing(str)

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


def test_adoption_is_idempotent_like_any_assignment():
    from test_second_opinions import corroborable

    inv, ledger, engine, _other = corroborable()
    inv.total = +inv.total                               # adopt: frozen
    pinned = len(ledger)
    inv.total = +inv.total                               # the replayed program
    assert len(ledger) == pinned
    assert +inv.total == 1249.5
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
        size = Thing(int)

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
    assert "total" in blame and "p 0.93 ±" in blame     # consensus view


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
    # every commit gets a mark: '*' along a chain, '+' at a chain's root
    assert all("*" in line or "+" in line for line in lines)
    assert sum("HEAD -> " in line for line in lines) == 1
    assert any("(memo-1)" in line for line in lines)   # every tip decorated
    assert any(line.startswith("| *") or line.startswith("* |")
               or line.startswith("| +") or line.startswith("+ |")
               for line in lines)                      # parallel lanes drawn


def test_graph_lanes_thread_the_full_format(store, capsys):
    main(["--store", store, "log", "--graph"])
    out = capsys.readouterr().out
    assert "* commit" in out
    assert "| Author:" in out or "Author:" in out


# --------------------------------------------------------------------------
# evaluate and beliefs: consultation from the terminal
# --------------------------------------------------------------------------

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
        assert spent >= 2                          # one consult per believed
                                                   # cell; frozen ones skipped

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
    """No store needed, none created, nothing on stdout but the text."""
    monkeypatch.chdir(tmp_path)
    assert main(["ground"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# thinair — The Measurement Space")
    assert "Pillar I" in out
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
        header = out.splitlines()[0]                     # that commit, not HEAD,
        assert header == f"evaluate small-fast @ {settle} (inv-1)"
        assert "qwen3-35b" not in header                 # and only that belief
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
        size = Thing(str)

    for name in ("card-a", "card-b", "card-c"):
        +Card(__entity__=name, __ledger__=ledger, text="same words").size
    assert len(history(ledger)) == 2                     # one shared chain

    second = FakeEngine([{"value": "big", "p": 0.7}])
    previous = set_config(None, engine=second)
    try:
        main(["--store", str(tmp_path / "o.db"), "evaluate"])
        spent = second.call_count
        assert spent == 1                                # size once, not per
                                                         # ref; text is frozen
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
        value = Thing(float, p=0.9)

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
        size = Thing(float)

    b = Box(__entity__="box-1", __ledger__=ledger)
    +b.size
    corroborate(b, attrs=["size"])                       # the dissenting note

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    line = [l for l in capsys.readouterr().out.splitlines()
            if "[settle]" in l][0]
    signature = line.split("[settle] ")[1]               # past the yellow hash
    assert any(code in signature for code in
               ("\x1b[92m", "\x1b[33m", "\x1b[2;31m"))  # dissent: mid-ramp

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
        size = Thing(float, deviation=0.1)

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
        level = Thing(float, eager=True)

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
    assert "\x1b[92m" in out                           # agreement in the
                                                         # one green


def test_ground_serves_the_builtin_roster_on_demand(tmp_path, monkeypatch,
                                                    capsys):
    """The roster is reference, not orientation: its own section (and
    --full), while the default tier says where to find it."""
    monkeypatch.chdir(tmp_path)
    main(["ground", "roster"])
    out = capsys.readouterr().out
    assert out.startswith("## Appendix: built-in beliefs")
    assert "`TokenSubsetBelief`" in out and "`RangeBelief`" in out

    main(["ground"])
    default = capsys.readouterr().out
    assert "`TokenSubsetBelief`" not in default
    assert "ground roster" in default                # discoverable from it
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
        size = Thing(float)

    b = Box(__entity__="box-3", __ledger__=ledger)
    +b.size
    corroborate(b, attrs=["size"])

    commit = history(ledger, entity="box-3")[-1]
    assert commit["consensus"]["size"]["similarity"] == \
        pytest.approx(1 - 89 / 99)

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    line = [l for l in capsys.readouterr().out.splitlines()
            if "[settle]" in l][0]
    assert "~" not in line                               # overlap is the
    assert "±" in line                                   # color, not text

    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    out = capsys.readouterr().out
    assert ", ~0.1" in out                               # detail on the note


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
        size = Thing(str)

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


def test_matrix_merges_scoped_rows_and_marks_the_unreachable(store, capsys):
    """One row per mechanism: scoped wrappers pool into their inner
    belief's row (the column names the attribute).  An empty cell says
    why: '?' the client could ask; 'x' it has no way to call this one."""
    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]
    main(["--store", store, "show", settle])
    out = capsys.readouterr().out
    matrix = out[out.index("matrix:"):out.index("readings:")]
    assert "@total" not in matrix                        # merged away
    assert "tokenSubsetBelief[source_text" in matrix           # the mechanism row
    jane = [l for l in matrix.splitlines()
            if l.strip().startswith("human:jane")][0]
    assert "x" in jane                                   # nobody to call


def test_evaluate_ends_with_the_filled_matrix_when_piped(store, capsys):
    """Piped evaluate: plain log lines, then the finished table -- with a
    row guaranteed for the consulted belief."""
    from thinair.beliefs import restore_config, set_config

    engine = FakeEngine([{"value": 1249.5, "p": 0.66}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate", "HEAD", "deepseek-v4-flash"])
        out = capsys.readouterr().out
    finally:
        restore_config(None, previous)
    assert "matrix:" in out
    assert "model:deepseek-v4-flash" in out               # its row exists
    assert out.index("agrees") < out.index("matrix:")     # logs, then table


def test_evaluate_redraws_the_matrix_live_on_a_tty(store, monkeypatch,
                                                   capsys):
    """On a terminal the matrix fills itself in: cursor-up erase sequences
    between readings, the table redrawn after each."""
    from thinair.beliefs import restore_config, set_config

    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    engine = FakeEngine([{"value": 1249.5, "p": 0.66}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate", "HEAD", "deepseek-v4-flash"])
        out = capsys.readouterr().out
    finally:
        restore_config(None, previous)
    assert "A\x1b[J" in out                               # erase-and-redraw
    assert out.count("matrix:") > 2                       # one per update


def test_evaluate_rebuilds_validators_from_the_record(store, capsys):
    """The belief table stores constructor configs, so evaluate '*' can
    rebuild built-in validators and have them re-judge the held tree --
    the ? cells in validator rows are fillable, not decorative."""
    from thinair.beliefs import restore_config, set_config

    engine = FakeEngine([{"value": 1249.5, "p": 0.66}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate"])
        out = capsys.readouterr().out
    finally:
        restore_config(None, previous)
    assert "rebuilt validators" in out.splitlines()[0]
    assert "judges p" in out                             # a judge spoke
    noted = [o for o in SqliteLedger(store).opinions(entity="inv-1",
                                                     attr="total")
             if (o.meta or {}).get("corroboration")
             and o.belief.startswith("tokenSubset")]
    assert noted                                         # and was recorded

    main(["--store", store, "evaluate"])                 # idempotent by hash
    again = capsys.readouterr().out
    assert "judges p" not in again


def test_matrix_shades_by_value_overlap(tmp_path, monkeypatch, capsys):
    """Not binary red/green: a partial text overlap lands mid-ramp."""
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": "urgent delivery failed", "p": 0.9}])
    other = FakeEngine([{"value": "urgent delivery arrived", "p": 0.8}])

    class Report(Thing):
        __beliefs__ = [model("small-fast", engine=engine),
                       model("qwen3-35b", engine=other), human("jane")]
        summary = Thing(str)

    r = Report(__entity__="rep-1", __ledger__=ledger)
    +r.summary
    corroborate(r, attrs=["summary"])                    # near-miss reading

    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    out = capsys.readouterr().out
    matrix = out[out.index("matrix:"):out.index("readings:")]
    assert "\x1b[92m" in matrix                        # exact: the one green
    assert "\x1b[33m" in matrix \
        or "\x1b[2;31m" in matrix                        # partial: between


# --------------------------------------------------------------------------
# custom beliefs: registered with the client, callable by evaluate
# --------------------------------------------------------------------------

CUSTOM_BELIEF = '''\
from thinair.beliefs import Discriminative


class EndsWithPeriod(Discriminative):
    """The candidate text finishes its sentence."""

    def judge(self, value, e, attr):
        if not isinstance(value, str):
            return None
        return 1.0 if value.endswith(".") else 0.25
'''


def test_belief_add_list_rm(store, tmp_path, capsys):
    source = tmp_path / "sentences.py"
    source.write_text(CUSTOM_BELIEF)

    main(["--store", store, "belief", "add", str(source)])
    assert "EndsWithPeriod" in capsys.readouterr().out

    main(["--store", store, "belief", "list"])
    listed = capsys.readouterr().out
    assert "sentences.py" in listed and "EndsWithPeriod" in listed

    main(["--store", store, "belief", "rm", "sentences"])
    capsys.readouterr()
    main(["--store", store, "belief", "list"])
    assert "sentences" not in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["--store", store, "belief", "rm", "sentences"])
    with pytest.raises(SystemExit):
        main(["--store", store, "belief", "add", str(tmp_path / "nope.py")])


def test_belief_add_refuses_files_without_beliefs(store, tmp_path):
    source = tmp_path / "empty.py"
    source.write_text("x = 1\n")
    with pytest.raises(SystemExit):
        main(["--store", store, "belief", "add", str(source)])


# --------------------------------------------------------------------------
# supported models: registered with the client, resolved by name
# --------------------------------------------------------------------------

CUSTOM_MODEL = '''\
from thinair.models import ModelDef

MODEL = ModelDef(match=("housemodel-9b",), version="9",
                 defaults=dict(temperature=0.1))
'''


def test_model_add_list_rm(store, tmp_path, capsys):
    source = tmp_path / "housemodel.py"
    source.write_text(CUSTOM_MODEL)

    main(["--store", store, "model", "add", str(source)])
    assert "housemodel-9b" in capsys.readouterr().out

    main(["--store", store, "model", "list"])
    listed = capsys.readouterr().out
    assert "housemodel.py" in listed and "housemodel-9b" in listed

    # the def resolves through the registry once the store's models load --
    # the same door evaluate walks
    from thinair.cli import _load_models
    from thinair.models import resolve

    _load_models(SqliteLedger(store))
    assert resolve("housemodel-9b").version == "9"

    main(["--store", store, "model", "rm", "housemodel"])
    capsys.readouterr()
    main(["--store", store, "model", "list"])
    assert "housemodel" not in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["--store", store, "model", "rm", "housemodel"])
    with pytest.raises(SystemExit):
        main(["--store", store, "model", "add", str(tmp_path / "nope.py")])


def test_model_add_refuses_files_without_a_def(store, tmp_path):
    source = tmp_path / "no_def.py"
    source.write_text("x = 1\n")
    with pytest.raises(SystemExit):
        main(["--store", store, "model", "add", str(source)])


def test_evaluate_consults_registered_custom_beliefs(store, tmp_path,
                                                     capsys):
    """A custom belief the app used is rebuildable once its file is
    registered: kind resolves to the class, config rebuilds it, and it
    re-judges the tree beside the built-ins."""
    from thinair.cli import _load_custom

    source = tmp_path / "sentences.py"
    source.write_text(CUSTOM_BELIEF)
    main(["--store", store, "belief", "add", str(source)])
    capsys.readouterr()

    cls = _load_custom(SqliteLedger(store))["EndsWithPeriod"]
    engine = FakeEngine([{"value": "All good here.", "p": 0.9}])

    class Report(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane"),
                       cls(necessary=False)]
        text = Thing(str)

    +Report(__entity__="rep-7", __ledger__=SqliteLedger(store)).text

    from thinair.beliefs import restore_config, set_config
    previous = set_config(None, engine=FakeEngine([{"value": "All good here.",
                                                    "p": 0.7}]))
    try:
        main(["--store", store, "evaluate", "rep-7"])
        out = capsys.readouterr().out
    finally:
        restore_config(None, previous)
    assert "endsWithPeriod judges p 1" in out


def test_judges_rebuild_cold_from_config(store, monkeypatch):
    """Without a warm registry, the config column alone rebuilds the
    instrument -- the truly-cold CLI path."""
    import thinair.beliefs as B
    from thinair.cli import _reconstructible_judges

    monkeypatch.setattr(B, "_REGISTRY", {})
    judges = _reconstructible_judges(SqliteLedger(store), custom={})
    assert any(j.id.startswith("tokenSubset") for j in judges)


def test_the_consensus_view_travels_everywhere(store, capsys):
    """(p 0.95 ±0.02) is the standard rendering of a believed cell: show's
    context lines carry it, and the matrix ends with a (held) footer row
    stating each cell's standing consensus."""
    main(["--store", store, "show", "HEAD"])             # the episode commit
    out = capsys.readouterr().out
    context = [l for l in out.splitlines()
               if l.strip().startswith("total =")][0]
    assert "±" in context                                # not just (p 0.93)

    matrix = out[out.index("matrix:"):out.index("readings:")]
    footer = [l for l in matrix.splitlines() if "(held)" in l]
    assert footer and "±" in footer[0]
    assert "p 1.00 ±0.00" in footer[0]                   # the frozen fiat,
                                                         # unmeasured so far

    from thinair.beliefs import restore_config, set_config
    previous = set_config(None, engine=FakeEngine([{"value": 1249.5,
                                                    "p": 0.8}]))
    try:
        main(["--store", store, "evaluate", "HEAD", "small-fast"])
        evaluated = capsys.readouterr().out
    finally:
        restore_config(None, previous)
    assert "(held)" in evaluated                         # footer in evaluate


def test_frozen_columns_are_skipped_unless_included(store, capsys):
    """A pinned cell is not a question: evaluate passes it by, says so,
    and the matrix shows 'frozen' on every row but the freezer's own.
    --include-frozen turns the question back on."""
    from thinair.beliefs import restore_config, set_config

    engine = FakeEngine([{"value": 1249.5, "p": 0.66}])
    previous = set_config(None, engine=engine)
    try:
        main(["--store", store, "evaluate", "HEAD", "small-fast"])
        out = capsys.readouterr().out
        assert "frozen, not consulted: source_text" in out
        assert "source_text        ⇒" not in out         # never consulted
        spent = engine.call_count

        matrix = out[out.index("matrix:"):]
        jane = [l for l in matrix.splitlines()
                if l.strip().startswith("human:jane")][0]
        assert "1.00" in jane                            # the freezer's number
        model_row = [l for l in matrix.splitlines()
                     if l.strip().startswith("model:small-fast")][0]
        assert "frozen" in model_row                     # everyone else

        main(["--store", store, "evaluate", "HEAD", "small-fast",
              "--include-frozen"])
        included = capsys.readouterr().out
        assert "frozen, not consulted" not in included
        assert "source_text" in included and engine.call_count > spent
    finally:
        restore_config(None, previous)


def test_a_frozen_fact_is_measurable_in_the_footer(tmp_path, capsys):
    """Readings show through frozen columns, and the footer prices the
    fact: p 1.00 with its deviation against what other beliefs read."""
    from thinair.beliefs import restore_config, set_config

    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 10.0, "p": 0.9}])

    class Box(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        size = Thing(float)

    b = Box(__entity__="box-9", __ledger__=ledger)
    b.size = +b.size

    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    out = capsys.readouterr().out
    matrix = out[out.index("matrix:"):]
    model_row = [l for l in matrix.splitlines()
                 if l.strip().startswith("model:small-fast")][0]
    assert "0.90" in model_row                           # the reading shows
    assert "p 1.00 ±0.00" in matrix                      # fact, unmeasured

    previous = set_config(None, engine=FakeEngine([{"value": 10.0,
                                                    "p": 0.7}]))
    try:
        main(["--store", str(tmp_path / "o.db"), "evaluate", "HEAD",
              "deepseek-v4-flash", "--include-frozen"])
        capsys.readouterr()
    finally:
        restore_config(None, previous)

    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    matrix = capsys.readouterr().out
    assert "p 1.00 ±0.15" in matrix                      # σ of [1.0, 0.7]:
                                                         # the fact, priced


def test_the_panel_unwinds_to_the_commit(tmp_path, capsys):
    """A later override must not haunt an earlier commit: show at the
    settle displays the panel as it stood then, and only HEAD knows the
    override happened."""
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 10.0, "p": 0.9}])

    class Box(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        size = Thing(float)

    b = Box(__entity__="box-t", __ledger__=ledger)
    +b.size                                              # settles at 10.0
    b.size = 99.0                                        # next week's override

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]

    main(["--store", str(tmp_path / "o.db"), "show", settle])
    then = capsys.readouterr().out
    assert "99" not in then                              # not yet spoken
    assert "10.0" in then

    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    now = capsys.readouterr().out
    assert "99" in now                                   # but HEAD knows


def test_the_signature_is_always_numeric(tmp_path, capsys):
    """(p 0.7 ±0.00), never a bare (p 0.7): the spread slot is always
    stated -- coverage (the parens) is what says whether anyone measured."""
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": "calm", "p": 0.7}])

    class Person(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]

    person = Person(__entity__="per-1", __ledger__=ledger)
    +person.mood                                         # undeclared: one voice

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    line = [l for l in capsys.readouterr().out.splitlines()
            if "[settle]" in l][0]
    assert "±0.00" in line and "±?" not in line


def test_readings_shade_by_agreement_with_the_held_value(tmp_path,
                                                         monkeypatch,
                                                         capsys):
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 10.0, "p": 0.9}])

    class Box(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        size = Thing(float)

    b = Box(__entity__="box-r", __ledger__=ledger)
    +b.size
    b.size = 99.0                                        # the override holds

    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    readings = capsys.readouterr().out.split("readings:")[1]
    jane = [l for l in readings.splitlines() if "human:jane" in l][0]
    assert "\x1b[4;92m" in jane                        # holds it -- under-
                                                         # lined, the one in use
    model_line = [l for l in readings.splitlines()
                  if "model:small-fast" in l][0]
    assert "\x1b[2;31m" in model_line or "\x1b[31m" in model_line
    assert "\x1b[4;" not in model_line                   # overruled: plain


def test_parens_shade_by_evaluation_coverage(store, monkeypatch, capsys):
    """The signature's parens say how much of the callable panel has
    spoken: mid-ramp while qwen3-35b is silent on a cell, green once
    every reachable belief has weighed in."""
    from thinair.beliefs import restore_config, set_config

    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", store, "log", "--oneline", "inv-1"])
    lines = capsys.readouterr().out.splitlines()
    settle = [l for l in lines if "[settle]" in l][0]
    assert "\x1b[92m(" in settle                         # total: both models
                                                         # + tokenSubset spoke
    episode = [l for l in lines if "[episode]" in l]
    assert episode                                       # (episodes carry no
                                                         # signature to wrap)

    engine = FakeEngine([{"value": 1249.5, "p": 0.66}])
    previous = set_config(None, engine=engine)
    try:                                                 # silence nobody:
        main(["--store", store, "evaluate", "HEAD"])     # every model reads
        capsys.readouterr()
    finally:
        restore_config(None, previous)
    main(["--store", store, "log", "--oneline", "inv-1"])
    after = [l for l in capsys.readouterr().out.splitlines()
             if "[settle]" in l][0]
    assert "\x1b[92m(" in after                          # still complete


def test_coverage_is_per_attribute(tmp_path, monkeypatch, capsys):
    """priority's enum never counts against customer: a cell whose own
    panel has fully spoken wears green parens, whatever other cells'
    validators are still owed elsewhere."""
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": "Anna", "p": 0.9},
                         {"value": "high", "p": 0.8}])

    class Ticket(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        source_text: str
        customer = Thing(str, extracted_from="source_text")
        priority = Thing(str, enum=["low", "high"])

    t = Ticket(__entity__="tick-1", __ledger__=ledger, source_text="From Anna")
    +t.customer
    +t.priority

    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    customer = [l for l in capsys.readouterr().out.splitlines()
                if "customer" in l][0]
    assert "\x1b[92m(" in customer                       # its panel is done


def test_ai_readable_says_the_colors_in_text(store, capsys):
    main(["--store", store, "--ai-readable", "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0]
    assert "agree=1.00" in settle and "asked=" in settle

    main(["--store", store, "log", "--oneline", "inv-1"])   # and off again
    assert "agree=" not in capsys.readouterr().out


def test_matrix_marks_follow_the_cells_panel(tmp_path, capsys):
    """priority's enum reads x under customer, not ? -- a mechanism
    claimed by scoped wrappers is only askable on the attributes its
    wrappers name, and evaluate consults by the same rule."""
    from thinair.beliefs import restore_config, set_config

    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": "Anna", "p": 0.9},
                         {"value": "high", "p": 0.8}])

    class Ticket(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        source_text: str
        customer = Thing(str, extracted_from="source_text")
        priority = Thing(str, enum=["low", "high"])

    t = Ticket(__entity__="tick-2", __ledger__=ledger, source_text="From Anna")
    +t.customer
    +t.priority

    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    out = capsys.readouterr().out
    matrix = out[out.index("matrix:"):out.index("readings:")]
    enum_row = [l for l in matrix.splitlines() if "enumBelief[" in l][0]
    columns = enum_row.split()
    assert columns[-3:] == ["x", "1.00", "frozen"]       # customer, priority,
                                                         # source_text
    previous = set_config(None, engine=FakeEngine([{"value": "high",
                                                    "p": 0.6}]))
    try:
        main(["--store", str(tmp_path / "o.db"), "evaluate", "HEAD"])
        evaluated = capsys.readouterr().out
    finally:
        restore_config(None, previous)
    judged = [l for l in evaluated.splitlines()
              if "enumBelief[" in l and "judges p" in l]
    assert all(l.strip().startswith("priority") for l in judged)


def test_assign_and_freeze_carry_the_frozen_parens(store, capsys):
    """Every commit line ends in parens: believed cells wear the
    signature, frozen ones a dim (frozen)."""
    main(["--store", store, "log", "--oneline"])
    lines = capsys.readouterr().out.splitlines()
    assign = [l for l in lines if "[assign]" in l][0]
    assert assign.rstrip().endswith("(frozen)")


def test_a_changed_panel_is_a_belief_commit(tmp_path, capsys):
    """Adding or removing a belief changes what the strategy is, and the
    hash chain says so: a [belief] commit, tree untouched.  The first
    declaration is a silent baseline; restating it commits nothing."""
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 5.0, "p": 0.9}])

    def build(extra=()):
        class Box(Thing):
            __beliefs__ = [model("small-fast", engine=engine),
                           human("jane"), *extra]
            size = Thing(float)
        return Box

    +build()(__entity__="box-p", __ledger__=ledger).size
    first = history(ledger, entity="box-p")
    assert [c["kind"] for c in first] == ["settle"]      # baseline invisible

    build()(__entity__="box-p", __ledger__=ledger)       # same panel again
    assert len(history(ledger, entity="box-p")) == len(first)

    second = FakeEngine([{"value": 5.0, "p": 0.7}])
    build((model("qwen3-35b", engine=second),))(
        __entity__="box-p", __ledger__=ledger)           # the strategy moved
    commits = history(ledger, entity="box-p")
    assert commits[-1]["kind"] == "belief"
    assert "+ model:qwen3-35b" in commits[-1]["message"]
    assert commits[-1]["parent"] == first[-1]["hash"]    # chained: hash moved
    assert commits[-1]["tree"] == first[-1]["tree"]      # the tree did not

    main(["--store", str(tmp_path / "o.db"), "log", "--oneline"])
    assert "[belief]" in capsys.readouterr().out


def test_frozen_footer_colors_by_prior_estimates(tmp_path, monkeypatch,
                                                 capsys):
    """The (held) row of a frozen column paints by whether the fiat
    landed where the other beliefs' readings did."""
    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": 10.0, "p": 0.9}])

    class Box(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        size = Thing(float)

    b = Box(__entity__="box-f", __ledger__=ledger)
    +b.size
    b.size = 99.0                                        # far from estimates

    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", str(tmp_path / "o.db"), "show", "HEAD"])
    footer = [l for l in capsys.readouterr().out.splitlines()
              if "(held)" in l][0]
    assert "\x1b[2;31m" in footer or "\x1b[31m" in footer


def test_the_resolving_reading_renders_underlined(store, monkeypatch, capsys):
    """The value the program actually served: its author's cell is the
    underlined one in its column -- emphasis by line, so the gradient
    keeps its one green, the branch green."""
    main(["--store", store, "log", "--oneline", "inv-1"])
    settle = [l for l in capsys.readouterr().out.splitlines()
              if "[settle]" in l][0].split()[0]
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", store, "show", settle])
    matrix = capsys.readouterr().out.split("matrix:")[1].split("readings:")[0]
    model_row = [l for l in matrix.splitlines()
                 if l.strip().startswith("model:small-fast")][0]
    assert "\x1b[4;92m" in model_row                   # underlined: resolved
    judge_row = [l for l in matrix.splitlines()
                 if l.strip().startswith("tokenSubset")][0]
    assert "\x1b[4;" not in judge_row                    # judges stay plain
    assert not any(second in matrix                      # no second green:
                   for second in ("\x1b[2;32m", "\x1b[32m", "\x1b[1;32m"))
                                                         # dim, plain, bold


def test_ground_teaches_the_client_to_agents(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["ground"])
    out = capsys.readouterr().out
    assert "the thinair client, for agents" in out
    assert "--ai-readable" in out                        # the blindness cure
    assert "thinair belief add" in out                   # how to extend
    assert "def judge(self, value, e, attr):" in out     # a working example
    assert not (tmp_path / ".thinair").exists()          # still store-free


def test_ground_demands_a_running_program(tmp_path, monkeypatch, capsys):
    """The manual must not let an agent stop at prose: it shows a
    complete runnable skeleton and says the strategy document is the
    plan, not the deliverable."""
    monkeypatch.chdir(tmp_path)
    main(["ground"])
    out = capsys.readouterr().out
    assert "class Ticket(Thing):" in out                 # runnable skeleton
    assert "__entity__=" in out                          # named branches
    assert "measures nothing until it executes" in " ".join(out.split())
    # grounding demands nothing of the project: no repo files named
    assert "SPEC.md" not in out
    # the verify loop: debug through the record, finish on consensus
    assert "not done until the record agrees" in out
    assert "`agree=` near 1.00" in out
    # and the theory document itself no longer ends at "no code at all"
    assert "needs no code at all" not in out
    assert "the strategy is a waypoint" in " ".join(out.split())


def test_ground_is_tiered(tmp_path, monkeypatch, capsys):
    """Default ground is the orientation tier -- pillars, moves, framework,
    manual; no meta explaining the file to its reader -- minus the
    strategy-design stretch (anchors + Part 2) and the reference stretch
    (belief roster, the spelled-out surface), sized to land whole in a
    harness's inline tool output.  --full restores every cut."""
    monkeypatch.chdir(tmp_path)
    main(["ground"])
    out = capsys.readouterr().out
    assert "Pillar I " in out                            # the pillars stay
    assert "not an oracle" in out                        # the reframing too
    assert "## The moves" in out
    assert "# Part 3 " in out                            # the framework
    assert "\n# Part 2 " not in out                      # protocol is --full
    assert "Mathematical anchors" not in out             # citations too
    assert "built-in beliefs" not in out                 # reference tier
    assert "The surface, spelled out" not in out         # reference tier
    assert "Link it, as-is" not in out                   # no meta preamble
    # the acting-critical rules survive the surface demotion in the manual
    assert "second opinion" in out and "corroborate" in out
    # fits inline: harness tool-output limits sit far above this; the budget
    # tracks the folded content (§13 rule 7 grows it) with modest headroom
    assert len(out.encode()) < 32_000

    main(["ground", "--full"])
    full = capsys.readouterr().out
    assert "\n# Part 2 " in full                         # protocol restored
    assert "Mathematical anchors" in full
    assert "## Appendix: built-in beliefs" in full       # reference restored
    assert "### The surface, spelled out" in full
    assert "the thinair client, for agents" in full      # manual still there


def test_ground_serves_single_sections(tmp_path, monkeypatch, capsys):
    """`thinair ground <section>` re-fetches one part on demand -- the
    growth valve: rule-7 folds land in a tier or a section, never back in
    one ever-fatter blob."""
    monkeypatch.chdir(tmp_path)
    main(["ground", "moves"])
    moves = capsys.readouterr().out
    assert moves.startswith("## The moves")
    assert "Prefer comparisons" in moves
    assert "Pillar I " not in moves                      # just the section

    main(["ground", "verify"])
    verify = capsys.readouterr().out
    assert verify.startswith("### Verify")
    assert "`agree=` near 1.00" in verify
    assert "### Extend" not in verify                    # sliced at siblings

    main(["ground", "protocol"])                         # a --full-only part
    assert capsys.readouterr().out.startswith("# Part 2 ")

    main(["ground", "surface"])
    surface = capsys.readouterr().out
    assert "never `t += belief`" in surface
    assert "## The mapping" not in surface

    with pytest.raises(SystemExit) as caught:
        main(["ground", "nonsense"])
    assert "roster" in str(caught.value)                 # the menu is the error
    assert not (tmp_path / ".thinair").exists()


def test_the_held_row_colors_earned_agreement_and_dims_unopposed(
        store, monkeypatch, capsys):
    """A cell with independent readers on record wears the gradient; an
    unopposed believed cell renders dim -- one reading that nothing could
    have contradicted earns no color either way.  Frozen facts stay green
    (a fiat nothing disagrees with)."""
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", store, "show", "HEAD"])
    footer = [l for l in capsys.readouterr().out.splitlines()
              if "(held)" in l][0]
    cells = footer.split("(held)")[1]
    assert "\x1b[92m" in cells               # corroborated: earned green
    assert "\x1b[2m" in cells                  # unopposed: dim, unearned


def test_each_signature_channel_earns_its_own_color(monkeypatch):
    """Every voice counts the same -- a validator's verdict colors the
    value channel exactly like a model's reading.  The ± colors by the
    recorded min-max range; a cell where nothing else spoke waits dim
    whole."""
    from thinair.cli import _signature

    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    commit = {"consensus": {"size": {"n": 2, "dissent": 0, "dev": 0.02,
                                     "range": 0.05}}, "expect": {}}
    out = _signature(commit, "size", 0.95)
    assert "\x1b[92mp 0.95" in out           # two voices agree: green
    assert "\x1b[92m±0.02" in out            # spread: tight, green

    far = {"consensus": {"size": {"n": 1, "dissent": 1, "similarity": 0.2,
                                  "dev": 0.3, "range": 0.6}}, "expect": {}}
    mid = _signature(far, "size", 0.95)
    assert "\x1b[33mp 0.95" in mid             # a voice landed elsewhere

    lone = _signature({"consensus": {"size": {"n": 1, "dissent": 0}},
                       "expect": {}}, "size", 0.95)
    assert "\x1b[2mp 0.95" in lone and "\x1b[2m±0.00" in lone


def test_a_failed_ask_is_a_note_but_never_a_reading(
        tmp_path, monkeypatch, capsys):
    """An empty completion (no value, p 0) stays on the record and renders
    as what it is -- an unanswered ask -- but it is no reading: the matrix
    and the readings table keep the belief's last real stance."""
    from thinair.ledger import Opinion

    ledger = SqliteLedger(tmp_path / "opinions.db")
    ledger.add(Opinion(belief="model:a", entity="e1", attr="size",
                       value=4, p=0.9, meta={"model": "a", "round": 1}))
    ledger.add(Opinion(belief="model:pro", entity="e1", attr="size",
                       value=4, p=0.95,
                       meta={"model": "pro", "corroboration": True}))
    ledger.add(Opinion(belief="model:pro", entity="e1", attr="size",
                       value=None, p=0.0,
                       meta={"model": "pro", "corroboration": True,
                             "reason": "completion carried no 'value'"}))
    monkeypatch.setattr("thinair.cli._tty", lambda: True)
    main(["--store", str(tmp_path / "opinions.db"), "show", "HEAD"])
    out = capsys.readouterr().out
    assert "note: model:pro was asked size; no answer came back" in out
    assert "read size as null" not in out          # never dressed as a stance
    stances = [l for l in out.splitlines()
               if "model:pro" in l and "note:" not in l]
    assert stances                                 # matrix row + readings row
    assert all("0.95" in l for l in stances)       # the last *real* reading
    assert not any("\x1b[31m" in l for l in stances)


def test_oneline_caps_at_the_screen(store, monkeypatch, capsys):
    """--oneline means one line: a row longer than the screen is cut with
    an ellipsis (ANSI codes pass through unmeasured).  A piped reader is
    never cut -- `_columns()` is None off-screen."""
    from thinair.cli import _columns
    assert _columns() is None                  # capsys is a pipe: no cap

    monkeypatch.setattr("thinair.cli._columns", lambda: 48)
    main(["--store", store, "log", "--oneline"])
    lines = capsys.readouterr().out.splitlines()
    assert any(line.endswith(" …") for line in lines)
    assert all(len(line) <= 48 for line in lines)


def test_ai_readable_counts_every_voice_and_unopposed_means_alone(
        tmp_path, capsys):
    """No castes: a validator's verdict earns `agree=` exactly like a
    second model would.  `agree=unopposed` is reserved for a cell where
    nothing else spoke at all."""
    from typing import Any

    ledger = SqliteLedger(tmp_path / "o.db")
    engine = FakeEngine([{"value": "Anna", "p": 0.9},
                         {"value": "hello", "p": 0.8}])

    class Ticket(Thing):
        __beliefs__ = [model("small-fast", engine=engine), human("jane")]
        source_text: str
        customer = Thing(str, extracted_from="source_text")
        note = Thing(Any)

    t = Ticket(__entity__="tick-9", __ledger__=ledger,
               source_text="From Anna")
    +t.customer
    +t.note
    main(["--store", str(tmp_path / "o.db"), "--ai-readable", "log",
          "--oneline", "tick-9"])
    lines = [l for l in capsys.readouterr().out.splitlines()
             if "[settle]" in l]
    customer = [l for l in lines if "customer" in l][0]
    assert "agree=1.00" in customer            # the validator is a voice
    note = [l for l in lines if "note" in l][0]
    assert "agree=unopposed" in note           # truly alone: no flattery


def test_forked_chains_collapse_with_a_slash(tmp_path, capsys):
    """Two refs sharing their first commit then diverging are a real fork:
    the graph bends the extra lane home with `/` just above the shared
    parent, git's `|/`."""
    ledger = SqliteLedger(tmp_path / "o.db")

    class Card(Thing):
        __beliefs__ = [human("jane")]
        text: str

    a = Card(__entity__="branch-a", __ledger__=ledger, text="same start")
    b = Card(__entity__="branch-b", __ledger__=ledger, text="same start")
    a.status = "open"
    b.status = "closed"

    main(["--store", str(tmp_path / "o.db"), "log", "--all", "--oneline",
          "--graph"])
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("|/") for line in lines)     # the collapse
    assert any(line.startswith("| *") for line in lines)    # parallel lanes
    assert lines[-1].startswith("+ ")                       # the shared root
                                                            # is a birth
