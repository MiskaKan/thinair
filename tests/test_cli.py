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


def test_identical_content_never_shares_a_commit_id(tmp_path):
    """Two entities with byte-identical states are different commits: the id
    chains entity | parent | tree, so identity is positional, like git's."""
    ledger = SqliteLedger(tmp_path / "o.db")

    class Card(Thing):
        __beliefs__ = [human("jane")]
        text: str

    Card(__entity__="card-1", __ledger__=ledger, text="same words")
    Card(__entity__="card-2", __ledger__=ledger, text="same words")
    commits = history(ledger)
    assert len(commits) == 2
    assert commits[0]["hash"] != commits[1]["hash"]
    assert commits[0]["tree"] == commits[1]["tree"]     # same content, though


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
