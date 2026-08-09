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

    inv = Invoice(__entity__="inv-1", __ledger__=ledger, source_text=SOURCE)
    +inv.total
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
