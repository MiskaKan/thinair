"""thinair -- the record, inspected like a repository.

Git's mapping, on the command line: the tree is the state hash, a commit is
whatever moved it, every entity is a branch with its own chain.  The
commands are deliberate copies:

    thinair log [entity] [--oneline] [-n N]     the commits, newest first
    thinair show <hash-prefix>                  one commit: diff, rounds,
                                                vetoes, notes
    thinair status                              the store, summarized
    thinair branch                              entities and their heads
    thinair blame <entity>                      every cell: who set it, when

``--store`` points anywhere: the default ``.thinair/opinions.db``, any
SQLite store, or a committed ``ledger.json`` archive -- the inspector is a
pure derivation (``thinair.evaluate.history``) and spends nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .evaluate import history
from .ledger import Ledger
from .store import DEFAULT_PATH, SqliteLedger

KINDS = {"assign": "assign", "settle": "settle", "episode": "episode",
         "code": "code", "freeze": "freeze", "fixture": "fixture"}


def open_store(path: str | None) -> Ledger:
    path = path or os.environ.get("THINAIR_STORE") or DEFAULT_PATH
    if not os.path.exists(path):
        sys.exit(f"fatal: not a thinair store: {path}")
    if path.endswith(".json"):
        return Ledger.load(path)
    return SqliteLedger(path)


def _value(v, limit=60):
    text = json.dumps(v, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 2] + " …"


def _message(commit) -> str:
    if commit["kind"] == "episode":
        return commit["message"]
    attr, (value, p, frozen) = next(iter(commit["changes"].items()))
    arrow = "=" if frozen else "⇒"
    stated = f"{attr} {arrow} {_value(value)}"
    if not frozen:
        stated += f" (p {p:g})"
    return stated


def cmd_log(ledger, args):
    commits = history(ledger, entity=args.entity)
    commits.reverse()                                  # newest first, like git
    if args.n:
        commits = commits[: args.n]
    for commit in commits:
        if args.oneline:
            print(f"{commit['hash']} {commit['entity']:<12} "
                  f"[{commit['kind']}] {_message(commit)}")
            continue
        print(f"commit {commit['hash']} ({commit['entity']})")
        print(f"Author: {commit['author']}")
        print(f"Date:   t={commit['t']:g}")
        print()
        print(f"    {_message(commit)}")
        detail = []
        if commit["rounds"] > 1 or commit["vetoes"]:
            detail.append(f"{commit['rounds']} rounds, "
                          f"{len(commit['vetoes'])} candidates vetoed")
        if commit.get("notes"):
            detail.append(f"{len(commit['notes'])} corroborations")
        if commit.get("unknown_judges"):
            detail.append("? unverifiable vetoes (judges undescribed)")
        for line in detail:
            print(f"    ({line})")
        print()


def cmd_show(ledger, args):
    commits = history(ledger)
    matches = [c for c in commits if c["hash"].startswith(args.commit)]
    if not matches:
        sys.exit(f"fatal: bad revision '{args.commit}'")
    for commit in matches:
        print(f"commit {commit['hash']} ({commit['entity']})")
        print(f"Author: {commit['author']}")
        print(f"Date:   t={commit['t']:g}")
        print(f"Parent: {commit['parent']}"
              + ("" if commit["kind"] != "episode" else
                 f" (recorded {commit['recorded_parent']}"
                 + (")" if commit["parent_matches"] else " -- MISMATCH)")))
        print()
        print(f"    {_message(commit)}")
        print()
        print(f"diff --thinair {commit['entity']}")
        for attr, (value, p, frozen) in sorted(commit["changes"].items()):
            mark = "frozen" if frozen else f"p={p:g}"
            print(f"+ {attr} = {_value(value)}   ({mark})")
        if commit["kind"] == "episode":
            value, p = commit["returned"]
            print(f"return {_value(value)}   (p={p:g})")
        for value, p in commit["vetoes"]:
            print(f"- {_value(value)}   (vetoed at p={p:g})")
        for belief, attr, value, p in commit.get("notes", ()):
            print(f"note: {belief} read {attr} as {_value(value)} (p={p:g})")
        for judge in commit.get("unknown_judges", ()):
            print(f"?     {judge} judged this; necessity unknown here")
        print()


def cmd_status(ledger, args):
    commits = history(ledger)
    entities = {c["entity"] for c in commits}
    beliefs = ledger.beliefs()
    print(f"On store {getattr(ledger, 'path', '(json archive)')}")
    print(f"{len(ledger)} opinions, {len(commits)} commits, "
          f"{len(entities)} entities, {len(beliefs)} beliefs")
    if commits:
        last = commits[-1]
        print(f"HEAD is at {last['hash']} ({last['entity']}) "
              f"{_message(last)}")


def cmd_branch(ledger, args):
    commits = history(ledger)
    heads: dict[str, dict] = {}
    for commit in commits:
        heads[commit["entity"]] = commit
    for entity in sorted(heads):
        head = heads[entity]
        count = sum(1 for c in commits if c["entity"] == entity)
        print(f"  {entity:<20} {head['hash']}  {count} commits")


def cmd_blame(ledger, args):
    commits = history(ledger, entity=args.entity)
    if not commits:
        sys.exit(f"fatal: no such entity '{args.entity}'")
    latest: dict[str, dict] = {}
    for commit in commits:
        for attr in commit["changes"]:
            latest[attr] = commit
    for attr in sorted(latest):
        commit = latest[attr]
        value, p, frozen = commit["changes"][attr]
        mark = "frozen" if frozen else f"p={p:g}"
        print(f"{commit['hash']} ({commit['author'][:34]:<34} t={commit['t']:g}) "
              f"{attr} = {_value(value)}   ({mark})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="thinair", description="inspect a thinair record, git-style")
    parser.add_argument("--store", help="path to opinions.db or a ledger.json")
    sub = parser.add_subparsers(dest="command", required=True)

    log = sub.add_parser("log", help="the commits, newest first")
    log.add_argument("entity", nargs="?", default=None)
    log.add_argument("--oneline", action="store_true")
    log.add_argument("-n", type=int, default=None)
    log.set_defaults(run=cmd_log)

    show = sub.add_parser("show", help="one commit in full")
    show.add_argument("commit")
    show.set_defaults(run=cmd_show)

    sub.add_parser("status", help="the store, summarized").set_defaults(
        run=cmd_status)
    sub.add_parser("branch", help="entities and their heads").set_defaults(
        run=cmd_branch)

    blame = sub.add_parser("blame", help="every cell: who set it, when")
    blame.add_argument("entity")
    blame.set_defaults(run=cmd_blame)

    args = parser.parse_args(argv)
    args.run(open_store(args.store), args)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
