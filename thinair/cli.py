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
    thinair beliefs [commit]                    who spoke (or could) there
    thinair evaluate [belief] [commit]          consult beliefs against a
                                                commit's state -- spends
                                                calls, records corroborations

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
from .ledger import Ledger, Opinion, values_equal
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


class _RecordSnapshot:
    """A sealed snapshot built from a commit's tree -- no live Thing needed.

    Carries the same surface a belief reads off a real snapshot.  Contracts
    and purpose are class code, absent here, so a reading taken this way has
    its *own* exposure -- a leaner context than the original run's -- and
    the stamp records exactly that; settlement weighs it accordingly.
    """

    def __init__(self, entity, cells, deriving):
        object.__setattr__(self, "_cells", {
            attr: Opinion(belief=author or "record", entity=entity, attr=attr,
                          value=value, p=p, t=1.0, frozen=frozen)
            for attr, (value, p, frozen, author) in cells.items()
            if attr != deriving})
        object.__setattr__(self, "__entity__", entity)

    __class_name__ = "Record"
    __purpose__ = ""
    __value__ = None
    __p__ = 0.0
    __provenance__ = ()
    __contracts__: dict = {}
    __methods__: tuple = ()
    __arguments__: dict = {}
    __objections__: tuple = ()
    __owner__ = None
    __episode__ = None
    __call_arguments__ = None
    __beliefs__: list = []

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_cells")[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("a snapshot is sealed")

    def __attrs__(self):
        return dict(object.__getattribute__(self, "_cells"))

    def __opinion__(self, attr):
        return object.__getattribute__(self, "_cells").get(attr)


def _commit_at(commits, prefix):
    if prefix is None:
        if not commits:
            sys.exit("fatal: empty record")
        return commits[-1]                                 # HEAD: the newest
    matches = [c for c in commits if c["hash"].startswith(prefix)]
    if not matches:
        sys.exit(f"fatal: bad revision '{prefix}'")
    return matches[-1]


def _tree_at(commits, commit):
    """attr -> (value, p, frozen, author) for the commit's entity, as of it."""
    cells: dict[str, tuple] = {}
    for earlier in commits:
        if earlier["entity"] != commit["entity"] or earlier["t"] > commit["t"]:
            continue
        for attr, (value, p, frozen) in earlier["changes"].items():
            already = cells.get(attr)
            if frozen or already is None or not already[2]:
                cells[attr] = (value, p, frozen, earlier["author"])
    return cells


def _model_names(ledger):
    """The reconstructible instruments the record knows: model belief names,
    parsed from durable ids (invariant 6 makes the id the configuration)."""
    names = []
    for belief_id in ledger.beliefs():
        if belief_id.startswith("model:") and "@T" in belief_id:
            name = belief_id[len("model:"):].split("@T")[0]
            if name not in names:
                names.append(name)
    return names


def cmd_beliefs(ledger, args):
    commits = history(ledger)
    scope = None
    if args.commit:
        commit = _commit_at(commits, args.commit)
        scope = commit["entity"]
        spoke = {o.belief for o in ledger.opinions(entity=scope)}
        print(f"beliefs on {commit['hash']} ({scope}):")
    else:
        spoke = set(ledger.beliefs())
        print("beliefs on record:")
    for belief_id in sorted(spoke):
        row = getattr(ledger, "belief_row", lambda _id: None)(belief_id) or {}
        marks = []
        if row.get("proposes") or belief_id.startswith("model:"):
            marks.append("proposes")
        if row.get("necessary"):
            marks.append(f"necessary@{row['veto_line']:g}")
        if belief_id.startswith("model:"):
            marks.append("reconstructible")
        tail = f"  [{', '.join(marks)}]" if marks else ""
        description = row.get("description")
        note = f"  -- {description}" if description and description != belief_id \
            else ""
        print(f"  {belief_id}{tail}{note}")


def cmd_evaluate(ledger, args):
    if not hasattr(ledger, "belief_row"):
        sys.exit("fatal: archives are read-only; evaluate needs a store")
    from .beliefs import model

    commits = history(ledger)
    commit = _commit_at(commits, args.commit)
    entity = commit["entity"]
    cells = _tree_at(commits, commit)
    if args.belief == "*":
        names = _model_names(ledger)
        if not names:
            sys.exit("fatal: no reconstructible belief has spoken here; "
                     "name one: thinair evaluate <model-name>")
    else:
        raw = args.belief
        names = [raw[len("model:"):].split("@T")[0]
                 if raw.startswith("model:") else raw]

    print(f"evaluate {', '.join(names)} @ {commit['hash']} ({entity})")
    for name in names:
        belief = model(name)
        for attr in sorted(cells):
            e = _RecordSnapshot(entity, cells, deriving=attr)
            stamp = belief.exposure(e, attr)
            prior = ledger.opinions(entity=entity, attr=attr, belief=belief.id)
            if any((o.meta or {}).get("exposure") == stamp for o in prior):
                print(f"  {attr:<18} {belief.id}: asked and answered")
                continue
            got = belief(e, attr)
            if got is None:
                continue
            ledger.add(Opinion(
                belief=belief.id, entity=entity, attr=attr,
                value=+got, p=~got, frozen=False,
                meta=dict(getattr(got, "meta", None) or {},
                          corroboration=True, at=commit["hash"])))
            held, _p, _frozen, _author = cells[attr]
            verdict = "agrees" if values_equal(+got, held) else \
                f"DIFFERS from {_value(held)}"
            print(f"  {attr:<18} ⇒ {_value(+got)} (p {~got:g})  {verdict}")


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

    beliefs = sub.add_parser("beliefs", help="who spoke (or could) at a commit")
    beliefs.add_argument("commit", nargs="?", default=None)
    beliefs.set_defaults(run=cmd_beliefs)

    evaluate = sub.add_parser(
        "evaluate", help="consult beliefs against a commit's state "
                         "(spends calls, records corroborations)")
    evaluate.add_argument("belief", nargs="?", default="*")
    evaluate.add_argument("commit", nargs="?", default=None)
    evaluate.set_defaults(run=cmd_evaluate)

    args = parser.parse_args(argv)
    args.run(open_store(args.store), args)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
