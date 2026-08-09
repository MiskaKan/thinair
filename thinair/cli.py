"""thinair -- the record, inspected like a repository.

Git's mapping, on the command line: the tree is the state hash, a commit is
whatever moved it, every entity is a branch with its own chain.  The
commands are deliberate copies:

    thinair log [entity] [--oneline] [-n N]     the commits, newest first
    thinair show [commit]                       one commit: diff, matrix,
                                                rounds, vetoes, notes
    thinair status                              the store, summarized
    thinair branch [-d name]                    entities and their heads
    thinair blame <entity>                      every cell: who set it, when
    thinair beliefs [commit]                    who spoke (or could) there
    thinair evaluate [commit] [belief]          consult beliefs against a
                                                commit's state -- spends
                                                calls, records corroborations
    thinair diff A...B | A B | A                two trees, cell by cell
    thinair source [commit]                     the tree as annotated source
    thinair ground                              print the measurement
                                                grounding (GROUNDING.md) --
                                                pipe it to an agent

A *commit* anywhere above is a hash prefix, a branch (entity) name, or
``HEAD`` -- resolved the way git resolves names.

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


# -- color, git's way: only to a terminal, and NO_COLOR wins ---------------

def _tty() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _paint(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _tty() else text


def yellow(text):
    return _paint("33", text)


def green(text):
    return _paint("32", text)


def red(text):
    return _paint("31", text)


def cyan(text):
    return _paint("1;36", text)


def branch_green(text):
    return _paint("1;32", text)


def dim(text):
    return _paint("2", text)


def _p_verdict(commit, attr, p):
    """``("p 0.93 ±0.04", flag)`` -- the stated probability with the cell's
    deviation, and how the record judges it: ``"violated"`` when a declared
    expectation (p bounds, max deviation) is missed, ``"dissent"`` when a
    recorded reading holds a different value, ``None`` when nothing objects.
    """
    view = (commit.get("consensus") or {}).get(attr) or {}
    expect = (commit.get("expect") or {}).get(attr) or {}
    dev = view.get("dev")
    text = f"p {p:g}" + (f" ±{dev:.2f}" if dev is not None else "")
    bounds = expect.get("p")
    if bounds and not (bounds[0] <= p <= bounds[1]):
        return text, "violated"
    if expect.get("deviation") is not None and dev is not None \
            and dev > expect["deviation"]:
        return text, "violated"
    if view.get("dissent"):
        return text, "dissent"
    return text, None


def _paint_verdict(text, flag):
    return red(text) if flag == "violated" else \
        yellow(text) if flag == "dissent" else text


def _message(commit) -> str:
    if commit["kind"] == "episode":
        return commit["message"]
    attr, (value, p, frozen) = next(iter(commit["changes"].items()))
    arrow = "=" if frozen else "⇒"
    stated = f"{attr} {arrow} {_value(value)}"
    if not frozen:
        text, flag = _p_verdict(commit, attr, p)
        stated += f" ({_paint_verdict(text, flag)})"
    return stated


def _label(commit) -> str:
    """The display name: the first ref, with a marker when others share it."""
    entities = commit["entities"]
    return entities[0] if len(entities) == 1 \
        else f"{entities[0]}(+{len(entities) - 1})"


def _decorations(commits):
    """Branch tips, git-style: refs at their head commits; the newest commit
    overall wears HEAD.  A collapsed chain shows every ref on one tip.
    Git's palette too: HEAD in cyan, branch names in green, parens yellow."""
    out: dict[str, str] = {}
    head_t = max((c["t"] for c in commits), default=None)
    for commit in commits:
        if not commit.get("heads"):
            continue
        label = yellow(", ").join(branch_green(h) for h in commit["heads"])
        if commit["t"] == head_t:
            label = cyan("HEAD -> ") + label
        out[commit["hash"]] = yellow(" (") + label + yellow(")")
    return out


def _lanes(commits_newest_first):
    """One lane per chain while it is on screen: '*' on the commit's own
    lane, '|' through the others -- git's graph, for parallel branches.
    A collapsed multi-ref chain is one lane; forks get their own."""
    oldest_row: dict[str, int] = {}
    for row, commit in enumerate(commits_newest_first):
        oldest_row[commit["entities"][0]] = row
    lanes: list[str | None] = []
    for row, commit in enumerate(commits_newest_first):
        entity = commit["entities"][0]
        if entity not in lanes:
            try:
                lanes[lanes.index(None)] = entity
            except ValueError:
                lanes.append(entity)
        star = " ".join("*" if slot == entity else ("|" if slot else " ")
                        for slot in lanes).rstrip()
        bar = " ".join("|" if slot else " " for slot in lanes).rstrip()
        if oldest_row[entity] == row:
            lanes[lanes.index(entity)] = None
            while lanes and lanes[-1] is None:
                lanes.pop()
        yield star, bar, commit


def cmd_log(ledger, args):
    commits = history(ledger, entity=None if args.all else args.entity)
    commits.reverse()                                  # newest first, like git
    if args.n:
        commits = commits[: args.n]
    decorations = _decorations(commits) if args.decorate else {}
    for star, bar, commit in _lanes(commits):
        head = f"{star} " if args.graph else ""
        body = f"{bar} " if args.graph else ""
        decor = decorations.get(commit["hash"], "")
        if args.oneline:
            print(f"{head}{yellow(commit['hash'])}{decor} "
                  f"{_label(commit):<12} "
                  f"[{commit['kind']}] {_message(commit)}")
            continue
        print(f"{head}{yellow('commit ' + commit['hash'])}{decor} "
              f"({_label(commit)})")
        print(f"{body}Author: {commit['author']}")
        print(f"{body}Date:   t={commit['t']:g}")
        print(body.rstrip())
        print(f"{body}    {_message(commit)}")
        detail = []
        if commit["rounds"] > 1 or commit["vetoes"]:
            detail.append(f"{commit['rounds']} rounds, "
                          f"{len(commit['vetoes'])} candidates vetoed")
        if commit.get("notes"):
            detail.append(f"{len(commit['notes'])} corroborations")
        if commit.get("unknown_judges"):
            detail.append("? unverifiable vetoes (judges undescribed)")
        for line in detail:
            print(f"{body}    ({line})")
        print(body.rstrip())


def _proposer_roster(ledger):
    """Every belief on record that answers cells (models, humans) -- the
    rows of the readings panel; judges verdict, they do not read."""
    roster = []
    for belief_id in ledger.beliefs():
        if belief_id.startswith(("policy:", "changeset:")):
            continue                       # runtime authors, not readers
        row = getattr(ledger, "belief_row", lambda _id: None)(belief_id)
        if row is not None and row["proposes"]:
            roster.append(belief_id)
        elif belief_id.startswith(("model:", "human:")):
            if belief_id not in roster:
                roster.append(belief_id)
    return roster


def _readings(ledger, commit):
    """Per changed cell: what every known proposer says -- ``-`` where one
    is silent.  Corroborations recorded by ``thinair evaluate`` live here
    from then on: the cache, made visible."""
    roster = _proposer_roster(ledger)
    if not roster:
        return
    print()
    print("readings:")
    many = len(commit["entities"]) > 1
    for entity in commit["entities"]:
        if many:
            print(f"  [{entity}]")
        for attr in sorted(commit["changes"]):
            print(f"  {attr}:")
            for belief_id in roster:
                label = belief_id[:44]
                latest = ledger.latest(entity, attr, belief=belief_id)
                if latest is None:
                    print(dim(f"    {label:<46} -"))
                    continue
                meta = latest.meta or {}
                tag = ("corroboration" if meta.get("corroboration") else
                       "frozen" if latest.frozen else
                       "resolving" if belief_id == commit["author"] else "")
                line = f"    {label:<46} {_value(latest.value, 40)}"
                if not latest.frozen:
                    line += f" (p {latest.p:g})"
                print(green(line) + (dim(f"   {tag}") if tag else ""))


def _matrix(ledger, commit):
    """Belief × attribute over the commit's cells: each cell is that
    belief's latest stated p -- green where its value matches what the
    commit holds, red where it differs, ``-`` where it is silent."""
    attrs = sorted(commit["changes"])
    rows: dict[str, dict] = {}
    for entity in commit["entities"]:
        for attr in attrs:
            held = commit["changes"][attr][0]
            for o in ledger.opinions(entity=entity, attr=attr):
                if o.belief.startswith(("policy:", "changeset:")):
                    continue
                rows.setdefault(o.belief, {})[attr] = \
                    (o.p, values_equal(o.value, held))
    if len(rows) < 2:
        return                       # a matrix of one voice says nothing
    width = max(8, *(min(len(a), 14) for a in attrs)) + 2
    print()
    print("matrix:  " + dim("(rows beliefs, columns attributes; "
                            "green agrees, red differs)"))
    print(dim(" " * 46 + "".join(a[:14].rjust(width) for a in attrs)))
    for belief_id, cells_ in rows.items():
        line = f"  {belief_id[:44]:<44}"
        for attr in attrs:
            got = cells_.get(attr)
            if got is None:
                line += dim("-".rjust(width))
                continue
            p, agrees = got
            cell = f"{p:.2f}".rjust(width)
            line += green(cell) if agrees else red(cell)
        print(line)


def cmd_show(ledger, args):
    commits = history(ledger)
    commit = _commit_at(commits, args.commit)
    print(f"{yellow('commit ' + commit['hash'])} "
          f"({', '.join(commit['entities'])})")
    print(f"Author: {commit['author']}")
    print(f"Date:   t={commit['t']:g}")
    print(f"Parent: {commit['parent'] or '(root)'}")
    if commit["kind"] == "episode":
        verdict = "" if commit["parent_matches"] else "  -- MISMATCH"
        print(f"Tree:   {commit['tree']} (parent tree "
              f"{commit['parent_tree']}, recorded "
              f"{commit['recorded_parent']}{verdict})")
    print()
    print(f"    {_message(commit)}")
    print()
    print(f"diff --thinair {_label(commit)}")
    for attr, (value, p, frozen) in sorted(commit["changes"].items()):
        if frozen:
            print(green(f"+ {attr} = {_value(value)}   (frozen)"))
            continue
        text, flag = _p_verdict(commit, attr, p)
        line = f"+ {attr} = {_value(value)}   ({text})"
        print(red(line) if flag == "violated" else
              yellow(line) if flag == "dissent" else green(line))
    if commit["kind"] == "episode":
        value, p = commit["returned"]
        print(f"return {_value(value)}   (p={p:g})")
    for value, p in commit["vetoes"]:
        print(red(f"- {_value(value)}   (vetoed at p={p:g})"))
    for belief, attr, value, p in commit.get("notes", ()):
        print(dim(f"note: {belief} read {attr} as {_value(value)} "
                  f"(p={p:g})"))
    for judge in commit.get("unknown_judges", ()):
        print(f"?     {judge} judged this; necessity unknown here")
    _matrix(ledger, commit)
    _readings(ledger, commit)
    print()


def cmd_status(ledger, args):
    commits = history(ledger)
    entities = {e for c in commits for e in c["entities"]}
    beliefs = ledger.beliefs()
    print(f"On store {getattr(ledger, 'path', '(json archive)')}")
    print(f"{len(ledger)} opinions, {len(commits)} commits, "
          f"{len(entities)} entities, {len(beliefs)} beliefs")
    if commits:
        last = commits[-1]
        print(f"HEAD is at {last['hash']} ({_label(last)}) "
              f"{_message(last)}")


def cmd_branch(ledger, args):
    commits = history(ledger)
    if args.delete:
        name = args.delete
        if not hasattr(ledger, "drop_entity"):
            sys.exit("fatal: archives are read-only; branch -d needs a store")
        if not any(name in c["entities"] for c in commits):
            sys.exit(f"error: branch '{name}' not found")
        dropped = ledger.drop_entity(name)
        print(f"Deleted branch {name} ({dropped} opinions dropped).")
        return
    heads: dict[str, dict] = {}
    for commit in commits:
        for entity in commit.get("heads", ()):
            heads[entity] = commit
    for entity in sorted(heads):
        head = heads[entity]
        count = sum(1 for c in commits if entity in c["entities"])
        print(f"  {branch_green(f'{entity:<20}')} {head['hash']}  "
              f"{count} commits")


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


def _try_commit(commits, rev):
    """Resolve a revision the way git names things, or ``None``:
    ``HEAD`` -> the newest commit overall; a branch name (any entity) ->
    that ref's tip; otherwise a hash prefix."""
    if not commits:
        return None
    if rev is None or rev == "HEAD":
        return commits[-1]
    on_branch = [c for c in commits if rev in c["entities"]]
    if on_branch:
        return on_branch[-1]                               # the ref's tip
    matches = [c for c in commits if c["hash"].startswith(rev)]
    return matches[-1] if matches else None


def _commit_at(commits, rev):
    if not commits:
        sys.exit("fatal: empty record")
    commit = _try_commit(commits, rev)
    if commit is None:
        sys.exit(f"fatal: bad revision '{rev}'")
    return commit


def _tree_at(commits, commit):
    """attr -> (value, p, frozen, author) as of the commit -- by walking the
    parent pointers, which need no entity: ancestry is in the hashes."""
    index = {c["hash"]: c for c in commits}
    chain, cursor = [], commit
    while cursor is not None:
        chain.append(cursor)
        cursor = index.get(cursor["parent"]) if cursor["parent"] else None
    cells: dict[str, tuple] = {}
    for earlier in reversed(chain):
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
        spoke = {o.belief for scope in commit["entities"]
                 for o in ledger.opinions(entity=scope)}
        print(f"beliefs on {commit['hash']} "
              f"({', '.join(commit['entities'])}):")
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
    rev, belief_arg = args.commit, args.belief
    commit = _try_commit(commits, rev)
    if commit is None and rev is not None and belief_arg == "*":
        # `thinair evaluate qwen3-35b`: not a revision, so it's the belief
        commit, belief_arg = _commit_at(commits, None), rev
    if commit is None:
        sys.exit(f"fatal: bad revision '{rev}'" if commits
                 else "fatal: empty record")
    cells = _tree_at(commits, commit)
    if belief_arg == "*":
        names = _model_names(ledger)
        if not names:
            sys.exit("fatal: no reconstructible belief has spoken here; "
                     "name one: thinair evaluate HEAD <model-name>")
    else:
        names = [belief_arg[len("model:"):].split("@T")[0]
                 if belief_arg.startswith("model:") else belief_arg]

    print(f"evaluate {', '.join(names)} @ {commit['hash']} "
          f"({', '.join(commit['entities'])})")
    # One reading per (commit, belief, attribute): a shared commit IS the
    # same content, so its refs share the evaluation -- the note lands on
    # the one commit every ref points at.
    entity = commit["entities"][0]
    for name in names:
        belief = model(name)
        for attr in sorted(cells):
            e = _RecordSnapshot(entity, cells, deriving=attr)
            stamp = belief.exposure(e, attr)
            prior = [o for ref in commit["entities"]
                     for o in ledger.opinions(entity=ref, attr=attr,
                                              belief=belief.id)]
            if any((o.meta or {}).get("at") == commit["hash"]
                   or (o.meta or {}).get("exposure") == stamp
                   for o in prior):
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


def cmd_diff(ledger, args):
    """Two trees, compared cell by cell -- ``thinair diff A...B``, ``A B``,
    or ``A`` alone against A's branch tip."""
    import re
    spec = args.commits
    if len(spec) == 1 and re.search(r"\.\.", spec[0]):
        first, second = re.split(r"\.{2,3}", spec[0], maxsplit=1)
    elif len(spec) == 2:
        first, second = spec
    else:
        first, second = spec[0], None
    commits = history(ledger)
    a = _commit_at(commits, first)
    if second:
        b = _commit_at(commits, second)
    else:
        b = [c for c in commits
             if set(c["entities"]) & set(a["entities"])][-1]
    if a["t"] > b["t"] and set(a["entities"]) & set(b["entities"]):
        a, b = b, a                        # oldest on the left, like a range
    tree_a, tree_b = _tree_at(commits, a), _tree_at(commits, b)
    print(f"diff --thinair a/{_label(a)}@{a['hash']} "
          f"b/{_label(b)}@{b['hash']}")
    for attr in sorted(set(tree_a) | set(tree_b)):
        va, vb = tree_a.get(attr), tree_b.get(attr)
        if va is not None and vb is not None \
                and values_equal(va[0], vb[0]) and va[1:3] == vb[1:3]:
            continue
        if va is not None:
            mark = "frozen" if va[2] else f"p={va[1]:g}"
            print(red(f"- {attr} = {_value(va[0])}   ({mark})"))
        if vb is not None:
            mark = "frozen" if vb[2] else f"p={vb[1]:g}"
            print(green(f"+ {attr} = {_value(vb[0])}   ({mark})"))


def _short_author(author: str) -> str:
    return author.split("@")[0]


def cmd_source(ledger, args):
    """The commit's tree rendered as source: frozen attributes plain,
    believed ones annotated -- ``source(thing)``, for any point in time."""
    commits = history(ledger)
    commit = _commit_at(commits, args.commit)
    tree = _tree_at(commits, commit)
    print(dim(f"# {_label(commit)} @ {commit['hash']} (t={commit['t']:g})"))
    for attr, (value, p, frozen, author) in sorted(tree.items()):
        line = f"{attr} = {_value(value, limit=70)}"
        if not frozen:
            line += dim(f"   # p={p:g} ← {_short_author(author)}")
        print(line)


def _builtin_roster() -> str:
    """The installed instruments, generated from the running package --
    what a strategy can reach for without writing a validator itself."""
    import inspect as _inspect
    from . import validators as V
    from .beliefs import Belief

    lines = [
        "", "---", "",
        "## Appendix: built-in beliefs in this installation",
        "",
        "Generated from the running package.  Generative: `model(name, "
        "think=, temperature=)` (an LLM), `human(name, interactive=)`, and "
        "any function via `thinair.fn` (code; its results freeze).  "
        "Validators (`from thinair.validators import <Name>`; most arrive "
        "automatically via `contract(...)` options):", ""]
    for name in getattr(V, "__all__", ()):
        obj = getattr(V, name, None)
        if not (_inspect.isclass(obj) and issubclass(obj, Belief)):
            continue
        doc = (obj.__doc__ or "").strip().splitlines()
        first = doc[0].rstrip(".") if doc else ""
        lines.append(f"- `{name}` -- {first}" if first else f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def cmd_ground(_ledger, _args):
    """The grounding -- GROUNDING.md verbatim, then a generated roster of
    the built-in beliefs this installation ships.  Nothing else on stdout,
    so the output pipes straight into an agent's context; the first command
    of an agentic session."""
    from importlib.resources import files
    sys.stdout.write(files("thinair").joinpath("GROUNDING.md").read_text(
        encoding="utf-8"))
    sys.stdout.write(_builtin_roster())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="thinair", description="inspect a thinair record, git-style")
    parser.add_argument("--store", help="path to opinions.db or a ledger.json")
    sub = parser.add_subparsers(dest="command", required=True)

    log = sub.add_parser("log", help="the commits, newest first")
    log.add_argument("entity", nargs="?", default=None)
    log.add_argument("--oneline", action="store_true")
    log.add_argument("--all", action="store_true",
                     help="every entity's chain (the default when no entity "
                          "is named)")
    log.add_argument("--decorate", action="store_true",
                     help="mark branch tips; the newest commit wears HEAD")
    log.add_argument("--graph", action="store_true",
                     help="draw the entity lanes")
    log.add_argument("-n", type=int, default=None)
    log.set_defaults(run=cmd_log)

    show = sub.add_parser("show", help="one commit in full")
    show.add_argument("commit", nargs="?", default=None,
                      help="a hash prefix, a branch (entity) name, or HEAD "
                           "(the default)")
    show.set_defaults(run=cmd_show)

    sub.add_parser("status", help="the store, summarized").set_defaults(
        run=cmd_status)
    branch = sub.add_parser("branch", help="entities and their heads")
    branch.add_argument("-d", "--delete", metavar="name", default=None,
                        help="delete a branch: drop the ref's opinions from "
                             "the store (archives are untouched)")
    branch.set_defaults(run=cmd_branch)

    blame = sub.add_parser("blame", help="every cell: who set it, when")
    blame.add_argument("entity")
    blame.set_defaults(run=cmd_blame)

    beliefs = sub.add_parser("beliefs", help="who spoke (or could) at a commit")
    beliefs.add_argument("commit", nargs="?", default=None)
    beliefs.set_defaults(run=cmd_beliefs)

    evaluate = sub.add_parser(
        "evaluate", help="consult beliefs against a commit's state "
                         "(spends calls, records corroborations)")
    evaluate.add_argument("commit", nargs="?", default=None,
                          help="hash prefix, branch name, or HEAD (default)")
    evaluate.add_argument("belief", nargs="?", default="*",
                          help="a model name, or * for every reconstructible "
                               "belief on record (default)")
    evaluate.set_defaults(run=cmd_evaluate)

    diff = sub.add_parser("diff", help="two trees, cell by cell")
    diff.add_argument("commits", nargs="+",
                      help="A...B, or A B, or A against its branch tip")
    diff.set_defaults(run=cmd_diff)

    source = sub.add_parser(
        "source", help="the commit's tree as annotated source")
    source.add_argument("commit", nargs="?", default=None)
    source.set_defaults(run=cmd_source)

    ground = sub.add_parser(
        "ground", help="print the measurement grounding; pipe it to an agent")
    ground.set_defaults(run=cmd_ground, needs_store=False)

    help_ = sub.add_parser("help", help="show help for thinair or a command")
    help_.add_argument("topic", nargs="?", default=None)

    def run_help(_ledger, args):
        if args.topic is None:
            parser.print_help()
            return
        chosen = sub.choices.get(args.topic)
        if chosen is None:
            sys.exit(f"fatal: no help for '{args.topic}'")
        chosen.print_help()

    help_.set_defaults(run=run_help, needs_store=False)

    args = parser.parse_args(argv)
    ledger = open_store(args.store) if getattr(args, "needs_store", True) \
        else None
    try:
        args.run(ledger, args)
    except KeyboardInterrupt:
        # Ctrl-C means *now*: flush what was already printed and recorded
        # (every ledger append is its own transaction), then leave without
        # giving any in-flight call a chance to keep the process alive.
        sys.stdout.flush()
        sys.stderr.write("\ninterrupted\n")
        sys.stderr.flush()
        os._exit(130)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
