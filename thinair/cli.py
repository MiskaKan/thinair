"""thinair -- the record, inspected like a repository.

Git's mapping, on the command line: the tree is the state hash, a commit is
whatever moved it, every entity is a branch with its own chain.  The
commands are deliberate copies:

    thinair log [entity] [--oneline] [-n N]     the commits, newest first
    thinair show [commit]                       one commit: the whole tree,
                                                the matrix, rounds, vetoes,
                                                notes
    thinair status                              the store, summarized
    thinair branch [-d name]                    entities and their heads
    thinair blame <entity>                      every cell: who set it, when
    thinair belief add|rm|list [file]           custom beliefs the client can
                                                call (.thinair/beliefs/)
    thinair evaluate [commit] [belief]          consult beliefs against a
                                                commit's state -- spends
                                                calls, records corroborations
    thinair diff A...B | A B | A                two trees, cell by cell
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

from .evaluate import history, similarity
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


def shade(overlap: float, text: str) -> str:
    """Paint by similarity, in the terminal's own palette: exact agreement
    is the theme's green, no overlap its red, and partial text or numeric
    overlap lands between as faded green, yellow, faded red -- typed
    matching via ``evaluate.similarity``, so "how green" means the same
    thing a settlement's ``~`` metric means."""
    if overlap >= 1.0:
        return green(text)
    if overlap >= 0.66:
        return _paint("2;32", text)
    if overlap >= 0.33:
        return yellow(text)
    if overlap > 0.0:
        return _paint("2;31", text)
    return red(text)


def _signature_text(commit, attr, p) -> str:
    """``p 0.95 ±0.02`` -- or ``p 0.95 ±?`` when fewer than two agreeing
    voices have measured the cell: an unmeasured spread says so instead of
    hiding, because ``(p 0.95)`` alone reads as more settled than it is."""
    view = (commit.get("consensus") or {}).get(attr) or {}
    dev = view.get("dev")
    return f"p {p:g} ±" + (f"{dev:.2f}" if dev is not None else "?")


def _signature(commit, attr, p, pad=0) -> str:
    """The trust signature of a believed cell: ``(p 0.95 ±0.02)``, painted
    on the overlap gradient.

    The number is the resolving belief's honest p with the agreeing
    voices' spread beside it; the color is the record's verdict on the
    *value* -- green when every recorded reading holds it, sliding toward
    red as dissenting readings land farther away (their mean
    ``similarity`` to the held value folds into the score).  A lone
    unchecked voice stays unpainted: green must mean "the record agrees",
    never "nobody looked".  A violated declared expectation is always the
    theme's red.
    """
    view = (commit.get("consensus") or {}).get(attr) or {}
    expect = (commit.get("expect") or {}).get(attr) or {}
    text = _signature_text(commit, attr, p)
    if pad:
        text = text.rjust(pad)
    dev = view.get("dev")
    bounds = expect.get("p")
    if (bounds and not (bounds[0] <= p <= bounds[1])) \
            or (expect.get("deviation") is not None and dev is not None
                and dev > expect["deviation"]):
        return red(text)
    voices = view.get("n", 1) + view.get("dissent", 0)
    if voices <= 1:
        return text
    score = (view.get("n", 1)
             + view.get("dissent", 0) * view.get("similarity", 0.0)) / voices
    return shade(score, text)


def _reach(ledger):
    """Coverage for the signature's parens: of the mechanisms this client
    could hear on a cell, the fraction that have spoken -- judged by the
    same rules as the matrix's ``?`` cells, so green parens mean "the
    panel is complete" and red parens mean "almost nobody has been
    asked".  ``None`` on archives, where askability is unknowable."""
    if not hasattr(ledger, "belief_rows"):
        return None
    custom = _load_custom(ledger)
    askable = {row["id"] for row in ledger.belief_rows()
               if _can_fill(ledger, row["id"], custom)}

    def coverage(commit, attr):
        bases = set(commit["changes"])
        spoken = set()
        for entity in commit["entities"]:
            for o in ledger.opinions(entity=entity, attr=attr):
                if o.belief.startswith(("policy:", "changeset:")):
                    continue
                if not _visible_at(o, commit):
                    continue
                spoken.add(_base_id(ledger, o.belief, bases))
        pool = spoken | askable
        return len(spoken) / len(pool) if pool else 1.0

    return coverage


def _signed(commit, attr, p, reach=None) -> str:
    """The signature in its parens, parens shaded by coverage."""
    body = _signature(commit, attr, p)
    if reach is None:
        return f"({body})"
    cov = reach(commit, attr)
    return shade(cov, "(") + body + shade(cov, ")")


def _message(commit, reach=None) -> str:
    if commit["kind"] == "episode":
        return commit["message"]
    attr, (value, p, frozen) = next(iter(commit["changes"].items()))
    arrow = "=" if frozen else "⇒"
    stated = f"{attr} {arrow} {_value(value)}"
    if not frozen:
        stated += f" {_signed(commit, attr, p, reach)}"
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
    # No per-commit branch column: membership IS ancestry -- a ref requires
    # its chain to exist, so the tips say everything.  Decorations are on by
    # default, as in git; --no-decorate turns them off.
    decorations = {} if args.no_decorate else _decorations(commits)
    reach = _reach(ledger)
    for star, bar, commit in _lanes(commits):
        head = f"{star} " if args.graph else ""
        body = f"{bar} " if args.graph else ""
        decor = decorations.get(commit["hash"], "")
        if args.oneline:
            print(f"{head}{yellow(commit['hash'])}{decor} "
                  f"[{commit['kind']}] {_message(commit, reach)}")
            continue
        print(f"{head}{yellow('commit ' + commit['hash'])}{decor}")
        print(f"{body}Author: {commit['author']}")
        print(f"{body}Date:   t={commit['t']:g}")
        print(body.rstrip())
        print(f"{body}    {_message(commit, reach)}")
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


def _visible_at(o, commit) -> bool:
    """Does this opinion belong in the commit's panel?

    The panel unwinds to the moment: an opinion on record by the commit's
    time is in; anything later is out -- a next week's override must not
    haunt last week's commit.  The one exception is a corroboration
    *about* this commit: evaluate stamps the hash it read (``meta.at``),
    and the ``corroborate`` verb's notes are time-attached to their commit
    by ``history`` -- both are later measurements of exactly this state.
    The cutoff is the commit's negotiation *end*: its own judges speak
    after the candidate whose t is the Date.
    """
    if o.t <= commit.get("end", commit["t"]):
        return True
    meta = o.meta or {}
    if not meta.get("corroboration"):
        return False
    if meta.get("at") == commit["hash"]:
        return True
    return any(o.belief == belief and o.attr == attr
               and values_equal(o.value, value) and o.p == p
               for belief, attr, value, p in commit.get("notes", ()))


def _readings(ledger, commit):
    """Per changed cell: what every known proposer says -- ``-`` where one
    is silent.  Opinions pool across the commit's refs (the commit is the
    content; refs are pointers to it), latest per belief.  Corroborations
    recorded by ``thinair evaluate`` live here from then on: the cache,
    made visible."""
    roster = _proposer_roster(ledger)
    if not roster:
        return
    print()
    print("readings:")
    for attr in sorted(commit["changes"]):
        print(f"  {attr}:")
        for belief_id in roster:
            label = belief_id[:44]
            latest = None
            for entity in commit["entities"]:
                for got in ledger.opinions(entity=entity, attr=attr,
                                           belief=belief_id):
                    if not _visible_at(got, commit):
                        continue
                    if latest is None or got.t > latest.t:
                        latest = got
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
            # shaded like the matrix: how far this reading sits from what
            # the commit holds
            held = commit["changes"][attr][0]
            print(shade(similarity(latest.value, held), line)
                  + (dim(f"   {tag}") if tag else ""))


def _base_id(ledger, belief_id, attrs):
    """A scoped wrapper's row is its inner mechanism's row: the ``@attr``
    suffix says nothing the matrix column doesn't already say.  The belief
    table knows (``kind == "Scoped"``); cold archives fall back to the id
    convention -- a model id's ``@T0.2/...`` tail never names an attribute
    of the tree."""
    if "@" not in belief_id:
        return belief_id
    head, tail = belief_id.rsplit("@", 1)
    row = getattr(ledger, "belief_row", lambda _id: None)(belief_id) or {}
    if row.get("kind") == "Scoped" or tail in attrs:
        return head
    return belief_id


def _can_fill(ledger, belief_id, custom):
    """Could ``thinair evaluate`` fill this row's empty cells from here?
    Models always (the id is the configuration); humans, code and runtime
    authors never (no one to call); anything else when its class -- built
    in or from a registered belief file -- plus its stored config rebuild
    it.  Deliberately durable-only: a warm registry must not make a cell
    look reachable that a fresh process could not reach."""
    row = getattr(ledger, "belief_row", lambda _id: None)(belief_id) or {}
    if belief_id.startswith("model:") or row.get("kind") == "ModelBelief":
        return True
    if belief_id.startswith(("human:", "code:", "policy:", "changeset:")) \
            or row.get("kind") in ("HumanBelief", "CodeBelief", "Scoped"):
        return False
    config = row.get("config")
    if not config or "__unjson__" in config:
        return False
    from . import validators as V
    cls = getattr(V, row.get("kind") or "", None) \
        or (custom or {}).get(row.get("kind"))
    return isinstance(cls, type)


def _matrix_lines(ledger, commit, held, extra=(), active=None, always=False,
                  custom=None, stats=None, frozen_by=None):
    """Belief × attribute over the commit's whole tree, as rendered lines:
    each cell is that belief's latest stated p, shaded by how much its
    value overlaps what the tree holds (``evaluate.similarity``): pure
    green is exact agreement, pure red is no overlap at all, and partial
    text or numeric matches land between.  A scoped wrapper pools into
    its inner mechanism's row -- the column already names the attribute.
    An empty cell says why it is empty: ``?`` where the belief could be
    asked and never was (``thinair evaluate`` fills those in; ``active``
    marks the cell being consulted right now with ``…``), ``x`` where
    this client has no way to call it.  ``extra`` guarantees a row for
    beliefs about to speak.  Opinions pool across the commit's refs: one
    commit, one panel."""
    attrs = sorted(held)
    bases = set(attrs)
    rows: dict[str, dict] = {}
    for entity in commit["entities"]:
        for attr in attrs:
            for o in ledger.opinions(entity=entity, attr=attr):
                if o.belief.startswith(("policy:", "changeset:")):
                    continue
                if not _visible_at(o, commit):
                    continue         # the panel unwinds to the moment
                rows.setdefault(_base_id(ledger, o.belief, bases), {})[attr] \
                    = (o.p, similarity(o.value, held[attr]))
    for belief_id in extra:
        rows.setdefault(_base_id(ledger, belief_id, bases), {})
    if not always and len(rows) < 2:
        return []                    # a matrix of one voice says nothing
    footer = {attr: (stats or {}).get(attr) for attr in attrs}
    footer_texts = ["p 1.00 ±0.00" if frozen
                    else _signature_text(owner, attr, p)
                    for attr, cell in footer.items() if cell is not None
                    for owner, p, frozen in (cell,)]
    width = max(8, *(min(len(a), 14) for a in attrs),
                *(len(text) for text in footer_texts)) + 2
    lines = ["matrix:  " + dim("(rows beliefs, columns attributes; shade = "
                               "value overlap, green exact ... red none; "
                               "? never asked, x unreachable from here)"),
             dim(" " * 46 + "".join(a[:14].rjust(width) for a in attrs))]
    for belief_id, cells_ in rows.items():
        fillable = _can_fill(ledger, belief_id, custom)
        line = f"  {belief_id[:44]:<44}"
        for attr in attrs:
            got = cells_.get(attr)
            if got is not None:
                # a recorded reading always shows, frozen column or not
                p, overlap = got
                line += shade(overlap, f"{p:.2f}".rjust(width))
                continue
            if active == (belief_id, attr):
                line += yellow("…".rjust(width))
                continue
            if (frozen_by or {}).get(attr) is not None:
                # pinned by fiat: an empty cell is not an invitation
                line += dim("frozen".rjust(width))
                continue
            line += dim(("?" if fillable else "x").rjust(width))
        lines.append(line)
    if stats is not None:
        # the bottom row: what the tree holds, with the cell's consensus.
        # The one place a frozen cell speaks in probability: the fiat is
        # p 1.00, and its deviation measures how close the declared fact
        # sits to what other beliefs have read -- ±0.00 until someone has.
        line = dim(f"  {'(held)':<44}")
        for attr in attrs:
            got = footer.get(attr)
            if got is None:
                line += dim("-".rjust(width))
                continue
            owner, p, frozen = got
            if frozen:
                view = (owner.get("consensus") or {}).get(attr) or {}
                text = f"p 1.00 ±{view.get('dev') or 0.0:.2f}".rjust(width)
                voices = view.get("n", 1) + view.get("dissent", 0)
                if voices <= 1:
                    line += dim(text)
                else:
                    score = (view.get("n", 1) + view.get("dissent", 0)
                             * view.get("similarity", 0.0)) / voices
                    line += shade(score, text)
                continue
            line += _signature(owner, attr, p, pad=width)
        lines.append(line)
    return lines


def _matrix(ledger, commit, held, stats=None, frozen_by=None):
    lines = _matrix_lines(ledger, commit, held, custom=_load_custom(ledger),
                          stats=stats, frozen_by=frozen_by)
    if lines:
        print()
        for line in lines:
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
    # the whole tree as of this commit, source-style; what this commit
    # moved is highlighted, the rest is context
    tree = _tree_at(commits, commit)
    stats = _cell_stats(commits, commit)
    reach = _reach(ledger)
    print(f"diff --thinair {_label(commit)}")
    for attr in sorted(set(tree) | set(commit["changes"])):
        if attr not in commit["changes"]:
            value, p, frozen, _author = tree[attr]
            if frozen:
                print(dim(f"  {attr} = {_value(value)}   (frozen)"))
                continue
            owner, held_p, _frozen = stats.get(attr) or (commit, p, frozen)
            print(dim(f"  {attr} = {_value(value)}")
                  + f"   {_signed(owner, attr, held_p, reach)}")
            continue
        value, p, frozen = commit["changes"][attr]
        if frozen:
            print(green(f"+ {attr} = {_value(value)}   (frozen)"))
            continue
        print(green(f"+ {attr} = {_value(value)}")
              + f"   {_signed(commit, attr, p, reach)}")
    if commit["kind"] == "episode":
        value, p = commit["returned"]
        print(f"return {_value(value)}   (p={p:g})")
    for value, p in commit["vetoes"]:
        print(red(f"- {_value(value)}   (vetoed at p={p:g})"))
    for belief, attr, value, p in commit.get("notes", ()):
        held = commit["changes"].get(attr)
        overlap = ""
        if held is not None and not values_equal(value, held[0]):
            overlap = f", ~{similarity(value, held[0]):.2f}"
        print(dim(f"note: {belief} read {attr} as {_value(value)} "
                  f"(p={p:g}{overlap})"))
    for judge in commit.get("unknown_judges", ()):
        print(f"?     {judge} judged this; necessity unknown here")
    _matrix(ledger, commit, {attr: cell[0] for attr, cell in tree.items()},
            stats=stats,
            frozen_by={attr: cell[3] for attr, cell in tree.items()
                       if cell[2]})
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
    reach = _reach(ledger)
    for attr in sorted(latest):
        commit = latest[attr]
        value, p, frozen = commit["changes"][attr]
        tail = "(frozen)" if frozen else _signed(commit, attr, p, reach)
        print(f"{commit['hash']} ({commit['author'][:34]:<34} t={commit['t']:g}) "
              f"{attr} = {_value(value)}   {tail}")


class _RecordSnapshot:
    """A sealed snapshot built from a commit's tree -- no live Thing needed.

    Carries the same surface a belief reads off a real snapshot.  Contracts
    and purpose are class code, absent here, so a reading taken this way has
    its *own* exposure -- a leaner context than the original run's -- and
    the stamp records exactly that; settlement weighs it accordingly.
    """

    def __init__(self, entity, cells, deriving, panel=()):
        object.__setattr__(self, "_cells", {
            attr: Opinion(belief=author or "record", entity=entity, attr=attr,
                          value=value, p=p, t=1.0, frozen=frozen)
            for attr, (value, p, frozen, author) in cells.items()
            if attr != deriving})
        object.__setattr__(self, "__entity__", entity)
        object.__setattr__(self, "__beliefs__", list(panel))

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


def _cell_stats(commits, commit):
    """``attr -> (owning commit, p, frozen)``: where each cell of the
    commit's tree last moved -- the trust signature renders from the
    owner's consensus."""
    index = {c["hash"]: c for c in commits}
    chain, cursor = [], commit
    while cursor is not None:
        chain.append(cursor)
        cursor = index.get(cursor["parent"]) if cursor["parent"] else None
    held: dict[str, tuple] = {}
    for owner in reversed(chain):
        for attr, (_value, p, frozen) in owner["changes"].items():
            already = held.get(attr)
            if frozen or already is None or not already[2]:
                held[attr] = (owner, p, frozen)
    return held


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


# -- custom beliefs: user instruments the client can rebuild ---------------

#: modules already imported this process, by real path -- re-importing a
#: belief file would mint duplicate classes and trip invariant 6's
#: same-id-same-type check.
_CUSTOM_MODULES: dict = {}


def _beliefs_dir(ledger):
    """``<store dir>/beliefs`` -- next to opinions.db; archives have none."""
    path = getattr(ledger, "path", None)
    if not path:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(path)), "beliefs")


def _import_module(path):
    real = os.path.realpath(path)
    if real in _CUSTOM_MODULES:
        return _CUSTOM_MODULES[real]
    import importlib.util
    name = "thinair_beliefs_" + os.path.splitext(os.path.basename(path))[0]
    try:
        spec = importlib.util.spec_from_file_location(name, real)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        module = None
    _CUSTOM_MODULES[real] = module
    return module


def _load_custom(ledger):
    """Import every registered belief file; return {class name: class}.

    Importing also registers any module-level belief *instances* under
    their durable ids, which is the other door ``evaluate`` walks through
    (``lookup``).  A file that fails to import is skipped -- ``thinair
    belief list`` shows the damage."""
    import glob

    classes: dict[str, type] = {}
    directory = _beliefs_dir(ledger)
    if not directory or not os.path.isdir(directory):
        return classes
    from .beliefs import Belief

    for path in sorted(glob.glob(os.path.join(directory, "*.py"))):
        module = _import_module(path)
        if module is None:
            continue
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Belief):
                classes[obj.__name__] = obj
    return classes


def cmd_belief(ledger, args):
    """``thinair belief add <file.py> | rm <name> | list`` -- the custom
    instruments this store's client can rebuild, kept in
    ``.thinair/beliefs/``."""
    import glob
    import shutil

    directory = _beliefs_dir(ledger)
    if directory is None:
        sys.exit("fatal: custom beliefs live beside a store; archives "
                 "have none")
    if args.action == "list":
        from .beliefs import Belief
        for path in sorted(glob.glob(os.path.join(directory, "*.py"))):
            module = _import_module(path)
            if module is None:
                print(f"  {os.path.basename(path):<24} (does not import)")
                continue
            names = [obj.__name__ for obj in vars(module).values()
                     if isinstance(obj, type) and issubclass(obj, Belief)]
            print(f"  {os.path.basename(path):<24} {', '.join(names) or '-'}")
        return
    if not args.name:
        sys.exit(f"fatal: thinair belief {args.action} needs a name")
    if args.action == "add":
        if not os.path.exists(args.name):
            sys.exit(f"fatal: no such file: {args.name}")
        os.makedirs(directory, exist_ok=True)
        target = os.path.join(directory, os.path.basename(args.name))
        shutil.copyfile(args.name, target)
        module = _import_module(target)
        if module is None:
            os.remove(target)
            sys.exit(f"fatal: {args.name} does not import; not registered")
        from .beliefs import Belief
        names = [obj.__name__ for obj in vars(module).values()
                 if isinstance(obj, type) and issubclass(obj, Belief)]
        if not names:
            os.remove(target)
            sys.exit(f"fatal: {args.name} defines no Belief subclass")
        print(f"Registered {os.path.basename(target)}: {', '.join(names)}")
        return
    target = os.path.join(directory, args.name if args.name.endswith(".py")
                          else args.name + ".py")
    if not os.path.exists(target):
        sys.exit(f"error: no registered belief file '{args.name}'")
    os.remove(target)
    print(f"Removed {os.path.basename(target)}.")


class _HeldValue:
    """A stub proposer serving the record's held value, so a rebuilt judge
    has a candidate to judge -- the tree speaks for itself."""

    id = "record"
    proposes = True
    necessary = False

    def __init__(self, cells):
        self._cells = cells

    def __call__(self, e, attr):
        spec = self._cells.get(attr)
        if spec is None:
            return None
        value, p, frozen, _author = spec
        return (value, 1.0 if frozen else p)


def _reconstructible_judges(ledger, custom=None):
    """Validators rebuilt from the record: the belief table stores each
    instrument's constructor configuration (invariant 6 -- the id names a
    fixed configuration; the config column makes it loadable), and the
    classes come from ``thinair.validators`` plus the store's own
    registered belief files (``thinair belief add``).  A live registered
    instance under the row's id is used as-is -- that is how module-level
    instances in belief files arrive.  Scoped wrappers are skipped: their
    inner mechanism is described separately and judges every cell.
    Anything whose config did not survive JSON is skipped, silently: a
    judge that cannot be rebuilt exactly must not be rebuilt at all.
    """
    import json as _json
    from . import validators as V
    from .beliefs import lookup

    rows = getattr(ledger, "belief_rows", lambda: [])()
    out = []
    for row in rows:
        if row["kind"] in (None, "Scoped", "ModelBelief", "HumanBelief",
                           "CodeBelief"):
            continue
        if not _can_fill(ledger, row["id"], custom):
            continue                 # the same gate the matrix's ? obeys:
                                     # warm and cold reach identically
        live = lookup(row["id"])
        if live is not None:
            out.append(live)         # already constructed here: use as-is
            continue
        try:
            spec = _json.loads(row["config"])
            cls = getattr(V, row["kind"], None) \
                or (custom or {}).get(row["kind"])
            belief = cls(*spec.get("args", ()),
                         **dict(spec.get("kwargs", {}), id=row["id"],
                                necessary=row["necessary"],
                                veto_line=row["veto_line"]))
        except Exception:
            continue
        out.append(belief)
    return out


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

    # One reading per (commit, belief, attribute): a shared commit IS the
    # same content, so its refs share the evaluation -- the note lands on
    # the one commit every ref points at.  On a terminal the commit's
    # matrix sits below the log and fills itself in, one cell at a time
    # (``…`` marks the consultation in flight); piped output stays plain
    # lines with the finished table at the end.  Models re-read; validators
    # rebuilt from the record re-judge the held tree.
    entity = commit["entities"][0]
    held = {attr: spec[0] for attr, spec in cells.items()}
    frozen_by = {attr: spec[3] for attr, spec in cells.items() if spec[2]}
    include = getattr(args, "include_frozen", None)
    custom = _load_custom(ledger)
    instruments = [model(name) for name in names]
    if belief_arg == "*":
        instruments += _reconstructible_judges(ledger, custom)
    extra = [b.id for b in instruments]
    print(f"evaluate {', '.join(names)}"
          + (f" + {len(instruments) - len(names)} rebuilt validators"
             if len(instruments) > len(names) else "")
          + f" @ {commit['hash']} ({', '.join(commit['entities'])})")
    live = _tty()
    block = 0

    def freshly():
        """Consensus for the footer, re-derived so each landed reading
        moves the (held) row while the table fills."""
        fresh = history(ledger)
        now = next((c for c in fresh if c["hash"] == commit["hash"]), commit)
        return _cell_stats(fresh, now)

    def redraw(active=None):
        nonlocal block
        if not live:
            return
        if block:
            sys.stdout.write(f"\x1b[{block}A\x1b[J")
        lines = _matrix_lines(ledger, commit, held, extra=extra,
                              active=active, always=True, custom=custom,
                              stats=freshly(), frozen_by=frozen_by)
        for line in lines:
            print(line)
        block = len(lines)

    def emit(line):
        nonlocal block
        if live and block:
            sys.stdout.write(f"\x1b[{block}A\x1b[J")
            block = 0
        print(line)
        redraw()

    skipped = [attr for attr in sorted(frozen_by)
               if include not in ("*", attr)]
    if skipped:
        print(dim(f"  frozen, not consulted: {', '.join(skipped)}   "
                  "(--include-frozen consults them)"))
    redraw()
    for instrument in instruments:
        reads = hasattr(instrument, "exposure")     # a model re-reads;
        for attr in sorted(cells):                  # a judge re-judges
            if attr in frozen_by and include not in ("*", attr):
                continue                            # pinned by fiat: not a
                                                    # question, by default
            scope = getattr(instrument, "scope", None)
            if scope is not None and scope != attr:
                continue
            panel = [] if reads else [_HeldValue(cells)]
            e = _RecordSnapshot(entity, cells, deriving=attr, panel=panel)
            stamp = instrument.exposure(e, attr) if reads else None
            prior = [o for ref in commit["entities"]
                     for o in ledger.opinions(entity=ref, attr=attr,
                                              belief=instrument.id)]
            if any((o.meta or {}).get("at") == commit["hash"]
                   or (stamp is not None
                       and (o.meta or {}).get("exposure") == stamp)
                   for o in prior):
                emit(f"  {attr:<18} {instrument.id}: asked and answered")
                continue
            redraw(active=(instrument.id, attr))
            try:
                got = instrument(e, attr)
            except Exception:                       # a judge choking on a
                got = None                          # cell is "no opinion"
            if got is None:
                redraw()
                continue
            ledger.add(Opinion(
                belief=instrument.id, entity=entity, attr=attr,
                value=+got, p=~got, frozen=False,
                meta=dict(getattr(got, "meta", None) or {},
                          corroboration=True, at=commit["hash"])))
            if reads:
                verdict = "agrees" if values_equal(+got, held[attr]) else \
                    f"DIFFERS from {_value(held[attr])}"
                emit(f"  {attr:<18} ⇒ {_value(+got)} (p {~got:g})  {verdict}")
            else:
                emit(f"  {attr:<18} {instrument.id[:40]} judges p {~got:g}")
    if not live:
        for line in _matrix_lines(ledger, commit, held, extra=extra,
                                  always=True, custom=custom,
                                  stats=freshly(), frozen_by=frozen_by):
            print(line)


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
                     help="mark branch tips (the default, as in git; kept "
                          "for the muscle memory)")
    log.add_argument("--no-decorate", action="store_true",
                     help="plain hashes, no ref names")
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

    belief = sub.add_parser(
        "belief", aliases=["beliefs"],
        help="custom beliefs the client can call: add <file.py> / "
             "rm <name> / list")
    belief.add_argument("action", choices=("add", "rm", "list"))
    belief.add_argument("name", nargs="?", default=None)
    belief.set_defaults(run=cmd_belief)

    evaluate = sub.add_parser(
        "evaluate", help="consult beliefs against a commit's state "
                         "(spends calls, records corroborations)")
    evaluate.add_argument("commit", nargs="?", default=None,
                          help="hash prefix, branch name, or HEAD (default)")
    evaluate.add_argument("belief", nargs="?", default="*",
                          help="a model name, or * for every reconstructible "
                               "belief on record (default)")
    evaluate.add_argument("--include-frozen", nargs="?", const="*",
                          default=None, metavar="attr",
                          help="consult frozen columns too -- every one, or "
                               "just the named attribute")
    evaluate.set_defaults(run=cmd_evaluate)

    diff = sub.add_parser("diff", help="two trees, cell by cell")
    diff.add_argument("commits", nargs="+",
                      help="A...B, or A B, or A against its branch tip")
    diff.set_defaults(run=cmd_diff)

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
