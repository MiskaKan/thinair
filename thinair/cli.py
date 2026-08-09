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
import contextlib
import json
import os
import subprocess
import sys

from .evaluate import history, similarity
from .ledger import Ledger, Opinion, values_equal
from .store import DEFAULT_PATH, SqliteLedger

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

#: True while stdout is a pager we spawned ourselves -- the reader is still
#: a terminal, so color survives the pipe (git's behavior).
_PAGING = False


def _tty() -> bool:
    return (_PAGING or sys.stdout.isatty()) and not os.environ.get("NO_COLOR")


@contextlib.contextmanager
def _pager(enabled: bool):
    """Long output lands in a pager, git's way: only when stdout is a
    terminal, quitting immediately if everything fits on one screen
    (``less -FRX``).  ``THINAIR_PAGER`` (then ``PAGER``) overrides the
    command; ``THINAIR_PAGER=`` or ``cat`` disables paging outright."""
    global _PAGING
    command = os.environ.get("THINAIR_PAGER", os.environ.get("PAGER") or "less")
    if not enabled or not sys.stdout.isatty() or command in ("", "cat"):
        yield
        return
    env = dict(os.environ)
    env.setdefault("LESS", "FRX")
    try:
        process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE,
                                   text=True, env=env)
    except OSError:                                    # pragma: no cover
        yield
        return
    previous, sys.stdout = sys.stdout, process.stdin
    _PAGING = True
    try:
        yield
    except BrokenPipeError:
        pass                                           # the reader quit: done
    finally:
        _PAGING = False
        sys.stdout = previous
        try:
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        process.wait()


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


def shade(overlap: float, text: str, bold: bool = False) -> str:
    """Paint by similarity, in the terminal's own palette: exact agreement
    is the theme's green, no overlap its red, and partial text or numeric
    overlap lands between as faded green, yellow, faded red -- typed
    matching via ``evaluate.similarity``, so "how green" means the same
    thing a settlement's ``~`` metric means.  ``bold`` marks the resolving
    reading (bold replaces the faint variants: emphasis and fading do not
    mix)."""
    if overlap >= 1.0:
        code = "32"
    elif overlap >= 0.66:
        code = "2;32"
    elif overlap >= 0.33:
        code = "33"
    elif overlap > 0.0:
        code = "2;31"
    else:
        code = "31"
    if bold:
        code = "1;" + code.removeprefix("2;")
    return _paint(code, text)


def _signature_text(commit, attr, p) -> str:
    """``p 0.95 ±0.02`` -- always numeric; an unmeasured spread reads
    ``±0.00``, and the parens' coverage color is what says nobody has
    measured yet."""
    view = (commit.get("consensus") or {}).get(attr) or {}
    dev = view.get("dev")
    return f"p {p:.2f} ±{dev if dev is not None else 0.0:.2f}"


def _verdict(commit, attr, p):
    """``(score, violated)``: agreement among the cell's independent
    *readers*, and whether a declared expectation is missed.

    ``score=None`` means the cell is **unopposed**: one reading, which
    nothing on record could have contradicted.  Judges never count --
    they verdict the candidate they were handed, so their concord is
    purchased by construction and earns nothing (Pillar III).  ``agree=``
    used to flatter exactly this way; now it refuses to.
    """
    view = (commit.get("consensus") or {}).get(attr) or {}
    expect = (commit.get("expect") or {}).get(attr) or {}
    dev = view.get("dev")
    bounds = expect.get("p")
    violated = (bool(bounds) and not (bounds[0] <= p <= bounds[1])) \
        or (expect.get("deviation") is not None and dev is not None
            and dev > expect["deviation"])
    readers = view.get("readers")
    if readers is None:
        # an archive from before readers were tallied: all voices, the
        # old flattering math -- better than pretending to know
        voices = view.get("n", 1) + view.get("dissent", 0)
        score = 1.0 if voices <= 1 else \
            (view.get("n", 1)
             + view.get("dissent", 0) * view.get("similarity", 0.0)) / voices
        return score, violated
    if readers <= 1:
        return None, violated
    dissenting = view.get("readers_dissent", 0)
    score = ((readers - dissenting)
             + dissenting * view.get("readers_similarity", 0.0)) / readers
    return score, violated


def _signature(commit, attr, p, pad=0) -> str:
    """The trust signature of a believed cell: ``(p 0.95 ±0.02)``, painted
    on the overlap gradient.

    The number is the resolving belief's honest p with the agreeing
    voices' spread beside it; the color is the record's verdict on the
    *value* -- green when independent readers agree, sliding toward red
    as dissenting readings land farther away, and plain (dim) while the
    cell is unopposed: a single reading earns no color either way.

    The ``±`` wears its own color, from the **min-max range** of the
    recorded ps rather than their deviation -- one voice far from the
    rest barely moves a deviation, and the range refuses to average it
    away.  The printed number stays the deviation.  A violated declared
    expectation is always the theme's red.
    """
    view = (commit.get("consensus") or {}).get(attr) or {}
    dev = view.get("dev")
    spread = view.get("range")
    head = f"p {p:.2f} "
    tail = f"±{dev if dev is not None else 0.0:.2f}"
    lead = " " * max(0, pad - len(head) - len(tail)) if pad else ""
    score, violated = _verdict(commit, attr, p)
    if violated:
        return lead + red(head + tail)
    if score is None:
        # unopposed: no independent reader has spoken, so neither the value
        # match nor the spread is evidence yet -- the whole signature waits
        # dim.  (Coloring the ± off the judges' ps here would contradict the
        # p beside it; the parens still say whether anyone could be asked.)
        return lead + dim(head + tail)
    head = shade(score, head)
    tail = dim(tail) if spread is None else shade(1.0 - spread, tail)
    return lead + head + tail


def _attachments(ledger, custom):
    """Which mechanisms belong to which cells -- the one notion of "the
    cell's panel" that the matrix's ``?`` marks, the parens' coverage and
    ``evaluate``'s consultations all obey.  ``(models, by_attr,
    unclaimed)``: every model reads any cell; a mechanism claimed by
    scoped wrappers belongs exactly to the attributes those wrappers
    name (priority's enum is never customer's business); a rebuildable
    mechanism no wrapper claims was attached directly, so it judges
    anywhere."""
    rows = getattr(ledger, "belief_rows", lambda: [])()
    models, claimed, by_attr = set(), set(), {}
    for row in rows:
        if row["kind"] == "ModelBelief" or row["id"].startswith("model:"):
            models.add(row["id"])
            continue
        if row["kind"] == "Scoped" and "@" in row["id"]:
            base, scoped_to = row["id"].rsplit("@", 1)
            claimed.add(base)
            if _can_fill(ledger, base, custom):
                by_attr.setdefault(scoped_to, set()).add(base)
    unclaimed = {row["id"] for row in rows
                 if row["id"] not in claimed and row["id"] not in models
                 and _can_fill(ledger, row["id"], custom)}
    return models, by_attr, unclaimed


def _askable(attachments, attr):
    models, by_attr, unclaimed = attachments
    return models | by_attr.get(attr, set()) | unclaimed


def _reach(ledger):
    """Coverage for the signature's parens: of the cell's panel
    (`_attachments`), the fraction that have spoken -- green parens mean
    the panel is complete, red parens mean almost nobody has been asked.
    ``None`` on archives, where askability is unknowable."""
    if not hasattr(ledger, "belief_rows"):
        return None
    attachments = _attachments(ledger, _load_custom(ledger))

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
        pool = spoken | _askable(attachments, attr)
        return (len(spoken), len(pool)) if pool else (1, 1)

    return coverage


#: --ai-readable: the signature's color channels, said in text -- for
#: readers (pipes, models) that cannot see a terminal's palette.
_AI_READABLE = False


def _signed(commit, attr, p, reach=None) -> str:
    """The signature in its parens, parens shaded by coverage."""
    body = _signature(commit, attr, p)
    cov = reach(commit, attr) if reach is not None else None
    if _AI_READABLE:
        score, violated = _verdict(commit, attr, p)
        # ``agree=unopposed``: one reader, nothing could have disagreed --
        # not evidence, and never printed as a flattering 1.00
        parts = ["agree=unopposed" if score is None else f"agree={score:.2f}"]
        if cov is not None:
            parts.append(f"asked={cov[0]}/{cov[1]}")
        if violated:
            parts.append("expect-violated")
        body += "  " + " ".join(parts)
    if cov is None:
        return f"({body})"
    fraction = cov[0] / cov[1]
    return shade(fraction, "(") + body + shade(fraction, ")")


def _message(commit, reach=None) -> str:
    if commit["kind"] in ("episode", "belief"):
        return commit["message"]
    attr, (value, p, frozen) = next(iter(commit["changes"].items()))
    arrow = "=" if frozen else "⇒"
    stated = f"{attr} {arrow} {_value(value)}"
    stated += f" {dim('(frozen)')}" if frozen \
        else f" {_signed(commit, attr, p, reach)}"
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


#: the graph's swim-lane palette, git's cycle: each chain keeps one color
#: for its whole life on screen.
_LANE_COLORS = ("1;32", "1;33", "1;34", "1;35", "1;36", "1;31")


def _lanes(commits_newest_first):
    """Git's graph, made literal: one colored column per live chain, ``*``
    on the commit's own column, ``|`` through the rest.  Columns follow
    the *parent hashes* -- a slot holds the hash it is waiting to print --
    so chains that fork from one commit really converge: the extra lanes
    bend home with ``/`` on a connector row just above their shared
    parent (``|/``, ``|/ /`` for a wider collapse).  No merges exist in
    the record, so ``\\`` never appears -- there is nothing it would say.

    A chain's *root* -- the commit that brought a new Thing into the
    record -- marks itself ``+`` in the lane's color instead of ``*``:
    lanes are reused once a chain ends, so without the mark two chains
    sharing a column read as one.  The birth of an object should look
    like one.

    Yields ``(pre, star, bar, commit)``: an optional connector row to
    print *before* the commit, the commit row prefix, and the
    continuation prefix for its body lines.
    """
    lanes: list[str | None] = []       # each slot: the hash it waits for
    colors: dict[int, str] = {}
    minted = 0

    def paint(slot, mark):
        return _paint(colors.get(slot, "0"), mark)

    for commit in commits_newest_first:
        matches = [i for i, waiting in enumerate(lanes)
                   if waiting == commit["hash"]]
        if matches:
            column = matches[0]
        else:
            # a branch tip: open a lane in the first free slot
            try:
                column = lanes.index(None)
            except ValueError:
                lanes.append(None)
                column = len(lanes) - 1
            lanes[column] = commit["hash"]
            colors[column] = _LANE_COLORS[minted % len(_LANE_COLORS)]
            minted += 1
        pre = None
        extras = matches[1:]
        if extras:
            # the fork, seen from above: every other lane waiting for this
            # commit bends home -- the slash sits in the separator column,
            # git's way (`|/`, `|/ /`)
            row = [" "] * (2 * len(lanes))
            for i in range(len(lanes)):
                if lanes[i] is not None and i not in extras:
                    row[2 * i] = paint(i, "|")
            for i in extras:
                row[2 * i - 1] = paint(i, "/")
            pre = "".join(row).rstrip()
            for i in extras:
                lanes[i] = None
        mark = "*" if commit["parent"] else paint(column, "+")
        star = " ".join(
            mark if i == column else
            (paint(i, "|") if lanes[i] is not None else " ")
            for i in range(len(lanes))).rstrip()
        lanes[column] = commit["parent"]           # may be None: the root
        bar = " ".join(paint(i, "|") if lanes[i] is not None else " "
                       for i in range(len(lanes))).rstrip()
        while lanes and lanes[-1] is None:
            colors.pop(len(lanes) - 1, None)
            lanes.pop()
        yield pre, star, bar, commit


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
    for pre, star, bar, commit in _lanes(commits):
        if args.graph and pre:
            print(pre)
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
               for belief, attr, value, p, _reads in commit.get("notes", ()))


def _readings(ledger, commit):
    """Per changed cell: what every known proposer says -- ``-`` where one
    is silent.  Opinions pool across the commit's refs (the commit is the
    content; refs are pointers to it), latest per belief.  Corroborations
    recorded by ``thinair evaluate`` live here from then on: the cache,
    made visible."""
    roster = _proposer_roster(ledger)
    if not roster or not commit["changes"]:
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
            # the commit holds -- and the reading in use renders bold
            held = commit["changes"][attr][0]
            print(shade(similarity(latest.value, held), line,
                        bold=belief_id == commit["author"])
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
    attachments = _attachments(ledger, custom)
    # the resolving belief per cell -- its reading is the value the
    # program actually served, so its number renders bold
    resolving = {}
    for attr, cell in (stats or {}).items():
        owner, _p, _frozen = cell
        resolving[attr] = _base_id(ledger, owner["author"], bases)
    footer = {attr: (stats or {}).get(attr) for attr in attrs}
    footer_texts = ["p 1.00 ±0.00" if frozen
                    else _signature_text(owner, attr, p)
                    for attr, cell in footer.items() if cell is not None
                    for owner, p, frozen in (cell,)]
    width = max(8, *(min(len(a), 14) for a in attrs),
                *(len(text) for text in footer_texts)) + 2
    lines = ["matrix:  " + dim("(rows beliefs, columns attributes; shade = "
                               "value overlap, green exact ... red none; "
                               "bold = the resolving reading; ? never "
                               "asked, x unreachable from here)"),
             dim(" " * 46 + "".join(a[:14].rjust(width) for a in attrs))]
    for belief_id, cells_ in rows.items():
        line = f"  {belief_id[:44]:<44}"
        for attr in attrs:
            askable = belief_id in _askable(attachments, attr) \
                or belief_id.startswith("model:")
            got = cells_.get(attr)
            if got is not None:
                # a recorded reading always shows, frozen column or not
                p, overlap = got
                line += shade(overlap, f"{p:.2f}".rjust(width),
                              bold=resolving.get(attr) == belief_id)
                continue
            if active == (belief_id, attr):
                line += yellow("…".rjust(width))
                continue
            if (frozen_by or {}).get(attr) is not None:
                # pinned by fiat: an empty cell is not an invitation
                line += dim("frozen".rjust(width))
                continue
            line += dim(("?" if askable else "x").rjust(width))
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
                # the fiat speaks by the same three channels as every other
                # signature: `p 1.00` colored by whether the declared fact
                # landed where the other beliefs' recorded readings did
                # (dim while nobody else has read the cell), `±` by the
                # p range of the agreeing voices
                view = (owner.get("consensus") or {}).get(attr) or {}
                head = "p 1.00 "
                tail = f"±{view.get('dev') or 0.0:.2f}"
                lead = " " * max(0, width - len(head) - len(tail))
                freezer = (frozen_by or {}).get(attr)
                freezer_base = _base_id(ledger, freezer, bases) \
                    if freezer else None
                estimates = [cells_[attr][1] for base, cells_ in rows.items()
                             if attr in cells_ and base != freezer_base]
                if estimates:
                    head = shade(sum(estimates) / len(estimates), head)
                    spread = view.get("range")
                    tail = dim(tail) if spread is None \
                        else shade(1.0 - spread, tail)
                    line += lead + head + tail
                else:
                    # an unmeasured fiat waits dim whole, like any
                    # unopposed signature
                    line += lead + dim(head + tail)
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
    reach = _reach(ledger)
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
    print(f"    {_message(commit, reach)}")
    print()
    # the whole tree as of this commit, source-style; what this commit
    # moved is highlighted, the rest is context
    tree = _tree_at(commits, commit)
    stats = _cell_stats(commits, commit)
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
    for belief, attr, value, p, reads in commit.get("notes", ()):
        held = commit["changes"].get(attr)
        overlap = ""
        if held is not None and not values_equal(value, held[0]):
            overlap = f", ~{similarity(value, held[0]):.2f}"
        verb = "read" if reads else "judged"
        print(dim(f"note: {belief} {verb} {attr} as {_value(value)} "
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
    attachments = _attachments(ledger, custom)
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
            if not reads and instrument.id not in _askable(attachments, attr):
                continue                            # priority's enum is not
                                                    # customer's business
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
        "## Appendix: built-in beliefs",
        "",
        "Generative: `model(name, think=, temperature=)`, `human(name, "
        "interactive=)`, and any function via `thinair.fn` (code; results "
        "freeze).  Validators (`from thinair.validators import <Name>`; "
        "the declaration options `enum=`, `range=`, ... are shorthand "
        "for these):", ""]
    for name in getattr(V, "__all__", ()):
        obj = getattr(V, name, None)
        if not (_inspect.isclass(obj) and issubclass(obj, Belief)):
            continue
        doc = (obj.__doc__ or "").strip().splitlines()
        first = doc[0].rstrip(".") if doc else ""
        lines.append(f"- `{name}` -- {first}" if first else f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def _client_manual() -> str:
    """The CLI, taught to the agent that will drive it."""
    return '''
---

## Appendix: the thinair client, for agents

You are likely a coding agent in a repository whose `.thinair/` holds a
record.  The `thinair` command is git for that record.  **Always pass
`--ai-readable`**: it states the display's color channels as text --
`agree=` (value agreement among *independent readings*; judges never
count, their concord is built in; `agree=unopposed` means one reading,
which nothing could have contradicted -- no evidence either way),
`asked=N/M` (how much of the cell's panel has spoken), and
`expect-violated` (a declared expectation was missed).  Without it you
are blind to half the output.

### Build -- the deliverable is a running program

Whatever plan you hold, it measures nothing until it executes: build
it as an ordinary thinair program and run it.  Everything needed is in
this output -- no other file is required reading, and none has to
exist.  (The experiment-design protocol, for when a measurement
strategy must be designed from raw data, is Part 2: `ground --full`.)
The whole skeleton:

    from thinair import Thing, model
    from thinair.validators import TokenSubsetBelief

    Thing.__default__ = model("deepseek-v4-flash")   # panels fall back here

    class Ticket(Thing):
        """A support ticket to be understood."""   # prompt material
        __beliefs__ = [TokenSubsetBelief("source_text")]
        source_text: str                           # you supply this
        customer = Thing(str, extracted_from="source_text")
        priority = Thing(str, enum=["low", "normal", "high"])

    t = Ticket(source_text=raw, __entity__="ticket-4417")
    print(+t.priority, ~t.priority)                # value, probability
    t += model("deepseek-v4-pro")                  # a dissimilar second voice

Every read is measured once and lands in `.thinair/opinions.db`;
reruns replay from the record at zero cost.  `__entity__` names the
branch you will see in `thinair log` (omit it and one is generated).
One Thing per document.  Uncertainty enters only at the readings
(Pillar II): pre-passes, aggregation, and every derivable quantity
stay exact Python -- beliefs are consulted only where epistemic
uncertainty genuinely enters.  **A value code already knows is
assigned, never asked**: `t.axis = value` records it frozen, free,
and consults no panel (designed-in ground for `evaluate` lives on a
parallel entity carrying the same axis names, e.g. `gold-ticket-4417`).
The record the run leaves *is* the strategy's evidence; the
commands below are how you read it back.

Env: `THINAIR_MODEL`, `THINAIR_BASE_URL`, `THINAIR_API_KEY`;
`THINAIR_OFFLINE=1` makes any would-be model call raise (wire your
pre-passes under it); `THINAIR_PROGRESS=1` narrates calls on stderr --
reasoning models think for minutes, and that is working, not hanging.

### Inspect

    thinair --ai-readable status                    the store, summarized
    thinair --ai-readable log --all --oneline --graph
    thinair --ai-readable show [rev]                whole tree + the matrix
    thinair --ai-readable blame <entity>            who set each cell, when
    thinair diff A...B                              two trees, cell by cell
    thinair branch [-d name]                        entities are branches

A `rev` is a hash prefix, a branch (entity) name, or `HEAD`.  A believed
cell renders as `(p 0.95 ±0.02)` -- the resolving belief's own p, the
agreeing voices' spread beside it (`±0.00` = unmeasured, not certain:
check `asked=`).  Matrix marks: `?` a consultation `evaluate` would
perform, `x` unreachable on that cell, `frozen` pinned by fiat.

### Measure

    thinair --ai-readable evaluate [rev] [belief|*]

Rebuilds the commit's tree, consults every reconstructible belief
(models by id; validators from their stored configurations), and
records the answers as corroborations -- idempotent per (commit,
belief, cell), so re-running is free.  This is the command that turns
`?` into numbers, `asked=1/3` into `asked=3/3`, and `agree=unopposed`
into an earned `agree=`: model corroborations are the independent
readers agreement is counted over (validator verdicts fill the matrix
but can never earn it).  Frozen cells are skipped by default -- a fact
is not a question -- and `--include-frozen` prices the facts too.
Models need an endpoint: `THINAIR_MODEL`, `THINAIR_BASE_URL`,
`THINAIR_API_KEY`.

### Verify -- the implementation is not done until the record agrees

Debug the application through its record, not print statements: the
store already holds what every belief saw and said.  After each run:

    thinair --ai-readable log --all --oneline    did it record what
                                                 you intended?
    thinair --ai-readable show <entity>          every cell + readings

- Branches are your entities, commits your reads.  A read with no new
  commit replayed from the record (free, by design); an attribute with
  no commit at all was assigned, not believed.  An unexpected
  `[belief]` commit means the panel changed under you.
- Fill the matrix before trusting it: run `evaluate` until `asked=` is
  full.  A cell at `±0.00 asked=1/3` is unmeasured, not settled, and
  `agree=unopposed` is exactly one reading -- nothing could have
  disagreed, so nothing was earned.
- Then read the signatures.  Done means `agree=` near 1.00 across the
  cells that matter -- computed over real independent readers, never
  over judges -- and no `expect-violated` anywhere.  Low `agree=`
  is a *located* disagreement, never a reason to blindly rerun: `show`
  the readings, `blame` the cell, then either improve the prompt
  material (docstrings, `Thing(..., doc=...)`), tighten the contract, or
  report the divergence as a finding -- disagreement between dissimilar
  instruments is information, not failure.

### Extend

Add your own instrument when nothing built-in checks what matters:

    # sentences.py
    from thinair.beliefs import Discriminative

    class EndsWithPeriod(Discriminative):
        """The candidate text finishes its sentence."""

        def judge(self, value, e, attr):
            if not isinstance(value, str):
                return None                    # no opinion off-type
            return 1.0 if value.endswith(".") else (0.2, "unfinished")

    $ thinair belief add sentences.py       # copies into .thinair/beliefs/
    $ thinair belief list                   # and: thinair belief rm <name>

`judge` returns `None` (no opinion), a probability, or `(p, reason)`;
`e` is a sealed snapshot -- read other cells as `e.attr.value`.
Registered classes rebuild from the record's stored configurations, so
`evaluate '*'` consults them beside the built-ins; module-level
*instances* register under durable ids and serve live.  `necessary=True`
at the attachment site gains a veto at read time; registered-only
beliefs corroborate, never gate.

### Declare (in the application's Python, not the CLI)

Docstrings are prompt material: the class docstring is the entity's
purpose, `Thing(..., doc=...)` joins the attribute's description, and a
method's first docstring line rides along with its signature.
Instruments attach declaratively -- `Thing(float,
beliefs=[RangeBelief(0, 10_000)])`: auto-scoped, veto terms kept, described
to the model; `enum=`, `range=`, ... are shorthand for this.
`Thing(str, p=0.9, deviation=0.1)` declares expectations -- stamped
into the record and judged there, never shown to the answering belief,
never a gate.  `eager=True` resolves at construction.  The panel is
part of the history: `t += belief` / `t -= belief` land as `[belief]`
commits.  The theory is above; this output is the whole grounding.
'''


def cmd_ground(_ledger, args):
    """The grounding, dumped as-is -- no meta, no explaining the file to
    its reader.  Default: GROUNDING.md minus the strategy-design-only
    stretches (the mathematical anchors, Part 2 -- the experiment
    protocol -- and the Layer 2 outlook), plus two generated appendices
    -- the built-in belief roster and the client manual for agents --
    sized so a harness shows it inline instead of truncating to a file.
    `--full` restores the cuts.  Nothing else on stdout, so the output
    pipes straight into an agent's context; the first command of an
    agentic session."""
    from importlib.resources import files
    text = files("thinair").joinpath("GROUNDING.md").read_text(
        encoding="utf-8")
    if not getattr(args, "full", False):
        head, cut, rest = text.partition("## Mathematical anchors")
        _skip, mark, tail = rest.partition("# Part 3 ")
        if cut and mark:
            text = head + mark + tail
        head, cut, rest = text.partition("Layer 2 — scoring beliefs")
        _skip, mark, tail = rest.partition("\n---\n")
        if cut and mark:
            text = head + tail.lstrip("\n")
    sys.stdout.write(text)
    sys.stdout.write(_builtin_roster())
    sys.stdout.write(_client_manual())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="thinair", description="inspect a thinair record, git-style")
    parser.add_argument("--store", help="path to opinions.db or a ledger.json")
    parser.add_argument("--ai-readable", action="store_true",
                        help="say the signature's color channels in text "
                             "(agree=, asked=) for readers without a palette")
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
    log.set_defaults(run=cmd_log, paged=True)

    show = sub.add_parser("show", help="one commit in full")
    show.add_argument("commit", nargs="?", default=None,
                      help="a hash prefix, a branch (entity) name, or HEAD "
                           "(the default)")
    show.set_defaults(run=cmd_show, paged=True)

    sub.add_parser("status", help="the store, summarized").set_defaults(
        run=cmd_status)
    branch = sub.add_parser("branch", help="entities and their heads")
    branch.add_argument("-d", "--delete", metavar="name", default=None,
                        help="delete a branch: drop the ref's opinions from "
                             "the store (archives are untouched)")
    branch.set_defaults(run=cmd_branch, paged=True)

    blame = sub.add_parser("blame", help="every cell: who set it, when")
    blame.add_argument("entity")
    blame.set_defaults(run=cmd_blame, paged=True)

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
    diff.set_defaults(run=cmd_diff, paged=True)

    ground = sub.add_parser(
        "ground", help="print the measurement grounding; pipe it to an agent")
    ground.add_argument("--full", action="store_true",
                        help="include the experiment protocol (Part 2), the "
                             "mathematical anchors and the Layer 2 outlook: "
                             "for designing a strategy from raw data")
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
    global _AI_READABLE
    _AI_READABLE = bool(getattr(args, "ai_readable", False))
    ledger = open_store(args.store) if getattr(args, "needs_store", True) \
        else None
    try:
        with _pager(getattr(args, "paged", False)):
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
