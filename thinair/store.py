"""The default ledger, made durable: every opinion lands in ``.thinair/``.

The ledger has been a relational table since day one -- one record shape
that never varies (invariant 1), append-only with no deletion (invariant 2),
durable belief ids (invariant 6).  ``ledger.json`` was a serialization
accident; this module is the same table in SQLite, installed as the
process-wide default ledger so that tracking is on unless switched off:

* ``THINAIR_STORE`` unset or empty -> ``.thinair/opinions.db`` in the
  working directory, created lazily on the first append (a read-only
  process never grows a ``.thinair``).
* ``THINAIR_STORE=off`` (or ``0`` / ``false`` / ``no``) -> the in-memory
  default, exactly as before.  ``disable()`` does the same in code.
* ``THINAIR_STORE=<path>`` -> that file instead.

Explicit ledgers are untouched: ``Thing(__ledger__=...)`` and
``use_ledger(...)`` override the default as they always did, which is also
how the test suite stays hermetic.

**Operational truth vs archived evidence.**  While an app runs, this store
is the source of truth: relaunch the program and every frozen opinion --
assignments, code results, fixtures -- short-circuits straight from the
record at zero calls; a runner that names its entities deterministically and
skips cells already answered (``evaluate.reading``) continues where it left
off, because continuation here is *replay against a warm record*, not
process checkpointing -- the program is not in the database, and which
believed reading "stands" is the resolution policy's call, recomputable from
the record rather than stored beside it.  The committed evidence of an
experiment remains a JSON extract (SPEC.md §13 rule 7): text, diffable,
hash-stampable.  ``dump()`` is inherited, so the export is one call; the
database itself is derived structure -- rebuildable from archives via
``extend(Ledger.load(...))``, deletable without losing anything archived,
and therefore free to evolve its schema.

**Storage slices; it never judges.**  SQL equality is not ``values_equal``
and ``AVG`` knows nothing of licensed statistics -- every judgment about
what rows *mean* stays in ``thinair.evaluate``, which consumes this class
through the plain Ledger interface.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable, Iterator

from .ledger import Ledger, Opinion, set_default_ledger

__all__ = ["SqliteLedger", "DEFAULT_PATH", "install", "disable"]

DEFAULT_PATH = os.path.join(".thinair", "opinions.db")

_OFF = ("off", "0", "false", "no")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS opinion (
    t      REAL NOT NULL,
    belief TEXT NOT NULL,
    entity TEXT NOT NULL,
    attr   TEXT NOT NULL,
    value  TEXT,
    p      REAL NOT NULL,
    frozen INTEGER NOT NULL,
    meta   TEXT
);
CREATE INDEX IF NOT EXISTS opinion_cell   ON opinion(entity, attr);
CREATE INDEX IF NOT EXISTS opinion_belief ON opinion(belief);
CREATE TABLE IF NOT EXISTS belief (
    id          TEXT PRIMARY KEY,
    kind        TEXT,
    necessary   INTEGER,
    veto_line   REAL,
    proposes    INTEGER,
    description TEXT,
    config      TEXT
);
"""


class SqliteLedger(Ledger):
    """A Ledger whose memory is a SQLite file.

    Only the kernel is overridden; ``latest``, ``dump`` and the rest arrive
    from the base class.  The connection opens lazily -- reads against a
    file that does not exist yet answer emptily rather than creating it, so
    merely importing thinair (or only ever reading) leaves no ``.thinair``
    behind.  ``t`` continues from ``MAX(t)`` on reopen: a relaunched process
    appends to the same tape it left.
    """

    def __init__(self, path: str | os.PathLike = DEFAULT_PATH) -> None:
        self.path = os.fspath(path)
        self._conn: sqlite3.Connection | None = None
        self._described: set[str] = set()

    # -- the connection, lazily -------------------------------------------
    def _db(self, create: bool) -> sqlite3.Connection | None:
        if self._conn is None:
            if not create and not os.path.exists(self.path):
                return None
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_SCHEMA)
            try:                       # stores from before the config column
                conn.execute("ALTER TABLE belief ADD COLUMN config TEXT")
            except sqlite3.OperationalError:
                pass
            conn.commit()
            self._conn = conn
        return self._conn

    # -- the kernel: writing ----------------------------------------------
    def next_t(self) -> float:
        db = self._db(create=True)
        got = db.execute("SELECT COALESCE(MAX(t), 0.0) FROM opinion").fetchone()
        return float(got[0]) + 1.0

    def add(self, opinion: Opinion) -> Opinion:
        if not isinstance(opinion, Opinion):
            raise TypeError(
                "the ledger accepts exactly one record type: Opinion (invariant 1)"
            )
        db = self._db(create=True)
        with db:                                   # one append, one transaction
            if not opinion.t:
                from dataclasses import replace
                opinion = replace(opinion, t=self.next_t())
            db.execute(
                "INSERT INTO opinion (t, belief, entity, attr, value, p, frozen, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (float(opinion.t), opinion.belief, opinion.entity, opinion.attr,
                 json.dumps(opinion.value, default=str),
                 float(opinion.p), int(opinion.frozen),
                 json.dumps(opinion.meta, default=str)))
            self._describe(db, opinion.belief)
        return opinion

    def _describe(self, db: sqlite3.Connection, belief_id: str) -> None:
        """Persist the speaking belief's description, once.

        The record can then answer "was this judge necessary, and where was
        its veto line?" without the analysing process reconstructing the
        strategy's classes -- the failure mode behind the original veto bug.
        Descriptions only, never bodies: the database stores what the system
        said and who said it; git stores what the system is.
        """
        if belief_id in self._described:
            return
        self._described.add(belief_id)
        from .beliefs import lookup
        belief = lookup(belief_id)
        if belief is None:
            return                            # unregistered here; a later
                                              # process may describe it
        describe = getattr(belief, "describe", None)
        try:
            description = describe() if callable(describe) else \
                getattr(belief, "short", belief_id)
        except Exception:                     # a describe() that throws never
            description = belief_id           # blocks the record

        config = None
        if hasattr(belief, "_args") or hasattr(belief, "_kwargs"):
            try:
                # The constructor configuration, stored so a later process
                # can rebuild the instrument (values that do not survive
                # JSON are marked, and mark the whole config unrebuildable).
                config = json.dumps(
                    {"args": list(getattr(belief, "_args", ()) or ()),
                     "kwargs": dict(getattr(belief, "_kwargs", {}) or {})},
                    default=lambda o: {"__unjson__": repr(o)})
            except Exception:
                config = None
        db.execute(
            "INSERT OR IGNORE INTO belief "
            "(id, kind, necessary, veto_line, proposes, description, config) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (belief_id, type(belief).__name__,
             int(bool(getattr(belief, "necessary", False))),
             float(getattr(belief, "veto_line", 0.5)),
             int(bool(getattr(belief, "proposes", False))),
             str(description), config))
        if config is not None:                # backfill pre-config rows
            db.execute("UPDATE belief SET config = ? "
                       "WHERE id = ? AND config IS NULL",
                       (config, belief_id))
        # a scoped wrapper's mechanism is its inner belief; describe it too,
        # even if it never speaks under its own id, so the instrument is
        # rebuildable from the record
        inner = getattr(belief, "inner", None)
        if inner is not None and getattr(inner, "id", None):
            self._describe(db, inner.id)

    @staticmethod
    def _belief_dict(row: tuple) -> dict:
        return dict(id=row[0], kind=row[1], necessary=bool(row[2]),
                    veto_line=row[3], proposes=bool(row[4]),
                    description=row[5], config=row[6])

    def belief_row(self, belief_id: str) -> dict | None:
        """The stored description of a belief, or ``None``."""
        db = self._db(create=False)
        if db is None:
            return None
        row = db.execute(
            "SELECT id, kind, necessary, veto_line, proposes, description, "
            "config FROM belief WHERE id = ?", (belief_id,)).fetchone()
        return self._belief_dict(row) if row is not None else None

    def belief_rows(self) -> list[dict]:
        """Every described belief -- speakers and the inner mechanisms of
        scoped wrappers alike."""
        db = self._db(create=False)
        if db is None:
            return []
        return [self._belief_dict(r) for r in db.execute(
            "SELECT id, kind, necessary, veto_line, proposes, description, "
            "config FROM belief ORDER BY rowid")]

    # -- the kernel: reading ----------------------------------------------
    @staticmethod
    def _row(row: tuple) -> Opinion:
        t, belief, entity, attr, value, p, frozen, meta = row
        return Opinion(belief=belief, entity=entity, attr=attr,
                       value=json.loads(value) if value is not None else None,
                       p=p, t=t, frozen=bool(frozen),
                       meta=json.loads(meta) if meta else {})

    def opinions(self, entity: str | None = None, attr: str | None = None,
                 belief: str | None = None, frozen: bool | None = None,
                 ) -> list[Opinion]:
        db = self._db(create=False)
        if db is None:
            return []
        where, args = [], []
        for column, wanted in (("entity", entity), ("attr", attr),
                               ("belief", belief)):
            if wanted is not None:
                where.append(f"{column} = ?")
                args.append(wanted)
        if frozen is not None:
            where.append("frozen = ?")
            args.append(int(frozen))
        sql = "SELECT t, belief, entity, attr, value, p, frozen, meta FROM opinion"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY t, rowid"        # insertion-stable under equal stamps
        return [self._row(r) for r in db.execute(sql, args)]

    def cells(self, entity: str | None = None) -> list[tuple[str, str]]:
        db = self._db(create=False)
        if db is None:
            return []
        sql = ("SELECT entity, attr FROM opinion "
               + ("WHERE entity = ? " if entity is not None else "")
               + "GROUP BY entity, attr ORDER BY MIN(t)")
        args = (entity,) if entity is not None else ()
        return [tuple(r) for r in db.execute(sql, args)]

    def beliefs(self) -> list[str]:
        db = self._db(create=False)
        if db is None:
            return []
        return [r[0] for r in db.execute(
            "SELECT belief FROM opinion GROUP BY belief ORDER BY MIN(t)")]

    def __len__(self) -> int:
        db = self._db(create=False)
        if db is None:
            return 0
        return int(db.execute("SELECT COUNT(*) FROM opinion").fetchone()[0])

    def __iter__(self) -> Iterator[Opinion]:
        return iter(self.opinions())

    def drop_entity(self, entity: str) -> int:
        """Delete a ref: every opinion this entity owns, episode sub-entities
        included -- ``git branch -d``, with the gc built in.

        The one deletion in the package, and it lives here rather than on the
        base :class:`Ledger` deliberately: the operational store is derived
        structure, rebuildable from archives (see the module docstring), so
        dropping a branch from it loses nothing archived.  Commits shared
        with other refs survive -- they are re-derived from those refs'
        opinions.  Returns the number of opinions dropped.
        """
        db = self._db(create=False)
        if db is None:
            return 0
        with db:
            cursor = db.execute(
                "DELETE FROM opinion WHERE entity = ? OR entity LIKE ?",
                (entity, entity + ".%"))
        return cursor.rowcount

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<SqliteLedger {self.path!r}, {len(self)} opinions>"


# --------------------------------------------------------------------------
# the default, and the off switch
# --------------------------------------------------------------------------

def install(path: str | os.PathLike | None = None) -> Ledger:
    """Make the durable store the process default ledger.

    Called once at package import.  ``THINAIR_STORE`` decides: off-words
    leave the in-memory default, a path relocates the file, anything else
    means ``.thinair/opinions.db``.  Returns the ledger now serving as
    default.
    """
    env = (os.environ.get("THINAIR_STORE") or "").strip()
    if path is None:
        if env.lower() in _OFF:
            return disable()
        path = env or DEFAULT_PATH
    ledger = SqliteLedger(path)
    set_default_ledger(ledger)
    return ledger


def disable() -> Ledger:
    """Back to a plain in-memory default ledger; the store file is untouched."""
    ledger = Ledger()
    set_default_ledger(ledger)
    return ledger
