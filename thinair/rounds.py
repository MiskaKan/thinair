"""Rounds: sealed, simultaneous turns over one shared ledger (SPEC.md §15).

Messages make minds address each other; rounds make them fair.  Inside
``with ledger.rounds():`` an undeclared method call on a Thing does not run
-- it *declares* a pending turn.  ``ledger.round()`` evaluates everything
pending against the state sealed at its start (commits are staged, so no
turn sees another's same-round work), then lands one atomic boundary:
staged writes apply, queued tells are recorded on their senders and
delivered as calls the addressees answer next round, and a frozen marker
on the ``__rounds__`` entity stamps the global clock.  It returns whether
anything happened -- ``while ledger.round():`` runs a cascade to
quiescence, and re-seeding an unchanged world costs nothing (every seed
replays from the record).

Outside the scope nothing here exists: calls evaluate immediately, tells
are recorded at commit and wait in the ledger for the next scope to
deliver them.  The ledger stays the only memory: delivery bookkeeping
lives in the round markers, so a society resumes mid-conversation from
its record alone.
"""

from __future__ import annotations

import contextvars

from .ledger import Opinion, values_equal

__all__ = ["rounds", "RoundScope", "current_scope", "staging",
           "record_tells", "CLOCK"]

#: the entity round markers land on -- the society's global clock.
CLOCK = "__rounds__"

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar(
    "thinair_round_scope", default=None)


def rounds(ledger=None):
    """Round-time over one shared ledger -- the module verb (SPEC.md §15).

        with thinair.rounds(ledger) as clock:
            anna.next_step(); dave.check_for_jobs()      # seeds, pending
            while clock.round():                         # steps, to quiescence
                pass

    Inside the scope a call on a Thing declares a sealed pending turn;
    outside, everything is immediate.  Leaving the block always lands a
    final boundary, however the block is left.  ``ledger`` defaults to the
    active default ledger.
    """
    from .ledger import default_ledger

    return RoundScope(ledger if ledger is not None else default_ledger())


def current_scope(ledger):
    """The active RoundScope for this ledger, or None."""
    scope = _ACTIVE.get()
    if scope is not None and scope.ledger is ledger:
        return scope
    return None


def staging(ledger):
    """The active scope iff a turn is being evaluated under it -- the seam
    where an episode's commit defers to the boundary."""
    scope = current_scope(ledger)
    if scope is not None and scope._evaluating:
        return scope
    return None


def record_tells(ledger, sender, tells, belief, p):
    """Land queued messages as opinions on the *sender's* record.

    A tell is speech: it lives on the speaker's branch (``__tell__:`` cells
    are framework records, never standing state), carries the addressee in
    ``meta`` and every named entity in ``refs``.  Delivery is separate --
    a boundary's job, marked on the round clock.
    """
    from .thing import record_refs

    out = []
    for t in tells:
        refs = [t["to"]]
        for r in record_refs(t.get("args"), ledger):
            if r not in refs:
                refs.append(r)
        out.append(ledger.add(Opinion(
            belief=belief, entity=sender, attr=f"__tell__:{len(ledger)}",
            value={"to": t["to"], "verb": t["verb"],
                   "args": list(t.get("args") or [])},
            p=p, frozen=False,
            meta={"tell": {"to": t["to"], "verb": t["verb"]}, "refs": refs})))
    return out


class PendingTurn:
    """A turn declared inside the scope: input sealed, evaluation deferred.

    Forcing its value (``+turn`` / ``~turn``) evaluates just this turn
    early, still against the seal; otherwise ``ledger.round()`` runs it.
    A delivery turn carries its ``mail`` -- the tell opinions it answers --
    stamped delivered only once the turn has actually been taken.
    """

    __slots__ = ("scope", "thing", "name", "args", "kwargs", "result", "done",
                 "mail", "error")

    def __init__(self, scope, thing, name, args=(), kwargs=None):
        self.scope, self.thing, self.name = scope, thing, name
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.result = None
        self.done = False
        self.mail = ()
        self.error = None

    def _force(self):
        if not self.done:
            self.scope._evaluate(self)
        if self.error is not None:
            # forcing asks for the value: a failed turn answers exactly as
            # the immediate path would -- by raising, never by a silent None.
            raise self.error
        return self.result

    def __pos__(self):
        result = self._force()
        return None if result is None else +result

    def __invert__(self):
        result = self._force()
        return 0.0 if result is None else ~result

    def __repr__(self):                                # pragma: no cover
        state = "done" if self.done else "pending"
        return f"<turn {self.thing.__entity__}.{self.name}(...) {state}>"


class RoundScope:
    """``with ledger.rounds():`` -- round-time, scoped and exception-safe.

    One scope may be active at a time; leaving it always lands a final
    boundary for anything staged (never evaluating what never ran), so the
    ledger is stateless between statements outside the block.
    """

    def __init__(self, ledger):
        self.ledger = ledger
        self.number = 0            # the global clock, resumed from the record
        self.failures: list = []
        self._declared: list = []  # seeds: declared, not yet evaluated
        self._queue: list = []     # reaction turns for the coming round
        self._mailbag: list = []   # recorded tell opinions awaiting delivery
        self._dead: list = []      # undeliverable or parked this scope
        self._staged_commits: list = []
        self._staged_tells: list = []
        self._turn_frozen: list = []    # the running turn's own assignments
        self._staged_frozen: list = []  # finished turns' assignments, to land
        self._heard: list = []     # mail whose delivery turn has been taken
        self._backoff: list = []   # livelock tells parked this round
        #: (sender, addressee) -> state-hash pairs already delivered on; a
        #: recurring pair is the livelock tell, and delivery backs off.
        self.pair_states: dict = {}
        self._evaluating = False
        self._live = 0
        self._token = None

    # -- scope --------------------------------------------------------------
    def __enter__(self):
        if _ACTIVE.get() is not None:
            raise RuntimeError("a rounds() scope is already active; "
                               "one society clock at a time")
        self._token = _ACTIVE.set(self)
        self._scan()
        return self

    def __exit__(self, exc_type, exc, tb):
        _ACTIVE.reset(self._token)
        # land anything staged by early-forced turns after the last round();
        # a queued delivery turn that never ran drops here *unstamped*, so
        # its mail stays in the ledger for the next scope's scan -- leaving
        # early never loses a message.
        applied, ops = self._apply_staged()
        delivered, self._heard = [[o.entity, o.attr] for o in self._heard], []
        if applied or ops or delivered or self.failures:
            self._mark(turns=0, applied=applied, delivered=delivered,
                       closing=True)
        return False

    def _scan(self):
        """Resume from the record: undelivered mail and the clock reading."""
        delivered = set()
        tells = []
        for op in self.ledger:
            if op.entity == CLOCK and isinstance(op.value, dict):
                self.number = max(self.number,
                                  int((op.meta or {}).get("round", 0)))
                for eid, attr in op.value.get("delivered", ()):
                    delivered.add((eid, attr))
            elif (op.meta or {}).get("tell"):
                tells.append(op)
        self._mailbag = [op for op in tells
                         if (op.entity, op.attr) not in delivered]

    # -- declaration and evaluation ----------------------------------------
    def declare(self, thing, name, args, kwargs):
        turn = PendingTurn(self, thing, name, args, kwargs)
        self._declared.append(turn)
        return turn

    def stage(self, thing, validated, tells, belief, p):
        """An episode's commit, deferred to the boundary (called from
        ``episode.run`` through :func:`staging`)."""
        self._staged_commits.append((thing, list(validated)))
        if tells:
            self._staged_tells.append((thing.__entity__, list(tells),
                                       belief, p))

    def stage_frozen(self, thing, opinion):
        """An assignment made during a turn -- a real method's side effect
        included (called from ``Thing.__setattr__`` through
        :func:`staging`).  Held turn-local until the turn finishes, then
        landed at the boundary with everything else."""
        prior = self.turn_frozen(opinion.entity, opinion.attr)
        if prior is not None and prior.belief == opinion.belief \
                and values_equal(prior.value, opinion.value):
            return
        self._turn_frozen.append((thing, opinion))

    def turn_frozen(self, entity, attr):
        """The running turn's own staged assignment for this cell, if any --
        visible to the turn that made it, invisible to its round-mates."""
        for _thing, opinion in reversed(self._turn_frozen):
            if opinion.entity == entity and opinion.attr == attr:
                return opinion
        return None

    def _evaluate(self, turn):
        from .episode import run
        from .policy import LowConfidence, Unresolvable

        before = len(self.ledger)
        was_evaluating = self._evaluating
        self._evaluating = True
        try:
            turn.result = run(turn.thing, turn.name, turn.args, turn.kwargs)
            if len(self.ledger) > before:      # a replay appends nothing
                self._live += 1
        except (Unresolvable, LowConfidence) as exc:
            turn.error = exc
            self.failures.append(
                {"turn": f"{turn.thing.__entity__}.{turn.name}",
                 "why": str(exc).splitlines()[0][:200]})
        finally:
            # restore, never clear: a turn forced from inside another turn
            # must not break the outer turn's staging seam.
            self._evaluating = was_evaluating
            if not was_evaluating:
                self._staged_frozen += self._turn_frozen
                self._turn_frozen = []
            turn.done = True
            if turn.mail:
                self._heard += list(turn.mail)

    # -- the boundary -------------------------------------------------------
    def _apply_staged(self):
        from .thing import commit

        applied = 0
        # assignments first: code authority lands before opinions (§6)
        for thing, opinion in self._staged_frozen:
            latest = self.ledger.latest(opinion.entity, opinion.attr,
                                        frozen=True)
            if latest is not None and latest.belief == opinion.belief \
                    and values_equal(latest.value, opinion.value):
                continue
            self.ledger.add(opinion)
            thing.__root__.__resolved__.pop(
                (opinion.entity, opinion.attr), None)
            thing.__root__.__coerced__.clear()
            applied += 1
        self._staged_frozen = []
        for thing, validated in self._staged_commits:
            commit(thing, validated)
            applied += len(validated)
        ops = []
        for sender, tells, belief, p in self._staged_tells:
            ops += record_tells(self.ledger, sender, tells, belief, p)
        self._staged_commits, self._staged_tells = [], []
        self._mailbag += ops
        return applied, ops

    def _deliver(self):
        """Group the mailbag by addressee into next round's reaction turns.

        One turn per mind per round: a single message arrives as an
        addressed turn (``verb(*args, sender=...)``, answered by the
        addressee's own episode -- never dispatched, SPEC.md §15); several
        arrive together as one ``receive([...])`` turn, so simultaneous
        voices are compared, never raced.  An addressee with no live handle
        on this ledger keeps its mail in the record for a later scope.  A
        recurring (addressee, sender-boundary) state-hash pair is the
        livelock tell: that mail is parked on the record instead of queued,
        and the boundary notes the backoff -- fresh words over unmoving
        state must quiesce, not spend forever.
        """
        from .thing import _LIVE, state_hash

        per: dict = {}
        undeliverable = []
        parked = []
        fresh_pairs = []
        for op in self._mailbag:
            to = str(op.value["to"]).split("#", 1)[0]
            target = _LIVE.get(to)
            if target is None or target.__ledger__ is not self.ledger:
                undeliverable.append(op)
                continue
            sender = str(op.entity).split("#", 1)[0]
            speaker = _LIVE.get(sender)
            # perception-aligned, like the sweep's wake check: the addressee's
            # own half is its full state, the sender's half its boundary view.
            half = (state_hash(speaker, boundary=True)
                    if speaker is not None
                    and speaker.__ledger__ is self.ledger else sender)
            pair = (state_hash(target), half)
            seen = self.pair_states.setdefault((sender, to), set())
            if pair in seen:
                parked.append(op)
                self._backoff.append([sender, to])
                continue
            fresh_pairs.append(((sender, to), pair))
            per.setdefault(target.__entity__, (target, []))[1].append(op)
        for key, pair in fresh_pairs:
            self.pair_states[key].add(pair)
        self._mailbag = []
        self._dead += undeliverable + parked
        for _eid, (target, ops) in per.items():
            messages = [{"sender": op.entity, "verb": op.value["verb"],
                         "args": op.value["args"]} for op in ops]
            if len(messages) == 1:
                m = messages[0]
                turn = PendingTurn(self, target, m["verb"], tuple(m["args"]),
                                   {"sender": m["sender"]})
            else:
                turn = PendingTurn(self, target, "receive", (messages,), {})
            turn.mail = list(ops)
            self._queue.append(turn)
        return len(per)

    def _mark(self, turns, applied, delivered, closing=False):
        self.number += 1
        value = {"turns": turns, "live": self._live, "writes": applied,
                 "delivered": delivered, "failures": list(self.failures)}
        if self._backoff:
            value["backoff"] = list(self._backoff)
        if closing:
            value["closing"] = True
        self.ledger.add(Opinion(
            belief="code:rounds", entity=CLOCK, attr=f"round#{self.number}",
            value=value, p=1.0, frozen=True, meta={"round": self.number}))
        self.failures = []
        self._backoff = []

    def round(self):
        """Advance the clock one round: evaluate everything pending against
        the seal, then land the boundary.  Returns whether anything
        happened -- ``while clock.round():`` runs a cascade to quiescence.
        A round whose only events are failures still lands a marker carrying
        them: the ledger, not the exception, is the story."""
        pending = self._queue + [t for t in self._declared if not t.done]
        self._queue, self._declared = [], []
        self.failures = []
        self._live = 0
        for turn in pending:
            if not turn.done:
                self._evaluate(turn)
        applied, _ops = self._apply_staged()
        queued = self._deliver()
        # a message is stamped delivered only once its turn has been taken,
        # so mail a scope queued but never answered stays on the record.
        delivered, self._heard = [[o.entity, o.attr] for o in self._heard], []
        happened = bool(self._live or applied or queued or delivered
                        or self.failures)
        if happened:
            self._mark(len(pending), applied, delivered)
        return happened
