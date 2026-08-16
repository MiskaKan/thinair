"""The society runtime: a sweep loop over one shared ledger (STRATEGY.md).

Ordinary code, ZERO framework surface: this module adds no Thing method, no
new noun, no channel.  It only decides *whose turn it is* -- the turn itself
is the framework's ordinary episode, `agent.consider(moved_peer,
acting=True)`, and everything the turn may do is what any episode may do.

Wake rules.  Mail first: an undelivered tell addressed to a cast member
arrives as a turn the addressee answers -- one inbox per mind per pass,
marked delivered on the shared clock entity (SPEC.md §15), so a rounds()
scope and a sweep never deliver the same message twice.  Then movement
(all three reduce to one check -- has the (agent, peer) pair of state
hashes moved since this agent last considered that peer?):

1. you moved                       -> the pair (you, you)
2. something you hold moved        -> pairs (you, each ref in your cells)
3. something naming you moved      -> pairs (you, each entity whose
                                       *standing public* record carries
                                       your id in ``meta.refs``)

The pair is perception-aligned (SPEC.md §4): your own half is your full
state hash -- self-knowledge is whole -- but the peer's half is its
*boundary* hash, public cells only, because movement you cannot perceive
must not wake you.  This makes the wake check exactly dual to the episode
repetition key (§9): a turn fires precisely when ``consider(peer)`` would
not replay from the record.  For the same reason rule 3 reads the peer's
standing public cells, not raw mentions: a mention that was private, or has
since been overwritten, is imperceptible to the named -- naming, like every
acquaintance, is a property of the present record, and a relationship both
sides have dropped wakes nobody.  Standing cells are *folded from the tape*
(frozen assignments, plus each successful episode's committed changes), so
the graph survives a restart: the ledger, never process memory, is the
society's story.

Quiescence is physics: an unchanged pair is skipped, an unchanged episode
replays from the record for free, a recurring state-hash pair -- for a wake
and for a delivery alike -- is the livelock tell and backs off, and the
loop ends on a full pass that takes no turns.  Every turn's cause is the
recorded movement (or message) that fired it; there is no cache-busting
anywhere.  Time, if a cast needs it, enters as committed ticks: assign
``clock.tick = n`` (code, frozen) and let the wake rules do the rest.
"""

from __future__ import annotations

from thinair.episode import run
from thinair.ledger import Opinion, normal_form
from thinair.policy import LowConfidence, Unresolvable
from thinair.rounds import CLOCK
from thinair.thing import public_attrs, state_hash


class Sweep:
    """The loop.  Construct it on the cast *before* injecting the first
    fact: construction records the baseline hashes, so the first sweep
    wakes exactly whoever the injection moved."""

    def __init__(self, cast, *, verb="consider", turns_per_sweep=None):
        self.cast = list(cast)
        self.by_id = {t.__entity__: t for t in self.cast}
        if len(self.by_id) != len(self.cast):
            raise ValueError("cast entity ids must be unique")
        self.ledger = self.cast[0].__ledger__
        for member in self.cast:
            if member.__ledger__ is not self.ledger:
                raise ValueError("a society shares one ledger")
        self.verb = verb
        self.turns_per_sweep = turns_per_sweep or len(self.cast)
        #: (agent id, peer id) -> (agent hash, peer hash) after the last turn
        self.pair_seen: dict = {}
        #: (agent id, peer id) -> set of state-hash pairs already acted on;
        #: a recurring pair is the livelock tell, and it backs off.  Shared
        #: by the wake check and mail delivery: same pair, same repetition.
        self.pair_states: dict = {}
        #: (agent id, peer id) -> when the agent last considered that peer;
        #: never-considered pairs (fresh introductions) go first, then the
        #: stalest -- so one busy acquaintance cannot starve the rest.
        self.turn_stamp: dict = {}
        self._clock = 0
        self.log: list = []
        #: the tape fold: entity id -> {attr: (standing value, refs)} --
        #: frozen assignments and committed changesets, replayed from the
        #: record so a restarted society keeps its acquaintances.
        self._cells: dict = {}
        #: (entity, attr, value key) -> refs of the latest such proposal;
        #: a committed change's refs live on its proposal opinion.
        self._proposed: dict = {}
        self._delivered: set = set()
        self.round_number = 0
        self.cursor = 0                       # how much of the tape is folded
        self._ingest()
        # the baseline: every candidate pair's hashes as of now, no turns
        for agent in self.cast:
            for peer_id in self._candidates(agent):
                self.pair_seen[(agent.__entity__, peer_id)] = \
                    self._pair(agent, self.by_id[peer_id])

    # -- what the record says about who is connected to whom ----------------
    def _ingest(self):
        """Fold the tape's new opinions into standing cells and delivery
        bookkeeping.  Restart-proof by construction: the fold's input is the
        ledger alone."""
        tape = list(self.ledger)
        for op in tape[self.cursor:]:
            meta = op.meta or {}
            if op.entity == CLOCK and isinstance(op.value, dict):
                self.round_number = max(self.round_number,
                                        int(meta.get("round", 0)))
                for eid, attr in op.value.get("delivered", ()):
                    self._delivered.add((eid, attr))
                continue
            if meta.get("tell"):
                continue          # mail is delivery's business, not naming
            base = str(op.entity).split(".", 1)[0].split("#", 1)[0]
            if base not in self.by_id:
                continue
            changes = meta.get("changes")
            if op.attr == "result" and isinstance(changes, dict):
                # a successful episode: its committed changes stand
                cells = self._cells.setdefault(base, {})
                for attr, value in changes.items():
                    key = (base, attr, repr(normal_form(value)))
                    cells[attr] = (value, self._proposed.get(key, ()))
                continue
            if op.entity != base or op.attr.startswith("__"):
                continue
            refs = tuple(meta.get("refs", ()))
            if op.frozen:
                self._cells.setdefault(base, {})[op.attr] = (op.value, refs)
            else:
                # a proposal: not standing by itself, but if a result commits
                # this value, these are its refs.  Every belief that spoke on
                # the candidate left an opinion; union, so a judge's bare
                # record cannot erase the proposer's stamp.
                key = (base, op.attr, repr(normal_form(op.value)))
                self._proposed[key] = tuple(dict.fromkeys(
                    self._proposed.get(key, ()) + refs))
        self.cursor = len(tape)

    def _pair(self, agent, peer):
        """The one check's material: my whole state, your perceivable one.

        Self-knowledge is full; perception of a peer is its boundary view
        (§4).  Hashing a peer's private cells into the pair would wake the
        agent on movement its episode could never see -- and re-run a
        byte-identical prompt.
        """
        mine = state_hash(agent)
        if peer is agent:
            return (mine, mine)
        return (mine, state_hash(peer, boundary=True))

    def _refs_of(self, entity_id, attrs=None):
        out = []
        for attr, (_value, refs) in self._cells.get(entity_id, {}).items():
            if attrs is not None and attr not in attrs:
                continue
            for ref in refs:
                base = str(ref).split("#", 1)[0]
                if base in self.by_id and base != entity_id \
                        and base not in out:
                    out.append(base)
        return out

    def _public_refs(self, peer):
        """Entity ids the peer's standing *public* cells carry -- the only
        naming another mind can perceive."""
        return self._refs_of(peer.__entity__, public_attrs(type(peer)))

    def _holds(self, agent):
        """Entity ids referenced from the agent's own standing cells --
        self-knowledge is whole, so private cells count here."""
        return self._refs_of(agent.__entity__)

    def _candidates(self, agent):
        """The peers this agent could be woken by: self, whoever publicly
        names it now, whatever it holds -- never-considered first, then
        stalest."""
        me = agent.__entity__
        out = [me]
        namers = [p.__entity__ for p in self.cast
                  if p is not agent and me in self._public_refs(p)]
        for peer_id in namers + self._holds(agent):
            if peer_id not in out:
                out.append(peer_id)
        out.sort(key=lambda pid: self.turn_stamp.get((me, pid), -1))
        return out

    # -- mail: an undelivered tell is a turn the addressee answers ----------
    def _deliver_mail(self, sweep_number=0):
        """Deliver undelivered tells to cast members: one inbox per
        addressee, the turn answered by the addressee's own episode --
        speech heard, never code invoked (SPEC.md §15).  Delivery is marked
        on the shared clock entity; a recurring state-hash pair is parked
        instead, so fresh words over unmoving state cannot spend forever."""
        per: dict = {}
        for op in self.ledger:
            if not (op.meta or {}).get("tell"):
                continue
            if (op.entity, op.attr) in self._delivered:
                continue
            to = str(op.value["to"]).split("#", 1)[0]
            if to in self.by_id:
                per.setdefault(to, []).append(op)
        delivered = []
        turns = 0
        for to, ops in per.items():
            target = self.by_id[to]
            fresh = []
            for op in ops:
                sender = str(op.entity).split("#", 1)[0]
                speaker = self.by_id.get(sender)
                half = (state_hash(speaker, boundary=True)
                        if speaker is not None else sender)
                pair = (state_hash(target), half)
                seen = self.pair_states.setdefault((to, sender), set())
                if pair in seen:
                    self.log.append({"sweep": sweep_number, "agent": to,
                                     "peer": sender, "outcome": "backoff"})
                    continue
                fresh.append((op, (to, sender), pair))
            if not fresh:
                continue
            for _op, key, pair in fresh:
                self.pair_states[key].add(pair)
            messages = [{"sender": str(op.entity).split("#", 1)[0],
                         "verb": op.value["verb"], "args": op.value["args"]}
                        for op, _key, _pair in fresh]
            entry = {"sweep": sweep_number, "agent": to,
                     "cause": "mail from " + ", ".join(
                         m["sender"] for m in messages)}
            try:
                if len(messages) == 1:
                    m = messages[0]
                    turn = run(target, m["verb"], tuple(m["args"]),
                               {"sender": m["sender"]})
                else:
                    turn = run(target, "receive", (messages,), {})
                entry.update(outcome="turn", value=+turn, p=round(~turn, 4))
            except (Unresolvable, LowConfidence) as exc:
                entry.update(outcome="unresolvable",
                             why=str(exc).splitlines()[0][:200])
            self.log.append(entry)
            turns += 1
            for op, _key, _pair in fresh:
                self._delivered.add((op.entity, op.attr))
                delivered.append([op.entity, op.attr])
        if delivered:
            self.round_number += 1
            self.ledger.add(Opinion(
                belief="code:sweep", entity=CLOCK,
                attr=f"round#{self.round_number}",
                value={"delivered": delivered}, p=1.0, frozen=True,
                meta={"round": self.round_number}))
        return turns

    # -- one pass over the cast --------------------------------------------
    def pass_once(self, sweep_number=0):
        self._ingest()
        turns = self._deliver_mail(sweep_number)
        for agent in self.cast:
            if turns >= self.turns_per_sweep:
                break
            self._ingest()      # incremental: an earlier turn's commit this
            me = agent.__entity__       # very pass is already a wake cause
            for peer_id in self._candidates(agent):
                peer = self.by_id[peer_id]
                key = (me, peer_id)
                pair = self._pair(agent, peer)
                if self.pair_seen.get(key) == pair:
                    continue                  # nothing moved for this pair
                states = self.pair_states.setdefault(key, set())
                if pair in states:
                    # a state-hash pair recurring after movement is a
                    # livelock tell: back off -- mark it seen and move on.
                    self.pair_seen[key] = pair
                    self.log.append({"sweep": sweep_number, "agent": me,
                                     "peer": peer_id, "outcome": "backoff"})
                    continue
                states.add(pair)
                entry = {"sweep": sweep_number, "agent": me, "peer": peer_id,
                         "cause": ("self moved" if peer_id == me
                                   else f"{peer_id} moved")}
                try:
                    turn = getattr(agent, self.verb)(peer, acting=True)
                    entry.update(outcome="turn", value=+turn,
                                 p=round(~turn, 4))
                except (Unresolvable, LowConfidence) as exc:
                    entry.update(outcome="unresolvable",
                                 why=str(exc).splitlines()[0][:200])
                self.log.append(entry)
                turns += 1
                self._clock += 1
                self.turn_stamp[key] = self._clock
                self.pair_seen[key] = self._pair(agent, peer)
                break                         # one turn per agent per sweep
        return turns

    def run(self, max_sweeps=20):
        """Sweep until a full quiet pass.  Returns the number of sweeps."""
        for sweep_number in range(1, max_sweeps + 1):
            if self.pass_once(sweep_number) == 0:
                self.log.append({"sweep": sweep_number, "outcome": "quiescent"})
                return sweep_number
        self.log.append({"sweep": max_sweeps, "outcome": "budget exhausted"})
        return max_sweeps
