"""The society runtime: a sweep loop over one shared ledger (STRATEGY.md).

Ordinary code, ZERO framework surface: this module adds no Thing method, no
new noun, no channel.  It only decides *whose turn it is* -- the turn itself
is the framework's ordinary episode, `agent.consider(moved_peer,
acting=True)`, and everything the turn may do is what any episode may do.

Wake rules (all three reduce to one check -- has the (agent, peer) pair of
state hashes moved since this agent last considered that peer?):

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
standing public cells, not the tape: a mention that was private, or has
since been overwritten, is imperceptible to the named -- naming, like every
acquaintance, is a property of the present record, and a relationship both
sides have dropped wakes nobody.

Quiescence is physics: an unchanged pair is skipped, an unchanged episode
replays from the record for free, and the loop ends on a full pass that
takes no turns.  Every turn's cause is the recorded movement that fired it;
there is no cache-busting anywhere.  Time, if a cast needs it, enters as
committed ticks: assign ``clock.tick = n`` (code, frozen) and let the wake
rules do the rest.
"""

from __future__ import annotations

from thinair.policy import LowConfidence, Unresolvable
from thinair.thing import public_attrs, snapshot, state_hash


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
        #: a recurring pair is the livelock tell, and it backs off.
        self.pair_states: dict = {}
        #: (agent id, peer id) -> when the agent last considered that peer;
        #: never-considered pairs (fresh introductions) go first, then the
        #: stalest -- so one busy acquaintance cannot starve the rest.
        self.turn_stamp: dict = {}
        self._clock = 0
        self.log: list = []
        # the baseline: every candidate pair's hashes as of now, no turns
        for agent in self.cast:
            for peer_id in self._candidates(agent):
                self.pair_seen[(agent.__entity__, peer_id)] = \
                    self._pair(agent, self.by_id[peer_id])

    # -- what the standing record says about who is connected to whom ------
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

    def _public_refs(self, peer):
        """Entity ids the peer's standing *public* cells carry -- the only
        naming another mind can perceive."""
        allowed = public_attrs(type(peer))
        out = []
        for attr, opinion in snapshot(peer).__attrs__().items():
            if attr not in allowed:
                continue
            for ref in (opinion.meta or {}).get("refs", ()):
                base = str(ref).split("#", 1)[0]
                if base in self.by_id and base != peer.__entity__ \
                        and base not in out:
                    out.append(base)
        return out

    def _holds(self, agent):
        """Entity ids referenced from the agent's own standing cells."""
        out = []
        for opinion in snapshot(agent).__attrs__().values():
            for ref in (opinion.meta or {}).get("refs", ()):
                base = str(ref).split("#", 1)[0]
                if base in self.by_id and base != agent.__entity__ \
                        and base not in out:
                    out.append(base)
        return out

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

    # -- one pass over the cast --------------------------------------------
    def pass_once(self, sweep_number=0):
        turns = 0
        for agent in self.cast:
            if turns >= self.turns_per_sweep:
                break
            me = agent.__entity__
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
