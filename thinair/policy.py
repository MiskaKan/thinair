"""Resolution policies and the route ladder (SPEC.md §5).

A :class:`ResolutionPolicy` **selects** the resolving opinion from a cell's
opinions.  It never blends, averages, or pools -- that is invariant 5, and it
is what keeps ``~t`` an honest statement by one identifiable belief rather
than a number nobody said.  ``Pooled`` and ``MostCredible`` are Layer 2 and
are not here.

:class:`Route` is the other half: which generative belief proposes, and what
happens when it runs out of attempts.  It lives here rather than in
``engine/`` so that ``thing.py`` can import it without importing a transport
(invariant 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .beliefs import generative_members
from .ledger import values_equal

__all__ = [
    "ResolutionPolicy", "Proposed", "Unanimous", "Threshold",
    "Route", "Attempt", "LowConfidence", "Unresolvable", "Disagreement",
    "ROUNDS_PER_ROUTE", "ESCALATED_ROUNDS",
]

#: The budget: 3 rounds, then escalate the route; 2 more, then
#: Unresolvable.  Runtime-owned -- a model can never grant itself
#: another attempt.
ROUNDS_PER_ROUTE = 3
ESCALATED_ROUNDS = 2


# --------------------------------------------------------------------------
# what a failed derivation raises
# --------------------------------------------------------------------------

class LowConfidence(Exception):
    """A resolved opinion fell below an active ``Thing.require`` bar."""

    def __init__(self, cell, p, floor):
        self.cell, self.p, self.floor = cell, p, floor
        entity, attr = cell
        super().__init__(
            f"{entity}.{attr} resolved at p={p:.2f}, below the required {floor:.2f}")


class Disagreement(Exception):
    """``Unanimous`` was active and the generative beliefs did not agree."""

    def __init__(self, cell, opinions):
        self.cell, self.opinions = cell, list(opinions)
        entity, attr = cell
        voices = ", ".join(f"{o.belief} says {o.value!r}" for o in self.opinions)
        super().__init__(f"{entity}.{attr} is disputed: {voices}")


class Unresolvable(Exception):
    """The budget ran out, or there was nobody who could propose.

    Carries the full attempt history, because a failed derivation is exactly
    the moment a reader needs to see every rejected candidate and the reason
    each was rejected.
    """

    def __init__(self, cell, attempts=(), reason=""):
        self.cell = cell
        self.attempts = list(attempts)
        entity, attr = cell
        lines = [f"{entity}.{attr} could not be resolved"]
        if reason:
            lines[0] += f": {reason}"
        for i, attempt in enumerate(self.attempts, start=1):
            lines.append(f"  attempt {i} ({attempt.route}): {attempt}")
        super().__init__("\n".join(lines))


# --------------------------------------------------------------------------
# one round's record
# --------------------------------------------------------------------------

@dataclass
class Attempt:
    """One round: what was proposed, and what the panel made of it.

    A plain record for diagnostics and for the next round's prompt.  It is
    *not* memory -- the ledger is the memory; this is discarded when
    the read finishes.
    """

    round: int
    route: str
    value: Any = None
    p: float = 0.0
    vetoes: list = field(default_factory=list)      # (belief_id, p, reason)
    opinions: list = field(default_factory=list)

    @property
    def vetoed(self) -> bool:
        return bool(self.vetoes)

    def objections(self) -> list[dict]:
        """The shape ``engine/prompts.py`` quotes verbatim."""
        return [{"value": self.value, "belief": belief, "p": p, "reason": reason}
                for belief, p, reason in self.vetoes]

    def __str__(self) -> str:
        if not self.vetoes:
            return f"{self.value!r} at p={self.p:.2f}, unvetoed"
        rejections = "; ".join(
            f"{belief} (p={p:.2f}){': ' + reason if reason else ''}"
            for belief, p, reason in self.vetoes)
        return f"{self.value!r} rejected by {rejections}"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

class ResolutionPolicy:
    """``resolve(cell_opinions) -> (value, p)``.

    Ledger-local and evaluative-math-free: a policy may *choose* among
    opinions and may *refuse*, but it may never invent a value or a
    probability that no belief stated (invariant 5).
    """

    name = "policy"

    def resolve(self, opinions, cell=None, proposer=None):
        raise NotImplementedError

    #: policies that need to see every generative belief's answer make the
    #: runtime consult them all; ``Proposed`` needs only the routed head.
    consults_all_proposers = False

    def __repr__(self) -> str:                      # pragma: no cover - cosmetic
        return f"<{self.name}>"


def _generative_opinions(opinions, panel):
    ids = {b.id for b in generative_members(panel or ())}
    return [o for o in opinions if o.belief in ids]


class Proposed(ResolutionPolicy):
    """The default: the routed proposer's vetted answer, its own p.

    "Its own p" is the whole point.  Three validators corroborating a 0.6 do
    not make it a 0.9; they make it a 0.6 that three validators did not
    object to, which the ledger records and the reader can weigh.
    """

    name = "proposed"

    def resolve(self, opinions, cell=None, proposer=None):
        if not opinions:
            return None
        if proposer is not None:
            for opinion in reversed(opinions):
                if opinion.belief == proposer:
                    return (opinion.value, opinion.p)
        return (opinions[-1].value, opinions[-1].p)


class Unanimous(ResolutionPolicy):
    """Every consulted generative belief agrees, per ``values_equal``.

    Raises :class:`Disagreement` otherwise -- deliberately, because the point
    of asking several beliefs and demanding unanimity is to find out when you
    cannot have it.  Dissimilar beliefs agreeing is the strongest signal Layer
    1 offers; this is how a caller insists on it.
    """

    name = "unanimous"
    consults_all_proposers = True

    def __init__(self, min_p: float = 0.0):
        self.min_p = float(min_p)
        self.name = f"unanimous(min_p={self.min_p:g})"

    def resolve(self, opinions, cell=None, proposer=None):
        if not opinions:
            return None
        voices = {}
        for opinion in opinions:
            voices[opinion.belief] = opinion              # latest per belief
        speakers = list(voices.values())
        first = speakers[0]
        dissent = [o for o in speakers if not values_equal(o.value, first.value)]
        if dissent:
            raise Disagreement(cell or (first.entity, first.attr), speakers)
        weakest = min(speakers, key=lambda o: o.p)
        if weakest.p < self.min_p:
            raise LowConfidence(cell or (first.entity, first.attr),
                                weakest.p, self.min_p)
        return (first.value, weakest.p)


class Threshold(ResolutionPolicy):
    """Below the bar, resolve to ``None`` -- carrying the diagnostic p.

    The quiet counterpart to ``Thing.require``, which raises.  Which one you
    want depends on whether a missing answer is an error or a fact.
    """

    name = "threshold"

    def __init__(self, min_p: float = 0.5):
        self.min_p = float(min_p)
        self.name = f"threshold(min_p={self.min_p:g})"

    def resolve(self, opinions, cell=None, proposer=None):
        chosen = Proposed().resolve(opinions, cell, proposer)
        if chosen is None:
            return None
        value, p = chosen
        return (value, p) if p >= self.min_p else (None, p)


DEFAULT_POLICY = Proposed()


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

class Route:
    """List-order routing with escalation -- the whole v1 router.

    The generative members of ``__beliefs__``, in order: the first proposes,
    the rest are the escalation ladder.  Escalation fires when the veto gate
    exhausts a route's budget, or when a resolved p lands below an active
    ``require``.

    The ladder is *runtime-owned*.  Nothing a model returns can extend it,
    and the budgets below are the only threading in the framework.
    """

    def __init__(self, beliefs: Iterable[Any], budgets=None):
        self.members = generative_members(beliefs)
        self.budgets = list(budgets) if budgets is not None else None
        self.index = 0
        self.rounds_spent = 0
        self.attempts: list[Attempt] = []

    # -- who is speaking ---------------------------------------------------
    @property
    def empty(self) -> bool:
        """No generative member: a deterministic-only Thing."""
        return not self.members

    @property
    def head(self):
        return self.members[self.index] if self.index < len(self.members) else None

    @property
    def name(self) -> str:
        head = self.head
        return getattr(head, "id", "none") if head is not None else "none"

    def budget(self) -> int:
        if self.budgets is not None:
            return self.budgets[min(self.index, len(self.budgets) - 1)]
        return ROUNDS_PER_ROUTE if self.index == 0 else ESCALATED_ROUNDS

    # -- walking the ladder ------------------------------------------------
    def spend(self, attempt: Attempt) -> None:
        self.rounds_spent += 1
        self.attempts.append(attempt)

    @property
    def exhausted(self) -> bool:
        return self.rounds_spent >= self.budget()

    @property
    def can_escalate(self) -> bool:
        return self.index + 1 < len(self.members)

    def escalate(self) -> bool:
        """Walk to the next generative member.  ``False`` when there is none."""
        if not self.can_escalate:
            return False
        self.index += 1
        self.rounds_spent = 0
        return True

    def panel(self, beliefs: Iterable[Any]) -> list:
        """``beliefs`` with the routed head first.

        Routing is expressed purely as list order, so a belief that reads
        ``e.__beliefs__[0]`` stays correct without knowing routing exists.
        """
        head = self.head
        rest = [b for b in beliefs if b is not head]
        return ([head] + rest) if head is not None else rest

    def __repr__(self) -> str:                      # pragma: no cover - cosmetic
        return (f"<route {self.name} "
                f"[{self.index + 1}/{len(self.members)}] "
                f"round {self.rounds_spent}/{self.budget()}>")
