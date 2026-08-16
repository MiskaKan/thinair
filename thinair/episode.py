"""Imagined method calls: transactional changesets (SPEC.md §9).

An episode is **a pure function from a state snapshot to
``(changeset, return_value, p)``**.  There is no narrative memory: state --
frozen attributes plus currently resolved cells -- is the whole story, and
the ledger is the only memory.

The engine conversation that produces the changeset is ``ModelBelief``'s
private implementation.  Everything in this module is model-free: it
builds the snapshot, hands it to the panel, validates whatever comes back
against the per-attribute pipeline, and commits atomically or not at all.
"""

from __future__ import annotations

import json
from typing import Any

from .beliefs import MemoBelief, as_judgment
from .ledger import Opinion, arguments_id
from .policy import Attempt, Route, Unresolvable
from .thing import (Cell, Thing, _build_snapshot, _plain, boundary_snapshot,
                    commit, propose, record_refs, references, review,
                    state_hash)

__all__ = ["Episode", "run", "call_expression", "ACTION_BUDGET", "CORRECTIONS",
           "MAX_DEPTH", "MAX_OBSERVATION"]

#: Runtime-owned, like every budget in this framework.
ACTION_BUDGET = 8
CORRECTIONS = 3
MAX_DEPTH = 1
MAX_OBSERVATION = 2000

#: how deep we currently are.  An imagined method may not call another
#: imagined method; the refusal carries the reason.
_depth = 0


class Refused(Exception):
    """An action the grammar does not permit.  Costs a correction, not a run."""


def call_expression(name: str, args=(), kwargs=None) -> str:
    """``summarize(1, style='terse')`` -- the call as a reader would write it.

    ``returns=`` never reaches here: it governs acceptance, not identity.
    """
    return f"{name}({arguments_id(tuple(args), dict(kwargs or {}))})"


class Episode:
    """The capability an episode's snapshot carries, and nothing else.

    An ordinary attribute snapshot has none of this -- ``__episode__`` is
    ``None`` there -- so the sealed-snapshot guarantee holds everywhere
    except the one sanctioned place an episode may reach live state, and only
    through the two actions the grammar defines.
    """

    def __init__(self, thing, name, args=(), kwargs=None, *, returns=None,
                 action_budget=ACTION_BUDGET, functions=None, acting=False):
        self.thing = thing
        self.functions = dict(functions or {})
        self.name = name
        #: an open agentic turn -- "you are this entity; act" -- framed by its
        #: own template version.  Same grammar, same budgets, same panel.
        self.acting = bool(acting)
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.returns = returns
        self.attr = name
        self.action_budget = action_budget
        self.expression = call_expression(name, self.args, self.kwargs)
        self.entity = f"{thing.__entity__}.{self.expression}"
        self.state = self._state()
        self.cell = f"{self.entity}#{self.state}"
        #: outgoing messages queued this attempt (SPEC.md §15): committed
        #: with the changeset, atomically -- a vetoed turn sends nothing.
        self.tells: list = []

    def _state(self) -> str:
        """The state this episode is a pure function of -- all of it, and
        nothing finer.

        A Thing-valued argument's snapshot is part of what the episode saw,
        so its state belongs in the repetition key: an argument that changed
        must invalidate the memo exactly like an interleaved assignment on
        the host does.  But what the episode saw of an argument is its
        *boundary* view (§4): identity, purpose, public cells.  Keying on
        the argument's full state would re-run the episode on private
        movement its prompt cannot carry -- a spurious miss on a pure
        function, byte-identical prompt and all -- so the key hashes exactly
        the perceivable: ``state_hash(argument, boundary=True)``.  The
        expression alone cannot carry this -- it renders a Thing by entity,
        which is stable across state changes on purpose.  With no Thing
        arguments the key is the host's hash alone, unchanged.
        """
        material = [state_hash(self.thing)]
        for argument in self.args:
            if isinstance(argument, Thing):
                material.append(state_hash(argument, boundary=True))
        for _name, argument in sorted(self.kwargs.items()):
            if isinstance(argument, Thing):
                material.append(state_hash(argument, boundary=True))
        #: the host's own half, so §12's history can re-derive and check the
        #: parent tree even when Thing arguments widen the repetition key
        self.host_state = material[0]
        if len(material) == 1:
            return material[0]
        import hashlib
        return hashlib.sha1("+".join(material).encode("utf-8")).hexdigest()[:12]

    # -- the two actions that touch live state ---------------
    def get(self, attr):
        """``get <attr>`` -- through the read pipeline, truncated if huge."""
        if not isinstance(attr, str) or not attr:
            return "refused", "get needs an attribute name"
        try:
            cell = getattr(self.thing, attr)
            value = +cell
        except Unresolvable as exc:
            return "unresolved", str(exc).splitlines()[0]
        except AttributeError:
            return "refused", f"there is no attribute {attr!r}"
        rendered = _short(value)
        return f"{attr}", {"value": rendered, "p": round(~cell, 4)}

    def tell(self, entity, verb, args=()):
        """``tell <entity>.<verb>(args)`` -- queue an addressed message.

        A tell is speech, not action: it lands on the *sender's* record and
        is delivered as a call the addressee answers on its own later turn
        (SPEC.md §15).  Queued here; recorded only if the turn commits.
        """
        import re

        if not isinstance(entity, str) or not entity:
            return "refused", "tell needs an entity id"
        if entity == self.thing.__entity__:
            return "refused", ("that is yourself; write your own attributes "
                               "instead of telling yourself things")
        if not isinstance(verb, str) or not re.match(r"^[A-Za-z_]\w*$", verb):
            return "refused", "tell needs a plain verb name (an identifier)"
        try:
            json.dumps(args)
        except (TypeError, ValueError):
            return "refused", "tell args must be plain JSON values"
        self.tells.append({"to": entity, "verb": verb,
                           "args": list(args or [])})
        return "queued", {"to": entity, "verb": verb,
                          "delivery": "on the addressee's own next turn; "
                                      "nothing comes back now"}

    def call(self, method, args=(), kwargs=None):
        """``call <real_method>(args)`` -- actual code; its writes freeze."""
        if not isinstance(method, str) or not method:
            return "refused", "call needs a method name"
        from .fn import Fn

        target = getattr(type(self.thing), method, None)
        if target is None:
            target = self.functions.get(method)
        if target is None or not callable(target):
            return "refused", (
                f"{method!r} is not a real method of "
                f"{type(self.thing).__name__} and is not a function you were "
                f"offered; you may not call an imagined one from inside an "
                f"episode")
        try:
            # A ``@fn`` is a call-cell, not a bound method: it takes its own
            # arguments and hands back a child Thing.  Resolving one
            # here is the same consult the runtime would do anywhere else.
            if isinstance(target, Fn):
                result = +target(*(args or []), **(kwargs or {}))
            else:
                result = target(self.thing, *(args or []), **(kwargs or {}))
        except Exception as exc:
            return "raised", f"{type(exc).__name__}: {exc}"
        return f"{method}(...)", _short(result)


def _short(value):
    """Oversized values are truncated with an explicit marker."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:                                  # pragma: no cover
        text = repr(value)
    if len(text) > MAX_OBSERVATION:
        return text[:MAX_OBSERVATION] + " …[truncated]"
    return value if len(text) <= MAX_OBSERVATION else text


# --------------------------------------------------------------------------
# the runtime
# --------------------------------------------------------------------------

def run(thing, name, args=(), kwargs=None):
    """Calling an undefined method: work out what it would do."""
    global _depth

    kwargs = dict(kwargs or {})
    returns_template = kwargs.pop("returns", None)     # reserved, and stripped
    acting = bool(kwargs.pop("acting", False))         # reserved, and stripped:
                                                       # framing, not identity

    if _depth >= MAX_DEPTH:
        raise Unresolvable(
            (thing.__entity__, name),
            reason=f"an imagined method may not call another (depth > {MAX_DEPTH})")

    from .thing import Contract

    returns = Contract(returns_template) if returns_template is not None else None
    episode = Episode(thing, name, args, kwargs, returns=returns, acting=acting)
    ledger = thing.__ledger__

    # 6. repetition: same entity, same frozen-state hash, same expression is
    #    a ledger lookup -- byte-identical, zero model calls.
    remembered = ledger.latest(episode.cell, "result")
    if remembered is not None:
        return _result(thing, episode, remembered)

    panel = list(thing.__beliefs__)
    route = Route(panel)
    if route.empty:
        raise Unresolvable((episode.cell, "result"),
                           reason="no generative belief can imagine a method call")

    arguments = _argument_snapshots(episode)
    corrections = CORRECTIONS
    objections: list[dict] = []

    while True:
        memo, active = {}, set()
        entries = [MemoBelief(b, memo, active) for b in route.panel(panel)]
        e = _build_snapshot(thing, name, beliefs=entries, objections=objections,
                            round=CORRECTIONS - corrections + 1,
                            episode=episode, arguments=arguments,
                            call_arguments=episode.args)
        _depth += 1
        try:
            got = as_judgment(entries[0](e, name))
        finally:
            _depth -= 1

        no_return = bool((getattr(got, "meta", None) or {}).get("no_return"))
        if got is None or no_return:
            corrections -= 1
            objections.append({
                "value": None, "belief": route.name, "p": 0.0,
                "reason": (getattr(got, "meta", None) or {}).get("reason")
                          if got is not None else "no return was produced"})
            if corrections > 0:
                continue
            if route.escalate():
                corrections = CORRECTIONS
                continue
            raise Unresolvable((episode.cell, "result"),
                               reason="the episode produced no return")

        # The return value is a candidate like any other: the discriminative
        # panel judges it, and a `necessary` member vetoes it.
        # ``NonEchoBelief`` earns its keep here.
        _, echoes = review(thing, name, +got, ~got, record=False,
                           entity=episode.cell, episode=episode,
                           arguments=arguments, call_arguments=episode.args)
        if echoes:
            corrections -= 1
            objections.extend({"value": +got, "belief": belief, "p": p_,
                               "reason": f"the return value was rejected: "
                                         f"{reason or 'no reason given'}"}
                              for belief, p_, reason in echoes)
            if corrections > 0:
                continue
            if route.escalate():
                corrections = CORRECTIONS
                continue
            raise Unresolvable((episode.cell, "result"),
                               reason="the return value was vetoed every time")

        changes = dict((getattr(got, "meta", None) or {}).get("changes") or {})
        # 4. atomic commit: validate every write before any of them land
        validated, refusals = _validate(thing, changes)
        from .debug import trace

        trace("changeset", (episode.cell, "result"), (validated, refusals))
        if refusals:
            corrections -= 1
            objections.extend(refusals)
            if corrections > 0:
                continue
            if route.escalate():
                corrections = CORRECTIONS
                continue
            raise Unresolvable(
                (episode.cell, "result"),
                [Attempt(1, route.name, value=+got, p=~got,
                         vetoes=[(r["belief"], r["p"], r["reason"]) for r in refusals])],
                "the changeset could not be validated; nothing was applied")

        if returns is not None:
            shape, shape_refusals = _check_return(thing, name, +got, ~got, returns)
            if shape_refusals:
                corrections -= 1
                objections.extend(shape_refusals)
                if corrections > 0:
                    continue
                raise Unresolvable(
                    (episode.cell, "result"),
                    reason="the return value never matched the declared shape")

        from .rounds import record_tells, staging

        scope = staging(ledger)
        if scope is not None:
            # round-time (§15): the changeset and outbox land at the boundary,
            # atomically with every other turn of this round.
            scope.stage(thing, validated, episode.tells, route.name, ~got)
        else:
            commit(thing, validated)
            if episode.tells:
                record_tells(ledger, thing.__entity__, episode.tells,
                             route.name, ~got)
        provenance = {
            "call": episode.expression,
            # the record holds plain values: a Thing argument reduces to its
            # entity id (§7), and ``refs`` below keeps the address durable
            "args": [_plain(a) for a in episode.args],
            "state": episode.state,
            "changes": {o.attr: o.value for o in validated},
        }
        if episode.host_state != episode.state:
            # the repetition key covered Thing-valued arguments; keep the
            # host's own tree separately so the record stays checkable
            provenance["host_state"] = episode.host_state
        # References travel in answers too: an entity id inside the return
        # value is an introduction, and the record keeps the address.
        refs = references((episode.args, episode.kwargs))
        refs += [r for r in record_refs(+got, ledger) if r not in refs]
        if refs:
            provenance["refs"] = refs
        if episode.tells:
            provenance["tells"] = [dict(t) for t in episode.tells]
        for key in ("model", "actions", "why", "template", "exposure"):
            if key in (getattr(got, "meta", None) or {}):
                provenance[key] = got.meta[key]
        opinion = ledger.add(Opinion(
            belief=route.name, entity=episode.cell, attr="result",
            value=+got, p=~got, frozen=False, meta=provenance))
        return _result(thing, episode, opinion)


def _validate(thing, changes):
    """Every write through the per-attribute pipeline."""
    validated, refusals = [], []
    for attr, value in changes.items():
        opinion, vetoes = propose(thing, attr, value, p=1.0,
                                  meta={"from_changeset": True})
        if vetoes:
            for belief, p, reason in vetoes:
                refusals.append({"value": value, "belief": belief, "p": p,
                                 "reason": f"the write to {attr!r} was rejected: "
                                           f"{reason or 'no reason given'}"})
        elif opinion is not None:
            validated.append(opinion)
    return validated, refusals


def _check_return(thing, name, value, p, returns):
    """A ``returns=``-typed call vetoes a wrong-shape return."""
    from .validators import SchemaBelief

    if returns.template is Any:
        return value, []
    verdict = SchemaBelief(returns.template).judge(value, None, name)
    if verdict == 1.0:
        return value, []
    reason = verdict[1] if isinstance(verdict, tuple) else "wrong shape"
    return value, [{"value": value, "belief": "returns", "p": 0.0,
                    "reason": f"the return value was rejected: {reason}"}]


def _argument_snapshots(episode):
    """Thing-valued arguments contribute their *boundary* views (SPEC.md §4).

    An argument is another entity, and its snapshot crosses an entity
    boundary: identity, purpose, public cells only -- no panel, no ledger
    slice, no private cells.  The full state still governs repetition
    (:meth:`Episode._state`); what changes here is perception, not identity.
    """
    out = {}
    for i, argument in enumerate(episode.args):
        if isinstance(argument, Thing):
            out[f"#{i}"] = boundary_snapshot(argument)
    for name, argument in episode.kwargs.items():
        if isinstance(argument, Thing):
            out[name] = boundary_snapshot(argument)
    return out


class CallSite(Thing):
    """The entity an episode result belongs to: the call itself.

    A call is a cell like any other, so its result is an ordinary child
    Thing -- which is why ``x.f().f().f()`` recurses on the *result* by
    construction: three links, three entities, three real episodes.
    """

    def __init__(self, thing, episode):
        set = object.__setattr__
        set(self, "__entity__", episode.cell)
        set(self, "__ledger__", thing.__ledger__)
        set(self, "__root__", thing.__root__)
        set(self, "__resolved__", thing.__root__.__resolved__)
        set(self, "__coerced__", {})
        set(self, "__beliefs__", list(thing.__beliefs__))
        set(self, "__episode_of__", episode.expression)


def _result(thing, episode, opinion):
    return Cell(CallSite(thing, episode), "result", opinion=opinion,
                entity=episode.cell)
