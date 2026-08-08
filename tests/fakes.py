"""Scripted stand-ins: zero network, fully deterministic.

Two of them:

* :class:`FakeSnapshot` — the sealed value a belief receives.  ``thing.py``
  ships the real one in M3; this is the same *interface* built by hand, so
  belief and validator tests never need a Thing.
* :class:`FakeEngine` — a scripted generative belief's transport.  Answers
  come from a list or a dict keyed by attribute; every call is recorded, so
  "frozen cells make zero engine calls" is a countable assertion rather than
  a hope.
"""

from __future__ import annotations

import json

from thinair.ledger import Ledger, Opinion


class FakeSnapshot:
    """An inert, sealed snapshot.

    Cells are passed as ``attr=value`` or ``attr=(value, p)`` and become
    ordinary :class:`~thinair.ledger.Opinion` objects, which is exactly what
    attribute access on a real snapshot yields.  Everything framework-side
    lives in dunders so no domain name is ever shadowed; an unresolved cell
    reads as ``None``.
    """

    def __init__(self, entity="e1", beliefs=(), ledger=None, value=None, p=0.0,
                 contracts=None, methods=(), arguments=None, objections=(),
                 owner=None, episode=None, call_arguments=None,
                 class_name="Fake", provenance=(), purpose="",
                 **cells):
        object.__setattr__(self, "__entity__", entity)
        object.__setattr__(self, "__beliefs__", list(beliefs))
        object.__setattr__(self, "__ledger__", ledger if ledger is not None else Ledger())
        object.__setattr__(self, "__value__", value)
        object.__setattr__(self, "__p__", p)
        object.__setattr__(self, "__contracts__", dict(contracts or {}))
        object.__setattr__(self, "__methods__", tuple(methods))
        object.__setattr__(self, "__arguments__", dict(arguments or {}))
        object.__setattr__(self, "__objections__", tuple(objections))
        object.__setattr__(self, "__owner__", owner)
        object.__setattr__(self, "__episode__", episode)
        object.__setattr__(self, "__call_arguments__", call_arguments)
        object.__setattr__(self, "__class_name__", class_name)
        object.__setattr__(self, "__provenance__", tuple(provenance))
        object.__setattr__(self, "__purpose__", purpose)   # the class docstring
        object.__setattr__(self, "_cells", {
            attr: self._as_opinion(entity, attr, spec) for attr, spec in cells.items()
        })

    @staticmethod
    def _as_opinion(entity, attr, spec):
        if isinstance(spec, Opinion):
            return spec
        if isinstance(spec, tuple) and len(spec) == 2:
            value, p = spec
        else:
            value, p = spec, 1.0
        return Opinion(belief="fake", entity=entity, attr=attr, value=value,
                       p=float(p), t=1.0, frozen=(p == 1.0))

    # -- the ordinary public interface a belief sees ----------------------
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_cells")[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):        # sealed
        raise AttributeError(f"{type(self).__name__} is sealed; cannot set {name!r}")

    def __attrs__(self):
        return dict(object.__getattribute__(self, "_cells"))

    def __opinion__(self, attr):
        return object.__getattribute__(self, "_cells").get(attr)

    def with_head(self, belief):
        """A copy whose panel leads with ``belief`` — routing is list order."""
        clone = object.__new__(FakeSnapshot)
        for key, value in self.__dict__.items():
            object.__setattr__(clone, key, value)
        panel = [belief] + [b for b in getattr(self, "__beliefs__") if b is not belief]
        object.__setattr__(clone, "__beliefs__", panel)
        return clone


def head(proposal, p=1.0, **kwargs):
    """A snapshot whose panel head proposes ``(proposal, p)`` — the common shape.

    The parameter is not called ``value`` because ``FakeSnapshot`` already has
    one: a snapshot's ``__value__`` is what the entity *carries*, which is a
    different thing from what its proposer is offering right now.
    """
    return FakeSnapshot(beliefs=[lambda e, a: (proposal, p)], **kwargs)


class ScriptedBelief:
    """A generative belief that reads its answers off a script.

    Duck-typed rather than subclassed on purpose: the runtime never checks
    ``isinstance``, so a plain object with ``id``, ``necessary`` and the
    call signature must work.  If that ever stops being true, this class
    fails first.
    """

    necessary = False
    veto_line = 0.5
    proposes = True

    def __init__(self, script, belief_id="scripted:1"):
        self.id = belief_id
        self.short = belief_id
        self.script = script
        self.calls = []

    def __call__(self, e, attr):
        self.calls.append(attr)
        answer = self.script
        if isinstance(answer, dict):
            answer = answer.get(attr)
        elif isinstance(answer, list):
            answer = answer[min(len(self.calls) - 1, len(answer) - 1)]
        if callable(answer):
            answer = answer(e, attr)
        return answer


class FakeEngine:
    """A scripted ``Engine``: ``complete(messages, ...) -> (text, meta)``.

    ``script`` is a list of replies served in order (the last one repeats), or
    a callable taking the message list.  A reply may be a string (sent
    verbatim) or any JSON-able object (dumped).  ``calls`` records every
    request, which is what makes engine-call counting testable.
    """

    def __init__(self, script=None, model="fake-model"):
        self.script = script if script is not None else [{"value": None, "p": 0.0}]
        self.model = model
        self.calls = []

    def complete(self, messages, schema=None, temperature=0.2, max_tokens=None, **extra):
        self.calls.append({"messages": messages, "schema": schema,
                           "temperature": temperature, "extra": extra})
        script = self.script
        reply = script(messages) if callable(script) else \
            script[min(len(self.calls) - 1, len(script) - 1)]
        text = reply if isinstance(reply, str) else json.dumps(reply)
        meta = {"model": self.model, "temperature": temperature,
                "call": len(self.calls), "transport": "fake"}
        return text, meta

    @property
    def call_count(self):
        return len(self.calls)


class FlakyFakeEngine(FakeEngine):
    """A ``FakeEngine`` that misbehaves first, to exercise the retry ladder.

    The first ``bad`` replies are unparseable prose; the script takes over
    afterwards, so a test can assert that a parse failure costs a retry and
    not an answer.
    """

    def __init__(self, script=None, bad=1, noise="I cannot help with that.", **kwargs):
        super().__init__(script, **kwargs)
        self.bad = int(bad)
        self.noise = noise

    def complete(self, messages, **kwargs):
        if len(self.calls) < self.bad:
            self.calls.append({"messages": messages, "kwargs": kwargs})
            return self.noise, {"model": self.model, "call": len(self.calls),
                                "transport": "fake"}
        return super().complete(messages, **kwargs)
