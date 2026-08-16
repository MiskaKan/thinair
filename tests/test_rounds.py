"""Messages and rounds (SPEC.md §15): tells, delivery, the society clock.

Each guarantee has a test that fails loudly:

* a ``tell`` is speech on the sender's record -- atomic with the changeset,
  refs stamped, never sent by a turn that failed;
* outside ``rounds()`` every call evaluates immediately (the default is
  untouched); inside, a call declares a sealed pending turn;
* no turn sees another's same-round work (the seal), and all effects land
  together at the boundary;
* delivery runs the message as a call the addressee answers next round --
  one turn per mind per round, simultaneous voices batched into one inbox;
* quiescence is the boundary that finds nothing; an unchanged re-seed
  replays for free; the clock markers are frozen code on ``__rounds__``.
"""

from __future__ import annotations

import pytest

from thinair import Thing, rounds
from thinair.beliefs import model
from thinair.ledger import Ledger
from thinair.policy import Unresolvable
from thinair.rounds import CLOCK, PendingTurn

from fakes import FakeEngine


def ret(value, changes=None, p=0.8):
    return {"action": "return", "changes": changes or {}, "value": value,
            "p": p}


def tell(entity, verb, args=()):
    return {"action": "tell", "entity": entity, "method": verb,
            "args": list(args)}


def mind(ledger, name, engine, **cells):
    class Mind(Thing):
        """A small mind."""

        __beliefs__ = [model(f"scripted-{name}", engine=engine)]
        note = Thing(str, public=True)

    Mind.__name__ = Mind.__qualname__ = "Mind"
    return Mind(__entity__=name, __ledger__=ledger, **cells)


def tells_of(ledger, sender=None):
    return [o for o in ledger if (o.meta or {}).get("tell")
            and (sender is None or o.entity == sender)]


def markers_of(ledger):
    return [o for o in ledger if o.entity == CLOCK]


# --------------------------------------------------------------------------
# the tell action: speech on the sender's record
# --------------------------------------------------------------------------

def test_a_tell_lands_on_the_sender_atomically_with_the_changeset():
    ledger = Ledger()
    engine = FakeEngine([tell("bob", "greet", ["hi"]),
                         ret("said hi", {"note": "greeted bob"})])
    alice = mind(ledger, "alice", engine)
    mind(ledger, "bob", FakeEngine())

    result = alice.next_step()
    assert +result == "said hi"
    sent = tells_of(ledger, "alice")
    assert len(sent) == 1
    assert sent[0].value == {"to": "bob", "verb": "greet", "args": ["hi"]}
    assert sent[0].attr.startswith("__tell__:")
    assert "bob" in sent[0].meta["refs"]
    assert result.__opinion__.meta["tells"] == [
        {"to": "bob", "verb": "greet", "args": ["hi"]}]


def test_a_failed_turn_sends_nothing():
    """The outbox is part of the atomic commit: a turn that never returns
    (budget exhausted) leaves no message behind."""
    ledger = Ledger()
    engine = FakeEngine([tell("bob", "greet")])       # tells forever, never
    alice = mind(ledger, "alice", engine)             # returns
    with pytest.raises(Unresolvable):
        alice.next_step()
    assert tells_of(ledger) == []


def test_a_tell_to_yourself_is_refused_but_costs_only_the_action():
    ledger = Ledger()
    engine = FakeEngine([tell("alice", "remind"), ret("done")])
    alice = mind(ledger, "alice", engine)
    assert +alice.next_step() == "done"
    assert tells_of(ledger) == []
    observation = engine.calls[1]["messages"][-1]["content"]
    assert "refused" in observation


def test_the_tell_is_not_resent_on_replay():
    """A repeated call replays from the record: same result, no second
    message, zero engine calls."""
    ledger = Ledger()
    engine = FakeEngine([tell("bob", "greet"), ret("said hi")])
    alice = mind(ledger, "alice", engine)
    alice.next_step()
    spent = engine.call_count
    alice.next_step()                                  # byte-identical state
    assert engine.call_count == spent
    assert len(tells_of(ledger)) == 1


# --------------------------------------------------------------------------
# the scope: immediate outside, declared inside, sealed within a round
# --------------------------------------------------------------------------

def test_outside_rounds_calls_evaluate_immediately():
    ledger = Ledger()
    engine = FakeEngine([ret("ran now")])
    alice = mind(ledger, "alice", engine)
    assert +alice.next_step() == "ran now"
    assert engine.call_count == 1


def test_inside_rounds_a_call_declares_and_round_evaluates():
    ledger = Ledger()
    engine = FakeEngine([ret("ran in round 1")])
    alice = mind(ledger, "alice", engine)
    with rounds(ledger) as clock:
        turn = alice.next_step()
        assert isinstance(turn, PendingTurn)
        assert engine.call_count == 0              # declared, not evaluated
        assert clock.round() is True
        assert engine.call_count == 1
    assert +turn == "ran in round 1"


def test_no_turn_sees_anothers_same_round_work():
    """The seal: B's prompt renders A's *pre-round* public state even though
    A's turn ran first; both commits land at the boundary together.

    A's starting note is set by an ordinary turn (a changeset), never by
    assignment -- code-assigned cells freeze, and a frozen cell rightly
    refuses a mind's later write."""
    ledger = Ledger()
    a_engine = FakeEngine([ret("init", {"note": "quiet"}),
                           ret("wrote", {"note": "CHANGED-BY-A"})])
    b_engine = FakeEngine([ret("looked")])
    alice = mind(ledger, "alice", a_engine)
    bob = mind(ledger, "bob", b_engine)
    assert +alice.next_step() == "init"            # immediate: note = quiet
    with rounds(ledger) as clock:
        alice.rethink()
        bob.consider(alice)
        clock.round()
    b_prompt = "\n".join(m["content"]
                         for m in b_engine.calls[0]["messages"])
    assert "quiet" in b_prompt
    assert "CHANGED-BY-A" not in b_prompt
    assert +alice.note == "CHANGED-BY-A"           # landed at the boundary


def test_forcing_a_pending_turn_evaluates_early_against_the_seal():
    ledger = Ledger()
    engine = FakeEngine([ret("forced")])
    alice = mind(ledger, "alice", engine)
    with rounds(ledger) as clock:
        turn = alice.next_step()
        assert +turn == "forced"                   # forced before round()
        assert engine.call_count == 1
        clock.round()
    assert engine.call_count == 1                  # not run twice


def test_round_outside_scope_is_not_a_thing():
    ledger = Ledger()
    with pytest.raises(Exception):
        ledger.round()                             # no such method: the verb
    with pytest.raises(RuntimeError):
        with rounds(ledger):
            with rounds(ledger):                   # one clock at a time
                pass


# --------------------------------------------------------------------------
# delivery: the message is the call, answered next round
# --------------------------------------------------------------------------

def test_a_tell_is_delivered_as_a_call_next_round():
    ledger = Ledger()
    a_engine = FakeEngine([tell("bob", "greet", ["hi"]), ret("greeted")])
    b_engine = FakeEngine([ret("heard", {"note": "alice said hi"})])
    alice = mind(ledger, "alice", a_engine)
    bob = mind(ledger, "bob", b_engine)
    with rounds(ledger) as clock:
        alice.next_step()
        assert clock.round() is True               # round 1: alice speaks
        assert b_engine.call_count == 0            # not yet delivered-as-run
        assert clock.round() is True               # round 2: bob answers
        assert clock.round() is False              # round 3: quiescent
    prompt = "\n".join(m["content"] for m in b_engine.calls[0]["messages"])
    assert "greet" in prompt and "alice" in prompt
    assert +bob.note == "alice said hi"


def test_simultaneous_voices_arrive_as_one_inbox():
    """Two minds address the same one in one round: a single ``receive``
    turn carries both messages -- compared, never raced."""
    ledger = Ledger()
    a_engine = FakeEngine([tell("carol", "offer", [480]), ret("offered")])
    b_engine = FakeEngine([tell("carol", "offer", [260]), ret("offered")])
    c_engine = FakeEngine([ret("chose", {"note": "took 260"})])
    alice = mind(ledger, "alice", a_engine)
    bob = mind(ledger, "bob", b_engine)
    carol = mind(ledger, "carol", c_engine)
    with rounds(ledger) as clock:
        alice.next_step()
        bob.next_step()
        while clock.round():
            pass
    assert c_engine.call_count == 1                # one turn, both voices
    prompt = "\n".join(m["content"] for m in c_engine.calls[0]["messages"])
    assert "receive" in prompt
    assert "480" in prompt and "260" in prompt
    assert +carol.note == "took 260"


def test_mail_to_an_absent_entity_stays_on_the_record():
    ledger = Ledger()
    engine = FakeEngine([tell("nobody-here", "hello"), ret("sent")])
    alice = mind(ledger, "alice", engine)
    with rounds(ledger) as clock:
        alice.next_step()
        while clock.round():
            pass
    sent = tells_of(ledger, "alice")
    assert len(sent) == 1                          # recorded, never delivered
    delivered = [d for m in markers_of(ledger)
                 for d in m.value.get("delivered", ())]
    assert delivered == []


def test_layer_one_mail_is_delivered_by_the_next_scope():
    """A tell recorded outside any scope waits in the ledger; a later
    ``rounds()`` scan delivers it -- the society resumes from its record."""
    ledger = Ledger()
    a_engine = FakeEngine([tell("bob", "ping"), ret("pinged")])
    b_engine = FakeEngine([ret("ponged", {"note": "pong"})])
    alice = mind(ledger, "alice", a_engine)
    bob = mind(ledger, "bob", b_engine)
    alice.next_step()                              # immediate: mail recorded
    with rounds(ledger) as clock:
        while clock.round():
            pass
    assert +bob.note == "pong"


# --------------------------------------------------------------------------
# the clock: quiescence, free re-seeding, frozen markers
# --------------------------------------------------------------------------

def test_an_idle_epoch_costs_nothing():
    ledger = Ledger()
    engine = FakeEngine([ret("did the thing")])
    alice = mind(ledger, "alice", engine)
    with rounds(ledger) as clock:
        alice.next_step()
        while clock.round():
            pass
        spent = engine.call_count
        alice.next_step()                          # epoch 2: unchanged world
        assert clock.round() is False              # replay; quiescent at once
        assert engine.call_count == spent
    assert engine.call_count == spent


def test_the_clock_is_frozen_code_on_the_record():
    ledger = Ledger()
    engine = FakeEngine([tell("bob", "ping"), ret("pinged")])
    alice = mind(ledger, "alice", engine)
    mind(ledger, "bob", FakeEngine([ret("pong")]))
    with rounds(ledger) as clock:
        alice.next_step()
        while clock.round():
            pass
    marks = markers_of(ledger)
    assert marks and all(m.frozen and m.belief == "code:rounds"
                         for m in marks)
    assert [m.meta["round"] for m in marks] == list(
        range(1, len(marks) + 1))
    delivered = [tuple(d) for m in marks
                 for d in m.value.get("delivered", ())]
    sent = tells_of(ledger, "alice")
    assert [(o.entity, o.attr) for o in sent] == delivered


def test_the_clock_resumes_from_the_record():
    """A second scope continues the numbering -- the ledger, not the scope,
    is the memory."""
    ledger = Ledger()
    engine = FakeEngine([ret("one", {"note": "first"})])
    alice = mind(ledger, "alice", engine)
    with rounds(ledger) as clock:
        alice.next_step()
        while clock.round():
            pass
    first = max(m.meta["round"] for m in markers_of(ledger))
    engine.script = [ret("two")]
    alice.mood = "poked"                           # code moves the world
    with rounds(ledger) as clock:
        alice.next_step()
        while clock.round():
            pass
    assert max(m.meta["round"] for m in markers_of(ledger)) > first


def test_leaving_the_scope_lands_what_was_forced():
    """Exception safety: a turn forced mid-round commits at scope exit even
    when the block is left early -- and nothing leaks past the block."""
    ledger = Ledger()
    engine = FakeEngine([ret("early", {"note": "forced-write"})])
    alice = mind(ledger, "alice", engine)
    try:
        with rounds(ledger):
            turn = alice.next_step()
            assert +turn == "early"                # staged, not landed
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    assert +alice.note == "forced-write"           # landed by __exit__
    after = alice.next_step                        # outside: plain immediate
    engine.script = [ret("plain")]
    assert +alice.next_step() == "plain"
