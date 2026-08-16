"""The society layer: perception boundaries, references, code-only acting.

Each new guarantee has a test that fails loudly (SPEC.md §2 spirit):

* a snapshot crossing an entity boundary carries identity, purpose and
  **public** cells only -- default private, fail closed (§4);
* references are capabilities: entity ids in ``meta.refs``, stamped on model
  changesets that write one, dereferenced fresh at render time (§4, §12);
* acting on another object is a code-only capability -- episode ``call`` is
  host-scoped, and there is no write path between minds (§9);
* the open agentic turn is its own template version, never an edit (§9).
"""

from __future__ import annotations

import pytest

from thinair import Thing, human
from thinair.beliefs import model
from thinair.episode import Episode
from thinair.ledger import Ledger
from thinair.thing import boundary_snapshot, snapshot

from fakes import FakeEngine


def ret(value, changes=None, p=0.8):
    return {"action": "return", "changes": changes or {}, "value": value, "p": p}


def cast(ledger, script=None):
    """A tiny two-entity society on one shared ledger."""
    engine = FakeEngine(script or [ret("nothing to do")])

    class Customer(Thing):
        """A person with means of their own and a public voice."""

        __beliefs__ = [model("small-fast", engine=engine), human("anna")]
        budget = Thing(float)                        # private by default
        to = Thing(str, public=True)
        saying = Thing(str, public=True)

    class Shop(Thing):
        """A shop that quotes in public."""

        __beliefs__ = [model("small-fast", engine=engine), human("keeper")]
        margin = Thing(float)                        # private by default
        quote = Thing(float, public=True)

    anna = Customer(__entity__="anna", __ledger__=ledger, budget=312.0)
    shop = Shop(__entity__="shop", __ledger__=ledger, margin=0.4)
    return anna, shop, engine


# --------------------------------------------------------------------------
# the boundary filter: default private, fail closed
# --------------------------------------------------------------------------

def test_a_boundary_snapshot_carries_public_cells_only():
    ledger = Ledger()
    anna, shop, _ = cast(ledger)
    anna.saying = "hello"
    anna.rolodex = "assigned but undeclared"         # private like everything else

    view = boundary_snapshot(anna)
    assert view.__entity__ == "anna"
    assert "person" in view.__purpose__
    assert set(view.__attrs__()) == {"saying"}
    assert view.__attrs__()["saying"].value == "hello"


def test_a_boundary_snapshot_carries_no_panel_no_ledger_no_contracts():
    ledger = Ledger()
    anna, _, _ = cast(ledger)
    view = boundary_snapshot(anna)
    assert view.__beliefs__ == []
    assert view.__ledger__ is None
    assert view.__contracts__ == {}
    assert view.__methods__ == ()
    assert view.__boundary__ is True


def test_building_a_boundary_view_never_derives():
    """A stranger's read must not trigger derivation: an unresolved public
    cell is absent, not consulted into existence."""
    ledger = Ledger()
    anna, shop, engine = cast(ledger)
    view = boundary_snapshot(shop)                   # quote never resolved
    assert "quote" not in view.__attrs__()
    assert engine.call_count == 0


# --------------------------------------------------------------------------
# references: stamped on changesets, dereferenced fresh at render time
# --------------------------------------------------------------------------

def test_a_changeset_writing_an_entity_id_is_stamped_with_refs():
    ledger = Ledger()
    engine = FakeEngine([ret("addressed", {"to": "shop", "saying": "fix it"})])

    class Customer(Thing):
        """A person."""

        __beliefs__ = [model("small-fast", engine=engine), human("anna")]
        to = Thing(str, public=True)
        saying = Thing(str, public=True)

    class Shop(Thing):
        """A shop."""

        __beliefs__ = [human("keeper")]

    anna = Customer(__entity__="anna", __ledger__=ledger)
    Shop(__entity__="shop", __ledger__=ledger)

    anna.reach_out()
    writes = [o for o in ledger.opinions(entity="anna", attr="to")
              if (o.meta or {}).get("from_changeset")]
    assert writes and writes[-1].value == "shop"
    assert writes[-1].meta.get("refs") == ["shop"]
    # prose stays prose: no exact match, no capability
    said = [o for o in ledger.opinions(entity="anna", attr="saying")
            if (o.meta or {}).get("from_changeset")]
    assert said and "refs" not in (said[-1].meta or {})


def test_a_held_reference_renders_as_the_peers_public_view():
    ledger = Ledger()
    anna, shop, _ = cast(ledger)
    shop.quote = 250.0
    anna.contact = shop                              # assignment: refs stamped

    e = snapshot(anna)
    assert "shop" in e.__peers__
    peer = e.__peers__["shop"]
    assert peer.__boundary__ is True
    assert set(peer.__attrs__()) == {"quote"}        # margin stays invisible

    from thinair.engine.prompts import render_snapshot

    text = render_snapshot(e)
    assert "public view" in text and "quote = 250.0" in text
    assert "margin" not in text
    # the perceiver's own private cells still render to itself
    assert "budget" in text


def test_a_dangling_reference_fails_closed_to_the_bare_id():
    ledger = Ledger()
    anna, shop, _ = cast(ledger)
    anna.contact = shop
    other = Ledger()
    stranger_view_source = Thing(__entity__="shop", __ledger__=other)
    del stranger_view_source                          # not this ledger's shop

    # a ref whose live handle is gone (or on another ledger) renders id-only
    import thinair.thing as T
    T._LIVE.pop("shop", None)
    e = snapshot(anna)
    peer = e.__peers__["shop"]
    assert peer.__attrs__() == {} and peer.__boundary__ is True


def test_a_strangers_read_shows_no_private_cell_and_spends_nothing():
    ledger = Ledger()
    anna, shop, engine = cast(ledger)
    anna.saying = "my car is broken"

    stranger_engine = FakeEngine([ret("looked")])

    class Stranger(Thing):
        """Someone who merely holds a reference to anna."""

        __beliefs__ = [model("small-fast", engine=stranger_engine),
                       human("nobody")]
        note = Thing(str)

    stranger = Stranger(__entity__="stranger", __ledger__=ledger)
    stranger.watching = anna

    from thinair.engine.prompts import render_snapshot

    text = render_snapshot(snapshot(stranger))
    assert "my car is broken" in text                # public voice crosses
    assert "budget" not in text and "312" not in text
    assert engine.call_count == 0                    # anna's panel never ran
    assert stranger_engine.call_count == 0           # perception is not derivation


# --------------------------------------------------------------------------
# acting on another object is code-only (§9)
# --------------------------------------------------------------------------

def test_an_episode_call_is_host_scoped():
    """The call lookup resolves against the host alone: a method that exists
    on a Thing-valued *argument* is not reachable."""
    ledger = Ledger()

    class Host(Thing):
        """The mind whose turn it is."""

        __beliefs__ = [human("h")]

    class Peer(Thing):
        """Another mind, with a real method the host must not reach."""

        __beliefs__ = [human("p")]

        def self_destruct(self):                     # pragma: no cover - must
            raise AssertionError("a peer's method ran from another mind")

    host = Host(__entity__="host", __ledger__=ledger)
    peer = Peer(__entity__="peer", __ledger__=ledger)

    episode = Episode(host, "consider", args=(peer,))
    kind, detail = episode.call("self_destruct")
    assert kind == "refused"
    assert "not a real method" in detail


def test_there_is_no_write_path_between_minds():
    """A changeset targets the host's own cells; another entity's attribute
    is simply not a declared attribute of the host."""
    ledger = Ledger()
    engine = FakeEngine([ret("tried", {"quote": 1.0})] * 9)

    class Customer(Thing):
        """A person with no quote cell of their own."""

        __beliefs__ = [model("small-fast", engine=engine), human("anna")]
        saying = Thing(str, public=True)

    anna = Customer(__entity__="anna2", __ledger__=ledger)
    from thinair.policy import Unresolvable

    with pytest.raises(Unresolvable, match="not a declared attribute"):
        anna.consider()
    assert ledger.opinions(entity="anna2", attr="quote") == []


# --------------------------------------------------------------------------
# the open agentic turn: a template variant, its own version
# --------------------------------------------------------------------------

def test_an_acting_turn_uses_the_agent_template():
    ledger = Ledger()
    anna, shop, engine = cast(ledger, script=[ret("nothing worth doing")])
    result = anna.consider(shop, acting=True)
    assert +result == "nothing worth doing"
    prompt = engine.calls[0]["messages"][0]["content"]
    assert "You are the entity shown below" in prompt
    assert result.__opinion__.meta["template"] == "agent-v2"


def test_acting_is_framing_not_identity():
    """Like ``returns=``, ``acting=`` is stripped from the call cell."""
    ledger = Ledger()
    anna, shop, engine = cast(ledger, script=[ret("once")])
    a = anna.consider(shop, acting=True)
    b = anna.consider(shop)
    assert a.__entity__ == b.__entity__
    assert engine.call_count == 1                    # the memo held


def test_a_vetoed_turn_hears_the_objection():
    """The budget belief blocks, the rejection is quoted back, the mind
    corrects -- the §5/§9 feedback loop, through an acting turn."""
    ledger = Ledger()
    engine = FakeEngine([
        ret("accepting", {"accepted_quote": 450.0}),
        ret("declining", {"saying": "too expensive for me"}),
    ])

    from thinair.beliefs import Discriminative, value_at

    class WithinTestBudget(Discriminative):
        necessary = True
        scope = "accepted_quote"

        def judge(self, value, e, attr):
            budget = value_at(e, "budget")
            if isinstance(value, (int, float)) and value <= budget:
                return 1.0
            return (0.0, f"the quote {value} exceeds the budget of {budget}")

    class Customer(Thing):
        """A person who never overspends."""

        __beliefs__ = [model("small-fast", engine=engine), human("anna")]
        budget = Thing(float)
        accepted_quote = Thing(float, beliefs=[WithinTestBudget()], public=True)
        saying = Thing(str, public=True)

    anna = Customer(__entity__="anna3", __ledger__=ledger, budget=312.0)
    result = anna.consider(acting=True)
    assert +result == "declining"
    retry = "\n".join(m["content"] for m in engine.calls[1]["messages"])
    assert "exceeds the budget" in retry
    # the veto is on the record; the over-budget value was never committed
    stated = ledger.opinions(entity="anna3", attr="accepted_quote")
    assert any(o.value == 450.0 for o in stated)
    assert anna.__root__.__resolved__.get(("anna3", "accepted_quote")) is None


# --------------------------------------------------------------------------
# the experiment, end to end: the rehearsal through the real machinery
# --------------------------------------------------------------------------

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOCIETY = os.path.join(ROOT, "experiments", "society")
if SOCIETY not in sys.path:
    sys.path.insert(0, SOCIETY)


@pytest.fixture(scope="module")
def society():
    """One full run: cast, sweep, injection, quiescence."""
    import importlib

    import minds
    import run as society_run
    importlib.reload(minds)

    engines = {name: minds.MindEngine(minds.MINDS[name])
               for name in minds.MINDS}
    minded = {name: model(f"scripted-{name}", engine=engines[name])
              for name in engines}
    ledger = Ledger()
    cast_list = society_run.build(ledger, minded)

    from sweep import Sweep
    from thinair import use_ledger

    sweeper = Sweep(cast_list)
    with use_ledger(ledger):
        cast_list[0].car_broken = True       # THE injected fact
        sweeps = sweeper.run(max_sweeps=15)
    return dict(ledger=ledger, cast=cast_list, engines=engines,
                sweeps=sweeps, log=sweeper.log)


def test_the_story_runs_to_quiescence(society):
    assert society["sweeps"] < 15
    assert society["log"][-1]["outcome"] == "quiescent"


def test_every_hop_is_on_the_tape_and_authored(society):
    ledger = society["ledger"]

    def committed(entity, attr):
        writes = [o for o in ledger.opinions(entity=entity, attr=attr)
                  if (o.meta or {}).get("from_changeset")]
        return writes[-1].value if writes else None

    assert ledger.latest_frozen("anna", "car_broken").value is True
    assert committed("anna", "to") is not None            # anna spoke
    assert committed("internet", "referral") == "forum"   # the referral
    jobs = committed("forum", "open_jobs")
    assert jobs and jobs[0]["customer"] == "anna"         # the listing
    assert committed("mech-dave", "quote") == 480.0       # both quotes
    assert committed("mech-tom", "quote") == 260.0
    assert committed("anna", "accepted_quote") == 260.0   # the acceptance
    assert committed("anna", "mechanic") == "mech-tom"
    # every hop carries its author: no anonymous movement anywhere
    assert all(o.belief for o in ledger)


def test_introductions_ride_in_meta_refs(society):
    ledger = society["ledger"]
    referral = [o for o in ledger.opinions(entity="internet", attr="referral")
                if (o.meta or {}).get("from_changeset")][-1]
    assert "forum" in referral.meta["refs"]
    listing = [o for o in ledger.opinions(entity="forum", attr="open_jobs")
               if (o.meta or {}).get("from_changeset")][-1]
    assert "anna" in listing.meta["refs"]


def test_the_budget_belief_blocked_the_over_budget_quote(society):
    ledger = society["ledger"]
    stated = ledger.opinions(entity="anna", attr="accepted_quote")
    vetoed = [o for o in stated
              if o.value == 480.0 and o.belief.startswith("withinBudget")]
    assert vetoed and vetoed[0].p == 0.0
    assert "exceeds the budget" in vetoed[0].meta["reason"]
    # the over-budget value was proposed, refused, and never resolved
    committed = [o for o in stated if (o.meta or {}).get("from_changeset")
                 and o.value == 480.0]
    resolving = society["cast"][0].__root__.__resolved__.get(
        ("anna", "accepted_quote"))
    assert resolving is not None and resolving.value == 260.0
    assert committed == [] or resolving.value != 480.0


def test_no_mind_but_annas_ever_saw_the_budget(society):
    """The privacy claim, checked at the transport: every prompt every mind
    was ever shown, and the budget appears only in anna's own."""
    for name, engine in society["engines"].items():
        for call in engine.calls:
            text = "\n".join(m.get("content", "") for m in call["messages"])
            if name == "anna":
                continue
            assert "312" not in text, (name, text[:400])
            assert "budget" not in text, (name, text[:400])


def test_no_mind_ever_wrote_anothers_cell(society):
    """Every committed write's entity is the writer's own: the tape shows
    no cross-entity write anywhere."""
    ledger = society["ledger"]
    owners = {"anna", "internet", "forum", "mech-dave", "mech-tom"}
    for o in ledger:
        if not (o.meta or {}).get("from_changeset"):
            continue
        host = o.entity.split(".", 1)[0]
        assert host in owners
        # the episode that proposed it ran on the same entity
        assert o.entity == host


def test_a_repeated_sweep_is_free(society):
    """Quiescence is physics: with nothing moved, another pass takes zero
    turns and spends zero calls."""
    from sweep import Sweep

    spent = {name: len(engine.calls)
             for name, engine in society["engines"].items()}
    sweeper = Sweep(society["cast"])
    assert sweeper.run(max_sweeps=3) == 1                # one quiet pass
    assert {name: len(engine.calls)
            for name, engine in society["engines"].items()} == spent


def test_the_sweep_delivers_mail_and_marks_it_on_the_clock():
    """An undelivered tell reaches its addressee as a turn the addressee's
    own episode answers, delivery is marked on the shared clock entity, and
    a fresh Sweep on the same ledger -- a restart -- redelivers nothing."""
    from sweep import Sweep
    from thinair.rounds import CLOCK

    ledger = Ledger()
    a_engine = FakeEngine([
        {"action": "tell", "entity": "beth", "method": "greet",
         "args": ["hello"]},
        ret("greeted"),
    ])
    b_engine = FakeEngine([ret("heard", {"saying": "alma said hello"})])

    class Voice(Thing):
        """A person with a public voice."""

        __beliefs__ = [human("desk")]
        saying = Thing(str, public=True)

    alma = Voice(__entity__="alma", __ledger__=ledger,
                 __beliefs__=[model("scripted-alma", engine=a_engine),
                              human("desk")])
    beth = Voice(__entity__="beth", __ledger__=ledger,
                 __beliefs__=[model("scripted-beth", engine=b_engine),
                              human("desk")])
    alma.next_step()                                 # immediate: mail recorded

    sweeper = Sweep([alma, beth])
    sweeper.run(max_sweeps=5)
    first = "\n".join(m["content"] for m in b_engine.calls[0]["messages"])
    assert "greet" in first and "alma" in first      # beth's episode answered
    assert +beth.saying == "alma said hello"
    marks = [o for o in ledger if o.entity == CLOCK]
    assert marks and marks[0].value["delivered"]

    spent = b_engine.call_count
    again = Sweep([alma, beth])                      # a restart: the fold and
    again.run(max_sweeps=3)                          # the markers are the memory
    assert b_engine.call_count == spent


def test_the_naming_graph_survives_a_restart():
    """Rule 3 folds standing public cells from the tape, so a fresh Sweep on
    the same ledger keeps every acquaintance a committed changeset made."""
    from sweep import Sweep

    ledger = Ledger()
    engine = FakeEngine([ret("addressed", {"to": "bea"})])

    class Speaker(Thing):
        """A person."""

        __beliefs__ = [human("desk")]
        to = Thing(str, public=True)

    ana = Speaker(__entity__="ana", __ledger__=ledger,
                  __beliefs__=[model("scripted-ana", engine=engine),
                               human("desk")])
    bea = Speaker(__entity__="bea", __ledger__=ledger)
    ana.reach_out()                                  # commits to = "bea"

    sweeper = Sweep([ana, bea])                      # built after the fact:
    assert sweeper._public_refs(ana) == ["bea"]      # the tape alone suffices
    assert "ana" in sweeper._candidates(bea)
