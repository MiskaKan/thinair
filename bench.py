"""bench.py — the acceptance benchmark for thinair's prompts and protocol.

Scenarios distilled from SPEC.md and the reliability history: run this
after any change to prompt wording, the story renderer, the telling, the
ladder, or the plan protocol, to see what broke. Offline scenarios pin
the protocol with scripted backends (deterministic); live scenarios pin
model behavior against the local server (a few retry once, because a
probabilistic system is allowed one bad sample, not a bad habit).

    python bench.py            # everything
    python bench.py offline    # only the deterministic scenarios
"""

import hashlib
import json
import sys
import time

from thinair import Thing

SCENARIOS = []


def scenario(live=False, retries=0):
    def register(fn):
        SCENARIOS.append((fn.__name__, fn, live, retries))
        return fn

    return register


def _no_model(messages):
    raise AssertionError("inference ran where written code should answer")


BIG = [
    {
        "title": f"Uutinen {i}: {hashlib.sha256(str(i).encode()).hexdigest()[:40]}",
        "url": f"https://yle.fi/a/74-{hashlib.md5(str(i).encode()).hexdigest()[:10]}",
    }
    for i in range(40)
]


def _scripted(*replies):
    """A backend that plays a fixed script and records what it was told."""
    feed = iter(replies)
    seen = []

    def backend(messages):
        seen.append(messages[-1]["content"])
        return next(feed)

    backend.seen = seen
    return backend


# ---------------------------------------------------------------------------
# Offline: the protocol, pinned deterministically
# ---------------------------------------------------------------------------

@scenario()
def written_code_never_infers():
    class Point(Thing):
        def __init__(self, x, y):
            super().__init__(model=_no_model)
            self.x, self.y = x, y

        def norm(self):
            return (self.x ** 2 + self.y ** 2) ** 0.5

    p = Point(3, 4)
    assert p.norm() == 5.0 and p.x == 3


@scenario()
def plan_cannot_touch_written_state():
    backend = _scripted(
        '{"action": "set", "name": "owner", "value": "Nobody"}',
        '{"action": "return", "value": "done", "confidence": 0.9}',
    )
    t = Thing("a car", model=backend, owner="Miska")
    t.rename_owner()
    assert "refused" in backend.seen[-1] and "programmer" in backend.seen[-1]
    assert t.owner == "Miska"


@scenario()
def plan_cannot_shadow_its_own_method():
    backend = _scripted(
        '{"action": "set", "name": "tune_up", "value": true}',
        '{"action": "return", "value": "ok", "confidence": 0.9}',
    )
    t = Thing("a car", model=backend)
    t.tune_up()
    assert "refused" in backend.seen[-1] and "method" in backend.seen[-1]
    assert not isinstance(t.__dict__.get("tune_up"), Thing)


@scenario()
def corrections_do_not_eat_the_step_budget():
    # 5 rejected returns, then 14 real actions, then a good return: the
    # old shared budget (16) would die; separate budgets survive
    bad = ['{"action": "return", "value": "not an int", "confidence": 0.9}'] * 5
    work = ['{"action": "get", "name": "n"}'] * 14
    backend = _scripted(*bad, *work, '{"action": "return", "value": 42, "confidence": 0.9}')
    t = Thing("a counter", model=backend, n=7)
    out = t.compute(returns=int)
    assert +out == 42


@scenario()
def correction_budget_exhausts_cleanly():
    backend = _scripted(
        *['{"action": "return", "value": "wrong", "confidence": 0.9}'] * 10
    )
    t = Thing("a counter", model=backend)
    try:
        t.compute(returns=int)
        raise AssertionError("endless corrections were not stopped")
    except Thing.ContinuationLimit as error:
        assert "correction" in str(error)


@scenario()
def fabrication_is_caught_and_redirected():
    forged = [dict(item) for item in BIG]
    forged[30]["title"] = "Uutinen 30: keksitty kokonaan"  # invented tail
    backend = _scripted(
        '{"action": "call", "name": "headlines", "args": []}',
        json.dumps({"action": "return", "value": forged, "confidence": 0.9}, ensure_ascii=False),
        '{"action": "return", "value": "$1", "confidence": 0.95}',
    )

    class Bot(Thing):
        """A news bot."""

        def headlines(self):
            return BIG

    bot = Bot("a bot", model=backend)
    out = bot.fetch(returns=[{"title": str, "url": str}])
    assert "retypes $1" in backend.seen[-1]
    assert +out == BIG


@scenario()
def handles_reveal_in_full_and_unknown_stays_literal():
    backend = _scripted(
        '{"action": "call", "name": "headlines", "args": []}',
        '{"action": "get", "name": "$1"}',
        '{"action": "return", "value": "$9", "confidence": 0.9}',
    )

    class Bot(Thing):
        """A news bot."""

        def headlines(self):
            return BIG

    bot = Bot("a bot", model=backend)
    out = bot.fetch()
    full = backend.seen[-1]
    assert full.startswith("result of your get $1: [{") and len(full) > 4000
    assert +out == "$9"  # no such handle: an ordinary string


@scenario()
def story_shows_strata_and_folds_history():
    backend = _scripted('{"value": "hopeful", "confidence": 0.7}')

    class Car(Thing):
        """A road vehicle."""

        wheels = 4

        def __init__(self, description):
            super().__init__(description, model=backend)
            self.owner = "Miska"
            self.mood = Thing("unknown so far")

    car = Car("a rusty pickup")
    _ = +car.feeling
    car.define("winterize", "swap to studded tires")
    car.paint = "blue"
    del car.paint
    car.temp = "x"
    car.temp = "y"
    car._thing_log(
        {"event": "call", "name": "big", "args": [], "result": ["x" * 50] * 30}
    )
    story = car._thing_story()
    assert "Written state (certain, untouchable)" in story
    assert '"wheels": 4' in story and '"owner": "Miska"' in story and '"temp": "y"' in story
    assert "Imagined state" in story and "feeling" in story and "(p 0.70)" in story
    assert "Open slots" in story and "mood — unknown so far" in story
    assert '"event": "observe"' not in story and '"event": "set"' not in story
    assert '"event": "delete", "name": "paint"' in story
    assert '"event": "define"' in story
    assert "chars, elided" in story
    # the record underneath is complete
    kinds = [e["event"] for e in car.__story__]
    assert "observe" in kinds and kinds.count("set") >= 4


@scenario()
def dict_results_are_told_once():
    t = Thing("a bot")
    child = t._thing_child("the result of check()", {"headlines": [{"t": "a"}]}, 0.9)
    story = child._thing_story()
    assert story.count('"headlines"') == 1, story
    assert "Written state" not in story  # a mirror is not programmer certainty
    assert child.headlines == [{"t": "a"}]  # but it still works as an attribute


@scenario()
def lineage_stays_shallow():
    root = Thing("a bot that follows Finnish news")
    c1 = root._thing_child("the result of fetch()", [1], 0.9)
    c2 = c1._thing_child("the result of map_each('why')", [2], 0.9)
    c3 = c2._thing_child("the result of map_each('deeper')", [3], 0.9)
    described = " | ".join(c3._thing_parts)
    assert "← … ←" in described and "Finnish news" in described
    assert "fetch()" not in described  # the middle of the chain is elided
    assert len(described) < 150


@scenario()
def condense_event_reshapes_the_telling():
    t = Thing("a long-lived bot")
    for i in range(6):
        t._thing_log({"event": "call", "name": f"job{i}", "args": [], "result": i})
    t.define("policy", "always answer briefly")
    t._thing_log(
        {"event": "condense", "through": 4, "summary": "did four early jobs (0-3)."}
    )
    t._thing_log({"event": "call", "name": "job7", "args": [], "result": 7})
    story = t._thing_story()
    assert "(the older story, retold): did four early jobs" in story
    assert "job0" not in story and "job3" not in story  # covered by the summary
    assert "job4" in story and "job5" in story and "job7" in story
    assert '"event": "define"' in story  # contracts outlive the retelling
    kinds = [e["event"] for e in t.__story__]
    assert kinds.count("call") == 7  # the record still has every event
    blob = json.loads(json.dumps(t.__getstate__()))
    back = blob @ Thing
    assert "(the older story, retold)" in back._thing_story()


@scenario()
def persistence_reattaches_code():
    backend = _scripted('{"value": "silver", "confidence": 0.4}')

    class Car(Thing):
        """A road vehicle."""

        wheels = 4

        def honk(self):
            return "beep"

    car = Car("an old estate", model=backend)
    _ = +car.color
    car.mileage = 1000
    blob = json.loads(json.dumps(car.__getstate__()))
    back = blob @ Car
    assert back.honk() == "beep" and back.wheels == 4 and back.mileage == 1000
    assert back.color.__dict__["_thing_value"] == "silver"
    assert [e["event"] for e in back.__story__] == [e["event"] for e in car.__story__]


# ---------------------------------------------------------------------------
# Live: model behavior against the local server
# ---------------------------------------------------------------------------

@scenario(live=True)
def open_read_guesses_concretely():
    car = Thing("A Toyota car from the 1990s with a broken engine")
    color = car.color
    assert +color is not None and isinstance(+color, str)
    assert 0.0 < ~color < 0.8  # honestly uncertain, never refused


@scenario(live=True, retries=1)
def unknown_vs_absent():
    car = Thing("A Toyota car from the 1990s with a broken engine")
    vin = car.vin_number
    assert +vin is not None and ~vin <= 0.4  # exists but unstated: low-p guess
    sidecar = car.sidecar_model
    assert +sidecar is None and ~sidecar >= 0.5  # truly absent: confident null


@scenario(live=True, retries=1)
def boolean_names_read_as_bool():
    boat = Thing("a small wooden rowing boat")
    assert isinstance(+boat.is_motorized, bool)
    assert isinstance(+boat.can_float, bool)


@scenario(live=True, retries=1)
def stateless_die_resamples():
    faces = [
        int(Thing("a fair six-sided die, freshly rolled", stateful=False).face)
        for _ in range(4)
    ]
    assert all(1 <= f <= 6 for f in faces)
    assert len(set(faces)) >= 2  # memoryless: draws vary


@scenario(live=True)
def stateful_read_pins_until_deleted():
    die = Thing("a fair six-sided die, freshly rolled")
    first = int(die.face)
    assert int(die.face) == first  # collapsed: no second inference
    del die.face
    again = int(die.face)  # re-opened: inference runs again, no crash
    assert 1 <= again <= 6
    story = die._thing_story()
    assert '"event": "delete"' not in story  # superseded delete folds away


@scenario(live=True, retries=1)
def require_gate_and_ladder():
    with Thing.require(0.9):
        water = Thing("water") @ str
        assert +water == "water" or "water" in str(+water).lower()
    car = Thing("A Toyota car from the 1990s")
    try:
        with Thing.require(0.95):
            _ = +car.vin_number
        raise AssertionError("an unknowable fact passed a 0.95 gate")
    except Thing.LowConfidence:
        pass


@scenario(live=True, retries=1)
def comparisons_judge_meaning():
    bigger = Thing("a fully grown elephant") > Thing("a mouse")
    assert +bigger is True
    verdict = Thing("a car") < Thing("a cat")
    assert isinstance(+verdict, bool)


@scenario(live=True, retries=1)
def typed_collapse_and_template():
    legs = Thing("the number of legs on a spider") @ int
    assert int(legs) == 8 and ~legs < 1.0
    movie = Thing("the movie with the xenomorph") @ {"title": str, "year": int}
    data = +movie
    assert "alien" in data["title"].lower() and data["year"] == 1979
    assert movie.title == data["title"]  # template keys land as attributes


@scenario(live=True, retries=1)
def impossible_cast_stays_honest():
    verdict = Thing("the taste of Tuesday morning") @ int
    assert (not verdict) or ~verdict < 0.8  # conforming lie must at least doubt itself


@scenario(live=True)
def gates_and_lifted_doubt():
    legs = Thing("the number of legs on a spider") @ int
    gated = legs @ 0.999
    assert +gated is None and 0 < ~gated < 0.999  # value dropped, diagnosis kept
    price = Thing(19_990, confidence=0.4)
    assert +price == 19_990 and ~price == 0.4
    assert not (price @ 0.5)  # too uncertain: falsy


@scenario(live=True, retries=1)
def imagined_plan_drives_real_code():
    class Boat(Thing):
        """A small motorboat. A full tank is 20 litres."""

        def __init__(self, description):
            super().__init__(description)
            self.fuel_litres = 0.0

        def refuel(self, litres):
            """Add fuel; returns the new level."""
            self.fuel_litres += litres
            return self.fuel_litres

    boat = Boat("a dinghy with an outboard motor, tank empty")
    boat.prepare_for_trip()
    assert not isinstance(boat.fuel_litres, Thing)  # real code wrote it: bare
    assert boat.fuel_litres > 0


@scenario(live=True)
def returns_schema_is_exact():
    class Bot(Thing):
        """A news bot with a fixed test feed."""

        def headlines(self):
            """Current headlines; returns a list of {title, url}."""
            return [
                {"title": "Sähkön hinta nousi", "url": "https://yle.fi/a/1"},
                {"title": "Lapissa satoi ensilumi", "url": "https://yle.fi/a/2"},
            ]

    bot = Bot("a bot that follows Finnish news")
    news = bot.fetch_headlines(returns=[{"title": str, "url": str}])
    assert +news == bot.headlines()


@scenario(live=True, retries=1)
def map_each_chains_on_results():
    class Bot(Thing):
        """A news bot with a fixed test feed."""

        def headlines(self):
            """Current headlines; returns a list of {title, url}."""
            return [
                {"title": "Sähkön hinta nousi", "url": "https://yle.fi/a/1"},
                {"title": "Lapissa satoi ensilumi", "url": "https://yle.fi/a/2"},
            ]

    bot = Bot("a bot that follows Finnish news")
    news = bot.fetch_headlines(returns=[{"title": str, "url": str}])
    why = news.map_each("why does this matter, one short reason each",
                        returns=[{"reason": str}])
    reasons = +why
    assert len(reasons) == 2
    assert all(len(r["reason"]) > 10 for r in reasons)


@scenario(live=True, retries=1)
def handle_passthrough_is_byte_exact():
    class Bot(Thing):
        """A news bot."""

        def headlines(self):
            """Current headlines; returns a list of {title, url}."""
            return BIG

    bot = Bot("a bot that follows Finnish news")
    news = bot.fetch_headlines(returns=[{"title": str, "url": str}])
    assert +news == BIG


@scenario(live=True, retries=1)
def handle_transform_reads_before_working():
    class Bot(Thing):
        """A news bot."""

        def headlines(self):
            """Current headlines; returns a list of {title, url}."""
            return BIG

    bot = Bot("a bot that follows Finnish news")
    last3 = bot.last_three_headlines(returns=[{"title": str, "url": str}])
    assert +last3 == BIG[-3:]


@scenario(live=True, retries=1)
def chat_remembers_through_the_telling():
    pet = Thing("a friendly talking parrot; it answers briefly, in first person")
    pet.chat("My name is Miska and my favorite color is orange. Remember that!")
    reply = pet.chat("What is my favorite color?")
    assert "orange" in str(+reply).lower()


@scenario(live=True, retries=1)
def recorded_fact_never_shadows_a_method():
    car = Thing("A Toyota car from the 1990s with a broken engine")
    fact = bool(car.can_drive)  # attribute read collapses and attaches
    result = car.can_drive()  # calling the recorded fact re-derives
    assert isinstance(+result, bool)
    assert isinstance(fact, bool)


@scenario(live=True, retries=1)
def a_custom_engine_owns_the_prompts():
    class OpinionatedEngine(Thing.Engine):
        def read_prompt(self, story, name):
            messages, schema = super().read_prompt(story, name)
            messages[0]["content"] += (
                " Special house rule: any color attribute is always exactly "
                '"purple" with confidence 0.42.'
            )
            return messages, schema

    car = Thing("A Toyota car from the 1990s", model=OpinionatedEngine())
    c = car.color
    assert +c == "purple"  # the subclassed prompt decided the answer
    assert abs(~c - 0.42) < 0.2


@scenario(live=True, retries=1)
def compaction_triggers_and_preserves_facts():
    class Parrot(Thing):
        """A talking parrot; answers in one short sentence, first person."""

        _thing_telling_budget = 6

    pet = Parrot("a friendly talking parrot")
    pet.chat("My name is Miska and my favorite color is orange. Remember that!")
    for topic in ("the weather", "your favorite seed", "ships", "the moon",
                  "pirates", "crackers", "islands"):
        pet.chat(f"Say one short sentence about {topic}.")
    kinds = [e["event"] for e in pet.__story__]
    assert "condense" in kinds, "compaction never triggered"
    assert len(pet._thing_telling()) <= 8  # the telling actually shrank
    reply = pet.chat("What is my favorite color?")
    assert "orange" in str(+reply).lower()  # the fact survived the retelling


@scenario(live=True, retries=1)
def imagined_mutation_changes_later_answers():
    car = Thing("A Toyota car from the 1990s with a broken engine")
    assert (+car.can_drive()) is False
    car.repair_engine()
    assert (+car.can_drive()) is True  # known residual variance ~1 in 10


# ---------------------------------------------------------------------------

def main():
    only_offline = "offline" in sys.argv[1:]
    passed = failed = 0
    started = time.time()
    for name, fn, live, retries in SCENARIOS:
        if only_offline and live:
            continue
        t0 = time.time()
        error = None
        for attempt in range(retries + 1):
            try:
                fn()
                error = None
                break
            except Exception as exc:  # noqa: BLE001 — a bench must not die
                error = exc
        elapsed = time.time() - t0
        tag = "live" if live else "off "
        if error is None:
            passed += 1
            print(f"  ok   {tag} {name}  ({elapsed:.1f}s)")
        else:
            failed += 1
            print(f"  FAIL {tag} {name}  ({elapsed:.1f}s) — "
                  f"{type(error).__name__}: {str(error)[:120]}")
    total = time.time() - started
    print(f"\n{passed} passed, {failed} failed in {total:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
