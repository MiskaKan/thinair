# thinair

Probabilistic Python objects. Invent attributes, methods, anything out of thin air; an LLM of your choice (local or hosted) fills in the blanks, with confidence attached.

```python
from thinair import Thing

car = Thing("A Toyota car from the 1990s with a broken engine")

car.color               # "silver"   (confidence 0.3 — a draw from the prior)
car.year                # 1995       (confidence 0.1 — one year out of a decade)
car.can_drive()         # False      (confidence 0.97)

car.repair_engine()     # no such method — a plan is imagined and *acted out*
car.can_drive()         # True — the object remembers
```

One axiom: **an object is a story, and every interaction is a continuation of it.** Everything else falls out — see [SPEC.md](SPEC.md) for the full spec.

## The principles

**1. Written code always wins.** Real attributes and methods run as ordinary Python, byte-for-byte, zero inference. The model is consulted only where your code is silent:

```python
class Car(Thing):
    """A road vehicle."""
    wheels = 4

    def honk(self):
        return "beep"

truck = Car("rusty 1970s pickup, flatbed full of firewood")
truck.wheels            # 4 — bare int; no inference ran, nothing was billed
truck.honk()            # "beep" — real code, really executed
truck.top_speed_kmh     # 105 — imagined; here the class was silent
```

**2. Certainty is a bare value.** Anything from code or assignment is a plain `str`/`int`/`bool`. Anything inferred is a child `Thing` that behaves as its value but carries `.confidence` — and never claims 1.0. The wrapper *is* the provenance:

```python
truck.engine_ok = False          # written down: authoritative, p = 1.0
truck.wheels.confidence          # AttributeError — bare values carry no doubt
truck.top_speed_kmh.confidence   # 0.6
```

**3. Everything imagined chains.** A read or call result is itself a `Thing`: real methods of its value execute for real, and names the value lacks continue the story:

```python
sedan.brands                        # ['Toyota', 'Honda', ...] — a child Thing
sedan.brands.count('Toyota')        # 1 — real list.count, free and exact
sedan.brands.overlaps(suv.brands)   # no such list method — imagined, chained
```

**4. Imagined methods act, they don't just answer.** A plan may read state, write state, and call your real methods (which actually execute) — but it can never generate or run Python source:

```python
class Boat(Thing):
    """A small motorboat."""
    def refuel(self, litres):
        self.fuel_litres = getattr(self, "fuel_litres", 0) + litres
        return self.fuel_litres

boat = Boat("a dinghy, tank empty")
boat.prepare_for_trip()   # imagined plan → calls the real refuel(20)
boat.fuel_litres          # 20.0 — a bare float, set by real code
```

**5. There is no "unknown".** An unstated fact comes back as a concrete guess with honestly low confidence; true absence comes back as `None`. Guard the branches where a guess is not good enough:

```python
car.vin_number               # "4T1BF1FK5CU123456" (confidence 0.02)
car.trailer_license_plate    # None (confidence 0.99 — confident absence)

with Thing.require(0.9):
    car.vin_number           # raises Thing.LowConfidence
```

## What it's for

**Typed answers from messy input.** Describe the object, demand a schema. `returns=` is enforced by the runtime — the imagination is made to correct itself until the value conforms:

```python
invoice = Thing("an invoice", raw_email_text)
data = invoice.extract(returns={"total_eur": float, "due_date": str, "items": [str]})
data.total_eur          # bare float, guaranteed by the schema
```

**Agents from a story plus a couple of real methods.** [`reddit_bot.py`](reddit_bot.py) in this repo is a working one: two written methods that fetch reddit over plain HTTP, and everything between them imagined:

```python
class RedditBot(Thing):
    """Checks reddit posts for the user. Browses reddit over plain HTTP."""
    def browse(self, url): ...      # real code: fetch and parse a page
    def search(self, query): ...    # real code: search all of reddit

bot = RedditBot("a bot that follows news about local LLMs and Apple MLX")
posts = bot.check_posts("Apple MLX only",
                        returns={"posts": [{"title": str, "url": str}], "mood": str})
posts.posts             # schema-guaranteed list — the plan drove browse()/search()
posts.summarize()       # the result is a Thing: chain straight into it
```

**Simulation and play.** Objects have memory by default and stay consistent with their own story; flip one flag for independent samples instead:

```python
npc = Thing("a tavern keeper who witnessed the robbery", suspicious_of="strangers")
npc.tell_story()                  # conditioned on everything said so far

die = Thing("a fair six-sided die, freshly rolled", stateful=False)
die.face; die.face; die.face      # 4, 2, 5 — rerolled on every read
```

**Sketch now, harden later.** Start with imagined names and let the story carry the prototype; when a name starts to matter, write it as real code. Call sites don't change — the answer just becomes free and certain:

```python
overlap.is_empty()      # imagined today: one LLM call, p ≈ 0.95

class Car(Thing):
    def is_empty(self):          # promoted tomorrow
        return len(self) == 0

overlap.is_empty()      # the identical call — now real code, p = 1.0, no LLM
```

**Objects are documents.** The story is text and the state is a dict, so persistence is trivial — and thawing into a subclass reattaches the written methods:

```python
db.put("car:1", json.dumps(car.freeze()))
car = Car.thaw(json.loads(db.get("car:1")))    # answers, story, state survive
pickle.dumps(car)                              # also just works
```

And you can always look at what an object has become — `__story__` is the journal of every event, and `__source__` renders the class as it looks right now:

```python
print(truck.__source__)
```
```python
class Car(Thing):
    """
    A road vehicle.

    rusty 1970s pickup, flatbed full of firewood
    """

    wheels = 4

    engine_ok = False  # written (p = 1.0)
    top_speed_kmh = 105  # imagined (p = 0.60)

    def honk(self):
        return "beep"
```

## Setup

```bash
pip install thinair
```

([thinair on PyPI](https://pypi.org/project/thinair/)) — no dependencies, one file, stdlib only. Point it at any OpenAI-compatible endpoint (defaults target a local server):

```bash
export THINAIR_BASE_URL="http://127.0.0.1:8000/v1"   # default
export THINAIR_API_KEY="1234"
export THINAIR_MODEL="Qwen3.6-35B-A3B-oQ6-mtp"
```

or in code: `Thing.defaults(model="...", base_url="...", api_key="...")` — a URL, a provider object with `complete(messages) -> text`, or a bare callable all work per instance too: `Thing("a car", model=...)`.

Then:

```bash
python demo.py         # the SPEC.md scenes, live
python reddit_bot.py   # an agent that browses reddit through two real methods
```

## Status

An experiment. Every unresolved attribute costs an inference call; answers are as good as your model. That's the fun part.

## License

MIT
