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

## The rules

- **Written code always wins.** Subclass `Thing`, write real methods and attributes — they run as ordinary Python, byte-for-byte, zero inference. The model is only consulted where your code is silent.
- **Certainty is a bare value.** Anything from code or explicit assignment is a plain `str`/`int`/`bool`. Anything inferred is a `Thing` that behaves as its value but carries `.confidence` — and never claims 1.0. Guard the branches that matter with `with Thing.require(0.9):` and low-confidence answers raise instead of flowing.
- **Imagined methods act, they don't just answer.** They read state, write state, and call your real methods (which actually execute). They can never generate or run Python code.

## What that buys you

**Objects that judge other objects.** Hand one Thing to another as an argument; its whole story travels with it:

```python
sedan    = Car("a compact sedan: efficient, cheap to run, small trunk")
suv      = Car("a full-size SUV: seven seats, tow hook, thirsty")
roadster = Car("a two-seat roadster: loud, fast, no luggage space")

customer = Thing("a retired couple with a caravan and two dogs, modest budget")

pick = customer.prefers(sedan, suv, roadster,
                        returns={"choice": str, "why": str})
pick.choice      # "a full-size SUV: seven seats, tow hook, thirsty"
pick.why         # "The caravan requires a tow hook, which only the SUV
                 #  has, and it provides necessary space for the two dogs."
pick.confidence  # 0.99
```

**Typed answers from messy input.** Describe the object, demand a schema. `returns=` is enforced by the runtime — the imagination corrects itself until the value conforms:

```python
invoice = Thing("an invoice", raw_email_text)
data = invoice.extract(returns={"total_eur": float, "due_date": str, "items": [str]})
data.total_eur          # bare float, guaranteed by the schema
```

**An agent from a story plus two real methods.** [`reddit_bot.py`](reddit_bot.py) in this repo is a working one: written methods fetch reddit over plain HTTP, imagination decides what to do with them:

```python
bot = RedditBot("a bot that follows news about local LLMs and Apple MLX")

posts = bot.check_posts("Apple MLX only",
                        returns={"posts": [{"title": str, "url": str}], "mood": str})
posts.posts             # real posts — the plan drove the real browse()/search()
posts.summarize()       # every result is a Thing: chain straight into it
```

**Simulation with memory — or without.** Objects stay consistent with their own story by default; flip one flag for independent samples:

```python
npc = Thing("a tavern keeper who witnessed the robbery", suspicious_of="strangers")
npc.tell_story()                  # conditioned on everything said so far

die = Thing("a fair six-sided die, freshly rolled", stateful=False)
die.face; die.face; die.face      # 4, 2, 5 — rerolled on every read
```

**Prototype now, promote later.** Start with imagined names; when one starts to matter, write it as real code. Call sites don't change — the answer just becomes free and certain:

```python
overlap.is_empty()      # imagined today: one LLM call, p ≈ 0.95

class Car(Thing):
    def is_empty(self):          # promoted tomorrow
        return len(self) == 0

overlap.is_empty()      # the identical call — now real code, p = 1.0, no LLM
```

**Objects are documents.** `freeze()` to a JSON blob, `thaw()` it back (written methods reattach), `pickle` just works. And `__source__` renders any object as the class it currently is:

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

or in code: `Thing.defaults(model="...", base_url="...", api_key="...")`. A URL, a provider object with `complete(messages) -> text`, or a bare callable work per instance too: `Thing("a car", model=...)`.

Then:

```bash
python demo.py         # the SPEC.md scenes, live
python reddit_bot.py   # an agent that browses reddit through two real methods
```

## Status

An experiment. Every unresolved attribute costs an inference call; answers are as good as your model. That's the fun part.

## License

MIT
