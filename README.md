# thinair

Probabilistic Python objects. Invent attributes, methods, anything out of thin air; an LLM of your choice (local or hosted) fills in the blanks, with confidence attached.

```python
from thinair import Thing

car = Thing("A Toyota car from the 1990s with a broken engine")

car.color               # "unknown"  (confidence 0.94)
car.year                # 1995       (confidence 0.1 — one year out of a decade)
car.can_drive()         # False      (confidence 0.97)

car.repair_engine()     # no such method — a plan is imagined and *acted out*
car.can_drive()         # True — the object remembers
```

One axiom: **an object is a story, and every interaction is a continuation of it.** Everything else falls out — see [SPEC.md](SPEC.md) for the full spec.

## The rules

- **Written code always wins.** Subclass `Thing`, write real methods and attributes — they run as ordinary Python, byte-for-byte, zero inference. The model is only consulted where your code is silent.
- **Certainty is a bare value.** Anything from code or explicit assignment is a plain `str`/`int`/`bool`. Inferred values duck-type as their value but carry `.confidence` — and never claim 1.0.
- **Imagined methods can act, not just answer.** They read state, write state, and call your real methods (which actually execute). They can never generate or run Python code.

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

Also in the box: `stateful=False` for memoryless objects (a die that rerolls every read), `with Thing.require(0.9):` to guard branches on confidence, and `freeze()`/`thaw()` to store objects in a database and restore them later.

## Setup

No dependencies — one file, stdlib only. Point it at any OpenAI-compatible endpoint (defaults target a local server):

```bash
export THINAIR_BASE_URL="http://127.0.0.1:8000/v1"   # default
export THINAIR_API_KEY="1234"
export THINAIR_MODEL="Qwen3.6-35B-A3B-oQ6-mtp"
```

or in code: `Thing.defaults(model="...", base_url="...", api_key="...")`.

Then:

```bash
python demo.py         # the SPEC.md scenes, live
python reddit_bot.py   # an agent that browses reddit through two real methods
```

## Status

An experiment. Every unresolved attribute costs an inference call; answers are as good as your model. That's the fun part.

## License

MIT
