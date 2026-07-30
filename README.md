# thinair

Probabilistic Python objects. Invent attributes, methods, anything out of thin air; an LLM of your choice (local or hosted) fills in the blanks, with confidence attached.

**Code the certain, imagine the rest.**

One axiom: **an object is a story, and every interaction is a continuation of it.** Everything else falls out — see [SPEC.md](SPEC.md) for the full spec.

## A Thing in sixty seconds

A `Thing` is any value with a probability, made from words:

```python
from thinair import Thing

spider = Thing("the number of legs on a spider")

+spider                  # "the number of legs on a spider" — your own words
                         # back; free, no inference
~spider                  # 1.0 — your words are certain

legs = spider @ int      # collapsing happens through typing: one inference
                         # call, and now legs is a Thing carrying 8
+legs                    # 8 — the value, a real int
~legs                    # 0.99 — the probability, a bare float
```

`+` takes the value, `~` takes the probability, `@` shapes the Thing — and only `@ <type>` ever costs inference. Requirements chain, and an unmet one drops the value but keeps the probability, so failures explain themselves:

```python
+(spider @ int @ 0.8)          # 8 — typed AND vouched for at p >= 0.8

car = Thing("a rusty 1990 Toyota Hilux")   # attributes you never defined
guess = car.price_eur @ float @ 0.9        # are imagined on first read —
                                           # so this is a typed, gated guess
+guess                         # None — didn't clear the bar...
~guess                         # 0.1 — ...and this is why
if guess: ...                  # failed Things are falsy: gate whole branches
```

The type operand scales from primitives through schemas to real classes:

```python
movie = Thing("the Ridley Scott movie with the xenomorph")

+(movie @ {"title": str, "year": int})   # {'title': 'Alien', 'year': 1979}
                                         # — a schema-guaranteed dict

@dataclass
class Record:
    title: str
    year: int

+(movie @ Record)        # Record(title='Alien', year=1979) — the model
                         # imagines the kwargs, YOUR constructor builds it
```

You can assert your own doubt, and comparisons happen in Thing space — they are judgments, not byte compares:

```python
price = Thing(19_990, confidence=0.4)    # lift a belief into Thing space
+(price @ 0.5)                           # None — you said so yourself

Thing("a car") < Thing("a cat")          # False (p 0.75) — an imagined judgment
+car.price < 20_000                      # take the value out first for plain,
                                         # free Python semantics
```

## Deterministic meets probabilistic

Subclass `Thing` and write the parts you're sure of. Written code and bare values are the certain skeleton — they run as ordinary Python, cost nothing, and **the model can never touch them**. Everything else is imagined on demand:

```python
class Car(Thing):
    """A road vehicle."""
    wheels = 4                      # certain by definition

    def horn(self):                 # real code: runs in CPython,
        return "beep"               # inference is never consulted

car = Car("a rusty 1990 Toyota Hilux, engine coughs, "
          "radio stuck on a Finnish schlager station")

car.wheels               # 4 — bare int; no inference ran, nothing was billed
car.horn()               # "beep" — real code, really executed
+car.color               # "brown" — imagined; here the class was silent
~car.color               # 0.1 — and honestly unsure about it

car.owner = "Miska"                  # bare assignment: authoritative, and
                                     # locked — no plan may overwrite it
car.mood = Thing("unknown so far")   # a slot the model MAY manage
```

Provenance is permission: bare values belong to the programmer, `Thing` values belong to the imagination.

## Call anything

Any method you never wrote is imagined at call time — and it *acts*: it reads state, writes state (with confidence, journaled), and calls your real methods, which actually execute. Results are Things, so everything chains:

```python
problems = car.list_your_problems(returns=[str])
+problems                # ['Engine is coughing', 'Radio is stuck on a
                         #  Finnish schlager station', 'High rust level']

car.repair_engine()      # no such method — a plan is imagined and ACTED
                         # out; the story now contains the repair

car.diagnose().severity_of_worst_issue()   # results are Things: chain
                                           # imagined calls on imagined calls
```

Guard the branches that matter: inside `with Thing.require(0.9):` any resolution below 0.9 raises `Thing.LowConfidence` instead of flowing.

## Talk to it

There is no chatbot framework here — and `chat` is not a built-in either: nobody wrote it, it's imagined at call time like any other missing method. The car is already a chatbot, because a conversation is just more story:

```python
while True:
    print(car.chat(input("> ")))   # `chat` appears out of thin air too
```
```
> Why did you break down on me this morning?
Look, mate, it's a 1990 Hilux. The rust is high, the engine was coughing
its guts out, and frankly, I was just trying to listen to some good
Finnish schlager while falling apart.
```

Every turn is journaled, so the car remembers what you said — and objects can size each other up the same way, by handing Things to a Thing:

```python
customer = Thing("a retired couple with a caravan and two dogs, modest budget")

pick = customer.prefers(sedan, suv, roadster,
                        returns={"choice": str, "why": str})
pick.choice      # "a full-size SUV: seven seats, tow hook, thirsty"
pick.why         # "The caravan requires a tow hook, which only the SUV
                 #  has, and it provides necessary space for the two dogs."
```

## Objects are documents

`car.__getstate__()` is a JSON-able blob — description, state, story, flags; no code, no weights, no client. `blob @ Car` casts it back to life with written methods reattached; `pickle` just works. And `__source__` renders any object as the class it currently is:

```python
print(car.__source__)
```
```python
class Car(Thing):
    """
    A road vehicle.

    a rusty 1990 Toyota Hilux, engine coughs, radio stuck on a Finnish schlager station
    """

    wheels = 4

    owner = 'Miska'  # written (p = 1.0)
    color = 'brown'  # imagined (p = 0.10)
    top_speed_kmh = 130  # imagined (p = 0.10)

    def horn(self):
        return "beep"
```

`__story__` is the other lens: the full journal of every event, answer, and imagined step, in order — consistency and provenance for free.

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

or in code: `Thing.defaults(model="...", base_url="...", api_key="...")`. A URL, a provider object with `complete(messages) -> text`, or a bare callable all work per instance too: `Thing("a car", model=...)`.

Then:

```bash
python demo.py         # the SPEC.md scenes, live
python car_chat.py     # talk to a rusty Hilux — `chat` is imagined, not written
```

## Status

An experiment. Every unresolved attribute costs an inference call; answers are as good as your model. That's the fun part.

## License

MIT
