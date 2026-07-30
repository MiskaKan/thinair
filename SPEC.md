# Dream-code spec

The code we wish we could run. This page is the acceptance test for any
implementation: when every scene below behaves as its comments claim, the
system is done.

The principles:

1. **Object = story.** Every interaction is a continuation of the story;
   the continuation is appended (unless the object is stateless — scene 9).
2. **Code before inference, at every seam.** Written code and recorded state
   are authoritative; inference is consulted only where they are silent.
3. **Certainty is a bare value.** p = 1.0 means an ordinary `str`/`int`/`bool`
   with no wrapper. Inference never returns exactly 1.0 — it returns a child
   `Thing` that behaves as its value and carries `.confidence`. Even state
   written by an imagined plan carries the plan's confidence; a bare value
   is proof the programmer wrote it, as code or explicit assignment.
4. **Closure through the protocol, never through code.** Imagined methods can
   do what written methods do, but only via a closed action vocabulary:
   *get, set, delete, call, define, return*. Imagination
   never writes, generates, or executes Python source. No exec, no eval,
   ever.
5. **One class.** The entire public surface is `Thing`. Everything inference
   produces is another `Thing`; the guards and exceptions are reached through
   it (`Thing.require`, `Thing.LowConfidence`, `Thing.ContinuationLimit`).

---

## Scene 1 — pure invention, rich construction

```python
from thinair import Thing          # the only import there is

car = Thing("A Toyota car from the 1990s with a broken engine")

car.color               # "silver" — a child Thing that behaves as the str
car.color.confidence    # 0.3 — always the probability the value is correct
car.year                # 1994 — an int; name and usage steer the type
car.can_drive()         # False, confidence ~0.97

car.color               # "silver" again — first read collapsed it;
                        # the answer joined the story, so it is now consistent
del car.color           # deletion is an event: re-opens the distribution
```

There is no "unknown". An unstated fact is a draw from the prior: a concrete
value of the natural type with honestly low confidence — gate it with
`Thing.require` (scene 7) when a guess is not good enough. True absence
("this car tows no trailer") is `None`, held with confidence like any other
fact. Sentinel strings never travel in the value channel.

The constructor absorbs anything. Positional args are woven into the story
(strings as description, other values by their content); keyword args become
authoritative state, p = 1.0:

```python
car = Thing("a Toyota", "from the 1990s", owner_manual_pdf_text,
            color="red", mileage_km=289_000)
car.color               # "red" — bare str; you wrote it down
car.mileage_km          # 289000 — bare int
```

(`stateful`, `model`, and `confidence` are the only reserved keyword names —
scenes 9–10 and 13.)

## Scene 2 — subclass mixing written and imagined

```python
class Car(Thing):
    """A road vehicle."""
    wheels = 4                      # stratum 1: certain by definition

    def honk(self):                 # real code — runs in CPython,
        return "beep"               # inference is never consulted

truck = Car("rusty 1970s pickup, flatbed full of firewood")

truck.wheels            # 4        — bare int; machinery never even fired
truck.honk()            # "beep"   — bare str
truck.top_speed_kmh     # 105      — imagined; a child Thing behaving as an int

truck.wheels.confidence          # AttributeError — bare values carry no doubt
truck.top_speed_kmh.confidence   # ~0.6 — the wrapper IS the provenance
```

## Scene 3 — deterministic writes are story events

```python
truck.engine_ok = False     # ordinary assignment; authoritative, p = 1.0
truck.can_drive()           # False — inference is conditioned on recorded
                            # state, and may not contradict it
```

## Scene 4 — imagined mutation changes later answers

```python
car.can_drive()             # False
car.repair_engine()         # no such method exists; a plan is imagined at
                            # call time — it acts, it doesn't just answer
car.can_drive()             # True, confidence ~0.93 — the story now
                            # contains the repair
```

## Scene 5 — an imagined method drives real code

```python
class Boat(Thing):
    """A small motorboat."""
    def __init__(self, description):
        super().__init__(description)
        self.fuel_litres = 0.0

    def refuel(self, litres):           # real method
        self.fuel_litres += litres
        return self.fuel_litres

boat = Boat("a dinghy with an outboard motor, tank empty")

boat.prepare_for_trip()
# Imagined plan, executed step by step through the object's own surface:
#   1. get  boat.fuel_litres       -> 0.0        (real state, p = 1.0)
#   2. call boat.refuel(20)        -> 20.0       (real code ACTUALLY RUNS —
#                                                 it may not be imagined,
#                                                 because it exists)
#   3. set  boat.safety_checked = True           (recorded state, carrying
#                                                 the plan's confidence)
#   4. return True, confidence ~0.88             (min over imagined steps;
#                                                 real steps contribute 1.0)

boat.fuel_litres        # 20.0 — bare float, mutated by real code that an
                        # imagined method invoked
boat.safety_checked     # True — persisted with the plan's confidence
                        # (p < 1: imagination wrote it, not the programmer);
                        # visible to written code later
boat.__story__          # full journal: every event, answer, and sub-step,
                        # in order — consistency and provenance for free

print(boat.__source__)  # the object rendered as Python-like source, as it
                        # looks right now: written code verbatim, imagined
                        # state and vocabulary as stubs annotated with p
```

## Scene 6 — the action vocabulary: all an imagination may do

An imagined plan is a sequence of these verbs, each executed by the runtime
through the normal Python attribute protocol — so principle 2 applies at
every single step:

```python
# get    x.attr                  read state or collapse a new attribute
# set    x.attr = value          write recorded state, carrying confidence
# delete del x.attr              supersede / re-open
# call   x.method(args)          real methods execute; missing ones recurse
#                                into another imagined plan
# define                         record what a name MEANS, as story text —
#                                never as Python source
# return value, confidence       terminate the plan; the value is always
#                                written out whole — one way to return,
#                                never an abbreviation or a reference
```

`define` is what "creating a function" means here. It adds a contract to the
story; later calls to that name are continuations conditioned on it:

```python
car.define("winterize", "drain the washer fluid, swap to studded tires, "
                        "store the battery indoors")
car.winterize()             # imagined plan, now guided by that definition
car.tires                   # "studded", confidence ~0.9
```

There is no verb for emitting code. A `Thing` can never extend the
deterministic stratum by itself; only the programmer writes stratum 1.

Recorded state does not lock the future — but only imagination's own
stratum is negotiable. A plan may create new attributes freely and may
overwrite or delete any value that carries confidence: anything imagination
wrote, or a slot the programmer explicitly opened by giving it a `Thing`
value. Bare values are the programmer's certainty and, like written code,
are untouchable — a plan that tries is refused and records its change under
a new name instead:

```python
class Car(Thing):
    def __init__(self):
        self.owner = "Miska"                  # bare: no plan may touch it
        self.mood = Thing("unknown so far")   # a slot imagination manages

car.cheer_up()      # may set mood — and may NOT set owner
```

What imagination writes it writes with its confidence (its own per `set`,
else the plan's floor so far), so certainty can still only originate from
the programmer.

## Scene 7 — confidence guards

```python
with Thing.require(0.9):
    if car.can_drive():     # 0.93 >= 0.9 — passes, branch is trusted
        plan_road_trip(car)
    car.vin_number          # confidence ~0.2 — raises Thing.LowConfidence
```

## Scene 8 — runaway imagination is an exception, not garbage

```python
weird = Thing("a machine that must inspect itself to know anything")
weird.deep_self_analysis()  # if the imagined plan exceeds the step budget:
                            # raises Thing.ContinuationLimit
                            # (cf. RecursionError)
```

## Scene 9 — statefulness is a flag

`stateful=True` (the default) is everything above: answers append, the story
accumulates, the object has memory. `stateful=False` freezes the story at
construction — inference still answers, but nothing it says is recorded.
Reads become independent samples:

```python
die = Thing("a fair six-sided die, freshly rolled", stateful=False)
die.face                # 4, confidence ~0.17
die.face                # 2 — no memory, resampled each read
die.face                # 5

pinned = Thing("a fair six-sided die, freshly rolled")   # stateful
pinned.face             # 3
pinned.face             # 3 — collapsed, forever (until an event)
```

Strata 1–2 are untouched by the flag — written code and explicit assignment
still work; you cannot turn off Python. Only imagination loses its pen.

## Scene 10 — the inference backend is pluggable

```python
Thing.defaults(model="claude-sonnet-5")        # class-wide default (else:
                                               # sensible built-in / env var)

a = Thing("a car")                             # uses the class default
b = Thing("a car", model="https://my-vllm.local:8000/v1")   # any OpenAI-ish
                                               # HTTP endpoint by URL
c = Thing("a car", model=my_client)            # any object implementing the
                                               # one-method provider protocol:
                                               # complete(messages) -> text
                                               # (OpenAI-style chat messages
                                               # in, completion text out)
d = Thing("a car", model=lambda messages: "…") # or just a callable — handy
                                               # for tests and stubs
e = Thing("a car", model="file://models/tiny.gguf")   # future: embedded
                                               # in-process model, no network
```

Subclasses inherit and may override the default the ordinary Python way:

```python
class Npc(Thing):
    """A tavern keeper."""
Npc.defaults(model="file://models/npc-8b.gguf")
```

## Scene 11 — the object is a document

The story is text and the state is a dict; `__getstate__()` is the dump
and `@` is the restore — no freeze/thaw vocabulary, just Python's own
protocol and the cast operator:

```python
blob = car.__getstate__()       # JSON-able dict: description, state, story,
                                # flags — no code, no weights, no client
db.put("car:1", json.dumps(blob))

later = json.loads(db.get("car:1")) @ Thing
later.color                     # same collapsed answer as before the dump
later.can_drive()               # True — the repair from scene 4 survived

pickle.dumps(car)               # also just works (same mechanism)
```

Casting a document into a subclass reattaches stratum 1: `blob @ Car`
gives the written methods back their body; the story never contained them,
only their effects.

## Scene 12 — reads and results are Things: chaining and schema guarantees

Everything inference produces — an attribute read or a call result — is a
child `Thing` born from its story. It behaves as its value (`bool`, `str`,
arithmetic, comparison, iteration, `.confidence`), real methods of the value
win over imagination, and anything the value lacks becomes a further
continuation. A child is always a plain `Thing`, never the parent's class:
a list of headlines is not a news bot. Imagined methods may mutate their
object, but what they return is a new Thing — recast it with `@ Class`
(scene 13) when the class genuinely applies:

```python
news = bot.check_news("related to Ukraine, headlines only",
                      returns={"headlines": [{"title": str, "url": str}]})
news.headlines          # bare list of dicts — guaranteed by the schema
news.keywords()         # the result is a Thing: chain another imagined call
bool(car.can_drive())   # scalar results still work as plain values

sedan.brands                        # ['Toyota', ...] — reads are Things too
sedan.brands.count('Toyota')        # real list.count executes — never imagined
sedan.brands.overlaps(suv.brands)   # no such list method: an imagined call,
                                    # chained on the result of a read
```

`returns=` is the reserved schema kwarg on any imagined call: a template of
types (`str`, `int`, `bool`, `float`), dicts, and lists. The runtime rejects
any return that does not match and makes the imagination correct itself —
within the step budget — so a value that arrives is a value that conforms.
Dict results also land as authoritative attributes on the child (state wins
over inference, as always).

## Scene 13 — two worlds, three operators

Things operate in Thing space: comparing them is a judgment, not a byte
compare, and returns a child Thing with confidence. `@` shapes a Thing
without ever leaving Thing space — even failure is a Thing — and the two
unary operators take a value or a probability out:

```python
t @ int     # approximate AS a type: a Thing whose value is guaranteed an
            # int (schema-corrected; Thing(None, 0.01) if it cannot conform)
t @ 0.8     # confidence gate: the same Thing if confidence >= 0.8, else a
            # Thing whose value is dropped but whose probability survives —
            # free, never runs inference; story Things pass any gate
+t          # the value, no questions asked: the carried value, or the
            # story text if nothing has collapsed — free, never infers
~t          # the probability: a bare float; 1.0 until inference has
            # spoken, because your own words are certain
```

Only `@ <type>` collapses — `+` and `~` just look. `@` chains; `+`/`~`
are the terminal steps (they bind tighter than `@`, so pipelines take
parens). A failed Thing is falsy and remembers why:

```python
+Thing("the number of legs on a spider")     # "the number of legs on a
                                             # spider" — your words, free
+(Thing("the number of legs on a spider") @ int)          # 8 — collapsing
                                                          # happens through
                                                          # typing
+(Thing("the number of legs on a spider") @ int @ 0.8)    # 8 — typed AND
                                                          # vouched for
~(thing @ int @ 0.8)              # the diagnosis: 0.67 tells you the cast
                                  # worked but fell short of the 0.8 bar
if car.price @ float @ 0.7:       # failure Things are falsy — gate whole
    ...                           # branches without unwrapping

# the type operand scales all the way up:
movie = Thing("the movie with the xenomorph")
movie @ {"title": str, "year": int, "cast": [str]}   # a JSON template —
                                  # enforced like returns=, keys land as
                                  # attributes on the child
movie @ MovieRecord               # any custom class: the imagination
                                  # supplies constructor kwargs, the REAL
                                  # constructor builds the instance
movie @ Film                      # a Thing subclass: re-classes the story,
                                  # free — no inference, code reattaches
                                  # (the scene-11 cast, in operator form)

Thing("a car") < Thing("a cat")   # False, confidence ~0.75 — an imagined
                                  # judgment; the stories decide what
                                  # "less" means here
+car.price < 20_000               # crossed the border first: plain, free,
                                  # deterministic int compare

price = Thing(19_990, confidence=0.4)   # lift your own doubt into Thing
+(price @ 0.5)                          # space ... None — too uncertain
+price                                  # 19990 — the value regardless
```

`==` and hashing stay in value space — containers and dedup depend on
them. No inferred value survives `@ 1` — only your own words do
(principle 3, as an operator).

## Scene 14 — the fixed point: stock Python, untouched

```python
class Point(Thing):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3, 4)
p.norm()                # 5.0 — byte-for-byte ordinary Python; every lookup
p.x                     # 3     resolves in strata 1–2, inference never runs,
                        #       no wrapper ever appears, nothing is billed
```

---

## Acceptance checklist

- [ ] A fully written subclass behaves identically to a plain `object`
      subclass (scene 14) — zero inference calls, zero wrappers.
- [ ] `Thing(*anything, **state)` instantiates directly: positional args
      join the story; keyword args become bare authoritative state
      (scene 1). Only `stateful` and `model` are reserved.
- [ ] First read of an undefined attribute collapses it; later reads agree
      (scene 1) until an event supersedes them (scenes 3–4, `del`).
- [ ] Written code and recorded state always win over inference (scene 3).
- [ ] Imagined plans are limited to the six verbs (scene 6): they can
      get, set, delete, call, define, and return — and can never generate
      or execute Python source.
- [ ] A real method reached from an imagined plan is executed, never
      simulated (scene 5, step 2).
- [ ] State written by an imagined plan carries confidence < 1 and may
      supersede only what imagination manages (its own writes, or slots the
      programmer opened with a `Thing` value); bare programmer state and
      written code are refused (scenes 5–6).
- [ ] Ordering comparisons on a Thing are imagined judgments returning a
      bool child with confidence; `==` and hashing stay in value space
      (scene 13).
- [ ] `+t` takes the value (or the story text when nothing has collapsed)
      and `~t` the probability (1.0 until inference has spoken) — both
      free. Only `t @ <type>` collapses, single-shot, as that
      type/schema/class; `t @ p` gates on confidence without inferring.
      `@` always returns a Thing: an unmet requirement drops the value but
      keeps the model's probability for diagnosis, and is falsy;
      `Thing(v, confidence=p)` lifts a programmer value into Thing space
      (scene 13).
- [ ] Bare value ⇔ p = 1.0 ⇔ provenance is code/state; child Thing ⇔ p < 1
      (scene 2). Inference never returns exactly 1.0.
- [ ] Confidence of a composite action = min over its imagined steps
      (scene 5).
- [ ] `Thing.require(t)` raises `Thing.LowConfidence` on any sub-threshold
      resolution (scene 7); step budgets raise `Thing.ContinuationLimit`
      (scene 8).
- [ ] `stateful=False` disables all appends by imagination — independent
      samples per read — while strata 1–2 behave normally (scene 9).
- [ ] The backend is injectable per instance or per class as a model name,
      URL, provider object, bare callable, or (future) embedded model file
      (scene 10); the provider protocol is a single method:
      complete(messages) -> text.
- [ ] `__getstate__()` dumps the full story and state as a JSON-able
      document; `blob @ Car` casts it back to life, reattaching written
      code; `pickle` works (scene 11).
- [ ] Every imagined read or call returns a child Thing carrying
      `.confidence` and its value (bool/str/arithmetic/iteration delegate to
      it); real methods of the value execute, never simulated; dict results
      land as bare attributes; further imagined calls chain on it (scene 12).
- [ ] A `returns=` schema is enforced: non-conforming returns are rejected
      and corrected within the step budget, so a delivered value always
      matches the template (scene 12).
- [ ] The public surface is exactly one name: `Thing` (principle 5).
- [ ] `obj.__story__` replays every event and answer in order;
      `obj.__source__` renders the object as Python-like source (scene 5).
