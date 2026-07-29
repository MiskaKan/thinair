"""Run the SPEC.md scenes against the real backend."""

import json
import pickle

from thinair import Thing


print("== Scene 12 — stock Python, zero inference ==")

class Point(Thing):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3, 4)
print(f"norm={p.norm()}  x={p.x}  type(x)={type(p.x).__name__}")


print("\n== Scene 1 — pure invention ==")
car = Thing("A Toyota car from the 1990s with a broken engine")
print(f"color={car.color!r}  confidence={car.color.confidence}")
collapsed = car.color  # second access: the stored Approx, not a proxy
print(f"color again={collapsed!r}  type={type(collapsed).__name__}"
      f"  base={type(collapsed).__mro__[1].__name__}  (collapsed, consistent)")
year = car.year
print(f"year={year}  confidence={year.confidence}")


print("\n== Scene 4 — imagined mutation ==")
d1 = car.can_drive()
print(f"can_drive() -> {bool(d1)}  confidence={d1.confidence}")
car.repair_engine()
d2 = car.can_drive()
print(f"repair_engine(); can_drive() -> {bool(d2)}  confidence={d2.confidence}")


print("\n== Scene 5 — imagined method drives real code ==")

class Boat(Thing):
    """A small motorboat."""

    def __init__(self, description):
        super().__init__(description)
        self.fuel_litres = 0.0

    def refuel(self, litres):
        """Add fuel to the tank; returns the new fuel level in litres."""
        self.fuel_litres += litres
        return self.fuel_litres

boat = Boat("a dinghy with an outboard motor, tank empty")
result = boat.prepare_for_trip()
print(f"prepare_for_trip() -> {result!r}  confidence={getattr(result, 'confidence', 1.0)}")
print(f"fuel_litres={boat.fuel_litres}  type={type(boat.fuel_litres).__name__}")
print("story:")
for event in boat.__story__:
    print("  ", json.dumps(event, default=repr))


print("\n== Scene 9 — stateless: independent samples ==")
die = Thing("a fair six-sided die, freshly rolled", stateful=False)
rolls = [die.face._resolve() for _ in range(3)]
print("rolls:", [(repr(r), r.confidence) for r in rolls])


print("\n== Scene 7 — confidence guard ==")
try:
    with Thing.require(0.9):
        vin = car.vin_number
        print(f"vin={vin!r}")
except Thing.LowConfidence as error:
    print(f"vin_number raised LowConfidence: {error}")


print("\n== Scene 11 — freeze / thaw ==")
blob = car.freeze()
clone = Thing.thaw(json.loads(json.dumps(blob, default=repr)))
print(f"thawed color={clone.color!r}  (survived the freeze)")
d3 = clone.can_drive()
print(f"thawed can_drive() -> {bool(d3)}  (repair survived)")
roundtrip = pickle.loads(pickle.dumps(car))
print(f"pickle roundtrip color={roundtrip.color!r}")
