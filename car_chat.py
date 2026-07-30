"""Talk to a car.

There is no chatbot framework in here, and `chat` is not a method anyone
wrote: it is imagined at call time, and the conversation is just the
object's story growing turn by turn. Written code (`horn`, `wheels`)
stays certain and free; everything else comes out of thin air.

    python car_chat.py
"""

from thinair import Thing


class Car(Thing):
    """A road vehicle."""

    wheels = 4                  # bare and written: the model may never touch it

    def __init__(self, description):
        super().__init__(description)
        self.mood = Thing("unknown so far")   # a Thing slot: an open field
                                              # the imagination MAY manage

    def horn(self):
        """Sound the horn."""
        return "beep"


car = Car(
    "a rusty 1990 Toyota Hilux: the engine coughs, the radio is stuck on a "
    "Finnish schlager station, and it has seen things. It answers as itself, "
    "in first person, a little tired but fond of its owner — and it keeps "
    "its `mood` state up to date as the conversation moves it."
)

print("You are standing next to a rusty 1990 Toyota Hilux. Say something.")
print("(empty line quits; the car remembers the whole conversation)\n")

while True:
    try:
        line = input("> ").strip()
    except EOFError:
        break
    if not line:
        break
    try:
        print(car.chat(line))
    except Thing.ContinuationLimit:
        print("(the engine coughs; the car lost its train of thought — ask again)")
    print()

print(f"(you pat the hood and leave; the car's mood right now: {+car.mood})")
