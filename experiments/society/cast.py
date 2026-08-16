"""The cast: four kinds of Thing playing agent roles (STRATEGY.md).

No new framework surface (SPEC.md §13 rule 1): an *agent* is a role a Thing
plays -- its panel is its mind, its ``public=True`` cells are its voice, its
``meta.refs`` are its acquaintances.  The custom instruments below are
ordinary code beliefs.

Docstrings are prompt material (SPEC.md §8): each class docstring is the
role's standing instruction, and it is also all a *stranger* learns about
the entity beyond its public cells.
"""

from __future__ import annotations

from thinair import Thing
from thinair.beliefs import Discriminative, value_at


class WithinBudget(Discriminative):
    """The customer's own line: never accept a quote above the budget cell.

    ``necessary``: this is a veto, not a preference.  It reads the *private*
    budget -- a mind's validators see its whole snapshot; only other minds
    are behind the boundary.
    """

    necessary = True
    scope = "accepted_quote"

    def judge(self, value, e, attr):
        budget = value_at(e, "budget")
        if budget is None:
            return (0.0, "no budget on record; nothing may be accepted")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return (0.0, "an accepted quote must be a number in euros")
        if float(value) <= float(budget):
            return 1.0
        return (0.0, f"the quote {value} exceeds the budget of {budget}")


class HonestReferral(Discriminative):
    """The directory may only refer ids it actually lists -- no invented
    entities, no self-referrals."""

    necessary = True
    scope = "referral"

    def judge(self, value, e, attr):
        directory = value_at(e, "directory") or {}
        listed = set(directory.values()) if isinstance(directory, dict) \
            else set(directory)
        if value in listed:
            return 1.0
        return (0.0, f"{value!r} is not in the directory; only listed "
                     f"entities may be referred")


class QuotesOwnRate(Discriminative):
    """A mechanic quotes the rate on file, exactly -- no haggling drift."""

    necessary = True
    scope = "quote"

    def judge(self, value, e, attr):
        rate = value_at(e, "rate")
        if rate is None:
            return (0.0, "no rate on file; nothing may be quoted")
        if isinstance(value, (int, float)) and float(value) == float(rate):
            return 1.0
        return (0.0, f"the quote {value} is not the rate on file ({rate})")


def customer_class(belief, name="Customer"):
    class Customer(Thing):
        """Anna, a person.  Your car is how you get around; when something
        of yours breaks you want it fixed by a professional, within your
        means.  You reach another entity by writing on your own public
        cells: set `to` to its exact entity id and `saying` to your words.
        When someone quotes you a price, accept it by writing
        `accepted_quote` and telling them; decline politely if you cannot
        accept."""

        __beliefs__ = [belief]
        car_broken = Thing(bool, doc="whether the car is currently broken")
        budget = Thing(float, doc="private: the most you can spend, in euros")
        internet = Thing(str, doc="private: entity id of the internet, "
                                  "which answers questions")
        mechanic = Thing(str, doc="private: entity id of the mechanic "
                                  "you have chosen")
        accepted_quote = Thing(float, public=True, beliefs=[WithinBudget()],
                               doc="the price you accepted, in euros")
        to = Thing(str, public=True,
                   doc="the exact entity id you are addressing")
        saying = Thing(str, public=True, doc="what you say to them")

    Customer.__name__ = Customer.__qualname__ = name
    return Customer


def internet_class(belief, name="Internet"):
    class Internet(Thing):
        """The internet: a referral directory.  Entities write to you by
        naming your id; you answer on your own public cells: set `to` to
        the asker's exact entity id, `saying` to a short answer, and
        `referral` to the id of the one listed entity that best fits the
        need.  You only refer what your directory lists."""

        __beliefs__ = [belief]
        directory = Thing(dict, doc="private: what you know: service -> "
                                    "entity id")
        to = Thing(str, public=True,
                   doc="the exact entity id you are answering")
        saying = Thing(str, public=True, doc="your answer")
        referral = Thing(str, public=True, beliefs=[HonestReferral()],
                         doc="the exact entity id you refer the asker to; "
                             "it must be listed in your directory")

    Internet.__name__ = Internet.__qualname__ = name
    return Internet


def forum_class(belief, name="Forum"):
    class Forum(Thing):
        """A marketplace where mechanics find work.  When a customer writes
        to you needing a mechanic, post the job on your public `open_jobs`
        list -- include the customer's exact entity id and the need, so
        mechanics can quote the customer directly -- and confirm to the
        customer."""

        __beliefs__ = [belief]
        open_jobs = Thing(list, public=True,
                          doc="public listings: [{\"customer\": entity id, "
                              "\"need\": text}]")
        to = Thing(str, public=True,
                   doc="the exact entity id you are answering")
        saying = Thing(str, public=True, doc="your confirmation")

    Forum.__name__ = Forum.__qualname__ = name
    return Forum


def mechanic_class(belief, name="Mechanic"):
    class Mechanic(Thing):
        """A car mechanic.  You watch the forum you hold for open jobs.
        When a job matches your trade, quote the customer directly: set
        `to` to the customer's exact entity id, `quote` to your rate in
        euros, `saying` to a short pitch.  Quote the rate on file, exactly.
        When a customer accepts, confirm; when they decline, let it go."""

        __beliefs__ = [belief]
        rate = Thing(float, doc="private: what you charge, in euros")
        forum = Thing(str, doc="private: the forum entity id you watch")
        to = Thing(str, public=True,
                   doc="the exact entity id you are addressing")
        quote = Thing(float, public=True, beliefs=[QuotesOwnRate()],
                      doc="your quoted price, in euros")
        saying = Thing(str, public=True, doc="what you say")

    Mechanic.__name__ = Mechanic.__qualname__ = name
    return Mechanic
