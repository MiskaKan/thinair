"""Versioned (de)serialization of Thing and Ledger (SPEC.md §11).

A Thing's state is not a dict of values -- it has none.  It is an
entity id and that entity's slice of the ledger, which is why saving one is
saving *opinions*, authorship and all.

    blob = invoice.__getstate__()
    revived = blob @ Invoice            # metaclass __rmatmul__

``pickle`` rides the same mechanism.  Ledgers save and load whole
(``ledger.dump(path)`` / ``Ledger.load(path)``) -- the ledger is the memory;
losing it forgets everything ever said.
"""

from __future__ import annotations

from .ledger import Opinion

__all__ = ["VERSION", "dump_thing", "revive", "load_into", "PersistError"]

#: bumped when the blob shape changes; a reader refuses what it cannot read.
VERSION = 1


class PersistError(ValueError):
    """A blob this version cannot read."""


def dump_thing(thing) -> dict:
    """``{"thinair": 1, ...}`` -- class name, entity id, and the ledger slice."""
    entity = thing.__entity__
    ledger = thing.__ledger__
    owned = [o for o in ledger if o.entity == entity or o.entity.startswith(entity + "#")
             or o.entity.startswith(entity + ".")]
    return {
        "thinair": VERSION,
        "class": type(thing).__name__,
        "module": type(thing).__module__,
        "entity": entity,
        "opinions": [o.to_json() for o in owned],
        # Which belief resolved each cell -- not the value.  Restoring by
        # "latest opinion at the cell" would hand the resolution to whichever
        # validator spoke last, and a validator's 1.0 means "I see nothing
        # wrong", never "this is certain".
        "resolved": {
            attr: opinion.belief
            for (ent, attr), opinion in getattr(thing.__root__, "__resolved__", {}).items()
            if ent == entity},
    }


def _check(blob) -> dict:
    if not isinstance(blob, dict) or "thinair" not in blob:
        raise PersistError("not a thinair blob")
    version = blob["thinair"]
    if version != VERSION:
        raise PersistError(
            f"blob version {version} cannot be read by thinair persist v{VERSION}")
    for key in ("class", "entity", "opinions"):
        if key not in blob:
            raise PersistError(f"blob is missing {key!r}")
    return blob


def revive(blob, cls):
    """``blob @ MyClass`` -- restore an entity into a class of your choosing.

    The class is yours to pick, exactly as with a recast: opinions and
    frozen state carry over, and the class you name governs all future
    consultation.  Reviving into a *different* class is therefore a recast
    and a load in one step, which is the honest reading of ``@``.
    """
    _check(blob)
    thing = cls.__new__(cls)
    set = object.__setattr__
    set(thing, "__entity__", blob["entity"])
    set(thing, "__root__", thing)
    set(thing, "__resolved__", {})
    set(thing, "__coerced__", {})
    # Reviving into a *different* class is a recast, and a recast invalidates
    # cached resolutions: the new class's beliefs and contracts govern
    # all future consultation, so the old class's answers are not answers to
    # the new class's questions.  Frozen state is not a cached resolution and
    # carries over either way.
    load_into(thing, blob, resolutions=(cls.__name__ == blob["class"]))
    return thing


def load_into(thing, blob, ledger=None, resolutions=True) -> None:
    """Replay a blob's opinions into a ledger and re-establish resolutions."""
    _check(blob)
    set = object.__setattr__
    if not hasattr(thing, "__entity__"):
        set(thing, "__entity__", blob["entity"])
        set(thing, "__root__", thing)
        set(thing, "__resolved__", {})
        set(thing, "__coerced__", {})

    if ledger is None:
        ledger = getattr(thing, "__ledger__", None)
    if ledger is None:
        from .thing import _ledgers
        from .ledger import default_ledger

        ledger = _ledgers[-1] if _ledgers else default_ledger()
    set(thing, "__ledger__", ledger)

    known = {(o.belief, o.entity, o.attr, o.t) for o in ledger}
    for record in blob["opinions"]:
        opinion = Opinion.from_json(record)
        if (opinion.belief, opinion.entity, opinion.attr, opinion.t) in known:
            continue                                 # already remembered
        ledger.add(opinion)

    # A standing resolution is not a fact, it is a *cached selection* -- so
    # it is restored by re-selecting the latest opinion of the belief that
    # resolved it, never by trusting a stored value.
    entity = blob["entity"]
    resolved = {}
    for attr, belief in ((blob.get("resolved") or {}) if resolutions else {}).items():
        latest = ledger.latest(entity, attr, belief=belief)
        if latest is not None:
            resolved[(entity, attr)] = latest
    thing.__root__.__resolved__.update(resolved)


# --------------------------------------------------------------------------
# pickle rides the same mechanism
# --------------------------------------------------------------------------

def __reduce_thing__(thing):
    return (_from_blob, (type(thing), thing.__getstate__()))


def _from_blob(cls, blob):
    return revive(blob, cls)
