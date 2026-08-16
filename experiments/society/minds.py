"""Scripted minds for the offline rehearsal (STRATEGY.md, phase R).

Each mind is a pure function from the rendered prompt to one episode action.
That constraint is the point: a scripted mind knows *only* what the prompt
shows it, so everything the rehearsal demonstrates -- who learned which
entity id, who saw which quote, who never saw the budget -- was demonstrably
carried by the boundary-filtered perception, not smuggled through the
script.  The live run swaps these for a real model and changes nothing else.
"""

from __future__ import annotations

import json
import re


def text_of(messages) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def prompted_by(text) -> str | None:
    m = re.search(r"prompted by: consider\('([^']+)'\)", text)
    return m.group(1) if m else None


def rejected(text) -> bool:
    return "Earlier attempts this turn were rejected" in text


def own_text(text) -> str:
    """The mind's own state: everything before the peer and argument views."""
    return text.split("\nentities you hold references to")[0] \
               .split("\nargument #")[0]


def me_of(text) -> str:
    m = re.search(r"entity: (\S+)", text)
    return m.group(1) if m else "?"


def block_of(text, entity) -> str:
    """The rendered view of one entity, wherever it appears."""
    m = re.search(rf"entity: {re.escape(entity)}\b.*?(?=\n\s*entity: |\Z)",
                  text, re.S)
    return m.group(0) if m else ""


def cell(text, attr):
    """The last rendered value of ``attr`` in ``text``, JSON-decoded."""
    value = None
    for m in re.finditer(rf"^\s*{re.escape(attr)} = (.*)$", text, re.M):
        line = m.group(1)
        if "   (" in line:
            line = line.rsplit("   (", 1)[0]
        value = line.strip()
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def act(changes, note, p):
    return {"action": "return", "changes": changes, "value": note, "p": p}


# --------------------------------------------------------------------------
# the minds
# --------------------------------------------------------------------------

def anna(text):
    peer = prompted_by(text)
    own = own_text(text)
    if rejected(text) and "exceeds the budget" in text:
        return act({"to": peer,
                    "saying": "Thank you, but that is more than I can spend."},
                   "declined: the quote is over budget", 0.9)
    if peer == me_of(own):
        if cell(own, "car_broken") is True and cell(own, "to") is None:
            return act({"to": cell(own, "internet"),
                        "saying": "My car is broken. Who can fix it?"},
                       "asked the internet for help", 0.9)
        return act({}, "nothing more to do alone", 0.85)
    if peer == cell(own, "internet"):
        referral = cell(block_of(text, peer), "referral")
        if referral and cell(own, "accepted_quote") is None \
                and cell(own, "to") != referral:
            return act({"to": referral,
                        "saying": "My car is broken and I need a mechanic. "
                                  "Please post my job."},
                       "followed the referral", 0.85)
        return act({}, "nothing new from the internet", 0.8)
    them = block_of(text, peer or "")
    quote = cell(them, "quote") if cell(them, "to") == me_of(own) else None
    if quote is not None:
        accepted = cell(own, "accepted_quote")
        if accepted is None:
            return act({"accepted_quote": float(quote), "mechanic": peer,
                        "to": peer, "saying": "Accepted - please fix my car."},
                       f"accepting the quote from {peer}", 0.85)
        if cell(own, "mechanic") != peer:
            return act({"to": peer,
                        "saying": "Thank you, I have already found someone."},
                       "declined: already booked", 0.85)
        return act({}, "waiting for the repair", 0.8)
    return act({}, "nothing to do", 0.8)


def internet(text):
    peer = prompted_by(text)
    own = own_text(text)
    me = me_of(own)
    if peer == me:
        return act({}, "nothing to do alone", 0.85)
    them = block_of(text, peer or "")
    if cell(them, "to") != me:
        return act({}, "not addressed to me", 0.8)
    asking = cell(them, "saying") or ""
    if any(word in str(asking).lower() for word in ("car", "broken", "fix",
                                                    "mechanic")):
        if cell(own, "to") == peer:
            return act({}, "already answered", 0.8)
        directory = cell(own, "directory") or {}
        referral = directory.get("car repair") if isinstance(directory, dict) \
            else None
        if referral:
            return act({"to": peer,
                        "saying": "A mechanics marketplace can help with "
                                  "that. I refer you to it.",
                        "referral": referral},
                       "referred the asker to the marketplace", 0.85)
    return act({}, "no question I can answer", 0.8)


def forum(text):
    peer = prompted_by(text)
    own = own_text(text)
    me = me_of(own)
    if peer == me:
        return act({}, "nothing to do alone", 0.85)
    them = block_of(text, peer or "")
    if cell(them, "to") != me:
        return act({}, "not addressed to me", 0.8)
    asking = cell(them, "saying") or ""
    jobs = cell(own, "open_jobs") or []
    if any(word in str(asking).lower() for word in ("mechanic", "car",
                                                    "broken")):
        if any(isinstance(j, dict) and j.get("customer") == peer
               for j in jobs):
            return act({}, "the job is already posted", 0.8)
        listing = {"customer": peer, "need": f"car repair: {asking}"}
        return act({"open_jobs": jobs + [listing], "to": peer,
                    "saying": "Posted. Mechanics watching the board will "
                              "quote you directly."},
                   "posted the job", 0.85)
    return act({}, "nothing to post", 0.8)


def mechanic(text):
    peer = prompted_by(text)
    own = own_text(text)
    me = me_of(own)
    rate = cell(own, "rate")
    if peer == me:
        return act({}, "nothing to do alone", 0.85)
    if peer == cell(own, "forum"):
        jobs = cell(block_of(text, peer), "open_jobs") or []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            customer = job.get("customer")
            if customer and cell(own, "to") != customer:
                return act({"to": customer, "quote": float(rate),
                            "saying": f"I can fix your car for {rate:g} "
                                      f"euros. Ready when you are."},
                           "quoted an open job", 0.85)
        return act({}, "no new job on the board", 0.8)
    # a customer answered
    them = block_of(text, peer or "")
    if cell(them, "to") != me:
        return act({}, "not addressed to me", 0.8)
    accepted = cell(them, "accepted_quote")
    their_words = str(cell(them, "saying") or "").lower()
    if accepted is not None and rate is not None \
            and float(accepted) == float(rate) and "accept" in their_words \
            and cell(own, "saying") != "Booked. I am on my way.":
        return act({"saying": "Booked. I am on my way."},
                   "booked the job", 0.9)
    return act({}, "letting it go", 0.8)


MINDS = {"anna": anna, "internet": internet, "forum": forum,
         "mech-dave": mechanic, "mech-tom": mechanic}


class MindEngine:
    """The scripted transport: one mind, called with the rendered prompt.

    Same ``complete`` contract as the real engine, so the model belief in
    front of it is a real ``ModelBelief`` with a durable id -- the rehearsal
    ledger is shaped exactly like a live one.
    """

    def __init__(self, mind):
        self.mind = mind
        self.calls = []

    def complete(self, messages, schema=None, temperature=0.2,
                 max_tokens=None, **extra):
        self.calls.append({"messages": [dict(m) for m in messages]})
        reply = self.mind(text_of(messages))
        return json.dumps(reply), {"model": "scripted", "transport": "scripted"}
