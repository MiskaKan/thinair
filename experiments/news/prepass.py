"""The zero-call pre-pass.  Exhaust the free structure before spending anything.

Nothing here calls a model.  Everything here is classical: counts, shingles,
regex datelines, and two of thinair's deterministic validators used directly as
what they are --- pure functions.
"""

from __future__ import annotations

import re

from thinair.validators import CalendarFact, IsoCountry
from thinair.validators.grounding import normalize_text, numbers_in

SHINGLE = 5
DUPLICATE_AT = 0.5

#: Three classical duplicate signals, pre-registered together so that none is a
#: threshold tuned after seeing the answer.  They differ in mechanism: word
#: order, vocabulary, and the numbers alone.
SIGNALS = ("shingle5", "unigram", "fingerprint")
UNIGRAM_AT = 0.5

_DATELINE = re.compile(r"^([A-ZÁÉÍÓÚÑÜÖÅÄ' -]+),\s*([^—]+)—\s*(\d{4}-\d{2}-\d{2})\s*\(([A-Za-z]+)\)")


def _shingles(text, n=SHINGLE):
    words = normalize_text(text).split()
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def jaccard(a, b, n=SHINGLE):
    x, y = _shingles(a, n), _shingles(b, n)
    return len(x & y) / len(x | y) if (x | y) else 0.0


def _fingerprint(record):
    """Dateline date plus the multiset of numbers -- vocabulary-free."""
    dateline = record.get("dateline") or {}
    return (dateline.get("date"), tuple(sorted(record["numbers"])))


def run(items):
    """``{...}`` -- everything derivable from the corpus for zero calls."""
    calendar, country = CalendarFact(), IsoCountry()
    per_item, dateline_faults = {}, []

    for item in items:
        text = item["text"]
        m = _DATELINE.match(text)
        record = {
            "words": len(text.split()),
            "numbers": numbers_in(text),
            "dateline": None,
        }
        if m:
            city, place, date, weekday = (g.strip() for g in m.groups())
            claim = f"{date} is a {weekday}"
            verdict = calendar.judge(claim, None, None)
            recognized = country.judge(place, None, None)
            record["dateline"] = {
                "city": city, "place": place, "date": date, "weekday": weekday,
                "calendar_p": verdict[0], "calendar_why": verdict[1],
                "table_p": recognized[0], "table_why": recognized[1],
            }
            if verdict[0] < 1.0:
                dateline_faults.append((item["id"], verdict[1]))
        per_item[item["id"]] = record

    pairs, ranked = [], []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            shingle5 = jaccard(a["text"], b["text"])
            unigram = jaccard(a["text"], b["text"], n=1)
            same = (_fingerprint(per_item[a["id"]])
                    == _fingerprint(per_item[b["id"]]))
            fired = []
            if shingle5 >= DUPLICATE_AT:
                fired.append("shingle5")
            if unigram >= UNIGRAM_AT:
                fired.append("unigram")
            if same:
                fired.append("fingerprint")
            row = dict(pair=(a["id"], b["id"]), shingle5=round(shingle5, 4),
                       unigram=round(unigram, 4), fingerprint=same, fired=fired)
            ranked.append(row)
            if fired:
                pairs.append(row)
    ranked.sort(key=lambda r: -r["unigram"])

    return {
        "items": per_item,
        "near_duplicates": pairs,
        "pairs_ranked": ranked[:5],
        "dateline_faults": dateline_faults,
        "shingle": SHINGLE,
        "duplicate_at": DUPLICATE_AT,
        "unigram_at": UNIGRAM_AT,
        "calls": 0,
    }
