"""Family D — pinned reference tables.

Two properties matter more than any single row: the tables are *vendored*, so
a lookup is deterministic and offline, and the family does not veto by
default, because absence from a hand-kept table is weak evidence.
"""

from __future__ import annotations

import pytest

from thinair.validators import CalendarFact, IsoCountry, IsoCurrency, Timezone
from thinair.validators.reference import (
    COUNTRIES,
    COUNTRIES_REVISION,
    CURRENCIES,
    CURRENCIES_REVISION,
)

from fakes import FakeSnapshot, head


def p_of(belief, value):
    got = belief(head(value), "x")
    return None if got is None else ~got


# --------------------------------------------------------------------------
# the tables themselves
# --------------------------------------------------------------------------

def test_the_tables_are_pinned_to_a_stated_revision():
    assert COUNTRIES_REVISION and CURRENCIES_REVISION


def test_the_tables_are_internally_consistent():
    assert all(len(code) == 2 and code.isupper() for code in COUNTRIES)
    assert all(len(a3) == 3 and len(num) == 3 for a3, num, _ in COUNTRIES.values())
    assert all(len(code) == 3 and code.isupper() for code in CURRENCIES)
    assert len({a3 for a3, _, _ in COUNTRIES.values()}) == len(COUNTRIES)


def test_a_lookup_never_reaches_the_network(monkeypatch):
    """Invariant 7, enforced where it would be tempting to break it."""
    import socket

    def forbidden(*a, **k):                       # pragma: no cover
        raise AssertionError("family D attempted a live lookup")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    assert p_of(IsoCountry(), "FI") == 1.0
    assert p_of(IsoCurrency(), "EUR") == 1.0
    assert p_of(Timezone(), "Europe/Helsinki") == 1.0
    assert p_of(CalendarFact(), "2024-02-29") == 1.0


# --------------------------------------------------------------------------
# IsoCountry / IsoCurrency
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("FI", 1.0), ("fi", 1.0), ("FIN", 1.0), ("246", 1.0), ("Finland", 1.0),
    ("finland", 1.0), (" FI ", 1.0),
    ("XX", 0.0), ("Atlantis", 0.0), ("", 0.0),
    (42, None),                                    # nothing string-shaped
])
def test_iso_country(value, expected):
    assert p_of(IsoCountry(), value) == expected


@pytest.mark.parametrize("value,expected", [
    ("EUR", 1.0), ("eur", 1.0), ("978", 1.0), ("Euro", 1.0),
    ("XQZ", 0.0), ("Galleons", 0.0),
])
def test_iso_currency(value, expected):
    assert p_of(IsoCurrency(), value) == expected


def test_a_list_of_codes_is_graded_not_all_or_nothing():
    assert p_of(IsoCurrency(), ["EUR", "USD"]) == 1.0
    assert p_of(IsoCurrency(), ["EUR", "XQZ"]) == 0.5
    assert p_of(IsoCurrency(), ["XQZ", "XYZ"]) == 0.0


def test_the_reason_names_the_revision_and_the_offender():
    got = IsoCountry()(head("Atlantis"), "country")
    assert COUNTRIES_REVISION in got.meta["reason"]
    assert "Atlantis" in got.meta["reason"]


def test_a_dict_candidate_is_searched_for_strings():
    assert p_of(IsoCountry(), {"code": "FI", "amount": 12}) == 1.0


# --------------------------------------------------------------------------
# Timezone
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("Europe/Helsinki", 1.0),
    ("UTC", 1.0),
    ("Middle-earth/Shire", 0.0),
    ("EEST", 0.0),                                 # an abbreviation is not a zone
])
def test_timezone(value, expected):
    assert p_of(Timezone(), value) == expected


# --------------------------------------------------------------------------
# CalendarFact — pure computation, so it may veto
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ({"date": "2024-02-29", "weekday": "Thursday"}, 1.0),
    ({"date": "2024-02-29", "weekday": "thursday"}, 1.0),
    ({"date": "2024-02-29", "weekday": "Friday"}, 0.0),
    ({"date": "2023-02-29", "weekday": "Wednesday"}, 0.0),     # not a leap year
    ("2024-02-29 was a Thursday", 1.0),
    ("2024-02-29 was a Friday", 0.0),
    ("2024-02-29", 1.0),
    ("2024-02-30", 0.0),
    ("2023-02-29", 0.0),
    ("no date here", 0.0),
])
def test_calendar_fact(value, expected):
    assert p_of(CalendarFact(), value) == expected


def test_calendar_fact_explains_itself():
    got = CalendarFact()(head({"date": "2024-02-29", "weekday": "Friday"}), "d")
    assert "Thursday" in got.meta["reason"] and "Friday" in got.meta["reason"]


def test_calendar_fact_may_be_given_veto_rights():
    """Date-to-weekday is computation, not a table, so this is honest."""
    assert CalendarFact().necessary is False
    assert CalendarFact(necessary=True).necessary is True


# --------------------------------------------------------------------------
# family behavior
# --------------------------------------------------------------------------

FAMILY_D = [IsoCountry(), IsoCurrency(), Timezone(), CalendarFact()]


@pytest.mark.parametrize("belief", FAMILY_D, ids=lambda b: b.id)
def test_family_d_does_not_veto_by_default(belief):
    assert belief.necessary is False


@pytest.mark.parametrize("belief", FAMILY_D, ids=lambda b: b.id)
def test_family_d_is_silent_without_a_candidate(belief):
    assert belief(FakeSnapshot(beliefs=[lambda e, a: None]), "x") is None


@pytest.mark.parametrize("belief", FAMILY_D, ids=lambda b: b.id)
def test_family_d_is_deterministic(belief):
    e = head("FI")
    first = belief(e, "x")
    assert all(belief(e, "x") == first for _ in range(3))
