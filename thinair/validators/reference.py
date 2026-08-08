"""Family D — pinned reference tables.

Deterministic *given a pinned snapshot*. Every table in this module is
vendored, tiny, and frozen at a stated revision; nothing here ever performs a
live lookup, because a belief whose answer depends on the network is not a
pure function of the snapshot (invariant 3).

Default ``necessary = False``: a name missing from a hand-vendored table is
weak evidence of a wrong answer, so these beliefs measure rather than veto.
``CalendarFact`` is the exception the plan allows — date-to-weekday is pure
computation, not a table, so it may be given veto rights at the attachment
site.
"""

from __future__ import annotations

import datetime as _dt
import re

from ..beliefs import Discriminative

__all__ = [
    "IsoCountry",
    "IsoCurrency",
    "Timezone",
    "CalendarFact",
    "COUNTRIES",
    "CURRENCIES",
    "COUNTRIES_REVISION",
    "CURRENCIES_REVISION",
]

# --------------------------------------------------------------------------
# Vendored tables. Pinned, not fetched.
# --------------------------------------------------------------------------

COUNTRIES_REVISION = "iso-3166-1:2020"
CURRENCIES_REVISION = "iso-4217:2015-amend-169"

#: alpha-2 -> (alpha-3, numeric, common name).  A deliberately partial table:
#: the point is a pinned snapshot, not completeness, and family D never vetoes
#: by default, so absence costs nothing but a low p.
COUNTRIES = {
    "AT": ("AUT", "040", "Austria"),
    "AU": ("AUS", "036", "Australia"),
    "BE": ("BEL", "056", "Belgium"),
    "BR": ("BRA", "076", "Brazil"),
    "CA": ("CAN", "124", "Canada"),
    "CH": ("CHE", "756", "Switzerland"),
    "CN": ("CHN", "156", "China"),
    "CZ": ("CZE", "203", "Czechia"),
    "DE": ("DEU", "276", "Germany"),
    "DK": ("DNK", "208", "Denmark"),
    "EE": ("EST", "233", "Estonia"),
    "ES": ("ESP", "724", "Spain"),
    "FI": ("FIN", "246", "Finland"),
    "FR": ("FRA", "250", "France"),
    "GB": ("GBR", "826", "United Kingdom"),
    "IE": ("IRL", "372", "Ireland"),
    "IN": ("IND", "356", "India"),
    "IS": ("ISL", "352", "Iceland"),
    "IT": ("ITA", "380", "Italy"),
    "JP": ("JPN", "392", "Japan"),
    "LT": ("LTU", "440", "Lithuania"),
    "LV": ("LVA", "428", "Latvia"),
    "MX": ("MEX", "484", "Mexico"),
    "NL": ("NLD", "528", "Netherlands"),
    "NO": ("NOR", "578", "Norway"),
    "NZ": ("NZL", "554", "New Zealand"),
    "PL": ("POL", "616", "Poland"),
    "PT": ("PRT", "620", "Portugal"),
    "RU": ("RUS", "643", "Russian Federation"),
    "SE": ("SWE", "752", "Sweden"),
    "SG": ("SGP", "702", "Singapore"),
    "TR": ("TUR", "792", "Turkey"),
    "UA": ("UKR", "804", "Ukraine"),
    "US": ("USA", "840", "United States"),
    "ZA": ("ZAF", "710", "South Africa"),
}

#: alpha-3 -> (numeric, minor units, common name).
CURRENCIES = {
    "AUD": ("036", 2, "Australian Dollar"),
    "BRL": ("986", 2, "Brazilian Real"),
    "CAD": ("124", 2, "Canadian Dollar"),
    "CHF": ("756", 2, "Swiss Franc"),
    "CNY": ("156", 2, "Yuan Renminbi"),
    "CZK": ("203", 2, "Czech Koruna"),
    "DKK": ("208", 2, "Danish Krone"),
    "EUR": ("978", 2, "Euro"),
    "GBP": ("826", 2, "Pound Sterling"),
    "INR": ("356", 2, "Indian Rupee"),
    "ISK": ("352", 0, "Iceland Krona"),
    "JPY": ("392", 0, "Yen"),
    "MXN": ("484", 2, "Mexican Peso"),
    "NOK": ("578", 2, "Norwegian Krone"),
    "NZD": ("554", 2, "New Zealand Dollar"),
    "PLN": ("985", 2, "Zloty"),
    "RUB": ("643", 2, "Russian Ruble"),
    "SEK": ("752", 2, "Swedish Krona"),
    "SGD": ("702", 2, "Singapore Dollar"),
    "TRY": ("949", 2, "Turkish Lira"),
    "UAH": ("980", 2, "Hryvnia"),
    "USD": ("840", 2, "US Dollar"),
    "ZAR": ("710", 2, "Rand"),
}

_BY_COUNTRY_NAME = {name.casefold(): code for code, (_, _, name) in COUNTRIES.items()}
_BY_COUNTRY_A3 = {a3: code for code, (a3, _, _) in COUNTRIES.items()}
_BY_CURRENCY_NAME = {name.casefold(): code for code, (_, _, name) in CURRENCIES.items()}


class _Table(Discriminative):
    """Shared plumbing: judge each string in the candidate against a table."""

    necessary = False
    revision = ""
    #: how a hit and a miss read in the opinion's reason; families D members
    #: differ enough in kind that one phrasing would be misleading.
    hit_phrase = "recognized in"
    miss_phrase = "unknown in"

    def _lookup(self, token):  # -> (bool, str) : recognized, canonical/why
        raise NotImplementedError

    def _tokens(self, value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [v for v in value.values() if isinstance(v, str)]
        if isinstance(value, (list, tuple, set)):
            out = []
            for item in value:
                out.extend(self._tokens(item))
            return out
        return []

    def judge(self, value, e, attr):
        tokens = self._tokens(value)
        if not tokens:
            return None
        hits, misses = [], []
        for token in tokens:
            ok, note = self._lookup(token)
            (hits if ok else misses).append(note)
        p = len(hits) / len(tokens)
        if misses:
            return p, f"{self.miss_phrase} {self.revision}: {'; '.join(misses[:3])}"
        return p, f"{self.hit_phrase} {self.revision}: {'; '.join(hits[:3])}"


class IsoCountry(_Table):
    """The candidate names a country in the pinned ISO 3166-1 snapshot.

    Accepts alpha-2, alpha-3, numeric, or the common English name — the point
    is recognition, not a canonical spelling.
    """

    revision = COUNTRIES_REVISION

    def _lookup(self, token):
        raw = token.strip()
        key = raw.upper()
        if key in COUNTRIES:
            return True, key
        if key in _BY_COUNTRY_A3:
            return True, _BY_COUNTRY_A3[key]
        folded = raw.casefold()
        if folded in _BY_COUNTRY_NAME:
            return True, _BY_COUNTRY_NAME[folded]
        for code, (_, numeric, _name) in COUNTRIES.items():
            if raw == numeric:
                return True, code
        return False, raw


class IsoCurrency(_Table):
    """The candidate names a currency in the pinned ISO 4217 snapshot."""

    revision = CURRENCIES_REVISION

    def _lookup(self, token):
        raw = token.strip()
        key = raw.upper()
        if key in CURRENCIES:
            return True, key
        folded = raw.casefold()
        if folded in _BY_CURRENCY_NAME:
            return True, _BY_CURRENCY_NAME[folded]
        for code, (numeric, _minor, _name) in CURRENCIES.items():
            if raw == numeric:
                return True, code
        return False, raw


class Timezone(_Table):
    """The candidate names an IANA zone the running interpreter knows.

    The table is ``zoneinfo``'s, i.e. the platform tzdata — pinned by the
    interpreter rather than by this file, which is the closest a timezone
    check gets to a vendored snapshot without shipping tzdata itself.
    """

    revision = "iana/zoneinfo"

    def _lookup(self, token):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        raw = token.strip()
        try:
            ZoneInfo(raw)
        except (ZoneInfoNotFoundError, ValueError, KeyError, ModuleNotFoundError):
            return False, raw
        return True, raw


_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


class CalendarFact(_Table):
    """Date arithmetic asserted by the candidate actually holds.

    Pure computation, not a table — so this one may reasonably be attached
    ``necessary=True``.  Understands three shapes:

    * ``{"date": "2024-02-29", "weekday": "Thursday"}`` — mapping form,
    * ``"2024-02-29 is a Thursday"`` — prose form,
    * ``"2024-02-30"`` — a bare date, checked for existence.
    """

    revision = "proleptic Gregorian calendar"
    hit_phrase = "holds in the"
    miss_phrase = "contradicted by the"

    def _tokens(self, value):  # keep dicts intact; this belief reads keys
        if isinstance(value, (str, dict)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            out = []
            for item in value:
                out.extend(self._tokens(item))
            return out
        return []

    @staticmethod
    def _parse_date(text):
        m = _DATE_RE.search(text or "")
        if not m:
            return None, "no ISO date found"
        y, mo, d = (int(g) for g in m.groups())
        try:
            return _dt.date(y, mo, d), None
        except ValueError as exc:
            return None, f"{m.group(0)} is not a real date ({exc})"

    @staticmethod
    def _weekday_in(text):
        low = (text or "").casefold()
        for name in _WEEKDAYS:
            if name in low:
                return name
        return None

    def _lookup(self, token):
        if isinstance(token, dict):
            date_text = str(token.get("date", ""))
            claimed = token.get("weekday")
            claimed = str(claimed).casefold().strip() if claimed else None
        else:
            date_text = str(token)
            claimed = self._weekday_in(date_text)

        date, why = self._parse_date(date_text)
        if date is None:
            return False, why
        actual = _WEEKDAYS[date.weekday()]
        if claimed is None:
            return True, f"{date.isoformat()} exists"
        if claimed != actual:
            return False, f"{date.isoformat()} is a {actual.title()}, not {claimed.title()}"
        return True, f"{date.isoformat()} is a {actual.title()}"
