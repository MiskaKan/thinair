"""thinair.evaluate -- settlement over the record.

thinair measures; this module settles.  A large language model is a
measurement instrument, not an oracle: it projects things onto axes you name,
and every reading is an opinion ``(value, p)``, never a fact.  Confidence is
not assumed, it is *earned* -- by laws that could have failed and held, by
dissimilar paths that could have disagreed and agreed, and by concordance
with the frozen record, which is the only place this space touches reality.

The four pillars, one sentence each:

  I    Anything is measurable on any nameable axis; a reading is an opinion,
       and no set of readings exhausts the thing.
  II   Uncertainty enters only at the readings -- downstream computation is
       exact -- but selection and differencing amplify it, and repetition
       shrinks variance, never bias.
  III  Confidence is unbuilt coherence: agreement is evidence exactly in
       proportion to how likely disagreement was.
  IV   Readings persist and structure becomes the instrument for the next
       question; freezing is stronger, deliberate, and code-or-human-only.

**The model is a column factory.**  It is consulted only on axes no
deterministic reader exists for -- turning liquid data into columns.
Everything downstream of a reading is classical, exact, and free: code sorts
numbers, computes correlations, clusters embeddings, aggregates ranks.  If a
regex can read the field, a regex is the belief.

The loop this module makes cheap:

  1. propose a small measurement strategy (GROUNDING.md, next to this file,
     carries the full theory; its Part 2 is the strategy format);
  2. run it as a thinair program -- Things, contracts, panels -- leaving a
     complete ledger;
  3. settle the ledger here: what was reliable, what discriminated, what
     concorded with ground, what the stated p was worth;
  4. let the grades pick the next strategy.

Everything below :func:`evaluate` is classical math over a
:class:`~thinair.ledger.Ledger` -- this module never talks to a model, and an
import-graph test holds it to that.  Only the :func:`evaluate` driver spends
calls, and only through beliefs the strategist already declared.

Deferred, and visibly so: ``earned`` / ``dissimilarity`` -- scoring agreement
by mechanism-and-exposure dissimilarity -- wait until the accumulated ledgers
span the moves (SPEC.md §14; GROUNDING.md Part 3).  Model opinions carry an
``exposure`` fingerprint in meta precisely so that math is computable, later,
from ledgers recorded now.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from .beliefs import lookup
from .ledger import Ledger, values_equal
from .policy import Disagreement, LowConfidence, Unresolvable

__all__ = [
    "evaluate",
    "reading", "verdicts", "coverage",
    "agree", "adjacent", "kappa",
    "ranks", "median", "spearman", "mannwhitney", "wilson", "n_eff",
    "bradley_terry",
    "reliability", "drift", "discrimination",
    "grounded", "concordance", "calibration", "separation", "tiers",
    "graph", "lineage", "invalidated",
    "LICENSED", "GRADES",
]


# --------------------------------------------------------------------------
# grounding as data
# --------------------------------------------------------------------------

#: Scale type -> the statistics licensed on readings of that type.  Each level
#: inherits everything above it.  This table is why a weighted sum of ordinal
#: values, a mean of ranks, or a Pearson on Likert scores never appears here:
#: compute the licensed ones from this module, compute the higher-scale ones
#: in plain code only when the axis's declared type licenses them.
LICENSED = {
    "nominal": ("counts", "mode", "agreement rate", "kappa"),
    "ordinal": ("median", "quantiles", "ranks", "spearman", "mannwhitney"),
    "interval": ("mean", "standard deviation", "differences", "pearson"),
    "ratio": ("ratios", "sums", "geometric mean", "coefficient of variation"),
}

#: The grading rule as data: every deliverable states where each conclusion
#: sits.  The scale runs from vibes to calibrated; nothing on it is a fact.
GRADES = (
    ("vibes", "no law checked, a single path, no ground"),
    ("claim", "laws checked or paths agreed -- but not both, or the paths "
              "share mechanism and exposure"),
    ("finding", "substantive laws held AND genuinely dissimilar paths agreed; "
                "no frozen ground reachable"),
    ("calibrated", "a finding, from an instrument whose concordance with the "
                   "frozen record is itself on record"),
)


# --------------------------------------------------------------------------
# the driver: run every belief a Thing declares
# --------------------------------------------------------------------------

def evaluate(thing: Any, order: Iterable[str] | None = None) -> dict:
    """Resolve every declared cell of ``thing`` and profile the result.

    Plain attribute access runs the full read pipeline -- panel, vetoes,
    rounds -- and never an episode, so the sweep measures without acting.
    ``order`` chooses the reading order, and order *is* context: each
    resolved cell enters the snapshots of the cells read after it, which is
    how a reliability twin gets its reshuffled context
    (``evaluate(twin, order=reversed(AXES))``).

    A cell that cannot resolve is a **coverage gap, not a clean negative**:
    it lands in the profile as ``status="gap"`` with the recorded reason,
    and the sweep continues.
    """
    contracts = type(thing).__contracts__
    names = list(order) if order is not None else list(contracts)
    unknown = [n for n in names if n not in contracts]
    if unknown:
        raise ValueError(f"not declared on {type(thing).__name__}: {unknown}")
    profile: dict[str, dict] = {}
    for attr in names:
        try:
            cell = getattr(thing, attr)
        except (Unresolvable, LowConfidence, Disagreement) as exc:
            profile[attr] = {"status": "gap", "value": None, "p": 0.0,
                             "reason": f"{type(exc).__name__}: {exc}"}
            continue
        opinion = cell.__opinion__
        if opinion is None:
            profile[attr] = {"status": "gap", "value": None, "p": 0.0,
                             "reason": "no resolving opinion"}
            continue
        profile[attr] = {
            "status": "frozen" if opinion.frozen else "resolved",
            "value": opinion.value, "p": opinion.p, "belief": opinion.belief,
        }
    return profile


# --------------------------------------------------------------------------
# reading the record
# --------------------------------------------------------------------------

def _generative(meta: dict) -> bool:
    """Was this opinion produced by an instrument, rather than a judge?"""
    return "model" in meta or "code" in meta


def reading(ledger: Ledger, entity: str, attr: str, belief: str | None = None,
            strict: bool = True) -> dict | None:
    """The instrument's last candidate at a cell, and whether it was vetoed.

    Veto detection needs each judge's ``necessary`` flag and ``veto_line``.
    They come from the constructed belief (the registry) or, failing that,
    from the ledger's stored belief descriptions (``belief_row``, which a
    store-backed ledger provides) -- so a fresh process can settle a record
    without reconstructing the strategy's classes.  When neither source
    knows the judge, ``strict`` makes the failure loud: an unknown judge
    raises instead of silently reading every veto as an ordinary low score,
    which is a bug this module exists to make unrepeatable.

    ``belief=None`` takes the cell's most recent instrument (a model or code
    opinion); frozen authority is not a reading -- ask
    ``ledger.latest_frozen`` for that.
    """
    everything = ledger.opinions(entity=entity, attr=attr)
    if belief is None:
        stated = [o for o in everything if not o.frozen and _generative(o.meta or {})]
        if not stated:
            return None
        belief = stated[-1].belief
    opinions = [o for o in everything if o.belief == belief]
    if not opinions:
        return None
    last = opinions[-1]
    vetoes = []
    for other in everything:
        if other.belief == belief or other.frozen:
            continue
        if _generative(other.meta or {}):
            continue                      # another instrument, not a judge
        if not values_equal(other.value, last.value):
            continue                      # a verdict on some earlier candidate
        judge = lookup(other.belief)
        if judge is not None:
            necessary = getattr(judge, "necessary", False)
            line = getattr(judge, "veto_line", 0.5)
        else:
            row = getattr(ledger, "belief_row", lambda _id: None)(other.belief)
            if row is None:
                if strict:
                    raise LookupError(
                        f"judge {other.belief!r} is neither registered nor "
                        f"described in the ledger; construct the strategy's "
                        f"classes, use a store-backed ledger, or pass "
                        f"strict=False")
                continue
            necessary, line = row["necessary"], row["veto_line"]
        if not necessary:
            continue
        if other.p < line:
            vetoes.append((other.belief, (other.meta or {}).get("reason")))
    return dict(value=last.value, p=last.p, rounds=len(opinions),
                vetoed=bool(vetoes), vetoes=vetoes, meta=last.meta or {})


def verdicts(ledger: Ledger, entity: str, attr: str,
             exclude: Iterable[str] = ()) -> dict:
    """Every judge's opinions at a cell, keyed by judge id.

    ``exclude`` names the instruments under test, so their proposals are not
    mistaken for verdicts about them.
    """
    skip = set(exclude)
    out: dict[str, list] = {}
    for o in ledger.opinions(entity=entity, attr=attr):
        if o.belief in skip or o.frozen:
            continue
        out.setdefault(o.belief, []).append(
            (o.value, o.p, (o.meta or {}).get("reason")))
    return out


def coverage(ledger: Ledger, entities: Iterable[str], axes: Iterable[str],
             belief: str | None = None, strict: bool = True) -> dict:
    """The census laws cannot take for themselves: what never resolved.

    A law can check only a reading that exists.  Cells whose last candidate
    was vetoed and cells never read at all are different absences -- a
    refusal is a judgement, a dead cell is the absence of one -- and both
    are counted, never silently dropped from denominators.
    """
    entities = list(entities)
    out: dict[str, dict] = {}
    for axis in axes:
        resolved, vetoed, absent = [], [], []
        for entity in entities:
            got = reading(ledger, entity, axis, belief=belief, strict=strict)
            if got is None:
                absent.append(entity)
            elif got["vetoed"]:
                vetoed.append((entity, [r for _b, r in got["vetoes"]]))
            else:
                resolved.append(entity)
        out[axis] = dict(n=len(entities), resolved=len(resolved),
                         vetoed=vetoed, absent=absent)
    return out


# --------------------------------------------------------------------------
# agreement -- licensed, and chance-honest
# --------------------------------------------------------------------------

def agree(a: Any, b: Any, scale: str, tol: float = 0.01) -> bool | None:
    """Scale-appropriate agreement between two readings of one axis.

    ``None`` when either reading is missing -- a gap is not a disagreement.
    Ratio readings agree within relative ``tol``; ordinal readings compare as
    integers; everything else falls to the framework's one normalizing
    comparator.
    """
    if a is None or b is None:
        return None
    if scale == "ratio":
        try:
            x, y = float(a), float(b)
        except (TypeError, ValueError):
            return values_equal(a, b)
        if x == y:
            return True
        return abs(x - y) <= tol * max(abs(x), abs(y))
    if scale == "ordinal":
        try:
            return int(a) == int(b)
        except (TypeError, ValueError):
            return values_equal(a, b)
    return values_equal(a, b)


def adjacent(a: Any, b: Any, scale: str) -> bool | None:
    """Ordinal near-miss: one step apart is a softer agreement, not a hit."""
    if scale != "ordinal" or a is None or b is None:
        return None
    try:
        return abs(int(a) - int(b)) <= 1
    except (TypeError, ValueError):
        return None


def _label(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def kappa(pairs: Iterable[tuple]) -> dict:
    """Chance-corrected agreement between two nominal readings of the same things.

    Agreement is evidence exactly in proportion to how likely disagreement
    was: two paths matching 80% of the time on an axis where one label covers
    90% of the things have agreed *less* than chance.  Raw rates flatter
    exactly when discrimination is low; kappa discounts by the agreement the
    marginals hand out for free.  ``kappa=None`` with reason means the
    marginals made disagreement impossible -- enforced coherence earns
    nothing (Pillar III).
    """
    got = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(got)
    if not n:
        return dict(kappa=None, po=None, pe=None, n=0, reason="no paired readings")
    hits = sum(1 for a, b in got if values_equal(a, b))
    po = hits / n
    left = Counter(_label(a) for a, _ in got)
    right = Counter(_label(b) for _, b in got)
    pe = sum(left[k] * right.get(k, 0) for k in left) / (n * n)
    if pe >= 1.0 - 1e-12:
        return dict(kappa=None, po=round(po, 3), pe=round(pe, 3), n=n,
                    reason="constant marginals: agreement was enforced, "
                           "not earned")
    return dict(kappa=round((po - pe) / (1 - pe), 3),
                po=round(po, 3), pe=round(pe, 3), n=n)


# --------------------------------------------------------------------------
# licensed statistics
# --------------------------------------------------------------------------

def ranks(values: list) -> list[float]:
    """Tie-averaged ranks, 1 = largest."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = mean
        i = j + 1
    return out


def median(values: Iterable) -> Any:
    got = sorted(v for v in values if v is not None)
    if not got:
        return None
    mid = len(got) // 2
    return got[mid] if len(got) % 2 else (got[mid - 1] + got[mid]) / 2


def spearman(xs: list, ys: list) -> tuple:
    """Rank correlation between two *paired* lists.  Licensed on ordinal
    readings; Pearson is not.  Returns ``(rho, n)``; ``rho=None`` under three
    usable pairs, because two points always correlate perfectly."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) *
                    sum((b - my) ** 2 for b in ry))
    return (round(num / den, 3) if den else None), n


def mannwhitney(xs: list, ys: list) -> dict:
    """Rank-sum comparison of two *unpaired* groups on one ordinal-or-better
    axis -- two clouds of readings, not two readings of the same things
    (that is :func:`spearman`).

    Tie-corrected normal approximation; treat ``p`` as indicative below
    about eight readings per group.  ``r`` is the rank-biserial effect:
    +1 means every x outranks every y.
    """
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    n1, n2 = len(xs), len(ys)
    if not n1 or not n2:
        return dict(u=None, z=None, p=None, r=None, n1=n1, n2=n2,
                    reason="a group is empty")
    pooled = xs + ys
    ascending = ranks([-v for v in pooled])         # 1 = smallest
    r1 = sum(ascending[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    counts = Counter(pooled)
    tie_term = sum(t ** 3 - t for t in counts.values())
    var = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return dict(u=u1, z=None, p=None, r=None, n1=n1, n2=n2,
                    reason="all readings tied: nothing could differ")
    z = (u1 - n1 * n2 / 2) / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return dict(u=u1, z=round(z, 3), p=round(p, 4),
                r=round(2 * u1 / (n1 * n2) - 1, 3), n1=n1, n2=n2)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple | None:
    """A proportion with its interval.  A dozen readings are not a point
    estimate, and the interval says so out loud."""
    if not n:
        return None
    phat = hits / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (round(phat, 3), round(max(0.0, centre - half), 3),
            round(min(1.0, centre + half), 3))


def n_eff(n: int, rho: float) -> float:
    """Design effect: what correlated readings are actually worth.

    N readings sharing intraclass correlation ``rho`` carry the information
    of about ``N / (1 + (N-1)·rho)`` independent ones -- bounded by ``1/rho``
    forever.  Confidence is bought with mechanism dissimilarity, never with
    repetition count.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho is an intraclass correlation, in [0, 1]")
    return n / (1 + (n - 1) * rho)


# --------------------------------------------------------------------------
# orders from comparisons
# --------------------------------------------------------------------------

def bradley_terry(wins: Iterable[tuple], prior: float = 0.1,
                  max_iter: int = 1000, tol: float = 1e-10) -> dict:
    """Interval strengths from pairwise winners -- the settlement of the
    "prefer comparisons" move.

    ``wins`` is ``(winner, loser)`` per comparison, repeats welcome.  Three
    diagnostics ride along, each a law:

    * ``connected`` -- strengths are identifiable only if the comparison
      graph is connected; across components the numbers mean nothing, and
      blocking without bridges is how graphs disconnect.
    * ``consistency`` -- the fraction of observed outcomes the fitted order
      agrees with: the principled transitivity reading (1.0 = perfectly
      transitive record), with the offending pairs listed.
    * ``se`` -- approximate standard error per log-strength; a strength
      without its interval is a selection amplifier waiting to fire.

    ``prior`` adds a small symmetric pseudo-count to each *observed* pairing
    so undefeated items keep finite strengths; it never invents an edge.
    """
    obs: dict[Any, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
    observed = 0
    for winner, loser in wins:
        if winner == loser:
            raise ValueError(f"{winner!r} cannot beat itself")
        obs[winner][loser] += 1.0
        observed += 1
    items = sorted(set(obs) | {b for row in obs.values() for b in row}, key=repr)
    if not items:
        return dict(scores={}, se={}, ranking=[], connected=True,
                    components=[], consistency=None, inconsistent=[], n=0)
    pairings = {frozenset((a, b)) for a in obs for b in obs[a]}
    sm: dict[Any, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
    for pairing in pairings:
        a, b = tuple(pairing)
        sm[a][b] = obs[a].get(b, 0.0) + prior
        sm[b][a] = obs[b].get(a, 0.0) + prior

    # connectivity of the observed graph (the prior never invents an edge)
    parent = {i: i for i in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in obs:
        for b in obs[a]:
            parent[find(a)] = find(b)
    components = defaultdict(list)
    for i in items:
        components[find(i)].append(i)
    parts = sorted(components.values(), key=len, reverse=True)

    # minorization-maximization on strengths
    pi = {i: 1.0 for i in items}
    n_ij = {(a, b): sm[a][b] + sm[b][a] for a in sm for b in sm[a]}
    for _ in range(max_iter):
        shift = 0.0
        for i in items:
            wins_i = sum(sm[i].values())
            denom = sum(n_ij[(i, j)] / (pi[i] + pi[j])
                        for j in items if (i, j) in n_ij)
            if not denom:
                continue
            new = wins_i / denom
            shift = max(shift, abs(new - pi[i]))
            pi[i] = new
        mean = math.exp(sum(math.log(p) for p in pi.values()) / len(pi))
        for i in pi:
            pi[i] /= mean
        if shift < tol:
            break

    scores = {i: round(math.log(p), 4) for i, p in pi.items()}
    se = {}
    for i in items:
        info = sum(n_ij[(i, j)] * (pi[i] * pi[j]) / (pi[i] + pi[j]) ** 2
                   for j in items if (i, j) in n_ij)
        se[i] = round(1 / math.sqrt(info), 4) if info > 0 else None

    consistent, inconsistent = 0.0, Counter()
    for a in obs:
        for b, times in obs[a].items():
            if pi[a] > pi[b]:
                consistent += times
            elif pi[a] == pi[b]:
                consistent += times / 2
            else:
                inconsistent[(a, b)] += int(times)
    return dict(
        scores=scores, se=se,
        ranking=sorted(items, key=lambda i: -scores[i]),
        connected=len(parts) <= 1,
        components=[sorted(p, key=repr) for p in parts] if len(parts) > 1 else [],
        consistency=round(consistent / observed, 3) if observed else None,
        inconsistent=sorted(((a, b, c) for (a, b), c in inconsistent.items()),
                            key=lambda x: -x[2]),
        n=observed)


# --------------------------------------------------------------------------
# the instrument, measured
# --------------------------------------------------------------------------

def reliability(ledger: Ledger, pairs: Iterable[tuple], axes: Iterable[str],
                scales: dict, belief: str | None = None, strict: bool = True,
                compare: Callable[[str, Any, Any], bool | None] | None = None,
                ) -> dict:
    """Test--retest over ``(base, twin)`` entity pairs.  This caps everything.

    The twin is a fresh entity measured under reshuffled context; agreement
    per axis is the ceiling on every downstream confidence.  A pair where
    both readings were vetoed is *reproducibly unreadable* -- a real property
    of the instrument, but not an agreement about a value -- and is reported
    separately rather than folded into either rate.

    ``compare(axis, a, b)`` overrides the default scale-typed :func:`agree`
    for experiments whose axes carry a bespoke equivalence (a dict reading
    compared by one field, a tolerance wider than the default).
    """
    if compare is None:
        compare = lambda axis, a, b: agree(a, b, scales[axis])  # noqa: E731
    pairs = list(pairs)
    rows: dict[str, dict] = {}
    for axis in axes:
        exact, near, n, detail = 0, 0, 0, []
        both_stuck, one_stuck = [], []
        for base, twin in pairs:
            a = reading(ledger, base, axis, belief=belief, strict=strict)
            b = reading(ledger, twin, axis, belief=belief, strict=strict)
            if a is None or b is None:
                continue
            stuck = (bool(a["vetoed"]), bool(b["vetoed"]))
            if all(stuck):
                both_stuck.append(base)
                continue
            if any(stuck):
                one_stuck.append(base)
                continue
            n += 1
            same = compare(axis, a["value"], b["value"])
            close = adjacent(a["value"], b["value"], scales[axis])
            exact += bool(same)
            near += bool(same or close)
            detail.append((base, a["value"], b["value"], bool(same)))
        rows[axis] = dict(n=n, exact=round(exact / n, 3) if n else None,
                          within_one=round(near / n, 3) if n else None,
                          detail=detail, unresolved_both=both_stuck,
                          unresolved_one=one_stuck)
    return rows


def drift(rel: dict, scales: dict) -> dict:
    """Is base/twin disagreement noise, or a direction?

    Signed twin-minus-base deltas per ordinal axis.  A difference that keeps
    its sign across items is a context effect, not variance -- and averaging
    more readings will never shrink it (Pillar II).  Agreements are evidence
    of no direction either way; they neither make nor break the claim.
    """
    rows: dict[str, dict] = {}
    for axis, row in rel.items():
        if scales.get(axis) != "ordinal":
            continue
        deltas = []
        for _entity, a, b, _same in row["detail"]:
            try:
                deltas.append(int(b) - int(a))
            except (TypeError, ValueError):
                continue
        if not deltas:
            continue
        up = sum(1 for d in deltas if d > 0)
        down = sum(1 for d in deltas if d < 0)
        moved = up + down
        rows[axis] = dict(n=len(deltas),
                          mean=round(sum(deltas) / len(deltas), 3),
                          up=up, down=down, moved=moved,
                          same_sign=(moved > 0 and (up == moved or
                                                    down == moved)),
                          deltas=deltas)
    return rows


def discrimination(values: Iterable, top: int = 3) -> dict:
    """Do the readings vary with the things measured?

    A constant instrument passes every law and is worthless -- its readings
    could never have disagreed across objects (Pillar III).  High
    ``modal_share`` also warns that raw agreement rates are about to
    flatter: check :func:`kappa` before believing them.
    """
    labels = [_label(v) for v in values if v is not None]
    counts = Counter(labels)
    return dict(n=len(labels), distinct=len(counts),
                modal_share=(round(max(counts.values()) / len(labels), 3)
                             if labels else None),
                top=counts.most_common(top))


# --------------------------------------------------------------------------
# the ground
# --------------------------------------------------------------------------

def grounded(ledger: Ledger, axes: Iterable[str] | None = None,
             entities: Iterable[str] | None = None,
             ground: Callable[[str], str] | None = None,
             belief: str | None = None, strict: bool = True):
    """Yield ``(entity, axis, reading, truth)`` wherever frozen ground exists.

    Two places ground lives, one gatherer:

    * ``ground=None`` -- the natural flow: the cell was measured, and the
      outcome was later frozen *on the same cell* (assignment, code, an
      explicit ``freeze``).  Calibration accrues from the ledger alone.
    * ``ground=fn`` -- designed-in ground on parallel entities (an experiment
      whose truth must stay invisible to the instrument); ``fn`` maps a
      measured entity to its ground entity.
    """
    if axes is not None and entities is not None:
        # both given: iterate in the given orders, axis-major, so reports
        # come out in pre-registered order
        cells = [(entity, axis) for axis in axes for entity in entities]
    else:
        wanted_axes = None if axes is None else set(axes)
        wanted_entities = None if entities is None else set(entities)
        cells = [(entity, axis) for entity, axis in ledger.cells()
                 if (wanted_axes is None or axis in wanted_axes)
                 and (wanted_entities is None or entity in wanted_entities)]
    for entity, axis in cells:
        if ground is not None:
            frozen = ledger.latest_frozen(ground(entity), axis)
        else:
            frozen = ledger.latest_frozen(entity, axis)
        if frozen is None:
            continue
        got = reading(ledger, entity, axis, belief=belief, strict=strict)
        if got is None:
            continue
        yield entity, axis, got, frozen.value


def concordance(cells: Iterable[tuple], scales: dict) -> tuple[dict, list]:
    """Readings against the frozen record: the only touch of reality.

    Coherence earns confidence; concordance earns calibration; nothing else
    earns anything.  Feed it :func:`grounded`.  Returns per-axis rows and the
    flat ``(entity, axis, p, hit)`` list that :func:`calibration` and
    :func:`separation` consume.
    """
    rows: dict[str, dict] = {}
    per_reading: list[tuple] = []
    for entity, axis, got, truth in cells:
        row = rows.setdefault(axis, dict(n=0, hits=0, misses=[]))
        row["n"] += 1
        ok = bool(agree(got["value"], truth, scales[axis]))
        row["hits"] += ok
        per_reading.append((entity, axis, got["p"], ok))
        if not ok:
            row["misses"].append((entity, got["value"], truth, got["p"],
                                  got["vetoed"]))
    for axis, row in rows.items():
        row["rate"] = wilson(row["hits"], row["n"])
    return rows, per_reading


def calibration(per_reading: Iterable[tuple]) -> dict:
    """The instrument's stated p against the record, bucketed.

    ``p`` is itself a reading: initially untrusted, calibratable over time.
    An instrument that said 0.8 should have been sustained about 8 times
    in 10.
    """
    buckets: dict[float, list] = defaultdict(lambda: [0, 0])
    for _entity, _axis, p, ok in per_reading:
        key = round(min(0.999, max(0.0, p)) * 10) / 10
        buckets[key][0] += ok
        buckets[key][1] += 1
    return {k: dict(hits=v[0], n=v[1], observed=wilson(v[0], v[1]))
            for k, v in sorted(buckets.items())}


def separation(per_reading: Iterable[tuple]) -> dict:
    """Does a low stated p actually mark the readings that turned out wrong?

    Per axis, because an instrument can be calibrated on one axis and asleep
    on another -- a pooled figure would hide both facts.  A p that is the
    same on hits and misses is a formality; one that is lower on misses is
    doing work.
    """
    rows: dict[str, dict] = {}
    for _entity, axis, p, ok in per_reading:
        row = rows.setdefault(axis, dict(hit=[], miss=[]))
        row["hit" if ok else "miss"].append(p)
    out: dict[str, dict] = {}
    for axis, row in rows.items():
        hits, misses = row["hit"], row["miss"]
        mean_hit = round(sum(hits) / len(hits), 3) if hits else None
        mean_miss = round(sum(misses) / len(misses), 3) if misses else None
        out[axis] = dict(
            n_hit=len(hits), n_miss=len(misses),
            mean_p_hit=mean_hit, mean_p_miss=mean_miss,
            separation=(round(mean_hit - mean_miss, 3)
                        if mean_hit is not None and mean_miss is not None
                        else None))
    return out


def tiers(ledger: Ledger, validation: Callable[[Any], bool],
          proposers: Iterable[str] | None = None) -> dict:
    """The two-tier budget, recomputed from the record rather than trusted.

    One instrument opinion is one call, so findings and validation are
    separately countable after the fact -- the buyer of confidence sees what
    confidence cost.  ``validation`` is the experiment's own predicate over
    opinions (twin entities, pair cells, held-out judges...).  Completions
    the transport re-asked for cost a call and left no opinion, so these
    counts are a floor.
    """
    ids = None if proposers is None else set(proposers)
    findings = checking = 0
    for o in ledger:
        if ids is not None:
            if o.belief not in ids:
                continue
        elif o.frozen or not _generative(o.meta or {}):
            continue
        if validation(o):
            checking += 1
        else:
            findings += 1
    total = findings + checking
    return dict(findings=findings, validation=checking, total=total,
                validation_share=round(checking / total, 3) if total else None)


# --------------------------------------------------------------------------
# the record as a graph
# --------------------------------------------------------------------------

_DEPENDENCY_KINDS = ("ref", "host", "child")


def graph(ledger: Ledger) -> dict:
    """The record's structure, derived: nodes and typed edges.

    Nodes are strings -- an entity id, a belief id, or a cell address
    ``entity#attr``.  Edges, deduplicated:

    * ``authored`` -- belief -> cell, with how many opinions and whether any
      froze.
    * ``ref`` -- cell -> what its value crossed the boundary with
      (``meta["refs"]``: entity ids and cell addresses; the provenance
      ``_plain`` and call rendering would otherwise destroy).
    * ``host`` -- an episode's call-cell entity -> the Thing it ran on,
      recovered exactly from provenance (``call`` + ``state``), never by
      parsing entity ids.
    * ``child`` -- entity ``a#b`` -> entity ``a``, when both appear as
      entities in the record.

    ``exposures`` groups readings by context fingerprint: the readings in
    one group saw the same rendered context, so their agreement shares
    exposure and is cheap (Pillar III).  The whole structure is a derived
    instrument (Pillar IV): rebuildable from any ledger, never stored,
    never authoritative.
    """
    entities: list[str] = []
    beliefs: list[str] = []
    cells: list[tuple[str, str]] = []
    seen_entity, seen_belief, seen_cell = set(), set(), set()
    authored: dict[tuple, dict] = {}
    seen_edge, links = set(), []
    exposures: dict[str, list] = {}
    for o in ledger:
        cell = f"{o.entity}#{o.attr}"
        if o.entity not in seen_entity:
            seen_entity.add(o.entity)
            entities.append(o.entity)
        if o.belief not in seen_belief:
            seen_belief.add(o.belief)
            beliefs.append(o.belief)
        if (o.entity, o.attr) not in seen_cell:
            seen_cell.add((o.entity, o.attr))
            cells.append((o.entity, o.attr))
        row = authored.setdefault((o.belief, cell), dict(n=0, frozen=False))
        row["n"] += 1
        row["frozen"] = row["frozen"] or o.frozen
        meta = o.meta or {}
        for ref in meta.get("refs") or ():
            key = ("ref", cell, ref)
            if key not in seen_edge:
                seen_edge.add(key)
                links.append(dict(kind="ref", src=cell, dst=ref))
        if "exposure" in meta:
            exposures.setdefault(meta["exposure"], []).append(
                (o.belief, o.entity, o.attr))
        if "call" in meta and "state" in meta:
            suffix = f".{meta['call']}#{meta['state']}"
            if o.entity.endswith(suffix) and len(o.entity) > len(suffix):
                key = ("host", o.entity, o.entity[:-len(suffix)])
                if key not in seen_edge:
                    seen_edge.add(key)
                    links.append(dict(kind="host", src=key[1], dst=key[2]))
    edges = [dict(kind="authored", src=belief, dst=cell, **row)
             for (belief, cell), row in authored.items()]
    edges.extend(links)
    for entity in entities:
        head, sep, _tail = entity.rpartition("#")
        if sep and head in seen_entity:
            edges.append(dict(kind="child", src=entity, dst=head))
    return dict(entities=entities, beliefs=beliefs, cells=cells,
                edges=edges, exposures=exposures)


def _cells_of(g: dict) -> dict:
    out: dict[str, list] = {}
    for entity, attr in g["cells"]:
        out.setdefault(entity, []).append(f"{entity}#{attr}")
    return out


def lineage(g: dict, node: str) -> list[str]:
    """Everything upstream of ``node``: what its value transitively rests on.

    ``node`` is an entity id or a cell address.  The walk follows dependency
    edges (``ref``, ``host``, ``child``) plus containment -- a cell rests on
    its entity, so an episode result's lineage includes the episode's host
    and everything the call was handed.
    """
    dep: dict[str, list] = {}
    for edge in g["edges"]:
        if edge["kind"] in _DEPENDENCY_KINDS:
            dep.setdefault(edge["src"], []).append(edge["dst"])
    entity_of = {f"{e}#{a}": e for e, a in g["cells"]}
    out: set[str] = set()
    frontier = [node]
    while frontier:
        n = frontier.pop()
        steps = list(dep.get(n, ()))
        if n in entity_of:
            steps.append(entity_of[n])
        for step in steps:
            if step != node and step not in out:
                out.add(step)
                frontier.append(step)
    return sorted(out)


def invalidated(g: dict, node: str) -> list[str]:
    """Everything downstream of ``node``: what a change to it calls into
    question.  The staleness question, generalized into a query -- "this
    entity's state moved; which episode results and derived cells were
    computed from the old world?"  A bare entity seeds with all its cells;
    reaching an entity spreads to the cells that belong to it (an
    invalidated episode invalidates its result).
    """
    rdep: dict[str, list] = {}
    for edge in g["edges"]:
        if edge["kind"] in _DEPENDENCY_KINDS:
            rdep.setdefault(edge["dst"], []).append(edge["src"])
    cells_of = _cells_of(g)
    seeds = {node, *cells_of.get(node, ())}
    out: set[str] = set()
    frontier = list(seeds)
    while frontier:
        n = frontier.pop()
        for step in list(rdep.get(n, ())) + cells_of.get(n, []):
            if step not in out and step not in seeds:
                out.add(step)
                frontier.append(step)
    return sorted(out)
