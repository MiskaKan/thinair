"""A 12-item synthetic news corpus with designed-in ground.

**These are not real news reports.**  Every item, organisation, place detail
and number below was written for this experiment.  Nothing here should be read
as reporting, quoted, or reused as if it described real events.

Why synthetic: the point of the experiment is to measure the *instrument*, and
that needs a frozen record to measure it against (PLAN_2, "The Ground").  A
corpus whose ground truth is authored is the only corpus where concordance is
computable at this size.  The price is stated once and never forgotten: the
absent population is *everything* -- these items were written by the same kind
of process that will read them, so any agreement between corpus and reader is
partly shared authorship, not evidence.  See STRATEGY.md §1.

Planted defects, so the validator paths have something real to catch:

* ``n05`` states a percentage that is wrong (30% where the figures give 24%),
  and a dateline weekday that is wrong (2026-08-03 was a Monday).
* ``n07`` states an average that is wrong (420 where 4200/11 = 381.8).
* ``n03`` and ``n06`` state the affected count only as a sum, so the correct
  reading is a number that appears nowhere in the source text.
* ``n11`` and ``n12`` are near-duplicate wires about one event.
* ``n02`` and ``n11``/``n12`` are set in countries absent from the ISO table
  thinair vendors, so the reference validator's own coverage is exercised.
* ``n10`` is about reefs, not people: its ``people_affected`` is 0.
"""

#: Ordinal rubric for ``certainty`` -- decidable from the text, not taste.
CERTAINTY = {
    1: "rumour: unnamed sources, social posts, 'early reports'",
    2: "attributed: a named non-official source (NGO, union, employee)",
    3: "official: a government body or the responsible company, on record",
    4: "record: a published document, register, paper or court filing",
}

#: Ordinal rubric for ``horizon`` -- when the central claim becomes checkable
#: against a public record.  Every item names its resolution event.
HORIZON = {
    1: "days: a public record settles it within a week",
    2: "weeks: within two months",
    3: "months: within a year",
    4: "no resolution is foreseeable",
}

#: ``people_affected`` measurand, stated so it is decidable:
#: the number of people the item says are *directly* affected.  Where sources
#: conflict, the official figure.  Where the item gives only components, their
#: sum.  Where no people are affected, 0.
MEASURAND = (
    "the number of people this item says are directly affected; where two "
    "sources conflict, the official figure; where only components are given, "
    "their sum; where no people are affected, 0"
)

ITEMS = [
    dict(
        id="n01",
        text=(
            "TAMPERE, Finland — 2026-08-05 (Wednesday). The municipal water "
            "utility Pirkanmaan Vesi said on Wednesday that coliform bacteria "
            "had been detected in the northern distribution zone and issued a "
            "boil-water notice covering 3,400 of the town district's 8,500 "
            "residents — 40 percent of the zone. The utility's operations "
            "director, Heli Rantanen, said in a statement that chlorination "
            "had been increased and that no illnesses had been reported. A "
            "laboratory report confirming or lifting the notice is due from "
            "the regional health authority on Friday."
        ),
        ground=dict(country="FI", people_affected=3400, certainty=3, horizon=1),
    ),
    dict(
        id="n02",
        text=(
            "NAKURU, Kenya — 2026-08-04 (Tuesday). Swarms of desert locusts "
            "have settled across roughly 12,000 hectares of cropland in the "
            "central Rift Valley, according to field teams from the "
            "agricultural charity Mashamba Trust, which estimates that 5,600 "
            "smallholder households, about 31,000 people, farm the affected "
            "land. The trust's regional coordinator said aerial spraying had "
            "not yet begun. The national agriculture ministry has not "
            "commented. A joint FAO and ministry survey that would put an "
            "authoritative figure on the infestation is scheduled for "
            "November."
        ),
        ground=dict(country="KE", people_affected=31000, certainty=2, horizon=3),
    ),
    dict(
        id="n03",
        text=(
            "KAWASAKI, Japan — 2026-08-03 (Monday). A fire in the coating "
            "shop of a Kawasaki auto-parts plant killed 2 workers and injured "
            "14 others out of the 96 on the night shift, according to the "
            "incident report published on Monday by the Kanagawa prefectural "
            "police, which lists the cause as an ignited solvent vapour. The "
            "plant's operator, Shirogane Kikai, has suspended the line. The "
            "fire service will publish its own findings within five days."
        ),
        ground=dict(country="JP", people_affected=16, certainty=4, horizon=1),
    ),
    dict(
        id="n04",
        text=(
            "BELO HORIZONTE, Brazil — 2026-08-04 (Tuesday). State civil "
            "defence ordered the precautionary evacuation of 8,500 residents "
            "downstream of the Córrego Fundo tailings dam on Tuesday after "
            "monitoring instruments recorded seepage on the eastern "
            "abutment. Those evacuated are 8,500 of the municipality's 34,000 "
            "inhabitants, a quarter of the population. The dam's operator "
            "said the structure remained within its safety envelope. An "
            "independent engineering assessment commissioned by the state has "
            "been given three weeks to report."
        ),
        ground=dict(country="BR", people_affected=8500, certainty=3, horizon=2),
    ),
    dict(
        id="n05",
        text=(
            "HANNOVER, Germany — 2026-08-03 (Wednesday). A signalling failure "
            "at Hannover Hauptbahnhof disrupted travel for 60,000 of the "
            "250,000 passengers who pass through the station daily, 30 "
            "percent of the day's traffic, the operator NordBahn said in a "
            "statement. Services were restored after six hours. The operator "
            "is required to file a fault report with the federal railway "
            "regulator within 48 hours, and the regulator publishes such "
            "filings on receipt."
        ),
        ground=dict(country="DE", people_affected=60000, certainty=3, horizon=1),
    ),
    dict(
        id="n06",
        text=(
            "PATNA, India — 2026-08-06 (Thursday). A first-floor corridor of "
            "a government secondary school in Patna district partially "
            "collapsed on Thursday morning, early reports circulating among "
            "parents said. The school enrols 340 pupils and employs 45 staff, "
            "all of whom were on the premises when the corridor gave way. "
            "Neither the district education office nor the police have "
            "confirmed the collapse or given any casualty figure. The "
            "district magistrate's office said a verified count would be "
            "released tomorrow."
        ),
        ground=dict(country="IN", people_affected=385, certainty=1, horizon=1),
    ),
    dict(
        id="n07",
        text=(
            "COLUMBUS, United States — 2026-08-05 (Wednesday). A ransomware "
            "intrusion has locked scheduling systems at 11 hospitals in an "
            "Ohio health network, forcing the postponement of roughly 4,200 "
            "appointments, an average of 420 per hospital, according to two "
            "people familiar with the response who were not authorised to "
            "speak publicly. The network has not confirmed an intrusion. "
            "Federal rules require a breach notification to the regulator "
            "within 30 days, and those notifications are published."
        ),
        ground=dict(country="US", people_affected=4200, certainty=1, horizon=2),
    ),
    dict(
        id="n08",
        text=(
            "PORT HARCOURT, Nigeria — 2026-08-02 (Sunday). A leak on a "
            "disused feeder pipeline has contaminated farmland used by 2,100 "
            "of the 7,000 registered farmers in two Rivers State "
            "communities, three in ten of them, according to a field survey "
            "released by the environmental group Delta Watch. The pipeline's "
            "operator disputes the survey's extent but not the leak. A joint "
            "investigation visit by the regulator, the operator and community "
            "representatives is scheduled for October."
        ),
        ground=dict(country="NG", people_affected=2100, certainty=2, horizon=3),
    ),
    dict(
        id="n09",
        text=(
            "PARIS, France — 2026-08-06 (Thursday). Marchers filled the "
            "Boulevard Saint-Michel on Thursday in the fourth day of protests "
            "against the pension reform bill. The organising union federation "
            "put the turnout at 250,000; the Paris prefecture of police, "
            "whose figure is the official one, counted 89,000. There were no "
            "reported injuries. The interior ministry will publish a "
            "consolidated national count on Monday."
        ),
        ground=dict(country="FR", people_affected=89000, certainty=3, horizon=1),
    ),
    dict(
        id="n10",
        text=(
            "TOWNSVILLE, Australia — 2026-08-04 (Tuesday). Aerial and in-water "
            "surveys recorded bleaching on 452 of the 1,050 reefs surveyed "
            "across the central Great Barrier Reef, 43 percent, according to "
            "a paper published on Tuesday in a peer-reviewed marine science "
            "journal by a team at the national reef institute. No settlements "
            "were affected and no people were displaced. The institute's next "
            "survey season begins in March, when the same reefs will be "
            "re-scored."
        ),
        ground=dict(country="AU", people_affected=0, certainty=4, horizon=3),
    ),
    dict(
        id="n11",
        text=(
            "TACLOBAN, Philippines — 2026-08-01 (Saturday). Typhoon Rosalia "
            "displaced 74,000 people across eastern Samar, the national "
            "disaster risk reduction council said on Saturday, with 312 "
            "evacuation centres opened. Power remains out in eleven "
            "municipalities. The council said damage assessments from local "
            "government units were still arriving and that a consolidated "
            "situation report would be issued within three weeks."
        ),
        ground=dict(country="PH", people_affected=74000, certainty=3, horizon=2),
    ),
    dict(
        id="n12",
        text=(
            "MANILA, Philippines — 2026-08-01 (Saturday). The national "
            "disaster risk reduction council reported on Saturday that 74,000 "
            "people had been displaced in eastern Samar by Typhoon Rosalia "
            "and that 312 evacuation centres were in use. Eleven "
            "municipalities are without power. A consolidated situation "
            "report, drawing on local government damage assessments still "
            "being submitted, is expected within three weeks."
        ),
        ground=dict(country="PH", people_affected=74000, certainty=3, horizon=2),
    ),
]

BY_ID = {item["id"]: item for item in ITEMS}


def ground(axis: str) -> dict:
    """``{item id: designed-in value}`` for one grounded axis."""
    return {item["id"]: item["ground"][axis] for item in ITEMS
            if axis in item["ground"]}
