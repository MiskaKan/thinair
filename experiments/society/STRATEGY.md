# The society experiment — pre-registration

Registered before the first run; `run.py` stamps this file's hash into the
runlog.  The design under test is SPEC.md's society layer: perception
boundaries (§4), references as capabilities (§4/§12), code-only acting (§9),
and quiescence as physics.

## The claim

A cast of ordinary Things, each playing an agent role — no Agent class, no
new Thing method, no channel between minds — can carry a multi-hop story
end-to-end over one shared ledger, driven by nothing but one injected frozen
fact and the three wake rules, and stop by itself.

## Cast (ids, order, and numbers are pre-registered)

| entity | class | private | public | panel law |
|---|---|---|---|---|
| `anna` | Customer | `budget=312.0`, `internet`, `mechanic`, `car_broken` | `to`, `saying`, `accepted_quote` | `WithinBudget` (necessary) vetoes any accepted quote above the budget |
| `internet` | Internet | `directory={"car repair": "forum"}` | `to`, `saying`, `referral` | `HonestReferral` (necessary): only listed ids may be referred |
| `forum` | Forum | — | `open_jobs`, `to`, `saying` | — |
| `mech-dave` | Mechanic | `rate=480.0`, `forum` | `to`, `quote`, `saying` | `QuotesOwnRate` (necessary) |
| `mech-tom` | Mechanic | `rate=260.0`, `forum` | `to`, `quote`, `saying` | `QuotesOwnRate` (necessary) |

Sweep order: anna, internet, forum, mech-dave, mech-tom — the expensive
mechanic first, so the budget veto fires before an acceptable quote exists.

Holdings at start (assignments by code, refs stamped): anna holds internet;
each mechanic holds the forum.  Nobody holds anna.

## The one injected fact

`anna.car_broken = True` — code, frozen, after the sweep records its
baseline.  Nothing else moves the world from outside.

## Runtime (ordinary code, zero framework surface)

Turn = `agent.consider(moved_peer, acting=True)`.  Wake rules 1–3 reduce to
one check: has the (agent, peer) state-hash pair moved since the agent last
considered that peer (self is a peer).  One turn per agent per sweep; a
recurring state-hash pair backs off (livelock watch); the run ends on a full
quiet pass or at 15 sweeps.

## Predictions

1. The chain: injected fact → anna asks internet → internet refers the forum
   (its referral vetted by `HonestReferral`, carrying its own p) → anna asks
   the forum → the forum posts a listing naming anna → both mechanics quote
   anna directly → `WithinBudget` vetoes dave's 480 (the rejection is quoted
   back; anna declines) → anna accepts tom's 260 → tom books → quiescence,
   within 15 sweeps.
2. Every hop is on one tape, authored, with `meta.refs` carrying every
   introduction — the story is a pure derivation from the ledger
   (`thinair log --graph`).
3. `312.0` (the budget) appears in no prompt shown to any mind but anna's
   own; no private cell of anyone crosses a boundary.
4. No mind ever writes another's cell or calls another's method; every
   cross-entity effect is "I wrote my own cells, you read them".

## Phases

- **R (rehearsal, offline):** scripted minds — pure functions of the rendered
  prompt, so everything they know demonstrably arrived through the boundary
  filter — through the identical cast, sweep, panels, and prompts.
  Establishes the mechanics; archived as `ledger.json` + `runlog.json`.
- **L (live):** one real model plays every mind (`run.py` without
  `--scripted`, endpoint from the environment).  Findings about model
  behaviour in the society frame wait for this phase.
