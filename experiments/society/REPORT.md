# The society experiment — report (phase R: rehearsal)

**Status: mechanics established offline; the live phase (L) awaits an
endpoint.**  Everything below is the scripted rehearsal — minds that are
pure functions of the rendered prompt, run through the identical cast,
panels, prompts, sweep, and ledger a live model would see.  Findings about
*model behaviour* in the society frame are deliberately absent: they belong
to phase L (`run.py` without `--scripted`).

## What the record shows (`ledger.json`, 114 opinions, quiescent at sweep 14)

The predicted chain ran whole, every hop authored on one tape
(`thinair log --graph --store experiments/society/ledger.json`):

1. `car_broken = true` — the one injected fact (code, frozen).
2. anna → internet: `to="internet"`, refs `["internet"]`.
3. internet → anna: `referral="forum"`, vetted by `HonestReferral`, refs
   `["forum"]` — the introduction.
4. anna → forum; forum posts `open_jobs=[{"customer": "anna", ...}]`, refs
   `["anna"]`.
5. Both mechanics quote anna directly (`QuotesOwnRate` vetted): dave 480,
   tom 260.
6. `WithinBudget` vetoed dave's 480 on the record —
   `withinBudget@accepted_quote p=0.00 "the quote 480.0 exceeds the budget
   of 312.0"` — the rejection was quoted back into the same turn, and the
   mind declined instead.  The 480 was never committed.
7. anna accepted tom's 260 (`accepted_quote=260.0`, `mechanic="mech-tom"`);
   tom booked; the sweep went quiet by itself.

Pre-registered predictions 1–4 (STRATEGY.md) all held, verified as tests in
`tests/test_society.py`: quiescence within budget; every hop authored with
`meta.refs` carrying each introduction; `312`/`budget` appearing in no
prompt shown to any mind but anna's own; no cross-entity write anywhere on
the tape; a repeat sweep after quiescence takes zero turns and zero calls.

## What the rehearsal taught (folded per §13 rule 7)

- **Queue fairness**: a static self-first wake queue let busy peers starve
  fresh introductions — anna never reached the mechanics.  Fixed:
  fresh-pairs-first, then stalest.  → GROUNDING.md Part 3.
- **Address discipline**: minds that answer any mention of their trade
  (rather than what is addressed to them) churn the society to its sweep
  cap.  → GROUNDING.md Part 3.
- **Framework gap found and fixed**: episode provenance recorded raw
  `args` (a live Thing is not JSON) and `history` checked the parent tree
  against the *combined* repetition key, flagging every cross-entity
  episode as a tree MISMATCH.  Both fixed in the package (`args` reduce per
  §7; `host_state` recorded when arguments widen the key).
- **Latent bug found and fixed**: episode retry prompts never carried the
  objections the system prompt promised ("you will be told why") — the old
  test passed on a substring coincidence.  Fixed with `episode-v2`.

## The record

- `STRATEGY.md` — pre-registration (hashes stamped in `runlog.json`).
- `cast.py`, `sweep.py`, `minds.py`, `run.py` — the code that ran.
- `ledger.json` — the complete tape; `runlog.json` — turns + outcome.
