# narrator-c5b.3.2 — Hypothesis board over live interpretations

## What was built

`hypothesis_board.py`, a standalone module (no dependencies beyond stdlib
`dataclasses`), following the repo's module convention: docstring explaining
the why, `if __name__ == "__main__": _self_check()` at the bottom, no
external test framework needed.

`HypothesisBoard`:

- Constructed from 3-5 `(id, statement)` pairs; rejects counts outside that
  range and duplicate ids. All hypotheses start at equal weight — no prior
  favorite before evidence arrives.
- `reweight(turn, weights, reason)` — soft belief update. Renormalizes over
  live hypotheses so weights always sum to 1. **Refuses to drive a live
  hypothesis's weight to zero** (raises `ValueError` telling the caller to use
  `rule_out` instead) and refuses a call with no `reason`. Every changed
  weight is appended to `history` as its own event — the granularity is
  per-hypothesis-per-call, not per-call, so the dump shows exactly which
  reading moved and by how much on which turn.
- `rule_out(turn, hypothesis_id, reason)` — the only way to drop a hypothesis
  out of the live set. Requires a non-empty `reason` (the evidence that
  killed it) and refuses to eliminate the last surviving hypothesis (a board
  can converge to 1, but never to 0).
- `dump()` — returns `{"hypotheses": [...], "history": [...]}`. `history` is
  the full list of `BoardEvent`s (turn, kind, hypothesis_id, old_weight,
  new_weight, reason) since the board was created, in order.

## How the acceptance criteria are met

- **"A conversation that reverses direction shows the weight shift in the
  dump rather than a silent switch."** Self-check builds a 4-hypothesis board
  about what a chat user wants (refund/exchange/complaint/info), reweights it
  toward refund+exchange on turn 1, then reweights it back toward
  complaint+info on turn 2 (a reversal). The test asserts on `dump()["history"]`
  directly: there are exactly two events for `"refund"`, the first raises its
  weight and the second lowers it — i.e. the reversal is a pair of opposing,
  timestamped entries on the record, not something you'd only see by diffing
  the final state against your memory of the first.
- **"Collapsing below two live hypotheses requires evidence that rules the
  others out."** `reweight` structurally cannot zero a hypothesis out (raises
  if asked to) — self-check exercises this directly. The only path to fewer
  live hypotheses is `rule_out`, which self-check drives all the way from 4
  live down to 1, checking at each step: no reason → rejected; already-dead
  id → `KeyError`; and the last survivor can never be ruled out even with a
  reason. Every `rule_out` in the chain carries a distinct reason string
  standing in for the evidence that killed that specific reading.

## Verification run

```
$ uv run python hypothesis_board.py
ok
$ uv run pytest -q
1 passed in 0.00s
```

(`tests/test_scaffold.py` is the pre-existing placeholder test; this bead
didn't touch it. No new file was added under `tests/` because the module's
own self-check is the runnable check the working rules ask for, matching
`ocean.py`/`agents.py`/`mystery.py`'s pattern of self-check-first, `tests/`
for anything broader.)

## Scope notes / what's deliberately not here

- This module does **not** read or write an evidence ledger — `narrator-c5b.3.1`
  (evidence ledger with provenance tags) is a sibling task, still open, and
  `narrator-c5b.3.2` has no formal `bd` dependency on it (checked via
  `bd show`). `reweight`'s `reason` and `rule_out`'s `reason` are plain
  strings today; when `.3.1` lands, the natural integration point is passing
  a ledger entry id/citation as (or alongside) that reason so admissibility
  checks in `.3.3` can trace a weight change back to a specific ledger line.
  Did not build that wiring speculatively — no ledger exists yet to wire
  against.
- Did not touch `.3.4` (discriminating-question selector) or `.3.5` (move
  set) — both are blocked-by this bead per `bd show`, now unblocked from this
  side, but their own logic is out of scope here.

## Unrelated issue found and filed separately

While spot-checking that other modules' self-checks still pass after this
change, found `motive.py` and `mystery.py` both `import networkx` but
`pyproject.toml` declares `dependencies = []`, so `uv run python mystery.py`
(or `motive.py`) fails with `ModuleNotFoundError` from a clean `uv sync`.
Reproduced, filed as `narrator-3c5` (P2 bug), not fixed here — out of this
bead's scope.

## Suggested next command

```
bd close narrator-c5b.3.2
```
