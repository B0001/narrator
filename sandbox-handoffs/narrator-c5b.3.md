# narrator-c5b.3 — Fair-play chat core: ledger, board, moves

## What was built

Three new stdlib-only modules, each following the repo's self-check
convention (`if __name__ == "__main__": _self_check()`, assert-based, no
framework), plus one integration module tying them to the already-closed
`hypothesis_board.py` (`narrator-c5b.3.2`):

- **`evidence_ledger.py`** — `EvidenceLedger`: one append-only JSONL file per
  conversation. Every entry has an id, turn, claim, provenance tag, and an
  optional `supports` tuple (ids of entries it was derived from). Provenance
  is checked at write time (`ValueError` on anything not in
  `stated_by_user` / `observed_artifact` / `inferred_by_model` / `assumed`),
  duplicate ids are rejected, and dangling `supports` are rejected. Writes
  flush per line. `load(path)` reads it back and reuses `agents.py`'s
  `load_transcript` pattern exactly: a `JSONDecodeError` is re-raised as
  `ValueError(f"{path}:{lineno}: {e}")`, naming the file and the exact line
  that didn't survive a mid-write kill, with blank lines skipped (not
  miscounted).

- **`admissibility.py`** — inverted from `mystery.EpistemicClueGraph
  .validate_solvability()`. `check(ledger, cited_ids)` walks each cited
  entry: a `DIRECT` entry (`stated_by_user`/`observed_artifact`) is grounded
  on its own; an `inferred_by_model` entry is grounded only if it cites
  support and *every* one of those is itself grounded (recursively); an
  `assumed` entry is never grounded no matter what it cites — laundering an
  assumption through a support chain would make the tag meaningless. Returns
  a `Verdict(admissible, missing)` where `missing` names exactly which
  entries broke the chain, mirroring `suspects_consistent_with_clues()`:
  reads only the ledger, never any model-internal state.

- **`moves.py`** — `choose_move(ledger, turn, cited_ids, requested_move)`.
  Four moves: `reveal`, `complicate`, `ask`, `abstain`. The one hard rule:
  a `reveal` request runs `admissibility.check` first, and if it fails, the
  move is downgraded to `abstain` with a reason naming the missing evidence
  — there is no code path from "I have a conclusion" to "I said it" that
  skips the check. Every returned `TurnLog` always carries both a move and
  a non-empty reason.

- **`chat_core.py`** — `ChatCore` wires ledger + board + admissibility +
  moves into one turn loop: `observe()` writes evidence, `conclude()` calls
  `choose_move` and, only when the move actually lands as `reveal`, applies
  a board `rule_out()` whose reason cites the ledger entry that licensed it.
  This closes the gap the `.3.2` handoff flagged ("the natural integration
  point is passing a ledger entry id ... as the reason") — a downgraded
  (abstained) turn is now structurally prevented from touching the board.

## How the acceptance criteria are met

Bead's own AC: *"Every conclusion the bot states cites ledger entries the
user has seen; conclusions that cannot are withheld with the missing
evidence named."*

`chat_core.py`'s self-check runs this as one scenario end to end (a
4-suspect board, deliberately shaped like `mystery.py`'s cast):

1. Turn 0: a direct sighting (`stated_by_user`) is logged, plus an
   unsupported inference drawn from nothing (`inferred_by_model`, no
   `supports`). A `reveal` attempt citing only the unsupported inference is
   downgraded to `abstain`, the missing entry is named in the log, and —
   critically — the hypothesis board is asserted **unchanged**: a blocked
   reveal must not be able to sneak a board update in on the side.
2. Turn 1: an `observed_artifact` entry arrives and a new inference cites it
   as support. The *same* conclusion shape is now admissible; `reveal` goes
   through, the board narrows via `rule_out`, and the board's own history
   records a reason string containing the ledger entry id that licensed it.
3. Turn 2: a bare `assumed` entry (no user ever saw or said this) is cited
   for another reveal attempt — blocked again, board unchanged.

Reading "cites ledger entries the user has seen" as *transitively* grounded
(an inference is admissible if its support chain bottoms out only in
`stated_by_user`/`observed_artifact` entries, never resting on an
inferred-only or assumed leg) rather than requiring literal direct-entry
citation is a deliberate design choice — see Scope notes below for why, and
compare `admissibility.py`'s own AC wording ("an inferred-*only* chain is
blocked"), which only forbids chains with no direct grounding anywhere, not
inference in general.

### Children's acceptance criteria (met by the same artifacts)

This bead's description enumerates exactly four structures — ledger, board,
admissibility check, move set — which are precisely what children `.3.1`,
`.3.2` (already closed), `.3.3`, and `.3.5` ask for individually. Since the
artifacts built here satisfy each child's own stated AC directly (not just
the parent's rollup AC), those three are closed alongside this bead rather
than left open to be rediscovered as already-done work by a future session:

- **`.3.1`** (evidence ledger with provenance tags) — `evidence_ledger.py`'s
  self-check drives a mid-write kill (`load()` raises naming
  `{path}:4:`, matching the AC's "offending line named") and drives an
  untagged-provenance write (rejected at write time, not read time).
- **`.3.3`** (admissibility check inverted from `validate_solvability()`) —
  `admissibility.py`'s self-check exercises: fully-grounded direct citation
  (admissible), a grounded multi-step inference (admissible), an
  unsupported inference (blocked, self-naming), an assumed premise laundered
  through an inference (blocked, names the assumption), and a mixed
  chain with one grounded leg and one assumed leg (blocked — one bad leg is
  enough, same shape as the seed-7 bug where one alibi wasn't enough to
  clear the ambiguity).
- **`.3.5`** (move set) — `moves.py`'s self-check asserts every `TurnLog`
  carries a move and a reason, and specifically that a `reveal` request
  whose citations fail admissibility comes back as `abstain`, never
  `reveal` — the AC's exact wording ("a turn whose conclusion fails
  admissibility cannot choose reveal").

## Verification run

```
$ uv run python evidence_ledger.py && uv run python admissibility.py \
  && uv run python moves.py && uv run python chat_core.py \
  && uv run python hypothesis_board.py
ok
ok
ok
ok
ok
$ uv run pytest -q
1 passed in 0.00s
```

Also re-ran every pre-existing module's self-check to confirm no regression:
`ocean.py` and `agents.py` still print `ok`; `metrics.py` still prints its
table. `motive.py`, `mystery.py`, and `chapters.py` fail with
`ModuleNotFoundError: No module named 'networkx'` — this is the pre-existing,
already-filed `narrator-3c5` bug (`pyproject.toml` missing the `networkx`
dependency), not something this bead touched or broke. Confirmed
pre-existing by checking it reproduces on a clean `uv sync` regardless of
these changes.

## Scope notes / design decisions

- **Grounding is transitive, not one-hop.** An `inferred_by_model` entry is
  admissible as a citation if its *entire* support chain bottoms out in
  direct evidence, not only if the conclusion cites a direct entry
  literally. This matches `.3.3`'s own AC wording ("an inferred-only chain
  is blocked") — a chain isn't inferred-only just because it has one
  inference step in it; it's inferred-only if no direct evidence appears
  anywhere in it. A stricter "conclusions may only cite DIRECT entries"
  reading would make `inferred_by_model` entries unusable for anything,
  which contradicts the ledger even having that tag.
- **`assumed` is a dead end by design, unconditionally.** No support chain
  can launder an assumption into groundedness — this was a deliberate
  design choice tested explicitly (`built_on_assumption` and the
  mixed-chain case in `admissibility.py`'s self-check), not an oversight.
  If a future bead wants "assumed" entries to become groundable once
  confirmed, that requires a distinct provenance transition (e.g.
  rewriting the tag once verified), not a support-chain workaround.
- **Board integration is intentionally minimal.** `chat_core.ChatCore
  .conclude()` only calls `board.rule_out()`, never `reweight()` — a
  `reveal` is exactly the kind of high-confidence, evidence-backed event
  `rule_out` was built for (`.3.2`'s handoff: "only `rule_out` ... can drop
  the live count"). Wiring `reweight()` to `ask`/`complicate` moves (soft
  evidence, not a conclusion) is real follow-on work; not built here since
  it's not required by any of this bead's or its children's acceptance
  criteria and would be speculative without a concrete calling pattern from
  the (still-open) persona/turn-loop work (`.3.8`, `.3.9`, `.3.10`).
- **Not touched, deliberately out of scope:** `.3.4` (discriminating-question
  selector), `.3.6` (Chekhov ledger / age counter), `.3.7` (metrics
  extension), `.3.8`–`.3.12` (two-channel turns, trait-weighted evidence,
  agreeableness sweep, mode flag, checker-subagent isolation). All of these
  build *on top of* the four structures closed here; none of their specific
  acceptance criteria are met by this session's artifacts, so they stay
  open.
- **No new file under `tests/`.** Matching `hypothesis_board.py`'s
  precedent: each module's own `_self_check()` is the runnable check the
  working rules ask for. `tests/test_scaffold.py` is untouched.

## Suggested next commands

```
bd close narrator-c5b.3.1 narrator-c5b.3.3 narrator-c5b.3.5 narrator-c5b.3
git status   # nothing committed — new files are untracked, per "do not commit unless asked"
```
