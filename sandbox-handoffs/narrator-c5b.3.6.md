# narrator-c5b.3.6 — Chekhov ledger: unresolved threads with an age counter

## What this bead asked for

Track evidence-ledger threads that were introduced and never resolved, with
the turn count since each appeared. Two uses named in the bead: a writing
tool (flag guns on the wall that never fire) and chat policy (pacing).
Acceptance criteria: a thread introduced and dropped shows up with its age,
and retirement is an explicit entry, not silence.

## What existed already

`metrics.unresolved_threads` (narrator-c5b.3.7) already computed the "used"
half by replaying a saved chat log: transitive citation closure over
grounded reveals, correctly excluding citations refused by admissibility.
Its own docstring pointed at this bead by name for the missing half:
retirement. There was no way, anywhere in the C5 code, to explicitly close a
thread — a dropped lead and one nobody had gotten to yet were both just
"missing from the used set," indistinguishable from outside.

## What I built

**`chekhov_ledger.py`** (new module, self-contained, self-checking):
`ChekhovLedger` with three write operations —
- `observe(entry_id, turn, supports=())` — register a thread, mirrors
  `EvidenceLedger.write`.
- `use(turn, cited_ids)` — pay off citations and everything they
  transitively rest on. Callers must only invoke this for a move that
  actually landed (never on an abstained/blocked turn) — same rule
  `admissibility`/`unresolved_threads` already enforced.
- `retire(entry_id, turn, reason)` — explicit closure, requires a reason,
  refuses to retire an already-used or already-retired thread, refuses an
  unobserved id. Same discipline as `hypothesis_board.rule_out`.

and three read-only queries: `status(entry_id)` (`open`/`used`/`retired`),
`open_threads()` (id → age for everything neither used nor retired, age
measured against the highest turn any operation has seen), `retirements()`
(the ordered list of explicit closures — this is the "not silence" half).
A used-after-retired thread reports `used`: a later real conclusion citing a
chain through an earlier retirement overrides it, and the retirement event
itself stays on the record either way (tested in the module's self-check).

**`chat_core.py`**: `ChatCore` now owns a live `self.chekhov`. `observe()`
registers every entry as it's written; `conclude()` calls `chekhov.use()`
only when the move actually lands as `reveal` (matching the existing
abstain/board-untouched rule exactly); a new `ChatCore.retire(entry_id, turn,
reason)` passthrough is the one operation with no automatic trigger, because
retirement is a narrative decision nothing else can infer. Self-check
extended to walk all three states (open after a blocked reveal, used after a
grounded one, explicitly retired) on the existing four-suspect scenario.

**`metrics.py`**: `unresolved_threads` now delegates to a new
`_replay_chekhov(records)` helper that rebuilds a `ChekhovLedger` from a
saved log (replacing the hand-rolled closure it used to do itself) —
satisfying the "when it lands, this should read that instead of re-deriving
it" note in its own docstring. New `retired_threads(records)` surfaces the
explicit-closure list. `chat_metrics()`/`chat_summary()` gained a `retired`
field/line. `chat_record()` gained an optional `chekhov=` kwarg (default
`None`, existing 3-arg call sites unaffected) that writes the cumulative
retirement list into each row, same idiom as the `ledger` field already
does. `_demo_chat`'s scripted conversation now retires the `rumour` thread
(assumed evidence from turn 1, never cited) on the final turn, so the
committed `chat_session.jsonl` carries one real example of both halves: three
threads still open with ages (`saw_margaret`: 5, `butler_key`: 2, `hunch`:
1) and one explicitly retired (`rumour` at turn 5, with its reason) instead
of just aging forever. Regenerated `chat_session.jsonl` to match (the
self-check compares it byte-for-byte and would have caught a stale copy).

## Evidence

- `python3 chekhov_ledger.py` — new self-check: age counting, all three
  status transitions, retire's four rejection paths (no reason, unknown id,
  already used, already retired), and the used-overrides-retired edge case.
- `python3 chat_core.py` — extended self-check: open/used/retired all
  demonstrated on the same scripted 4-suspect conversation already there.
- `python3 metrics.py` — extended self-check: hand-built rows proving
  `unresolved_threads`/`retired_threads` agree and are mutually exclusive,
  plus the scripted chat log's retirement of `rumour` matched exactly
  (entry, turn, reason) against `chat_metrics()["retired"]` and
  `retired_threads()`.
- Full sweep: every top-level `*.py` and everything in `backends/` runs
  clean standalone; `uv run pytest` passes (1 test, the scaffold placeholder
  — this repo's real tests are the self-checks per `CLAUDE.md`).

## Scope notes / what I deliberately did not do

- Did not touch `turn.py`'s `run_turn` or `TurnOutput` — nothing in this
  bead's AC needed a new field there, and `turn.py` already carries the
  3.9/3.10 trait-evidence wiring; adding to its surface area wasn't
  necessary to satisfy "a thread shows up with its age" + "retirement is an
  entry, not silence."
- Did not make retirement automatic on `hypothesis_board.rule_out` (e.g.
  auto-retiring evidence that only supported a ruled-out hypothesis). That
  would be inferring a narrative decision from a structural one, which is
  exactly the kind of silent inference this bead's own AC is against —
  retirement stays an explicit call.
- Did not attempt "which evidence answered which question" (that gap is
  explicitly filed as its own bead, `narrator-5ob`, which this bead blocks
  and which can now proceed).

## State

Working tree at close: `chekhov_ledger.py` (new), `chat_core.py` (modified),
`metrics.py` (modified), `chat_session.jsonl` (regenerated). Untouched
uncommitted changes from other sandbox sessions (`turn.py`,
`backends/ocean_fable.py`, `trait_evidence.py`, etc. — bd shows these as
already-closed beads 3.9/3.10/opf not yet committed) were left as found.
Per the conservative git profile, nothing was committed or pushed. `bd
close narrator-c5b.3.6` run after this handoff was written.

Suggested next commands (not run):
```
git add chekhov_ledger.py chat_core.py metrics.py chat_session.jsonl
git commit -m "..."
```
