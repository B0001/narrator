# narrator-5ob: question grounding, closed

## State found

The bead was already `in_progress` (claimed 2026-09-27) with no notes and no
handoff. The working tree already contained a complete, working
implementation, uncommitted:

- `evidence_ledger.py`: `LedgerEntry.prompted_by` (id of the question that
  drew an entry out, or `None`), threaded through `EvidenceLedger.write` and
  `load`, with write-time validation (blank string rejected — `None` is the
  "nothing prompted this" value).
- `question_selector.py`: `question_id(selection_log)` builds the id as
  `f"{turn}:{chosen}"` — turn-prefixed because a bare candidate id is only
  unique within one generation call.
- `chat_core.py`: `ChatCore.observe` takes `prompted_by` and passes it
  straight through to the ledger; `ChatCore` itself tracks no notion of "the
  outstanding question" (a chat can ask, get no answer, ask something else,
  and have the user circle back later — that bookkeeping is the caller's).
- `metrics.py`: `question_grounding(records)` walks the same supports-closure
  `unresolved_threads`/Chekhov ledger already walk, asking whether any entry
  stamped with a chosen question's id is reachable — directly or
  transitively — from a landed reveal's citations. `chat_metrics` folds it
  in (`questions_grounded`, `question_grounding_rate`,
  `grounded_question_ids`) and `chat_summary` prints it as `answer grounded a
  reveal` next to the existing `question yield`/`user answered` lines.

I verified rather than assumed: ran every module's self-check
(`uv run python <module>.py` for all top-level `.py` files and everything
under `backends/`) and `uv run pytest`, all green. I did not find a case the
existing self-checks missed.

## What I did this session

1. Confirmed the implementation actually satisfies the bead's acceptance
   criteria, clause by clause:
   - "A chat log carries, for each ledger entry, the id of the question that
     prompted it (or `None`)" — `LedgerEntry.prompted_by`, round-trips
     through JSONL (`evidence_ledger.py` self-check, lines ~155-203).
   - "metrics.py reports how often a chosen question's elicited evidence
     appears in the citation chain of a later landed reveal" —
     `question_grounding` (`metrics.py:415`), reachable via `chat_metrics`
     and printed in `chat_summary`.
   - "a self-check covers a question whose answer grounds a reveal and one
     whose answer never gets used" — `metrics.py` self-check has both:
     `grounded_rows` (answer cited by a landed reveal, `grounded: 1`) and
     `unused_rows` (answer written, never cited, `grounded: 0`) around line
     908-928, plus a third case proving grounding walks `supports`
     transitively, not just direct citations.
2. Fixed one stale doc comment: `evidence_ledger.py`'s `prompted_by` field
   comment pointed at a nonexistent `turn.question_id_for`; the real
   function is `question_selector.question_id` (the module docstring above
   it already said so correctly). Corrected the inline comment to match.
3. Re-ran the full self-check sweep (every top-level module +
   `backends/*.py`) and `uv run pytest` after the fix — all pass.

## What I did not touch

The working tree also carries substantial uncommitted work unrelated to this
bead (`trait_evidence.py`, `trait_role_conditions.py`,
`agreeableness_sweep.py`, `chekhov_ledger.py` itself, and diffs across
`turn.py`/`ocean.py`/`backends/`/etc. spanning several other closed beads
like narrator-c5b.3.6 through 3.10). None of it conflicts with or depends on
narrator-5ob's change beyond the already-closed Chekhov ledger dependency, so
I left it as-is — it is not this bead's scope to commit or clean up, and the
CLAUDE.md profile here is conservative (no commit/push without explicit
authority).

## Follow-up

None needed for this bead. If a future session wants to actually exercise
`prompted_by` end-to-end from a live conversation (rather than
self-check-constructed `ChatCore.observe` calls), that would need a real chat
driver loop threading `question_selector.question_id(selection_log)` into
the `observe()` call when the user's next message answers an outstanding
ask — no such driver exists yet in this repo (everything is exercised
through self-checks), and building one is out of scope here.

## Validation

```
uv run pytest                         # 1 passed
uv run python evidence_ledger.py      # ok
uv run python question_selector.py    # ok
uv run python chat_core.py            # ok
uv run python metrics.py              # prints chat summary table incl. "answer grounded a reveal"
uv run python turn.py                 # ok
# full sweep: every top-level *.py and backends/*.py self-check passes
```

## Git status at handoff

Working tree still has the same uncommitted changes as at session start
(this bead's work included), plus the one-line comment fix in
`evidence_ledger.py`. Nothing committed or pushed — conservative profile,
no explicit authority given to do either.
