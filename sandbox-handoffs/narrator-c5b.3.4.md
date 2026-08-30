# narrator-c5b.3.4 — Discriminating-question selector

## What was built

`question_selector.py`, a standalone module (depends only on
`hypothesis_board.HypothesisBoard`'s public `dump()`/live-id surface),
following the repo's module convention: docstring-first, no external test
framework, `if __name__ == "__main__": _self_check()` at the bottom.

- `Candidate(id, text, predicted_answers)` — one candidate clarifying
  question, tagged with what every *live* hypothesis would predict the
  answer to be (`predicted_answers`: hypothesis id -> predicted answer).
- `score_candidate(board, candidate)` — weighted Gini impurity of the
  predicted answers over the board's live hypotheses, using each
  hypothesis's *current* board weight, not a bare count of distinct answers.
  0.0 exactly when every live hypothesis predicts the same answer (the
  "boring for everyone" case); approaches 1.0 as predictions split cleanly
  across hypotheses of comparable current weight. Returns a
  `ScoredCandidate(id, text, score, discriminates, reason)` — `reason` names
  either "every live hypothesis predicts the same answer" or the answer
  groups and their weight mass, so the score is legible without re-deriving
  it.
- `select_question(board, turn, candidates)` — scores every candidate
  (logging all of them, not just the winner), then picks the
  highest-scoring one that actually discriminates (`score > 0`). Returns a
  `SelectionLog(turn, scored, chosen)` where `chosen` is the winning
  candidate's id or `None` if nothing on the table splits the board — that
  absence is itself the logged outcome, never a silent fallback to whatever
  candidate came first. Always picks at most one winner (`chosen` is a
  single id, never a list) — that's "cap at one question per turn."
- Raises `KeyError` if a candidate is missing a predicted answer for any
  currently-live hypothesis (a caller bug, not something to score around),
  and `ValueError` if asked to select from zero candidates.

## How the acceptance criteria are met

- **"Given a board, the selector prefers the question that splits it"** —
  self-check builds a 4-suspect board, offers three candidates (uniform
  "boring" answer, a full 4-way split, and a Columbo-shaped 3-vs-1 split),
  and asserts the full split wins with the highest score, ahead of the 3-vs-1
  split, ahead of the boring one at exactly 0.0. A second scenario reweights
  the board toward one hypothesis (`blackwood=0.7`) and shows a question that
  separates the heavyweight hypothesis from the rest outscores one that only
  tells two already-marginal hypotheses apart — i.e. the selector reads the
  board's *current* weights, not just its live-id set, which is the "given a
  board" part of the criteria (a static per-hypothesis-count scorer would
  have missed this).
- **"and rejects questions every hypothesis answers the same way"** — the
  uniform-answer candidate scores exactly `0.0` and `discriminates=False`;
  `select_question` never lets a non-discriminating candidate win even if it
  is the only one offered (`log3` in the self-check: every candidate is
  uninformative, `chosen is None`, not a fallback pick).
- **"Selection scores are logged"** — `SelectionLog.scored` carries a
  `ScoredCandidate` (score, discriminates, reason) for every candidate
  passed in, winner or not; self-check asserts on `by_id` built from the
  full `scored` tuple, not just the chosen id.

## Verification run

```
$ uv run python question_selector.py
ok
$ uv run pytest -q
1 passed in 0.00s
```

Also re-ran every other C5 module's self-check to confirm nothing was
disturbed: `evidence_ledger.py`, `hypothesis_board.py`, `admissibility.py`,
`moves.py`, `chat_core.py`, `turn.py` — all `ok`.

## Scope notes / what's deliberately not here

- **No wiring into `moves.py`/`chat_core.py`/`turn.py`.** The bead's
  acceptance criteria only exercise the selector "given a board" —
  standalone, same as `hypothesis_board.py` (`.3.2`) was standalone before
  `.3.3`'s admissibility check and `chat_core.py` wired it into the ledger.
  The natural integration point when someone wants the bot to actually *ask*
  the winning question is `moves.ASK`/`turn.py`'s reasoning prompt: today the
  reasoning channel decides `move="ask"` off model judgment alone with no
  access to a scored candidate list. Wiring that would mean generating
  candidate questions with per-live-hypothesis predicted answers (itself a
  model call, or authored data in a test harness) before the reasoning
  prompt runs, then feeding `select_question`'s `chosen` id into the prompt
  or directly into the `ask` move's reason. Not built here — no bead asked
  for it, and it's a real design decision (who generates candidates and
  their predictions?) that deserves its own scoping rather than a
  speculative stub.
- Score formula is weighted Gini impurity over live-hypothesis board weight,
  not a plain count of distinct answers. Chose this because the bead's
  framing ("given a board") and the Columbo motivating example both hinge on
  *current belief*, not just how many suspects exist — a question that only
  discriminates between two hypotheses the board has already nearly ruled
  out is worth less than one that splits the leading hypothesis from the
  rest, and only a weighted score can tell those apart. This is a design
  choice, not dictated by the bead text; an alternative (unweighted,
  one-hypothesis-one-vote) would also satisfy the letter of the acceptance
  criteria on an equal-weight board, but would give the same answer on the
  reweighted-board scenario in the self-check regardless of which candidate
  actually mattered more right now, which seemed like the wrong lesson for a
  learner to take from this module.
- Did not touch `.3.6`–`.3.11` (Chekhov ledger, metrics, traits-weighting,
  agreeableness sweep, mode flag) — all separate, still-open siblings under
  `narrator-c5b.3`.

## Suggested next command

```
bd close narrator-c5b.3.4
```
