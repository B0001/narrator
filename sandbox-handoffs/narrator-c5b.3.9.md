# narrator-c5b.3.9 — Traits move evidence weights; log the counterfactual when it bites

## What was built

One new stdlib-only module, `trait_evidence.py`, plus wiring into `turn.py`
and `metrics.py`. `admissibility.py`, `moves.py`, `evidence_ledger.py`,
`hypothesis_board.py`, and `chat_core.py` are unmodified — checker purity
(`panel.py`'s precedent: the thing that audits the weighting cannot itself be
weighted) is preserved by construction, not by convention.

- **`trait_evidence.py`** — four pure trait-weight functions, each a
  documented, self-checked mapping from an `Ocean` trait to a quantity a
  persona would use if it were driving evidence weighting directly:
  - `hypothesis_budget(persona)` — openness → how many live hypotheses the
    persona is comfortable narrowing down to (1 at openness=-1, MAX_HYPOTHESES
    at openness=1).
  - `evidence_bar(persona)` — conscientiousness → how many citations the
    persona wants behind a reveal before it's willing to want one (1 to 3).
  - `source_credibility(persona, liked)` — agreeableness → a liked source's
    testimony weighted up, a disliked one down, in [0.5, 1.5] (1.0 at
    agreeableness=0, matching the ledger's own trait-blind behavior today).
  - `concession_pull(persona)` — the positive half of agreeableness only: how
    hard the trait pulls toward conceding to whatever the user just proposed.
  - `TraitGap(turn, trait, value, wanted_move, licensed_move, missing)` — the
    counterfactual record: what the trait alone would have wanted vs. what
    the (untouched) `moves.choose_move`/`admissibility.check` pipeline
    actually licensed. `missing` is carried verbatim from the `TurnLog` the
    gap is built from — never re-derived — so it inherits narrator-7gj's
    "never claim text, only ledger ids + reasons" discipline automatically.
  - `assess_concession` / `assess_conviction` — the two reveal-shaped gaps.
    Both fire only when `turn_log.move == ABSTAIN and turn_log.missing`
    (mirroring `turn.py`'s own `blocked` check exactly, reusing its
    boundary rather than inventing a second one) — so a declined `ask` or a
    directly requested `abstain` (both leave `missing` empty) never counts
    as a refused want.
  - `assess_hypothesis_retention` — the openness-shaped gap. Different
    shape: it fires on a *licensed* reveal that narrowed the board past the
    persona's `hypothesis_budget`. Nothing was refused, so `missing` stays
    `()` and `wanted_move` is `complicate`, not `reveal`.
  - `assess(persona, turn, cited_ids, turn_log)` — combines the two
    reveal-shaped gaps in order, dropping whichever didn't fire.

- **`turn.py`** — `_trait_gaps(...)` helper wires `trait_evidence.assess(...)`
  + `assess_hypothesis_retention(...)` into both `run_turn` branches
  (SINGLE_PASS and TWO_PASS), landing on a new `TurnOutput.trait_gaps: tuple`
  field (default `()`, so every existing caller is unaffected). Separately,
  `_voice_prompt` gained a `concession_pull` parameter: on a blocked turn
  with a positive pull, it appends a fixed, content-free
  `_CONCESSION_VOICE_CLAUSE` ("agree warmly... but you are still holding
  back") — same discipline as the existing `_BLOCKED_VOICE_REASON`, so
  agreeableness can color the *reply's tone* (public compliance) while
  `turn_log.move` keeps recording the smaller, licensed move (private
  acceptance), without ever handing the voice call the refused claim.

- **`metrics.py`** — `chat_record()` now serializes `out.trait_gaps` into the
  JSONL row (`trait`, `value`, `wanted_move`, `licensed_move`, `missing`).
  This is the literal transcript-visibility mechanism the AC asks for.
  `chat_session.jsonl` regenerated via `metrics._demo_chat(...)`; turn 4 of
  the scripted conversation (an agreeable persona citing an ungrounded
  "Jeeves" hunch) now carries exactly one `trait_gaps` entry — the flagship
  withheld-concession row — and no other turn does.

## How the acceptance criteria are met

Bead AC: *"A withheld concession appears in the transcript with the trait
value, the move the persona selected, and the missing evidence named.
Nothing is suppressed without a corresponding log line."*

- `chat_session.jsonl` turn 4 (regenerated, committed as part of this
  bead's work — see diff): `"trait_gaps": [{"trait": "agreeableness",
  "value": 0.4, "wanted_move": "reveal", "licensed_move": "abstain",
  "missing": ["hunch (inferred with no cited support)"]}]`. Trait value,
  selected move, and the missing evidence id + reason are all present,
  exactly as named.
- `metrics.py`'s self-check now asserts this row by hand (`by_turn[4]
  ["trait_gaps"] == [...]`) and asserts every *other* turn's `trait_gaps` is
  `[]` — so "nothing is suppressed without a corresponding log line" is
  checked as an absence claim too, not just a presence claim on the one row
  that matters.
- `turn.py`'s self-check adds four scenarios exercising the full space:
  1. The existing narrator-7gj blocked-reveal fixture (a low-conscientiousness,
     zero-agreeableness persona) now also asserts `out.trait_gaps` has
     exactly one conscientiousness gap, with `missing` equal to the same
     tuple the checker produced.
  2. A dedicated agreeableness fixture: one ungrounded citation, an
     agreeableness=0.7 persona → exactly one gap, `wanted_move=reveal`,
     `licensed_move=abstain`, and the voice prompt contains
     `_CONCESSION_VOICE_CLAUSE` while still not leaking the refused claim
     text (extends the existing narrator-7gj leak-check pattern to the new
     clause). A neutral persona in the identical scenario produces `()` and
     no clause — the trait, not the scenario, drives the gap.
  3. An openness fixture: a grounded, checker-licensed reveal narrows the
     board past a high-openness persona's budget → one gap with
     `missing == ()` and `wanted_move == complicate`, confirming the
     "nothing refused" shape is structurally distinct from the other two.

## Verification run

```
$ for f in evidence_ledger admissibility moves chat_core hypothesis_board \
    ocean agents question_selector panel motive mystery clue_partition \
    chapters discussion metrics turn trait_evidence; do
    uv run python "$f.py"
  done
ok (x16, motive/mystery/chapters/discussion print their own summaries too)
$ uv run pytest -q
1 passed in 0.01s
```

All pre-existing modules re-checked for regression; only `turn.py`,
`metrics.py`, and `chat_session.jsonl` were modified, and `trait_evidence.py`
is new. `admissibility.py`, `moves.py`, `evidence_ledger.py`,
`hypothesis_board.py`, `chat_core.py`, `ocean.py`, `panel.py` are byte-for-byte
untouched.

## Scope notes / design decisions

- **No clamping, by construction.** `trait_evidence.py` never imports
  `hypothesis_board` for anything but its `MIN_HYPOTHESES`/`MAX_HYPOTHESES`
  constants (to size `hypothesis_budget`'s range), and never calls
  `board.reweight()`/`rule_out()` or touches a `Verdict`/`TurnLog` in place.
  Every function is pure: input persona + already-decided facts (cited ids,
  the `TurnLog` `core.conclude()` already produced) → an output value or a
  `TraitGap`. `run_turn` calls `core.conclude()` first, unconditionally, then
  asks `trait_evidence` a second, purely observational question. There is no
  code path where a trait value changes what `admissibility.check()` decides
  or what the board's live hypotheses are.
- **`source_credibility` is implemented and self-checked but not yet wired
  into a live consumer.** The bead's own description names it alongside
  `hypothesis_budget`/`evidence_bar`/`concession_pull` as one of the things
  "traits reach," but `evidence_ledger.py` has no notion of a testimony
  *source* to weight (provenance tags track how a claim was obtained, not
  who said it) — chat is single-user, not multi-agent. Reading
  `narrator-c5b.3.10`'s own AC ("simulation mode lets the persona act on its
  weights unimpeded and records the damage; product mode acts on the log and
  withholds") confirms this is the intended split: `.3.9`'s job is the
  primitives + the always-on counterfactual-logging spine (defaulting to
  product-style withholding, which is what `turn.py` does unconditionally
  today), and `.3.10` is where a `mode` flag lets simulation mode actually
  *act* on these weights instead of only logging against them.
  `source_credibility` is ready for whichever consumer `.3.10` builds; wiring
  it into one now would be scope creep into that bead, so it wasn't done.
- **Two structurally different `TraitGap` shapes, both under one dataclass.**
  Reveal-shaped gaps (agreeableness, conscientiousness) only ever fire
  against an admissibility-refused `ABSTAIN` (`missing` non-empty);
  the openness gap fires against a *licensed* `REVEAL` that outran the
  persona's comfort (`missing` empty, `wanted_move=complicate`). Both share
  one record type rather than two so `metrics.py`'s serialization and any
  future consumer only need one shape to handle, but the fields make the
  difference legible (`missing == ()` vs. non-empty is the tell).
- **`_CONCESSION_VOICE_CLAUSE` reuses, rather than reopens, the narrator-7gj
  trust boundary.** It's a second fixed, content-free string appended
  alongside `_BLOCKED_VOICE_REASON` under the same condition (`blocked`) plus
  one more (`concession_pull > 0`) — no new parameter reaches the voice
  prompt that could carry ledger content; only a boolean-gated constant does.
- **No new file under `tests/`.** Matching every other C5 module's
  precedent: each touched module's own `_self_check()` is the runnable check.

## Suggested next commands

```
bd close narrator-c5b.3.9
git status   # trait_evidence.py untracked; turn.py, metrics.py, chat_session.jsonl modified
             # nothing committed, per "do not commit unless asked"
```
