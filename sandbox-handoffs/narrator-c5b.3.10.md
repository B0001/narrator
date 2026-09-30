# narrator-c5b.3.10 — Mode flag: simulation and product read the sweep opposite ways

## What was built

An addition to `trait_evidence.py` (the module `.3.9` built), plus wiring
into `turn.py` and `metrics.py`. `admissibility.py`, `moves.py`,
`evidence_ledger.py`, `hypothesis_board.py`, and `chat_core.py` remain
byte-for-byte untouched — the checker still never learns which mode is
active, and still never departs from what the ledger licenses.

- **`trait_evidence.py`** — new `SIMULATION`/`PRODUCT` constants, a
  `Damage` dataclass (`turn`, `trait`, `value`, `move`, `missing` — one row
  per turn a trait's want overrode the checker's refusal), an `Actuation`
  dataclass (`move`, `damage` — what actually happens this turn under one
  mode), and `TraitMode`:
  - `TraitMode(mode)` — no default; `mode` must be `SIMULATION` or
    `PRODUCT` or it raises `ValueError` naming the bad value. Calling
    `TraitMode()` with no argument raises `TypeError` (Python's own
    required-positional-argument behavior — nothing bespoke needed for that
    half of the AC).
  - `TraitMode.actuate(persona, turn, cited_ids, turn_log)` — recomputes
    `assess_concession`/`assess_conviction` (unchanged from `.3.9`; the
    openness/`assess_hypothesis_retention` gap is deliberately excluded, see
    Scope notes) against the *already-decided* `turn_log`. Under `PRODUCT`,
    or when no reveal-shaped gap fired, returns `Actuation(turn_log.move,
    ())` — identical to pre-`.3.10` behavior byte-for-byte. Under
    `SIMULATION`, if a gap fired, returns `Actuation(gap.wanted_move,
    damage)` where `damage` has one `Damage` entry per gap that fired,
    letting the trait's want (always `REVEAL`, the only move a gap ever
    wants that the checker didn't already grant) become the effective move
    for the turn.

- **`turn.py`** — `run_turn(core, persona, turn, user_message, generate_fn,
  *, trait_mode, model=..., mode=TWO_PASS)`: `trait_mode` is now a required
  keyword-only parameter with no default, so a caller cannot silently get a
  guess in either direction. `core.conclude()` still runs unconditionally
  in both branches (SINGLE_PASS and TWO_PASS) — `turn_log` is the checker's
  verdict and does not move with the mode. Both branches then call
  `trait_mode.actuate(...)` and populate two new `TurnOutput` fields:
  `effective_move` (what actually happened) and `damage` (the override
  record, `()` under PRODUCT or when nothing fired). In TWO_PASS,
  `_voice_prompt` gained an `unimpeded_reveal` parameter: when `actuate()`
  produced damage, the voice is told it "spoke with full confidence, citing
  <ids>" instead of receiving `_BLOCKED_VOICE_REASON` — the same ids-only
  shape a checker-*licensed* reveal's own reason already exposes, so this
  does not reopen narrator-7gj (the refused claim's sentence, and any
  assumed-premise text, still never reach the voice; only entry ids do,
  exactly as before).

- **`metrics.py`** — `chat_record()` now serializes `out.effective_move` and
  `out.damage` (`trait`, `value`, `move`, `missing` per damage row) into the
  JSONL transcript row, following the `trait_gaps` precedent. `_demo_chat()`
  and the SINGLE_PASS self-check scenario now construct and pass a
  `trait_evidence.TraitMode(trait_evidence.PRODUCT)` (the mode the scripted
  demo conversation has always implicitly run under). `chat_session.jsonl`
  regenerated via `metrics._demo_chat(...)` — every row picked up the two
  new keys (`effective_move` equal to `move`, `damage` equal to `[]`, on
  every one of the six turns, including turn 4's agreeableness gap, since
  the demo runs in PRODUCT mode).

## How the acceptance criteria are met

Bead AC: *"Simulation mode lets the persona act on its weights unimpeded
and records the damage. Product mode acts on the log and withholds.
Constructing without a mode raises rather than defaulting."*

- **"raises rather than defaulting"** — `TraitMode()` → `TypeError`
  (missing required positional argument); `TraitMode("wing it")` →
  `ValueError: not a mode: 'wing it'; must be one of ['product',
  'simulation']`. `run_turn(...)` without `trait_mode=` → `TypeError`
  (missing required keyword-only argument). Both checked in
  `trait_evidence.py`'s and (implicitly, by every call site needing the
  argument) `turn.py`'s self-checks.
- **"product mode acts on the log and withholds"** — every one of
  `turn.py`'s ~19 pre-existing `run_turn(...)` self-check call sites now
  passes `trait_mode=product`, and the self-check still passes unchanged —
  PRODUCT mode is provably behavior-identical to pre-`.3.10` `run_turn`. A
  dedicated assertion on the flagship conscientiousness scenario
  (`turn.py` `_self_check`, narrator-7gj block) checks `out.effective_move
  == out.turn_log.move and out.damage == ()` explicitly. `metrics.py`'s
  self-check separately asserts this across the full six-turn scripted
  demo: `by_turn[t]["effective_move"] == by_turn[t]["move"]` and
  `by_turn[t]["damage"] == []` for every turn, including turn 4's
  agreeableness gap.
- **"simulation mode lets the persona act on its weights unimpeded and
  records the damage"** — `turn.py`'s self-check adds three SIMULATION-mode
  scenarios, one per trait shape:
  1. The conscientiousness flagship (a blocked two-citation reveal),
     replayed under `trait_mode=simulation`: `out.turn_log.move` is still
     `ABSTAIN` (the checker's own verdict, unmoved by mode) but
     `out.effective_move == moves.REVEAL` and `out.damage` carries one
     `Damage(trait="conscientiousness", move=REVEAL, missing=<the
     checker's own missing tuple>)`. The voice prompt now contains the
     cited ledger ids (`"hunch"`, `"premise"`) instead of
     `_BLOCKED_VOICE_REASON` — while the refused claim's actual sentence
     and the assumed premise's text still do not appear anywhere in it
     (narrator-7gj re-checked under the new code path, not just assumed to
     still hold).
  2. The agreeableness flagship (a single ungrounded citation, a
     persona with agreeableness=0.7), replayed the same way: same
     override/damage/voice-content shape, `damage[0].trait ==
     "agreeableness"`.
  3. The openness scenario (a *licensed* reveal that outran the persona's
     `hypothesis_budget`): replayed under SIMULATION on a fresh board (the
     PRODUCT run already consumed the only `rule_out` this fixture
     supports) and asserted identical to PRODUCT — `effective_move ==
     turn_log.move == REVEAL`, `damage == ()`. This is deliberate: openness's
     gap has no checker refusal to override, so SIMULATION has nothing to
     act on here, and the AC's "unimpeded... and records the damage" only
     describes the two reveal-shaped gaps in the first place.

## Verification run

```
$ for f in ocean motive agents metrics mystery evidence_ledger \
    hypothesis_board admissibility moves chat_core trait_evidence turn; do
    uv run python "$f.py"
  done
ok (x11; motion/metrics/mystery print their own summaries too)
$ uv run pytest -q
1 passed in 0.00s
```

All pre-existing self-checking modules re-run for regression; only
`turn.py`, `metrics.py`, and `chat_session.jsonl` were modified this
session, and `trait_evidence.py` (already new/untracked from `.3.9`, not
yet committed) gained the mode-flag addition. `admissibility.py`,
`moves.py`, `evidence_ledger.py`, `hypothesis_board.py`, `chat_core.py`,
`ocean.py` are byte-for-byte untouched by this bead.

## Scope notes / design decisions

- **The checker never sees the mode, by construction.** `core.conclude()`
  (and therefore `admissibility.check()`) is called identically in both
  `run_turn` branches regardless of `trait_mode` — `trait_mode.actuate(...)`
  is only ever consulted *after* `turn_log` already exists, using it as a
  read-only input. This keeps the repo's foundational invariant intact:
  "the checker cannot see the producer's internals" now also reads as "the
  checker cannot see which success-criteria mode the producer is running
  under" — a mode that could leak into `admissibility.check()` would let
  simulation mode manufacture its own admissible reveals instead of
  overriding a real refusal, which is a different (and much weaker) bug
  than the one this bead asks for.
- **The board is never mutated by an override.** `TraitOutput.effective_move`
  and `.damage` are an observability layer, not a second write path into
  `ChatCore`. `core.conclude(...)` is always called with the *requested*
  move (from the reasoning/decision channel), never the trait-overridden
  one, so `HypothesisBoard.rule_out` only ever fires for a checker-licensed
  reveal in both modes. This was a deliberate choice: the alternative
  (SIMULATION mode actually narrowing the board on an inadmissible reveal)
  would require `run_turn` to call `conclude()` a second time with a
  different move, doubling the ledger/board write surface for a turn and
  making "what did the board actually see" ambiguous. The bead's AC talks
  about the persona "acting... and the damage" being recorded, which the
  `Damage` record + overridden `effective_move`/voice content already
  satisfies without that risk. `.3.11`'s sweep (which this bead blocks) can
  measure the override rate and voice-content directly from
  `effective_move`/`damage`; if it turns out to need board-level
  consequences too, that's server for a new bead, not a retrofit here.
- **Only the two reveal-shaped gaps are actuatable.** `assess_concession`
  and `assess_conviction` both fire against a checker *refusal*
  (`turn_log.move == ABSTAIN`, `missing` non-empty) with `wanted_move ==
  REVEAL` — there is an obvious "unimpeded" action to take: let the reveal
  happen anyway. `assess_hypothesis_retention` (openness) fires against a
  checker *licensed* reveal that already happened; there is no refusal to
  override, only a "the board narrowed further than I'd have liked"
  observation with `wanted_move == COMPLICATE`. Reading `wanted_move ==
  COMPLICATE` as an instruction to *un-narrow* the board defies the
  ledger/board's append-only, monotonic-narrowing design (`HypothesisBoard`
  has no "re-add a ruled-out hypothesis" operation, and shouldn't grow one
  just for this). `TraitMode.actuate` therefore only ever consults the two
  reveal-shaped gaps; openness's gap remains purely observational (via the
  unchanged `_trait_gaps` helper) in both modes, confirmed identical by the
  openness self-check scenario above.
- **`_voice_prompt`'s `unimpeded_reveal` reuses the existing ids-only
  vocabulary, not new content.** The override reason string ("citing
  <ids>, and speaking with full confidence") deliberately mirrors
  `moves.choose_move`'s own REVEAL reason shape (`"citing <ids>, all
  grounded..."`) rather than inventing new leak-prone phrasing — the
  ledger entry ids a SIMULATION-mode override exposes to the voice are
  exactly the ids a checker-*licensed* reveal already would have exposed
  in the exact same call; only the refused claim's own sentence (and any
  assumed-premise text) is still withheld, matching narrator-7gj.
- **`chat_record()`'s new keys cost a PRODUCT transcript nothing.** Every
  row still shows `effective_move == move` and `damage == []`; the fields
  only start diverging once a caller actually runs `_demo_chat`-shaped
  code under `SIMULATION`, which nothing in this repo does yet (no bead
  asked for a simulation-mode demo transcript; `.3.11`'s sweep is the
  first consumer, and it's free to build its own scripted conversation
  under `trait_mode=simulation` rather than reusing `_demo_chat`'s
  PRODUCT-flavored fixture).
- **No new file under `tests/`.** Matching every other C5 module's
  precedent: each touched module's own `_self_check()` is the runnable
  check.

## Suggested next commands

```
bd close narrator-c5b.3.10
git status   # trait_evidence.py untracked (carried over from .3.9, still uncommitted);
             # turn.py, metrics.py, chat_session.jsonl modified this session
             # nothing committed, per "do not commit unless asked"
```
