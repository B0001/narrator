# narrator-c5b.3.8 — Two-channel turn: persona-neutral reasoning, persona voice

## What was built

One new stdlib-only module, `turn.py`, sitting on top of the already-closed
`chat_core.py`/`moves.py`/`admissibility.py`/`evidence_ledger.py` stack. It
adds the piece those modules deliberately left out: the actual model calls a
turn needs. Everything below it stays model-agnostic (`ChatCore.conclude()`
still just takes a move and citations from whoever calls it); `turn.py` is
the "whoever."

- **`ReasoningProfile`** — an `Ocean` subclass with every trait left at the
  `Ocean()` default (so `system_prompt()` renders the persona-neutral
  "Respond neutrally, with no pronounced personality" text unconditionally),
  and `options()` overridden outright to a fixed `{"temperature": 0.2, ...}`
  rather than computed from traits — so it can't drift back toward
  persona-driven sampling even if someone later constructs it with non-zero
  fields. `REASONING_PROFILE` is the module-level singleton used for every
  reasoning call.
- **`run_turn(core, persona, turn, user_message, generate_fn, model=..., mode=...)`**
  — two modes:
  - `TWO_PASS` (default): call 1 uses `REASONING_PROFILE` to decide the move
    + citations (JSON: `move`, `cited`, `rule_out`); call 2 uses the real
    `persona`'s `system_prompt()`/`options()` to render the chosen move in
    character. `ChatCore.conclude()` — and therefore
    `admissibility.check()` — runs on the decision from call 1 exactly as it
    would from any other caller.
  - `SINGLE_PASS`: one call, using `persona` throughout, that returns a
    combined JSON (`move`, `cited`, `rule_out`, `reply`) — the same
    admissibility-gated `conclude()` path runs on citations that were
    decided at the *persona's* sampler settings, not a neutral one. This
    mode isn't a deprecated fallback; it's kept specifically so the
    acceptance criteria's "watch the difference" is something a self-check
    (or a learner) can point at directly.
  - `Call` (label, profile, options, prompt, raw) is recorded for every
    model invocation in `TurnOutput.calls`, so "which profile's options
    reached which call" is inspectable data, not a claim in a docstring.
  - `generate_fn` follows `ocean.generate`'s shape (`profile, prompt,
    model=...`), matching `agents.converse`'s injectable pattern — the real
    Ollama backend and the self-check's stub are interchangeable.

## How the acceptance criteria are met

Bead AC: *"Sampler settings from a profile reach the voice call only. A
single-pass mode stays available so a learner can watch the difference."*

`turn.py`'s self-check asserts this directly, not just that some fixed
number was used somewhere:

1. **Two-pass, same reasoning options across different personas.** The same
   scenario is run once with a highly neurotic/low-conscientiousness persona
   and once with a highly disciplined one. Both times the reasoning call's
   recorded `options()` are captured; the test asserts they are *identical*
   to each other and equal to `REASONING_PROFILE.options()` — not just "some
   low number," but the same fixed dict regardless of who's speaking. The
   voice call's options are asserted to equal that persona's own
   `options()`, and the fake `generate_fn` echoes the temperature it
   received back into the reply text so the test can check the persona's
   real temperature actually reached that call.
2. **Single-pass, persona options reach the one call that both decides and
   speaks.** The fake `generate_fn` asserts it was handed the persona object
   itself (not `REASONING_PROFILE`), the recorded call's `options` equal
   `persona.options()`, and that temperature differs from the fixed
   reasoning temperature — demonstrating the exact contrast the bead is
   asking a learner to be able to see.
3. **Admissibility gating is unconditional, not mode-dependent.** In
   two-pass mode, an ungrounded citation triggers the same
   reveal-to-abstain downgrade `moves.py`/`chat_core.py` already test, board
   untouched; then grounding the same conclusion (an `observed_artifact`
   entry backs the inference) lets the identical shape of reveal go
   through and narrow the board. This is the same scenario shape as
   `chat_core.py`'s own self-check, run through the new two-call path
   instead of a caller supplying the move directly.
4. **Malformed reasoning output is a named failure.** Non-JSON from the
   reasoning channel raises `ValueError` naming what was actually returned,
   not a raw `JSONDecodeError` pointing at a character offset. A
   single-pass response missing `reply` is rejected the same way.
5. **An unknown mode is rejected outright**, matching the `moves.py`/
   `admissibility.py` convention of failing loud on an unrecognized enum
   value rather than silently defaulting.

## Verification run

```
$ uv run python evidence_ledger.py && uv run python admissibility.py \
  && uv run python moves.py && uv run python chat_core.py \
  && uv run python hypothesis_board.py && uv run python ocean.py \
  && uv run python agents.py && uv run python turn.py
ok
ok
ok
ok
ok
ok
ok
ok
$ uv run pytest -q
1 passed in 0.00s
```

All pre-existing modules re-checked for regression; none touched by this
bead (`moves.py`, `admissibility.py`, `evidence_ledger.py`, `chat_core.py`,
`hypothesis_board.py`, `ocean.py` are unmodified — `turn.py` is additive
only). `motive.py`/`mystery.py`/`chapters.py` still fail with
`ModuleNotFoundError: No module named 'networkx'` — this is the
pre-existing, already-filed `narrator-3c5` gap in `pyproject.toml`, not
something this session touched.

## Scope notes / design decisions

- **Why an `Ocean` subclass instead of a bare dict of options.** `ocean
  .generate()` (and `agents.converse`'s `generate_fn` convention) expects
  something with `system_prompt()` and `options()`. Making `ReasoningProfile`
  a real `Ocean` subclass means the exact same `generate_fn` that would call
  a live Ollama backend for the voice pass also works unmodified for the
  reasoning pass — no parallel "options-only" call path to keep in sync with
  the real one. Overriding `options()` outright (rather than picking trait
  values that happen to compute a low temperature through the existing
  formula) is deliberate: reverse-engineering neutral-looking trait values
  out of `0.7 + 0.4*n - 0.3*c` would still leave the reasoning temperature
  formally *derived from* a personality vector, which is exactly the
  coupling this bead exists to break.
- **`ChatCore.conclude()` is untouched and unconditionally in the loop.**
  Both modes route the decided move through the exact same admissibility
  check `chat_core.py`/`moves.py` already had. This bead does not add a
  second gate or a mode-specific bypass — the only thing that varies by mode
  is which profile's sampler settings produced the citations being checked.
- **Single-pass is a first-class mode, not a deprecated code path.** The
  bead's AC explicitly asks for it to "stay available," and the
  `Call`-recording design exists specifically so a real run (or the
  self-check) can diff the two modes' recorded options side by side.
- **Deliberately not solved here, per the bead's own scope note:** trait
  effects on evidence weighting (`narrator-c5b.3.9`), the simulation/product
  mode flag (`.3.10`), and the checker-subagent-carries-no-profile
  constraint (`.3.12`). All three are already filed as children of the
  parent bead and depend on this one; nothing in `turn.py` tries to weight
  evidence by trait or pick a mode automatically — `mode` is an explicit
  caller-supplied argument with no default inference from persona.
- **No new file under `tests/`.** Matching every other C5 module's
  precedent: `turn.py`'s own `_self_check()` is the runnable check the
  working rules ask for.

## Suggested next commands

```
bd close narrator-c5b.3.8
git status   # turn.py is untracked — nothing committed, per "do not commit unless asked"
```
