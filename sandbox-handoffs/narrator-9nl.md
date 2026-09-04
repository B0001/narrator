# narrator-9nl — Wire question_selector + panel into the C5 turn loop

## What was built

Two acceptance-criteria branches, both addressed:

**1. `ask` move now calls `question_selector.select_question()` over real
board state, with observable-only candidate generation.**

`question_selector.py` gained three new functions, additive only (no
existing function changed):

- `candidate_prompt(board, n=3)` — builds the candidate-generation prompt
  from `board.dump()`'s *live* hypotheses (id + statement) alone. The
  function's signature is the proof: it takes a board and a count, nothing
  else — no ledger, no persona, no hidden ground truth object can reach the
  string it returns, because there is no parameter for any of those to
  travel through.
- `parse_candidates(board, raw)` — turns the model's JSON response into
  `Candidate`s, checking every live hypothesis got a predicted answer
  (raising `ValueError` immediately, not a `KeyError` later inside
  `select_question`).
- `generate_candidates(board, profile, generate_fn, model=..., n=3)` — the
  one model call, composing the two above. Follows the repo-wide
  `generate_fn(profile, prompt, model=...)` shape (`ocean.generate` /
  `agents.converse` / `panel.run_panel`'s convention), so it's interchangeable
  with a real backend or a scripted stub.

`turn.py` wires this into `run_turn`'s `TWO_PASS` branch only: when the
reasoning call's decision resolves to `moves.ASK`, `_ask_candidate_call()`
runs `generate_candidates` against `core.board` (the real, live
`HypothesisBoard`, not a mock) pinned to `REASONING_PROFILE` — the same
persona-neutral, fixed-temperature profile the reasoning call itself already
uses, on the same rationale the module's docstring already gives for the
reasoning call: deciding *which question is worth asking* is a "what
actually splits the board" judgment, not a voice one, so it must not run at
a persona's drifting sampler settings either. `select_question()`'s winner
(if any) is what actually reaches the voice prompt (`ask_text` in
`_voice_prompt`) — the model doesn't get to invent its own question when one
was already selected. The whole selection (winner and rejects both) comes
back on `TurnOutput.question_log`, a new field (default `None`, only set
when the move is `ask`) — the observable record the acceptance criteria ask
for.

`SINGLE_PASS` is deliberately **not** wired: it makes exactly one call by
design (asserted in its own pre-existing self-check), and this module exists
specifically so a learner can see what a real turn looks like *without* a
separated reasoning step. Adding a second, board-reading call there would
erase that contrast. Documented in `turn.py`'s module docstring, not just
here.

**2. `panel.run_panel()` — written decision that it stays a standalone
diagnostic.**

Added to `panel.py`'s module docstring (not just this handoff, so it's
discoverable by reading the module): `Checker.verdict()` already *is*
`admissibility.check()`, unmodified — the identical call `moves.choose_move`
runs, unconditionally, on every attempted `reveal`. Running `run_panel()`
inside `chat_core.conclude()` or `turn.run_turn` would invoke the same
checker a second time; it adds two persona voices around an unchanged
yes/no, not additional verification. It would also give the deliberately
model-agnostic `ChatCore.conclude()` a `generate_fn` dependency it doesn't
otherwise need, just to call a checker it already calls. Decision: `panel.py`
remains standalone — useful on its own terms (the propose/critique/check
contrast a self-check can run directly), not invoked from the turn loop.
A real "let the user watch proposer/critic argue before reveal" feature is a
legitimate future bead, but it's a presentation feature, not a verification
one, so it doesn't belong under this bead's "verify path" criterion.

## How the acceptance criteria are met

> An 'ask' move in the C5 loop that calls `question_selector.select_question()`
> over real board state, with candidate generation that provably reads only
> observable state

- `turn.py`'s new two-pass self-check block runs a full `ChatCore` +
  `HypothesisBoard` (the real classes, not stand-ins), scripts the reasoning
  call to decide `move: "ask"`, and asserts: the `ask-candidates` call is
  pinned to `REASONING_PROFILE` (`ask_call.profile is REASONING_PROFILE`,
  its `options()` match, `temperature == REASONING_TEMPERATURE` — never the
  persona's own drifting sampler); every live hypothesis id reaches the
  candidate prompt; the discriminating candidate (`whereabouts`, a clean
  4-way split) wins over the uninformative one (`boring`, same answer under
  every hypothesis); and the winning question's *text*, not the board or
  ledger, is what lands in the voice call's prompt.
- A second scenario rules two hypotheses out via real `core.conclude()`
  calls (not hand-set state) down to `{blackwood, ellis}`, offers only a
  candidate that's uninformative between the two survivors, and asserts
  `question_log.chosen is None` and the voice prompt never claims a
  "specific question" that doesn't exist — the selector's honest-abstention
  behavior (`chosen=None` is itself the logged outcome) survives the trip
  through the turn loop, not just in `question_selector.py`'s own
  self-check.
- "Provably reads only observable state" is checked two ways:
  1. **Signature-level**, in `question_selector.py`'s own self-check:
     `candidate_prompt`/`generate_candidates` take a board and nothing else
     that could carry hidden state.
  2. **Leak-check**, mirroring `chapters.py`'s pre-existing technique for
     `sim.culprit`: rule out a hypothesis (`jeeves`), then assert its id and
     statement never appear in `candidate_prompt`'s output or in the prompt
     a scripted `generate_fn` actually receives. A ruled-out hypothesis is
     the one piece of prior board state this function must not see, and the
     test proves it doesn't, on the actual string sent, not on a promise in
     a docstring.
  3. A malformed candidate-generator response (`"not json"`) is a named
     `ValueError` propagated all the way out of `run_turn`, same discipline
     `turn.py` already applies to a malformed reasoning-channel response.

> and either `panel.run_panel()` invoked somewhere in the verify path or a
> written decision that it stays a standalone diagnostic

- Written decision, in `panel.py`'s module docstring (see above) — not just
  this handoff.

> Turn/chat_core self-checks cover the new wiring

- `turn.py`'s self-check covers all three scenarios above (discriminating
  win, honest no-discrimination, malformed response). `chat_core.py` itself
  is untouched — the wiring reads `core.board`, `ChatCore`'s existing public
  surface, so no new method was needed there, and its own self-check was
  re-run unchanged to confirm nothing regressed.

## Verification run

```
$ for f in evidence_ledger.py hypothesis_board.py admissibility.py moves.py \
    chat_core.py question_selector.py ocean.py agents.py turn.py panel.py; do
    uv run python "$f"; done
ok  (x10)
$ uv run pytest -q
1 passed in 0.00s
```

Every C5 module's self-check was re-run, not just the three files touched,
to confirm nothing else regressed.

## Scope notes / what was deliberately not done

- **No new Ocean profile, no change to `moves.py` or `chat_core.py`.** The
  `ask` move's legality (never gated by admissibility, unlike `reveal`) was
  already correct in `moves.py`; nothing about that changed. `ChatCore`
  needed no new method — `core.board` was already public.
- **`SINGLE_PASS` mode is not wired to the selector**, on purpose — see
  above and `turn.py`'s docstring.
- **`panel.run_panel()` is not called from the turn loop** — a documented
  decision, not an oversight. A future "show the proposer/critic debate
  before revealing" feature is real scope for a new bead, not this one.
- **Did not touch** `narrator-c5b.3.6`–`.3.11` (Chekhov ledger, metrics,
  traits-weighting, agreeableness sweep, mode flag) or `.3.9`'s
  trait-weighted evidence work — all separate, unrelated siblings.
- **`ASK_CANDIDATE_COUNT = 3`** is a fixed constant in `turn.py`, matching
  `question_selector.select_question`'s own "cap at one question per turn"
  discipline: several candidates go in, at most one question comes out.

## Suggested next commands

```
bd close narrator-9nl
git status   # question_selector.py, turn.py, panel.py modified; nothing else
             # -- per "do not commit unless asked", left for review
```
