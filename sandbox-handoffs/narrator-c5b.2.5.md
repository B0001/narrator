# narrator-c5b.2.5 — Token budget and key policy for the teaching audience

## What the bead asked for

`agents.converse` round-robins N agents over a shared transcript that every
turn's prompt re-reads in full, so tokens-sent-per-run grow quadratically in
turn count. Add a per-run ceiling that stops rather than silently spends,
and document (in `prd.md`) which components default to which backend and
why some are cheap enough for a metered one and others aren't.

## What I found in the working tree first

Before touching anything I ran `git status`: several files were already
modified or untracked (`agents.py` itself had an unrelated `evidence` field
add from `narrator-cby.3.5.2`, plus `backends/ocean_fable.py`,
`clue_partition.py`, `discussion.py`, `panel.py`, `question_selector.py`,
`pyproject.toml`/`uv.lock` changes, and several `sandbox-handoffs/*.md`
files). None of it referenced `narrator-c5b.2.5` in `bd show`'s notes, and
`bd show` confirmed this bead itself was still `open` (not previously
claimed). I read that as other bead sessions' in-progress work in a shared
workspace, not a prior attempt at this bead — so I left all of it alone and
only added to it, never reverted or rewrote any of it. My diff to `agents.py`
sits on top of the pre-existing `evidence` field change, not against a clean
base.

## What I changed

**`agents.py`**
- Added `DEFAULT_TOKEN_BUDGET = 20_000` and `_estimate_tokens(text)`, a
  stdlib-only `len(text) // 4` heuristic (no tokenizer dependency, per
  `prd.md`'s legibility constraint) — good enough to catch a runaway
  quadratic transcript, not good enough to bill against, and the docstring
  says so.
- Added `TokenBudgetExceeded(RuntimeError)`.
- `converse()` now takes `token_budget=DEFAULT_TOKEN_BUDGET`. Before each
  turn it estimates that turn's prompt cost, and if the running total would
  exceed the budget it raises `TokenBudgetExceeded` naming the turn number,
  the projected token count, the configured limit, and how much was spent
  over how many turns already — before generating or writing that turn, so
  a run that hits the ceiling leaves a transcript with exactly the turns it
  actually paid for and no partial/corrupt final line. `token_budget=None`
  disables the ceiling explicitly (documented as the escape hatch for a run
  you know is free, e.g. local Ollama).
- Extended `_self_check()` with `_check_token_budget()`: a generous budget
  doesn't interfere with a normal run; a budget too tight for even turn 0
  stops immediately with both the projected count and the limit in the
  message, and writes nothing to the transcript file; `token_budget=None`
  runs the same as before this change existed.
- No existing caller breaks: `token_budget` is a new keyword-only-by-default
  argument appended after `generate_fn`, and the one other caller I found in
  the tree (`discussion.py`, from someone else's in-progress work) calls
  `converse()` entirely by keyword.

**`prd.md`**
- Replaced the forward-reference "see `narrator-c5b.2.5` for which
  components are cheap enough" with the actual answer, as a new bullet
  under the local-by-default/cloud-opt-in decision: `ocean.py generate()`,
  `motive.py prose()`, and `chapters.py write_chapters()` are one
  independent call per unit of work (no growing shared context) so they're
  linear and fine against Fable by default; `turn.py run_turn()` is bounded
  by the evidence ledger's live entries, not turn count, so a long
  conversation costs about the same per turn throughout and is also fine
  against Fable by default; `agents.converse()` is the outlier — quadratic
  in turns — and is the one place the new `token_budget` ceiling lives.

I did not touch `turn.py`, `chapters.py`, or `motive.py` — I verified their
call shapes (grep + read) to confirm the "linear, not quadratic" claim in
`prd.md` is actually true of the code today, but the bead's fix (the
ceiling) belongs on `agents.converse`, which is the component actually named
in both the bead and the pre-existing `prd.md` paragraph it was resolving a
forward-reference in.

## Verification

- `uv run python agents.py` → `ok` (self-check, includes the three new
  token-budget assertions).
- `uv run pytest -q` → 1 passed (unchanged).
- Self-checked every other module with a `__main__` block
  (`ocean.py`, `motive.py`, `mystery.py`, `backends/base.py`,
  `backends/ocean_ollama.py`, `backends/ocean_fable.py`) — all still `ok`,
  confirming the change didn't ripple anywhere.
- `uv run python discussion.py` (someone else's in-progress file that calls
  `converse()`) → still `ok`, confirming the new keyword argument doesn't
  break that caller.

## Acceptance criteria check

- "A run that would exceed the ceiling stops with the count and the limit
  named." → `TokenBudgetExceeded` message names the turn, the projected
  token count, and the configured `token_budget`; asserted directly in
  `_check_token_budget`.
- "prd.md says which components default to which backend." → the existing
  "What stays local by default" bullet already named the components; the
  new "Cost" bullet now says *why* each one is or isn't cheap enough to run
  against Fable, closing the forward-reference this bead existed to answer.

## What I left alone / follow-ups

- I did not touch the other uncommitted files in the tree (see above) —
  they belong to other beads' sessions.
- Not filed as new beads because they're out of scope for this bead but not
  new problems I found: whether `turn.py`'s chat loop should eventually get
  its own ceiling if the evidence ledger's live-id bound turns out not to
  hold under some future change is worth someone's eye when that surface
  gets touched next, but nothing in this bead's scope calls for it now.
