# narrator-c5b.3.12 — Checker subagent carries no Ocean profile

## What was built

One new stdlib-only module, `panel.py`, sitting alongside `turn.py` on top of
the already-closed C5 core (`evidence_ledger.py`, `admissibility.py`,
`moves.py`, `chat_core.py`) and reusing C3's `agents.py` roster.

- **`PROPOSER` / `CRITIC`** — no new profiles. Per the bead text ("Vale and
  Wren from C3 already cover the first two"), these are aliases for
  `agents.ACCOMMODATING` (Wren) and `agents.CONTRARIAN` (Vale). Vale's
  `agreeableness=-0.8` matches the bead's "low-agreeableness critic"
  description trait-for-trait; Wren covers the proposer role by the bead's
  own instruction, though its defining trait is agreeableness rather than
  openness — noted in the module docstring rather than silently glossed
  over, since inventing a new "high-openness" profile wasn't what the bead
  asked for.
- **`Checker`** — the new piece. A class with `__init__(self, profile=None)`
  that raises `TypeError` for *any* non-`None` profile, including a bare
  `Ocean()` with every trait at its default. That last case is deliberate:
  the bead's claim is "cannot be weighted," not "currently weighted at
  zero," so a neutral-looking profile is rejected on the same footing as an
  extreme one — same principle `turn.py`'s `ReasoningProfile` already
  applies to sampler options, pushed one step further here since the
  checker doesn't even get an Ocean subclass, just a bare guard. `verdict()`
  delegates straight to `admissibility.check()`, unmodified — the ledger and
  citation list are the only inputs, exactly as `admissibility.py`'s own
  docstring already requires of every caller.
- **`run_panel(ledger, cited_ids, proposer_prompt, critic_prompt,
  generate_fn, model=...)`** — one round showing the three roles doing three
  different jobs (propose, critique, check) rather than three turns of the
  same job, which is the contrast the bead draws against
  `agents.converse()`'s round-robin conversationalists. `generate_fn` follows
  the same injectable shape as `ocean.generate` / `agents.converse` /
  `turn.run_turn`.

## How the acceptance criteria are met

Bead AC: *"Passing a profile to the checker raises rather than being
ignored. Verdicts are reproducible across runs at fixed seed while proposer
and critic vary."*

1. **Construction-time guard.** The self-check constructs `Checker()` and
   `Checker(profile=None)` (both fine), then loops over `Ocean()` (default,
   "neutral-looking"), `PROPOSER.profile`, `CRITIC.profile`, and
   `Ocean(neuroticism=0.9)`, asserting each one raises `TypeError` naming
   "profile" in the message — tried both positionally and as a keyword, so
   there's no back door through argument style.
2. **Reproducible verdicts, varying proposer/critic.** A fake
   `sampled_generate` stands in for a real backend: it draws from Python's
   module-level `random` state, scaled by the persona's `options()["temperature"]`
   — a stand-in for real sampling variance. `random.seed(42)` is set once,
   then `run_panel()` is called 5 times over the same ledger and citation
   set. The proposer and critic texts differ across all 5 calls (asserted via
   `len(set(...)) > 1`, since each call consumes RNG state). The checker's
   `Verdict` is asserted identical across all 5 (`all(v == verdicts[0] ...)`)
   — it never touches the RNG at all, so there was never a mechanism by
   which it *could* vary. To make the causal claim concrete rather than
   coincidental, the seed is reset to 42 and the whole round run again: the
   checker verdict matches the very first run's, and — as a side effect of
   using the same seeded RNG — the proposer/critic draw reproduces too. The
   point being demonstrated is the asymmetry: the checker's reproducibility
   never depended on the seed in the first place, while the proposer/critic
   reproducibility exists only because the RNG was reset.
3. An ungrounded citation (`"guess"`, an `inferred_by_model` entry with no
   `supports`) run through `run_panel()` still comes back inadmissible,
   confirming the panel's checker path is the same `admissibility.check()`
   everything else in C5 already goes through, not a parallel weaker rule.

## Verification run

```
$ uv run python evidence_ledger.py && uv run python admissibility.py \
  && uv run python moves.py && uv run python chat_core.py \
  && uv run python hypothesis_board.py && uv run python ocean.py \
  && uv run python agents.py && uv run python turn.py \
  && uv run python panel.py
ok
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

All pre-existing C5 modules re-checked for regression; none touched by this
bead (`panel.py` is additive only, imports from `admissibility.py` and
`agents.py` without modifying either).

## Scope notes / design decisions

- **No new Ocean profile for the proposer.** The bead explicitly says Vale
  and Wren already cover the first two roles; inventing a "true"
  high-openness profile would be scope creep this bead didn't ask for, and
  would also duplicate `agents.py`'s existing reference pair rather than
  reuse it. Flagged the trait mismatch in the docstring instead of hiding it.
- **`Checker` is a bare class, not an `Ocean` subclass.** `turn.py`'s
  `ReasoningProfile` is an `Ocean` subclass specifically because it still
  needs to satisfy `ocean.generate()`'s `system_prompt()`/`options()`
  contract — the reasoning channel is still a model call, just a
  persona-neutral one. The checker never calls a model at all (`verdict()`
  is a pure wrapper over `admissibility.check()`), so there's no shared
  interface to subclass; making it inherit from `Ocean` just to reuse its
  shape would let a future caller accidentally hand it to `generate()` and
  get a plausible-looking but meaningless response back. A bare class with
  a hostile constructor is the tighter fit.
- **Rejects a default `Ocean()`, not just "obviously biased" ones.** This is
  the acceptance criterion's real teeth. It would have been easy to write a
  guard that only rejects profiles with non-zero fields (mirroring "can this
  actually bias anything") — that's the "convention" version the bead
  explicitly asks to avoid ("make it a construction-time error rather than
  a convention"). Any `Ocean` instance is refused, independent of its
  values.
- **`run_panel()` is deliberately thin.** It does not integrate with
  `turn.py`'s two-channel machinery or `chat_core.py`'s `conclude()` — that
  wiring (deciding *when* a panel round runs during a real conversation,
  vs. the existing single-reasoning-channel path) isn't what this bead's AC
  asks for and would be new scope. `run_panel()` exists to make the
  reproducibility contrast something a self-check (or a learner) can call
  directly, matching `turn.run_turn`'s own precedent.
- **Deliberately not touched:** `narrator-c5b.3.9`'s trait-weighted evidence
  work and `.3.10`'s mode flag are unrelated siblings under the same parent;
  nothing here anticipates or blocks them. `panel.py` has no mode concept
  and no trait-weighting hook.
- **Pre-existing uncommitted state in the tree, not from this session:**
  `agents.py`, `backends/__init__.py`, `pyproject.toml`, `uv.lock`,
  `discussion.py`, `clue_partition.py`, `backends/ocean_fable.py`, and
  several `sandbox-handoffs/*.md` files were already modified/untracked
  before this session started (visible via `git diff`/`git status` before
  any edit here). None of it was touched by this bead's work; `panel.py` is
  the only file this session created.

## Suggested next commands

```
bd close narrator-c5b.3.12
git status   # panel.py is untracked, alongside pre-existing untracked/modified
             # files from other in-progress beads -- nothing committed here,
             # per "do not commit unless asked"
```
