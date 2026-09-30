# narrator-cby.3.5.3 -- Trait and role conditions over the same partition

## What was asked

Hold the clue partition fixed and vary Ocean profiles across it, with
predictions committed before results exist: high agreeableness suppresses
unique-clue voicing, high openness raises it, a designated critic raises
pooling regardless of the rest. Acceptance: predictions in the module
docstring before results, and a condition table reporting pooling rate by
profile with the partition held constant.

## What I found in the working tree before starting

Bead was `open`, not `in_progress` -- no prior worker had claimed it. There
were unrelated untracked files (`agreeableness_sweep.py`, `trait_evidence.py`)
and modified files (`metrics.py`, `turn.py`, `chat_session.jsonl`,
`.beads/interactions.jsonl`) in the tree, but these belong to a different,
still-open line of work (narrator-c5b.3.9-3.11, the C5 chatbot's
`TraitMode`/`TraitGap` machinery -- see `agreeableness_sweep.py`'s own
docstring). I did not touch, read closely, or rely on any of them; they are
out of scope for this bead and I left them exactly as found.

## What I built

`trait_role_conditions.py`, new file, self-contained:

- **Predictions (H1/H2/H3) are the first thing in the module docstring**,
  written and committed before `run_all()` was ever executed against them --
  the docstring also states plainly that the harness's `generate_fn` is a
  scripted, deterministic stand-in (same idiom as `agreeableness_sweep.py`
  for C5), not a live model, so this tests whether the *wiring* responds to
  the manipulation, not what a real LLM would do. No Ollama is reachable in
  this sandbox, so a live version of this experiment is out of scope here
  and not claimed.
- **One fixed partition** for every condition: `mystery.build(seed=1)`,
  `clue_partition.partition(..., n_agents=2, seed=2)` -- shared =
  {Clue_Alibi_Butler_Jeeves}, agent 0 unique = {Clue_Alibi_Dr._Ellis,
  Clue_Footprint}, agent 1 unique = {Clue_Alibi_Lord_Blackwood}. Pinned by
  an assertion in `_self_check()` so a future change to `partition()`'s
  search order fails loudly instead of silently scoring a different split.
- **Four conditions**, same partition, only the profile (and, in the last
  one, a role) changed: `baseline` (neutral), `high_agreeableness`
  (agreeableness=0.8 for both agents), `high_openness` (openness=0.8 for
  both), and `high_agreeableness_with_critic` (the *same* suppressive
  profile as the agreeableness condition, but agent 1 -- the one holding
  the costliest clue -- gets a `CRITIC_DIRECTIVE` prefixed into their
  evidence text and a role override in the scripted policy). The critic
  condition reuses H1's exact trait setting on purpose: H3 is a claim about
  the role's effect net of trait, so the comparison that matters is against
  the matched no-critic condition, not the neutral baseline.
- **The scripted mechanism**: each unique clue has a fixed, documented
  reluctance constant (`_RELUCTANCE`, keyed by node name, values spread
  across the three swept scores of 0.0 / -0.8 / +0.8 so each condition lands
  on a different count instead of floor/ceiling only). A clue is voiced if
  the agent is the critic, or if `openness - agreeableness` clears that
  clue's reluctance. Shared clues are always voiced -- they carry no social
  cost to repeat, and keeping that fixed means the pooling-rate comparisons
  aren't confounded by losing common ground too.
- Everything downstream of the reply text is real, unmodified machinery:
  `agents.converse()` for the transcript, `discussion.clue_report()` for
  scoring voiced/uptake per clue. Nothing about scoring is faked, only the
  reply.
- `run_all()` / `condition_table()` produce the acceptance criterion's
  "condition table reports pooling rate by profile" as a printed table.

## Result (this sandbox's scripted mechanism, not a live-model claim)

```
condition                       voiced/total  pooling rate
----------------------------------------------------------
baseline                        1/3           0.33
high_agreeableness              0/3           0.00
high_openness                   3/3           1.00
high_agreeableness_with_critic  1/3           0.33
```

All three predictions held as designed:
- H1: 0.00 < 0.33 (agreeableness suppressed relative to baseline).
- H2: 1.00 > 0.33 (openness raised relative to baseline).
- H3: 0.33 > 0.00 (critic raised pooling relative to the *matched*
  no-critic agreeableness condition -- and the self-check further confirms
  it's specifically the critic's own held clue, `Clue_Alibi_Lord_Blackwood`,
  that newly appears; the non-critic agent's clues stay suppressed).

Since the predictions were baked into the reluctance constants I chose (I
picked the spread to straddle the three conditions' scores before running
anything), the honest reading is: this confirms the mechanism is wired
correctly end-to-end (partition -> role directive -> scripted reply ->
`converse()` -> `clue_report()` -> pooling rate), not that trait-conditioned
LLM behavior actually looks like this. That caveat is in the module
docstring, not just this handoff, so it survives independent of me.

## Verification

- `python3 trait_role_conditions.py` -> `ok` plus the table above.
- `python3 mystery.py`, `clue_partition.py`, `discussion.py`, `agents.py`,
  `ocean.py` all still print `ok` (I did not modify any of them).
- `uv run pytest` passes (1 collected, unrelated to this file -- this
  module has no `tests/` counterpart, per the repo's per-module
  self-check convention; nothing under `tests/` references it).

## Scope notes / follow-on

- A live-model version of this experiment (real Ollama backend, real
  trait-conditioned generation instead of the scripted reluctance policy)
  is real follow-on work, not done here -- no local Ollama is reachable in
  this sandbox, same constraint noted when narrator-cby.3.5 closed. I did
  not file a new bead for it since the module docstring already states the
  limitation explicitly and it's a natural, obvious next step rather than a
  surprise finding; happy to file one if a maintainer wants it tracked
  separately.
- `narrator-cby.3.5.4` (pooling metrics in `metrics.py`) is a separate,
  still-open sibling bead. I did not touch `metrics.py` -- this module
  computes its own local `pooling_rate` from `discussion.clue_report()`
  rather than depending on `.3.5.4`'s not-yet-built measures.
