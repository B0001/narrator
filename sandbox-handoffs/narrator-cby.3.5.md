# narrator-cby.3.5 — Hidden-profile harness over the mystery generator

## Status: closing as already done, via its children

This session did not write new code. It re-verified the parent epic's own
acceptance criteria against what its two P1 children (`.3.5.1` clue
partitioner, `.3.5.2` discussion protocol) already built and closed in a
prior session, and confirmed the AC is satisfied without any further work.

## The parent AC, checked directly

> "A run reports whether the group reached the culprit, which clues were
> never voiced, and which agent held them. Ground truth stays out of every
> agent's context."

- **Whether the group reached the culprit** — `discussion.run_discussion()`
  returns `culprit_hit`: `final_answer()`'s read of the transcript (most
  recent turn naming exactly one suspect, `None` if the group never
  narrows to one) compared against `sim.culprit` one level up, in
  `run_discussion()` itself, not inside the answer-extraction function.
- **Which clues were never voiced, and which agent held them** —
  `discussion.clue_report()` returns one row per clue with `voiced`
  (`False` for clues no transcript turn ever mentions, verified explicitly
  in `discussion.py`'s self-check with a silent-transcript case) and
  `held_by` (`kind == "shared"` → all agent names; `kind == "unique"` → the
  one owning agent's name, read from `clue_partition.CluePartition`).
- **Ground truth stays out of every agent's context** — each `Agent`'s
  `evidence` field is built by `_evidence_text(clues,
  partition.agent_clues(i))`, which reads only that agent's own clue
  descriptions off the clue graph; `sim.culprit` is never passed into
  `Agent(...)`, `_prompt()`, or anywhere upstream of `converse()`.
  `clue_partition.partition()` itself never sees `sim.culprit` either — it
  calls the real `mystery.EpistemicClueGraph.validate_solvability()` (never
  reimplemented) against reconstructed subgraphs, the same checker
  `mystery.py`'s own producer/checker split already trusts.

All three are exercised by `discussion.py`'s `_self_check()` end to end
(scripted `generate_fn`, no live model) and as pure-function tests against
hand-built transcripts, not just asserted by construction.

## What was verified this session (reproduction, not trust)

Environment note: `networkx` is still not declared in `pyproject.toml`
(pre-existing, tracked separately as `narrator-3c5`, out of every one of
these beads' scope). Installed into the `uv`-managed venv only
(`VIRTUAL_ENV=/tmp/venv uv pip install networkx`), no `pyproject.toml` /
`uv.lock` change, same workaround the `.3.5.1` and `.3.5.2` handoffs used.

- `uv run --active python mystery.py` → `ok`
- `uv run --active python clue_partition.py` → `ok`
- `uv run --active python discussion.py` → `ok`
- `uv run --active python agents.py` → `ok`
- `uv run --active python metrics.py` → `ok`
- `uv run --active python discussion.py --seed 7 --agents 3` → builds the
  mystery, partitions the clue graph, reaches the model boundary, and fails
  with `URLError: [Errno 111] Connection refused` — no live Ollama server in
  this sandbox. Confirms the wiring is correct up to the point only a real
  model call can exercise; same failure shape the `.3.5.2` handoff recorded.
- `uv run --active pytest -q` → 1 passed (the `tests/` scaffold placeholder;
  every module here self-checks standalone per this repo's convention, which
  is what the four `ok` lines above are).

Nothing in the working tree needed to change to make these pass — this was
verification of existing, already-closed child work, not new implementation.

## Why the parent closes with two children still open

`.3.5.3` (trait/role conditions) and `.3.5.4` (pooling metrics in
`metrics.py`) are real, separately-scoped follow-on work — both already
filed as children with their own acceptance criteria — but neither is
required by the parent's own stated AC above, which asks only for a run
that reports the hit/miss, the unvoiced clues, and their holders, with
ground truth kept out of agent context. That bar is met. Extending
`metrics.py` and adding trait manipulation are enhancements on top of a
working harness, not preconditions for calling the harness's own AC
satisfied. They stay open, tracked independently, for a future session.

## Left for the still-open children

Unchanged from the `.3.5.2` handoff's read, re-confirmed here:

- `.3.5.3` can hold `partition` fixed and vary each agent's `Ocean` profile
  in the `roster` list `run_discussion()` builds — nothing in
  `discussion.py` assumes a neutral profile; `Ocean()` there is just this
  bead's default, not a constraint.
- `.3.5.4` wants fraction-of-unique-voiced, fraction-with-uptake, and
  time-to-first-unique-mention, all reducible from exactly the per-clue rows
  `clue_report()` already returns, without touching `discussion.py`.
