# narrator-cby.3.5.2 — Discussion protocol and the unshared-clue measure

Artifact: `discussion.py` (`run_discussion()`, `clue_report()`, `final_answer()`,
`report_table()`), plus a small extension to `agents.py` (`Agent.evidence`).

## What it does

`run_discussion(seed, n_agents, turns, ...)` builds a mystery
(`mystery.build()`), partitions its clue graph (`clue_partition.partition()`),
and runs `agents.converse()` over the split: each agent's `Agent.evidence` is
set to exactly `partition.agent_clues(i)`'s descriptions, so their prompt
carries only the clues they were dealt, never the pooled set and never
`sim.culprit`. From the resulting transcript, two things come out:

- `clue_report(clues, partition, records, agent_names)` — one row per clue,
  reporting whether it's `shared` or `unique` (and to whom, from the
  partition — the manipulation), whether it was `voiced` (read from the
  transcript text), the `first_speaker`/`first_turn` of that first mention,
  the total `mention_count`, and `uptake`.
- `final_answer(records, suspects)` — the group's own answer, read as the
  most recent turn that names exactly one suspect. `None` (undecided) if no
  turn ever narrows to one name — abstention over a coin flip, matching the
  repo's `validate_solvability()` precedent.

`run_discussion()`'s return dict scores `group_answer` against `sim.culprit`
as `culprit_hit`, computed after the fact from two things that never talked
to each other: `final_answer()` only ever sees `records` and `sim.suspects`
(never `sim.culprit`), and the culprit comparison happens one level up, in
`run_discussion()`, not inside the answer-extraction function itself.

## The one non-obvious design decision: uptake ≠ mention count > 1

The bead's description flags this directly: "mention count alone is the
wrong measure." A clue an agent repeats three times to itself is still
information that never left that agent's head. `uptake` is defined as *a
later mention by a speaker other than whoever first raised it* — the textual
signature of the group actually engaging with the clue, not just its holder
talking to themselves. `mention_count` is reported alongside it rather than
being the uptake signal, so a learner can see the two diverge on the same
row (a unique clue mentioned twice by its one holder: `mention_count=2`,
`uptake=False`).

Voicing detection itself is a marker match (`_mentions_clue`), same spirit
and same ceiling as `metrics.py`'s agreement regex: alibi clues match on the
witness's name (unique per clue, since there's one alibi clue per witness);
the footprint clue has no name, so it requires the room name *and* a
footprint/mud word together, to avoid false-positiving on a turn that
mentions the room for some other reason. Documented as a "ponytail"-grade
matcher in the module docstring, same convention as `metrics.py`.

## Why `agents.py` needed a one-field extension

The bead says "Run `agents.converse()` over the partition" — literally reuse
the existing turn loop, not reimplement it with per-agent context bolted on
outside it. `agents.converse()`'s prompt builder only ever had `topic` (same
string to every agent) and `log` (shared) to work with; there was no way to
give one agent private information without changing what `Agent` carries.

Added `Agent.evidence: str = ""`, injected into `_prompt()`'s template only
when non-empty. Default value keeps every existing caller (including
`CONTRARIAN`/`ACCOMMODATING` and `agents.py`'s own self-check) byte-for-byte
unaffected — verified by re-running `agents.py`'s self-check unchanged plus
one new assertion pinning that a briefed agent sees their evidence and an
unbriefed one sees no such section at all.

## AC check

- **"Per-clue voiced, uptake and turn-first-mentioned columns come out of a
  run."** `discussion.py`'s `_self_check()` runs the full pipeline —
  `run_discussion()` with a scripted `generate_fn` that echoes each agent's
  own evidence back verbatim on their first-round turn, plus a scripted
  final-turn verdict — and asserts `first_speaker`/`first_turn`/`uptake` on
  the resulting `clue_report` for a shared clue (voiced by three distinct
  speakers → `uptake=True`) and three unique clues (voiced once each, by
  their one holder → `uptake=False`). Same pattern is exercised again as a
  pure-function test directly against a hand-built transcript, including the
  edge cases: a clue nobody voices (`voiced=False`, `first_speaker=None`,
  `first_turn=None`), and a same-speaker repeat (`mention_count=2`,
  `uptake=False` — the case the bead's "mention count alone is wrong"
  warning is about). `report_table()` prints these as columns, same style as
  `metrics.summary()`.
- **"Group answer is scored against ground truth the agents never saw."**
  `final_answer()` takes only `records` and `sim.suspects` — never
  `sim.culprit` — and is tested standalone against four transcripts: no
  suspect named anywhere (`None`), the deciding turn naming exactly one
  suspect, a later turn overriding an earlier different guess, and a
  deciding turn naming two suspects at once (`None` — ambiguity is not a
  verdict). The full-pipeline test then confirms `run_discussion()` wires
  the comparison correctly: `sim.culprit` for seed 3 is "Lady Margaret", the
  scripted transcript's answer is "Lord Blackwood", and `culprit_hit` comes
  back `False` — the mismatch is caught, not silently agreed with.

## Verified

`networkx` still isn't in `pyproject.toml` (pre-existing, tracked as
`narrator-3c5`, out of this bead's scope — same note as the
`narrator-cby.3.5.1` handoff). Installed it into the `uv run`-managed venv
(`uv pip install --python $(uv run python3 -c 'import sys; print(sys.executable)') networkx pytest`)
to run anything that imports `mystery`; no `pyproject.toml`/`uv.lock` change.

- `python3 discussion.py` → `ok` (self-check above).
- `python3 discussion.py --seed 7 --agents 3` → builds the mystery, partitions
  it, and gets as far as the network call to Ollama before failing with
  `ConnectionRefusedError` (no local Ollama server in this environment) —
  confirms the wiring up to the model boundary is correct, same shape as
  `agents.py --run` failing the same way for the same reason.
- `python3 agents.py` → `ok`, including the new evidence-injection assertion.
- `python3 metrics.py`, `python3 clue_partition.py`, `python3 mystery.py` →
  `ok`, unaffected by this bead's changes.
- `uv run pytest` → 1 passed (the scaffold placeholder; nothing under
  `tests/` yet exercises `mystery`-dependent modules, same pre-existing gap
  noted in the `.3.5.1` handoff).

## Left for narrator-cby.3.5.3 and .3.5.4

- `.3.5.3` (trait/role conditions) can hold `partition` fixed and vary each
  agent's `Ocean` profile directly in the `roster` list `run_discussion()`
  builds — nothing in `discussion.py` assumes a neutral profile, `Ocean()`
  there is just the default for this bead, not a constraint.
- `.3.5.4` (pooling metrics in `metrics.py`) wants fraction-of-unique-voiced,
  fraction-with-uptake, and time-to-first-unique-mention, aggregated from
  exactly the per-clue rows `clue_report()` already returns — it should be
  able to import `discussion.clue_report`'s output shape and reduce over it
  without touching `discussion.py`.
