# narrator-cby.3.5.1 — Clue partitioner with a solvability contract

Artifact: `clue_partition.py` (`partition()`, `CluePartition`, `PartitionError`).

## What it does

`partition(clues, n_agents, seed=...)` takes the `EpistemicClueGraph` C4's
`mystery.build()` already produced and splits its clue nodes (`kind in
{"trace", "alibi"}`) across `n_agents`: each clue is either shared (goes to
everyone) or unique (goes to exactly one, seeded rng choice). It keeps
sampling seeded candidates until one satisfies both halves of the contract,
or raises `PartitionError` naming which half is unreachable.

Both halves are checked by reconstructing a same-shape `EpistemicClueGraph`
restricted to a candidate node set (`_subgraph_view`) and calling the real,
unmodified `mystery.EpistemicClueGraph.validate_solvability()` on it — not a
local reimplementation. This mirrors the producer/checker split the rest of
the repo already enforces: the partitioner is a producer of splits, and the
thing that judges a split is the same checker `mystery.py` trusts for the
whole graph, so a change to what "solvable" means never has to be kept in
sync in two places.

## Why retry-and-check instead of building the split by construction

`suspects_consistent_with_clues()` currently only cares about which `alibi`
clue nodes are present in the graph (the footprint clue doesn't move it).
I could have hard-coded "spread the alibi clues across agents, never let one
agent hold them all" and skipped verification — but that bakes in today's
solvability rule. If a later bead changes what makes a subset solvable (e.g.
footprint starts mattering), a construction-only partitioner would keep
producing splits that look fair and are not, silently. Sampling + checking
via the actual validator can't go stale that way; it's the same argument
`narrator-cby.4.1`'s postmortem made for the mystery checker itself.

## AC check

- **"Union is solvable, every subset is not, both checked by the existing
  validator rather than by construction."** `_self_check()` builds a
  3-agent split and asserts `_subgraph_view(clues,
  result.pooled_clues()).validate_solvability()` is `True` and
  `_subgraph_view(clues, result.agent_clues(i)).validate_solvability()` is
  `False` for every agent — both calls go through
  `mystery.EpistemicClueGraph.validate_solvability()` unmodified. Swept over
  seeds 1–5, 42 × agent counts 2–4 by hand outside the self-check (see
  Verified below); every combination held.
- **"Rejected partitions name which half failed."** Three rejection paths
  are exercised in `_self_check()`, one per way the contract can fail:
  - `n_agents=1` — individual half, one agent always equals the pooled set —
    asserts `"individual half failed"`.
  - A clue set with one alibi node removed from a normal `mystery.build()`
    graph — pooled set can no longer validate at all, so no split can save
    it — asserts `"union half failed"`.
  - A deliberately unsplittable clue set, built by shrinking `sim.suspects`
    to `[culprit, one witness]` so the single remaining alibi clue is
    simultaneously necessary and sufficient to solve — whichever agent gets
    it violates the individual half no matter how the rest of the split
    goes — asserts `"individual half failed"`.
- **"Same seed reproduces the same split."** `_self_check()` calls
  `partition()` twice with identical arguments and asserts `.shared` and
  `.unique` are equal; a 6-seed × 3-agent-count sweep (below) additionally
  asserts this across fresh Python processes, not just repeated in-process
  calls.

## Verified

`mystery.py` imports `networkx`, which is not in `pyproject.toml`'s
dependency list (tracked separately as `narrator-3c5`, pre-existing, out of
this bead's scope). `uv run pytest` currently can't import `mystery` or
`clue_partition` at all for that reason — the scaffold test is the only
thing that's actually running under that invocation right now. I installed
`networkx` directly into the `uv`-managed venv (`uv pip install --python
$(uv python find) networkx pytest`, no `pyproject.toml`/`uv.lock` change) to
verify this bead's own work; that installation does not persist as a repo
change.

- `python clue_partition.py` → `ok` (self-check above).
- `python clue_partition.py --seed 7 --agents 3` → prints a partition,
  eyeballed: 1 shared alibi, unique alibis and the footprint spread over 3
  agents, pooled validates, no agent validates alone.
- Manual sweep, seeds `{1,2,3,4,5,42}` × `n_agents ∈ {2,3,4}`: pooled
  solvable, no individual agent solvable, and a second `partition()` call
  with the same seed reproduces `.shared`/`.unique` exactly, for all 18
  combinations.
- `python mystery.py` → `ok` (unchanged, confirms the dependency I installed
  locally didn't mask a real regression in the module this one builds on).

## Left for narrator-cby.3.5.2

`CluePartition.agent_clues(i)` (what to hand each agent for the
conversation) and the shared/unique breakdown (`.shared`, `.unique[i]`, for
the "was this a shared or unique clue" half of the unshared-clue measure)
are both already exposed on the return value — that bead shouldn't need to
touch `clue_partition.py` to consume it, only to import `partition()` and
`CluePartition`.
