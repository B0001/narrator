# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## What this repo is

A teaching project at the intersection of code generation and psychology, for
an audience of three teenagers learning by building. Personality becomes a
data structure, motive a graph traversal, and pathology an observable
multi-agent interaction. See `prd.md` for the full spec and non-goals.

## The claim this repo makes

That a conclusion the system presents — a mystery's solution, or (in the
newer C5 chatbot work) a deduction stated mid-conversation — actually follows
from evidence that was checked independently of the thing that produced it,
rather than merely asserted by it.

`sandbox-prompt.md` states this standard in full, including the real bug
(`narrator-cby.4.1`) that motivates it. Read it before writing code.

## Producer / checker split

The generating half and the verifying half are separate modules, and the
checker does not get access to the producer's internals:
`WorldSimulation`/the clue graph never gets read by `validate_solvability()`
as ground truth, only as the clue set a real solver would see. C5's runtime
admissibility check follows the same rule against the evidence ledger. A
checker that can see the producer's internal state will agree with it and
verify nothing.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
uv sync
uv run pytest
```

Every module also self-checks standalone: `python3 <module>.py`.

## Architecture Overview

- `ocean.py` (C1) — Five-Factor trait vector compiled into Ollama sampler
  settings and prompt injections.
- `motive.py` (C2) — Seeded graph traversal over bias/trigger nodes; the
  deterioration path doubles as a story outline.
- `agents.py` / `metrics.py` (C3) — Round-robin multi-agent turn loop over a
  shared transcript, plus interaction metrics. `narrator-cby.3.5` extends
  this into a Stasser-Titus hidden-profile harness (clue partitioning +
  pooling measures).
- `mystery.py` (C4) — `WorldSimulation` + causal graph is the producer;
  `validate_solvability()` is the checker, run against the clue graph only,
  never the generator's internal claim of solvability.
- C5 (`narrator-c5b`, in progress) — turns the same producer/checker shape
  around into a fair-play chatbot. `evidence_ledger.py` is the producer's
  visible surface; `admissibility.py` (inverted from `validate_solvability()`)
  is the checker; `hypothesis_board.py`, `moves.py`, `chat_core.py`, and
  `turn.py` are the turn loop around them. The `backends/` split and the
  Fable/Anthropic backend are follow-on beads.

## Conventions & Patterns

- Stdlib first, few dependencies, for anything a learner reads (`prd.md`'s
  legibility constraint). Dev-only tooling (`pytest`, `uv`) is exempt — it
  never ships to a learner.
- Every module ends with `if __name__ == "__main__": _self_check()` —
  assert-based, no framework. `tests/` holds anything broader.
- Producer and checker stay in separate functions/modules; the checker never
  reads the producer's internal state, only its observable output.
