# narrator-c5b.1 — Reverse the 'local Ollama only' non-goal, on the record

## Status: done

## What was built

`prd.md`'s Non-goals section no longer contains "No cloud model providers.
Local Ollama only." — that line contradicted the Fable backend
(`narrator-c5b.2.2`) already scaffolded for in `backends/base.py`'s
docstring. In its place, a new section, **"Decision: local-by-default,
cloud opt-in (2026-08-29, `narrator-c5b.1`)"**, records:

- **What stays local by default** — Ollama (`backends/ocean_ollama.py`) is
  what every existing entry point (`ocean.py`, `agents.py`, `motive.py`,
  `chapters.py`, `turn.py`) calls with no key set. No caller changes
  behavior unless a learner deliberately switches backends.
- **What the cloud path is for** — Fable is opt-in per learner (a stronger
  model without local hardware, or comparison against a local model for the
  same profile), a second backend behind the same `Backend` protocol, not a
  replacement. Points forward to `narrator-c5b.2.2` (implementation) and
  `narrator-c5b.2.5` (which components are cheap enough to run against it —
  `agents.converse`'s transcript cost is quadratic in turns).
- **Who holds the key** — the learner, via `ANTHROPIC_API_KEY` from the
  environment only, never hardcoded/profile-file/source-checked-in. A run
  with no key set must fail naming the missing variable, not a raw
  connection error.
- **Why this is a real change** — API keys, per-token cost, a network
  dependency, and a teenager's conversation text leaving the machine were
  the actual stakes the old non-goal fenced off. This decision moves the
  fence rather than removing it: cloud use is possible but never silent,
  never default, never authorized by anyone but the person running it.

This is a docs-only change. No backend code was touched or added — writing
`backends/ocean_fable.py` is explicitly out of scope for this bead and
belongs to `narrator-c5b.2.2`, which was blocked on this bead landing first.

## Verification

- `grep -n "Ollama only\|cloud model" prd.md` — only match left is inside
  the new decision section's own "Supersedes ..." sentence, quoting the old
  line for the record. The non-goal itself is gone.
- `uv run pytest -q` — 1 passed (scaffold test; unaffected by a docs-only
  change).
- No other module was touched, so no other self-check applies.

## Files touched

- `prd.md` (Non-goals bullet removed, decision section added)
- `sandbox-handoffs/narrator-c5b.1.md` (new, this file)

No commits made — repo policy is not to commit/push unless the bead says
to, and this bead doesn't.

## Next steps (for whoever picks up `narrator-c5b.2.2`)

The blocker this bead existed to clear is gone. `narrator-c5b.2.2`
(`backends/ocean_fable.py`, Anthropic `/v1/messages` against
`claude-fable-5`) can now start — see `sandbox-handoffs/narrator-c5b.2.md`'s
"Next steps" for the shape differences already scoped out (system as a
top-level param, `max_tokens` required, content-block-list response, key
from `ANTHROPIC_API_KEY` only, name-the-missing-var failure).
