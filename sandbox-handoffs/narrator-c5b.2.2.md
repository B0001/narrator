# narrator-c5b.2.2 — Anthropic backend: /v1/messages against claude-fable-5

## Status: done

## What was built

`backends/ocean_fable.py` — the second `Backend` implementation, mirroring
`ocean_ollama.py`'s exact shape and philosophy (stdlib only: `json`, `os`,
`urllib.request`; same self-check style, monkeypatched transport, no
third-party HTTP client or the `anthropic` SDK). It handles the three shape
differences the bead named:

- **`system` is a top-level request field**, not a message role —
  `profile.system_prompt()`'s string goes straight into `body["system"]`.
- **`max_tokens` is required** and has no equivalent in `profile.options()`,
  so the module picks a default (`DEFAULT_MAX_TOKENS = 1024`) and documents
  it as a callable override, same as `model` and `host`.
- **The response is a content-block list**, not a single string —
  `generate()` joins the `text` field of every block with `type == "text"`
  in `reply["content"]`, rather than indexing `reply["response"]` the way
  Ollama does.

`options()`'s `repeat_penalty` has no Anthropic equivalent (the exact
example `base.py`'s docstring already names), so it's popped before the
remaining keys (`temperature`, `top_p`) are spread into the top-level
request body.

**Key handling**: `os.environ.get("ANTHROPIC_API_KEY")` is checked first,
before any request is built or any import of a live transport path runs. If
unset, `generate()` raises `RuntimeError("ANTHROPIC_API_KEY is not set --
...")` — the variable name is in the message. No profile file, no
source-level constant, no fallback ever holds the key.

## How the acceptance criteria were verified

*"A conversation runs end to end against Fable; with no key set the failure
names the missing variable instead of raising a connection error."*

This sandbox has no `ANTHROPIC_API_KEY` set (`pyproject.toml`'s own
`[tool.sandbox.forward-env]` comment says as much: "unset means that work
can't be tested" against a live endpoint) — so a live call to the real
Anthropic API was not possible here, and I did not fabricate one.

Both halves of the AC are exercised by `_self_check()` following the
*existing* convention `ocean_ollama.py` already set for this exact
situation — its own self-check never contacts a live Ollama server either,
it monkeypatches `urllib.request.urlopen` and asserts on the captured
request/response. `ocean_fable.py`'s self-check does the same, in two parts:

1. **No-key path**: pops `ANTHROPIC_API_KEY` from the environment, replaces
   `urlopen` with a function that raises `AssertionError` if called at all,
   and asserts `generate()` raises `RuntimeError` naming
   `ANTHROPIC_API_KEY` — proving the failure happens before any network
   attempt, not as a caught connection error.
2. **Keyed, end-to-end path**: sets a fake key, mocks `urlopen` to capture
   the outgoing request and return a two-block fake response
   (`{"type": "text", "text": "generated "}`, `{"type": "text", "text":
   "text"}`), and asserts the full round trip: correct URL
   (`https://api.anthropic.com/v1/messages`), correct headers (`x-api-key`,
   `anthropic-version: 2023-06-01`), a request body with `system` top-level,
   `max_tokens` present, `repeat_penalty` absent, and the two text blocks
   joined into `"generated text"`.

If a maintainer wants a genuine live call against the real API, that just
means exporting `ANTHROPIC_API_KEY` and running
`backends/ocean_fable.py`'s `generate()` directly (e.g. via a small
`--demo` flag the way `ocean.py` has one for Ollama) — no code change is
needed for that, the function already takes real key/host/model as its only
inputs and does not know it's being mocked. Wiring that demo entry point
into `ocean.py` is out of scope here (see below).

## Verification run this session

- `uv run python3 backends/ocean_fable.py` — `ok`
- `uv run python3 backends/base.py` — `ok` (unaffected, unmodified)
- `uv run python3 backends/ocean_ollama.py` — `ok` (unaffected, unmodified)
- `uv run python3 ocean.py` — `ok` (unaffected; still imports
  `backends.ocean_ollama.generate`, per the parent bead's constraint that
  Ollama stays the literal default — this bead does not touch that import)
- `uv run pytest -q` — 1 passed (scaffold test; unaffected)

## Files touched

- `backends/ocean_fable.py` (new)
- `backends/__init__.py` (docstring updated: named the module that now
  exists instead of forward-referencing this bead by ID)
- `sandbox-handoffs/narrator-c5b.2.2.md` (new, this file)

No commits made — repo policy is not to commit/push unless the bead says
to, and this bead doesn't.

## Explicitly out of scope (not done here, not filed as new beads because
they're already tracked)

`ocean.py:16`'s `from backends.ocean_ollama import generate` is untouched —
wiring backend *selection* into any caller is `narrator-c5b.2.5`'s job
("Token budget and key policy for the teaching audience"), not this one.
Likewise `narrator-c5b.2.3` (re-derive the trait-to-sampler mapping without
`repeat_penalty` — this bead just drops the key, it doesn't redesign what
replaces its effect) and `narrator-c5b.2.4` (don't let persona code assert a
model identity) are untouched; all three were already open and blocked on
this bead, and now they're unblocked.

## Next steps (for whoever picks up `narrator-c5b.2`'s closure)

Both backends now satisfy `Backend`. The parent bead's AC ("Both backends
satisfy one interface; every existing entry point runs unchanged against
Ollama with no key set") looks met, but I'm leaving the parent bead's own
closure to a session that deliberately checks it end to end (starting with
the children it still blocks: `.2.3`, `.2.4`, `.2.5`) rather than closing it
as a side effect of this one.
