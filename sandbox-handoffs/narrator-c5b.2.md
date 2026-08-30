# narrator-c5b.2 — Model backend split: Ollama stays default, Fable opt-in

## Status: partial, left open on purpose

The interface is formalised and the Ollama path is migrated behind it. The
second backend (Fable/Anthropic) is deliberately **not** built this session
— see "Why this stays open" below. Closing narrator-c5b.2 requires that
backend to exist ("Both backends satisfy one interface"), so the bead stays
open until a future session clears its upstream blocker and does that work.

## What was built

- **`backends/base.py`** — `Backend`, a `typing.Protocol`:
  `generate(profile, prompt, model=...) -> str`. Docstring states the
  contract callers rely on: personality compiles *inside* the call
  (`profile.system_prompt()`, `profile.options()`), never before it, so a
  backend missing an equivalent of one of `options()`'s keys (Anthropic has
  no `repeat_penalty`) just drops that key — the caller doesn't need to know
  which backend it's talking to. `conforms()` is a small runtime helper used
  by self-checks, not an import-time gate. Stdlib only (`typing`).

- **`backends/ocean_ollama.py`** — the exact HTTP call that used to live in
  `ocean.generate()`, moved verbatim (same URL construction, same request
  body, same `timeout=120`, same default model `qwen2.5-coder:14b` and host
  `http://localhost:11434`). Self-check monkeypatches `urllib.request.urlopen`
  so it runs with no live Ollama and asserts the exact request body sent
  (model, prompt, compiled system prompt, compiled options, `stream: False`)
  and that the `response` field of the reply is what comes back.

- **`ocean.py`** — no longer imports `urllib`, no longer contains an HTTP
  call. `generate` is now `from backends.ocean_ollama import generate`, so
  every existing caller (`agents.converse`'s `generate_fn` default,
  `motive.prose`, `chapters.write_chapters`'s default, `ocean.py`'s own
  `--demo` path) keeps importing `ocean.generate` with the identical
  signature and identical behaviour. No caller was touched.

Verified: `python3 backends/base.py`, `python3 backends/ocean_ollama.py`,
`python3 ocean.py`, `python3 agents.py`, `python3 turn.py`,
`python3 evidence_ledger.py`, `python3 admissibility.py`, `python3 moves.py`,
`python3 chat_core.py`, `python3 metrics.py`, `python3 hypothesis_board.py`
all print `ok` (or, for `metrics.py`, its normal table) unchanged, and
`uv run pytest` passes. `chapters.py`, `motive.py`, `mystery.py` fail at
`import networkx` — pre-existing, unrelated to this change, already tracked
as `narrator-3c5` (networkx missing from `pyproject.toml`).

`narrator-c5b.2.1` ("Extract a Backend interface behind ocean.generate()")
was closed as part of this work — its acceptance criteria (`ocean.py` has no
HTTP call; existing self-checks pass untouched) are exactly what the above
satisfies, and leaving it open once done would just make a future session
redo it.

## Why this stays open

The parent bead's acceptance criteria is *"Both backends satisfy one
interface; every existing entry point runs unchanged against Ollama with no
key set."* Only one backend exists after this session. The second
(`narrator-c5b.2.2`, Anthropic `/v1/messages` against `claude-fable-5`) is
explicitly blocked on `narrator-c5b.1` ("Reverse the 'local Ollama only'
non-goal, on the record"), and `narrator-c5b.1`'s own acceptance criteria
says the prd.md amendment must land **before any backend code lands**.
`prd.md` still lists "No cloud model providers. Local Ollama only." under
Non-goals (line 22) as of this session. Writing the Fable backend now would
be exactly the thing that bead exists to gate against — landing code before
the decision it depends on is on the record. So `narrator-c5b.2.2` (and the
three P2 children that depend on it in turn — `.2.3` sampler mapping, `.2.4`
model identity in transcripts, `.2.5` token budget/key policy) were left
untouched this session.

## Next steps (for whoever picks this up)

1. `narrator-c5b.1` first — amend `prd.md`'s non-goal, record what stays
   local by default, what the cloud path is for, who holds the key.
2. `narrator-c5b.2.2` — `backends/ocean_fable.py` implementing the same
   `Backend` interface built here: `system` as a top-level param (not a
   message role — `profile.system_prompt()` already produces the right
   string, it just moves to a different JSON key here), `max_tokens`
   required (Ollama's call has no equivalent — pick a default and document
   it), response is a content-block list to flatten rather than a single
   `response` string, key from `ANTHROPIC_API_KEY` only (never source or a
   profile file), and a clear `RuntimeError`/similar naming the missing env
   var when unset — not a raw connection error.
3. Then `.2.3` (sampler mapping without `repeat_penalty`), `.2.4` (no
   prompt names a model; transcripts record which model actually answered),
   `.2.5` (token ceiling for `agents.converse`'s quadratic cost, plus
   `prd.md` saying which components default to which backend) — all three
   already depend on `.2.2` in the tracker.
4. Once `.2.2`–`.2.5` are done, close `narrator-c5b.2` itself — its
   acceptance criteria will finally be fully met by two backends behind
   `backends/base.py`'s `Backend` interface.

## Files touched

- `backends/__init__.py` (new)
- `backends/base.py` (new)
- `backends/ocean_ollama.py` (new)
- `ocean.py` (HTTP call removed, `generate` now imported from
  `backends.ocean_ollama`)

No commits made — repo policy is not to commit/push unless the bead says to,
and this bead doesn't.
