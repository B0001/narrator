# narrator-c5b.2 — Model backend split: Ollama stays default, Fable opt-in

## Status: complete, closing this session

All five children (`.2.1`–`.2.5`) are closed. The parent's acceptance
criteria — *"Both backends satisfy one interface; every existing entry point
runs unchanged against Ollama with no key set."* — now holds: two backends
(`backends/ocean_ollama.py`, `backends/ocean_fable.py`) satisfy the same
`Backend` protocol (`backends/base.py`), `ocean.py` still imports its
default `generate` from the Ollama backend with no key required, and every
existing caller (`agents.converse`, `motive.prose`, `chapters.write_chapters`,
`turn.run_turn`) is unchanged at the call site.

## What was built, across the full arc of this bead

- **`backends/base.py`** — `Backend`, a `typing.Protocol`:
  `generate(profile, prompt, model=...) -> str`. Personality compiles
  *inside* the call (`profile.system_prompt()`, `profile.options()`), never
  before it, so a backend missing an equivalent of one of `options()`'s keys
  (Anthropic has no `repeat_penalty`) just drops that key. Also carries
  **`Reply(str)`** (added for `.2.4`): a `str` subclass with an additive
  `.model` attribute, so every existing string-consuming caller (`.strip()`,
  `json.loads()`, equality, f-strings) keeps working unmodified, while a
  caller that wants to know which model actually answered reads `.model`.
  `getattr(x, "model", "unknown")` is the shim for code that may see either
  a `Reply` or a plain `str` (e.g. a self-check stub with nothing to
  report).

- **`backends/ocean_ollama.py`** — the original `ocean.generate()` HTTP
  call, moved verbatim, now wrapping its response in `Reply(text,
  model=data.get("model", model))` — Ollama's own response names the model
  that answered; falls back to the requested name only if the server ever
  omits it.

- **`backends/ocean_fable.py`** (`.2.2`) — the second `Backend`
  implementation: Anthropic `/v1/messages` against `claude-fable-5`. `system`
  is a top-level request field, not a message role. `max_tokens` has no
  `options()` equivalent, so this module picks one flat default
  (`DEFAULT_MAX_TOKENS = 4096`) — documented as a ceiling, not a spend, and
  generous enough for the longest current caller (`chapters.write_chapters`).
  The reply is a content-block list, flattened to text. The key comes from
  `ANTHROPIC_API_KEY` only, never source or a profile file; with no key set
  this raises naming that variable before any network attempt. Server-side
  fallbacks are on by default (`anthropic-beta: server-side-fallback-2026-07-01`
  header, `fallbacks: "default"` body field) — a policy decline gets one
  retry on a fallback model inside the same call; a refusal that survives
  the whole chain still raises. Also wraps its return in `Reply(text,
  model=reply.get("model", model))` — the response's `model` field is
  whichever model actually answered, which can differ from what was
  requested when a fallback swap fires. `claude-fable-5` rejects
  `temperature`/`top_p`/`top_k` outright (HTTP 400), so `profile.options()`
  is never forwarded on this backend at all — the persona reaches the model
  through the system prompt only (`.2.3`'s finding, below).

- **`ocean.py`** — no `urllib` import, no HTTP call. `generate` is
  `from backends.ocean_ollama import generate`. `SAMPLER_TARGETS` +
  `_check_sampler_targets()` document each of the five traits' compilation
  target *per backend* (`.2.3`): Ollama gets temperature/top_p/top_k/
  repeat_penalty; Fable gets none of them (forwards no sampler settings —
  `backends/ocean_fable.py`'s self-check asserts `profile.options()` is
  never even called, so the "no target" case fails loud rather than
  silently degrading). `prd.md`'s trait-to-sampler table documents the same
  mapping for a reader.

- **Model-identity discipline (`.2.4`)** — no prompt string built anywhere
  in the repo names a model (verified by grep, scoped to actual
  prompt-building functions' returned strings, excluding docstrings/
  comments/CLI hints). Every persisted transcript record carries the
  responding model or an explicit `"unknown"`:
  - `agents.converse()`'s per-turn record gets `"model":
    getattr(raw, "model", "unknown")`.
  - `turn.py`'s `Call` dataclass gained a `model: str = "unknown"` field,
    populated the same way at each of its three `generate_fn` call sites
    (`reasoning`, `voice`, `ask-candidates`, and the `SINGLE_PASS`
    `reasoning+voice` call).
  - `metrics.chat_record()` reads `out.calls[-1].model` — always the call
    that produced the visible reply (the voice call in `TWO_PASS` even when
    an ask-candidates call ran first; the single reasoning+voice call in
    `SINGLE_PASS`).
  - `chat_session.jsonl`'s fixture was regenerated via
    `metrics._demo_chat('chat_session.jsonl')` to carry `"model": "unknown"`
    on every row (its stub `generate()` returns a plain string).

- **Token budget and key policy (`.2.5`, closed by an earlier session,
  re-verified intact this session)** — `agents.converse()`'s
  `DEFAULT_TOKEN_BUDGET` / `TokenBudgetExceeded` ceiling on the quadratic
  transcript cost described in `prd.md`'s cost table, plus the key-policy
  documentation in `backends/ocean_fable.py`'s module docstring.

Also removed the last Ollama-shaped hardcoded model default
(`model="llama3"`, `model="qwen2.5-coder:14b"`) from every caller that had
one — `agents.converse`, `discussion.run_discussion`, `turn.run_turn`,
`turn._ask_candidate_call`, `question_selector.generate_candidates` — each
now takes `model=None` and builds `kwargs = {} if model is None else
{"model": model}`, so the *backend's* `generate()` supplies its own correct
default only when the caller doesn't override. Which model answers is the
backend's call, not any caller's.

## Verification

Full self-check sweep, all `ok` (or, for `metrics.py`, `ok` plus its normal
demo table):

```
ocean.py, motive.py, chapters.py, question_selector.py, panel.py,
agents.py, discussion.py, turn.py, metrics.py, chat_core.py,
evidence_ledger.py, backends/base.py, backends/ocean_ollama.py,
backends/ocean_fable.py
```

`uv run pytest -q` from `/workspace` → `1 passed`.

`turn.py`'s self-check specifically proves the `Reply`/`.model` propagation
end to end: a `NamedReply` (same shape as `backends.base.Reply`) fed through
`run_turn()` in `SINGLE_PASS` mode against a *fresh* `ChatCore` (a hypothesis
already ruled out by a preceding scenario in the same self-check can't be
ruled out twice — `hypothesis_board.rule_out()` raises `KeyError` on a
non-live hypothesis; the fix this session was giving that scenario its own
`ChatCore` rather than reusing one already past that state) shows up as
`named_out.calls[0].model == "claude-fallback-model"`, proving a mid-call
fallback swap on Fable would reach the transcript, not just the "unknown"
degrade path a plain-str stub exercises.

## Children (all closed)

- `narrator-c5b.2.1` — Backend interface extracted behind `ocean.generate()`.
- `narrator-c5b.2.2` — `backends/ocean_fable.py` against `claude-fable-5`.
- `narrator-c5b.2.3` — sampler mapping re-derived per backend, no
  `repeat_penalty` pretending on Fable.
- `narrator-c5b.2.4` — no prompt names a model; transcripts record who
  answered or say `unknown`.
- `narrator-c5b.2.5` — token budget ceiling + key policy.

## Files touched (this session's increment on top of prior sessions)

- `backends/base.py` — added `Reply(str)`.
- `backends/ocean_ollama.py`, `backends/ocean_fable.py` — wrapped return
  values in `Reply`; added the try/relative-then-absolute import fallback
  for `Reply` (needed because `backends/` is a real package — `from .base
  import Reply` works when imported as `backends.ocean_ollama`, but running
  `python ocean_ollama.py` directly has no parent package context, so it
  falls back to `from base import Reply`).
- `agents.py`, `discussion.py`, `turn.py`, `question_selector.py` —
  `model="<hardcoded>"` → `model=None` plus the `kwargs = {} if model is
  None else {"model": model}` pattern at each call site.
- `metrics.py` — `chat_record()` gained a `"model"` field;
  `chat_session.jsonl` regenerated.
- Self-checks extended in all of the above to assert the new behavior, not
  just implement it.

No commits made — repo policy (CLAUDE.md, conservative profile) is not to
commit/push without explicit authority. `git status` at session end below.
