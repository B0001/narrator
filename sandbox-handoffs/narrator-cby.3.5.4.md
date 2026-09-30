# narrator-cby.3.5.4 -- Pooling metrics alongside the existing transcript measures

## What was asked

Extend `metrics.py` with the pooling measures: fraction of unique clues ever
voiced, fraction that received uptake, and time-to-first-unique-mention.
Report them next to turn share, agreement rate and concession, same
command-line output style as the existing metrics. Acceptance: all three
pooling measures print alongside the existing three from one saved
transcript.

## What I found in the working tree before starting

Bead was `in_progress` (claimed 2026-08-30) but had no notes, no comments,
and no trace of prior work: `grep -n "pooling\|unique_clue\|uptake" metrics.py`
turned up nothing before I started, and `sandbox-handoffs/narrator-cby.3.5.3.md`
(the sibling bead closed just before this one was claimed) says explicitly
"`narrator-cby.3.5.4` ... is a separate, still-open sibling bead. I did not
touch `metrics.py`" -- so whoever claimed this bead never got to it. I
treated it as unstarted and re-claimed it.

`metrics.py`, `discussion.py`, `agents.py`, `chat_session.jsonl`, and about a
dozen other files were already modified-but-uncommitted in the tree, all
`narrator-c5b`/`narrator-5ob` work (the C5 chatbot line) plus one unrelated
`discussion.py` diff (`model="llama3"` -> `model=None`, narrator-c5b.2.4). I
did not touch, revert, or rely on any of it -- confirmed by diffing
`metrics.py` before and after my change and checking the only file this bead
touched is `metrics.py` itself (`git status --porcelain` after my edit shows
no new untracked files and no other file newly modified).

## What I built

All in `metrics.py`, no other file touched:

- **`pooling_metrics(report)`** -- pure function over a
  `discussion.clue_report()` list. Denominator is unique clues only (a
  shared clue starts out known to everyone, so "did it get voiced" isn't the
  pooling-bias question). Returns `unique_total`, `unique_voiced`,
  `fraction_unique_voiced`, `unique_uptake`, `fraction_unique_uptake`, and
  `mean_time_to_first_unique_mention` (averaged over unique clues that were
  voiced at all -- a clue never mentioned has no turn to average in, same
  idiom as `movement` returning `None` for a speaker who never named a
  split). Zero unique clues -> fractions are `None`, not a crash. Empty
  report -> `ValueError`, same convention as `metrics([])`.
- **`pooling_summary(report)`** -- printable block, same key/value idiom as
  `chat_summary` (raw counts beside every rate).
- **`discussion_summary(records, report)`** -- `summary(records)` +
  `pooling_summary(report)`, which is the acceptance criterion made literal:
  one call, one block of text, both sets of measures from the same run.
- Self-check additions: hand-built `clue()` fixtures pin the pure-function
  edge cases (shared clues excluded from the denominator, zero-unique ->
  `None`, never-voiced excluded from the mean, empty report rejected) the
  same way `row()`/`entry()` pin the chat measures above them. Then a real
  pipeline run: `discussion.run_discussion(seed=3, n_agents=3, turns=4, ...)`
  with the same scripted echo-your-evidence `fake_generate` discussion.py's
  own self-check uses for this seed, so the pinned per-clue facts
  (`Clue_Alibi_Lord_Blackwood`/`Butler_Jeeves`/`Dr._Ellis` each voiced once
  by their sole holder, no uptake) are values already verified correct by
  `discussion.py`'s own test, not re-derived here. The transcript is written
  to a real temp file and reloaded via `load_transcript` before being scored
  -- a genuinely saved-and-reread transcript, not an in-memory shortcut.
  `discussion_summary`'s rendered output is asserted to contain both
  `summary`'s column headers (`turn%`, `agree%`, `vs even`) and
  `pooling_summary`'s labels (`unique clues`, `received uptake`, `mean
  time-to-first-mention`) in one string.
- Module docstring updated to describe the addition and to say plainly that
  this is *not* a new single-file CLI mode -- a discussion's pooling measures
  need both the transcript and the `clue_report` `run_discussion` returns
  alongside it, so scoring one is `metrics.discussion_summary(records,
  result["clue_report"])`, not `python3 metrics.py <path>`. I chose not to
  add sidecar-file CLI plumbing (`discussion.py` writing `clue_report` to a
  file next to the transcript, `metrics.py`'s `__main__` detecting it) --
  that would touch `discussion.py`, which is out of this bead's stated scope
  ("Extend metrics.py..."), and the acceptance criterion only asks for the
  numbers to print alongside each other, not for a new CLI invocation shape.
  If a maintainer wants real single-path CLI usage for discussion logs,
  that's a small, clearly separable follow-on -- I did not file a bead for it
  since it's an obvious next step stated in the docstring, not a surprise
  finding.

## Verification

```
$ uv run python metrics.py
ok
<debate table for scarce_resource.jsonl>

<chat table for chat_session.jsonl>

<debate table for the discussion self-check's own agents>

unique clues                  3
  ever voiced                 100.0%  (3/3)
  received uptake             0.0%  (0/3)
  mean time-to-first-mention  0.667
```

- `uv run python metrics.py` -> `ok`, all new assertions pass.
- `uv run python metrics.py scarce_resource.jsonl` and
  `uv run python metrics.py chat_session.jsonl` -> unchanged output, CLI
  behavior for the two existing log kinds is untouched.
- `uv run python agents.py`, `discussion.py`, `chat_core.py`,
  `evidence_ledger.py` all still print `ok` (untouched by this change).
- `uv run pytest` -> 1 passed.
- `git status --porcelain` after the change: only `metrics.py` newly
  modified beyond what was already dirty in the tree; no new files.

## Scope notes / follow-on

- Real single-path CLI support for discussion logs (sidecar clue-report file
  written by `discussion.py`, detected by `metrics.py`'s `__main__`) is a
  natural follow-on, not filed as its own bead per the note above -- happy to
  file one if a maintainer wants it tracked separately.
