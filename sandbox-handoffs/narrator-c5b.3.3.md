# narrator-c5b.3.3 — Invert validate_solvability() into a runtime admissibility check

Built as part of the parent bead's session. Full writeup, design decisions,
and verification run are in [`narrator-c5b.3.md`](narrator-c5b.3.md); this
file exists only to keep one handoff per closed bead.

Artifact: `admissibility.py` (`check()`, `Verdict`).

AC check:
- "A conclusion resting on an inferred-only chain is blocked and reported" —
  `_self_check()` exercises an unsupported inference, an assumed premise,
  and a mixed chain (one grounded leg, one assumed leg); all three are
  blocked and `Verdict.missing` names the offending entry.
- "The check runs on the ledger, with no access to the generator's ground
  truth" — `check()`/`_grounded()` only ever call `ledger.get()`/`ledger
  .__contains__`; there is no other input.

Verified: `uv run python admissibility.py` → `ok`.
