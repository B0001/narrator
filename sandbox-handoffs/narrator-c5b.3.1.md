# narrator-c5b.3.1 — Evidence ledger with provenance tags

Built as part of the parent bead's session. Full writeup, design decisions,
and verification run are in [`narrator-c5b.3.md`](narrator-c5b.3.md); this
file exists only to keep one handoff per closed bead.

Artifact: `evidence_ledger.py` (`EvidenceLedger`, `load()`).

AC check:
- "A ledger survives a run killed mid-write and reads back with the
  offending line named" — `_self_check()`'s truncated-write case asserts the
  `ValueError` names `{path}:4:` exactly (blank-line skip doesn't shift the
  count), mirroring `agents.load_transcript`'s same guarantee.
- "Every entry has a provenance tag; untagged entries are rejected at write
  time" — `EvidenceLedger.write()` raises `ValueError` before the entry is
  ever created if `provenance not in PROVENANCE_TAGS`.

Verified: `uv run python evidence_ledger.py` → `ok`.
