# narrator-c5b.3.5 — Explicit move set: reveal, complicate, ask, abstain

Built as part of the parent bead's session. Full writeup, design decisions,
and verification run are in [`narrator-c5b.3.md`](narrator-c5b.3.md); this
file exists only to keep one handoff per closed bead.

Artifact: `moves.py` (`choose_move()`, `TurnLog`).

AC check:
- "Every logged turn carries a move and a reason" — `TurnLog` always has
  both fields populated; `_self_check()` asserts this across reveal, ask,
  complicate, and abstain.
- "A turn whose conclusion fails admissibility cannot choose reveal" —
  `choose_move()` runs `admissibility.check()` before ever returning
  `REVEAL`; on failure the move is downgraded to `ABSTAIN` with the missing
  evidence in the reason. `_self_check()` asserts this downgrade directly.

Verified: `uv run python moves.py` → `ok`.
