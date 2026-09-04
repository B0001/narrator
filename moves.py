"""Explicit move set: reveal, complicate, ask, abstain (C5).

Most chatbots have exactly one move -- say the thing -- which is exactly
wrong for this genre: a fair-play mystery is mostly about what you hold
back and when. Every turn picks a move before it picks words, and the move
plus the reason for it is logged next to the reply, because the PRD's
"every component's output is observable" rule applies to *why* a turn said
what it said, not just to what it said.

The four moves:

  reveal      -- state a conclusion. Only legal when admissibility.check()
                 says the citations are grounded; this is the one hard rule.
  complicate  -- surface a competing hypothesis or a wrinkle in the current
                 one, without asserting either is settled.
  ask         -- request the specific evidence the ledger is missing.
  abstain     -- explicitly decline to conclude anything yet, naming what's
                 missing, rather than staying silent about why.

The bug this guards against is the same one narrator-cby.4.1 already showed
once: a producer that can assert whatever it wants and a checker nobody
consults. Here the checker (`admissibility.check`) sits directly in the
`reveal` path -- there is no route from "I have a conclusion" to "I said it
out loud" that skips the check.

That claim needed one repair to stay true. A downgraded reveal's `reason`
below is built from the checker's `missing` tuple, which used to quote the
refused claim, and `turn._voice_prompt` handed the reason straight to the
persona -- so the sentence the checker had just refused arrived at the
speaking model anyway, while `move` still read `abstain` (narrator-7gj). The
route existed; it ran through the reason string. Two things close it, and
both matter: `admissibility` now names ids rather than quoting claims, and
`turn._voice_prompt` withholds a blocked turn's reason from the voice
entirely. `TurnLog.missing` below is unchanged -- the audit record is not
the leak, the persona's copy of it was.

    python3 moves.py   # self-check
"""

from dataclasses import dataclass

import admissibility

REVEAL = "reveal"
COMPLICATE = "complicate"
ASK = "ask"
ABSTAIN = "abstain"
MOVES = frozenset({REVEAL, COMPLICATE, ASK, ABSTAIN})


@dataclass
class TurnLog:
    turn: int
    move: str
    reason: str
    cited: tuple = ()
    missing: tuple = ()


def choose_move(ledger, turn, cited_ids, requested_move):
    """Pick the move for this turn and log it.

    `requested_move` is what the caller (persona layer, or a human driving
    the self-check) wants to do. The only move this function will ever
    downgrade is `reveal`: if the citations don't pass admissibility, the
    move becomes `abstain` and the reason names the missing evidence. Every
    other requested move is honored as asked, since only a conclusion needs
    gatekeeping -- asking a question or raising a complication doesn't
    assert anything the ledger has to back up.
    """
    if requested_move not in MOVES:
        raise ValueError(f"not a move: {requested_move!r}; must be one of {sorted(MOVES)}")

    if requested_move == REVEAL:
        verdict = admissibility.check(ledger, cited_ids)
        if not verdict.admissible:
            return TurnLog(
                turn, ABSTAIN,
                f"conclusion blocked by admissibility check; missing: {'; '.join(verdict.missing)}",
                cited=tuple(cited_ids), missing=verdict.missing,
            )
        return TurnLog(
            turn, REVEAL,
            f"citing {', '.join(cited_ids)}, all grounded in evidence the user has seen",
            cited=tuple(cited_ids),
        )

    if requested_move == ASK:
        return TurnLog(turn, ASK, "requesting evidence the ledger does not yet have", cited=tuple(cited_ids))
    if requested_move == COMPLICATE:
        return TurnLog(turn, COMPLICATE, "surfacing a competing reading, not settling one", cited=tuple(cited_ids))
    # ABSTAIN requested directly (not via a downgraded reveal).
    return TurnLog(turn, ABSTAIN, "declining to conclude yet", cited=tuple(cited_ids))


def _self_check():
    import tempfile

    from evidence_ledger import EvidenceLedger

    with tempfile.TemporaryDirectory() as d:
        with EvidenceLedger(f"{d}/l.jsonl") as ledger:
            ledger.write("seen1", 0, "user: 'the lights were off'", "stated_by_user")
            ledger.write("solid", 1, "so nobody was reading in there", "inferred_by_model", supports=("seen1",))
            ledger.write("shaky", 1, "the room was empty all evening", "inferred_by_model")

            # A grounded reveal goes through as reveal.
            log = choose_move(ledger, 1, ["solid"], REVEAL)
            assert log.move == REVEAL
            assert "solid" in log.reason

            # An ungrounded reveal is downgraded to abstain, and names what's missing.
            log = choose_move(ledger, 2, ["shaky"], REVEAL)
            assert log.move == ABSTAIN
            assert "shaky" in log.reason
            assert log.missing and any("shaky" in m for m in log.missing)

            # The other three moves are never gatekept by admissibility --
            # only an assertion needs backing, a question or a complication doesn't.
            log = choose_move(ledger, 3, ["shaky"], ASK)
            assert log.move == ASK
            log = choose_move(ledger, 4, [], COMPLICATE)
            assert log.move == COMPLICATE
            log = choose_move(ledger, 5, [], ABSTAIN)
            assert log.move == ABSTAIN

            # An unknown move is rejected outright, not silently coerced.
            try:
                choose_move(ledger, 6, [], "shrug")
            except ValueError:
                pass
            else:
                raise AssertionError("unknown move should have been rejected")

            # Every log carries both a move and a reason -- the observable
            # record the acceptance criteria asks for.
            for log in (
                choose_move(ledger, 7, ["solid"], REVEAL),
                choose_move(ledger, 8, ["shaky"], REVEAL),
                choose_move(ledger, 9, [], ASK),
            ):
                assert log.move and log.reason

    print("ok")


if __name__ == "__main__":
    _self_check()
