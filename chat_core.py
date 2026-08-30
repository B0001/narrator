"""Fair-play chat core: wires the ledger, board, admissibility check, and move
set into one turn loop (C5).

Each piece already stands on its own and self-checks standalone:

  evidence_ledger.py  -- what has actually been said or shown to the user,
                          with provenance, append-only.
  hypothesis_board.py -- which readings of "what's going on" are still live,
                          and how much weight each one carries.
  admissibility.py    -- inverted from mystery.validate_solvability(): is a
                          proposed conclusion actually derivable from things
                          the user has seen, or does the chain run out into
                          an inference or assumption nobody verified?
  moves.py            -- reveal / complicate / ask / abstain, with reveal
                          gated by the admissibility check.

`ChatCore` is the thin layer that makes them act as one conversation rather
than four unrelated data structures: board updates cite the ledger entries
that justified them (closing the gap `narrator-c5b.3.2`'s handoff flagged --
"the natural integration point is passing a ledger entry id ... as the
reason"), and a `conclude()` call is the only path from "I have a
conclusion" to "the board narrows," so a premature reveal can never
silently collapse a hypothesis the checker would have blocked.

    python3 chat_core.py   # self-check
"""

from dataclasses import dataclass

import moves
from evidence_ledger import EvidenceLedger
from hypothesis_board import HypothesisBoard


@dataclass
class TurnResult:
    turn_log: "moves.TurnLog"
    ruled_out: tuple = ()  # hypothesis ids the board dropped this turn, if any


class ChatCore:
    def __init__(self, ledger_path, hypotheses):
        self.ledger = EvidenceLedger(ledger_path)
        self.board = HypothesisBoard(hypotheses)

    def close(self):
        self.ledger.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def observe(self, entry_id, turn, claim, provenance, supports=()):
        """Record one piece of evidence. Returns the entry id for later citation."""
        self.ledger.write(entry_id, turn, claim, provenance, supports=supports)
        return entry_id

    def conclude(self, turn, cited_ids, requested_move, rule_out=None):
        """Attempt a conclusion. `rule_out`, if given, is (hypothesis_id,)
        and is only ever applied to the board when the move actually lands
        as `reveal` -- a downgraded-to-abstain turn must not narrow the
        board, because nothing was actually established.
        """
        log = moves.choose_move(self.ledger, turn, cited_ids, requested_move)
        ruled_out = ()
        if log.move == moves.REVEAL and rule_out is not None:
            hyp_id = rule_out
            self.board.rule_out(turn, hyp_id, f"ruled out by {log.reason}")
            ruled_out = (hyp_id,)
        return TurnResult(log, ruled_out)


def _self_check():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", [
            ("blackwood", "Lord Blackwood did it"),
            ("margaret", "Lady Margaret did it"),
            ("ellis", "Dr. Ellis did it"),
            ("jeeves", "Butler Jeeves did it"),
        ]) as core:
            for h in core.board.dump()["hypotheses"]:
                assert h["weight"] == 0.25

            # Turn 0: user reports a sighting. On its own this is real
            # evidence, but not yet enough to convict anyone -- an inference
            # drawn from it alone would be an inferred-only chain.
            core.observe("saw_margaret", 0, "user: 'I saw Lady Margaret near the Library'", "stated_by_user")
            weak = core.observe("weak_inference", 0, "Margaret must be the culprit", "inferred_by_model")

            # A reveal attempted on the unsupported inference must be blocked
            # and downgraded to abstain, naming exactly what's missing.
            result = core.conclude(0, [weak], moves.REVEAL, rule_out="margaret")
            assert result.turn_log.move == moves.ABSTAIN, "premature reveal must be downgraded"
            assert result.ruled_out == (), "an abstained turn must not touch the board"
            assert any("weak_inference" in m for m in result.turn_log.missing)
            assert core.board.live_ids() == ["blackwood", "margaret", "ellis", "jeeves"], (
                "board must be untouched by a blocked reveal"
            )

            # Turn 1: an artifact arrives that actually grounds the inference.
            core.observe("photo", 1, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            strong = core.observe(
                "strong_inference", 1, "Margaret could not have been at the scene", "inferred_by_model",
                supports=("photo",),
            )

            # Now the same shape of conclusion -- "rule out Margaret" -- is
            # admissible, and the board actually narrows.
            result = core.conclude(1, [strong], moves.REVEAL, rule_out="margaret")
            assert result.turn_log.move == moves.REVEAL, result.turn_log.reason
            assert result.ruled_out == ("margaret",)
            assert set(core.board.live_ids()) == {"blackwood", "ellis", "jeeves"}

            # The board's own record shows *why*, and it traces back to a
            # ledger citation, not a bare assertion.
            last_event = core.board.dump()["history"][-1]
            assert last_event["kind"] == "rule_out"
            assert "strong_inference" in last_event["reason"]

            # A conclusion citing nothing the user has ever seen is
            # inadmissible even if it sounds confident.
            baseless = core.observe("baseless", 2, "it was Jeeves all along", "assumed")
            result = core.conclude(2, [baseless], moves.REVEAL, rule_out="jeeves")
            assert result.turn_log.move == moves.ABSTAIN
            assert result.ruled_out == ()
            assert "jeeves" in core.board.live_ids(), "an assumption must never narrow the board"

        # The ledger on disk is the full observable record of the conversation.
        from evidence_ledger import load
        entries = load(f"{d}/ledger.jsonl")
        assert [e.id for e in entries] == [
            "saw_margaret", "weak_inference", "photo", "strong_inference", "baseless",
        ]

    print("ok")


if __name__ == "__main__":
    _self_check()
