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

`chekhov_ledger.py` (narrator-c5b.3.6) rides along the same two operations:
every `observe()` registers a thread, and `conclude()` pays off its citations
in the Chekhov ledger the instant (and only the instant) a reveal actually
lands -- an abstained or downgraded turn touches neither the hypothesis board
nor the Chekhov ledger, for the same reason in both places: nothing was
actually established. `retire()` is the one operation with no automatic
trigger, because retirement is a narrative decision (a lead named as a red
herring, a thread the story is done with) that nothing in the ledger or the
board can infer on its own; it has to be called explicitly, with a reason.

    python3 chat_core.py   # self-check
"""

from dataclasses import dataclass

import moves
from chekhov_ledger import ChekhovLedger
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
        self.chekhov = ChekhovLedger()

    def close(self):
        self.ledger.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def observe(self, entry_id, turn, claim, provenance, supports=(), prompted_by=None):
        """Record one piece of evidence. Returns the entry id for later citation.

        `prompted_by` (narrator-5ob), if given, is the id of the question
        (`question_selector.question_id`) that drew this entry out -- the
        caller's business to know, since `ChatCore` itself does not track
        which question is outstanding at any given moment (a chat could ask,
        get no answer, ask something else, and only later have the user
        circle back). Passed straight to the ledger; the Chekhov ledger has
        no use for it, since it tracks payoff through `supports`, not through
        why an entry exists.
        """
        self.ledger.write(entry_id, turn, claim, provenance, supports=supports, prompted_by=prompted_by)
        self.chekhov.observe(entry_id, turn, supports=supports)
        return entry_id

    def conclude(self, turn, cited_ids, requested_move, rule_out=None):
        """Attempt a conclusion. `rule_out`, if given, is (hypothesis_id,)
        and is only ever applied to the board when the move actually lands
        as `reveal` -- a downgraded-to-abstain turn must not narrow the
        board, because nothing was actually established. The Chekhov ledger
        follows the same rule: citations only pay off threads when the move
        that cited them actually landed as `reveal`.
        """
        log = moves.choose_move(self.ledger, turn, cited_ids, requested_move)
        ruled_out = ()
        if log.move == moves.REVEAL:
            self.chekhov.use(turn, log.cited)
            if rule_out is not None:
                hyp_id = rule_out
                self.board.rule_out(turn, hyp_id, f"ruled out by {log.reason}")
                ruled_out = (hyp_id,)
        return TurnResult(log, ruled_out)

    def retire(self, entry_id, turn, reason):
        """Explicitly close a thread without paying it off (narrator-c5b.3.6).
        Requires a reason for the same cause `hypothesis_board.rule_out`
        does: dropping a lead is a decision about the story, and the record
        should show what decided it rather than just the lead's later
        absence from the open set."""
        self.chekhov.retire(entry_id, turn, reason)


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
            # A blocked reveal must leave the Chekhov ledger untouched too --
            # the same "abstained means nothing was established" rule the
            # board is held to.
            assert set(core.chekhov.open_threads()) == {"saw_margaret", "weak_inference"}

            # Turn 1: an artifact arrives that actually grounds the inference.
            # narrator-5ob: it arrives because turn 0 asked about it -- passed
            # through to the ledger untouched, ChatCore itself tracks nothing
            # about which question is outstanding.
            core.observe(
                "photo", 1, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact",
                prompted_by="0:whereabouts",
            )
            assert core.ledger.get("photo").prompted_by == "0:whereabouts"
            assert core.ledger.get("saw_margaret").prompted_by is None, "an unprompted entry stays unprompted"
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

            # The reveal landing pays off "strong_inference" in the Chekhov
            # ledger, and "photo" with it -- the thing the citation
            # transitively rests on is exactly as resolved as the citation
            # itself. "saw_margaret" and "weak_inference" are still open:
            # nothing has cited either of them yet.
            assert core.chekhov.status("strong_inference") == "used"
            assert core.chekhov.status("photo") == "used"
            assert set(core.chekhov.open_threads()) == {"saw_margaret", "weak_inference"}

            # The board's own record shows *why*, and it traces back to a
            # ledger citation, not a bare assertion. The elimination is no
            # longer the last entry -- narrator-eeb made rule_out record the
            # survivors' renormalization too -- so find it rather than
            # assuming the board stopped moving when it fired.
            history = core.board.dump()["history"]
            rule_outs = [e for e in history if e["kind"] == "rule_out"]
            assert len(rule_outs) == 1
            assert "strong_inference" in rule_outs[-1]["reason"]

            # ...and the weight the eliminated hypothesis shed is accounted
            # for: the entries after it name the rule_out that moved them.
            tail = history[history.index(rule_outs[-1]) + 1:]
            assert tail, "ruling out margaret must leave the survivors' renormalization on the record"
            assert all(e["kind"] == "reweight" and "ruling out margaret" in e["reason"] for e in tail), tail

            # A conclusion citing nothing the user has ever seen is
            # inadmissible even if it sounds confident.
            baseless = core.observe("baseless", 2, "it was Jeeves all along", "assumed")
            result = core.conclude(2, [baseless], moves.REVEAL, rule_out="jeeves")
            assert result.turn_log.move == moves.ABSTAIN
            assert result.ruled_out == ()
            assert "jeeves" in core.board.live_ids(), "an assumption must never narrow the board"
            # A blocked reveal pays off nothing in the Chekhov ledger either.
            assert core.chekhov.status("baseless") == "open"

            # narrator-c5b.3.6: "weak_inference" is a thread that was
            # introduced on turn 0 and never paid off by anything -- exactly
            # the gun-on-the-mantel case the ledger exists to surface, aged
            # two turns by now. Retiring it is a decision the story makes
            # explicitly, not something that happens by falling out of the
            # open set.
            assert core.chekhov.open_threads()["weak_inference"] == 2
            core.retire("weak_inference", 2, "the sighting alone never led anywhere; nothing was ever built on it")
            assert core.chekhov.status("weak_inference") == "retired"
            assert "weak_inference" not in core.chekhov.open_threads(), (
                "a retired thread must not also count as still-open"
            )
            assert set(core.chekhov.open_threads()) == {"saw_margaret", "baseless"}
            retirements = core.chekhov.retirements()
            assert len(retirements) == 1
            assert retirements[0].entry_id == "weak_inference" and retirements[0].turn == 2
            assert "never led anywhere" in retirements[0].reason, (
                "retirement is an entry on the record, not silence -- the reason has to be there to read"
            )

        # The ledger on disk is the full observable record of the conversation.
        from evidence_ledger import load
        entries = load(f"{d}/ledger.jsonl")
        assert [e.id for e in entries] == [
            "saw_margaret", "weak_inference", "photo", "strong_inference", "baseless",
        ]

    print("ok")


if __name__ == "__main__":
    _self_check()
