"""Board of live interpretations for what the chat user wants (C5).

A chatbot that locks onto the first plausible reading of what the user is
after is the thing that makes it feel stupid, and the genre's whole engine is
that the obvious suspect is wrong. So instead of one running guess, the board
holds three to five weighted hypotheses at once and moves weight between them
as evidence comes in.

The two rules that keep this honest:

  - Every weight change is appended to history, never just applied in place.
    A conversation that reverses direction has to show both the swing and the
    swing back in the dump -- a silent switch from A to B is indistinguishable
    from having been wrong about A the whole time, and the dump exists so a
    learner can tell those apart.
  - Weight alone can never retire a hypothesis. `reweight` refuses to drive a
    live hypothesis to zero; only `rule_out`, which demands a reason, can drop
    the live count. That is the difference between "unlikely" and "dead" --
    conflating them is how a system talks itself into premature certainty.

    python3 hypothesis_board.py   # self-check
"""

from dataclasses import dataclass

MIN_HYPOTHESES = 3
MAX_HYPOTHESES = 5


@dataclass
class Hypothesis:
    id: str
    statement: str
    weight: float
    live: bool = True


@dataclass
class BoardEvent:
    turn: int
    kind: str  # "reweight" or "rule_out"
    hypothesis_id: str
    old_weight: float
    new_weight: float
    reason: str


class HypothesisBoard:
    def __init__(self, hypotheses):
        """hypotheses: list of (id, statement) pairs, 3 to 5 of them.

        All start at equal weight -- the board has no prior favorite before
        any evidence has arrived.
        """
        if not (MIN_HYPOTHESES <= len(hypotheses) <= MAX_HYPOTHESES):
            raise ValueError(
                f"hold {MIN_HYPOTHESES} to {MAX_HYPOTHESES} live readings, got {len(hypotheses)}"
            )
        ids = [hid for hid, _ in hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError(f"hypothesis ids must be unique, got {ids}")
        n = len(hypotheses)
        self._order = ids
        self._hyps = {hid: Hypothesis(hid, statement, 1.0 / n) for hid, statement in hypotheses}
        self.history = []

    def live_ids(self):
        return [hid for hid in self._order if self._hyps[hid].live]

    def reweight(self, turn, weights, reason):
        """Shift belief across live hypotheses without eliminating anyone.

        `weights` maps hypothesis id -> new relative weight, for a subset of
        the live hypotheses; anyone left out keeps their current weight going
        into renormalization. The result always sums to 1 over live
        hypotheses. This can push a hypothesis arbitrarily close to zero but
        never to it -- driving it out entirely is `rule_out`'s job, because
        that is the operation that has to justify itself with evidence.
        """
        if not reason:
            raise ValueError("reweight requires a reason citing the evidence")
        live = self.live_ids()
        unknown = set(weights) - set(live)
        if unknown:
            raise KeyError(f"not live hypotheses: {sorted(unknown)}")
        for hid, w in weights.items():
            if w <= 0:
                raise ValueError(f"reweight cannot zero out {hid!r}; rule it out with evidence instead")

        raw = {hid: weights.get(hid, self._hyps[hid].weight) for hid in live}
        total = sum(raw.values())
        for hid in live:
            old = self._hyps[hid].weight
            new = raw[hid] / total
            if abs(new - old) > 1e-12:
                self.history.append(BoardEvent(turn, "reweight", hid, old, new, reason))
            self._hyps[hid].weight = new

    def rule_out(self, turn, hypothesis_id, reason):
        """Retire a hypothesis for cause. Refuses to touch the last survivor:
        a board is allowed to converge on one answer, but only by naming, on
        the record, the evidence that killed every alternative.
        """
        if not reason:
            raise ValueError("rule_out requires evidence naming why this reading is dead")
        live = self.live_ids()
        if hypothesis_id not in live:
            raise KeyError(f"{hypothesis_id!r} is not a live hypothesis")
        if len(live) <= 1:
            raise ValueError("cannot rule out the last surviving hypothesis")

        hyp = self._hyps[hypothesis_id]
        old = hyp.weight
        hyp.live = False
        hyp.weight = 0.0
        self.history.append(BoardEvent(turn, "rule_out", hypothesis_id, old, 0.0, reason))

        remaining = self.live_ids()
        total = sum(self._hyps[hid].weight for hid in remaining)
        if total > 0:
            for hid in remaining:
                self._hyps[hid].weight /= total

    def dump(self):
        """Full board state plus the weight-shift history. A reversal shows
        up here as two opposing entries on the same hypothesis id, not as a
        board that just quietly points somewhere else than it used to.
        """
        return {
            "hypotheses": [
                {
                    "id": hid,
                    "statement": self._hyps[hid].statement,
                    "weight": self._hyps[hid].weight,
                    "live": self._hyps[hid].live,
                }
                for hid in self._order
            ],
            "history": [
                {
                    "turn": e.turn,
                    "kind": e.kind,
                    "hypothesis_id": e.hypothesis_id,
                    "old_weight": e.old_weight,
                    "new_weight": e.new_weight,
                    "reason": e.reason,
                }
                for e in self.history
            ],
        }


def _self_check():
    # 3 to 5 hypotheses, nothing outside that range.
    try:
        HypothesisBoard([("a", "wants X"), ("b", "wants Y")])
    except ValueError:
        pass
    else:
        raise AssertionError("2 hypotheses should have been rejected")

    try:
        HypothesisBoard([(str(i), "x") for i in range(6)])
    except ValueError:
        pass
    else:
        raise AssertionError("6 hypotheses should have been rejected")

    try:
        HypothesisBoard([("a", "x"), ("a", "y"), ("c", "z")])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate ids should have been rejected")

    board = HypothesisBoard([
        ("refund", "user wants a refund"),
        ("exchange", "user wants an exchange"),
        ("complaint", "user just wants to vent"),
        ("info", "user wants the return policy explained"),
    ])
    for h in board.dump()["hypotheses"]:
        assert h["weight"] == 0.25, "no evidence yet: every reading starts equally weighted"
        assert h["live"]

    # Turn 1: user says the item arrived broken -- favors refund and exchange.
    board.reweight(1, {"refund": 0.4, "exchange": 0.4, "complaint": 0.1, "info": 0.1}, "user: 'it arrived broken'")
    after_t1 = {h["id"]: h["weight"] for h in board.dump()["hypotheses"]}
    assert after_t1["refund"] > 0.25 and after_t1["exchange"] > 0.25

    # Turn 2: the conversation reverses -- user says they don't want to send it back.
    board.reweight(2, {"refund": 0.1, "exchange": 0.1, "complaint": 0.5, "info": 0.3}, "user: 'I don't want to ship it back'")
    after_t2 = {h["id"]: h["weight"] for h in board.dump()["hypotheses"]}
    assert after_t2["refund"] < after_t1["refund"], "reversal must actually move the weight back down"
    assert after_t2["complaint"] > after_t1["complaint"]

    # The reversal must be legible in history, not just visible as a changed
    # final state: refund has to show one entry going up and a later one
    # going down, on the record.
    refund_events = [e for e in board.dump()["history"] if e["hypothesis_id"] == "refund"]
    assert len(refund_events) == 2
    assert refund_events[0]["new_weight"] > refund_events[0]["old_weight"], "turn 1 raised refund"
    assert refund_events[1]["new_weight"] < refund_events[1]["old_weight"], "turn 2 walked it back down"

    for h in board.dump()["hypotheses"]:
        assert h["live"], "reweight alone must never retire a hypothesis"
    total = sum(h["weight"] for h in board.dump()["hypotheses"])
    assert abs(total - 1.0) < 1e-9, "live weights must sum to 1"

    # Weight cannot be used to quietly zero a hypothesis out.
    try:
        board.reweight(3, {"info": 0.0}, "no such evidence")
    except ValueError:
        pass
    else:
        raise AssertionError("reweight should refuse to drive a live hypothesis to zero")

    # rule_out demands a reason.
    try:
        board.rule_out(3, "info", "")
    except ValueError:
        pass
    else:
        raise AssertionError("rule_out without a reason should have been rejected")

    # rule_out on a hypothesis that isn't live, or doesn't exist.
    try:
        board.rule_out(3, "nonexistent", "made up")
    except KeyError:
        pass
    else:
        raise AssertionError("rule_out on an unknown id should have been rejected")

    # Turn 3: user explicitly says the item isn't broken -- kills info-only reading.
    board.rule_out(3, "info", "user: 'nothing's wrong with the return policy, I just don't like the color'")
    live_after = board.live_ids()
    assert set(live_after) == {"refund", "exchange", "complaint"}
    assert abs(sum(board._hyps[h].weight for h in live_after) - 1.0) < 1e-9

    try:
        board.rule_out(4, "info", "already dead")
    except KeyError:
        pass
    else:
        raise AssertionError("cannot rule out a hypothesis twice")

    # Collapse down to two, then to the last one -- each step needs its own
    # evidence-bearing reason, and the very last survivor can't be touched.
    board.rule_out(4, "complaint", "user: 'no I'm not upset, I just want it swapped'")
    assert set(board.live_ids()) == {"refund", "exchange"}

    board.rule_out(5, "refund", "user: 'I don't want my money back, I want the right size'")
    assert board.live_ids() == ["exchange"]
    assert board._hyps["exchange"].weight == 1.0

    try:
        board.rule_out(6, "exchange", "nothing left to rule it out against")
    except ValueError:
        pass
    else:
        raise AssertionError("the last surviving hypothesis must not be ruleable-out")

    kinds = [e["kind"] for e in board.dump()["history"]]
    assert kinds.count("rule_out") == 3, "one rule_out per elimination, each with its own reason"
    assert kinds.count("reweight") > 0, "the two evidence turns must leave reweight events behind"

    print("ok")


if __name__ == "__main__":
    _self_check()
