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
            # Recorded whenever it actually changed. This used to skip changes
            # under 1e-12 while still applying them -- the same hole rule_out
            # had, just smaller: a weight the record cannot account for
            # (narrator-eeb). An unchanged weight writes nothing, so a no-op
            # reweight still leaves no entry.
            if new != old:
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

        # Renormalizing the survivors is a weight change like any other, so it
        # goes on the record like any other (narrator-eeb). Skipping it left
        # dump()["history"] unable to reconstruct dump()["hypotheses"] after
        # every single elimination, and the gap is not small: retiring a
        # hypothesis that held most of the mass can lift a survivor from last
        # place to board leader with nothing in the record attributing the move
        # to anything. The reason names the rule_out that caused it, which is
        # the entry immediately before these.
        remaining = self.live_ids()
        total = sum(self._hyps[hid].weight for hid in remaining)
        if total > 0:
            for hid in remaining:
                old_w = self._hyps[hid].weight
                new_w = old_w / total
                if new_w != old_w:
                    self.history.append(BoardEvent(
                        turn, "reweight", hid, old_w, new_w,
                        f"renormalized after ruling out {hypothesis_id}",
                    ))
                self._hyps[hid].weight = new_w

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

    # --- narrator-eeb: the record has to reconstruct the board. ---
    # This is the module's first stated rule -- "Every weight change is
    # appended to history, never just applied in place" -- turned into
    # something runnable rather than asserted in a docstring. rule_out used to
    # renormalize the survivors silently, so replaying the record produced a
    # board that summed to less than 1 with every survivor understated, on
    # every single elimination.
    def replay(dumped):
        """Rebuild the board from its own history: uniform prior, then every
        event applied in order.

        Each event's `old_weight` is checked against the replayed state
        *before* being applied, so a change that reached the board without
        reaching the record surfaces at the next event touching that
        hypothesis, instead of quietly absorbing into the final total.
        """
        n = len(dumped["hypotheses"])
        weights = {h["id"]: 1.0 / n for h in dumped["hypotheses"]}
        live = {h["id"]: True for h in dumped["hypotheses"]}
        for e in dumped["history"]:
            hid = e["hypothesis_id"]
            assert e["old_weight"] == weights[hid], (
                f"history gap on {hid}: the record replays to {weights[hid]}, "
                f"but the next event says it was {e['old_weight']}"
            )
            weights[hid] = e["new_weight"]
            if e["kind"] == "rule_out":
                live[hid] = False
        return weights, live

    dumped = board.dump()
    replayed_weights, replayed_live = replay(dumped)
    for h in dumped["hypotheses"]:
        assert replayed_weights[h["id"]] == h["weight"], (
            f"{h['id']}: record replays to {replayed_weights[h['id']]}, board holds {h['weight']}"
        )
        assert replayed_live[h["id"]] == h["live"], f"{h['id']}: liveness diverged from the record"

    # An elimination must leave the survivors' renormalization on the record,
    # not just the retired hypothesis's own entry -- and it has to say which
    # rule_out moved them, since a bare "reweight" next to an elimination is
    # exactly the silent switch the dump exists to rule out.
    fresh = HypothesisBoard([("a", "A"), ("b", "B"), ("c", "C")])
    fresh.reweight(1, {"a": 0.9, "b": 0.05, "c": 0.05}, "early evidence for a")
    before = {h["id"]: h["weight"] for h in fresh.dump()["hypotheses"]}
    n_before = len(fresh.dump()["history"])
    fresh.rule_out(2, "a", "user: 'definitely not A'")
    new_events = fresh.dump()["history"][n_before:]
    assert [e["kind"] for e in new_events] == ["rule_out", "reweight", "reweight"], new_events
    assert all("ruling out a" in e["reason"] for e in new_events[1:]), new_events
    # The move it records is not cosmetic: b goes from last place to leading.
    after = {h["id"]: h["weight"] for h in fresh.dump()["hypotheses"]}
    assert before["b"] == 0.05 and after["b"] == 0.5, (before, after)
    w, lv = replay(fresh.dump())
    for h in fresh.dump()["hypotheses"]:
        assert w[h["id"]] == h["weight"] and lv[h["id"]] == h["live"]

    # A move smaller than the 1e-12 that used to gate the record is still a
    # move: it changes the board, so the record has to carry it or the two
    # disagree from here on. Same hole rule_out had, at a scale that looks
    # ignorable -- which is exactly why it was left in.
    hair = HypothesisBoard([("a", "A"), ("b", "B"), ("c", "C")])
    hair.reweight(1, {"a": 1.0, "b": 1.0, "c": 1.0 + 1e-12}, "a hair of evidence for c")
    moved = [h for h in hair.dump()["hypotheses"] if h["weight"] != 1.0 / 3]
    assert len(moved) == 3, "this fixture is meant to move every weight"
    assert all(abs(h["weight"] - 1.0 / 3) < 1e-12 for h in moved), (
        "...and to move each by less than the epsilon that used to gate the record"
    )
    assert len(hair.dump()["history"]) == 3, "every moved weight needs its own entry"
    w, lv = replay(hair.dump())
    for h in hair.dump()["hypotheses"]:
        assert w[h["id"]] == h["weight"], f"{h['id']}: a sub-epsilon move left the record behind"

    # The other half of the same rule: a reweight that changes nothing writes
    # nothing. An entry claiming a move that never happened is the same kind
    # of lie as a move with no entry, just pointing the other way.
    quiet = HypothesisBoard([("a", "A"), ("b", "B"), ("c", "C")])
    quiet.reweight(1, {"a": 1.0, "b": 1.0, "c": 1.0}, "evidence that favours nobody")
    assert quiet.dump()["history"] == [], quiet.dump()["history"]

    print("ok")


if __name__ == "__main__":
    _self_check()
