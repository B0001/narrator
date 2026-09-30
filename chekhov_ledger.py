"""Chekhov ledger: unresolved threads with an age counter (C5, narrator-c5b.3.6).

Every entry `evidence_ledger.py` writes is a thread: introduced on some turn,
then either paid off (cited, directly or transitively, by a conclusion that
actually landed) or left hanging. Chekhov's rule turned into a number: a gun
on the mantel that never fires is a thread whose age keeps climbing with no
use in sight. Two uses for the same count -- as a writing tool it flags guns
that never go off, and as chat policy it is a pacing signal: a conversation
that racks up old, unfired threads is losing pressure, independent of how
much context it still has.

`metrics.unresolved_threads` (narrator-c5b.3.7) already computed the "used"
half of this by replaying a saved log, and it says so in its own docstring:
"c5b.3.6's Chekhov ledger is where a richer definition belongs." The richer
half is retirement. A thread that was cited into a grounded conclusion and a
thread nobody has gotten to yet both look the same from outside -- neither is
in the "still open" set. But a thread the narrator *decided* to drop (a red
herring named as one, a lead that turned out to be nothing) is a third
thing, and conflating it with "not resolved yet" erases the decision. So
retirement is its own operation here, `retire()`, and it demands a reason
the same way `hypothesis_board.rule_out` does: a thread being dropped is a
claim about the story, and the claim needs a citable cause, not just an
absence from the open set.

This ledger is built to be driven two ways, matching the two use cases:
live, wired into `ChatCore` as evidence is observed and conclusions land
(so a running conversation can ask "what's still open, and how old" at any
turn); or replayed after the fact from a saved chat log, the way
`metrics.py` already replays citations. Either way the object is the same
three operations -- `observe`, `use`, `retire` -- plus three read-only
queries -- `status`, `open_threads`, `retirements`.

    python3 chekhov_ledger.py   # self-check
"""

from dataclasses import dataclass

OPEN = "open"
USED = "used"
RETIRED = "retired"


@dataclass
class RetirementEvent:
    entry_id: str
    turn: int
    reason: str


class ChekhovLedger:
    def __init__(self):
        self._introduced = {}   # entry_id -> turn first observed
        self._supports = {}     # entry_id -> tuple of ids it rests on
        self._used = set()      # ids paid off by a grounded conclusion, transitively
        self._retired = {}      # entry_id -> RetirementEvent, in first-retired order
        self._last_turn = 0

    def observe(self, entry_id, turn, supports=()):
        """Register a thread as of `turn`. Mirrors `EvidenceLedger.write`:
        every entry gets exactly one thread, so a duplicate id is a bug in
        the caller, not something to paper over silently."""
        if entry_id in self._introduced:
            raise ValueError(f"duplicate thread id {entry_id!r}")
        self._introduced[entry_id] = turn
        self._supports[entry_id] = tuple(supports)
        self._last_turn = max(self._last_turn, turn)

    def use(self, turn, cited_ids):
        """Pay off `cited_ids` and everything they transitively rest on, as
        of `turn`. Call this only for a move that actually concluded
        something -- a downgraded-to-abstain or directly-abstained turn must
        never call this, because admissibility refused the citation and
        nothing was actually established. Counting a refusal as a use would
        delete the exact case this ledger exists to keep: an ungrounded
        claim the checker rejected, still sitting there unfired.
        """
        self._last_turn = max(self._last_turn, turn)
        stack = list(cited_ids)
        while stack:
            eid = stack.pop()
            if eid in self._used:
                continue
            self._used.add(eid)
            stack.extend(self._supports.get(eid, ()))

    def retire(self, entry_id, turn, reason):
        """Explicitly close a thread without paying it off. The reason is
        required and becomes part of the permanent record (`retirements()`)
        -- retirement has to be legible as a decision, not inferable only
        from the thread's later absence from `open_threads()`.
        """
        if entry_id not in self._introduced:
            raise KeyError(f"cannot retire an unobserved thread {entry_id!r}")
        if not reason:
            raise ValueError("retire requires a reason naming why the thread is being dropped")
        if entry_id in self._used:
            raise ValueError(f"{entry_id!r} was already paid off by a conclusion; nothing to retire")
        if entry_id in self._retired:
            raise ValueError(f"{entry_id!r} was already retired")
        self._last_turn = max(self._last_turn, turn)
        self._retired[entry_id] = RetirementEvent(entry_id, turn, reason)

    def status(self, entry_id):
        """"used", "retired", or "open" for one thread."""
        if entry_id not in self._introduced:
            raise KeyError(f"no such thread {entry_id!r}")
        if entry_id in self._used:
            return USED
        if entry_id in self._retired:
            return RETIRED
        return OPEN

    def open_threads(self):
        """Every thread neither used nor retired, with its age in turns
        since introduction -- the guns still on the mantel. Age is measured
        against the highest turn this ledger has seen from any operation,
        not the introducing row, so it keeps climbing turn over turn even if
        the thread itself is never touched again.
        """
        return {
            eid: self._last_turn - turn
            for eid, turn in self._introduced.items()
            if eid not in self._used and eid not in self._retired
        }

    def retirements(self):
        """Every explicit retirement, in the order it happened. This is the
        half `open_threads()` cannot show: a thread missing from the open
        set because it was decided against, not because nobody has looked at
        it -- retirement is an entry here, never just silence.
        """
        return list(self._retired.values())


def _self_check():
    ck = ChekhovLedger()

    # A thread introduced and never touched again shows up open, aging with
    # the latest turn the ledger has seen -- the plain "gun on the mantel".
    ck.observe("footprint", 0)
    ck.observe("seen1", 0)
    ck.use(0, [])  # a turn happens; nothing was cited
    assert ck.open_threads() == {"footprint": 0, "seen1": 0}

    ck.observe("inferred", 1, supports=("seen1",))
    ck.use(1, ["inferred"])
    # Citing "inferred" pays off "seen1" too, transitively -- the thing it
    # rests on is exactly as resolved as the conclusion that used it.
    assert ck.status("inferred") == USED
    assert ck.status("seen1") == USED
    assert ck.status("footprint") == OPEN
    assert ck.open_threads() == {"footprint": 1}, ck.open_threads()

    # Age keeps climbing on a thread nobody ever comes back to.
    ck.use(5, [])
    assert ck.open_threads() == {"footprint": 5}

    # --- retirement: an explicit entry, not silence ---
    ck.observe("red_herring", 2)
    assert ck.status("red_herring") == OPEN
    assert ck.retirements() == []

    ck.retire("red_herring", 6, "the locked window turned out to have a mundane latch fault")
    assert ck.status("red_herring") == RETIRED
    assert "red_herring" not in ck.open_threads(), (
        "a retired thread must not also read as still open -- that would count it twice, "
        "once as dropped and once as merely unresolved"
    )
    events = ck.retirements()
    assert len(events) == 1
    assert events[0].entry_id == "red_herring"
    assert events[0].turn == 6
    assert "latch fault" in events[0].reason, "the reason is the point -- retirement without one is just deletion"

    # Retiring without a reason is refused, same discipline as
    # hypothesis_board.rule_out.
    ck.observe("unreasoned", 6)
    try:
        ck.retire("unreasoned", 7, "")
    except ValueError as e:
        assert "reason" in str(e)
    else:
        raise AssertionError("retire without a reason should have been rejected")
    assert ck.status("unreasoned") == OPEN, "a rejected retire must leave the thread untouched"

    # Retiring an id this ledger never saw is refused -- there is no thread
    # to close.
    try:
        ck.retire("never_observed", 7, "does not exist")
    except KeyError:
        pass
    else:
        raise AssertionError("retiring an unobserved id should have been rejected")

    # A used thread cannot be retired after the fact -- it was already paid
    # off, and pretending otherwise would erase that it fired.
    try:
        ck.retire("inferred", 7, "changed my mind")
    except ValueError as e:
        assert "already paid off" in str(e)
    else:
        raise AssertionError("retiring a used thread should have been rejected")

    # A thread cannot be retired twice.
    try:
        ck.retire("red_herring", 8, "again")
    except ValueError as e:
        assert "already retired" in str(e)
    else:
        raise AssertionError("double retirement should have been rejected")

    # A duplicate observe is a bug in the caller, same as EvidenceLedger.write.
    try:
        ck.observe("footprint", 9)
    except ValueError:
        pass
    else:
        raise AssertionError("re-observing an existing thread id should have been rejected")

    # An unknown id has no status to report.
    try:
        ck.status("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("status on an unobserved id should have been rejected")

    # Citing an id transitively through a support that was itself retired
    # still pays the support off -- a use always overrides an earlier
    # retirement decision, the same direction hypothesis_board lets evidence
    # override an earlier weight but never a rule_out.
    ck2 = ChekhovLedger()
    ck2.observe("clue", 0)
    ck2.retire("clue", 1, "looked irrelevant at the time")
    ck2.observe("payoff", 2, supports=("clue",))
    ck2.use(2, ["payoff"])
    assert ck2.status("payoff") == USED
    assert ck2.status("clue") == USED, (
        "a later conclusion citing a chain through a retired thread pays it off; "
        "the ledger tracks what actually happened, not what looked settled earlier"
    )
    assert ck2.retirements() == [RetirementEvent("clue", 1, "looked irrelevant at the time")], (
        "the retirement event itself stays on the record even after the thread "
        "is later used -- 'retired, then used anyway' is real history, not something "
        "to erase because the status moved on"
    )

    print("ok")


if __name__ == "__main__":
    _self_check()
