"""Discriminating-question selector (C5).

A chatbot that asks a generic clarifying question -- "can you say more?" --
burns a turn without buying any information. Columbo's one-more-thing works
because it isn't generic: it's boring under every hypothesis except one, so
the answer he gets back actually moves the case, regardless of which
hypothesis was right. This module is that move, made scoreable: given the
live `HypothesisBoard` and a set of candidate questions, each tagged with
what every live hypothesis would predict the answer to be, score each
candidate by how much the live hypotheses' predictions actually disagree and
hand back the one with the widest spread.

Spread is weighted Gini impurity over the live hypotheses' predicted
answers, using each hypothesis's *current* board weight -- not a bare count
of distinct answers. A question that splits two heavyweight hypotheses apart
is worth more right now than one that only separates a hypothesis the board
has nearly ruled out already; the selector reads the board's live belief
state, not just its live id list, which is what "given a board" in the
acceptance criteria is asking for. A question every live hypothesis answers
identically scores exactly 0 -- it is uninformative regardless of who is
right -- and is rejected outright rather than merely scored low, so a caller
can never accidentally spend a turn on it.

Failing to ask is not neutral. Anthropic's persona-vector work found that
responses to underspecified queries push a model toward hallucination --
guessing fills the silence a real question would have used instead. So this
module's other half is the same abstention spine as `admissibility.py`: if
nothing on the table actually splits the board, that is itself the
observable, logged outcome (`chosen=None`), not a silent fallback to asking
whatever came first in the list.

Cap at one question per turn: `select_question` always returns a single
winner (or none), never a ranked list to ask through -- Columbo gets one
more thing, not several.

    python3 question_selector.py   # self-check
"""

from dataclasses import dataclass

_EPS = 1e-9


@dataclass
class Candidate:
    id: str
    text: str
    predicted_answers: dict  # hypothesis_id -> predicted answer, for every live hypothesis


@dataclass
class ScoredCandidate:
    id: str
    text: str
    score: float
    discriminates: bool
    reason: str


@dataclass
class SelectionLog:
    turn: int
    scored: tuple  # ScoredCandidate, in input order -- the full observable record
    chosen: str  # candidate id, or None if nothing on the table splits the board


def _live_weights(board):
    return {h["id"]: h["weight"] for h in board.dump()["hypotheses"] if h["live"]}


def _spread(live_weights, predicted_answers):
    """Weighted Gini impurity of predicted answers over live hypotheses.

    0.0 exactly when every live hypothesis predicts the same answer -- the
    "boring for everyone" case, which carries no information no matter which
    hypothesis turns out to be true. Approaches 1.0 as predictions split
    cleanly across many hypotheses of comparable current weight.
    """
    missing = sorted(set(live_weights) - set(predicted_answers))
    if missing:
        raise KeyError(f"candidate missing a predicted answer for live hypotheses: {missing}")

    groups = {}
    for hid, w in live_weights.items():
        answer = predicted_answers[hid]
        groups[answer] = groups.get(answer, 0.0) + w
    total = sum(groups.values())
    return 1.0 - sum((mass / total) ** 2 for mass in groups.values()), groups


def score_candidate(board, candidate):
    """Score one candidate question against the board's live hypotheses.

    Reads only `board`'s live ids and current weights -- the same "given a
    board" surface `HypothesisBoard.dump()` already exposes, nothing about
    how those weights got there.
    """
    live_weights = _live_weights(board)
    spread, groups = _spread(live_weights, candidate.predicted_answers)
    discriminates = spread > _EPS
    if not discriminates:
        (only_answer,) = groups.keys()
        reason = f"every live hypothesis predicts the same answer ({only_answer!r}); spread=0.0"
    else:
        by_mass = sorted(groups.items(), key=lambda kv: -kv[1])
        breakdown = ", ".join(f"{answer!r} ({mass:.2f})" for answer, mass in by_mass)
        reason = f"splits live hypotheses into {len(groups)} answer groups: {breakdown}; spread={spread:.4f}"
    return ScoredCandidate(candidate.id, candidate.text, spread, discriminates, reason)


def select_question(board, turn, candidates):
    """Score every candidate and choose the one with the widest spread.

    Every candidate is scored and logged, discriminating or not -- the
    rejected ones are exactly what makes this auditable rather than a black
    box that just hands back an answer. Only a discriminating candidate
    (spread > 0) can win; if none discriminates, `chosen` is `None` and that
    absence is itself the logged outcome, not a silent pick of whatever
    candidate happened to come first.
    """
    if not candidates:
        raise ValueError("no candidate questions to select from")

    scored = tuple(score_candidate(board, c) for c in candidates)
    winners = [s for s in scored if s.discriminates]
    chosen = max(winners, key=lambda s: s.score).id if winners else None
    return SelectionLog(turn, scored, chosen)


def _self_check():
    from hypothesis_board import HypothesisBoard

    board = HypothesisBoard([
        ("blackwood", "Lord Blackwood did it"),
        ("margaret", "Lady Margaret did it"),
        ("ellis", "Dr. Ellis did it"),
        ("jeeves", "Butler Jeeves did it"),
    ])

    boring = Candidate(
        "breakfast", "What did you have for breakfast?",
        {"blackwood": "doesn't matter", "margaret": "doesn't matter",
         "ellis": "doesn't matter", "jeeves": "doesn't matter"},
    )
    # Splits everyone apart evenly -- the widest possible spread on an
    # equal-weight board.
    full_split = Candidate(
        "whereabouts", "Where were you at the time of the murder?",
        {"blackwood": "study", "margaret": "garden", "ellis": "library", "jeeves": "kitchen"},
    )
    # Columbo's actual shape: boring under every hypothesis except one --
    # three suspects give the same throwaway answer, one gives a different
    # one. Still discriminating, just less so than a full four-way split.
    one_off = Candidate(
        "umbrella", "Did you notice the umbrella stand by the door?",
        {"blackwood": "no", "margaret": "no", "ellis": "no", "jeeves": "yes, I moved it"},
    )

    log = select_question(board, 0, [boring, full_split, one_off])
    assert log.chosen == "whereabouts", log.chosen

    by_id = {s.id: s for s in log.scored}
    assert by_id["breakfast"].discriminates is False
    assert by_id["breakfast"].score == 0.0
    assert by_id["whereabouts"].discriminates is True
    assert by_id["umbrella"].discriminates is True
    # A clean four-way split on an equal-weight board beats a 3-vs-1 split.
    assert by_id["whereabouts"].score > by_id["umbrella"].score > 0.0
    # Every candidate is logged, not just the winner.
    assert {s.id for s in log.scored} == {"breakfast", "whereabouts", "umbrella"}
    assert "doesn't matter" in by_id["breakfast"].reason

    # A board that's already leaning hard on one hypothesis changes which
    # question is worth asking: a question that only tells margaret and
    # jeeves apart is worth little once they're both nearly dead, even
    # though it looked fine on the equal-weight board above.
    board.reweight(1, {"blackwood": 0.7, "ellis": 0.2, "margaret": 0.05, "jeeves": 0.05},
                    "user: 'I'm almost certain it was Lord Blackwood'")
    margaret_vs_jeeves = Candidate(
        "silver", "Was the silver polished that evening?",
        {"blackwood": "no idea", "ellis": "no idea", "margaret": "yes", "jeeves": "no"},
    )
    blackwood_vs_rest = Candidate(
        "study_light", "Was the study light on?",
        {"blackwood": "off", "ellis": "on", "margaret": "on", "jeeves": "on"},
    )
    log2 = select_question(board, 1, [margaret_vs_jeeves, blackwood_vs_rest])
    assert log2.chosen == "study_light", (
        "splitting the heavyweight hypothesis apart from the rest must beat "
        "splitting two hypotheses the board has already mostly ruled out"
    )

    # A candidate that omits a prediction for a live hypothesis is a bug in
    # the caller, not something to silently score around.
    incomplete = Candidate("bad", "text", {"blackwood": "x"})
    try:
        select_question(board, 2, [incomplete])
    except KeyError as e:
        assert "ellis" in str(e) or "margaret" in str(e) or "jeeves" in str(e)
    else:
        raise AssertionError("a candidate missing a live hypothesis's prediction should have been rejected")

    # No candidates at all is a caller error, not an empty-but-valid result.
    try:
        select_question(board, 3, [])
    except ValueError:
        pass
    else:
        raise AssertionError("selecting from zero candidates should have been rejected")

    # A turn where every candidate is uninformative logs that honestly --
    # `chosen` is None, not a fallback pick.
    all_boring = [
        Candidate("a", "text a", {"blackwood": "same", "ellis": "same", "margaret": "same", "jeeves": "same"}),
        Candidate("b", "text b", {"blackwood": "same2", "ellis": "same2", "margaret": "same2", "jeeves": "same2"}),
    ]
    log3 = select_question(board, 4, all_boring)
    assert log3.chosen is None
    assert all(not s.discriminates for s in log3.scored)

    # Ruling a hypothesis out narrows what "live" means for scoring --
    # predictions for a dead hypothesis are no longer required.
    board.rule_out(5, "jeeves", "user: 'Jeeves was in London that whole week'")
    still_valid = Candidate(
        "alibi", "Where were you at the time of the murder?",
        {"blackwood": "study", "margaret": "garden", "ellis": "library"},
    )
    log4 = select_question(board, 5, [still_valid])
    assert log4.chosen == "alibi"

    print("ok")


if __name__ == "__main__":
    _self_check()
