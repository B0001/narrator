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

import json
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


def candidate_prompt(board, n=3):
    """Build the prompt for the candidate-generation call.

    The only state that reaches this string is `board.dump()`'s *live*
    hypotheses -- id and statement, nothing else. No ledger entry, no
    ruled-out hypothesis, no persona, no hidden ground truth is in scope,
    because this function's signature does not admit any of those: it takes
    a board and a count, full stop. That is the same discipline
    `suspects_consistent_with_clues()` applies to `sim.culprit` and
    `chapters.py` applies to the culprit's name before the deduction chapter
    -- provable by what the function *can* reach, not by a promise about
    what it won't.
    """
    live = [h for h in board.dump()["hypotheses"] if h["live"]]
    lines = "\n".join(f"- {h['id']}: {h['statement']}" for h in live)
    return (
        "You are the question-generation channel of a fair-play mystery chatbot.\n"
        f"Propose up to {n} short clarifying questions to ask the user next. For each "
        "question, predict what the user would answer under every live hypothesis "
        "below -- you do not know which hypothesis is true, only what each one, if "
        "it were true, would predict the answer to be.\n\n"
        f"Live hypotheses:\n{lines}\n\n"
        "Return JSON only, no other text: "
        '{"candidates": [{"id": "<short id>", "text": "<question>", '
        '"predicted_answers": {"<hypothesis id>": "<predicted answer>", '
        "... one entry per live hypothesis id listed above}}]}"
    )


def parse_candidates(board, raw):
    """Parse the candidate-generation call's raw output into `Candidate`s.

    Checks the same completeness rule `score_candidate` enforces --
    every live hypothesis needs a predicted answer -- but here, so a
    malformed or lazy model response is a named `ValueError` at the source,
    not a `KeyError` raised later, deep inside `select_question`.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"candidate generator did not return JSON: {raw!r}") from e

    raw_candidates = data.get("candidates") if isinstance(data, dict) else None
    if not raw_candidates:
        raise ValueError(f"candidate generator returned no candidates: {raw!r}")

    live_ids = {h["id"] for h in board.dump()["hypotheses"] if h["live"]}
    candidates = []
    for c in raw_candidates:
        predicted = c.get("predicted_answers", {})
        missing = sorted(live_ids - set(predicted))
        if missing:
            raise ValueError(
                f"candidate {c.get('id')!r} missing predictions for live hypotheses: {missing}"
            )
        candidates.append(Candidate(c["id"], c["text"], {hid: predicted[hid] for hid in live_ids}))
    return tuple(candidates)


def generate_candidates(board, profile, generate_fn, model="qwen2.5-coder:14b", n=3):
    """One model call, over `candidate_prompt(board, n)` alone, producing
    scoreable candidates. `generate_fn` follows the repo-wide injectable
    shape (`profile, prompt, model=...`), same as `ocean.generate` /
    `turn.run_turn` / `panel.run_panel`, so a caller can pin `profile` to a
    persona-neutral one (`turn.REASONING_PROFILE`) the same way the
    reasoning channel itself is pinned, and a self-check can swap in a
    scripted stub with no model at all.
    """
    raw = generate_fn(profile, candidate_prompt(board, n=n), model=model)
    return parse_candidates(board, raw)


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

    # --- candidate generation: the prompt can only ever carry live hypotheses.
    # jeeves was ruled out above; a leaked reference to jeeves or his statement
    # would mean this function saw more than board.dump() exposes as live --
    # the same leak check chapters.py runs for sim.culprit before the
    # deduction chapter, aimed at the one piece of state this module must
    # never see: a hypothesis the board itself has already killed.
    prompt = candidate_prompt(board, n=2)
    assert "jeeves" not in prompt.lower()
    assert "Butler Jeeves" not in prompt
    for live_id in ("blackwood", "margaret", "ellis"):
        assert live_id in prompt

    def fake_candidate_generate(profile, prompt, model=None):
        assert "jeeves" not in prompt.lower(), "candidate generator must never see a ruled-out hypothesis"
        return json.dumps({"candidates": [
            {
                "id": "whereabouts2",
                "text": "Where were you at the time of the murder?",
                "predicted_answers": {"blackwood": "study", "margaret": "garden", "ellis": "library"},
            },
            {
                "id": "boring2",
                "text": "What did you have for breakfast?",
                "predicted_answers": {"blackwood": "eggs", "margaret": "eggs", "ellis": "eggs"},
            },
        ]})

    generated = generate_candidates(board, profile=None, generate_fn=fake_candidate_generate)
    assert {c.id for c in generated} == {"whereabouts2", "boring2"}
    log5 = select_question(board, 6, list(generated))
    assert log5.chosen == "whereabouts2"

    # A candidate generator that skips a live hypothesis's prediction is a
    # named failure at parse time, not a KeyError raised later inside scoring.
    def fake_incomplete_generate(profile, prompt, model=None):
        return json.dumps({"candidates": [
            {"id": "bad", "text": "text", "predicted_answers": {"blackwood": "x"}},
        ]})

    try:
        generate_candidates(board, profile=None, generate_fn=fake_incomplete_generate)
    except ValueError as e:
        assert "margaret" in str(e) or "ellis" in str(e)
    else:
        raise AssertionError("a candidate missing a live hypothesis's prediction should have been rejected")

    # A candidate generator that returns no candidates at all, or non-JSON,
    # is a named failure too -- never a silent empty selection.
    try:
        generate_candidates(board, profile=None, generate_fn=lambda p, t, model=None: json.dumps({"candidates": []}))
    except ValueError as e:
        assert "no candidates" in str(e)
    else:
        raise AssertionError("zero candidates from the generator should have been rejected")

    try:
        generate_candidates(board, profile=None, generate_fn=lambda p, t, model=None: "not json")
    except ValueError as e:
        assert "did not return JSON" in str(e)
    else:
        raise AssertionError("non-JSON candidate-generator output should have been rejected")

    print("ok")


if __name__ == "__main__":
    _self_check()
