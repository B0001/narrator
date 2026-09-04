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

# A question earns its turn only if enough of the board's belief actually sits
# behind a differing answer. This floor is stated in belief units -- the share
# of live weight held by every hypothesis that does *not* give the majority
# answer -- rather than as a threshold on the Gini score, because a Gini value
# is not a quantity a reader can judge and a fraction of the board is: 1e-3 is
# "a tenth of a percent". An epsilon on the score instead would be doing two
# unrelated jobs at once, guarding float noise and deciding what is worth
# asking, and would set the second by accident: for a two-way split Gini is
# ~2x the minority fraction, so a 1e-9 score floor rejects only a minority mass
# below ~5e-10 while calling every board above that worth a turn.
# ponytail: one flat repo-wide floor, no per-board calibration. If a scenario
# ever needs the selector to chase a long shot, make this a select_question()
# argument rather than a second constant here.
_MIN_MINORITY_MASS = 1e-3


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


def _as_group_key(hid, answer):
    """Predicted answers become dict keys in `_spread`, so they have to be
    hashable. A model hedging with a JSON array (`"blackwood": ["study",
    "library"]`) or an object is perfectly valid JSON and ordinary schema
    drift, so it earns a named `ValueError` here rather than a `TypeError`
    raised three frames down inside a dict update (narrator-fdp).
    """
    try:
        hash(answer)
    except TypeError as e:
        raise ValueError(
            f"predicted answer for {hid!r} cannot be compared to other answers: {answer!r}"
        ) from e
    return answer


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
        answer = _as_group_key(hid, predicted_answers[hid])
        groups[answer] = groups.get(answer, 0.0) + w
    total = sum(groups.values())
    return 1.0 - sum((mass / total) ** 2 for mass in groups.values()), groups


def _minority_mass(groups):
    """Share of live board weight held by every hypothesis that does *not*
    give the single most-believed predicted answer.

    0.0 exactly when they all agree. This is the quantity that says whether a
    question is worth a turn: not how many distinct answers exist, but how
    much belief actually rides on the difference.
    """
    total = sum(groups.values())
    return 1.0 - max(groups.values()) / total


def score_candidate(board, candidate):
    """Score one candidate question against the board's live hypotheses.

    Reads only `board`'s live ids and current weights -- the same "given a
    board" surface `HypothesisBoard.dump()` already exposes, nothing about
    how those weights got there.
    """
    live_weights = _live_weights(board)
    spread, groups = _spread(live_weights, candidate.predicted_answers)
    minority = _minority_mass(groups)

    if len(groups) == 1:
        # Uninformative regardless of who is right: nobody answers differently.
        (only_answer,) = groups
        discriminates = False
        reason = f"every live hypothesis predicts the same answer ({only_answer!r}); spread=0.0"
    elif minority <= _MIN_MINORITY_MASS:
        # A split on paper only. The hypotheses that would answer differently
        # are ones the board has already all but abandoned, so the answer moves
        # essentially no belief -- same practical outcome as a uniform question,
        # said honestly rather than by pretending there was only one answer.
        # This is also the branch that keeps the single-answer unpack above
        # from being reached with several groups (narrator-n0x).
        discriminates = False
        reason = (
            f"nominally splits live hypotheses into {len(groups)} answer groups, but only "
            f"{minority:.3e} of the board's belief sits behind a differing answer "
            f"(floor {_MIN_MINORITY_MASS:g}): every hypothesis that would answer differently "
            f"is already all but ruled out; spread={spread:.3e}"
        )
    else:
        discriminates = True
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
    ids = [c.id for c in candidates]
    if len(set(ids)) != len(ids):
        # `chosen` below is a bare id, so two candidates sharing one makes the
        # log ambiguous: a caller looking the winner back up by id can get the
        # other one, and would ask a question this function rejected while the
        # log reports the one it picked (narrator-fdp). Same uniqueness rule
        # evidence_ledger and hypothesis_board already apply to their own ids.
        raise ValueError(f"candidate ids must be unique, got {ids}")

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
        '{"candidates": [{"id": "<short id, unique across these candidates>", "text": "<question>", '
        '"predicted_answers": {"<hypothesis id>": "<predicted answer>", '
        "... one entry per live hypothesis id listed above}}]}"
    )


def parse_candidates(board, raw):
    """Parse the candidate-generation call's raw output into `Candidate`s.

    This is the module's trust boundary: the one place raw model output
    becomes `Candidate`s for every caller. Everything it declines to check
    here surfaces later as some other exception from deeper in, which is the
    failure this function exists to prevent -- so a malformed or lazy
    response is a named `ValueError` naming the generator, never a
    `KeyError`, `AttributeError` or `TypeError` from inside
    `select_question` (narrator-fdp).

    Checked, in order: the payload is JSON with a non-empty `candidates`
    list; each entry is an object with a non-blank string `id` and `text`;
    ids are unique, because `SelectionLog.chosen` is a bare id and cannot
    name one of two candidates sharing it; `predicted_answers` is an object
    covering every live hypothesis; and each predicted answer is hashable,
    since `_spread` groups by it.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"candidate generator did not return JSON: {raw!r}") from e

    raw_candidates = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"candidate generator returned no candidates: {raw!r}")

    live_ids = {h["id"] for h in board.dump()["hypotheses"] if h["live"]}
    candidates = []
    seen = set()
    for c in raw_candidates:
        if not isinstance(c, dict):
            raise ValueError(f"candidate is not an object: {c!r}")
        for field in ("id", "text"):
            value = c.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate needs a non-empty {field!r}, got {value!r}")
        cid = c["id"]
        if cid in seen:
            raise ValueError(
                f"candidate generator repeated the id {cid!r}; ids must be unique, "
                "since the selection log names its winner by id"
            )
        seen.add(cid)

        predicted = c.get("predicted_answers")
        if not isinstance(predicted, dict):
            raise ValueError(f"candidate {cid!r} needs a predicted_answers object, got {predicted!r}")
        missing = sorted(live_ids - set(predicted))
        if missing:
            raise ValueError(
                f"candidate {cid!r} missing predictions for live hypotheses: {missing}"
            )
        answers = {hid: _as_group_key(hid, predicted[hid]) for hid in live_ids}
        candidates.append(Candidate(cid, c["text"], answers))
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

    # --- narrator-fdp: this function is the module's trust boundary, so every
    # shape of drift it lets through becomes some other exception from deeper
    # in. Each of these used to escape as KeyError, AttributeError or
    # TypeError; all must be a named ValueError naming the generator. ---
    live_now = sorted(board.live_ids())  # blackwood, ellis, margaret

    def payload(**over):
        c = {"id": "q", "text": "a question?",
             "predicted_answers": {hid: "same" for hid in live_now}}
        c.update(over)
        return json.dumps({"candidates": [c]})

    def rejects(raw, expect, why):
        try:
            parse_candidates(board, raw)
        except ValueError as e:
            assert expect in str(e), f"{why}: wrong message {str(e)!r}"
        else:
            raise AssertionError(f"{why} should have been rejected")

    # A model hedging with a list or object is valid JSON and ordinary drift.
    # It used to reach _spread and die on `groups[answer]` as an unhashable
    # dict key, aborting the whole turn with no reply and no abstention.
    hedged = dict.fromkeys(live_now, "same")
    hedged[live_now[0]] = ["study", "library"]
    rejects(payload(predicted_answers=hedged), "cannot be compared", "a list-valued prediction")
    hedged[live_now[0]] = {"maybe": "study"}
    rejects(payload(predicted_answers=hedged), "cannot be compared", "an object-valued prediction")

    # Drifted or missing key names: hard-indexed c["id"] / c["text"] raised
    # KeyError, and a bare string in the list raised AttributeError on .get.
    rejects(json.dumps({"candidates": [{"question": "q", "text": "t",
                                        "predicted_answers": dict.fromkeys(live_now, "x")}]}),
            "non-empty 'id'", "a candidate with a drifted id key")
    rejects(payload(text="   "), "non-empty 'text'", "a blank question")
    rejects(payload(id=None), "non-empty 'id'", "a null id")
    rejects(json.dumps({"candidates": ["just a string"]}), "not an object", "a bare string candidate")
    rejects(payload(predicted_answers="not an object"), "predicted_answers object",
            "a non-object predicted_answers")

    # Duplicate ids: SelectionLog.chosen is a bare id, so two candidates
    # sharing one make the log unable to name its own winner. A generator
    # emitting generic q1/q2 ids is exactly how this arrives.
    dup = json.dumps({"candidates": [
        {"id": "q1", "text": "boring?", "predicted_answers": dict.fromkeys(live_now, "same")},
        {"id": "q1", "text": "splitting?", "predicted_answers": {
            hid: hid for hid in live_now}},
    ]})
    rejects(dup, "repeated the id", "a repeated candidate id")

    # ...and select_question enforces the same rule itself, because it is
    # callable with hand-built Candidates that never passed through parse --
    # its own return value is what goes ambiguous.
    twins = [Candidate("q1", "boring?", dict.fromkeys(live_now, "same")),
             Candidate("q1", "splitting?", {hid: hid for hid in live_now})]
    try:
        select_question(board, 11, twins)
    except ValueError as e:
        assert "unique" in str(e)
    else:
        raise AssertionError("select_question must refuse candidates it cannot name a winner among")

    # Same for an unhashable answer arriving by the hand-built route: a named
    # ValueError from the scorer, not a TypeError from a dict update.
    try:
        select_question(board, 12, [Candidate("q", "?", {**dict.fromkeys(live_now, "x"),
                                                         live_now[0]: ["a", "b"]})])
    except ValueError as e:
        assert "cannot be compared" in str(e)
    else:
        raise AssertionError("an unhashable predicted answer should have been rejected")

    # The prompt now states the uniqueness rule, rather than only punishing
    # its absence after the fact.
    assert "unique" in candidate_prompt(board)

    # --- narrator-n0x: a lopsided board. `reweight` refuses only w <= 0, so a
    # live hypothesis can carry arbitrarily small positive mass while two
    # genuinely distinct predicted answers exist. The scorer used to assume
    # "not discriminating" meant "exactly one answer group" and died with
    # ValueError: too many values to unpack. It must report the honest
    # outcome instead -- uninformative, because everyone who would answer
    # differently is already all but ruled out -- and still return a log. ---
    def lopsided_board(c_weight):
        b = HypothesisBoard([("a", "A did it"), ("b", "B did it"), ("c", "C did it")])
        b.reweight(1, {"a": 1.0, "b": 1.0, "c": c_weight}, "c is all but excluded")
        return b

    only_c_differs = Candidate(
        "hairline", "Did you notice the hairline crack in the vase?",
        {"a": "no", "b": "no", "c": "yes"},
    )
    log6 = select_question(lopsided_board(1e-12), 7, [only_c_differs])
    assert isinstance(log6, SelectionLog), "a lopsided board must yield a log, not a raise"
    assert log6.chosen is None, "a split nobody has weight behind cannot win"
    (scored6,) = log6.scored
    assert scored6.discriminates is False
    assert "2 answer groups" in scored6.reason and "already all but ruled out" in scored6.reason
    assert scored6.score > 0.0, "the real spread is still reported, not flattened to 0"

    # The floor has to be doing real work, not passing because the case above
    # picked an absurd constant. At c=1e-4 the minority mass is ~5e-5: still
    # nothing worth a turn, but ~5 orders of magnitude above where a
    # float-noise epsilon on the Gini score would have sat, so this case
    # abstains only because the gate is stated in belief units. If the gate
    # ever reverts to `spread > 1e-9`, this assertion is what fails.
    faint = select_question(lopsided_board(1e-4), 8, [only_c_differs])
    assert faint.chosen is None, "a 5e-5 minority is a split on paper, not one worth asking about"
    assert faint.scored[0].score > 1e-9, "...and it is not float noise that rejected it"

    # ...and the floor must not swallow a real long shot. c=0.05 relative is a
    # hypothesis the board has demoted but not abandoned; a question only it
    # answers differently is exactly Columbo's one-more-thing.
    longshot = select_question(lopsided_board(0.05), 9, [only_c_differs])
    assert longshot.chosen == "hairline", "a live minority still deserves the turn"
    assert longshot.scored[0].discriminates is True

    # The genuinely-uniform case keeps its own distinct explanation -- the two
    # rejections must not collapse into one vague "uninformative".
    (uniform6,) = select_question(
        lopsided_board(1e-12), 10, [Candidate("u", "", {"a": "x", "b": "x", "c": "x"})]
    ).scored
    assert uniform6.score == 0.0 and "every live hypothesis predicts the same answer" in uniform6.reason
    assert "answer groups" not in uniform6.reason

    print("ok")


if __name__ == "__main__":
    _self_check()
