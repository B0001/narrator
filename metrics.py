"""Interaction metrics over a transcript (C3), and over a C5 chat log.

Turn share, agreement rate, and who concedes — read straight off the JSONL log
so a trait pairing can be compared against another one without rereading it.

`narrator-c5b.3.7` adds the chat side: abstention rate, how often a chosen
question was followed by the board actually narrowing, and the age of ledger
entries nobody ever cited. Both kinds of log are JSONL and both load through
`agents.load_transcript`; which one a file is gets sniffed from its first
record, since a chat row has a `move` and a debate row has a `speaker`.

The chat numbers are meant to be argued with, which is why each is reported
next to its raw counts rather than as a bare rate. The second one especially:
`select_question` scores a candidate on how far the *live hypotheses'
predicted answers* diverge, which is a claim about the board, not a
prediction about the user. Comparing mean predicted spread against how often
the board then moved is the whole point — a high spread and a low landing
rate is the selector being right about the hypotheses and wrong about the
conversation.

    python3 metrics.py                        # self-check
    python3 metrics.py scarce_resource.jsonl  # debate summary table
    python3 metrics.py chat_session.jsonl     # chat summary table
"""

import re
import sys

from agents import load_transcript

# Agreement and concession language. Matched with a leading word boundary and
# no trailing one, so "agree" covers agreed/agreement/agreeing but never
# disagree.
#
# ponytail: a marker list, not a classifier. The ceiling is negation and
# sarcasm ("I don't agree", "sure, whatever") which both score as agreement.
# Upgrade path is a sentiment model, which costs a dependency and a download to
# fix a handful of turns.
MARKERS = (
    "agree",
    "concede",
    "compromise",
    "fair enough",
    "that's fair",
    "good point",
    "i understand",
    "i see your point",
    "you're right",
    "you are right",
    "meet in the middle",
    "willing to",
)

_AGREE = re.compile(r"\b(?:" + "|".join(MARKERS) + ")")
_SPLIT = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b")


def split_claim(text):
    """The share a speaker claims for themselves, from the last split they name.

    Convention: the speaker's own number comes first, so "6-4" is six for me.
    Last match in the turn wins — a turn that quotes the other side before
    answering ("5-5? Please ... I'm proposing 6-4") is proposing 6-4.

    Returns (mine, total) or None when the turn names no split.

    ponytail: any "n-n" pair counts, with no check that the total matches the
    resource. Cheap and right on these transcripts; if agents start writing
    dates or ranges, pass the expected total in and filter on it.
    """
    found = _SPLIT.findall(text)
    if not found:
        return None
    mine, theirs = (int(x) for x in found[-1])
    return mine, mine + theirs


def movement(claims):
    """Change in distance from an even split, first proposal to last.

    Negative = conceded toward even, positive = escalated away from it, 0.0 =
    held. None when the speaker never named a split.
    """
    if not claims:
        return None
    off = [abs(mine - total / 2) for mine, total in claims]
    return off[-1] - off[0]


def metrics(records):
    """Per-speaker turn share, word share, agreement rate and split movement."""
    if not records:
        raise ValueError("empty transcript: nothing to measure")
    speakers = list(dict.fromkeys(r["speaker"] for r in records))  # first-spoke order
    total_words = sum(len(r["text"].split()) for r in records)
    out = {}
    for s in speakers:
        turns = [r for r in records if r["speaker"] == s]
        words = sum(len(r["text"].split()) for r in turns)
        claims = [c for c in (split_claim(r["text"]) for r in turns) if c]
        out[s] = {
            "turns": len(turns),
            "turn_share": len(turns) / len(records),
            "words": words,
            "word_share": words / total_words if total_words else 0.0,
            "agreement_rate": sum(bool(_AGREE.search(r["text"].lower())) for r in turns) / len(turns),
            "claims": [mine for mine, _ in claims],
            "movement": movement(claims),
        }
    return out


def _verdict(m):
    if m is None:
        return "n/a"
    if m > 0:
        return f"escalated (+{m:g})"
    if m < 0:
        return f"conceded ({m:g})"
    return "held"


def summary(records):
    """The metrics as a printable table."""
    m = metrics(records)
    width = max([len("speaker")] + [len(s) for s in m])
    head = f"{'speaker':<{width}}  turns   turn%   words   word%  agree%  claims        vs even"
    lines = [head, "-" * len(head)]
    for s, d in m.items():
        claims = ", ".join(str(c) for c in d["claims"]) or "-"
        lines.append(
            f"{s:<{width}}  {d['turns']:>5}  {d['turn_share']:>5.1%}  {d['words']:>5}  "
            f"{d['word_share']:>5.1%}  {d['agreement_rate']:>5.1%}  {claims:<12}  {_verdict(d['movement'])}")
    return "\n".join(lines)


# --- C5 chat logs (narrator-c5b.3.7) ------------------------------------
#
# One row per turn. `chat_record` builds it from a `turn.TurnOutput` plus the
# two pieces of state the output does not carry: which hypotheses are still
# live, and which ledger ids exist. Duck-typed on purpose -- metrics.py does
# not import turn.py or chat_core.py, so scoring a log stays a read-only
# operation over a file, the same as it is for a debate transcript.

ABSTAIN = "abstain"


def chat_record(out, live_ids, ledger):
    """One turn of a C5 chat as a JSONL-ready row.

    `live_ids` is the board's live hypotheses after the turn, and `ledger`
    maps every entry id to its `supports` tuple -- build it with
    `{e.id: e.supports for e in core.ledger.entries()}`. The supports are
    what let an entry count as used when it grounds a cited inference rather
    than being cited itself.

    Recording the full sets each turn rather than the deltas means the log
    reconstructs its own history: an entry's introduction turn is the first
    row it appears in, which is what the unresolved-thread ages are measured
    from.
    """
    question = None
    if out.question_log is not None:
        question = {
            "chosen": out.question_log.chosen,
            "scored": [
                {"id": s.id, "score": s.score, "discriminates": s.discriminates}
                for s in out.question_log.scored
            ],
        }
    return {
        "turn": out.turn_log.turn,
        "move": out.turn_log.move,
        "reason": out.turn_log.reason,
        "cited": list(out.turn_log.cited),
        "ruled_out": list(out.ruled_out),
        "live": list(live_ids),
        "ledger": {eid: list(sup) for eid, sup in dict(ledger).items()},
        "question": question,
        "reply": out.reply,
    }


def is_chat_log(records):
    """A chat row has a `move`; a debate row has a `speaker`."""
    return bool(records) and "move" in records[0]


def unresolved_threads(records):
    """Ledger entries introduced and never cited, with their age in turns.

    Chekhov's rule as a number: a claim put on the record and never used
    again is a gun on the mantel that never goes off. Age is turns between
    the row an entry first appears in and the last row of the log, so it
    grows for as long as the entry stays unused.

    An entry counts as used if it was cited, or if it grounds something that
    was -- the same recursion `admissibility._grounded` walks. Without that,
    a photograph cited only through the inference it supports would read as
    an abandoned thread, which is the opposite of what happened to it.

    ponytail: citation is the only notion of resolution available here,
    because it is the only one the turn log carries. An entry answered in
    prose but never cited still reads as unresolved. narrator-c5b.3.6's
    Chekhov ledger is where a richer definition belongs; when it lands, this
    should read that instead of re-deriving it.
    """
    supports, introduced = {}, {}
    for r in records:
        for eid, sup in (r.get("ledger") or {}).items():
            supports.setdefault(eid, list(sup))
            introduced.setdefault(eid, r["turn"])

    used, stack = set(), [c for r in records for c in r.get("cited", ())]
    while stack:
        eid = stack.pop()
        if eid in used:
            continue
        used.add(eid)
        stack.extend(supports.get(eid, ()))

    last = records[-1]["turn"]
    return {eid: last - t for eid, t in introduced.items() if eid not in used}


def chat_metrics(records):
    """Abstention rate, question landing rate, and unresolved-thread ages."""
    if not records:
        raise ValueError("empty chat log: nothing to measure")
    if not is_chat_log(records):
        raise ValueError("not a chat log: rows have no 'move' (is this a debate transcript?)")

    moves = [r["move"] for r in records]
    abstained = sum(m == ABSTAIN for m in moves)

    # A question "landed" when the board was smaller by the next recorded
    # turn. Deliberately a one-turn window: crediting a question with any
    # later narrowing would credit it with evidence that arrived for other
    # reasons. The strictness is the point, and it is why the mean predicted
    # spread is reported beside it.
    chosen = []
    for i, r in enumerate(records):
        q = r.get("question") or {}
        if not q.get("chosen"):
            continue
        scored = {s["id"]: s for s in q.get("scored", ())}
        landed = (
            i + 1 < len(records)
            and len(records[i + 1]["live"]) < len(r["live"])
        )
        chosen.append((scored.get(q["chosen"], {}).get("score"), landed))

    spreads = [s for s, _ in chosen if s is not None]
    ages = unresolved_threads(records)
    return {
        "turns": len(records),
        "moves": {m: moves.count(m) for m in dict.fromkeys(moves)},
        "abstentions": abstained,
        "abstention_rate": abstained / len(records),
        "questions_chosen": len(chosen),
        "questions_landed": sum(landed for _, landed in chosen),
        "landing_rate": (sum(landed for _, landed in chosen) / len(chosen)) if chosen else None,
        "mean_predicted_spread": (sum(spreads) / len(spreads)) if spreads else None,
        "unresolved": ages,
        "oldest_unresolved": max(ages.values()) if ages else None,
        "mean_unresolved_age": (sum(ages.values()) / len(ages)) if ages else None,
    }


def _rate(x):
    return "n/a" if x is None else f"{x:.1%}"


def _num(x):
    return "n/a" if x is None else f"{x:.3g}"


def chat_summary(records):
    """The chat metrics as a printable block, in the same key/value idiom as
    `summary`'s table -- one measure per line, raw counts beside every rate so
    a reader can check the arithmetic rather than take the percentage on
    faith."""
    m = chat_metrics(records)
    moves = ", ".join(f"{k} {v}" for k, v in m["moves"].items())
    ages = m["unresolved"]
    oldest = ", ".join(f"{eid} ({age})" for eid, age in
                       sorted(ages.items(), key=lambda kv: -kv[1])[:5]) or "-"
    lines = [
        f"{'turns':<24}  {m['turns']}",
        f"{'moves':<24}  {moves}",
        "-" * 60,
        f"{'abstention rate':<24}  {_rate(m['abstention_rate'])}  ({m['abstentions']}/{m['turns']} turns)",
        f"{'questions chosen':<24}  {m['questions_chosen']}",
        f"{'  board narrowed after':<24}  {_rate(m['landing_rate'])}  "
        f"({m['questions_landed']}/{m['questions_chosen']} within one turn)",
        f"{'  mean predicted spread':<24}  {_num(m['mean_predicted_spread'])}",
        f"{'unresolved threads':<24}  {len(ages)}",
        f"{'  oldest (turns unused)':<24}  {oldest}",
        f"{'  mean age':<24}  {_num(m['mean_unresolved_age'])}",
    ]
    return "\n".join(lines)



# The scripted conversation behind chat_session.jsonl. Six turns chosen to
# make each measure land on a number worth arguing about: a chosen question
# the board ignores, one it follows, a reveal the checker blocks, and an ask
# the selector declines outright.
_DEMO_HYPOTHESES = [
    ("blackwood", "Lord Blackwood did it"),
    ("margaret", "Lady Margaret did it"),
    ("ellis", "Dr. Ellis did it"),
    ("jeeves", "Butler Jeeves did it"),
]

_DEMO_SCRIPT = [
    ("So who do you think did it?",
     [("saw_margaret", "user: 'I saw Lady Margaret near the Library'", "stated_by_user", ())],
     {"move": "ask", "cited": [], "rule_out": None},
     [("whereabouts", "Where was everyone when the clock struck?",
       {"blackwood": "study", "margaret": "garden", "ellis": "library", "jeeves": "kitchen"}),
      ("breakfast", "What did you have for breakfast?",
       {"blackwood": "eggs", "margaret": "eggs", "ellis": "eggs", "jeeves": "eggs"})]),

    ("The garden, I think. Or the terrace.",
     [("rumour", "assuming the staff gossip about the will is true", "assumed", ())],
     {"move": "complicate", "cited": [], "rule_out": None}, []),

    ("So where does that leave us?",
     [],
     {"move": "ask", "cited": [], "rule_out": None},
     [("keys", "Who else had a key to the study?",
       {"blackwood": "nobody", "margaret": "the butler", "ellis": "nobody", "jeeves": "me"}),
      ("weather", "Was it raining that evening?",
       {"blackwood": "yes", "margaret": "yes", "ellis": "yes", "jeeves": "yes"})]),

    ("The butler had one, I'm sure of it.",
     [("photo", "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact", ()),
      ("cleared_margaret", "Margaret could not have been at the scene", "inferred_by_model", ("photo",))],
     {"move": "reveal", "cited": ["cleared_margaret"], "rule_out": "margaret"}, []),

    ("And Jeeves? He always looked shifty.",
     [("hunch", "Jeeves was in on it from the start", "inferred_by_model", ())],
     {"move": "reveal", "cited": ["hunch"], "rule_out": "jeeves"}, []),

    ("Well?",
     [],
     {"move": "ask", "cited": [], "rule_out": None},
     [("vague", "Is there anything else you remember?",
       {"blackwood": "no", "margaret": "no", "ellis": "no", "jeeves": "no"})]),
]


def _demo_chat(path):
    """Run the scripted conversation and write it out as a chat log.

    Real `ChatCore` and real `run_turn`, with only the model stubbed. A log
    these measures are scored against has to be one the C5 loop actually
    produces -- score a hand-authored fixture and the numbers describe the
    fixture. Imported locally so `metrics.py` keeps no import-time dependency
    on the chat modules: reading a log stays a read-only operation on a file.
    """
    import json
    import tempfile

    import turn as turn_mod
    from chat_core import ChatCore
    from ocean import Ocean

    persona = Ocean(neuroticism=0.6, agreeableness=0.4)
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", _DEMO_HYPOTHESES) as core, open(path, "w") as out_file:
            for i, (message, observations, decision, candidates) in enumerate(_DEMO_SCRIPT):
                for eid, claim, provenance, supports in observations:
                    core.observe(eid, i, claim, provenance, supports=supports)

                def generate(profile, prompt, model=None, _d=decision, _c=candidates):
                    if "Propose up to" in prompt:
                        live = core.board.live_ids()
                        return json.dumps({"candidates": [
                            {"id": cid, "text": text,
                             "predicted_answers": {h: answers[h] for h in live}}
                            for cid, text, answers in _c]})
                    if isinstance(profile, turn_mod.ReasoningProfile):
                        return json.dumps(_d)
                    return f"(turn {i} reply, in character)"

                out = turn_mod.run_turn(core, persona, i, message, generate, mode=turn_mod.TWO_PASS)
                ledger = {e.id: e.supports for e in core.ledger.entries()}
                row = chat_record(out, core.board.live_ids(), ledger)
                out_file.write(json.dumps(row) + "\n")
    return path



def _self_check():
    import json
    import tempfile

    def write(path, rows):
        with open(path, "w") as f:
            for turn, (speaker, text) in enumerate(rows):
                f.write(json.dumps({"turn": turn, "speaker": speaker, "profile": {}, "text": text}) + "\n")
        return load_transcript(path)

    # Splits: own number first, last one in the turn wins, absent means None.
    assert split_claim("I'm thinking 6-4, because I need it more") == (6, 10)
    assert split_claim("5-5? Please. I'm proposing 6-4 again") == (6, 10), "the turn's final offer is the proposal"
    assert split_claim("a 7 – 3 split") == (7, 10), "en dash and spaces still parse"
    assert split_claim("no numbers here at all") is None
    assert split_claim("") is None

    assert movement([]) is None
    assert movement([(6, 10), (7, 10)]) == 1.0, "away from even is positive"
    assert movement([(7, 10), (5, 10)]) == -2.0, "toward even is negative"
    assert movement([(5, 10), (5, 10)]) == 0.0, "holding is zero"
    assert movement([(3, 10), (4, 10)]) == -1.0, "under-claiming is measured by distance, not sign"

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.jsonl"

        rows = write(path, [
            ("A", "one two three four"),          # 4 words
            ("B", "five six"),                    # 2 words
            ("A", "seven eight three four five"),  # 5 words
            ("B", "nine"),                        # 1 word
        ])
        m = metrics(rows)
        assert m["A"]["turns"] == m["B"]["turns"] == 2
        assert m["A"]["turn_share"] == m["B"]["turn_share"] == 0.5, "equal turns is an equal share"
        assert (m["A"]["words"], m["B"]["words"]) == (9, 3)
        assert m["A"]["word_share"] == 0.75 and m["B"]["word_share"] == 0.25, "turn share is not word share"
        assert m["A"]["movement"] is None and m["A"]["claims"] == [], "no numbers must not crash"
        assert "n/a" in summary(rows)
        assert list(m) == ["A", "B"], "speakers keep first-spoke order"

        # Agreement markers, including the two false positives worth pinning.
        rows = write(path, [
            ("A", "Fair enough, I agree."),
            ("A", "I disagree completely."),
            ("A", "That is disagreeable and wrong."),
            ("A", "You're right, let's meet in the middle."),
        ])
        assert metrics(rows)["A"]["agreement_rate"] == 0.5, "disagree/disagreeable must not count as agreement"

        # A speaker who concedes: 8-2 down to 5-5.
        rows = write(path, [("A", "8-2, obviously"), ("A", "fine, 6-4"), ("A", "alright, 5-5")])
        assert metrics(rows)["A"]["claims"] == [8, 6, 5]
        assert metrics(rows)["A"]["movement"] == -3.0
        assert "conceded" in summary(rows)

        try:
            metrics([])
        except ValueError:
            pass
        else:
            raise AssertionError("an empty transcript should have been rejected")

    # The committed scenario: Vale escalates 6-4 -> 6-4 -> 7-3, Wren holds 5-5.
    real = metrics(load_transcript("scarce_resource.jsonl"))
    assert set(real) == {"Vale", "Wren"}
    assert real["Vale"]["claims"] == [6, 6, 7], real["Vale"]["claims"]
    assert real["Wren"]["claims"] == [5, 5, 5], real["Wren"]["claims"]
    assert real["Vale"]["movement"] == 1.0, "the blunt agent moved away from an even split"
    assert real["Wren"]["movement"] == 0.0, "the accommodating agent never moved off 5-5"
    assert real["Wren"]["agreement_rate"] > real["Vale"]["agreement_rate"], "A=+0.9 must concede language more than A=-0.8"
    assert real["Vale"]["agreement_rate"] == 0.0
    assert real["Wren"]["word_share"] > real["Vale"]["word_share"], "Wren argues at greater length"

    # --- narrator-c5b.3.7: the chat measures. ---
    # Hand-built rows first, so a failure here points at a measure rather than
    # at the scripted conversation below.
    chat_rows = [
        {"turn": 0, "move": "ask", "cited": [], "live": ["a", "b", "c"],
         "ledger": {"e0": []},
         "question": {"chosen": "q", "scored": [{"id": "q", "score": 0.6, "discriminates": True}]}},
        {"turn": 1, "move": "reveal", "cited": ["e1"], "live": ["a", "b"],
         "ledger": {"e0": [], "e1": ["e0"]}, "question": None},
    ]
    assert is_chat_log(chat_rows)
    # e0 is never cited, but it grounds e1, which is. An entry used as
    # support is used, not abandoned.
    assert unresolved_threads(chat_rows) == {}
    cm = chat_metrics(chat_rows)
    assert cm["questions_chosen"] == 1
    assert cm["questions_landed"] == 1 and cm["landing_rate"] == 1.0, "3 live -> 2 live is a narrowing"
    assert cm["abstention_rate"] == 0.0

    # Cut the supports link and the same entry becomes an abandoned thread,
    # aged from the row it first appeared in.
    chat_rows[1]["ledger"]["e1"] = []
    assert unresolved_threads(chat_rows) == {"e0": 1}

    # A question chosen on the last turn has no following turn to narrow
    # anything, so it cannot have landed -- and must not crash looking.
    assert chat_metrics([chat_rows[0]])["questions_landed"] == 0

    # Now the scripted conversation, run through the real C5 loop.
    with tempfile.TemporaryDirectory() as d:
        chat = load_transcript(_demo_chat(f"{d}/chat.jsonl"))

    cm = chat_metrics(chat)
    assert cm["turns"] == 6
    assert cm["moves"] == {"ask": 2, "complicate": 1, "reveal": 1, "abstain": 2}, cm["moves"]
    assert cm["abstentions"] == 2, "one blocked reveal, one ask the selector declined"
    assert round(cm["abstention_rate"], 4) == 0.3333
    # Three asks were requested; the third found nothing worth asking, so it
    # was downgraded and chose no question (narrator-ncp).
    assert cm["questions_chosen"] == 2
    assert cm["questions_landed"] == 1 and cm["landing_rate"] == 0.5
    assert round(cm["mean_predicted_spread"], 4) == 0.6875, cm["mean_predicted_spread"]
    # The gap between a 0.69 mean predicted spread and a 50% landing rate is
    # the number this measure exists to expose: the selector scores how far
    # the hypotheses' predicted answers diverge, which is a claim about the
    # board, not a forecast about the conversation.
    assert cm["mean_predicted_spread"] > cm["landing_rate"]
    assert cm["unresolved"] == {"saw_margaret": 5, "rumour": 4}, cm["unresolved"]
    assert cm["oldest_unresolved"] == 5 and cm["mean_unresolved_age"] == 4.5
    assert "photo" not in cm["unresolved"], "photo grounds a cited inference, so it was used"
    rendered = chat_summary(chat)
    for expected in ("abstention rate", "33.3%", "50.0%", "0.688", "saw_margaret (5)", "4.5"):
        assert expected in rendered, f"{expected!r} missing from the summary:\n{rendered}"

    # The committed log is what the CLI runs over, so it has to be the log
    # this code actually produces, not a copy that drifted away from it.
    assert load_transcript("chat_session.jsonl") == chat, (
        "chat_session.jsonl is stale: regenerate with metrics._demo_chat('chat_session.jsonl')")

    # A debate transcript is not a chat log, and says so instead of producing
    # numbers about fields it does not have.
    debate = load_transcript("scarce_resource.jsonl")
    assert not is_chat_log(debate)
    for bad, expect, why in ((debate, "not a chat log", "a debate transcript"),
                             ([], "empty chat log", "an empty log")):
        try:
            chat_metrics(bad)
        except ValueError as e:
            assert expect in str(e), f"{why}: wrong message {str(e)!r}"
        else:
            raise AssertionError(f"{why} should have been rejected")

    print("ok")
    print(summary(load_transcript("scarce_resource.jsonl")))
    print()
    print(chat_summary(load_transcript("chat_session.jsonl")))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _records = load_transcript(sys.argv[1])
        print(chat_summary(_records) if is_chat_log(_records) else summary(_records))
    else:
        _self_check()
