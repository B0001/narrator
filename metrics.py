"""Interaction metrics over a transcript (C3), and over a C5 chat log.

Turn share, agreement rate, and who concedes — read straight off the JSONL log
so a trait pairing can be compared against another one without rereading it.

`narrator-c5b.3.7` adds the chat side: abstention rate, what the question
selector actually produced and whether the user answered it, and the age of
ledger entries nobody ever used. The repo writes three JSONL formats and
`log_kind` names which one it was handed -- a chat row has a `move`, a debate
row a `speaker`, a ledger row a `provenance` -- so pointing the CLI at the
ledger sitting next to a chat log says so instead of failing on a missing
field.

The chat numbers are meant to be argued with, which is why each is reported
next to its raw counts rather than as a bare rate.

One measure is deliberately absent. The bead asked how often a chosen
question "actually split the board", and that is not recoverable from this
log: the board narrows only via `rule_out`, which `chat_core.conclude` calls
only on a reveal that passed admissibility, and nothing links a reveal's
citations back to a question asked earlier. A first version of this module
scored "the live set was smaller by the next turn" and called that the
question landing -- it was measuring whether the following turn happened to
be a successful reveal, on evidence that could predate the question
entirely, and it was capped at (live-1) landings per conversation no matter
how good the questions were. What is honestly measurable is how often the
selector found a question worth asking at all, and whether new
`stated_by_user` evidence arrived on the turn after it did. Closing the gap
needs the log to record which evidence answered which question --
`narrator-c5b.3.6`'s Chekhov ledger territory, filed as its own bead.

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
    maps every entry id to its entry -- build it with
    `{e.id: e for e in core.ledger.entries()}`. Both fields are recorded and
    both are load-bearing: `supports` lets an entry count as used when it
    grounds a cited inference rather than being cited itself, and
    `provenance` is how new `stated_by_user` evidence on the next turn is
    recognised as the user answering the question this turn asked.

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
        "ledger": {
            eid: {"supports": list(e.supports), "provenance": e.provenance}
            for eid, e in dict(ledger).items()
        },
        "question": question,
        "reply": out.reply,
    }


# The three JSONL formats this repo writes, in the order they are tested for.
# Positive identification of each, rather than "chat, else debate": a
# ChatCore run leaves its evidence ledger in the same directory as its chat
# log, so the ledger is the likeliest wrong file to be handed, and falling
# through to the debate path made that a bare KeyError: 'speaker'.
_LOG_KINDS = (("move", "chat"), ("speaker", "debate"), ("provenance", "ledger"))


def log_kind(records):
    """Which format this is -- "chat", "debate", "ledger", or None."""
    if not records:
        return None
    return next((kind for field, kind in _LOG_KINDS if field in records[0]), None)


def is_chat_log(records):
    return log_kind(records) == "chat"


def unresolved_threads(records):
    """Ledger entries introduced and never used, with their age in turns.

    Chekhov's rule as a number: a claim put on the record and never used
    again is a gun on the mantel that never goes off. Age is the highest turn
    in the log minus the turn the entry first appears on, so it grows for as
    long as the entry stays unused.

    An entry counts as used if it was cited on a turn that concluded
    something, or if it grounds something that was -- the same recursion
    `admissibility._grounded` walks. The transitive half matters: without it,
    a photograph cited only through the inference it supports would read as
    an abandoned thread, the opposite of what happened to it.

    Citations on an *abstained* turn are not uses. `moves.choose_move` keeps
    the refused ids on the TurnLog when admissibility downgrades a reveal, so
    counting them would mark an inference the checker **rejected** as
    resolved -- and since the mark is transitive, everything supporting it
    too. An ungrounded claim the checker refused and nobody ever grounded is
    the purest case of the gun left on the mantel, so it is the one case this
    must not delete. A directly requested abstain concluded nothing either,
    and is treated the same way.

    ponytail: citation is the only notion of use available here, because it
    is the only one the turn log carries. An entry answered in prose but
    never cited still reads as unresolved. narrator-c5b.3.6's Chekhov ledger
    is where a richer definition belongs; when it lands, this should read
    that instead of re-deriving it.
    """
    if not records:
        raise ValueError("empty chat log: nothing to measure")

    supports, introduced = {}, {}
    for r in records:
        for eid, meta in (r.get("ledger") or {}).items():
            supports.setdefault(eid, list(meta.get("supports", ())))
            introduced.setdefault(eid, r["turn"])

    used = set()
    stack = [c for r in records if r.get("move") != ABSTAIN for c in r.get("cited", ())]
    while stack:
        eid = stack.pop()
        if eid in used:
            continue
        used.add(eid)
        stack.extend(supports.get(eid, ()))

    # max, not records[-1]: a log that is not turn-monotonic (resumed, merged,
    # re-sorted) would otherwise produce negative ages.
    last = max(r["turn"] for r in records)
    return {eid: last - t for eid, t in introduced.items() if eid not in used}


def chat_metrics(records):
    """Abstention rate, what the question selector produced and whether the
    user answered it, and unresolved-thread ages."""
    if not records:
        raise ValueError("empty chat log: nothing to measure")
    kind = log_kind(records)
    if kind != "chat":
        named = {"debate": "an agents.py debate transcript",
                 "ledger": "an evidence_ledger.py ledger"}.get(kind, "an unrecognised format")
        raise ValueError(f"not a chat log: rows have no 'move' -- this looks like {named}")

    moves = [r["move"] for r in records]
    abstained = sum(m == ABSTAIN for m in moves)

    # Two things about questions are honestly in this log. First, yield: a
    # `question` block exists on every turn the reasoning channel asked, and
    # `chosen` is None on the ones where the selector found nothing worth
    # asking, so the ratio is the selector's own hit rate. Second, whether
    # the user answered: new `stated_by_user` evidence on the following turn
    # is the observable trace of an answer reaching the record.
    #
    # Whether the answer *split the board* is not in here -- see the module
    # docstring. The board narrows only on a reveal that passed
    # admissibility, and nothing ties a reveal's citations to an earlier
    # question, so any such number would be measuring the next turn's move.
    asked = [i for i, r in enumerate(records) if r.get("question") is not None]
    chosen, answered, spreads = [], 0, []
    for i in asked:
        q = records[i]["question"]
        if not q.get("chosen"):
            continue
        chosen.append(q["chosen"])
        score = next((c["score"] for c in q.get("scored", ()) if c["id"] == q["chosen"]), None)
        if score is not None:
            spreads.append(score)
        if i + 1 < len(records):
            before = set(records[i].get("ledger") or {})
            after = (records[i + 1].get("ledger") or {}).items()
            if any(eid not in before and meta.get("provenance") == "stated_by_user"
                   for eid, meta in after):
                answered += 1

    ages = unresolved_threads(records)
    return {
        "turns": len(records),
        "moves": {m: moves.count(m) for m in dict.fromkeys(moves)},
        "abstentions": abstained,
        "abstention_rate": abstained / len(records),
        "asked": len(asked),
        "questions_chosen": len(chosen),
        "question_yield": (len(chosen) / len(asked)) if asked else None,
        "questions_answered": answered,
        "answer_rate": (answered / len(chosen)) if chosen else None,
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
        f"{'asks':<24}  {m['asked']}",
        f"{'  question chosen':<24}  {_rate(m['question_yield'])}  "
        f"({m['questions_chosen']}/{m['asked']} found one worth asking)",
        f"{'  user answered':<24}  {_rate(m['answer_rate'])}  "
        f"({m['questions_answered']}/{m['questions_chosen']} drew new user evidence)",
        f"{'  mean predicted spread':<24}  {_num(m['mean_predicted_spread'])}",
        f"{'unresolved threads':<24}  {len(ages)}",
        f"{'  oldest (turns unused)':<24}  {oldest}",
        f"{'  mean age':<24}  {_num(m['mean_unresolved_age'])}",
    ]
    return "\n".join(lines)



# The scripted conversation behind chat_session.jsonl. Six turns chosen to
# make each measure land on a number worth arguing about: a question the user
# answers with new evidence, one they answer with an assumption nobody can
# use, an ask the selector declines outright, and a reveal the checker blocks
# whose refused citation has to stay counted as an open thread.
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
     [("butler_key", "user: 'the butler had a key to the study'", "stated_by_user", ()),
      ("photo", "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact", ()),
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
                ledger = {e.id: e for e in core.ledger.entries()}
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
    # Hand-built rows first, so a failure points at a measure rather than at
    # the scripted conversation below.
    def row(turn, move, **kw):
        base = {"turn": turn, "move": move, "cited": [], "live": ["a", "b", "c"],
                "ledger": {}, "question": None}
        base.update(kw)
        return base

    def entry(supports=(), provenance="stated_by_user"):
        return {"supports": list(supports), "provenance": provenance}

    q_chosen = {"chosen": "q", "scored": [{"id": "q", "score": 0.5, "discriminates": True}]}

    assert log_kind([]) is None
    assert log_kind([{"nothing": 1}]) is None

    # An entry used only as support for a cited inference is used, not abandoned.
    photo, inference = entry(provenance="observed_artifact"), entry(("e0",), "inferred_by_model")
    chat_rows = [
        row(0, "ask", ledger={"e0": photo}, question=q_chosen),
        row(1, "reveal", cited=["e1"], live=["a", "b"], ledger={"e0": photo, "e1": inference}),
    ]
    assert log_kind(chat_rows) == "chat" and is_chat_log(chat_rows)
    assert unresolved_threads(chat_rows) == {}

    # Cut the supports link and it becomes an abandoned thread, aged from the
    # row it first appeared in.
    chat_rows[1]["ledger"]["e1"] = entry(provenance="inferred_by_model")
    assert unresolved_threads(chat_rows) == {"e0": 1}

    # A citation on an ABSTAINED turn is not a use -- admissibility refused
    # it. This is the case the measure most has to get right: an ungrounded
    # claim the checker rejected, and everything under it, is the gun left on
    # the mantel, and counting the refusal as resolution deletes exactly that.
    ledger = {"g": entry(provenance="assumed"), "h": entry(("g",), "inferred_by_model")}
    blocked = [row(0, "complicate", ledger=ledger), row(9, "abstain", cited=["h"], ledger=ledger)]
    assert unresolved_threads(blocked) == {"g": 9, "h": 9}, unresolved_threads(blocked)
    # The same citation on a reveal that landed IS a use, transitively.
    assert unresolved_threads([blocked[0], dict(blocked[1], move="reveal")]) == {}

    # Ages come from the largest turn, not the last row, so a log that is not
    # turn-monotonic cannot produce a negative age.
    assert unresolved_threads([row(9, "complicate", ledger={"x": entry()}),
                               row(0, "complicate", ledger={"x": entry()})]) == {"x": 0}
    try:
        unresolved_threads([])
    except ValueError:
        pass
    else:
        raise AssertionError("an empty log should have been rejected by unresolved_threads too")

    # The user answered iff NEW stated_by_user evidence arrives next turn.
    def answered_with(provenance):
        return chat_metrics([
            row(0, "ask", question=q_chosen, ledger={"old": entry()}),
            row(1, "complicate", ledger={"old": entry(), "new": entry(provenance=provenance)}),
        ])["questions_answered"]

    assert answered_with("stated_by_user") == 1
    assert answered_with("observed_artifact") == 0, "only the user's own evidence answers a question"
    assert answered_with("inferred_by_model") == 0, "the model answering itself is not an answer"
    # Evidence that was already on the record is not an answer.
    assert chat_metrics([row(0, "ask", question=q_chosen, ledger={"old": entry()}),
                         row(1, "complicate", ledger={"old": entry()})])["questions_answered"] == 0
    # A question chosen on the final turn has no next turn, and must not crash.
    assert chat_metrics([row(0, "ask", question=q_chosen)])["questions_answered"] == 0

    # Yield is over asks, not turns: a declined ask still carries a question
    # block, with chosen=None.
    yielded = chat_metrics([row(0, "ask", question=q_chosen),
                            row(1, "abstain", question={"chosen": None, "scored": []})])
    assert (yielded["asked"], yielded["questions_chosen"]) == (2, 1)
    assert yielded["question_yield"] == 0.5

    # Now the scripted conversation, run through the real C5 loop.
    with tempfile.TemporaryDirectory() as d:
        chat = load_transcript(_demo_chat(f"{d}/chat.jsonl"))

    cm = chat_metrics(chat)
    assert cm["turns"] == 6
    assert cm["moves"] == {"ask": 2, "complicate": 1, "reveal": 1, "abstain": 2}, cm["moves"]
    assert cm["abstentions"] == 2, "one blocked reveal, one ask the selector declined"
    assert round(cm["abstention_rate"], 4) == 0.3333
    assert (cm["asked"], cm["questions_chosen"]) == (3, 2), "the third ask found nothing worth asking"
    assert round(cm["question_yield"], 4) == 0.6667
    assert cm["questions_answered"] == 1 and cm["answer_rate"] == 0.5
    assert round(cm["mean_predicted_spread"], 4) == 0.6875, cm["mean_predicted_spread"]
    assert cm["unresolved"] == {"saw_margaret": 5, "rumour": 4, "butler_key": 2, "hunch": 1}, cm["unresolved"]
    assert cm["oldest_unresolved"] == 5 and cm["mean_unresolved_age"] == 3.0
    # photo and the inference it grounds were both used, by a reveal that landed.
    assert not {"photo", "cleared_margaret"} & set(cm["unresolved"])
    # hunch WAS cited -- by the reveal admissibility refused -- and stays open.
    assert "hunch" in cm["unresolved"], "a refused citation is the thread this measure exists to find"

    rendered = chat_summary(chat)
    for expected in ("abstention rate", "33.3%", "66.7%", "50.0%", "0.688",
                     "saw_margaret (5)", "3"):
        assert expected in rendered, f"{expected!r} missing from the summary:\n{rendered}"

    # SINGLE_PASS is the one mode _demo_chat never exercises -- it makes a
    # single call and its TurnOutput carries no question_log at all -- and a
    # record is only worth anything if the file survives the round trip the
    # CLI puts it through.
    with tempfile.TemporaryDirectory() as d:
        import turn as _turn
        from chat_core import ChatCore as _ChatCore
        from ocean import Ocean as _Ocean
        with _ChatCore(f"{d}/l.jsonl", _DEMO_HYPOTHESES) as _core:
            _core.observe("e0", 0, "user: 'the door was locked'", "stated_by_user")
            _out = _turn.run_turn(
                _core, _Ocean(neuroticism=0.5), 0, "hm",
                lambda profile, prompt, model=None: json.dumps(
                    {"move": "complicate", "cited": ["e0"], "rule_out": None,
                     "reply": "Curious. Say more about that door."}),
                mode=_turn.SINGLE_PASS)
            _rec = chat_record(_out, _core.board.live_ids(),
                               {e.id: e for e in _core.ledger.entries()})
        assert _rec["question"] is None, "a single-pass turn has no question log to record"
        assert _rec["ledger"] == {"e0": {"supports": [], "provenance": "stated_by_user"}}, _rec["ledger"]
        with open(f"{d}/one.jsonl", "w") as f:
            f.write(json.dumps(_rec) + "\n")
        assert load_transcript(f"{d}/one.jsonl") == [_rec], "a record must survive its own round trip"

    # The committed log is what the CLI runs over, so it has to be the log
    # this code actually produces, not a copy that drifted away from it.
    assert load_transcript("chat_session.jsonl") == chat, (
        "chat_session.jsonl is stale: regenerate with metrics._demo_chat('chat_session.jsonl')")

    # Each of the repo's three JSONL formats is identified positively, so the
    # ledger sitting next to a chat log is named rather than crashing the
    # debate path on a missing 'speaker'.
    debate = load_transcript("scarce_resource.jsonl")
    assert log_kind(debate) == "debate" and not is_chat_log(debate)
    assert log_kind([{"id": "e", "turn": 0, "claim": "x", "provenance": "assumed",
                      "supports": []}]) == "ledger"
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
        _kind = log_kind(_records)
        if _kind == "chat":
            print(chat_summary(_records))
        elif _kind == "debate":
            print(summary(_records))
        elif _kind == "ledger":
            sys.exit(f"{sys.argv[1]}: that is an evidence ledger, not a transcript or chat log")
        else:
            sys.exit(f"{sys.argv[1]}: not a recognised transcript, chat log or ledger")
    else:
        _self_check()
