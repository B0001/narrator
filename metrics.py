"""Interaction metrics over an agents.py transcript (C3).

Turn share, agreement rate, and who concedes — read straight off the JSONL log
so a trait pairing can be compared against another one without rereading it.

    python3 metrics.py                        # self-check
    python3 metrics.py scarce_resource.jsonl  # summary table
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

    print("ok")
    print(summary(load_transcript("scarce_resource.jsonl")))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(summary(load_transcript(sys.argv[1])))
    else:
        _self_check()
