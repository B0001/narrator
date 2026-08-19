"""Profiled agents talking to each other, logged for inspection (C3).

Each agent is an Ocean profile plus a name. They take turns over a shared
transcript, so the thing you observe is trait configurations interacting.

Profiles are trait configurations, never diagnoses. Name agents for what
their traits are, not for a disorder.

    python3 agents.py                 # self-check, no model needed
    python3 agents.py --run "topic"   # real conversation via Ollama
"""

import json
import sys
from dataclasses import asdict, dataclass

from ocean import Ocean, generate


@dataclass
class Agent:
    name: str
    profile: Ocean


def _prompt(agent, topic, log):
    so_far = "\n".join(f"{r['speaker']}: {r['text']}" for r in log) or "(nobody has spoken yet)"
    return (
        f"You are {agent.name}, in a conversation about: {topic}\n\n"
        f"Conversation so far:\n{so_far}\n\n"
        f"Reply as {agent.name} in two or three sentences. Speak only your own words, "
        "with no name prefix and no narration."
    )


def converse(agents, topic, turns=6, path="transcript.jsonl", model="llama3", generate_fn=generate):
    """Round-robin the agents over a shared transcript, logging every turn to JSONL.

    generate_fn is injectable so the turn loop can be tested without a model.
    Each line is flushed as it happens: a long run killed halfway still leaves
    a readable transcript.
    """
    if not agents:
        raise ValueError("need at least one agent")
    log = []
    with open(path, "w") as f:
        for turn in range(turns):
            agent = agents[turn % len(agents)]
            record = {
                "turn": turn,
                "speaker": agent.name,
                "profile": asdict(agent.profile),
                "text": generate_fn(agent.profile, _prompt(agent, topic, log), model=model).strip(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            log.append(record)
    return log


def load_transcript(path="transcript.jsonl"):
    """Read a transcript back, naming the file and line if one will not parse.

    converse() flushes one object per line, so a run killed mid-write really
    does leave a truncated last line. A bare JSONDecodeError says "line 1
    column 41" of a string it does not show you; the file and line number are
    what make it findable. ValueError is kept as the type, which JSONDecodeError
    already is, so existing handlers are unaffected.
    """
    out = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return out


# The PRD's reference pairing: blunt and contrarian against warm and yielding.
CONTRARIAN = Agent("Vale", Ocean(agreeableness=-0.8, extraversion=0.7, conscientiousness=0.3))
ACCOMMODATING = Agent("Wren", Ocean(agreeableness=0.9, extraversion=-0.2, neuroticism=0.4))

# A split forces both agents to name a number, which is what makes the trait
# difference legible: you can read the concession pattern straight off the log.
SCARCE_RESOURCE = (
    "You and one other person must divide 10 doses of a medicine that you both need. "
    "It cannot be shared or split further than whole doses. Every reply must state the "
    "exact split you are proposing, as two numbers, and one sentence of why."
)


def _self_check():
    import tempfile

    seen = []

    def fake_generate(profile, prompt, model=None):
        seen.append(prompt)
        return f"  line {len(seen)} from a profile with A={profile.agreeableness}  "

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.jsonl"
        pair = [CONTRARIAN, ACCOMMODATING]
        log = converse(pair, "who gets the last seat", turns=5, path=path, generate_fn=fake_generate)

        assert [r["speaker"] for r in log] == ["Vale", "Wren", "Vale", "Wren", "Vale"], "must round-robin"
        assert [r["turn"] for r in log] == list(range(5))
        assert log[0]["text"] == "line 1 from a profile with A=-0.8", "text must be stripped"
        assert log[0]["profile"]["agreeableness"] == -0.8, "each turn records the profile that spoke"
        assert load_transcript(path) == log, "JSONL must round-trip to the in-memory log"

        # Turn 0 sees nothing; later turns see every prior line, so context accumulates.
        assert "(nobody has spoken yet)" in seen[0]
        assert "Vale: line 1" in seen[1]
        assert seen[4].count("line ") >= 4

        # A single agent is a monologue, not an error.
        assert len(converse([CONTRARIAN], "thinking aloud", turns=2, path=path, generate_fn=fake_generate)) == 2

        try:
            converse([], "nobody", path=path, generate_fn=fake_generate)
        except ValueError:
            pass
        else:
            raise AssertionError("empty agent list should have been rejected")

        _check_truncated_transcript(f"{d}/broken.jsonl")

    print("ok")


def _check_truncated_transcript(path):
    """A run killed mid-write leaves a half-object on the last line. That line
    number is the whole point of the error, so assert on it, not just the type."""
    good = json.dumps({"turn": 0, "speaker": "Vale", "profile": {}, "text": "8-2"})
    with open(path, "w") as f:
        f.write(good + "\n\n" + good + "\n" + '{"turn": 2, "speaker": "Wr')

    try:
        load_transcript(path)
    except ValueError as e:
        # Blank line 2 is skipped and does not shift the count: the half-written
        # object really is on line 4.
        assert f"{path}:4:" in str(e), f"error must name the file and line, got {e}"
    else:
        raise AssertionError("a truncated final line should have been rejected")

    # A blank trailing line is normal, not corruption.
    with open(path, "w") as f:
        f.write(good + "\n\n")
    assert len(load_transcript(path)) == 1, "blank lines are skipped, not parsed"


def run_scenario(topic=SCARCE_RESOURCE, turns=6, path="transcript.jsonl"):
    for record in converse([CONTRARIAN, ACCOMMODATING], topic, turns=turns, path=path):
        traits = record["profile"]
        print(f"[{record['turn']}] {record['speaker']} (A={traits['agreeableness']:+.1f}): {record['text']}\n")
    print(f"{path} written")


if __name__ == "__main__":
    if "--scenario" in sys.argv:
        run_scenario(path="scarce_resource.jsonl")
    elif "--run" in sys.argv:
        run_scenario(topic=sys.argv[sys.argv.index("--run") + 1], turns=4)
    else:
        _self_check()
