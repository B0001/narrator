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
    # What this agent alone has been given to know, e.g. a hidden-profile clue
    # subset (narrator-cby.3.5.2). Empty by default: a plain C3 agent shares
    # nothing beyond the transcript, same as before this field existed.
    evidence: str = ""


def _prompt(agent, topic, log):
    so_far = "\n".join(f"{r['speaker']}: {r['text']}" for r in log) or "(nobody has spoken yet)"
    evidence = (
        f"\n\nWhat you have personally observed, and no one else has told you:\n{agent.evidence}"
        if agent.evidence else ""
    )
    return (
        f"You are {agent.name}, in a conversation about: {topic}\n\n"
        f"Conversation so far:\n{so_far}{evidence}\n\n"
        f"Reply as {agent.name} in two or three sentences. Speak only your own words, "
        "with no name prefix and no narration."
    )


# Every turn's prompt carries the whole transcript so far, so the tokens sent
# grow with turn count, and the tokens sent *across a run* grow quadratically
# in it -- see narrator-c5b.2.5 / prd.md's "Cost: which components are cheap
# enough for a metered backend" section. This default is a stdlib-only
# characters-per-token heuristic (see _estimate_tokens), sized to stop a run
# well before it would matter on a metered backend, not to model any real
# provider's billing.
DEFAULT_TOKEN_BUDGET = 20_000


class TokenBudgetExceeded(RuntimeError):
    """Raised by converse() when the next turn would spend past its ceiling."""


def _estimate_tokens(text):
    """~4 characters per token for English text -- a rough stand-in, not a
    real tokenizer (stdlib-first per prd.md). Good enough to catch a runaway
    quadratic transcript long before the ceiling actually matters; not good
    enough to bill against.
    """
    return max(1, len(text) // 4)


def converse(agents, topic, turns=6, path="transcript.jsonl", model="llama3", generate_fn=generate,
             token_budget=DEFAULT_TOKEN_BUDGET):
    """Round-robin the agents over a shared transcript, logging every turn to JSONL.

    generate_fn is injectable so the turn loop can be tested without a model.
    Each line is flushed as it happens: a long run killed halfway still leaves
    a readable transcript.

    token_budget is a per-run ceiling on estimated tokens sent to the model
    (prompt and reply combined). Once the next turn's prompt would push the
    running total past it, converse() stops and raises TokenBudgetExceeded
    naming both the projected count and the limit, rather than silently
    continuing to spend on a backend that bills per token. Pass token_budget=
    None to disable the ceiling (e.g. for a backend you know is free, like a
    local Ollama run with time to spare).
    """
    if not agents:
        raise ValueError("need at least one agent")
    log = []
    spent = 0
    with open(path, "w") as f:
        for turn in range(turns):
            agent = agents[turn % len(agents)]
            prompt = _prompt(agent, topic, log)
            projected = spent + _estimate_tokens(prompt)
            if token_budget is not None and projected > token_budget:
                raise TokenBudgetExceeded(
                    f"turn {turn} would bring this run to an estimated {projected} tokens, "
                    f"over the {token_budget}-token ceiling ({spent} spent across {turn} turns so far). "
                    "Raise token_budget, or use fewer turns/agents, to continue."
                )
            reply = generate_fn(agent.profile, prompt, model=model).strip()
            spent = projected + _estimate_tokens(reply)
            record = {
                "turn": turn,
                "speaker": agent.name,
                "profile": asdict(agent.profile),
                "text": reply,
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

        # An agent with evidence gets it injected into the prompt; one without
        # (the default) sees no such section, so old callers are unaffected.
        seen.clear()
        briefed = Agent("Nym", CONTRARIAN.profile, evidence="- the vault was open at t=1")
        converse([briefed, ACCOMMODATING], "the missing ledger", turns=2, path=path, generate_fn=fake_generate)
        assert "the vault was open at t=1" in seen[0], "briefed agent must see their own evidence"
        assert "personally observed" not in seen[1], "an agent with no evidence gets no evidence section"

        try:
            converse([], "nobody", path=path, generate_fn=fake_generate)
        except ValueError:
            pass
        else:
            raise AssertionError("empty agent list should have been rejected")

        _check_token_budget(path, fake_generate)
        _check_truncated_transcript(f"{d}/broken.jsonl")

    print("ok")


def _check_token_budget(path, fake_generate):
    """A run that would exceed its token_budget stops before spending past
    it, naming the projected count and the limit -- not a silent overspend
    and not a hard stop after the fact.
    """
    pair = [CONTRARIAN, ACCOMMODATING]

    # A generous budget does not get in the way of a normal run.
    log = converse(pair, "a modest budget", turns=4, path=path, generate_fn=fake_generate, token_budget=10_000)
    assert len(log) == 4, "a run comfortably inside budget must complete normally"

    # A ceiling too tight for even the first turn must stop on turn 0, before
    # any line is written, and name both the projected spend and the limit.
    try:
        converse(pair, "a tight budget", turns=50, path=path, generate_fn=fake_generate, token_budget=1)
    except TokenBudgetExceeded as e:
        assert "turn 0" in str(e), f"must name the turn it stopped at, got: {e}"
        assert "1-token ceiling" in str(e), f"must name the configured limit, got: {e}"
    else:
        raise AssertionError("a ceiling smaller than the first turn's prompt should have stopped the run")
    assert load_transcript(path) == [], "a run that never got past its first turn must write nothing"

    # token_budget=None is the explicit opt-out: unlimited, same as before this ceiling existed.
    log = converse(pair, "no ceiling", turns=6, path=path, generate_fn=fake_generate, token_budget=None)
    assert len(log) == 6, "token_budget=None must disable the ceiling entirely"


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
