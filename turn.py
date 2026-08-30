"""Two-channel turn: persona-neutral reasoning, persona voice (C5).

Every prior C5 module (`evidence_ledger.py`, `admissibility.py`, `moves.py`,
`chat_core.py`) is model-agnostic: `ChatCore.conclude()` takes a move that
something else already decided. This module is the something else -- it
makes the two model calls a real turn needs and keeps them apart on purpose:

  reasoning call -- decides the move (reveal / complicate / ask / abstain)
                    and which ledger entries license it. Always run against
                    `REASONING_PROFILE`: a persona-neutral system prompt
                    (`Ocean()`'s own "Respond neutrally" text) at a fixed low
                    temperature, regardless of which persona is speaking.
  voice call     -- renders the move `ChatCore.conclude()` actually returned
                    (post-admissibility-check) in character, using the real
                    persona's compiled `system_prompt()`/`options()`.

The failure this prevents already has a name in `ocean.py`: neuroticism
widens `temperature`. If that widened sampler were also driving the step
that decides *what is true* -- which citations exist, whether a chain is
grounded enough to reveal -- a jittery persona would not just sound anxious,
it would reason less reliably, and admissibility.check() would be auditing a
noisier decision every time the speaker got more neurotic. Pinning the
reasoning call to one fixed profile makes that impossible by construction:
`REASONING_PROFILE.options()` does not read the persona's traits at all.

Scope limit, stated in the bead: this splits *sampler settings* from
*reasoning*, not traits from evidence. A persona's traits still don't get a
say in which evidence is admissible, or how heavily to weight it -- that is
`narrator-c5b.3.9`'s job, deliberately not started here. Nor does this
module gate anything moves.py doesn't already gate: `ChatCore.conclude()`
runs the identical admissibility check regardless of mode, so a reveal that
should be blocked is blocked whether the decision came from the neutral
profile or, in single-pass mode, from the persona itself.

SINGLE_PASS stays alongside TWO_PASS, not as a deprecated fallback, but so a
learner can point both modes at the same neurotic persona and read the
sampler options straight off the recorded `Call`s -- that comparison *is*
the lesson, and deleting single-pass would delete the ability to see it.

    python3 turn.py   # self-check
"""

import json
from dataclasses import dataclass

from ocean import Ocean

TWO_PASS = "two_pass"
SINGLE_PASS = "single_pass"
MODES = frozenset({TWO_PASS, SINGLE_PASS})

REASONING_TEMPERATURE = 0.2


class ReasoningProfile(Ocean):
    """Stand-in persona for the reasoning channel.

    Same interface `ocean.generate()` expects (`system_prompt()`,
    `options()`), so a real backend needs no special-casing -- but every
    field is left at the `Ocean()` default, so `system_prompt()` renders the
    persona-neutral "Respond neutrally" text no matter what, and `options()`
    is overridden outright rather than computed from traits, so it cannot
    drift even if someone constructs this with non-zero fields later.
    """

    def options(self):
        return {"temperature": REASONING_TEMPERATURE, "top_p": 0.9, "repeat_penalty": 1.1}


REASONING_PROFILE = ReasoningProfile()


@dataclass
class Call:
    """One model invocation, recorded so the acceptance criteria -- sampler
    settings from a profile reach the voice call only -- is something a
    self-check can assert on, not just a claim in a docstring."""

    label: str  # "reasoning", "voice", or "reasoning+voice" (single-pass)
    profile: Ocean
    options: dict
    prompt: str
    raw: str


@dataclass
class TurnOutput:
    mode: str
    calls: tuple  # Call, in call order
    turn_log: "moves.TurnLog"  # noqa: F821 -- forward ref, moves imported lazily below
    ruled_out: tuple
    reply: str


def _ledger_summary(ledger):
    return "\n".join(f"- {e.id} [{e.provenance}]: {e.claim}" for e in ledger.entries()) or "(empty)"


def _decision_instructions(want_reply):
    fields = '"move": "reveal|complicate|ask|abstain", "cited": ["<ledger id>", ...], "rule_out": "<hypothesis id or null>"'
    if want_reply:
        fields += ', "reply": "<what to say, in character>"'
    return (
        "Decide the move for this turn and cite exactly the ledger entries you "
        f"are relying on. Return JSON only, no other text: {{{fields}}}"
    )


def _reasoning_prompt(ledger, turn, user_message, live_ids):
    return (
        "You are the reasoning channel of a fair-play mystery chatbot.\n"
        f"{_decision_instructions(want_reply=False)}\n\n"
        f"Ledger so far:\n{_ledger_summary(ledger)}\n\n"
        f"Live hypotheses: {', '.join(live_ids)}\n"
        f"Turn {turn}. User just said: {user_message!r}\n"
    )


def _combined_prompt(ledger, turn, user_message, live_ids):
    return (
        "Decide this turn's move and write the reply in character, in one pass.\n"
        f"{_decision_instructions(want_reply=True)}\n\n"
        f"Ledger so far:\n{_ledger_summary(ledger)}\n\n"
        f"Live hypotheses: {', '.join(live_ids)}\n"
        f"Turn {turn}. User just said: {user_message!r}\n"
    )


def _voice_prompt(turn_log, user_message):
    return (
        f"The reasoning channel chose to {turn_log.move} this turn, because "
        f"{turn_log.reason}. Write the reply in character, in your own voice, "
        f"responding to: {user_message!r}. Do not mention the reasoning "
        "channel, the ledger, or admissibility -- just speak."
    )


def _parse_decision(raw, require_reply):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"reasoning channel did not return JSON: {raw!r}") from e
    if not isinstance(data, dict) or "move" not in data:
        raise ValueError(f"reasoning channel response missing 'move': {data!r}")
    if require_reply and not data.get("reply"):
        raise ValueError(f"single-pass response missing 'reply': {data!r}")
    data.setdefault("cited", [])
    data.setdefault("rule_out", None)
    return data


def run_turn(core, persona, turn, user_message, generate_fn, model="qwen2.5-coder:14b", mode=TWO_PASS):
    """Run one turn of the fair-play chat, in either mode.

    `generate_fn` follows `ocean.generate`'s shape (`profile, prompt,
    model=...`), so the real backend and this self-check's stub are
    interchangeable -- see `agents.converse`'s `generate_fn` for the same
    pattern. `core.conclude()` (and therefore `admissibility.check()`) runs
    unconditionally in both modes; only which profile produced the decision
    differs.
    """
    if mode not in MODES:
        raise ValueError(f"not a mode: {mode!r}; must be one of {sorted(MODES)}")

    live_ids = core.board.live_ids()
    calls = []

    if mode == SINGLE_PASS:
        prompt = _combined_prompt(core.ledger, turn, user_message, live_ids)
        raw = generate_fn(persona, prompt, model=model)
        calls.append(Call("reasoning+voice", persona, persona.options(), prompt, raw))
        decision = _parse_decision(raw, require_reply=True)
        result = core.conclude(turn, decision["cited"], decision["move"], rule_out=decision["rule_out"])
        reply = decision["reply"].strip()
        return TurnOutput(mode, tuple(calls), result.turn_log, result.ruled_out, reply)

    reasoning_prompt = _reasoning_prompt(core.ledger, turn, user_message, live_ids)
    raw = generate_fn(REASONING_PROFILE, reasoning_prompt, model=model)
    calls.append(Call("reasoning", REASONING_PROFILE, REASONING_PROFILE.options(), reasoning_prompt, raw))
    decision = _parse_decision(raw, require_reply=False)
    result = core.conclude(turn, decision["cited"], decision["move"], rule_out=decision["rule_out"])

    voice_prompt = _voice_prompt(result.turn_log, user_message)
    reply_raw = generate_fn(persona, voice_prompt, model=model)
    calls.append(Call("voice", persona, persona.options(), voice_prompt, reply_raw))

    return TurnOutput(mode, tuple(calls), result.turn_log, result.ruled_out, reply_raw.strip())


def _self_check():
    import tempfile

    import moves
    from chat_core import ChatCore

    hypotheses = [
        ("blackwood", "Lord Blackwood did it"),
        ("margaret", "Lady Margaret did it"),
        ("ellis", "Dr. Ellis did it"),
        ("jeeves", "Butler Jeeves did it"),
    ]

    neurotic = Ocean(neuroticism=0.9, conscientiousness=-0.6)
    disciplined = Ocean(conscientiousness=0.9)
    assert neurotic.options()["temperature"] != REASONING_TEMPERATURE
    assert disciplined.options()["temperature"] != REASONING_TEMPERATURE

    # --- two_pass: the reasoning call's options never move with the persona. ---
    with tempfile.TemporaryDirectory() as d:
        seen_reasoning_options = []

        def scripted_two_pass(decisions):
            remaining = list(decisions)

            def fake_generate(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    seen_reasoning_options.append(profile.options())
                    return json.dumps(remaining.pop(0))
                return f"(reply at temperature={profile.options()['temperature']})"

            return fake_generate

        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            core.observe("saw_margaret", 0, "user: 'I saw Lady Margaret near the Library'", "stated_by_user")
            core.observe("weak_inference", 0, "Margaret must be the culprit", "inferred_by_model")

            # Turn 0: an ungrounded reveal request must still be downgraded to
            # abstain -- admissibility gating is unconditional, mode or no mode.
            out = run_turn(
                core, neurotic, 0, "so was it Margaret?",
                scripted_two_pass([{"move": "reveal", "cited": ["weak_inference"], "rule_out": "margaret"}]),
                mode=TWO_PASS,
            )
            assert out.turn_log.move == moves.ABSTAIN, "ungrounded reveal must be downgraded regardless of mode"
            assert out.ruled_out == ()
            assert core.board.live_ids() == ["blackwood", "margaret", "ellis", "jeeves"]
            assert len(out.calls) == 2 and out.calls[0].label == "reasoning" and out.calls[1].label == "voice"

            # Turn 1: ground the inference, then reveal through the disciplined persona.
            core.observe("photo", 1, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            core.observe(
                "strong_inference", 1, "Margaret could not have been at the scene", "inferred_by_model",
                supports=("photo",),
            )
            out = run_turn(
                core, disciplined, 1, "come on, who was it?",
                scripted_two_pass([{"move": "reveal", "cited": ["strong_inference"], "rule_out": "margaret"}]),
                mode=TWO_PASS,
            )
            assert out.turn_log.move == moves.REVEAL, out.turn_log.reason
            assert out.ruled_out == ("margaret",)
            assert set(core.board.live_ids()) == {"blackwood", "ellis", "jeeves"}

            # The two personas above have very different traits, but the
            # reasoning call's options were identical both times -- that
            # identity, not just "some fixed number", is the claim under test.
            assert seen_reasoning_options[0] == seen_reasoning_options[1] == REASONING_PROFILE.options()
            assert seen_reasoning_options[0]["temperature"] == REASONING_TEMPERATURE

            # The voice call, by contrast, actually carried each persona's own
            # options -- that's "reach the voice call only", the other half.
            voice_call = out.calls[1]
            assert voice_call.profile is disciplined
            assert voice_call.options == disciplined.options()
            assert str(disciplined.options()["temperature"]) in voice_call.raw

    # --- single_pass: persona options reach the one call that both decides
    # and speaks -- this is the mode the bead keeps around so the difference
    # from two_pass is something a learner can actually see, not just read. ---
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            core.observe("photo", 0, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            core.observe("strong_inference", 0, "Margaret could not have been at the scene", "inferred_by_model", supports=("photo",))

            def fake_single(profile, prompt, model=None):
                assert profile is neurotic, "single-pass must hand the persona to the one call it makes"
                return json.dumps({
                    "move": "reveal", "cited": ["strong_inference"], "rule_out": "margaret",
                    "reply": "It wasn't Margaret.",
                })

            out = run_turn(core, neurotic, 0, "who did it?", fake_single, mode=SINGLE_PASS)
            assert len(out.calls) == 1, "single-pass makes exactly one call"
            call = out.calls[0]
            assert call.profile is neurotic
            assert call.options == neurotic.options(), "persona sampler settings reached the decision, not just the voice"
            assert call.options["temperature"] != REASONING_TEMPERATURE
            assert out.turn_log.move == moves.REVEAL
            assert out.ruled_out == ("margaret",)
            assert out.reply == "It wasn't Margaret."

            # A single-pass response with no reply is malformed, not silently accepted.
            def fake_missing_reply(profile, prompt, model=None):
                return json.dumps({"move": "ask", "cited": []})

            try:
                run_turn(core, neurotic, 1, "hm?", fake_missing_reply, mode=SINGLE_PASS)
            except ValueError as e:
                assert "reply" in str(e)
            else:
                raise AssertionError("single-pass response missing 'reply' should have been rejected")

    # Non-JSON from the reasoning channel is a named failure, not a crash
    # that leaves no trace of what the model actually said.
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            def fake_garbage(profile, prompt, model=None):
                return "sure, it was the butler probably"

            try:
                run_turn(core, Ocean(), 0, "well?", fake_garbage, mode=TWO_PASS)
            except ValueError as e:
                assert "did not return JSON" in str(e)
            else:
                raise AssertionError("non-JSON reasoning output should have been rejected")

    # An unknown mode is rejected outright, not silently coerced to a default.
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            try:
                run_turn(core, Ocean(), 0, "hi", lambda *a, **k: "{}", mode="just_wing_it")
            except ValueError as e:
                assert "not a mode" in str(e)
            else:
                raise AssertionError("unknown mode should have been rejected")

    print("ok")


if __name__ == "__main__":
    _self_check()
