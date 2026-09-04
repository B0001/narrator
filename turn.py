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

narrator-9nl wires `question_selector.py` in on the same principle: when the
reasoning call's decision resolves to `ask`, deciding *which* question is
worth asking is another "what actually splits the board" judgment, not a
voice one, so candidate generation (`question_selector.generate_candidates`)
runs at `REASONING_PROFILE` too, over `core.board` -- the same live board
`ChatCore.conclude()` reads, never a copy -- and `select_question()`'s
winner, not the model's free-form invention, is what the voice call is told
to ask. `TWO_PASS` only: `SINGLE_PASS` makes exactly one call by design (see
its own self-check), and inserting a second, board-reading call there would
break that invariant for a mode that exists specifically to demonstrate what
happens *without* a separated reasoning step.

    python3 turn.py   # self-check
"""

import json
from dataclasses import dataclass

import moves
import question_selector
from ocean import Ocean

TWO_PASS = "two_pass"
SINGLE_PASS = "single_pass"
MODES = frozenset({TWO_PASS, SINGLE_PASS})

REASONING_TEMPERATURE = 0.2
ASK_CANDIDATE_COUNT = 3


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
    question_log: "question_selector.SelectionLog | None" = None  # only set when this turn's move is "ask"


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


# What the voice is told when the checker refused the reveal it was asked for.
# Fixed text: it names no entry, cites nothing, and says only what the persona
# actually needs to act on -- that it is holding back.
_BLOCKED_VOICE_REASON = "the evidence on the ledger does not support the conclusion it was about to state"


def _voice_prompt(turn_log, user_message, ask_text=None):
    """Build the persona call's prompt. This is a trust boundary, not a
    formatting helper (narrator-7gj).

    A reveal that `admissibility.check()` refused is the one turn whose
    reason must not reach the voice. `moves.choose_move` builds that reason
    out of the checker's `missing` tuple, which names the very entries it
    just declined to let the bot assert -- and the voice call is handed no
    ledger and no board, so that string would be nearly the whole prompt,
    which makes paraphrasing the refused claim the model's most available
    continuation. The reply then states the conclusion while `turn_log.move`
    still records `abstain`: the audit record says the bot held back on a
    turn where it did not. That is exactly the narrator-cby.4.1 shape, and it
    would falsify moves.py's own invariant that "there is no route from 'I
    have a conclusion' to 'I said it out loud' that skips the check" -- the
    route would run through the reason string.

    The full `missing` tuple stays on the `TurnLog`, which is where the audit
    record wants it. Only the persona's copy is withheld. A directly
    requested abstain keeps its own reason: nothing was refused there, so
    there is nothing to withhold, and `missing` being empty is what tells
    the two apart.
    """
    blocked = turn_log.move == moves.ABSTAIN and turn_log.missing
    reason = _BLOCKED_VOICE_REASON if blocked else turn_log.reason
    ask_clause = f"The specific question to ask the user is: {ask_text!r}. " if ask_text else ""
    return (
        f"The reasoning channel chose to {turn_log.move} this turn, because "
        f"{reason}. {ask_clause}Write the reply in character, in your own "
        f"voice, responding to: {user_message!r}. Do not mention the reasoning "
        "channel, the ledger, or admissibility -- just speak."
    )


def _ask_candidate_call(core, turn, generate_fn, model):
    """When the reasoning channel's move resolves to `ask`, generate and
    score candidate questions over the board's *live* hypotheses -- the
    same board `core.conclude()` already reads, not a mock -- and hand back
    both the `Call` (so this model invocation is as observable as reasoning
    and voice) and the `SelectionLog` (so a caller can see every candidate
    considered, not just the winner).

    Pinned to `REASONING_PROFILE`, same as the reasoning call: choosing
    *which* question is worth asking is a "what actually splits the board"
    decision, not a persona-voice one, so it must not be run at a persona's
    drifting sampler settings either.
    """
    prompt = question_selector.candidate_prompt(core.board, n=ASK_CANDIDATE_COUNT)
    raw = generate_fn(REASONING_PROFILE, prompt, model=model)
    call = Call("ask-candidates", REASONING_PROFILE, REASONING_PROFILE.options(), prompt, raw)
    candidates = question_selector.parse_candidates(core.board, raw)
    selection = question_selector.select_question(core.board, turn, candidates)
    return call, selection


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

    question_log = None
    ask_text = None
    if result.turn_log.move == moves.ASK:
        ask_call, question_log = _ask_candidate_call(core, turn, generate_fn, model)
        calls.append(ask_call)
        if question_log.chosen is not None:
            ask_text = next(s.text for s in question_log.scored if s.id == question_log.chosen)

    voice_prompt = _voice_prompt(result.turn_log, user_message, ask_text=ask_text)
    reply_raw = generate_fn(persona, voice_prompt, model=model)
    calls.append(Call("voice", persona, persona.options(), voice_prompt, reply_raw))

    return TurnOutput(mode, tuple(calls), result.turn_log, result.ruled_out, reply_raw.strip(), question_log)


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
            assert out.question_log is None, "question_log is only populated when the move actually resolves to ask"

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

    # --- two_pass ask: when the reasoning channel's move resolves to `ask`,
    # the actual question comes from question_selector.select_question() run
    # over the real board (core.board, not a stand-in), and the
    # candidate-generation call is pinned to REASONING_PROFILE exactly like
    # the decision call -- deciding what's worth asking is the same kind of
    # "what actually splits the board" judgment, not a persona-voice one. ---
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            core.observe("saw_margaret", 0, "user: 'I saw Lady Margaret near the Library'", "stated_by_user")

            seen_ask_prompts, seen_ask_options = [], []

            def fake_ask_generate(profile, prompt, model=None):
                if "Propose up to" in prompt:
                    seen_ask_prompts.append(prompt)
                    seen_ask_options.append(profile.options())
                    return json.dumps({"candidates": [
                        {
                            "id": "whereabouts",
                            "text": "Where were you at the time of the murder?",
                            "predicted_answers": {
                                "blackwood": "study", "margaret": "garden",
                                "ellis": "library", "jeeves": "kitchen",
                            },
                        },
                        {
                            "id": "boring",
                            "text": "What did you have for breakfast?",
                            "predicted_answers": {
                                "blackwood": "eggs", "margaret": "eggs",
                                "ellis": "eggs", "jeeves": "eggs",
                            },
                        },
                    ]})
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "ask", "cited": [], "rule_out": None})
                return f"(voice reply for prompt of length {len(prompt)})"

            out = run_turn(core, neurotic, 0, "hmm, not sure who to suspect", fake_ask_generate, mode=TWO_PASS)
            assert out.turn_log.move == moves.ASK
            assert [c.label for c in out.calls] == ["reasoning", "ask-candidates", "voice"]

            # Candidate generation ran at the reasoning channel's fixed
            # profile, never the neurotic persona's own (drifting) sampler.
            ask_call = out.calls[1]
            assert ask_call.profile is REASONING_PROFILE
            assert ask_call.options == REASONING_PROFILE.options()
            assert seen_ask_options[0]["temperature"] == REASONING_TEMPERATURE

            # The candidate prompt saw every currently-live hypothesis and
            # nothing else -- this board has no ruled-out hypotheses yet, so
            # this just confirms the wiring reads core.board, not a mock.
            for hid in ("blackwood", "margaret", "ellis", "jeeves"):
                assert hid in seen_ask_prompts[0]

            # Every candidate is logged, and the discriminating one won.
            assert out.question_log is not None
            assert {s.id for s in out.question_log.scored} == {"whereabouts", "boring"}
            assert out.question_log.chosen == "whereabouts"

            # The winning question's text, not the board or the ledger, is
            # what actually reaches the voice call.
            voice_call = out.calls[2]
            assert "Where were you at the time of the murder?" in voice_call.prompt

            # Rule out every hypothesis but one candidate can't discriminate
            # between (blackwood vs. ellis) -- no candidate on offer splits
            # the survivors, so `chosen` must honestly be None, and the voice
            # call must not claim a specific question that doesn't exist.
            core.observe("photo", 1, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            core.observe("strong_inference", 1, "Margaret could not have been at the scene", "inferred_by_model", supports=("photo",))
            core.conclude(1, ["strong_inference"], moves.REVEAL, rule_out="margaret")
            core.observe("alibi", 1, "user: 'Jeeves was in London all week'", "stated_by_user")
            core.conclude(1, ["alibi"], moves.REVEAL, rule_out="jeeves")
            assert set(core.board.live_ids()) == {"blackwood", "ellis"}

            def fake_ask_boring(profile, prompt, model=None):
                if "Propose up to" in prompt:
                    return json.dumps({"candidates": [
                        {
                            "id": "boring",
                            "text": "What did you have for breakfast?",
                            "predicted_answers": {"blackwood": "eggs", "ellis": "eggs"},
                        },
                    ]})
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "ask", "cited": [], "rule_out": None})
                return "(voice reply)"

            out2 = run_turn(core, disciplined, 2, "well?", fake_ask_boring, mode=TWO_PASS)
            assert out2.turn_log.move == moves.ASK
            assert out2.question_log.chosen is None
            assert "specific question" not in out2.calls[-1].prompt

            # A candidate generator that produces malformed JSON is a named
            # failure, same discipline as the reasoning channel's own parse.
            def fake_ask_garbage(profile, prompt, model=None):
                if "Propose up to" in prompt:
                    return "not json"
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "ask", "cited": [], "rule_out": None})
                return "(voice reply)"

            try:
                run_turn(core, disciplined, 3, "?", fake_ask_garbage, mode=TWO_PASS)
            except ValueError as e:
                assert "did not return JSON" in str(e)
            else:
                raise AssertionError("a candidate generator returning non-JSON should have been rejected")

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

    # --- narrator-7gj: a reveal the checker refused must not smuggle the
    # refused claim into the voice call. Same leak-check shape chapters.py
    # uses for sim.culprit: name the one thing that must not cross the
    # boundary, then assert on the actual string that crossed it -- not on a
    # docstring promise that it won't. ---
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            # A claim with distinctive words, so a leak cannot hide in
            # vocabulary the prompt would have contained anyway.
            refused = "Lady Margaret poisoned the sherry and Blackwood is covering for her"
            core.observe("hunch", 0, refused, "inferred_by_model")
            core.observe("premise", 0, "the decanter was tampered with", "assumed")

            def fake_blocked_reveal(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "reveal", "cited": ["hunch", "premise"], "rule_out": None})
                return "(voice reply)"

            out = run_turn(core, neurotic, 0, "so who was it?", fake_blocked_reveal, mode=TWO_PASS)

            # The checker did its job: the reveal was downgraded.
            assert out.turn_log.move == moves.ABSTAIN
            assert out.turn_log.missing, "a blocked reveal must name what was missing"

            # The audit record keeps everything -- ids and the reason string.
            assert any("hunch" in m for m in out.turn_log.missing)
            assert any("premise" in m for m in out.turn_log.missing)
            assert "hunch" in out.turn_log.reason

            # ...and neither the audit record nor the voice prompt carries the
            # refused sentence itself. The ledger already holds it under
            # `hunch`; an auditor looks it up there.
            voice_prompt = out.calls[-1].prompt
            assert out.calls[-1].label == "voice"
            for leaked in (refused, "poisoned", "sherry", "covering"):
                assert leaked not in voice_prompt, f"refused claim leaked to the voice: {leaked!r}"
                assert all(leaked not in m for m in out.turn_log.missing), f"checker copied {leaked!r}"
            assert "tampered" not in voice_prompt, "an assumed premise's text leaked to the voice"

            # The voice is told it is holding back, and told nothing else --
            # not which entries failed, since a model-chosen entry id can be
            # self-describing too.
            assert _BLOCKED_VOICE_REASON in voice_prompt
            assert "hunch" not in voice_prompt and "premise" not in voice_prompt

            # A directly requested abstain was refused nothing, so it keeps its
            # own reason -- the boundary must not flatten every abstain alike.
            def fake_plain_abstain(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "abstain", "cited": [], "rule_out": None})
                return "(voice reply)"

            out2 = run_turn(core, neurotic, 1, "anything?", fake_plain_abstain, mode=TWO_PASS)
            assert out2.turn_log.move == moves.ABSTAIN and not out2.turn_log.missing
            assert "declining to conclude yet" in out2.calls[-1].prompt
            assert _BLOCKED_VOICE_REASON not in out2.calls[-1].prompt

            # A reveal the checker ALLOWED still reaches the voice with its own
            # reason: this boundary withholds refused content, not all content.
            core.observe("photo", 2, "photo shows the decanter unsealed", "observed_artifact")
            core.observe("sound", 2, "the seal was broken before dinner", "inferred_by_model", supports=("photo",))

            def fake_ok_reveal(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "reveal", "cited": ["sound"], "rule_out": None})
                return "(voice reply)"

            out3 = run_turn(core, neurotic, 2, "well?", fake_ok_reveal, mode=TWO_PASS)
            assert out3.turn_log.move == moves.REVEAL
            assert "sound" in out3.calls[-1].prompt, "an admissible reveal's own reason still reaches the voice"

    print("ok")


if __name__ == "__main__":
    _self_check()
