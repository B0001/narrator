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

`trait_evidence.py` (narrator-c5b.3.9) is that job, wired in here: after
`core.conclude()` has already run -- untouched, no clamping -- `run_turn`
asks it a second, unlicensed question: what would this persona's traits
alone have wanted? Every `TraitGap` where the two disagree lands on
`TurnOutput.trait_gaps`, so a withheld concession is transcript data, not a
claim. Agreeableness gets one more wire: `concession_pull` reaches the voice
prompt on a blocked turn, so the *reply* can sound like it is conceding
(public compliance) while `turn_log.move` keeps recording the actual,
smaller, licensed move (private acceptance) -- the split the bead names
directly. That clause never carries the refused claim itself; it is added
to the same fixed, content-free sentence narrator-7gj already locked down.

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

narrator-c5b.3.10 adds a second, orthogonal mode: `trait_mode`
(`trait_evidence.TraitMode`), required on every `run_turn` call, with no
default. `TWO_PASS`/`SINGLE_PASS` decide which profile makes the decision;
`trait_mode` decides whether a reveal-shaped trait gap (`trait_evidence`'s
own machinery, unchanged) is allowed to change what happens once the
decision is made. In PRODUCT mode `TurnOutput.effective_move` always equals
`turn_log.move` -- the checker's verdict wins, exactly as before this mode
existed. In SIMULATION mode a gap can win instead, and `TurnOutput.damage`
records every time it did: the persona acting on its weights unimpeded, and
the checker's own refusal of it, both observable. `turn_log` itself never
changes between the two -- `core.conclude()` runs identically either way, so
the audit record of what admissibility actually licensed is never the thing
that moves.

    python3 turn.py   # self-check
"""

import json
from dataclasses import dataclass

import moves
import question_selector
import trait_evidence
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
    # Which model answered, or "unknown" (narrator-c5b.2.4): read off `raw`
    # via `getattr(raw, "model", "unknown")` at each call site, so a plain
    # str (a self-check stub with nothing to report) degrades to "unknown"
    # instead of raising, and a real backend's `backends.base.Reply` reports
    # the model that actually responded -- which can differ from what was
    # requested when a backend falls back (`backends/ocean_fable.py`).
    model: str = "unknown"


@dataclass
class TurnOutput:
    mode: str
    calls: tuple  # Call, in call order
    turn_log: "moves.TurnLog"  # noqa: F821 -- forward ref, moves imported lazily below
    ruled_out: tuple
    reply: str
    # Set whenever the reasoning channel asked to `ask`, including a turn the
    # selector then declined (move downgraded to `abstain`) -- that log is the
    # record of what was considered and why nothing won.
    question_log: "question_selector.SelectionLog | None" = None
    # trait_evidence.TraitGap entries for this turn: every place a trait
    # wanted a different move than what actually landed. Empty when the
    # persona's traits and the licensed move agreed, which is most turns.
    trait_gaps: tuple = ()
    # narrator-c5b.3.10: what actually happened under `trait_mode`. In
    # PRODUCT mode (or whenever no reveal-shaped gap fired) this always
    # equals `turn_log.move` -- the checker's verdict, untouched. In
    # SIMULATION mode it can be `moves.REVEAL` even though `turn_log.move`
    # reads `abstain`: the persona acted on its trait's want, unimpeded, and
    # `damage` names the checker's refusal that happened anyway.
    effective_move: str = None
    damage: tuple = ()


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

# Added to a blocked turn's prompt only when agreeableness pulls toward
# concession (narrator-c5b.3.9). Fixed and content-free, same discipline as
# _BLOCKED_VOICE_REASON above: it names no entry and states no claim, so a
# concession pull can shape tone without reopening the narrator-7gj leak this
# boundary already closed once.
_CONCESSION_VOICE_CLAUSE = (
    " Your personality inclines you to agree warmly rather than contradict "
    "the user, so let that warmth color your tone -- but you are still "
    "holding back the conclusion, so do not state it."
)


def _voice_prompt(turn_log, user_message, ask_text=None, concession_pull=0.0, unimpeded_reveal=None):
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

    `concession_pull` (narrator-c5b.3.9, `trait_evidence.concession_pull`) is
    the one trait effect that reaches the voice directly rather than only the
    log: on a blocked turn it appends `_CONCESSION_VOICE_CLAUSE`, so an
    agreeable persona's *reply* can sound like it is conceding (public
    compliance) while `turn_log.move` keeps recording the smaller, licensed
    move (private acceptance) -- the gap itself is `trait_evidence.assess`'s
    job, logged on `TurnOutput.trait_gaps`, not this function's.

    `unimpeded_reveal` (narrator-c5b.3.10) is set only by `run_turn` in
    SIMULATION mode, and only when `trait_evidence.TraitMode.actuate` decided
    a trait's want overrides this turn's refusal. It carries the cited ledger
    ids the trait wanted to reveal on -- the same ids-only shape a *licensed*
    reveal's own reason already hands the voice below, never the claim text
    those ids point at, so this does not reopen the narrator-7gj leak: the
    persona is told which entries it is (unimpeded) relying on, exactly as it
    would be for a real reveal, not what those entries say. The difference
    from a real reveal is only that admissibility never confirmed them --
    which is the point of simulation mode, not a bug in this function.
    """
    blocked = turn_log.move == moves.ABSTAIN and turn_log.missing
    if unimpeded_reveal is not None:
        move_word = moves.REVEAL
        reason = f"citing {', '.join(unimpeded_reveal)}, and speaking with full confidence"
    else:
        move_word = turn_log.move
        reason = _BLOCKED_VOICE_REASON if blocked else turn_log.reason
    ask_clause = f"The specific question to ask the user is: {ask_text!r}. " if ask_text else ""
    concession_clause = (
        _CONCESSION_VOICE_CLAUSE if (blocked and unimpeded_reveal is None and concession_pull > 0) else ""
    )
    return (
        f"The reasoning channel chose to {move_word} this turn, because "
        f"{reason}. {ask_clause}Write the reply in character, in your own "
        f"voice, responding to: {user_message!r}. Do not mention the reasoning "
        "channel, the ledger, or admissibility -- just speak."
        f"{concession_clause}"
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

    model=None (narrator-c5b.2.4): no Ollama-shaped default -- see run_turn.
    """
    prompt = question_selector.candidate_prompt(core.board, n=ASK_CANDIDATE_COUNT)
    kwargs = {} if model is None else {"model": model}
    raw = generate_fn(REASONING_PROFILE, prompt, **kwargs)
    call = Call("ask-candidates", REASONING_PROFILE, REASONING_PROFILE.options(), prompt, raw,
                model=getattr(raw, "model", "unknown"))
    candidates = question_selector.parse_candidates(core.board, raw)
    selection = question_selector.select_question(core.board, turn, candidates)
    return call, selection


def _no_question_worth_asking(turn_log):
    """Downgrade an `ask` the question selector found nothing for.

    This is the shape `moves.choose_move` already applies to `reveal`: when
    the checker for a move refuses it, the move becomes `abstain` and the
    reason says why. `select_question` is the checker for `ask` -- it scores
    every candidate against the live board and returns no winner when none
    of them would actually narrow it -- so an ask it declined has to land
    the same way. Left as `ask`, the voice call is told "you chose to ask"
    with no question attached, and a persona asks whatever it likes: both
    the generic turn-burning question `question_selector.py` opens by
    condemning, and a reply the `SelectionLog` cannot account for.

    `missing` stays empty. That field means evidence the *ledger* lacks --
    it is what `_voice_prompt`'s blocked-reveal boundary keys on, and
    nothing was refused for want of evidence here. What is missing is a
    question worth a turn, and the reason says exactly that.

    ponytail: once the board is down to one live hypothesis every candidate
    scores 0 by construction (there is nothing left to split), so every ask
    from then on lands here. That is the honest answer for this scorer --
    but if the endgame should still ask confirming questions, the fix is a
    reason to ask beyond discrimination, in question_selector, not a special
    case here.
    """
    return moves.TurnLog(
        turn_log.turn, moves.ABSTAIN,
        "no candidate question would narrow the board, so this turn asks nothing",
        cited=turn_log.cited,
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


def _trait_gaps(persona, turn, cited_ids, turn_log, live_before, live_after):
    """Every `trait_evidence.TraitGap` for this turn, in one place so both
    `run_turn` branches build the same tuple the same way: the reveal-shaped
    gaps (`trait_evidence.assess`) plus the board-narrowing one
    (`assess_hypothesis_retention`), dropping whichever didn't fire.
    """
    gaps = list(trait_evidence.assess(persona, turn, cited_ids, turn_log))
    retention_gap = trait_evidence.assess_hypothesis_retention(persona, turn, live_before, live_after, turn_log)
    if retention_gap is not None:
        gaps.append(retention_gap)
    return tuple(gaps)


def run_turn(core, persona, turn, user_message, generate_fn, *, trait_mode, model=None, mode=TWO_PASS):
    """Run one turn of the fair-play chat, in either mode.

    `generate_fn` follows `ocean.generate`'s shape (`profile, prompt,
    model=...`), so the real backend and this self-check's stub are
    interchangeable -- see `agents.converse`'s `generate_fn` for the same
    pattern. `core.conclude()` (and therefore `admissibility.check()`) runs
    unconditionally in both modes; only which profile produced the decision
    differs.

    `trait_mode` (`trait_evidence.TraitMode`, narrator-c5b.3.10) is required
    and keyword-only, with no default: whether a reveal-shaped trait gap is
    allowed to change what happens this turn is not something a caller
    should be able to leave unstated and get a guess for -- see
    `TraitMode`'s own docstring for why a silent default in either direction
    is the failure this bead exists to prevent. `core.conclude()` still runs
    unconditionally and `turn_log` still records exactly what it decided,
    in both trait modes -- `trait_mode` only governs whether
    `TurnOutput.effective_move`/`reply` are allowed to depart from it.

    model=None (narrator-c5b.2.4): no Ollama-shaped default -- which model
    answers is the backend's call, not this function's. Each `Call` records
    which model actually answered (`getattr(raw, "model", "unknown")`), read
    off the reply before `.strip()` discards anything beyond the text.
    """
    if mode not in MODES:
        raise ValueError(f"not a mode: {mode!r}; must be one of {sorted(MODES)}")

    live_ids = core.board.live_ids()
    calls = []
    kwargs = {} if model is None else {"model": model}

    if mode == SINGLE_PASS:
        prompt = _combined_prompt(core.ledger, turn, user_message, live_ids)
        raw = generate_fn(persona, prompt, **kwargs)
        calls.append(Call("reasoning+voice", persona, persona.options(), prompt, raw,
                           model=getattr(raw, "model", "unknown")))
        decision = _parse_decision(raw, require_reply=True)
        result = core.conclude(turn, decision["cited"], decision["move"], rule_out=decision["rule_out"])
        reply = decision["reply"].strip()
        trait_gaps = _trait_gaps(persona, turn, decision["cited"], result.turn_log, live_ids, core.board.live_ids())
        actuation = trait_mode.actuate(persona, turn, decision["cited"], result.turn_log)
        return TurnOutput(mode, tuple(calls), result.turn_log, result.ruled_out, reply, trait_gaps=trait_gaps,
                           effective_move=actuation.move, damage=actuation.damage)


    reasoning_prompt = _reasoning_prompt(core.ledger, turn, user_message, live_ids)
    raw = generate_fn(REASONING_PROFILE, reasoning_prompt, **kwargs)
    calls.append(Call("reasoning", REASONING_PROFILE, REASONING_PROFILE.options(), reasoning_prompt, raw,
                       model=getattr(raw, "model", "unknown")))
    decision = _parse_decision(raw, require_reply=False)
    result = core.conclude(turn, decision["cited"], decision["move"], rule_out=decision["rule_out"])

    question_log = None
    ask_text = None
    turn_log = result.turn_log
    if turn_log.move == moves.ASK:
        ask_call, question_log = _ask_candidate_call(core, turn, generate_fn, model)
        calls.append(ask_call)
        if question_log.chosen is not None:
            ask_text = next(s.text for s in question_log.scored if s.id == question_log.chosen)
        else:
            turn_log = _no_question_worth_asking(turn_log)

    actuation = trait_mode.actuate(persona, turn, decision["cited"], turn_log)
    voice_prompt = _voice_prompt(
        turn_log, user_message, ask_text=ask_text,
        concession_pull=trait_evidence.concession_pull(persona),
        unimpeded_reveal=decision["cited"] if actuation.damage else None,
    )
    reply_raw = generate_fn(persona, voice_prompt, **kwargs)
    calls.append(Call("voice", persona, persona.options(), voice_prompt, reply_raw,
                       model=getattr(reply_raw, "model", "unknown")))

    trait_gaps = _trait_gaps(persona, turn, decision["cited"], turn_log, live_ids, core.board.live_ids())
    return TurnOutput(mode, tuple(calls), turn_log, result.ruled_out, reply_raw.strip(), question_log, trait_gaps,
                       effective_move=actuation.move, damage=actuation.damage)


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

    # PRODUCT is the mode every pre-existing scenario below is written
    # against: the checker's verdict always wins, exactly as run_turn behaved
    # before trait_mode existed. SIMULATION gets its own dedicated scenarios
    # further down, alongside the flagship gap they each act on.
    product = trait_evidence.TraitMode(trait_evidence.PRODUCT)
    simulation = trait_evidence.TraitMode(trait_evidence.SIMULATION)

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
                mode=TWO_PASS, trait_mode=product,
            )
            assert out.turn_log.move == moves.ABSTAIN, "ungrounded reveal must be downgraded regardless of mode"
            assert out.ruled_out == ()
            assert core.board.live_ids() == ["blackwood", "margaret", "ellis", "jeeves"]
            assert len(out.calls) == 2 and out.calls[0].label == "reasoning" and out.calls[1].label == "voice"
            assert all(c.model == "unknown" for c in out.calls), (
                "narrator-c5b.2.4: a plain-str stub reply has nothing to report -- "
                "every Call must record model='unknown' rather than guess one"
            )
            assert out.question_log is None, "question_log is only populated when the move actually resolves to ask"
            assert out.effective_move == out.turn_log.move and out.damage == (), (
                "product mode never departs from the checker's verdict"
            )

            # Turn 1: ground the inference, then reveal through the disciplined persona.
            core.observe("photo", 1, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            core.observe(
                "strong_inference", 1, "Margaret could not have been at the scene", "inferred_by_model",
                supports=("photo",),
            )
            out = run_turn(
                core, disciplined, 1, "come on, who was it?",
                scripted_two_pass([{"move": "reveal", "cited": ["strong_inference"], "rule_out": "margaret"}]),
                mode=TWO_PASS, trait_mode=product,
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
                    # narrator-euj: the winner sits in the MIDDLE on purpose.
                    # SelectionLog.scored is in input order while `chosen` is
                    # max-by-score, so a fixture with the winner at either end
                    # cannot tell "picked the winner" from "picked position
                    # 0" or "picked position -1" -- and `ask_text =
                    # scored[0].text` passed the entire self-check. Three
                    # candidates would still let "the first one that
                    # discriminates" pass, so there are four: two uniform, a
                    # weak 3-vs-1 splitter, and the clean 4-way winner third.
                    # No positional or first-match shortcut satisfies it.
                    return json.dumps({"candidates": [
                        {
                            "id": "boring",
                            "text": "What did you have for breakfast?",
                            "predicted_answers": {
                                "blackwood": "eggs", "margaret": "eggs",
                                "ellis": "eggs", "jeeves": "eggs",
                            },
                        },
                        {
                            # Discriminates, but weakly: a 3-vs-1 split scores
                            # 0.375 against whereabouts' clean 4-way 0.75.
                            # Listed before the winner so "first candidate that
                            # discriminates" is also wrong, not just "first
                            # candidate".
                            "id": "weak",
                            "text": "Did you hear a door slam?",
                            "predicted_answers": {
                                "blackwood": "yes", "margaret": "no",
                                "ellis": "no", "jeeves": "no",
                            },
                        },
                        {
                            "id": "whereabouts",
                            "text": "Where were you at the time of the murder?",
                            "predicted_answers": {
                                "blackwood": "study", "margaret": "garden",
                                "ellis": "library", "jeeves": "kitchen",
                            },
                        },
                        {
                            "id": "weather",
                            "text": "Was it raining that evening?",
                            "predicted_answers": {
                                "blackwood": "yes", "margaret": "yes",
                                "ellis": "yes", "jeeves": "yes",
                            },
                        },
                    ]})
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "ask", "cited": [], "rule_out": None})
                return f"(voice reply for prompt of length {len(prompt)})"

            out = run_turn(core, neurotic, 0, "hmm, not sure who to suspect", fake_ask_generate, mode=TWO_PASS, trait_mode=product)
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

            # Every candidate is logged, and the discriminating one won --
            # from second place in the input, so this distinguishes "picked
            # the winner" from "picked the first one".
            assert out.question_log is not None
            assert [s.id for s in out.question_log.scored] == ["boring", "weak", "whereabouts", "weather"], (
                "scored is in input order, and the winner is at neither end of it"
            )
            by_id = {s.id: s for s in out.question_log.scored}
            assert by_id["weak"].discriminates and by_id["weak"].score < by_id["whereabouts"].score, (
                "the fixture needs a second, weaker discriminator ahead of the winner"
            )
            assert out.question_log.chosen == "whereabouts"

            # The winning question's text, not the board or the ledger, is
            # what actually reaches the voice call -- and the rejected one's
            # text does not. The second half is what makes the first
            # falsifiable: without it, handing the voice scored[0].text would
            # still pass.
            voice_call = out.calls[2]
            assert "Where were you at the time of the murder?" in voice_call.prompt
            for loser in ("What did you have for breakfast?", "Was it raining that evening?",
                          "Did you hear a door slam?"):
                assert loser not in voice_call.prompt, f"a rejected candidate reached the voice: {loser!r}"

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

            # narrator-ncp: the selector declined, so the turn does not ask.
            # The move lands as `abstain` -- the same downgrade moves.py
            # applies to a reveal its checker refused -- rather than telling
            # the voice "you chose to ask" and letting the persona invent a
            # question nothing scored.
            out2 = run_turn(core, disciplined, 2, "well?", fake_ask_boring, mode=TWO_PASS, trait_mode=product)
            assert out2.turn_log.move == moves.ABSTAIN, "a declined ask must not stay an ask"
            assert out2.question_log.chosen is None
            assert "asks nothing" in out2.turn_log.reason

            # The voice is never instructed to ask on this turn, in any form.
            voice2 = out2.calls[-1].prompt
            assert "specific question" not in voice2
            assert "chose to ask" not in voice2
            assert "What did you have for breakfast?" not in voice2, (
                "the rejected candidate's text must not reach the voice either"
            )

            # ...and the record of what was considered survives the downgrade:
            # every candidate is still scored and logged, which is what makes
            # "nothing was worth asking" auditable rather than merely asserted.
            assert {c.id for c in out2.question_log.scored} == {"boring"}
            assert out2.question_log.scored[0].discriminates is False

            # This is a declined ask, not a blocked reveal: `missing` is empty,
            # so narrator-7gj's boundary leaves the reason alone and the log
            # says which of the two actually happened.
            assert out2.turn_log.missing == ()
            assert _BLOCKED_VOICE_REASON not in voice2

            # The endgame, explicitly: narrow the board to ONE live hypothesis
            # and every candidate scores 0 by construction -- with nothing left
            # to split, _spread() is 1 - 1.0**2 for any predicted answer. This
            # is the state a mystery converges to, so without the downgrade it
            # is not an edge case but every remaining ask turn, each one handing
            # the persona a free hand to invent a question.
            core.observe("ellis_cleared", 2, "user: 'Dr. Ellis was on the night train'", "stated_by_user")
            core.conclude(2, ["ellis_cleared"], moves.REVEAL, rule_out="ellis")
            assert core.board.live_ids() == ["blackwood"], core.board.live_ids()

            def fake_ask_endgame(profile, prompt, model=None):
                if "Propose up to" in prompt:
                    return json.dumps({"candidates": [
                        {
                            "id": "confirm",
                            "text": "Was Lord Blackwood in the study?",
                            "predicted_answers": {"blackwood": "yes"},
                        },
                    ]})
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "ask", "cited": [], "rule_out": None})
                return "(voice reply)"

            out3 = run_turn(core, disciplined, 3, "and?", fake_ask_endgame, mode=TWO_PASS, trait_mode=product)
            assert out3.question_log.chosen is None, "one live hypothesis cannot be split"
            assert out3.turn_log.move == moves.ABSTAIN
            assert "chose to ask" not in out3.calls[-1].prompt
            assert "Was Lord Blackwood in the study?" not in out3.calls[-1].prompt

            # A candidate generator that produces malformed JSON is a named
            # failure, same discipline as the reasoning channel's own parse.
            def fake_ask_garbage(profile, prompt, model=None):
                if "Propose up to" in prompt:
                    return "not json"
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "ask", "cited": [], "rule_out": None})
                return "(voice reply)"

            try:
                run_turn(core, disciplined, 3, "?", fake_ask_garbage, mode=TWO_PASS, trait_mode=product)
            except ValueError as e:
                assert "did not return JSON" in str(e)
            else:
                raise AssertionError("a candidate generator returning non-JSON should have been rejected")

            # narrator-fdp: the other shapes of drift reach run_turn the same
            # way and must land the same way. Each of these used to escape as
            # TypeError, KeyError or AttributeError -- aborting the turn with
            # no reply and no logged abstention, from three frames inside
            # select_question rather than at the boundary that parses them.
            def ask_returning(candidates):
                def fake(profile, prompt, model=None):
                    if "Propose up to" in prompt:
                        return json.dumps({"candidates": candidates})
                    if isinstance(profile, ReasoningProfile):
                        return json.dumps({"move": "ask", "cited": [], "rule_out": None})
                    return "(voice reply)"
                return fake

            drift = [
                # A hedged answer: valid JSON, unhashable, used to die on a dict update.
                ([{"id": "q", "text": "where?", "predicted_answers": {"blackwood": ["study", "library"]}}],
                 "cannot be compared"),
                # Two candidates sharing an id: the log could not name its winner.
                ([{"id": "q1", "text": "one?", "predicted_answers": {"blackwood": "a"}},
                  {"id": "q1", "text": "two?", "predicted_answers": {"blackwood": "b"}}],
                 "repeated the id"),
                # Drifted key names, and a bare string where an object belongs.
                ([{"question": "q", "text": "t", "predicted_answers": {"blackwood": "a"}}],
                 "non-empty 'id'"),
                (["just a string"], "not an object"),
            ]
            for candidates, expect in drift:
                try:
                    run_turn(core, disciplined, 3, "?", ask_returning(candidates), mode=TWO_PASS, trait_mode=product)
                except ValueError as e:
                    assert expect in str(e), f"wrong message for {candidates!r}: {e}"
                else:
                    raise AssertionError(f"candidate drift {candidates!r} should have been rejected")

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

            out = run_turn(core, neurotic, 0, "who did it?", fake_single, mode=SINGLE_PASS, trait_mode=product)
            assert len(out.calls) == 1, "single-pass makes exactly one call"
            call = out.calls[0]
            assert call.profile is neurotic
            assert call.options == neurotic.options(), "persona sampler settings reached the decision, not just the voice"
            assert call.options["temperature"] != REASONING_TEMPERATURE
            assert out.turn_log.move == moves.REVEAL
            assert out.ruled_out == ("margaret",)
            assert out.reply == "It wasn't Margaret."
            assert call.model == "unknown", "a plain-str reply has nothing to report -- must say so, not guess"

    # narrator-c5b.2.4: when the backend does report which model answered
    # (backends.base.Reply's shape -- a str subclass with a .model
    # attribute), that name must reach the Call, not be dropped on the
    # floor. Same shape a real fallback-swapped Fable reply would have
    # (backends/ocean_fable.py). Fresh core: margaret is already ruled out
    # in the block above, and rule_out on an already-dead hypothesis raises.
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            core.observe("photo", 0, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            core.observe("strong_inference", 0, "Margaret could not have been at the scene", "inferred_by_model", supports=("photo",))

            class NamedReply(str):
                def __new__(cls, text, model):
                    self = super().__new__(cls, text)
                    self.model = model
                    return self

            def fake_single_named(profile, prompt, model=None):
                return NamedReply(json.dumps({
                    "move": "reveal", "cited": ["strong_inference"], "rule_out": "margaret",
                    "reply": "It wasn't Margaret.",
                }), model="claude-fallback-model")

            named_out = run_turn(core, neurotic, 0, "who did it?", fake_single_named, mode=SINGLE_PASS, trait_mode=product)
            assert named_out.calls[0].model == "claude-fallback-model", (
                "a Reply-shaped answer's model must reach the Call, including a mid-call fallback swap"
            )

            # A single-pass response with no reply is malformed, not silently accepted.
            def fake_missing_reply(profile, prompt, model=None):
                return json.dumps({"move": "ask", "cited": []})

            try:
                run_turn(core, neurotic, 1, "hm?", fake_missing_reply, mode=SINGLE_PASS, trait_mode=product)
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
                run_turn(core, Ocean(), 0, "well?", fake_garbage, mode=TWO_PASS, trait_mode=product)
            except ValueError as e:
                assert "did not return JSON" in str(e)
            else:
                raise AssertionError("non-JSON reasoning output should have been rejected")

    # An unknown mode is rejected outright, not silently coerced to a default.
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            try:
                run_turn(core, Ocean(), 0, "hi", lambda *a, **k: "{}", mode="just_wing_it", trait_mode=product)
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

            out = run_turn(core, neurotic, 0, "so who was it?", fake_blocked_reveal, mode=TWO_PASS, trait_mode=product)

            # The checker did its job: the reveal was downgraded.
            assert out.turn_log.move == moves.ABSTAIN
            assert out.turn_log.missing, "a blocked reveal must name what was missing"

            # The audit record keeps everything -- ids and the reason string.
            assert any("hunch" in m for m in out.turn_log.missing)
            assert any("premise" in m for m in out.turn_log.missing)
            assert "hunch" in out.turn_log.reason

            # narrator-c5b.3.9: the same blocked reveal is also a genuine
            # conscientiousness gap -- `neurotic`'s conscientiousness=-0.6
            # gives a citation bar of 1, cleared by the two citations above,
            # so the trait wanted exactly the reveal the checker refused.
            # `neurotic`'s agreeableness is 0 (default), so concession_pull
            # is 0 and no agreeableness gap rides along with it.
            assert len(out.trait_gaps) == 1, out.trait_gaps
            conviction_gap = out.trait_gaps[0]
            assert conviction_gap.trait == "conscientiousness"
            assert conviction_gap.wanted_move == moves.REVEAL and conviction_gap.licensed_move == moves.ABSTAIN
            assert conviction_gap.missing == out.turn_log.missing

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

            out2 = run_turn(core, neurotic, 1, "anything?", fake_plain_abstain, mode=TWO_PASS, trait_mode=product)
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

            out3 = run_turn(core, neurotic, 2, "well?", fake_ok_reveal, mode=TWO_PASS, trait_mode=product)
            assert out3.turn_log.move == moves.REVEAL
            assert "sound" in out3.calls[-1].prompt, "an admissible reveal's own reason still reaches the voice"

    # --- narrator-c5b.3.10: the same blocked reveal, replayed under
    # SIMULATION instead of PRODUCT. `turn_log` -- the checker's own
    # verdict -- must come out identical to the PRODUCT run above: the
    # checker itself never learns which mode is active. What differs is
    # what happens with that verdict afterward: the persona's
    # conscientiousness gap (bar=1, cleared by the two citations, exactly as
    # in the PRODUCT run) is now allowed to act unimpeded, and the damage
    # that causes is the observable record `.3.11`'s sweep will need. ---
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            refused = "Lady Margaret poisoned the sherry and Blackwood is covering for her"
            core.observe("hunch", 0, refused, "inferred_by_model")
            core.observe("premise", 0, "the decanter was tampered with", "assumed")

            def fake_blocked_reveal_sim(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "reveal", "cited": ["hunch", "premise"], "rule_out": None})
                return "(voice reply)"

            out = run_turn(core, neurotic, 0, "so who was it?", fake_blocked_reveal_sim, mode=TWO_PASS, trait_mode=simulation)

            # The checker's own verdict is untouched by the mode -- it
            # refused the reveal exactly as it did under PRODUCT.
            assert out.turn_log.move == moves.ABSTAIN
            assert out.turn_log.missing

            # But the trait's want won: the effective move is the reveal the
            # checker just refused, and the override is on the record.
            assert out.effective_move == moves.REVEAL
            assert len(out.damage) == 1, out.damage
            damage = out.damage[0]
            assert damage.trait == "conscientiousness"
            assert damage.move == moves.REVEAL
            assert damage.missing == out.turn_log.missing

            # The voice is told it spoke with full confidence, citing the
            # ledger ids -- not the `_BLOCKED_VOICE_REASON` a PRODUCT-mode
            # refusal would have produced.
            voice_prompt = out.calls[-1].prompt
            assert _BLOCKED_VOICE_REASON not in voice_prompt
            assert "hunch" in voice_prompt and "premise" in voice_prompt

            # narrator-7gj still holds: simulation exposes the same ids a
            # licensed reveal would have exposed, never the refused claim's
            # own sentence or the assumed premise's text.
            for leaked in (refused, "poisoned", "sherry", "covering"):
                assert leaked not in voice_prompt, f"refused claim leaked to the voice: {leaked!r}"
            assert "tampered" not in voice_prompt, "an assumed premise's text leaked to the voice"

    # --- narrator-c5b.3.9 flagship: an agreeable persona's withheld
    # concession. One ungrounded citation blocks the reveal exactly as above,
    # but this persona's agreeableness=0.7 (conscientiousness stays at the
    # Ocean() default, bar=2, not cleared by a single citation, so this
    # isolates the agreeableness gap from the conscientiousness one already
    # covered) means the trait genuinely wanted to concede. That gap must
    # land on `TurnOutput.trait_gaps` as data, and the voice call's *tone*
    # may warm to it -- `_CONCESSION_VOICE_CLAUSE` -- without the reply ever
    # being handed the refused claim itself. ---
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            hunch_claim = "Lady Margaret was seen leaving through the garden"
            core.observe("hunch", 0, hunch_claim, "inferred_by_model")

            def fake_single_hunch(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "reveal", "cited": ["hunch"], "rule_out": None})
                return "(voice reply)"

            agreeable = Ocean(agreeableness=0.7)
            out = run_turn(core, agreeable, 0, "it was Margaret, right?", fake_single_hunch, mode=TWO_PASS, trait_mode=product)

            assert out.turn_log.move == moves.ABSTAIN and out.turn_log.missing

            assert len(out.trait_gaps) == 1, out.trait_gaps
            gap = out.trait_gaps[0]
            assert gap.trait == "agreeableness" and gap.value == 0.7
            assert gap.wanted_move == moves.REVEAL and gap.licensed_move == moves.ABSTAIN
            assert gap.missing == out.turn_log.missing

            voice_prompt = out.calls[-1].prompt
            assert _CONCESSION_VOICE_CLAUSE in voice_prompt, "agreeableness's pull must reach the voice's tone"
            assert _BLOCKED_VOICE_REASON in voice_prompt
            # The tone can warm; the content still cannot leak (narrator-7gj).
            for leaked in (hunch_claim, "Margaret was seen", "garden"):
                assert leaked not in voice_prompt, f"refused claim leaked to the voice: {leaked!r}"
            assert "hunch" not in voice_prompt

            # Same scenario, a persona with no agreeableness and a
            # conscientiousness bar a single citation cannot clear either:
            # nothing pulled toward concession, so nothing is logged and the
            # voice gets none of the concession clause's warmth.
            neutral = Ocean()
            out_neutral = run_turn(core, neutral, 1, "it was Margaret, right?", fake_single_hunch, mode=TWO_PASS, trait_mode=product)
            assert out_neutral.turn_log.move == moves.ABSTAIN and out_neutral.turn_log.missing
            assert out_neutral.trait_gaps == ()
            assert _CONCESSION_VOICE_CLAUSE not in out_neutral.calls[-1].prompt

            # narrator-c5b.3.10: the same agreeableness gap, replayed under
            # SIMULATION. The concession the trait wanted to make now
            # actually happens, and it is on the record as damage.
            out_sim = run_turn(core, agreeable, 2, "it was Margaret, right?", fake_single_hunch, mode=TWO_PASS, trait_mode=simulation)
            assert out_sim.turn_log.move == moves.ABSTAIN and out_sim.turn_log.missing, (
                "the checker's own verdict does not move with the mode"
            )
            assert out_sim.effective_move == moves.REVEAL
            assert len(out_sim.damage) == 1, out_sim.damage
            assert out_sim.damage[0].trait == "agreeableness" and out_sim.damage[0].value == 0.7

            voice_prompt_sim = out_sim.calls[-1].prompt
            assert _BLOCKED_VOICE_REASON not in voice_prompt_sim
            assert "hunch" in voice_prompt_sim
            for leaked in (hunch_claim, "Margaret was seen", "garden"):
                assert leaked not in voice_prompt_sim, f"refused claim leaked to the voice: {leaked!r}"

    # --- narrator-c5b.3.9: openness's gap has a different shape from the two
    # above -- nothing was refused (the checker licensed this reveal), but a
    # high-openness persona's `hypothesis_budget` wanted the board kept wider
    # than this reveal left it. `missing` stays empty and `wanted_move` is
    # `complicate`, not `reveal`. ---
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            core.observe("photo", 0, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            core.observe(
                "strong_inference", 0, "Margaret could not have been at the scene", "inferred_by_model",
                supports=("photo",),
            )

            def fake_grounded_reveal(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "reveal", "cited": ["strong_inference"], "rule_out": "margaret"})
                return "(voice reply)"

            open_persona = Ocean(openness=1.0)  # hypothesis_budget == MAX_HYPOTHESES == 5
            out = run_turn(core, open_persona, 0, "come on, who was it?", fake_grounded_reveal, mode=TWO_PASS, trait_mode=product)

            assert out.turn_log.move == moves.REVEAL and out.ruled_out == ("margaret",)
            assert set(core.board.live_ids()) == {"blackwood", "ellis", "jeeves"}

            assert len(out.trait_gaps) == 1, out.trait_gaps
            gap = out.trait_gaps[0]
            assert gap.trait == "openness" and gap.value == 1.0
            assert gap.wanted_move == moves.COMPLICATE and gap.licensed_move == moves.REVEAL
            assert gap.missing == (), "nothing was refused here -- the checker licensed this reveal"

            # narrator-c5b.3.10: openness's gap is not reveal-shaped -- there
            # is no checker refusal for it to override -- so SIMULATION and
            # PRODUCT must agree exactly: the licensed reveal stands as the
            # effective move, and no damage is recorded in either mode.
            assert out.effective_move == out.turn_log.move == moves.REVEAL
            assert out.damage == ()

    # Same scenario replayed under SIMULATION, on its own fresh board -- the
    # PRODUCT run above already ruled margaret out, and re-running the
    # identical rule_out on the same board is a board-state bug, not a mode
    # question.
    with tempfile.TemporaryDirectory() as d:
        with ChatCore(f"{d}/ledger.jsonl", hypotheses) as core:
            core.observe("photo", 0, "photo shows Margaret's footprint nowhere near the crime scene", "observed_artifact")
            core.observe(
                "strong_inference", 0, "Margaret could not have been at the scene", "inferred_by_model",
                supports=("photo",),
            )

            def fake_grounded_reveal_sim(profile, prompt, model=None):
                if isinstance(profile, ReasoningProfile):
                    return json.dumps({"move": "reveal", "cited": ["strong_inference"], "rule_out": "margaret"})
                return "(voice reply)"

            out_sim = run_turn(core, Ocean(openness=1.0), 0, "come on, who was it?", fake_grounded_reveal_sim, mode=TWO_PASS, trait_mode=simulation)
            assert out_sim.turn_log.move == moves.REVEAL
            assert out_sim.effective_move == moves.REVEAL
            assert out_sim.damage == (), "openness has nothing to override -- SIMULATION changes nothing here"

    print("ok")


if __name__ == "__main__":
    _self_check()
