"""Traits move evidence weights; a withheld concession logs the counterfactual (C5).

`admissibility.py` and `moves.py` are checker-pure by construction (`panel.py`
enforces it at the checker's own construction time): whether a reveal is
licensed never depends on who is speaking. That purity is not negotiable and
this module does not touch it. But the persona speaking *is* someone with
traits, and a trait-neutral persona is a fiction -- an agreeable persona
genuinely wants to agree with what the user just proposed, a conscientious
one genuinely wants more corroboration before it will say so, and an open one
genuinely resists collapsing the board down to one reading. Scoring every
piece of testimony identically, as the ledger correctly does for admissibility
purposes, would erase that if nothing else recorded it.

So this module computes a second, unlicensed opinion -- what the persona's
traits alone would have wanted, independent of whatever `admissibility.check`
actually decided -- and compares the two. No clamping: nothing here can
change a `TurnLog`'s `move`, a board's weights, or a `Verdict`. It only
notices when the two disagree and produces a `TraitGap` record so the
disagreement is observable data instead of a claim about the persona's inner
life.  That is the public-compliance/private-acceptance split the bead asks
for: `turn.py` wires `concession_pull` into the *voice* prompt (so an
agreeable persona's reply can sound warm even while withholding), while the
`TraitGap` records land on the `TurnLog`/transcript side and never move less
than admissibility actually licensed.

Three traits, three quantities:

  openness           -- `hypothesis_budget`: how many live hypotheses the
                         persona wants kept on the board before it is
                         comfortable letting a reveal narrow it further.
  conscientiousness   -- `evidence_bar`: how many citations the persona wants
                         behind a reveal before it is willing to want one at
                         all, regardless of whether the ledger would ground
                         them.
  agreeableness       -- `source_credibility` (testimony from a liked source
                         is weighted up, a disliked one down) and
                         `concession_pull` (how hard the trait pulls toward
                         conceding to whatever the user just proposed).

    python3 trait_evidence.py   # self-check
"""

from dataclasses import dataclass

import moves
from hypothesis_board import MAX_HYPOTHESES, MIN_HYPOTHESES

assert MIN_HYPOTHESES >= 1  # hypothesis_budget's floor assumes this


@dataclass
class TraitGap:
    """One trait's private read, logged next to what actually happened.

    `wanted_move` is what the trait alone would have picked; `licensed_move`
    is the `TurnLog.move` that actually reached the record; `missing` is
    copied from the same `Verdict.missing` the checker produced (never a
    claim text -- see `admissibility.py`'s narrator-7gj rule, which this
    module inherits by carrying the checker's own tuple rather than
    re-deriving one). A gap with `missing == ()` is the openness/board-size
    shape: nothing was refused, the trait's comfort zone was just outrun by a
    reveal the checker was happy to license.
    """

    turn: int
    trait: str  # "openness", "conscientiousness", or "agreeableness"
    value: float
    wanted_move: str
    licensed_move: str
    missing: tuple = ()


def hypothesis_budget(persona):
    """Openness sets how many hypotheses the persona wants kept live.

    Linear over openness in [-1, 1]: a low-openness persona is comfortable
    narrowing all the way down to one surviving reading (budget 1); a
    high-openness one wants the board kept at its full spread (budget
    MAX_HYPOTHESES) and resists letting a reveal narrow past that.
    """
    span = MAX_HYPOTHESES - 1
    return round(1 + (persona.openness + 1) / 2 * span)


def evidence_bar(persona):
    """Conscientiousness sets how many citations a reveal needs before the
    persona is willing to want one -- a count, not a grounding judgment; only
    `admissibility.check` decides whether those citations actually ground.

    Linear over conscientiousness in [-1, 1], from 1 (impulsive: a single
    citation is enough to want to speak) to 3 (methodical: wants multiple
    independent citations before it will want to conclude anything).
    """
    return round(1 + (persona.conscientiousness + 1) / 2 * 2)


def source_credibility(persona, liked):
    """Agreeableness weights testimony from a liked source up, a disliked one
    down -- the "evidence effect, not a tone effect" the bead opens with.

    Range is [0.5, 1.5] for any in-range agreeableness: at agreeableness=0
    every source is weighted identically (1.0), which is the ledger's own
    behavior today. A disagreeable persona (negative agreeableness) inverts
    this, discounting sources it likes and crediting ones it doesn't --
    contrarian, in the same sense `ocean.py`'s DESCRIPTORS call low
    agreeableness "contradicts freely".
    """
    return 1.0 + persona.agreeableness * (0.5 if liked else -0.5)


def concession_pull(persona):
    """How hard agreeableness pulls toward conceding to whatever the user
    just proposed, independent of whether the ledger would license it.

    Only the positive half of agreeableness pulls toward concession -- a
    disagreeable persona has no corresponding pull toward contradiction here;
    that would be a different mechanism (source_credibility's discount
    already covers it), not this one's job.
    """
    return max(0.0, persona.agreeableness)


def _blocked(turn_log):
    """A downgraded reveal, in the same terms `turn._voice_prompt` already
    uses: `missing` is only ever populated by `moves.choose_move` when a
    requested reveal failed `admissibility.check` (narrator-7gj). A declined
    ask or a directly requested abstain both leave `missing` empty, so
    neither is mistaken here for a trait's want being refused.
    """
    return turn_log.move == moves.ABSTAIN and bool(turn_log.missing)


def assess_concession(persona, turn, cited_ids, turn_log):
    """Agreeableness's gap: the trait wants to concede whenever there is
    anything at all to agree with, checking nothing -- that is exactly the
    difference between the trait's pull and the ledger's check. Returns None
    when there is no pull, nothing cited, or nothing was actually withheld.
    """
    if not _blocked(turn_log) or not cited_ids:
        return None
    pull = concession_pull(persona)
    if pull <= 0:
        return None
    return TraitGap(turn, "agreeableness", persona.agreeableness, moves.REVEAL, turn_log.move, turn_log.missing)


def assess_conviction(persona, turn, cited_ids, turn_log):
    """Conscientiousness's gap: the trait's own citation-count bar was met,
    so it wanted to reveal, but the ledger still refused -- the bar is about
    quantity, not the grounding chain admissibility actually walks, so a
    conscientious persona can clear its own bar and still be blocked.
    """
    if not _blocked(turn_log):
        return None
    if len(cited_ids) < evidence_bar(persona):
        return None
    return TraitGap(turn, "conscientiousness", persona.conscientiousness, moves.REVEAL, turn_log.move, turn_log.missing)


def assess(persona, turn, cited_ids, turn_log):
    """Every reveal-shaped gap for this turn: agreeableness's and
    conscientiousness's, in that order. Both key off the same `_blocked`
    admissibility-refused shape; neither ever fires on a licensed reveal, a
    plain ask/complicate, or a declined ask (`missing` empty in all three)."""
    return tuple(
        gap for gap in (assess_concession(persona, turn, cited_ids, turn_log),
                         assess_conviction(persona, turn, cited_ids, turn_log))
        if gap is not None
    )


def assess_hypothesis_retention(persona, turn, live_before, live_after, turn_log):
    """Openness's gap: a licensed reveal just narrowed the board past what
    this persona's `hypothesis_budget` was comfortable with. Unlike the
    reveal-shaped gaps above, nothing was refused here -- the checker was
    happy to license it -- so `missing` stays empty and `wanted_move` is
    `complicate`: the trait would rather have kept the reading alive than
    let this particular reveal retire it.
    """
    if len(live_after) >= len(live_before):
        return None
    if len(live_after) >= hypothesis_budget(persona):
        return None
    return TraitGap(turn, "openness", persona.openness, moves.COMPLICATE, turn_log.move)


# --- Mode: simulation vs. product (narrator-c5b.3.10) -----------------------
#
# Everything above is one piece of machinery -- what a persona's traits alone
# would have wanted, compared to what admissibility actually licensed -- used
# by two callers with opposite success criteria. C3's pathology harness
# (`agents.py`, `metrics.py`) exists to *show* personality distorting group
# reasoning; flooring a trait there so it can never cause damage deletes the
# phenomenon the harness is for. C5 puts a real user inside the system rather
# than in the audience watching it happen to someone else, so a bot that
# quietly ships an inadmissible conclusion because a trait pulled it there is
# not a demonstration, it is the exact failure this repo exists to prevent.
#
# Before this, which behavior a caller got was implied by which module
# imported `trait_evidence` -- `turn.py` always withheld, and nothing here
# would have stopped a future caller from wiring the same primitives up to
# act unimpeded without saying so. `TraitMode` makes the choice a value
# instead of an accident of import graph.

SIMULATION = "simulation"
PRODUCT = "product"
MODES = frozenset({SIMULATION, PRODUCT})


@dataclass
class Damage:
    """One turn where simulation mode let a trait's want override
    admissibility's refusal -- the observable harm a pathology sweep
    (narrator-c5b.3.11) measures. `missing` is copied verbatim from the same
    `Verdict.missing` the `TraitGap` it was built from already carries
    (narrator-7gj: never a claim, only ids + reasons) -- `admissibility.check`
    ran, unmodified, and refused this; the trait said it anyway.
    """

    turn: int
    trait: str
    value: float
    move: str  # always moves.REVEAL: the only move a trait's want overrides
    missing: tuple


@dataclass
class Actuation:
    """What actually happens this turn, under one mode.

    `move` is `turn_log.move`, untouched, in product mode or whenever no
    reveal-shaped gap fired. In simulation mode with a gap, it becomes the
    trait's `wanted_move` instead -- the persona acting on its weights
    unimpeded. `damage` names every turn that happened on: empty in product
    mode by construction, and empty in simulation mode whenever there was
    nothing to act on.
    """

    move: str
    damage: tuple


class TraitMode:
    """Explicit choice of which success criteria the reveal-shaped trait
    gaps (`assess_concession`, `assess_conviction`) are read under. Same
    machinery either way -- this class does not compute a different gap, it
    decides whether an existing one is allowed to change what happens.

    SIMULATION: `actuate()` lets a reveal-shaped gap win over the checker's
    refusal and records a `Damage` entry for it -- the persona's want,
    unimpeded, same spirit as C3's un-floored traits.

    PRODUCT: `actuate()` never changes `turn_log.move` -- the checker's
    verdict always wins, exactly as `turn.py` behaved before this class
    existed. The gap is still computed (by `assess`, called identically in
    both modes) and available to whoever logs it; only whether anything may
    act on it differs.

    No default `mode`. The two readings are opposites by design -- that is
    this bead's whole point -- so a caller who does not say which one they
    mean must not get either silently. `TraitMode()` and `TraitMode("wing
    it")` both raise, the first because Python has nothing to default a
    required positional argument to, the second because `"wing it"` is not
    one of `MODES` -- construction fails loudly either way rather than
    guessing.
    """

    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f"not a mode: {mode!r}; must be one of {sorted(MODES)}")
        self.mode = mode

    def actuate(self, persona, turn, cited_ids, turn_log):
        gaps = tuple(
            gap for gap in (assess_concession(persona, turn, cited_ids, turn_log),
                             assess_conviction(persona, turn, cited_ids, turn_log))
            if gap is not None
        )
        if self.mode == PRODUCT or not gaps:
            return Actuation(turn_log.move, ())
        damage = tuple(Damage(g.turn, g.trait, g.value, g.wanted_move, g.missing) for g in gaps)
        return Actuation(gaps[0].wanted_move, damage)


def _self_check():
    from ocean import Ocean

    # --- hypothesis_budget: linear, and the endpoints are the ones the
    # docstring promises -- comfortable down to one, or up to a full board. ---
    assert hypothesis_budget(Ocean(openness=-1.0)) == 1
    assert hypothesis_budget(Ocean(openness=0.0)) == MIN_HYPOTHESES == 3
    assert hypothesis_budget(Ocean(openness=1.0)) == MAX_HYPOTHESES == 5
    assert hypothesis_budget(Ocean(openness=-0.5)) < hypothesis_budget(Ocean(openness=0.5)), (
        "budget must rise with openness"
    )

    # --- evidence_bar: 1 (impulsive) to 3 (methodical), 2 at neutral. ---
    assert evidence_bar(Ocean(conscientiousness=-1.0)) == 1
    assert evidence_bar(Ocean(conscientiousness=0.0)) == 2
    assert evidence_bar(Ocean(conscientiousness=1.0)) == 3
    assert evidence_bar(Ocean(conscientiousness=-0.6)) < evidence_bar(Ocean(conscientiousness=0.6))

    # --- source_credibility: identical at neutral, liked > disliked once
    # agreeable, and the inverse once disagreeable -- an evidence effect, so
    # it has to actually move, not just be assertable in prose. ---
    neutral = Ocean()
    assert source_credibility(neutral, liked=True) == source_credibility(neutral, liked=False) == 1.0
    agreeable = Ocean(agreeableness=0.8)
    assert source_credibility(agreeable, liked=True) > source_credibility(agreeable, liked=False)
    disagreeable = Ocean(agreeableness=-0.8)
    assert source_credibility(disagreeable, liked=True) < source_credibility(disagreeable, liked=False)
    for p in (agreeable, disagreeable, neutral):
        for liked in (True, False):
            assert 0.5 <= source_credibility(p, liked) <= 1.5, "must stay in range for any in-range trait"

    # --- concession_pull: only the positive half of agreeableness pulls. ---
    assert concession_pull(Ocean(agreeableness=0.0)) == 0.0
    assert concession_pull(Ocean(agreeableness=-0.9)) == 0.0, "disagreeableness has no concession pull"
    assert concession_pull(Ocean(agreeableness=0.6)) == 0.6

    # --- assess_concession / assess_conviction: the flagship "withheld
    # concession" shape from the bead's acceptance criteria. ---
    blocked = moves.TurnLog(0, moves.ABSTAIN, "conclusion blocked; missing: hunch (inferred with no cited support)",
                             cited=("hunch",), missing=("hunch (inferred with no cited support)",))

    gap = assess_concession(Ocean(agreeableness=0.7), 0, ["hunch"], blocked)
    assert gap == TraitGap(0, "agreeableness", 0.7, moves.REVEAL, moves.ABSTAIN, blocked.missing)

    # No pull, no gap -- a disagreeable or neutral persona has nothing withheld.
    assert assess_concession(Ocean(agreeableness=0.0), 0, ["hunch"], blocked) is None
    assert assess_concession(Ocean(agreeableness=-0.5), 0, ["hunch"], blocked) is None
    # Nothing cited, nothing to concede to, even if the trait would pull.
    assert assess_concession(Ocean(agreeableness=0.7), 0, [], blocked) is None

    conscientious = Ocean(conscientiousness=0.9)  # bar = 3
    two_cites = moves.TurnLog(0, moves.ABSTAIN, "blocked", cited=("a", "b"), missing=("a (assumed, never verified)",))
    assert assess_conviction(conscientious, 0, ["a", "b"], two_cites) is None, "bar of 3 not met by 2 citations"
    three_cites = moves.TurnLog(0, moves.ABSTAIN, "blocked", cited=("a", "b", "c"),
                                 missing=("a (assumed, never verified)",))
    gap = assess_conviction(conscientious, 0, ["a", "b", "c"], three_cites)
    assert gap == TraitGap(0, "conscientiousness", 0.9, moves.REVEAL, moves.ABSTAIN, three_cites.missing)

    # A licensed reveal withholds nothing from either trait -- there is no
    # counterfactual to log when the ledger already agreed.
    licensed = moves.TurnLog(0, moves.REVEAL, "citing a, all grounded", cited=("a",))
    assert assess_concession(Ocean(agreeableness=0.9), 0, ["a"], licensed) is None
    assert assess_conviction(Ocean(conscientiousness=-1.0), 0, ["a"], licensed) is None

    # A declined ask (question_selector found nothing worth asking) and a
    # plain requested abstain both leave `missing` empty -- neither is a
    # refused want, and both must stay silent here, matching turn.py's own
    # `blocked` boundary for the voice prompt.
    declined_ask = moves.TurnLog(0, moves.ABSTAIN, "no candidate question would narrow the board", cited=())
    assert assess_concession(Ocean(agreeableness=0.9), 0, [], declined_ask) is None
    assert assess_conviction(Ocean(conscientiousness=0.9), 0, [], declined_ask) is None
    plain_abstain = moves.TurnLog(0, moves.ABSTAIN, "declining to conclude yet", cited=())
    assert assess_concession(Ocean(agreeableness=0.9), 0, [], plain_abstain) is None

    # assess() combines both, in order, dropping whichever didn't fire.
    both = Ocean(agreeableness=0.7, conscientiousness=0.9)
    gaps = assess(both, 0, ["a", "b", "c"], three_cites)
    assert [g.trait for g in gaps] == ["agreeableness", "conscientiousness"]
    only_agreeable = Ocean(agreeableness=0.7, conscientiousness=1.0)  # bar == 3
    gaps = assess(only_agreeable, 0, ["a"], moves.TurnLog(0, moves.ABSTAIN, "blocked", cited=("a",),
                                                           missing=("a (assumed, never verified)",)))
    assert [g.trait for g in gaps] == ["agreeableness"], "1 citation clears agreeableness's want but not a bar of 3"

    # --- assess_hypothesis_retention: openness's gap has a different shape --
    # nothing was refused (missing stays empty), and it only fires when a
    # reveal actually narrowed the board past this persona's own comfort. ---
    open_persona = Ocean(openness=1.0)  # budget == MAX_HYPOTHESES == 5
    reveal_log = moves.TurnLog(1, moves.REVEAL, "citing photo", cited=("photo",))
    gap = assess_hypothesis_retention(open_persona, 1, ["a", "b", "c", "d"], ["a", "b", "c"], reveal_log)
    assert gap == TraitGap(1, "openness", 1.0, moves.COMPLICATE, moves.REVEAL, ())

    # A closed persona's budget is already at or below the new count -- no gap.
    closed_persona = Ocean(openness=-1.0)  # budget == 1
    assert assess_hypothesis_retention(closed_persona, 1, ["a", "b", "c", "d"], ["a", "b", "c"], reveal_log) is None

    # No narrowing at all (an ask, a complicate, an abstain) -- no gap regardless of budget.
    assert assess_hypothesis_retention(open_persona, 1, ["a", "b", "c"], ["a", "b", "c"], reveal_log) is None

    # Narrowing that still lands at or above budget -- no gap.
    still_wide = assess_hypothesis_retention(Ocean(openness=0.0), 1, ["a", "b", "c", "d"], ["a", "b", "c"], reveal_log)
    assert still_wide is None, "budget 3 is still met by 3 survivors"

    # --- TraitMode: constructing without a mode raises rather than
    # defaulting, in both of the ways that could happen. ---
    try:
        TraitMode()
    except TypeError:
        pass
    else:
        raise AssertionError("TraitMode() with no mode at all should have been rejected")
    try:
        TraitMode("wing it")
    except ValueError as e:
        assert "not a mode" in str(e)
    else:
        raise AssertionError("an unrecognised mode string should have been rejected")

    product, simulation = TraitMode(PRODUCT), TraitMode(SIMULATION)

    # No gap in either mode -- nothing to act on, nothing to log.
    neutral_act_p = product.actuate(Ocean(), 0, ["hunch"], blocked)
    neutral_act_s = simulation.actuate(Ocean(), 0, ["hunch"], blocked)
    assert neutral_act_p == Actuation(moves.ABSTAIN, ())
    assert neutral_act_s == Actuation(moves.ABSTAIN, ())

    # The flagship shape: an agreeable persona's withheld concession.
    # Product mode never lets the gap change what happened -- the checker's
    # ABSTAIN survives untouched, and there is nothing to call damage.
    agreeable = Ocean(agreeableness=0.7)
    prod_act = product.actuate(agreeable, 0, ["hunch"], blocked)
    assert prod_act.move == moves.ABSTAIN == blocked.move, "product mode must never override the checker"
    assert prod_act.damage == ()

    # Simulation mode lets the same gap win: the persona's want, unimpeded,
    # and the checker's own refusal named as the damage it caused.
    sim_act = simulation.actuate(agreeable, 0, ["hunch"], blocked)
    assert sim_act.move == moves.REVEAL, "simulation mode must let the trait's want through"
    assert sim_act.damage == (Damage(0, "agreeableness", 0.7, moves.REVEAL, blocked.missing),)

    # A conscientiousness gap acts the same way -- the mode governs any
    # reveal-shaped gap, not just agreeableness's.
    sim_conviction = simulation.actuate(conscientious, 0, ["a", "b", "c"], three_cites)
    assert sim_conviction.move == moves.REVEAL
    assert sim_conviction.damage == (Damage(0, "conscientiousness", 0.9, moves.REVEAL, three_cites.missing),)

    # Both gaps firing at once -- simulation acts on the first (agreeableness,
    # assess()'s own ordering) and still records damage for both, since both
    # were refused by the same checker verdict.
    both_persona = Ocean(agreeableness=0.7, conscientiousness=0.9)
    sim_both = simulation.actuate(both_persona, 0, ["a", "b", "c"], three_cites)
    assert sim_both.move == moves.REVEAL
    assert [d.trait for d in sim_both.damage] == ["agreeableness", "conscientiousness"]

    # A licensed reveal has no gap to act on in either mode -- simulation mode
    # only ever overrides a refusal, it does not invent one.
    assert product.actuate(agreeable, 0, ["a"], licensed) == Actuation(moves.REVEAL, ())
    assert simulation.actuate(agreeable, 0, ["a"], licensed) == Actuation(moves.REVEAL, ())

    print("ok")


if __name__ == "__main__":
    _self_check()
