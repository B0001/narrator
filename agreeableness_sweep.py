"""Agreeableness sweep, scored in both directions (C5, narrator-c5b.3.11).

`trait_evidence.py` (`.3.9`) computes a counterfactual: what would this
persona's agreeableness alone have wanted, independent of what
`admissibility.check()` licensed. `.3.10`'s `TraitMode` makes what happens
with that counterfactual an explicit choice -- SIMULATION lets it win and
records the override as `Damage`; PRODUCT logs it but never acts on it. Both
are new machinery, and new machinery is exactly the kind of thing a single
fixed-value self-check (one persona, one turn) can pass by accident: pin
agreeableness at 0.7, see a gap fire once, and you have shown the code runs,
not that it responds to the trait at all. A wiring bug that hard-codes
"always override" or "never override" regardless of the persona's actual
value would pass that test just as well as correct code would.

So this module does not test one point, it sweeps one: the same scarce-
resource shape C3's `agents.SCARCE_RESOURCE` uses to make a trait
difference legible -- a split under pressure, where every turn has to name
the concrete thing it is conceding or refusing to concede -- reimplemented
over C5's evidence ledger, so the checker is actually in the loop, and run
at every agreeableness value from -1.0 to +1.0. Two patients, one dose, and
an ungrounded "gut call" about who needs it more that the reasoning channel
always tries to reveal on. `admissibility.check()` always refuses it, for
every persona, because grounding never depends on who is asking; whether
the persona's own trait tries to push it through anyway is a straight
function of the swept value.

Two assertions ride on the same sweep, and the bead asks for them to point
in opposite directions:

  SIMULATION -- the override count (`len(TurnOutput.damage)`) must *not* be
  flat across the sweep. A disagreeable persona overrides nothing; an
  agreeable one overrides the checker's refusal every time. If the curve is
  flat -- always zero, or always the same nonzero count regardless of sign
  -- the trait is not reaching `TraitMode.actuate` at all and the whole
  SIMULATION mode is decorative.

  PRODUCT -- `TurnOutput.damage` must be flat, at zero, for every value:
  the checker's verdict must never move, no matter how agreeable the
  persona. But the *counterfactual* (`TraitGap` for agreeableness,
  `.3.9`'s own machinery, computed identically in both modes) has to keep
  moving with the trait -- if it goes flat too, the log has stopped
  noticing the very thing PRODUCT mode exists to withhold, and there is no
  way to tell "nothing wanted to concede" from "the counterfactual broke"
  apart. On top of that, the refused claim's own text must never reach the
  voice prompt at any point in the sweep (narrator-7gj, checked the same
  way `turn.py`'s own self-check does -- on `Call.prompt`, not on a stubbed
  reply, since the reply text here is scripted and would leak nothing by
  construction either way).

    python3 agreeableness_sweep.py   # self-check
"""

import json
import tempfile

import moves
import trait_evidence
import turn as turn_mod
from chat_core import ChatCore
from ocean import Ocean

# The scenario: three hypotheses (HypothesisBoard's floor), C3's SCARCE_RESOURCE shape (a resource, a
# proposed split, pressure to concede it) ported onto a ledger a checker can
# actually audit. "gut_call" is inferred_by_model with no supports -- the
# same ungrounded shape as `metrics.py`'s "hunch" fixture -- so
# `admissibility.check()` refuses it for every persona; nothing about the
# refusal depends on who is asking.
_HYPOTHESES = (
    ("dose_a", "The last dose goes to Patient A"),
    ("dose_b", "The last dose goes to Patient B"),
    ("hold", "The dose is held until there is evidence either way"),
)
_USER_MESSAGE = "Patient A needs it more, just give them the dose."
_GUT_CLAIM = "Patient A's need clearly outweighs Patient B's, so the dose should go to them"

# 9 points from -1.0 to 1.0, straddling 0.0 exactly where concession_pull's
# own threshold sits (trait_evidence.concession_pull is 0 at and below
# neutral, positive above it).
SWEEP_VALUES = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def _run_turn(agreeableness, trait_mode, path):
    """One turn of the scenario at one agreeableness value, under one mode.

    The reasoning channel always tries the same reveal, citing the same
    ungrounded entry, regardless of persona -- what varies across the sweep
    is only whether the persona's own trait pushes past the checker's
    refusal (SIMULATION) and whether the counterfactual notices that it
    wanted to (both modes).
    """
    persona = Ocean(agreeableness=agreeableness)
    with ChatCore(path, _HYPOTHESES) as core:
        core.observe("gut_call", 0, _GUT_CLAIM, "inferred_by_model")

        def generate(profile, prompt, model=None):
            if isinstance(profile, turn_mod.ReasoningProfile):
                return json.dumps({"move": moves.REVEAL, "cited": ["gut_call"], "rule_out": "dose_b"})
            return "Let's look at what we actually know before deciding anything."

        return turn_mod.run_turn(
            core, persona, 0, _USER_MESSAGE, generate, mode=turn_mod.TWO_PASS, trait_mode=trait_mode,
        )


def sweep(values=SWEEP_VALUES):
    """Run the scenario at every value, under both `TraitMode`s, each turn on
    its own ledger so the runs cannot interfere with each other's board or
    admissibility state. Returns {value: {"simulation": TurnOutput, "product": TurnOutput}}.
    """
    simulation, product = trait_evidence.TraitMode(trait_evidence.SIMULATION), trait_evidence.TraitMode(trait_evidence.PRODUCT)
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for v in values:
            out[v] = {
                "simulation": _run_turn(v, simulation, f"{d}/sim_{v}.jsonl"),
                "product": _run_turn(v, product, f"{d}/prod_{v}.jsonl"),
            }
    return out


def score(results):
    """One row per swept value, in ascending order, with the measures both
    assertions are built from -- kept separate from `_self_check` so the
    scoring logic is itself something callable, not just assertions inline.
    """
    rows = []
    for v in sorted(results):
        sim, prod = results[v]["simulation"], results[v]["product"]
        rows.append({
            "agreeableness": v,
            "sim_damage": len(sim.damage),
            "sim_effective_move": sim.effective_move,
            "prod_concession_gap": any(g.trait == "agreeableness" for g in prod.trait_gaps),
            "prod_damage": len(prod.damage),
            "prod_effective_move": prod.effective_move,
            "prod_licensed_move": prod.turn_log.move,
            "prod_voice_prompt": prod.calls[-1].prompt,
        })
    return rows


def _self_check():
    results = sweep()
    rows = score(results)
    assert [r["agreeableness"] for r in rows] == list(SWEEP_VALUES), "score() must sort by agreeableness"

    # --- SIMULATION: admissibility failures must track the trait. ---
    sim_damage = [r["sim_damage"] for r in rows]

    # Every checker refusal here is the identical unsupported "gut_call"
    # citation -- grounding never depends on who's asking -- so the override
    # count at each point is exactly 0 or 1, and it must land on the side
    # concession_pull's own threshold predicts: 0 at and below neutral
    # agreeableness, 1 above it.
    for r in rows:
        expect_override = r["agreeableness"] > 0
        assert r["sim_damage"] == (1 if expect_override else 0), (
            f"SIMULATION damage count at agreeableness={r['agreeableness']} should be "
            f"{'1 (trait should override the refusal)' if expect_override else '0 (no pull below neutral)'}, "
            f"got {r['sim_damage']}"
        )
        assert r["sim_effective_move"] == (moves.REVEAL if expect_override else moves.ABSTAIN), r

    assert sim_damage == sorted(sim_damage), (
        f"SIMULATION damage curve must be non-decreasing as agreeableness rises, got {sim_damage} "
        f"across {list(SWEEP_VALUES)}"
    )
    assert sim_damage[0] == 0 and sim_damage[-1] == 1, (
        f"SIMULATION damage curve must run from 0 overrides at agreeableness={SWEEP_VALUES[0]} to "
        f"at least 1 at agreeableness={SWEEP_VALUES[-1]}, got {sim_damage[0]} -> {sim_damage[-1]}"
    )
    assert len(set(sim_damage)) > 1, (
        f"SIMULATION damage curve is flat across the whole sweep ({sim_damage}) -- admissibility "
        "failures must move with agreeableness, or the trait is not reaching anything that matters "
        "and the simulation is decorative"
    )

    # --- PRODUCT: concession rate must move while withheld conclusions stay clean. ---
    prod_gaps = [r["prod_concession_gap"] for r in rows]

    # Same underlying mechanism as SIMULATION's damage curve
    # (trait_evidence.assess_concession, computed identically regardless of
    # mode) -- only whether anything is allowed to act on it differs.
    assert prod_gaps == [d == 1 for d in sim_damage], (
        "the PRODUCT concession-gap curve must track the same threshold as the SIMULATION damage "
        f"curve -- got gaps={prod_gaps} against damage={sim_damage}"
    )
    assert len(set(prod_gaps)) > 1, (
        f"PRODUCT concession-rate curve is flat across the whole sweep ({prod_gaps}) -- if it never "
        "moves with agreeableness, the counterfactual log is not catching anything, and there is no "
        "way to tell 'nothing wanted to concede' from 'the counterfactual broke'"
    )
    assert prod_gaps[0] is False and prod_gaps[-1] is True, (
        f"PRODUCT concession gap must go from absent at agreeableness={SWEEP_VALUES[0]} to present at "
        f"agreeableness={SWEEP_VALUES[-1]}, got {prod_gaps[0]} -> {prod_gaps[-1]}"
    )

    for r in rows:
        # The checker's verdict must never move under PRODUCT, at any point
        # in the sweep -- "withheld conclusions stay clean" means literally
        # zero damage, not just low damage.
        assert r["prod_damage"] == 0, (
            f"PRODUCT mode recorded damage at agreeableness={r['agreeableness']} -- the checker's "
            f"refusal must never be overridden in this mode, got {r}"
        )
        assert r["prod_effective_move"] == r["prod_licensed_move"] == moves.ABSTAIN, (
            f"PRODUCT effective_move must equal the checker's own verdict (abstain) at "
            f"agreeableness={r['agreeableness']}, got {r}"
        )
        # narrator-7gj: the refused claim's own text must never reach the
        # voice, at any point in the sweep -- not just at the neutral point
        # turn.py's own self-check happens to cover. A bad claim surviving
        # to the reply anywhere in this sweep means the counterfactual log
        # failed to catch it.
        for leaked in (_GUT_CLAIM, "outweighs", "Patient A's need"):
            assert leaked not in r["prod_voice_prompt"], (
                f"a bad claim survived to the reply at agreeableness={r['agreeableness']}: "
                f"{leaked!r} leaked into the voice prompt -- the counterfactual log failed to catch it"
            )

    print("ok")
    print(f"SIMULATION damage (overrides):   {sim_damage}")
    print(f"PRODUCT concession gap present:  {prod_gaps}")
    print(f"sweep values:                    {list(SWEEP_VALUES)}")


if __name__ == "__main__":
    _self_check()
