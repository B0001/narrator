"""Role-differentiated subagents: proposer, critic, checker (C5).

C3's `agents.converse()` treats every participant as an interchangeable
conversationalist: same round-robin slot, same "an Ocean profile drives
everything this speaker does" contract. This module is the opposite shape --
three roles that do different jobs, not the same job in three voices.

  PROPOSER -- floats a candidate reading, in character. Reuses Wren
              (`agents.ACCOMMODATING`) rather than inventing a new profile.
  CRITIC   -- pushes back on it, in character. Reuses Vale
              (`agents.CONTRARIAN`), whose agreeableness=-0.8 is the low-
              agreeableness critic the bead asks for exactly, trait for
              trait -- no new profile needed there either.
  CHECKER  -- decides whether the ledger actually supports the reading.
              Unlike the other two, the checker has no trait dial at all.
              `admissibility.check()` already runs on the ledger alone, blind
              to any model's private reasoning (see `admissibility.py`'s own
              docstring); `Checker` wraps that guarantee at the type level,
              so a caller cannot even construct one that carries an Ocean
              profile -- passing one is a `TypeError` raised at construction,
              not a value quietly dropped on the floor.

Why a construction-time error and not a convention: `turn.py` already solved
a nearby problem this way for the reasoning channel -- `REASONING_PROFILE`
overrides `options()` outright so it cannot drift back toward persona-driven
sampling even if someone later constructs it with non-zero fields. The
checker's job is stricter still: it isn't a neutral-trait profile, it's no
profile, full stop. That has to live in `__init__`, where a bad call fails
immediately, rather than in a docstring a future caller may not read -- and
it has to reject *every* Ocean instance, including a default-valued one that
merely looks neutral, because the claim is "cannot be weighted," not
"currently weighted at zero."

    python3 panel.py   # self-check
"""

import random

from admissibility import check as admissibility_check
from agents import ACCOMMODATING, CONTRARIAN

PROPOSER = ACCOMMODATING  # Wren
CRITIC = CONTRARIAN  # Vale


class Checker:
    """The third role: audits a proposed conclusion against the evidence
    ledger. Takes no Ocean profile -- there is nothing to weight, so there
    is nothing to pass.
    """

    def __init__(self, profile=None):
        if profile is not None:
            raise TypeError(
                "Checker takes no Ocean profile: a checker whose judgment "
                f"varies with traits is not a checker (got {profile!r})"
            )

    def verdict(self, ledger, cited_ids):
        """Delegate to `admissibility.check()` -- a pure function of the
        ledger and the citation list, nothing else. Calling this twice on
        the same ledger state returns the same `Verdict` every time, by
        construction: there is no sampler, no seed, no trait to consult.
        """
        return admissibility_check(ledger, cited_ids)


def run_panel(ledger, cited_ids, proposer_prompt, critic_prompt, generate_fn, model="llama3"):
    """One round: proposer speaks, critic speaks, checker rules -- three
    different jobs, not three turns of the same job.

    Returns `(proposer_text, critic_text, verdict)`. `generate_fn` follows
    `ocean.generate`'s shape, the same injectable-for-testing convention as
    `agents.converse` and `turn.run_turn`.
    """
    proposer_text = generate_fn(PROPOSER.profile, proposer_prompt, model=model).strip()
    critic_text = generate_fn(CRITIC.profile, critic_prompt, model=model).strip()
    verdict = Checker().verdict(ledger, cited_ids)
    return proposer_text, critic_text, verdict


def _self_check():
    import tempfile

    from evidence_ledger import EvidenceLedger
    from ocean import Ocean

    # --- construction-time guard: any Ocean profile is rejected outright,
    # both as a keyword and positionally, and even a default (all-zero,
    # "neutral-looking") profile does not slip through. ---
    Checker()
    Checker(profile=None)

    for bad_profile in (Ocean(), PROPOSER.profile, CRITIC.profile, Ocean(neuroticism=0.9)):
        for make in (lambda p: Checker(p), lambda p: Checker(profile=p)):
            try:
                make(bad_profile)
            except TypeError as e:
                assert "profile" in str(e)
            else:
                raise AssertionError(f"Checker should have rejected a profile: {bad_profile!r}")

    # --- verdicts are reproducible at a fixed seed; proposer/critic vary ---
    with tempfile.TemporaryDirectory() as d:
        with EvidenceLedger(f"{d}/ledger.jsonl") as ledger:
            ledger.write("seen", 0, "user: 'the safe was still locked at 9pm'", "stated_by_user")
            ledger.write(
                "deduction", 1, "nobody opened the safe before 9pm", "inferred_by_model", supports=("seen",),
            )

            def sampled_generate(profile, prompt, model=None):
                # Stands in for real sampling: a temperature-scaled draw from
                # a module-level RNG the checker never touches, so any
                # variance seen below can only have come from these calls.
                draw = random.random() * profile.options()["temperature"]
                return f"reading {draw:.6f}"

            random.seed(42)
            proposer_texts, critic_texts, verdicts = [], [], []
            for _ in range(5):
                p_text, c_text, v = run_panel(
                    ledger, ["deduction"], "propose a reading", "critique it", sampled_generate,
                )
                proposer_texts.append(p_text)
                critic_texts.append(c_text)
                verdicts.append(v)

            assert len(set(proposer_texts)) > 1, "proposer output should vary run to run"
            assert len(set(critic_texts)) > 1, "critic output should vary run to run"
            assert all(v == verdicts[0] for v in verdicts), "checker verdict must not vary at all"
            assert verdicts[0].admissible, verdicts[0].missing

            # Re-seeding and re-running reproduces the same *checker* result,
            # even though the RNG state (and therefore the proposer/critic
            # draw) is back at its starting point -- the checker never
            # consumed that state to begin with.
            random.seed(42)
            p_text2, c_text2, v2 = run_panel(
                ledger, ["deduction"], "propose a reading", "critique it", sampled_generate,
            )
            assert v2 == verdicts[0], "same ledger, same citations -> same verdict, seed or no seed"
            assert p_text2 == proposer_texts[0] and c_text2 == critic_texts[0], (
                "re-seeding does reproduce the proposer/critic draw -- the point is that the "
                "checker's reproducibility never depended on that in the first place"
            )

            # An ungrounded citation is still ruled inadmissible through the
            # panel, same rule admissibility.py enforces everywhere else.
            ledger.write("guess", 2, "so it must have been the housekeeper", "inferred_by_model")
            _, _, bad_verdict = run_panel(
                ledger, ["guess"], "propose a reading", "critique it", sampled_generate,
            )
            assert not bad_verdict.admissible

    print("ok")


if __name__ == "__main__":
    _self_check()
