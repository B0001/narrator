"""Runtime admissibility check, inverted from mystery.validate_solvability() (C5).

`EpistemicClueGraph.validate_solvability()` asked: given the clues, does
exactly one suspect survive? It ran on the clue graph alone, blind to
`WorldSimulation`'s ground truth, because a checker that could see the
answer would just agree with it.

Here the direction reverses. The bot is the one proposing a conclusion; the
user holds ground truth. So the question becomes: given the ledger, is this
conclusion something the user has actually been shown grounds for -- or is
it resting, however many inference steps down, on something nobody
observed? This module only ever reads `EvidenceLedger` entries and the
citation list a proposed conclusion offers. It has no way to ask the model
what it "really meant"; if the ledger doesn't show the chain, the check has
nothing else to go on, same as `suspects_consistent_with_clues()` never gets
to peek at `sim.culprit`.

Grounding is recursive, mirroring `traces_to_murder()`'s "every event must
connect" but for evidence instead of causation:

  - a DIRECT entry (stated_by_user / observed_artifact) is grounded on its
    own -- it *is* something the user has seen.
  - an `inferred_by_model` entry is grounded only if it cites at least one
    supporting entry and every one of those is itself grounded.
  - an `assumed` entry is never grounded, no matter what it cites. An
    assumption is by definition a premise nobody verified; letting it
    launder itself through a support chain would make "assumed" meaningless.

    python3 admissibility.py   # self-check
"""

from dataclasses import dataclass

from evidence_ledger import DIRECT


@dataclass
class Verdict:
    admissible: bool
    missing: tuple  # ids (or descriptions) of the evidence that would be needed


def _grounded(ledger, entry_id, trail, missing):
    """Is this single entry grounded in something the user has seen?

    `trail` guards against a cycle in hand-authored `supports` data turning
    this into infinite recursion; a cycle is treated as ungrounded rather
    than crashing, since a chain that only ever points at itself never
    reaches the user either way.

    `missing` names entry *ids* and why each failed, never the claim text
    (narrator-7gj). The ledger already holds the claim under that id, so
    copying it in here adds nothing an auditor cannot look up -- but it does
    put the exact sentence this checker just refused into a string that
    downstream code renders. It reached the persona voice prompt that way
    once already. A checker's output travels further than its author expects;
    the safe thing for it to carry is a pointer, not the refused content.
    """
    if entry_id in trail:
        missing.append(f"{entry_id} (circular support chain)")
        return False
    if entry_id not in ledger:
        missing.append(f"{entry_id} (not in ledger)")
        return False

    entry = ledger.get(entry_id)
    if entry.provenance in DIRECT:
        return True
    if entry.provenance == "assumed":
        missing.append(f"{entry_id} (assumed, never verified)")
        return False

    # inferred_by_model: grounded only if every one of its own supports is.
    if not entry.supports:
        missing.append(f"{entry_id} (inferred with no cited support)")
        return False
    ok = True
    for support_id in entry.supports:
        if not _grounded(ledger, support_id, trail | {entry_id}, missing):
            ok = False
    return ok


def check(ledger, cited_ids):
    """Is a conclusion citing `cited_ids` admissible?

    Runs on the ledger alone -- no access to whatever the model's private
    reasoning "really" concluded, only to what it wrote down and tagged.
    A conclusion resting on an inferred-only chain that bottoms out in
    nothing the user has seen is blocked, and the missing evidence is named
    so the bot's next move can go ask for it instead of asserting anyway.
    """
    if not cited_ids:
        return Verdict(False, ("no evidence cited",))
    missing = []
    all_grounded = True
    for cid in cited_ids:
        if not _grounded(ledger, cid, frozenset(), missing):
            all_grounded = False
    return Verdict(all_grounded, tuple(missing))


def _self_check():
    import tempfile

    from evidence_ledger import EvidenceLedger

    with tempfile.TemporaryDirectory() as d:
        with EvidenceLedger(f"{d}/l.jsonl") as ledger:
            ledger.write("seen1", 0, "user: 'the study door was locked from inside'", "stated_by_user")
            ledger.write("seen2", 0, "photo shows a key in the lock", "observed_artifact")
            ledger.write("deduction_ok", 1, "the killer was already inside", "inferred_by_model",
                         supports=("seen1", "seen2"))

            # A conclusion citing only direct evidence is admissible.
            v = check(ledger, ["seen1", "seen2"])
            assert v.admissible and v.missing == ()

            # A conclusion citing a properly-grounded inference is admissible.
            v = check(ledger, ["deduction_ok"])
            assert v.admissible, v.missing

            # An unsupported inference is not admissible, and names itself.
            ledger.write("guess", 2, "the butler did it", "inferred_by_model")
            v = check(ledger, ["guess"])
            assert not v.admissible
            assert any("guess" in m for m in v.missing)

            # An assumption can never ground anything, even indirectly.
            ledger.write("premise", 2, "assuming the will was already changed", "assumed")
            ledger.write("built_on_assumption", 3, "so the heir had motive", "inferred_by_model",
                         supports=("premise",))
            v = check(ledger, ["built_on_assumption"])
            assert not v.admissible
            assert any("premise" in m for m in v.missing), v.missing

            # A mixed chain: one grounded leg, one assumed leg -- still blocked,
            # same as the seed-7 bug: one alibi isn't enough if any other leg
            # of the same conclusion is unverified.
            ledger.write("mixed", 4, "partly grounded, partly assumed", "inferred_by_model",
                         supports=("seen1", "premise"))
            v = check(ledger, ["mixed"])
            assert not v.admissible
            assert any("premise" in m for m in v.missing)

            # No citations at all is trivially inadmissible.
            v = check(ledger, [])
            assert not v.admissible

            # Citing an id that doesn't exist names itself as missing rather
            # than crashing -- the check has no ground truth to fall back on.
            v = check(ledger, ["never_written"])
            assert not v.admissible
            assert any("never_written" in m for m in v.missing)

            # narrator-7gj: `missing` names ids and the reason each failed --
            # never the refused claim itself. This is checked over every
            # blocked shape at once, on the strings the checker actually
            # emits, because `missing` is rendered downstream by callers this
            # module cannot see: it reached the persona voice prompt that way
            # once. An id is a pointer into the ledger; the claim is the
            # content the checker just refused, and a checker's output should
            # not carry the thing it declined to let anyone say.
            ledger.write("loud", 5, "Colonel Mustard strangled the parlourmaid", "inferred_by_model")
            ledger.write("loud_premise", 5, "assuming the conservatory key was copied", "assumed")
            ledger.write("loud_chain", 6, "so the Colonel had the run of the house", "inferred_by_model",
                         supports=("loud_premise",))
            # `missing` names the leg that actually failed, not every id on the
            # way down to it -- same rule the assumed-chain case above relies
            # on, so an ask can go after the evidence that would fix it.
            for cited, must_name in (
                (["loud"], "loud"),
                (["loud_chain"], "loud_premise"),
                (["loud", "loud_chain"], "loud_premise"),
            ):
                v = check(ledger, cited)
                assert not v.admissible
                blob = " | ".join(v.missing)
                assert must_name in blob, f"{must_name} must be named as missing, got {v.missing}"
                for leaked in ("Mustard", "strangled", "parlourmaid", "conservatory", "copied"):
                    assert leaked not in blob, f"claim text leaked into missing: {leaked!r}"

    print("ok")


if __name__ == "__main__":
    _self_check()
