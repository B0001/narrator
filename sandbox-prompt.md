You are working autonomously in the narrator repo. Make aggressive, real
progress. Do not stop to ask permission; do not stop early because you are
unsure whether there is work left.

## What this repo is

A teaching project at the intersection of code generation and psychology, for
an audience of three teenagers learning by building. Personality is a data
structure (`ocean.py`), motive is a graph traversal (`motive.py`), pathology
is an observable multi-agent interaction (`agents.py`), and a mystery is a
generated causal graph checked for fairness (`mystery.py`). `narrator-cby` is
the original epic (C1-C4, mostly closed); `narrator-c5b` turns the same
machinery around into a fair-play chatbot. Run `bd ready` — do not assume a
single epic id, there are several live ones.

## The standard everything is held to

**The output of this system is a claim that a conclusion — a mystery's
solution, or a chatbot's deduction — actually follows from evidence that was
independently checked, not just asserted by the thing that produced it.**

This has already failed once for real: `validate_solvability()` originally
only proved a path existed from Root to Deduction_Culprit, which is
structural, not logical. A seed-7 run generated a footprint in the Library
where Lady Margaret also stood at t=2 — clues consistent with more than one
suspect — and the structural check passed it anyway (`narrator-cby.4.1`). It
was only caught once an independent solver enumerated suspects consistent
with the clue set instead of trusting the generator's own claim that it had
built a solvable mystery. C5's admissibility check is the same shape again:
the model's private reasoning is not evidence, only what the ledger says the
user has actually seen is, and a checker that could read the model's hidden
state would rubber-stamp it exactly like the old structural check did.

So:

- **A result is a candidate until something measured says otherwise.** Never
  let a producer's output be phrased, logged, or reported as if it were
  verified. State the scope you actually covered: *this* seed, *these*
  turns, *this* threshold. Coverage you did not measure is not coverage you
  have.
- **Prefer abstention to a confident answer.** A component that returns
  "cannot tell" (or, in C5, withholds and names the missing evidence) on the
  cases it cannot separate is worth more than one that guesses and is right
  most of the time — because the consumer of the output cannot tell which
  mode they are in.
- **A number is only allowed to exist in a document if the code produces it,
  or the document says where it came from.** When `prd.md` and the code
  disagree you have two honest moves: fix the code, or fix the document.
  Never a third.
- **A passing test with a name is evidence. Your reasoning is not.** Every
  module self-checks standalone (`if __name__ == "__main__":`, assert-based)
  in addition to anything under `tests/`. Make it fail first if you can — an
  assertion you never saw fail is an assertion you have not verified.
- **The checker must not be able to see the producer's internals.**
  `validate_solvability()` and the C5 admissibility check both run on the
  same rule: no access to ground truth, only to what an independent party
  could actually observe. If the verifying half can read the generating
  half's state, it will agree with it, and you will have tested nothing.

## Working rules

- The bead is the specification. Where `prd.md` and a bead disagree, the bead
  wins; where the code and a bead disagree, say so rather than silently
  following one.
- Reproduce before you file. A bead asserting a problem you did not observe
  wastes a whole worker session.
- Work outside the current bead's scope gets filed as a new bead, not done now.
- Leave one runnable check behind for any non-trivial logic. It does not need
  a framework; it needs to fail if the logic breaks.
- Stdlib first, few dependencies — `prd.md`'s legibility constraint for the
  taught code applies to `ocean.py`/`mystery.py`/`agents.py`/etc. It does not
  apply to dev tooling (`pytest`, `uv`) that never ships to a learner.
- Do not commit or push unless the bead explicitly says to.
