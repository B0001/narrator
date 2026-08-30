"""Clue partitioner with a solvability contract (narrator-cby.3.5.1).

Stasser and Titus's hidden-profile paradigm needs the clue set split so that
no individual agent can solve the mystery alone but the pooled set can. This
module is the experimental manipulation: it splits an `EpistemicClueGraph`'s
clues across N agents and checks the result the same way `mystery.py` checks
the generator -- by running the real `validate_solvability()` against a
reconstructed clue graph, never by asserting the split is fair because of how
it was built.

    python3 clue_partition.py            # self-check
    python3 clue_partition.py --seed 7 --agents 3   # print one partition

Contract (see narrator-cby.3.5.1):
  - Every clue is either shared (goes to every agent) or unique (goes to
    exactly one agent) -- no clue is dropped.
  - The union of everyone's clues must validate_solvability().
  - Every individual agent's own subset must NOT validate_solvability().
  - Both halves are checked with mystery.EpistemicClueGraph.validate_solvability(),
    not reimplemented here -- a checker that re-derives its own notion of
    "solvable" could drift from the one the rest of the codebase trusts.
  - Same seed reproduces the same split.
"""

import random
import sys

import mystery


class PartitionError(ValueError):
    """No split of this clue set satisfies the contract. Names which half failed."""


class CluePartition:
    """A seeded split of one EpistemicClueGraph's clues across N agents."""

    def __init__(self, seed, n_agents, shared, unique):
        self.seed = seed
        self.n_agents = n_agents
        self.shared = frozenset(shared)
        # agent index -> frozenset of clues seen by that agent alone.
        self.unique = {i: frozenset(unique.get(i, ())) for i in range(n_agents)}

    def agent_clues(self, i):
        """Everything agent `i` actually holds: their shared clues plus their own."""
        return self.shared | self.unique[i]

    def pooled_clues(self):
        """Everything the group holds once everyone's clues are on the table."""
        return self.shared.union(*self.unique.values())


def _clue_nodes(clues):
    """Every clue under investigation -- not Root, not the deduction itself."""
    return [n for n, d in clues.dag.nodes(data=True) if d.get("kind") in ("trace", "alibi")]


def _subgraph_view(clues, node_names):
    """A same-shape EpistemicClueGraph restricted to `node_names`.

    Reconstructs the Root -> clue -> Deduction_Culprit edges build_clues()
    would have made, so the real validate_solvability() runs unmodified
    against a subset instead of the full graph. This is the whole point: the
    checker used here is the one mystery.py already trusts, not a copy of it.
    """
    view = mystery.EpistemicClueGraph(clues.sim)
    view.dag.add_node("Root", **clues.dag.nodes["Root"])
    view.dag.add_node("Deduction_Culprit", **clues.dag.nodes["Deduction_Culprit"])
    for n in node_names:
        view.dag.add_node(n, **clues.dag.nodes[n])
        view.dag.add_edge("Root", n)
        view.dag.add_edge(n, "Deduction_Culprit")
    return view


def partition(clues, n_agents, seed=None, shared_probability=0.3, max_attempts=500):
    """Split clues.dag's clues across n_agents under the solvability contract.

    Every clue is assigned, seeded by `seed`, to "shared" (probability
    `shared_probability`) or to exactly one randomly chosen agent. The split
    is accepted only once the pooled set validates and no individual agent's
    set does; both checked via mystery.EpistemicClueGraph.validate_solvability().
    Retries are drawn from the same seeded stream, so a given seed always
    walks the same sequence of candidate splits and lands on the same result.

    Raises PartitionError, naming which half of the contract is unreachable,
    if no candidate within max_attempts satisfies it.
    """
    if n_agents < 2:
        raise PartitionError(
            f"individual half failed: {n_agents} agent(s) can't hide anything from itself"
        )

    all_clues = _clue_nodes(clues)
    rng = random.Random(seed)

    last_union_ok = None
    last_solved_agents = None

    for _ in range(max_attempts):
        shared = set()
        unique = {i: set() for i in range(n_agents)}
        for clue in all_clues:
            if rng.random() < shared_probability:
                shared.add(clue)
            else:
                unique[rng.randrange(n_agents)].add(clue)

        candidate = CluePartition(seed, n_agents, shared, unique)

        union_ok = _subgraph_view(clues, candidate.pooled_clues()).validate_solvability()
        if not union_ok:
            last_union_ok = False
            continue
        last_union_ok = True

        solved_agents = [
            i for i in range(n_agents)
            if _subgraph_view(clues, candidate.agent_clues(i)).validate_solvability()
        ]
        if solved_agents:
            last_solved_agents = solved_agents
            continue

        return candidate

    if not last_union_ok:
        raise PartitionError(
            f"union half failed: no split of {len(all_clues)} clue(s) across "
            f"{n_agents} agent(s) produced a pooled set that validate_solvability() "
            f"accepts, after {max_attempts} seeded attempts"
        )
    raise PartitionError(
        f"individual half failed: every seeded split left at least one agent "
        f"(e.g. agent {last_solved_agents[0]}) able to validate_solvability() alone, "
        f"after {max_attempts} attempts -- this clue set can't be hidden across "
        f"{n_agents} agent(s)"
    )


def _self_check():
    sim, clues = mystery.build(seed=3)
    all_clues = set(_clue_nodes(clues))
    assert len(all_clues) >= 2, "self-check needs a clue set with something to hide"

    result = partition(clues, n_agents=3, seed=11)

    # Recorded, not asserted: every clue landed exactly once, shared or unique.
    assert result.shared | set().union(*result.unique.values()) == all_clues
    assert result.pooled_clues() == all_clues
    for i in range(3):
        for j in range(i + 1, 3):
            assert result.unique[i].isdisjoint(result.unique[j]), "a unique clue leaked to a second agent"
    for i in range(3):
        assert result.shared <= result.agent_clues(i), "a shared clue is missing from an agent"

    # Both halves of the contract, checked by the real validator.
    assert _subgraph_view(clues, result.pooled_clues()).validate_solvability(), \
        "pooled clue set must validate_solvability()"
    for i in range(3):
        assert not _subgraph_view(clues, result.agent_clues(i)).validate_solvability(), \
            f"agent {i} must not be able to validate_solvability() alone"

    # Same seed, same split.
    again = partition(clues, n_agents=3, seed=11)
    assert again.shared == result.shared
    assert again.unique == result.unique

    # A different seed is allowed to land on a different split.
    varied = [partition(clues, n_agents=3, seed=s).shared for s in range(20)]
    assert len(set(varied)) > 1, "every seed produced the same shared set -- rng isn't wired in"

    # Drop one alibi clue so the full set can no longer validate at all -- no
    # split can save a pooled set that isn't solvable to begin with.
    unsolvable_sim, unsolvable_clues = mystery.build(seed=3)
    dropped = next(n for n, d in unsolvable_clues.dag.nodes(data=True) if d.get("kind") == "alibi")
    unsolvable_clues.dag.remove_node(dropped)
    assert not unsolvable_clues.validate_solvability(), "test setup: expected an unsolvable clue set"
    try:
        partition(unsolvable_clues, n_agents=2, seed=1, max_attempts=20)
        assert False, "an unsolvable clue set must be rejected, not silently split"
    except PartitionError as e:
        assert "union half failed" in str(e)

    # One agent can't hide anything from itself: reject, don't degrade.
    try:
        partition(clues, n_agents=1, seed=1)
        assert False, "n_agents=1 must be rejected, not silently accepted"
    except PartitionError as e:
        assert "individual half failed" in str(e)

    # Force a genuinely unsplittable clue set: shrink to one witness, so the
    # single alibi clue is both necessary and sufficient to solve alone.
    # Whichever agent receives it violates the individual half no matter how
    # the rest of the split goes, so every seed must raise -- there is no
    # legal partition to find, and the function must say so rather than hand
    # back a split that quietly breaks the contract.
    tiny_sim, _ = mystery.build(seed=5)
    tiny_sim.suspects = [tiny_sim.culprit, tiny_sim.witnesses()[0]]
    tiny_clues = mystery.EpistemicClueGraph(tiny_sim)
    tiny_clues.build_clues()
    assert len(_clue_nodes(tiny_clues)) == 2, "expected exactly one alibi clue plus the footprint"
    assert tiny_clues.validate_solvability(), "the shrunk mystery must still be solvable as a whole"
    for s in range(10):
        try:
            partition(tiny_clues, n_agents=2, seed=s, max_attempts=50)
            assert False, f"seed {s}: an unsplittable clue set must always be rejected"
        except PartitionError as e:
            assert "individual half failed" in str(e)

    print("ok")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        n_agents = int(sys.argv[sys.argv.index("--agents") + 1]) if "--agents" in sys.argv else 3
        sim, clues = mystery.build(seed=seed)
        result = partition(clues, n_agents=n_agents, seed=seed)
        print(f"Culprit (never shown to any agent): {sim.culprit}\n")
        print(f"Shared clues ({len(result.shared)}):")
        for n in sorted(result.shared):
            print(f"  {n:24} {clues.dag.nodes[n]['description']}")
        for i in range(n_agents):
            print(f"\nAgent {i} unique clues ({len(result.unique[i])}):")
            for n in sorted(result.unique[i]):
                print(f"  {n:24} {clues.dag.nodes[n]['description']}")
        pooled_ok = _subgraph_view(clues, result.pooled_clues()).validate_solvability()
        print(f"\nPooled set validates: {pooled_ok}")
        for i in range(n_agents):
            solo_ok = _subgraph_view(clues, result.agent_clues(i)).validate_solvability()
            print(f"Agent {i} alone validates: {solo_ok}")
    else:
        _self_check()
