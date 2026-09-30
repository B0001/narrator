"""Trait and role conditions over one fixed partition (narrator-cby.3.5.3).

`discussion.py` (.3.5.2) runs one discussion over one partition and reports
whether each clue got voiced. This module holds that partition fixed --
same mystery seed, same `clue_partition.partition()` seed, same clue split
-- and varies only the Ocean profile the agents carry, plus one binary role
(designated critic), so any change in the unique-clue pooling rate has
nowhere to come from except the manipulation.

PREDICTIONS -- written before any condition below has been run, so the
harness can be wrong:

  H1 (agreeableness suppresses).  Raising every agent's agreeableness,
     traits otherwise neutral, should LOWER the fraction of unique clues
     ever voiced, relative to the neutral baseline. Unshared information is
     socially costly to raise -- it cannot be nodded along with -- and
     agreeableness is a preference for not paying that cost.

  H2 (openness raises it).  Raising every agent's openness, traits
     otherwise neutral, should RAISE that same fraction, relative to
     baseline. Openness is a preference for novelty; a fact nobody else has
     said is, structurally, the most novel thing available to say.

  H3 (a designated critic raises pooling regardless of the rest).  Take the
     H1 condition -- every agent, including the one who will be critic,
     already set to the trait that H1 predicts suppresses voicing -- and
     assign one agent the critic role: instructed to explicitly name
     anything nobody else has said yet. That agent's unique-clue pooling
     rate should RISE relative to the matched no-critic condition, even
     though its own trait profile is the suppressive one. This is the
     argumentative-theory prediction: an assigned role to challenge
     consensus can override personal disposition. The comparison is against
     the *matched* H1 condition, not the neutral baseline -- H3 is a claim
     about the role's effect net of trait, not about where the number ends
     up absolutely.

WHAT THE HARNESS ACTUALLY MEASURES, AND WHAT IT DOESN'T:
  Every condition below runs through the real `agents.converse()` loop and
  the real `discussion.clue_report()` scorer -- nothing about scoring is
  faked. What IS scripted is the *reply*: `generate_fn` is a small,
  documented, fully deterministic function of (trait score, a fixed
  per-clue reluctance constant, critic flag), not a live model. That makes
  this a test of the wiring -- does a trait/role manipulation actually
  reach the transcript and move `clue_report()`'s count -- the same thing
  `agreeableness_sweep.py` tests for C5's `TraitMode`. It is NOT evidence
  that a real LLM voicing decisions would move this way; that would need a
  live backend (no Ollama is reachable in this sandbox) and is out of scope
  here. Do not read the table below as a claim about model behavior, only
  about whether this specific mechanism, once wired in, responds to the
  manipulation in the predicted direction.

THE FIXED PARTITION:
  mystery seed 1, `clue_partition.partition(..., n_agents=2, seed=2)`:
  shared = {Clue_Alibi_Butler_Jeeves}, agent 0 unique = {Clue_Alibi_Dr._Ellis,
  Clue_Footprint}, agent 1 unique = {Clue_Alibi_Lord_Blackwood}. Pinned by
  `_self_check()`; if `clue_partition.partition()`'s search order ever
  changes, that assertion fails loudly instead of silently scoring a
  different split under the same seed.

  python3 trait_role_conditions.py   # self-check, no model needed
"""

import re
import sys
import tempfile

import clue_partition
import discussion
import mystery
from agents import Agent, converse
from ocean import Ocean

MYSTERY_SEED = 1
PARTITION_SEED = 2
N_AGENTS = 2
AGENT_NAMES = ("Investigator A", "Investigator B")

# Fixed, arbitrary-but-constant per-clue reluctance: how far a persona's
# score (openness - agreeableness) has to clear before that specific clue
# gets voiced. Spread across the swept conditions' scores (0.0 at baseline,
# -0.8 under high agreeableness, +0.8 under high openness) so each
# condition lands on a different count instead of all-or-nothing. Keyed by
# node name because the partition above is fixed -- these are not
# recomputed per run.
_RELUCTANCE = {
    "Clue_Alibi_Dr._Ellis": -0.6,
    "Clue_Footprint": 0.0,
    "Clue_Alibi_Lord_Blackwood": 0.6,
}

CRITIC_DIRECTIVE = (
    "Your role in this discussion is designated critic: explicitly name anything "
    "nobody else has said yet, even if it slows down agreement."
)


def _evidence_text(clues, nodes, is_critic):
    """One agent's private briefing, with the critic directive prefixed when it applies.

    The directive is real prompt text, not just an internal flag -- a live
    model given this same roster would see the same instruction a scripted
    run acts on.
    """
    lines = [clues.dag.nodes[n]["description"] for n in sorted(nodes)]
    body = "\n".join(f"- {l}" for l in lines) if lines else "(you have nothing concrete to report)"
    return f"{CRITIC_DIRECTIVE}\n{body}" if is_critic else body


def _score(profile):
    """openness pulls toward voicing unshared information, agreeableness pulls away."""
    return profile.openness - profile.agreeableness


def _build_generate_fn(clues, partition, names, profiles, critic_index):
    """Precompute each agent's one scripted reply for this condition.

    The reply is static across an agent's turns (the decision is per
    condition, not per turn): shared clues are always included -- common
    ground carries no social cost to repeat -- and each unique clue is
    included if the agent is the critic, or if their score clears that
    clue's fixed reluctance constant.
    """
    replies = {}
    for i, name in enumerate(names):
        is_critic = i == critic_index
        score = _score(profiles[i])
        voice = list(sorted(partition.shared))
        for node in sorted(partition.unique[i]):
            if is_critic or score > _RELUCTANCE.get(node, 0.0):
                voice.append(node)
        replies[name] = (
            " ".join(clues.dag.nodes[n]["description"] for n in voice) + "."
            if voice else "I don't have anything else to add right now."
        )

    def generate_fn(profile, prompt, model=None):
        name = re.match(r"You are ([^,]+),", prompt).group(1)
        return replies[name]

    return generate_fn


def run_condition(profiles, critic_index=None, turns=4, path=None):
    """Run one discussion over the fixed partition under one trait/role condition.

    profiles is one Ocean per agent, positional against AGENT_NAMES.
    critic_index, if given, names which agent gets the critic directive and
    the role override in the scripted policy.
    """
    if len(profiles) != N_AGENTS:
        raise ValueError(f"need {N_AGENTS} profiles, got {len(profiles)}")
    sim, clues = mystery.build(seed=MYSTERY_SEED)
    partition = clue_partition.partition(clues, n_agents=N_AGENTS, seed=PARTITION_SEED)
    roster = [
        Agent(
            AGENT_NAMES[i], profiles[i],
            evidence=_evidence_text(clues, partition.agent_clues(i), i == critic_index),
        )
        for i in range(N_AGENTS)
    ]
    generate_fn = _build_generate_fn(clues, partition, AGENT_NAMES, profiles, critic_index)
    records = converse(
        roster, "discuss who was where when the crime happened",
        turns=turns, path=path or tempfile.mktemp(suffix=".jsonl"), generate_fn=generate_fn,
    )
    report = discussion.clue_report(clues, partition, records, AGENT_NAMES)
    unique_report = [r for r in report if r["kind"] == "unique"]
    voiced = sum(r["voiced"] for r in unique_report)
    return {
        "report": report,
        "unique_voiced": voiced,
        "unique_total": len(unique_report),
        "pooling_rate": voiced / len(unique_report),
    }


# Named conditions, in report order. "profiles" is one Ocean applied to
# every agent alike (the manipulation the bead asks for: vary the profile,
# not which agent gets which profile). "critic_index" is None except in
# the H3 condition, which reuses H1's exact trait setting on purpose.
CONDITIONS = (
    ("baseline", (Ocean(), Ocean()), None),
    ("high_agreeableness", (Ocean(agreeableness=0.8), Ocean(agreeableness=0.8)), None),
    ("high_openness", (Ocean(openness=0.8), Ocean(openness=0.8)), None),
    ("high_agreeableness_with_critic", (Ocean(agreeableness=0.8), Ocean(agreeableness=0.8)), 1),
)


def run_all(conditions=CONDITIONS):
    """Every condition, same partition, returned in table order."""
    return [(name, run_condition(profiles, critic_index=critic)) for name, profiles, critic in conditions]


def condition_table(results):
    """One row per condition: pooling rate plus the raw voiced/total it came from."""
    width = max(len("condition"), *(len(name) for name, _ in results))
    head = f"{'condition':<{width}}  voiced/total  pooling rate"
    lines = [head, "-" * len(head)]
    for name, r in results:
        lines.append(f"{name:<{width}}  {r['unique_voiced']}/{r['unique_total']:<10}  {r['pooling_rate']:.2f}")
    return "\n".join(lines)


def _self_check():
    # Pin the partition this whole module is built on -- if partition()'s
    # search order ever changes, this fails loudly instead of silently
    # scoring a different split under the same seed.
    sim, clues = mystery.build(seed=MYSTERY_SEED)
    partition = clue_partition.partition(clues, n_agents=N_AGENTS, seed=PARTITION_SEED)
    assert sorted(partition.shared) == ["Clue_Alibi_Butler_Jeeves"]
    assert partition.unique[0] == frozenset({"Clue_Alibi_Dr._Ellis", "Clue_Footprint"})
    assert partition.unique[1] == frozenset({"Clue_Alibi_Lord_Blackwood"})

    by_name = dict(run_all())

    # -- H1: agreeableness suppresses unique-clue voicing, relative to baseline. --
    assert by_name["high_agreeableness"]["pooling_rate"] < by_name["baseline"]["pooling_rate"], (
        "H1 failed: high agreeableness did not suppress pooling relative to baseline -- "
        f"got {by_name['high_agreeableness']['pooling_rate']} vs baseline "
        f"{by_name['baseline']['pooling_rate']}"
    )
    assert by_name["high_agreeableness"]["pooling_rate"] == 0.0, (
        "expected agreeableness=0.8 to clear none of the fixed reluctance constants, got "
        f"{by_name['high_agreeableness']}"
    )

    # -- H2: openness raises it, relative to baseline. --
    assert by_name["high_openness"]["pooling_rate"] > by_name["baseline"]["pooling_rate"], (
        "H2 failed: high openness did not raise pooling relative to baseline -- "
        f"got {by_name['high_openness']['pooling_rate']} vs baseline "
        f"{by_name['baseline']['pooling_rate']}"
    )
    assert by_name["high_openness"]["pooling_rate"] == 1.0, (
        f"expected openness=0.8 to clear every fixed reluctance constant, got {by_name['high_openness']}"
    )

    # -- H3: a designated critic raises pooling against the *matched* trait
    # condition (same profile, no critic) -- the comparison H3 is actually about. --
    assert (
        by_name["high_agreeableness_with_critic"]["pooling_rate"]
        > by_name["high_agreeableness"]["pooling_rate"]
    ), (
        "H3 failed: assigning a critic under the same suppressive trait setting did not "
        f"raise pooling -- got {by_name['high_agreeableness_with_critic']['pooling_rate']} vs "
        f"matched no-critic {by_name['high_agreeableness']['pooling_rate']}"
    )
    # The critic's own held clue (Blackwood, index 1) is what should now be
    # voiced -- not the other agent's, who is still under the same
    # suppressive score and no role override.
    crit_report = {r["clue"]: r for r in by_name["high_agreeableness_with_critic"]["report"]}
    assert crit_report["Clue_Alibi_Lord_Blackwood"]["voiced"] is True, (
        "the critic (agent 1, holding Blackwood) must be the one whose clue newly appears"
    )
    assert crit_report["Clue_Alibi_Dr._Ellis"]["voiced"] is False, (
        "the non-critic agent's own clues must stay suppressed -- only the critic's role changed"
    )

    # Shared information is never the thing being suppressed or raised --
    # every condition above must still voice it, or the H1/H2/H3 comparisons
    # would be confounded by losing common ground too.
    for name, r in by_name.items():
        shared_report = next(x for x in r["report"] if x["kind"] == "shared")
        assert shared_report["voiced"], f"{name}: shared clue must always be voiced"

    table = condition_table(list(by_name.items()))
    assert "high_agreeableness_with_critic" in table
    assert "baseline" in table

    print("ok")
    print(table)


if __name__ == "__main__":
    if "--table" in sys.argv:
        print(condition_table(run_all()))
    else:
        _self_check()
