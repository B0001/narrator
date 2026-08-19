"""Narrative DAG -> one chapter of prose per node (C5).

Walks the clue graph in topological order and asks the model for one chapter
per node. The logic engine stays deterministic: the model never invents a clue,
a suspect or a deduction, it only dramatises the JSON state it is handed.

The epistemic boundary is structural. A node's payload is built from that node
plus nx.ancestors(dag, node) -- nothing else. Descendants and sibling branches
are unreachable by construction, so the culprit cannot appear in any chapter
before the deduction node, where the solver (not the model, and not ground
truth) names them.

    python3 chapters.py            # self-check, no model, no network
    python3 chapters.py --seed 7   # real run against local Ollama
"""

import json
import sys

import networkx as nx

from mystery import build
from ocean import Ocean, generate

# Atmospheric but disciplined: high openness for the prose, high
# conscientiousness so it stays inside the facts it was given.
NARRATOR = Ocean(openness=0.7, conscientiousness=0.7, neuroticism=-0.3)


def _facts(dag, node):
    """A node's own attributes, exactly as the logic layer wrote them."""
    return {"node": node, **dag.nodes[node]}


def _payload(clues, node, order, prose, sim):
    """Everything this chapter is allowed to know, and nothing else.

    `known` is the node's ancestors in the DAG, in chapter order. Nothing is
    hand-filtered: if a fact is not upstream of this node it is not in here.
    """
    ancestors = nx.ancestors(clues.dag, node)
    known = [n for n in order if n in ancestors]
    this = _facts(clues.dag, node)
    if this["kind"] == "deduction":
        # The solver reads the clue set and is blind to sim.culprit -- see
        # mystery.suspects_consistent_with_clues. The name is derived, not leaked.
        this["solution"] = sorted(clues.suspects_consistent_with_clues())
    return {
        "chapter": order.index(node) + 1,
        "of": len(order),
        # sim is read for scene-setting only. sim.culprit and sim.suspects are
        # never touched in this module; the self-check enforces it.
        "setting": {
            "victim": sim.victim,
            "crime_scene": sim.crime_scene,
            "time_steps": sim.time_steps,
        },
        "established": [_facts(clues.dag, n) for n in known],
        "prior_chapters": [prose[n] for n in known if n in prose],
        "this_chapter": this,
    }


def _prompt(payload):
    return (
        "You are writing one chapter of a mystery novel.\n\n"
        "The JSON below is the complete state of the investigation at this point. "
        "It is everything you know and everything the reader may learn here.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n\n"
        f"Write chapter {payload['chapter']} of {payload['of']} in 150-250 words of prose. "
        "Dramatise 'this_chapter'; treat 'established' and 'prior_chapters' as already told. "
        "Invent no clue, character, room or deduction that is not in the JSON, and name no "
        "culprit unless 'solution' appears above."
    )


def write_chapters(sim, clues, model="llama3", generate_fn=generate, narrator=NARRATOR):
    """One chapter per DAG node, in topological order, earliest event first.

    generate_fn is injectable so the whole pipeline runs without a model.
    Ties break on event time then node name, so chapter order is reproducible
    and the investigation uncovers sightings oldest-first instead of
    alphabetically. Chapter one is still the discovery of the body: Root is an
    ancestor of every clue, so no ordering can precede it, and opening on the
    corpse is what a mystery does anyway.
    """
    order = list(nx.lexicographical_topological_sort(
        clues.dag, key=lambda n: (clues.dag.nodes[n].get("t", 0), n)))
    prose, chapters = {}, []
    for node in order:
        payload = _payload(clues, node, order, prose, sim)
        prose[node] = generate_fn(narrator, _prompt(payload), model=model).strip()
        chapters.append({"node": node, "payload": payload, "prose": prose[node]})
    return chapters


def _self_check():
    for seed in range(12):
        sim, clues = build(seed=seed)
        prompts = []

        def fake_generate(profile, prompt, model=None):
            prompts.append(prompt)
            return f"  chapter {len(prompts)} in a voice with O={profile.openness}  "

        chapters = write_chapters(sim, clues, generate_fn=fake_generate)
        nodes = [c["node"] for c in chapters]

        assert len(chapters) == clues.dag.number_of_nodes() == len(prompts), "one chapter per node"
        assert set(nodes) == set(clues.dag), "every node must get a chapter"
        rank = {n: i for i, n in enumerate(nodes)}
        for u, v in clues.dag.edges:
            assert rank[u] < rank[v], f"seed {seed}: {u} must be told before {v}"
        assert chapters[0]["prose"] == "chapter 1 in a voice with O=0.7", "prose is stripped"

        # Chronology: within what the DAG allows, earlier events are narrated
        # first. Root is everyone's ancestor so it stays chapter one; the clues
        # between it and the deduction must run oldest sighting first.
        assert nodes[0] == "Root" and nodes[-1] == "Deduction_Culprit"
        times = [clues.dag.nodes[n]["t"] for n in nodes[1:-1]]
        assert times == sorted(times), f"seed {seed}: clues out of chronological order: {list(zip(nodes[1:-1], times))}"

        for c in chapters:
            established = {f["node"] for f in c["payload"]["established"]}
            assert established == nx.ancestors(clues.dag, c["node"]), (
                f"seed {seed}: {c['node']} payload must hold exactly its ancestors")

        # The one that matters: the culprit's name -- in full or in any part of
        # it -- must not reach the model before the deduction chapter, and must
        # reach it there. Checked against the prompt actually sent, so adding
        # e.g. the cast list or sim.culprit to the payload fails this.
        parts = [p for p in sim.culprit.split() if len(p) > 3] + [sim.culprit]
        deduction = rank["Deduction_Culprit"]
        for i, prompt in enumerate(prompts):
            leaked = [p for p in parts if p in prompt]
            if i < deduction:
                assert not leaked, f"seed {seed}: chapter {i + 1} leaks the culprit {leaked}"
            else:
                assert sim.culprit in prompt, f"seed {seed}: the deduction must name the culprit"
        assert deduction == len(prompts) - 1, "the deduction is the last chapter"

    # A payload with no ancestors carries no prior chapters: Root starts cold.
    sim, clues = build(seed=3)
    root = write_chapters(sim, clues, generate_fn=lambda p, t, model=None: "x")[0]
    assert root["node"] == "Root" and root["payload"]["established"] == []
    assert root["payload"]["prior_chapters"] == []
    assert sim.victim in _prompt(root["payload"]), "the victim is public knowledge from page one"

    print("ok")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        sim, clues = build(seed=int(sys.argv[sys.argv.index("--seed") + 1]))
        for c in write_chapters(sim, clues):
            print(f"--- {c['payload']['chapter']}. {c['node']} ---\n{c['prose']}\n")
    else:
        _self_check()
