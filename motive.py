"""Load the motive vocabulary into a NetworkX digraph.

Nodes are psychological states, edges are escalations. Traversal from a
trigger to a terminal is the murderer's deterioration path (C2).

    python3 motive.py            # self-check + graph summary
"""

import json
import random
import sys

import networkx as nx

KINDS = ("trigger", "bias", "terminal")
DEFAULT_GRAPH = "motive_graph.json"


def load_graph(path=DEFAULT_GRAPH):
    """Build a validated DiGraph. Raises ValueError on an unusable vocabulary."""
    with open(path) as f:
        data = json.load(f)

    g = nx.DiGraph()
    for n in data["nodes"]:
        if n["kind"] not in KINDS:
            raise ValueError(f"{path}: node {n['id']} has kind {n['kind']!r}, expected one of {list(KINDS)}")
        if n["id"] in g:
            raise ValueError(f"{path}: duplicate node id {n['id']!r}")
        g.add_node(n["id"], kind=n["kind"], desc=n["desc"])

    for e in data["edges"]:
        for end in ("from", "to"):
            if e[end] not in g:
                raise ValueError(f"{path}: edge {e['from']}->{e['to']} references unknown node {e[end]!r}")
        if not 0 < e["weight"] <= 1:
            raise ValueError(f"{path}: edge {e['from']}->{e['to']} weight {e['weight']} outside (0, 1]")
        g.add_edge(e["from"], e["to"], weight=e["weight"], note=e.get("note", ""))

    triggers = nodes_of_kind(g, "trigger")
    terminals = nodes_of_kind(g, "terminal")
    if not triggers or not terminals:
        raise ValueError(f"{path}: need at least one trigger and one terminal node")

    # A trigger that cannot reach an ending is a dead start: the generator would
    # pick it and then have no story to tell.
    for t in triggers:
        if not any(nx.has_path(g, t, end) for end in terminals):
            raise ValueError(f"{path}: trigger {t!r} cannot reach any terminal node")

    return g


def nodes_of_kind(g, kind):
    return sorted(n for n, d in g.nodes(data=True) if d["kind"] == kind)


def deterioration_path(g, seed=None, start=None):
    """Walk trigger -> terminal, picking each step weighted by escalation strength.

    Same seed, same path. States are never revisited: a motive that loops back
    through a bias it already passed reads as repetition rather than descent.
    """
    rng = random.Random(seed)
    terminals = nodes_of_kind(g, "terminal")
    node = start if start is not None else rng.choice(nodes_of_kind(g, "trigger"))
    if node not in g:
        raise ValueError(f"unknown start node {node!r}")

    path, seen = [node], {node}
    while g.nodes[node]["kind"] != "terminal":
        options = [n for n in g.successors(node) if n not in seen]
        if not options:
            # Walked into a corner. Finish along the shortest remaining route so
            # the path still lands on an ending rather than trailing off.
            routes = [nx.shortest_path(g, node, t) for t in terminals if nx.has_path(g, node, t)]
            if not routes:
                raise ValueError(f"no route from {node!r} to a terminal")
            path += min(routes, key=len)[1:]
            break
        node = rng.choices(options, weights=[g[path[-1]][n]["weight"] for n in options])[0]
        path.append(node)
        seen.add(node)
    return path


def describe(g, path):
    return "\n".join(f"{i}. {n} — {g.nodes[n]['desc']}" for i, n in enumerate(path, 1))


def _self_check():
    import tempfile

    g = load_graph()
    triggers, terminals = nodes_of_kind(g, "trigger"), nodes_of_kind(g, "terminal")
    assert triggers and terminals
    assert all(g.in_degree(t) == 0 for t in triggers), "triggers must be entry points"
    assert all(g.out_degree(n) > 0 for n in g if g.nodes[n]["kind"] != "terminal"), "non-terminals need an exit"
    assert all("desc" in g.nodes[n] and g.nodes[n]["desc"] for n in g), "every node needs a description"

    base = json.load(open(DEFAULT_GRAPH))
    breakages = {
        "dangling edge": lambda d: d["edges"].append({"from": "rumination", "to": "ghost", "weight": 0.5}),
        "bad weight": lambda d: d["edges"].append({"from": "rumination", "to": "sunk_cost", "weight": 1.7}),
        "duplicate id": lambda d: d["nodes"].append(dict(d["nodes"][0])),
        "unknown kind": lambda d: d["nodes"].append({"id": "x", "kind": "vibe", "desc": "d"}),
        "unreachable trigger": lambda d: d["nodes"].append({"id": "orphan", "kind": "trigger", "desc": "goes nowhere"}),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for label, break_it in breakages.items():
            data = json.loads(json.dumps(base))
            break_it(data)
            path = f"{tmp}/broken.json"
            with open(path, "w") as f:
                json.dump(data, f)
            try:
                load_graph(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{label} should have been rejected")

    _check_traversal(g)
    print(f"ok — {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    for kind in KINDS:
        print(f"  {kind}: {', '.join(nodes_of_kind(g, kind))}")


def _check_traversal(g):
    triggers, terminals = nodes_of_kind(g, "trigger"), nodes_of_kind(g, "terminal")

    assert deterioration_path(g, seed=7) == deterioration_path(g, seed=7), "same seed must reproduce the path"
    paths = [deterioration_path(g, seed=s) for s in range(40)]
    assert len({tuple(p) for p in paths}) > 1, "different seeds must not all collapse to one path"

    for p in paths:
        assert p[0] in triggers, f"path must start at a trigger, got {p[0]}"
        assert p[-1] in terminals, f"path must end at a terminal, got {p[-1]}"
        assert len(set(p)) == len(p), f"path revisits a state: {p}"
        for a, b in zip(p, p[1:]):
            assert g.has_edge(a, b), f"path uses non-existent edge {a}->{b}"

    assert deterioration_path(g, seed=1, start="exposure_threat")[0] == "exposure_threat"
    try:
        deterioration_path(g, seed=1, start="nonexistent")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown start node should have been rejected")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        graph = load_graph()
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        print(describe(graph, deterioration_path(graph, seed=seed)))
    else:
        _self_check()
