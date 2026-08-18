"""Three-layer mystery pipeline: space, simulation, clues (C4).

Ported from the Mystery_Novel_Graph_Discussion.md script, with the bugs fixed
and the randomness seeded. See FIXES below.

    python3 mystery.py            # self-check
    python3 mystery.py --seed 7   # print one mystery

FIXES vs. the source script:
  1. validate_solvability() was missing `self` and could never be called.
  2. The culprit fled Study -> Conservatory, which is not an edge of the
     mansion; they now flee to the Library, which is.
  3. Innocent rerouting picked a neighbour of the crime scene, which could
     teleport them off their own path. Steps are now chosen from neighbours of
     where they actually were.
  4. random was unseeded, so no mystery could be reproduced.
"""

import random
import sys

import networkx as nx

ROOMS = ["Foyer", "Hallway", "Study", "Library", "Conservatory", "Kitchen", "Dining Room"]
DOORS = [
    ("Foyer", "Hallway"),
    ("Hallway", "Study"),
    ("Hallway", "Dining Room"),
    ("Study", "Library"),
    ("Dining Room", "Kitchen"),
    ("Kitchen", "Conservatory"),
    ("Library", "Conservatory"),
]


class SpatialGraph:
    """Physical layout. Nodes are rooms, edges are doors."""

    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(ROOMS)
        self.graph.add_edges_from(DOORS)

    def get_adjacent(self, room):
        return sorted(self.graph.neighbors(room))


class WorldSimulation:
    """Ground truth: who was where, and who did it."""

    def __init__(self, spatial, seed=None, time_steps=4):
        self.spatial = spatial
        self.time_steps = time_steps
        self.rng = random.Random(seed)
        self.suspects = ["Lord Blackwood", "Lady Margaret", "Dr. Ellis", "Butler Jeeves"]
        self.victim = "Professor Plum"
        self.crime_scene = "Study"
        self.murder_time = 2
        self.culprit = self.rng.choice(self.suspects)
        self.trajectories = {}

    def generate_timeline(self):
        """Build continuous paths for everyone; only the culprit meets the victim."""
        self.trajectories = {
            self.victim: {0: "Foyer", 1: "Hallway", 2: self.crime_scene, 3: self.crime_scene},
            # Flees to the Library: it is the only room adjacent to the Study.
            self.culprit: {0: "Dining Room", 1: "Hallway", 2: self.crime_scene, 3: "Library"},
        }
        for suspect in self.suspects:
            if suspect != self.culprit:
                self.trajectories[suspect] = self._innocent_path()
        return self.trajectories

    def _innocent_path(self):
        """A continuous walk that is never at the crime scene when it matters."""
        path = {0: self.rng.choice([r for r in ROOMS if r != self.crime_scene])}
        for t in range(1, self.time_steps):
            options = self.spatial.get_adjacent(path[t - 1])
            if t == self.murder_time:
                # Standing still beats teleporting when every door leads to the body.
                options = [r for r in options if r != self.crime_scene] or [path[t - 1]]
            path[t] = self.rng.choice(options)
        return path

    def witnesses(self):
        """Anyone whose own position at the murder puts them beyond suspicion."""
        return sorted(
            s for s in self.suspects
            if s != self.culprit and self.trajectories[s][self.murder_time] != self.crime_scene
        )


class EpistemicClueGraph:
    """What the investigation can know, and whether that is enough to solve it."""

    def __init__(self, sim):
        self.sim = sim
        self.dag = nx.DiGraph()

    def build_clues(self):
        sim = self.sim
        self.dag.add_node("Root", description=f"Body found in the {sim.crime_scene} at t={sim.time_steps - 1}")

        flee_room = sim.trajectories[sim.culprit][sim.time_steps - 1]
        self.dag.add_node("Clue_Footprint", description=f"Muddy footprint found in the {flee_room}")
        self.dag.add_edge("Root", "Clue_Footprint")

        witness = sim.witnesses()[0]
        witness_room = sim.trajectories[witness][sim.murder_time]
        self.dag.add_node("Clue_Witness", description=f"{witness} was seen in the {witness_room} at t={sim.murder_time}")
        self.dag.add_edge("Root", "Clue_Witness")

        self.dag.add_node("Deduction_Culprit", description=f"The culprit is {sim.culprit}")
        self.dag.add_edge("Clue_Footprint", "Deduction_Culprit")
        self.dag.add_edge("Clue_Witness", "Deduction_Culprit")
        return self.dag

    def validate_solvability(self):
        """A mystery nobody can reason their way to is not a mystery."""
        return nx.is_directed_acyclic_graph(self.dag) and nx.has_path(self.dag, "Root", "Deduction_Culprit")


def build(seed=None):
    sim = WorldSimulation(SpatialGraph(), seed=seed)
    sim.generate_timeline()
    clues = EpistemicClueGraph(sim)
    clues.build_clues()
    if not clues.validate_solvability():
        raise ValueError(f"seed {seed} produced an unsolvable mystery")
    return sim, clues


def _self_check():
    spatial = SpatialGraph()

    sim, clues = build(seed=7)
    again, _ = build(seed=7)
    assert sim.trajectories == again.trajectories, "same seed must reproduce the timeline"
    assert sim.culprit == again.culprit
    assert len({build(seed=s)[0].culprit for s in range(30)}) > 1, "culprit must not always be the same person"

    for s in range(20):
        sim, clues = build(seed=s)
        for who, path in sim.trajectories.items():
            assert sorted(path) == list(range(sim.time_steps)), f"{who} has a gap in their timeline"
            for t in range(1, sim.time_steps):
                a, b = path[t - 1], path[t]
                assert a == b or spatial.graph.has_edge(a, b), f"{who} teleported {a}->{b} at t={t}"

        at_scene = [w for w, p in sim.trajectories.items() if p[sim.murder_time] == sim.crime_scene]
        assert set(at_scene) == {sim.culprit, sim.victim}, f"seed {s}: only culprit and victim at the scene, got {at_scene}"
        assert sim.culprit not in sim.witnesses(), "the culprit cannot be their own alibi"
        assert clues.validate_solvability()
        assert nx.has_path(clues.dag, "Root", "Deduction_Culprit")
        assert clues.dag.nodes["Deduction_Culprit"]["description"].endswith(sim.culprit)

    # Every clue must be derived from the simulation, never invented.
    sim, clues = build(seed=3)
    footprint = clues.dag.nodes["Clue_Footprint"]["description"]
    assert sim.trajectories[sim.culprit][sim.time_steps - 1] in footprint

    print("ok")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        sim, clues = build(seed=int(sys.argv[sys.argv.index("--seed") + 1]))
        print(f"Victim: {sim.victim} — killed in the {sim.crime_scene} at t={sim.murder_time}")
        print(f"Culprit: {sim.culprit}\n")
        for who in sorted(sim.trajectories):
            print(f"  {who:16} {' -> '.join(sim.trajectories[who][t] for t in range(sim.time_steps))}")
        print("\nInvestigation:")
        for n in nx.topological_sort(clues.dag):
            print(f"  {n:20} {clues.dag.nodes[n]['description']}")
        print(f"\nSolvable: {clues.validate_solvability()}")
    else:
        _self_check()
