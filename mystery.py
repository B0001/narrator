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


def can_reach(spatial, origin, destination, steps):
    """Could someone standing in `origin` be in `destination` `steps` later?

    Loitering is allowed, so only the shortest path matters, not its parity.
    """
    if steps < 0:
        return False
    try:
        return nx.shortest_path_length(spatial.graph, origin, destination) <= steps
    except nx.NetworkXNoPath:
        return False


class AlibiEngine:
    """Turns sightings into opportunity, using the map rather than authored alibis.

    A sighting exonerates only when the geometry says so: too far from the
    Study, with too few steps left to get there.
    """

    def __init__(self, sim):
        self.sim = sim

    def had_opportunity(self, room, t):
        """Could a person seen in `room` at time `t` have done the killing?"""
        return can_reach(self.sim.spatial, room, self.sim.crime_scene, self.sim.murder_time - t)

    def could_have_left_trace(self, room, t, flee_room):
        """...and then reached `flee_room`, where the footprint was found.

        `flee_room` is a parameter because this is the one method on this
        class the *checker* calls (narrator-jtv). It used to read the room
        off `sim.trajectories[sim.culprit]`, which meant
        `suspects_consistent_with_clues()` dereferenced the very answer it
        exists to derive independently. The caller supplies it from the
        published footprint clue instead.

        Contrast `exonerating_sighting` below, which reads trajectories
        freely and correctly: that one is only ever called from
        `build_clues()`, on the producer's side of the line. The rule is not
        "never touch ground truth", it is "the checker never touches it".
        """
        if not self.had_opportunity(room, t):
            return False
        if flee_room is None:
            # No trace clue in this clue set: there is no flight to place, so
            # opportunity is the whole test.
            return True
        last = self.sim.time_steps - 1
        return can_reach(
            self.sim.spatial, self.sim.crime_scene, flee_room, last - self.sim.murder_time)

    def exonerating_sighting(self, who):
        """Earliest sighting of `who` that the map alone proves innocent.

        Falls back to the moment of the murder, where being anywhere but the
        Study is its own alibi — innocents are never at the scene then, so this
        always terminates.
        """
        for t in range(self.sim.murder_time + 1):
            room = self.sim.trajectories[who][t]
            if not self.had_opportunity(room, t):
                return t, room
        return self.sim.murder_time, self.sim.trajectories[who][self.sim.murder_time]


class CausalRealityGraph:
    """What actually happened. Events are nodes, causal links are edges.

    Held strictly apart from the epistemic layer: this is ground truth, and
    nothing here is available to the investigation until a clue exposes it.
    """

    def __init__(self, sim):
        self.sim = sim
        self.graph = nx.DiGraph()

    def build(self):
        sim = self.sim
        last = sim.time_steps - 1
        flee_room = sim.trajectories[sim.culprit][last]

        def event(node, t, description, *causes):
            self.graph.add_node(node, t=t, description=description)
            for c in causes:
                self.graph.add_edge(c, node)

        event("victim_arrives", sim.murder_time, f"{sim.victim} enters the {sim.crime_scene}")
        event("culprit_arrives", sim.murder_time, f"{sim.culprit} enters the {sim.crime_scene}")
        event("murder", sim.murder_time, f"{sim.culprit} kills {sim.victim} in the {sim.crime_scene}",
              "victim_arrives", "culprit_arrives")
        event("flight", last, f"{sim.culprit} flees to the {flee_room}", "murder")
        event("footprint_left", last, f"Mud from the flight is left in the {flee_room}", "flight")
        event("discovery", last, f"{sim.victim}'s body is found in the {sim.crime_scene}", "murder")
        return self.graph

    def traces_to_murder(self):
        """Every event must connect to the killing, or it does not belong in the story."""
        g = self.graph.to_undirected()
        return all(nx.has_path(g, n, "murder") for n in self.graph)

    def causes_precede_effects(self):
        """A cause cannot happen after its effect. Cheap check, catches real errors."""
        return all(self.graph.nodes[u]["t"] <= self.graph.nodes[v]["t"] for u, v in self.graph.edges)


class EpistemicClueGraph:
    """What the investigation can know, and whether that is enough to solve it."""

    def __init__(self, sim):
        self.sim = sim
        self.dag = nx.DiGraph()
        self.engine = AlibiEngine(sim)

    def build_clues(self):
        sim = self.sim
        last = sim.time_steps - 1
        self.dag.add_node("Root", kind="root", t=last,
                          description=f"Body found in the {sim.crime_scene} at t={last}")
        # Declared before anything points at it: add_edge would otherwise create
        # it as a bare, attribute-less node that mid-build inspection would see.
        # t sits past the last event because the deduction is reached after them.
        self.dag.add_node("Deduction_Culprit", kind="deduction", t=sim.time_steps,
                          description="The culprit is whoever has no alibi")

        flee_room = sim.trajectories[sim.culprit][sim.time_steps - 1]
        self.dag.add_node(
            "Clue_Footprint", kind="trace", room=flee_room, t=sim.time_steps - 1,
            description=f"Muddy footprint found in the {flee_room}")
        self.dag.add_edge("Root", "Clue_Footprint")
        self.dag.add_edge("Clue_Footprint", "Deduction_Culprit")

        # One sighting per innocent. A single witness leaves the rest of the cast
        # equally guilty, which is the unfairness this layer exists to prevent.
        for who in sim.witnesses():
            t, room = self.engine.exonerating_sighting(who)
            node = f"Clue_Alibi_{who.replace(' ', '_')}"
            self.dag.add_node(
                node, kind="alibi", who=who, room=room, t=t,
                description=f"{who} was seen in the {room} at t={t}")
            self.dag.add_edge("Root", node)
            self.dag.add_edge(node, "Deduction_Culprit")

        return self.dag

    def _flee_room_from_clues(self):
        """Where the flight ended, read off the published trace clue.

        The clue graph already carries this: `Clue_Footprint` is built with
        `room=flee_room` and its description names the room out loud. Taking
        it from here rather than from `sim.trajectories[sim.culprit]` is what
        keeps the solver on the clue set (narrator-jtv).

        `None` when this clue set holds no trace at all, which is the honest
        answer rather than an error: `clue_partition._subgraph_view` hands
        this checker arbitrary subsets, and a solver that has not been shown
        the footprint has no flight to place. It is also what the old code
        did on such a subset -- it read the real room off ground truth, and
        `can_reach(crime_scene, flee_room, 1)` came back True -- so the
        hidden-profile harness's results are preserved, not quietly moved.
        """
        traces = [d["room"] for _, d in self.dag.nodes(data=True) if d.get("kind") == "trace"]
        if len(traces) > 1:
            raise ValueError(f"expected at most one trace clue to locate the flight, got {len(traces)}")
        return traces[0] if traces else None

    def suspects_consistent_with_clues(self):
        """Independent solver: who survives the clue set?

        Reads only the clues and the cast — never sim.culprit, and never the
        trajectories it could be recovered from. If this returns anything but
        a single name, the mystery is unfair and the generator is wrong,
        which is the whole point of checking it this way.
        """
        flee_room = self._flee_room_from_clues()
        alibied = {
            d["who"] for _, d in self.dag.nodes(data=True)
            if d.get("kind") == "alibi"
            and not self.engine.could_have_left_trace(d["room"], d["t"], flee_room)
        }
        return set(self.sim.suspects) - alibied

    def validate_solvability(self):
        """A mystery nobody can reason their way to is not a mystery."""
        return (
            nx.is_directed_acyclic_graph(self.dag)
            and nx.has_path(self.dag, "Root", "Deduction_Culprit")
            and len(self.suspects_consistent_with_clues()) == 1
        )


def build(seed=None):
    sim = WorldSimulation(SpatialGraph(), seed=seed)
    sim.generate_timeline()
    # Ground truth hangs off the sim rather than widening the return value,
    # which several callers already unpack as a pair.
    sim.causal = CausalRealityGraph(sim)
    sim.causal.build()
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

        causal = sim.causal
        assert nx.is_directed_acyclic_graph(causal.graph), "ground truth cannot contain a causal loop"
        assert causal.traces_to_murder(), "every event must trace back to the killing"
        assert causal.causes_precede_effects(), "a cause must not happen after its effect"
        assert nx.has_path(causal.graph, "murder", "footprint_left"), "the footprint must descend from the murder"
        assert sim.culprit in causal.graph.nodes["murder"]["description"]

        # Alibis must be earned from the map, not asserted. Every sighting used
        # as an alibi has to be one the geometry actually rules out, and the
        # culprit's own position must never produce one.
        engine = clues.engine
        flee_room = clues._flee_room_from_clues()
        assert flee_room == sim.trajectories[sim.culprit][sim.time_steps - 1], (
            "the published trace clue must name the room the culprit actually fled to")
        for _, d in clues.dag.nodes(data=True):
            if d.get("kind") == "alibi":
                assert not engine.could_have_left_trace(d["room"], d["t"], flee_room), (
                    f"seed {s}: {d['who']} seen in {d['room']} at t={d['t']} could still have done it")
        culprit_room = sim.trajectories[sim.culprit][sim.murder_time]
        assert engine.could_have_left_trace(culprit_room, sim.murder_time, flee_room), (
            "the culprit must have had opportunity")
        # The solver never sees sim.culprit; it must land on them anyway.
        assert clues.suspects_consistent_with_clues() == {sim.culprit}, (
            f"seed {s}: clues admit {clues.suspects_consistent_with_clues()}, culprit is {sim.culprit}")

    # Every clue must be derived from the simulation, never invented.
    sim, clues = build(seed=3)
    footprint = clues.dag.nodes["Clue_Footprint"]["description"]
    assert sim.trajectories[sim.culprit][sim.time_steps - 1] in footprint

    # Under-clued mysteries must fail loudly: drop one alibi and two suspects
    # remain equally guilty, so solvability has to go false.
    dropped = next(n for n, d in clues.dag.nodes(data=True) if d.get("kind") == "alibi")
    clues.dag.remove_node(dropped)
    assert len(clues.suspects_consistent_with_clues()) == 2, "removing an alibi must widen the suspect set"
    assert not clues.validate_solvability(), "an under-clued mystery must not validate"

    # --- narrator-jtv: the solver must not be able to read the answer. ---
    # could_have_left_trace() used to fetch the flee room from
    # sim.trajectories[sim.culprit], so suspects_consistent_with_clues() --
    # the function whose entire job is to derive the culprit independently --
    # dereferenced the culprit to do it.
    #
    # Output-equivalence cannot prove this fixed, and that is worth stating:
    # the leak was inert. can_reach(crime_scene, flee_room, 1) is the same
    # constant for every alibi clue, so it factored out of the comparison and
    # the answer came out {culprit} whether it was True or False. A test that
    # moved ground truth and checked the answer did not move would have passed
    # against the broken code. So the proof is denial of access instead: run
    # the solver against a sim that refuses the two attributes outright. This
    # is the same standard chapters.py holds itself to for sim.culprit, made
    # enforceable rather than promised.
    class _NoGroundTruth:
        FORBIDDEN = ("culprit", "trajectories", "causal")

        def __init__(self, sim):
            self._sim = sim

        def __getattr__(self, name):
            if name in _NoGroundTruth.FORBIDDEN:
                raise AssertionError(f"the solver reached for ground truth: sim.{name}")
            return getattr(self._sim, name)

    for seed in range(20):
        sim, clues = build(seed=seed)
        blindfolded = EpistemicClueGraph(_NoGroundTruth(sim))
        blindfolded.dag = clues.dag  # the published clue set, unchanged
        assert blindfolded.suspects_consistent_with_clues() == {sim.culprit}, (
            f"seed {seed}: the solver must land on the culprit from the clues alone")
        assert blindfolded.validate_solvability()

        # Exercise the far side of the short-circuit as well. Every alibi
        # clue fails had_opportunity by construction -- that is what makes it
        # an alibi -- so the checker on its own returns before reaching the
        # can_reach branch, and a ground-truth read hiding there would never
        # run under the guard. Someone standing in the crime scene at the
        # murder time is the case that does reach it.
        blind_engine = blindfolded.engine
        assert blind_engine.could_have_left_trace(
            sim.crime_scene, sim.murder_time, blindfolded._flee_room_from_clues())

        # ...and with no trace clue in the set, that person is still not
        # alibied. Returning False here instead would alibi everyone holding
        # an alibi clue, which would let a solo agent who never saw the
        # footprint "solve" the case -- collapsing the hidden-profile
        # contract clue_partition.py is built to enforce.
        assert blind_engine.could_have_left_trace(sim.crime_scene, sim.murder_time, None)

    # The guard has to be able to fail, or it proves nothing: reaching for a
    # forbidden attribute through it must raise.
    probe = _NoGroundTruth(sim)
    assert probe.crime_scene == sim.crime_scene, "the guard must pass observable state through"
    # Named explicitly: an empty FORBIDDEN would make the loop below vacuous
    # and the whole proof above would pass while guarding nothing.
    assert set(_NoGroundTruth.FORBIDDEN) >= {"culprit", "trajectories"}
    for forbidden in _NoGroundTruth.FORBIDDEN:
        try:
            getattr(probe, forbidden)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"the ground-truth guard failed to catch sim.{forbidden}")

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
        deduced = clues.suspects_consistent_with_clues()
        print(f"\nSolver (clues only, blind to ground truth): {sorted(deduced)}")
        print(f"Solvable: {clues.validate_solvability()}")
    else:
        _self_check()
