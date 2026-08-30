"""Discussion protocol and the unshared-clue measure (narrator-cby.3.5.2).

Runs `agents.converse()` over a `clue_partition.partition()` split: each
agent's prompt carries only the clues they were dealt (their `evidence`),
never the rest of the pooled set and never `sim.culprit`. What comes out is
the transcript Stasser and Titus's paradigm needs measured -- per clue,
whether it was ever voiced, by whom, on which turn, and whether the group
built on it after the first mention -- plus the group's own final answer,
scored against ground truth no agent ever saw.

    python3 discussion.py                      # self-check, no model needed
    python3 discussion.py --seed 7 --agents 3   # real run via Ollama

Why uptake is not "mentioned more than once":
  A clue repeated three times by the one agent who holds it is still
  information that never left that agent's head -- the group never engaged
  with it. `uptake` is true only when a clue is mentioned again by someone
  OTHER than whoever first raised it: that is the textual signature of the
  group actually building on it, not just the holder repeating themselves.
  `mention_count` is kept alongside it, but per the classic finding (shared
  information is mentioned earlier and more often; unique information is
  voiced once, if at all, and then dropped) mention count alone conflates
  "the group used this" with "one agent said it twice," so it is reported
  as a separate column rather than folded into a single score.

Why the group answer is found in the transcript, not asked for out-of-band:
  A dedicated "state your verdict" round would be a second channel the
  agents could use differently than ordinary discussion, and scoring it
  would require trusting that the model actually treated it specially. The
  group's answer is instead read off the same transcript the clue measures
  come from: the most recent turn that names exactly one suspect. If no
  turn ever narrows to one name, the group is scored as undecided rather
  than credited with a guess it did not actually commit to -- abstention
  over a confident wrong read, same as `validate_solvability()` itself.
"""

import re
import sys

import clue_partition
import mystery
from agents import Agent, converse
from ocean import Ocean, generate

_FOOTPRINT_WORDS = re.compile(r"\b(?:footprint|mud|muddy)\b", re.I)


def _evidence_text(clues, node_names):
    """What one agent is told: their own clue descriptions, nothing else."""
    lines = [clues.dag.nodes[n]["description"] for n in sorted(node_names)]
    if not lines:
        return "(you have nothing concrete to report)"
    return "\n".join(f"- {line}" for line in lines)


def _mentions_clue(text, data):
    """Did this turn's text voice this clue's content?

    A marker match, same spirit as metrics.py's agreement regex: cheap, and
    wrong on paraphrase that avoids the room/name entirely. Alibi clues are
    keyed on the witness's name (one alibi clue per witness, so it is a
    unique marker); the footprint clue has no name, so it needs the room
    together with a footprint word to avoid matching any turn that merely
    names the room in passing.
    """
    kind = data.get("kind")
    if kind == "alibi":
        return re.search(rf"\b{re.escape(data['who'])}\b", text) is not None
    if kind == "trace":
        return data["room"].lower() in text.lower() and _FOOTPRINT_WORDS.search(text) is not None
    raise ValueError(f"unknown clue kind: {kind!r}")


def clue_report(clues, partition, records, agent_names):
    """Per clue: shared or unique, voiced, first speaker/turn, uptake, mention count.

    `partition` decides "shared" vs. "unique" by construction (it is the
    manipulation); "voiced" and "uptake" are read from `records`, which is
    the transcript agents.converse() actually produced -- an agent handed a
    unique clue is not thereby credited with having said it.
    """
    all_nodes = partition.shared | set().union(*partition.unique.values())
    report = []
    for node in sorted(all_nodes):
        data = clues.dag.nodes[node]
        mentions = [(r["turn"], r["speaker"]) for r in records if _mentions_clue(r["text"], data)]
        voiced = bool(mentions)
        first_turn, first_speaker = mentions[0] if voiced else (None, None)
        uptake = voiced and any(speaker != first_speaker for _, speaker in mentions[1:])

        if node in partition.shared:
            kind, held_by = "shared", tuple(agent_names)
        else:
            owner = next(i for i in range(partition.n_agents) if node in partition.unique[i])
            kind, held_by = "unique", (agent_names[owner],)

        report.append({
            "clue": node,
            "kind": kind,
            "held_by": held_by,
            "voiced": voiced,
            "first_speaker": first_speaker,
            "first_turn": first_turn,
            "mention_count": len(mentions),
            "uptake": uptake,
        })
    return report


def final_answer(records, suspects):
    """The group's answer: the most recent turn that names exactly one suspect.

    Scanned back-to-front so a late turn that narrows the field overrides an
    earlier one that named a different single suspect (or several). Returns
    None -- undecided -- if no turn ever names exactly one.
    """
    for r in reversed(records):
        mentioned = [s for s in suspects if re.search(rf"\b{re.escape(s)}\b", r["text"])]
        if len(mentioned) == 1:
            return mentioned[0]
    return None


def run_discussion(seed=None, n_agents=3, turns=8, path="discussion_transcript.jsonl",
                    model="llama3", generate_fn=generate, agent_names=None):
    """Build a mystery, partition its clues, and run the discussion over the split.

    Every agent's `evidence` is exactly their `partition.agent_clues(i)` --
    never the pooled set, never sim.culprit. The group's answer is scored
    against sim.culprit only after the fact, by a function (`final_answer`)
    that never received it either.
    """
    sim, clues = mystery.build(seed=seed)
    partition = clue_partition.partition(clues, n_agents=n_agents, seed=seed)
    names = agent_names or [f"Investigator {chr(65 + i)}" for i in range(n_agents)]
    if len(names) != n_agents:
        raise ValueError(f"need {n_agents} agent_names, got {len(names)}")

    roster = [
        Agent(names[i], Ocean(), evidence=_evidence_text(clues, partition.agent_clues(i)))
        for i in range(n_agents)
    ]
    topic = (
        f"{sim.victim} has been found dead in the {sim.crime_scene}. Share only what you "
        "personally observed, weigh what the others say, and try to agree on a single suspect."
    )
    records = converse(roster, topic, turns=turns, path=path, model=model, generate_fn=generate_fn)

    report = clue_report(clues, partition, records, names)
    answer = final_answer(records, sim.suspects)

    return {
        "sim": sim,
        "clues": clues,
        "partition": partition,
        "agent_names": names,
        "records": records,
        "clue_report": report,
        "group_answer": answer,
        "culprit_hit": answer is not None and answer == sim.culprit,
    }


def report_table(report):
    """The per-clue report as a printable table, same style as metrics.summary()."""
    width = max([len("clue")] + [len(r["clue"]) for r in report])
    head = f"{'clue':<{width}}  kind      held by              voiced  first        mentions  uptake"
    lines = [head, "-" * len(head)]
    for r in report:
        held = ",".join(r["held_by"]) if r["kind"] == "unique" else "all"
        first = f"{r['first_speaker']}@{r['first_turn']}" if r["voiced"] else "-"
        lines.append(
            f"{r['clue']:<{width}}  {r['kind']:<8}  {held:<20}  {str(r['voiced']):<6}  "
            f"{first:<12}  {r['mention_count']:>8}  {str(r['uptake'])}"
        )
    return "\n".join(lines)


def _self_check():
    import tempfile

    # -- clue_report / final_answer as pure functions over a hand-built transcript --
    sim, clues = mystery.build(seed=3)
    partition = clue_partition.partition(clues, n_agents=3, seed=11)
    names = ["Ana", "Bo", "Cy"]
    # This seed/partition combo (checked once, pinned here) gives one shared
    # alibi clue and unique clues split across agents 1 and 2, agent 0 empty --
    # exactly the shape the differential shared-vs-unique test below needs.
    assert sorted(partition.shared) == ["Clue_Alibi_Lord_Blackwood"]
    assert partition.unique[0] == frozenset()
    assert partition.unique[1] == frozenset({"Clue_Alibi_Butler_Jeeves"})
    assert partition.unique[2] == frozenset({"Clue_Alibi_Dr._Ellis", "Clue_Footprint"})

    shared_desc = clues.dag.nodes["Clue_Alibi_Lord_Blackwood"]["description"]
    jeeves_desc = clues.dag.nodes["Clue_Alibi_Butler_Jeeves"]["description"]
    ellis_desc = clues.dag.nodes["Clue_Alibi_Dr._Ellis"]["description"]
    footprint_desc = clues.dag.nodes["Clue_Footprint"]["description"]

    records = [
        {"turn": 0, "speaker": "Ana", "text": f"I noticed: {shared_desc}."},
        {"turn": 1, "speaker": "Bo", "text": f"Right, and also: {jeeves_desc}. {shared_desc}, agreed."},
        {"turn": 2, "speaker": "Cy", "text": f"{ellis_desc}. Also {footprint_desc}."},
        {"turn": 3, "speaker": "Ana", "text": "Not sure who that leaves."},
    ]
    report = {r["clue"]: r for r in clue_report(clues, partition, records, names)}

    shared = report["Clue_Alibi_Lord_Blackwood"]
    assert shared["kind"] == "shared" and shared["held_by"] == ("Ana", "Bo", "Cy")
    assert shared["voiced"] and shared["first_speaker"] == "Ana" and shared["first_turn"] == 0
    assert shared["mention_count"] == 2, "voiced by Ana at turn 0 and repeated by Bo at turn 1"
    assert shared["uptake"] is True, "a different speaker (Bo) mentioned it after Ana raised it"

    jeeves = report["Clue_Alibi_Butler_Jeeves"]
    assert jeeves["kind"] == "unique" and jeeves["held_by"] == ("Bo",)
    assert jeeves["voiced"] and jeeves["first_speaker"] == "Bo" and jeeves["first_turn"] == 1
    assert jeeves["mention_count"] == 1
    assert jeeves["uptake"] is False, "nobody but Bo ever mentioned it"

    ellis = report["Clue_Alibi_Dr._Ellis"]
    assert ellis["kind"] == "unique" and ellis["voiced"] and ellis["uptake"] is False

    footprint = report["Clue_Footprint"]
    assert footprint["kind"] == "unique" and footprint["voiced"]
    assert footprint["first_speaker"] == "Cy" and footprint["first_turn"] == 2
    assert footprint["uptake"] is False

    # A clue nobody voices at all is reported, not silently dropped.
    silent_records = [{"turn": 0, "speaker": "Ana", "text": "So, thoughts?"}]
    silent_report = {r["clue"]: r for r in clue_report(clues, partition, silent_records, names)}
    never = silent_report["Clue_Alibi_Butler_Jeeves"]
    assert not never["voiced"] and never["first_speaker"] is None and never["first_turn"] is None
    assert never["mention_count"] == 0 and never["uptake"] is False

    # A same-speaker repeat is not uptake -- only a different speaker counts.
    same_speaker = [
        {"turn": 0, "speaker": "Bo", "text": jeeves_desc},
        {"turn": 1, "speaker": "Bo", "text": f"As I said, {jeeves_desc}"},
    ]
    repeated = {r["clue"]: r for r in clue_report(clues, partition, same_speaker, names)}["Clue_Alibi_Butler_Jeeves"]
    assert repeated["mention_count"] == 2, "both turns count as mentions"
    assert repeated["uptake"] is False, "same speaker repeating themselves is not the group building on it"

    # final_answer: most recent turn naming exactly one suspect wins; ties or
    # silence on the point are undecided, not a coin flip.
    no_name_records = [
        {"turn": 0, "speaker": "Ana", "text": "I noticed something odd."},
        {"turn": 1, "speaker": "Bo", "text": "Interesting, tell me more."},
    ]
    assert final_answer(no_name_records, sim.suspects) is None, "no turn here names a suspect at all"
    assert final_answer(records, sim.suspects) == "Dr. Ellis", (
        "turn 2 names Dr. Ellis alone; turn 1 names two suspects and turn 3 names none"
    )
    named = records[:-1] + [{"turn": 3, "speaker": "Ana", "text": "It has to be Lord Blackwood."}]
    assert final_answer(named, sim.suspects) == "Lord Blackwood"
    reversed_last = named[:-1] + [{"turn": 3, "speaker": "Ana", "text": "Wait -- Dr. Ellis, not Blackwood."}]
    assert final_answer(reversed_last, sim.suspects) == "Dr. Ellis", "the later turn overrides the earlier guess"
    ambiguous = [{"turn": 0, "speaker": "Ana", "text": "Lord Blackwood or Dr. Ellis, hard to say."}]
    assert final_answer(ambiguous, sim.suspects) is None, "two names in the deciding turn is not a verdict"

    # -- full pipeline: agents.converse() actually runs over the partition --
    with tempfile.TemporaryDirectory() as d:
        turns = 4  # one full round (3 agents) plus one extra turn for the verdict
        call_index = 0

        def fake_generate(profile, prompt, model=None):
            nonlocal call_index
            call_index += 1
            if call_index == turns:
                return "It has to be Lord Blackwood."
            m = re.search(r"personally observed, and no one else has told you:\n(.*?)\n\nReply as", prompt, re.S)
            return m.group(1).strip() if m else "Nothing new from me."

        result = run_discussion(
            seed=3, n_agents=3, turns=turns, path=f"{d}/t.jsonl",
            generate_fn=fake_generate, agent_names=names,
        )

        # run_discussion partitions with seed=3 too, which lands on a different
        # split than the seed=11 one pinned above: Clue_Footprint is the shared
        # clue this time, and agent 2 (Cy) holds no unique clue at all.
        assert result["sim"].culprit == "Lady Margaret", "pinned: seed 3's culprit"
        assert len(result["records"]) == turns
        assert result["group_answer"] == "Lord Blackwood", "the scripted final turn names one suspect"
        assert result["culprit_hit"] is False, "Lord Blackwood != Lady Margaret -- scored against ground truth"

        by_clue = {r["clue"]: r for r in result["clue_report"]}
        assert by_clue["Clue_Footprint"]["kind"] == "shared"
        assert by_clue["Clue_Footprint"]["voiced"], "every agent echoes their evidence, shared included"
        assert by_clue["Clue_Footprint"]["uptake"] is True, (
            "agents 0 and 1 both hold and echo the shared clue on their own turns -- two distinct speakers"
        )
        assert by_clue["Clue_Alibi_Lord_Blackwood"]["kind"] == "unique"
        assert by_clue["Clue_Alibi_Lord_Blackwood"]["uptake"] is False, (
            "agent 0 (Ana) is its only holder; the scripted verdict at turn 3 names Blackwood "
            "again but it is Ana speaking both times -- same speaker, not uptake"
        )
        assert by_clue["Clue_Alibi_Butler_Jeeves"]["kind"] == "unique"
        assert by_clue["Clue_Alibi_Butler_Jeeves"]["uptake"] is False, "only its one owner ever says it"
        assert by_clue["Clue_Alibi_Dr._Ellis"]["uptake"] is False

        assert "Clue_Alibi_Lord_Blackwood" in report_table(result["clue_report"])

    print("ok")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        n_agents = int(sys.argv[sys.argv.index("--agents") + 1]) if "--agents" in sys.argv else 3
        turns = int(sys.argv[sys.argv.index("--turns") + 1]) if "--turns" in sys.argv else n_agents * 3
        result = run_discussion(seed=seed, n_agents=n_agents, turns=turns, path="discussion_transcript.jsonl")
        print(f"Culprit (never shown to any agent): {result['sim'].culprit}\n")
        print(report_table(result["clue_report"]))
        print(f"\nGroup answer: {result['group_answer']}")
        print(f"Matches culprit: {result['culprit_hit']}")
    else:
        _self_check()
