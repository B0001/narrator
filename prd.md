# Narrator — PRD

Source: `codegen_psychology_chat.md`

## Premise

A teaching project that sits at the intersection of code generation and psychology. Learners
build behavioral systems rather than reading about them: personality becomes a data structure,
motive becomes a graph traversal, and pathology becomes an observable interaction between
agents. Everything runs locally against Ollama.

The audience is three teenagers learning by building. That drives two constraints:

- **Legible over clever.** Stdlib first, few dependencies, code that can be read start to finish.
- **Observable.** Every component must produce output a learner can inspect and argue with.
  A simulation nobody can watch teaches nothing.

## Non-goals

- Not a psychometrics research tool. Outputs are illustrative, not clinically valid.
- No claims about real people or populations. See "Cultural transformation" below.

## Decision: local-by-default, cloud opt-in (2026-08-29, `narrator-c5b.1`)

Supersedes the former non-goal "No cloud model providers. Local Ollama only."
C5 (`narrator-c5b`) adds a second generation backend behind the `Backend`
protocol (`backends/base.py`): Anthropic's `/v1/messages`, model
`claude-fable-5`. That's a real reversal for an audience of three teenagers,
not a footnote, so it's recorded here rather than silently overwritten.

- **What stays local by default.** Ollama (`backends/ocean_ollama.py`) is
  what every existing entry point calls with no key set — `ocean.py`,
  `agents.py`, `motive.py`, `chapters.py`, `turn.py`. Every caller keeps
  running unchanged against a local model unless a learner deliberately
  switches backends; no component is silently pointed at the network.
- **What the cloud path is for.** Fable is opt-in per learner: for whoever
  wants a stronger model without provisioning local hardware, or wants to
  compare `claude-fable-5`'s output against a local model for the same
  profile. It's a second backend behind the same protocol, not a
  replacement for the local one. See `narrator-c5b.2.2` for the
  implementation and `narrator-c5b.2.5` for which components are cheap
  enough to actually run against it — `agents.converse`'s round-robin
  transcript grows quadratically in turns, which is a real cost surface on
  a metered backend, not a hypothetical.
- **Who holds the key.** The learner. `ANTHROPIC_API_KEY` comes from the
  environment only — never hardcoded, never read from a profile file, never
  checked into source. A run against Fable with no key set must fail naming
  the missing variable, not a raw connection error.
- **Why this is a real change, not a formality.** Cloud calls mean API
  keys, per-token cost, a network dependency, and a teenager's conversation
  text leaving their machine. Those are the actual stakes the old non-goal
  was fencing off. This decision doesn't remove the fence, it moves it:
  cloud use stays possible, but never silent, never the default, and never
  authorized by anyone but the person running it.

---

## C1 — Personality parameterization (built: `ocean.py`)

Five-Factor traits as a five-dimensional state vector compiled into model behavior.

- `Ocean` dataclass; five traits as scalars validated to [-1.0, 1.0].
- Traits past ±0.4 compile into system-prompt injections; both poles are expressible.
- Traits map to sampling parameters: neuroticism widens temperature, conscientiousness
  tightens it and raises repeat penalty, openness nudges top_p.
- `generate()` POSTs to Ollama at `localhost:11434`.

**Remaining:** JSON schema for profiles + load/save, so profiles are shareable artifacts
rather than literals in source.

### Cultural transformation

A 5×5 matrix re-weights the trait vector before compilation, expressing that a construct
does not carry identical behavioral meaning across cultural contexts.

Shipped as **mechanism only** — the default is identity and matrices load from JSON at
runtime. No population coefficients are bundled. The cross-cultural FFM literature is real
(the Chinese Personality Assessment Inventory and its interpersonal-relatedness factor being
the standard example), but numbers we invented and keyed to an ethnicity would be stereotypes
dressed as psychometrics. Coefficients enter this project only with a citation attached, and
each shipped matrix carries its source in a sidecar field.

---

## C2 — Graph-based motive generation

A procedural mystery generator where graph structure encodes psychology.

- NetworkX graph; nodes are cognitive biases and psychological triggers, not rooms.
- Edges are weighted by how plausibly one state escalates into the next.
- Traversal from baseline to the crime yields the murderer's path — a deterioration
  sequence that doubles as the story outline.
- Seeded and reproducible: same seed, same mystery.
- Renders the path as readable prose beats via Ollama, and the graph as an image.

The learning payload: a character arc is a path through a state space, and "motive" is an
edge weight you chose.

---

## C3 — Multi-agent pathology simulation

Agents instantiated from C1 profiles, talking to each other.

- Spawn N agents, each with an `Ocean` profile and a name.
- Round-robin turn loop over a shared transcript; each agent sees the conversation so far.
- Full transcript logged to JSONL — who said what, under which profile, at which turn.
- Reference scenario: a high-disagreeableness/high-extraversion profile against a
  high-agreeableness one, in a task requiring them to split a scarce resource.
- Basic metrics over the transcript: turn share, agreement rate, who concedes.

**Framing constraint:** profiles are trait configurations, never diagnoses. Agents get
descriptive names ("low-agreeableness profile"), not clinical labels. Simulated behavior is
a property of the prompt and the sampler, not evidence about people who hold those traits.

---

## Success criteria

A learner can, in one sitting: define a profile, watch it change model output, generate a
mystery whose motive graph they can read, and run two profiles against each other and
explain the transcript. Each component runs standalone from the command line.
