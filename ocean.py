"""OCEAN personality vector -> Ollama generation parameters.

Five-Factor traits are scalars in [-1.0, 1.0]. They compile into a system
prompt and sampling options for a local Ollama model.

    python3 ocean.py                 # self-check
    python3 ocean.py --demo "..."    # generate against local ollama
"""

import json
import sys
import urllib.request
from dataclasses import asdict, astuple, dataclass

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

# Descriptors injected when a trait clears THRESHOLD in either direction.
DESCRIPTORS = {
    "openness": ("incurious, concrete, sticks to the familiar", "imaginative, abstract, chases novelty"),
    "conscientiousness": ("impulsive, disorganized, leaves things unfinished", "methodical, precise, finishes what it starts"),
    "extraversion": ("reserved, low-energy, speaks only when asked", "outgoing, talkative, fills silences"),
    "agreeableness": ("blunt, skeptical, contradicts freely", "warm, accommodating, seeks agreement"),
    "neuroticism": ("calm, unbothered, steady under pressure", "anxious, defensive, hedges and second-guesses"),
}

THRESHOLD = 0.4
IDENTITY = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]


@dataclass
class Ocean:
    openness: float = 0.0
    conscientiousness: float = 0.0
    extraversion: float = 0.0
    agreeableness: float = 0.0
    neuroticism: float = 0.0

    def __post_init__(self):
        for t in TRAITS:
            v = getattr(self, t)
            if not -1.0 <= v <= 1.0:
                raise ValueError(f"{t}={v} outside [-1.0, 1.0]")

    def save(self, path):
        """Write the profile as JSON so it can be shared and diffed."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path):
        """Read a profile, rejecting anything that is not all five traits in range.

        ponytail: the dataclass is the schema. A JSON Schema file to describe
        five floats would be a second copy of these rules to keep in sync.
        """
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
        if unknown := sorted(set(data) - set(TRAITS)):
            raise ValueError(f"{path}: unknown traits {unknown}; expected {list(TRAITS)}")
        if missing := sorted(set(TRAITS) - set(data)):
            raise ValueError(f"{path}: missing traits {missing}")
        for t in TRAITS:
            # bool is an int subclass, and True as a trait value is a mistake worth naming.
            if isinstance(data[t], bool) or not isinstance(data[t], (int, float)):
                raise ValueError(f"{path}: {t} must be a number, got {data[t]!r}")
        try:
            return cls(**data)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e

    def transform(self, matrix):
        """Re-weight traits through a 5x5 matrix, e.g. a cultural adjustment.

        Takes a CultureMatrix, whose citation is announced on stderr so a
        learner always sees which coefficients moved the profile, or a bare 5x5
        list for the identity and test cases that make no claim about anyone.

        ponytail: no coefficients shipped -- the default is identity and real
        matrices are yours to source. Baking in per-population numbers we made
        up would be stereotype-encoding, not psychometrics. See load_matrix().
        """
        if isinstance(matrix, CultureMatrix):
            print(f"applying {matrix}", file=sys.stderr)
            matrix = matrix.matrix
        v = astuple(self)
        out = [sum(matrix[i][j] * v[j] for j in range(5)) for i in range(5)]
        return Ocean(*[max(-1.0, min(1.0, x)) for x in out])

    def system_prompt(self):
        lines = []
        for t in TRAITS:
            v = getattr(self, t)
            if abs(v) > THRESHOLD:
                lines.append(f"- {DESCRIPTORS[t][v > 0]} (strength {abs(v):.2f})")
        if not lines:
            return "Respond neutrally, with no pronounced personality."
        return "Adopt this personality in every response:\n" + "\n".join(lines)

    def options(self):
        """Neuroticism widens sampling; conscientiousness tightens it."""
        temperature = 0.7 + 0.4 * self.neuroticism - 0.3 * self.conscientiousness
        return {
            "temperature": round(max(0.1, min(1.5, temperature)), 3),
            "top_p": round(max(0.3, min(0.99, 0.9 + 0.09 * self.openness)), 3),
            "repeat_penalty": round(1.1 + 0.2 * max(0.0, self.conscientiousness), 3),
        }


@dataclass
class CultureMatrix:
    """A 5x5 trait re-weighting plus the citation it came from.

    The citation is not metadata, it is the admission ticket. A matrix keyed to
    a population is a claim about that population; numbers we invented would be
    stereotypes dressed as psychometrics. So source is required and unsourced
    matrices do not load -- see prd.md, "Cultural transformation".
    """
    matrix: list
    source: str

    def __str__(self):
        return f"culture matrix from {self.source}"


def load_matrix(path):
    """Load a sourced 5x5 transformation matrix from JSON.

    Expects {"matrix": [[...] x5], "source": "citation"}. A bare 5x5 array is
    rejected: that is the old unsourced shape, and silently accepting it is the
    exact hole this gate exists to close.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected an object with 'matrix' and 'source', got a bare "
            f"{type(data).__name__}; every matrix must carry its citation")
    if missing := sorted({"matrix", "source"} - set(data)):
        raise ValueError(f"{path}: missing {missing}; every matrix must carry its citation")
    source = data["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{path}: source must be a non-empty citation, got {source!r}")
    m = data["matrix"]
    if len(m) != 5 or any(len(row) != 5 for row in m):
        raise ValueError(f"{path}: expected 5x5 matrix, got {len(m)} rows")
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{path}: matrix[{i}][{j}] must be a number, got {v!r}")
    return CultureMatrix(m, source.strip())


def generate(profile, prompt, model="qwen2.5-coder:14b", host="http://localhost:11434"):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": profile.system_prompt(),
        "options": profile.options(),
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["response"]


def _self_check():
    neurotic = Ocean(neuroticism=0.9, conscientiousness=-0.5)
    assert "anxious" in neurotic.system_prompt()
    assert neurotic.options()["temperature"] > 0.7, "neuroticism must widen sampling"

    careful = Ocean(conscientiousness=0.9)
    assert careful.options()["temperature"] < 0.7, "conscientiousness must tighten sampling"
    assert "methodical" in careful.system_prompt()

    assert Ocean().system_prompt().startswith("Respond neutrally")
    assert Ocean(openness=THRESHOLD).system_prompt().startswith("Respond neutrally"), "at threshold is not over it"

    # Low pole reads as the negative descriptor, not the positive one.
    assert "incurious" in Ocean(openness=-0.9).system_prompt()

    # Identity is a no-op; a matrix actually mixes traits; results stay clamped.
    p = Ocean(openness=0.5, neuroticism=-0.3)
    assert p.transform(IDENTITY) == p
    swap = [r[:] for r in IDENTITY]
    swap[0], swap[4] = swap[4], swap[0]
    assert p.transform(swap).openness == -0.3
    blowup = [[3.0] * 5 for _ in range(5)]
    assert all(abs(v) <= 1.0 for v in astuple(p.transform(blowup))), "must clamp to [-1, 1]"

    for bad in (1.5, -2.0):
        try:
            Ocean(openness=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"openness={bad} should have been rejected")

    _check_profile_io()
    _check_matrix_gate()
    print("ok")


def _check_matrix_gate():
    """No matrix loads without a citation. This is a values constraint, not a
    style one: the failing cases below are the point of the feature."""
    import io
    import json as _json
    import tempfile
    from contextlib import redirect_stderr

    good = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
    good[0][4] = 0.2

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/m.json"

        def write(obj):
            with open(path, "w") as f:
                _json.dump(obj, f)

        rejected = [
            (good, "a bare 5x5 array -- the old unsourced shape"),
            ({"matrix": good}, "matrix with no source"),
            ({"source": "Smith 2019"}, "source with no matrix"),
            ({"matrix": good, "source": ""}, "empty source"),
            ({"matrix": good, "source": "   "}, "whitespace-only source"),
            ({"matrix": good, "source": 42}, "non-string source"),
            ({"matrix": good[:4], "source": "Smith 2019"}, "4x5 matrix"),
            ({"matrix": [["x"] * 5] + good[1:], "source": "Smith 2019"}, "non-numeric entry"),
        ]
        for obj, why in rejected:
            write(obj)
            try:
                load_matrix(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{why} should have been rejected")

        # A sourced matrix loads, keeps its citation, and announces it when applied.
        write({"matrix": good, "source": "  Smith et al. 2019, J. Cross-Cult. Psych.  "})
        cm = load_matrix(path)
        assert cm.source == "Smith et al. 2019, J. Cross-Cult. Psych.", "citation is stripped and kept"

    err = io.StringIO()
    with redirect_stderr(err):
        moved = Ocean(openness=0.1, neuroticism=0.5).transform(cm)
    assert "Smith et al. 2019" in err.getvalue(), "the citation must be announced on use"
    assert abs(moved.openness - 0.2) < 1e-9, "the matrix must actually re-weight"

    # The identity path still takes a bare list, so tests make no claim about anyone.
    assert Ocean(openness=0.5).transform(IDENTITY).openness == 0.5


def _check_profile_io():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/p.json"
        original = Ocean(0.3, -0.7, 0.0, 0.95, -0.25)
        original.save(path)
        assert Ocean.load(path) == original, "save/load must round-trip exactly"

        rejects = {
            "not an object": "[1, 2, 3]",
            "unknown trait": '{"openness": 0, "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0, "charisma": 1}',
            "missing trait": '{"openness": 0.5}',
            "out of range": '{"openness": 4.0, "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0}',
            "wrong type": '{"openness": "high", "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0}',
            "bool sneaking in": '{"openness": true, "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0}',
        }
        for label, blob in rejects.items():
            with open(path, "w") as f:
                f.write(blob)
            try:
                Ocean.load(path)
            except ValueError as e:
                assert path in str(e), f"{label}: error must name the file, got {e}"
            else:
                raise AssertionError(f"{label} should have been rejected")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        if "--profile" in sys.argv:
            profile = Ocean.load(sys.argv[sys.argv.index("--profile") + 1])
        else:
            profile = Ocean(neuroticism=0.9, conscientiousness=-0.6, agreeableness=-0.4)
        print(generate(profile, sys.argv[sys.argv.index("--demo") + 1]))
    else:
        _self_check()
