"""OCEAN personality vector -> Ollama generation parameters.

Five-Factor traits are scalars in [-1.0, 1.0]. They compile into a system
prompt and sampling options for a local Ollama model.

    python3 ocean.py                 # self-check
    python3 ocean.py --demo "..."    # generate against local ollama
"""

import json
import sys
import urllib.request
from dataclasses import astuple, dataclass

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

    def transform(self, matrix):
        """Re-weight traits through a 5x5 matrix, e.g. a cultural adjustment.

        ponytail: no coefficients shipped -- the default is identity and real
        matrices are yours to source. Baking in per-population numbers we made
        up would be stereotype-encoding, not psychometrics. See load_matrix().
        """
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


def load_matrix(path):
    """Load a 5x5 transformation matrix from JSON."""
    with open(path) as f:
        m = json.load(f)
    if len(m) != 5 or any(len(row) != 5 for row in m):
        raise ValueError(f"{path}: expected 5x5 matrix, got {len(m)} rows")
    return m


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

    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        profile = Ocean(neuroticism=0.9, conscientiousness=-0.6, agreeableness=-0.4)
        print(generate(profile, sys.argv[sys.argv.index("--demo") + 1]))
    else:
        _self_check()
