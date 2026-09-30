"""Ollama backend: the default `Backend` (see `base.py`), local, no key.

This is exactly the HTTP call `ocean.generate()` used to make inline, moved
here so `ocean.py` (and every caller that imports `generate` from it) keeps
working unchanged while a second backend (`narrator-c5b.2.2`) becomes
possible to add without editing a single caller.

    python3 backends/ocean_ollama.py   # self-check (no live Ollama needed)
"""

import json
import urllib.request

try:
    from .base import Reply  # imported as backends.ocean_ollama (e.g. from ocean.py)
except ImportError:
    from base import Reply  # run directly: `python ocean_ollama.py`, backends/ is on sys.path


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
        data = json.load(r)
    # Ollama's own response names the model that actually answered; fall back
    # to the requested name only if the server ever omits it (narrator-c5b.2.4
    # -- a caller must be able to record which model answered, not assume it).
    return Reply(data["response"], model=data.get("model", model))


def _self_check():
    from base import conforms

    assert conforms(generate), "generate() must satisfy the Backend interface"

    class FakeProfile:
        def system_prompt(self):
            return "a system prompt"

        def options(self):
            return {"temperature": 0.5, "top_p": 0.9, "repeat_penalty": 1.1}

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": "generated text"}).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data)
        return FakeResponse()

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        result = generate(FakeProfile(), "a prompt", model="test-model")
    finally:
        urllib.request.urlopen = real_urlopen

    assert result == "generated text", "must return the 'response' field of the Ollama reply"
    assert captured["url"] == "http://localhost:11434/api/generate", "must hit the default host"
    assert captured["timeout"] == 120
    assert captured["body"] == {
        "model": "test-model",
        "prompt": "a prompt",
        "system": "a system prompt",
        "options": {"temperature": 0.5, "top_p": 0.9, "repeat_penalty": 1.1},
        "stream": False,
    }, "request body must carry the compiled profile, not the profile object itself"

    # narrator-c5b.2.4: the reply names the model that actually answered.
    assert isinstance(result, Reply), "generate() must return a Reply, not a plain str"
    assert result.model == "test-model", "Ollama's response 'model' field is what answered"

    # If a server response ever omits "model", fall back to what was
    # requested rather than crashing or guessing something else.
    urllib.request.urlopen = fake_urlopen
    captured.clear()

    class FakeResponseNoModel:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": "other text"}).encode()

    def fake_urlopen_no_model(req, timeout=None):
        return FakeResponseNoModel()

    urllib.request.urlopen = fake_urlopen_no_model
    try:
        result2 = generate(FakeProfile(), "a prompt", model="fallback-model")
    finally:
        urllib.request.urlopen = real_urlopen
    assert result2.model == "fallback-model", "with no 'model' in the response, fall back to what was requested"

    print("ok")


if __name__ == "__main__":
    _self_check()
