"""Ollama backend: the default `Backend` (see `base.py`), local, no key.

This is exactly the HTTP call `ocean.generate()` used to make inline, moved
here so `ocean.py` (and every caller that imports `generate` from it) keeps
working unchanged while a second backend (`narrator-c5b.2.2`) becomes
possible to add without editing a single caller.

    python3 backends/ocean_ollama.py   # self-check (no live Ollama needed)
"""

import json
import urllib.request


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

    print("ok")


if __name__ == "__main__":
    _self_check()
