"""Gemini backend: a third `Backend` (see `base.py`), cloud, needs a key.

Same seam as `ocean_ollama.py` and `ocean_fable.py`. Gemini's
`generateContent` puts the system prompt in `systemInstruction`, the prompt
in `contents`, and sampler knobs in `generationConfig` (camelCase). Unlike
Fable, Gemini accepts `temperature` and `topP`, so `prd.md` C1's
trait-to-sampler path is live on this backend; `repeat_penalty` has no
equivalent and is dropped, per `base.py`.

The key comes from `GEMINI_API_KEY` in the environment, never from source.
With no key set this fails naming that variable, before any network attempt.

    python3 backends/ocean_gemini.py   # self-check (no key, no live call needed)
"""

import json
import os
import urllib.error
import urllib.request

try:
    from .base import Reply  # imported as backends.ocean_gemini
except ImportError:
    from base import Reply  # run directly: `python ocean_gemini.py`, backends/ is on sys.path

# Ollama option name -> Gemini generationConfig name. Anything not here is dropped.
SAMPLER_KEYS = {"temperature": "temperature", "top_p": "topP", "top_k": "topK"}


def generate(profile, prompt, model="gemini-flash-latest",
             host="https://generativelanguage.googleapis.com"):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set -- the Gemini backend needs it in the environment.")

    config = {SAMPLER_KEYS[k]: v for k, v in profile.options().items() if k in SAMPLER_KEYS}
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": profile.system_prompt()}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": config,
    }).encode()
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    req = urllib.request.Request(f"{host}/v1beta/models/{model}:generateContent", body, headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as e:
        # Google's JSON error body says *why* (unknown model, retired model,
        # bad key); urllib's HTTPError alone only says the status code.
        raise RuntimeError(f"{model}: HTTP {e.code}: {e.read().decode(errors='replace')}") from e

    # A blocked prompt or response comes back 200 with no text -- surface it
    # rather than returning "" as if the model had answered.
    candidates = reply.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
    if not parts:
        reason = candidates[0].get("finishReason") if candidates else reply.get("promptFeedback")
        raise RuntimeError(f"{model} returned no text ({reason}); nothing was generated.")
    text = "".join(p.get("text", "") for p in parts)
    return Reply(text, model=reply.get("modelVersion", model))


def _self_check():
    from base import conforms

    assert conforms(generate), "generate() must satisfy the Backend interface"

    class FakeProfile:
        def system_prompt(self):
            return "a system prompt"

        def options(self):
            return {"temperature": 0.5, "top_p": 0.9, "repeat_penalty": 1.1}

    real_urlopen = urllib.request.urlopen
    old_key = os.environ.pop("GEMINI_API_KEY", None)

    def exploding_urlopen(*a, **k):
        raise AssertionError("must not attempt a network call with no key set")

    urllib.request.urlopen = exploding_urlopen
    try:
        generate(FakeProfile(), "a prompt")
    except RuntimeError as e:
        assert "GEMINI_API_KEY" in str(e), "error must name the missing variable"
    else:
        raise AssertionError("must raise when GEMINI_API_KEY is unset")

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(captured["response"]).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data)
        return FakeResponse()

    urllib.request.urlopen = fake_urlopen
    os.environ["GEMINI_API_KEY"] = "test-key"
    try:
        captured["response"] = {
            "candidates": [{"content": {"parts": [{"text": "generated "}, {"text": "text"}]}}],
            "modelVersion": "gemini-test-001",
        }
        result = generate(FakeProfile(), "a prompt", model="gemini-test")
        assert result == "generated text", "must join the text parts"
        assert isinstance(result, Reply) and result.model == "gemini-test-001", "must report modelVersion"
        assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
        assert captured["headers"]["x-goog-api-key"] == "test-key"
        assert captured["body"] == {
            "systemInstruction": {"parts": [{"text": "a system prompt"}]},
            "contents": [{"role": "user", "parts": [{"text": "a prompt"}]}],
            "generationConfig": {"temperature": 0.5, "topP": 0.9},
        }, "sampler keys renamed to camelCase, repeat_penalty dropped"

        # A block must raise, not return "".
        captured["response"] = {"candidates": [{"finishReason": "SAFETY"}]}
        try:
            generate(FakeProfile(), "a prompt")
        except RuntimeError as e:
            assert "SAFETY" in str(e)
        else:
            raise AssertionError("an empty candidate must raise")

        # No modelVersion: fall back to what was requested.
        captured["response"] = {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}
        assert generate(FakeProfile(), "p", model="requested").model == "requested"
    finally:
        urllib.request.urlopen = real_urlopen
        if old_key is not None:
            os.environ["GEMINI_API_KEY"] = old_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)

    print("ok")


if __name__ == "__main__":
    _self_check()
