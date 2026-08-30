"""Anthropic backend: the opt-in `Backend` (see `base.py`), cloud, needs a key.

Second implementation of the same seam `ocean_ollama.py` was moved behind --
this is the backend `prd.md`'s "Decision: local-by-default, cloud opt-in"
(`narrator-c5b.1`) reverses the old "local Ollama only" non-goal for. Three
shapes differ from Ollama's `/api/generate`: `system` is a top-level request
field here, not folded into the prompt; `max_tokens` is required by
Anthropic and has no equivalent in `profile.options()`, so this module picks
a default; and a reply is a list of content blocks, not one `response`
string, so the text has to be pulled out of it.

`claude-fable-5` rejects `temperature` / `top_p` / `top_k` outright (HTTP
400 -- the model removed all sampler controls), so -- per `base.py`'s "drop
what the backend has no equivalent for" -- `profile.options()` is not
forwarded at all. On Fable the persona reaches the model through the system
prompt only; `prd.md` C1's trait-to-sampler path is inert on this backend.

The key comes from `ANTHROPIC_API_KEY` in the environment, never from source
or a profile file (prd.md, `narrator-c5b.1`). With no key set this fails by
naming that variable, before any network attempt -- not a connection error.

    python3 backends/ocean_fable.py   # self-check (no key, no live call needed)
"""

import json
import os
import urllib.request

DEFAULT_MAX_TOKENS = 4096
ANTHROPIC_VERSION = "2023-06-01"


def generate(profile, prompt, model="claude-fable-5", max_tokens=DEFAULT_MAX_TOKENS,
             host="https://api.anthropic.com"):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set -- the Fable backend needs it in the "
            "environment (never in source or a profile file)."
        )

    # No sampling params: claude-fable-5 400s on temperature/top_p/top_k, and
    # profile.options() carries exactly those. The system prompt does the work.
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": profile.system_prompt(),
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(f"{host}/v1/messages", body, headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.load(r)

    # A safety refusal is HTTP 200 with stop_reason "refusal" and usually no
    # text -- surface it rather than returning "" as if the model had answered.
    if reply.get("stop_reason") == "refusal":
        raise RuntimeError(
            "claude-fable-5 declined this request (stop_reason 'refusal'); "
            "nothing was generated."
        )
    return "".join(block["text"] for block in reply["content"] if block.get("type") == "text")


def _self_check():
    from base import conforms

    assert conforms(generate), "generate() must satisfy the Backend interface"

    class FakeProfile:
        def system_prompt(self):
            return "a system prompt"

        def options(self):
            return {"temperature": 0.5, "top_p": 0.9, "repeat_penalty": 1.1}

    # No key set: must fail naming the variable, without touching the network.
    real_urlopen = urllib.request.urlopen

    def exploding_urlopen(*a, **k):
        raise AssertionError("must not attempt a network call with no key set")

    urllib.request.urlopen = exploding_urlopen
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        try:
            generate(FakeProfile(), "a prompt")
        except RuntimeError as e:
            assert "ANTHROPIC_API_KEY" in str(e), "error must name the missing variable"
        else:
            raise AssertionError("must raise when ANTHROPIC_API_KEY is unset")
    finally:
        urllib.request.urlopen = real_urlopen
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key

    # Key set: request shape is right, response is parsed out of content blocks.
    captured = {}

    def fake_response(payload):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(payload).encode()

        return FakeResponse()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data)
        return captured["response"]

    urllib.request.urlopen = fake_urlopen
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        captured["response"] = fake_response({
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": "generated "},
                {"type": "text", "text": "text"},
            ],
        })
        result = generate(FakeProfile(), "a prompt", model="test-model")

        assert result == "generated text", "must join the 'text' fields of the content blocks"
        assert captured["url"] == "https://api.anthropic.com/v1/messages"
        assert captured["timeout"] == 120
        assert captured["headers"]["x-api-key"] == "test-key"
        assert captured["headers"]["anthropic-version"] == ANTHROPIC_VERSION
        assert captured["body"] == {
            "model": "test-model",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": "a system prompt",
            "messages": [{"role": "user", "content": "a prompt"}],
        }, "request body: system top-level, no sampling params (Fable rejects them)"

        # A refusal must raise, not return "".
        captured["response"] = fake_response({"stop_reason": "refusal", "content": []})
        try:
            generate(FakeProfile(), "a prompt")
        except RuntimeError as e:
            assert "refusal" in str(e), "a refusal must be surfaced, got: " + str(e)
        else:
            raise AssertionError("stop_reason 'refusal' must raise")
    finally:
        urllib.request.urlopen = real_urlopen
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    print("ok")


if __name__ == "__main__":
    _self_check()
